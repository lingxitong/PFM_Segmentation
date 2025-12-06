import math
import timm
from timm.layers import use_fused_attn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import repeat
from .build_conch_v1_5 import Conch_V1_5_Attention
from typing import Optional


class DoraLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, r=4, lora_alpha=1.0):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.r = r
        self.lora_alpha = lora_alpha
        # original linear layer (frozen)
        self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        
        # LoRA: low-rank adaptor
        # Note: lora_a: [in_features, r], lora_b: [r, out_features]
        # So lora_a @ lora_b: [in_features, r] @ [r, out_features] = [in_features, out_features]
        self.lora_a = nn.Parameter(torch.zeros(in_features, r), requires_grad=True)
        self.lora_b = nn.Parameter(torch.zeros(r, out_features), requires_grad=True)
        self.scale = lora_alpha

        # Dora magnitude scaling parameter
        self.m = nn.Parameter(torch.ones(1), requires_grad=True)  # Scalar magnitude

        # initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize original weight and bias (frozen)
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

        # Initialize magnitude to 1
        nn.init.ones_(self.m)
    
    def forward(self, x):  # shape [batch, seq_len, in_features]
        # compute LoRA update: lora_a @ lora_b
        lora_update = self.lora_a @ self.lora_b  # Shape: [in_features, out_features]
        
        # Apply scale
        lora_update = lora_update * self.scale  # Shape: [in_features, out_features]
        
        # compute original weight norm and updated weight norm (Frobenius norm)
        weight_norm = self.weight.norm(p='fro')
        updated_weight = self.weight + lora_update.t()  # Shape: [out_features, in_features]
        updated_norm = updated_weight.norm(p='fro')
        
        # compute direction (unit vector in matrix space)
        direction = updated_weight / updated_norm  # Shape: [out_features, in_features]
        
        # apply Dora: magnitude * direction * original_norm
        dora_weight = self.m * direction * weight_norm
        
        # compute output using Dora-adjusted weight
        output = F.linear(x, dora_weight, self.bias)
        return output


class Dora_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            dora_r: int = 16,
            dora_alpha: float = 1.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = DoraLinear(dim, dim * 3, bias=qkv_bias, r=dora_r, lora_alpha=dora_alpha)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DoraLinear(dim, dim, r=dora_r, lora_alpha=dora_alpha)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                try:
                    attn = attn + attn_mask
                except Exception:
                    # attn_mask might be boolean mask; use masked_fill for that case
                    attn = attn.masked_fill(attn_mask, float('-inf'))

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def equip_model_with_dora(pfm_name, model, rank, alpha):
    """
    Equip a PFM model with Dora by replacing its attention layers with Dora_Attention layers.
    This version also copies original attention weights into the Dora-Attention module.

    Args:
        pfm_name (str): Name of the PFM model.
        model (nn.Module): The PFM model to be equipped with Dora.
        rank (int): Rank of the low-rank adaptation.
        alpha (float): Scaling factor for the Dora output.

    Returns:
        nn.Module: The PFM model with Dora applied to its attention layers.
    """
    def copy_weights(src_attn, dst_dora_attn):
        with torch.no_grad():
            # Copy original weights to the frozen part of Dora
            dst_dora_attn.qkv.weight.copy_(src_attn.qkv.weight)
            if src_attn.qkv.bias is not None:
                dst_dora_attn.qkv.bias.copy_(src_attn.qkv.bias)

            dst_dora_attn.proj.weight.copy_(src_attn.proj.weight)
            if src_attn.proj.bias is not None:
                dst_dora_attn.proj.bias.copy_(src_attn.proj.bias)

            if hasattr(src_attn, 'q_norm') and hasattr(dst_dora_attn, 'q_norm') and isinstance(dst_dora_attn.q_norm, nn.LayerNorm):
                dst_dora_attn.q_norm.load_state_dict(src_attn.q_norm.state_dict())
            if hasattr(src_attn, 'k_norm') and hasattr(dst_dora_attn, 'k_norm') and isinstance(dst_dora_attn.k_norm, nn.LayerNorm):
                dst_dora_attn.k_norm.load_state_dict(src_attn.k_norm.state_dict())

    if pfm_name in ['uni_v1', 'uni_v2', 'virchow_v2', 'gigapath','virchow_v1','phikon','phikon_v2','hibou_l','musk','lunit_vits8','midnight12k','hoptimus_0','hoptimus_1','patho3dmatrix-vision','kaiko-vitl14','conch_v1']:
        for name, module in model.named_modules():
            if isinstance(module, timm.models.vision_transformer.Attention):
                dora_attn = Dora_Attention(
                    dim=module.qkv.in_features,
                    num_heads=module.num_heads,
                    qkv_bias=module.qkv.bias is not None,
                    qk_norm=isinstance(module.q_norm, nn.LayerNorm),
                    attn_drop=module.attn_drop.p,
                    proj_drop=module.proj_drop.p,
                    dora_r=rank,
                    dora_alpha=alpha,
                )

                copy_weights(module, dora_attn)

                parent_module = dict(model.named_modules())[name.rsplit(".", 1)[0]]
                setattr(parent_module, name.rsplit('.', 1)[-1], dora_attn)

    elif pfm_name == 'conch_v1_5':
        for name, module in model.named_modules():
            if isinstance(module, Conch_V1_5_Attention):
                dora_attn = Dora_Attention(
                    dim=module.qkv.in_features,
                    num_heads=module.num_heads,
                    qkv_bias=module.qkv.bias is not None,
                    qk_norm=isinstance(module.q_norm, nn.LayerNorm),
                    attn_drop=module.attn_drop.p,
                    proj_drop=module.proj_drop.p,
                    dora_r=rank,
                    dora_alpha=alpha,
                )

                copy_weights(module, dora_attn)

                parent_module = dict(model.named_modules())[name.rsplit(".", 1)[0]]
                setattr(parent_module, name.rsplit('.', 1)[-1], dora_attn)

    # Freeze all original parameters
    for param in model.parameters():
        param.requires_grad = False

    # Only train Dora-specific parameters: lora_a, lora_b, and m
    for name, param in model.named_parameters():
        if 'lora_a' in name or 'lora_b' in name or name.endswith('.m'):
            param.requires_grad = True

    return model