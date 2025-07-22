import math
import timm
from timm.layers import use_fused_attn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import repeat
from .build_conch_v1_5 import Conch_V1_5_Attention



class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, r=4, lora_alpha=1.0):
        super().__init__()
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.r = r
        self.lora_alpha = lora_alpha
        # original linear layer
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        
        # LoRA: low-rank adaptor
        self.lora_a = nn.Parameter(torch.zeros(in_features, r), requires_grad=True)
        self.lora_b = nn.Parameter(torch.zeros(r, out_features), requires_grad=True)
        self.scale = lora_alpha

        # initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
        
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
    
    def forward(self, x):  # shape [10000, 197, 1024]
        # compute original output
        ori_output = F.linear(x, self.weight, self.bias)
        lora_output = ((x @ self.lora_a) @ self.lora_b) * self.scale
        return ori_output + lora_output


class LoRA_Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            lora_r: int = 16,
            lora_alpha: float = 1.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = LoRALinear(dim, dim * 3, bias=qkv_bias, r=lora_r, lora_alpha=lora_alpha)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LoRALinear(dim, dim, r=lora_r, lora_alpha=lora_alpha)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 
                0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    



def equip_model_with_lora(pfm_name, model, rank, alpha):
    """
    Equip a PFM model with LoRA by replacing its attention layers with LoRA_Attention layers.
    This version also copies original attention weights into the LoRA-Attention module.

    Args:
        pfm_name (str): Name of the PFM model.
        model (nn.Module): The PFM model to be equipped with LoRA.
        rank (int): Rank of the low-rank adaptation.
        alpha (float): Scaling factor for the LoRA output.

    Returns:
        nn.Module: The PFM model with LoRA applied to its attention layers.
    """
    def copy_weights(src_attn, dst_lora_attn):
        with torch.no_grad():
            dst_lora_attn.qkv.weight.copy_(src_attn.qkv.weight)
            if src_attn.qkv.bias is not None:
                dst_lora_attn.qkv.bias.copy_(src_attn.qkv.bias)

            dst_lora_attn.proj.weight.copy_(src_attn.proj.weight)
            if src_attn.proj.bias is not None:
                dst_lora_attn.proj.bias.copy_(src_attn.proj.bias)

            if hasattr(src_attn, 'q_norm') and hasattr(dst_lora_attn, 'q_norm') and isinstance(dst_lora_attn.q_norm, nn.LayerNorm):
                dst_lora_attn.q_norm.load_state_dict(src_attn.q_norm.state_dict())
            if hasattr(src_attn, 'k_norm') and hasattr(dst_lora_attn, 'k_norm') and isinstance(dst_lora_attn.k_norm, nn.LayerNorm):
                dst_lora_attn.k_norm.load_state_dict(src_attn.k_norm.state_dict())

    if pfm_name in ['uni_v1', 'uni_v2', 'virchow_v2', 'gigapath']:
        for name, module in model.named_modules():
            if isinstance(module, timm.models.vision_transformer.Attention):
                lora_attn = LoRA_Attention(
                    dim=module.qkv.in_features,
                    num_heads=module.num_heads,
                    qkv_bias=module.qkv.bias is not None,
                    qk_norm=isinstance(module.q_norm, nn.LayerNorm),
                    attn_drop=module.attn_drop.p,
                    proj_drop=module.proj_drop.p,
                    lora_r=rank,
                    lora_alpha=alpha,
                )

                copy_weights(module, lora_attn)

                parent_module = dict(model.named_modules())[name.rsplit(".", 1)[0]]
                setattr(parent_module, name.rsplit('.', 1)[-1], lora_attn)

    elif pfm_name == 'conch_v1_5':
        for name, module in model.named_modules():
            if isinstance(module, Conch_V1_5_Attention):
                lora_attn = LoRA_Attention(
                    dim=module.qkv.in_features,
                    num_heads=module.num_heads,
                    qkv_bias=module.qkv.bias is not None,
                    qk_norm=isinstance(module.q_norm, nn.LayerNorm),
                    attn_drop=module.attn_drop.p,
                    proj_drop=module.proj_drop.p,
                    lora_r=rank,
                    lora_alpha=alpha,
                )

                copy_weights(module, lora_attn)

                parent_module = dict(model.named_modules())[name.rsplit(".", 1)[0]]
                setattr(parent_module, name.rsplit('.', 1)[-1], lora_attn)

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if 'lora_a' in name or 'lora_b' in name:
            param.requires_grad = True

    return model
