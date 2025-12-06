"""
Transformer Adapter Module for PFM Segmentation

This module implements a Transformer Adapter inspired by DINOv2's architecture.
The Transformer Adapter adds extra Vision Blocks (Transformer layers) after 
the frozen ViT encoder to enable efficient fine-tuning for segmentation tasks.

The strategy:
1. Freeze the original PFM (ViT encoder) parameters
2. Add trainable Vision Blocks after the ViT encoder
3. Train only: Vision Blocks + Decoder + Segmentation Head

Reference: DINOv2 Vision Transformer architecture

Author: @chenwm
"""

import math
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import use_fused_attn, DropPath, Mlp


class VisionBlockAttention(nn.Module):
    """
    Multi-head Self-Attention for Vision Block.
    
    Follows the standard ViT attention mechanism with optional QK normalization.
    
    Args:
        dim (int): Input dimension
        num_heads (int): Number of attention heads
        qkv_bias (bool): Whether to use bias in QKV projection
        qk_norm (bool): Whether to apply layer norm to Q and K
        attn_drop (float): Dropout rate for attention weights
        proj_drop (float): Dropout rate for output projection
        norm_layer (nn.Module): Normalization layer class
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
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


class VisionBlock(nn.Module):
    """
    Vision Block (Transformer Block) for adapter.
    
    Standard Transformer block with:
    - Pre-normalization
    - Multi-head Self-Attention
    - MLP (Feed-Forward Network)
    - Residual connections
    - Optional DropPath for regularization
    
    This follows the DINOv2/ViT-style architecture.
    
    Args:
        dim (int): Input/output dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        qkv_bias (bool): Whether to use bias in QKV projection
        qk_norm (bool): Whether to apply layer norm to Q and K
        drop (float): Dropout rate
        attn_drop (float): Attention dropout rate
        drop_path (float): Stochastic depth rate
        init_values (float): Initial value for layer scale (None to disable)
        act_layer (nn.Module): Activation layer class
        norm_layer (nn.Module): Normalization layer class
        mlp_layer (nn.Module): MLP layer class
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        init_values: Optional[float] = None,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        
        # Pre-normalization
        self.norm1 = norm_layer(dim)
        
        # Self-attention
        self.attn = VisionBlockAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        
        # Layer scale for attention
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        
        # Drop path for stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Pre-normalization for MLP
        self.norm2 = norm_layer(dim)
        
        # MLP (Feed-Forward Network)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        
        # Layer scale for MLP
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        
        # Drop path for stochastic depth
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # MLP with residual
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class LayerScale(nn.Module):
    """
    Layer Scale module for improved training stability.
    
    Reference: "Going deeper with Image Transformers" (CaiT paper)
    
    Args:
        dim (int): Dimension of the input
        init_values (float): Initial value for the scale parameters
    """
    
    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class TransformerAdapter(nn.Module):
    """
    Transformer Adapter: A stack of Vision Blocks added after the frozen ViT encoder.
    
    This adapter processes the features from the frozen PFM encoder and learns
    task-specific representations through additional transformer layers.
    
    Args:
        dim (int): Input/output embedding dimension (must match PFM output)
        depth (int): Number of Vision Blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim
        qkv_bias (bool): Whether to use bias in QKV projection
        qk_norm (bool): Whether to apply layer norm to Q and K
        drop_rate (float): Dropout rate
        attn_drop_rate (float): Attention dropout rate
        drop_path_rate (float): Stochastic depth rate
        init_values (float): Initial value for layer scale
        act_layer (nn.Module): Activation layer class
        norm_layer (nn.Module): Normalization layer class
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        init_values: Optional[float] = 1e-5,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Stack of Vision Blocks
        self.blocks = nn.ModuleList([
            VisionBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                init_values=init_values,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        
        # Final normalization
        self.norm = norm_layer(dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights following ViT conventions."""
        if isinstance(m, nn.Linear):
            # Use truncated normal initialization
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer Adapter.
        
        Args:
            x (torch.Tensor): Input features from PFM encoder, shape (B, N, dim)
            
        Returns:
            torch.Tensor: Adapted features, shape (B, N, dim)
        """
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x


# Number of tokens to skip for each PFM model (CLS token + register tokens)
# This is used to extract patch tokens after vision blocks processing
PFM_SKIP_TOKENS = {
    'gigapath': 1,           # CLS only
    'uni_v1': 1,             # CLS only
    'uni_v2': 9,             # CLS + 8 register tokens
    'virchow_v1': 1,         # CLS only
    'virchow_v2': 5,         # CLS + 4 register tokens
    'conch_v1': 1,           # CLS only
    'conch_v1_5': 1,         # CLS only
    'phikon': 1,             # CLS only
    'phikon_v2': 1,          # CLS only
    'patho3dmatrix-vision': 1,  # CLS only
    'hoptimus_0': 5,         # CLS + 4 register tokens
    'hoptimus_1': 5,         # CLS + 4 register tokens
    'kaiko-vits8': 5,        # CLS + 4 register tokens
    'kaiko-vits16': 5,       # CLS + 4 register tokens
    'kaiko-vitb8': 5,        # CLS + 4 register tokens
    'kaiko-vitb16': 5,       # CLS + 4 register tokens
    'kaiko-vitl14': 5,       # CLS + 4 register tokens
    'midnight12k': 1,        # CLS only
    'lunit_vits8': 1,        # CLS only
    'hibou_l': 5,            # CLS + 4 register tokens
    'musk': 1,               # CLS only
    'PathOrchestra': 1,      # CLS only
}


def get_skip_tokens(pfm_name: str) -> int:
    """
    Get the number of tokens to skip (CLS + register tokens) for a specific PFM model.
    
    Args:
        pfm_name (str): Name of the PFM model
        
    Returns:
        int: Number of tokens to skip at the beginning of the sequence
    """
    if pfm_name in PFM_SKIP_TOKENS:
        return PFM_SKIP_TOKENS[pfm_name]
    else:
        # Default: skip only CLS token
        return 1


def get_num_heads_from_dim(emb_dim: int) -> int:
    """
    Infer the number of attention heads from embedding dimension.
    
    Common ViT configurations:
    - dim=384: 6 heads (head_dim=64)
    - dim=768: 12 heads (head_dim=64)
    - dim=1024: 16 heads (head_dim=64)
    - dim=1280: 16 heads (head_dim=80) or 20 heads (head_dim=64)
    - dim=1536: 24 heads (head_dim=64)
    
    Args:
        emb_dim (int): Embedding dimension
        
    Returns:
        int: Number of attention heads
    """
    # Standard head_dim is 64 for most ViT models
    head_dim = 64
    num_heads = emb_dim // head_dim
    
    # Ensure num_heads is valid (at least 1 and divides evenly)
    if emb_dim % head_dim != 0:
        # Try common head dimensions
        for hd in [64, 80, 96, 128]:
            if emb_dim % hd == 0:
                num_heads = emb_dim // hd
                break
    
    return max(1, num_heads)


def equip_model_with_transformer_adapter(
    model: nn.Module, 
    adapter_config: Dict[str, Any]
) -> nn.Module:
    """
    Equip a PFMSegmentationModel with Transformer Adapter.
    
    Data flow: Image -> PFM (frozen, full output with CLS) -> Transformer Adapter (trainable) 
               -> Extract patch tokens -> Decoder (trainable) -> Segmentation Head (trainable)
    
    The Transformer Adapter adds extra Vision Blocks after the frozen PFM encoder.
    All PFM output tokens (including CLS and register tokens) are passed to Vision Blocks,
    then patch tokens are extracted before the decoder.
    
    Args:
        model (nn.Module): PFMSegmentationModel instance
        adapter_config (dict): Transformer adapter configuration with keys:
            - depth (int): Number of Vision Blocks (default: 4)
            - num_heads (int): Number of attention heads (default: auto-inferred from emb_dim)
            - mlp_ratio (float): MLP hidden dim ratio (default: 4.0)
            - drop_rate (float): Dropout rate (default: 0.0)
            - attn_drop_rate (float): Attention dropout rate (default: 0.0)
            - drop_path_rate (float): Stochastic depth rate (default: 0.1)
            - init_values (float): Layer scale init value (default: 1e-5)
            - qk_norm (bool): Whether to use QK normalization (default: False)
            
    Returns:
        nn.Module: Model equipped with Transformer Adapter
    """
    import types
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Get PFM model properties - read directly from existing model
    PFM_name = model.PFM_name
    emb_dim = model.decoder.conv_more[0].in_channels  # Get emb_dim from existing decoder
    
    # Get number of tokens to skip (CLS + register tokens)
    skip_tokens = get_skip_tokens(PFM_name)
    
    # Get adapter config
    depth = adapter_config.get('depth', 4)
    mlp_ratio = adapter_config.get('mlp_ratio', 4.0)
    drop_rate = adapter_config.get('drop_rate', 0.0)
    attn_drop_rate = adapter_config.get('attn_drop_rate', 0.0)
    drop_path_rate = adapter_config.get('drop_path_rate', 0.1)
    init_values = adapter_config.get('init_values', 1e-5)
    qk_norm = adapter_config.get('qk_norm', False)
    
    # Get num_heads: from config if provided, otherwise infer from emb_dim
    num_heads = adapter_config.get('num_heads', None)
    if num_heads is None:
        num_heads = get_num_heads_from_dim(emb_dim)
    
    # Create Transformer Adapter
    transformer_adapter = TransformerAdapter(
        dim=emb_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        qk_norm=qk_norm,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        init_values=init_values,
    )
    
    # Add transformer adapter to model
    model.transformer_adapter = transformer_adapter
    
    # Store skip_tokens for use in forward method
    model._transformer_adapter_skip_tokens = skip_tokens
    
    logger.info(f"Transformer Adapter created: depth={depth}, num_heads={num_heads}, "
                f"dim={emb_dim}, mlp_ratio={mlp_ratio}, skip_tokens={skip_tokens}")
    
    # Define new forward method that uses transformer adapter
    def forward_with_transformer_adapter(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Transformer Adapter.
        
        Flow: Image -> PFM (frozen, full output) -> Transformer Adapter (all tokens) 
              -> Extract patch tokens -> Decoder -> Seg Head
        
        Key difference from original: PFM outputs ALL tokens (including CLS and register tokens)
        to Vision Blocks, then we extract patch tokens after Vision Blocks processing.
        """
        # Handle single channel images
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Step 1: Extract FULL features from frozen PFM (including CLS and register tokens)
        # Different PFM models have different output structures
        if self.PFM_name == 'virchow_v1':
            # Virchow v1: returns all tokens including CLS
            features = self.pfm(x)  # (B, N+1, dim) where N is num_patches
        elif self.PFM_name == 'virchow_v2':
            # Virchow v2: returns all tokens including CLS + register tokens
            features = self.pfm(x)  # (B, N+5, dim)
        elif self.PFM_name == 'conch_v1':
            # CONCH v1: access through visual.trunk
            features = self.pfm.visual.trunk.forward_features(x)  # (B, N+1, dim)
        elif self.PFM_name == 'conch_v1_5':
            # CONCH v1.5: access through trunk
            features = self.pfm.trunk.forward_features(x)  # (B, N+1, dim)
        elif self.PFM_name == 'phikon' or self.PFM_name == 'phikon_v2':
            # Phikon: transformers ViTModel wrapper
            features = self.pfm(x)  # (B, N+1, dim)
        elif self.PFM_name == 'hibou_l':
            # Hibou-L: transformers AutoModel wrapper
            features = self.pfm(x)  # (B, N+5, dim)
        elif self.PFM_name == 'musk':
            # MUSK: wrapper returns all tokens
            features = self.pfm.forward(x)  # (B, N+1, dim)
        elif self.PFM_name == 'lunit_vits8':
            # Lunit: standard timm ViT
            features = self.pfm.forward_features(x)  # (B, N+1, dim)
        elif self.PFM_name == 'midnight12k':
            # Midnight-12k: transformers AutoModel wrapper
            features = self.pfm.forward_features(x)  # (B, N+1, dim)
        elif self.PFM_name.startswith('kaiko-'):
            # Kaiko models: standard timm ViT with register tokens
            features = self.pfm.forward_features(x)  # (B, N+5, dim)
        elif self.PFM_name == 'hoptimus_0' or self.PFM_name == 'hoptimus_1':
            # H-Optimus: standard timm ViT with register tokens
            features = self.pfm.forward_features(x)  # (B, N+5, dim)
        elif self.PFM_name == 'patho3dmatrix-vision':
            # Patho3DMatrix: standard timm ViT
            features = self.pfm.forward_features(x)  # (B, N+1, dim)
        elif self.PFM_name == 'uni_v2':
            # UNI v2: standard timm ViT with register tokens
            features = self.pfm.forward_features(x)  # (B, N+9, dim)
        elif self.PFM_name == 'PathOrchestra':
            # PathOrchestra: standard timm ViT
            features = self.pfm.forward_features(x)  # (B, N+1, dim)
        else:
            # Default: assume standard timm ViT structure
            features = self.pfm.forward_features(x)  # (B, N+1, dim)
        
        # Step 2: Apply Transformer Adapter to ALL tokens (including CLS and register tokens)
        features = self.transformer_adapter(features)
        
        # Step 3: Extract patch tokens (skip CLS and register tokens)
        skip_tokens = self._transformer_adapter_skip_tokens
        patch_features = features[:, skip_tokens:, :]  # (B, N, dim)
        
        # Step 4: Decode features (only patch tokens)
        decoded_features = self.decoder(patch_features)
        
        # Step 5: Generate final predictions
        logits = self.segmentation_head(decoded_features)
        
        return {'out': logits}
    
    # Bind the new forward method to the model
    model.forward = types.MethodType(forward_with_transformer_adapter, model)
    
    return model

