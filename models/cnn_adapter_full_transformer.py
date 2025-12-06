"""
CNN Adapter Module for PFM Segmentation

This module implements a CNN adapter based on TransUNet's ResNetV2 architecture.
The CNN adapter extracts multi-scale features before the ViT encoder and provides
skip connections to the decoder for better segmentation performance.

Reference: TransUNet (https://arxiv.org/abs/2102.04306)

Author: @chenwm
"""

import math
import logging
from collections import OrderedDict
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# Patch size mapping for different PFM models
# Models with patch_size=14 need special handling
PFM_PATCH_SIZE = {
    'gigapath': 16,  # config says 16
    'uni_v1': 16,
    'uni_v2': 14,
    'virchow_v1': 14,
    'virchow_v2': 14,
    'conch_v1': 16,
    'conch_v1_5': 16,
    'phikon': 16,
    'phikon_v2': 16,
    'patho3dmatrix-vision': 16,
    'hoptimus_0': 14,
    'hoptimus_1': 14,
    'kaiko-vits8': 8,
    'kaiko-vits16': 16,
    'kaiko-vitb8': 8,
    'kaiko-vitb16': 16,
    'kaiko-vitl14': 14,
    'midnight12k': 14,
    'lunit_vits8': 8,
    'hibou_l': 14,
    'musk': 16,
    'PathOrchestra': 14,
}


class StdConv2d(nn.Conv2d):
    """
    Standard convolution with weight standardization.
    Weight standardization improves training stability and convergence.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin: int, cout: int, stride: int = 1, groups: int = 1, bias: bool = False) -> StdConv2d:
    """3x3 convolution with weight standardization."""
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin: int, cout: int, stride: int = 1, bias: bool = False) -> StdConv2d:
    """1x1 convolution with weight standardization."""
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """
    Pre-activation (v2) bottleneck block.
    
    This block uses pre-activation design where batch normalization and ReLU
    are applied before the convolution layers.
    """

    def __init__(self, cin: int, cout: int = None, cmid: int = None, stride: int = 1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or cin != cout:
            # Projection with pre-activation
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y


class ResNetV2Adapter(nn.Module):
    """
    ResNet V2 Adapter for CNN-based feature extraction before ViT.
    
    This adapter extracts multi-scale features that can be used as skip connections
    in the decoder. It follows the TransUNet architecture design.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB images)
        width_factor (float): Width multiplier for the network (default: 1)
        block_units (Tuple[int, ...]): Number of units in each block (default: (3, 4, 9))
        output_channels (int): Number of output channels to match ViT input
    """

    def __init__(self, in_channels: int = 3, width_factor: float = 1, 
                 block_units: Tuple[int, ...] = (3, 4, 9), output_channels: int = None):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.in_channels = in_channels
        self.output_channels = output_channels

        # Root block: Initial convolution
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
        ]))

        # Body blocks
        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) 
                 for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) 
                 for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) 
                 for i in range(2, block_units[2] + 1)],
            ))),
        ]))
        
        # Output projection to match ViT embedding dimension if specified
        self.output_proj = None
        if output_channels is not None:
            self.output_proj = nn.Sequential(
                nn.Conv2d(width * 16, output_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, output_channels, eps=1e-6),
                nn.ReLU(inplace=True)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through the CNN adapter.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: 
                - Output features for ViT input
                - List of skip connection features [1/8, 1/4, 1/2 scale] (reversed order for decoder)
        """
        features = []
        b, c, in_size, _ = x.size()
        
        # Root block: 1/2 scale
        x = self.root(x)
        features.append(x)  # 1/2 scale, width channels
        
        # MaxPool: 1/4 scale (padding=0 to match original TransUNet)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        
        # Block 1: 1/4 scale
        x = self.body.block1(x)
        # Handle potential size mismatch due to padding=0 MaxPool (following original TransUNet)
        right_size = int(in_size / 4)
        if x.size()[2] != right_size:
            pad = right_size - x.size()[2]
            assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
            feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
            feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
        else:
            feat = x
        features.append(feat)  # 1/4 scale, width*4 channels
        
        # Block 2: 1/8 scale
        x = self.body.block2(x)
        right_size = int(in_size / 8)
        if x.size()[2] != right_size:
            pad = right_size - x.size()[2]
            assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
            feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
            feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
        else:
            feat = x
        features.append(feat)  # 1/8 scale, width*8 channels
        
        # Block 3: 1/16 scale (for ViT input)
        x = self.body.block3(x)
        
        # Project to ViT embedding dimension if specified
        if self.output_proj is not None:
            x = self.output_proj(x)
        
        # Return features in reverse order for decoder (from coarse to fine)
        # features: [1/2, 1/4, 1/8] -> reversed: [1/8, 1/4, 1/2]
        return x, features[::-1]
    
    def get_skip_channels(self) -> List[int]:
        """Get the number of channels for each skip connection."""
        return [self.width * 8, self.width * 4, self.width]


class Conv2dReLU(nn.Sequential):
    """Convolution layer with batch normalization and ReLU activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 padding: int = 0, stride: int = 1, use_batchnorm: bool = True):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlockWithSkip(nn.Module):
    """Decoder block with skip connection support for CNN adapter."""
    
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int = 0,
                 use_batchnorm: bool = True, scale: float = 2):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            # Handle size mismatch between x and skip
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCupWithSkip(nn.Module):
    """
    Decoder with skip connections for CNN adapter integration.
    
    This decoder takes features from the ViT encoder and skip connections
    from the CNN adapter to produce the final segmentation output.
    
    Args:
        emb_dim (int): Embedding dimension from ViT
        decoder_channels (Tuple[int, ...]): Number of channels in each decoder block
        skip_channels (List[int]): Number of channels from CNN adapter skip connections
    """
    
    def __init__(self, emb_dim: int, decoder_channels: Tuple[int, ...] = (256, 128, 64, 16),
                 skip_channels: List[int] = None):
        super().__init__()
        head_channels = 512
        self.decoder_channels = decoder_channels
        
        # Initial convolution to reduce ViT embedding dimension
        self.conv_more = Conv2dReLU(
            emb_dim,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        
        # Default skip channels if not provided
        if skip_channels is None:
            skip_channels = [0, 0, 0, 0]
        else:
            # Pad skip_channels to match decoder depth
            while len(skip_channels) < len(decoder_channels):
                skip_channels.append(0)
        
        # Create decoder blocks with skip connections
        blocks = []
        scales = [2, 2, 2, 2]  # Default scales
            
        for i, (in_ch, out_ch, sk_ch, scale) in enumerate(zip(in_channels, out_channels, skip_channels, scales)):
            blocks.append(DecoderBlockWithSkip(in_ch, out_ch, sk_ch, scale=scale))
        
        self.blocks = nn.ModuleList(blocks)
        self.skip_channels = skip_channels

    def forward(self, hidden_states: torch.Tensor, 
                skip_features: Optional[List[torch.Tensor]] = None,
                feature_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Forward pass through decoder with skip connections.
        
        Args:
            hidden_states (torch.Tensor): Encoded features from transformer (B, n_patch, hidden)
            skip_features (Optional[List[torch.Tensor]]): Skip connection features from CNN adapter
            feature_hw (Optional[Tuple[int, int]]): Height and width of feature grid (h, w). 
                If None, assumes square grid.
            
        Returns:
            torch.Tensor: Decoded feature maps
        """
        B, n_patch, hidden = hidden_states.size()
        
        # Use provided feature_hw or compute from n_patch (assuming square)
        if feature_hw is not None:
            h, w = feature_hw
        else:
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        
        # Reshape from (B, n_patch, hidden) to (B, hidden, h, w)
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        # Apply decoder blocks with skip connections
        for i, decoder_block in enumerate(self.blocks):
            skip = None
            if skip_features is not None and i < len(skip_features) and self.skip_channels[i] > 0:
                skip = skip_features[i]
            x = decoder_block(x, skip=skip)
            
        return x


def _get_position_embeddings(model: nn.Module, pfm_name: str, num_patches: int, emb_dim: int) -> Optional[nn.Parameter]:
    """
    Try to extract position embeddings from PFM model or create new ones.
    
    Following TransUNet's approach, we add position embeddings to CNN tokens
    before feeding them into the transformer blocks.
    
    Args:
        model: PFM segmentation model
        pfm_name: Name of the PFM model
        num_patches: Expected number of patches (H/16 * W/16)
        emb_dim: Embedding dimension
        
    Returns:
        Position embeddings parameter or None if not applicable
    """
    try:
        # Try to get position embeddings from different PFM structures
        pos_embed = None
        
        if pfm_name in ['virchow_v1', 'virchow_v2']:
            if hasattr(model.pfm, 'pos_embed'):
                pos_embed = model.pfm.pos_embed
        elif pfm_name == 'conch_v1':
            if hasattr(model.pfm.visual.trunk, 'pos_embed'):
                pos_embed = model.pfm.visual.trunk.pos_embed
        elif pfm_name == 'conch_v1_5':
            if hasattr(model.pfm.trunk, 'pos_embed'):
                pos_embed = model.pfm.trunk.pos_embed
        elif pfm_name in ['hoptimus_0', 'hoptimus_1', 'uni_v2', 'patho3dmatrix-vision', 'PathOrchestra']:
            if hasattr(model.pfm, 'pos_embed'):
                pos_embed = model.pfm.pos_embed
        elif pfm_name.startswith('kaiko-') or pfm_name == 'lunit_vits8':
            if hasattr(model.pfm, 'pos_embed'):
                pos_embed = model.pfm.pos_embed
        elif pfm_name in ['phikon', 'phikon_v2']:
            if hasattr(model.pfm.vit_model.embeddings, 'position_embeddings'):
                pos_embed = model.pfm.vit_model.embeddings.position_embeddings
        elif pfm_name == 'hibou_l':
            if hasattr(model.pfm.hibou_model.embeddings, 'position_embeddings'):
                pos_embed = model.pfm.hibou_model.embeddings.position_embeddings
        elif pfm_name == 'midnight12k':
            if hasattr(model.pfm.midnight_model.embeddings, 'position_embeddings'):
                pos_embed = model.pfm.midnight_model.embeddings.position_embeddings
        elif pfm_name == 'musk':
            # MUSK uses BEiT3 which has its own position encoding mechanism
            # Position embeddings are handled internally in the encoder
            return None
        else:
            # Default: try to find pos_embed directly
            if hasattr(model.pfm, 'pos_embed'):
                pos_embed = model.pfm.pos_embed
        
        if pos_embed is not None:
            # Check if we need to adjust for CLS token
            # Most ViT models have shape (1, num_patches + 1, emb_dim) where +1 is for CLS token
            if pos_embed.shape[1] > num_patches:
                # Skip CLS token position embedding
                pos_embed = pos_embed[:, 1:num_patches+1, :]
            elif pos_embed.shape[1] < num_patches:
                # Need to interpolate position embeddings
                logger.warning(f"Position embeddings size mismatch: {pos_embed.shape[1]} vs {num_patches}, skipping.")
                return None
            return pos_embed
            
    except Exception as e:
        logger.warning(f"Could not extract position embeddings from {pfm_name}: {e}")
    
    return None


def equip_model_with_cnn_adapter(model: nn.Module, cnn_config: dict) -> nn.Module:
    """
    Equip a PFMSegmentationModel with CNN adapter following TransUNet-like architecture.
    
    Data flow: Image -> CNN -> CNN output + pos_embed -> PFM (ViT transformer) -> Decoder (with skip) -> Segmentation
    
    The CNN processes the image first, outputs feature map (B, emb_dim, H/16, W/16),
    which is then converted to token format, position embeddings are added (following TransUNet),
    and fed into PFM's transformer blocks (bypassing patch embedding). 
    Skip connections from CNN are used in decoder.
    
    Args:
        model (nn.Module): PFMSegmentationModel instance
        cnn_config (dict): CNN adapter configuration with keys:
            - width_factor (float): Width multiplier for the network (default: 1)
            - block_units (list): Number of units in each block (default: [3, 4, 9])
            
    Returns:
        nn.Module: Model equipped with CNN adapter
    """
    # Get CNN adapter config
    width_factor = cnn_config.get('width_factor', 1)
    block_units = tuple(cnn_config.get('block_units', [3, 4, 9]))
    
    # Determine model properties
    PFM_name = model.PFM_name
    emb_dim = model.decoder.conv_more[0].in_channels  # Get emb_dim from existing decoder
    decoder_channels = model.decoder_channels
    
    # Create CNN adapter with output projection to ViT embedding dimension
    cnn_adapter = ResNetV2Adapter(
        in_channels=3,
        width_factor=width_factor,
        block_units=block_units,
        output_channels=emb_dim  # Project CNN output to match ViT embedding dimension
    )
    
    # Get skip channels from CNN adapter
    skip_channels = cnn_adapter.get_skip_channels()
    
    # NOTE: CNN adapter always produces 1/16 scale features (due to ResNet's fixed downsampling pattern:
    # root stride 2 + maxpool stride 2 + block2 stride 2 + block3 stride 2 = 16x total)
    # Therefore, we always need 16x upsampling in the decoder, regardless of ViT's patch_size.
    # Do NOT use special_scale or is_lunit here - those are only for when ViT's patch embedding
    # produces non-1/16 scale features (e.g., patch_size=14 produces 1/14 scale).
    
    # Create new decoder with skip connections
    new_decoder = DecoderCupWithSkip(
        emb_dim=emb_dim,
        decoder_channels=decoder_channels,
        skip_channels=skip_channels,
    )
    
    # Add CNN adapter to model
    model.cnn_adapter = cnn_adapter
    
    # Replace decoder with new decoder that supports skip connections
    model.decoder = new_decoder
    
    # Create learnable position embeddings (following TransUNet design)
    # Default to 224x224 input -> 14x14 patches for 1/16 scale
    # This will be resized during forward if input size differs
    default_num_patches = 14 * 14  # 224 / 16 = 14
    
    # Try to get position embeddings from PFM model, or create new learnable ones
    pos_embed = _get_position_embeddings(model, PFM_name, default_num_patches, emb_dim)
    if pos_embed is not None:
        # Use extracted position embeddings (frozen by default, user can unfreeze if needed)
        model.register_buffer('cnn_position_embeddings', pos_embed.clone().detach())
        model.position_embeddings_from_pfm = True
        logger.info(f"Using position embeddings from {PFM_name} (shape: {pos_embed.shape})")
    else:
        # Create new learnable position embeddings
        model.cnn_position_embeddings = nn.Parameter(torch.zeros(1, default_num_patches, emb_dim))
        nn.init.trunc_normal_(model.cnn_position_embeddings, std=0.02)
        model.position_embeddings_from_pfm = False
        logger.info(f"Created new learnable position embeddings (shape: {model.cnn_position_embeddings.shape})")
    
    # Dropout after adding position embeddings (following TransUNet)
    model.cnn_pos_drop = nn.Dropout(p=0.1)
    
    # Define new forward method that uses CNN adapter
    # Following the same PFM structure handling as pfm_seg_models.py
    def forward_with_cnn_adapter(self, x: torch.Tensor) -> dict:
        # Handle single channel images
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Step 1: CNN processes image first
        # cnn_features: (B, emb_dim, H/16, W/16), skip_features: list of multi-scale features
        cnn_features, skip_features = self.cnn_adapter(x)
        
        # Step 2: Convert CNN features to token format for ViT
        # (B, emb_dim, H/16, W/16) -> (B, H/16*W/16, emb_dim)
        B, C, H, W = cnn_features.shape
        cnn_tokens = cnn_features.flatten(2).permute(0, 2, 1)  # (B, N, emb_dim)
        
        # Step 2.5: Add position embeddings (following TransUNet design)
        # This is critical for transformer to understand spatial relationships
        num_patches = H * W
        pos_embed = self.cnn_position_embeddings
        
        # Handle size mismatch by interpolating position embeddings
        if pos_embed.shape[1] != num_patches:
            # Resize position embeddings to match current input size
            pos_embed = pos_embed.permute(0, 2, 1)  # (1, emb_dim, N_original)
            pos_embed = pos_embed.reshape(1, -1, int(np.sqrt(pos_embed.shape[2])), int(np.sqrt(pos_embed.shape[2])))
            pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.flatten(2).permute(0, 2, 1)  # (1, N_new, emb_dim)
        
        cnn_tokens = cnn_tokens + pos_embed
        cnn_tokens = self.cnn_pos_drop(cnn_tokens)
        
        # Step 3: Feed CNN tokens into PFM's transformer blocks
        # Different PFM structures require different access patterns (same as pfm_seg_models.py)
        if self.PFM_name == 'virchow_v1':
            # Virchow v1: direct blocks and norm access
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        elif self.PFM_name == 'virchow_v2':
            # Virchow v2: direct blocks and norm access
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        elif self.PFM_name == 'conch_v1':
            # CONCH v1: access through visual.trunk
            features = self.pfm.visual.trunk.blocks(cnn_tokens)
            features = self.pfm.visual.trunk.norm(features)
        elif self.PFM_name == 'conch_v1_5':
            # CONCH v1.5: access through trunk
            features = self.pfm.trunk.blocks(cnn_tokens)
            features = self.pfm.trunk.norm(features)
        elif self.PFM_name == 'phikon':
            # Phikon: transformers ViTModel wrapper, access encoder layers
            features = self.pfm.vit_model.encoder(cnn_tokens)[0]
            features = self.pfm.vit_model.layernorm(features)
        elif self.PFM_name == 'phikon_v2':
            # Phikon v2: similar to phikon
            features = self.pfm.vit_model.encoder(cnn_tokens)[0]
            features = self.pfm.vit_model.layernorm(features)
        elif self.PFM_name == 'hibou_l':
            # Hibou-L: transformers AutoModel wrapper
            features = self.pfm.hibou_model.encoder(cnn_tokens)[0]
            features = self.pfm.hibou_model.layernorm(features)
        elif self.PFM_name == 'musk':
            # MUSK: BEiT3-based model, access transformer through beit3
            # BEiT3's encoder handles position embeddings internally via token_embeddings parameter
            outputs = self.pfm.musk_model.beit3.encoder(
                src_tokens=None,
                token_embeddings=cnn_tokens
            )
            features = outputs["encoder_out"]
        elif self.PFM_name == 'lunit_vits8':
            # Lunit: standard timm ViT
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        elif self.PFM_name == 'midnight12k':
            # Midnight-12k: transformers AutoModel wrapper
            features = self.pfm.midnight_model.encoder(cnn_tokens)[0]
            features = self.pfm.midnight_model.layernorm(features)
        elif self.PFM_name.startswith('kaiko-'):
            # Kaiko models: standard timm ViT
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        elif self.PFM_name == 'hoptimus_0' or self.PFM_name == 'hoptimus_1':
            # H-Optimus: standard timm ViT
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        elif self.PFM_name == 'patho3dmatrix-vision':
            # Patho3DMatrix: standard timm ViT
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        elif self.PFM_name == 'uni_v2':
            # UNI v2: standard timm ViT
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        elif self.PFM_name == 'PathOrchestra':
            # PathOrchestra: standard timm ViT
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        else:
            # Default: assume standard timm ViT structure
            features = self.pfm.blocks(cnn_tokens)
            features = self.pfm.norm(features)
        
        # features: (B, N, emb_dim)
        
        # Step 4: Decode features with CNN skip connections
        decoded_features = self.decoder(features, skip_features)
        
        # Step 5: Generate final predictions
        logits = self.segmentation_head(decoded_features)
        
        return {'out': logits}
    
    # Bind the new forward method to the model
    import types
    model.forward = types.MethodType(forward_with_cnn_adapter, model)
    
    return model

