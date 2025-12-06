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
    Equip a PFMSegmentationModel with CNN adapter for skip connections only.
    
    Data flow: 
    - Path 1: Image -> PFM (ViT transformer) -> encoded features
    - Path 2: Image -> CNN -> skip features (parallel)
    - Decoder: PFM features + CNN skip connections -> Segmentation
    
    The CNN processes the image in parallel with PFM to extract multi-scale features
    for skip connections. The image is directly fed into PFM using its own patch embedding.
    
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
    
    # Create CNN adapter WITHOUT output projection (only for skip connections)
    cnn_adapter = ResNetV2Adapter(
        in_channels=3,
        width_factor=width_factor,
        block_units=block_units,
        output_channels=None  # No projection needed, only skip connections
    )
    
    # Get skip channels from CNN adapter
    skip_channels = cnn_adapter.get_skip_channels()
    
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
    
    # Store original forward method
    model._original_forward = model.forward
    
    # Define encoding method that extracts features from PFM
    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image using PFM's patch embedding and transformer.
        Returns encoded features ready for decoder.
        Follows the same logic as PFMSegmentationModel.forward()
        """
        # Extract features from pathology foundation model
        # This code mirrors pfm_seg_models.py forward() logic
        if self.PFM_name == 'virchow_v1':
            # Skip CLS token and use patch tokens
            features = self.pfm(x)[:, 1:, :]
        elif self.PFM_name == 'virchow_v2':
            # Skip first 5 tokens (CLS and register tokens)
            features = self.pfm(x)[:, 5:, :]
        elif self.PFM_name == 'conch_v1':
            # Skip CLS token and get features before projection head
            features = self.pfm.visual.trunk.forward_features(x)[:, 1:, :]
        elif self.PFM_name == 'conch_v1_5':
            # Skip CLS token
            features = self.pfm.trunk.forward_features(x)[:, 1:, :]
        elif self.PFM_name == 'phikon' or self.PFM_name == 'phikon_v2':
            # Phikon / Phikon-v2 use transformers ViT-like models - skip CLS token
            features = self.pfm(x)[:, 1:, :]
        elif self.PFM_name == 'hibou_l':
            # Hibou-Large: transformers AutoModel wrapper - Skip first 5 tokens (CLS and register tokens)
            features = self.pfm(x)[:, 5:, :]
        elif self.PFM_name == 'musk':
            # MUSK: wrapper returns all tokens (including CLS token)
            # Skip CLS token to get patch tokens only
            features = self.pfm.forward(x)[:, 1:, :]
        elif self.PFM_name == 'lunit_vits8':
            # Lunit-S8: standard ViT - skip CLS token
            features = self.pfm.forward_features(x)[:, 1:, :]
        elif self.PFM_name == 'midnight12k':
            # Midnight-12k: use forward_features to get full token sequence, skip CLS token
            features = self.pfm.forward_features(x)[:, 1:, :]
        elif self.PFM_name.startswith('kaiko-'):
            # Kaiko models (vits8, vits16, vitb8, vitb16, vitl14): standard ViT - skip CLS token
            features = self.pfm.forward_features(x)[:, 5:, :]
        elif self.PFM_name == 'hoptimus_0' or self.PFM_name == 'hoptimus_1':
            # H-Optimus-0/1: ViT-Giant models - skip CLS token, keep patch tokens
            features = self.pfm.forward_features(x)[:, 5:, :]
        elif self.PFM_name == 'patho3dmatrix-vision':
            # Skip CLS token - standard ViT with forward_features
            features = self.pfm.forward_features(x)[:, 1:, :]
        elif self.PFM_name == 'uni_v2':
            # Skip first 9 tokens (CLS and register tokens)
            features = self.pfm.forward_features(x)[:, 9:, :]
        elif self.PFM_name == 'PathOrchestra':
            # PathOrchestra: skip CLS token, keep patch tokens
            features = self.pfm.forward_features(x)[:, 1:, :]
        else:
            # Standard ViT - skip CLS token
            features = self.pfm.forward_features(x)[:, 1:, :]
        
        return features
    
    # Define new forward method that uses CNN adapter for skip connections only
    def forward_with_cnn_adapter(self, x: torch.Tensor) -> dict:
        # Handle single channel images
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Step 1: Extract skip connection features from CNN (parallel path)
        # cnn_features: (B, width*16, H/16, W/16) - not used for PFM input
        # skip_features: list of multi-scale features [1/8, 1/4, 1/2 scale]
        _, skip_features = self.cnn_adapter(x)
        
        # Step 2: Feed image directly into PFM using original forward method
        # This uses PFM's own patch embedding and transformer
        # We need to extract the encoded features before decoder
        # Call the original PFM encoding (before decoder)
        features = self._encode_image(x)
        
        # Step 3: Decode features with CNN skip connections
        decoded_features = self.decoder(features, skip_features)
        
        # Step 4: Generate final predictions
        logits = self.segmentation_head(decoded_features)
        
        return {'out': logits}
    
    # Bind the new methods to the model
    import types
    model._encode_image = types.MethodType(_encode_image, model)
    model.forward = types.MethodType(forward_with_cnn_adapter, model)
    
    logger.info(f"Equipped {PFM_name} with CNN adapter (skip connections only)")
    logger.info(f"  CNN width factor: {width_factor}")
    logger.info(f"  CNN block units: {block_units}")
    logger.info(f"  Skip channels: {skip_channels}")
    
    return model

