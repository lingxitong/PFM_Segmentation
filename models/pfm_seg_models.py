"""
Pathology Foundation Models (PFM) for Semantic Segmentation

This module integrates multiple pathology foundation models including
Gigapath, UNI v1/v2, Virchow v2, and Conch V1.5 for segmentation tasks.

Author: @Toby
Function: Segmentation models using PFMs (pathology foundation models)
"""

import copy
import logging
import math
from os.path import join as pjoin
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from typing import Optional, Dict, Any, Tuple
from .lora import equip_model_with_lora

logger = logging.getLogger(__name__)

# Vision Transformer component names for loading pretrained weights
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def get_PFM_model(PFM_name: str, PFM_weights_path: str) -> nn.Module:
    """
    Load and configure a Pathology Foundation Model.
    
    Args:
        PFM_name (str): Name of the PFM model
        PFM_weights_path (str): Path to model weights
        
    Returns:
        nn.Module: Configured PFM model
    """
    if PFM_name == 'gigapath':
        gig_config = {
            "architecture": "vit_giant_patch14_dinov2",
            "num_classes": 0,
            "num_features": 1536,
            "global_pool": "token",
            "model_args": {
                "img_size": 224,
                "in_chans": 3,
                "patch_size": 16,
                "embed_dim": 1536,
                "depth": 40,
                "num_heads": 24,
                "init_values": 1e-05,
                "mlp_ratio": 5.33334,
                "num_classes": 0,
                "dynamic_img_size": True
            }
        } 
        model = timm.create_model("vit_giant_patch14_dinov2", pretrained=False, **gig_config['model_args'])
        state_dict = torch.load(PFM_weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        
    elif PFM_name == 'uni_v1':
        model = timm.create_model(
            "vit_large_patch16_224", 
            img_size=224, 
            patch_size=16, 
            init_values=1e-5, 
            num_classes=0, 
            dynamic_img_size=True
        )
        model.load_state_dict(torch.load(PFM_weights_path, map_location='cpu', weights_only=True), strict=True)
        
    elif PFM_name == 'virchow_v2':
        from timm.layers import SwiGLUPacked
        virchow_v2_config = {
            "img_size": 224,
            "init_values": 1e-5,
            "num_classes": 0,
            "mlp_ratio": 5.3375,
            "reg_tokens": 4,
            "global_pool": "",
            "dynamic_img_size": True
        }
        model = timm.create_model(
            "vit_huge_patch14_224", 
            pretrained=False,
            mlp_layer=SwiGLUPacked, 
            act_layer=torch.nn.SiLU,
            **virchow_v2_config
        )
        state_dict = torch.load(PFM_weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        
    elif PFM_name == 'conch_v1_5':
        try:
            from .conch_v1_5_config import ConchConfig
            from .build_conch_v1_5 import build_conch_v1_5
            conch_v1_5_config = ConchConfig()
            model = build_conch_v1_5(conch_v1_5_config, PFM_weights_path)
        except ImportError:
            raise ImportError("Conch V1.5 dependencies not found.")
            
    elif PFM_name == 'uni_v2':
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model('vit_giant_patch14_224', pretrained=False, **timm_kwargs)
        state_dict = torch.load(PFM_weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        
    else:
        raise ValueError(f"Unsupported PFM model: {PFM_name}")
            
    return model


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


class DecoderBlock(nn.Module):
    """Decoder block for upsampling and feature fusion."""
    
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
        # self.up = nn.Upsample(scale_factor=scale, mode='nearest')



    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    """Segmentation head for final prediction."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, upsampling: int = 1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # upsampling = nn.Upsample(scale_factor=upsampling, mode='nearest') if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    """Decoder network for feature reconstruction and upsampling."""
    
    def __init__(self, emb_dim: int, decoder_channels: Tuple[int, ...], is_virchow_v2_or_uni_v2: bool = False):
        super().__init__()
        head_channels = 512
        self.decoder_channels = decoder_channels
        
        self.conv_more = Conv2dReLU(
            emb_dim,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = [0, 0, 0, 0]  # No skip connections in current implementation

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) 
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        
        # Special handling for Virchow v2 model
        if is_virchow_v2_or_uni_v2:
            blocks[-1] = DecoderBlock(in_channels[-1], out_channels[-1], skip_channels[-1], scale=1.75)
            
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states: torch.Tensor, features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            hidden_states (torch.Tensor): Encoded features from transformer (B, n_patch, hidden)
            features (Optional[torch.Tensor]): Skip connection features (not used currently)
            
        Returns:
            torch.Tensor: Decoded feature maps
        """
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        
        # Reshape from (B, n_patch, hidden) to (B, hidden, h, w)
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        for decoder_block in self.blocks:
            x = decoder_block(x, skip=None)
            
        return x


class PFMSegmentationModel(nn.Module):
    """
    Pathology Foundation Model for Semantic Segmentation.
    
    This model integrates various pathology foundation models with a decoder
    network for pixel-level segmentation tasks.
    
    Args:
        PFM_name (str): Name of the pathology foundation model
        PFM_weights_path (str): Path to pretrained weights
        emb_dim (int): Embedding dimension of the PFM
        num_classes (int): Number of segmentation classes
    """
    
    def __init__(self, PFM_name: str, PFM_weights_path: str, emb_dim: int,  num_classes: int = 2):
        super(PFMSegmentationModel, self).__init__()
        
        self.num_classes = num_classes
        self.PFM_name = PFM_name
        self.decoder_channels = (256, 128, 64, 16)
        
        # Create decoder
        if PFM_name == 'virchow_v2' or PFM_name == 'uni_v2':
            self.decoder = DecoderCup(emb_dim, self.decoder_channels, is_virchow_v2_or_uni_v2 = True)
        else:
            self.decoder = DecoderCup(emb_dim, self.decoder_channels)
            
        # Create segmentation head
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
        )
        
        # Load pathology foundation model
        self.pfm = get_PFM_model(PFM_name, PFM_weights_path)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing output predictions
        """
        # Handle single channel images by repeating to 3 channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Extract features from pathology foundation model
        if self.PFM_name == 'virchow_v2':
            # Skip first 5 tokens (CLS and register tokens)
            features = self.pfm(x)[:, 5:, :]
        elif self.PFM_name == 'conch_v1_5':
            # Skip CLS token
            features = self.pfm.trunk.forward_features(x)[:, 1:, :]
        elif self.PFM_name == 'uni_v2':
            # Skip first 9 tokens (CLS and register tokens)
            features = self.pfm.forward_features(x)[:, 9:, :]
        else:
            # Standard ViT - skip CLS token
            features = self.pfm.forward_features(x)[:, 1:, :]
        
        # Decode features
        decoded_features = self.decoder(features)
        
        # Generate final predictions
        logits = self.segmentation_head(decoded_features)
        
        return {'out': logits}

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate feature maps for visualization.
        
        Args:
            x (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Intermediate feature maps
        """
        with torch.no_grad():
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
                
            if self.PFM_name == 'virchow_v2':
                features = self.pfm(x)[:, 5:, :]
            elif self.PFM_name == 'conch_v1_5':
                features = self.pfm.trunk.forward_features(x)[:, 1:, :]
            elif self.PFM_name == 'uni_v2':
                features = self.pfm.forward_features(x)[:, 9:, :]
            else:
                features = self.pfm.forward_features(x)[:, 1:, :]
                
            # Reshape to spatial format
            B, n_patch, hidden = features.size()
            h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
            features = features.permute(0, 2, 1).contiguous().view(B, hidden, h, w)
            
        return features


def create_pfm_segmentation_model(model_config: Dict[str, Any]) -> PFMSegmentationModel:
    """
    Factory function to create PFM segmentation model.
    
    Args:
        model_config (Dict[str, Any]): Model configuration dictionary
        
    Returns:
        PFMSegmentationModel: Configured PFM segmentation model
    """
    required_keys = ['pfm_name', 'pfm_weights_path', 'emb_dim','num_classes','finetune_mode']
    for key in required_keys:
        if key not in model_config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    pfm_seg_model =  PFMSegmentationModel(
        PFM_name=model_config['pfm_name'],
        PFM_weights_path=model_config['pfm_weights_path'],
        emb_dim=model_config['emb_dim'],
        num_classes=model_config.get('num_classes', 2))
    finetune_mode = model_config['finetune_mode'].get('type')
    if finetune_mode == 'frozen':
        for param in pfm_seg_model.pfm.parameters():
            param.requires_grad = False
    elif finetune_mode == 'lora':
        lora_rank = model_config['finetune_mode'].get('rank')
        lora_alpha = model_config['finetune_mode'].get('alpha')
        for param in pfm_seg_model.pfm.parameters():
            param.requires_grad = False
        pfm_seg_model.pfm = equip_model_with_lora(model_config['pfm_name'], pfm_seg_model.pfm, rank=lora_rank, alpha=lora_alpha)
    elif finetune_mode == 'full':
        pass
    return pfm_seg_model
        

