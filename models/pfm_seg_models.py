"""
Pathology Foundation Models (PFM) for Semantic Segmentation

This module integrates multiple pathology foundation models including
"Gigapath, UNI v1/v2, Virchow v1/v2, Conch V1/V1.5, patho3dmatrix-vision, Phikon, Phikon-v2,
H-Optimus-0/1, MUSK, Midnight-12k, and Kaiko (vits8/vits16/vitb8/vitb16/vitl14)"
for segmentation tasks.

Author: @Toby and @chenwm
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
from .dora import equip_model_with_dora

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
        
    elif PFM_name == 'virchow_v1':#CWM
        from timm.layers import SwiGLUPacked
        # Create Virchow v1 model structure
        # Note: Using hf-hub model name to get correct architecture
        # Then load weights from local path
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=False,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )
        # Load weights from local path
        # Handle different weight file formats
        weights = torch.load(PFM_weights_path, map_location="cpu", weights_only=True)
        if isinstance(weights, dict):
            # If weights is a dict, try to extract state_dict
            if 'model' in weights:
                state_dict = weights['model']
            elif 'state_dict' in weights:
                state_dict = weights['state_dict']
            else:
                state_dict = weights
        else:
            state_dict = weights
        model.load_state_dict(state_dict, strict=True)
        model = model.eval()
        
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
        
    elif PFM_name == 'patho3dmatrix-vision':
        patho3dmatrix_vision_kwargs = {
            'model_name': 'vit_large_patch16_224',
            'img_size': 224,
            'patch_size': 16,
            'init_values': 1e-5,
            'num_classes': 0,
            'dynamic_img_size': True
        }
        model = timm.create_model(**patho3dmatrix_vision_kwargs)
        state_dict = torch.load(PFM_weights_path, map_location="cpu")
        new_state_dict = OrderedDict({k.replace('backbone.', ''): v for k, v in state_dict['teacher'].items()})
        model.load_state_dict(new_state_dict, strict=False)
            
    elif PFM_name == 'conch_v1':#CWM
        from conch.open_clip_custom import create_model_from_pretrained
        model, _ = create_model_from_pretrained('conch_ViT-B-16', PFM_weights_path)
            
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
        
    elif PFM_name == 'phikon':#CWM
        # Build Phikon ViT-B/16 locally, then load local bin weights (no HF repo needed)
        from transformers import ViTModel
        import os
        model_dir = os.path.dirname(PFM_weights_path)
        model = ViTModel.from_pretrained(model_dir, add_pooling_layer=False, local_files_only=True)
        model = PhikonWrapper(model)

    elif PFM_name == 'phikon_v2':#CWM
        # Phikon v2 uses a different HF repo and model format; load via AutoModel
        from transformers import AutoModel
        import os
        try:
            model_dir = os.path.dirname(PFM_weights_path)
            model = AutoModel.from_pretrained(model_dir)
        except Exception:
            raise Exception(
                f"Failed to create Phikon v2 model from local checkpoint at '{PFM_weights_path}'. "
                "You can download the required `model.safetensors` and `config.json` from: https://huggingface.co/owkin/phikon-v2."
            )

        # Wrap to provide a uniform interface (returns last_hidden_state)
        model = PhikonWrapper(model)

    elif PFM_name == 'hoptimus_0':
        # H-Optimus-0: ViT-Giant model with DinoV2 backbone
        hoptimus_0_config = {
            "num_classes": 0,
            "img_size": 224,
            "global_pool": "token",
            "init_values": 1e-5,
            "dynamic_img_size": False
        }
        model = timm.create_model("vit_giant_patch14_reg4_dinov2", **hoptimus_0_config)
        
        try:
            state_dict = torch.load(PFM_weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            raise Exception(
                f"Failed to create H-Optimus-0 model from local checkpoint at '{PFM_weights_path}'. "
                "You can download the required `pytorch_model.bin` from: https://huggingface.co/bioptimus/H-optimus-0."
            )

    elif PFM_name == 'hoptimus_1':
        # H-Optimus-1: ViT-Giant model with DinoV2 backbone (similar architecture to H-Optimus-0)
        hoptimus_1_config = {
            "num_classes": 0,
            "img_size": 224,
            "global_pool": "token",
            "init_values": 1e-5,
            "dynamic_img_size": False
        }
        
        model = timm.create_model("vit_giant_patch14_reg4_dinov2", **hoptimus_1_config)
        try:
            state_dict = torch.load(PFM_weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            raise Exception(
                f"Failed to create H-Optimus-1 model from local checkpoint at '{PFM_weights_path}'. "
                "You can download the required `pytorch_model.bin` from: https://huggingface.co/bioptimus/H-optimus-1."
            )

    elif PFM_name.startswith('kaiko-'):
        # Kaiko model family: vits8, vits16, vitb8, vitb16, vitl14
        # Map model names to their timm identifiers and image sizes
        kaiko_configs = {
            'kaiko-vits8': {'hf_hub_id': 'vit_small_patch8_224', 'img_size': 224},
            'kaiko-vits16': {'hf_hub_id': 'vit_small_patch16_224', 'img_size': 224},
            'kaiko-vitb8': {'hf_hub_id': 'vit_base_patch8_224', 'img_size': 224},
            'kaiko-vitb16': {'hf_hub_id': 'vit_base_patch16_224', 'img_size': 224},
            'kaiko-vitl14': {'hf_hub_id': 'vit_large_patch14_reg4_dinov2', 'img_size': 518},
        }
        
        if PFM_name not in kaiko_configs:
            raise ValueError(f"Unknown Kaiko model: {PFM_name}. Available: {list(kaiko_configs.keys())}")
        
        config = kaiko_configs[PFM_name]
        hf_hub_id = config['hf_hub_id']
        img_size = config['img_size']
        
        try:
            model = timm.create_model(
                hf_hub_id,
                num_classes=0,
                checkpoint_path=PFM_weights_path,
                img_size=img_size,
                dynamic_img_size=True
            )
        except Exception:
            raise Exception(
                f"Failed to create Kaiko model from local checkpoint at '{PFM_weights_path}'. "
                "You can download the required `model.safetensors` and `config.yaml` from: https://huggingface.co/collections/1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795."
            )

    elif PFM_name == 'midnight12k':
        # Midnight-12k by Kaiko: Uses AutoModel.from_pretrained with HuggingFace Hub
        from transformers import AutoModel
        import os
        try:
            model_dir = os.path.dirname(PFM_weights_path)
            model = AutoModel.from_pretrained(model_dir)
        except Exception:
            raise Exception(
                f"Failed to create Midnight-12k model from local checkpoint at '{PFM_weights_path}'. "
                "You can download the required `model.safetensors` and `config.json` from: https://huggingface.co/kaiko-ai/midnight."
            )
        
        # Wrap to provide a uniform interface (returns last_hidden_state with CLS token handling)
        model = Midnight12kWrapper(model)

    elif PFM_name == 'lunit_vits8':
        # Lunit-S8: ViT-Small with Dino pretraining, using timm
        try:
            timm_kwargs = {"img_size": 224}
            model = timm.create_model("vit_small_patch8_224", checkpoint_path=PFM_weights_path, **timm_kwargs)
        except Exception:
            raise Exception(
                f"Failed to create Lunit-S8 model from local checkpoint at '{PFM_weights_path}'. "
                "You can download the required `model.safetensors` and `config.yaml` from: https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino."
            )

    elif PFM_name == 'hibou_l':
        # Hibou-Large: uses transformers.AutoModel with trust_remote_code
        from transformers import AutoModel
        import os
        if PFM_weights_path:
            raise NotImplementedError("Hibou-Large doesn't support local model loading. PR welcome!")
        else:
            try:
                model = AutoModel.from_pretrained("histai/hibou-L", trust_remote_code=True)
            except:
                raise Exception("Failed to download Hibou-L model, make sure that you were granted access and that you correctly registered your token")
        

        # Wrap to provide last_hidden_state as token sequence
        model = HibouWrapper(model)

    elif PFM_name == 'musk':#CWM
        # MUSK: specialized model with custom loading utilities
        try:
            from musk import utils as musk_utils, modeling
        except Exception:
            raise Exception("Please install MUSK `pip install fairscale git+https://github.com/lilab-stanford/MUSK`")

        # MUSK upstream loader in reference does not support local checkpoint loading
        # and uses their `load_model_and_may_interpolate` helper to fetch from HF dataset.
        if PFM_weights_path:
            # Mirror reference behavior: local loading is not implemented for MUSK
            raise NotImplementedError("MUSK doesn't support local model loading. PR welcome!")
        else:
            try:
                # load weights from hf hub dataset (this function will download and load)
                model = timm.create_model("musk_large_patch16_384")
                musk_utils.load_model_and_may_interpolate("hf_hub:xiangjx/musk", model, 'model|module', '')
            except Exception:
                raise Exception("Failed to download MUSK model, make sure that you were granted access and that you correctly registered your token")

        # Wrap to provide a uniform interface for segmentation (try to expose patch tokens)
        model = MuskWrapper(model)

    elif PFM_name == 'PathOrchestra':
        pathOrchestra_config = {
            "num_classes": 0, 
            "init_values": 1e-5, 
            "dynamic_img_size": True
        }
        model = timm.create_model(
            "hf-hub:AI4Pathology/PathOrchestra",
            checkpoint_path=PFM_weights_path,
            **pathOrchestra_config
        )

    else:
        raise ValueError(f"Unsupported PFM model: {PFM_name}")

    return model


class PhikonWrapper(nn.Module):
    """
    Wrapper class for Phikon model to make it compatible with the segmentation framework.
    
    Phikon uses transformers library's ViTModel, which expects different input format.
    This wrapper converts the input tensor format and extracts features properly.
    """
    
    def __init__(self, vit_model: nn.Module):
        """
        Initialize Phikon wrapper.
        
        Args:
            vit_model: ViTModel from transformers library
        """
        super(PhikonWrapper, self).__init__()
        self.vit_model = vit_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Phikon model.
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Features of shape (B, num_tokens, hidden_dim)
        """
        # Phikon expects input in format (B, C, H, W) where values are in [0, 1]
        # Transformers ViTModel expects pixel_values in format (B, C, H, W)
        # The model will handle normalization internally based on its config
        
        # Get outputs from ViTModel
        outputs = self.vit_model(pixel_values=x)
        # Extract last_hidden_state: (B, num_tokens, hidden_dim)
        features = outputs.last_hidden_state
        return features
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get features (alias for forward).
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Features of shape (B, num_tokens, hidden_dim)
        """
        return self.forward(x)


class HibouWrapper(nn.Module):
    """
    Wrapper for Hibou models (histai/hibou-L). Returns token sequence
    (last_hidden_state) so segmentation code can use patch tokens.
    """

    def __init__(self, hibou_model: nn.Module):
        super(HibouWrapper, self).__init__()
        self.hibou_model = hibou_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # hibou expects pixel_values keyword
        out = self.hibou_model(pixel_values=x)
        # return token sequence (B, num_tokens, hidden)
        return out.last_hidden_state

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hibou_model(pixel_values=x)
        return out.last_hidden_state


class MuskWrapper(nn.Module):
    """
    Wrapper for MUSK model to provide a consistent interface for
    feature extraction. MUSK's forward method returns (vision_cls, language_cls).
    By setting return_global=False, we can get patch-level tokens for segmentation.
    """

    def __init__(self, musk_model: nn.Module):
        super(MuskWrapper, self).__init__()
        self.musk_model = musk_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return per-token/patch features for segmentation.
        
        Uses MUSK's forward method with return_global=False to get all patch tokens
        instead of just the CLS token.
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Patch tokens of shape (B, num_tokens, hidden_dim)
        """
        # Call MUSK forward with return_global=False to get all tokens (not just CLS)
        # with_head=False and out_norm=False to get raw features
        vision_cls, _ = self.musk_model(
            image=x,
            return_global=False,  # Return all tokens, not just CLS
            with_head=False,      # Don't apply vision_head
            out_norm=False,       # Don't normalize output
            ms_aug=False          # Disable multiscale augmentation for segmentation
        )
        
        if vision_cls is None:
            raise RuntimeError("MUSK model returned None for vision features")
        
        if not isinstance(vision_cls, torch.Tensor):
            raise RuntimeError("MUSK model returned non-tensor output; cannot extract features")

        if vision_cls.ndim == 2:
            # Only global embeddings available (should not happen with return_global=False)
            raise RuntimeError(
                "MUSK returned global embeddings only; expected patch tokens with return_global=False."
            )

        return vision_cls
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get features (alias for forward).
        
        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Patch tokens of shape (B, num_tokens, hidden_dim)
        """
        return self.forward(x)


class Midnight12kWrapper(nn.Module):
    """
    Wrapper for Midnight-12k model (Kaiko) to provide a consistent interface
    for feature extraction. Midnight-12k uses transformers' AutoModel which
    returns a BaseModelOutput with last_hidden_state containing patch tokens.
    """

    def __init__(self, midnight_model: nn.Module, return_type: str = "cls+mean"):
        """
        Initialize Midnight-12k wrapper.

        Args:
            midnight_model: AutoModel instance from transformers
            return_type: "cls_token" returns only CLS token,
                        "cls+mean" concatenates CLS token with mean of patch tokens
        """
        super(Midnight12kWrapper, self).__init__()
        self.midnight_model = midnight_model
        self.return_type = return_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Midnight-12k model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Features - either CLS token only or CLS+mean of patches,
                         shape (B, hidden_dim) or (B, 2*hidden_dim)
        """
        out = self.midnight_model(x)
        last_hidden_state = out.last_hidden_state  # (B, num_tokens, hidden_dim)
        cls_token = last_hidden_state[:, 0, :]     # (B, hidden_dim)

        if self.return_type == "cls_token":
            return cls_token
        elif self.return_type == "cls+mean":
            patch_embeddings = last_hidden_state[:, 1:, :]  # skip CLS token
            patch_mean = patch_embeddings.mean(1)           # (B, hidden_dim)
            # Concatenate CLS with mean of patches
            return torch.cat([cls_token, patch_mean], dim=-1)  # (B, 2*hidden_dim)
        else:
            raise ValueError(f"Unknown return_type: {self.return_type}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get full token sequence (for segmentation compatibility).

        For segmentation we need the full token sequence, not the aggregated output.
        This method returns the last_hidden_state directly.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Full last_hidden_state of shape (B, num_tokens, hidden_dim)
        """
        out = self.midnight_model(x)
        return out.last_hidden_state


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
    
    def __init__(self, emb_dim: int, decoder_channels: Tuple[int, ...], is_virchow_v2_or_is_virchow_v1_or_uni_v2_or_midnight_or_hoptimus_or_hibou_or_kaiko: bool = False,is_lunit: bool = False):
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
        if is_virchow_v2_or_is_virchow_v1_or_uni_v2_or_midnight_or_hoptimus_or_hibou_or_kaiko:
            blocks[-1] = DecoderBlock(in_channels[-1], out_channels[-1], skip_channels[-1], scale=1.75)
        if is_lunit:
            blocks[-1] = DecoderBlock(in_channels[-1], out_channels[-1], skip_channels[-1], scale=1)
            
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
        if PFM_name == 'virchow_v2' or PFM_name == 'virchow_v1' or PFM_name == 'uni_v2' or PFM_name=='midnight12k' or PFM_name=='hoptimus_0' or PFM_name=='hoptimus_1' or PFM_name=='hibou_l' or PFM_name.startswith('kaiko-'):
            self.decoder = DecoderCup(emb_dim, self.decoder_channels, is_virchow_v2_or_is_virchow_v1_or_uni_v2_or_midnight_or_hoptimus_or_hibou_or_kaiko = True)
        elif PFM_name=='lunit_vits8':
            self.decoder = DecoderCup(emb_dim, self.decoder_channels, is_lunit=True)
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
        if self.PFM_name == 'virchow_v1':
            # Skip CLS token and use patch tokens
            features = self.pfm(x)[:, 1:, :]  # size: (B, 256, 1280)
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
            features = self.pfm(x)[:, 1:, :]  # size: (B, num_patches, hidden)
        elif self.PFM_name == 'hibou_l':
            # Hibou-Large: transformers AutoModel wrapper - Skip first 5 tokens (CLS and register tokens)
            features = self.pfm(x)[:, 5:, :]
        elif self.PFM_name == 'musk':
            # MUSK: wrapper returns all tokens (including CLS token)
            # Skip CLS token to get patch tokens only
            features = self.pfm.forward(x)[:, 1:, :]  # Skip first token (CLS token)
        elif self.PFM_name == 'lunit_vits8':
            # Lunit-S8: standard ViT - skip CLS token
            features = self.pfm.forward_features(x)[:, 1:, :]  # size: (B, num_patches, hidden)
        elif self.PFM_name == 'midnight12k':
            # Midnight-12k: use forward_features to get full token sequence, skip CLS token
            features = self.pfm.forward_features(x)[:, 1:, :]  # size: (B, num_patches, hidden)
        elif self.PFM_name.startswith('kaiko-'):
            # Kaiko models (vits8, vits16, vitb8, vitb16, vitl14): standard ViT - skip CLS token
            features = self.pfm.forward_features(x)[:, 5:, :]  # size: (B, num_patches, hidden)
        elif self.PFM_name == 'hoptimus_0' or self.PFM_name == 'hoptimus_1':
            # H-Optimus-0/1: ViT-Giant models - skip CLS token, keep patch tokens
            features = self.pfm.forward_features(x)[:, 5:, :]  # size: (B, num_patches, hidden)
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
                
            if self.PFM_name == 'virchow_v1':
                features = self.pfm(x)[:, 1:, :] 
            elif self.PFM_name == 'virchow_v2':
                features = self.pfm(x)[:, 5:, :]
            elif self.PFM_name == 'conch_v1':
                features = self.pfm.visual.trunk.forward_features(x)[:, 1:, :]
            elif self.PFM_name == 'conch_v1_5':
                features = self.pfm.trunk.forward_features(x)[:, 1:, :]
            elif self.PFM_name == 'phikon' or self.PFM_name == 'phikon_v2':
                features = self.pfm(x)[:, 1:, :]  # Skip CLS token
            elif self.PFM_name == 'hibou_l':
                features = self.pfm(x)[:, 5:, :]
            elif self.PFM_name == 'musk':
                # MUSK: wrapper returns all tokens (including CLS token)
                # Skip CLS token to get patch tokens only
                features = self.pfm.forward(x)[:, 1:, :]  # Skip first token (CLS token)
            elif self.PFM_name == 'lunit_vits8':
                features = self.pfm.forward_features(x)[:, 1:, :]  # Skip CLS token
            elif self.PFM_name == 'midnight12k':
                features = self.pfm.forward_features(x)[:, 1:, :]  # Skip CLS token
            elif self.PFM_name.startswith('kaiko-'):
                features = self.pfm.forward_features(x)[:, 5:, :]  # Skip CLS token
            elif self.PFM_name == 'hoptimus_0' or self.PFM_name == 'hoptimus_1':
                features = self.pfm.forward_features(x)[:, 5:, :]  # Skip CLS token
            elif self.PFM_name == 'patho3dmatrix-vision':
                features = self.pfm.forward_features(x)[:, 1:, :]
            elif self.PFM_name == 'uni_v2':
                features = self.pfm.forward_features(x)[:, 9:, :]
            elif self.PFM_name == 'PathOrchestra':
                features = self.pfm.forward_features(x)[:, 5:, :]
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
    elif finetune_mode == 'dora':
        dora_rank = model_config['finetune_mode'].get('rank')
        dora_alpha = model_config['finetune_mode'].get('alpha')
        for param in pfm_seg_model.pfm.parameters():
            param.requires_grad = False
        pfm_seg_model.pfm = equip_model_with_dora(model_config['pfm_name'], pfm_seg_model.pfm, rank=dora_rank, alpha=dora_alpha)
    elif finetune_mode == 'full':
        pass
    return pfm_seg_model


def create_segmentation_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Unified factory function to create segmentation models (PFM or UNet).
    
    Args:
        model_config (Dict[str, Any]): Model configuration dictionary
            For PFM models, required keys: pfm_name, pfm_weights_path, emb_dim, num_classes, finetune_mode
            For UNet, required keys: num_classes (and optionally model_type: 'unet' or pfm_name: 'unet')
        
    Returns:
        nn.Module: Configured segmentation model (PFMSegmentationModel or UNet)
    """
    # Check if UNet is requested
    pfm_name = model_config.get('pfm_name', '').lower()
    model_type = model_config.get('model_type', '').lower()
    
    if pfm_name == 'unet' or model_type == 'unet':
        from .unet import create_unet_model
        return create_unet_model(model_config)
    else:
        # Use PFM model
        return create_pfm_segmentation_model(model_config)
        

