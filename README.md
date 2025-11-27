# 🩺 Pathology Foundation Models Meet Semantic Segmentation

A comprehensive semantic segmentation framework based on Pathology Foundation Models (PFMs), designed specifically for pathological image analysis, supporting multiple state-of-the-art pathology foundation models with complete training, inference, and evaluation capabilities.

## 🌟 Features

- 🧬 **Support for SOTA Pathology Foundation Models**: uni_v1, uni_v2, conch_v1_5, gigapath, virchow_v2
- 🔧 **Flexible Fine-tuning Strategies**: LoRA, full parameter fine-tuning, frozen backbone
- 📊 **Complete Training Pipeline**: Mixed precision training, learning rate scheduling, gradient accumulation
- 🎯 **Advanced Data Augmentation**: Integrated 10+ advanced data augmentations including spatial, color, and noise transformations
- 📈 **Comprehensive Evaluation Metrics**: Integrated 10+ evaluation metrics including IoU/Dice and more
- ⚡ **Advanced Inference Pipeline**: Support for arbitrary resolution sliding window inference

## 📋 Table of Contents

- [Dataset Format](#-dataset-format)
- [Configuration File Details](#-configuration-file-details)
- [Training Script Usage](#-training-script-usage)
- [Inference Script Usage](#-inference-script-usage)
- [Pathology Foundation Models Details](#-pathology-foundation-models-details)

## 📁 Dataset Format

### JSON Configuration File Format

The dataset uses JSON format for configuration, supporting train, validation, and test set splits:

```json
{
  "num_classes": 3,
  "data": {
    "train": [
      {
        "image_path": "/path/to/train/image1.jpg",
        "mask_path": "/path/to/train/mask1.png"
      },
    ],
    "val": [
      {
        "image_path": "/path/to/val/image1.jpg",
        "mask_path": "/path/to/val/mask1.png"
      }
    ],
    "test": [
      {
        "image_path": "/path/to/test/image1.jpg",
        "mask_path": "/path/to/test/image2.png" 
      }
    ]
  }
}
```

During training, only the `train` and `val` fields are used. The `test` field is used when executing inference scripts. The `mask_path` in the test field can be null or missing, in which case the model will not compute metrics. If `mask_path` exists, metrics will be automatically calculated after inference.

## ⚙️ Configuration File Details

The configuration file uses YAML format and includes the following main sections:

### Dataset Configuration (dataset)

```yaml
dataset:
  json_file: "/path/to/dataset.json"  # Path to dataset JSON configuration file
  num_classes: 3                      # Number of classes, must match JSON file
  ignore_index: 255                   # Pixel value to ignore for uncertain regions
```

### System Configuration (system)

```yaml
system:
  num_workers: 4          # Number of processes for data loading
  pin_memory: true        # Whether to use pin_memory for faster data transfer
  seed: 42               # Random seed for reproducible experiments
  device: "cuda:0"       # Device to use
```

### Pathology Foundation Model Configuration (model) 🧬

This is the most important section, controlling the selection and configuration of pathology foundation models:

```yaml
model:
  # === Base Model Selection ===
  pfm_name: "uni_v1"                    # Pathology foundation model name
  # Options:
  # - "uni_v1"       : UNI model version 1 (1024 dim)
  # - "uni_v2"       : UNI model version 2 (1536 dim)
  # - "conch_v1"     : Conch model version 1 (1024 dim)
  # - "conch_v1_5"   : Conch model version 1.5 (768 dim)
  # - "virchow_v1"   : Virchow model version 2 (1280 dim)
  # - "virchow_v2"   : Virchow model version 2 (1280 dim)
  # - "phikon"       : Phikon model (768 dim)
  # - "phikon_v2"    : Phikon-v2 model (1024 dim)
  # - "hoptimus_0"   : H-Optimus-0 model (1536 dim)
  # - "hoptimus_1"   : H-Optimus-1 model (1536 dim)
  # - "gigapath"     : Gigapath model (1536 dim)
  # - "midnight12k"  : Midnight-12k model (1536 dim)
  # - "kaiko-vitl14" : Kaiko-ViT-L14 model (1024 dim)
  # - "lunit_vits8"  : Lunit-S8 model (384 dim)
  # - 'musk'         : MUSK model (1024 dim)
  # - "patho3dmatrix-vision": Patho3DMatrix-Vision model (1536 dim)
  # - "pathorchestra": PathOrchestra model (1536 dim)
  # - "hibou_l"     : Hibou-Large model (1024 dim)
  
  
  # === Model Parameter Configuration ===
  emb_dim: 1024                         # Embedding dimension, must match selected PFM model
  # Corresponding embedding dimensions for each model:
  # midnight12k/hoptimus_0/hoptimus_1/uni_v2/gigapath: 1536
  # virchow_v1/virchow_v2: 1280
  # uni_v1/hibou_l/musk/phikon_v2/kaiko-vitl14/conch_v1/patho3dmatrix-vision: 1024
  # conch_v1_5/phikon: 768
  # lunit_vits8: 384
  
  pfm_weights_path: '/path/to/pytorch_model.bin'  # Path to pre-trained weights file
  
  # === Fine-tuning Strategy Configuration ===
  finetune_mode:
    type: "lora"          # Fine-tuning mode
    # Options:
    # - "lora"   : LoRA low-rank adaptation, parameter efficient
    # - "full"   : Full parameter fine-tuning, best performance but requires more memory
    # - "frozen" : Frozen backbone, only train segmentation head
    
    rank: 16              # LoRA rank, only used when type is "lora"
    alpha: 1.0            # LoRA scaling factor, only used when type is "lora"
  
  num_classes: 3                  # Number of segmentation classes, must match dataset.num_classes
```

### Training Configuration (training)

```yaml
training:
  # === Basic Training Parameters ===
  batch_size: 8           # Batch size
  epochs: 100             # Number of training epochs
  learning_rate: 0.001     # Initial learning rate
  weight_decay: 0.0001    # Weight decay
  
  # === Training Optimization Settings ===
  use_amp: true                    # Whether to use mixed precision training
  accumulate_grad_batches: 1       # Number of gradient accumulation steps
  clip_grad_norm: 5.0              # Gradient clipping threshold
  
  # === Data Augmentation Configuration ===
  augmentation:
    RandomResizedCropSize: 512     # Random crop size
    # Note: Different PFM models have input size requirements
    # virchow_v1,virchow_v2,uni_v2,midnight12k,kaiko-vitl14,hibou_l,H-optimus-1,H-optimus-0: must be a multiple of 14 (token_size) 
    # uni_v1,conch_v1_5,gigapath,conch_v1 ,phikon,phikon_v2,patho3dmatrix-vision: must be a multiple of 16 (token_size) 
    # lunit_vits8: must be a multiple of 8 (token_size)
    # special: musk: 384
  
  # === Optimizer Configuration ===
  optimizer:
    type: "SGD"                    # Optimizer type: SGD, Adam, AdamW
    momentum: 0.9                  # SGD momentum (SGD only)
    nesterov: true                 # Whether to use Nesterov momentum
  
  # === Learning Rate Scheduler ===
  scheduler:
    type: "cosine"                 # Scheduler type: cosine, step
    warmup_epochs: 2               # Number of warmup epochs
  
  # === Loss Function ===
  loss:
    type: "dice"          # Loss function: cross_entropy, dice, ohem, iou
```

### Validation Configuration (validation)

```yaml
validation:
  eval_interval: 1        # Validate every N epochs
  batch_size: 16          # Validation batch size
  augmentation:
    ResizedSize: 512      # Image size during validation
    # Note: Different PFM models have input size requirements
    # virchow_v1,virchow_v2,uni_v2,midnight12k,kaiko-vitl14,hibou_l,H-optimus-1,H-optimus-0: must be a multiple of 14 (token_size) 
    # uni_v1,conch_v1_5,gigapath,conch_v1 ,phikon,phikon_v2,patho3dmatrix-vision: must be a multiple of 16 (token_size) 
    # lunit_vits8: must be a multiple of 8 (token_size)
    # special: musk: 384
```

### Logging and Visualization Configuration

```yaml
logging:
  log_dir: "/path/to/logs"           # Log save directory
  experiment_name: "pfm_segmentation" # Experiment name

visualization:
  save_interval: 2        # Save visualization results every N epochs
  num_vis_samples: 8      # Number of visualization samples to save
```

## 🚀 Training Script Usage

### Basic Training Command

```bash
python scripts/train.py --config configs/config.yaml
```

### Training Script Parameters Details

```bash
python scripts/train.py \
    --config configs/config.yaml \      # Configuration file path
    --resume checkpoints/model.pth \    # Resume training from checkpoint (optional)
    --device cuda:0                     # Specify device (optional, overrides config file)
```

### Parameter Description

- `--config`: **Required** Configuration file path containing all training settings
- `--resume`: **Optional** Checkpoint file path for resuming interrupted training
- `--device`: **Optional** Training device, overrides device setting in config file

### Training Output

During training, the following files will be generated:

```
logs/experiment_name/
├── config.yaml                 # Saved copy of configuration file
├── training.log                # Training log
├── checkpoints/                # Model checkpoints
│   ├── best_model.pth          # Best model
├── visualizations/             # Visualization results
│   ├── epoch_010_sample_00.png
│   └── ...
└── training_history.png        # Training curve plot
```

### Training Monitoring

During training, the following will be displayed:
- Training loss and validation loss
- Validation metrics (mIoU, Pixel Accuracy, etc.)
- Learning rate changes
- Time consumption per epoch

## 🔍 Inference Script Usage

### Basic Inference Command

```bash
python scripts/infer.py \
    --config logs/experiment_name/config.yaml \
    --checkpoint logs/experiment_name/checkpoints/best_model.pth \
    --input_json dataset/test.json \
    --output_dir results/
```

### Inference Script Parameters Details

```bash
python scripts/infer.py \
    --config CONFIG_PATH \              # Configuration file used during training
    --checkpoint CHECKPOINT_PATH \      # Trained model weights
    --input_json INPUT_JSON \           # Input data JSON file
    --output_dir OUTPUT_DIR \           # Results save directory
    --device cuda:0 \                   # Inference device
    --input_size 512 \                  # Input image size
    --resize_or_windowslide windowslide \ # Inference mode
    --batch_size 4                      # Inference batch size
```

### Detailed Parameter Description

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `--config` | str | ✅ | Configuration file path used during training |
| `--checkpoint` | str | ✅ | Path to model checkpoint file or checkpoint directory. For LoRA/DoRA mode, will automatically load both base model and LoRA/DoRA weights. |
| `--input_json` | str | ✅ | JSON file containing data to be inferred |
| `--output_dir` | str | ✅ | Inference results save directory |
| `--device` | str | ✅ | Inference device, default cuda:0 |
| `--input_size` | int | ✅ | Input image size for model, not original image size |
| `--resize_or_windowslide` | str | ✅ | Inference mode, default windowslide |
| `--batch_size` | int | ✅ | Inference batch size, default 2 |

### Inference Mode Selection

1. **Resize Mode** (`--resize_or_windowslide resize`)
   - Resize input images to fixed size (input_size) for inference
   - Resize prediction results back to original image size after inference

2. **Window Slide Mode** (`--resize_or_windowslide windowslide`)  
   - Use sliding window (input_size) strategy to process large images
   - Maintains original resolution with higher accuracy
   - Merge back to original image size after inference

### Inference Output

After inference completion, the following will be generated:

```
output_dir/
├── predictions_masks/          # Prediction masks (grayscale images)
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── predictions_overlays/       # Prediction result visualizations (colored overlay images)
    ├── image001.png
    ├── image002.png
    └── ...
```

### Inference Result Format

- **Prediction Masks**: Grayscale PNG images with pixel values corresponding to class indices
- **Visualization Overlays**: Colored overlays of original images with prediction results for intuitive viewing

## 🧬 Pathology Foundation Models Details

### Supported Models List

| Model Name | Parameters | Embedding Dim | Token Size | HuggingFace |
|------------|------------|---------------|------------|-------------|
| UNI | 307M | 1024 | 16×16 | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| UNI2-h | 1.1B | 1536 | 14×14 | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| CONCH | 90M | 768 | 16×16 | [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) |
| CONCHv1.5 | 307M | 1024 | 16×16 | [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5) |
| Virchow | 632M | 2560 | 14×14 | [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) |
| Virchow2 | 632M | 2560 | 14×14 | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |
| Phikon | 85.8M | 768 | 16×16 | [owkin/phikon](https://huggingface.co/owkin/phikon) |
| Phikon-v2 | 300M | 1024 | 16×16 | [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2) |
| Prov-Gigapath | 1.1B | 1536 | 16×16 | [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| H-Optimus-0 | 1.1B | 1536 | 14×14 | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |
| H-Optimus-1 | 1.1B | 1536 | 14×14 | [bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) |
| MUSK | - | 1024 | 32×32 | [xiangjx/musk](https://huggingface.co/xiangjx/musk) |
| Midnight-12k | - | 3072 | 14×14 | [kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight) |
| Kaiko | Various | 384/768/1024 | Various (8×8 or 16×16 or 14×14) | [1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795](https://huggingface.co/collections/1aurent/kaikoai-models) |
| Lunit | 21.7M | 384 | 8×8 | [1aurent/vit_small_patch8_224.lunit_dino](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino) |
| Hibou | - | 1024 | 14×14 | [histai/hibou-L](https://huggingface.co/histai/hibou-L) |
| PathOrchestra | 307M | 1024 | 16×16 | [AI4Pathology/PathOrchestra](https://huggingface.co/AI4Pathology/PathOrchestra) |
| patho3dmatrix-vision | 307M | 1024 | 16×16 | - |

## 🤝 Contributing

Welcome to submit issues and feature requests! Please check the contribution guidelines for more information.

## 📞 Contact

If you have questions or suggestions, please contact us through:
- Submit GitHub Issue
- Send email to: [lingxt23@mails.tsinghua.edu.cn] or [cwm25@mails.tsinghua.edu.cn]

---

