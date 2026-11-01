# QAPE-DETR

Core implementation of QAPE-DETR for calf behavior detection, built on DEIMv2.
This repository contains model code, training/evaluation utilities, and a single-image inference script. No datasets, annotations, pretrained weights, experiment outputs, or example images are distributed.

## Architecture

- **Backbone:** HGNetv2-Atto, using its stride-16 feature map.
- **Encoder:** `LiteEncoder` with a learned quality vector and the LGP, ECR, and QAF-lite modules enabled. It outputs stride-16 and stride-32 feature maps with 64 channels.
- **Decoder:** DEIMTransformer with 3 layers and 100 object queries.

The main custom modules are in [`engine/deim/lite_encoder.py`](engine/deim/lite_encoder.py). The complete configuration is [`configs/qape_detr.yml`](configs/qape_detr.yml). Internal DEIM class names are retained for compatibility with the original checkpoints.

## Installation

Use Python 3.11. Install a CUDA-compatible PyTorch build if GPU training is required; the dependency file retains the upstream torch 2.5.1 / torchvision 0.20.1 pair.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Prepare your own data

Use COCO object detection annotations and arrange your own files as follows (these folders are intentionally absent):

```text
data/
  train/
  val/
  test/
  annotations/
    train_coco.json
    val_coco.json
    test_coco.json
```

Set `num_classes` and paths in `configs/dataset/calf_detection.yml`. With `remap_mscoco_category: False`, category IDs must be contiguous and start at **0**. The supplied configuration retains the original two-class setting; assign the behavior names and IDs according to your annotations. Image `file_name` values are relative to each split folder. If changing validation/test paths, also update their overrides in `configs/qape_detr.yml`.

## Train and evaluate

Run all commands from the repository root:

```bash
python train.py -c configs/qape_detr.yml --device cuda:0
python train.py -c configs/qape_detr.yml --device cuda:0 -r outputs/qape_detr/last.pth
python train.py -c configs/qape_detr.yml --device cuda:0 --test-only --eval-set val -r path/to/checkpoint.pth
python train.py -c configs/qape_detr.yml --device cuda:0 --test-only --eval-set test -r path/to/checkpoint.pth
```

The training entry point also evaluates the selected best checkpoint on the test split after training when `test_dataloader` is configured. Keep test data held out when selecting models. The retained experiment configuration uses 100 epochs and 640 × 640 inputs. Batch size and worker count can be adjusted to your hardware:

```bash
python train.py -c configs/qape_detr.yml --device cuda:0 -u train_dataloader.total_batch_size=8 train_dataloader.num_workers=0 val_dataloader.num_workers=0 test_dataloader.num_workers=0
```

## Inference

```bash
python infer.py -c configs/qape_detr.yml -r path/to/checkpoint.pth -i path/to/image.jpg -o outputs/prediction.jpg --device cpu
```

Predictions are drawn with numeric class IDs and confidence scores. Use checkpoints from a trusted source. No checkpoint is included.

## Repository layout

```text
configs/       Model, dataset, optimizer and augmentation settings
engine/        Backbone, custom encoder, decoder, losses and training engine
train.py       Training, resume and validation/test evaluation
infer.py       Single-image inference
```

## Provenance and license

Based on [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), with components originating from DEIM, D-FINE, and RT-DETR. Original source copyright notices and the upstream Apache-2.0 LICENSE are preserved. The original experiment used `deimv2_hgnetv2_n_coco_backup.yml`; its actual backbone is **Atto**, despite the filename. The publication configuration retains its architecture and training settings while replacing local paths. No accuracy claims or paper citation metadata have been added.

## Validation status

The release has been checked for Python syntax, local module dependencies, configuration include paths, and excluded data/artifacts. Full training and checkpoint inference require the dependencies, your own dataset, and a compatible checkpoint; they have not been run as part of this source-only cleanup.
