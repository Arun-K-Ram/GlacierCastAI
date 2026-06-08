# Experiment Registry

## Naming Convention
exp<NNN>_<backbone>_<modality>_<temporal>_<notes>

## Modality codes
- IO  : image only
- IC  : image + climate
- ICD : image + climate + DEM (full multimodal)

## Temporal codes
- lstm : ConvLSTM
- trf  : Temporal Transformer

## Example
exp001_resnet50_IO_lstm_baseline
exp002_resnet50_IC_lstm_add_climate
exp003_resnet50_ICD_lstm_full_multimodal
exp004_prithvi_ICD_trf_foundation_model
exp005_resnet50_ICD_trf_ablation_no_boundary_loss

## Results Table

| Exp | Backbone | Modality | Temporal | IoU | BF1 | RR-RMSE | Notes |
|-----|----------|----------|----------|-----|-----|---------|-------|
| 001 | ResNet50 | IO | ConvLSTM | - | - | - | Baseline |
| 002 | ResNet50 | IC | ConvLSTM | - | - | - | +Climate |
| 003 | ResNet50 | ICD | ConvLSTM | - | - | - | +DEM |
| 004 | Prithvi | ICD | Transformer | - | - | - | Full model |
| 005 | Swin-B | ICD | Transformer | - | - | - | |
| 006 | ConvNeXt | ICD | Transformer | - | - | - | |
| 007 | MaxViT | ICD | Transformer | - | - | - | |

## Ablation Table (for paper Table 2)

| Modality | IoU | BF1 | RR-RMSE |
|----------|-----|-----|---------|
| Image only | - | - | - |
| + Climate | - | - | - |
| + DEM | - | - | - |
| + Climate + DEM | - | - | - |

## Acceleration Detection Experiments

| Exp | Description | Precision | Recall | F1 |
|-----|-------------|-----------|--------|-----|
| A01 | Baseline acceleration head | - | - | - |