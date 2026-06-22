# GlacierCastAI

GlacierCastAI predicts glacier retreat acceleration before it becomes visible in satellite imagery. By fusing Landsat time series, ERA5 climate signals, and Copernicus DEM terrain features in a multi-modal deep learning model, it forecasts glacier boundary changes at 1, 3, and 5-year horizons - and uses SHAP attribution to identify which climate drivers are responsible, making predictions interpretable and trustworthy.

## Research Question

> Can climate signals predict glacier retreat acceleration before it becomes detectable in satellite imagery?

## Study Glaciers

| Glacier | Region | Climate Regime |
|---------|--------|----------------|
| Aletsch | Swiss Alps | Alpine |
| Gangotri | Himalayas | Monsoon |
| Grey | Patagonia | Maritime |
| Columbia | Alaska | Maritime/Subarctic |
| Athabasca | Canadian Rockies | Continental |

## Model Architecture

- **Backbone**: ResNet50 (pretrained, ImageNet)
- **Temporal**: ConvLSTM (3 layers, hidden_dim=256, T=4 timesteps)
- **Climate encoder**: Cross-attention transformer (16-dim ERA5 features × 4 seasons)
- **Decoder**: UNet-style with skip connections
- **Heads**: Boundary mask, retreat rate, risk score
- **Parameters**: 56.1M total

## Data

| Source | Variables | Resolution |
|--------|-----------|------------|
| Landsat 5/7/8/9 Collection 2 | Green, SWIR1, NDSI | 30m |
| Copernicus DEM GLO-30 | Elevation, slope, aspect | 30m |
| ERA5 Monthly Means | T2m, precipitation, snowfall, solar radiation | ~31km |

- **64 Landsat scenes** across 5 glaciers (2000–2023)
- **29,810 patches** (256×256px, 64px overlap)
- **40,476 sequences** (T=4 input timesteps, horizons 1/2/3yr)
- Train/Val/Test split: 27,725 / 5,861 / 6,890

## Experiment Results

| Exp | Modality | test/IoU | test/BF1 | Notes |
|-----|----------|----------|----------|-------|
| 001 | Image only | **0.326** | **0.158** | Baseline |
| 002 | Image + Climate | 0.313 | 0.133 | Re-run (checkpoint bug fix) |
| 003 | Image + Climate + DEM | - | - |  Running |
| 004 | Prithvi foundation model | - | - | Pending |
| 005 | Climate only (MLP) | - | - | Pending |

See `experiments/README.md` for full training logs and ablation table.

## Setup

```bash
conda create -n glaciercastai python=3.12
conda activate glaciercastai
```

Required API keys (copy `.env.example` to `.env`):
- `WANDB_API_KEY` - Weights & Biases
- CDS API key in `~/.cdsapirc` - ERA5 download
- NASA Earthdata credentials - Landsat download

## Training

```bash
# Train from scratch
python scripts/train.py --config configs/model.yaml

# Resume from checkpoint
python scripts/train.py --config configs/model.yaml --resume experiments/checkpoints/exp002/last.ckpt
```

## Repository Structure

```
GlacierCastAI/
├── configs/          # Model and training configs
├── scripts/          # Training, preprocessing, sequence building
├── src/
│   ├── models/       # GlacierCastAI model architecture
│   └── training/     # Lightning module, datamodule, losses
├── experiments/      # Checkpoints and results (gitignored)
└── data/             # Raw and processed data (gitignored)
```