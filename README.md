# Classification Methods on SpaceNet-7 Satellite Imagery

Final project for MATH 7339 (Northeastern University), exploring seasonal
classification and urban-sector segmentation on multi-temporal satellite
imagery from the SpaceNet-7 Multi-Temporal Urban Development Challenge.

**Author:** Maeve Wolfenden
**Course:** MATH 7339, Fall 2025

## Overview

The SpaceNet-7 dataset provides two years of monthly satellite imagery and
building-footprint labels for 60+ Areas of Interest (AOIs) worldwide. Rather
than tackling the original building-footprint change-detection task, this
project asks two different questions:

1. **Season classification** — can supervised and unsupervised models infer
   the season an image was taken in from RGB bands alone?
2. **Urban-sector segmentation** — can Gaussian Mixture Models, using
   engineered spectral/texture features, separate man-made from natural
   areas, and dense from sparse development, without ground-truth land-use
   labels?

Brief exploratory work on time-series forecasting (ARIMA/SARIMA/Prophet on
green-band trends) and building-count regression is also included.

## Methods

- **Image ingestion:** Rasterio (spectral bands) and a pretrained ResNet-18
  (512-d embeddings)
- **Dimensionality reduction:** Incremental PCA (96.9% variance) on flattened
  RGB pixels; PCA (90% variance) on ResNet-18 embeddings
- **Supervised season classification:** Random Forest, Bagged RF, LDA+RF,
  KNN, XGBoost, SVM, Naive Bayes
- **AOI-level meta-modeling:** Random Forest Regressor + SHAP to explain why
  season-classification accuracy varies so much by AOI/biome
- **Unsupervised seasonality:** Gaussian Hidden Markov Models (2-state
  global, 4-state single-AOI case study)
- **Urban-sector segmentation:** Three-stage Gaussian Mixture Model pipeline
  (man-made vs. natural → density → enhanced sector mask), using Sobel edge
  density, GLCM texture, corner density, and connected-component features
- **Exploratory forecasting:** ARIMA/SARIMA/Prophet on stacked per-AOI
  green-band time series; Random Forest regression for building counts

## Key Findings

- Season classification is viable but highly region-dependent — some AOIs
  predict near-perfectly, others barely above chance, with no clean
  correlation to spectral distributions.
- Unsupervised latent-state modeling recovers seasonal structure only for
  AOIs with strong, traditional (temperate) seasonality; tropical/arid
  biomes with wet/dry cycles confound the hard 4-season encoding.
- Feature-engineered GMMs reliably separate man-made from natural terrain,
  and can further distinguish dense vs. sparse development.
- ResNet-18 embeddings did **not** outperform raw spectral PCA features for
  this task.

Full writeup: [`reports/MATH_7339_final_paper.pdf`](reports/MATH_7339_final_paper.pdf)
Slides: [`reports/MATH_7339_Final_Presentation.pdf`](reports/MATH_7339_Final_Presentation.pdf)

## Data

This project uses the [SpaceNet-7 Multi-Temporal Urban Development
Challenge](https://spacenet.ai/sn7-challenge/) dataset. It is not included in
this repo due to size.

To reproduce:
1. Download the SpaceNet-7 training set (`SN7_buildings_train`) from the
   [SpaceNet AWS bucket](https://spacenet.ai/sn7-challenge/) or Kaggle mirror.
2. Place it locally and update `BASE_DIR` at the top of the notebook to point
   to your `.../SN7_buildings_train/train` folder.

Expected structure per AOI:
```
train/
└── L15-<AOI_ID>/
├── images/            # raw monthly GeoTIFFs
├── images_masked/     # masked monthly GeoTIFFs
└── labels/            # per-timestamp GeoJSON building footprints
```
## Setup

```bash
git clone https://github.com/<your-username>/spacenet7-classification.git
cd spacenet7-classification
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then open `notebooks/Final_Code.ipynb` and update `BASE_DIR` to point at your
local copy of the dataset.

## Repo Structure
```
├── notebooks/Final_Code.ipynb   # full analysis pipeline
├── reports/                     # final paper + slide deck (PDF)
└── figures/                     # exported plots referenced in the paper
```
## Limitations & Future Work

- RGB-only imagery limits access to vegetation/spectral indices (e.g. NDVI)
- Two-year temporal coverage and global AOI diversity limit seasonal
  inference for non-temperate biomes
- No ground truth for land-use segmentation categories
- Future directions: longer temporal archives, biome-aware latent models,
  transfer learning across AOIs, and supervised sector segmentation with
  manually drawn masks

## References

- Van Etten, A. *The SpaceNet 7 Multi-Temporal Urban Development Challenge:
  Dataset Release.*
- MathWorks. *ResNet-18 Convolutional Neural Network.*
