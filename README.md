# Predicting future extreme rainfall over urban environments: systematic selection of climate projections and CNN-based downscaling

**Authors**: Wegayehu Asfaw* (from UT-ITC, NL), Tom Rientjes (from UT-ITC, NL), Alemseged Tamiru Haile (from IWMI, ETH), and Hui Tang (from GFZ, GER)  
**Last updated**: March 2026

## Abstract
This repository contains a script for implementing a residual Convolutional Neural Network (CNN) with k-fold cross-validation for statistical downscaling of daily rainfall from three CMIP6 Global Climate Models — **MRI-ESM2-0**, **EC-Earth3**, and **INM-CM5-0** — from ~0.5° to 0.05° resolution.  

The workflow:
- Historical data obtained by blending satellite estimates with ground-based observations using a dynamic Bayesian Model Averaging approach (Asfaw et al., 2023)
- Uses 12 large-scale predictors (clt, huss, hurs, pr, psl, sfcWind, rlds, rsds, tas, uas, vas, and wap) from three CMIP6 GCMs (MRI-ESM2-0, EC-Earth3, and INM-CM5-0)
- Applies log1p transformation + intensity-weighted Huber loss (quantile-based exponential weighting)
- Produces single-GCM downscaled fields for historical and future periods (SSP2-4.5 / SSP5-8.5)
- Generates multi-model ensemble means and pairwise ensembles
- Performance of the downscaled datasets and future predictions of extreme rainfall analysed

Main methodological choices
- CNN architecture: residual blocks + bilinear upsampling ×10
- Loss: Huber (δ=10) with intensity-dependent sample weights
- Target transformation: log1p + threshold at 1 mm/day
- Weighting: quantile bins + exponential function (different k and quantile points per GCM)
- Ensemble: 5-fold CV mean per GCM → multi-GCM mean & pairwise means
- ETCCDI indices applied to characterise extreme rainfall from the downscaled datasets

## Folder Structure
GCM_Downscaling_Study/

├── data/

│   ├── target/

│   │   └── historical_rainfall_05deg.nc           ← high-resolution observed rainfall (target)

│   ├── MRI-ESM2-0/

│   │   ├── historical/

│   │   ├── ssp245/

│   │   └── ssp585/

│   ├── EC-Earth3/          … same structure …

│   └── INM-CM5-0/          … same structure …

│
├── models/

│   ├── MRI-ESM2-0/

│   │   ├── MRI-ESM2-0_historical_downscaled.nc

│   │   ├── MRI-ESM2-0_ssp245_downscaled.nc

│   │   └── MRI-ESM2-0_ssp585_downscaled.nc

│   ├── EC-Earth3/

│   │   ├── EC-Earth3_historical_downscaled.nc

│   │   ├── EC-Earth3_ssp245_downscaled.nc

│   │   └── EC-Earth3_ssp585_downscaled.nc

│   └── INM-CM5-0/

│       ├── INM-CM5-0_historical_downscaled.nc

│       ├── INM-CM5-0_ssp245_downscaled.nc

│       └── INM-CM5-0_ssp585_downscaled.nc

│
└── ensembles/

├── historical/

│   ├── ensemble-mean_3models.nc

│   ├── ensemble-mean_EC-Earth3_INM-CM5-0.nc

│   ├── ensemble-mean_EC-Earth3_MRI-ESM2-0.nc

│   └── ensemble-mean_INM-CM5-0_MRI-ESM2-0.nc

├── ssp245/           … same pattern …

└── ssp585/           … same pattern …

## How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow xarray numpy scikit-learn matplotlib
2. Organise your data exactly as shown in the folder structure.
3. Run the script: python cnn_gcm_downscaling.py
4. Outputs appear automatically in models/ and ensembles/.

References:
Asfaw, W., Rientjes, T., Haile, A.T., 2023. Blending high-resolution satellite rainfall estimates over urban catchment using Bayesian Model Averaging approach. J. Hydrol.: Reg. Stud. 45, 101287. https://doi.org/10.1016/j.ejrh.2022.101287
