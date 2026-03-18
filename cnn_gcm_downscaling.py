import os
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Cropping2D, ReLU, Add, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from sklearn.model_selection import KFold
from itertools import combinations

# ────────────────────────────────────────────────
#  1. CONFIGURATION ─ per-GCM differences only
# ────────────────────────────────────────────────

GCM_CONFIGS = {
    "MRI-ESM2-0": {
        "hist_predictor_dir":   "data/MRI-ESM2-0/historical",
        "ssp245_predictor_dir": "data/MRI-ESM2-0/ssp245",
        "ssp585_predictor_dir": "data/MRI-ESM2-0/ssp585",
        "output_dir":           "models/MRI-ESM2-0",              # final files go directly here
        "quantiles":            [0.15, 0.30, 0.50, 0.65, 0.80],
        "weight_k":             0.15,
    },
    "EC-Earth3": {
        "hist_predictor_dir":   "data/EC-Earth3/historical",
        "ssp245_predictor_dir": "data/EC-Earth3/ssp245",
        "ssp585_predictor_dir": "data/EC-Earth3/ssp585",
        "output_dir":           "models/EC-Earth3",
        "quantiles":            [0.15, 0.35, 0.50, 0.65, 0.75],
        "weight_k":             0.10,
    },
    "INM-CM5-0": {
        "hist_predictor_dir":   "data/INM-CM5-0/historical",
        "ssp245_predictor_dir": "data/INM-CM5-0/ssp245",
        "ssp585_predictor_dir": "data/INM-CM5-0/ssp585",
        "output_dir":           "models/INM-CM5-0",
        "quantiles":            [0.05, 0.15, 0.40, 0.60, 0.70],
        "weight_k":             0.10,
    }
}

# Shared settings
TARGET_FILE         = "data/target/historical_rainfall_05deg.nc"
PREDICTOR_VARS      = ['clt', 'huss', 'hurs', 'pr', 'psl', 'sfcWind',
                       'rlds', 'rsds', 'tas', 'uas', 'vas', 'wap']
PREDICTOR_SHAPE     = (3, 3, len(PREDICTOR_VARS))
TARGET_SHAPE        = (20, 18, 1)
EPOCHS              = 100
BATCH_SIZE          = 32
RAIN_THRESHOLD      = 1.0


# ────────────────────────────────────────────────
#  2. SHARED UTILITY FUNCTIONS
# ────────────────────────────────────────────────

def load_and_align_data(directory, var_name=None, target_time=None):
    """Load multiple .nc files and align them."""
    files = sorted(f for f in os.listdir(directory) if f.endswith('.nc'))
    ds = xr.open_mfdataset([os.path.join(directory, f) for f in files],
                           combine='by_coords', compat='override')
    for dim in ['height', 'lev', 'level']:
        if dim in ds.dims:
            ds = ds.drop_dims(dim)
    if var_name:
        da = ds[var_name]
        if target_time is not None:
            da = da.sel(time=target_time, method='nearest')
        return da
    return ds


def preprocess(X, y=None):
    """Standardize predictors; log1p transform target if provided."""
    X_norm = (X - np.mean(X, axis=(0,1,2), keepdims=True)) / \
             (np.std(X, axis=(0,1,2), keepdims=True) + 1e-8)
    if y is not None:
        return X_norm, np.log1p(y)
    return X_norm


def residual_block(x, filters, kernel_size=(3,3)):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])
    x = ReLU()(x)
    return x


def build_downscaling_cnn():
    inputs = Input(shape=PREDICTOR_SHAPE)
    x = Conv2D(64, (3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = UpSampling2D((10,10), interpolation='bilinear')(x)
    crop_vert = (x.shape[1] - TARGET_SHAPE[0]) // 2
    crop_horz = (x.shape[2] - TARGET_SHAPE[1]) // 2
    x = Cropping2D(((crop_vert, crop_vert), (crop_horz, crop_horz)))(x)
    x = residual_block(x, 64)
    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    output = Conv2D(1, (3,3), padding='same', activation='linear', name='pr')(x)
    model = Model(inputs, output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(delta=10),
        weighted_metrics=[MeanAbsoluteError(name='mae'), MeanSquaredError(name='mse')]
    )
    return model


def calculate_sample_weights(y_log, bins, weights):
    y_orig = np.expm1(y_log).squeeze()
    bin_idx = np.digitize(y_orig, bins=bins, right=False)
    bin_idx = np.clip(bin_idx, 0, len(weights)-1)
    return np.array(weights)[bin_idx][..., np.newaxis]


def save_netcdf(data, time_coords, lat, lon, filename, outdir, is_prediction=True):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    if is_prediction:
        data = np.expm1(data).squeeze()
        data[data < RAIN_THRESHOLD] = 0.0
        data = np.round(data, 1)
    else:
        data = data.squeeze()
    ds = xr.Dataset(
        {'pr': (('time', 'lat', 'lon'), data)},
        coords={'time': time_coords, 'lat': lat, 'lon': lon}
    )
    ds.to_netcdf(path)
    print(f"→ Saved: {path}")


# ────────────────────────────────────────────────
#  3. MAIN PROCESSING – per GCM
# ────────────────────────────────────────────────

# Load target once (shared across all GCMs)
target_ds = load_and_align_data(TARGET_FILE, 'rainfall')
target_data = target_ds.values[..., np.newaxis]
target_time = target_ds.time.values
target_lat  = target_ds.lat.values
target_lon  = target_ds.lon.values

for gcm, cfg in GCM_CONFIGS.items():
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━ {gcm} ━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    outdir = cfg["output_dir"]
    os.makedirs(outdir, exist_ok=True)

    # ── Compute GCM-specific bins & exponential weights ──
    nonzero = target_data[target_data > 0].flatten()
    qvals = np.quantile(nonzero, cfg["quantiles"]).tolist()
    rain_bins = [RAIN_THRESHOLD] + qvals
    mid_points = [0.0] + [(rain_bins[i] + rain_bins[i+1])/2 for i in range(len(rain_bins)-1)] + [rain_bins[-1] * 1.5]
    raw_w = [0.0 if m == 0 else np.exp(cfg["weight_k"] * m) for m in mid_points]
    rain_weights = np.round([0.0] + raw_w[1:], 3).tolist()

    print(f"  Bins:    {np.round(rain_bins, 2)}")
    print(f"  Weights: {rain_weights}\n")

    # ── Historical predictors & 5-fold training ──
    pred_data = []
    for var in PREDICTOR_VARS:
        da = load_and_align_data(cfg["hist_predictor_dir"], var, target_time)
        if len(da.dims) > 3:
            da = da.isel({d:0 for d in da.dims if d not in ['time','lat','lon']})
        pred_data.append(da.values)
    pred_data = np.stack(pred_data, axis=-1)
    X_hist = preprocess(pred_data)

    kf = KFold(n_splits=5, shuffle=False)
    models = []

    for fold, (train_idx, _) in enumerate(kf.split(pred_data), 1):
        print(f"  Fold {fold}/5")
        X_tr = pred_data[train_idx]
        y_tr = target_data[train_idx]
        X_tr, y_tr_log = preprocess(X_tr, y_tr)
        tr_weights = calculate_sample_weights(y_tr_log, rain_bins, rain_weights)

        model = build_downscaling_cnn()
        model.fit(
            X_tr, y_tr_log,
            sample_weight=tr_weights,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        models.append(model)

    # Save historical downscaled product (fold ensemble mean)
    preds_hist = [m.predict(X_hist) for m in models]
    mean_hist = np.mean(preds_hist, axis=0)
    save_netcdf(mean_hist, target_time, target_lat, target_lon,
                f"{gcm}_historical_downscaled.nc", outdir)

    # ── Future scenarios (SSP2-4.5 & SSP5-8.5) ──
    for scen, dir_key in [("ssp245", "ssp245_predictor_dir"),
                          ("ssp585", "ssp585_predictor_dir")]:
        print(f"  → {scen.upper()}")
        pred_data = []
        for var in PREDICTOR_VARS:
            da = load_and_align_data(cfg[dir_key], var)
            if len(da.dims) > 3:
                da = da.isel({d:0 for d in da.dims if d not in ['time','lat','lon']})
            pred_data.append(da.values)
        pred_data = np.stack(pred_data, axis=-1)
        X_fut = preprocess(pred_data)
        fut_time = da.time.values   # time from last variable

        preds_fut = [m.predict(X_fut) for m in models]
        mean_fut = np.mean(preds_fut, axis=0)
        save_netcdf(mean_fut, fut_time, target_lat, target_lon,
                    f"{gcm}_{scen}_downscaled.nc", outdir)

    print(f"  Completed {gcm}\n")


# ────────────────────────────────────────────────
#  4. MULTI-GCM ENSEMBLES
# ────────────────────────────────────────────────

ENSEMBLE_ROOT = "ensembles"
os.makedirs(ENSEMBLE_ROOT, exist_ok=True)

def create_multi_model_ensembles(scenario: str):
    print(f"\n→ Creating {scenario.upper()} multi-model ensembles")

    outdir = os.path.join(ENSEMBLE_ROOT, scenario)
    os.makedirs(outdir, exist_ok=True)

    preds = {}
    for gcm, cfg in GCM_CONFIGS.items():
        fname = f"{gcm}_historical_downscaled.nc" if scenario == "historical" else \
                f"{gcm}_{scenario}_downscaled.nc"
        path = os.path.join(cfg["output_dir"], fname)
        if os.path.isfile(path):
            ds = xr.open_dataset(path)
            preds[gcm] = ds['pr']

    if len(preds) == 0:
        print("  No files found → skipping")
        return

    da = xr.concat(list(preds.values()), dim="gcm")
    da.attrs["gcms"] = ", ".join(preds.keys())

    # Full 3-model mean
    da.mean("gcm").to_netcdf(os.path.join(outdir, "ensemble-mean_3models.nc"))

    # Pairwise means
    for pair in combinations(preds.keys(), 2):
        pair_name = "_".join(sorted(pair))
        xr.concat([preds[a] for a in pair], dim="gcm").mean("gcm").to_netcdf(
            os.path.join(outdir, f"ensemble-mean_{pair_name}.nc")
        )

    print(f"  Saved in: {outdir}\n")


# Run for all periods
for scen in ["historical", "ssp245", "ssp585"]:
    create_multi_model_ensembles(scen)


print("\nAll downscaling and multi-model ensemble steps completed.\n")