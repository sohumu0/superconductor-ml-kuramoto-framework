#!/usr/bin/env python3
"""
Support Vector Regression for predicting superconductor Tc from the 3DSC_MP dataset.
Uses MAGPIE + structural features with GroupShuffleSplit cross-validation.

Outputs: R-squared, MAE, RMSE, MSLE, SMAPE (train & test).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "data" / "3dsc_data" / "3DSC_MP_with_MAGPIE.csv"

RANDOM_SEED = 58
N_REPS = 10
TRAIN_FRAC = 0.8
TARGET = 'tc'
GROUP_COL = 'chemical_composition_sc'
WEIGHT_COL = 'weight'


def get_features(df):
    magpie = [c for c in df.columns if c.startswith('MAGPIE_')]
    electronic = ['band_gap_2', 'energy_per_atom_2', 'formation_energy_per_atom_2',
                  'total_magnetization_2', 'num_unique_magnetic_sites_2', 'true_total_magnetization_2']
    lattice = ['lata_2', 'latb_2', 'latc_2']
    symmetry = ['cubic', 'hexagonal', 'monoclinic', 'orthorhombic', 'tetragonal',
                'triclinic', 'trigonal', 'primitive', 'base-centered', 'body-centered', 'face-centered']
    other = ['crystal_temp_2', 'density_2', 'e_above_hull_2', 'efermi_2',
             'cell_volume_2', 'nsites_2', 'num_elements_sc']
    features = magpie + electronic + lattice + symmetry + other
    return [f for f in features if f in df.columns]


def arcsinh_transform(y):
    return np.arcsinh(y)

def sinh_transform(y):
    return np.sinh(y)

def smape(y_true, y_pred, min_tc=0):
    mask = (y_true > min_tc) & (y_pred > min_tc)
    if mask.sum() == 0:
        return 0.0
    diff = np.abs(y_true[mask] - y_pred[mask])
    denom = y_true[mask] + y_pred[mask]
    return np.mean(diff / denom)

def evaluate(y_true, y_pred, sample_weight=None):
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
    mae = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
    try:
        msle = mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight)
    except ValueError:
        y_pred_clipped = np.clip(y_pred, 0, None)
        msle = mean_squared_log_error(y_true, y_pred_clipped, sample_weight=sample_weight)
    s = smape(y_true, y_pred)
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MSLE': msle, 'SMAPE': s}


def main():
    np.random.seed(RANDOM_SEED)
    df = pd.read_csv(DATASET)
    FEATURES = get_features(df)
    print(f"Dataset: {len(df)} samples, {len(FEATURES)} features "
          f"({len([c for c in FEATURES if c.startswith('MAGPIE')])} MAGPIE + "
          f"{len(FEATURES) - len([c for c in FEATURES if c.startswith('MAGPIE')])} structural)")

    X = df[FEATURES].values
    y = df[TARGET].values
    groups = df[GROUP_COL].values
    weights = df[WEIGHT_COL].values

    scaler_X = StandardScaler()
    splitter = GroupShuffleSplit(n_splits=N_REPS, train_size=TRAIN_FRAC, random_state=RANDOM_SEED)

    all_train_scores = []
    all_test_scores = []

    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train, w_test = weights[train_idx], weights[test_idx]

        X_train_sc = scaler_X.fit_transform(X_train)
        X_test_sc = scaler_X.transform(X_test)

        y_train_t = arcsinh_transform(y_train)

        model = SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale')
        model.fit(X_train_sc, y_train_t, sample_weight=w_train)

        y_pred_train = sinh_transform(model.predict(X_train_sc))
        y_pred_test = sinh_transform(model.predict(X_test_sc))
        y_pred_train = np.clip(y_pred_train, 0, None)
        y_pred_test = np.clip(y_pred_test, 0, None)

        train_scores = evaluate(y_train, y_pred_train, w_train)
        test_scores = evaluate(y_test, y_pred_test, w_test)
        all_train_scores.append(train_scores)
        all_test_scores.append(test_scores)

        print(f"  Rep {i+1}/{N_REPS}: test R2={test_scores['R2']:.4f}, MAE={test_scores['MAE']:.4f}")

    print("\n" + "="*70)
    print("SVR Results (mean +/- std over {} repetitions)".format(N_REPS))
    print("="*70)
    for metric in ['R2', 'MAE', 'RMSE', 'MSLE', 'SMAPE']:
        train_vals = [s[metric] for s in all_train_scores]
        test_vals = [s[metric] for s in all_test_scores]
        print(f"  {metric:>6s}:  Train {np.mean(train_vals):.4f} +/- {np.std(train_vals):.4f}  |  "
              f"Test {np.mean(test_vals):.4f} +/- {np.std(test_vals):.4f}")


if __name__ == '__main__':
    main()
