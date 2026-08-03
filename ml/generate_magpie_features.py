#!/usr/bin/env python3
"""
Generate MAGPIE features for the 3DSC_MP dataset using matminer.
Parses chemical formulas via pymatgen and computes 132 MAGPIE descriptors
per composition. Saves the enriched dataset with MAGPIE columns appended.

Input:  data/3dsc_data/3DSC_MP.csv  (raw 3DSC dataset)
Output: data/3dsc_data/3DSC_MP_with_MAGPIE.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "data" / "3dsc_data" / "3DSC_MP.csv"
OUTPUT = PROJECT_ROOT / "data" / "3dsc_data" / "3DSC_MP_with_MAGPIE.csv"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET, header=1)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    print("Parsing compositions from formula_sc...")
    compositions = []
    failed = []
    for i, formula in enumerate(tqdm(df['formula_sc'], desc="Parsing")):
        try:
            comp = Composition(formula)
            compositions.append(comp)
        except Exception as e:
            print(f"  Failed to parse '{formula}': {e}")
            compositions.append(None)
            failed.append(i)

    if failed:
        print(f"WARNING: {len(failed)} formulas failed to parse, dropping them")
        df = df.drop(index=failed).reset_index(drop=True)
        compositions = [c for c in compositions if c is not None]

    df['composition'] = compositions

    print("Generating MAGPIE features (this may take a few minutes)...")
    magpie = ElementProperty.from_preset("magpie")
    magpie_features = magpie.featurize_dataframe(
        df, col_id='composition', ignore_errors=True, inplace=False
    )

    magpie_cols = [c for c in magpie_features.columns if c not in df.columns]
    print(f"Generated {len(magpie_cols)} MAGPIE features")

    rename_map = {}
    for col in magpie_cols:
        new_name = f"MAGPIE_{col.replace(' ', '_')}"
        rename_map[col] = new_name
    magpie_features = magpie_features.rename(columns=rename_map)
    magpie_cols_renamed = list(rename_map.values())

    nan_counts = magpie_features[magpie_cols_renamed].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"WARNING: {len(nan_cols)} MAGPIE columns have NaN values:")
        for col, count in nan_cols.items():
            print(f"  {col}: {count} NaN")
        nan_rows = magpie_features[magpie_cols_renamed].isna().any(axis=1)
        print(f"Dropping {nan_rows.sum()} rows with NaN MAGPIE features")
        magpie_features = magpie_features[~nan_rows].reset_index(drop=True)

    magpie_features = magpie_features.drop(columns=['composition'])

    print(f"Final dataset: {len(magpie_features)} rows, {len(magpie_features.columns)} columns")
    print(f"  Original columns: {len(df.columns) - 1}")
    print(f"  MAGPIE columns: {len(magpie_cols_renamed)}")

    magpie_features.to_csv(OUTPUT, index=False)
    print(f"Saved to: {OUTPUT}")

    print("\nMAGPIE feature names:")
    for i, col in enumerate(magpie_cols_renamed, 1):
        print(f"  {i:3d}. {col}")


if __name__ == '__main__':
    main()
