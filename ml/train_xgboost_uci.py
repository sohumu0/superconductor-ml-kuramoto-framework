#!/usr/bin/env python3
"""
XGBoost regression on the UCI/SuperCon (Hamidieh) dataset.
25-iteration Monte Carlo cross-validation with the paper's tuned hyperparameters.
Demonstrates the high R-squared (>0.90) achievable on the compositionally redundant
UCI dataset — contrasted with the ~0.50 ceiling on the structurally diverse 3DSC dataset.

Outputs: RMSE, R-squared, and top-15 feature importance plot.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET = PROJECT_ROOT / "data" / "hamideih_data" / "train.csv"
FIGURE_DIR = PROJECT_ROOT / "figures"


def main():
    df = pd.read_csv(DATASET)
    X = df.drop(columns=['critical_temp'])
    y = df['critical_temp']

    print("Starting XGBoost 25-iteration Cross-Validation...")

    mses = []
    r2s = []
    feature_importances = np.zeros(X.shape[1])

    for i in range(25):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42 + i)

        model = XGBRegressor(
            n_estimators=374,
            max_depth=16,
            learning_rate=0.02,
            colsample_bytree=0.5,
            min_child_weight=1,
            random_state=42 + i,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mses.append(mean_squared_error(y_test, preds))
        r2s.append(r2_score(y_test, preds))
        feature_importances += model.feature_importances_

    final_rmse = np.sqrt(np.mean(mses))
    final_r2 = np.mean(r2s)
    feature_importances /= 25

    print("\n=== XGBoost Results (UCI/SuperCon) ===")
    print(f"RMSE: {final_rmse:.2f} K")
    print(f"R2:   {final_r2:.3f}")

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    print("\n=== Top 15 Most Important Features ===")
    print(importance_df.head(15).to_string(index=False))

    top15 = importance_df.head(15).copy()
    top15['Relative Importance'] = top15['Importance'] / top15['Importance'].sum()
    top15 = top15.sort_values('Relative Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        top15['Feature'], top15['Relative Importance'],
        color=plt.cm.Blues_r(np.linspace(0.2, 0.8, len(top15))),
        edgecolor='white', linewidth=0.6, height=0.72,
    )
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}", va='center', ha='left', fontsize=9.5, color='#333333')

    ax.set_xlabel("Relative Importance (XGBoost Gain)", fontsize=12, labelpad=8)
    ax.set_title("Top 15 Features (XGBoost, UCI/SuperCon, 25-fold CV)",
                 fontsize=13, fontweight='bold', pad=14)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0, top15['Relative Importance'].max() * 1.18)
    plt.tight_layout()

    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / "feature_importance_uci_xgboost.png", dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {FIGURE_DIR / 'feature_importance_uci_xgboost.png'}")


if __name__ == "__main__":
    main()
