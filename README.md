# Superconductor Screening via Materials Informatics and Phase-Dynamics Simulation

Code and analysis for a two-stage superconductor screening framework that combines machine learning critical-temperature prediction with RCSJ-derived Kuramoto phase-coherence simulation.

**Stage 1 (Materials Informatics):** XGBoost, SVR, and MLP models trained on the 3DSC dataset (5,773 samples, 159 structural + MAGPIE features) with GroupShuffleSplit cross-validation to prevent data leakage from compositional redundancy. Benchmarked against the UCI/SuperCon dataset to quantify the feature-intrinsic predictive ceiling (R-squared ~ 0.50 on 3DSC vs. > 0.90 on UCI).

**Stage 2 (Phase-Dynamics Simulation):** Kuramoto phase-oscillator equations derived from the RCSJ circuit model under the overdamped approximation (Stewart-McCumber parameter beta_c < 1). Stochastic simulations on 2D lattices with honeycomb, square, and triangular coordination probe the Berezinskii-Kosterlitz-Thouless (BKT) vortex-unbinding crossover and the effect of lattice geometry and coupling disorder on phase coherence.

## Repository Structure

```
superconductor-screening/
├── simulation/
│   ├── generate_bkt_lattice_figures.py   # BKT transition, lattice geometry, composition figures
│   └── dataset_kuramoto_bridge.py        # Composition-score to Kuramoto bridge (Figure 6)
├── ml/
│   ├── train_xgboost_3dsc.py             # XGBoost on 3DSC with GroupShuffleSplit
│   ├── train_svr_3dsc.py                 # SVR on 3DSC with GroupShuffleSplit
│   ├── train_nn_3dsc.py                  # MLP on 3DSC with GroupShuffleSplit
│   ├── train_xgboost_uci.py             # XGBoost on UCI/SuperCon (baseline)
│   └── generate_magpie_features.py       # MAGPIE feature generation via matminer
├── visualization/
│   ├── plot_ml_performance.py            # Figure 2: ML performance comparison
│   └── plot_feature_importance.py        # Figure 3: XGBoost feature importance
├── data/
│   ├── README.md                         # Dataset download instructions
│   ├── hamideih_data/                    # UCI/SuperCon data (user-provided)
│   └── 3dsc_data/                        # 3DSC data (user-provided)
├── figures/                              # Generated figure output
├── analysis_results/                     # Generated analysis CSV output
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd superconductor-screening
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data

The datasets are too large for Git. See [`data/README.md`](data/README.md) for download instructions.

**UCI/SuperCon** (Hamidieh 2018): Place `train.csv` and `unique_m.csv` in `data/hamideih_data/`.

**3DSC** (Sommer et al. 2023): Place `3DSC_MP_with_MAGPIE.csv` in `data/3dsc_data/`. To generate from raw data, run:
```bash
python ml/generate_magpie_features.py
```

## Reproducing Results

### ML benchmarking (Stage 1)

Train all three models on 3DSC with grouped cross-validation:
```bash
python ml/train_xgboost_3dsc.py
python ml/train_svr_3dsc.py
python ml/train_nn_3dsc.py
```

UCI/SuperCon baseline (demonstrates inflated R-squared from compositional redundancy):
```bash
python ml/train_xgboost_uci.py
```

### Phase-dynamics simulation (Stage 2)

Generate BKT transition, lattice geometry, and composition figures:
```bash
python simulation/generate_bkt_lattice_figures.py
```

Run the dataset-Kuramoto bridge analysis (requires both datasets):
```bash
python simulation/dataset_kuramoto_bridge.py
```

### Visualization

Generate publication figures (requires analysis_results/ from ML training):
```bash
python visualization/plot_ml_performance.py
python visualization/plot_feature_importance.py
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K (coupling) | 0.01 per bond | Places BKT crossover within accessible noise range |
| beta_c | < 1 | Overdamped RCSJ regime |
| Lattice size | 32 x 32 (BKT, composition), 24 x 24 (lattice) | Square grid |
| Noise range | T_tilde = [0.05, 1.00, 2.00] | Dimensionless thermal noise |
| GroupShuffleSplit | 10 splits, 80/20 | Prevents compositional data leakage |
| Operating temperature | 77 K | Liquid nitrogen (composition analysis) |

## License

This project is released for academic and research purposes.
