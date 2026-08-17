# Superconductor Phase-Coherence Screening via RCSJ-Kuramoto Simulation

Code repository for reproducing all simulation results, figures, and statistical analyses in:

> S. Uttamchandani and K. Walker, "Phase Coherence and Vortex Dynamics in Superconducting Grain Networks: An RCSJ-Kuramoto Simulation," *Journal of High School Science*, 2026. (Submitted)

## Overview

This project uses stochastic Kuramoto phase-oscillator simulations derived from the Resistively and Capacitively Shunted Junction (RCSJ) circuit model to study phase coherence in two-dimensional models of superconducting grain networks. The simulations probe:

1. **Noise-driven coherence-to-disorder crossover** consistent with BKT vortex unbinding
2. **Coordination number and coupling disorder effects** on vortex suppression
3. **Composition-score bridge** connecting chemical uniformity to simulated phase coherence at 77 K

## Repository Structure

```
superconductor-ml-kuramoto-framework/
├── simulation/
│   ├── run_enhanced_simulations.py      # JHSS paper: noise sweep, finite-size scaling,
│   │                                    #   time-series, Tables 1-2, Kruskal-Wallis, sensitivity
│   ├── generate_bkt_lattice_figures.py  # BKT phase transition, lattice geometry, composition
│   └── dataset_kuramoto_bridge.py       # Composition-score bridge using real datasets
├── ml/
│   ├── train_xgboost_3dsc.py            # XGBoost on 3DSC with GroupShuffleSplit
│   ├── train_svr_3dsc.py               # SVR on 3DSC with GroupShuffleSplit
│   ├── train_nn_3dsc.py                # MLP on 3DSC with GroupShuffleSplit
│   ├── train_xgboost_uci.py            # XGBoost on UCI/SuperCon (baseline)
│   └── generate_magpie_features.py      # MAGPIE feature generation via matminer
├── visualization/
│   ├── plot_ml_performance.py           # ML performance comparison
│   └── plot_feature_importance.py       # XGBoost feature importance
├── data/
│   ├── README.md                        # Dataset download instructions
│   ├── hamideih_data/                   # UCI/SuperCon data (user-provided)
│   └── 3dsc_data/                       # 3DSC data (user-provided)
├── figures/                             # Generated figure output
├── analysis_results/                    # Generated analysis output (JSON, CSV)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/sohumu0/superconductor-ml-kuramoto-framework.git
cd superconductor-ml-kuramoto-framework

# For macOS and Linux:
python3 -m venv venv && source venv/bin/activate

# For Windows:
# python -m venv venv && .\venv\Scripts\activate

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

### Figures 1, 5, 6: BKT transition, lattice geometry, composition heatmaps

```bash
python simulation/generate_bkt_lattice_figures.py
```

Generates `fig_bkt.png` (Figure 1), `fig_lattice.png` (Figure 5), and `fig_composition.png` (Figure 6) in `figures/`.

### Figures 2-4, Tables 1-2, statistical tests

```bash
python simulation/run_enhanced_simulations.py
```

This single script runs all quantitative analyses reported in the paper:

| Output | Paper element | Description |
|--------|---------------|-------------|
| `fig_noise_sweep.png` | Figure 2 | Dense 10-point noise sweep, 3 geometries, 10 seeds |
| `fig_timeseries.png` | Figure 3 | Time-series convergence over 5,000 steps |
| `fig_finite_size.png` | Figure 4 | Finite-size scaling, L = 16 to 128, 8 seeds |
| `simulation_results.json` | Tables 1-2 | All numerical values with standard deviations |
| (console output) | Section 4.4 | Kruskal-Wallis H = 3.85, p = 0.15; ANOVA F = 2.83, p = 0.07 |
| (console output) | Section 4.4 | Sensitivity analysis on S_comp mapping slope (+/-30%) |

### Dataset-Kuramoto bridge (requires datasets)

```bash
python simulation/dataset_kuramoto_bridge.py
```

### ML benchmarking

```bash
python ml/train_xgboost_3dsc.py
python ml/train_svr_3dsc.py
python ml/train_nn_3dsc.py
python ml/train_xgboost_uci.py
```

## Key Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K_0 (coupling) | 0.01 | Places BKT crossover near T~ ~ 1 |
| beta_c | < 1 | Overdamped RCSJ regime |
| dt | 0.02 | Euler-Maruyama time step |
| Steps | 5,000 | Integration steps per simulation |
| Lattice sizes | 16-128 (scaling), 24 (geometry), 32 (composition) | Square grid side length |
| Noise range | T~ = 0.1 to 2.0 | Dimensionless thermal noise |
| Operating temperature | 77 K | Liquid nitrogen (composition analysis) |
| Seeds | 8-20 per condition | Deterministic for reproducibility |

## Runtime Estimates

- `generate_bkt_lattice_figures.py`: ~2 minutes
- `run_enhanced_simulations.py`: ~15-30 minutes (128x128 finite-size scaling dominates)
- `dataset_kuramoto_bridge.py`: ~1 minute (requires datasets)
  
## License

This project is released for academic and research purposes.
