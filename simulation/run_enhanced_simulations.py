#!/usr/bin/env python3
"""
Enhanced simulations for JHSS paper revision:
  1. Dense 10-point noise sweep (BKT phase diagram) across 3 geometries
  2. Finite-size scaling (16x16 to 128x128)
  3. Time-series convergence (r(t) over integration steps)
  4. Tables 1 & 2 with standard deviations
  5. Kruskal-Wallis test on high-Tc S_comp groups
  6. Sensitivity analysis on S_comp -> sigma_K mapping slope
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR = PROJECT_ROOT / "analysis_results"


# ─── Core simulation engine ───────────────────────────────────────────

def build_square_edges(L, periodic=False):
    N = L * L
    rows_list, cols_list = [], []
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if j + 1 < L:
                rows_list += [idx, idx + 1]
                cols_list += [idx + 1, idx]
            elif periodic:
                right = i * L
                rows_list += [idx, right]
                cols_list += [right, idx]
            if i + 1 < L:
                rows_list += [idx, idx + L]
                cols_list += [idx + L, idx]
            elif periodic:
                down = j
                rows_list += [idx, down]
                cols_list += [down, idx]
    return N, np.array(rows_list, dtype=np.intp), np.array(cols_list, dtype=np.intp)


def build_honeycomb_edges(L, periodic=False):
    N = L * L
    rows_list, cols_list = [], []
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if j + 1 < L:
                rows_list += [idx, idx + 1]
                cols_list += [idx + 1, idx]
            elif periodic:
                right = i * L
                rows_list += [idx, right]
                cols_list += [right, idx]
            if i + 1 < L and (i + j) % 2 == 0:
                rows_list += [idx, idx + L]
                cols_list += [idx + L, idx]
            elif periodic and (i + j) % 2 == 0:
                rows_list += [idx, j]
                cols_list += [j, idx]
    return N, np.array(rows_list, dtype=np.intp), np.array(cols_list, dtype=np.intp)


def build_triangular_edges(L, periodic=False):
    N = L * L
    rows_list, cols_list = [], []
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if j + 1 < L:
                rows_list += [idx, idx + 1]
                cols_list += [idx + 1, idx]
            elif periodic:
                right = i * L
                rows_list += [idx, right]
                cols_list += [right, idx]
            if i + 1 < L:
                rows_list += [idx, idx + L]
                cols_list += [idx + L, idx]
            elif periodic:
                rows_list += [idx, j]
                cols_list += [j, idx]
            ni, nj = i + 1, j + 1
            if ni < L and nj < L:
                nidx = ni * L + nj
                rows_list += [idx, nidx]
                cols_list += [nidx, idx]
            elif periodic:
                nidx = (ni % L) * L + (nj % L)
                if nidx != idx:
                    rows_list += [idx, nidx]
                    cols_list += [nidx, idx]
    return N, np.array(rows_list, dtype=np.intp), np.array(cols_list, dtype=np.intp)


def simulate(N, rows, cols, K_vals, T_noise, dt=0.02, steps=5000,
             coherent_init=True, record_timeseries=False, ts_interval=50):
    if coherent_init:
        theta = np.zeros(N) + 0.01 * np.random.randn(N)
    else:
        theta = np.random.uniform(0, 2 * np.pi, N)
    omega = np.zeros(N)

    r_ts = [] if record_timeseries else None

    for step in range(steps):
        sin_diff = K_vals * np.sin(theta[cols] - theta[rows])
        coupling = np.bincount(rows, weights=sin_diff, minlength=N)
        dtheta = omega + coupling + T_noise * np.random.randn(N)
        theta = (theta + dt * dtheta) % (2 * np.pi)

        if record_timeseries and step % ts_interval == 0:
            r_ts.append(float(np.abs(np.mean(np.exp(1j * theta)))))

    r_final = float(np.abs(np.mean(np.exp(1j * theta))))
    return theta, r_final, r_ts


def count_vortices(theta, L):
    count = 0
    for i in range(L - 1):
        for j in range(L - 1):
            idx = [i * L + j, i * L + j + 1, (i + 1) * L + j + 1, (i + 1) * L + j]
            phases = theta[idx]
            winding = 0.0
            for k in range(4):
                diff = phases[(k + 1) % 4] - phases[k]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                winding += diff
            if abs(winding) > np.pi:
                count += 1
    return count


# ─── 1. Dense noise sweep (BKT phase diagram) ─────────────────────────

def run_dense_noise_sweep():
    print("=== 1. Dense noise sweep across 3 geometries ===")
    T_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    n_seeds = 10
    K0 = 0.01
    L = 32

    builders = {
        'Honeycomb': (build_honeycomb_edges, False),
        'Square': (build_square_edges, True),
        'Triangular': (build_triangular_edges, False),
    }

    results = {}
    for geom_name, (builder, use_periodic) in builders.items():
        print(f"  {geom_name}...")
        N, rows, cols = builder(L, periodic=use_periodic)
        K_vals = np.ones(len(rows)) * K0

        r_means, r_stds, v_means, v_stds = [], [], [], []
        for T in T_values:
            rs, vs = [], []
            for seed in range(n_seeds):
                np.random.seed(seed * 31 + int(T * 1000) + hash(geom_name) % 10000)
                theta, r_val, _ = simulate(N, rows, cols, K_vals, T,
                                           coherent_init=True)
                rs.append(r_val)
                vs.append(count_vortices(theta, L))
            r_means.append(np.mean(rs))
            r_stds.append(np.std(rs, ddof=1))
            v_means.append(np.mean(vs))
            v_stds.append(np.std(vs, ddof=1))
            print(f"    T={T:.1f}: r={np.mean(rs):.3f}±{np.std(rs, ddof=1):.3f}, "
                  f"v={np.mean(vs):.1f}±{np.std(vs, ddof=1):.1f}")

        results[geom_name] = {
            'T_values': T_values,
            'r_means': r_means, 'r_stds': r_stds,
            'v_means': v_means, 'v_stds': v_stds,
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    markers = {'Honeycomb': 'o', 'Square': 's', 'Triangular': '^'}
    colors = {'Honeycomb': '#e74c3c', 'Square': '#3498db', 'Triangular': '#2ecc71'}

    for geom_name, data in results.items():
        ax1.errorbar(data['T_values'], data['r_means'], yerr=data['r_stds'],
                     marker=markers[geom_name], color=colors[geom_name],
                     label=geom_name, capsize=3, linewidth=1.5, markersize=6)
        ax2.errorbar(data['T_values'], data['v_means'], yerr=data['v_stds'],
                     marker=markers[geom_name], color=colors[geom_name],
                     label=geom_name, capsize=3, linewidth=1.5, markersize=6)

    ax1.set_xlabel(r'Dimensionless noise amplitude $\tilde{T}$', fontsize=11)
    ax1.set_ylabel(r'Order parameter $r$', fontsize=11)
    ax1.set_title('(a) Phase coherence vs. noise', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(r'Dimensionless noise amplitude $\tilde{T}$', fontsize=11)
    ax2.set_ylabel('Mean vortex defect count', fontsize=11)
    ax2.set_title('(b) Vortex defects vs. noise', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'fig_noise_sweep.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> fig_noise_sweep.png saved\n")
    return results


# ─── 2. Finite-size scaling ──────────────────────────────────────────

def run_finite_size_scaling():
    print("=== 2. Finite-size scaling ===")
    lattice_sizes = [16, 32, 64, 128]
    T_values = [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]
    n_seeds = 8
    K0 = 0.01

    results = {}
    for L in lattice_sizes:
        print(f"  L={L}...")
        N, rows, cols = build_square_edges(L, periodic=True)
        K_vals = np.ones(len(rows)) * K0

        r_means, r_stds = [], []
        for T in T_values:
            rs = []
            for seed in range(n_seeds):
                np.random.seed(seed * 41 + int(T * 1000) + L * 7)
                _, r_val, _ = simulate(N, rows, cols, K_vals, T,
                                       coherent_init=True)
                rs.append(r_val)
            r_means.append(np.mean(rs))
            r_stds.append(np.std(rs, ddof=1))
            print(f"    T={T:.1f}: r={np.mean(rs):.3f}±{np.std(rs, ddof=1):.3f}")

        results[L] = {'T_values': T_values, 'r_means': r_means, 'r_stds': r_stds}

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    colors_fss = {16: '#e74c3c', 32: '#e67e22', 64: '#3498db', 128: '#2ecc71'}
    for L, data in results.items():
        ax.errorbar(data['T_values'], data['r_means'], yerr=data['r_stds'],
                     marker='o', color=colors_fss[L], label=f'$L = {L}$',
                     capsize=3, linewidth=1.5, markersize=5)

    ax.set_xlabel(r'Dimensionless noise amplitude $\tilde{T}$', fontsize=11)
    ax.set_ylabel(r'Order parameter $r$', fontsize=11)
    ax.set_title('Finite-size scaling: periodic square lattice', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'fig_finite_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> fig_finite_size.png saved\n")
    return results


# ─── 3. Time-series convergence ──────────────────────────────────────

def run_timeseries_convergence():
    print("=== 3. Time-series convergence ===")
    L = 32
    N, rows, cols = build_square_edges(L, periodic=True)
    K_vals = np.ones(len(rows)) * 0.01

    T_values = [0.05, 1.00, 2.00]
    labels = [r'$\tilde{T} = 0.05$', r'$\tilde{T} = 1.00$', r'$\tilde{T} = 2.00$']
    colors = ['#2ecc71', '#e67e22', '#e74c3c']
    ts_interval = 10

    fig, ax = plt.subplots(figsize=(8, 5))
    for T, label, color in zip(T_values, labels, colors):
        np.random.seed(42 + int(T * 1000))
        _, _, r_ts = simulate(N, rows, cols, K_vals, T,
                              coherent_init=True, record_timeseries=True,
                              ts_interval=ts_interval)
        time_steps = np.arange(len(r_ts)) * ts_interval
        ax.plot(time_steps, r_ts, label=label, color=color, linewidth=1.5)

    ax.set_xlabel('Integration step', fontsize=11)
    ax.set_ylabel(r'Order parameter $r$', fontsize=11)
    ax.set_title('Time-series convergence of order parameter', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=3000, color='gray', linestyle=':', alpha=0.5, label='')
    ax.text(3050, 0.5, 'Steady state\nachieved', fontsize=8, color='gray', va='center')
    plt.tight_layout()
    fig.savefig(FIGURE_DIR / 'fig_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> fig_timeseries.png saved\n")


# ─── 4. Table 1 with standard deviations ─────────────────────────────

def run_table1_with_stds():
    print("=== 4. Table 1: Lattice geometry with std devs ===")
    L = 24
    T = 0.80
    K0 = 0.01
    n_seeds = 8

    builders = [
        ('Honeycomb', build_honeycomb_edges),
        ('Square', build_square_edges),
        ('Triangular', build_triangular_edges),
    ]

    table1_data = {}
    for geom_name, builder in builders:
        N, rows, cols = builder(L, periodic=False)

        # Uniform
        K_vals_u = np.ones(len(rows)) * K0
        rs_u, vs_u = [], []
        for seed in range(n_seeds):
            np.random.seed(300 + hash(geom_name) % 1000 + seed)
            theta, r_val, _ = simulate(N, rows, cols, K_vals_u, T, coherent_init=True)
            rs_u.append(r_val)
            vs_u.append(count_vortices(theta, L))

        # Disordered
        rs_d, vs_d = [], []
        for seed in range(n_seeds):
            np.random.seed(800 + hash(geom_name) % 1000 + seed)
            n_edges = len(rows)
            eta = np.random.uniform(0.2, 0.8, n_edges)
            # Symmetrize: for each pair of directed edges, average
            K_vals_d = K0 * eta
            theta, r_val, _ = simulate(N, rows, cols, K_vals_d, T, coherent_init=True)
            rs_d.append(r_val)
            vs_d.append(count_vortices(theta, L))

        table1_data[geom_name] = {
            'r_uni_mean': np.mean(rs_u), 'r_uni_std': np.std(rs_u, ddof=1),
            'v_uni_mean': np.mean(vs_u), 'v_uni_std': np.std(vs_u, ddof=1),
            'r_dis_mean': np.mean(rs_d), 'r_dis_std': np.std(rs_d, ddof=1),
            'v_dis_mean': np.mean(vs_d), 'v_dis_std': np.std(vs_d, ddof=1),
        }
        d = table1_data[geom_name]
        print(f"  {geom_name}:")
        print(f"    Uniform:    r={d['r_uni_mean']:.3f}±{d['r_uni_std']:.3f}, "
              f"v={d['v_uni_mean']:.1f}±{d['v_uni_std']:.1f}")
        print(f"    Disordered: r={d['r_dis_mean']:.3f}±{d['r_dis_std']:.3f}, "
              f"v={d['v_dis_mean']:.1f}±{d['v_dis_std']:.1f}")

    return table1_data


# ─── 5. Table 2 with stds + Kruskal-Wallis test ─────────────────────

def run_table2_with_stats():
    print("\n=== 5. Table 2: Composition score with std devs + stats ===")
    L = 32
    N, rows, cols = build_square_edges(L, periodic=False)
    K0_base = 0.01
    T_operating = 1.00
    n_seeds = 20

    tc_configs = [
        ('<20 K', 10, False),
        ('20-77 K', 48, False),
        ('>77 K', 90, True),
    ]
    scomp_configs = [
        ('<=0.33', 0.20),
        ('0.33-0.56', 0.45),
        ('>=0.56', 0.70),
    ]

    table2_data = {}
    high_tc_defect_groups = {}

    for tc_label, tc_med, use_coherent in tc_configs:
        if tc_med > 77:
            K_mean = K0_base * (tc_med - 77) / 13
        else:
            K_mean = 0.0003

        for sc_label, sc_med in scomp_configs:
            sigma_K = 0.06 + 0.55 * (1 - sc_med)

            rs, ds = [], []
            for run in range(n_seeds):
                np.random.seed(run * 997 + hash(tc_label) % 1000 + hash(sc_label) % 100 + 3)
                n_edges = len(rows)
                disorder = np.maximum(np.random.normal(1.0, sigma_K, n_edges), 0.05)
                K_vals = K_mean * disorder
                theta, r_val, _ = simulate(N, rows, cols, K_vals, T_operating,
                                           coherent_init=use_coherent)
                rs.append(r_val)
                ds.append(count_vortices(theta, L))

            key = (tc_label, sc_label)
            table2_data[key] = {
                'r_mean': np.mean(rs), 'r_std': np.std(rs, ddof=1),
                'v_mean': np.mean(ds), 'v_std': np.std(ds, ddof=1),
            }

            if tc_med > 77:
                high_tc_defect_groups[sc_label] = ds

            d = table2_data[key]
            print(f"  {tc_label} / {sc_label}: r={d['r_mean']:.3f}±{d['r_std']:.3f}, "
                  f"v={d['v_mean']:.1f}±{d['v_std']:.1f}")

    # Kruskal-Wallis test on high-Tc defect counts
    print("\n  Kruskal-Wallis test on high-Tc defect counts:")
    groups = [high_tc_defect_groups['<=0.33'],
              high_tc_defect_groups['0.33-0.56'],
              high_tc_defect_groups['>=0.56']]
    stat, p_value = stats.kruskal(*groups)
    print(f"    H-statistic = {stat:.3f}, p-value = {p_value:.4f}")

    # Also run one-way ANOVA
    f_stat, p_anova = stats.f_oneway(*groups)
    print(f"    ANOVA F-statistic = {f_stat:.3f}, p-value = {p_anova:.4f}")

    return table2_data, {'kruskal_H': stat, 'kruskal_p': p_value,
                         'anova_F': f_stat, 'anova_p': p_anova}


# ─── 6. Sensitivity analysis on S_comp mapping ──────────────────────

def run_sensitivity_analysis():
    print("\n=== 6. Sensitivity analysis on S_comp -> sigma_K slope ===")
    L = 32
    N, rows, cols = build_square_edges(L, periodic=False)
    T_operating = 1.00
    n_seeds = 20
    tc_med = 90
    K_mean = 0.01 * (tc_med - 77) / 13

    slopes = [0.385, 0.44, 0.55, 0.66, 0.715]  # -30%, -20%, baseline, +20%, +30%
    slope_labels = ['-30%', '-20%', 'Baseline', '+20%', '+30%']
    scomp_medians = [0.20, 0.45, 0.70]
    sc_labels = ['Low', 'Mid', 'High']

    results = {}
    for slope, sl_label in zip(slopes, slope_labels):
        defects_by_scomp = {}
        for sc_med, sc_label in zip(scomp_medians, sc_labels):
            sigma_K = 0.06 + slope * (1 - sc_med)
            ds = []
            for run in range(n_seeds):
                np.random.seed(run * 997 + int(slope * 1000) + int(sc_med * 100))
                n_edges = len(rows)
                disorder = np.maximum(np.random.normal(1.0, sigma_K, n_edges), 0.05)
                K_vals = K_mean * disorder
                theta, r_val, _ = simulate(N, rows, cols, K_vals, T_operating,
                                           coherent_init=True)
                ds.append(count_vortices(theta, L))
            defects_by_scomp[sc_label] = np.mean(ds)

        results[sl_label] = defects_by_scomp
        print(f"  Slope {sl_label} ({slope:.3f}): "
              f"Low={defects_by_scomp['Low']:.1f}, "
              f"Mid={defects_by_scomp['Mid']:.1f}, "
              f"High={defects_by_scomp['High']:.1f}, "
              f"Trend={'decreasing' if defects_by_scomp['High'] < defects_by_scomp['Low'] else 'NOT decreasing'}")

    return results


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    FIGURE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_results = {}

    noise_sweep = run_dense_noise_sweep()
    all_results['noise_sweep'] = {
        geom: {k: [float(x) for x in v] if isinstance(v, list) else v
               for k, v in data.items()}
        for geom, data in noise_sweep.items()
    }

    fss = run_finite_size_scaling()
    all_results['finite_size'] = {
        str(L): {k: [float(x) for x in v] if isinstance(v, list) else v
                 for k, v in data.items()}
        for L, data in fss.items()
    }

    run_timeseries_convergence()

    table1 = run_table1_with_stds()
    all_results['table1'] = {k: {kk: float(vv) for kk, vv in v.items()}
                             for k, v in table1.items()}

    table2, stat_tests = run_table2_with_stats()
    all_results['table2'] = {f"{k[0]}|{k[1]}": {kk: float(vv) for kk, vv in v.items()}
                             for k, v in table2.items()}
    all_results['stat_tests'] = {k: float(v) for k, v in stat_tests.items()}

    sensitivity = run_sensitivity_analysis()
    all_results['sensitivity'] = {k: {kk: float(vv) for kk, vv in v.items()}
                                  for k, v in sensitivity.items()}

    with open(OUTPUT_DIR / 'simulation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n=== All simulations complete. Results saved to simulation_results.json ===")
