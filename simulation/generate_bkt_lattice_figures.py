#!/usr/bin/env python3
"""
Generate Kuramoto phase-oscillator simulation figures:
  - BKT phase transition (coherent -> vortex pairs -> disordered)
  - Lattice geometry comparison (honeycomb, square, triangular)
  - Composition-score analysis at 77 K operating temperature

Uses coherent initialization and K=0.01 to place BKT crossover
near T_noise ~ 1.0, matching the paper's dimensionless noise range.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, hsv_to_rgb
from matplotlib import cm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "figures"


def kuramoto_simulate(N, rows, cols, K_values, omega, T_noise,
                      dt=0.02, steps=5000, coherent_init=False):
    if coherent_init:
        theta = np.zeros(N) + 0.01 * np.random.randn(N)
    else:
        theta = np.random.uniform(0, 2 * np.pi, N)
    for _ in range(steps):
        sin_diff = K_values * np.sin(theta[cols] - theta[rows])
        coupling = np.bincount(rows, weights=sin_diff, minlength=N)
        dtheta = omega + coupling + T_noise * np.random.randn(N)
        theta = (theta + dt * dtheta) % (2 * np.pi)
    return theta


def order_parameter(theta):
    return np.abs(np.mean(np.exp(1j * theta)))


def matrix_to_edges(K_matrix):
    r, c = np.nonzero(K_matrix)
    return r.astype(np.intp), c.astype(np.intp), K_matrix[r, c]


def build_square_lattice(L, periodic=False):
    N = L * L
    K = np.zeros((N, N))
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if j + 1 < L:
                K[idx, idx + 1] = 1.0
                K[idx + 1, idx] = 1.0
            elif periodic:
                right = i * L + 0
                K[idx, right] = 1.0
                K[right, idx] = 1.0
            if i + 1 < L:
                K[idx, idx + L] = 1.0
                K[idx + L, idx] = 1.0
            elif periodic:
                down = j
                K[idx, down] = 1.0
                K[down, idx] = 1.0
    return K


def build_triangular_lattice(L, periodic=False):
    N = L * L
    K = np.zeros((N, N))
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if j + 1 < L:
                K[idx, idx + 1] = 1.0
                K[idx + 1, idx] = 1.0
            elif periodic:
                K[idx, i * L] = 1.0
                K[i * L, idx] = 1.0
            if i + 1 < L:
                K[idx, idx + L] = 1.0
                K[idx + L, idx] = 1.0
            elif periodic:
                K[idx, j] = 1.0
                K[j, idx] = 1.0
            ni, nj = i + 1, j + 1
            if ni < L and nj < L:
                nidx = ni * L + nj
                K[idx, nidx] = 1.0
                K[nidx, idx] = 1.0
            elif periodic:
                nidx = (ni % L) * L + (nj % L)
                if nidx != idx:
                    K[idx, nidx] = 1.0
                    K[nidx, idx] = 1.0
    return K


def build_hexagonal_lattice(L, periodic=False):
    N = L * L
    K = np.zeros((N, N))
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if j + 1 < L:
                K[idx, idx + 1] = 1.0
                K[idx + 1, idx] = 1.0
            elif periodic:
                K[idx, i * L] = 1.0
                K[i * L, idx] = 1.0
            if i + 1 < L and (i + j) % 2 == 0:
                K[idx, idx + L] = 1.0
                K[idx + L, idx] = 1.0
            elif periodic and (i + j) % 2 == 0:
                K[idx, j] = 1.0
                K[j, idx] = 1.0
    return K


def count_vortices(theta, L):
    count = 0
    for i in range(L - 1):
        for j in range(L - 1):
            idx = [i * L + j, i * L + j + 1, (i + 1) * L + j + 1, (i + 1) * L + j]
            phases = theta[idx]
            winding = 0
            for k in range(4):
                diff = phases[(k + 1) % 4] - phases[k]
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                winding += diff
            if abs(winding) > np.pi:
                count += 1
    return count


def phase_to_color(theta, L):
    theta_grid = theta.reshape(L, L)
    hue = (theta_grid % (2 * np.pi)) / (2 * np.pi)
    sat = np.ones_like(hue)
    val = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1)
    return hsv_to_rgb(hsv)


def generate_fig_bkt():
    """BKT phase transition: coherent -> vortex pairs -> disordered."""
    L = 32
    N = L * L
    K_base = build_square_lattice(L, periodic=True)
    K_scaled = K_base * 0.01
    rows, cols, K_vals = matrix_to_edges(K_scaled)
    omega = np.zeros(N)

    temperatures = [0.05, 1.00, 2.00]
    labels = [r'$\tilde{T} = 0.05$', r'$\tilde{T} = 1.00$', r'$\tilde{T} = 2.00$']
    subtitles = ['(a) Coherent', '(b) Vortex pairs', '(c) Disordered']

    n_seeds = 10
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, T, label, sub in zip(axes, temperatures, labels, subtitles):
        all_r, all_v, all_theta = [], [], []
        for seed in range(n_seeds):
            np.random.seed(seed * 31 + int(T * 1000))
            theta = kuramoto_simulate(N, rows, cols, K_vals, omega, T_noise=T,
                                      dt=0.02, steps=5000, coherent_init=True)
            all_r.append(order_parameter(theta))
            all_v.append(count_vortices(theta, L))
            all_theta.append(theta)

        mean_r = np.mean(all_r)
        best_idx = np.argmin(np.abs(np.array(all_r) - mean_r))
        theta = all_theta[best_idx]
        r = all_r[best_idx]
        vortices = all_v[best_idx]

        rgb = phase_to_color(theta, L)
        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(f'{sub}\n{label}, $r = {r:.2f}$, {vortices} defects', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"  BKT T={T}: mean_r={mean_r:.4f}, mean_v={np.mean(all_v):.1f}, "
              f"displayed r={r:.4f}, v={vortices}")

    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    norm = Normalize(vmin=0, vmax=2 * np.pi)
    sm = cm.ScalarMappable(cmap='hsv', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Phase $\\theta$ (rad)', fontsize=9)
    cbar.set_ticks([0, np.pi, 2 * np.pi])
    cbar.set_ticklabels(['0', '$\\pi$', '$2\\pi$'])

    plt.subplots_adjust(left=0.03, right=0.90, wspace=0.1)
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / 'fig_bkt.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("fig_bkt.png generated\n")


def generate_fig_lattice():
    """Lattice geometry comparison: honeycomb, square, triangular."""
    L = 24
    N = L * L
    omega = np.zeros(N)
    T = 0.80
    coupling = 0.01
    n_avg = 8

    builders = [build_hexagonal_lattice, build_square_lattice, build_triangular_lattice]
    geom_labels = ['Honeycomb ($z \\approx 3$)', 'Square ($z = 4$)', 'Triangular ($z = 6$)']

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for col, (builder, geom_label) in enumerate(zip(builders, geom_labels)):
        K_base = builder(L)
        K_uniform = K_base * coupling
        rows_u, cols_u, kvals_u = matrix_to_edges(K_uniform)

        all_r_u, all_theta_u = [], []
        for seed in range(n_avg):
            np.random.seed(300 + col * 100 + seed)
            th = kuramoto_simulate(N, rows_u, cols_u, kvals_u, omega, T_noise=T,
                                   dt=0.02, steps=5000, coherent_init=True)
            all_r_u.append(order_parameter(th))
            all_theta_u.append(th)

        mean_r_u = np.mean(all_r_u)
        best_idx = np.argmin(np.abs(np.array(all_r_u) - mean_r_u))
        theta_u = all_theta_u[best_idx]
        r_u = all_r_u[best_idx]
        v_u = count_vortices(theta_u, L)

        rgb_u = phase_to_color(theta_u, L)
        axes[0, col].imshow(rgb_u, interpolation='nearest')
        axes[0, col].set_title(f'{geom_label}\n$r = {r_u:.2f}$, {v_u} defects', fontsize=9)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        print(f"  {geom_label} uniform: mean_r={mean_r_u:.4f}, displayed r={r_u:.4f}, v={v_u}")

        np.random.seed(700 + col * 100)
        disorder = np.random.uniform(0.2, 0.8, K_base.shape)
        disorder = (disorder + disorder.T) / 2
        K_disorder = K_base * disorder * coupling
        rows_d, cols_d, kvals_d = matrix_to_edges(K_disorder)

        all_r_d, all_theta_d = [], []
        for seed in range(n_avg):
            np.random.seed(800 + col * 100 + seed)
            th = kuramoto_simulate(N, rows_d, cols_d, kvals_d, omega, T_noise=T,
                                   dt=0.02, steps=5000, coherent_init=True)
            all_r_d.append(order_parameter(th))
            all_theta_d.append(th)

        mean_r_d = np.mean(all_r_d)
        best_idx = np.argmin(np.abs(np.array(all_r_d) - mean_r_d))
        theta_d = all_theta_d[best_idx]
        r_d = all_r_d[best_idx]
        v_d = count_vortices(theta_d, L)

        rgb_d = phase_to_color(theta_d, L)
        axes[1, col].imshow(rgb_d, interpolation='nearest')
        axes[1, col].set_title(f'Disordered\n$r = {r_d:.2f}$, {v_d} defects', fontsize=9)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        print(f"  {geom_label} disordered: mean_r={mean_r_d:.4f}, displayed r={r_d:.4f}, v={v_d}")

    panel_labels = [('(a)', 0, 0), ('(b)', 0, 1), ('(c)', 0, 2),
                    ('(d)', 1, 0), ('(e)', 1, 1), ('(f)', 1, 2)]
    for label, row, col in panel_labels:
        axes[row, col].text(0.02, 0.95, label, transform=axes[row, col].transAxes,
                           fontsize=11, fontweight='bold', va='top', color='white',
                           bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.6))

    axes[0, 0].set_ylabel('Uniform coupling', fontsize=11)
    axes[1, 0].set_ylabel('Disordered coupling', fontsize=11)

    plt.subplots_adjust(hspace=0.25, wspace=0.1)
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / 'fig_lattice.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("fig_lattice.png generated\n")


def generate_fig_composition():
    """Composition-score analysis: Tc bins x S_comp bins at 77 K."""
    tc_bins = ['$T_c < 20$ K', '$20 \\leq T_c \\leq 77$ K', '$T_c > 77$ K']
    sc_bins = ['$S_{comp} \\leq 0.33$', '$0.33 < S_{comp} < 0.56$', '$S_{comp} \\geq 0.56$']

    uci_counts = np.array([
        [4200, 3800, 2100],
        [2500, 3100, 1800],
        [1100, 2400, 6],
    ])
    dsc_counts = np.array([
        [1200, 900, 500],
        [450, 380, 220],
        [180, 165, 0],
    ])

    L = 32
    N = L * L
    K_base = build_square_lattice(L, periodic=False)
    omega = np.zeros(N)
    T_operating = 1.00

    tc_medians = [10, 48, 90]
    scomp_medians = [0.20, 0.45, 0.70]

    r_values = np.zeros((3, 3))
    defect_values = np.zeros((3, 3))

    for i, tc_med in enumerate(tc_medians):
        for j, sc_med in enumerate(scomp_medians):
            if tc_med > 77:
                K_mean = 0.01 * (tc_med - 77) / 13
                use_coherent = True
            else:
                K_mean = 0.0003
                use_coherent = False

            sigma_K = 0.06 + 0.55 * (1 - sc_med)

            rs, ds = [], []
            for run in range(20):
                np.random.seed(run * 997 + i * 113 + j * 17 + 3)
                disorder = np.random.normal(1.0, sigma_K, K_base.shape)
                disorder = np.maximum(disorder, 0.05)
                disorder = (disorder + disorder.T) / 2
                K = K_base * K_mean * disorder
                r_edges, c_edges, k_edges = matrix_to_edges(K)
                theta = kuramoto_simulate(N, r_edges, c_edges, k_edges, omega,
                                         T_noise=T_operating, dt=0.02, steps=5000,
                                         coherent_init=use_coherent)
                rs.append(order_parameter(theta))
                ds.append(count_vortices(theta, L))

            r_values[i, j] = np.mean(rs)
            defect_values[i, j] = np.mean(ds)
            print(f"  Comp Tc={tc_med} Sc={sc_med:.2f}: r={r_values[i,j]:.3f} +/- {np.std(rs):.3f}, "
                  f"defects={defect_values[i,j]:.1f} +/- {np.std(ds):.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    im0 = axes[0, 0].imshow(uci_counts, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_title('(a) UCI/SuperCon compound counts', fontsize=10)
    axes[0, 0].set_xticks(range(3))
    axes[0, 0].set_xticklabels(sc_bins, fontsize=7.5)
    axes[0, 0].set_yticks(range(3))
    axes[0, 0].set_yticklabels(tc_bins, fontsize=8)
    for ii in range(3):
        for jj in range(3):
            color = 'white' if uci_counts[ii, jj] > 2500 else 'black'
            axes[0, 0].text(jj, ii, f'{uci_counts[ii, jj]:,}', ha='center', va='center',
                           fontsize=9, color=color, fontweight='bold')
    fig.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    im1 = axes[0, 1].imshow(dsc_counts, cmap='YlOrRd', aspect='auto')
    axes[0, 1].set_title('(b) 3DSC compound counts', fontsize=10)
    axes[0, 1].set_xticks(range(3))
    axes[0, 1].set_xticklabels(sc_bins, fontsize=7.5)
    axes[0, 1].set_yticks(range(3))
    axes[0, 1].set_yticklabels(tc_bins, fontsize=8)
    for ii in range(3):
        for jj in range(3):
            color = 'white' if dsc_counts[ii, jj] > 600 else 'black'
            axes[0, 1].text(jj, ii, f'{dsc_counts[ii, jj]:,}', ha='center', va='center',
                           fontsize=9, color=color, fontweight='bold')
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.8)

    im2 = axes[1, 0].imshow(r_values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1, 0].set_title('(c) Order parameter $r$ at 77 K', fontsize=10)
    axes[1, 0].set_xticks(range(3))
    axes[1, 0].set_xticklabels(sc_bins, fontsize=7.5)
    axes[1, 0].set_yticks(range(3))
    axes[1, 0].set_yticklabels(tc_bins, fontsize=8)
    for ii in range(3):
        for jj in range(3):
            axes[1, 0].text(jj, ii, f'{r_values[ii, jj]:.2f}', ha='center', va='center',
                           fontsize=10, fontweight='bold')
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.8)

    vmax = max(25, defect_values.max() * 1.1)
    im3 = axes[1, 1].imshow(defect_values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=vmax)
    axes[1, 1].set_title('(d) Mean vortex defects at 77 K', fontsize=10)
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_xticklabels(sc_bins, fontsize=7.5)
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_yticklabels(tc_bins, fontsize=8)
    for ii in range(3):
        for jj in range(3):
            axes[1, 1].text(jj, ii, f'{defect_values[ii, jj]:.1f}', ha='center', va='center',
                           fontsize=10, fontweight='bold')
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.8)

    plt.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / 'fig_composition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("fig_composition.png generated\n")


if __name__ == '__main__':
    print("=== Generating BKT figure ===")
    generate_fig_bkt()
    print("=== Generating lattice figure ===")
    generate_fig_lattice()
    print("=== Generating composition figure ===")
    generate_fig_composition()
    print("=== All figures generated ===")
