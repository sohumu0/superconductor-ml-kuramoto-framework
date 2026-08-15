#!/usr/bin/env python3
"""
Dataset-Kuramoto bridge: connects ML dataset composition profiles to
Kuramoto phase-coherence simulations. Computes a composition-based
uniformity score S_comp from five formula-derived disorder descriptors,
then runs Kuramoto simulations parameterized by Tc and S_comp to
produce order-parameter and vortex-defect predictions.

Generates Figure 3 (dataset-Kuramoto bridge matrix).
"""
from pathlib import Path
import math

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.utils.data import MagpieData
from PIL import Image, ImageDraw, ImageFont
from pymatgen.core import Composition


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULT_DIR = PROJECT_ROOT / "analysis_results"

OPERATING_TEMP_K = 77.0
LOW_TC_MAX_K = 20.0
HIGH_TC_MIN_K = OPERATING_TEMP_K

N = 32
DT = 0.04
STEPS = 2200
REPLICATES = 6
SEED = 2406

CANVAS_W = 2200
CANVAS_H = 1180
BG = (255, 255, 255, 255)
INK = (12, 18, 28, 255)
MUTED = (78, 88, 104, 255)
GRID = (221, 227, 236, 255)
AXIS = (45, 55, 70, 255)

TC_LEVELS = ["Low Tc", "Intermediate Tc", "High Tc"]
ORDER_LEVELS = ["low material order", "intermediate material order", "high material order"]
PROFILE_ORDER = [f"{tc_level} / {order_level}" for tc_level in TC_LEVELS for order_level in ORDER_LEVELS]

PROFILE_COLORS = {
    "Low Tc / high material order": (80, 143, 185, 255),
    "Low Tc / low material order": (213, 117, 69, 255),
    "High Tc / high material order": (77, 156, 118, 255),
    "High Tc / low material order": (156, 105, 178, 255),
    "middle": (170, 176, 186, 75),
}

DATASETS = [
    {
        "name": "UCI / SuperCon",
        "path": PROJECT_ROOT / "data" / "hamideih_data" / "unique_m.csv",
        "formula_col": "material",
        "tc_col": "critical_temp",
    },
    {
        "name": "3DSC",
        "path": PROJECT_ROOT / "data" / "3dsc_data" / "3DSC_MP_with_MAGPIE.csv",
        "formula_col": "formula_sc",
        "tc_col": "tc",
    },
]

DISORDER_FEATURES = [
    "number_of_elements",
    "normalized_mixing_entropy",
    "avg_dev_covalent_radius",
    "avg_dev_electronegativity",
    "avg_dev_valence_electrons",
]


def load_font(size, bold=False):
    names = ["Arial Bold.ttf", "Arial.ttf"] if bold else ["Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text(draw, xy, value, font, fill=INK):
    draw.text(xy, value, font=font, fill=fill)


def centered_text(draw, box, value, font, fill=INK):
    left, top, right, bottom = draw.multiline_textbbox((0, 0), value, font=font, spacing=5)
    width = right - left
    height = bottom - top
    x0, y0, x1, y1 = box
    draw.multiline_text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        value,
        font=font,
        fill=fill,
        align="center",
        spacing=5,
    )


def robust_scale(values):
    series = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = series.median()
    series = series.fillna(0.0 if pd.isna(median) else median)
    lo, hi = series.quantile([0.05, 0.95])

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index)

    return ((series - lo) / (hi - lo)).clip(0, 1)


def label_profile(tc_k, material_order_score, low_order_cut, high_order_cut):
    if tc_k <= LOW_TC_MAX_K:
        tc_level = "Low Tc"
    elif tc_k < HIGH_TC_MIN_K:
        tc_level = "Intermediate Tc"
    else:
        tc_level = "High Tc"

    if material_order_score <= low_order_cut:
        order_level = "low material order"
    elif material_order_score < high_order_cut:
        order_level = "intermediate material order"
    else:
        order_level = "high material order"
    return f"{tc_level} / {order_level}"


def load_raw_dataset(spec):
    df = pd.read_csv(spec["path"], usecols=[spec["formula_col"], spec["tc_col"]]).copy()
    df["Tc_K"] = pd.to_numeric(df[spec["tc_col"]], errors="coerce")
    df["formula"] = df[spec["formula_col"]].astype(str).str.strip()
    df = df[df["Tc_K"].notna() & (df["Tc_K"] > 0) & df["formula"].ne("")].copy()
    df["dataset"] = spec["name"]
    return df[["dataset", "formula", "Tc_K"]]


def formula_disorder_metrics(formula, featurizer):
    composition = Composition(formula)
    amounts = np.array(list(composition.get_el_amt_dict().values()), dtype=float)
    fractions = amounts / amounts.sum()
    n_elements = len(fractions)
    if n_elements > 1:
        mixing_entropy = float(-(fractions * np.log(fractions)).sum() / np.log(n_elements))
    else:
        mixing_entropy = 0.0

    radius_dev, electronegativity_dev, valence_dev = featurizer.featurize(composition)
    return {
        "number_of_elements": float(n_elements),
        "normalized_mixing_entropy": mixing_entropy,
        "avg_dev_covalent_radius": float(radius_dev),
        "avg_dev_electronegativity": float(electronegativity_dev),
        "avg_dev_valence_electrons": float(valence_dev),
    }


def add_shared_material_order_score(all_data):
    featurizer = ElementProperty(
        data_source=MagpieData(impute_nan=True),
        features=["CovalentRadius", "Electronegativity", "NValence"],
        stats=["avg_dev"],
    )
    metrics_by_formula = {
        formula: formula_disorder_metrics(formula, featurizer)
        for formula in all_data["formula"].unique()
    }
    metrics = pd.DataFrame.from_dict(metrics_by_formula, orient="index")
    metrics.index.name = "formula"
    data = all_data.join(metrics, on="formula")

    scaled_columns = []
    for feature in DISORDER_FEATURES:
        scaled_name = f"scaled_{feature}"
        data[scaled_name] = robust_scale(data[feature])
        scaled_columns.append(scaled_name)

    data["composition_disorder_score"] = data[scaled_columns].mean(axis=1)
    data["material_order_score"] = 1.0 - data["composition_disorder_score"]
    low_cut, high_cut = data["material_order_score"].quantile([1 / 3, 2 / 3])
    data["profile"] = [
        label_profile(tc, score, low_cut, high_cut)
        for tc, score in zip(data["Tc_K"], data["material_order_score"])
    ]
    data["low_material_order_cut"] = float(low_cut)
    data["high_material_order_cut"] = float(high_cut)
    data["score_definition"] = "1 - pooled mean of five robust-scaled composition-disorder features"
    return data


def reduced_temperature(tc_k):
    return OPERATING_TEMP_K / max(float(tc_k), 1e-9)


def model_temperature(tc_k):
    reduced = reduced_temperature(tc_k)
    scaled = (reduced - 0.25) / (1.35 - 0.25)
    return float(np.clip(0.12 + 1.08 * scaled, 0.12, 1.20))


def coupling_sigma(material_order_score):
    disorder_level = 1.0 - float(material_order_score)
    return float(np.clip(0.06 + 0.55 * disorder_level, 0.06, 0.62))


def build_profiles(all_data):
    rows = []
    for profile in PROFILE_ORDER:
        subset = all_data[all_data["profile"] == profile].copy()
        if subset.empty:
            raise ValueError(f"No rows found for profile: {profile}")

        dataset_counts = subset["dataset"].value_counts().to_dict()
        median_tc = float(subset["Tc_K"].median())
        median_order = float(subset["material_order_score"].median())
        rows.append(
            {
                "profile": profile,
                "n_samples": int(len(subset)),
                "uci_samples": int(dataset_counts.get("UCI / SuperCon", 0)),
                "d3sc_samples": int(dataset_counts.get("3DSC", 0)),
                "median_Tc_K": median_tc,
                "median_material_order_score": median_order,
                "reduced_operating_temperature_77K": reduced_temperature(median_tc),
                "model_temperature": model_temperature(median_tc),
                "coupling_sigma_K": coupling_sigma(median_order),
            }
        )
    return pd.DataFrame(rows)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def defect_grid(theta):
    dx = wrap(np.roll(theta, -1, axis=1) - theta)
    dy = wrap(np.roll(theta, -1, axis=0) - theta)
    phase_sum = dx + np.roll(dy, -1, axis=1) - np.roll(dx, -1, axis=0) - dy
    return np.round(phase_sum / (2 * np.pi))


def order_parameter(theta):
    return float(abs(np.exp(1j * theta).mean()))


def simulate_kuramoto(model_t, sigma_k, seed):
    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.08, size=(N, N)) % (2 * np.pi)

    k_right = np.clip(rng.normal(1.0, sigma_k, size=(N, N)), 0.05, 2.5)
    k_down = np.clip(rng.normal(1.0, sigma_k, size=(N, N)), 0.05, 2.5)
    noise_scale = math.sqrt(2 * model_t * DT)

    for _ in range(STEPS):
        torque = (
            k_right * np.sin(np.roll(theta, -1, axis=1) - theta)
            + np.roll(k_right, 1, axis=1) * np.sin(np.roll(theta, 1, axis=1) - theta)
            + k_down * np.sin(np.roll(theta, -1, axis=0) - theta)
            + np.roll(k_down, 1, axis=0) * np.sin(np.roll(theta, 1, axis=0) - theta)
        )
        theta = (theta + DT * torque + noise_scale * rng.normal(size=(N, N))) % (2 * np.pi)

    q = defect_grid(theta)
    defects = int(np.abs(q).sum())
    return {
        "theta": theta,
        "q": q,
        "r": order_parameter(theta),
        "defects": defects,
        "defect_density": defects / (N * N),
        "k_right": k_right,
        "k_down": k_down,
    }


def run_profile_simulations(profile_df):
    summary_rows = []

    for idx, row in profile_df.reset_index(drop=True).iterrows():
        profile = row["profile"]
        results = []
        for rep in range(REPLICATES):
            result = simulate_kuramoto(
                row["model_temperature"],
                row["coupling_sigma_K"],
                SEED + idx * 100 + rep,
            )
            results.append(result)

        r_values = np.array([item["r"] for item in results])
        defect_values = np.array([item["defects"] for item in results])
        density_values = np.array([item["defect_density"] for item in results])
        summary_rows.append(
            {
                **row.to_dict(),
                "r_mean": float(r_values.mean()),
                "r_std": float(r_values.std(ddof=1)),
                "defects_mean": float(defect_values.mean()),
                "defects_std": float(defect_values.std(ddof=1)),
                "defect_density_mean": float(density_values.mean()),
                "defect_density_std": float(density_values.std(ddof=1)),
                "replicates": REPLICATES,
                "lattice_size": N,
                "simulation_steps": STEPS,
            }
        )

    return pd.DataFrame(summary_rows)


def blend_color(start, end, amount):
    amount = float(np.clip(amount, 0, 1))
    return tuple(int(round(a + amount * (b - a))) for a, b in zip(start, end)) + (255,)


def draw_matrix_headers(draw, all_data, box):
    axis_font = load_font(16, bold=True)
    x0, y0, x1, y1 = box
    low_cut = float(all_data["low_material_order_cut"].iloc[0])
    high_cut = float(all_data["high_material_order_cut"].iloc[0])

    row_label_w = 170
    header_h = 72
    matrix_x0 = x0 + row_label_w
    matrix_y0 = y0 + header_h
    matrix_x1 = x1
    matrix_y1 = y1
    cell_w = (matrix_x1 - matrix_x0) / 3
    cell_h = (matrix_y1 - matrix_y0) / 3

    column_labels = [
        f"Low composition score\n(<= {low_cut:.2f})",
        f"Intermediate score\n({low_cut:.2f} to {high_cut:.2f})",
        f"High composition score\n(>= {high_cut:.2f})",
    ]
    row_labels = [
        "Low Tc\n(Tc <= 20 K)",
        "Intermediate Tc\n(20 < Tc < 77 K)",
        "High Tc\n(Tc >= 77 K)",
    ]
    for col_idx, label in enumerate(column_labels):
        left = matrix_x0 + col_idx * cell_w
        centered_text(draw, (left, y0, left + cell_w, matrix_y0 - 4), label, axis_font, MUTED)
    for row_idx, label in enumerate(row_labels):
        top = matrix_y0 + row_idx * cell_h
        centered_text(draw, (x0, top, matrix_x0 - 10, top + cell_h), label, axis_font, MUTED)

    return matrix_x0, matrix_y0, matrix_x1, matrix_y1, cell_w, cell_h


def draw_count_matrix_panel(draw, all_data, dataset_name, box, panel_label, title):
    heading = load_font(28, bold=True)
    count_font = load_font(29, bold=True)
    percent_font = load_font(16)

    x0, y0, x1, y1 = box
    dataset = all_data[all_data["dataset"] == dataset_name]
    total = len(dataset)
    text(draw, (x0, y0 - 52), f"{panel_label} {title} (n = {total:,})", heading)
    matrix_x0, matrix_y0, matrix_x1, matrix_y1, cell_w, cell_h = draw_matrix_headers(
        draw, all_data, box
    )

    for row_idx, tc_level in enumerate(TC_LEVELS):
        for col_idx, order_level in enumerate(ORDER_LEVELS):
            profile = f"{tc_level} / {order_level}"
            left = matrix_x0 + col_idx * cell_w
            top = matrix_y0 + row_idx * cell_h
            right = left + cell_w
            bottom = top + cell_h
            count = int((dataset["profile"] == profile).sum())
            percent = 100.0 * count / total
            strength = math.sqrt(min(percent / 50.0, 1.0))
            fill = blend_color((244, 247, 250), (58, 126, 166), strength)
            label_color = (255, 255, 255, 255) if strength > 0.58 else INK

            draw.rectangle((left, top, right, bottom), fill=fill, outline=(255, 255, 255, 255), width=4)
            centered_text(draw, (left, top + 8, right, top + cell_h * 0.66), f"{count:,}", count_font, label_color)
            centered_text(
                draw,
                (left, top + cell_h * 0.57, right, bottom - 5),
                f"{percent:.1f}%",
                percent_font,
                label_color,
            )

    draw.rectangle((matrix_x0, matrix_y0, matrix_x1, matrix_y1), outline=AXIS, width=2)


def draw_simulation_matrix_panel(
    draw,
    all_data,
    results_df,
    box,
    panel_label,
    title,
    metric,
    error,
    maximum,
    high_color,
):
    heading = load_font(28, bold=True)
    value_font = load_font(27, bold=True)
    error_font = load_font(15)
    note_font = load_font(15)

    x0, y0, x1, y1 = box
    text(draw, (x0, y0 - 52), f"{panel_label} {title}", heading)
    matrix_box = (x0, y0, x1, y1 - 28)
    matrix_x0, matrix_y0, matrix_x1, matrix_y1, cell_w, cell_h = draw_matrix_headers(
        draw, all_data, matrix_box
    )

    for row_idx, tc_level in enumerate(TC_LEVELS):
        for col_idx, order_level in enumerate(ORDER_LEVELS):
            profile = f"{tc_level} / {order_level}"
            row = results_df[results_df["profile"] == profile].iloc[0]
            value = float(row[metric])
            uncertainty = float(row[error])
            strength = min(value / maximum, 1.0)
            fill = blend_color((245, 247, 248), high_color, strength)
            label_color = (255, 255, 255, 255) if strength > 0.56 else INK

            left = matrix_x0 + col_idx * cell_w
            top = matrix_y0 + row_idx * cell_h
            right = left + cell_w
            bottom = top + cell_h
            draw.rectangle((left, top, right, bottom), fill=fill, outline=(255, 255, 255, 255), width=4)

            value_label = f"{value:.2f}" if metric == "r_mean" else f"{value:.1f}"
            error_label = f"+/- {uncertainty:.2f}" if metric == "r_mean" else f"+/- {uncertainty:.1f}"
            centered_text(draw, (left, top + 8, right, top + cell_h * 0.66), value_label, value_font, label_color)
            centered_text(
                draw,
                (left, top + cell_h * 0.58, right, bottom - 5),
                error_label,
                error_font,
                label_color,
            )

    draw.rectangle((matrix_x0, matrix_y0, matrix_x1, matrix_y1), outline=AXIS, width=2)
    centered_text(draw, (matrix_x0, y1 - 23, matrix_x1, y1), "mean +/- SD across 6 runs", note_font, MUTED)


def make_figure(all_data, results_df):
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas, "RGBA")
    draw_count_matrix_panel(
        draw, all_data, "UCI / SuperCon",
        (90, 125, 1030, 590), "(a)", "UCI/SuperCon records",
    )
    draw_count_matrix_panel(
        draw, all_data, "3DSC",
        (1170, 125, 2110, 590), "(b)", "3DSC records",
    )
    draw_simulation_matrix_panel(
        draw, all_data, results_df,
        (90, 730, 1030, 1120), "(c)", "Phase coherence, r",
        "r_mean", "r_std", 1.0, (54, 145, 105),
    )
    draw_simulation_matrix_panel(
        draw, all_data, results_df,
        (1170, 730, 2110, 1120), "(d)", "Mean vortex-defect count",
        "defects_mean", "defects_std", 120.0, (197, 79, 52),
    )

    FIGURE_DIR.mkdir(exist_ok=True)
    png_path = FIGURE_DIR / "figure6_dataset_kuramoto_bridge.png"
    pdf_path = FIGURE_DIR / "figure6_dataset_kuramoto_bridge.pdf"

    rgb_canvas = canvas.convert("RGB")
    rgb_canvas.save(png_path, dpi=(300, 300))

    pdf_saved = False
    try:
        Image.init()
        rgb_canvas.save(pdf_path, "PDF", resolution=300)
        pdf_saved = True
    except Exception as exc:
        print(f"PDF export skipped: {exc}")

    print(f"Saved {png_path}")
    if pdf_saved:
        print(f"Saved {pdf_path}")


def main():
    FIGURE_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    data_frames = [load_raw_dataset(spec) for spec in DATASETS]
    all_data = add_shared_material_order_score(pd.concat(data_frames, ignore_index=True))
    profiles = build_profiles(all_data)
    results = run_profile_simulations(profiles)

    proxy_path = RESULT_DIR / "figure6_dataset_proxy_table.csv"
    profile_path = RESULT_DIR / "figure6_kuramoto_bridge_results.csv"
    all_data.to_csv(proxy_path, index=False)
    results.to_csv(profile_path, index=False)

    make_figure(all_data, results)

    print(f"Saved {proxy_path}")
    print(f"Saved {profile_path}")
    print("\nBridge profiles")
    print(
        results[
            [
                "profile",
                "n_samples",
                "uci_samples",
                "d3sc_samples",
                "median_Tc_K",
                "median_material_order_score",
                "model_temperature",
                "coupling_sigma_K",
                "r_mean",
                "defects_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
