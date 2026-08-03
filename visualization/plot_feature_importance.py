#!/usr/bin/env python3
"""
Generate Figure 3: XGBoost feature importance across three regimes
(UCI/SuperCon composition, 3DSC structural, 3DSC structural+MAGPIE).

Reads pre-computed feature importance from analysis_results/.
"""
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPORTANCE_PATH = PROJECT_ROOT / "analysis_results" / "figure3_xgboost_feature_importance.csv"
FIGURE_DIR = PROJECT_ROOT / "figures"

CANVAS_W = 1900
CANVAS_H = 610

BG = (255, 255, 255, 255)
INK = (12, 18, 28, 255)
MUTED = (76, 86, 101, 255)
GRID = (220, 227, 236, 255)
AXIS = (45, 55, 70, 255)
COMPOSITION = (79, 141, 185, 255)
STRUCTURAL = (91, 179, 188, 255)
MAGPIE = (214, 119, 72, 255)

PANELS = [
    ("(a)", "UCI/SuperCon composition", "UCI / SuperCon", "Composition features"),
    ("(b)", "3DSC structural", "3DSC", "Structural only"),
    ("(c)", "3DSC structural+MAGPIE", "3DSC", "Structural + MAGPIE"),
]


LABELS = {
    "range_ThermalConductivity": "spread in thermal conductivity",
    "range_atomic_radius": "spread in atomic radius",
    "wtd_gmean_ThermalConductivity": "weighted thermal conductivity",
    "wtd_std_ElectronAffinity": "variation in electron affinity",
    "wtd_std_ThermalConductivity": "variation in thermal cond.",
    "std_atomic_mass": "variation in atomic mass",
    "wtd_mean_ThermalConductivity": "mean thermal conductivity",
    "std_Density": "variation in density",
    "gmean_ElectronAffinity": "typical electron affinity",
    "wtd_gmean_Valence": "weighted valence",
    "num_elements_sc": "number of elements",
    "base-centered": "base-centered lattice",
    "formation_energy_per_atom_2": "formation energy per atom",
    "efermi_2": "Fermi energy",
    "latb_2": "lattice constant b",
    "density_2": "crystal density",
    "primitive": "primitive lattice",
    "latc_2": "lattice constant c",
    "cell_volume_2": "unit-cell volume",
    "e_above_hull_2": "energy above hull",
    "MAGPIE_MagpieData_range_Column": "spread in periodic-table group",
    "MAGPIE_MagpieData_mode_MeltingT": "most common melting point",
    "MAGPIE_MagpieData_mode_SpaceGroupNumber": "most common space group",
    "MAGPIE_MagpieData_minimum_MeltingT": "lowest elemental melting point",
    "MAGPIE_MagpieData_mode_NUnfilled": "typical unfilled electrons",
    "MAGPIE_MagpieData_maximum_SpaceGroupNumber": "largest space-group number",
    "MAGPIE_MagpieData_avg_dev_NfUnfilled": "variation in unfilled f states",
    "MAGPIE_MagpieData_avg_dev_NUnfilled": "variation in unfilled electrons",
    "MAGPIE_MagpieData_mean_NfUnfilled": "mean unfilled f states",
}


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


def centered(draw, box, value, font, fill=INK):
    left, top, right, bottom = draw.textbbox((0, 0), value, font=font)
    width = right - left
    height = bottom - top
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        value,
        font=font,
        fill=fill,
    )


def load_importances():
    with open(IMPORTANCE_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["importance_mean"] = float(row["importance_mean"])
        row["importance_std"] = float(row["importance_std"])
    return rows


def feature_label(feature):
    if feature in LABELS:
        return LABELS[feature]
    label = feature.replace("MAGPIE_MagpieData_", "MAGPIE ")
    label = label.replace("_", " ")
    return label


def category(feature, dataset, feature_space):
    if feature.startswith("MAGPIE_"):
        return "MAGPIE", MAGPIE
    if dataset == "UCI / SuperCon":
        return "composition", COMPOSITION
    return "structural", STRUCTURAL


def rounded_axis_max(max_percent):
    if max_percent <= 20:
        return 20
    if max_percent <= 40:
        return 40
    if max_percent <= 70:
        return 70
    return 100


def draw_panel(draw, box, panel_label, title, rows):
    title_font = load_font(26, bold=True)
    label_font = load_font(16)
    tick_font = load_font(15)
    value_font = load_font(15, bold=True)

    x0, y0, x1, y1 = box
    text(draw, (x0, y0), f"{panel_label} {title}", title_font)

    top_rows = sorted(rows, key=lambda r: r["importance_mean"], reverse=True)[:8]
    max_percent = max(r["importance_mean"] for r in top_rows) * 100
    axis_max = rounded_axis_max(max_percent)

    label_w = 250
    plot_x = x0 + label_w
    plot_y = y0 + 58
    plot_w = x1 - plot_x - 34
    row_h = 40
    bar_h = 22
    plot_h = row_h * len(top_rows)

    for tick in [0, axis_max / 2, axis_max]:
        tx = plot_x + (tick / axis_max) * plot_w
        draw.line((tx, plot_y - 10, tx, plot_y + plot_h + 7), fill=GRID, width=1)
        label = f"{tick:.0f}"
        left, top, right, bottom = draw.textbbox((0, 0), label, font=tick_font)
        text(draw, (tx - (right - left) / 2, plot_y + plot_h + 14), label, tick_font, MUTED)

    draw.line((plot_x, plot_y + plot_h + 4, plot_x + plot_w, plot_y + plot_h + 4), fill=AXIS, width=2)

    for idx, row in enumerate(top_rows):
        y = plot_y + idx * row_h
        percent = row["importance_mean"] * 100
        err = row["importance_std"] * 100
        _, color = category(row["feature"], row["dataset"], row["feature_space"])

        label = feature_label(row["feature"])
        text(draw, (x0, y + 2), label, label_font)

        bar_w = (percent / axis_max) * plot_w
        draw.rounded_rectangle((plot_x, y, plot_x + bar_w, y + bar_h), radius=5, fill=color)

        err_x = plot_x + (min(axis_max, percent + err) / axis_max) * plot_w
        center_y = y + bar_h / 2
        draw.line((plot_x + bar_w, center_y, err_x, center_y), fill=INK, width=2)
        draw.line((err_x, center_y - 7, err_x, center_y + 7), fill=INK, width=2)

        value = f"{percent:.1f}"
        value_x = max(plot_x + bar_w + 8, err_x + 8)
        text(draw, (value_x, y + 1), value, value_font)

    centered(draw, (plot_x, plot_y + plot_h + 42, plot_x + plot_w, plot_y + plot_h + 70), "relative importance (%)", tick_font, MUTED)


def draw_legend(draw):
    font = load_font(17)
    items = [
        ("composition descriptor", COMPOSITION),
        ("structural descriptor", STRUCTURAL),
        ("MAGPIE descriptor", MAGPIE),
    ]
    x = 70
    y = 555
    for label, color in items:
        draw.rounded_rectangle((x, y, x + 30, y + 19), radius=5, fill=color)
        text(draw, (x + 42, y - 2), label, font)
        x += 360


def main():
    if not IMPORTANCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {IMPORTANCE_PATH}. Run the ML training scripts first to generate feature importance data."
        )

    rows = load_importances()
    canvas = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(canvas, "RGBA")

    panel_w = 560
    panel_h = 470
    x_positions = [70, 670, 1270]

    for x, (panel_label, title, dataset, feature_space) in zip(x_positions, PANELS):
        subset = [
            row
            for row in rows
            if row["dataset"] == dataset and row["feature_space"] == feature_space
        ]
        draw_panel(draw, (x, 60, x + panel_w, 60 + panel_h), panel_label, title, subset)

    draw_legend(draw)

    FIGURE_DIR.mkdir(exist_ok=True)
    png_path = FIGURE_DIR / "figure3_xgboost_feature_importance.png"
    pdf_path = FIGURE_DIR / "figure3_xgboost_feature_importance.pdf"

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


if __name__ == "__main__":
    main()
