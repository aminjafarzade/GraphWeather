#!/usr/bin/env python3
"""Add the dense-L3-k24 experiment to slides 10 and 11 of the deck.

The original PowerPoint stores the two comparison charts as PNGs.  This script
rebuilds those PNGs in the same six-panel layout, reads the new experiment's
scores directly from its WeatherBench2 CSVs, and replaces only those two media
files in a copy of the deck.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import tempfile
import zipfile

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


VARIABLES = ["z500", "t2m", "t850", "q700", "u850", "msl"]
TITLES = {
    "z500": "z500  (m²/s²)",
    "t2m": "t2m  (K)",
    "t850": "t850  (K)",
    "q700": "q700  (kg/kg)",
    "u850": "u850  (m/s)",
    "msl": "msl  (Pa)",
}


# The source deck contains only raster plots for the two earlier GNN runs.
# These traces were recovered from those plots' marker locations; their exact
# day-10 values are retained from the deck's accompanying summary slide.
PREVIOUS_RMSE = {
    "GNN 3.8M": {
        "z500": [86.85, 153.58, 229.81, 316.38, 400.84, 484.68, 556.26, 606.94, 659.73, 687.46],
        "t2m": [0.8059, 1.1119, 1.3230, 1.5237, 1.7215, 1.9207, 2.0689, 2.1948, 2.3133, 2.4147],
        "t850": [0.8402, 1.1238, 1.3730, 1.6393, 1.9287, 2.2172, 2.4459, 2.6344, 2.8270, 2.9422],
        "q700": [0.0004603, 0.0006094, 0.0007284, 0.0008298, 0.0009294, 0.0010085, 0.0010716, 0.0011272, 0.0011739, 0.0012059],
        "u850": [1.1947, 1.7186, 2.2324, 2.7714, 3.2362, 3.6507, 4.0012, 4.2902, 4.5038, 4.5948],
        "msl": [99.61, 162.30, 235.59, 315.64, 384.54, 452.35, 512.66, 555.61, 590.34, 611.10],
    },
    "GNN 2.5M": {
        "z500": [103.11, 176.59, 257.26, 344.46, 431.67, 507.90, 571.67, 617.92, 656.14, 683.83],
        "t2m": [0.9245, 1.2800, 1.5000, 1.7007, 1.8844, 2.0622, 2.1956, 2.2993, 2.4089, 2.4827],
        "t850": [0.9975, 1.3377, 1.5779, 1.8320, 2.0844, 2.3475, 2.5598, 2.7246, 2.8861, 2.9720],
        "q700": [0.0005206, 0.0006867, 0.0008038, 0.0008986, 0.0009890, 0.0010612, 0.0011159, 0.0011677, 0.0012053, 0.0012316],
        "u850": [1.3794, 1.9309, 2.4636, 2.9887, 3.4372, 3.8015, 4.1156, 4.3178, 4.4698, 4.6081],
        "msl": [117.89, 187.16, 262.82, 341.59, 410.13, 472.64, 527.10, 561.64, 590.89, 613.98],
    },
}

PREVIOUS_ACC = {
    "GNN 3.8M": {
        "z500": [0.9931, 0.9803, 0.9547, 0.9099, 0.8490, 0.7786, 0.7017, 0.6281, 0.5576, 0.5041],
        "t2m": [0.9451, 0.8875, 0.8394, 0.7914, 0.7306, 0.6633, 0.6025, 0.5480, 0.4904, 0.4381],
        "t850": [0.9675, 0.9352, 0.9003, 0.8551, 0.7972, 0.7200, 0.6467, 0.5791, 0.5029, 0.4422],
        "q700": [0.9425, 0.8910, 0.8398, 0.7821, 0.7184, 0.6569, 0.6021, 0.5452, 0.4971, 0.4511],
        "u850": [0.9781, 0.9429, 0.8968, 0.8359, 0.7690, 0.6915, 0.6175, 0.5445, 0.4817, 0.4259],
        "msl": [0.9931, 0.9774, 0.9416, 0.8852, 0.8196, 0.7453, 0.6624, 0.5865, 0.5176, 0.4569],
    },
    "GNN 2.5M": {
        "z500": [0.9931, 0.9739, 0.9451, 0.8971, 0.8330, 0.7626, 0.6889, 0.6153, 0.5480, 0.4926],
        "t2m": [0.9243, 0.8526, 0.7917, 0.7341, 0.6732, 0.6050, 0.5445, 0.4901, 0.4353, 0.3881],
        "t850": [0.9477, 0.9064, 0.8676, 0.8164, 0.7555, 0.6784, 0.6012, 0.5307, 0.4645, 0.4091],
        "q700": [0.9220, 0.8618, 0.8013, 0.7405, 0.6732, 0.6092, 0.5548, 0.5003, 0.4526, 0.4112],
        "u850": [0.9637, 0.9227, 0.8711, 0.8045, 0.7338, 0.6595, 0.5826, 0.5189, 0.4581, 0.4086],
        "msl": [0.9864, 0.9608, 0.9220, 0.8615, 0.7882, 0.7133, 0.6306, 0.5634, 0.4962, 0.4316],
    },
}


def read_wide_csv(path: Path) -> tuple[list[int], dict[str, list[float]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    lead_times = [int(row["lead_time"]) for row in rows]
    values = {var: [float(row[var]) for row in rows] for var in VARIABLES}
    if lead_times != list(range(1, 11)):
        raise ValueError(f"Expected lead times 1..10 in {path}, found {lead_times}")
    return lead_times, values


def read_kai(path: Path, metric: str) -> dict[str, list[float]]:
    result = {var: [] for var in VARIABLES}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            var = row["variable"]
            if var in result:
                result[var].append(float(row[metric]))
    for var, values in result.items():
        if len(values) != 10:
            raise ValueError(f"Expected 10 KAI values for {var} in {path}")
    return result


def style_axis(ax: plt.Axes, var: str, metric: str) -> None:
    ax.set_title(TITLES[var], fontsize=20, fontweight="bold", pad=8)
    ax.set_xlim(0.55, 10.45)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.grid(True, color="#b0b0b0", alpha=0.30, linewidth=1.0)
    ax.tick_params(axis="both", labelsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if metric == "acc":
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        return

    limits_ticks = {
        "z500": ((40, 760), [100, 200, 300, 400, 500, 600, 700]),
        "t2m": ((0.70, 3.05), [1.0, 1.5, 2.0, 2.5, 3.0]),
        "t850": ((0.60, 3.25), [1.0, 1.5, 2.0, 2.5, 3.0]),
        "q700": ((0.00040, 0.00142), [0.0006, 0.0008, 0.0010, 0.0012, 0.0014]),
        "u850": ((0.90, 4.95), [1.0, 2.0, 3.0, 4.0]),
        "msl": ((60, 670), [100, 200, 300, 400, 500, 600]),
    }
    ylim, yticks = limits_ticks[var]
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    if var == "q700":
        formatter = ScalarFormatter(useMathText=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)


def make_plot(
    output: Path,
    metric: str,
    lead_times: list[int],
    new_values: dict[str, list[float]],
    kai_values: dict[str, list[float]],
) -> None:
    previous = PREVIOUS_RMSE if metric == "rmse" else PREVIOUS_ACC
    # A tiny epsilon avoids floating-point truncation to 1648 px on some
    # Matplotlib builds; the source deck's embedded plots are 1649 x 930 px.
    fig, axes = plt.subplots(2, 3, figsize=(16.491, 9.301), dpi=100)
    axes = axes.ravel()
    colors = {"GNN 3.8M": "#1f6feb", "GNN 2.5M": "#e0662b"}

    handles = []
    labels = []
    for ax, var in zip(axes, VARIABLES):
        line = ax.plot(
            lead_times,
            kai_values[var],
            "o--",
            color="#111111",
            linewidth=2.8,
            markersize=5.2,
            label="KAI-α baseline",
            zorder=2,
        )[0]
        if not handles:
            handles.append(line)
            labels.append(line.get_label())

        for model in ("GNN 3.8M", "GNN 2.5M"):
            line = ax.plot(
                lead_times,
                previous[model][var],
                "o-",
                color=colors[model],
                linewidth=2.7,
                markersize=5.2,
                label=model,
                zorder=3,
            )[0]
            if ax is axes[0]:
                handles.append(line)
                labels.append(line.get_label())

        line = ax.plot(
            lead_times,
            new_values[var],
            "D-",
            color="#7c3aed",
            markeredgecolor="white",
            markeredgewidth=0.7,
            linewidth=3.0,
            markersize=5.8,
            label="Before curriculum · 2.5M",
            zorder=4,
        )[0]
        if ax is axes[0]:
            handles.append(line)
            labels.append(line.get_label())

        style_axis(ax, var, metric)

    for ax in axes[3:]:
        ax.set_xlabel("lead time (days)", fontsize=14)

    descriptor = "RMSE" if metric == "rmse" else "ACC"
    direction = "lower is better" if metric == "rmse" else "higher is better"
    fig.suptitle(
        f"2.5° models — {descriptor} vs lead time ({direction})",
        fontsize=28,
        fontweight="bold",
        y=0.987,
    )
    fig.supylabel(descriptor, fontsize=22, fontweight="bold", x=0.012)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.951),
        ncol=4,
        frameon=True,
        framealpha=0.96,
        fontsize=13,
        handlelength=2.4,
        columnspacing=1.7,
    )
    fig.subplots_adjust(left=0.083, right=0.991, bottom=0.080, top=0.840, wspace=0.16, hspace=0.29)
    fig.savefig(output, dpi=100, facecolor="white", metadata={"Software": "matplotlib"})
    plt.close(fig)


def replace_media(source: Path, output: Path, rmse_png: Path, acc_png: Path) -> None:
    replacements = {
        "ppt/media/image2.png": rmse_png.read_bytes(),
        "ppt/media/image6.png": acc_png.read_bytes(),
    }
    subtitle_replacements = {
        "ppt/slides/slide10.xml": (
            b"Lower is better. Solid = GNN models, dashed = KAI-\xce\xb1 baseline.",
            b"Lower is better. Purple diamonds = before-curriculum 2.5M model.",
        ),
        "ppt/slides/slide11.xml": (
            b"Higher is better. Both GNN models hold skill markedly longer than the baseline.",
            b"Higher is better. Purple diamonds = before-curriculum 2.5M model.",
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".pptx", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with zipfile.ZipFile(source, "r") as zin, zipfile.ZipFile(tmp_path, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename in replacements:
                    data = replacements[info.filename]
                if info.filename in subtitle_replacements:
                    old, new = subtitle_replacements[info.filename]
                    if old not in data:
                        raise ValueError(f"Expected subtitle not found in {info.filename}")
                    data = data.replace(old, new, 1)
                zout.writestr(info, data)
        os.replace(tmp_path, output)
        os.chmod(output, source.stat().st_mode & 0o777)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = args.experiment / "evaluation_test_weekly52_weatherbench2_comparison"
    lead_rmse, new_rmse = read_wide_csv(comparison / "weatherbench2_rollout_rmse.csv")
    lead_acc, new_acc = read_wide_csv(comparison / "weatherbench2_rollout_acc.csv")
    if lead_rmse != lead_acc:
        raise ValueError("RMSE and ACC lead times differ")
    kai_rmse = read_kai(args.baseline, "rmse")
    kai_acc = read_kai(args.baseline, "acc")

    with tempfile.TemporaryDirectory(prefix="comparison_slides_") as tmp:
        tmp_dir = Path(tmp)
        rmse_png = tmp_dir / "comp_2p5_rmse.png"
        acc_png = tmp_dir / "comp_2p5_acc.png"
        make_plot(rmse_png, "rmse", lead_rmse, new_rmse, kai_rmse)
        make_plot(acc_png, "acc", lead_acc, new_acc, kai_acc)
        replace_media(args.source, args.output, rmse_png, acc_png)

    print(args.output)


if __name__ == "__main__":
    main()
