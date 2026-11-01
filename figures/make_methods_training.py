#!/usr/bin/env python3
"""Build Figure 1: training of the two 3D Rectified Flow generators.

The vector geometry and text are kept editable in the SVG. Only the example
MRI, mask, and noise tiles are raster images. The illustrative case is selected
deterministically from the training split without exposing its identifier in
the exported figure.

Run from the repository root, for example:

    python figures/make_methods_training.py --data-root <data_dir>/train
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import nibabel as nib
import numpy as np


WIDTH_MM = 192.0
HEIGHT_MM = 104.0
EMBEDDED_RASTER_DPI = 450

NAVY = "#1F4E79"
NAVY_DARK = "#15344F"
MAGENTA = "#A51C61"
MAGENTA_LIGHT = "#F7D9E8"
NOISE_BLUE = "#56B4E9"
NOISE_DARK = "#143B5B"
GREEN = "#009E73"
ORANGE = "#E69F00"
ORANGE_LIGHT = "#FFF1D6"
GREY_900 = "#20242A"
GREY_700 = "#4C5661"
GREY_500 = "#8D98A5"
GREY_300 = "#CFD6DD"
GREY_200 = "#E7EBEF"
GREY_100 = "#F6F8FA"
WHITE = "#FFFFFF"


@dataclass(frozen=True)
class ExampleSlice:
    bravo: np.ndarray
    mask: np.ndarray
    case_dir: Path
    axial_index: int
    lesion_area: int
    tissue_area: int


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "image.composite_image": False,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
        }
    )


def _load_pair(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    bravo_path = case_dir / "bravo.nii.gz"
    mask_path = case_dir / "seg.nii.gz"
    if not bravo_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Incomplete BRAVO/mask pair: {case_dir}")

    bravo_nii = nib.load(str(bravo_path))
    mask_nii = nib.load(str(mask_path))
    bravo = np.asarray(bravo_nii.dataobj, dtype=np.float32)
    mask = np.asarray(mask_nii.dataobj, dtype=np.float32) > 0.5
    if bravo.shape != mask.shape or bravo.ndim != 3:
        raise ValueError(f"Unexpected pair shape in {case_dir}: {bravo.shape}, {mask.shape}")
    if not np.allclose(bravo_nii.affine, mask_nii.affine):
        raise ValueError(f"BRAVO/mask affine mismatch in {case_dir}")
    return bravo, mask


def select_example(data_root: Path) -> ExampleSlice:
    """Select a visible but non-extreme training example deterministically.

    The target is the 75th percentile of each case's maximum axial lesion area.
    Ties are resolved by the amount of visible tissue on the selected slice,
    then by the sorted internal case identifier. The latter is never drawn.
    """

    records: list[tuple[Path, int, int, int, np.ndarray, np.ndarray]] = []
    for case_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        bravo_path = case_dir / "bravo.nii.gz"
        mask_path = case_dir / "seg.nii.gz"
        if not bravo_path.is_file() or not mask_path.is_file():
            continue
        bravo, mask = _load_pair(case_dir)
        axial_areas = mask.sum(axis=(0, 1))
        axial_index = int(np.argmax(axial_areas))
        lesion_area = int(axial_areas[axial_index])
        if lesion_area <= 0:
            continue

        value_range = float(bravo.max() - bravo.min())
        bravo_scaled = (bravo - bravo.min()) / max(value_range, 1e-8)
        tissue_area = int((bravo_scaled[:, :, axial_index] > 0.05).sum())
        records.append(
            (case_dir, axial_index, lesion_area, tissue_area, bravo_scaled, mask)
        )

    if not records:
        raise RuntimeError(f"No complete foreground-containing pairs found in {data_root}")

    target_area = float(np.percentile([record[2] for record in records], 75))
    closest_distance = min(abs(record[2] - target_area) for record in records)
    closest = [record for record in records if abs(record[2] - target_area) == closest_distance]
    chosen = sorted(closest, key=lambda record: (-record[3], record[0].name))[0]
    case_dir, axial_index, lesion_area, tissue_area, bravo, mask = chosen

    slice_indices = [
        max(0, axial_index - 2),
        max(0, axial_index - 1),
        axial_index,
    ]
    return ExampleSlice(
        bravo=np.stack([np.rot90(bravo[:, :, index]) for index in slice_indices]),
        mask=np.stack([np.rot90(mask[:, :, index]) for index in slice_indices]),
        case_dir=case_dir,
        axial_index=axial_index,
        lesion_area=lesion_area,
        tissue_area=tissue_area,
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = WHITE,
    edgecolor: str = GREY_300,
    linewidth: float = 0.8,
    radius: float = 1.5,
    zorder: float = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.35,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GREY_900,
    linewidth: float = 0.9,
    linestyle: str = "-",
    mutation_scale: float = 7.5,
    connectionstyle: str = "arc3",
    zorder: float = 6,
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        mutation_scale=mutation_scale,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 8.0,
    color: str = GREY_900,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    zorder: float = 10,
    linespacing: float = 0.95,
    rotation: float = 0,
) -> mpl.text.Text:
    return ax.text(
        x,
        y,
        text,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=zorder,
        linespacing=linespacing,
        rotation=rotation,
    )


def add_volume_tile(
    ax: plt.Axes,
    image: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    tile_label: str,
    *,
    cmap: str | mpl.colors.Colormap = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    mask_contour: np.ndarray | None = None,
    border: str = GREY_700,
    label_size: float = 7.4,
    label_offset: float = 2.4,
    zorder: float = 4,
) -> None:
    def image_frames(array: np.ndarray) -> list[np.ndarray]:
        if array.ndim == 2 or (array.ndim == 3 and array.shape[-1] in (3, 4)):
            return [array, array, array]
        if array.ndim == 3 or (array.ndim == 4 and array.shape[-1] in (3, 4)):
            if array.shape[0] != 3:
                raise ValueError(f"Expected exactly three display slices, got {array.shape}")
            return [array[index] for index in range(3)]
        raise ValueError(f"Unsupported image-tile shape: {array.shape}")

    frames = image_frames(image)
    if mask_contour is None:
        contour_frames: list[np.ndarray | None] = [None, None, None]
    else:
        contour_frames = image_frames(mask_contour)

    # The front image is the selected slice. The two visible sheets behind it
    # are the two immediately preceding axial slices from the same volume.
    layers = [
        (1.50, frames[0], contour_frames[0], zorder - 2),
        (0.75, frames[1], contour_frames[1], zorder - 1),
        (0.00, frames[2], contour_frames[2], zorder),
    ]
    for offset, frame, contour, layer_zorder in layers:
        layer_x = x + offset
        layer_y = y + offset
        ax.imshow(
            frame,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=(layer_x, layer_x + width, layer_y, layer_y + height),
            interpolation="bilinear",
            zorder=layer_zorder,
        )
        if contour is not None and np.any(contour):
            yy, xx = np.mgrid[0 : contour.shape[0], 0 : contour.shape[1]]
            x_coords = layer_x + xx / (contour.shape[1] - 1) * width
            y_coords = layer_y + (1.0 - yy / (contour.shape[0] - 1)) * height
            ax.contour(
                x_coords,
                y_coords,
                contour.astype(float),
                levels=[0.5],
                colors=[MAGENTA],
                linewidths=0.8,
                zorder=layer_zorder + 0.2,
            )
        ax.add_patch(
            Rectangle(
                (layer_x, layer_y),
                width,
                height,
                fill=False,
                edgecolor=border if offset == 0 else GREY_500,
                linewidth=0.65 if offset == 0 else 0.45,
                zorder=layer_zorder + 0.4,
            )
        )
    label(ax, x + width / 2, y - label_offset, tile_label, size=label_size, va="top")


def add_mask_tile(
    ax: plt.Axes,
    mask: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    tile_label: str,
    *,
    label_size: float = 7.4,
    label_offset: float = 2.4,
) -> None:
    background = np.zeros((*mask.shape, 4), dtype=float)
    background[..., :3] = mpl.colors.to_rgb(GREY_900)
    background[..., 3] = 1.0
    foreground = mask.astype(bool)
    background[foreground, :3] = mpl.colors.to_rgb(MAGENTA)
    background[foreground, 3] = 1.0
    add_volume_tile(
        ax,
        background,
        x,
        y,
        width,
        height,
        tile_label,
        border=MAGENTA,
        label_size=label_size,
        label_offset=label_offset,
    )


def add_unet(ax: plt.Axes, x: float, y: float, width: float, height: float, symbol: str) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=WHITE,
        edgecolor=NAVY,
        linewidth=1.15,
        radius=1.8,
        zorder=4,
    )
    # Six encoder and decoder stages mirror the production MONAI 3D U-Net.
    # Horizontal links show the skip paths. Exact widths remain in the appendix.
    levels = list(range(6))
    block_w = 1.25
    block_h = 2.15
    x_step = 1.55
    y_step = 1.70
    x_left = [x + 2.2 + x_step * level for level in levels]
    y_left = [y + height - 4.2 - y_step * level for level in levels]
    x_right = [x + width - 2.2 - block_w - x_step * level for level in levels]
    y_right = y_left

    for bx_left, bx_right, by in zip(x_left, x_right, y_left):
        ax.plot(
            [bx_left + block_w, bx_right],
            [by + block_h / 2, by + block_h / 2],
            color=mpl.colors.to_rgba(NAVY, 0.36),
            linewidth=0.55,
            zorder=4.5,
        )

    for index, (bx, by) in enumerate(zip(x_left, y_left)):
        ax.add_patch(
            Rectangle(
                (bx, by),
                block_w,
                block_h,
                facecolor=mpl.colors.to_rgba(NAVY, 0.12 + 0.065 * index),
                edgecolor=NAVY,
                linewidth=0.48,
                zorder=6,
            )
        )
        if index:
            ax.plot(
                [x_left[index - 1] + block_w, bx],
                [y_left[index - 1] + block_h / 2, by + block_h / 2],
                color=NAVY,
                linewidth=0.48,
                zorder=5,
            )
    for index, (bx, by) in enumerate(zip(x_right, y_right)):
        ax.add_patch(
            Rectangle(
                (bx, by),
                block_w,
                block_h,
                facecolor=mpl.colors.to_rgba(NAVY, 0.12 + 0.065 * index),
                edgecolor=NAVY,
                linewidth=0.48,
                zorder=6,
            )
        )
        if index:
            ax.plot(
                [x_right[index - 1], bx + block_w],
                [y_right[index - 1] + block_h / 2, by + block_h / 2],
                color=NAVY,
                linewidth=0.48,
                zorder=5,
            )
    ax.plot(
        [x_left[-1] + block_w, x_right[-1]],
        [y_left[-1] + block_h / 2, y_right[-1] + block_h / 2],
        color=NAVY,
        linewidth=0.65,
        zorder=5,
    )
    label(ax, x + width / 2, y + 3.0, symbol, size=9.0, color=NAVY_DARK)


def add_mix_node(ax: plt.Axes, x: float, y: float, time_symbol: str) -> None:
    rounded_box(
        ax,
        x,
        y,
        17.6,
        11.8,
        facecolor="#EEF7FC",
        edgecolor=NOISE_DARK,
        linewidth=0.8,
        radius=2.0,
        zorder=3,
    )
    label(ax, x + 8.8, y + 7.5, "Rectified Flow", size=6.7, color=NOISE_DARK, weight="bold")
    label(ax, x + 8.8, y + 3.4, f"interpolation at ${time_symbol}$", size=6.1, color=GREY_700)


def add_time_input(ax: plt.Axes, x: float, y: float, symbol: str) -> None:
    ax.add_patch(
        mpl.patches.Circle(
            (x, y),
            radius=2.25,
            facecolor=WHITE,
            edgecolor=NAVY,
            linewidth=0.75,
            zorder=8,
        )
    )
    label(ax, x, y + 0.1, f"${symbol}$", size=8.2, color=NAVY_DARK)


def add_time_sampling_axis(ax: plt.Axes) -> None:
    """Draw where the training time comes from: one draw per example over [0,1].

    Figure 2's strip shows $t$ being swept, and without a counterpart here a reader
    can carry that reading into training, where $t$ is instead drawn afresh for
    every example. The axis is deliberately NOT an arrow for that reason -- it is a
    range sampled from, not a direction integrated along.

    The curve is the exact logistic-normal density with location 0 and scale 1,
    p(t) = phi(logit t) / (t(1-t)), which is what MONAI's RFlowScheduler draws from
    under sample_method='logit-normal'. Drawing it uniform instead would be the
    natural assumption and would hide why the low-$t$ specialists of Section 3.2.3
    exist at all -- only 13.6% of draws land below t=0.25, against 25% for a uniform
    draw. That figure is left to the prose. Shading and labelling the tail here
    would put a second argument into a strip whose job is to say where $t$ comes
    from, and the taper is visible in the curve without it.

    Orientation matches Figure 2 ($t=1$ left, $t=0$ right), so the sparse tail here
    sits at the same end as the interval the specialists take over there.
    """
    # Occupies the band right of the title and above the three-stage row, so the
    # roadmap below it survives and the strip keeps its height. The axis stops at
    # 160 to leave the caption a column of its own rather than stacking it.
    axis_y = 96.4
    x_left, x_right = 72.0, 160.0
    peak_mm = 5.0
    value_y = 94.9

    def to_x(t):
        return x_left + (1.0 - t) * (x_right - x_left)

    def density(t):
        z = np.log(t / (1.0 - t))
        return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi) / (t * (1.0 - t))

    grid = np.linspace(1e-4, 1 - 1e-4, 1200)
    dens = density(grid)
    curve_y = axis_y + dens / dens.max() * peak_mm
    ax.fill_between(to_x(grid), axis_y, curve_y,
                    facecolor=mpl.colors.to_rgba(NAVY, 0.13), edgecolor="none", zorder=4)
    ax.plot(to_x(grid), curve_y, color=NAVY, linewidth=0.75, zorder=6)

    ax.plot([x_left, x_right], [axis_y, axis_y], color=GREY_700, linewidth=0.8, zorder=6)
    for t in (1.0, 0.75, 0.5, 0.25, 0.0):
        ax.plot([to_x(t), to_x(t)], [axis_y - 0.6, axis_y + 0.6],
                color=GREY_700, linewidth=0.6, zorder=7, solid_capstyle="butt")
        text = f"$t={t:.2f}$" if t == 1.0 else f"${t:.2f}$"
        label(ax, to_x(t), value_y, text, size=5.6, color=GREY_700)

    # Individual draws were shown here as marks on the axis and removed: twelve
    # points cannot look like a sample without a seed chosen to make them look like
    # one, and the caption already states that a draw is per example. The curve is
    # the claim, so it carries it alone.
    label(ax, x_right + 2.5, axis_y + 2.2, "logit-normal,\none draw per example",
          size=5.6, color=GREY_700, ha="left", linespacing=1.2)


def build_figure(example: ExampleSlice, seed_mask: int, seed_bravo: int,
                 time_value: float) -> plt.Figure:
    rng_mask = np.random.default_rng(seed_mask)
    rng_bravo = np.random.default_rng(seed_bravo)
    eps_mask = rng_mask.standard_normal(example.mask.shape).astype(np.float32)
    eps_bravo = rng_bravo.standard_normal(example.bravo.shape).astype(np.float32)

    clean_mask = example.mask.astype(np.float32)
    noisy_mask = (1.0 - time_value) * clean_mask + time_value * eps_mask
    noisy_bravo = (1.0 - time_value) * example.bravo + time_value * eps_bravo

    def display_scale(array: np.ndarray) -> np.ndarray:
        low, high = np.percentile(array, [1.0, 99.0])
        return np.clip((array - low) / max(float(high - low), 1e-8), 0.0, 1.0)

    noise_cmap = LinearSegmentedColormap.from_list(
        "figure_noise", [NOISE_DARK, NOISE_BLUE, WHITE]
    )

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(0, HEIGHT_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    rounded_box(
        ax,
        1.0,
        88.5,
        190.0,
        14.2,
        facecolor=GREY_100,
        edgecolor=GREY_300,
        linewidth=0.75,
        radius=2.0,
    )
    label(ax, 8.0, 98.4, "Shared Rectified Flow training", size=8.3, weight="bold", ha="left")
    add_time_sampling_axis(ax)
    # Dropped from 93.4 to clear the time axis above it.
    label(ax, 39.0, 91.6, "Clean target and Gaussian noise", size=7.6, color=NAVY_DARK)
    add_arrow(ax, (61.5, 91.6), (73.0, 91.6), color=GREY_500, linewidth=0.7)
    label(ax, 96.0, 91.6, "Interpolate at a sampled time", size=7.6, color=NAVY_DARK)
    add_arrow(ax, (118.5, 91.6), (130.0, 91.6), color=GREY_500, linewidth=0.7)
    label(ax, 154.0, 91.6, "Regress denoising velocity", size=7.6, color=NAVY_DARK)

    rounded_box(ax, 1.0, 54.3, 190.0, 31.2, facecolor="#FBFCFD", edgecolor=GREY_200, linewidth=0.65)
    rounded_box(ax, 1.0, 2.0, 190.0, 49.2, facecolor="#FBFCFD", edgecolor=GREY_200, linewidth=0.65)
    ax.add_patch(Rectangle((1.0, 54.3), 4.0, 31.2, facecolor=MAGENTA, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((1.0, 2.0), 4.0, 49.2, facecolor=NAVY, edgecolor="none", zorder=2))
    label(ax, 8.2, 69.9, "MASK\nTRAINING", size=7.5, color=MAGENTA, weight="bold", rotation=90)
    label(ax, 8.2, 31.1, "BRAVO\nTRAINING", size=7.5, color=NAVY, weight="bold", rotation=90)

    tile_w = 14.5
    tile_h = 14.5

    add_mask_tile(ax, example.mask, 14.0, 64.0, tile_w, tile_h, "Real mask\n$m_0$")
    add_volume_tile(
        ax, eps_mask, 35.0, 64.0, tile_w, tile_h, r"Noise" + "\n" + r"$\epsilon_m$",
        cmap=noise_cmap, vmin=-2.5, vmax=2.5, border=NOISE_DARK,
    )
    add_mix_node(ax, 55.5, 65.8, "t_m")
    add_volume_tile(
        ax, display_scale(noisy_mask), 77.0, 64.0, tile_w, tile_h,
        "Noisy mask\n$m_{t_m}$", cmap="gray", vmin=0, vmax=1,
    )
    label(ax, 32.4, 71.3, "+", size=13.0, color=GREY_700)
    add_arrow(ax, (51.8, 71.3), (55.5, 71.3))
    add_arrow(ax, (73.9, 71.3), (77.0, 71.3))
    add_arrow(ax, (93.8, 71.3), (112.0, 71.3))
    label(ax, 102.9, 73.1, "1 channel", size=6.8, color=GREY_700)
    add_unet(ax, 112.0, 61.0, 25.0, 20.0, r"$f_{\theta_m}$")
    label(ax, 124.5, 58.0, "Unconditional mask generator", size=7.3, color=NAVY_DARK)
    add_time_input(ax, 124.5, 83.1, "t_m")
    add_arrow(ax, (124.5, 80.9), (124.5, 78.4), color=NAVY, linewidth=0.8)
    add_arrow(ax, (137.8, 71.3), (144.0, 71.3))
    rounded_box(ax, 144.0, 64.0, 13.8, 14.5, facecolor=WHITE, edgecolor=NAVY, linewidth=0.8)
    label(ax, 150.9, 73.4, "Predicted", size=6.9, color=GREY_700)
    label(ax, 150.9, 69.7, "velocity", size=7.2, color=GREY_700)
    label(ax, 150.9, 65.9, r"$\hat v_m$", size=9.0, color=NAVY_DARK)
    add_arrow(ax, (158.5, 71.3), (163.0, 71.3))
    rounded_box(ax, 163.0, 64.0, 23.4, 14.5, facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=0.9)
    label(ax, 174.7, 74.8, "Mask loss", size=7.4, weight="bold")
    label(ax, 174.7, 68.8, "MSE velocity\nregression", size=6.9, linespacing=0.86)
    ax.add_patch(Rectangle((100.0, 51.7), 49.0, 2.3, facecolor=WHITE, edgecolor="none", zorder=8))
    label(
        ax,
        124.5,
        52.85,
        "Shared six-stage backbone  •  independent weights",
        size=6.5,
        color=NAVY_DARK,
    )

    add_volume_tile(
        ax, example.bravo, 14.0, 26.0, tile_w, tile_h, "Real BRAVO\n$b_0$",
        cmap="gray", vmin=0, vmax=1, mask_contour=example.mask, label_offset=0.8,
    )
    add_volume_tile(
        ax, eps_bravo, 35.0, 26.0, tile_w, tile_h, r"Noise" + "\n" + r"$\epsilon_b$",
        cmap=noise_cmap, vmin=-2.5, vmax=2.5, border=NOISE_DARK,
    )
    add_mix_node(ax, 55.5, 27.8, "t_b")
    add_volume_tile(
        ax, display_scale(noisy_bravo), 77.0, 26.0, tile_w, tile_h,
        "Noisy BRAVO\n$b_{t_b}$", cmap="gray", vmin=0, vmax=1,
    )
    label(ax, 32.4, 33.3, "+", size=13.0, color=GREY_700)
    add_arrow(ax, (51.8, 33.3), (55.5, 33.3))
    add_arrow(ax, (73.9, 33.3), (77.0, 33.3))
    add_arrow(ax, (93.8, 33.3), (97.0, 33.3))

    add_mask_tile(
        ax,
        example.mask,
        97.75,
        5.8,
        9.0,
        9.0,
        "Paired mask  $m_0$",
        label_size=5.8,
        label_offset=1.5,
    )
    rounded_box(ax, 97.0, 26.2, 10.5, 14.2, facecolor=MAGENTA_LIGHT, edgecolor=MAGENTA, linewidth=0.9)
    label(ax, 102.25, 36.4, "Concatenate", size=5.5, color=MAGENTA, weight="bold")
    label(ax, 102.25, 32.0, r"$[b_{t_b},m_0]$", size=7.1, color=MAGENTA)
    label(ax, 102.25, 28.2, "2 channels", size=6.0, color=MAGENTA)
    add_arrow(
        ax, (102.25, 16.3), (102.25, 26.2), color=MAGENTA, linewidth=1.0,
        linestyle="--",
    )
    add_arrow(ax, (108.3, 33.3), (112.0, 33.3), color=MAGENTA, linewidth=1.0)
    add_unet(ax, 112.0, 23.0, 25.0, 20.0, r"$f_{\theta_b}$")
    label(ax, 124.5, 20.0, "Conditional BRAVO generator", size=7.3, color=NAVY_DARK)
    add_time_input(ax, 124.5, 45.0, "t_b")
    add_arrow(ax, (124.5, 42.8), (124.5, 40.4), color=NAVY, linewidth=0.8)
    add_arrow(ax, (137.8, 33.3), (144.0, 33.3))
    rounded_box(ax, 144.0, 26.0, 13.3, 14.5, facecolor=WHITE, edgecolor=NAVY, linewidth=0.8)
    label(ax, 150.65, 35.4, "Predicted", size=6.9, color=GREY_700)
    label(ax, 150.65, 31.7, "velocity", size=7.2, color=GREY_700)
    label(ax, 150.65, 27.9, r"$\hat v_b$", size=9.0, color=NAVY_DARK)
    add_arrow(ax, (158.0, 33.3), (163.0, 33.3))
    rounded_box(ax, 163.0, 21.8, 23.4, 21.2, facecolor=ORANGE_LIGHT, edgecolor=ORANGE, linewidth=0.9)
    label(ax, 174.7, 39.2, "BRAVO loss\ncomponents", size=6.8, weight="bold", linespacing=0.82)
    label(ax, 174.7, 34.6, "Velocity regression", size=6.8)
    label(ax, 174.7, 31.1, "+", size=8.0, color=GREY_700)
    label(
        ax,
        174.7,
        26.7,
        "Optional RadImageNet\nfeature term on $\\hat b_0$",
        size=6.1,
        color=GREY_700,
        # Wider than the other labels: at 0.84 the hat of b_0 met the descender of the g above and vanished.
        linespacing=1.12,
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path,
        default=Path("/path/to/MedicalDataSets/brainmetshare-3/train"),
        help="Training split containing <case>/{bravo,seg}.nii.gz pairs.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "build",
        help="Working directory for methods_training.{svg,pdf} and PNG preview.",
    )
    parser.add_argument("--mask-seed", type=int, default=1042)
    parser.add_argument("--bravo-seed", type=int, default=2042)
    parser.add_argument("--display-time", type=float, default=0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.display_time <= 1.0:
        raise ValueError("--display-time must be in [0, 1]")
    configure_matplotlib()
    example = select_example(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure = build_figure(example, args.mask_seed, args.bravo_seed, args.display_time)
    common = {"bbox_inches": None, "pad_inches": 0, "facecolor": WHITE, "edgecolor": WHITE}
    figure.savefig(
        args.output_dir / "methods_training.svg", format="svg",
        dpi=EMBEDDED_RASTER_DPI,
        metadata={"Title": "Training of the 3D Rectified Flow generators"}, **common,
    )
    figure.savefig(
        args.output_dir / "methods_training.pdf", format="pdf",
        dpi=EMBEDDED_RASTER_DPI,
        metadata={
            "Title": "Training of the 3D Rectified Flow generators",
            "Author": "Modestas Sukarevicius et al.",
            "Subject": "Generator training schematic",
        }, **common,
    )
    figure.savefig(
        args.output_dir / "methods_training_preview.png", format="png", dpi=220,
        metadata={"Title": "Training of the 3D Rectified Flow generators"}, **common,
    )
    plt.close(figure)

    print("Figure 1 assets written to", args.output_dir)
    print(
        "Illustrative training slice selected deterministically:",
        f"case={example.case_dir.name}, axial_index={example.axial_index},",
        f"lesion_area={example.lesion_area}, tissue_area={example.tissue_area}",
    )
    print(
        f"Display parameters: t={args.display_time}, mask_seed={args.mask_seed},",
        f"bravo_seed={args.bravo_seed}",
    )


if __name__ == "__main__":
    main()
