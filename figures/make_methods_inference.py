#!/usr/bin/env python3
"""Build Figure 2: two-stage Rectified Flow inference.

The SVG remains editable. Only the example BRAVO, mask, and Gaussian-noise
tiles are raster images. The displayed pair is an illustrative accepted output
and is not identified as part of either matched evaluation cohort.

Run from the repository root, for example:

    python figures/make_methods_inference.py --case-dir <generated_dir>/<route>/00000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import nibabel as nib
import numpy as np


WIDTH_MM = 180.0
HEIGHT_MM = 96.0
Y_MIN_MM = -2.0
Y_MAX_MM = 94.0
EMBEDDED_RASTER_DPI = 450

NAVY = "#1F4E79"
NAVY_DARK = "#15344F"
MAGENTA = "#A51C61"
MAGENTA_LIGHT = "#F7D9E8"
NOISE_BLUE = "#56B4E9"
NOISE_DARK = "#143B5B"
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
class ExamplePair:
    bravo: np.ndarray
    mask: np.ndarray
    case_dir: Path
    slice_indices: tuple[int, int, int]


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


def load_example(case_dir: Path, slice_indices: tuple[int, int, int]) -> ExamplePair:
    bravo_path = case_dir / "bravo.nii.gz"
    mask_path = case_dir / "seg.nii.gz"
    if not bravo_path.is_file() or not mask_path.is_file():
        raise FileNotFoundError(f"Incomplete BRAVO/mask pair: {case_dir}")

    bravo_nii = nib.load(str(bravo_path))
    mask_nii = nib.load(str(mask_path))
    bravo = np.asarray(bravo_nii.dataobj, dtype=np.float32)
    mask = np.asarray(mask_nii.dataobj, dtype=np.float32)
    if bravo.ndim != 3 or bravo.shape != mask.shape:
        raise ValueError(f"Unexpected pair shape: {bravo.shape}, {mask.shape}")
    if not np.allclose(bravo_nii.affine, mask_nii.affine):
        raise ValueError(f"BRAVO/mask affine mismatch in {case_dir}")
    if min(slice_indices) < 0 or max(slice_indices) >= bravo.shape[2]:
        raise ValueError(f"Slice indices {slice_indices} outside depth {bravo.shape[2]}")

    # Back sheet, middle sheet, front sheet. Slice 70 is deliberately shown in
    # front because it contains the clearest view of the same lesion present in
    # the two adjacent slices.
    bravo_frames = np.stack(
        [np.rot90(bravo[:, :, index], k=1) for index in slice_indices]
    )
    mask_frames = np.stack(
        [np.rot90(mask[:, :, index] > 0.5, k=1) for index in slice_indices]
    )
    return ExamplePair(
        bravo=bravo_frames,
        mask=mask_frames,
        case_dir=case_dir,
        slice_indices=slice_indices,
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
    zorder: float = 7,
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


def _frames(array: np.ndarray) -> list[np.ndarray]:
    if array.ndim == 3 and array.shape[0] == 3:
        return [array[index] for index in range(3)]
    if array.ndim == 4 and array.shape[0] == 3 and array.shape[-1] in (3, 4):
        return [array[index] for index in range(3)]
    raise ValueError(f"Expected three display slices, got {array.shape}")


def add_volume_tile(
    ax: plt.Axes,
    image: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    cmap: str | mpl.colors.Colormap = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
    border: str = GREY_700,
    interpolation: str = "bilinear",
    zorder: float = 4,
) -> None:
    frames = _frames(image)
    layers = [
        (1.50, frames[0], zorder - 2),
        (0.75, frames[1], zorder - 1),
        (0.00, frames[2], zorder),
    ]
    for offset, frame, layer_zorder in layers:
        layer_x = x + offset
        layer_y = y + offset
        ax.imshow(
            frame,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            extent=(layer_x, layer_x + width, layer_y, layer_y + height),
            interpolation=interpolation,
            zorder=layer_zorder,
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


def add_mask_tile(
    ax: plt.Axes,
    mask: np.ndarray,
    x: float,
    y: float,
    width: float,
    height: float,
) -> None:
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[..., :3] = mpl.colors.to_rgb(GREY_900)
    rgba[..., 3] = 1.0
    foreground = mask.astype(bool)
    rgba[foreground, :3] = mpl.colors.to_rgb(MAGENTA)
    add_volume_tile(
        ax,
        rgba,
        x,
        y,
        width,
        height,
        border=MAGENTA,
        interpolation="nearest",
    )


def add_unet(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    symbol: str,
) -> None:
    """Draw the same six-stage 3D U-Net glyph used in Figure 1."""
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
    # This is deliberately identical to Figure 1. Six encoder and decoder
    # stages mirror the production MONAI 3D U-Net, and the horizontal links
    # show the skip paths.
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


def add_sampler_core(
    ax: plt.Axes,
    *,
    center_y: float,
    title: str,
    symbol: str,
    state_symbol: str,
    conditional: bool,
) -> None:
    """Draw one sampler on the shared mask/BRAVO column grid."""
    container_x = 38.0
    container_y = center_y - 12.8
    container_width = 70.0
    container_height = 23.8
    state_x = 40.0
    adapter_x = 42.0
    adapter_width = 14.0
    unet_x = 60.0
    unet_width = 25.0
    unet_height = 20.0
    euler_x = 90.0
    euler_width = 14.0
    euler_height = 12.4

    rounded_box(
        ax,
        container_x,
        container_y,
        container_width,
        container_height,
        facecolor=GREY_100,
        edgecolor=GREY_300,
        linewidth=0.75,
        radius=1.8,
        zorder=3,
    )
    label(
        ax,
        73.0,
        center_y + 12.8,
        title,
        size=7.2,
        weight="bold",
    )

    # Noise initializes the recurrent state at the first update. Subsequent
    # Euler outputs return to this same junction before the next evaluation.
    ax.add_patch(
        mpl.patches.Circle(
            (state_x, center_y),
            radius=0.8,
            facecolor=WHITE,
            edgecolor=GREY_700,
            linewidth=0.75,
            zorder=9,
        )
    )

    if conditional:
        rounded_box(
            ax,
            adapter_x,
            center_y - 6.9,
            adapter_width,
            13.8,
            facecolor=MAGENTA_LIGHT,
            edgecolor=MAGENTA,
            linewidth=0.9,
            radius=1.0,
            zorder=5,
        )
        add_arrow(
            ax,
            (state_x + 1.0, center_y),
            (adapter_x - 0.3, center_y),
            color=GREY_900,
            linewidth=0.85,
            mutation_scale=6.5,
        )
        label(
            ax,
            adapter_x + adapter_width / 2,
            center_y + 3.6,
            "Concatenate",
            size=5.9,
            color=MAGENTA,
            weight="bold",
        )
        label(
            ax,
            adapter_x + adapter_width / 2,
            center_y - 2.2,
            r"$[b_t,m^{\mathrm{cond}}]$",
            size=7.1,
            color=MAGENTA,
        )
        add_arrow(
            ax,
            (adapter_x + adapter_width + 0.6, center_y),
            (unet_x - 0.3, center_y),
            color=MAGENTA,
            linewidth=0.9,
        )
    else:
        label(
            ax,
            adapter_x + adapter_width / 2,
            center_y + 3.2,
            "1 channel",
            size=6.4,
            color=GREY_700,
        )
        add_arrow(
            ax,
            (state_x + 1.0, center_y),
            (unet_x - 0.3, center_y),
            color=GREY_900,
            linewidth=0.9,
        )

    add_unet(
        ax,
        unet_x,
        center_y - unet_height / 2,
        unet_width,
        unet_height,
        symbol,
    )
    add_arrow(
        ax,
        (unet_x + unet_width + 0.8, center_y),
        (euler_x - 0.3, center_y),
        color=NAVY,
        linewidth=0.8,
    )
    rounded_box(
        ax,
        euler_x,
        center_y - euler_height / 2,
        euler_width,
        euler_height,
        facecolor="#EEF7FC",
        edgecolor=NOISE_DARK,
        linewidth=0.75,
        radius=1.0,
        zorder=5,
    )
    label(
        ax,
        euler_x + euler_width / 2,
        center_y + 3.1,
        "Euler update",
        size=5.9,
        color=NOISE_DARK,
        weight="bold",
    )
    label(
        ax,
        euler_x + euler_width / 2,
        center_y - 2.5,
        state_symbol,
        size=7.5,
        color=NAVY_DARK,
    )

    footer_y = center_y - 11.85
    return_path = MplPath(
        [
            (euler_x + euler_width / 2, center_y - euler_height / 2),
            (euler_x + euler_width / 2, footer_y),
            (state_x, footer_y),
            (state_x, center_y - 1.0),
        ],
        [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
    )
    ax.add_patch(
        FancyArrowPatch(
            path=return_path,
            arrowstyle="-|>",
            color=GREY_500,
            linewidth=0.65,
            mutation_scale=6.0,
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
    )
    footer_label = label(
        ax,
        73.0,
        footer_y,
        "100 sequential updates",
        size=5.7,
        color=GREY_700,
        zorder=8,
    )
    footer_label.set_bbox({"facecolor": GREY_100, "edgecolor": "none", "pad": 0.2})


def add_mask_sampler(ax: plt.Axes) -> None:
    add_sampler_core(
        ax,
        center_y=62.75,
        title="Unconditional mask sampler",
        symbol=r"$f_{\theta_m}$",
        state_symbol=r"$m_{t-\Delta t}$",
        conditional=False,
    )


def add_bravo_sampler(ax: plt.Axes) -> None:
    add_sampler_core(
        ax,
        center_y=30.25,
        title="Conditional BRAVO sampler",
        symbol=r"$f_{\theta_b}$",
        state_symbol=r"$b_{t-\Delta t}$",
        conditional=True,
    )


def add_step_axis(ax: plt.Axes) -> None:
    """Draw the shared time axis as a chain of discrete Euler hops.

    A single long arrow says the sampler runs backwards in time but not that it
    does so in fixed jumps. The hops make the discretisation visible: the state
    lands on $t=1.00,0.99,\\ldots,0.01,0$ and never between, so the sequence has
    101 values and 100 hops and the network is never evaluated at $t=0$. The
    elision in the middle stands for the hops that will not fit at this scale, and
    the straight arrow is kept underneath the labels so the overall direction
    still reads at a glance.
    """
    axis_y = 86.4
    x_start, x_end = 66.0, 158.0
    hop = 7.0
    value_y = 84.6
    # Ticks per end. Six is the practical maximum: it leaves a 22 mm elision gap,
    # where seven would leave 8 mm and the two chains would read as very nearly
    # contiguous. The gap is not to scale in any case -- the values either side of
    # it say how much is elided, which a proportionate gap could not.
    n_ticks = 6
    left = [x_start + hop * index for index in range(n_ticks)]
    right = [x_end - hop * (n_ticks - 1 - index) for index in range(n_ticks)]
    values = [f"${1.0 - 0.01 * index:.2f}$" for index in range(n_ticks)]
    values[0] = "$t=1.00$"
    values += [f"${0.01 * (n_ticks - 1 - index):.2f}$" for index in range(n_ticks)]

    # The head runs past the final tick so it reads as the axis terminus rather
    # than colliding with the arrowhead of the hop that lands on $t=0$.
    add_arrow(
        ax,
        (x_start, axis_y),
        (x_end + 2.6, axis_y),
        color=GREY_700,
        linewidth=0.8,
        mutation_scale=6.5,
    )
    for x_tick, text in zip(left + right, values):
        ax.plot(
            [x_tick, x_tick],
            [axis_y - 0.75, axis_y + 0.75],
            color=GREY_700,
            linewidth=0.6,
            zorder=8,
            solid_capstyle="butt",
        )
        label(ax, x_tick, value_y, text, size=5.6, color=GREY_700)

    # Everything above the line is motion, everything below it is a value. Negative
    # rad bows to the left of the travel direction, which is upward here, so each
    # hop arches over the interval it crosses and lands on the next tick.
    for cluster in (left, right):
        for x_from, x_to in zip(cluster, cluster[1:]):
            add_arrow(
                ax,
                (x_from, axis_y + 0.4),
                (x_to, axis_y + 0.4),
                color=GREY_500,
                linewidth=0.65,
                mutation_scale=5.0,
                connectionstyle="arc3,rad=-0.45",
                zorder=8,
            )
    # Sits at hop height rather than on the line, so it elides 95 more hops and
    # not a stretch of the axis. The axis itself stays unbroken underneath.
    label(
        ax,
        (left[-1] + right[0]) / 2,
        axis_y + 1.25,
        r"$\cdots$",
        size=8.0,
        color=GREY_500,
    )


def add_routing_inset(ax: plt.Axes) -> None:
    # Keep the inset fully inside the BRAVO lane. Its left edge follows the
    # sampler container and its right edge follows the QC block.
    rounded_box(
        ax,
        38.0,
        3.2,
        98.5,
        12.4,
        facecolor=WHITE,
        edgecolor=GREY_300,
        linewidth=0.7,
        radius=1.6,
        zorder=3,
    )
    label(ax, 41.5, 14.0, "BRAVO checkpoint routing", size=6.8, weight="bold", ha="left")
    bar_x, bar_w = 68.0, 65.5

    single_y, single_h = 9.45, 2.20
    handoff_y, handoff_h = 6.35, 2.20
    label(ax, 41.5, single_y + single_h / 2, "Single checkpoint", size=6.4, ha="left")
    label(ax, 41.5, handoff_y + handoff_h / 2, "Handoff", size=6.4, ha="left")

    ax.add_patch(
        Rectangle(
            (bar_x, single_y),
            bar_w,
            single_h,
            facecolor=mpl.colors.to_rgba(NAVY, 0.78),
            edgecolor=NAVY_DARK,
            linewidth=0.55,
            zorder=5,
        )
    )
    label(
        ax,
        bar_x + bar_w / 2,
        single_y + single_h / 2,
        "one checkpoint  ·  100",
        size=6.0,
        color=WHITE,
        weight="bold",
    )

    baseline_w = bar_w * 0.75
    specialist_w = bar_w - baseline_w
    ax.add_patch(
        Rectangle(
            (bar_x, handoff_y),
            baseline_w,
            handoff_h,
            facecolor=mpl.colors.to_rgba(NAVY, 0.62),
            edgecolor=NAVY_DARK,
            linewidth=0.55,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (bar_x + baseline_w, handoff_y),
            specialist_w,
            handoff_h,
            facecolor="#C9DAE7",
            edgecolor=NAVY_DARK,
            linewidth=0.55,
            zorder=5,
        )
    )
    label(
        ax,
        bar_x + baseline_w / 2,
        handoff_y + handoff_h / 2,
        "Original MSE  ·  75",
        size=5.8,
        color=WHITE,
        weight="bold",
    )
    label(
        ax,
        bar_x + baseline_w + specialist_w / 2,
        handoff_y + handoff_h / 2,
        "specialist  ·  25",
        size=5.2,
        color=NAVY_DARK,
        weight="bold",
    )
    label(
        ax,
        bar_x + bar_w / 2,
        4.55,
        r"switching $f_{\theta_b}$ at $t=0.25$ while the evolving $b_t$ state continues",
        size=5.5,
        color=GREY_700,
        weight="bold",
    )


def build_figure(example: ExamplePair, seed_mask: int, seed_bravo: int) -> plt.Figure:
    rng_mask = np.random.default_rng(seed_mask)
    rng_bravo = np.random.default_rng(seed_bravo)
    eps_mask = rng_mask.standard_normal(example.mask.shape).astype(np.float32)
    eps_bravo = rng_bravo.standard_normal(example.bravo.shape).astype(np.float32)
    noise_cmap = LinearSegmentedColormap.from_list(
        "figure_noise_inference", [NOISE_DARK, NOISE_BLUE, WHITE]
    )

    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH_MM)
    ax.set_ylim(Y_MIN_MM, Y_MAX_MM)
    ax.set_aspect("equal")
    ax.axis("off")

    # Shared update strip.
    rounded_box(
        ax,
        1.0,
        79.5,
        178.0,
        13.5,
        facecolor=GREY_100,
        edgecolor=GREY_300,
        linewidth=0.75,
        radius=2.0,
    )
    label(ax, 8.0, 89.2, "Reverse-time Rectified Flow sampling", size=8.3, weight="bold", ha="left")
    add_step_axis(ax)
    label(ax, 66.0, 82.0, "Gaussian noise", size=6.5, color=GREY_700)
    label(ax, 112.0, 82.0, r"reverse-time integration, $\Delta t=0.01$", size=6.5, color=GREY_700)
    label(ax, 158.0, 82.0, "synthetic sample", size=6.5, color=GREY_700)
    # Lane containers.
    rounded_box(ax, 1.0, 48.5, 178.0, 28.5, facecolor="#FBFCFD", edgecolor=GREY_200, linewidth=0.65)
    rounded_box(ax, 1.0, 2.0, 178.0, 43.5, facecolor="#FBFCFD", edgecolor=GREY_200, linewidth=0.65)
    ax.add_patch(Rectangle((1.0, 48.5), 4.0, 28.5, facecolor=MAGENTA, edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((1.0, 2.0), 4.0, 43.5, facecolor=NAVY, edgecolor="none", zorder=2))
    label(ax, 8.2, 62.75, "MASK\nSAMPLING", size=7.4, color=MAGENTA, weight="bold", rotation=90)
    label(ax, 8.2, 24.0, "BRAVO\nSAMPLING", size=7.4, color=NAVY, weight="bold", rotation=90)

    # The two source outputs land directly inside one shared pair container.
    # This avoids redrawing either volume or suggesting an additional copy
    # operation after sampling.
    rounded_box(
        ax,
        143.0,
        18.8,
        32.5,
        57.4,
        facecolor=WHITE,
        edgecolor=GREY_300,
        linewidth=0.85,
        radius=2.0,
        zorder=1.5,
    )
    label(ax, 159.25, 50.2, "Paired output", size=7.7, weight="bold")
    label(ax, 159.25, 44.6, r"$(b_0^{\mathrm{syn}},m^{\mathrm{cond}})$", size=9.0, color=NAVY_DARK)

    tile_w = 14.5
    tile_h = 14.5

    # Mask lane.
    add_volume_tile(
        ax,
        eps_mask,
        15.0,
        55.5,
        tile_w,
        tile_h,
        cmap=noise_cmap,
        vmin=-2.5,
        vmax=2.5,
        border=NOISE_DARK,
    )
    label(ax, 22.25, 73.9, r"Independent mask noise  $\epsilon_m$", size=6.8)
    add_arrow(ax, (31.5, 62.75), (39.0, 62.75))
    add_mask_sampler(ax)
    add_arrow(ax, (104.5, 62.75), (114.0, 62.75))
    rounded_box(
        ax,
        114.5,
        55.8,
        22.0,
        13.8,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        linewidth=0.8,
        radius=1.4,
        zorder=4,
    )
    label(ax, 125.5, 66.5, "Clip and binarize", size=6.3, weight="bold")
    label(ax, 125.5, 60.7, r"strict threshold $>0.5$", size=6.1)
    label(ax, 125.5, 72.4, r"Continuous estimate  $\tilde m_0$", size=6.4, color=GREY_700)
    add_arrow(ax, (137.2, 62.75), (145.0, 62.75))
    add_mask_tile(ax, example.mask, 145.5, 55.5, tile_w, tile_h)
    label(
        ax,
        169.5,
        62.75,
        "Fixed binary\n" + r"mask  $m^{\mathrm{cond}}$",
        size=6.1,
        color=MAGENTA,
        weight="bold",
        linespacing=0.85,
    )
    # Fixed conditioning route into every BRAVO update.
    ax.plot(
        [147.0, 147.0, 49.0],
        [55.2, 47.0, 47.0],
        color=MAGENTA,
        linewidth=0.9,
        linestyle="--",
        zorder=6,
    )
    rounded_box(
        ax,
        109.5,
        44.9,
        31.5,
        4.2,
        facecolor=MAGENTA_LIGHT,
        edgecolor=MAGENTA,
        linewidth=0.65,
        radius=0.75,
        zorder=7,
    )
    label(
        ax,
        125.25,
        47.0,
        "same fixed mask at every update",
        size=5.8,
        color=MAGENTA,
        weight="bold",
    )
    add_arrow(
        ax,
        (49.0, 47.0),
        (49.0, 37.45),
        color=MAGENTA,
        linewidth=0.9,
        linestyle="--",
        mutation_scale=6.5,
    )

    # BRAVO lane.
    add_volume_tile(
        ax,
        eps_bravo,
        15.0,
        23.0,
        tile_w,
        tile_h,
        cmap=noise_cmap,
        vmin=-2.5,
        vmax=2.5,
        border=NOISE_DARK,
    )
    label(ax, 22.25, 41.4, r"Independent BRAVO noise  $\epsilon_b$", size=6.8)
    add_arrow(ax, (31.5, 30.25), (39.0, 30.25))
    add_bravo_sampler(ax)
    add_arrow(ax, (104.5, 30.25), (114.0, 30.25))
    rounded_box(
        ax,
        114.5,
        23.35,
        22.0,
        13.8,
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        linewidth=0.8,
        radius=1.4,
        zorder=4,
    )
    label(ax, 125.5, 34.45, "QC and masking", size=6.0, weight="bold")
    label(ax, 125.5, 31.35, r"Clamp to $[0,1]$", size=5.8)
    label(ax, 125.5, 28.45, "PCA containment", size=5.8)
    label(ax, 125.5, 25.55, "Zero outside brain", size=5.8)
    add_arrow(ax, (137.2, 30.25), (145.0, 30.25))
    add_volume_tile(
        ax,
        example.bravo,
        145.5,
        23.0,
        tile_w,
        tile_h,
        cmap="gray",
        vmin=0.0,
        vmax=0.5,
        border=NAVY_DARK,
    )
    label(
        ax,
        169.5,
        30.25,
        "Generated\n" + r"BRAVO  $b_0^{\mathrm{syn}}$",
        size=6.1,
        color=NAVY_DARK,
        weight="bold",
        linespacing=0.85,
    )

    add_routing_inset(ax)

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path(
            "/path/to/MedicalDataSets/generated/"
            "exp48c_handoff_exp32/00000"
        ),
        help="Illustrative accepted <case> directory containing bravo.nii.gz and seg.nii.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build",
        help="Working directory for methods_inference.{svg,pdf} and PNG preview.",
    )
    parser.add_argument("--mask-noise-seed", type=int, default=3042)
    parser.add_argument("--bravo-noise-seed", type=int, default=4042)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    example = load_example(args.case_dir, (69, 71, 70))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure = build_figure(example, args.mask_noise_seed, args.bravo_noise_seed)
    common = {"bbox_inches": None, "pad_inches": 0, "facecolor": WHITE, "edgecolor": WHITE}
    figure.savefig(
        args.output_dir / "methods_inference.svg",
        format="svg",
        dpi=EMBEDDED_RASTER_DPI,
        metadata={"Title": "Two-stage Rectified Flow inference"},
        **common,
    )
    figure.savefig(
        args.output_dir / "methods_inference.pdf",
        format="pdf",
        dpi=EMBEDDED_RASTER_DPI,
        metadata={
            "Title": "Two-stage Rectified Flow inference",
            "Author": "Modestas Sukarevicius et al.",
            "Subject": "Generator inference schematic",
        },
        **common,
    )
    figure.savefig(
        args.output_dir / "methods_inference_preview.png",
        format="png",
        dpi=220,
        metadata={"Title": "Two-stage Rectified Flow inference"},
        **common,
    )
    plt.close(figure)

    print("Figure 2 assets written to", args.output_dir)
    print(
        "Illustrative accepted output:",
        f"case={example.case_dir.name}, slices={example.slice_indices}",
    )
    print(
        "Independent display-noise seeds:",
        f"mask={args.mask_noise_seed}, BRAVO={args.bravo_noise_seed}",
    )


if __name__ == "__main__":
    main()
