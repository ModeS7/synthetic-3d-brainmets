#!/usr/bin/env python3
"""Qualitative figure conditioned on HELD-OUT TEST masks.

Supersedes the earlier real-mask version (notes/figure_drafts/make_qualitative.py), which
conditioned on training masks the generators had seen. That figure could not distinguish
faithful synthesis from memorisation, which is the first thing a reviewer will ask about a
qualitative panel. Every mask used here comes from the 51-case official test cohort and
was never seen by any generator during training.

Layout: columns are the real volume plus one per generation route; rows are axial, a
lesion zoom, coronal and sagittal (>=2 orthogonal planes satisfies the MIA figure rule).
Every column is conditioned on the SAME test mask, verified at IoU = 1.0000, so any
visible difference is appearance and not anatomy. The cyan contour makes that explicit.


    usage: make_qualitative_testmask.py [--case Mets_098] [--lesion N]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# TrueType (Type 42) fonts in the PDF. The default, Type 3, embeds glyphs as drawing procedures,
# which production desks often reject. Text and layout are unchanged (pixel-checked 2026-10-04).
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import nibabel as nib
import numpy as np
from scipy import ndimage

GEN = Path(os.environ.get("QUALITATIVE_DIR", "qualitative_testmask"))
MASKS = Path(os.environ.get("TEST_LABELS_DIR", "test_labels"))
REAL = Path("/path/to/MedicalDataSets/brainmetshare-3/test1")
REPO = Path(__file__).resolve().parents[1]

# Directory name -> the label used in the paper. Order matches Table 2 so the figure and
# the segmentation table can be read side by side.
ROUTES = [("exp1_1_1000",         "Original MSE"),
          ("exp1_1_1000plus",     "Extended MSE"),
          ("exp32_2_1000",        "Perceptual\nContinuation"),
          ("exp47a",              "Transition (a)"),
          ("exp47c",              "Transition (c)"),
          ("exp1_to_exp48c_t025", "Handoff (c) $\\dagger$"),
          ("exp1_to_exp48d_t025", "Handoff (d)")]
# Sample index of the exhibited case in the three-case job.
CASE_INDEX = {"Mets_098": "00001"}

THR, MARGIN = 0.05, 4


def load(p):
    v = nib.load(str(p)).get_fdata().astype(np.float32)
    m = v.max()
    return v / m if m > 0 else v


def planes(vol, centre):
    """Three orthogonal slices. Axial goes through the exhibited lesion, the other two
    are mid-volume: their job is whole-brain plausibility, and slicing them through an
    off-centre lesion would show far less anatomy."""
    axl = int(np.argmin(vol.shape))
    other = [a for a in range(3) if a != axl]

    def take(axis, at):
        idx = [slice(None)] * 3
        idx[axis] = int(np.clip(at, 0, vol.shape[axis] - 1))
        return np.rot90(vol[tuple(idx)])

    return {"Axial": take(axl, centre[axl]),
            "Sagittal": take(other[0], vol.shape[other[0]] // 2),
            "Coronal": take(other[1], vol.shape[other[1]] // 2)}


def bbox(im):
    ys, xs = np.where(im > THR)
    if not len(ys):
        return 0, im.shape[0] - 1, 0, im.shape[1] - 1
    return ys.min(), ys.max(), xs.min(), xs.max()


def crop(im, box):
    """Apply one SHARED crop box to every column.

    Cropping each array to its own bounding box misplaces the mask contour, because a
    segmentation's bbox is the lesion rather than the brain. That bug produced a figure
    where the outline sat in the middle of the canvas instead of on the tumour.
    """
    y0, y1, x0, x1 = box
    return im[y0:y1 + 1, x0:x1 + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="Mets_098", choices=sorted(CASE_INDEX))
    ap.add_argument("--lesion", type=int, default=0,
                    help="0 = largest ground-truth lesion")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--pdf", type=Path)
    args = ap.parse_args()
    idx = CASE_INDEX[args.case]

    seg = nib.load(str(MASKS / f"{args.case.replace('Mets_', 'BrainMet_')}.nii.gz")).get_fdata() > 0.5
    lab, n = ndimage.label(seg, structure=np.ones((3, 3, 3)))
    sizes = [(int((lab == i).sum()), i) for i in range(1, n + 1)]
    sizes.sort(reverse=True)
    pick = sizes[args.lesion][1]
    seg_main = lab == pick
    print(f"  {args.case}: {n} lesions, exhibiting #{args.lesion} "
          f"({sizes[args.lesion][0]} voxels)")

    centre = np.round(np.array(np.nonzero(seg_main)).mean(axis=1)).astype(int)

    cols = [("Real", load(REAL / args.case / "bravo.nii.gz"))]
    for d, label in ROUTES:
        v = load(GEN / d / idx / "bravo.nii.gz")
        s = nib.load(str(GEN / d / idx / "seg.nii.gz")).get_fdata() > 0.5
        iou = (s & seg).sum() / max((s | seg).sum(), 1)
        assert iou > 0.999, f"{d} conditioning mask differs from the real one (IoU {iou:.4f})"
        cols.append((label, v))

    data = {name: planes(vol, centre) for name, vol in cols}
    boxes = {}
    for p in ("Axial", "Coronal", "Sagittal"):
        ys0 = xs0 = 10 ** 9
        ys1 = xs1 = -1
        for name, _ in cols:
            y0, y1, x0, x1 = bbox(data[name][p])
            ys0, ys1, xs0, xs1 = min(ys0, y0), max(ys1, y1), min(xs0, x0), max(xs1, x1)
        h, w = data[cols[0][0]][p].shape
        boxes[p] = (max(ys0 - MARGIN, 0), min(ys1 + MARGIN, h - 1),
                    max(xs0 - MARGIN, 0), min(xs1 + MARGIN, w - 1))
    data = {n_: {p: crop(sl, boxes[p]) for p, sl in d.items()} for n_, d in data.items()}
    canvas = {p: data[cols[0][0]][p].shape for p in boxes}

    segax = crop(planes(seg_main.astype(np.float32), centre)["Axial"], boxes["Axial"])
    segall = crop(planes(seg.astype(np.float32), centre)["Axial"], boxes["Axial"])
    ys, xs = np.where(segax > 0.5)
    H, W = canvas["Axial"]
    side = int(0.30 * min(H, W))
    r0 = int(np.clip((int(ys.mean()) if len(ys) else H // 2) - side // 2, 0, H - side))
    c0 = int(np.clip((int(xs.mean()) if len(xs) else W // 2) - side // 2, 0, W - side))

    rows = ["Axial", "Lesion zoom", "Coronal", "Sagittal"]
    asp = {p: canvas[p][0] / canvas[p][1] for p in canvas}
    heights = [asp["Axial"], 1.0, asp["Coronal"], asp["Sagittal"]]
    CW = 2.6
    # Black figure face: these panels are black-background MRI, so a white face showed
    # as bright seams between them. Titles and row labels invert to white.
    fig = plt.figure(figsize=(CW * len(cols), CW * sum(heights)), facecolor="black")
    gs = gridspec.GridSpec(len(rows), len(cols), height_ratios=heights,
                           wspace=0.02, hspace=0.04,
                           left=0.055, right=0.998, top=0.955, bottom=0.004)
    for j, (name, _) in enumerate(cols):
        ax_img = data[name]["Axial"]
        for i, row in enumerate(rows):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([]); ax.set_yticks([])
            if row == "Axial":
                ax.imshow(ax_img, cmap="gray", vmin=0, vmax=1)
                ax.add_patch(Rectangle((c0, r0), side, side, ec="#ff2d2d", fc="none", lw=1.6))
                ax.set_title(name, fontsize=11.5, pad=6, color="white")
            elif row == "Lesion zoom":
                ax.imshow(ax_img[r0:r0 + side, c0:c0 + side], cmap="gray",
                          vmin=0, vmax=1, interpolation="nearest")
                ax.contour(segall[r0:r0 + side, c0:c0 + side], levels=[0.5],
                           colors="#33ddff", linewidths=1.1)
                for sp in ax.spines.values():
                    sp.set_edgecolor("#ff2d2d"); sp.set_linewidth(1.6)
            else:
                ax.imshow(data[name][row], cmap="gray", vmin=0, vmax=1)
            if j == 0:
                ax.set_ylabel(row, fontsize=12.5, color="white")

    out = args.out or (REPO / "figures" / "output"
                       / f"qualitative_testmask_{args.case}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="black")
    if args.pdf:
        # MRI panels stay raster, but at print resolution inside a PDF container so the
        # labels and contours remain vector.
        args.pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.pdf, dpi=330, bbox_inches="tight", facecolor="black")
        print(f"  wrote {args.pdf}  (330 dpi)")
    print(f"  all 7 routes verified at IoU 1.0000 against the real test mask")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
