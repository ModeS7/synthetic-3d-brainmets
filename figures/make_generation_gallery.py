#!/usr/bin/env python3
"""Appendix gallery: every generation route on the same synthetic-mask candidates.

The qualitative figure in Results shows the seven routes that reached the downstream
panel. Table 2 reports fourteen, including the two weakest, and nothing in the paper
shows what a weak route looks like. This fills that gap.

Source is the strict common-96 cohort -- the 96 candidate identifiers accepted by all 14
routes -- so it is the same cohort the Table 2 metrics were computed on, and one
identifier gives 14 volumes conditioned on one identical synthetic mask.

The masks are SYNTHETIC, so there is no paired real volume and the gallery has no real
reference column. That is the honest framing: this compares generators with each other,
which is exactly what the common-96 metrics do.

Fourteen rows do not fit a page (~25 in at a readable cell size, against ~9 in of usable
height), so the gallery splits on the seam Table 1 already uses: the baseline and its
direct continuations, transitions, handoffs. Note the seam is lineage, not training
range -- transitions also train over the full trajectory, so "full" below is a group key
only. Each group is coherent on its own, and putting the five transitions and the five
handoffs in separate figures lets them be compared directly.

    usage: make_generation_gallery.py [--group full|transition|handoff|all] [--n 6]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# TrueType (Type 42) fonts in the PDF. The default, Type 3, embeds glyphs as drawing procedures,
# which production desks often reject. Text and layout are unchanged (pixel-checked 2026-10-04).
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path("/path/to/MedicalDataSets/evalModels"
            "/source_selection_synthmask525_seed42_euler100_pca2p323mm_try3"
            "/matched_common96_all14")
REPO = Path(__file__).resolve().parents[1]

# Order and naming follow Table 1 of the manuscript.
GROUPS = {
    "full": [("exp1_1_1000", "Original MSE"),
             ("exp1_1_1000plus", "Extended MSE"),
             ("exp32_2_1000", "Perceptual\nContinuation"),
             ("exp32_3_1000", "Pseudo-Huber\nContinuation")],
    "transition": [("exp47a", "Transition (a)"), ("exp47b", "Transition (b)"),
                   ("exp47c", "Transition (c)"), ("exp47d", "Transition (d)"),
                   ("exp47e", "Transition (e)")],
    "handoff": [("exp1_to_exp48a_t025", "Handoff (a)"),
                ("exp1_to_exp48b_t025", "Handoff (b)"),
                ("exp1_to_exp48c_t025", "Handoff (c)"),
                ("exp1_to_exp48d_t025", "Handoff (d)"),
                ("exp1_to_exp48e_t025", "Handoff (e)")],
}
ALL = [r for g in GROUPS.values() for r in g]
THR, MARGIN = 0.05, 4


def pick_candidates(n):
    """Deterministic spread over lesion burden, not a hand-picked set.

    Sorting the 96 by largest-lesion size and taking evenly spaced ranks gives the same
    selection on every run and covers the range instead of flattering it.
    """
    ids = sorted(p.name for p in (ROOT / ALL[0][0]).iterdir() if p.name.isdigit())
    sizes = []
    for i in ids:
        seg = nib.load(str(ROOT / ALL[0][0] / i / "seg.nii.gz")).get_fdata() > 0.5
        lab, k = ndimage.label(seg, structure=np.ones((3, 3, 3)))
        big = max((int((lab == j).sum()) for j in range(1, k + 1)), default=0)
        sizes.append((big, i))
    sizes.sort(reverse=True)
    ranks = np.linspace(0, len(sizes) - 1, n).round().astype(int)
    return [sizes[r][1] for r in ranks]


def lesion_slice(seg):
    """Axial index through the largest lesion, which is what each cell exhibits."""
    lab, k = ndimage.label(seg, structure=np.ones((3, 3, 3)))
    if k == 0:
        return seg.shape[2] // 2
    big = lab == (1 + int(np.argmax([(lab == j).sum() for j in range(1, k + 1)])))
    return int(np.round(np.nonzero(big)[2].mean()))


def brain_box(slices):
    """One crop box per candidate, shared by every route in that column.

    Cropping each route to its own bounding box would misalign the column and make
    routes look different in scale rather than in appearance.
    """
    y0 = x0 = 10 ** 9
    y1 = x1 = -1
    for im in slices:
        ys, xs = np.where(im > THR)
        if not len(ys):
            continue
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
    if y1 < 0:
        return 0, slices[0].shape[0] - 1, 0, slices[0].shape[1] - 1
    h, w = slices[0].shape
    return (max(y0 - MARGIN, 0), min(y1 + MARGIN, h - 1),
            max(x0 - MARGIN, 0), min(x1 + MARGIN, w - 1))


def build(group, cand, out_png, out_pdf=None):
    routes = GROUPS[group]
    cols = {}
    for c in cand:
        seg = nib.load(str(ROOT / routes[0][0] / c / "seg.nii.gz")).get_fdata() > 0.5
        z = lesion_slice(seg)
        ims, pool, segs = [], [], seg[:, :, z]
        for d, _ in ALL:                       # box over ALL 14 so groups stay aligned
            s2 = nib.load(str(ROOT / d / c / "seg.nii.gz")).get_fdata() > 0.5
            iou = (s2 & seg).sum() / max((s2 | seg).sum(), 1)
            assert iou > 0.999, f"{d}/{c} conditioning mask differs (IoU {iou:.4f})"
            v = nib.load(str(ROOT / d / c / "bravo.nii.gz")).get_fdata().astype(np.float32)
            ims.append(np.rot90(v[:, :, z]))
            pool.append(v[v > THR])
        box = brain_box(ims)
        y0, y1, x0, x1 = box
        keep = {lbl: ims[i][y0:y1 + 1, x0:x1 + 1]
                for i, (d, lbl) in enumerate(ALL) if (d, lbl) in routes}
        # One display window per candidate, from brain voxels pooled over all 14 routes.
        # Scaling each volume by its own max instead let a single outlier voxel make a
        # route look ~10% brighter than its neighbours, which reads as a real difference.
        vmax = float(np.percentile(np.concatenate(pool), 99.5))
        cols[c] = (keep, np.rot90(segs)[y0:y1 + 1, x0:x1 + 1], vmax)

    h, w = next(iter(cols.values()))[0][routes[0][1]].shape
    asp = h / w
    CW = 1.28
    # Black throughout: the panels are black-background MRI, so a white figure face
    # showed as bright seams between them. Row labels invert to white to stay legible.
    fig = plt.figure(figsize=(CW * len(cand) + 1.15, CW * asp * len(routes) + 0.18),
                     facecolor="black")
    gs = gridspec.GridSpec(len(routes), len(cand), wspace=0.02, hspace=0.02,
                           left=1.15 / (CW * len(cand) + 1.15), right=0.999,
                           top=0.999, bottom=0.001)
    for i, (_, lbl) in enumerate(routes):
        for j, c in enumerate(cand):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(cols[c][0][lbl], cmap="gray", vmin=0, vmax=cols[c][2])
            ax.contour(cols[c][1], levels=[0.5], colors="#33ddff", linewidths=0.55)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(lbl, fontsize=8.5, color="white")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, facecolor="black", bbox_inches="tight")
    if out_pdf:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, dpi=330, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    print(f"  {group}: {len(routes)} routes x {len(cand)} candidates -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all", choices=[*GROUPS, "all"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()

    cand = pick_candidates(args.n)
    print(f"  candidates (spread over lesion burden): {' '.join(cand)}")
    groups = list(GROUPS) if args.group == "all" else [args.group]
    for g in groups:
        build(g, cand,
              REPO / "figures" / "output" / f"gallery_{g}.png",
              (REPO / "figures" / "output" / f"gallery_{g}.pdf")
              if args.pdf else None)


if __name__ == "__main__":
    raise SystemExit(main())
