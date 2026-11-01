#!/usr/bin/env python3
"""Qualitative segmentation panel for the downstream results.

The generation findings already appear in two orthogonal planes (Figure 3). The
SEGMENTATION findings were reported only as tables, so a reviewer answering the MIA
questionnaire had no figure to point at for the paper's primary result. This is that
figure: predicted masks against ground truth, axial and coronal, on the real test volumes.

Three conditions, chosen because they are the ones the argument turns on:
    real reference        what synthetic data has to beat
    handoff (c)           the validation-selected synthetic source, best of seven
    hybrid +25            real plus synthetic, the level with the most favourable interval

Five cases, chosen by the same rule the generation gallery uses: the 51 official test
patients sorted by real-reference volumetric Dice (ties by case id), then the ranks of
np.round(np.linspace(0, 50, 5)). That spans the cohort instead of flattering it, and it keeps
rank 0, a patient the reference scores zero on. A panel showing only the cases a model handles
is a highlight reel, not a finding.

DATA. Ground truth, predictions and per-case Dice are read from the directories named by the
environment variables below. The script asserts same shape, same orientation, and per-case Dice.

    usage: make_downstream_qualitative.py [--pdf]
"""

from __future__ import annotations

import argparse
import os
import csv
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

REPO = Path(__file__).resolve().parents[1]
PRED = Path(os.environ.get("NNUNET_RESULTS", "nnunet_results"))
GT = Path(os.environ.get("TEST_LABELS_DIR", "test_labels"))
REAL = Path("/path/to/MedicalDataSets/brainmetshare-3/test")
EXPERIMENT = {
    "real_reference": "exp3_baseline_v2_d600",
    "original_mse": "exp16_2_synthetic_105_common105_exp1_1_1000_d650",
    "extended_mse": "exp16_2_synthetic_105_common105_exp1_1_1000plus_d651",
    "perceptual_continuation": "exp16_2_synthetic_105_common105_exp32_2_1000_d652",
    "strong_perceptual_continuation": "exp16_2_synthetic_105_common105_exp47a_d654",
    "weighted_huber_transition": "exp16_2_synthetic_105_common105_exp47c_d656",
    "weighted_huber_handoff": "exp16_2_synthetic_105_common105_exp1_to_exp48c_t025_d661",
    "pseudo_huber_perceptual_handoff": "exp16_2_synthetic_105_common105_exp1_to_exp48d_t025_d662",
    "hybrid_25": "exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663",
    "hybrid_50": "exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663",
    "hybrid_105": "exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663",
    "hybrid_210": "exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663",
}
ARCH = Path(os.environ.get("RESULTS_DIR", "results"))

CONDITIONS = [("real_reference", "Real reference"),
              ("weighted_huber_handoff", "Handoff (c)"),
              ("hybrid_25", "Hybrid $+$25")]
CASES = ["BrainMet_232", "BrainMet_203", "BrainMet_065", "BrainMet_097", "BrainMet_260"]

# --gallery: rows are training sources and columns are cases, matching the generation
# galleries so the appendix reads the same way. Cases are the ranks of
# np.round(np.linspace(0, 48, 6)) over the 49 patients some model detects, sorted by
# real-reference Dice (ties by case id). Drawn from the detected set because a patient no
# condition detects would fill its column with annotation-only panels.
GALLERY_CASES = ["BrainMet_232", "BrainMet_203", "BrainMet_315",
                 "BrainMet_166", "BrainMet_241", "BrainMet_260"]
# The replacement family is split on the same boundary the generation galleries use --
# baseline and direct continuations, then the routed objectives. Not a stylistic choice:
# panel width is textwidth/ncols and the panels are square, so eight rows over six cases
# is a 9.2 in figure and a page cannot hold it beside a caption. Four and five rows can,
# at full panel size, which is what tiny lesion contours need. The reference row appears
# in both so each figure stands on its own.
REPLACEMENT_A = [("real_reference", "Real reference"),
                 ("original_mse", "Original MSE"),
                 ("extended_mse", "Extended MSE"),
                 ("perceptual_continuation", "Perceptual Cont.")]
REPLACEMENT_B = [("real_reference", "Real reference"),
                 ("strong_perceptual_continuation", "Transition (a)"),
                 ("weighted_huber_transition", "Transition (c)"),
                 ("weighted_huber_handoff", "Handoff (c)"),
                 ("pseudo_huber_perceptual_handoff", "Handoff (d)")]
AUGMENTATION = [("real_reference", "Real reference"),
                ("hybrid_25", "Hybrid $+$25"), ("hybrid_50", "Hybrid $+$50"),
                ("hybrid_105", "Hybrid $+$105"), ("hybrid_210", "Hybrid $+$210")]
# all twelve, deduplicated with the reference kept once -- the galleries pick their cut
# over the whole set so that the three of them cut each patient at the same depth
ALL_CONDITIONS = REPLACEMENT_A + REPLACEMENT_B[1:] + AUGMENTATION[1:]

# Ground truth and prediction must stay distinguishable in grayscale and to a reader with
# impaired colour vision, so the pair is blue/orange rather than the usual red/green.
GT_COLOR, PRED_COLOR = "#4DA6FF", "#FF9E1B"
SPACING = (0.9375, 0.9375, 1.0)      # from the prediction headers, matches the paper
MARGIN = 6


def archive_dice() -> dict:
    d = {}
    for job in ("synthetic_only", "hybrid"):
        with (ARCH / job / "per_case.csv").open() as fh:
            for r in csv.DictReader(fh):
                d[(r["condition"], r["case_id"])] = float(r["volumetric_dice"])
    return d


def load_case(case: str, arch: dict, conds):
    """Real volume, ground truth and the three predictions, with alignment verified."""
    real = np.asarray(nib.load(str(REAL / case.replace("BrainMet_", "Mets_") / "bravo.nii.gz")).dataobj,
                      dtype=np.float32)
    gt_n = nib.load(str(GT / f"{case}.nii.gz"))
    gt = np.asarray(gt_n.dataobj) > 0.5
    preds = {}
    for key, _ in conds:
        e = EXPERIMENT[key]
        pn = nib.load(str(PRED / e / f"eval_{e}" / "predictions" / f"{case}.nii.gz"))
        p = np.asarray(pn.dataobj) > 0.5
        if p.shape != gt.shape:
            raise ValueError(f"{key}/{case}: shape {p.shape} vs ground truth {gt.shape}")
        if nib.aff2axcodes(pn.affine) != nib.aff2axcodes(gt_n.affine):
            raise ValueError(f"{key}/{case}: orientation differs from ground truth")
        dice = 2 * (p & gt).sum() / max(p.sum() + gt.sum(), 1)
        want = arch[(key, case)]
        if abs(dice - want) > 5e-4:
            raise ValueError(f"{key}/{case}: Dice {dice:.4f} does not reproduce archive {want:.4f}")
        preds[key] = p
    return real, gt, preds


def slices(gt: np.ndarray, preds: dict):
    """Axial and coronal indices for Figure 4, one per plane.

    Same rule as gallery_slice, applied independently to each plane: stay inside the largest
    reference lesion, and among those slices take the one where the most displayed conditions
    have a prediction to show. The centroid rule this replaced put the cut one slice away
    from the prediction on two of the five cases, and both times the empty column was the
    validation-selected source, on patients where it is within 0.007 Dice of the reference.
    Read as a block, the axial half said the synthetic source misses lesions the real
    reference finds. Moving the cut one slice fixes both. It also makes this figure agree
    with the appendix galleries, which share two of its cases and previously cut them
    elsewhere, so the same case under the same condition gave opposite impressions.
    """
    lab, n = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    if n == 0:
        return gt.shape[2] // 2, gt.shape[1] // 2
    big = lab == (1 + int(np.argmax([(lab == i).sum() for i in range(1, n + 1)])))

    def pick(axis):
        extent = big.sum(axis=tuple(a for a in (0, 1, 2) if a != axis))
        def visible(i):
            index = [slice(None)] * 3
            index[axis] = i
            return sum(p[tuple(index)].any() for p in preds.values())
        return max((i for i in range(gt.shape[axis]) if extent[i] > 0),
                   key=lambda i: (visible(i), extent[i]))

    return pick(2), pick(1)


def gallery_slice(gt: np.ndarray, preds: dict):
    """Axial index for the galleries: inside an annotated lesion, at the depth where the most
    conditions have a prediction to show, preferring the largest lesion where that ties.

    The obvious rule -- cut at the centroid of the largest reference lesion -- is what the
    other panels use, and on two of these six patients it produced a figure that argued the
    opposite of the result. Case 257 has non-zero Dice under all twelve conditions, and the
    centroid cut showed a prediction in two of them, because the lesion spans z 76 to 90
    and the predictions sit in its upper half. A reader would have concluded that ten
    conditions missed a lesion the other two found, on a patient where they are tied.
    Cutting at the largest lesion cross-section does not fix it. The prediction is simply
    somewhere else in a lesion that is long in z.

    Choosing the depth by pooled prediction presence does fix it, and it cannot flatter any
    single condition because every displayed condition votes. The candidate slices are
    still confined to the reference lesion, so the annotation is always in frame. On the six
    gallery cases the result is a figure a reader can trust at a glance: a prediction
    contour appears in a panel exactly when that condition scored above zero on that
    patient. build_gallery asserts that, so it cannot quietly stop being true.
    """
    lab, n = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    if n == 0:
        return gt.shape[2] // 2
    big = lab == (1 + int(np.argmax([(lab == i).sum() for i in range(1, n + 1)])))
    areas = big.sum(axis=(0, 1))
    all_areas = gt.sum(axis=(0, 1))
    # Every annotated slice is a candidate. On patients with many lesions, a condition can
    # score above zero on a lesion the largest one never meets, and no cut through the
    # largest lesion then shows it. Pooled prediction presence decides first, the largest
    # lesion breaks ties, so a patient with one dominant lesion is cut through it as before.
    candidates = [z for z in range(gt.shape[2]) if all_areas[z] > 0]
    return max(candidates,
               key=lambda z: (sum(p[:, :, z].any() for p in preds.values()), areas[z] > 0,
                              areas[z], all_areas[z]))


def crop_box(mask2d, shape, aspect=1.0):
    """Square-in-display crop around the brain.

    A tight bounding box gives every case a different image aspect, and with equal grid
    cells that leaves ragged gaps between rows. Squaring the box in DISPLAY space -- so
    the anisotropic coronal is squared after its aspect correction, not before -- makes
    every panel fill its cell identically.
    """
    ys, xs = np.where(mask2d)
    if not len(ys):
        y0, y1, x0, x1 = 0, shape[0] - 1, 0, shape[1] - 1
    else:
        y0, y1 = max(ys.min() - MARGIN, 0), min(ys.max() + MARGIN, shape[0] - 1)
        x0, x1 = max(xs.min() - MARGIN, 0), min(xs.max() + MARGIN, shape[1] - 1)
    h, w = (y1 - y0 + 1) * aspect, x1 - x0 + 1
    side = max(h, w)
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    half_y, half_x = side / aspect / 2, side / 2
    y0 = int(max(round(cy - half_y), 0)); y1 = int(min(round(cy + half_y), shape[0] - 1))
    x0 = int(max(round(cx - half_x), 0)); x1 = int(min(round(cx + half_x), shape[1] - 1))
    return y0, y1, x0, x1


def panel(ax, img, gt2d, pred2d, aspect):
    vmax = float(np.percentile(img[img > 0.02], 99.5)) if (img > 0.02).any() else 1.0
    ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="bilinear", aspect=aspect)
    for m, c, w in ((gt2d, GT_COLOR, 1.15), (pred2d, PRED_COLOR, 0.7)):
        if m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=c, linewidths=w)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#3A3A3A"); s.set_linewidth(0.5)


def build(arch: dict):
    ncol = 2 * len(CONDITIONS)
    fig = plt.figure(figsize=(7.1, 6.2), facecolor="black")
    gs = gridspec.GridSpec(len(CASES), ncol, wspace=0.03, hspace=0.03,
                           left=0.085, right=0.995, top=0.90, bottom=0.045)
    for row, case in enumerate(CASES):
        real, gt, preds = load_case(case, arch, CONDITIONS)
        za, yc = slices(gt, preds)
        # brain box per case, shared by every column so a difference is never a crop artefact
        ax_img = np.rot90(real[:, :, za]); ax_gt = np.rot90(gt[:, :, za])
        co_img = np.rot90(real[:, yc, :]); co_gt = np.rot90(gt[:, yc, :])
        ab = crop_box(ax_img > 0.05, ax_img.shape, 1.0)
        cb = crop_box(co_img > 0.05, co_img.shape, SPACING[2] / SPACING[0])
        for col, (key, _) in enumerate(CONDITIONS):
            p = preds[key]
            for j, (img, g, pr, box, asp) in enumerate((
                    (ax_img, ax_gt, np.rot90(p[:, :, za]), ab, 1.0),
                    (co_img, co_gt, np.rot90(p[:, yc, :]), cb, SPACING[2] / SPACING[0]))):
                y0, y1, x0, x1 = box
                ax = fig.add_subplot(gs[row, col + j * len(CONDITIONS)])
                panel(ax, img[y0:y1 + 1, x0:x1 + 1], g[y0:y1 + 1, x0:x1 + 1],
                      pr[y0:y1 + 1, x0:x1 + 1], asp)
                if row == 0:
                    ax.set_title(CONDITIONS[col][1], fontsize=6.4, color="white", pad=3)
                if col == 0 and j == 0:
                    ax.set_ylabel(f"{case.replace('BrainMet_', 'Case ')}\n"
                                  f"ref {arch[('real_reference', case)]:.3f}",
                                  fontsize=6.0, color="white", labelpad=2, linespacing=1.3)
    # plane headers over each block of three, and the contour key
    for x, txt in ((0.085 + 0.455 * 0.5, "Axial"), (0.54 + 0.455 * 0.5, "Coronal")):
        fig.text(x, 0.955, txt, ha="center", fontsize=8.0, color="white", weight="bold")
    fig.text(0.5, 0.012, "ground truth annotation", ha="right", fontsize=6.6, color=GT_COLOR)
    fig.text(0.505, 0.012, "prediction", ha="left", fontsize=6.6, color=PRED_COLOR)
    return fig


def build_gallery(arch: dict, conds, cases, title):
    """Rows are training sources, columns are cases, one axial slice each.

    Same shape as the generation galleries, so the two appendices read alike. A single
    plane here is deliberate: Figure~4 already carries two orthogonal planes for the
    same comparison, and what this figure is for is how the prediction changes when the
    training source changes, which one consistent cut shows more clearly than two.
    """
    fig = plt.figure(figsize=(7.1, 1.06 * len(conds) + 0.55), facecolor="black")
    gs = gridspec.GridSpec(len(conds), len(cases), wspace=0.03, hspace=0.03,
                           left=0.115, right=0.995, top=0.955, bottom=0.035)
    cache = {}
    for col, case in enumerate(cases):
        # every condition is loaded, not only the displayed rows, so that the three
        # galleries cut each patient at the same depth and stay comparable with each other
        real, gt, preds = load_case(case, arch, ALL_CONDITIONS)
        za = gallery_slice(gt, preds)
        # The property the reader will assume: no orange means this condition found
        # nothing on this patient. It is only true because of how za was chosen, so it is
        # checked rather than hoped for.
        for key, _ in ALL_CONDITIONS:
            visible = bool(preds[key][:, :, za].any())
            scored = arch[(key, case)] > 0
            if visible != scored:
                raise ValueError(
                    f"{case}/{key}: prediction {'visible' if visible else 'absent'} at "
                    f"z={za} but volumetric Dice is {arch[(key, case)]:.4f}. The panel "
                    f"would tell the reader the opposite of the table.")
        img = np.rot90(real[:, :, za]); g2 = np.rot90(gt[:, :, za])
        box = crop_box(img > 0.05, img.shape, 1.0)
        cache[case] = (img, g2, preds, za, box)
    for row, (key, label) in enumerate(conds):
        for col, case in enumerate(cases):
            img, g2, preds, za, box = cache[case]
            y0, y1, x0, x1 = box
            ax = fig.add_subplot(gs[row, col])
            panel(ax, img[y0:y1 + 1, x0:x1 + 1], g2[y0:y1 + 1, x0:x1 + 1],
                  np.rot90(preds[key][:, :, za])[y0:y1 + 1, x0:x1 + 1], 1.0)
            if col == 0:
                ax.set_ylabel(label, fontsize=6.2, color="white", labelpad=2)
            if row == 0:
                # three decimals, not two: the lowest-ranked case scores 0.0012 and two
                # decimals print it as "0.00", which in this paper names a defined set --
                # the patients no source detects at all, which are a different figure
                ax.set_title(f"{case.replace('BrainMet_', 'Case ')}\n"
                             f"ref {arch[('real_reference', case)]:.3f}",
                             fontsize=5.8, color="white", pad=2, linespacing=1.2)
    fig.text(0.5, 0.008, "ground truth annotation", ha="right", fontsize=6.4, color=GT_COLOR)
    fig.text(0.505, 0.008, "prediction", ha="left", fontsize=6.4, color=PRED_COLOR)
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", action="store_true", help="also write into the inner repo")
    ap.add_argument("--which", default="main",
                    choices=["main", "replacement", "augmentation", "all"],
                    help="main = Figure 4; the rest are the appendix galleries")
    args = ap.parse_args()
    arch = archive_dice()

    global CASES
    jobs = []
    if args.which in ("main", "all"):
        jobs.append(("downstream_qualitative", lambda: build(arch),
                     f"{len(CASES)} cases x {len(CONDITIONS)} conditions x 2 planes"))
    if args.which in ("replacement", "all"):
        jobs.append(("downstream_gallery_continuations",
                     lambda: build_gallery(arch, REPLACEMENT_A, GALLERY_CASES, "replacement"),
                     f"{len(REPLACEMENT_A)} sources x {len(GALLERY_CASES)} cases"))
        jobs.append(("downstream_gallery_routed",
                     lambda: build_gallery(arch, REPLACEMENT_B, GALLERY_CASES, "replacement"),
                     f"{len(REPLACEMENT_B)} sources x {len(GALLERY_CASES)} cases"))
    if args.which in ("augmentation", "all"):
        jobs.append(("downstream_gallery_augmentation",
                     lambda: build_gallery(arch, AUGMENTATION, GALLERY_CASES, "augmentation"),
                     f"{len(AUGMENTATION)} levels x {len(GALLERY_CASES)} cases"))

    for name, make, desc in jobs:
        fig = make()
        out = REPO / "figures" / "output" / f"{name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, facecolor="black", bbox_inches="tight")
        if args.pdf:
            fig.savefig(REPO / "figures" / "output" / f"{name}.pdf",
                        dpi=400, facecolor="black", bbox_inches="tight")
        plt.close(fig)
        print(f"  {name:34s} {desc}")
    print("  every alignment and per-case Dice verified against the per-case results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
