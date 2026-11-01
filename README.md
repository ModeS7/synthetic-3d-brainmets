# Synthetic 3D brain MRI for brain metastasis segmentation

Code for the paper "Synthetic 3D Brain MRI for Brain Metastasis Segmentation: Utility as Training Data and Value of Generation Metrics".

## Contents

| Path | What it holds | Paper |
|---|---|---|
| `src/medgen/scripts/train.py`, `configs/` | Training of the mask and BRAVO generators | §3.2, App. A, B |
| `src/medgen/scripts/generate.py` | Sampling, single-checkpoint and handoff routing, containment check | §3.2.2, §3.2.4, App. C.1 |
| `src/medgen/scripts/compute_brain_atlas.py`, `compute_brain_pca.py` | Brain atlas and PCA shape model, built from the training data | §3.2.4 |
| `src/medgen/scripts/eval_all_metrics.py` | FID, KID, BiomedCLIP-MMD and diversity | §3.4.1 |
| `src/medgen/downstream/nnunet/`, `scripts/train_nnunet.py`, `scripts/eval_nnunet.py` | Dataset conversion, trainer, inference | §3.3 |
| `src/medgen/scripts/recompute_nnunet_metrics.py` | Segmentation metrics and paired statistics | §3.4.2, §3.4.3 |
| `src/medgen/scripts/select_exp16_2_validation_source.py` | Selection of the source on validation Dice | §3.3.2 |
| `figures/` | Scripts that draw the paper's figures | Figures 1 to 4, App. C, D |

## Route names

| Paper | Run name in code |
|---|---|
| Mask generator | `exp14_1_pixel_seg` |
| Original MSE | `exp1_1_1000` |
| Extended MSE | `exp1_1_1000plus` |
| Perceptual Continuation | `exp32_2_1000` |
| Pseudo-Huber Continuation | `exp32_3_1000` |
| Transitions (a) to (e) | `exp47a` to `exp47e` |
| Handoffs (a) to (e) | `exp48a` to `exp48e` specialists, sampled as `exp1_to_exp48{a..e}_t025` |

## Data

The images come from BrainMetShare, available from the [Stanford AIMI repository](https://aimi.stanford.edu/datasets/brainmetshare) under its research use agreement. No images, trained models or generated volumes are included here. The one derived file is the brain atlas in `data/`, a binary union of the brain masks of the 105 training volumes. The public release has no masks for its 51 test cases, so those are not included either.

## Environment

The paper used two environments, each pinned in its own file. Training and sampling need a GPU.

```
# Generators and generation metrics: PyTorch 2.9.0, torchvision 0.24.0, MONAI 1.5.1, open_clip_torch 3.2.0
pip install -r requirements.txt
pip install -e .

# Segmentation, in a separate environment: nnU-Net v2 2.6.2, PyTorch 2.10.0, torchvision 0.25.0, MONAI 1.5.1
pip install -r requirements-segmentation.txt
pip install -e .
```

## Setup

Set the data locations in `configs/paths/local.yaml`. Each case is expected as `<split>/<case>/bravo.nii.gz` and `seg.nii.gz`. Generator training needs only `train/` and skips `val/` and `test_new/` if they are absent. `convert_dataset.py` needs all three, because it reads the test cases from `val/` and `test_new/`, so divide the test cases between those two folders. Test cases without `seg.nii.gz` cannot be scored.

The brain atlas used by the mask filter is in `data/`. The PCA model used by the containment check on generated BRAVO volumes is not bundled. Build it from the training data and pass its path to `generate.py`:

```
python -m medgen.scripts.compute_brain_pca --data-root <data_dir> --output data/brain_support_pca.npz --n-components 88 --fit-splits train --heldout-splits <another_split>
python -m medgen.scripts.generate ... brain_support_pca_path=data/brain_support_pca.npz brain_atlas_path=null conditioning_brain_qc_mode=reject
```

`recompute_nnunet_metrics.py` checks how each prediction pool was produced, so save the output of `eval_nnunet.py` for each condition and pass it with `--condition`, with standard output in a `.out` file and standard error in a `.err` file beside it.

Figure scripts need generated and real volumes. They take their locations from command line arguments, or from the environment variables and constants at the top of each script.

## Example runs

These commands show how the pieces fit together. The values are examples, not the settings of any route in the paper. Each run writes its checkpoints under `paths.model_dir`.

### Training

```
COMMON="--config-name=diffusion_3d paths=local strategy=rflow model=default_3d volume.height=256 volume.width=256 volume.pad_depth_to=160"

# Mask generator
python -m medgen.scripts.train $COMMON mode=seg \
    training.epochs=500 training.learning_rate=1e-4 training.warmup_epochs=10 training.name=mask_

# BRAVO generator, conditioned on the mask
python -m medgen.scripts.train $COMMON mode=bravo \
    training.epochs=500 training.learning_rate=1e-4 training.warmup_epochs=10 training.name=bravo_

# Continue a BRAVO run with a perceptual term on the low-noise steps
python -m medgen.scripts.train $COMMON mode=bravo \
    training.resume_from=<bravo_run>/checkpoint_latest.pt +training.reset_scheduler=true \
    training.epochs=200 training.learning_rate=1e-5 training.warmup_epochs=10 \
    training.perceptual_weight=0.1 training.perceptual_max_timestep=250 training.name=continued_

# Specialist that trains only on the low-noise steps, for use in a handoff
python -m medgen.scripts.train $COMMON mode=bravo \
    training.resume_from=<bravo_run>/checkpoint_latest.pt +training.reset_scheduler=true \
    training.epochs=200 training.learning_rate=1e-5 training.warmup_epochs=10 \
    strategy.loss_type=weighted_huber training.curriculum.enabled=true training.curriculum.warmup_epochs=1 \
    training.curriculum.min_t_start=0.0 training.curriculum.min_t_end=0.0 \
    training.curriculum.max_t_start=0.25 training.curriculum.max_t_end=0.25 training.name=specialist_
```

### Inference

```
COMMON="paths=local spatial_dims=3 gen_mode=bravo image_size=256 depth=160 trim_slices=10 validate_size_bins=false"

# Masks and images from one BRAVO checkpoint
python -m medgen.scripts.generate $COMMON \
    seg_model=<mask_run>/checkpoint_latest.pt image_model=<bravo_run>/checkpoint_latest.pt \
    num_images=100 num_steps_seg=50 num_steps_bravo=50 seed=42 output_subdir=single

# Handoff: the general model samples until handoff_t, then the specialist takes over
python -m medgen.scripts.generate $COMMON \
    seg_model=<mask_run>/checkpoint_latest.pt \
    image_model=<specialist_run>/checkpoint_latest.pt image_model_high_t=<bravo_run>/checkpoint_latest.pt handoff_t=0.25 \
    num_images=100 num_steps_seg=50 num_steps_bravo=50 seed=42 output_subdir=handoff

# Images for a fixed set of masks, with the containment check from Setup
python -m medgen.scripts.generate $COMMON \
    image_model=<bravo_run>/checkpoint_latest.pt real_seg_dir=<mask_folder> num_images=100 num_steps_bravo=50 seed=42 \
    validate_brain_mask=true conditioning_brain_qc_mode=reject brain_atlas_path=null \
    brain_support_pca_path=data/brain_support_pca.npz output_subdir=fixed_masks
```

### Generation metrics

```
python -m medgen.scripts.eval_all_metrics \
    --reference train:<data_dir>/train --dataset handoff:<generated_dir>/handoff \
    --pattern bravo.nii.gz --pca-model none --output metrics.json
```

### Segmentation

Run these in the segmentation environment. `<base>` holds the nnU-Net folders `nnUNet_raw` and `nnUNet_preprocessed`.

```
export nnUNet_raw=<base>/nnUNet_raw nnUNet_preprocessed=<base>/nnUNet_preprocessed nnUNet_results=<results>

# Real training data. Add --synthetic-dir to train on generated volumes instead
python -m medgen.downstream.nnunet.convert_dataset --real-dir <data_dir> --nnunet-raw $nnUNet_raw --dataset-id 501 --modality bravo
nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEncL -c 3d_fullres --verify_dataset_integrity

# Train one fold. Repeat for folds 0 to 4
python -m medgen.scripts.train_nnunet --experiment baseline --dataset-id 501 --fold 0 \
    --nnunet-base <base> --nnunet-results <results> \
    --trainer nnUNetTrainerBrainMets --plans nnUNetResEncUNetLPlans

# Five-fold inference and evaluation, keeping the logs for recompute_nnunet_metrics.py
python -m medgen.scripts.eval_nnunet --experiment baseline --dataset-id 501 \
    --nnunet-base <base> --nnunet-results <results> \
    --trainer nnUNetTrainerBrainMets --plans nnUNetResEncUNetLPlans > baseline.out 2> baseline.err
```

## Licence

MIT. See `LICENSE`.
