# Image-to-Image Translation on MoNuSeg with Pix2Pix

Master in Artificial Intelligence — *Computer Vision II* course, 2025/2026.  
Universidade de Santiago de Compostela.

**Authors:** Hugo García Souto, Adrián Martínez Balea.

This repository contains the full training code, ablation study, and report for a paired image-to-image translation system on the **MoNuSeg** histology dataset. The model takes a nuclei label map as input and generates a realistic H&E-stained histology image.

A separate, **inference-only public demo** is available at <https://github.com/hugogs7/pix2pix-monuseg-demo>. The trained generator checkpoints are hosted on Hugging Face at <https://huggingface.co/hugogs-7/pix2pix-monuseg>.

---

## Project setup

The project dependencies are managed with Conda through the `environment.yml` file provided in the repository root.

From the repository root, create the environment with:

```bash
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate cv2
```

If the environment already exists and you only want to update it from `environment.yml`, run:

```bash
conda env update -f environment.yml --prune
```

To make the environment available as a Jupyter kernel, run:

```bash
python -m ipykernel install --user --name cv2 --display-name "Python (cv2)"
```

Finally, launch the notebook with:

```bash
jupyter notebook project/Project_Hugo_Adrian.ipynb
```

---

## Project structure

```text
project/
├── ablation_results.csv          ← test-set metrics for all variants
├── checkpoints/                  ← saved best/last checkpoints for each experiment
│   ├── baseline_pix2pix_best.pt
│   ├── baseline_pix2pix_last.pt
│   ├── improved_all_best.pt
│   ├── improved_all_last.pt
│   ├── improved_aug_best.pt
│   ├── improved_aug_last.pt
│   ├── improved_lsgan_best.pt
│   ├── improved_lsgan_last.pt
│   ├── improved_multi_best.pt
│   └── improved_multi_last.pt
├── Project_Hugo_Adrian.ipynb     ← main notebook (run from here)
├── qualitative_comparison.png    ← side-by-side figure used in the analysis
└── src/
    ├── datasets.py               ← paired MoNuSeg datasets (plain and augmented)
    ├── evaluation.py             ← test-set evaluation and qualitative figures
    ├── experiments.py            ← run_pix2pix_experiment helper
    ├── __init__.py               ← package marker for local imports
    ├── models.py                 ← U-Net generator and (multi-scale) PatchGAN
    ├── training.py               ← train/validate loops, metrics, checkpointing
    └── utils.py                  ← seeding, paths, plotting, initialization
```

The MoNuSeg PNG dataset is expected at `../MoNuSeg/` relative to the `project/` folder. In other words, from the repository root, the expected layout is:

```text
CV-II-PROJECT/
├── environment.yml
├── MoNuSeg/
└── project/
```

---

## What the project does

The notebook implements a complete ablation study around Pix2Pix on MoNuSeg:

| # | Model | Description |
| --- | --- | --- |
| 1 | Baseline | Pix2Pix with U-Net generator (54.4M params) and PatchGAN discriminator (2.77M params), BCE adversarial loss, L1 reconstruction with λ=100. |
| 2 | + Augmentation | Baseline plus synchronized geometric augmentations and mild photometric jitter. |
| 3 | + LSGAN | Baseline with the binary cross-entropy adversarial loss replaced by Least Squares GAN. |
| 4 | + Multi-scale D | Baseline with a two-scale PatchGAN discriminator. |
| 5 | Improved (all) | Combined model with augmentation + LSGAN + multi-scale discriminator. |

All five models share the same data split (350 / 75 / 75), same hyperparameters (Adam, lr=2e-4, betas=(0.5, 0.999), batch size 8, 30 epochs), and the same random seed for a fair comparison.

### Test-set results

| Model | L1 ↓ | PSNR (dB) ↑ | SSIM ↑ |
| --- | :---: | :---: | :---: |
| Baseline | 0.3023 | 14.84 | 0.4077 |
| + Augmentation | 0.2959 | 14.85 | 0.3910 |
| + LSGAN | 0.2818 | 15.42 | 0.4275 |
| + Multi-scale D | 0.3095 | 14.61 | 0.3647 |
| Improved (all) | **0.2761** | **15.49** | **0.4280** |

See the project report for the full discussion.

---

## Reproducing the results

1. Create and activate the Conda environment using `environment.yml`.
2. Place the MoNuSeg dataset at `MoNuSeg/` in the repository root, so that from the notebook folder it is available as `../MoNuSeg/`.
3. Open `project/Project_Hugo_Adrian.ipynb` with the `Python (cv2)` kernel.
4. Run all cells in order.

The first run trains all five models sequentially. Each training run takes approximately 15–25 minutes on a single GPU (~1.5 hours total). Subsequent runs **detect existing checkpoints in `project/checkpoints/` and skip retraining**, so the notebook can be re-executed in seconds for figures, tables, and analysis.

To force retraining of a specific model, delete its corresponding `*_best.pt` and `*_last.pt` files from `project/checkpoints/`.

The final cell automatically copies four test-set images to the demo repository (`../demo/samples/`) so the public demo always works on images that were never seen during training.

---

## Reproducibility

- **Same seed (42)** is set for Python's `random`, NumPy, and PyTorch.
- **Same data split** is reused across all experiments.
- **`torch.backends.cudnn.deterministic = True`** is enabled, at the cost of slightly slower training.
- All hyperparameters and design choices are documented inline in the notebook and in the report.

---

## Demo and trained models

For inference (running a trained model on new images), please use the dedicated demo repository:

- **Demo (GitHub):** <https://github.com/hugogs7/pix2pix-monuseg-demo>
- **Checkpoints (Hugging Face):** <https://huggingface.co/hugogs-7/pix2pix-monuseg>

The demo does **not** require this training repository: it is a standalone, inference-only package with its own command-line script and Jupyter notebook.

---

## References

- P. Isola et al., "Image-to-image translation with conditional adversarial networks," CVPR 2017.
- N. Kumar et al., "A dataset and a technique for generalized nuclear segmentation for computational pathology," IEEE TMI 36:1550–1560, 2017.
- X. Mao et al., "Least squares generative adversarial networks," ICCV 2017.
