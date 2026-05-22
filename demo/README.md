# Pix2Pix MoNuSeg — Image-to-Image Translation Demo

Inference demo for the Pix2Pix image-to-image translation models trained on the **MoNuSeg** dataset of H&E-stained histology images.

Given a **label map** of nuclei segmentations, the model generates a **realistic histology image**. This repository contains the demo code (inference only — no training) and pointers to the trained checkpoints hosted on Hugging Face Hub.

This work was developed for the *Computer Vision II* course (Master in Artificial Intelligence, 2025/2026).

## Example output

Each demo run produces a side-by-side figure with the input label map, the generated image, and the ground-truth real image:

| Input label map | Generated image | Real (ground truth) |
| :---: | :---: | :---: |
| *label* | *generated* | *real* |

## Available models

Five trained models are available, corresponding to the ablation study reported in the project:

| Model name (`--model`) | Description | Test L1 ↓ | Test PSNR ↑ | Test SSIM ↑ |
| --- | --- | :---: | :---: | :---: |
| `baseline` | Plain Pix2Pix (BCE loss, single-scale D) | 0.3024 | 14.84 | 0.4074 |
| `aug` | Baseline + data augmentation | 0.2969 | 14.73 | 0.3307 |
| `lsgan` | Baseline + LSGAN loss | 0.2813 | 15.39 | 0.4309 |
| `multi` | Baseline + multi-scale discriminator | 0.3009 | 14.72 | 0.3877 |
| `improved` *(default)* | All three improvements combined | **0.2759** | **15.48** | 0.4278 |

L1 is in normalized [-1, 1] space (lower is better). PSNR (higher is better) and SSIM (higher is better) are computed in the [0, 1] image space.

## Installation

Tested with Python 3.10+.

```bash
git clone https://github.com/YOUR_USERNAME/pix2pix-monuseg-demo.git
cd pix2pix-monuseg-demo
pip install -r requirements.txt
```

That's it. The trained checkpoints (~200 MB each) are **not** stored in this repository; they are downloaded automatically from Hugging Face Hub the first time you request a given model.

## Quick start (CLI)

Run inference on a single sample image with the best improved model:

```bash
python demo/demo.py --input samples/sample_01.png --output outputs/result.png
```

Pick a different model:

```bash
python demo/demo.py --input samples/sample_01.png --model baseline --output outputs/result_baseline.png
```

Process all images in a directory at once:

```bash
python demo/demo.py --input samples/ --model improved --output outputs/
```

### All CLI options

```
--input PATH               Single paired PNG, or a directory of paired PNGs (required).
--model NAME               One of: baseline, aug, lsgan, multi, improved (default: improved).
--output PATH              Output file or directory (default: outputs/).
--checkpoints-dir PATH     Optional local directory with .pt files (skips download).
--image-size INT           Square size to resize inputs to (default: 256).
--device {auto,cpu,cuda}   Compute device (default: auto).
```

## Quick start (Jupyter notebook)

If you prefer a visual walkthrough:

```bash
jupyter notebook demo/demo_notebook.ipynb
```

The notebook reuses the same inference functions as the CLI script (`demo/inference.py`), so results are identical. It also includes an optional cell that runs **all five trained models** on the same input image and plots them side by side.

## Input format

Each input image must be a **paired** PNG in the MoNuSeg style:
- **Left half**: real H&E histology image.
- **Right half**: corresponding label map (nuclei segmentation).

The CLI automatically splits the paired image, feeds the right half to the generator, and compares the prediction against the left half.

The `samples/` directory contains a few examples from the **test set** of MoNuSeg (images never seen during training or validation).

## Project structure

```
pix2pix-monuseg-demo/
├── README.md                   ← you are here
├── requirements.txt
├── models.py                   ← U-Net generator definition (needed to load checkpoints)
├── demo/
│   ├── inference.py            ← reusable inference library
│   ├── demo.py                 ← command-line script
│   └── demo_notebook.ipynb     ← Jupyter notebook demo
├── samples/                    ← test-set images
└── outputs/                    ← demo results go here
```

## Where are the checkpoints hosted?

The five trained generators are published on Hugging Face Hub:

**[`REPLACE_WITH_YOUR_HF_REPO_ID`](https://huggingface.co/REPLACE_WITH_YOUR_HF_REPO_ID)**

You do **not** need to manually download them. The script and notebook do it for you the first time each model is used, and cache the files locally under `~/.cache/huggingface/hub/`.

If you prefer to use local files instead, place them in a folder and pass `--checkpoints-dir /path/to/folder` to the CLI.

## Authors

Hugo *(surname)* and Adrián Martínez Balea — Master in Artificial Intelligence (Universidade da Coruña), 2025/2026.

## License

Code: MIT.
Model checkpoints: same as the MoNuSeg dataset license (research use).

## Citation

If you use this code, please cite the original Pix2Pix paper:

> Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros.
> *Image-to-image translation with conditional adversarial networks*.
> CVPR, 2017.

And the MoNuSeg dataset:

> Neeraj Kumar et al. *A dataset and a technique for generalized nuclear segmentation for computational pathology*.
> IEEE Transactions on Medical Imaging 36:1550–1560, 2017.