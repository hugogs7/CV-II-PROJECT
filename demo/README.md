# Pix2Pix MoNuSeg — Image-to-Image Translation Demo

Inference demo for Pix2Pix-based image-to-image translation models trained on the **MoNuSeg** dataset of H&E-stained histology images.

Given a paired MoNuSeg image, the demo uses the **label map** as input and generates a **realistic histology image**. Each output is saved as a side-by-side figure showing:

1. the input label map,
2. the generated histology image,
3. the real ground-truth histology image.

This work was developed for the **Computer Vision II** course, Master in Artificial Intelligence, Universidade de Santiago de Compostela, 2025/2026.

---

## Available models

Five trained models are available. They correspond to the ablation study reported in the project:

| Model name (`--model`) | Description | Test L1 ↓ | Test PSNR ↑ | Test SSIM ↑ |
| --- | --- | :---: | :---: | :---: |
| `baseline` | Plain Pix2Pix with BCE loss and single-scale PatchGAN discriminator | 0.3023 | 14.84 | 0.4077 |
| `aug` | Baseline + data augmentation | 0.2959 | 14.85 | 0.3910 |
| `lsgan` | Baseline + Least Squares GAN loss | 0.2818 | 15.42 | 0.4275 |
| `multi` | Baseline + multi-scale discriminator | 0.3095 | 14.61 | 0.3647 |
| `improved` *(default)* | Data augmentation + LSGAN + multi-scale discriminator | **0.2761** | **15.49** | **0.4280** |

L1 is computed in the normalized `[-1, 1]` image space. PSNR and SSIM are computed in the `[0, 1]` image space.

---

## Checkpoints

The trained generator checkpoints are hosted on Hugging Face Hub:

<https://huggingface.co/hugogs-7/pix2pix-monuseg>

The checkpoints are **not stored in this GitHub repository**. They are downloaded automatically the first time a model is used and cached locally by Hugging Face under the user cache directory.

No Hugging Face token is required because the checkpoint repository is public.

---

## Installation

Tested with Python 3.10+.

```bash
git clone https://github.com/hugogs7/pix2pix-monuseg-demo.git
cd pix2pix-monuseg-demo
pip install -r requirements.txt
```

---

## Quick start: command-line demo

Run inference on all sample images using the best model:

```bash
python demo/demo.py --input demo/samples --model improved --output demo/outputs
```

Run inference on a single image:

```bash
python demo/demo.py --input demo/samples/image_099.png --model improved --output demo/outputs/image_099_demo.png
```

Use another model:

```bash
python demo/demo.py --input demo/samples --model baseline --output demo/outputs_baseline
```

The output figures contain:

```text
Input label map | Generated image | Real ground truth
```

---

## CLI options

| Option | Description |
| --- | --- |
| `--input PATH` | Single paired PNG image, or a directory of paired PNG images. |
| `--model NAME` | One of: `baseline`, `aug`, `lsgan`, `multi`, `improved`. |
| `--output PATH` | Output file or directory. |
| `--checkpoints-dir PATH` | Optional local directory with `.pt` checkpoints. If omitted, checkpoints are downloaded from Hugging Face. |
| `--image-size INT` | Square size used for inference. Default: `256`. |
| `--device {auto,cpu,cuda}` | Compute device. Default: `auto`. |

---

## Quick start: Jupyter notebook

A visual notebook demo is also included:

```bash
jupyter notebook demo/demo_notebook.ipynb
```

The notebook reuses the same inference functions as the command-line script, so both demos follow the same preprocessing, checkpoint loading, inference, and visualization pipeline.

The notebook also includes an optional comparison cell that runs all five trained models on the same input image.

---

## Input format

Each input image must be a paired MoNuSeg-style PNG:

- **Left half:** real H&E histology image.
- **Right half:** corresponding nuclei label map.

The demo automatically splits the image, feeds the right half to the generator, and compares the generated output with the left half.

The images in `demo/samples/` are examples from the MoNuSeg test split.

---

## Repository structure

```text
pix2pix-monuseg-demo/
├── README.md
├── requirements.txt
├── .gitignore
├── models.py
└── demo/
    ├── demo.py
    ├── inference.py
    ├── demo_notebook.ipynb
    ├── samples/
    │   ├── image_099.png
    │   ├── image_186.png
    │   ├── image_296.png
    │   └── image_317.png
    └── outputs/
        └── .gitkeep
```

---

## Authors

Hugo García Souto and Adrián Martínez Balea  
Master in Artificial Intelligence  
Universidade de Santiago de Compostela  
2025/2026

---

## References

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros.  
*Image-to-image translation with conditional adversarial networks.*  
CVPR, 2017.

Neeraj Kumar et al.  
*A dataset and a technique for generalized nuclear segmentation for computational pathology.*  
IEEE Transactions on Medical Imaging, 36:1550–1560, 2017.
