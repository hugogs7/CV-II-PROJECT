"""
Command-line demo for the Pix2Pix MoNuSeg image-to-image translation.

Given a paired MoNuSeg image (real on the left, label map on the right), runs
inference with one of the trained models and saves a side-by-side figure
showing the input label map, the generated image, and the ground-truth real
image.

Usage examples
--------------

Single image:

    python demo/demo.py \\
        --input samples/TCGA-XX-YYYY-ZZ.png \\
        --model improved \\
        --output outputs/result.png

Batch over a directory:

    python demo/demo.py \\
        --input samples/ \\
        --model improved \\
        --output outputs/

Available --model values: baseline, aug, lsgan, multi, improved (default).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the inference module importable when running this script directly.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from inference import (  # noqa: E402
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    load_generator,
    resolve_checkpoint_path,
    run_inference_on_paired_image,
    save_side_by_side,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pix2Pix MoNuSeg image-to-image translation demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to a single paired PNG image, or to a directory containing "
            "paired PNG images. Each image must have the real histology on the "
            "left half and the label map on the right half."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=list(AVAILABLE_MODELS),
        default=DEFAULT_MODEL,
        help="Which trained model to use for inference.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help=(
            "Where to save the side-by-side figure(s). If --input is a single "
            "file, this can be a file path; if --input is a directory, this "
            "must be a directory."
        ),
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help=(
            "Optional local directory containing the checkpoint .pt files. "
            "If a matching file is found, it is used directly; otherwise the "
            "checkpoint is downloaded from Hugging Face Hub."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Square size to which input images are resized.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Compute device. 'auto' picks GPU if available, otherwise CPU.",
    )

    return parser.parse_args()


def pick_device(choice: str) -> torch.device:
    if choice == "cuda":
        if not torch.cuda.is_available():
            print("Warning: --device cuda requested but no GPU is available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_input_images(input_path: Path) -> list[Path]:
    """
    Return the list of paired PNG images to process.
    """
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        png_paths = sorted(input_path.glob("*.png"))
        if not png_paths:
            raise FileNotFoundError(
                f"No PNG files found in directory: {input_path}"
            )
        return png_paths

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def resolve_output_path(
    output_arg: Path,
    input_image_path: Path,
    is_batch: bool,
) -> Path:
    """
    Decide where to save the result for a given input image.

    - If we are processing a directory, the output must also be a directory.
      Each result is saved as <output_dir>/<input_stem>_demo.png.
    - If we are processing a single file:
        - if --output ends with an image extension, use it as-is;
        - otherwise treat it as a directory and write
          <output_dir>/<input_stem>_demo.png inside it.
    """
    image_extensions = {".png", ".jpg", ".jpeg"}
    treat_as_file = (not is_batch) and (output_arg.suffix.lower() in image_extensions)

    if treat_as_file:
        return output_arg

    output_arg.mkdir(parents=True, exist_ok=True)
    return output_arg / f"{input_image_path.stem}_demo.png"


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Resolve and load the checkpoint
    print("Resolving checkpoint...")
    checkpoint_path = resolve_checkpoint_path(
        model_name=args.model,
        local_checkpoints_dir=args.checkpoints_dir,
    )
    print(f"Checkpoint: {checkpoint_path}")

    generator, device = load_generator(checkpoint_path, device=device)

    # Gather inputs
    input_images = collect_input_images(args.input)
    is_batch = len(input_images) > 1 or args.input.is_dir()

    print(f"Found {len(input_images)} image(s) to process.")

    for image_path in input_images:
        print(f"  Processing: {image_path.name}")
        result = run_inference_on_paired_image(
            paired_image_path=image_path,
            generator=generator,
            device=device,
            image_size=args.image_size,
        )

        output_path = resolve_output_path(
            output_arg=args.output,
            input_image_path=image_path,
            is_batch=is_batch,
        )
        save_side_by_side(
            inference_result=result,
            output_path=output_path,
            model_name=args.model,
        )
        print(f"    Saved: {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()