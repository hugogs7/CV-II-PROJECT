from __future__ import annotations

# =========================
# Evaluation helpers
# =========================

import csv
from pathlib import Path

import torch

from src.models import UNetGenerator
from src.training import load_generator_weights, evaluate_on_test_set


def evaluate_checkpoint_on_test(
    checkpoint_path: Path,
    test_loader,
    device: torch.device,
    base_channels: int = 64,
) -> tuple[dict, torch.nn.Module]:
    """
    Build a generator, load checkpoint weights, and evaluate it on the test set.

    Only the generator is needed for inference and quantitative evaluation.
    Therefore, the discriminator architecture used during training does not
    matter when loading a checkpoint.
    """
    generator = UNetGenerator(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
    ).to(device)

    generator = load_generator_weights(
        generator=generator,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    test_metrics = evaluate_on_test_set(
        generator=generator,
        test_loader=test_loader,
        device=device,
    )

    return test_metrics, generator


def format_ablation_results_table(
    rows: list[dict],
    columns: list[str] | None = None,
) -> str:
    """
    Format ablation results as a plain-text table for notebook output.
    """
    if columns is None:
        columns = ["Model", "Test L1", "Test PSNR (dB)", "Test SSIM"]

    col_widths = {
        col: max(len(col), max(len(str(row[col])) for row in rows))
        for col in columns
    }

    def format_row(values):
        return "  ".join(
            str(value).ljust(col_widths[col])
            for col, value in zip(columns, values)
        )

    lines = []
    lines.append("Test-set ablation results")
    lines.append("")
    lines.append(format_row(columns))
    lines.append("  ".join("-" * col_widths[col] for col in columns))

    for row in rows:
        lines.append(format_row([row[col] for col in columns]))

    return "\n".join(lines)


def save_ablation_results_csv(
    rows: list[dict],
    output_path: Path,
    columns: list[str] | None = None,
) -> None:
    """
    Save ablation results to a CSV file.
    """
    if columns is None:
        columns = ["Model", "Test L1", "Test PSNR (dB)", "Test SSIM"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        
def compare_models_qualitatively(
    generators_by_name: dict,
    test_loader,
    device: torch.device,
    output_path: Path,
    num_samples: int = 3,
) -> None:
    """
    Plot a qualitative side-by-side comparison on test examples.

    For each selected test example, the figure shows:
    - the input label map,
    - the generated output of each model,
    - the real ground-truth image.

    The resulting figure is saved to `output_path`.
    """
    import matplotlib.pyplot as plt

    from src.utils import denormalize_image

    batch = next(iter(test_loader))

    labels = batch["label"][:num_samples].to(device)
    reals = batch["real"][:num_samples].to(device)

    model_names = list(generators_by_name.keys())
    num_models = len(model_names)
    num_cols = 2 + num_models  # label + each model + real

    fig, axes = plt.subplots(
        num_samples,
        num_cols,
        figsize=(3 * num_cols, 3 * num_samples),
    )

    if num_samples == 1:
        axes = axes[None, :]

    # Pre-compute generations for each model.
    with torch.no_grad():
        outputs_per_model = {
            name: generators_by_name[name](labels)
            for name in model_names
        }

    label_np = denormalize_image(labels).cpu()
    real_np = denormalize_image(reals).cpu()

    outputs_np = {
        name: denormalize_image(output).cpu()
        for name, output in outputs_per_model.items()
    }

    for row in range(num_samples):
        # Column 0: input label map.
        axes[row, 0].imshow(label_np[row].permute(1, 2, 0))
        axes[row, 0].set_title("Label map" if row == 0 else "")
        axes[row, 0].axis("off")

        # Columns 1..N: each model output.
        for col, name in enumerate(model_names, start=1):
            axes[row, col].imshow(outputs_np[name][row].permute(1, 2, 0))
            axes[row, col].set_title(name if row == 0 else "")
            axes[row, col].axis("off")

        # Last column: ground truth.
        axes[row, -1].imshow(real_np[row].permute(1, 2, 0))
        axes[row, -1].set_title("Real (ground truth)" if row == 0 else "")
        axes[row, -1].axis("off")

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Figure saved to: {output_path}")