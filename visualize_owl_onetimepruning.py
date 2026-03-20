import os

import matplotlib.pyplot as plt
import numpy as np

from experiment_helpers import load_data, load_model_and_processor
from utils.WandA_profiler import WAprofiler
from utils.evaluator import Evaluator
from utils.pruning_tools import PruningTool
from utils.scorer import Scorer


def one_time_owl_pruning(model, processor, dataloader, scorer, weights, stats, level,
                         relative_difference, average_retention_ratio, device, dtype):
    _, retention_ratio = scorer.owl(
        weights,
        stats,
        level=level,
        relative_difference=relative_difference,
        average_retention_ratio=average_retention_ratio,
    )
    sparsity_dict = {name: 1.0 - score for name, score in retention_ratio.items()}

    pruner = PruningTool()
    pruner.wanda_unstructured_pruning(weights, stats, sparsity=sparsity_dict)
    pruner.apply_to_model(model, log=False)

    evaluator = Evaluator(model, processor, dataloader, device, language=DATASET_NAME, dtype=dtype)
    cer, wer, avg_loss = evaluator.evaluate(log=False)
    return cer, wer, avg_loss


def plot_fixed_level(target_level, levels, relative_differences, average_retention_ratios, results, save_dir):
    x_index = levels.index(target_level)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ["CER", "WER", "Avg Loss"]

    for ratio_index, ratio in enumerate(average_retention_ratios):
        data_slice = results[x_index, :, ratio_index, :]
        for metric_index in range(3):
            axes[metric_index].plot(
                relative_differences,
                data_slice[:, metric_index],
                marker="o",
                label=f"ARR = {ratio}",
            )

    for index, ax in enumerate(axes):
        ax.set_title(f"{metrics_names[index]} (Level = {target_level})")
        ax.set_xlabel("Relative Difference")
        ax.set_ylabel(metrics_names[index])
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fixed_level_{target_level}_owl.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 成功保存对比图至: {save_path}")


def plot_fixed_arr(target_arr, levels, relative_differences, average_retention_ratios, results, save_dir):
    z_index = average_retention_ratios.index(target_arr)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ["CER", "WER", "Avg Loss"]

    for level_index, level in enumerate(levels):
        data_slice = results[level_index, :, z_index, :]
        for metric_index in range(3):
            axes[metric_index].plot(
                relative_differences,
                data_slice[:, metric_index],
                marker="s",
                label=f"Level = {level}",
            )

    for index, ax in enumerate(axes):
        ax.set_title(f"{metrics_names[index]} (ARR = {target_arr})")
        ax.set_xlabel("Relative Difference")
        ax.set_ylabel(metrics_names[index])
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fixed_arr_{target_arr}_owl.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 成功保存对比图至: {save_path}")


# ============================================================
# 只改这里
# ============================================================
MODEL_NAME = "whisper-large-v3-original"
DATASET_NAME = "en"
DTYPE = "float16"
DEVICE = None

MODEL_ROOT = "./models"
DATA_ROOT = "./data/fleurs_full"
SAVE_DIR = f"./outputs/visualize_owl_onetimepruning/{MODEL_NAME}"

PROFILE_SPLIT = "train"
PROFILE_BATCH_SIZE = 64
PROFILE_NUM_SAMPLES = 64

EVAL_SPLIT = "test"
EVAL_BATCH_SIZE = 32
EVAL_NUM_SAMPLES = None

LEVELS = [8, 9]
RELATIVE_DIFFERENCES = [x * 0.03 for x in range(0, 11)]
AVERAGE_RETENTION_RATIOS = [x * 0.05 for x in range(6, 9)]
# ============================================================


def main():
    model, processor, device, torch_dtype = load_model_and_processor(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        device=DEVICE,
        model_root=MODEL_ROOT,
    )

    profile_loader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=PROFILE_SPLIT,
        batch_size=PROFILE_BATCH_SIZE,
        num_samples=PROFILE_NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=True,
    )

    profiler = WAprofiler(model, profile_loader, device=device, dtype=torch_dtype)
    weights, stats = profiler.getWA()
    original_weights_backup = {key: value.clone() for key, value in model.state_dict().items()}

    eval_loader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=EVAL_SPLIT,
        batch_size=EVAL_BATCH_SIZE,
        num_samples=EVAL_NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=False,
    )

    scorer = Scorer()
    results = np.zeros((len(LEVELS), len(RELATIVE_DIFFERENCES), len(AVERAGE_RETENTION_RATIOS), 3))

    for x, level in enumerate(LEVELS):
        for y, relative_difference in enumerate(RELATIVE_DIFFERENCES):
            for z, average_retention_ratio in enumerate(AVERAGE_RETENTION_RATIOS):
                cer, wer, avg_loss = one_time_owl_pruning(
                    model=model,
                    processor=processor,
                    dataloader=eval_loader,
                    scorer=scorer,
                    weights=weights,
                    stats=stats,
                    level=level,
                    relative_difference=relative_difference,
                    average_retention_ratio=average_retention_ratio,
                    device=device,
                    dtype=torch_dtype,
                )
                results[x, y, z] = [cer, wer, avg_loss]
                model.load_state_dict(original_weights_backup)

    for level in LEVELS:
        plot_fixed_level(
            target_level=level,
            levels=LEVELS,
            relative_differences=RELATIVE_DIFFERENCES,
            average_retention_ratios=AVERAGE_RETENTION_RATIOS,
            results=results,
            save_dir=SAVE_DIR,
        )

    for average_retention_ratio in AVERAGE_RETENTION_RATIOS:
        plot_fixed_arr(
            target_arr=average_retention_ratio,
            levels=LEVELS,
            relative_differences=RELATIVE_DIFFERENCES,
            average_retention_ratios=AVERAGE_RETENTION_RATIOS,
            results=results,
            save_dir=SAVE_DIR,
        )


if __name__ == "__main__":
    main()
