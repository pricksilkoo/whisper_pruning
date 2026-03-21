import os
import re

import matplotlib.pyplot as plt
import numpy as np

from experiment_helpers import load_data, load_model_and_processor
from utils.signal_collector import SignalCollector
from utils.scorer import Scorer


def visualize_network_scores(
    scores,
    save_dir="score_plots",
    filename="network_scores.png",
    title="Layer-wise Scores across the Network",
):
    """按 Whisper 的层顺序画出每层的分数。"""
    if not scores:
        print("❌ 传入的 scores 字典为空。")
        return

    def whisper_sort_key(layer_name):
        if "model.encoder" in layer_name or ".encoder." in layer_name:
            module_priority = 0
        elif "model.decoder" in layer_name or ".decoder." in layer_name:
            module_priority = 1
        else:
            module_priority = 2

        match = re.search(r"layers\.(\d+)", layer_name)
        layer_num = int(match.group(1)) if match else -1

        if "self_attn.q_proj" in layer_name:
            sub_order = 1
        elif "self_attn.k_proj" in layer_name:
            sub_order = 2
        elif "self_attn.v_proj" in layer_name:
            sub_order = 3
        elif "self_attn.out_proj" in layer_name:
            sub_order = 4
        elif "encoder_attn.q_proj" in layer_name:
            sub_order = 5
        elif "encoder_attn.k_proj" in layer_name:
            sub_order = 6
        elif "encoder_attn.v_proj" in layer_name:
            sub_order = 7
        elif "encoder_attn.out_proj" in layer_name:
            sub_order = 8
        elif "fc1" in layer_name:
            sub_order = 9
        elif "fc2" in layer_name:
            sub_order = 10
        else:
            sub_order = 99

        sub_components = [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", layer_name)]
        return (module_priority, layer_num, sub_order, sub_components)

    sorted_layer_names = sorted(scores.keys(), key=whisper_sort_key)
    sorted_scores = [scores[name] for name in sorted_layer_names]

    encoder_decoder_boundary = None
    layer_boundaries = []
    current_module = "encoder" if "model.encoder" in sorted_layer_names[0] else None
    current_layer_num = re.search(r"layers\.(\d+)", sorted_layer_names[0])
    current_layer_num = int(current_layer_num.group(1)) if current_layer_num else -1

    for index, name in enumerate(sorted_layer_names):
        is_decoder = "model.decoder" in name or ".decoder." in name
        if is_decoder and current_module == "encoder":
            encoder_decoder_boundary = index
            current_module = "decoder"

        match = re.search(r"layers\.(\d+)", name)
        layer_num = int(match.group(1)) if match else -1
        if layer_num != current_layer_num and layer_num != -1:
            layer_boundaries.append(index)
            current_layer_num = layer_num

    plt.figure(figsize=(24, 6))
    x_indices = np.arange(len(sorted_scores))

    plt.plot(x_indices, sorted_scores, marker="o", markersize=4, linestyle="-", linewidth=1.5)
    plt.fill_between(x_indices, sorted_scores, 0, alpha=0.1)

    for boundary in layer_boundaries:
        plt.axvline(x=boundary - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)

    if encoder_decoder_boundary is not None:
        plt.axvline(
            x=encoder_decoder_boundary - 0.5,
            color="red",
            linestyle="-",
            linewidth=2,
            alpha=0.9,
            label="Encoder-Decoder Boundary",
        )

    plt.title(title)
    plt.ylabel("Score Value")
    plt.xlabel("Layer Index")
    plt.xlim(-1, len(sorted_layer_names))
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    if encoder_decoder_boundary is not None:
        plt.legend(loc="upper right")

    step = max(1, len(sorted_layer_names) // 40)
    plt.xticks(x_indices[::step], x_indices[::step], fontsize=10)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ 图片已保存至: {os.path.abspath(save_path)}")
    plt.close()

# ==========================================
# 只改这里
MODEL_NAME = "whisper-medium-original"
DATASET_NAME = "en"
DTYPE = "float32"
DEVICE = None

MODEL_ROOT = "./models"
DATA_ROOT = "./data/fleurs_full"
SAVE_DIR = f"./outputs/visualize_scores/{MODEL_NAME}"

SPLIT = "test"
BATCH_SIZE = 4
NUM_SAMPLES = 8
RANDOM_SUBSET = False
SAMPLE_SEED = 42

SCORE_METHOD = "owl"  # owl / cv
LEVEL = 7
RELATIVE_DIFFERENCE = 0.0
AVERAGE_RETENTION_RATIO = 0.4

PLOT_RETENTION_RATIO = True
FILENAME = "owl_retention.png"


def main():
    model, processor, device, torch_dtype = load_model_and_processor(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        device=DEVICE,
        model_root=MODEL_ROOT,
    )

    dataloader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=SPLIT,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=False,
        random_subset=RANDOM_SUBSET,
        seed=SAMPLE_SEED,
    )

    collector = SignalCollector(model, dataloader, device=device, dtype=torch_dtype)
    weights, activations, gradients = collector.collect()

    scorer = Scorer()
    scores, retention_ratio = scorer.compute(
        method=SCORE_METHOD,
        weights=weights,
        activations=activations,
        gradients=gradients,
        level=LEVEL,
        relative_difference=RELATIVE_DIFFERENCE,
        average_retention_ratio=AVERAGE_RETENTION_RATIO,
    )

    values = retention_ratio if PLOT_RETENTION_RATIO and retention_ratio else scores
    title_name = "retention ratio" if PLOT_RETENTION_RATIO and retention_ratio else "scores"
    visualize_network_scores(
        values,
        SAVE_DIR,
        FILENAME,
        title=f"{SCORE_METHOD} {title_name} across Whisper",
    )


if __name__ == "__main__":
    main()
