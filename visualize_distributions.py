import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from experiment_helpers import load_data, load_model_and_processor
from utils.WandA_profiler import WAprofiler


def to_numpy_flat(data):
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.array(data).flatten()


def visualize_distributions(weights, stats, save_dir, model_name):
    common_layers = sorted(list(set(weights.keys()) & set(stats.keys())))
    if not common_layers:
        print("❌ 错误：weights 和 stats 字典中没有匹配的层名。")
        return

    print("\n" + "=" * 50)
    print(" 📊 可用的网络层对照表 ")
    print("=" * 50)
    for index, layer_name in enumerate(common_layers):
        print(f"[{index:03d}] {layer_name}")
    print("=" * 50 + "\n")

    while True:
        choice = input(f"请输入想要可视化的层编号 (0-{len(common_layers)-1})，或输入 'q' 退出: ")
        if choice.lower() == "q":
            print("已退出可视化。")
            return

        try:
            layer_index = int(choice)
        except ValueError:
            print("⚠️ 输入无效，请输入数字编号。")
            continue

        if 0 <= layer_index < len(common_layers):
            selected_layer = common_layers[layer_index]
            break

        print("⚠️ 编号超出范围，请重新输入。")

    print(f"\n⏳ 正在处理并绘制 [{selected_layer}] 的数据，请稍候...")

    w_raw = torch.abs(weights[selected_layer])
    s_raw = (stats[selected_layer]["sq_sum"] / stats[selected_layer]["count"]).sqrt()
    prod_raw = w_raw * s_raw

    w_data = to_numpy_flat(w_raw)
    s_data = to_numpy_flat(s_raw)
    prod_data = to_numpy_flat(prod_raw)

    w_mean = np.mean(w_data)
    s_mean = np.mean(s_data)
    prod_mean = np.mean(prod_data)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f"Distribution Analysis: {model_name} -- {selected_layer}", fontsize=16, fontweight="bold")

    def plot_with_mean(ax, data, mean_value, title, color):
        sns.histplot(data, bins=100, kde=True, color=color, ax=ax, stat="probability")
        ax.set_title(f"{title}\nMean: {mean_value:.6f}", fontsize=13)
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability (Log Scale)")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1)
        ax.axvline(x=mean_value, color="red", linestyle="--", linewidth=2, alpha=0.8)

    plot_with_mean(axes[0], w_data, w_mean, "Weights (|W|) Distribution", "steelblue")
    plot_with_mean(axes[1], s_data, s_mean, "Activations (||X||_l2) Distribution", "darkorange")
    plot_with_mean(axes[2], prod_data, prod_mean, "Product (|W| * ||X||_l2) Distribution", "seagreen")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)

    safe_layer_name = selected_layer.replace(".", "_").replace("/", "_")
    save_path = os.path.join(save_dir, f"dist_{safe_layer_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 图片已保存至: {os.path.abspath(save_path)}")
    plt.close(fig)


# 只改这里
MODEL_NAME = "whisper-large-v3-original"
DATASET_NAME = "en"
DTYPE = "float32"
DEVICE = None

MODEL_ROOT = "./models"
DATA_ROOT = "./data/fleurs_full"
SAVE_DIR = f"./outputs/visualize_distributions/{MODEL_NAME}"

SPLIT = "test"
BATCH_SIZE = 4
NUM_SAMPLES = 8


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
    )

    profiler = WAprofiler(model, dataloader, device=device, dtype=torch_dtype)
    weights, stats = profiler.getWA()
    visualize_distributions(weights, stats, SAVE_DIR, MODEL_NAME)


if __name__ == "__main__":
    main()
