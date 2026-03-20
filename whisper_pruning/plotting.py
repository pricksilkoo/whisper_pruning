from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def _to_numpy_flat(data):
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.array(data).flatten()


def visualize_network_scores(
    scores,
    save_dir="score_plots",
    filename="network_scores.png",
    title="Layer-wise Scores across the Network",
):
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

    plt.plot(
        x_indices,
        sorted_scores,
        marker="o",
        markersize=4,
        linestyle="-",
        linewidth=1.5,
        color="dodgerblue",
        alpha=0.8,
        zorder=3,
    )
    plt.fill_between(x_indices, sorted_scores, 0, color="dodgerblue", alpha=0.1)

    for boundary in layer_boundaries:
        plt.axvline(x=boundary - 0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6, zorder=1)

    if encoder_decoder_boundary is not None:
        plt.axvline(
            x=encoder_decoder_boundary - 0.5,
            color="red",
            linestyle="-",
            linewidth=2.5,
            alpha=0.9,
            zorder=2,
            label="Encoder-Decoder Boundary",
        )

    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("Score Value", fontsize=14)
    plt.xlabel("Layer Index (Topology Order)", fontsize=14)
    plt.xlim(-1, len(sorted_layer_names))
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    if encoder_decoder_boundary is not None:
        plt.legend(fontsize=12, loc="upper right")

    step = max(1, len(sorted_layer_names) // 40)
    plt.xticks(x_indices[::step], x_indices[::step], fontsize=10)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n✅ 拓扑分数图已保存至: {os.path.abspath(save_path)}")
    plt.close()


def select_layer(weights, stats, selected_layer=None):
    common_layers = sorted(list(set(weights.keys()) & set(stats.keys())))
    if not common_layers:
        raise ValueError("weights 和 stats 没有匹配的层名。")

    if selected_layer is not None:
        if selected_layer not in common_layers:
            raise ValueError(f"指定层不存在: {selected_layer}")
        return selected_layer

    print("\n" + "=" * 50)
    print(" 📊 可用的网络层对照表 ")
    print("=" * 50)
    for index, layer_name in enumerate(common_layers):
        print(f"[{index:03d}] {layer_name}")
    print("=" * 50 + "\n")

    while True:
        choice = input(f"请输入想要可视化的层编号 (0-{len(common_layers)-1})，或输入 'q' 退出: ")
        if choice.lower() == "q":
            raise KeyboardInterrupt("用户取消了可视化。")
        try:
            layer_index = int(choice)
        except ValueError:
            print("⚠️ 输入无效，请输入数字编号。")
            continue
        if 0 <= layer_index < len(common_layers):
            return common_layers[layer_index]
        print("⚠️ 编号超出范围，请重新输入。")


def visualize_distributions(weights, stats, save_dir, model_name, selected_layer=None):
    selected_layer = select_layer(weights, stats, selected_layer=selected_layer)
    print(f"\n⏳ 正在处理并绘制 [{selected_layer}] 的数据，请稍候...")

    w_raw = torch.abs(weights[selected_layer])
    s_raw = (stats[selected_layer]["sq_sum"] / stats[selected_layer]["count"]).sqrt()
    prod_raw = w_raw * s_raw

    w_data = _to_numpy_flat(w_raw)
    s_data = _to_numpy_flat(s_raw)
    prod_data = _to_numpy_flat(prod_raw)

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
