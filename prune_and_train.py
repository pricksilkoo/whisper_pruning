"""
这个脚本用于:
1. 先做一次 one-shot pruning
2. 再挂着固定 mask 继续训练
3. 训练完成后评测

直接运行:
    python prune_and_train.py
"""

import os

import torch

from experiment_helpers import configure_torch_runtime, load_data, load_model_and_processor
from utils.evaluator import Evaluator
from utils.pruning_basemethod import PruningBaseMethod
from utils.scorer import Scorer
from utils.signal_collector import SignalCollector


# ============================================================
# 只改这里
# ============================================================
MODEL_NAME = "whisper-large-v3-original"
DATASET_NAME = "en"
DTYPE = "bfloat16"  # 重新训练更推荐 bfloat16；不支持时可改成 float16 / float32
DEVICE = "cuda:1"

MODEL_ROOT = "./models"
DATA_ROOT = "./data/fleurs_full"
TEXT_FIELD = "raw_transcription"

# 第一步: prune once 用的数据
PROFILE_SPLIT = "train"
PROFILE_BATCH_SIZE = 64
PROFILE_NUM_SAMPLES = 256
PROFILE_RANDOM_SUBSET = True
PROFILE_SAMPLE_SEED = 42
PROFILE_NUM_WORKERS = 4
PROFILE_GRADIENTS = False

# 第二步: 重新训练用的数据
TRAIN_SPLIT = "train"
TRAIN_BATCH_SIZE = 16
TRAIN_NUM_SAMPLES = None
TRAIN_RANDOM_SUBSET = False
TRAIN_SAMPLE_SEED = 42
TRAIN_NUM_WORKERS = 4
NUM_EPOCHS = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
LOG_EVERY = 10

# 第三步: 评测用的数据
EVAL_SPLIT = "test"
EVAL_BATCH_SIZE = 16
EVAL_NUM_SAMPLES = None
EVAL_NUM_WORKERS = 4
EVAL_GENERATION_BATCH_SIZE = 8
EVAL_COMPUTE_LOSS = False

# 剪枝配置
PRUNING_METHOD = "wanda"  # wanda / sparsegpt
SPARSITY_MODE = "uniform"  # uniform / layerwise
UNIFORM_SPARSITY = 0.6

# 只有 layerwise 时生效
SCORE_METHOD = "owl"  # owl / cv
LEVEL = 7
RELATIVE_DIFFERENCE = 0.0
AVERAGE_RETENTION_RATIO = 0.4

# 只有 sparsegpt 时生效
SPARSEGPT_BLOCKSIZE = 128
SPARSEGPT_DAMPING = 0.01
SPARSEGPT_MAX_BATCHES = None

# 保存
SAVE_DIR = f"./outputs/prune_and_train/{MODEL_NAME}"
SAVE_MODEL = True
SAVE_MASKS = True
# ============================================================


def build_sparsity(weights, activations, gradients):
    """
    根据配置生成这次 one-shot pruning 要用的总稀疏度。
    """
    if SPARSITY_MODE == "uniform":
        print(f"✂️ 使用 uniform {PRUNING_METHOD}，统一稀疏度 = {UNIFORM_SPARSITY:.2%}")
        return UNIFORM_SPARSITY

    if SPARSITY_MODE == "layerwise":
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
        sparsity = {name: 1.0 - ratio for name, ratio in retention_ratio.items()}
        print(f"💻 {SCORE_METHOD} 打分完成，共得到 {len(scores)} 层分数")
        return sparsity

    raise ValueError(f"不支持的 SPARSITY_MODE: {SPARSITY_MODE}")


def evaluate_current_model(model, processor, device, torch_dtype):
    eval_loader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=EVAL_SPLIT,
        batch_size=EVAL_BATCH_SIZE,
        num_samples=EVAL_NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=False,
        num_workers=EVAL_NUM_WORKERS,
        text_field=TEXT_FIELD,
    )

    evaluator = Evaluator(
        model=model,
        processor=processor,
        dataloader=eval_loader,
        device=device,
        language=DATASET_NAME,
        task="transcribe",
        dtype=torch_dtype,
        generation_kwargs={
            "generation_batch_size": EVAL_GENERATION_BATCH_SIZE,
        },
        compute_loss=EVAL_COMPUTE_LOSS,
    )
    return evaluator.evaluate()


def build_module_masks(model, masks):
    """
    把 pruner 产出的 mask 对齐到当前模型模块上。
    """
    module_masks = {}
    for name, module in model.named_modules():
        if name not in masks:
            continue
        module_masks[name] = masks[name].to(device=module.weight.device)
    return module_masks


def apply_masks_to_weights(model, module_masks):
    """
    确保被剪掉的位置始终保持为 0。
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if name not in module_masks:
                continue
            mask = module_masks[name].to(dtype=module.weight.data.dtype)
            module.weight.data.mul_(mask)


def apply_masks_to_gradients(model, module_masks):
    """
    让被剪掉的位置在训练时梯度恒为 0。
    """
    for name, module in model.named_modules():
        if name not in module_masks or module.weight.grad is None:
            continue
        mask = module_masks[name].to(dtype=module.weight.grad.dtype)
        module.weight.grad.mul_(mask)


def measure_model_sparsity(model):
    total_model_params = sum(parameter.numel() for parameter in model.parameters())
    linear_params_total = 0
    total_pruned_zeros = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if "proj_out" in name:
            continue
        weight = module.weight.data
        linear_params_total += weight.numel()
        total_pruned_zeros += (weight == 0).sum().item()

    linear_sparsity = total_pruned_zeros / linear_params_total if linear_params_total else 0.0
    global_sparsity = total_pruned_zeros / total_model_params if total_model_params else 0.0
    return linear_sparsity, global_sparsity


def train_with_fixed_masks(model, processor, device, torch_dtype, module_masks):
    train_loader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=TRAIN_SPLIT,
        batch_size=TRAIN_BATCH_SIZE,
        num_samples=TRAIN_NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=True,
        random_subset=TRAIN_RANDOM_SUBSET,
        seed=TRAIN_SAMPLE_SEED,
        num_workers=TRAIN_NUM_WORKERS,
        text_field=TEXT_FIELD,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    original_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = False
    apply_masks_to_weights(model, module_masks)

    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            optimizer.zero_grad(set_to_none=True)

            input_features = batch["input_features"].to(device, dtype=torch_dtype)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_features=input_features,
                labels=labels,
                attention_mask=attention_mask,
            )
            loss = outputs.loss
            loss.backward()

            apply_masks_to_gradients(model, module_masks)
            if MAX_GRAD_NORM is not None and MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            apply_masks_to_weights(model, module_masks)

            loss_value = loss.item()
            running_loss += loss_value

            if step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                print(
                    f"Epoch {epoch}/{NUM_EPOCHS} | "
                    f"Step {step} | "
                    f"Global Step {global_step} | "
                    f"Loss {avg_loss:.4f}"
                )
                running_loss = 0.0

        linear_sparsity, global_sparsity = measure_model_sparsity(model)
        print(
            f"📍 Epoch {epoch} 完成 | "
            f"线性层稀疏度 {linear_sparsity:.2%} | "
            f"整网全局稀疏度 {global_sparsity:.2%}"
        )

    model.config.use_cache = original_use_cache
    apply_masks_to_weights(model, module_masks)


def main():
    configure_torch_runtime()

    model, processor, device, torch_dtype = load_model_and_processor(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        device=DEVICE,
        model_root=MODEL_ROOT,
    )

    # 第一步: 收集 pruning 统计
    profile_loader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=PROFILE_SPLIT,
        batch_size=PROFILE_BATCH_SIZE,
        num_samples=PROFILE_NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=True,
        random_subset=PROFILE_RANDOM_SUBSET,
        seed=PROFILE_SAMPLE_SEED,
        num_workers=PROFILE_NUM_WORKERS,
        text_field=TEXT_FIELD,
    )

    collector = SignalCollector(model, profile_loader, device=device, dtype=torch_dtype)
    weights, activations, gradients = collector.collect(
        collect_activations=True,
        collect_gradients=PROFILE_GRADIENTS,
    )

    sparsity = build_sparsity(weights, activations, gradients)

    # 第二步: one-shot pruning
    pruner = PruningBaseMethod()
    pruner.prune(
        method=PRUNING_METHOD,
        weights=weights,
        activations=activations,
        gradients=gradients,
        sparsity=sparsity,
        model=model,
        dataloader=profile_loader,
        device=device,
        dtype=torch_dtype,
        blocksize=SPARSEGPT_BLOCKSIZE,
        damping=SPARSEGPT_DAMPING,
        max_batches=SPARSEGPT_MAX_BATCHES,
    )
    pruner.apply_to_model(model)

    module_masks = build_module_masks(model, pruner.masks)
    linear_sparsity, global_sparsity = measure_model_sparsity(model)
    print(
        f"📍 One-shot pruning 完成 | "
        f"线性层稀疏度 {linear_sparsity:.2%} | "
        f"整网全局稀疏度 {global_sparsity:.2%}"
    )

    # 第三步: 挂 mask 继续训练
    train_with_fixed_masks(
        model=model,
        processor=processor,
        device=device,
        torch_dtype=torch_dtype,
        module_masks=module_masks,
    )

    # 第四步: 保存
    if SAVE_MODEL or SAVE_MASKS:
        os.makedirs(SAVE_DIR, exist_ok=True)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": {
                "model_name": MODEL_NAME,
                "dataset_name": DATASET_NAME,
                "dtype": DTYPE,
                "pruning_method": PRUNING_METHOD,
                "sparsity_mode": SPARSITY_MODE,
            },
        }
        if SAVE_MASKS:
            checkpoint["masks"] = {name: mask.cpu() for name, mask in module_masks.items()}

        checkpoint_path = os.path.join(SAVE_DIR, "pruned_retrained_checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 已保存到: {os.path.abspath(checkpoint_path)}")

    # 第五步: 最终评测
    evaluate_current_model(model, processor, device, torch_dtype)


if __name__ == "__main__":
    main()
