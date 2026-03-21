"""
这个脚本用于“多轮 iterative prune once”。

你只需要给一个目标累计稀疏度，
脚本就会自动拆成多轮，每一轮都重新收集统计再剪一次，
直到达到目标。

直接运行：
    python prune_to_target.py
"""

import os
import tempfile

import torch
import torch.multiprocessing as mp

from experiment_helpers import configure_torch_runtime, load_data, load_model_and_processor
from utils.evaluator import Evaluator, compute_metrics, get_text_normalizer, print_evaluation_summary
from utils.pruning_basemethod import PruningBaseMethod
from utils.scorer import Scorer
from utils.signal_collector import SignalCollector


# ============================================================
# 只改这里
# ============================================================
MODEL_NAME = "whisper-large-v3-original"
DATASET_NAME = "en"
DTYPE = "float16"
DEVICE = None

MODEL_ROOT = "./models"
DATA_ROOT = "./data/fleurs_full"
GPU_IDS = [0, 1, 2, 3]
USE_MULTI_GPU_EVAL = True
TEXT_FIELD = "raw_transcription"
EVAL_GENERATION_BATCH_SIZE = 8
EVAL_COMPUTE_LOSS = False

# iterative pruning 目标
TARGET_LINEAR_SPARSITY = 0.8
NUM_PRUNING_ROUNDS = 4
EVALUATE_EACH_ROUND = False
TARGET_TOLERANCE = 1e-4

# 用来收集统计的小样本数据
PROFILE_SPLIT = "train"
PROFILE_BATCH_SIZE = 64
PROFILE_NUM_SAMPLES = 256
PROFILE_RANDOM_SUBSET = True
PROFILE_SAMPLE_SEED = 42
PROFILE_NUM_WORKERS = 4
PROFILE_GRADIENTS = False

# 用来最终评测的数据
EVAL_SPLIT = "test"
EVAL_BATCH_SIZE = 32
EVAL_NUM_SAMPLES = None
EVAL_NUM_WORKERS = 4

# 剪枝方式
PRUNING_METHOD = "wanda"  # wanda / sparsegpt
SPARSITY_MODE = "uniform"  # uniform / layerwise
SCORE_METHOD = "owl"  # 只有 layerwise 时生效
LEVEL = 7
RELATIVE_DIFFERENCE = 0.0

# SparseGPT 相关参数
SPARSEGPT_BLOCKSIZE = 128
SPARSEGPT_DAMPING = 0.01
SPARSEGPT_MAX_BATCHES = None
# ============================================================


def get_profile_device():
    if DEVICE is not None:
        return DEVICE
    if torch.cuda.is_available() and GPU_IDS:
        return f"cuda:{GPU_IDS[0]}"
    return None


def measure_sparsity(model):
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

    global_sparsity = total_pruned_zeros / total_model_params if total_model_params else 0.0
    linear_sparsity = total_pruned_zeros / linear_params_total if linear_params_total else 0.0
    return linear_sparsity, global_sparsity


def plan_round_targets(current_linear_sparsity, target_linear_sparsity, num_rounds):
    if target_linear_sparsity < current_linear_sparsity:
        raise ValueError("目标稀疏度不能小于当前稀疏度。")
    if not (0.0 <= target_linear_sparsity <= 1.0):
        raise ValueError("TARGET_LINEAR_SPARSITY 必须在 [0, 1] 内。")
    if num_rounds <= 0:
        raise ValueError("NUM_PRUNING_ROUNDS 必须大于 0。")

    current_remaining = 1.0 - current_linear_sparsity
    target_remaining = 1.0 - target_linear_sparsity
    ratio = target_remaining / current_remaining if current_remaining > 0 else 0.0

    targets = []
    for round_index in range(1, num_rounds + 1):
        remaining = current_remaining * (ratio ** (round_index / num_rounds))
        targets.append(1.0 - remaining)
    return targets


def build_round_sparsity(
    weights,
    activations,
    gradients,
    current_linear_sparsity,
    target_linear_sparsity,
):
    if target_linear_sparsity <= current_linear_sparsity + TARGET_TOLERANCE:
        return 0.0 if SPARSITY_MODE == "uniform" else {name: 0.0 for name in weights}

    current_remaining = max(1e-12, 1.0 - current_linear_sparsity)
    target_remaining = max(0.0, 1.0 - target_linear_sparsity)
    average_retention_ratio = target_remaining / current_remaining
    average_retention_ratio = float(min(max(average_retention_ratio, 0.0), 1.0))

    if SPARSITY_MODE == "uniform":
        additional_sparsity = 1.0 - average_retention_ratio
        print(
            f"✂️ 这一轮使用 uniform {PRUNING_METHOD} | "
            f"当前线性稀疏度 {current_linear_sparsity:.2%} -> 目标 {target_linear_sparsity:.2%} | "
            f"额外稀疏度 {additional_sparsity:.2%}"
        )
        return additional_sparsity

    if SPARSITY_MODE == "layerwise":
        scorer = Scorer()
        scores, retention_ratio = scorer.compute(
            method=SCORE_METHOD,
            weights=weights,
            activations=activations,
            gradients=gradients,
            level=LEVEL,
            relative_difference=RELATIVE_DIFFERENCE,
            average_retention_ratio=average_retention_ratio,
        )
        print(
            f"💻 这一轮使用 {SCORE_METHOD} 打分 | "
            f"当前线性稀疏度 {current_linear_sparsity:.2%} -> 目标 {target_linear_sparsity:.2%} | "
            f"额外平均保留率 {average_retention_ratio:.2%} | 共 {len(scores)} 层"
        )
        return {name: 1.0 - ratio for name, ratio in retention_ratio.items()}

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


def _evaluate_checkpoint_worker(worker_rank, gpu_id, checkpoint_path, result_dir, num_shards):
    configure_torch_runtime()
    device_name = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    model, processor, device, torch_dtype = load_model_and_processor(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        device=device_name,
        model_root=MODEL_ROOT,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)

    eval_loader = load_data(
        dataset_name=DATASET_NAME,
        processor=processor,
        split=EVAL_SPLIT,
        batch_size=EVAL_BATCH_SIZE,
        num_samples=EVAL_NUM_SAMPLES,
        data_root=DATA_ROOT,
        shuffle=False,
        num_workers=0,
        shard_id=worker_rank,
        num_shards=num_shards,
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
    result = evaluator.evaluate(log=False, return_details=True)
    torch.save(result, os.path.join(result_dir, f"worker_{worker_rank}.pt"))


def evaluate_checkpoint_multi_gpu(checkpoint_path):
    gpu_ids = GPU_IDS if torch.cuda.is_available() else []
    if not gpu_ids:
        raise RuntimeError("多卡评测要求当前环境可见 CUDA GPU。")

    if EVAL_NUM_SAMPLES is not None:
        gpu_ids = gpu_ids[: max(1, min(len(gpu_ids), EVAL_NUM_SAMPLES))]

    num_shards = len(gpu_ids)

    with tempfile.TemporaryDirectory(prefix="whisper_eval_") as result_dir:
        ctx = mp.get_context("spawn")
        processes = []

        for worker_rank, gpu_id in enumerate(gpu_ids):
            process = ctx.Process(
                target=_evaluate_checkpoint_worker,
                args=(worker_rank, gpu_id, checkpoint_path, result_dir, num_shards),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(f"评测子进程失败，exit code = {process.exitcode}")

        all_references = []
        all_predictions = []
        total_loss = 0.0
        total_loss_batches = 0

        for worker_rank in range(num_shards):
            result = torch.load(os.path.join(result_dir, f"worker_{worker_rank}.pt"))
            all_references.extend(result["references"])
            all_predictions.extend(result["predictions"])
            total_loss += result["total_loss"]
            total_loss_batches += result.get("num_loss_batches", result["num_batches"])

    normalizer = get_text_normalizer(DATASET_NAME)
    cer, wer, _, _ = compute_metrics(
        references=all_references,
        predictions=all_predictions,
        normalizer=normalizer,
    )
    avg_loss = total_loss / total_loss_batches if total_loss_batches > 0 else float("nan")
    print_evaluation_summary(avg_loss, cer, wer, all_references, all_predictions, log=True)
    return cer, wer, avg_loss


def main():
    configure_torch_runtime()

    model, processor, device, torch_dtype = load_model_and_processor(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        device=get_profile_device(),
        model_root=MODEL_ROOT,
    )

    current_linear_sparsity, current_global_sparsity = measure_sparsity(model)
    print(
        f"📍 初始稀疏度 | 线性层: {current_linear_sparsity:.2%} | "
        f"整网全局: {current_global_sparsity:.2%}"
    )

    round_targets = plan_round_targets(
        current_linear_sparsity=current_linear_sparsity,
        target_linear_sparsity=TARGET_LINEAR_SPARSITY,
        num_rounds=NUM_PRUNING_ROUNDS,
    )
    print("📌 每轮目标线性稀疏度:")
    for index, target in enumerate(round_targets, start=1):
        print(f"Round {index}: {target:.2%}")

    for round_index, round_target in enumerate(round_targets, start=1):
        current_linear_sparsity, current_global_sparsity = measure_sparsity(model)
        if current_linear_sparsity >= TARGET_LINEAR_SPARSITY - TARGET_TOLERANCE:
            print(f"✅ 已达到目标累计稀疏度 {TARGET_LINEAR_SPARSITY:.2%}，提前停止。")
            break

        print("\n" + "=" * 80)
        print(
            f"Round {round_index}/{NUM_PRUNING_ROUNDS} | "
            f"当前线性稀疏度 {current_linear_sparsity:.2%} | "
            f"本轮目标 {round_target:.2%}"
        )
        print("=" * 80)

        profile_loader = load_data(
            dataset_name=DATASET_NAME,
            processor=processor,
            split=PROFILE_SPLIT,
            batch_size=PROFILE_BATCH_SIZE,
            num_samples=PROFILE_NUM_SAMPLES,
            data_root=DATA_ROOT,
            shuffle=True,
            random_subset=PROFILE_RANDOM_SUBSET,
            seed=PROFILE_SAMPLE_SEED + round_index - 1,
            num_workers=PROFILE_NUM_WORKERS,
            text_field=TEXT_FIELD,
        )

        collector = SignalCollector(model, profile_loader, device=device, dtype=torch_dtype)
        weights, activations, gradients = collector.collect(
            collect_activations=True,
            collect_gradients=PROFILE_GRADIENTS,
        )

        sparsity = build_round_sparsity(
            weights=weights,
            activations=activations,
            gradients=gradients,
            current_linear_sparsity=current_linear_sparsity,
            target_linear_sparsity=round_target,
        )

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

        current_linear_sparsity, current_global_sparsity = measure_sparsity(model)
        print(
            f"📍 Round {round_index} 完成 | 线性层: {current_linear_sparsity:.2%} | "
            f"整网全局: {current_global_sparsity:.2%}"
        )

        if EVALUATE_EACH_ROUND:
            evaluate_current_model(model, processor, device, torch_dtype)

    print("\n" + "=" * 80)
    print(f"🎯 最终目标线性稀疏度: {TARGET_LINEAR_SPARSITY:.2%}")
    final_linear_sparsity, final_global_sparsity = measure_sparsity(model)
    print(
        f"📍 最终实际稀疏度 | 线性层: {final_linear_sparsity:.2%} | "
        f"整网全局: {final_global_sparsity:.2%}"
    )
    print("=" * 80 + "\n")

    if USE_MULTI_GPU_EVAL and torch.cuda.is_available() and len(GPU_IDS) > 1:
        with tempfile.TemporaryDirectory(prefix="whisper_pruned_model_") as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "pruned_state_dict.pt")
            torch.save(model.state_dict(), checkpoint_path)

            model.to("cpu")
            del model
            torch.cuda.empty_cache()

            evaluate_checkpoint_multi_gpu(checkpoint_path)
    else:
        evaluate_current_model(model, processor, device, torch_dtype)


if __name__ == "__main__":
    main()
