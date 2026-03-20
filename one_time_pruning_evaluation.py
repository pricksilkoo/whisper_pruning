"""
这个文件对应你的核心实验：
1. 收集激活统计
2. 计算剪枝稀疏度
3. 执行一次性剪枝
4. 评测剪枝后的模型

以后你主要改下面的配置块，然后直接运行：
    python one_time_pruning_evaluation.py
"""

import os
import tempfile

import torch
import torch.multiprocessing as mp

from experiment_helpers import configure_torch_runtime, load_data, load_model_and_processor
from utils.WandA_profiler import WAprofiler
from utils.evaluator import Evaluator, compute_metrics, get_text_normalizer, print_evaluation_summary
from utils.pruning_tools import PruningTool
from utils.scorer import Scorer


# ============================================================
# 只改这里
# ============================================================
MODEL_NAME = "whisper-large-v3-original"
DATASET_NAME = "en"
DTYPE = "float16"  # A40 上更快；如果你想做更稳的统计，可以改回 float32
DEVICE = None

MODEL_ROOT = "./models"
DATA_ROOT = "./data/fleurs_full"
GPU_IDS = [0, 1, 2, 3]
USE_MULTI_GPU_EVAL = True

# 用来收集激活值的小样本数据
PROFILE_SPLIT = "train"
PROFILE_BATCH_SIZE = 64
PROFILE_NUM_SAMPLES = 256
PROFILE_RANDOM_SUBSET = True
PROFILE_SAMPLE_SEED = 42
PROFILE_NUM_WORKERS = 4

# 用来最终评测的数据
EVAL_SPLIT = "test"
EVAL_BATCH_SIZE = 32
EVAL_NUM_SAMPLES = 512
EVAL_NUM_WORKERS = 4

# 剪枝模式
# - "uniform": 真正的统一剪枝，所有层都用同一个 sparsity
# - "layerwise": 先打分，再给不同层分配不同 sparsity
SPARSITY_MODE = "uniform"  # uniform / layerwise
UNIFORM_SPARSITY = 0.6

# 只有在 SPARSITY_MODE="layerwise" 时，这组参数才会生效
SCORE_METHOD = "owl"  # owl / cv
LEVEL = 7
RELATIVE_DIFFERENCE = 0
AVERAGE_RETENTION_RATIO = 0.4

# ============================================================


def get_profile_device():
    if DEVICE is not None:
        return DEVICE
    if torch.cuda.is_available() and GPU_IDS:
        return f"cuda:{GPU_IDS[0]}"
    return None


def build_sparsity(weights, stats):
    """
    根据配置决定这次到底是:
    - 真 uniform 剪枝
    - 还是 layerwise 剪枝
    """
    if SPARSITY_MODE == "uniform":
        print(f"✂️ 使用 uniform Wanda，统一稀疏度 = {UNIFORM_SPARSITY:.2%}")
        return UNIFORM_SPARSITY

    if SPARSITY_MODE == "layerwise":
        scorer = Scorer()
        scores, retention_ratio = scorer.compute(
            method=SCORE_METHOD,
            weights=weights,
            activations_stats=stats,
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
    )

    evaluator = Evaluator(
        model=model,
        processor=processor,
        dataloader=eval_loader,
        device=device,
        language=DATASET_NAME,
        task="transcribe",
        dtype=torch_dtype,
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
        num_workers=EVAL_NUM_WORKERS,
        shard_id=worker_rank,
        num_shards=num_shards,
    )

    evaluator = Evaluator(
        model=model,
        processor=processor,
        dataloader=eval_loader,
        device=device,
        language=DATASET_NAME,
        task="transcribe",
        dtype=torch_dtype,
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
        total_batches = 0

        for worker_rank in range(num_shards):
            result = torch.load(os.path.join(result_dir, f"worker_{worker_rank}.pt"))
            all_references.extend(result["references"])
            all_predictions.extend(result["predictions"])
            total_loss += result["total_loss"]
            total_batches += result["num_batches"]

    normalizer = get_text_normalizer(DATASET_NAME)
    cer, wer, _, _ = compute_metrics(
        references=all_references,
        predictions=all_predictions,
        normalizer=normalizer,
    )
    avg_loss = total_loss / total_batches
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

    # 第一步：收集激活统计
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
    )

    profiler = WAprofiler(model, profile_loader, device=device, dtype=torch_dtype)
    weights, stats = profiler.getWA()

    # 第二步：生成这次要用的 sparsity
    sparsity = build_sparsity(weights, stats)

    # 第三步：执行剪枝
    pruner = PruningTool()
    pruner.wanda_unstructured_pruning(weights, stats, sparsity=sparsity)
    pruner.apply_to_model(model)

    # 第四步：评测剪枝后的模型
    if USE_MULTI_GPU_EVAL and torch.cuda.is_available() and len(GPU_IDS) > 1:
        with tempfile.TemporaryDirectory(prefix="whisper_pruned_model_") as temp_dir:
            checkpoint_path = os.path.join(temp_dir, "pruned_state_dict.pt")
            torch.save(model.state_dict(), checkpoint_path)

            # 主进程释放显存，让 4 张卡都能留给评测子进程。
            model.to("cpu")
            del model, pruner, weights, stats
            torch.cuda.empty_cache()

            evaluate_checkpoint_multi_gpu(checkpoint_path)
    else:
        evaluate_current_model(model, processor, device, torch_dtype)


if __name__ == "__main__":
    main()
