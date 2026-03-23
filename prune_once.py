"""
这个文件对应你的核心实验：
1. 收集激活统计
2. 计算剪枝稀疏度
3. 执行一次性剪枝
4. 评测剪枝后的模型

以后你主要改下面的配置块，然后直接运行：
    python prune_once.py
"""

import os
import tempfile

import torch
import torch.multiprocessing as mp

from experiment_helpers import configure_torch_runtime, load_data, load_model_and_processor
from utils.evaluator import Evaluator, compute_metrics, get_text_normalizer, print_evaluation_summary
from utils.signal_collector import SignalCollector
from utils.pruning_basemethod import PruningBaseMethod
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
TEXT_FIELD = "raw_transcription"
EVAL_GENERATION_BATCH_SIZE = 8
EVAL_COMPUTE_LOSS = False

# 用来收集激活值的小样本数据
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

# 剪枝模式
# - "uniform": 真正的统一剪枝，所有层都用同一个 sparsity
# - "layerwise": 先打分，再给不同层分配不同 sparsity
SPARSITY_MODE = "uniform"  # uniform / layerwise
PRUNING_METHOD = "sparsegpt"  # wanda / sparsegpt
UNIFORM_SPARSITY = 0.7
SPARSEGPT_BLOCKSIZE = 256
SPARSEGPT_DAMPING = 0.01
SPARSEGPT_MAX_BATCHES = 16

# 保存
SAVE_DIR = f"./outputs/prune_once/{MODEL_NAME}"
SAVE_MODEL = True
SAVE_MASKS = True

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


def build_sparsity(weights, activations, gradients):
    """
    根据配置决定这次到底是:
    - 真 uniform 剪枝
    - 还是 layerwise 剪枝
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
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
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


def build_module_masks(model, masks):
    """
    把 pruner 产出的 mask 对齐到当前模型模块设备上，便于保存。
    """
    module_masks = {}
    for name, module in model.named_modules():
        if name not in masks:
            continue
        module_masks[name] = masks[name].to(device=module.weight.device)
    return module_masks


def build_save_filename(sparsity):
    """
    生成带有剪枝方法和稀疏度信息的文件名。
    """
    method_tag = PRUNING_METHOD.lower()
    mode_tag = SPARSITY_MODE.lower()

    if isinstance(sparsity, dict):
        avg_sparsity = (
            sum(float(value) for value in sparsity.values()) / len(sparsity) if sparsity else 0.0
        )
        return f"pruned_{method_tag}_{mode_tag}_{SCORE_METHOD.lower()}_avgsp{avg_sparsity:.4f}.pt"

    return f"pruned_{method_tag}_{mode_tag}_sp{float(sparsity):.4f}.pt"


def save_pruned_checkpoint(model, module_masks, sparsity):
    """
    保存一次性剪枝后的 checkpoint 和可选 mask。
    """
    if not SAVE_MODEL and not SAVE_MASKS:
        return None

    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint = {
        "config": {
            "model_name": MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "dtype": DTYPE,
            "pruning_method": PRUNING_METHOD,
            "sparsity_mode": SPARSITY_MODE,
            "sparsity": sparsity,
            "sparsegpt_blocksize": SPARSEGPT_BLOCKSIZE,
            "sparsegpt_damping": SPARSEGPT_DAMPING,
            "sparsegpt_max_batches": SPARSEGPT_MAX_BATCHES,
        },
    }

    if SAVE_MODEL:
        checkpoint["model_state_dict"] = {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }
    if SAVE_MASKS:
        checkpoint["masks"] = {name: mask.detach().cpu() for name, mask in module_masks.items()}

    checkpoint_filename = build_save_filename(sparsity)
    checkpoint_path = os.path.join(SAVE_DIR, checkpoint_filename)
    torch.save(checkpoint, checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)
    print(f"💾 已保存到: {checkpoint_path}")
    return checkpoint_path


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
        text_field=TEXT_FIELD,
    )

    collector = SignalCollector(model, profile_loader, device=device, dtype=torch_dtype)
    weights, activations, gradients = collector.collect(
        collect_activations=True,
        collect_gradients=PROFILE_GRADIENTS,
    )

    # 第二步：生成这次要用的 sparsity
    sparsity = build_sparsity(weights, activations, gradients)

    # 第三步：执行剪枝
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
    saved_checkpoint_path = save_pruned_checkpoint(
        model=model,
        module_masks=module_masks,
        sparsity=sparsity,
    )

    # 第四步：评测剪枝后的模型
    if USE_MULTI_GPU_EVAL and torch.cuda.is_available() and len(GPU_IDS) > 1:
        if saved_checkpoint_path is not None and SAVE_MODEL:
            checkpoint_path = saved_checkpoint_path
            temp_dir = None
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="whisper_pruned_model_")
            checkpoint_path = os.path.join(temp_dir.name, "pruned_state_dict.pt")
            torch.save(model.state_dict(), checkpoint_path)

        try:
            # 主进程释放显存，让 4 张卡都能留给评测子进程。
            model.to("cpu")
            del model, pruner, weights, activations, gradients, module_masks
            torch.cuda.empty_cache()

            evaluate_checkpoint_multi_gpu(checkpoint_path)
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
    else:
        evaluate_current_model(model, processor, device, torch_dtype)


if __name__ == "__main__":
    main()
