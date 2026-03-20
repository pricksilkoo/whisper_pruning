"""
这个文件用于“只做基线评测”。

以后你最常改的地方，就是下面这段配置。
改完后直接运行：
    python evaluate_model.py
"""

import os
import tempfile

import torch
import torch.multiprocessing as mp

from experiment_helpers import configure_torch_runtime, load_data, load_model_and_processor
from utils.evaluator import Evaluator, compute_metrics, get_text_normalizer, print_evaluation_summary


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

SPLIT = "test"
BATCH_SIZE = 64
NUM_SAMPLES = None
NUM_WORKERS = 4
# ============================================================


def evaluate_single_gpu():
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
        num_workers=NUM_WORKERS,
    )

    evaluator = Evaluator(
        model=model,
        processor=processor,
        dataloader=dataloader,
        device=device,
        language=DATASET_NAME,
        task="transcribe",
        dtype=torch_dtype,
    )
    return evaluator.evaluate()


def _evaluate_worker(worker_rank, gpu_id, result_dir, num_shards):
    configure_torch_runtime()
    device_name = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    model, processor, device, torch_dtype = load_model_and_processor(
        model_name=MODEL_NAME,
        dtype=DTYPE,
        device=device_name,
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
        num_workers=0,
        shard_id=worker_rank,
        num_shards=num_shards,
    )

    evaluator = Evaluator(
        model=model,
        processor=processor,
        dataloader=dataloader,
        device=device,
        language=DATASET_NAME,
        task="transcribe",
        dtype=torch_dtype,
    )
    result = evaluator.evaluate(log=False, return_details=True)
    torch.save(result, os.path.join(result_dir, f"worker_{worker_rank}.pt"))


def evaluate_multi_gpu():
    gpu_ids = GPU_IDS if torch.cuda.is_available() else []
    if not gpu_ids:
        raise RuntimeError("多卡评测要求当前环境可见 CUDA GPU。")

    if NUM_SAMPLES is not None:
        gpu_ids = gpu_ids[: max(1, min(len(gpu_ids), NUM_SAMPLES))]

    num_shards = len(gpu_ids)

    with tempfile.TemporaryDirectory(prefix="whisper_eval_") as result_dir:
        ctx = mp.get_context("spawn")
        processes = []

        for worker_rank, gpu_id in enumerate(gpu_ids):
            process = ctx.Process(
                target=_evaluate_worker,
                args=(worker_rank, gpu_id, result_dir, num_shards),
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

    if USE_MULTI_GPU_EVAL and torch.cuda.is_available() and len(GPU_IDS) > 1:
        evaluate_multi_gpu()
        return

    evaluate_single_gpu()


if __name__ == "__main__":
    main()
