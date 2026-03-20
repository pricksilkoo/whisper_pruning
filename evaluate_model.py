"""
这个文件用于“只做基线评测”。

以后你最常改的地方，就是下面这段配置。
改完后直接运行：
    python evaluate_model.py
"""

from experiment_helpers import load_data, load_model_and_processor
from utils.evaluator import Evaluator


# ============================================================
# 只改这里
# ============================================================
MODEL_NAME = "whisper-large-v3-original"
DATASET_NAME = "en"
DTYPE = "float16"
DEVICE = None

MODEL_ROOT = "./models"
DATA_ROOT = "./data/fleurs_full"

SPLIT = "test"
BATCH_SIZE = 64
NUM_SAMPLES = None
# ============================================================


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

    evaluator = Evaluator(
        model=model,
        processor=processor,
        dataloader=dataloader,
        device=device,
        language=DATASET_NAME,
        task="transcribe",
        dtype=torch_dtype,
    )
    evaluator.evaluate()


if __name__ == "__main__":
    main()
