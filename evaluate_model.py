"""
这是一个“快捷脚本”。

它的作用不是自己实现评测逻辑，
而是提前写好一组默认参数，然后调用统一流水线 `run_evaluation(...)`。

如果你现在不想学习 CLI，用这个文件也可以直接跑基线评测。
"""

from whisper_pruning.config import DataLoaderConfig, EvaluationRunConfig, ExperimentConfig
from whisper_pruning.pipelines import run_evaluation


if __name__ == "__main__":
    # 这里的 config 就是在描述:
    # “我要评测哪个模型、哪个数据集、用什么 batch size”
    config = EvaluationRunConfig(
        experiment=ExperimentConfig(
            model_name="whisper-large-v3-original",
            dataset_name="en",
            dtype="float16",
        ),
        data=DataLoaderConfig(
            split="test",
            batch_size=64,
            num_samples=None,
            shuffle=False,
        ),
    )

    # 真正的评测实现不在这个文件里，
    # 而是在 whisper_pruning/pipelines.py 里。
    run_evaluation(config)
