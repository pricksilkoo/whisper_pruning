"""
这也是一个“快捷脚本”。

它对应的是你原来的核心实验:
1. 先收集激活统计
2. 再计算每层保留率
3. 然后执行一次性剪枝
4. 最后做评测

现在这些逻辑都被放进统一流水线 `run_one_shot_pruning(...)` 里了，
这个文件只负责提供一组默认参数。
"""

from whisper_pruning.config import (
    DataLoaderConfig,
    ExperimentConfig,
    OneShotPruningRunConfig,
    PruningConfig,
    ScoringConfig,
)
from whisper_pruning.pipelines import run_one_shot_pruning


if __name__ == "__main__":
    # 这部分是在描述“一次剪枝实验”的全部输入参数。
    config = OneShotPruningRunConfig(
        experiment=ExperimentConfig(
            model_name="whisper-large-v3-original",
            dataset_name="en",
            dtype="float32",
        ),
        profile_data=DataLoaderConfig(
            split="train",
            batch_size=32,
            num_samples=64,
        ),
        eval_data=DataLoaderConfig(
            split="test",
            batch_size=16,
            num_samples=512,
            shuffle=False,
        ),
        pruning=PruningConfig(
            method="wanda_unstructured",
            scoring=ScoringConfig(
                method="owl",
                level=7,
                relative_difference=0,
                average_retention_ratio=0.4,
            ),
        ),
    )

    # 真正执行流程的是 run_one_shot_pruning。
    run_one_shot_pruning(config)
