from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# 这个常量表示“项目根目录”。
# 因为当前文件位于 whisper_pruning/config.py，
# 所以 parents[1] 就是仓库根目录 whisper_pruning/。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return PROJECT_ROOT


@dataclass(slots=True)
class ProjectPaths:
    """
    这个类只负责“路径管理”。

    你可以把它理解成:
    - 模型默认放哪
    - 数据默认放哪
    - 输出结果默认放哪

    以前这些路径散落在每个脚本里，现在统一收口到这里。
    """

    project_root: Path = field(default_factory=_project_root)
    model_root: Optional[Path] = None
    data_root: Optional[Path] = None
    output_root: Optional[Path] = None

    def __post_init__(self) -> None:
        # 如果用户没有手动传路径，就使用仓库里的默认目录。
        if self.model_root is None:
            self.model_root = self.project_root / "models"
        if self.data_root is None:
            self.data_root = self.project_root / "data" / "fleurs_full"
        if self.output_root is None:
            self.output_root = self.project_root / "outputs"

    def model_path(self, model_name: str) -> Path:
        # 根据模型名拼出完整模型路径。
        return self.model_root / model_name

    def dataset_path(self, dataset_name: str) -> Path:
        # 根据数据集名拼出完整数据路径。
        return self.data_root / dataset_name


@dataclass(slots=True)
class ExperimentConfig:
    """
    这个类描述“一次实验的公共信息”。

    这些信息无论是评测、剪枝还是画图都会用到，比如:
    - 模型名
    - 数据集名
    - dtype
    - device
    - 语言

    可以把它理解成“全局实验配置”。
    """

    model_name: str
    dataset_name: str
    dtype: str = "float16"
    device: Optional[str] = None
    language: Optional[str] = None
    task: str = "transcribe"
    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def __post_init__(self) -> None:
        # 如果没手动指定 language，就默认让语言和数据集名字一致。
        # 例如 dataset_name="en"，那 language 默认也是 "en"。
        if self.language is None:
            self.language = self.dataset_name

    @property
    def model_path(self) -> Path:
        # 这是一个“动态属性”。
        # 外部直接写 experiment.model_path，就能得到模型完整路径。
        return self.paths.model_path(self.model_name)

    @property
    def dataset_path(self) -> Path:
        # 同理，这里返回数据集完整路径。
        return self.paths.dataset_path(self.dataset_name)


@dataclass(slots=True)
class DataLoaderConfig:
    """
    这个类只描述 dataloader 相关参数。

    之所以单独拆出来，是因为同一个实验里可能有两套 dataloader:
    - 一套用来收集激活值
    - 一套用来最终评测
    """

    split: str
    batch_size: int
    num_samples: Optional[int] = None
    shuffle: Optional[bool] = None

    @property
    def should_shuffle(self) -> bool:
        # 如果用户明确指定了 shuffle，就优先用用户的。
        if self.shuffle is not None:
            return self.shuffle
        # 否则保持原来的常见约定:
        # train 默认打乱，test/validation 默认不打乱。
        return self.split == "train"


@dataclass(slots=True)
class ScoringConfig:
    """
    这个类描述“怎么打分”。

    对你现在的项目来说，打分的作用是:
    先根据权重和激活值，算出每层到底该保留多少。
    """

    method: str = "owl"
    level: float = 7.0
    relative_difference: float = 0.0
    average_retention_ratio: float = 0.4


@dataclass(slots=True)
class PruningConfig:
    """
    这个类描述“怎么剪枝”。

    包括:
    - 用哪种剪枝方法
    - 用哪种打分方法
    - 如果是 N:M 剪枝，n 和 m 分别是多少
    - 如果是统一稀疏度，uniform_sparsity 是多少
    """

    method: str = "wanda_unstructured"
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    uniform_sparsity: Optional[float] = None
    n: int = 2
    m: int = 4


@dataclass(slots=True)
class EvaluationRunConfig:
    """只做评测时，需要的总配置。"""

    experiment: ExperimentConfig
    data: DataLoaderConfig


@dataclass(slots=True)
class ProfileRunConfig:
    """只做 profiler 采样时，需要的总配置。"""

    experiment: ExperimentConfig
    data: DataLoaderConfig


@dataclass(slots=True)
class OneShotPruningRunConfig:
    """
    一次性剪枝实验的总配置。

    注意这里有两套 dataloader 配置:
    - profile_data: 用来收集激活值
    - eval_data: 用来做最终评测
    """

    experiment: ExperimentConfig
    profile_data: DataLoaderConfig
    eval_data: DataLoaderConfig
    pruning: PruningConfig = field(default_factory=PruningConfig)


@dataclass(slots=True)
class OwlSweepConfig:
    """
    用来做 OWL 参数扫描的总配置。

    扫描的含义是:
    用多组 level / relative_difference / average_retention_ratio
    反复做“剪枝 + 评测”，最后得到一堆结果用于画图对比。
    """

    experiment: ExperimentConfig
    profile_data: DataLoaderConfig
    eval_data: DataLoaderConfig
    levels: list[float] = field(default_factory=lambda: [8.0, 9.0])
    relative_differences: list[float] = field(
        default_factory=lambda: [step * 0.03 for step in range(11)]
    )
    average_retention_ratios: list[float] = field(
        default_factory=lambda: [step * 0.05 for step in range(6, 9)]
    )
    output_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        # 不传输出目录时，就放到 outputs/visualize_owl_onetimepruning/<model_name>
        if self.output_dir is None:
            self.output_dir = (
                self.experiment.paths.output_root
                / "visualize_owl_onetimepruning"
                / self.experiment.model_name
            )
        else:
            self.output_dir = Path(self.output_dir)
