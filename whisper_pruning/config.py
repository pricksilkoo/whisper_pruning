from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_root() -> Path:
    return PROJECT_ROOT


@dataclass(slots=True)
class ProjectPaths:
    project_root: Path = field(default_factory=_project_root)
    model_root: Optional[Path] = None
    data_root: Optional[Path] = None
    output_root: Optional[Path] = None

    def __post_init__(self) -> None:
        if self.model_root is None:
            self.model_root = self.project_root / "models"
        if self.data_root is None:
            self.data_root = self.project_root / "data" / "fleurs_full"
        if self.output_root is None:
            self.output_root = self.project_root / "outputs"

    def model_path(self, model_name: str) -> Path:
        return self.model_root / model_name

    def dataset_path(self, dataset_name: str) -> Path:
        return self.data_root / dataset_name


@dataclass(slots=True)
class ExperimentConfig:
    model_name: str
    dataset_name: str
    dtype: str = "float16"
    device: Optional[str] = None
    language: Optional[str] = None
    task: str = "transcribe"
    paths: ProjectPaths = field(default_factory=ProjectPaths)

    def __post_init__(self) -> None:
        if self.language is None:
            self.language = self.dataset_name

    @property
    def model_path(self) -> Path:
        return self.paths.model_path(self.model_name)

    @property
    def dataset_path(self) -> Path:
        return self.paths.dataset_path(self.dataset_name)


@dataclass(slots=True)
class DataLoaderConfig:
    split: str
    batch_size: int
    num_samples: Optional[int] = None
    shuffle: Optional[bool] = None

    @property
    def should_shuffle(self) -> bool:
        if self.shuffle is not None:
            return self.shuffle
        return self.split == "train"


@dataclass(slots=True)
class ScoringConfig:
    method: str = "owl"
    level: float = 7.0
    relative_difference: float = 0.0
    average_retention_ratio: float = 0.4


@dataclass(slots=True)
class PruningConfig:
    method: str = "wanda_unstructured"
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    uniform_sparsity: Optional[float] = None
    n: int = 2
    m: int = 4


@dataclass(slots=True)
class EvaluationRunConfig:
    experiment: ExperimentConfig
    data: DataLoaderConfig


@dataclass(slots=True)
class ProfileRunConfig:
    experiment: ExperimentConfig
    data: DataLoaderConfig


@dataclass(slots=True)
class OneShotPruningRunConfig:
    experiment: ExperimentConfig
    profile_data: DataLoaderConfig
    eval_data: DataLoaderConfig
    pruning: PruningConfig = field(default_factory=PruningConfig)


@dataclass(slots=True)
class OwlSweepConfig:
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
        if self.output_dir is None:
            self.output_dir = (
                self.experiment.paths.output_root
                / "visualize_owl_onetimepruning"
                / self.experiment.model_name
            )
        else:
            self.output_dir = Path(self.output_dir)
