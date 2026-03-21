import torch

from utils.signal_collector import stats_rms


class Scorer:
    """
    非均匀剪枝打分器。

    约定所有打分函数都接收:
    - weights
    - activations
    - gradients

    这样你后面新增方法时，接口可以保持稳定。
    """

    def __init__(self):
        self.scores = {}
        self.retention_ratio = {}
        self.custom_methods = {}

    def register(self, name, fn):
        """
        注册自定义打分函数。

        参数:
        - name: 方法名，后续通过 `compute(method=name)` 调用。
        - fn: 自定义函数，签名建议接收 `weights/activations/gradients`。
        """
        self.custom_methods[name.lower()] = fn

    def _resolve_activations(self, activations=None, activations_stats=None):
        return activations if activations is not None else activations_stats

    def _finalize_scores(
        self,
        scores,
        weights,
        relative_difference=0.2,
        average_retention_ratio=0.6,
        retention_ratio=None,
    ):
        self.scores = dict(scores or {})
        if retention_ratio is not None:
            self.retention_ratio = dict(retention_ratio)
            return self.scores, self.retention_ratio

        counts = {name: weight.numel() for name, weight in weights.items()}
        self.retention_ratio = self.scores_to_retention_ratios(
            scores=self.scores,
            counts=counts,
            relative_difference=relative_difference,
            average_retention_ratio=average_retention_ratio,
        )
        return self.scores, self.retention_ratio

    def compute(
        self,
        method,
        weights,
        activations=None,
        gradients=None,
        activations_stats=None,
        level=7,
        relative_difference=0.2,
        average_retention_ratio=0.6,
        **kwargs,
    ):
        """
        统一调度打分方法。

        参数:
        - method: 打分方法名，当前内置 `owl` 和 `cv`。
        - weights: `{layer_name: weight}` 权重字典。
        - activations: `{layer_name: stats}` 激活统计字典。
        - gradients: `{layer_name: stats}` 梯度统计字典，供后续自定义方法扩展。
        - activations_stats: `activations` 的兼容别名。
        - level: OWL 方法的阈值参数。
        - relative_difference: 各层保留率允许拉开的相对差值。
        - average_retention_ratio: 期望的平均保留率。
        - **kwargs: 传给具体打分函数的额外参数。
        """
        activations = self._resolve_activations(
            activations=activations,
            activations_stats=activations_stats,
        )
        method = method.lower()

        if method in self.custom_methods:
            result = self.custom_methods[method](
                weights=weights,
                activations=activations,
                gradients=gradients,
                **kwargs,
            )
            if isinstance(result, tuple) and len(result) == 2:
                return self._finalize_scores(
                    scores=result[0],
                    retention_ratio=result[1],
                    weights=weights,
                    relative_difference=relative_difference,
                    average_retention_ratio=average_retention_ratio,
                )
            return self._finalize_scores(
                scores=result,
                weights=weights,
                relative_difference=relative_difference,
                average_retention_ratio=average_retention_ratio,
            )

        if method == "owl":
            return self.owl(
                weights=weights,
                activations=activations,
                gradients=gradients,
                level=level,
                relative_difference=relative_difference,
                average_retention_ratio=average_retention_ratio,
            )
        if method == "cv":
            scores, _ = self.cv(
                weights=weights,
                activations=activations,
                gradients=gradients,
            )
            return self._finalize_scores(
                scores=scores,
                weights=weights,
                relative_difference=relative_difference,
                average_retention_ratio=average_retention_ratio,
            )

        raise ValueError(f"不支持的打分方法: {method}")

    def scores_to_retention_ratios(
        self,
        scores,
        counts,
        relative_difference=0.2,
        average_retention_ratio=0.6,
    ):
        """
        把层分数映射成层保留率。

        参数:
        - scores: `{layer_name: score}` 分数字典。
        - counts: `{layer_name: param_count}` 参数量字典。
        - relative_difference: 层间差异幅度。
        - average_retention_ratio: 目标平均保留率。
        """
        self.retention_ratio.clear()
        if not scores:
            return self.retention_ratio

        if not (
            0 <= relative_difference <= 1
            and 0 <= average_retention_ratio <= 1
            and 0 <= average_retention_ratio - relative_difference
            and average_retention_ratio + relative_difference <= 1
        ):
            raise ValueError("rd 或 avg 值不在规定范围内")

        score_values = list(scores.values())
        score_max = max(score_values)
        score_min = min(score_values)
        if score_max == score_min:
            self.retention_ratio = {
                name: float(average_retention_ratio) for name in scores
            }
            return self.retention_ratio

        weighted_delta_sum = sum(
            (scores[name] - score_min) * counts.get(name, 1) for name in scores
        )
        total_count = sum(counts.get(name, 1) for name in scores)
        base = average_retention_ratio - (
            relative_difference * weighted_delta_sum
        ) / ((score_max - score_min) * total_count)

        for name, score in scores.items():
            ratio = base + relative_difference * (score - score_min) / (score_max - score_min)
            self.retention_ratio[name] = float(min(max(ratio, 0.0), 1.0))

        return self.retention_ratio

    def cv(
        self,
        weights,
        activations=None,
        gradients=None,
        activations_stats=None,
    ):
        """
        基于变异系数的层打分。

        参数:
        - weights: `{layer_name: weight}` 权重字典。
        - activations: `{layer_name: stats}` 激活统计字典。
        - gradients: 预留参数，当前不使用。
        - activations_stats: `activations` 的兼容别名。
        """
        del gradients
        activations = self._resolve_activations(
            activations=activations,
            activations_stats=activations_stats,
        )
        if not activations:
            raise ValueError("CV 打分需要 activations。")

        self.scores.clear()
        self.retention_ratio.clear()
        for name, weight in weights.items():
            activation_rms = stats_rms(activations[name])
            if activation_rms is None:
                raise ValueError(f"层 {name} 缺少可用的激活统计。")
            signal = torch.abs(weight).float() * activation_rms.unsqueeze(0)
            signal_mean = signal.mean()
            signal_std = signal.std()
            self.scores[name] = (signal_std / (signal_mean + 1e-9)).item()

        return self.scores, self.retention_ratio

    def owl(
        self,
        weights,
        activations=None,
        gradients=None,
        activations_stats=None,
        level=5,
        relative_difference=0.2,
        average_retention_ratio=0.6,
    ):
        """
        基于 OWL 异常值占比的层打分。

        参数:
        - weights: `{layer_name: weight}` 权重字典。
        - activations: `{layer_name: stats}` 激活统计字典。
        - gradients: 预留参数，当前不使用。
        - activations_stats: `activations` 的兼容别名。
        - level: 判断异常值的阈值。
        - relative_difference: 层间保留率差异幅度。
        - average_retention_ratio: 目标平均保留率。
        """
        del gradients
        activations = self._resolve_activations(
            activations=activations,
            activations_stats=activations_stats,
        )
        if not activations:
            raise ValueError("OWL 打分需要 activations。")

        self.scores.clear()
        self.retention_ratio.clear()
        if not (
            0 <= level
            and 0 <= relative_difference <= 1
            and 0 <= average_retention_ratio <= 1
            and 0 <= average_retention_ratio - relative_difference
            and average_retention_ratio + relative_difference <= 1
        ):
            raise ValueError("level 或 rd 或 avg 值不在规定范围内")

        counts = {}
        for name, weight in weights.items():
            activation_rms = stats_rms(activations[name])
            if activation_rms is None:
                raise ValueError(f"层 {name} 缺少可用的激活统计。")

            signal = torch.abs(weight).float() * activation_rms.unsqueeze(0)
            signal_mean = signal.mean()
            normalized_signal = signal / (signal_mean + 1e-9)
            mask = (normalized_signal >= level).float()
            counts[name] = signal.numel()
            self.scores[name] = mask.mean().item()

        self.retention_ratio = self.scores_to_retention_ratios(
            scores=self.scores,
            counts=counts,
            relative_difference=relative_difference,
            average_retention_ratio=average_retention_ratio,
        )
        return self.scores, self.retention_ratio


__all__ = ["Scorer"]
