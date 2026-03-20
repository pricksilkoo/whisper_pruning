import torch

class Scorer:
    """
    这个类负责“给每一层打分”。

    你可以把它理解成:
    输入是 weights + activations_stats，
    输出是:
    - scores: 每层的原始分数
    - retention_ratio: 每层最终保留率
    """

    def __init__(self):
        self.scores={}
        self.retention_ratio={}

    def compute(
        self,
        method,
        weights,
        activations_stats,
        level=7,
        relative_difference=0.2,
        average_retention_ratio=0.6,
    ):
        """
        统一调度入口。

        现在只保留两种打分方法：
        - owl
        - cv

        现在统一写成 scorer.compute(method=...) 即可。
        """
        method = method.lower()
        if method == "owl":
            return self.owl(
                weights=weights,
                activations_stats=activations_stats,
                level=level,
                relative_difference=relative_difference,
                average_retention_ratio=average_retention_ratio,
            )
        if method == "cv":
            scores, _ = self.cv(weights, activations_stats)
        else:
            raise ValueError(f"不支持的打分方法: {method}")

        counts = {name: weight.numel() for name, weight in weights.items()}
        retention_ratio = self.scores_to_retention_ratios(
            scores=scores,
            counts=counts,
            relative_difference=relative_difference,
            average_retention_ratio=average_retention_ratio,
        )
        return scores, retention_ratio

    def scores_to_retention_ratios(
        self,
        scores,
        counts,
        relative_difference=0.2,
        average_retention_ratio=0.6,
    ):
        """
        把原始 scores 映射成 retention_ratio。

        average_retention_ratio 可以理解成“总体平均保留率想控制在多少”。
        relative_difference 可以理解成“不同层之间允许拉开多大差距”。
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
            # 所有层分数都一样时，说明无法区分层间重要性。
            # 这时最稳妥的做法是所有层都给同样的保留率。
            self.retention_ratio = {
                name: float(average_retention_ratio) for name in scores
            }
            return self.retention_ratio

        # 这里是在算一个“基准保留率” base，
        # 让最后所有层的平均保留率仍然接近 average_retention_ratio。
        weighted_delta_sum = sum(
            (scores[name] - score_min) * counts.get(name, 1) for name in scores
        )
        total_count = sum(counts.get(name, 1) for name in scores)
        base = average_retention_ratio - (
            relative_difference * weighted_delta_sum
        ) / ((score_max - score_min) * total_count)

        for name, score in scores.items():
            # 分数越高，保留率越高。
            ratio = base + relative_difference * (score - score_min) / (score_max - score_min)
            self.retention_ratio[name] = float(min(max(ratio, 0.0), 1.0))

        return self.retention_ratio

    def cv(self,weights,activations_stats):
        """
            基于wanda计算变异系数(Coefficient of Variation)的打分方法
        """
        self.scores.clear()
        self.retention_ratio.clear()
        for name,weight in weights.items():
            x_l2=(activations_stats[name]["sq_sum"]/activations_stats[name]["count"]).sqrt()
            k = torch.abs(weight) * x_l2
            k_mean=k.mean()
            k_std=k.std()
            # CV = 标准差 / 均值，用来衡量这一层分布的离散程度。
            k_cv=k_std/(k_mean + 1e-9)
            self.scores[name] = k_cv.item()
        return self.scores, self.retention_ratio
    
    def owl(self,weights,activations_stats,level=5,relative_difference=0.2,average_retention_ratio=0.6):
        """
            基于owl的层级打分方法
            level为系数owl实验结果是5和7比较好,寻找异常值的系数
        """
        self.scores.clear()
        self.retention_ratio.clear()
        if not (0 <= level and 0 <= relative_difference <=1
                and 0 <= average_retention_ratio <=1 
                and 0 <= average_retention_ratio-relative_difference
                and average_retention_ratio+relative_difference <=1):
            raise ValueError(f"level或rd或avg值不在规定范围内")
        
        counts = {}
        for name,weight in weights.items():
            #计算owl的分数
            x_l2=(activations_stats[name]["sq_sum"]/activations_stats[name]["count"]).sqrt()
            k = torch.abs(weight) * x_l2
            k_mean=k.mean()
            s =  k / (k_mean + 1e-9)

            # OWL 的核心直觉:
            # 如果一层里“明显高于均值”的权重占比更高，
            # 那这层更值得保留。
            mask=(s>=level).float()
            counts[name] = k.numel()
            self.scores[name] = mask.mean().item()

        self.retention_ratio = self.scores_to_retention_ratios(
            scores=self.scores,
            counts=counts,
            relative_difference=relative_difference,
            average_retention_ratio=average_retention_ratio,
        )

        return self.scores,self.retention_ratio    
