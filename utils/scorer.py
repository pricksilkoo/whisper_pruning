import torch

class Scorer:
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
        method = method.lower()
        if method == "owl":
            return self.owl(
                weights=weights,
                activations_stats=activations_stats,
                level=level,
                relative_difference=relative_difference,
                average_retention_ratio=average_retention_ratio,
            )
        if method == "hhr1":
            scores = self.hhr1(weights, activations_stats, level=level)
        elif method == "cv":
            scores, _ = self.cv(weights, activations_stats)
        elif method == "mean":
            scores, _ = self.mean(weights, activations_stats)
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

    def hhr1(self,weights,activations_stats,level=0.1):
        """
            基于wanda的层级打分方法
            level表示在该层的相对重要性基线,即重要程度小于level的权重越多分数越低
            其中level范围(0,1)
        """
        self.scores.clear()
        if not (0 <= level <= 1):
            raise ValueError(f"level值不在规定范围内")
        
        for name,weight in weights.items():
            x_l2=activations_stats[name]["sq_sum"].sqrt()
            k = torch.abs(weight) * x_l2
            k_biweight_mean=self._tukey_biweight_mean(k,c=0.01)
            s =  k / (k_biweight_mean + 1e-9)
            mask=(s>=level).float()
            self.scores[name] = mask.mean().sqrt().item()
        return self.scores
    

    def _tukey_biweight_mean(self, x, c=4.685):
        """
        使用 Tukey Biweight 算法计算鲁棒均值
        x: 输入的 Tensor (例如你代码中的 k)
        c: 阈值常数，c越大越包容，c越小越严格
        """
        if x.numel() == 0:
            return torch.tensor(0.0)
        
        x_flat = x.view(-1)
        median = x_flat.median()
        # MAD = median(|x_i - median|)
        abs_deviation = torch.abs(x_flat - median)
        mad = abs_deviation.median()
        s = mad * 1.4826 + 1e-6
        
        u = abs_deviation / (c * s)
        
        # Tukey 权重
        # 当 |u| <= 1 时，权重为 (1 - u^2)^2；否则为 0
        weights = torch.where(
            u <= 1, 
            (1 - u**2)**2, 
            torch.zeros_like(u)
        )
        
        biweight_mean = torch.sum(weights * x_flat) / (torch.sum(weights) + 1e-10)
        
        return biweight_mean

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
            k_cv=k_std/k_mean
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

    def mean(self,weights,activations_stats):
        """
            基于wanda计算平均值的打分方法
        """
        self.scores.clear()
        self.retention_ratio.clear()
        for name,weight in weights.items():
            x_l2=(activations_stats[name]["sq_sum"]/activations_stats[name]["count"]).sqrt()
            k = torch.abs(weight) * x_l2
            self.scores[name]=k.mean().item()
        return self.scores,self.retention_ratio   
