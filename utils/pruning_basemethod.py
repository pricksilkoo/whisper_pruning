import torch

from utils.signal_collector import stats_rms


class PruningBaseMethod:
    """
    统一封装底层剪枝方法。

    当前内置:
    - wanda
    - sparsegpt
    """

    def __init__(self):
        self.masks = {}
        self.pruned_weights = {}
        self.last_method = None

    def _resolve_sparsity(self, sparsity, layer_name):
        if isinstance(sparsity, dict):
            current_sparsity = sparsity.get(layer_name, 0.0)
        else:
            current_sparsity = float(sparsity)
        return float(min(max(current_sparsity, 0.0), 1.0))

    def _apply_metric_pruning(self, weights, metrics, sparsity):
        self.masks = {}
        self.pruned_weights = {}

        for name, weight in weights.items():
            current_sparsity = self._resolve_sparsity(sparsity, name)
            prune_num = min(weight.shape[-1], int(weight.shape[-1] * current_sparsity))

            if prune_num <= 0:
                mask = torch.ones_like(weight, dtype=torch.bool)
            elif prune_num >= weight.shape[-1]:
                mask = torch.zeros_like(weight, dtype=torch.bool)
            else:
                _, bottomk_indices = torch.topk(
                    metrics[name],
                    k=prune_num,
                    dim=-1,
                    largest=False,
                )
                mask = torch.ones_like(weight, dtype=torch.bool)
                mask.scatter_(
                    dim=-1,
                    index=bottomk_indices,
                    src=torch.zeros_like(bottomk_indices, dtype=torch.bool),
                )

            self.masks[name] = mask
            self.pruned_weights[name] = weight * mask

        return self.pruned_weights, self.masks

    def prune(
        self,
        method,
        weights,
        activations=None,
        gradients=None,
        sparsity=0.5,
        **kwargs,
    ):
        """
        统一调度剪枝方法。

        参数:
        - method: 剪枝方法名，当前支持 `wanda` 和 `sparsegpt`。
        - weights: `{layer_name: weight}` 权重字典。
        - activations: `{layer_name: stats}` 激活统计字典。
        - gradients: `{layer_name: stats}` 梯度统计字典，当前内置方法默认不用。
        - sparsity: 全局稀疏度 float，或逐层稀疏度字典。
        - **kwargs: 传给具体剪枝方法的额外参数。
        """
        method = method.lower()
        if method == "wanda":
            return self.wanda(
                weights=weights,
                activations=activations,
                gradients=gradients,
                sparsity=sparsity,
                **kwargs,
            )
        if method in {"sparsegpt", "sparesgpt"}:
            return self.sparsegpt(
                weights=weights,
                activations=activations,
                gradients=gradients,
                sparsity=sparsity,
                **kwargs,
            )
        raise ValueError(f"不支持的剪枝方法: {method}")

    def apply_to_model(self, model, log=True):
        """
        把剪枝后的权重写回模型。

        参数:
        - model: 目标模型。
        - log: 是否打印剪枝报告。
        """
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in self.pruned_weights:
                    module.weight.data.copy_(self.pruned_weights[name])

        if log:
            self.print_report(model)

    def print_report(self, model):
        """
        打印当前剪枝结果的参数统计。

        参数:
        - model: 已应用剪枝结果的模型。
        """
        total_model_params = sum(parameter.numel() for parameter in model.parameters())
        linear_params_total = 0
        total_pruned_zeros = 0

        for mask in self.masks.values():
            layer_total = mask.numel()
            layer_retained = mask.sum().item()
            linear_params_total += layer_total
            total_pruned_zeros += layer_total - layer_retained

        global_sparsity = total_pruned_zeros / total_model_params if total_model_params else 0.0
        linear_sparsity = total_pruned_zeros / linear_params_total if linear_params_total else 0.0

        print("mask已经挂在权重上了")
        print("\n" + "=" * 40)
        print("剪枝报告")
        print("=" * 40)
        print(f"Whisper 全模型总参数量: {total_model_params:,}")
        print(f"其中参与剪枝的线性层参数量: {linear_params_total:,}")
        print(f"实际被物理抹除的参数量: {total_pruned_zeros:,}")
        print("-" * 40)
        print(f"线性层内部稀疏度: {linear_sparsity:.2%}")
        print(f"整网全局真稀疏度: {global_sparsity:.2%}")
        print("=" * 40 + "\n")

    def wanda(
        self,
        weights,
        activations=None,
        gradients=None,
        sparsity=0.5,
    ):
        """
        Wanda 非结构化剪枝。

        参数:
        - weights: `{layer_name: weight}` 权重字典。
        - activations: `{layer_name: stats}` 激活统计字典。
        - gradients: 预留参数，当前 Wanda 不使用。
        - sparsity: 全局稀疏度 float，或逐层稀疏度字典。
        """
        del gradients
        if not activations:
            raise ValueError("Wanda 需要输入激活统计 activations。")

        metrics = {}
        for name, weight in weights.items():
            activation_rms = stats_rms(activations[name])
            if activation_rms is None:
                raise ValueError(f"层 {name} 缺少可用的激活统计。")
            metrics[name] = torch.abs(weight).float() * activation_rms.unsqueeze(0)

        self.last_method = "wanda"
        return self._apply_metric_pruning(weights, metrics, sparsity)

    def sparsegpt(
        self,
        weights,
        activations=None,
        gradients=None,
        sparsity=0.5,
        damping=0.01,
        eps=1e-8,
    ):
        """
        这里实现的是对角 Hessian 近似版本的 SparseGPT。

        真正的 full Hessian 在 Whisper 这类大模型上线性层里非常吃显存，
        所以这里保留 SparseGPT 风格的二阶打分接口，但只使用输入激活的对角二阶矩。

        参数:
        - weights: `{layer_name: weight}` 权重字典。
        - activations: `{layer_name: stats}` 激活统计字典。
        - gradients: 预留参数，当前实现不使用。
        - sparsity: 全局稀疏度 float，或逐层稀疏度字典。
        - damping: 对角近似里的阻尼项。
        - eps: 数值稳定项，避免除零。
        """
        del gradients
        if not activations:
            raise ValueError("SparseGPT 需要输入激活统计 activations。")

        metrics = {}
        for name, weight in weights.items():
            activation_rms = stats_rms(activations[name])
            if activation_rms is None:
                raise ValueError(f"层 {name} 缺少可用的激活统计。")

            hessian_diag = activation_rms.square().clamp_min(eps) + damping
            inv_hessian_diag = torch.reciprocal(hessian_diag)
            metrics[name] = weight.float().square() / (inv_hessian_diag.unsqueeze(0).square() + eps)

        self.last_method = "sparsegpt"
        return self._apply_metric_pruning(weights, metrics, sparsity)

    def wanda_unstructured_pruning(self, weights, stats=None, sparsity=0.5, **kwargs):
        """
        Wanda 的兼容调用入口。

        参数:
        - weights: `{layer_name: weight}` 权重字典。
        - stats: 激活统计字典。
        - sparsity: 全局稀疏度 float，或逐层稀疏度字典。
        - **kwargs: 传给 `wanda` 的额外参数。
        """
        activations = kwargs.pop("activations", None) or stats
        return self.wanda(
            weights=weights,
            activations=activations,
            sparsity=sparsity,
            **kwargs,
        )

    def sparsegpt_unstructured_pruning(self, weights, stats=None, sparsity=0.5, **kwargs):
        """
        SparseGPT 的兼容调用入口。

        参数:
        - weights: `{layer_name: weight}` 权重字典。
        - stats: 激活统计字典。
        - sparsity: 全局稀疏度 float，或逐层稀疏度字典。
        - **kwargs: 传给 `sparsegpt` 的额外参数。
        """
        activations = kwargs.pop("activations", None) or stats
        return self.sparsegpt(
            weights=weights,
            activations=activations,
            sparsity=sparsity,
            **kwargs,
        )


__all__ = ["PruningBaseMethod"]
