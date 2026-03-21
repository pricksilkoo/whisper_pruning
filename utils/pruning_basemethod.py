import math

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.signal_collector import stats_rms


class _SparseGPTKernel:
    """
    基于 SparseGPT 官方实现改写的单层剪枝器。
    """

    def __init__(self, layer):
        self.layer = layer
        self.device = layer.weight.device
        weight = layer.weight.data.detach().clone().float()
        self.rows, self.columns = weight.shape
        self.hessian = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, layer_input):
        if len(layer_input.shape) == 2:
            layer_input = layer_input.unsqueeze(0)

        batch_size = layer_input.shape[0]
        if len(layer_input.shape) == 3:
            layer_input = layer_input.reshape((-1, layer_input.shape[-1]))

        layer_input = layer_input.t().float()
        self.hessian *= self.nsamples / (self.nsamples + batch_size)
        self.nsamples += batch_size
        layer_input = math.sqrt(2.0 / self.nsamples) * layer_input
        self.hessian += layer_input.matmul(layer_input.t())

    def faster_prune(self, sparsity, blocksize=128, percdamp=0.01):
        weight = self.layer.weight.data.detach().clone().float()
        hessian = self.hessian
        sparsity = float(min(max(sparsity, 0.0), 1.0))

        dead = torch.diag(hessian) == 0
        hessian[dead, dead] = 1
        weight[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(hessian))
        diagonal = torch.arange(self.columns, device=self.device)
        hessian[diagonal, diagonal] += damp

        hessian = torch.linalg.cholesky(hessian)
        hessian = torch.cholesky_inverse(hessian)
        hessian = torch.linalg.cholesky(hessian, upper=True)
        hessian_inv = hessian

        keep_mask = torch.ones_like(weight, dtype=torch.bool)

        for start in range(0, self.columns, blocksize):
            end = min(start + blocksize, self.columns)
            count = end - start

            weight_block = weight[:, start:end].clone()
            pruned_block = torch.zeros_like(weight_block)
            error_block = torch.zeros_like(weight_block)
            hessian_inv_block = hessian_inv[start:end, start:end]

            if sparsity <= 0:
                pruned_mask = torch.zeros_like(weight_block, dtype=torch.bool)
            elif sparsity >= 1:
                pruned_mask = torch.ones_like(weight_block, dtype=torch.bool)
            else:
                metric = weight_block.square() / (
                    torch.diag(hessian_inv_block).reshape((1, -1)).square()
                )
                threshold_index = min(max(int(metric.numel() * sparsity), 1), metric.numel()) - 1
                threshold = torch.sort(metric.flatten())[0][threshold_index]
                pruned_mask = metric <= threshold
            keep_mask[:, start:end] = ~pruned_mask

            for offset in range(count):
                column_weight = weight_block[:, offset]
                diagonal_value = hessian_inv_block[offset, offset]

                pruned_column = column_weight.clone()
                pruned_column[pruned_mask[:, offset]] = 0

                pruned_block[:, offset] = pruned_column
                error = (column_weight - pruned_column) / diagonal_value
                error_block[:, offset] = error

                weight_block[:, offset:] -= error.unsqueeze(1).matmul(
                    hessian_inv_block[offset, offset:].unsqueeze(0)
                )

            weight[:, start:end] = pruned_block
            if end < self.columns:
                weight[:, end:] -= error_block.matmul(hessian_inv[start:end, end:])

        self.layer.weight.data.copy_(weight.to(self.layer.weight.data.dtype))
        return self.layer.weight.data.detach().clone(), keep_mask

    def free(self):
        self.hessian = None
        torch.cuda.empty_cache()


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
        **kwargs,
    ):
        """
        Wanda 非结构化剪枝。

        参数:
        - weights: `{layer_name: weight}` 权重字典。
        - activations: `{layer_name: stats}` 激活统计字典。
        - gradients: 预留参数，当前 Wanda 不使用。
        - sparsity: 全局稀疏度 float，或逐层稀疏度字典。
        """
        del kwargs
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
        model=None,
        dataloader=None,
        device=None,
        dtype=None,
        blocksize=128,
        include_output_projection=False,
        max_batches=None,
        log=True,
    ):
        """
        论文风格的 SparseGPT。

        这里按层收集输入二阶矩 Hessian 近似，再做块式剪枝和误差补偿。
        为了控制显存，当前实现是“逐层重新跑 calibration dataloader”。

        参数:
        - weights: `{layer_name: weight}` 权重字典。
        - activations: 预留参数，当前实现不使用。
        - gradients: 预留参数，当前实现不使用。
        - sparsity: 全局稀疏度 float，或逐层稀疏度字典。
        - damping: Hessian 对角阻尼比例。
        - eps: 数值稳定项，保留该参数位以兼容调用。
        - model: 待剪枝模型。
        - dataloader: calibration dataloader。
        - device: 模型运行设备。
        - dtype: 输入特征送入模型时使用的 dtype。
        - blocksize: 块式剪枝的列块大小。
        - include_output_projection: 是否把 `proj_out` 也纳入剪枝。
        - max_batches: 最多使用多少个 calibration batch。
        - log: 是否打印进度。
        """
        del activations, gradients, eps
        if model is None or dataloader is None:
            raise ValueError("SparseGPT 需要额外传入 model 和 dataloader。")

        if device is None:
            device = next(model.parameters()).device
        if dtype is None:
            dtype = next(model.parameters()).dtype

        target_layers = []
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not include_output_projection and "proj_out" in name:
                continue
            if name not in weights:
                continue
            target_layers.append((name, module))

        self.masks = {}
        self.pruned_weights = {}

        iterator = target_layers
        if log:
            iterator = tqdm(iterator, desc="正在执行 SparseGPT")

        model.eval()
        for layer_name, layer in iterator:
            kernel = _SparseGPTKernel(layer)

            def hook(module, layer_inputs, layer_outputs):
                del module, layer_outputs
                kernel.add_batch(layer_inputs[0])

            handle = layer.register_forward_hook(hook)
            try:
                with torch.inference_mode():
                    for step, batch in enumerate(dataloader):
                        if max_batches is not None and step >= max_batches:
                            break

                        input_features = batch["input_features"].to(device, dtype=dtype)
                        labels = batch["labels"].to(device)
                        attention_mask = batch.get("attention_mask")
                        if attention_mask is not None:
                            attention_mask = attention_mask.to(device)

                        model(
                            input_features=input_features,
                            labels=labels,
                            attention_mask=attention_mask,
                        )
            finally:
                handle.remove()

            current_sparsity = self._resolve_sparsity(sparsity, layer_name)
            pruned_weight, keep_mask = kernel.faster_prune(
                sparsity=current_sparsity,
                blocksize=blocksize,
                percdamp=damping,
            )
            self.pruned_weights[layer_name] = pruned_weight
            self.masks[layer_name] = keep_mask
            kernel.free()

        self.last_method = "sparsegpt"
        return self.pruned_weights, self.masks

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
