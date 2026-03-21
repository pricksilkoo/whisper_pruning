from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


def create_running_stats():
    """
    创建累计统计容器。
    """
    return {"sum": None, "sq_sum": None, "count": 0}


def update_feature_stats(stats, tensor):
    """
    按特征维累加张量统计量。

    参数:
    - stats: 由 `create_running_stats` 创建的统计字典。
    - tensor: 待累计的输入张量，最后一维会被视为特征维。
    """
    tensor = tensor.detach().float()
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.reshape(-1, tensor.shape[-1])

    tensor_sum = tensor.sum(dim=0)
    tensor_sq_sum = tensor.square().sum(dim=0)

    if stats["sum"] is None:
        stats["sum"] = tensor_sum
        stats["sq_sum"] = tensor_sq_sum
    else:
        stats["sum"] += tensor_sum
        stats["sq_sum"] += tensor_sq_sum

    stats["count"] += tensor.shape[0]


def update_tensor_stats(stats, tensor):
    """
    对整块张量做逐元素累计统计。

    参数:
    - stats: 由 `create_running_stats` 创建的统计字典。
    - tensor: 待累计的张量，形状通常与某层权重一致。
    """
    tensor = tensor.detach().float()
    tensor_sq = tensor.square()

    if stats["sum"] is None:
        stats["sum"] = tensor.clone()
        stats["sq_sum"] = tensor_sq
    else:
        stats["sum"] += tensor
        stats["sq_sum"] += tensor_sq

    stats["count"] += 1


def stats_mean(stats) -> Optional[torch.Tensor]:
    """
    从统计字典里恢复均值张量。

    参数:
    - stats: 单层统计字典。
    """
    if not stats or stats["sum"] is None or stats["count"] <= 0:
        return None
    return stats["sum"] / stats["count"]


def stats_rms(stats) -> Optional[torch.Tensor]:
    """
    从统计字典里恢复 RMS 张量。

    参数:
    - stats: 单层统计字典。
    """
    if not stats or stats["sq_sum"] is None or stats["count"] <= 0:
        return None
    return torch.sqrt(stats["sq_sum"] / stats["count"])


def stats_dict_mean(stats_dict):
    """
    把一组层统计转换成均值字典。

    参数:
    - stats_dict: `{layer_name: stats}` 形式的统计字典。
    """
    return {
        name: stats_mean(stats)
        for name, stats in stats_dict.items()
    }


def stats_dict_rms(stats_dict):
    """
    把一组层统计转换成 RMS 字典。

    参数:
    - stats_dict: `{layer_name: stats}` 形式的统计字典。
    """
    return {
        name: stats_rms(stats)
        for name, stats in stats_dict.items()
    }


class SignalCollector:
    """
    统一收集线性层的:
    - weights
    - activations
    - gradients
    """

    def __init__(
        self,
        model,
        dataloader,
        device,
        dtype=None,
        include_output_projection=False,
    ) -> None:
        """
        参数:
        - model: 待分析的模型。
        - dataloader: 用来跑前向或反向的数据。
        - device: 模型运行设备。
        - dtype: 输入特征送入模型时使用的 dtype。
        - include_output_projection: 是否把 `proj_out` 层也纳入统计。
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.dtype = dtype or next(model.parameters()).dtype
        self.include_output_projection = include_output_projection
        self.target_layers = self._find_target_layers()

    def _find_target_layers(self) -> Dict[str, nn.Linear]:
        layers = {}
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not self.include_output_projection and "proj_out" in name:
                continue
            layers[name] = module
        return layers

    def _build_activation_hook(self, layer_name, activations):
        def hook(module, inputs, output):
            del module, output
            update_feature_stats(activations[layer_name], inputs[0])

        return hook

    def _snapshot_weights(self):
        return {
            name: module.weight.detach().clone()
            for name, module in self.target_layers.items()
        }

    def _move_batch(self, batch):
        input_features = batch["input_features"].to(self.device, dtype=self.dtype)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        return input_features, labels, attention_mask

    def collect(
        self,
        collect_activations=True,
        collect_gradients=False,
        max_batches=None,
        log=True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, dict], Dict[str, dict]]:
        """
        统一收集权重、激活统计、梯度统计。

        参数:
        - collect_activations: 是否收集输入激活统计。
        - collect_gradients: 是否通过 backward 收集权重梯度统计。
        - max_batches: 最多处理多少个 batch；`None` 表示跑完整个 dataloader。
        - log: 是否显示进度条。
        """
        weights = self._snapshot_weights()
        activations = (
            {name: create_running_stats() for name in self.target_layers}
            if collect_activations
            else {}
        )
        gradients = (
            {name: create_running_stats() for name in self.target_layers}
            if collect_gradients
            else {}
        )

        if not collect_activations and not collect_gradients:
            return weights, activations, gradients

        hooks = []
        if collect_activations:
            for name, module in self.target_layers.items():
                hooks.append(module.register_forward_hook(self._build_activation_hook(name, activations)))

        iterator = self.dataloader
        if log:
            iterator = tqdm(iterator, desc="正在收集层信号")

        self.model.eval()
        try:
            if collect_gradients:
                for step, batch in enumerate(iterator):
                    if max_batches is not None and step >= max_batches:
                        break

                    self.model.zero_grad(set_to_none=True)
                    input_features, labels, attention_mask = self._move_batch(batch)
                    outputs = self.model(
                        input_features=input_features,
                        labels=labels,
                        attention_mask=attention_mask,
                    )
                    outputs.loss.backward()

                    for name, module in self.target_layers.items():
                        if module.weight.grad is not None:
                            update_tensor_stats(gradients[name], module.weight.grad)

                    del outputs
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                with torch.inference_mode():
                    for step, batch in enumerate(iterator):
                        if max_batches is not None and step >= max_batches:
                            break

                        input_features, labels, attention_mask = self._move_batch(batch)
                        self.model(
                            input_features=input_features,
                            labels=labels,
                            attention_mask=attention_mask,
                        )

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        finally:
            for hook in hooks:
                hook.remove()
            self.model.zero_grad(set_to_none=True)

        return weights, activations, gradients


__all__ = [
    "SignalCollector",
    "create_running_stats",
    "stats_dict_mean",
    "stats_dict_rms",
    "stats_mean",
    "stats_rms",
    "update_feature_stats",
    "update_tensor_stats",
]
