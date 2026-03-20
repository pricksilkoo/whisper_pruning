import torch
import torch.nn as nn
from tqdm import tqdm

class WAprofiler:
    def __init__(self, model, dataloader, device, dtype=torch.float32) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.profiler_layers = {}
        self.hooks = []
        self.weights = {}
        self.activations_stats = {}
        self.dtype=dtype

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if "proj_out" not in name: #输出层是否需要
                    self.profiler_layers[name] = module

    def _get_hook(self, layer_name):
        def hook(module, input, output):
            x = input[0].detach().float()
            x = x.view(-1, x.shape[-1])
            self.activations_stats[layer_name]["sum"] += x.sum(dim=0)
            self.activations_stats[layer_name]["sq_sum"] += (x ** 2).sum(dim=0)
            self.activations_stats[layer_name]["count"] += x.shape[0]
        return hook

    def getWA(self):
        """
        启动前向传播体检循环。支持多次调用，每次调用都会重新读取当前模型的最新权重和激活值。
        Returns:
        Tuple[dict, dict]: (weights_dict, activations_stats)
        - weights_dict: { 'layer_name': torch.Tensor(out, in) }
        - activations_stats: { 'layer_name': {'sum': T, 'sq_sum': T, 'count': int} }
        """
        self.weights = {}
        self.activations_stats = {}
        for name, module in self.profiler_layers.items():
            self.activations_stats[name] = {"sum": 0.0, "sq_sum": 0.0, "count": 0}
            self.weights[name] = module.weight.data.clone()

        for name, module in self.profiler_layers.items():
            self.hooks.append(module.register_forward_hook(self._get_hook(name)))
            
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="正在收集最新激活值"):
                input_features = batch["input_features"].to(self.device,dtype=self.dtype)
                labels = batch["labels"].to(self.device)
                self.model(input_features=input_features, labels=labels)

        for h in self.hooks:
            h.remove()
        self.hooks = []

        return self.weights, self.activations_stats