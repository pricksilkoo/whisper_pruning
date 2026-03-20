import torch
import torch.nn as nn
from tqdm import tqdm

class WAprofiler:
    """
    WA = Weights + Activations。

    这个类做的事情很单纯:
    1. 找出模型里所有需要观察的线性层
    2. 在这些层上挂 forward hook
    3. 跑一遍前向传播
    4. 收集每层的权重和激活统计量

    最后返回:
    - weights: 每层当前权重
    - stats:   每层输入激活的统计量
    """

    def __init__(self, model, dataloader, device, dtype=None) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.profiler_layers = {}
        self.hooks = []
        self.weights = {}
        self.activations_stats = {}
        self.dtype = dtype or next(model.parameters()).dtype

        # 这里只筛选 nn.Linear，是因为你当前的剪枝对象主要就是线性层。
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if "proj_out" not in name: # 输出层是否纳入剪枝，可按实验需要调整。
                    self.profiler_layers[name] = module

    def _get_hook(self, layer_name):
        """
        为某一层生成一个 hook 函数。

        hook 会在该层每次 forward 时自动触发，
        这里记录的是“该层输入张量”的统计量。
        """
        def hook(module, input, output):
            # input[0] 就是进入该线性层的输入。
            x = input[0].detach().float()

            # 把除了最后一维以外的其他维度全部展平，
            # 最终得到 [N, hidden_size]，方便按特征维统计。
            x = x.view(-1, x.shape[-1])
            layer_stats = self.activations_stats[layer_name]
            x_sum = x.sum(dim=0)
            x_sq_sum = (x ** 2).sum(dim=0)

            # 第一次见到该层时，直接初始化统计量；
            # 后续 batch 则继续累加。
            if layer_stats["sum"] is None:
                layer_stats["sum"] = x_sum
                layer_stats["sq_sum"] = x_sq_sum
            else:
                layer_stats["sum"] += x_sum
                layer_stats["sq_sum"] += x_sq_sum

            layer_stats["count"] += x.shape[0]
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
            # 这里存的是“激活统计量”，不是原始激活本身。
            # 这样更省内存，也足够支持你当前的 scoring 逻辑。
            self.activations_stats[name] = {"sum": None, "sq_sum": None, "count": 0}
            self.weights[name] = module.weight.data.clone()

        # 给每一层挂上 hook。
        for name, module in self.profiler_layers.items():
            self.hooks.append(module.register_forward_hook(self._get_hook(name)))
            
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="正在收集最新激活值"):
                input_features = batch["input_features"].to(self.device,dtype=self.dtype)
                labels = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                # 不需要手动读 output。
                # 只要 forward 执行了，hook 就会自动帮我们把统计量记下来。
                self.model(
                    input_features=input_features,
                    labels=labels,
                    attention_mask=attention_mask,
                )

        # 用完一定要 remove hook，否则重复调用时会重复统计。
        for h in self.hooks:
            h.remove()
        self.hooks = []

        return self.weights, self.activations_stats
