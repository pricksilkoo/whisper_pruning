import torch

class PruningTool():
    def __init__(self):
        self.masks={}
        self.pruned_weights={}

    def restore_model_weights(self, model, backup_state_dict):
        #original_weights_backup = {k: v.clone() for k, v in model.state_dict().items()}用于获取原始权重
        #传入原始权重！！！
        model.load_state_dict(backup_state_dict)

    def apply_to_model(self, model,log=True):
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in self.pruned_weights:
                    module.weight.data.copy_(self.pruned_weights[name])
        if (log == True):
            print("mask已经挂在权重上了")
            total_model_params = sum(p.numel() for p in model.parameters())
            linear_params_total = 0
            total_pruned_zeros = 0
            for name, mask in self.masks.items():
                layer_total = mask.numel()
                layer_retained = mask.sum().item()
                
                linear_params_total += layer_total
                total_pruned_zeros += (layer_total - layer_retained)
            global_sparsity = total_pruned_zeros / total_model_params
            linear_sparsity = total_pruned_zeros / linear_params_total
            print("\n" + "="*40)
            print("📊 剪枝报告")
            print("="*40)
            print(f" Whisper 全模型总参数量: {total_model_params:,}")
            print(f" 其中参与剪枝的线性层参数量: {linear_params_total:,}")
            print(f" 实际被物理抹除的参数量: {total_pruned_zeros:,}")
            print("-" * 40)
            print(f" 局部指标 -> 线性层内部稀疏度: {linear_sparsity:.2%}")
            print(f" 核心指标 -> 整网全局真稀疏度: {global_sparsity:.2%}")
            print("="*40 + "\n")


    def wanda_nm_pruning(self, weights, stats, n=2, m=4):
        """
        Wanda 半结构化 (N:M) 剪枝核心逻辑
        参数:
            weights (dict): WAprofiler 收集到的各层权重字典 {layer_name: tensor}
            stats (dict): WAprofiler 收集到的激活值统计字典 {layer_name: {'sq_sum': tensor, ...}}
            n (int): 每 M 个元素中保留 N 个非零值
            m (int): 分块的大小 (通常是 4 或 8)  
        返回:
            self.pruned_weights (dict): 剪枝后(含大量0)的权重字典
            self.masks (dict): 对应的 0/1 掩码矩阵，方便后续接入 PyTorch 官方 API
        """
        self.masks={}
        self.pruned_weights={}
        
        for name, W in weights.items():
            out_channels, in_channels = W.shape
  
            if in_channels % m != 0:
                print(f"⚠️ 警告: {name} 的输入维度 {in_channels} 无法被 {m} 整除，跳过该层！")
                self.pruned_weights[name] = W.clone()
                continue

            x_l2_norm = torch.sqrt(stats[name]["sq_sum"])
            wanda_metric = torch.abs(W) * x_l2_norm.unsqueeze(0)
            metric_blocks = wanda_metric.view(out_channels, in_channels // m, m)
            _, topk_indices = torch.topk(metric_blocks, k=n, dim=-1)
            mask_blocks = torch.zeros_like(metric_blocks, dtype=torch.bool)
            mask_blocks.scatter_(dim=-1, index=topk_indices, src=torch.ones_like(topk_indices, dtype=torch.bool))
            mask = mask_blocks.view(out_channels, in_channels)
            self.pruned_weights[name] = W * mask
            self.masks[name] = mask
        return self.pruned_weights, self.masks
    
    def wanda_unstructured_pruning(self, weights, stats, sparsity=0.5):
        """
        Wanda 常规非结构化剪枝核心逻辑 (支持均匀与非均匀剪枝)
        参数:
            weights (dict): WAprofiler 收集到的各层权重字典 {layer_name: tensor}
            stats (dict): WAprofiler 收集到的激活值统计字典 {layer_name: {'sq_sum': tensor, ...}}
            sparsity (float 或 dict): 
                - 传入 float (如 0.5): 全局均匀剪枝，所有层统一砍掉 50%。
                - 传入 dict (如 {'layer1': 0.3...}): 非均匀剪枝，按层定制稀疏度。未在字典中的层默认不剪。
        返回:
            self.pruned_weights (dict): 剪枝后的权重字典
            self.masks (dict): 对应的 0/1 掩码矩阵
        """
        self.masks = {}
        self.pruned_weights = {}
        
        for name, W in weights.items():
            if isinstance(sparsity, dict):
                current_sparsity = sparsity.get(name, 0.0)
            else:
                current_sparsity = float(sparsity)
                
            x_l2_norm = torch.sqrt(stats[name]["sq_sum"])
            wanda_metric = torch.abs(W) * x_l2_norm.unsqueeze(0)
            
            out_channels, in_channels = W.shape
            prune_num = int(in_channels * current_sparsity)
            
            if prune_num <= 0:
                self.pruned_weights[name] = W.clone()
                self.masks[name] = torch.ones_like(W, dtype=torch.bool)
                continue
                
            _, bottomk_indices = torch.topk(wanda_metric, k=prune_num, dim=-1, largest=False)
            
            mask = torch.ones_like(W, dtype=torch.bool)
            mask.scatter_(dim=-1, index=bottomk_indices, src=torch.zeros_like(bottomk_indices, dtype=torch.bool))
            
            self.pruned_weights[name] = W * mask
            self.masks[name] = mask
            
        return self.pruned_weights, self.masks