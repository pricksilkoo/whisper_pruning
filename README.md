# Whisper Pruning

现在项目分成两层:

- 顶层脚本: 负责实验编排，尽量保持简单
- `utils/`: 负责通用工具，后面你扩展新方法主要改这里

## 推荐入口

- 基线评测: [`eval.py`](/Users/hehaoran/Developer/whisper_pruning/eval.py)
- 一次性剪枝评测: [`prune_once.py`](/Users/hehaoran/Developer/whisper_pruning/prune_once.py)
- 画层分数: [`plot_scores.py`](/Users/hehaoran/Developer/whisper_pruning/plot_scores.py)
- 画分布: [`plot_distributions.py`](/Users/hehaoran/Developer/whisper_pruning/plot_distributions.py)
- 扫描 OWL 参数: [`sweep_owl.py`](/Users/hehaoran/Developer/whisper_pruning/sweep_owl.py)

## 怎么跑

```bash
python eval.py
python prune_once.py
python plot_scores.py
python plot_distributions.py
python sweep_owl.py
```

## 工具层结构

你后面主要看这些文件:

- [`utils/dataloader.py`](/Users/hehaoran/Developer/whisper_pruning/utils/dataloader.py)
  数据准备和 dataloader。

- [`utils/evaluator.py`](/Users/hehaoran/Developer/whisper_pruning/utils/evaluator.py)
  负责评测 CER / WER / Loss。

- [`utils/signal_collector.py`](/Users/hehaoran/Developer/whisper_pruning/utils/signal_collector.py)
  负责统一收集 `weights / activations / gradients`。激活和梯度默认是统计字典，文件里也提供了均值/RMS 的辅助函数。

- [`utils/pruning_basemethod.py`](/Users/hehaoran/Developer/whisper_pruning/utils/pruning_basemethod.py)
  底层剪枝方法，当前内置 `wanda` 和 `sparsegpt`。

- [`utils/scorer.py`](/Users/hehaoran/Developer/whisper_pruning/utils/scorer.py)
  非均匀剪枝打分器。后面你新增方法时，统一按 `weights / activations / gradients` 这个接口写。

如果你的新打分方法需要梯度，记得在 [`prune_once.py`](/Users/hehaoran/Developer/whisper_pruning/prune_once.py) 里把 `PROFILE_GRADIENTS = True`。

## 推荐阅读顺序

1. [`prune_once.py`](/Users/hehaoran/Developer/whisper_pruning/prune_once.py)
2. [`utils/signal_collector.py`](/Users/hehaoran/Developer/whisper_pruning/utils/signal_collector.py)
3. [`utils/scorer.py`](/Users/hehaoran/Developer/whisper_pruning/utils/scorer.py)
4. [`utils/pruning_basemethod.py`](/Users/hehaoran/Developer/whisper_pruning/utils/pruning_basemethod.py)
5. [`utils/evaluator.py`](/Users/hehaoran/Developer/whisper_pruning/utils/evaluator.py)
