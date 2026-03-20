# Whisper Pruning

这版结构尽量保持简单。

你以后主要只需要看和改这几个顶层脚本：

- [`evaluate_model.py`](/Users/hehaoran/Developer/whisper_pruning/evaluate_model.py)
- [`one_time_pruning_evaluation.py`](/Users/hehaoran/Developer/whisper_pruning/one_time_pruning_evaluation.py)
- [`visualize_scores.py`](/Users/hehaoran/Developer/whisper_pruning/visualize_scores.py)
- [`visualize_distributions.py`](/Users/hehaoran/Developer/whisper_pruning/visualize_distributions.py)
- [`visualize_owl_onetimepruning.py`](/Users/hehaoran/Developer/whisper_pruning/visualize_owl_onetimepruning.py)

这些脚本的开头都有一块：

```python
# 只改这里
```

你平时基本只改那一块就够了。

## 怎么跑

基线评测：

```bash
python evaluate_model.py
```

一次性剪枝评测：

```bash
python one_time_pruning_evaluation.py
```

画层分数：

```bash
python visualize_scores.py
```

画层分布：

```bash
python visualize_distributions.py
```

扫描 OWL 参数并画图：

```bash
python visualize_owl_onetimepruning.py
```

## 每个文件是干什么的

- [`experiment_helpers.py`](/Users/hehaoran/Developer/whisper_pruning/experiment_helpers.py)
  只放很少量的公共辅助函数，比如加载模型、加载数据。平时一般不用改。

- [`utils/WandA_profiler.py`](/Users/hehaoran/Developer/whisper_pruning/utils/WandA_profiler.py)
  收集线性层权重和激活统计。

- [`utils/scorer.py`](/Users/hehaoran/Developer/whisper_pruning/utils/scorer.py)
  根据权重和激活值给每层打分。

- [`utils/pruning_tools.py`](/Users/hehaoran/Developer/whisper_pruning/utils/pruning_tools.py)
  真正执行剪枝。

- [`utils/evaluator.py`](/Users/hehaoran/Developer/whisper_pruning/utils/evaluator.py)
  负责评测 CER / WER / Loss。

## 最推荐的阅读顺序

如果你现在只想先看懂主流程，建议按这个顺序读：

1. [`one_time_pruning_evaluation.py`](/Users/hehaoran/Developer/whisper_pruning/one_time_pruning_evaluation.py)
2. [`utils/WandA_profiler.py`](/Users/hehaoran/Developer/whisper_pruning/utils/WandA_profiler.py)
3. [`utils/scorer.py`](/Users/hehaoran/Developer/whisper_pruning/utils/scorer.py)
4. [`utils/pruning_tools.py`](/Users/hehaoran/Developer/whisper_pruning/utils/pruning_tools.py)
5. [`utils/evaluator.py`](/Users/hehaoran/Developer/whisper_pruning/utils/evaluator.py)
