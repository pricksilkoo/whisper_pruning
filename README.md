# Whisper Pruning

这个仓库现在整理成了“统一配置 + 统一入口 + 薄脚本 wrapper”的结构，重点解决原来每个脚本都要单独改路径、dtype、batch size 和剪枝参数的问题。

如果你现在对这套结构还不熟，建议先看这份阅读说明：

- [`docs/beginner_guide.md`](/Users/hehaoran/Developer/whisper_pruning/docs/beginner_guide.md)

## 现在怎么用

统一入口是 [`main.py`](/Users/hehaoran/Developer/whisper_pruning/main.py)。

评测基线：

```bash
python main.py evaluate --model whisper-large-v3-original --dataset en --batch-size 64
```

做一次剪枝并评测：

```bash
python main.py prune-once \
  --model whisper-large-v3-original \
  --dataset en \
  --dtype float32 \
  --profile-batch-size 32 \
  --profile-num-samples 64 \
  --eval-batch-size 16 \
  --eval-num-samples 512 \
  --score-method owl \
  --level 7 \
  --average-retention-ratio 0.4
```

画层分数：

```bash
python main.py plot-scores --model whisper-medium-original --dataset en --score-method owl
```

画层分布：

```bash
python main.py plot-distributions --model whisper-large-v3-original --dataset en
```

扫描 OWL 参数：

```bash
python main.py sweep-owl --model whisper-large-v3-original --dataset en
```

如果模型和数据不在默认目录，可以额外传：

```bash
--models-root /path/to/models --data-root /path/to/data
```

## 目录说明

- [`whisper_pruning/config.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/config.py)：统一管理实验配置、路径和 dataloader 参数。
- [`whisper_pruning/runtime.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/runtime.py)：统一加载模型、processor、device、dtype 和 evaluator。
- [`whisper_pruning/pipelines.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/pipelines.py)：封装评测、一次性剪枝、OWL sweep 等实验流程。
- [`whisper_pruning/plotting.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/plotting.py)：集中放可视化逻辑。
- 顶层脚本如 [`evaluate_model.py`](/Users/hehaoran/Developer/whisper_pruning/evaluate_model.py) 和 [`one_time_pruning_evaluation.py`](/Users/hehaoran/Developer/whisper_pruning/one_time_pruning_evaluation.py) 现在只是预置参数的快捷入口。

## 推荐阅读顺序

如果你是为了“看懂项目”，建议按这个顺序读：

1. [`main.py`](/Users/hehaoran/Developer/whisper_pruning/main.py)
2. [`whisper_pruning/cli.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/cli.py)
3. [`whisper_pruning/config.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/config.py)
4. [`whisper_pruning/pipelines.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/pipelines.py)
5. [`utils/WandA_profiler.py`](/Users/hehaoran/Developer/whisper_pruning/utils/WandA_profiler.py)
6. [`utils/scorer.py`](/Users/hehaoran/Developer/whisper_pruning/utils/scorer.py)
7. [`utils/pruning_tools.py`](/Users/hehaoran/Developer/whisper_pruning/utils/pruning_tools.py)
8. [`utils/evaluator.py`](/Users/hehaoran/Developer/whisper_pruning/utils/evaluator.py)

## 这次顺手修的点

- 去掉了硬编码 `CUDA_VISIBLE_DEVICES`，显卡选择改为外部环境控制。
- `Evaluator` 和 `WAprofiler` 默认跟随模型 dtype，不再靠脚本手工对齐。
- `Scorer` 新增统一 `compute(...)` 接口，所有打分方法都能走同一套入口。
- `data_loader` 支持显式控制 `shuffle`。
- 非英文评测默认不再强制走英文 normalizer。
