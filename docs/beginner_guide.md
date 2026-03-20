# Beginner Guide

这份文档是专门给“刚开始读这套重构后代码”的自己看的。

目标不是解释所有细节，而是回答 3 个最重要的问题：

1. 这个项目现在到底是怎么跑起来的？
2. 每个文件分别负责什么？
3. 我如果想改参数，应该去哪里改？

## 1. 从外到内，代码是怎么执行的

最外层入口是 [`main.py`](/Users/hehaoran/Developer/whisper_pruning/main.py)。

比如你执行：

```bash
python main.py prune-once --model whisper-large-v3-original --dataset en
```

实际执行顺序大致是：

1. 先进入 [`main.py`](/Users/hehaoran/Developer/whisper_pruning/main.py)
2. `main.py` 调用 [`whisper_pruning/cli.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/cli.py) 里的 `main()`
3. [`whisper_pruning/cli.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/cli.py) 解析命令行参数
4. CLI 把零散参数整理成配置对象，比如 [`ExperimentConfig`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/config.py) 和 [`OneShotPruningRunConfig`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/config.py)
5. CLI 调用 [`whisper_pruning/pipelines.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/pipelines.py) 里的 `run_one_shot_pruning(...)`
6. `run_one_shot_pruning(...)` 再去调用底层工具：
   - [`utils/WandA_profiler.py`](/Users/hehaoran/Developer/whisper_pruning/utils/WandA_profiler.py)
   - [`utils/scorer.py`](/Users/hehaoran/Developer/whisper_pruning/utils/scorer.py)
   - [`utils/pruning_tools.py`](/Users/hehaoran/Developer/whisper_pruning/utils/pruning_tools.py)
   - [`utils/evaluator.py`](/Users/hehaoran/Developer/whisper_pruning/utils/evaluator.py)

一句话总结：

`main.py` 负责进门，`cli.py` 负责看你要干什么，`config.py` 负责打包参数，`pipelines.py` 负责真正串流程，`utils/` 负责干具体活。

## 2. 每个文件的职责

### 入口层

- [`main.py`](/Users/hehaoran/Developer/whisper_pruning/main.py)
  最外层入口，只做一件事：调用 CLI。

- [`whisper_pruning/__main__.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/__main__.py)
  让你可以用 `python -m whisper_pruning ...` 运行。

### 参数层

- [`whisper_pruning/config.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/config.py)
  这里不做计算，只定义“参数长什么样”。

  最常见的几个对象：
  - `ExperimentConfig`
    描述公共实验信息，比如模型名、数据集名、dtype。
  - `DataLoaderConfig`
    描述 dataloader 参数，比如 `split`、`batch_size`、`num_samples`。
  - `OneShotPruningRunConfig`
    把“一次剪枝实验”需要的所有配置打包在一起。

### 运行时辅助层

- [`whisper_pruning/runtime.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/runtime.py)
  负责那些“很多实验都会重复做”的事：
  - 加载模型
  - 加载 processor
  - 自动决定 device
  - 构造 dataloader
  - 构造 evaluator

### 流水线层

- [`whisper_pruning/pipelines.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/pipelines.py)
  这是最核心的调度层。

  你可以把它理解成“实验剧本”：
  - `run_evaluation(...)`
    只评测
  - `run_profile(...)`
    只收集激活统计
  - `run_one_shot_pruning(...)`
    一次剪枝
  - `run_owl_sweep(...)`
    扫 OWL 参数

### 底层算法层

- [`utils/WandA_profiler.py`](/Users/hehaoran/Developer/whisper_pruning/utils/WandA_profiler.py)
  从线性层收集权重和激活统计。

- [`utils/scorer.py`](/Users/hehaoran/Developer/whisper_pruning/utils/scorer.py)
  根据权重和激活统计，计算每层的分数或保留率。

- [`utils/pruning_tools.py`](/Users/hehaoran/Developer/whisper_pruning/utils/pruning_tools.py)
  真正执行剪枝，把不保留的权重置 0。

- [`utils/evaluator.py`](/Users/hehaoran/Developer/whisper_pruning/utils/evaluator.py)
  负责评测，输出 CER / WER / Loss。

## 3. 你以后改参数，应该去哪改

如果你只是想“直接跑一个固定实验”，最简单的是改快捷脚本：

- 基线评测改 [`evaluate_model.py`](/Users/hehaoran/Developer/whisper_pruning/evaluate_model.py)
- 一次剪枝改 [`one_time_pruning_evaluation.py`](/Users/hehaoran/Developer/whisper_pruning/one_time_pruning_evaluation.py)

如果你想学更通用的方式，就改命令行参数，例如：

```bash
python main.py prune-once \
  --model whisper-large-v3-original \
  --dataset en \
  --dtype float32 \
  --profile-batch-size 32 \
  --profile-num-samples 64 \
  --eval-batch-size 16
```

如果你想改“默认值”，就去看 [`whisper_pruning/cli.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/cli.py) 里每个参数的 `default=...`。

## 4. 为什么我把原来代码拆成这样

原来的问题主要是：

1. 每个脚本都自己写一遍“加载模型、加载数据、构造 evaluator”的重复代码。
2. 路径、batch size、dtype、split 散在很多文件里，不容易统一改。
3. 一旦你以后想加新实验，很容易复制粘贴出第 5 个、第 6 个脚本。

现在的思路是：

1. 把“重复逻辑”收进 `runtime.py`
2. 把“实验流程”收进 `pipelines.py`
3. 把“参数长什么样”收进 `config.py`
4. 让顶层脚本只负责“给默认参数”

这样以后你要新加一个实验，通常只需要：

1. 在 `pipelines.py` 里写一个新流程函数
2. 在 `cli.py` 里加一个新命令

不需要重新复制一整套加载逻辑。

## 5. 如果你现在只想先看懂“一次剪枝”

建议只盯着这 4 个文件：

1. [`one_time_pruning_evaluation.py`](/Users/hehaoran/Developer/whisper_pruning/one_time_pruning_evaluation.py)
2. [`whisper_pruning/pipelines.py`](/Users/hehaoran/Developer/whisper_pruning/whisper_pruning/pipelines.py) 里的 `run_one_shot_pruning(...)`
3. [`utils/scorer.py`](/Users/hehaoran/Developer/whisper_pruning/utils/scorer.py)
4. [`utils/pruning_tools.py`](/Users/hehaoran/Developer/whisper_pruning/utils/pruning_tools.py)

这样理解会最快，因为这条线就是你的核心实验线。

## 6. 你现在可以先不用理解的部分

如果你现在还不熟，可以先不管这些：

- `__main__.py`
- `plotting.py` 里的画图细节
- `sweep-owl` 的三重循环细节
- CLI 里的所有命令参数细节

先看懂“一次剪枝怎么跑通”，再回头看其它模块，会轻松很多。
