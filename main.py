"""
这个文件是最外层入口。

你在命令行里执行:
    python main.py evaluate ...
或者:
    python main.py prune-once ...

最终都会先进入这里，然后再转到 `whisper_pruning/cli.py` 去解析参数。
"""

from whisper_pruning.cli import main


if __name__ == "__main__":
    # main() 负责:
    # 1. 读取命令行参数
    # 2. 按命令类型组装 config
    # 3. 调用对应的实验流水线
    main()
