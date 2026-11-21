"""读取配置，集中管理所有路径与参数。"""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 相对路径改为相对项目根目录解析
    if cfg["store"]["path"].startswith("."):
        cfg["store"]["path"] = str(ROOT / cfg["store"]["path"])
    # 允许用环境变量重定向存储目录（测试隔离用，不影响正常使用）
    if os.environ.get("VM_STORE_PATH"):
        cfg["store"]["path"] = os.environ["VM_STORE_PATH"]
    if cfg["model"]["local_path"]:
        p = Path(cfg["model"]["local_path"])
        if not p.is_absolute():
            p = ROOT / p
        cfg["model"]["local_path"] = str(p)
    return cfg
