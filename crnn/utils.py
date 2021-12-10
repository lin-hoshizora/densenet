"""
Helper functions
"""

from pathlib import Path
from datetime import datetime
import yaml


conf_folder = Path(__file__).resolve().parent.parent / "conf"


def load_conf(p: str) -> dict:
  with open(p) as f:
    res = yaml.safe_load(f)
  return res


def get_timestamp() -> str:
  timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  return timestamp


def ensure_exist(path) -> None:
  if not Path(path).exists():
    raise ValueError(f"{path} does not exists.")
