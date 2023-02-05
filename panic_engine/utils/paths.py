import os
from pathlib import Path

MAIN_CONFIG_PATH = "configs/main_config.env"
LOADER_CONFIG_PATH = "configs/load_config.env"

def get_project_root() -> str:
    return Path(os.path.abspath(__file__)).parent.parent.__str__()