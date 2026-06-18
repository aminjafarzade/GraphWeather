from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

import yaml


class YParams:
    """Small YAML config loader with KAI-style dot and dict access."""

    def __init__(self, yaml_filename: str, config_name: str, print_params: bool = False):
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        self.params: dict[str, Any] = {}

        with open(yaml_filename, "r", encoding="utf-8") as f:
            root = yaml.safe_load(f)
        if config_name not in root:
            available = ", ".join(sorted(root.keys()))
            raise KeyError(f"Config '{config_name}' not found in {yaml_filename}. Available: {available}")

        for key, value in root[config_name].items():
            if value == "None":
                value = None
            self.params[key] = value
            setattr(self, key, value)
            if print_params:
                print(key, value)

    def __getitem__(self, key: str) -> Any:
        return self.params[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.params[key] = value
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        return key in self.params

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def update_params(self, config: dict[str, Any]) -> None:
        for key, value in config.items():
            self[key] = value

    def log(self) -> None:
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: %s", self._yaml_filename)
        logging.info("Configuration name: %s", self._config_name)
        for key, value in self.params.items():
            logging.info("%s %s", key, value)
        logging.info("---------------------------------------------------")


_LOG_FORMAT = "%(asctime)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(rank: int = 0, log_file: Optional[str] = None) -> None:
    root = logging.getLogger()
    if root.hasHandlers():
        root.handlers.clear()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        root.addHandler(ch)

