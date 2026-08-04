from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Optional

import yaml

from .architecture import normalize_model_config_dict
from .resolution import apply_resolution_profile


# Surgical environment-variable interpolation for config values (portability).
# Only the explicit ``${VAR}`` / ``${VAR:-default}`` form is expanded — bare
# ``$VAR`` is left untouched, so a value must opt in. When VAR is unset the
# default (after ``:-``) is used, so configs that pin the default to the current
# absolute path resolve byte-identically to before (see configs/base/paths.yaml
# and .env.example). Set the env var to relocate data without editing configs.
_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` in string values."""
    if isinstance(value, str):
        if "${" not in value:
            return value
        return _ENV_VAR_RE.sub(
            lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


from .layers import ATTENTION_IMPL_DEFAULT  # noqa: E402


def normalize_training_config_dict(params: dict[str, Any]) -> dict[str, Any]:
    """Expose nested training rollout settings as stable top-level keys."""
    resolved = dict(params)
    training = dict(resolved.get("training", {}) or {})

    rollout_mode = str(training.get("rollout_mode", resolved.get("rollout_mode", "curriculum"))).strip().lower()
    fixed_steps = training.get("fixed_train_rollout_steps", resolved.get("fixed_train_rollout_steps", 10))
    loss_weights = training.get("rollout_loss_weights", resolved.get("rollout_loss_weights", "uniform"))
    load_only_current_rollout = bool(
        training.get("load_only_current_rollout", resolved.get("load_only_current_rollout", False))
    )
    activation_checkpointing = bool(
        training.get("activation_checkpointing", resolved.get("activation_checkpointing", False))
    )
    checkpoint_rollout_steps = bool(
        training.get("checkpoint_rollout_steps", resolved.get("checkpoint_rollout_steps", False))
    )
    edge_projection_cache = bool(
        training.get("edge_projection_cache", resolved.get("edge_projection_cache", False))
    )
    attention_impl = str(
        training.get("attention_impl", resolved.get("attention_impl", ATTENTION_IMPL_DEFAULT))
    ).strip().lower()
    random_rollout_raw = dict(resolved.get("random_rollout", {}) or {})
    random_rollout_raw.update(dict(training.get("random_rollout", {}) or {}))
    max_steps = int(resolved.get("max_rollout_steps", random_rollout_raw.get("max_horizon", fixed_steps)))
    random_rollout = {
        "min_horizon": int(random_rollout_raw.get("min_horizon", 1)),
        "max_horizon": int(random_rollout_raw.get("max_horizon", max_steps)),
        "distribution": str(random_rollout_raw.get("distribution", "uniform_integer")).strip().lower(),
        "loss_type": str(random_rollout_raw.get("loss_type", "all_steps_mean")).strip().lower(),
        "final_step_weight": float(random_rollout_raw.get("final_step_weight", 0.0)),
        "detach_between_steps": bool(random_rollout_raw.get("detach_between_steps", False)),
    }
    scheduled_rollout_raw = dict(resolved.get("scheduled_rollout", {}) or {})
    scheduled_rollout_raw.update(dict(training.get("scheduled_rollout", {}) or {}))
    scheduled_phases = []
    for phase in list(scheduled_rollout_raw.get("phases", []) or []):
        item = dict(phase or {})
        normalized = {
            "name": str(item.get("name", "")),
            "start_epoch": int(item.get("start_epoch", 1)),
            "end_epoch": None if item.get("end_epoch", None) is None else int(item.get("end_epoch")),
            "mode": str(item.get("mode", "")).strip().lower(),
        }
        if normalized["mode"] == "random":
            normalized.update(
                {
                    "min_horizon": int(item.get("min_horizon", 1)),
                    "max_horizon": int(item.get("max_horizon", max_steps)),
                    "distribution": str(item.get("distribution", "uniform_integer")).strip().lower(),
                }
            )
        elif normalized["mode"] == "fixed":
            normalized["horizon"] = int(item.get("horizon", max_steps))
        else:
            normalized.update({key: item[key] for key in item if key not in normalized})
        scheduled_phases.append(normalized)
    scheduled_rollout = {"phases": scheduled_phases}
    if rollout_mode in {"random", "scheduled"}:
        load_only_current_rollout = False

    resolved["rollout_mode"] = rollout_mode
    resolved["fixed_train_rollout_steps"] = int(fixed_steps)
    resolved["rollout_loss_weights"] = loss_weights
    resolved["load_only_current_rollout"] = load_only_current_rollout
    resolved["activation_checkpointing"] = activation_checkpointing
    resolved["checkpoint_rollout_steps"] = checkpoint_rollout_steps
    resolved["edge_projection_cache"] = edge_projection_cache
    resolved["attention_impl"] = attention_impl
    resolved["random_rollout"] = dict(random_rollout)
    resolved["scheduled_rollout"] = dict(scheduled_rollout)
    resolved["training"] = {
        **training,
        "rollout_mode": rollout_mode,
        "fixed_train_rollout_steps": int(fixed_steps),
        "rollout_loss_weights": loss_weights,
        "load_only_current_rollout": load_only_current_rollout,
        "activation_checkpointing": activation_checkpointing,
        "checkpoint_rollout_steps": checkpoint_rollout_steps,
        "edge_projection_cache": edge_projection_cache,
        "attention_impl": attention_impl,
        "random_rollout": dict(random_rollout),
        "scheduled_rollout": dict(scheduled_rollout),
    }
    return resolved


DEFAULT_TARGET_HANDLING: dict[str, Any] = {
    "enabled": True,
    "copy_variables": ["orog"],
    "exclude_loss_variables": ["orog"],
    "known_future_variables": [],
}


DEFAULT_DIAGNOSTICS: dict[str, Any] = {
    "enabled": False,
    "baseline_compare": {
        "enabled": False,
        "baseline_name": None,
        "baseline_rollout_curve_csv": None,
        "baseline_scalars_json": None,
    },
}


def normalize_target_handling_config_dict(params: dict[str, Any]) -> dict[str, Any]:
    """Make fixed orography the project default while honoring explicit overrides."""
    resolved = dict(params)
    raw = resolved.get("target_handling", None)
    if raw is None:
        resolved["target_handling"] = dict(DEFAULT_TARGET_HANDLING)
        return resolved
    if not isinstance(raw, dict):
        raise ValueError("target_handling must be a mapping when provided.")
    merged = dict(DEFAULT_TARGET_HANDLING)
    merged.update(raw)
    merged["enabled"] = bool(merged.get("enabled", True))
    for key in ("copy_variables", "exclude_loss_variables", "known_future_variables"):
        value = merged.get(key, [])
        if value is None:
            merged[key] = []
        elif isinstance(value, (list, tuple)):
            merged[key] = [str(item) for item in value]
        else:
            merged[key] = [chunk.strip() for chunk in str(value).replace(",", " ").split() if chunk.strip()]
    resolved["target_handling"] = merged
    return resolved


def normalize_diagnostics_config_dict(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize optional diagnostics settings without changing disabled defaults."""
    resolved = dict(params)
    raw = resolved.get("diagnostics", None)
    if raw is None:
        resolved["diagnostics"] = dict(DEFAULT_DIAGNOSTICS)
        return resolved
    if not isinstance(raw, dict):
        raise ValueError("diagnostics must be a mapping when provided.")
    if not bool(raw.get("enabled", False)):
        diagnostics = dict(DEFAULT_DIAGNOSTICS)
        baseline_compare = dict(DEFAULT_DIAGNOSTICS["baseline_compare"])
        baseline_compare.update(dict(raw.get("baseline_compare", {}) or {}))
        diagnostics["baseline_compare"] = baseline_compare
        resolved["diagnostics"] = diagnostics
        return resolved

    defaults = {
        "enabled": True,
        "log_every_epochs": 1,
        "heavy_every_epochs": 5,
        "run_after_training": True,
        "rollout_horizons": [1, 2, 4, 6, 8, 10],
        "max_train_diag_batches": 2,
        "max_valid_diag_batches": 4,
        "max_full_diag_batches": 32,
        "embedding_sample_nodes": 2048,
        "pairwise_sample_nodes": 1024,
        "collect_embeddings": True,
        "collect_attention": True,
        "spectral_rmse": True,
        "spectral_variables": ["t2m", "u10", "v10", "msl", "z500"],
        "grid_shape": None,
        "baseline_compare": {
            "enabled": False,
            "baseline_name": None,
            "baseline_rollout_curve_csv": None,
            "baseline_scalars_json": None,
        },
        "wandb": {
            "enabled": True,
            "project": None,
            "entity": "amin1jafarzade-kaist",
            "run_name": None,
            "tags": ["diagnostics"],
            "log_scalars": True,
            "log_tables": True,
            "log_plots": False,
            "log_images": False,
            "log_histograms": False,
            "log_artifacts": False,
        },
        "plots": {
            "enabled": False,
            "save_local": True,
            "output_dir": "diagnostics",
            "plot_every_epochs": 5,
            "plot_after_training": True,
            "map_horizons": [1, 4, 10],
            "spectral_horizons": [1, 4, 10],
            "variables": ["t2m", "u10", "v10", "msl", "z500"],
            "max_map_batches": 2,
            "max_map_samples": 4,
            "max_hist_values": 200000,
            "rollout_plots": True,
            "layer_plots": True,
            "attention_plots": True,
            "spectral_plots": True,
            "spatial_maps": True,
            "optimization_plots": True,
        },
        "output_dir": "diagnostics",
    }
    merged = dict(defaults)
    merged.update(raw)
    baseline_compare = dict(defaults["baseline_compare"])
    baseline_compare.update(dict(raw.get("baseline_compare", {}) or {}))
    merged["baseline_compare"] = baseline_compare
    wandb = dict(defaults["wandb"])
    wandb.update(dict(raw.get("wandb", {}) or {}))
    merged["wandb"] = wandb
    plots = dict(defaults["plots"])
    plots.update(dict(raw.get("plots", {}) or {}))
    merged["plots"] = plots
    merged["enabled"] = True
    merged["rollout_horizons"] = [int(x) for x in list(merged.get("rollout_horizons", [])) if int(x) > 0]
    for list_key in ("map_horizons", "spectral_horizons"):
        plots[list_key] = [int(x) for x in list(plots.get(list_key, [])) if int(x) > 0]
    plots["variables"] = [str(x) for x in list(plots.get("variables", []) or [])]
    for key in (
        "log_every_epochs",
        "heavy_every_epochs",
        "max_train_diag_batches",
        "max_valid_diag_batches",
        "max_full_diag_batches",
        "embedding_sample_nodes",
        "pairwise_sample_nodes",
    ):
        merged[key] = int(merged[key])
    for key in ("plot_every_epochs", "max_map_batches", "max_map_samples", "max_hist_values"):
        plots[key] = int(plots[key])
    for key in ("run_after_training", "collect_embeddings", "collect_attention", "spectral_rmse"):
        merged[key] = bool(merged.get(key, False))
    for key in ("log_scalars", "log_tables", "log_plots", "log_images", "log_histograms", "log_artifacts"):
        wandb[key] = bool(wandb.get(key, False))
    for key in (
        "enabled",
        "save_local",
        "plot_after_training",
        "rollout_plots",
        "layer_plots",
        "attention_plots",
        "spectral_plots",
        "spatial_maps",
        "optimization_plots",
    ):
        plots[key] = bool(plots.get(key, False))
    resolved["diagnostics"] = merged
    return resolved


class YParams:
    """Small YAML config loader with KAI-style dot and dict access."""

    def __init__(
        self,
        yaml_filename: str,
        config_name: str,
        print_params: bool = False,
        resolution_mode: str | None = None,
    ):
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
        self.apply_resolution_mode(resolution_mode)
        # Expand ${VAR:-default} in all (possibly profile-injected) string values
        # last, so env-overridable paths work everywhere. No-op when no ${...}.
        self.params = _expand_env_vars(self.params)
        self._sync_attrs()
        if print_params:
            for key, value in self.params.items():
                print(key, value)

    def _sync_attrs(self) -> None:
        for key, value in self.params.items():
            setattr(self, key, value)

    def apply_resolution_mode(self, resolution_mode: str | None = None) -> None:
        self.params = apply_resolution_profile(self.params, cli_resolution_mode=resolution_mode)
        self.params = normalize_model_config_dict(self.params)
        self.params = normalize_training_config_dict(self.params)
        self.params = normalize_target_handling_config_dict(self.params)
        self.params = normalize_diagnostics_config_dict(self.params)
        self._sync_attrs()

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
