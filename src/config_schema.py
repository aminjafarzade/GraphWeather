"""Typed, read-only projection over an already-resolved config dict.

STATUS (P5.3): additive, NOT yet wired into production. The only importer is
tests/test_config_schema_projection.py; YParams does not use this projection yet.
Kept intentionally as a staged typed read-model — do not remove as dead code.

Governing principle (why this is behavior-preserving): the five reducers that
resolve a config -- ``apply_resolution_profile`` -> ``normalize_model_config_dict``
-> ``normalize_training_config_dict`` -> ``normalize_target_handling_config_dict``
-> ``normalize_diagnostics_config_dict`` -- remain the single behavior spine and
run untouched. This module does **not** re-derive, coerce, or default any value;
it is a *view* built AFTER the spine, over the resolved flat dict. ``AppConfig.flat``
is therefore byte-identical to ``YParams.params`` today, and section views satisfy
``view.to_dict() == flat[section]`` by construction (see
``tests/test_config_schema_projection.py``).

Nothing imports this yet -- it is additive. A later slice rewires ``YParams`` to
build an ``AppConfig`` and expose ``self.params = app.flat`` as a thin shim, and
migrates read sites from stringly-typed ``_get(params, "key")`` to
``app.<section>.<field>`` attribute access.
"""

from __future__ import annotations

from typing import Any, Optional


class _Section:
    """Base read-only view over a config sub-dict.

    Stores the raw section verbatim so ``to_dict()`` round-trips exactly. Typed
    access is provided by subclass ``@property`` accessors that read ``_raw``.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: Optional[dict[str, Any]] = None) -> None:
        self._raw: dict[str, Any] = dict(raw) if raw else {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._raw.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self._raw

    def to_dict(self) -> dict[str, Any]:
        """Return the raw section verbatim (lossless: ``== flat[section]``)."""
        return dict(self._raw)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Section) and other._raw == self._raw

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, self._raw)


# --------------------------------------------------------------------------- #
# Nested model sub-sections                                                    #
# --------------------------------------------------------------------------- #
class SkipFusionConfig(_Section):
    @property
    def type(self) -> Any:
        return self._raw.get("type", "default")

    @property
    def init_scale(self) -> Any:
        return self._raw.get("init_scale", 1.0)

    @property
    def max_scale(self) -> Any:
        return self._raw.get("max_scale", 2.0)


class PoolingConfig(_Section):
    @property
    def type(self) -> Any:
        return self._raw.get("type", "default")

    @property
    def init_scale(self) -> Any:
        return self._raw.get("init_scale", 1.0)

    @property
    def max_scale(self) -> Any:
        return self._raw.get("max_scale", 2.0)

    @property
    def mean_type(self) -> Any:
        return self._raw.get("mean_type", "mean")

    @property
    def include_max(self) -> Any:
        return self._raw.get("include_max", True)


class L0RefineConfig(_Section):
    @property
    def type(self) -> Any:
        return self._raw.get("type", "nodewise_mlp")

    @property
    def mlp_expansion(self) -> Any:
        return self._raw.get("mlp_expansion", 2)

    @property
    def dropout(self) -> Any:
        return self._raw.get("dropout", 0.0)

    @property
    def residual_scale_init(self) -> Any:
        return self._raw.get("residual_scale_init", 0.1)

    @property
    def learnable_residual_scale(self) -> Any:
        return self._raw.get("learnable_residual_scale", True)


class LeadConditioningConfig(_Section):
    @property
    def enabled(self) -> Any:
        return self._raw.get("enabled", False)

    @property
    def type(self) -> Any:
        return self._raw.get("type", "none")

    @property
    def max_lead(self) -> Any:
        return self._raw.get("max_lead", 0)

    @property
    def added_input_channels(self) -> Any:
        return self._raw.get("added_input_channels", 0)


class ModelConfig(_Section):
    """View over the ``model`` sub-dict of a resolved config."""

    _SCALARS = (
        "input_channels", "output_channels", "hidden_dim", "num_heads", "head_dim",
        "k_neighbors", "level_k_neighbors", "use_l3", "use_l4", "use_l4_ratio15",
        "num_graph_levels", "hierarchy_type", "l0_blocks", "l1_blocks", "l2_blocks",
        "l3_blocks", "l4_blocks", "l0_refine_blocks", "l1_refine_blocks",
        "l2_refine_after_l3_blocks", "l3_refine_after_l4_blocks",
    )

    def __getattr__(self, name: str) -> Any:
        # scalar model fields read straight from the raw section
        if name in type(self)._SCALARS:
            return self._raw.get(name)
        raise AttributeError(name)

    @property
    def skip_fusion(self) -> SkipFusionConfig:
        return SkipFusionConfig(self._raw.get("skip_fusion") or {})

    @property
    def pooling(self) -> PoolingConfig:
        return PoolingConfig(self._raw.get("pooling") or {})

    @property
    def l0_refine(self) -> Optional[L0RefineConfig]:
        raw = self._raw.get("l0_refine")
        return L0RefineConfig(raw) if isinstance(raw, dict) else None

    @property
    def lead_conditioning(self) -> LeadConditioningConfig:
        return LeadConditioningConfig(self._raw.get("lead_conditioning") or {})


# --------------------------------------------------------------------------- #
# Rollout (flattened top-level keys, mirrors normalize_training_config_dict)   #
# --------------------------------------------------------------------------- #
class RandomRolloutConfig(_Section):
    @property
    def min_horizon(self) -> Any:
        return self._raw.get("min_horizon", 1)

    @property
    def max_horizon(self) -> Any:
        return self._raw.get("max_horizon", 1)

    @property
    def distribution(self) -> Any:
        return self._raw.get("distribution", "uniform_integer")

    @property
    def loss_type(self) -> Any:
        return self._raw.get("loss_type", "all_steps_mean")

    @property
    def final_step_weight(self) -> Any:
        return self._raw.get("final_step_weight", 0.0)

    @property
    def detach_between_steps(self) -> Any:
        return self._raw.get("detach_between_steps", False)


class ScheduledRolloutConfig(_Section):
    @property
    def phases(self) -> list:
        return list(self._raw.get("phases", []) or [])


class RolloutConfig:
    """Projection of the flattened top-level rollout keys of a resolved config.

    These keys live at the top level (not under a single sub-dict), so this view
    reads selected keys from the whole flat dict; it deliberately does NOT expose
    a section ``to_dict()``.
    """

    __slots__ = ("_flat",)

    def __init__(self, flat: dict[str, Any]) -> None:
        self._flat = flat

    @property
    def rollout_mode(self) -> Any:
        return self._flat.get("rollout_mode", "curriculum")

    @property
    def fixed_train_rollout_steps(self) -> Any:
        return self._flat.get("fixed_train_rollout_steps", 10)

    @property
    def rollout_loss_weights(self) -> Any:
        return self._flat.get("rollout_loss_weights", "uniform")

    @property
    def load_only_current_rollout(self) -> Any:
        return self._flat.get("load_only_current_rollout", False)

    @property
    def activation_checkpointing(self) -> Any:
        return self._flat.get("activation_checkpointing", False)

    @property
    def checkpoint_rollout_steps(self) -> Any:
        return self._flat.get("checkpoint_rollout_steps", False)

    @property
    def random_rollout(self) -> RandomRolloutConfig:
        return RandomRolloutConfig(self._flat.get("random_rollout") or {})

    @property
    def scheduled_rollout(self) -> ScheduledRolloutConfig:
        return ScheduledRolloutConfig(self._flat.get("scheduled_rollout") or {})


# --------------------------------------------------------------------------- #
# Target handling / diagnostics                                               #
# --------------------------------------------------------------------------- #
class TargetHandlingConfig(_Section):
    @property
    def enabled(self) -> Any:
        return self._raw.get("enabled", True)

    @property
    def copy_variables(self) -> list:
        return list(self._raw.get("copy_variables", []) or [])

    @property
    def exclude_loss_variables(self) -> list:
        return list(self._raw.get("exclude_loss_variables", []) or [])

    @property
    def known_future_variables(self) -> list:
        return list(self._raw.get("known_future_variables", []) or [])


class BaselineCompareConfig(_Section):
    @property
    def enabled(self) -> Any:
        return self._raw.get("enabled", False)


class WandbConfig(_Section):
    @property
    def enabled(self) -> Any:
        return self._raw.get("enabled", True)

    @property
    def project(self) -> Any:
        return self._raw.get("project")

    @property
    def entity(self) -> Any:
        return self._raw.get("entity")


class PlotsConfig(_Section):
    @property
    def enabled(self) -> Any:
        return self._raw.get("enabled", False)


class DiagnosticsConfig(_Section):
    @property
    def enabled(self) -> Any:
        return self._raw.get("enabled", False)

    @property
    def rollout_horizons(self) -> list:
        return list(self._raw.get("rollout_horizons", []) or [])

    @property
    def baseline_compare(self) -> BaselineCompareConfig:
        return BaselineCompareConfig(self._raw.get("baseline_compare") or {})

    @property
    def wandb(self) -> WandbConfig:
        return WandbConfig(self._raw.get("wandb") or {})

    @property
    def plots(self) -> PlotsConfig:
        return PlotsConfig(self._raw.get("plots") or {})


# --------------------------------------------------------------------------- #
# Data loader (selected top-level keys)                                        #
# --------------------------------------------------------------------------- #
class DataLoaderConfig:
    __slots__ = ("_flat",)

    _KEYS = (
        "batch_size", "gradient_accumulation_steps", "num_data_workers", "pin_memory",
        "persistent_workers", "prefetch_factor", "train_data_path", "valid_data_path",
        "test_dataset_path", "normalization",
    )

    def __init__(self, flat: dict[str, Any]) -> None:
        self._flat = flat

    def __getattr__(self, name: str) -> Any:
        if name in type(self)._KEYS:
            return self._flat.get(name)
        raise AttributeError(name)


# --------------------------------------------------------------------------- #
# Top-level container                                                          #
# --------------------------------------------------------------------------- #
class AppConfig:
    """Typed read-model over a fully-resolved config dict.

    ``flat`` remains the single source of truth (identical to ``YParams.params``);
    the section views are additive, read-only lenses that never re-derive values.
    """

    __slots__ = ("_flat",)

    def __init__(self, flat: dict[str, Any]) -> None:
        self._flat = flat

    @classmethod
    def from_resolved(cls, flat: dict[str, Any]) -> "AppConfig":
        """Build from an already-resolved flat dict (e.g. ``YParams.params``)."""
        return cls(flat)

    @classmethod
    def from_yparams(cls, yparams: Any) -> "AppConfig":
        return cls(yparams.params)

    @property
    def flat(self) -> dict[str, Any]:
        """The resolved dict, verbatim (single source of truth)."""
        return self._flat

    # --- typed section views ------------------------------------------------ #
    @property
    def model(self) -> ModelConfig:
        return ModelConfig(self._flat.get("model") or {})

    @property
    def rollout(self) -> RolloutConfig:
        return RolloutConfig(self._flat)

    @property
    def data(self) -> DataLoaderConfig:
        return DataLoaderConfig(self._flat)

    @property
    def target_handling(self) -> TargetHandlingConfig:
        return TargetHandlingConfig(self._flat.get("target_handling") or {})

    @property
    def diagnostics(self) -> DiagnosticsConfig:
        return DiagnosticsConfig(self._flat.get("diagnostics") or {})

    # --- dict-style delegation (lets AppConfig back a YParams shim later) ---- #
    def get(self, key: str, default: Any = None) -> Any:
        return self._flat.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._flat[key]

    def __contains__(self, key: str) -> bool:
        return key in self._flat
