from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import torch

from .features import VariableResolver, _as_string_list, canonical_name


def _get(params: Any, name: str, default: Any = None) -> Any:
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


@dataclass(frozen=True)
class TargetHandlingSettings:
    enabled: bool = False
    copy_variables: tuple[str, ...] = ()
    known_future_variables: tuple[str, ...] = ()
    exclude_loss_variables: tuple[str, ...] = ()

    @classmethod
    def from_params(cls, params: Any) -> "TargetHandlingSettings":
        raw = _get(params, "target_handling", {}) or {}
        if not isinstance(raw, dict):
            raise ValueError("target_handling must be a mapping when provided.")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            copy_variables=tuple(_as_string_list(raw.get("copy_variables", []))),
            known_future_variables=tuple(_as_string_list(raw.get("known_future_variables", []))),
            exclude_loss_variables=tuple(_as_string_list(raw.get("exclude_loss_variables", []))),
        )


def _canonical_ordered(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        canonical = canonical_name(value) or str(value)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def _normal_metadata(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}
    return {
        "enabled": bool(payload.get("enabled", False)),
        "copy_variables": _canonical_ordered(tuple(_as_string_list(payload.get("copy_variables", [])))),
        "known_future_variables": _canonical_ordered(tuple(_as_string_list(payload.get("known_future_variables", [])))),
        "exclude_loss_variables": _canonical_ordered(tuple(_as_string_list(payload.get("exclude_loss_variables", [])))),
    }


class TargetHandling:
    """Apply configured state-channel copy/override rules after each forecast step."""

    version = 1

    def __init__(
        self,
        *,
        settings: TargetHandlingSettings,
        resolver: VariableResolver,
        logger: logging.Logger | Any = logging,
        context_name: str = "target_handling",
    ):
        self.settings = settings
        self.resolver = resolver
        self.logger = logger
        self.context_name = str(context_name)
        self.enabled = bool(settings.enabled)
        self.copy_channels: dict[str, int] = {}
        self.known_future_channels: dict[str, int] = {}
        self.exclude_loss_channels: dict[str, int] = {}
        if self.enabled:
            self._resolve_channels()

    @classmethod
    def from_params(
        cls,
        params: Any,
        *,
        channel_names: list[str] | None,
        out_channels: list[int],
        logger: logging.Logger | Any = logging,
    ) -> "TargetHandling":
        settings = TargetHandlingSettings.from_params(params)
        resolver = VariableResolver(params, channel_names, out_channels, logger=logger)
        return cls(settings=settings, resolver=resolver, logger=logger)

    @classmethod
    def from_settings(
        cls,
        settings: TargetHandlingSettings,
        params: Any,
        *,
        channel_names: list[str] | None,
        out_channels: list[int],
        logger: logging.Logger | Any = logging,
        context_name: str = "target_handling",
    ) -> "TargetHandling":
        resolver = VariableResolver(params, channel_names, out_channels, logger=logger)
        return cls(settings=settings, resolver=resolver, logger=logger, context_name=context_name)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "copy_variables": list(self.copy_channels.keys()) if self.enabled else [],
            "known_future_variables": list(self.known_future_channels.keys()) if self.enabled else [],
            "exclude_loss_variables": list(self.exclude_loss_channels.keys()) if self.enabled else [],
        }

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {"target_handling": dict(self.metadata)}

    def _resolve_group(self, variables: tuple[str, ...], field_name: str) -> dict[str, int]:
        resolved: dict[str, int] = {}
        requested = list(variables)
        for variable in variables:
            item = self.resolver.resolve(variable, required=False)
            if item.local_index is None:
                raise ValueError(
                    f"{self.context_name} requested {field_name}={json.dumps(requested)}, "
                    f"but variable {variable} was not found in channel list."
                )
            resolved[item.canonical] = int(item.local_index)
        return resolved

    def _resolve_channels(self) -> None:
        self.copy_channels = self._resolve_group(self.settings.copy_variables, "copy_variables")
        self.known_future_channels = self._resolve_group(
            self.settings.known_future_variables,
            "known_future_variables",
        )
        self.exclude_loss_channels = self._resolve_group(
            self.settings.exclude_loss_variables,
            "exclude_loss_variables",
        )

    def log_startup(self, output_channels: int | None = None) -> None:
        self.logger.info("Target handling:")
        self.logger.info("  enabled: %s", str(bool(self.enabled)).lower())
        if not self.enabled:
            if output_channels is not None:
                self.logger.info("Loss channels: %d/%d", int(output_channels), int(output_channels))
            return
        self.logger.info("  copy_variables:")
        for name, idx in self.copy_channels.items():
            self.logger.info("    %s -> channel %d", name, int(idx))
        if self.known_future_channels:
            self.logger.info("  known_future_variables:")
            for name, idx in self.known_future_channels.items():
                self.logger.info("    %s -> channel %d", name, int(idx))
        else:
            self.logger.info("  known_future_variables: []")
        self.logger.info("  exclude_loss_variables:")
        for name, idx in self.exclude_loss_channels.items():
            self.logger.info("    %s -> channel %d", name, int(idx))
        if output_channels is not None:
            included = int(output_channels) - len({int(idx) for idx in self.exclude_loss_channels.values()})
            self.logger.info("Loss channels: %d/%d", included, int(output_channels))
            if self.exclude_loss_channels:
                self.logger.info("Excluded from loss: %s", ", ".join(self.exclude_loss_channels))

    def loss_channel_mask(self, output_channels: int, device: torch.device | None = None) -> torch.Tensor | None:
        if not self.enabled or not self.exclude_loss_channels:
            return None
        mask = torch.ones(int(output_channels), dtype=torch.float32, device=device)
        for idx in self.exclude_loss_channels.values():
            if 0 <= int(idx) < int(output_channels):
                mask[int(idx)] = 0.0
        if float(mask.sum().item()) <= 0.0:
            raise ValueError("Loss channel mask excludes all output channels.")
        return mask

    def apply(
        self,
        *,
        pred_next: torch.Tensor,
        current_state: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        target_sequence: torch.Tensor | None = None,
        lead: int | None = None,
        target_norm: torch.Tensor | None = None,
        known_future_values: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return pred_next
        out = pred_next
        changed = False

        copy_source = initial_state if initial_state is not None else current_state
        for _, idx in self.copy_channels.items():
            idx = int(idx)
            if 0 <= idx < pred_next.shape[1]:
                if not changed:
                    out = pred_next.clone()
                    changed = True
                out[:, idx] = copy_source[:, idx]

        if self.known_future_channels:
            if target_norm is None and target_sequence is not None and lead is not None:
                target_index = int(lead) - 1
                if target_index < 0 or target_index >= int(target_sequence.shape[1]):
                    raise ValueError(
                        f"Target handling lead={lead} is outside target_sequence length {target_sequence.shape[1]}."
                    )
                target_norm = target_sequence[:, target_index]
            if target_norm is not None:
                for _, idx in self.known_future_channels.items():
                    idx = int(idx)
                    if 0 <= idx < pred_next.shape[1]:
                        if not changed:
                            out = pred_next.clone()
                            changed = True
                        out[:, idx] = target_norm[:, idx]
            elif known_future_values is not None:
                # REAL-INFERENCE INTERFACE: no future ERA5 target is available, so the
                # caller supplies normalized per-variable fields for this step's valid
                # time (e.g. src.solar.AnalyticSolarForcing.known_future_values for tisr).
                for name, idx in self.known_future_channels.items():
                    if name not in known_future_values:
                        raise ValueError(
                            f"Target handling known_future_values is missing variable {name!r}; "
                            f"got {sorted(known_future_values)}."
                        )
                    idx = int(idx)
                    if 0 <= idx < pred_next.shape[1]:
                        values = known_future_values[name]
                        if values.dim() == 4 and values.shape[1] == 1:
                            values = values[:, 0]
                        if not changed:
                            out = pred_next.clone()
                            changed = True
                        out[:, idx] = values.to(device=out.device, dtype=out.dtype)
            else:
                raise ValueError(
                    "Target handling known_future_variables require target_sequence and 1-based lead, "
                    "an explicit target_norm, or known_future_values (for real inference compute the "
                    "forcing analytically, e.g. src.solar.AnalyticSolarForcing for tisr)."
                )
        return out


def combine_loss_channel_masks(
    *masks: torch.Tensor | None,
    output_channels: int,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    combined: torch.Tensor | None = None
    for mask in masks:
        if mask is None:
            continue
        candidate = mask.to(device=device, dtype=torch.float32)
        if candidate.numel() != int(output_channels):
            raise ValueError(f"channel_mask length {candidate.numel()} does not match channels {output_channels}")
        combined = candidate if combined is None else combined * candidate
    if combined is None:
        return None
    if float(combined.sum().item()) <= 0.0:
        raise ValueError("Loss channel mask excludes all output channels.")
    return combined


def target_handling_metadata_matches(active: dict[str, Any], checkpoint_metadata: dict[str, Any]) -> tuple[bool, str]:
    active_payload = _normal_metadata(active)
    checkpoint_has = "target_handling" in checkpoint_metadata
    checkpoint_payload = _normal_metadata(checkpoint_metadata.get("target_handling", None))
    if not active_payload["enabled"] and not checkpoint_has:
        return True, ""
    if active_payload != checkpoint_payload:
        return (
            False,
            f"target_handling mismatch: active={active_payload} checkpoint={checkpoint_payload}. "
            "Use a checkpoint trained with the same target_handling configuration.",
        )
    return True, ""
