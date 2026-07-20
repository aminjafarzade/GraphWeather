from __future__ import annotations

import calendar
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ALIASES: dict[str, list[str]] = {
    "t2m": ["t2m", "2m_temperature", "temperature_2m"],
    "msl": ["msl", "mslp", "mean_sea_level_pressure"],
    "t850": ["t850", "t_850", "temperature_850", "t@850"],
    "z500": ["z500", "z_500", "geopotential_500", "z@500"],
    "tisr": ["tisr", "top_incoming_solar_radiation"],
    "orog": ["orog", "orography", "geopotential_at_surface", "surface_geopotential"],
    "lsm": ["lsm", "land_sea_mask", "land_mask"],
    "q700": ["q700", "q_700", "specific_humidity_700"],
    "u10": ["u10", "10u", "u_10m", "u_component_of_wind_10m", "u_component_wind_10m"],
    "v10": ["v10", "10v", "v_10m", "v_component_of_wind_10m", "v_component_wind_10m"],
    "u500": ["u500", "u_500", "u_component_wind_500", "u_component_of_wind_500"],
    "v500": ["v500", "v_500", "v_component_wind_500", "v_component_of_wind_500"],
    "u850": ["u850", "u_850", "u_component_wind_850"],
    "v850": ["v850", "v_850", "v_component_wind_850"],
}


HARD_CODED_FALLBACK_CHANNELS: dict[str, int] = {
    # ERA5-67 order used by this repository. These are only used after all
    # metadata sources fail, and a warning is always logged.
    "t2m": 0,
    "msl": 1,
    "tisr": 5,
    "u850": 8,
    "v850": 20,
    "t850": 32,
    "q700": 46,
    "z500": 60,
    "orog": 66,
}


def _get(params: Any, name: str, default: Any = None) -> Any:
    if isinstance(params, dict):
        return params.get(name, default)
    return getattr(params, name, default)


def norm_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def canonical_name(name: str) -> str | None:
    norm = norm_name(name)
    for canonical, aliases in ALIASES.items():
        if norm == norm_name(canonical) or norm in {norm_name(alias) for alias in aliases}:
            return canonical
    return None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [value]
    parsed: list[str] = []
    for item in values:
        parsed.extend(chunk.strip() for chunk in str(item).replace(",", " ").split() if chunk.strip())
    return parsed


def _nested(params: Any, *keys: str, default: Any = None) -> Any:
    current = params
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _names_from_payload(payload: Any) -> list[str] | None:
    if isinstance(payload, list) and payload:
        if all(isinstance(item, str) for item in payload):
            return [str(item) for item in payload]
        if all(isinstance(item, dict) for item in payload):
            names: dict[int, str] = {}
            for item in payload:
                idx = item.get("channel", item.get("index", item.get("original_channel_idx")))
                name = item.get("name", item.get("actual_name", item.get("variable_name")))
                if idx is not None and name is not None:
                    names[int(idx)] = str(name)
            if names:
                return [names.get(idx, f"Var{idx}") for idx in range(max(names) + 1)]
    if not isinstance(payload, dict):
        return None
    for key in ("channel_names", "variable_names", "variables"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            nested = _names_from_payload(value)
            if nested:
                return nested
        if isinstance(value, dict) and value:
            names: dict[int, str] = {}
            for raw_idx, item in value.items():
                if isinstance(item, dict):
                    idx = item.get("channel", item.get("index", raw_idx))
                    name = item.get("actual_name", item.get("name", item.get("variable_name")))
                else:
                    idx = raw_idx
                    name = item
                try:
                    names[int(idx)] = str(name)
                except (TypeError, ValueError):
                    continue
            if names:
                return [names.get(idx, f"Var{idx}") for idx in range(max(names) + 1)]
    return None


def _channel_names_from_json(path: str | os.PathLike[str] | None) -> list[str] | None:
    if not path:
        return None
    candidate = Path(str(path)).expanduser()
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    return _names_from_payload(payload)


def _metadata_candidates(params: Any) -> list[str]:
    candidates: list[str] = []
    direct = _nested(params, "variable_metadata", "path", default=None)
    if direct:
        candidates.append(str(direct))
    for stat_path in (_get(params, "global_means_path", None), _get(params, "global_stds_path", None)):
        if not stat_path:
            continue
        root = Path(str(stat_path)).expanduser().parent
        for name in ("variable_channel_mapping.json", "channel_names.json", "variable_metadata.json", "metadata.json"):
            candidates.append(str(root / name))
    experiment_dir = _get(params, "experiment_dir", None)
    if experiment_dir:
        root = Path(str(experiment_dir)).expanduser()
        for name in ("variable_channel_mapping.json", "channel_names.json", "variable_metadata.json", "metadata.json"):
            candidates.append(str(root / name))
    return candidates


@dataclass
class ResolvedVariable:
    canonical: str
    channel: int | None
    local_index: int | None
    source: str
    actual_name: str | None = None


class VariableResolver:
    def __init__(
        self,
        params: Any,
        channel_names: list[str] | None,
        out_channels: list[int],
        logger: logging.Logger | Any = logging,
    ):
        self.params = params
        self.out_channels = [int(x) for x in out_channels]
        self.logger = logger
        self.sources: list[tuple[str, list[str]]] = []

        if channel_names:
            self.sources.append(("dataset metadata / channel names", [str(x) for x in channel_names]))

        for key in ("variable_names", "channel_names", "variables"):
            names = _names_from_payload(_get(params, key, None))
            if names:
                self.sources.append((f"config {key}", names))

        for candidate in _metadata_candidates(params):
            names = _channel_names_from_json(candidate)
            if names:
                source = "normalization metadata" if "stats" in candidate else "experiment metadata"
                self.sources.append((f"{source}: {candidate}", names))

    def _resolve_from_names(self, variable: str, names: list[str]) -> tuple[int, str] | None:
        canonical = canonical_name(variable) or variable
        candidates = [norm_name(variable), norm_name(canonical)]
        candidates.extend(norm_name(alias) for alias in ALIASES.get(canonical, []))
        name_to_idx = {norm_name(name): idx for idx, name in enumerate(names)}
        for candidate in candidates:
            if candidate in name_to_idx:
                idx = int(name_to_idx[candidate])
                actual = names[idx] if idx < len(names) else f"Var{idx}"
                return idx, actual
        return None

    def resolve(
        self,
        variable: str,
        *,
        required: bool = False,
        allow_hardcoded_fallback: bool = True,
    ) -> ResolvedVariable:
        canonical = canonical_name(variable) or str(variable)
        for source, names in self.sources:
            match = self._resolve_from_names(variable, names)
            if match is None:
                continue
            channel, actual = match
            local = self.out_channels.index(channel) if channel in self.out_channels else None
            return ResolvedVariable(canonical, channel, local, source, actual)

        if allow_hardcoded_fallback and canonical in HARD_CODED_FALLBACK_CHANNELS:
            channel = int(HARD_CODED_FALLBACK_CHANNELS[canonical])
            local = self.out_channels.index(channel) if channel in self.out_channels else None
            self.logger.warning(
                "WARNING: using hardcoded fallback variable channel for %s -> channel %d. "
                "Verify channel order before using these results.",
                canonical,
                channel,
            )
            return ResolvedVariable(canonical, channel, local, "hardcoded fallback", f"Var{channel}")

        message = f"Could not resolve variable {variable!r} from dataset/config/normalization/experiment metadata."
        if required:
            raise ValueError(message)
        self.logger.warning("WARNING: %s", message)
        return ResolvedVariable(canonical, None, None, "unresolved", None)

    def log_resolved(self, variables: list[str], required: set[str] | None = None) -> dict[str, ResolvedVariable]:
        required = required or set()
        resolved: dict[str, ResolvedVariable] = {}
        self.logger.info("Resolved variables:")
        for variable in variables:
            item = self.resolve(variable, required=variable in required)
            resolved[variable] = item
            channel = "unresolved" if item.channel is None else str(item.channel)
            local = "" if item.local_index is None else f" local_output={item.local_index}"
            self.logger.info("  %-5s -> channel %s%s (%s)", variable, channel, local, item.source)
        return resolved


@dataclass
class ExtraFeatureSettings:
    enabled: bool = False
    lat_lon_sincos: bool = False
    dayofyear_sincos: bool = False
    use_target_time: bool = True
    orography: bool = False
    land_sea_mask: bool = False
    known_forcings_enabled: bool = False
    known_forcing_variables: tuple[str, ...] = ()
    copy_variables: tuple[str, ...] = ()
    exclude_loss_variables: tuple[str, ...] = ()
    override_prediction_with_known: tuple[str, ...] = ()
    require_static_features: bool = False
    static_path: str | None = None
    orography_name: str = "orog"
    land_sea_mask_name: str = "lsm"

    @classmethod
    def from_params(cls, params: Any) -> "ExtraFeatureSettings":
        extra = _get(params, "extra_features", {}) or {}
        static_fields = _get(params, "static_fields", {}) or {}
        enabled = bool(_nested(extra, "enabled", default=False))
        return cls(
            enabled=enabled,
            lat_lon_sincos=bool(_nested(extra, "spatial", "lat_lon_sincos", default=False)),
            orography=bool(_nested(extra, "spatial", "orography", default=False)),
            land_sea_mask=bool(_nested(extra, "spatial", "land_sea_mask", default=False)),
            dayofyear_sincos=bool(_nested(extra, "temporal", "dayofyear_sincos", default=False)),
            use_target_time=bool(_nested(extra, "temporal", "use_target_time", default=True)),
            known_forcings_enabled=bool(_nested(extra, "known_forcings", "enabled", default=False)),
            known_forcing_variables=tuple(_as_string_list(_nested(extra, "known_forcings", "variables", default=[]))),
            copy_variables=tuple(_as_string_list(_nested(extra, "static_handling", "copy_variables", default=[]))),
            exclude_loss_variables=tuple(_as_string_list(_nested(extra, "static_handling", "exclude_loss_variables", default=[]))),
            override_prediction_with_known=tuple(
                _as_string_list(_nested(extra, "static_handling", "override_prediction_with_known", default=[]))
            ),
            require_static_features=bool(_get(params, "require_static_features", False)),
            static_path=_nested(static_fields, "path", default=None),
            orography_name=str(_nested(static_fields, "orography_name", default="orog")),
            land_sea_mask_name=str(_nested(static_fields, "land_sea_mask_name", default="lsm")),
        )


def _days_in_year_from_time(value: Any) -> int:
    year = int(getattr(value, "year"))
    return 366 if calendar.isleap(year) else 365


def dayofyear_sincos(
    day_of_year: torch.Tensor,
    days_in_year: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    day = day_of_year.to(device=device, dtype=dtype)
    days = days_in_year.to(device=device, dtype=dtype).clamp_min(1.0)
    angle = 2.0 * math.pi * day / days
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def _coordinate_to_radians(values: torch.Tensor, *, coordinate: str) -> torch.Tensor:
    values = values.to(torch.float32)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return values
    max_abs = float(finite.abs().max().item())
    if coordinate == "lat":
        threshold = math.pi / 2.0 + 1.0e-4
    else:
        threshold = 2.0 * math.pi + 1.0e-4
    return values if max_abs <= threshold else torch.deg2rad(values)


def _load_static_field(path: str, names: list[str]) -> np.ndarray | None:
    candidate = Path(str(path)).expanduser()
    if not candidate.exists():
        return None
    suffix = candidate.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(candidate), dtype=np.float32).squeeze()
    if suffix == ".npz":
        payload = np.load(candidate)
        for name in names:
            if name in payload:
                return np.asarray(payload[name], dtype=np.float32).squeeze()
        return None
    try:
        import netCDF4 as nc
    except ModuleNotFoundError:
        return None
    with nc.Dataset(candidate, "r") as ds:
        for name in names:
            if name in ds.variables:
                return np.asarray(ds.variables[name][:], dtype=np.float32).squeeze()
    return None


class RolloutFeatureBuilder:
    version = 1

    def __init__(
        self,
        *,
        settings: ExtraFeatureSettings,
        graph: Any,
        resolver: VariableResolver,
        out_channels: list[int],
        output_means: np.ndarray | None = None,
        output_stds: np.ndarray | None = None,
        logger: logging.Logger | Any = logging,
    ):
        self.settings = settings
        self.graph = graph
        self.resolver = resolver
        self.out_channels = [int(x) for x in out_channels]
        self.output_means = None if output_means is None else np.asarray(output_means, dtype=np.float32)
        self.output_stds = None if output_stds is None else np.asarray(output_stds, dtype=np.float32)
        self.logger = logger
        self.enabled = bool(settings.enabled)
        self.feature_names: list[str] = []
        self.aux_feature_dim = 0
        self.known_forcing_channels: dict[str, int] = {}
        self.copy_channels: dict[str, int] = {}
        self.exclude_loss_channels: dict[str, int] = {}
        self.resolved_variables: dict[str, ResolvedVariable] = {}
        self.static_fields: dict[str, torch.Tensor] = {}
        self._state_static_variables: set[str] = set()
        self._static_node_features: torch.Tensor | None = None
        self._static_feature_columns: dict[str, int] = {}
        self._lat_lon_node_features: torch.Tensor | None = None
        self._warned_missing_time = False

        if not self.enabled:
            return

        self._resolve_channels()
        self._prepare_spatial_features()
        self.aux_feature_dim = len(self.feature_names)

    def _resolved(self, variable: str, *, required: bool = False) -> ResolvedVariable:
        return self.resolved_variables.get(variable) or self.resolver.resolve(variable, required=required)

    @classmethod
    def from_params(
        cls,
        params: Any,
        *,
        graph: Any,
        channel_names: list[str] | None,
        out_channels: list[int],
        output_means: np.ndarray | None = None,
        output_stds: np.ndarray | None = None,
        logger: logging.Logger | Any = logging,
    ) -> "RolloutFeatureBuilder":
        settings = ExtraFeatureSettings.from_params(params)
        resolver = VariableResolver(params, channel_names, out_channels, logger=logger)
        return cls(
            settings=settings,
            graph=graph,
            resolver=resolver,
            out_channels=out_channels,
            output_means=output_means,
            output_stds=output_stds,
            logger=logger,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "extra_features_enabled": bool(self.enabled),
            "aux_feature_names": list(self.feature_names),
            "aux_feature_dim": int(self.aux_feature_dim),
            "known_forcing_variables": list(self.known_forcing_channels.keys()),
            "copy_variables": list(self.copy_channels.keys()),
            "exclude_loss_variables": list(self.exclude_loss_channels.keys()),
            "feature_builder_version": int(self.version),
        }

    def checkpoint_metadata(self, base_input_channels: int, total_input_channels: int) -> dict[str, Any]:
        payload = dict(self.metadata)
        payload.update(
            {
                "base_input_channels": int(base_input_channels),
                "total_input_channels": int(total_input_channels),
            }
        )
        return payload

    def _resolve_channels(self) -> None:
        required = set(self.settings.known_forcing_variables) if self.settings.known_forcings_enabled else set()
        variables = sorted(
            {
                *self.settings.known_forcing_variables,
                *self.settings.copy_variables,
                *self.settings.exclude_loss_variables,
                *self.settings.override_prediction_with_known,
                "orog" if self.settings.orography else "",
                "lsm" if self.settings.land_sea_mask else "",
            }
            - {""}
        )
        resolved = self.resolver.log_resolved(variables, required=required)
        self.resolved_variables.update(resolved)

        if self.settings.known_forcings_enabled:
            for variable in self.settings.known_forcing_variables:
                item = resolved.get(variable) or self._resolved(variable, required=True)
                if item.local_index is None:
                    raise ValueError(
                        f"known_forcings.variables includes {variable!r}, but it is not present in out_channels."
                    )
                self.known_forcing_channels[item.canonical] = int(item.local_index)

        for variable in self.settings.copy_variables:
            item = resolved.get(variable) or self._resolved(variable)
            if item.local_index is None:
                message = f"copy_variables includes {variable!r}, but it could not be resolved in model output channels."
                if self.settings.require_static_features:
                    raise ValueError(message)
                self.logger.warning("WARNING: %s Skipping static copy for this variable.", message)
                continue
            self.copy_channels[item.canonical] = int(item.local_index)

        for variable in self.settings.exclude_loss_variables:
            item = resolved.get(variable) or self._resolved(variable)
            if item.local_index is None:
                self.logger.warning(
                    "WARNING: exclude_loss_variables includes %r, but it could not be resolved; loss mask will skip this entry.",
                    variable,
                )
                continue
            self.exclude_loss_channels[item.canonical] = int(item.local_index)

        for variable in self.settings.override_prediction_with_known:
            item = resolved.get(variable) or self._resolved(variable, required=variable in required)
            if item.local_index is None:
                if variable in required:
                    raise ValueError(f"override_prediction_with_known includes required variable {variable!r}, but it is unresolved.")
                self.logger.warning("WARNING: override_prediction_with_known variable %r is unresolved; skipping.", variable)
                continue
            self.known_forcing_channels[item.canonical] = int(item.local_index)

    def _prepare_spatial_features(self) -> None:
        if self.settings.lat_lon_sincos:
            lat_lon = self.graph.L0.lat_lon.detach().to(torch.float32)
            lat_rad = _coordinate_to_radians(lat_lon[:, 0], coordinate="lat")
            lon_rad = _coordinate_to_radians(lat_lon[:, 1], coordinate="lon")
            self._lat_lon_node_features = torch.stack(
                [torch.sin(lat_rad), torch.cos(lat_rad), torch.sin(lon_rad), torch.cos(lon_rad)],
                dim=-1,
            )
            self.feature_names.extend(["sin_lat", "cos_lat", "sin_lon", "cos_lon"])

        static_parts: list[torch.Tensor] = []
        if self.settings.orography:
            orog = self._static_field_or_none(
                canonical="orog",
                names=[self.settings.orography_name, *ALIASES["orog"]],
                normalize=True,
            )
            if orog is not None:
                self._static_feature_columns["orography"] = len(static_parts)
                static_parts.append(orog)
                self.feature_names.append("orography")
            elif self._resolved("orog").local_index is not None:
                self._state_static_variables.add("orog")
                self.feature_names.append("orography")
        if self.settings.land_sea_mask:
            lsm = self._static_field_or_none(
                canonical="lsm",
                names=[self.settings.land_sea_mask_name, *ALIASES["lsm"]],
                normalize=False,
            )
            if lsm is not None:
                self._static_feature_columns["land_sea_mask"] = len(static_parts)
                static_parts.append(lsm.clamp(0.0, 1.0))
                self.feature_names.append("land_sea_mask")
            elif self._resolved("lsm").local_index is not None:
                self._state_static_variables.add("lsm")
                self.feature_names.append("land_sea_mask")

        self._static_node_features = torch.cat(static_parts, dim=-1) if static_parts else None

        if self.settings.dayofyear_sincos:
            self.feature_names.extend(["sin_dayofyear", "cos_dayofyear"])
        if self.settings.known_forcings_enabled:
            for variable in self.settings.known_forcing_variables:
                canonical = canonical_name(variable) or variable
                if canonical in self.known_forcing_channels:
                    self.feature_names.append(f"known_{canonical}")

    def _static_field_or_none(self, canonical: str, names: list[str], normalize: bool) -> torch.Tensor | None:
        field: np.ndarray | None = None
        source = None
        if self.settings.static_path:
            field = _load_static_field(self.settings.static_path, names)
            if field is not None:
                source = self.settings.static_path

        if field is None:
            item = self._resolved(canonical)
            if item.local_index is not None:
                self.logger.info(
                    "Static feature %s will be read from normalized state channel %d at rollout time.",
                    canonical,
                    item.local_index,
                )
                return None

        if field is None:
            message = f"Requested static feature {canonical!r} was not found in static_fields.path or state channels."
            if self.settings.require_static_features:
                raise ValueError(message)
            self.logger.warning("WARNING: %s Skipping this auxiliary feature.", message)
            return None

        height, width = int(self.graph.L0.height), int(self.graph.L0.width)
        field = np.asarray(field, dtype=np.float32).squeeze()
        if field.shape != (height, width):
            message = f"Static feature {canonical!r} shape {field.shape} does not match grid {(height, width)}."
            if self.settings.require_static_features:
                raise ValueError(message)
            self.logger.warning("WARNING: %s Skipping this auxiliary feature.", message)
            return None

        if normalize:
            item = self._resolved(canonical)
            if item.local_index is not None and self.output_means is not None and self.output_stds is not None:
                mean = float(self.output_means[item.local_index])
                std = float(self.output_stds[item.local_index])
                field = (field - mean) / (std + 1.0e-8)
            else:
                field = (field - float(np.nanmean(field))) / (float(np.nanstd(field)) + 1.0e-8)
        tensor = torch.as_tensor(field.reshape(height * width, 1), dtype=torch.float32)
        self.logger.info("Loaded static feature %s from %s", canonical, source)
        return tensor

    def loss_channel_mask(self, output_channels: int, device: torch.device | None = None) -> torch.Tensor | None:
        if not self.exclude_loss_channels:
            return None
        mask = torch.ones(int(output_channels), dtype=torch.float32, device=device)
        for idx in self.exclude_loss_channels.values():
            if 0 <= int(idx) < int(output_channels):
                mask[int(idx)] = 0.0
        if float(mask.sum().item()) <= 0.0:
            raise ValueError("Loss channel mask excludes all output channels.")
        return mask

    def _current_state_feature(self, current: torch.Tensor, canonical: str) -> torch.Tensor | None:
        item = self._resolved(canonical)
        if item.local_index is None:
            return None
        values = current[:, int(item.local_index) : int(item.local_index) + 1]
        return values.permute(0, 2, 3, 1).reshape(values.shape[0], values.shape[2] * values.shape[3], 1)

    def build_step_features(
        self,
        *,
        current: torch.Tensor,
        target_norm: torch.Tensor | None = None,
        target_dayofyear: torch.Tensor | None = None,
        target_days_in_year: torch.Tensor | None = None,
        step_idx: int = 0,
    ) -> torch.Tensor | None:
        if not self.enabled or self.aux_feature_dim <= 0:
            return None
        bsz = int(current.shape[0])
        device = current.device
        dtype = current.dtype
        height, width = int(current.shape[-2]), int(current.shape[-1])
        num_nodes = height * width
        parts: list[torch.Tensor] = []

        if self._lat_lon_node_features is not None:
            parts.append(self._lat_lon_node_features.to(device=device, dtype=dtype).unsqueeze(0).expand(bsz, -1, -1))

        if self.settings.orography:
            if self._static_node_features is not None and "orography" in self._static_feature_columns:
                orog_pos = self._static_feature_columns["orography"]
                parts.append(
                    self._static_node_features[:, orog_pos : orog_pos + 1]
                    .to(device=device, dtype=dtype)
                    .unsqueeze(0)
                    .expand(bsz, -1, -1)
                )
            elif "orog" in self._state_static_variables:
                current_orog = self._current_state_feature(current, "orog")
                if current_orog is not None:
                    parts.append(current_orog.to(device=device, dtype=dtype))

        if self.settings.land_sea_mask:
            if self._static_node_features is not None and "land_sea_mask" in self._static_feature_columns:
                lsm_pos = self._static_feature_columns["land_sea_mask"]
                parts.append(
                    self._static_node_features[:, lsm_pos : lsm_pos + 1]
                    .to(device=device, dtype=dtype)
                    .unsqueeze(0)
                    .expand(bsz, -1, -1)
                )
            elif "lsm" in self._state_static_variables:
                current_lsm = self._current_state_feature(current, "lsm")
                if current_lsm is not None:
                    parts.append(current_lsm.to(device=device, dtype=dtype))

        if self.settings.dayofyear_sincos:
            if target_dayofyear is None or target_days_in_year is None:
                if not self._warned_missing_time:
                    self.logger.warning(
                        "WARNING: day-of-year auxiliary features requested but target timestamp metadata is missing; using zeros."
                    )
                    self._warned_missing_time = True
                doy_features = torch.zeros((bsz, 2), dtype=dtype, device=device)
            else:
                doy = target_dayofyear
                days = target_days_in_year
                if doy.dim() == 2:
                    doy = doy[:, int(step_idx)]
                if days.dim() == 2:
                    days = days[:, int(step_idx)]
                doy_features = dayofyear_sincos(doy, days, dtype=dtype, device=device)
            parts.append(doy_features.unsqueeze(1).expand(bsz, num_nodes, 2))

        if self.settings.known_forcings_enabled:
            if target_norm is None:
                raise ValueError("Known forcing features require target_norm for the current rollout step.")
            for variable in self.settings.known_forcing_variables:
                canonical = canonical_name(variable) or variable
                local_idx = self.known_forcing_channels.get(canonical)
                if local_idx is None:
                    continue
                values = target_norm[:, int(local_idx) : int(local_idx) + 1]
                parts.append(values.permute(0, 2, 3, 1).reshape(bsz, num_nodes, 1))

        if not parts:
            return None
        features = torch.cat(parts, dim=-1)
        if int(features.shape[-1]) != int(self.aux_feature_dim):
            raise RuntimeError(
                f"Auxiliary feature dimension mismatch: built {features.shape[-1]} but expected {self.aux_feature_dim}. "
                f"Feature names: {self.feature_names}"
            )
        return features

    def apply_overrides(
        self,
        pred: torch.Tensor,
        *,
        current: torch.Tensor,
        target_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled:
            return pred
        out = pred
        changed = False
        for _, idx in self.copy_channels.items():
            if 0 <= int(idx) < pred.shape[1]:
                if not changed:
                    out = pred.clone()
                    changed = True
                out[:, int(idx)] = current[:, int(idx)]
        if self.known_forcing_channels:
            if target_norm is None:
                raise ValueError("Known forcing override requires target_norm.")
            for _, idx in self.known_forcing_channels.items():
                if 0 <= int(idx) < pred.shape[1]:
                    if not changed:
                        out = pred.clone()
                        changed = True
                    out[:, int(idx)] = target_norm[:, int(idx)]
        return out

    def log_startup(self, base_input_channels: int, total_input_channels: int, output_channels: int) -> None:
        self.logger.info("Extra features enabled: %s", bool(self.enabled))
        if self.enabled:
            self.logger.info("Auxiliary features:")
            for name in self.feature_names:
                self.logger.info("  - %s", name)
        self.logger.info("Base input channels: %d", int(base_input_channels))
        self.logger.info("Auxiliary input channels: %d", int(self.aux_feature_dim))
        self.logger.info("Total input channels: %d", int(total_input_channels))
        self.logger.info("Output channels: %d", int(output_channels))
        if self.known_forcing_channels:
            self.logger.info("Known forcing variables:")
            for name, idx in self.known_forcing_channels.items():
                self.logger.info("  %s -> channel %d", name, idx)
        if self.copy_channels:
            self.logger.info("Copied static variables:")
            for name, idx in self.copy_channels.items():
                self.logger.info("  %s -> channel %d", name, idx)
        if self.exclude_loss_channels:
            self.logger.info("Excluded from loss: %s", ", ".join(self.exclude_loss_channels))


def feature_metadata_matches(active: dict[str, Any], checkpoint: dict[str, Any]) -> tuple[bool, str]:
    active_enabled = bool(active.get("extra_features_enabled", False))
    checkpoint_has_feature_meta = "extra_features_enabled" in checkpoint or "total_input_channels" in checkpoint
    checkpoint_enabled = bool(checkpoint.get("extra_features_enabled", False))
    if active_enabled and not checkpoint_has_feature_meta:
        return False, "active config enables extra_features, but checkpoint has no feature metadata"
    if active_enabled != checkpoint_enabled:
        return False, (
            f"extra_features mismatch: active={active_enabled} checkpoint={checkpoint_enabled}. "
            "Use a checkpoint trained with the same feature configuration."
        )
    if active_enabled:
        for key in ("aux_feature_dim", "base_input_channels", "total_input_channels"):
            if int(active.get(key, -1)) != int(checkpoint.get(key, -2)):
                return False, f"feature metadata mismatch for {key}: active={active.get(key)} checkpoint={checkpoint.get(key)}"
    return True, ""
