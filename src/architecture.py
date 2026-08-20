from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MODEL_CONFIG_KEYS = {
    "hierarchy_type",
    "use_l4_ratio15",
    "level_shapes",
    "use_l4",
    "input_channels",
    "output_channels",
    "hidden_dim",
    "level_dims",
    "level_heads",
    "edge_dim",
    "num_heads",
    "heads",
    "head_dim",
    "k_neighbors",
    "level_k_neighbors",
    "l3_k_neighbors",
    "skip_fusion",
    "pooling",
    "l0_refine",
    "lead_conditioning",
    "mesh_encoder",
    "edge_encoding",
    "boundary_mlp",
    "head_init_std",
    "encoder_blocks",
    "decoder_blocks",
    "l0_blocks",
    "l1_blocks",
    "l2_blocks",
    "l1_refine_blocks",
    "l0_refine_blocks",
    "num_graph_levels",
    "use_l3",
    "l3_blocks",
    "l4_blocks",
    "l3_refine_after_l4_blocks",
    "l2_refine_after_l3_blocks",
    "l1_refine_after_l2_blocks",
}

SUPPORTED_SKIP_FUSION_TYPES = {"default", "scalar_gated", "sum"}
SUPPORTED_POOLING_TYPES = {"default", "parent_index_meanmax", "scalar_gated_meanmax"}
SUPPORTED_POOLING_MEAN_TYPES = {"mean", "area_weighted"}
SUPPORTED_L0_REFINE_TYPES = {"attention", "nodewise_mlp"}
SUPPORTED_LEAD_CONDITIONING_TYPES = {"sincos_concat"}
SUPPORTED_HIERARCHY_TYPES = {"standard", "ratio15_l4", "l4_72_36_24_18_9"}
SUPPORTED_BIPARTITE_AGGREGATIONS = {"sum", "mean"}
SUPPORTED_MESH_BOUNDARY_TYPES = {"legacy", "graphcast_mlp", "fixed_spherical"}
SUPPORTED_MESH_COARSE_CONNECTIVITY = {"native_icosphere", "full_m1"}
MESH_ENCODER_KEYS = {
    "enabled",
    "refinement",
    "g2m_radius_factor",
    "mlp_hidden_ratio",
    "aggregation",
    "boundary_type",
    "grid_skip_mlp",
    "grid_attention_encoder_blocks",
    "grid_attention_decoder_blocks",
    "grid_attention_k_neighbors",
    "coarse_level_connectivity",
    # HEALPix mesh (mesh_type: healpix). See SUPPORTED_MESH_TYPES.
    "mesh_type",
    "nside",
    "regrid_oversample",
}
SUPPORTED_MESH_TYPES = {"icosphere", "healpix"}
# Keys that only mean something for an icosphere mesh; setting them alongside
# mesh_type: healpix is a config mistake, not a harmless extra.
ICOSPHERE_ONLY_MESH_ENCODER_KEYS = {
    "refinement",
    "g2m_radius_factor",
    "coarse_level_connectivity",
}
# ...and the reverse.
HEALPIX_ONLY_MESH_ENCODER_KEYS = {"nside", "regrid_oversample"}


@dataclass(frozen=True)
class SkipFusionConfig:
    type: str = "default"
    init_scale: float = 1.0
    max_scale: float = 2.0

    def asdict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "init_scale": float(self.init_scale),
            "max_scale": float(self.max_scale),
        }


@dataclass(frozen=True)
class PoolingConfig:
    type: str = "default"
    init_scale: float = 1.0
    max_scale: float = 2.0
    mean_type: str = "mean"
    include_max: bool = True

    def asdict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "init_scale": float(self.init_scale),
            "max_scale": float(self.max_scale),
            "mean_type": self.mean_type,
            "include_max": bool(self.include_max),
        }


@dataclass(frozen=True)
class LeadConditioningConfig:
    enabled: bool = False
    type: str = "none"
    max_lead: int = 0

    @property
    def added_input_channels(self) -> int:
        return 2 if self.enabled and self.type == "sincos_concat" else 0

    def asdict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "type": self.type,
            "max_lead": int(self.max_lead),
            "added_input_channels": int(self.added_input_channels),
        }


@dataclass(frozen=True)
class MeshEncoderConfig:
    enabled: bool = False
    refinement: int = 5
    g2m_radius_factor: float = 0.6
    mlp_hidden_ratio: int = 2
    aggregation: str = "sum"
    boundary_type: str = "legacy"
    grid_skip_mlp: bool = False
    grid_attention_encoder_blocks: int = 0
    grid_attention_decoder_blocks: int = 0
    grid_attention_k_neighbors: int = 8
    coarse_level_connectivity: str = "native_icosphere"
    mesh_type: str = "icosphere"
    nside: int = 32
    regrid_oversample: int = 8

    def asdict(self) -> dict[str, Any]:
        resolved = {
            "enabled": bool(self.enabled),
            "refinement": int(self.refinement),
            "g2m_radius_factor": float(self.g2m_radius_factor),
            "mlp_hidden_ratio": int(self.mlp_hidden_ratio),
            "aggregation": self.aggregation,
            "boundary_type": self.boundary_type,
            "grid_skip_mlp": bool(self.grid_skip_mlp),
            "grid_attention_encoder_blocks": int(self.grid_attention_encoder_blocks),
            "grid_attention_decoder_blocks": int(self.grid_attention_decoder_blocks),
            "grid_attention_k_neighbors": int(self.grid_attention_k_neighbors),
            "coarse_level_connectivity": self.coarse_level_connectivity,
        }
        if self.mesh_type == "icosphere":
            return resolved
        # HEALPix. The mesh keys are emitted only here, for the same reason the whole
        # mesh_encoder block is emitted only when configured: every existing icosphere
        # config must resolve byte-for-byte as before.
        #
        # The icosphere-only keys are DROPPED rather than left at their defaults, and
        # that is load-bearing, not tidiness: resolve_mesh_encoder is applied to its
        # own output (architecture_metadata and validate_checkpoint_architecture both
        # re-resolve an already-resolved config), so emitting `refinement: 5` here
        # would make the next pass see an explicitly-set icosphere key and reject it.
        # resolve_mesh_encoder must be idempotent.
        for key in ICOSPHERE_ONLY_MESH_ENCODER_KEYS:
            resolved.pop(key, None)
        resolved.update(
            {
                "mesh_type": self.mesh_type,
                "nside": int(self.nside),
                "regrid_oversample": int(self.regrid_oversample),
            }
        )
        return resolved


@dataclass(frozen=True)
class L0RefineConfig:
    type: str = "attention"
    mlp_expansion: int = 2
    dropout: float = 0.0
    residual_scale_init: float = 0.1
    learnable_residual_scale: bool = True

    def asdict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "mlp_expansion": int(self.mlp_expansion),
            "dropout": float(self.dropout),
            "residual_scale_init": float(self.residual_scale_init),
            "learnable_residual_scale": bool(self.learnable_residual_scale),
        }


@dataclass(frozen=True)
class GraphArchitectureConfig:
    hierarchy_type: str = "standard"
    use_l4_ratio15: bool = False
    hidden_dim: int = 96
    num_heads: int = 4
    head_dim: int = 24
    k_neighbors: int = 8
    level_k_neighbors: tuple[int, ...] = (8, 8, 8)
    skip_fusion: SkipFusionConfig = SkipFusionConfig()
    pooling: PoolingConfig = PoolingConfig()
    l0_refine: L0RefineConfig = L0RefineConfig()
    lead_conditioning: LeadConditioningConfig = LeadConditioningConfig()
    num_graph_levels: int = 3
    use_l3: bool = False
    use_l4: bool = False
    l0_blocks: int = 2
    l1_blocks: int = 2
    l2_blocks: int = 1
    l3_blocks: int = 1
    l4_blocks: int = 1
    l3_refine_after_l4_blocks: int = 1
    l2_refine_after_l3_blocks: int = 1
    l1_refine_blocks: int = 1
    l0_refine_blocks: int = 1

    def asdict(self) -> dict[str, Any]:
        data = asdict(self)
        data["level_k_neighbors"] = list(self.level_k_neighbors)
        data["skip_fusion"] = self.skip_fusion.asdict()
        data["pooling"] = self.pooling.asdict()
        data["l0_refine"] = self.l0_refine.asdict()
        data["lead_conditioning"] = self.lead_conditioning.asdict()
        return data


_MISSING = object()


def _get_value(source: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(source, dict):
        return source.get(key, default)
    return getattr(source, key, default)


def _extract_settings(source: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    model = _get_value(source, "model", None)
    if isinstance(model, dict):
        for key, value in model.items():
            if key in MODEL_CONFIG_KEYS:
                settings[key] = value
    for key in MODEL_CONFIG_KEYS:
        value = _get_value(source, key, _MISSING)
        if value is _MISSING:
            continue
        if key in settings and settings[key] != value:
            raise ValueError(
                f"Conflicting model config for {key}: model.{key}={settings[key]!r}, top-level {key}={value!r}."
            )
        settings[key] = value
    return settings


def _to_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _positive_int(value: Any, name: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise ValueError(f"{name} must be >= 1, got {ivalue}.")
    return ivalue


def _non_negative_int(value: Any, name: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise ValueError(f"{name} must be >= 0, got {ivalue}.")
    return ivalue


def _positive_int_list(value: Any, name: str) -> list[int]:
    if isinstance(value, str):
        chunks = [chunk for chunk in value.replace(",", " ").split() if chunk]
    else:
        chunks = list(value)
    values = [_positive_int(chunk, name) for chunk in chunks]
    if not values:
        raise ValueError(f"{name} must not be empty.")
    return values


def _positive_float(value: Any, name: str) -> float:
    fvalue = float(value)
    if fvalue <= 0.0:
        raise ValueError(f"{name} must be > 0, got {fvalue}.")
    return fvalue


def _non_negative_float(value: Any, name: str) -> float:
    fvalue = float(value)
    if fvalue < 0.0:
        raise ValueError(f"{name} must be >= 0, got {fvalue}.")
    return fvalue


def _resolve_width_settings(settings: dict[str, Any]) -> tuple[int, int, int, int]:
    hidden_dim = _positive_int(settings.get("hidden_dim", 96), "hidden_dim")
    has_num_heads = "num_heads" in settings
    has_heads = "heads" in settings
    if has_num_heads and has_heads and int(settings["num_heads"]) != int(settings["heads"]):
        raise ValueError(
            f"Conflicting model config for heads: num_heads={settings['num_heads']!r}, "
            f"heads={settings['heads']!r}."
        )
    num_heads = _positive_int(settings.get("num_heads", settings.get("heads", 4)), "num_heads")
    if hidden_dim % num_heads != 0:
        raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
    head_dim = hidden_dim // num_heads
    if "head_dim" in settings and int(settings["head_dim"]) != head_dim:
        raise ValueError(
            f"head_dim={settings['head_dim']} is inconsistent with hidden_dim={hidden_dim} "
            f"and num_heads={num_heads}; expected {head_dim}."
        )
    k_neighbors = _positive_int(settings.get("k_neighbors", 8), "k_neighbors")
    return hidden_dim, num_heads, head_dim, k_neighbors


def resolve_level_k_neighbors(
    source: Any | None = None,
    num_graph_levels: int | None = None,
    k_neighbors: int | None = None,
) -> tuple[int, ...]:
    """Resolve per-level k while preserving legacy single-k defaults."""

    settings = _extract_settings(source or {})
    levels = int(num_graph_levels) if num_graph_levels is not None else int(settings.get("num_graph_levels", 3))
    if levels not in {3, 4, 5}:
        raise ValueError(f"num_graph_levels must be 3, 4, or 5, got {levels}.")
    base_k = _positive_int(
        k_neighbors if k_neighbors is not None else settings.get("k_neighbors", 8),
        "k_neighbors",
    )
    if "level_k_neighbors" in settings:
        values = _positive_int_list(settings["level_k_neighbors"], "level_k_neighbors")
        if len(values) != levels:
            raise ValueError(
                f"level_k_neighbors must have length num_graph_levels={levels}; got {len(values)}."
            )
        return tuple(values)
    if "l3_k_neighbors" in settings:
        if levels < 4:
            raise ValueError("l3_k_neighbors requires num_graph_levels=4/use_l3=true.")
        values = [base_k for _ in range(levels)]
        values[3] = _positive_int(settings["l3_k_neighbors"], "l3_k_neighbors")
        return tuple(values)
    return tuple(base_k for _ in range(levels))


def _raw_type_or_dict(source: Any, key: str) -> dict[str, Any]:
    """Coerce a ``str | dict | None`` model sub-section into a plain dict.

    Shared preamble for the type-keyed resolvers (skip_fusion, pooling,
    l0_refine): a bare string becomes ``{"type": <string>}``, a mapping is
    copied, ``None`` becomes ``{}``, and anything else raises the
    section-specific ``ValueError`` (message identical to the inlined versions).
    """
    settings = _extract_settings(source or {})
    raw = settings.get(key, None)
    if raw is None:
        return {}
    if isinstance(raw, str):
        return {"type": raw}
    if isinstance(raw, dict):
        return dict(raw)
    raise ValueError(f"{key} must be a mapping, string, or null.")


def resolve_skip_fusion(source: Any | None = None) -> SkipFusionConfig:
    raw_dict = _raw_type_or_dict(source, "skip_fusion")

    fusion_type = str(raw_dict.get("type", "default")).strip().lower()
    if fusion_type not in SUPPORTED_SKIP_FUSION_TYPES:
        available = ", ".join(sorted(SUPPORTED_SKIP_FUSION_TYPES))
        raise ValueError(f"Unsupported skip_fusion.type={fusion_type!r}. Expected one of: {available}.")
    init_scale = _positive_float(raw_dict.get("init_scale", 1.0), "skip_fusion.init_scale")
    max_scale = _positive_float(raw_dict.get("max_scale", 2.0), "skip_fusion.max_scale")
    if init_scale >= max_scale:
        raise ValueError(
            f"skip_fusion.init_scale={init_scale:g} must be smaller than "
            f"skip_fusion.max_scale={max_scale:g}."
        )
    return SkipFusionConfig(type=fusion_type, init_scale=init_scale, max_scale=max_scale)


def resolve_pooling(source: Any | None = None) -> PoolingConfig:
    raw_dict = _raw_type_or_dict(source, "pooling")

    pooling_type = str(raw_dict.get("type", "default")).strip().lower()
    if pooling_type not in SUPPORTED_POOLING_TYPES:
        available = ", ".join(sorted(SUPPORTED_POOLING_TYPES))
        raise ValueError(f"Unsupported pooling.type={pooling_type!r}. Expected one of: {available}.")
    init_scale = _positive_float(raw_dict.get("init_scale", 1.0), "pooling.init_scale")
    max_scale = _positive_float(raw_dict.get("max_scale", 2.0), "pooling.max_scale")
    mean_type = str(raw_dict.get("mean_type", "mean")).strip().lower()
    if mean_type not in SUPPORTED_POOLING_MEAN_TYPES:
        available = ", ".join(sorted(SUPPORTED_POOLING_MEAN_TYPES))
        raise ValueError(f"Unsupported pooling.mean_type={mean_type!r}. Expected one of: {available}.")
    include_max = _to_bool(raw_dict.get("include_max", True), "pooling.include_max")
    if init_scale >= max_scale:
        raise ValueError(
            f"pooling.init_scale={init_scale:g} must be smaller than "
            f"pooling.max_scale={max_scale:g}."
        )
    return PoolingConfig(
        type=pooling_type,
        init_scale=init_scale,
        max_scale=max_scale,
        mean_type=mean_type,
        include_max=include_max,
    )


def resolve_lead_conditioning(source: Any | None = None) -> LeadConditioningConfig:
    settings = _extract_settings(source or {})
    raw = settings.get("lead_conditioning", None)
    if raw is None:
        raw_dict: dict[str, Any] = {}
    elif isinstance(raw, bool):
        raw_dict = {"enabled": raw}
    elif isinstance(raw, str):
        raw_dict = {"enabled": raw.strip().lower() not in {"", "false", "0", "no", "off"}, "type": raw}
    elif isinstance(raw, dict):
        raw_dict = dict(raw)
    else:
        raise ValueError("lead_conditioning must be a mapping, boolean, string, or null.")

    enabled = _to_bool(raw_dict.get("enabled", False), "lead_conditioning.enabled")
    if not enabled:
        return LeadConditioningConfig(enabled=False, type="none", max_lead=0)

    conditioning_type = str(raw_dict.get("type", "sincos_concat")).strip().lower()
    if conditioning_type not in SUPPORTED_LEAD_CONDITIONING_TYPES:
        available = ", ".join(sorted(SUPPORTED_LEAD_CONDITIONING_TYPES))
        raise ValueError(
            f"Unsupported lead_conditioning.type={conditioning_type!r}. Expected one of: {available}."
        )
    max_lead = _positive_int(raw_dict.get("max_lead", 10), "lead_conditioning.max_lead")
    return LeadConditioningConfig(enabled=True, type=conditioning_type, max_lead=max_lead)


def resolve_mesh_encoder(source: Any | None = None) -> MeshEncoderConfig:
    """Resolve the optional GraphCast-style grid<->icosphere domain crossing.

    Absence is equivalent to ``enabled: false``. Callers that serialize resolved
    configs should only emit this block when it was explicitly configured, which
    keeps legacy resolved configs byte-for-byte unchanged.
    """

    settings = _extract_settings(source or {})
    raw = settings.get("mesh_encoder", None)
    if raw is None:
        raw_dict: dict[str, Any] = {}
    elif isinstance(raw, dict):
        raw_dict = dict(raw)
    else:
        raise ValueError("mesh_encoder must be a mapping or null.")
    unknown = sorted(set(raw_dict) - MESH_ENCODER_KEYS)
    if unknown:
        raise ValueError(f"Unsupported mesh_encoder keys: {unknown}. Expected: {sorted(MESH_ENCODER_KEYS)}.")
    aggregation = str(raw_dict.get("aggregation", "sum")).strip().lower()
    if aggregation not in SUPPORTED_BIPARTITE_AGGREGATIONS:
        available = ", ".join(sorted(SUPPORTED_BIPARTITE_AGGREGATIONS))
        raise ValueError(
            f"Unsupported mesh_encoder.aggregation={aggregation!r}. Expected one of: {available}."
        )
    boundary_type = str(raw_dict.get("boundary_type", "legacy")).strip().lower()
    if boundary_type not in SUPPORTED_MESH_BOUNDARY_TYPES:
        available = ", ".join(sorted(SUPPORTED_MESH_BOUNDARY_TYPES))
        raise ValueError(
            f"Unsupported mesh_encoder.boundary_type={boundary_type!r}. Expected one of: {available}."
        )
    enabled = _to_bool(raw_dict.get("enabled", False), "mesh_encoder.enabled")
    grid_skip_mlp = _to_bool(
        raw_dict.get("grid_skip_mlp", False),
        "mesh_encoder.grid_skip_mlp",
    )
    grid_attention_encoder_blocks = _non_negative_int(
        raw_dict.get("grid_attention_encoder_blocks", 0),
        "mesh_encoder.grid_attention_encoder_blocks",
    )
    grid_attention_decoder_blocks = _non_negative_int(
        raw_dict.get("grid_attention_decoder_blocks", 0),
        "mesh_encoder.grid_attention_decoder_blocks",
    )
    grid_attention_k_neighbors = _positive_int(
        raw_dict.get("grid_attention_k_neighbors", 8),
        "mesh_encoder.grid_attention_k_neighbors",
    )
    if grid_skip_mlp and not enabled:
        raise ValueError("mesh_encoder.grid_skip_mlp=true requires mesh_encoder.enabled=true.")
    if grid_skip_mlp and boundary_type != "graphcast_mlp":
        raise ValueError(
            "mesh_encoder.grid_skip_mlp=true requires "
            "mesh_encoder.boundary_type='graphcast_mlp'."
        )
    if (grid_attention_encoder_blocks or grid_attention_decoder_blocks) and not enabled:
        raise ValueError(
            "mesh_encoder grid-attention blocks require mesh_encoder.enabled=true."
        )
    if boundary_type == "fixed_spherical" and grid_skip_mlp:
        raise ValueError(
            "mesh_encoder.boundary_type='fixed_spherical' is parameter-free and "
            "does not support grid_skip_mlp."
        )
    # 'fixed_spherical' + grid-attention blocks is supported and changes what the
    # fixed remap carries: without blocks it interpolates RAW channels onto the mesh
    # and interpolates the output delta back; with blocks the grid is embedded first,
    # the remap moves HIDDEN features both ways, and the head runs on the grid. The
    # conservative weights themselves are identical either way. See models.py
    # (grid_message_passing) and mesh_builder.build_healpix_mesh_bundle.
    coarse_level_connectivity = str(
        raw_dict.get("coarse_level_connectivity", "native_icosphere")
    ).strip().lower()
    if coarse_level_connectivity not in SUPPORTED_MESH_COARSE_CONNECTIVITY:
        available = ", ".join(sorted(SUPPORTED_MESH_COARSE_CONNECTIVITY))
        raise ValueError(
            "Unsupported mesh_encoder.coarse_level_connectivity="
            f"{coarse_level_connectivity!r}. Expected one of: {available}."
        )
    mesh_type = str(raw_dict.get("mesh_type", "icosphere")).strip().lower()
    if mesh_type not in SUPPORTED_MESH_TYPES:
        available = ", ".join(sorted(SUPPORTED_MESH_TYPES))
        raise ValueError(
            f"Unsupported mesh_encoder.mesh_type={mesh_type!r}. Expected one of: {available}."
        )
    nside = _positive_int(raw_dict.get("nside", 32), "mesh_encoder.nside")
    regrid_oversample = _positive_int(
        raw_dict.get("regrid_oversample", 8),
        "mesh_encoder.regrid_oversample",
    )
    if mesh_type == "healpix":
        if nside & (nside - 1):
            raise ValueError(
                f"mesh_encoder.nside must be a power of two, got {nside}: HEALPix "
                "pooling is a NEST quadtree (parent = pixel >> 2)."
            )
        if boundary_type != "fixed_spherical":
            raise ValueError(
                "mesh_encoder.mesh_type='healpix' requires boundary_type='fixed_spherical'; "
                f"the conservative regrid is a fixed, parameter-free remap, got {boundary_type!r}."
            )
        conflicting = sorted(set(raw_dict) & ICOSPHERE_ONLY_MESH_ENCODER_KEYS)
        if conflicting:
            raise ValueError(
                f"mesh_encoder keys {conflicting} only apply to mesh_type='icosphere'; "
                "a HEALPix mesh has no refinement level, no radius rule and no icosphere "
                "connectivity. Remove them."
            )
    else:
        conflicting = sorted(set(raw_dict) & HEALPIX_ONLY_MESH_ENCODER_KEYS)
        if conflicting:
            raise ValueError(
                f"mesh_encoder keys {conflicting} require mesh_type='healpix'."
            )
    return MeshEncoderConfig(
        enabled=enabled,
        refinement=_positive_int(raw_dict.get("refinement", 5), "mesh_encoder.refinement"),
        g2m_radius_factor=_positive_float(
            raw_dict.get("g2m_radius_factor", 0.6),
            "mesh_encoder.g2m_radius_factor",
        ),
        mlp_hidden_ratio=_positive_int(
            raw_dict.get("mlp_hidden_ratio", 2),
            "mesh_encoder.mlp_hidden_ratio",
        ),
        aggregation=aggregation,
        boundary_type=boundary_type,
        grid_skip_mlp=grid_skip_mlp,
        grid_attention_encoder_blocks=grid_attention_encoder_blocks,
        grid_attention_decoder_blocks=grid_attention_decoder_blocks,
        grid_attention_k_neighbors=grid_attention_k_neighbors,
        coarse_level_connectivity=coarse_level_connectivity,
        mesh_type=mesh_type,
        nside=nside,
        regrid_oversample=regrid_oversample,
    )


def resolve_l0_refine(source: Any | None = None) -> L0RefineConfig:
    raw_dict = _raw_type_or_dict(source, "l0_refine")

    refine_type = str(raw_dict.get("type", "attention")).strip().lower()
    if refine_type not in SUPPORTED_L0_REFINE_TYPES:
        available = ", ".join(sorted(SUPPORTED_L0_REFINE_TYPES))
        raise ValueError(f"Unsupported l0_refine.type={refine_type!r}. Expected one of: {available}.")

    mlp_expansion = _positive_int(raw_dict.get("mlp_expansion", 2), "l0_refine.mlp_expansion")
    dropout = _non_negative_float(raw_dict.get("dropout", 0.0), "l0_refine.dropout")
    if dropout >= 1.0:
        raise ValueError(f"l0_refine.dropout must be < 1, got {dropout}.")
    residual_scale_init = float(raw_dict.get("residual_scale_init", 0.1))
    learnable_residual_scale = _to_bool(
        raw_dict.get("learnable_residual_scale", True),
        "l0_refine.learnable_residual_scale",
    )
    return L0RefineConfig(
        type=refine_type,
        mlp_expansion=mlp_expansion,
        dropout=dropout,
        residual_scale_init=residual_scale_init,
        learnable_residual_scale=learnable_residual_scale,
    )


def resolve_graph_architecture(source: Any | None = None) -> GraphArchitectureConfig:
    """Resolve the graph U-Net level configuration.

    Defaults are intentionally the legacy 3-level model. If only one of
    use_l3/num_graph_levels is provided, the other is derived from it. If both
    are provided and disagree, fail early with a config error. A five-level
    model still has an L3 path; use_l3 therefore means "includes L3".
    """

    settings = _extract_settings(source or {})
    has_num_levels = "num_graph_levels" in settings
    has_use_l3 = "use_l3" in settings
    has_use_l4 = "use_l4" in settings
    has_use_l4_ratio15 = "use_l4_ratio15" in settings

    hierarchy_type = str(settings.get("hierarchy_type", "standard")).strip().lower()
    if hierarchy_type in {"", "none"}:
        hierarchy_type = "standard"
    use_l4_ratio15 = _to_bool(settings["use_l4_ratio15"], "use_l4_ratio15") if has_use_l4_ratio15 else False
    if use_l4_ratio15 and hierarchy_type == "standard":
        hierarchy_type = "ratio15_l4"
    if hierarchy_type not in SUPPORTED_HIERARCHY_TYPES:
        available = ", ".join(sorted(SUPPORTED_HIERARCHY_TYPES))
        raise ValueError(f"Unsupported hierarchy_type={hierarchy_type!r}. Expected one of: {available}.")
    if hierarchy_type == "ratio15_l4":
        use_l4_ratio15 = True

    num_graph_levels = int(settings["num_graph_levels"]) if has_num_levels else None
    if num_graph_levels is not None and num_graph_levels not in {3, 4, 5}:
        raise ValueError(f"num_graph_levels must be 3, 4, or 5, got {num_graph_levels}.")
    if use_l4_ratio15:
        if num_graph_levels is None:
            num_graph_levels = 5
        elif num_graph_levels != 5:
            raise ValueError(
                f"hierarchy_type='ratio15_l4' requires num_graph_levels=5, got {num_graph_levels}."
            )
    elif hierarchy_type == "l4_72_36_24_18_9":
        if num_graph_levels is None:
            num_graph_levels = 5
        elif num_graph_levels != 5:
            raise ValueError(
                f"hierarchy_type='l4_72_36_24_18_9' requires num_graph_levels=5, got {num_graph_levels}."
            )
    use_l3 = _to_bool(settings["use_l3"], "use_l3") if has_use_l3 else None
    use_l4 = _to_bool(settings["use_l4"], "use_l4") if has_use_l4 else None

    if num_graph_levels is not None and use_l3 is not None:
        if (num_graph_levels >= 4) != bool(use_l3):
            raise ValueError(
                f"Conflicting model graph-level config: num_graph_levels={num_graph_levels}, use_l3={use_l3}."
            )
    if num_graph_levels is not None and use_l4 is not None:
        if (num_graph_levels >= 5) != bool(use_l4):
            raise ValueError(
                f"Conflicting model graph-level config: num_graph_levels={num_graph_levels}, use_l4={use_l4}."
            )
    elif num_graph_levels is not None:
        use_l3 = num_graph_levels >= 4
    elif use_l3 is not None:
        num_graph_levels = 4 if use_l3 else 3
    elif use_l4 is not None:
        num_graph_levels = 5 if use_l4 else 3
    else:
        num_graph_levels = 3
        use_l3 = False
    if use_l4 is None:
        use_l4 = int(num_graph_levels) >= 5
    if use_l4 and not use_l3:
        use_l3 = True

    hidden_dim, num_heads, head_dim, k_neighbors = _resolve_width_settings(settings)
    level_k_neighbors = resolve_level_k_neighbors(
        settings,
        num_graph_levels=int(num_graph_levels),
        k_neighbors=k_neighbors,
    )
    skip_fusion = resolve_skip_fusion(settings)
    pooling = resolve_pooling(settings)
    l0_refine_config = resolve_l0_refine(settings)
    lead_conditioning = resolve_lead_conditioning(settings)
    l0_blocks = _positive_int(settings.get("l0_blocks", 2), "l0_blocks")
    l1_blocks = _positive_int(settings.get("l1_blocks", 2), "l1_blocks")
    l2_blocks = _positive_int(settings.get("l2_blocks", 1), "l2_blocks")
    l3_blocks = _positive_int(settings.get("l3_blocks", 1), "l3_blocks")
    l4_blocks = _positive_int(settings.get("l4_blocks", 1), "l4_blocks")
    l3_refine_after_l4 = _positive_int(
        settings.get("l3_refine_after_l4_blocks", 1),
        "l3_refine_after_l4_blocks",
    )
    l2_refine = _positive_int(
        settings.get("l2_refine_after_l3_blocks", 1),
        "l2_refine_after_l3_blocks",
    )
    if "l1_refine_blocks" in settings and "l1_refine_after_l2_blocks" in settings:
        if int(settings["l1_refine_blocks"]) != int(settings["l1_refine_after_l2_blocks"]):
            raise ValueError(
                "Conflicting model config for L1 refinement: "
                f"l1_refine_blocks={settings['l1_refine_blocks']!r}, "
                f"l1_refine_after_l2_blocks={settings['l1_refine_after_l2_blocks']!r}."
            )
    l1_refine = _positive_int(
        settings.get("l1_refine_blocks", settings.get("l1_refine_after_l2_blocks", 1)),
        "l1_refine_blocks",
    )
    l0_refine = _positive_int(settings.get("l0_refine_blocks", 1), "l0_refine_blocks")
    return GraphArchitectureConfig(
        hierarchy_type=hierarchy_type,
        use_l4_ratio15=bool(use_l4_ratio15),
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        k_neighbors=k_neighbors,
        level_k_neighbors=level_k_neighbors,
        skip_fusion=skip_fusion,
        pooling=pooling,
        l0_refine=l0_refine_config,
        lead_conditioning=lead_conditioning,
        num_graph_levels=int(num_graph_levels),
        use_l3=bool(use_l3),
        use_l4=bool(use_l4),
        l0_blocks=l0_blocks,
        l1_blocks=l1_blocks,
        l2_blocks=l2_blocks,
        l3_blocks=l3_blocks,
        l4_blocks=l4_blocks,
        l3_refine_after_l4_blocks=l3_refine_after_l4,
        l2_refine_after_l3_blocks=l2_refine,
        l1_refine_blocks=l1_refine,
        l0_refine_blocks=l0_refine,
    )


def normalize_model_config_dict(params: dict[str, Any]) -> dict[str, Any]:
    """Flatten optional nested model config while preserving old top-level keys."""

    resolved = dict(params)
    model = resolved.get("model", None)
    has_mesh_encoder = "mesh_encoder" in resolved or (
        isinstance(model, dict) and "mesh_encoder" in model
    )
    if isinstance(model, dict):
        for key, value in model.items():
            if key not in MODEL_CONFIG_KEYS:
                continue
            # Nested model settings are the canonical architecture override.
            # This lets a YAML section inherit legacy top-level defaults while
            # specializing the Graph U-Net block layout under model:.
            resolved[key] = value

    arch = resolve_graph_architecture(resolved)
    resolved.update(arch.asdict())
    model_dict = dict(model or {})
    model_dict.update(arch.asdict())
    if has_mesh_encoder:
        mesh_encoder = resolve_mesh_encoder(resolved).asdict()
        resolved["mesh_encoder"] = mesh_encoder
        model_dict["mesh_encoder"] = dict(mesh_encoder)
        if bool(mesh_encoder["enabled"]):
            levels = int(arch.num_graph_levels)
            if mesh_encoder.get("mesh_type", "icosphere") == "healpix":
                # HEALPix quadtree: 12*nside^2 pixels, halving nside per level.
                nside = int(mesh_encoder["nside"])
                coarsest_nside = nside >> (levels - 1)
                if coarsest_nside < 1:
                    raise ValueError(
                        f"mesh_encoder.nside={nside} is too small for "
                        f"num_graph_levels={levels}; the coarsest level would have "
                        f"nside={coarsest_nside}. Need nside >= 2**{levels - 1}."
                    )
                mesh_node_counts = [12 * (nside >> idx) ** 2 for idx in range(levels)]
                # True HEALPix adjacency is 8 slots per pixel (24 pixels have 7 real
                # neighbours and one masked self-pad), so 8 is the default on every
                # level. Unlike the icosphere branch this does NOT force the mesh
                # degree: an explicit level_k_neighbors is honoured so a densified
                # coarsest level (the dense-L3K24 recipe) can be reproduced on a
                # HEALPix mesh. Levels at k=8 keep true adjacency; a level asking for
                # more falls back to spherical kNN, matching graph_builder's
                # per-level dispatch on the HEALPix-native path.
                if "level_k_neighbors" in resolved:
                    level_k_neighbors = [int(x) for x in arch.level_k_neighbors]
                    if len(level_k_neighbors) != levels:
                        raise ValueError(
                            f"level_k_neighbors must have length num_graph_levels={levels}; "
                            f"got {len(level_k_neighbors)}."
                        )
                    for idx, (level_k, count) in enumerate(
                        zip(level_k_neighbors, mesh_node_counts)
                    ):
                        if level_k < 8:
                            raise ValueError(
                                f"level_k_neighbors[{idx}]={level_k} is below the HEALPix "
                                "mesh degree; use 8 for true adjacency or more than 8 to "
                                "densify that level with spherical kNN."
                            )
                        if level_k >= count:
                            raise ValueError(
                                f"level_k_neighbors[{idx}]={level_k} needs fewer neighbours "
                                f"than that level has pixels ({count})."
                            )
                else:
                    level_k_neighbors = [8] * levels
                # Must match mesh_builder.HEALPIX_MESH_FORMAT_VERSION; pinned by
                # tests/test_healpix_mesh.py so the two cannot drift.
                mesh_format_version: Any = "hpx_mesh_v1"
            else:
                refinement = int(mesh_encoder["refinement"])
                if refinement < levels - 1:
                    raise ValueError(
                        f"mesh_encoder.refinement={refinement} is too small for "
                        f"num_graph_levels={levels}; expected at least {levels - 1}."
                    )
                mesh_node_counts = [10 * (4 ** (refinement - idx)) + 2 for idx in range(levels)]
                level_k_neighbors = [6] * levels
                if mesh_encoder["coarse_level_connectivity"] == "full_m1":
                    coarsest_refinement = refinement - (levels - 1)
                    if coarsest_refinement != 1:
                        raise ValueError(
                            "mesh_encoder.coarse_level_connectivity='full_m1' requires "
                            f"the coarsest level to be M1, got M{coarsest_refinement} "
                            f"(refinement={refinement}, num_graph_levels={levels})."
                        )
                    level_k_neighbors[-1] = mesh_node_counts[-1] - 1
                mesh_format_version = 1
            resolved["node_counts"] = mesh_node_counts
            resolved["level_k_neighbors"] = level_k_neighbors
            model_dict["level_k_neighbors"] = list(level_k_neighbors)
            resolved["edge_counts"] = [
                count * level_k
                for count, level_k in zip(mesh_node_counts, level_k_neighbors)
            ]
            resolved["mesh_format_version"] = mesh_format_version
    if arch.lead_conditioning.enabled:
        output_channels = int(resolved.get("output_channels", model_dict.get("output_channels", 67)))
        n_history = int(resolved.get("n_history", 1))
        expected_input_channels = (n_history + 1) * output_channels + arch.lead_conditioning.added_input_channels
        configured_input_channels = resolved.get("input_channels", model_dict.get("input_channels", None))
        if configured_input_channels is not None and int(configured_input_channels) != expected_input_channels:
            raise ValueError(
                f"lead_conditioning adds {arch.lead_conditioning.added_input_channels} input channels; "
                f"expected input_channels={expected_input_channels}, got {configured_input_channels}."
            )
        resolved["input_channels"] = expected_input_channels
        model_dict["input_channels"] = expected_input_channels
    resolved["model"] = model_dict
    return resolved


def checkpoint_architecture_metadata(metadata: dict[str, Any] | None) -> GraphArchitectureConfig:
    """Read architecture metadata, treating legacy checkpoints as 3-level."""

    metadata = metadata or {}
    if "num_graph_levels" not in metadata and "use_l3" not in metadata:
        return GraphArchitectureConfig()
    return resolve_graph_architecture(metadata)


def architecture_metadata(
    source: Any | None = None,
    graph_metadata: dict[str, Any] | None = None,
    num_parameters: int | None = None,
) -> dict[str, Any]:
    arch = resolve_graph_architecture(source)
    metadata = arch.asdict()
    if graph_metadata:
        for key in ("level_shapes", "node_counts", "edge_counts"):
            if key in graph_metadata:
                metadata[key] = graph_metadata[key]
    mesh_encoder = resolve_mesh_encoder(source)
    if mesh_encoder.enabled:
        metadata["graph_mode"] = "mesh"
        metadata["mesh_encoder"] = mesh_encoder.asdict()
        if graph_metadata:
            for key in (
                "refinement",
                "g2m_radius_factor",
                "coarse_level_connectivity",
                "mesh_format_version",
            ):
                if key in graph_metadata:
                    metadata[key] = graph_metadata[key]
    for key in ("input_channels", "output_channels"):
        value = _get_value(source, key, _MISSING)
        if value is not _MISSING:
            metadata[key] = int(value)
    if num_parameters is not None:
        metadata["num_parameters"] = int(num_parameters)
    return metadata


def validate_checkpoint_architecture(
    checkpoint_metadata: dict[str, Any] | None,
    current: Any,
) -> None:
    validate_checkpoint_graph_mode(checkpoint_metadata, current)
    current_mesh = resolve_mesh_encoder(current)
    checkpoint_mesh_raw = checkpoint_metadata.get("mesh_encoder", {}) if checkpoint_metadata else {}
    if current_mesh.enabled:
        checkpoint_mesh = resolve_mesh_encoder({"mesh_encoder": checkpoint_mesh_raw})
        if checkpoint_mesh != current_mesh:
            raise RuntimeError(
                "Checkpoint architecture mismatch: "
                f"checkpoint mesh_encoder={checkpoint_mesh.asdict()!r} but current config has "
                f"mesh_encoder={current_mesh.asdict()!r}. Train from scratch or use explicit partial initialization."
            )

    checkpoint_arch = checkpoint_architecture_metadata(checkpoint_metadata)
    current_arch = resolve_graph_architecture(current)
    if checkpoint_arch != current_arch:
        checkpoint_values = checkpoint_arch.asdict()
        current_values = current_arch.asdict()
        mismatches: list[str] = []
        for key in checkpoint_values:
            if checkpoint_values[key] == current_values[key]:
                continue
            if key == "skip_fusion":
                checkpoint_skip = dict(checkpoint_values[key])
                current_skip = dict(current_values[key])
                for skip_key in ("type", "init_scale", "max_scale"):
                    if checkpoint_skip.get(skip_key) != current_skip.get(skip_key):
                        mismatches.append(
                            f"checkpoint skip_fusion.{skip_key}={checkpoint_skip.get(skip_key)!r} "
                            f"but current config has skip_fusion.{skip_key}={current_skip.get(skip_key)!r}"
                        )
                continue
            if key == "pooling":
                checkpoint_pooling = dict(checkpoint_values[key])
                current_pooling = dict(current_values[key])
                for pooling_key in ("type", "init_scale", "max_scale", "mean_type", "include_max"):
                    if checkpoint_pooling.get(pooling_key) != current_pooling.get(pooling_key):
                        mismatches.append(
                            f"checkpoint pooling.{pooling_key}={checkpoint_pooling.get(pooling_key)!r} "
                            f"but current config has pooling.{pooling_key}={current_pooling.get(pooling_key)!r}"
                        )
                continue
            if key == "lead_conditioning":
                checkpoint_lead = dict(checkpoint_values[key])
                current_lead = dict(current_values[key])
                for lead_key in ("enabled", "type", "max_lead", "added_input_channels"):
                    if checkpoint_lead.get(lead_key) != current_lead.get(lead_key):
                        mismatches.append(
                            f"checkpoint lead_conditioning.{lead_key}={checkpoint_lead.get(lead_key)!r} "
                            f"but current config has lead_conditioning.{lead_key}={current_lead.get(lead_key)!r}"
                        )
                continue
            if key == "l0_refine":
                checkpoint_refine = dict(checkpoint_values[key])
                current_refine = dict(current_values[key])
                for refine_key in (
                    "type",
                    "mlp_expansion",
                    "dropout",
                    "residual_scale_init",
                    "learnable_residual_scale",
                ):
                    if checkpoint_refine.get(refine_key) != current_refine.get(refine_key):
                        mismatches.append(
                            f"checkpoint l0_refine.{refine_key}={checkpoint_refine.get(refine_key)!r} "
                            f"but current config has l0_refine.{refine_key}={current_refine.get(refine_key)!r}"
                        )
                continue
            mismatches.append(
                f"checkpoint has {key}={checkpoint_values[key]!r} but current config has {key}={current_values[key]!r}"
            )
        details = "; ".join(mismatches)
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"{details}. "
            "Train from scratch or use explicit partial initialization."
        )

    current_lead = current_arch.lead_conditioning
    current_input = _get_value(current, "input_channels", _MISSING)
    checkpoint_input = _get_value(checkpoint_metadata or {}, "input_channels", _MISSING)
    if checkpoint_input is _MISSING:
        checkpoint_input = _get_value(checkpoint_metadata or {}, "N_in_channels", _MISSING)
    if current_input is not _MISSING and checkpoint_input is not _MISSING:
        if int(checkpoint_input) != int(current_input):
            raise RuntimeError(
                "Checkpoint architecture mismatch: "
                f"Checkpoint input_channels={int(checkpoint_input)}, current input_channels={int(current_input)}. "
                "Train from scratch or use explicit partial initialization."
            )
    elif current_lead.enabled and current_input is not _MISSING:
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"Checkpoint input_channels=<missing>, current input_channels={int(current_input)}. "
            "Train from scratch or use explicit partial initialization."
        )

    current_output = _get_value(current, "output_channels", _MISSING)
    checkpoint_output = _get_value(checkpoint_metadata or {}, "output_channels", _MISSING)
    if current_output is not _MISSING and checkpoint_output is not _MISSING:
        if int(checkpoint_output) != int(current_output):
            raise RuntimeError(
                "Checkpoint architecture mismatch: "
                f"Checkpoint output_channels={int(checkpoint_output)}, current output_channels={int(current_output)}. "
                "Train from scratch or use explicit partial initialization."
            )


def validate_checkpoint_graph_mode(
    checkpoint_metadata: dict[str, Any] | None,
    current: Any,
) -> None:
    """Reject grid↔mesh checkpoint reuse, including partial initialization."""

    current_mesh = resolve_mesh_encoder(current)
    checkpoint_mesh_raw = checkpoint_metadata.get("mesh_encoder", {}) if checkpoint_metadata else {}
    checkpoint_mode = str((checkpoint_metadata or {}).get("graph_mode", "grid"))
    checkpoint_mesh_enabled = bool(
        isinstance(checkpoint_mesh_raw, dict) and checkpoint_mesh_raw.get("enabled", False)
    )
    if checkpoint_mode == "mesh":
        checkpoint_mesh_enabled = True
    if bool(current_mesh.enabled) != checkpoint_mesh_enabled:
        raise RuntimeError(
            "Checkpoint architecture mismatch: "
            f"checkpoint graph_mode={'mesh' if checkpoint_mesh_enabled else 'grid'} but current config has "
            f"graph_mode={'mesh' if current_mesh.enabled else 'grid'}. Grid and mesh modes use separate "
            "checkpoint families."
        )
