"""Tests for the attention contraction paths and the scoped edge-projection cache.

Everything here exists to keep a *performance* change from becoming a *skill*
change, so almost every assertion is an equivalence assertion. The ones that
matter most:

* ``TestNumericalEquivalence.test_float64_matches_reference`` -- both production
  contraction paths reproduce ``LocalGraphAttention._forward_reference``, the
  verbatim pre-refactor implementation, exactly in float64.
* ``TestEdgeCacheRollout.test_four_configurations_agree`` -- the cache is
  invisible to both outputs and gradients, including inside a checkpointed
  rollout, where a cache whose lifetime ends before backward makes non-reentrant
  checkpoint recompute a different op sequence than it saved.
* ``TestPerfControlParity`` -- the new control is architecturally identical to the
  one it must reproduce, down to the parameter count.
"""
from __future__ import annotations

import contextlib
import inspect
import sys
import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.checkpoint import checkpoint as torch_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import normalize_training_config_dict  # noqa: E402
from src.graph_builder import HYBRID_ROW_AWARE_KNN, build_graph_bundle, cell_center_lat_lon  # noqa: E402
from src.graph_bundle import GraphBundle, GraphLevel  # noqa: E402
from src.layers import (  # noqa: E402
    ATTENTION_IMPL_DEFAULT,
    ATTENTION_IMPLS,
    LocalGraphAttention,
)
from src.models import GraphWeatherModel  # noqa: E402
from src.trainer import COMPILE_SCOPES, Trainer  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
S1_BASE = CONFIG_DIR / "config_2p5_l3_hidden160_base_S1_200epoch_dense_l3k24.yaml"
S1_PERFCTL = CONFIG_DIR / "config_2p5_l3_hidden160_base_S1_200epoch_dense_l3k24_perfctl.yaml"
CURR_BASE = CONFIG_DIR / "config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt_s1_200ep.yaml"
CURR_PERFCTL = CONFIG_DIR / "config_2p5_l3_hidden160_dense_l3k24_curriculum_S2toS10_3ep_initckpt_s1_200ep_perfctl.yaml"
S1_500_FULL = CONFIG_DIR / "config_2p5_l3_hidden160_base_S1_500epoch_dense_l3k24_perfctl_ema_compilefull.yaml"
S1_500_PROC = CONFIG_DIR / "config_2p5_l3_hidden160_base_S1_500epoch_dense_l3k24_perfctl_ema_compileproc.yaml"
THREE_ARM = CONFIG_DIR / (
    "config_2p5_l3_hidden160_s1x200_currS2toS10x3_dense_l3k24_perfctl_compilefull_ema_gate_nomax_itv.yaml"
)

# The 200-epoch hidden-160 CTL this experiment must reproduce.
# runs/2p5_l3_h160_densel3k24_currS2toS10x3_initckpt_s1x200/model_summary.txt
# reports trainable_parameters: 3844932 with pooling.include_max=true.
OLD_CTL_PARAMETER_COUNT = 3_844_932

_CACHE: dict[str, object] = {}


def _only_config(path: Path) -> dict:
    return list(yaml.safe_load(path.read_text()).values())[0]


def _small_graph_level(num_nodes: int = 64, k: int = 8, edge_dim: int = 6) -> GraphLevel:
    """One synthetic level with exactly k sorted incoming edges per target node."""
    generator = torch.Generator().manual_seed(1234)
    target = torch.arange(num_nodes).repeat_interleave(k)
    source = torch.randint(0, num_nodes, (num_nodes * k,), generator=generator)
    return GraphLevel(
        {
            "height": 8,
            "width": 8,
            "num_nodes": num_nodes,
            "k": k,
            "coords": torch.randn(num_nodes, 3, generator=generator),
            "lat_lon": torch.randn(num_nodes, 2, generator=generator),
            "edge_index": torch.stack([source, target]),
            "edge_attr": torch.randn(num_nodes * k, edge_dim, generator=generator),
        }
    )


def _attention(dim: int = 32, heads: int = 4, edge_dim: int = 6, **kwargs) -> LocalGraphAttention:
    torch.manual_seed(7)
    return LocalGraphAttention(dim, edge_dim=edge_dim, heads=heads, **kwargs)


def _impls():
    """Both production contraction paths. Neither may drift from the reference."""
    return ATTENTION_IMPLS


def _small_bundle_l3() -> GraphBundle:
    """5.625-degree 4-level bundle: 2048 / 512 / 128 / 32 nodes. Cheap enough for float64."""
    if "small" not in _CACHE:
        latitudes, longitudes = cell_center_lat_lon(32, 64)
        _CACHE["small"] = build_graph_bundle(
            latitudes,
            longitudes,
            k=8,
            resolution=5.625,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            num_graph_levels=4,
            level_k_neighbors=[8, 8, 8, 8],
        )
    return GraphBundle(_CACHE["small"])


def _bundle_2p5_l3k24() -> GraphBundle:
    """The real 2.5-degree L3 hierarchy the control runs on (10368/2592/648/162)."""
    if "2p5" not in _CACHE:
        latitudes, longitudes = cell_center_lat_lon(72, 144)
        _CACHE["2p5"] = build_graph_bundle(
            latitudes,
            longitudes,
            k=8,
            resolution=2.5,
            connectivity_strategy=HYBRID_ROW_AWARE_KNN,
            resolution_mode="2p5",
            num_graph_levels=4,
            level_k_neighbors=[8, 8, 8, 24],
        )
    return GraphBundle(_CACHE["2p5"])


def _model_from_config(config: dict, graph: GraphBundle, grid_shape: tuple[int, int]) -> GraphWeatherModel:
    """Build the model the way Trainer does (src/trainer.py:439), from YAML params."""
    torch.manual_seed(0)
    return GraphWeatherModel(
        graph=graph,
        grid_shape=grid_shape,
        input_channels=int(config["input_channels"]),
        output_channels=int(config["output_channels"]),
        n_history=int(config["n_history"]),
        hidden_dim=int(config["hidden_dim"]),
        edge_dim=int(config["edge_dim"]),
        heads=int(config["num_heads"]),
        k_neighbors=int(config["k_neighbors"]),
        level_k_neighbors=config["level_k_neighbors"],
        encoder_blocks=int(config["encoder_blocks"]),
        decoder_blocks=int(config["decoder_blocks"]),
        l0_blocks=int(config["l0_blocks"]),
        l1_blocks=int(config["l1_blocks"]),
        l2_blocks=int(config["l2_blocks"]),
        l1_refine_blocks=int(config["l1_refine_blocks"]),
        l0_refine_blocks=int(config["l0_refine_blocks"]),
        num_graph_levels=int(config["num_graph_levels"]),
        use_l3=bool(config["use_l3"]),
        l3_blocks=int(config["l3_blocks"]),
        l2_refine_after_l3_blocks=int(config["l2_refine_after_l3_blocks"]),
        skip_fusion=dict(config["skip_fusion"]),
        pooling=dict(config["pooling"]),
        lead_conditioning=dict(config["lead_conditioning"]),
    )


class TestNumericalEquivalence(unittest.TestCase):
    """7.1 / 7.2 -- both contraction paths reproduce the pre-refactor reference."""

    def test_float64_matches_reference(self):
        graph = _small_graph_level().double()
        h = torch.randn(2, 64, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        for impl in _impls():
            with self.subTest(impl=impl):
                module = _attention(attention_impl=impl).double()
                new = module(h, graph)
                reference = module._forward_reference(h, graph)
                self.assertEqual(new.shape, reference.shape)
                self.assertTrue(torch.allclose(new, reference, atol=1e-10, rtol=0.0))

    def test_default_impl_is_bit_exact_with_the_reference(self):
        """The default path must not perturb the control at all, not even in float32.

        The elementwise default performs the same ops in the same order as the code
        the 200-epoch control was trained with, so equality is exact, not approximate.
        """
        self.assertEqual(ATTENTION_IMPL_DEFAULT, "elementwise")
        graph = _small_graph_level()
        module = _attention()
        h = torch.randn(2, 64, 32, generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(module(h, graph), module._forward_reference(h, graph)))

    def test_float64_matches_reference_for_every_edge_encoding(self):
        """The gate and bias-MLP branches are off in this control but must not rot."""
        graph = _small_graph_level().double()
        h = torch.randn(2, 64, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        for impl in _impls():
            for encoding in (
                {"gate": True},
                {"bias_mlp_hidden": 8, "rbf_bins": 4, "bearing_harmonics": 2},
            ):
                with self.subTest(impl=impl, encoding=encoding):
                    module = _attention(edge_encoding=encoding, attention_impl=impl).double()
                    self.assertTrue(
                        torch.allclose(
                            module(h, graph),
                            module._forward_reference(h, graph),
                            atol=1e-10,
                            rtol=0.0,
                        )
                    )

    def test_the_two_impls_agree_with_each_other(self):
        graph = _small_graph_level().double()
        h = torch.randn(2, 64, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        outs = [_attention(attention_impl=impl).double()(h, graph) for impl in _impls()]
        self.assertTrue(torch.allclose(outs[0], outs[1], atol=1e-10, rtol=0.0))

    def test_bfloat16_matches_reference(self):
        """Loose by construction: the reduction order differs. Structural check only."""
        graph = _small_graph_level().bfloat16()
        h = torch.randn(2, 64, 32, generator=torch.Generator().manual_seed(3)).bfloat16()
        for impl in _impls():
            with self.subTest(impl=impl):
                module = _attention(attention_impl=impl).bfloat16()
                new = module(h, graph).float()
                reference = module._forward_reference(h, graph).float()
                self.assertTrue(
                    torch.allclose(new, reference, atol=2e-2),
                    f"max |delta| = {(new - reference).abs().max():.3e}",
                )

    def test_unknown_impl_is_rejected(self):
        with self.assertRaises(ValueError):
            _attention(attention_impl="sdpa")


class TestSoftmaxAxis(unittest.TestCase):
    """7.3 / 7.4 -- the softmax normalizes over neighbours, not heads."""

    def _dominant_neighbour_module(self, dominant, num_nodes, k, edge_dim, impl):
        graph = _small_graph_level(num_nodes=num_nodes, k=k, edge_dim=edge_dim)
        # Channel 0 flags the dominant neighbour slot; edge_bias turns it into +50.
        edge_attr = graph.edge_attr.reshape(num_nodes, k, edge_dim).clone()
        edge_attr[:, :, 0] = 0.0
        edge_attr[:, dominant, 0] = 1.0
        graph.edge_attr = edge_attr.reshape(num_nodes * k, edge_dim)
        module = _attention(edge_dim=edge_dim, attention_impl=impl)
        with torch.no_grad():
            module.edge_bias.weight.zero_()
            module.edge_bias.weight[:, 0] = 50.0
            module.edge_bias.bias.zero_()
            # Zero the value-side edge term so the expected output is exactly the
            # dominant neighbour's projected value.
            module.edge_v.weight.zero_()
            module.edge_v.bias.zero_()
        return graph.double(), module.double()

    def test_dominant_neighbour_wins(self):
        num_nodes, k, edge_dim, dominant = 64, 8, 6, 3
        for impl in _impls():
            with self.subTest(impl=impl):
                graph, module = self._dominant_neighbour_module(dominant, num_nodes, k, edge_dim, impl)
                h = torch.randn(
                    2, num_nodes, 32, dtype=torch.float64, generator=torch.Generator().manual_seed(5)
                )
                out = module(h, graph)

                src = graph.edge_index[0].reshape(num_nodes, k)
                v = module.v_proj(h).reshape(2, num_nodes, module.heads, module.head_dim)
                expected = module.out_proj(v[:, src[:, dominant]].reshape(2, num_nodes, 32))
                # A softmax over heads instead of neighbours moves this by O(1).
                self.assertTrue(torch.allclose(out, expected, atol=1e-5))

    def test_attention_weights_sum_to_one_over_neighbours(self):
        num_nodes, k, dim, heads, bsz = 64, 8, 32, 4, 2
        graph = _small_graph_level(num_nodes=num_nodes, k=k)
        h = torch.randn(bsz, num_nodes, dim, generator=torch.Generator().manual_seed(11))
        for impl in _impls():
            with self.subTest(impl=impl):
                module = _attention(dim=dim, heads=heads, attention_impl=impl)
                collector = _StubCollector()
                module(h, graph, diagnostics_collector=collector, diagnostics_name="attn")

                # Collectors see [B,N,k,H]; transpose to the head-major internal layout.
                attn = collector.attention["attn"].transpose(2, 3)
                self.assertEqual(tuple(attn.shape), (bsz, num_nodes, heads, k))
                self.assertTrue(
                    torch.allclose(attn.sum(dim=-1), torch.ones(bsz, num_nodes, heads), atol=1e-6)
                )


class _StubCollector:
    def __init__(self):
        self.attention: dict[str, torch.Tensor] = {}

    def add_attention(self, name, tensor, edge_index=None, num_nodes=None):
        self.attention[name] = tensor.detach().clone()


class TestDiagnosticsShape(unittest.TestCase):
    """7.5 -- src/diagnostics/ still receives [B, N, k, H]."""

    def test_collector_receives_neighbour_major_layout(self):
        num_nodes, k, dim, heads, bsz = 64, 8, 32, 4, 2
        graph = _small_graph_level(num_nodes=num_nodes, k=k)
        h = torch.randn(bsz, num_nodes, dim, generator=torch.Generator().manual_seed(13))
        captured = {}
        for impl in _impls():
            with self.subTest(impl=impl):
                module = _attention(dim=dim, heads=heads, attention_impl=impl)
                collector = _StubCollector()
                module(h, graph, diagnostics_collector=collector,
                       diagnostics_name="processor/l0_blocks.0")
                tensor = collector.attention["processor/l0_blocks.0"]
                self.assertEqual(tuple(tensor.shape), (bsz, num_nodes, k, heads))
                captured[impl] = tensor
        # Same weights, not just the same shape: the diagnostics stay comparable
        # across a change of contraction path.
        self.assertTrue(torch.allclose(captured["elementwise"], captured["matmul"], atol=1e-6))


def _rollout(model, previous, current, steps, *, use_cache, checkpoint, backward_loss=None):
    """Mirror of Trainer.train_one_epoch + Trainer._rollout_loss.

    Scope order matters and is the whole point of this helper: the cache scope wraps
    forward AND backward, warming happens once before the loop and outside every
    checkpointed step. Clearing the cache before backward makes non-reentrant
    checkpoint recompute a different op sequence than it saved, which raises
    CheckpointError -- so the shape here must stay in step with the trainer.
    """
    scope = model.edge_cache_scope(warm=False) if use_cache else contextlib.nullcontext()
    prediction = None
    with scope:
        if use_cache:
            model.warm_edge_cache()
        for _ in range(steps):
            def run_step(prev, cur):
                return model.forward_steps(prev, cur)

            prediction = (
                torch_checkpoint(run_step, previous, current, use_reentrant=False)
                if checkpoint
                else run_step(previous, current)
            )
            next_step = current.clone()
            next_step[:, : model.output_channels] = prediction
            previous, current = current, next_step
        if backward_loss is not None:
            (prediction * backward_loss).sum().backward()
    return prediction


class TestEdgeCacheRollout(unittest.TestCase):
    """7.6 / 7.7 -- cache correctness and lifetime."""

    GRAD_PARAMS = ("edge_k.weight", "edge_v.weight", "edge_bias.weight", "q_proj.weight")

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = GraphWeatherModel(
            graph=_small_bundle_l3(),
            grid_shape=(32, 64),
            input_channels=4,
            output_channels=2,
            n_history=1,
            hidden_dim=16,
            edge_dim=6,
            heads=4,
            num_graph_levels=4,
            use_l3=True,
            l3_blocks=1,
            l2_refine_after_l3_blocks=1,
        ).double()
        generator = torch.Generator().manual_seed(21)
        cls.previous = torch.randn(1, 2, 32, 64, dtype=torch.float64, generator=generator)
        cls.current = torch.randn(1, 2, 32, 64, dtype=torch.float64, generator=generator)
        cls.loss_weight = torch.randn(1, 2, 32, 64, dtype=torch.float64, generator=generator)

    def _set_impl(self, impl):
        for module in self.model.modules():
            if isinstance(module, LocalGraphAttention):
                module.attention_impl = impl

    def _run(self, *, use_cache, checkpoint, impl="elementwise"):
        self._set_impl(impl)
        self.model.zero_grad(set_to_none=True)
        prediction = _rollout(
            self.model,
            self.previous,
            self.current,
            3,
            use_cache=use_cache,
            checkpoint=checkpoint,
            backward_loss=self.loss_weight,
        )
        grads = {
            f"{name}.{suffix}": module.get_parameter(suffix).grad.detach().clone()
            for name, module in self.model.named_modules()
            if isinstance(module, LocalGraphAttention)
            for suffix in self.GRAD_PARAMS
        }
        return prediction.detach().clone(), grads

    def test_four_configurations_agree(self):
        """{cache on, off} x {rollout checkpoint on, off} x {both impls}."""
        baseline_out, baseline_grads = self._run(use_cache=False, checkpoint=False)
        self.assertTrue(torch.isfinite(baseline_out).all())
        self.assertTrue(all(g.abs().sum() > 0 for g in baseline_grads.values()))

        for impl in _impls():
            for use_cache in (False, True):
                for checkpoint in (False, True):
                    if impl == "elementwise" and not use_cache and not checkpoint:
                        continue
                    with self.subTest(impl=impl, cache=use_cache, checkpoint=checkpoint):
                        out, grads = self._run(
                            use_cache=use_cache, checkpoint=checkpoint, impl=impl
                        )
                        self.assertTrue(
                            torch.allclose(out, baseline_out, atol=1e-9, rtol=0.0),
                            f"output mismatch: max |delta| = "
                            f"{(out - baseline_out).abs().max():.3e}",
                        )
                        self.assertEqual(set(grads), set(baseline_grads))
                        for name, grad in grads.items():
                            self.assertTrue(
                                torch.allclose(grad, baseline_grads[name], atol=1e-9, rtol=0.0),
                                f"gradient mismatch for {name} "
                                f"(impl={impl}, cache={use_cache}, checkpoint={checkpoint}): "
                                f"max |delta| = {(grad - baseline_grads[name]).abs().max():.3e}",
                            )
        self._set_impl(ATTENTION_IMPL_DEFAULT)

    def test_cache_is_warm_before_the_rollout_runs(self):
        """Every attention module must be populated before step 1, not lazily."""
        modules = [m for m in self.model.modules() if isinstance(m, LocalGraphAttention)]
        self.assertEqual(len(modules), 11)
        with self.model.edge_cache_scope():
            self.assertTrue(
                all(m._edge_cache for m in modules),
                "warm_edge_cache() left some attention module unpopulated; a lazily "
                "filled cache is the stale-graph footgun this scope exists to avoid.",
            )

    def test_rollout_loss_warms_but_does_not_own_the_cache_lifetime(self):
        """Regression guard for the ordering constraint the four-config test found.

        _rollout_loss returns before backward() runs, so it must not close the cache
        scope: with checkpoint_rollout_steps on, each step is recomputed during
        backward and non-reentrant checkpoint pairs saved tensors positionally. A
        cache that is live during the forward and gone during the recompute changes
        the op sequence and raises CheckpointError. The lifetime belongs to
        train_one_epoch, which wraps forward and backward together.
        """
        source = inspect.getsource(Trainer._rollout_loss)
        self.assertIn("_warm_edge_cache()", source)
        self.assertNotIn("_edge_cache_scope", source)

        train_source = inspect.getsource(Trainer.train_one_epoch)
        self.assertIn("_edge_cache_scope(warm=False)", train_source)
        scope_at = train_source.index("self._edge_cache_scope(warm=False)")
        backward_at = train_source.index(".backward()")
        # ...and closes again before the optimizer sees the accumulated gradients.
        self.assertLess(scope_at, backward_at)
        self.assertLess(backward_at, train_source.index("self._optimizer_step()"))

    def test_warm_covers_every_attention_module(self):
        self.model.enable_edge_cache()
        try:
            warmed = self.model.warm_edge_cache()
        finally:
            self.model.clear_edge_cache()
        self.assertEqual(warmed, 11)

    def test_cache_does_not_survive_the_step_boundary(self):
        modules = [m for m in self.model.modules() if isinstance(m, LocalGraphAttention)]
        _rollout(self.model, self.previous, self.current, 2, use_cache=True, checkpoint=False)
        self.assertTrue(all(m._edge_cache is None for m in modules))

        with self.assertRaises(RuntimeError):
            with self.model.edge_cache_scope():
                self.model.forward_steps(self.previous, self.current)
                raise RuntimeError("mid-rollout failure")
        self.assertTrue(all(m._edge_cache is None for m in modules))


class TestEmaUnderCompile(unittest.TestCase):
    """EMA keys must be compile-invariant, or a compiled run's shadow is unusable.

    torch.compile renames model.processor's parameters to ``*._orig_mod.*``. The
    EMA shadow is keyed by parameter name, but ``_restore_checkpoint`` runs *before*
    ``_maybe_compile_processor``, so it matches saved EMA keys against *uncompiled*
    names. If the shadow carried the compile prefix, every processor parameter would
    be dropped on resume by a silent ``if name in param_names`` filter -- the run
    would continue with an EMA covering only embed/encoder/decoder/head.
    """

    class _Stub:
        """Minimal stand-in: Trainer.__init__ needs data loaders we do not have here."""

        _canonical_named_parameters = Trainer._canonical_named_parameters
        _ema_update = Trainer._ema_update
        _ema_swap_in = Trainer._ema_swap_in
        _ema_swap_out = Trainer._ema_swap_out

        def __init__(self, model, compiled):
            self.model = model
            self._processor_compiled = compiled
            self.ema_enabled = True
            self.ema_decay = 0.999
            self._ema_state = None
            self._ema_backup = None

    def _model(self):
        torch.manual_seed(0)
        return GraphWeatherModel(
            graph=_small_bundle_l3(), grid_shape=(32, 64),
            input_channels=4, output_channels=2, n_history=1,
            hidden_dim=16, edge_dim=6, heads=4,
            num_graph_levels=4, use_l3=True,
        )

    def test_shadow_keys_do_not_carry_the_compile_prefix(self):
        model = self._model()
        # Emulate torch.compile's rename without paying for a real compile.
        model.processor = _OrigModWrapper(model.processor)
        stub = self._Stub(model, compiled=True)
        stub._ema_update()

        keys = set(stub._ema_state)
        self.assertTrue(keys, "EMA shadow was never populated")
        offenders = sorted(k for k in keys if "_orig_mod" in k)
        self.assertEqual(offenders, [], f"{len(offenders)} shadow keys carry the compile prefix")

        # The names a pre-compile resume would match against.
        eager_names = {n for n, p in self._model().named_parameters() if p.requires_grad}
        self.assertEqual(keys, eager_names)
        self.assertTrue(any(k.startswith("processor.") for k in keys))

    def test_shadow_is_identical_compiled_and_eager(self):
        eager = self._Stub(self._model(), compiled=False)
        compiled_model = self._model()
        compiled_model.processor = _OrigModWrapper(compiled_model.processor)
        compiled = self._Stub(compiled_model, compiled=True)
        for stub in (eager, compiled):
            stub._ema_update()
            stub._ema_update()
        self.assertEqual(set(eager._ema_state), set(compiled._ema_state))
        for name, tensor in eager._ema_state.items():
            self.assertTrue(torch.allclose(tensor, compiled._ema_state[name], atol=0, rtol=0))

    def test_swap_in_and_out_round_trip_under_compile(self):
        model = self._model()
        model.processor = _OrigModWrapper(model.processor)
        stub = self._Stub(model, compiled=True)
        stub._ema_update()
        for shadow in stub._ema_state.values():
            shadow.add_(1.0)  # make the shadow visibly different from the live weights

        live_before = {n: p.detach().clone() for n, p in stub._canonical_named_parameters()}
        stub._ema_swap_in()
        swapped = dict(stub._canonical_named_parameters())
        # Every parameter moved, including the ones behind the compile wrapper.
        moved = [n for n, p in swapped.items() if not torch.allclose(p, live_before[n])]
        self.assertEqual(len(moved), len(live_before))
        self.assertTrue(any(n.startswith("processor.") for n in moved))

        stub._ema_swap_out()
        for name, param in stub._canonical_named_parameters():
            self.assertTrue(torch.equal(param, live_before[name]), f"{name} not restored")
        self.assertIsNone(stub._ema_backup)

    def test_perfctl_enables_ema(self):
        for path in (S1_PERFCTL, CURR_PERFCTL):
            with self.subTest(config=path.name):
                ema = _only_config(path).get("ema", {}) or {}
                self.assertTrue(bool(ema.get("enabled", False)))
                self.assertGreater(float(ema.get("decay", 0.0)), 0.0)


class _OrigModWrapper(torch.nn.Module):
    """What torch.compile's OptimizedModule does to parameter names, minus the compile."""

    def __init__(self, module):
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_mod, name)


class TestCompileScope(unittest.TestCase):
    """compile_scope selects how much of the hot path torch.compile wraps."""

    class _Stub:
        _resolve_compile_scope = Trainer._resolve_compile_scope

        def __init__(self, **params):
            self.params = type("P", (), params)()

    def test_back_compat_with_the_boolean(self):
        self.assertEqual(self._Stub()._resolve_compile_scope(), "none")
        self.assertEqual(self._Stub(compile_processor=True)._resolve_compile_scope(), "processor")
        self.assertEqual(self._Stub(compile_processor=False)._resolve_compile_scope(), "none")

    def test_scope_wins_over_the_boolean(self):
        stub = self._Stub(compile_scope="none", compile_processor=True)
        self.assertEqual(stub._resolve_compile_scope(), "none")
        stub = self._Stub(compile_scope="full", compile_processor=False)
        self.assertEqual(stub._resolve_compile_scope(), "full")

    def test_case_insensitive_and_validated(self):
        self.assertEqual(self._Stub(compile_scope="FULL")._resolve_compile_scope(), "full")
        with self.assertRaises(ValueError):
            self._Stub(compile_scope="sdpa")._resolve_compile_scope()
        self.assertEqual(set(COMPILE_SCOPES), {"none", "processor", "full"})

    def test_full_scope_targets_forward_steps_not_forward(self):
        """The hot path is forward_steps; compiling the module would be a no-op.

        torch.compile on an nn.Module only intercepts __call__/forward, and nothing
        in training or evaluation calls GraphWeatherModel.forward during a rollout.
        """
        source = inspect.getsource(Trainer._maybe_compile_model)
        self.assertIn("torch.compile(forward_steps", source)
        self.assertIn('getattr(self.model, "forward_steps", None)', source)


class TestFiveHundredEpochConfigs(unittest.TestCase):
    """The two 500-epoch S1 configs differ only in compile scope."""

    def test_both_exist(self):
        for path in (S1_500_FULL, S1_500_PROC):
            self.assertTrue(path.is_file(), f"missing {path.name}")

    def test_horizon_and_stage_epochs_agree(self):
        # CosineAnnealingLR uses T_max=max_epochs, and a stage schedule that stops
        # short of it would silently leave the tail epochs on the last stage.
        for path in (S1_500_FULL, S1_500_PROC):
            with self.subTest(config=path.name):
                config = _only_config(path)
                self.assertEqual(int(config["max_epochs"]), 500)
                self.assertEqual(sum(int(x) for x in config["rollout_stage_epochs"]), 500)
                self.assertEqual(list(config["rollout_schedule"]), [1])

    def test_perf_stack_and_ema(self):
        for path, scope in ((S1_500_FULL, "full"), (S1_500_PROC, "processor")):
            with self.subTest(config=path.name):
                config = normalize_training_config_dict(_only_config(path))
                self.assertEqual(int(config["batch_size"]), 12)
                self.assertEqual(int(config["gradient_accumulation_steps"]), 1)
                self.assertEqual(config["attention_impl"], "elementwise")
                self.assertTrue(bool(config["edge_projection_cache"]))
                self.assertEqual(str(config["compile_scope"]), scope)
                # compile_scope supersedes the boolean; keep one switch per file.
                self.assertNotIn("compile_processor", config)
                self.assertTrue(bool((config.get("ema") or {}).get("enabled")))

    def test_architecture_is_untouched(self):
        base = _only_config(S1_BASE)
        for path in (S1_500_FULL, S1_500_PROC):
            # max_epochs and rollout_stage_epochs are the horizon change itself.
            for key in TestPerfControlParity.GUARDED_KEYS:
                if key in {"max_epochs", "rollout_stage_epochs"}:
                    continue
                with self.subTest(config=path.name, key=key):
                    self.assertEqual(base.get(key), _only_config(path).get(key))

    def test_the_pair_differs_only_in_compile_scope(self):
        full, proc = _only_config(S1_500_FULL), _only_config(S1_500_PROC)
        identity = {"experiment_name", "run_name", "name", "wandb", "compile_scope"}
        differing = {k for k in set(full) | set(proc) if full.get(k) != proc.get(k)}
        self.assertEqual(differing, {"compile_scope"} | (differing & identity))
        self.assertEqual(full["compile_scope"], "full")
        self.assertEqual(proc["compile_scope"], "processor")


class TestThreeArmCombinedConfig(unittest.TestCase):
    """One config, both phases, three arms stacked on the full-compile perf stack."""

    def setUp(self):
        self.raw = _only_config(THREE_ARM)
        self.resolved = normalize_training_config_dict(self.raw)

    def test_single_config_covers_both_phases(self):
        # S1 for 200 epochs, then horizons 2..10 for 3 epochs each.
        self.assertEqual(list(self.raw["rollout_schedule"]), list(range(1, 11)))
        self.assertEqual(list(self.raw["rollout_stage_epochs"]), [200] + [3] * 9)
        # CosineAnnealingLR uses T_max=max_epochs; a schedule that under- or
        # over-runs it would silently mis-shape the tail.
        self.assertEqual(int(self.raw["max_epochs"]), sum(self.raw["rollout_stage_epochs"]))
        self.assertEqual(int(self.raw["max_epochs"]), 227)
        self.assertEqual(self.resolved["rollout_mode"], "curriculum")
        # Per-stage loading: _ensure_train_loader_rollout rebuilds the train loader at
        # each transition. Leaving this false makes all 227 epochs load 10-step
        # targets, and the 200 S1 epochs then spend ~60% of their wall clock blocked
        # on the dataloader.
        self.assertTrue(bool(self.resolved["load_only_current_rollout"]))
        self.assertEqual(self.raw["checkpoint_metric"], "valid_S10_final")
        self.assertEqual(int(self.raw["valid_rollout_steps"]), 10)

    def test_perf_stack_is_inherited(self):
        self.assertEqual(int(self.resolved["batch_size"]), 12)
        self.assertEqual(int(self.resolved["gradient_accumulation_steps"]), 1)
        self.assertEqual(self.resolved["attention_impl"], "elementwise")
        self.assertTrue(bool(self.resolved["edge_projection_cache"]))
        self.assertEqual(self.raw["compile_scope"], "full")
        self.assertTrue(bool(self.raw["ema"]["enabled"]))

    def test_all_three_arms_are_present_and_agree_flat_and_nested(self):
        # Trainer flattens the `model:` block over the flat keys, so a value set in
        # only one place silently wins; keep both in sync.
        gate = {"rbf_bins": 16, "bearing_harmonics": 2, "gate": True, "bias_mlp_hidden": 32}
        self.assertEqual(self.raw["edge_encoding"], gate)
        self.assertEqual(self.raw["model"]["edge_encoding"], gate)

        for pooling in (self.raw["pooling"], self.raw["model"]["pooling"]):
            self.assertFalse(bool(pooling["include_max"]))
            self.assertEqual(pooling["mean_type"], "area_weighted")

        lossw = self.raw["loss_channel_weighting"]
        self.assertTrue(bool(lossw["enabled"]))
        self.assertTrue(bool(lossw["inverse_tendency_variance"]))

    def test_arms_match_their_source_arm_configs(self):
        """Copied verbatim, so a change to an arm config surfaces here."""
        arm_dir = CONFIG_DIR
        gate_arm = _only_config(arm_dir / "config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24_edge_gate.yaml")
        nomax_arm = _only_config(arm_dir / "config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24_pool_nomax.yaml")
        itv_arm = _only_config(arm_dir / "config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24_lossw_itv.yaml")
        self.assertEqual(self.raw["model"]["edge_encoding"], gate_arm["model"]["edge_encoding"])
        self.assertEqual(self.raw["model"]["pooling"], nomax_arm["model"]["pooling"])
        self.assertEqual(self.raw["loss_channel_weighting"], itv_arm["loss_channel_weighting"])

    def test_substrate_matches_the_arm_family(self):
        """Every arm config -- and the arm family's own control -- feeds ground-truth
        tisr at each rollout step and drops it from the scored loss. Stacking arms on
        a different substrate would make the result incomparable to their individual
        numbers, so pin it to the arm configs rather than to a literal.
        """
        arm = _only_config(
            CONFIG_DIR / "config_2p5_l3_hidden160_base_S1_100epoch_dense_l3k24_edge_gate.yaml"
        )
        self.assertEqual(self.raw["target_handling"], arm["target_handling"])
        self.assertEqual(self.raw["target_handling"]["known_future_variables"], ["tisr"])
        self.assertEqual(self.raw["target_handling"]["exclude_loss_variables"], ["orog", "tisr"])

    def test_architecture_dimensions_are_still_the_control_s(self):
        base = _only_config(S1_BASE)
        for key in ("hidden_dim", "num_heads", "edge_dim", "level_k_neighbors",
                    "num_graph_levels", "use_l3", "l0_blocks", "l1_blocks", "l2_blocks",
                    "l3_blocks", "encoder_blocks", "decoder_blocks", "graph_path"):
            with self.subTest(key=key):
                self.assertEqual(base.get(key), self.raw.get(key))


class TestEffectiveBatchInvariance(unittest.TestCase):
    """7.8 -- (4, 3) and (12, 1) both give an effective batch of 12."""

    def test_arithmetic(self):
        for batch_size, accumulation in ((4, 3), (12, 1)):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(batch_size * accumulation, 12)

    def test_configs_on_disk(self):
        for path in (S1_BASE, S1_PERFCTL, CURR_BASE, CURR_PERFCTL):
            with self.subTest(config=path.name):
                config = normalize_training_config_dict(_only_config(path))
                self.assertEqual(
                    int(config["batch_size"]) * int(config["gradient_accumulation_steps"]), 12
                )

    def test_perfctl_uses_batch_12_accum_1_and_the_cache(self):
        for path in (S1_PERFCTL, CURR_PERFCTL):
            with self.subTest(config=path.name):
                config = normalize_training_config_dict(_only_config(path))
                self.assertEqual(int(config["batch_size"]), 12)
                self.assertEqual(int(config["gradient_accumulation_steps"]), 1)
                self.assertTrue(bool(config["edge_projection_cache"]))
                self.assertFalse(bool(config["activation_checkpointing"]))
                self.assertFalse(bool(config["checkpoint_rollout_steps"]))

    def test_edge_cache_defaults_off_so_existing_runs_are_unchanged(self):
        for path in (S1_BASE, CURR_BASE):
            with self.subTest(config=path.name):
                config = normalize_training_config_dict(_only_config(path))
                self.assertFalse(bool(config["edge_projection_cache"]))


class TestPerfControlParity(unittest.TestCase):
    """7.9 -- the new CTL is architecturally identical to the old CTL."""

    GUARDED_KEYS = (
        "resolution_mode", "hidden_dim", "num_heads", "edge_dim", "level_k_neighbors",
        "num_graph_levels", "use_l3", "encoder_blocks", "decoder_blocks", "l0_blocks",
        "l1_blocks", "l2_blocks", "l3_blocks", "l2_refine_after_l3_blocks",
        "l1_refine_blocks", "l0_refine_blocks", "graph_connectivity_strategy",
        "row_aware_knn", "skip_fusion", "pooling", "model", "lr", "min_lr",
        "weight_decay", "scheduler", "max_epochs", "max_gradient_norm", "enable_amp",
        "amp_dtype", "rollout_schedule", "rollout_stage_epochs", "lr_schedule_type",
        "warmup_epochs", "warmup_start_factor", "checkpoint_metric", "target_handling",
        "graph_path", "head_dim", "dt", "n_history", "max_rollout_steps",
    )

    def test_guarded_keys_are_byte_identical(self):
        for base_path, new_path in ((S1_BASE, S1_PERFCTL), (CURR_BASE, CURR_PERFCTL)):
            base, new = _only_config(base_path), _only_config(new_path)
            for key in self.GUARDED_KEYS:
                with self.subTest(config=new_path.name, key=key):
                    self.assertEqual(base.get(key), new.get(key))

    def test_max_epochs_is_still_200_for_s1(self):
        # CosineAnnealingLR is built with T_max=max_epochs (src/trainer.py), so
        # moving the horizon rewrites the whole LR trajectory.
        self.assertEqual(int(_only_config(S1_PERFCTL)["max_epochs"]), 200)

    def test_parameter_count_matches_the_old_ctl(self):
        graph = _bundle_2p5_l3k24()
        base_model = _model_from_config(_only_config(S1_BASE), graph, (72, 144))
        new_model = _model_from_config(_only_config(S1_PERFCTL), graph, (72, 144))
        base_count = sum(p.numel() for p in base_model.parameters())
        new_count = sum(p.numel() for p in new_model.parameters())
        self.assertEqual(new_count, base_count)
        self.assertEqual(new_count, OLD_CTL_PARAMETER_COUNT)

    def test_block_inventory(self):
        """11 attention blocks, 3 pools, 3 unpools -- the shape the brief assumes."""
        model = _model_from_config(_only_config(S1_PERFCTL), _bundle_2p5_l3k24(), (72, 144))
        attention = [m for m in model.modules() if isinstance(m, LocalGraphAttention)]
        self.assertEqual(len(attention), 11)
        self.assertEqual(len(model.edge_cache_levels()), 11)  # grid enc/dec included but empty in grid mode
        self.assertEqual(model.warm_edge_cache.__self__ is model, True)

    def test_perfctl_pins_the_measured_faster_attention_impl(self):
        """The default is set from measurement; the config must not silently drift.

        matmul is 0.45x eager elementwise at the L0 block shape (batch gemv, m=1),
        so shipping it enabled would make the perf control slower than the control
        it replaces. See ATTENTION_IMPL_DEFAULT in src/layers.py.
        """
        self.assertEqual(ATTENTION_IMPL_DEFAULT, "elementwise")
        for path in (S1_PERFCTL, CURR_PERFCTL):
            with self.subTest(config=path.name):
                config = normalize_training_config_dict(_only_config(path))
                self.assertEqual(config["attention_impl"], "elementwise")

    def test_perfctl_compiles_the_processor(self):
        """compile_processor is read flat (src/trainer.py:_maybe_compile_processor).

        Under `training:` it would parse fine and do nothing, so pin the location as
        well as the value. Measured 2.65x the old CTL at S1 and 1.63x eager at S10.
        """
        for path in (S1_PERFCTL, CURR_PERFCTL):
            with self.subTest(config=path.name):
                raw = _only_config(path)
                self.assertIs(raw.get("compile_processor"), True)
                self.assertNotIn("compile_processor", raw.get("training", {}))


if __name__ == "__main__":
    unittest.main()
