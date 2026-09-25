# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Is ``propagate_spyre_tensor_layouts`` re-runnable on an already-propagated graph?

This is a FEASIBILITY PROBE, not a guard on current behaviour. It exists to
answer one design question before anything is built on the answer:

    Can the layout pass be run more than once on the same GraphLowering, with
    the same input seeds, and produce the same result?

Why it matters. A proposed treatment of ``invoke_subgraph`` boundaries would
make the subgraph a layout TRANSFER FUNCTION rather than a fixed boundary: for
each candidate layout of an operand, seed the body and propagate it to learn
which output layout that candidate implies, then let the parent's optimizer
choose an (operand, output) pair -- with the constraint that every call site of
one body picks the same pair. That requires propagating the body once PER
CANDIDATE, i.e. repeatedly, on a graph that has already been propagated.

Two properties have to hold for that to be viable:

- IDEMPOTENCE: re-running with identical seeds yields identical layouts. The
  pass writes ``op.layouts`` / ``op.restick_cost_fn`` as plain overwrites, which
  SUGGESTS idempotence, but ``split_multi_ops`` has already mutated the graph by
  then and several passes stash attributes on ops, so it needs demonstrating.
- SENSITIVITY: re-running with DIFFERENT seeds yields different layouts. Without
  this, "propagate per candidate" is meaningless -- the transfer function would
  be constant and could not distinguish candidates.

A failure here does not mean the design is dead; it means the exploratory runs
need real isolation (deep-copying the body's GraphLowering, or re-lowering it
from FX per candidate), which changes the cost calculus considerably. That is
the decision this probe informs.

Scope: the pass is intercepted mid-pipeline and re-run on the real graph; no
backend codegen runs. Non-Spyre-device graphs return early from the pipeline, so
the tensors here are on the Spyre device.

The last two tests run the same probes against a REAL ``invoke_subgraph`` body
(``nested_compile_region`` called twice), which is the case the design targets.
The body is a separate ``GraphLowering`` whose pipeline runs nested inside parent
codegen, so it is identified by ``graph.parent is not None``.
"""

from unittest.mock import patch

import torch
from torch._inductor import config as t_inductor_config
from torch.compiler import nested_compile_region

import torch_spyre._inductor.passes as _passes
import torch_spyre._inductor.wsr.propagate_named_dims as _pnd
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.propagate_layouts import propagate_spyre_tensor_layouts
from utils_inductor import _compile_and_run, mock_backend_compiler

DEVICE = torch.device("spyre")


def _snapshot_layouts(graph) -> dict:
    """Record every op's candidate layouts as comparable plain data.

    ``SpyreTensorLayout`` compares by value, but keying on ``str`` keeps a
    mismatch readable in the failure output and avoids depending on __eq__
    semantics for the diff itself.
    """
    snap = {}
    for op in graph.operations:
        layouts = getattr(op, "layouts", None)
        if layouts is None:
            continue
        snap[op.get_operation_name()] = [str(stl) for stl in layouts]
    return snap


def _snapshot_input_seeds(graph) -> dict:
    """Record the seeded layouts on graph inputs, which drive everything else."""
    seeds = {}
    for name in graph.graph_input_names:
        tb = graph.graph_inputs.get(name)
        layouts = getattr(tb, "layouts", None)
        if layouts is not None:
            seeds[name] = [str(stl) for stl in layouts]
    return seeds


def _snapshot_committed(graph) -> dict:
    """Record the STL the optimizer committed for each op and graph input.

    ``optimize_restickify_locations`` writes ``committed_stl`` onto buffers
    (``optimize_restickify.py``: ``op.committed_stl = stl``, and
    ``tb.data.data.committed_stl`` for graph inputs). That attribute is the
    optimizer's entire output -- ``finalize_layouts`` then does
    ``stl = committed if cost_fn else op_layouts[0]`` -- so comparing it across
    runs is what tells us whether the beam is deterministic.
    """
    snap = {}
    for op in graph.operations:
        stl = getattr(op, "committed_stl", None)
        if stl is not None:
            snap[op.get_operation_name()] = str(stl)
    for name in graph.graph_input_names:
        tb = graph.graph_inputs.get(name)
        buf = getattr(getattr(tb, "data", None), "data", None)
        stl = getattr(buf, "committed_stl", None)
        if stl is not None:
            snap[f"input:{name}"] = str(stl)
    return snap


def _capture_rerun(fn, args):
    """Compile ``fn``, and at layout-propagation time re-run the pass.

    Returns a dict with the layout snapshot after the first (real) run and after
    a second run performed immediately afterwards on the same graph, plus the
    input seeds observed each time.
    """
    captured: dict = {}
    real_propagate = _passes.propagate_spyre_tensor_layouts

    def capturing(graph):
        # Run 1: the genuine pass, exactly as the pipeline would.
        real_propagate(graph)
        captured["run1"] = _snapshot_layouts(graph)
        captured["seeds1"] = _snapshot_input_seeds(graph)

        # Run 2: same graph, same seeds, no intervening passes. _named_dims is a
        # module-level dict with declare-once (setdefault) semantics, so a second
        # propagation could otherwise observe the first run's declarations; save
        # and restore it to isolate the variable under test.
        saved_named_dims = dict(_pnd._named_dims)
        try:
            propagate_spyre_tensor_layouts(graph)
        finally:
            _pnd._named_dims.clear()
            _pnd._named_dims.update(saved_named_dims)
        captured["run2"] = _snapshot_layouts(graph)
        captured["seeds2"] = _snapshot_input_seeds(graph)

    with (
        patch.object(_passes, "propagate_spyre_tensor_layouts", capturing),
        patch("torch_spyre.execution.kernel_runner.prepare_kernel"),
        patch("torch_spyre.execution.kernel_runner.launch_jobplan"),
        mock_backend_compiler(),
    ):
        try:
            _compile_and_run(fn, args, DEVICE)
        except Exception:
            # Later passes (work division, codegen) are irrelevant here and are
            # allowed to fail; the probe only needs the two snapshots. If the
            # pass itself raised, the assertions below report the missing keys.
            pass

    return captured


def _assert_ran(captured):
    assert "run1" in captured, "layout propagation never ran"
    assert "run2" in captured, "second propagation did not complete"
    assert captured["run1"], "first run recorded no layouts at all"


def _diff(a: dict, b: dict) -> list[str]:
    """Human-readable per-op differences between two snapshots."""
    out = []
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            out.append(f"  {key}: {a.get(key)} -> {b.get(key)}")
    return out


def test_rerun_with_same_seeds_is_idempotent():
    """Re-running the pass unchanged must reproduce every op's layouts.

    This is the gate on "propagate the body once per candidate": if a second
    run drifts, exploratory propagation corrupts the graph it explores.
    """

    def fn(x, w):
        return torch.relu(x @ w)

    x = torch.randn(64, 128, dtype=torch.float16, device=DEVICE)
    w = torch.randn(128, 256, dtype=torch.float16, device=DEVICE)

    captured = _capture_rerun(fn, (x, w))
    _assert_ran(captured)

    assert captured["seeds1"] == captured["seeds2"], (
        "graph-input seeds changed between runs, so the two runs were not "
        "given the same starting point:\n"
        + "\n".join(_diff(captured["seeds1"], captured["seeds2"]))
    )

    diffs = _diff(captured["run1"], captured["run2"])
    assert not diffs, (
        "propagate_spyre_tensor_layouts is NOT idempotent: a second run with "
        "identical seeds produced different layouts. Per-candidate propagation "
        "of an invoke_subgraph body would therefore need an isolated copy of "
        "the graph.\n" + "\n".join(diffs)
    )


def test_rerun_is_idempotent_for_a_matmul_chain():
    """Same question on a graph with several layout-constrained ops in sequence.

    A single matmul may be too simple to expose order-dependent drift: chained
    matmuls give each op an input whose layouts were themselves computed by the
    previous run, which is where accumulated state would show up.
    """

    def fn(x, w1, w2):
        return torch.relu(x @ w1) @ w2

    x = torch.randn(64, 128, dtype=torch.float16, device=DEVICE)
    w1 = torch.randn(128, 256, dtype=torch.float16, device=DEVICE)
    w2 = torch.randn(256, 64, dtype=torch.float16, device=DEVICE)

    captured = _capture_rerun(fn, (x, w1, w2))
    _assert_ran(captured)

    diffs = _diff(captured["run1"], captured["run2"])
    assert not diffs, (
        "propagate_spyre_tensor_layouts drifted on a matmul chain:\n" + "\n".join(diffs)
    )


def _capture_optimizer_rerun(fn, args, reprop_between: bool = False):
    """Compile ``fn``, and at optimizer time re-run the beam on the same graph.

    ``reprop_between`` also re-runs layout propagation between the two optimizer
    runs, which is the ordering the transfer-function design would actually
    produce: exploratory propagation of a subgraph body happening after the
    parent's beam has already run once.
    """
    captured: dict = {}
    real_optimize = _passes.optimize_restickify_locations

    def capturing(graph):
        real_optimize(graph)
        captured["opt1"] = _snapshot_committed(graph)
        captured["layouts1"] = _snapshot_layouts(graph)

        if reprop_between:
            saved = dict(_pnd._named_dims)
            try:
                propagate_spyre_tensor_layouts(graph)
            finally:
                _pnd._named_dims.clear()
                _pnd._named_dims.update(saved)
            captured["layouts_mid"] = _snapshot_layouts(graph)

        real_optimize(graph)
        captured["opt2"] = _snapshot_committed(graph)
        captured["layouts2"] = _snapshot_layouts(graph)

    with (
        patch.object(_passes, "optimize_restickify_locations", capturing),
        patch("torch_spyre.execution.kernel_runner.prepare_kernel"),
        patch("torch_spyre.execution.kernel_runner.launch_jobplan"),
        mock_backend_compiler(),
    ):
        try:
            _compile_and_run(fn, args, DEVICE)
        except Exception:
            pass

    return captured


def _fn_matmul_chain(x, w1, w2):
    return torch.relu(x @ w1) @ w2


_CHAIN_ARGS = (
    (64, 128),
    (128, 256),
    (256, 64),
)


def _chain_tensors():
    return tuple(
        torch.randn(*shape, dtype=torch.float16, device=DEVICE) for shape in _CHAIN_ARGS
    )


def test_optimizer_rerun_is_idempotent():
    """Re-running the beam on an unchanged graph must commit the same STLs.

    ``optimize_restickify_locations`` writes ``committed_stl``, which is the only
    thing ``finalize_layouts`` reads when a cost_fn is present. If a second beam
    run on identical candidates commits something different, then any design that
    lets exploratory work interleave with the optimizer is unsound.
    """
    captured = _capture_optimizer_rerun(_fn_matmul_chain, _chain_tensors())

    assert "opt1" in captured, "optimize_restickify_locations never ran"
    assert "opt2" in captured, "second optimizer run did not complete"
    assert captured["opt1"], "first optimizer run committed nothing at all"

    diffs = _diff(captured["opt1"], captured["opt2"])
    assert not diffs, (
        "optimize_restickify_locations is NOT idempotent: a second beam run on "
        "the same candidates committed different STLs. Exploratory propagation "
        "must then be fully separated from the optimizer.\n" + "\n".join(diffs)
    )


def test_optimizer_rerun_after_repropagation_is_stable():
    """Propagation between two beam runs must not change what the beam commits.

    This is the ordering the transfer-function design creates: the parent's beam
    runs, then a subgraph body is propagated per candidate, then selection
    continues. Propagation overwrites ``op.layouts`` (42 assignment sites), and
    the beam reads exactly those lists -- so if propagation perturbs them, the
    second beam run would diverge.
    """
    captured = _capture_optimizer_rerun(
        _fn_matmul_chain, _chain_tensors(), reprop_between=True
    )

    assert "opt1" in captured, "optimize_restickify_locations never ran"
    assert "layouts_mid" in captured, "intervening propagation did not run"
    assert "opt2" in captured, "second optimizer run did not complete"

    layout_diffs = _diff(captured["layouts1"], captured["layouts_mid"])
    assert not layout_diffs, (
        "re-propagating after the optimizer changed the candidate lists the "
        "optimizer had already consumed:\n" + "\n".join(layout_diffs)
    )

    diffs = _diff(captured["opt1"], captured["opt2"])
    assert not diffs, (
        "the beam committed different STLs after an intervening propagation, so "
        "exploratory propagation cannot be interleaved with selection:\n"
        + "\n".join(diffs)
    )


def test_rerun_reseeds_graph_inputs_from_real_inputs():
    """A re-run RESETS graph-input layouts; it does not honour a perturbed seed.

    This is the finding that shapes the transfer-function design, and it is why
    the previous two tests' idempotence is not by itself sufficient.

    ``propagate_spyre_tensor_layouts`` begins by seeding every graph input from
    ``V.real_inputs`` (or, for a subgraph, from ``_subgraph_input_stls``). So
    writing a candidate layout onto ``graph_inputs[name].layouts`` and re-running
    does NOT evaluate that candidate: the pass overwrites it with the same seed
    as before and reproduces the original result. Demonstrated here by permuting
    a weight's two outermost device axes, re-running, and observing that the seed
    comes back unchanged and nothing downstream moves.

    Consequence for the design: per-candidate evaluation of an ``invoke_subgraph``
    body cannot be done by poking ``.layouts`` and re-running. The candidate has
    to be injected through the pass's own seeding path -- i.e. the body's seed
    source (``_subgraph_input_stls``) must become a parameter, so the parent can
    say "propagate this body AS IF its operands had layout L" rather than having
    the pass rediscover the committed layouts each time.
    """
    captured: dict = {}
    real_propagate = _passes.propagate_spyre_tensor_layouts

    def capturing(graph):
        real_propagate(graph)
        captured["baseline"] = _snapshot_layouts(graph)

        # Reseed the first graph input that carries layouts with a permuted
        # variant, then re-propagate and see whether anything downstream moved.
        reseeded = None
        for name in graph.graph_input_names:
            tb = graph.graph_inputs.get(name)
            layouts = getattr(tb, "layouts", None)
            if not layouts:
                continue
            stl = layouts[0]
            dev_size = list(stl.device_size)
            stride_map = list(stl.stride_map)
            if len(dev_size) < 3:
                continue
            # Swap the two outermost device axes: same extents and same stick
            # (the last axis is untouched), different axis placement -- the exact
            # distinction stick_compatible ignores and require_exact catches.
            dev_size[0], dev_size[1] = dev_size[1], dev_size[0]
            stride_map[0], stride_map[1] = stride_map[1], stride_map[0]
            try:
                permuted = SpyreTensorLayout(
                    dev_size,
                    stride_map,
                    stl.device_dtype,
                    stl.element_arrangement,
                )
            except Exception:
                continue
            tb.layouts = [permuted]
            reseeded = name
            break

        captured["reseeded_input"] = reseeded
        if reseeded is None:
            return
        captured["injected_seed"] = str(permuted)

        saved_named_dims = dict(_pnd._named_dims)
        try:
            propagate_spyre_tensor_layouts(graph)
        finally:
            _pnd._named_dims.clear()
            _pnd._named_dims.update(saved_named_dims)
        captured["perturbed"] = _snapshot_layouts(graph)
        captured["seed_after"] = [
            str(stl) for stl in graph.graph_inputs[reseeded].layouts
        ]

    def fn(x, w):
        return torch.relu(x @ w)

    x = torch.randn(8, 64, 128, dtype=torch.float16, device=DEVICE)
    w = torch.randn(128, 256, dtype=torch.float16, device=DEVICE)

    with (
        patch.object(_passes, "propagate_spyre_tensor_layouts", capturing),
        patch("torch_spyre.execution.kernel_runner.prepare_kernel"),
        patch("torch_spyre.execution.kernel_runner.launch_jobplan"),
        mock_backend_compiler(),
    ):
        try:
            _compile_and_run(fn, (x, w), DEVICE)
        except Exception:
            pass

    assert "baseline" in captured, "layout propagation never ran"
    if captured.get("reseeded_input") is None:
        # Nothing suitable to perturb on this graph shape, so the probe would
        # report nothing; fail loudly rather than pass vacuously.
        raise AssertionError(
            "no graph input had a rank-3+ layout to permute; adjust the "
            "fixture shapes so this probe exercises something"
        )
    assert "perturbed" in captured, "re-propagation after reseeding did not run"

    # The pass re-seeded the input from its own source, discarding the injected
    # candidate. This is the mechanism being documented.
    assert captured["seed_after"] != [captured["injected_seed"]], (
        "the injected seed SURVIVED the re-run, which contradicts this test's "
        "premise that propagate_spyre_tensor_layouts re-seeds graph inputs. If "
        "the pass has changed to honour a pre-set .layouts on a graph input, "
        "per-candidate evaluation becomes much cheaper -- update the design "
        "notes in this module's docstring."
    )

    # ...and because the seed was reset, the result is unchanged.
    diffs = _diff(captured["baseline"], captured["perturbed"])
    assert not diffs, (
        "layouts changed even though the input seed was reset to its original "
        "value, which means the re-run is not deterministic:\n" + "\n".join(diffs)
    )


# --------------------------------------------------------------------------
# The same probes against a real invoke_subgraph body.
#
# A nested_compile_region called twice lowers to an invoke_subgraph HOP, so the
# body becomes its own GraphLowering with graph.parent set. Its pipeline runs
# nested inside parent codegen, and -- unlike a top-level graph -- its inputs are
# seeded by _subgraph_input_stls rather than from V.real_inputs. That difference
# is exactly what these tests exist to check, since _subgraph_input_stls is the
# function the transfer-function design would have to parameterize.
# --------------------------------------------------------------------------


@nested_compile_region
def _region(h, w):
    return torch.relu(h @ w)


def _region_caller(h, w):
    # Two call sites: fewer and Dynamo inlines the region, leaving no subgraph.
    for _ in range(2):
        h = _region(h, w)
    return h


def _capture_subgraph(on_body, h_shape=(64, 128)):
    """Compile a two-site region and invoke ``on_body(graph)`` for the body only.

    ``on_body`` runs immediately after the body's real propagation. Returns
    whatever ``on_body`` stored in the dict it is handed.
    """
    captured: dict = {}
    real_propagate = _passes.propagate_spyre_tensor_layouts

    def capturing(graph):
        real_propagate(graph)
        if getattr(graph, "parent", None) is None:
            return  # the parent graph; covered by the tests above
        captured["subgraph_name"] = graph.name
        on_body(graph, captured)

    h = torch.randn(*h_shape, dtype=torch.float16, device=DEVICE)
    w = torch.randn(h_shape[-1], h_shape[-1], dtype=torch.float16, device=DEVICE)

    with (
        patch.object(_passes, "propagate_spyre_tensor_layouts", capturing),
        t_inductor_config.patch("fx_graph_cache", False),
        patch("torch_spyre.execution.kernel_runner.prepare_kernel"),
        patch("torch_spyre.execution.kernel_runner.launch_jobplan"),
        mock_backend_compiler(),
    ):
        torch._dynamo.reset()
        try:
            torch.compile(_region_caller, dynamic=False, fullgraph=True)(h, w)
        except Exception:
            # Later passes may fail; only the body snapshots matter here.
            pass
        finally:
            torch._dynamo.reset()

    return captured


def test_subgraph_body_propagation_is_idempotent():
    """Re-running propagation on a real invoke_subgraph body reproduces it.

    The top-level tests above cannot cover this: a body's inputs are seeded from
    ``_subgraph_input_stls`` (reading the parent's committed operand layouts), a
    different code path from the ``V.real_inputs`` seeding a top-level graph uses.
    """

    def on_body(graph, captured):
        captured["run1"] = _snapshot_layouts(graph)
        captured["seeds1"] = _snapshot_input_seeds(graph)
        saved = dict(_pnd._named_dims)
        try:
            propagate_spyre_tensor_layouts(graph)
        finally:
            _pnd._named_dims.clear()
            _pnd._named_dims.update(saved)
        captured["run2"] = _snapshot_layouts(graph)
        captured["seeds2"] = _snapshot_input_seeds(graph)

    captured = _capture_subgraph(on_body)

    assert "run1" in captured, (
        "no invoke_subgraph body was propagated -- the region was probably "
        "inlined, so this test guarded nothing"
    )
    assert captured["run1"], "the body recorded no layouts at all"

    assert captured["seeds1"] == captured["seeds2"], (
        "the body's input seeds changed between runs:\n"
        + "\n".join(_diff(captured["seeds1"], captured["seeds2"]))
    )
    diffs = _diff(captured["run1"], captured["run2"])
    assert not diffs, (
        f"propagation of subgraph body {captured.get('subgraph_name')!r} is NOT "
        "idempotent, so per-candidate propagation of a body would need an "
        "isolated copy:\n" + "\n".join(diffs)
    )


def test_subgraph_body_reseeds_from_parent_operands_discarding_injection():
    """A body re-run RESEEDS from the parent's operands, discarding an injection.

    This is the finding that constrains the design, confirmed on the path that
    actually matters. Writing a candidate layout onto a body input's ``.layouts``
    and re-propagating does NOT evaluate that candidate: ``_subgraph_input_stls``
    re-derives the seed from the parent's committed operand layouts and overwrites
    it, so the body reproduces its original result.

    Consequence: to evaluate "what would this body do with operand layout L",
    ``propagate_spyre_tensor_layouts`` needs L injected through its own seeding
    path -- i.e. an optional ``input_stls`` parameter, with
    ``_subgraph_input_stls`` as the default -- rather than by poking ``.layouts``.
    """

    def on_body(graph, captured):
        captured["baseline"] = _snapshot_layouts(graph)
        for name in graph.graph_input_names:
            tb = graph.graph_inputs.get(name)
            layouts = getattr(tb, "layouts", None)
            if not layouts:
                continue
            stl = layouts[0]
            dev_size = list(stl.device_size)
            stride_map = list(stl.stride_map)
            if len(dev_size) < 3:
                continue
            # Swap the two outermost device axes: same extents, same stick.
            dev_size[0], dev_size[1] = dev_size[1], dev_size[0]
            stride_map[0], stride_map[1] = stride_map[1], stride_map[0]
            permuted = SpyreTensorLayout(
                dev_size, stride_map, stl.device_dtype, stl.element_arrangement
            )
            tb.layouts = [permuted]
            captured["target"] = name
            captured["injected"] = str(permuted)
            break
        if "target" not in captured:
            return
        saved = dict(_pnd._named_dims)
        try:
            propagate_spyre_tensor_layouts(graph)
        finally:
            _pnd._named_dims.clear()
            _pnd._named_dims.update(saved)
        captured["after"] = _snapshot_layouts(graph)
        captured["seed_after"] = [
            str(s) for s in graph.graph_inputs[captured["target"]].layouts
        ]

    # Rank-3 hidden state so a body input has >=3 device axes to permute.
    captured = _capture_subgraph(on_body, h_shape=(8, 64, 128))

    assert "baseline" in captured, (
        "no invoke_subgraph body was propagated -- the region was probably inlined"
    )
    if "target" not in captured:
        raise AssertionError(
            "no body input had a rank-3+ layout to permute; adjust h_shape so "
            "this probe exercises something"
        )

    assert captured["seed_after"] != [captured["injected"]], (
        "the injected seed SURVIVED on the subgraph path, contradicting this "
        "test's premise that _subgraph_input_stls reseeds body inputs. If the "
        "pass now honours a pre-set .layouts on a body input, per-candidate "
        "evaluation is much cheaper -- update this module's docstring."
    )
    diffs = _diff(captured["baseline"], captured["after"])
    assert not diffs, (
        "the body's layouts changed even though its seed was reset to the "
        "original value, so the re-run is not deterministic:\n" + "\n".join(diffs)
    )
