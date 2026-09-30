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

"""End-to-end coverage for ``nested_compile_region`` bodies on Spyre.

A ``torch.compiler.nested_compile_region`` block called N times lowers to an
``invoke_subgraph`` HOP, so the region body is compiled once as a *separate*
Inductor graph and called from the parent. That split graph is what these
tests exercise:

- ``TestInvokeSubgraphSplit`` -- regression guard for cross-graph buffer
  origins in ``split_multi_ops`` (issue #3883).
- ``TestInvokeSubgraphAttention`` -- a realistic Granite-8B prefill attention
  body (SDPA + o_proj) inside a region: compiles, and matches CPU numerically.
- ``TestInvokeSubgraphEmbeddingFedOperand`` -- the same attention body, but with
  the layer-0 hidden state produced by an in-graph embedding, so the call sites
  disagree about the operand's device layout.
- ``TestSubgraphOutputLayoutValidation`` -- the RESULT side: forces the parent's
  declared result layout to disagree with what the body produces, and checks that
  ``_validate_subgraph_output_stls`` raises instead of miscompiling.

These assert on the FX graph Inductor actually receives: if Dynamo inlined the
regions into the parent there is no subgraph at all, and a test that only
checked shapes or numerics would pass while guarding nothing.
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F
from torch import nn
from torch._inductor import config as t_inductor_config
from torch.compiler import nested_compile_region

import torch_spyre  # noqa: F401  registers "spyre" + installs the inductor passes
from torch_spyre.constants import DEVICE_NAME

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from utils_inductor import compare_with_pytorch  # noqa: E402


class _Block(nn.Module):
    def forward(self, h):
        # Fused multi-op pointwise body: mul -> add -> relu in one loop body,
        # forcing split_multi_ops to materialize an intermediate and hit the
        # FX-node insertion that used to assert.
        return torch.relu(h * 3.0 + 1.0)


def _region(block):
    # nested_compile_region cannot mark a bound method, so wrap it.
    def wrapper(*args, **kwargs):
        return block.forward(*args, **kwargs)

    return nested_compile_region(wrapper)


class _RegionTestCase(unittest.TestCase):
    """Base for region tests: disables the FX graph cache, resets Dynamo."""

    def setUp(self):
        super().setUp()
        # Load-bearing, not hygiene: a cached FX graph is replayed without
        # re-running the Spyre pre-scheduling passes, so the passes under test
        # never fire and these tests would pass against broken code.
        patcher = t_inductor_config.patch("fx_graph_cache", False)
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def _compile_counting_hops(self, fn, seen_hops):
        """Compile ``fn``, recording the parent graph's invoke_subgraph nodes.

        Appends one list of HOP nodes per backend invocation to ``seen_hops``,
        so a caller can assert the regions were not inlined away.
        """

        def backend(gm, example_inputs):
            seen_hops.append(
                [
                    n
                    for n in gm.graph.nodes
                    if "invoke_subgraph" in str(getattr(n, "target", ""))
                ]
            )
            from torch._inductor.compile_fx import compile_fx

            return compile_fx(gm, example_inputs)

        return torch.compile(fn, backend=backend, dynamic=False, fullgraph=True)

    def _assert_regions_not_inlined(self, seen_hops, expected=2):
        self.assertTrue(seen_hops, "compile backend never ran")
        self.assertGreaterEqual(
            len(seen_hops[0]),
            expected,
            "expected repeated invoke_subgraph calls, got "
            f"{[n.name for n in seen_hops[0]]}",
        )


class TestInvokeSubgraphSplit(_RegionTestCase):
    """Regression test for invoke_subgraph + split_multi_ops graph identity.

    A region with a *fused multi-op* pointwise body (``relu(h * 3 + 1)`` ->
    mul, add, relu in one loop body, so ``split_multi_ops`` fires and reaches
    its FX-node insertion), on Spyre-device tensors, called N times so the
    region lowers to an ``invoke_subgraph`` HOP. Before the fix this failed
    during ``torch.compile`` with::

        torch._inductor.exc.InductorError: AssertionError:
            Node to insert before is not in graph.

    Root cause: the subgraph ComputedBuffer's ``origins`` set spans TWO
    ``fx.Graph`` objects -- the parent's ``invoke_subgraph`` / ``get_attr``
    nodes AND the subgraph-local ``mul`` node. ``split_multi_ops`` picked
    ``next(iter(op.origins))``, which could return the parent's
    invoke_subgraph node; that node is not in the subgraph's ``gl.graph``, so
    ``gl.graph.inserting_before(orig_node)`` asserted. The fix
    (``pass_utils.origin_in_graph``) selects the origin whose ``.graph
    is gl.graph``.
    """

    def test_nested_region_multi_op_compiles(self):
        """A reused region with a fused multi-op body must compile and run.

        Used to raise InductorError("Node to insert before is not in graph.")
        from split_multi_ops' FX insertion, during compilation.
        """
        blocks = [_region(_Block()) for _ in range(3)]

        def outer(h):
            for b in blocks:
                h = b(h)
            return h

        # Assert on what Inductor actually receives: if the regions were
        # inlined into the parent graph there is no cross-graph origin set,
        # and the test would silently guard nothing.
        seen_hops = []
        compiled = self._compile_counting_hops(outer, seen_hops)

        h = torch.randn(2, 64, dtype=torch.float16, device=DEVICE_NAME)
        out = compiled(h)

        self.assertEqual(tuple(out.shape), (2, 64))
        self._assert_regions_not_inlined(seen_hops)


# Granite 3.3 2B prefill geometry, after hf_adapters' stick-alignment padding
# of head_dim 64 -> 128 (prepare_rope_and_heads -> pad_attention_heads).
_BATCH = 1
_SEQLEN = 512  # == _SDPA_MAX_SEQUENCE_TILE_SIZE, so SDPA picks work_divided
_HIDDEN = 2048
_NUM_HEADS = 32
_NUM_KVHEADS = 8
_HEAD_DIM = 128
# Granite scales attention by an explicit multiplier, not by head_dim**-0.5.
_ATTENTION_MULTIPLIER = 0.015625
_NUM_LAYERS = 2  # >= 2 call sites, else Dynamo inlines the region


class _AttentionTail(nn.Module):
    """Granite attention tail: SDPA -> o_proj, hidden_state in and out.

    A trimmed ``StandardGQABlock._region_attention_tail`` (hf_common.py) --
    the residual/norm/MLP stages are dropped, keeping the SDPA -> o_proj pair
    that carries the failure. Taking and returning a hidden state is what lets
    the block stack: layer N's output is layer N+1's input, at identical
    shapes, so Dynamo shares one subgraph across the call sites.

    Deliberately more than a pointwise chain -- SDPA is a *Spyre
    decomposition*, so a region containing it exercises the decomp table
    threading in torch_spyre/_monkey_patch.py::_patch_invoke_subgraph
    _decompositions. The ``transpose(1, 2).reshape(...)`` view feeding o_proj
    is kept verbatim: it is the shape the failing o_proj batchmatmul's
    provenance points at, and it gives the linear a factorized layout coming
    out of SDPA's internal head tiling.
    """

    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(_NUM_HEADS * _HEAD_DIM, _HIDDEN, bias=False)

    def forward(self, hidden_states, q, key_cache, value_cache, attn_mask):
        attn_out = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=_ATTENTION_MULTIPLIER,
            enable_gqa=True,
        )
        # Collapse heads before o_proj: the combined H*D dim is the linear's
        # contraction dim, spanning both num_heads and head_dim.
        attn_out = attn_out.transpose(1, 2).reshape(_BATCH, _SEQLEN, -1)
        # hidden_states threads through so the block composes as a layer; the
        # add also keeps o_proj's result from being the only thing on the wire.
        return hidden_states + self.o_proj(attn_out)


@nested_compile_region
def _shared_attention_tail(block, hidden_states, q, key_cache, value_cache, mask):
    """Mirrors ``hf_common._shared_region_attention_tail``.

    ``block`` is passed POSITIONALLY and never closed over: a
    ``nested_compile_region`` body is traced once, so a closed-over block
    would bind every call site to the first layer's weights.
    """
    return block(hidden_states, q, key_cache, value_cache, mask)


class TestInvokeSubgraphAttention(_RegionTestCase):
    """A Granite-2B prefill attention tail in a region must compile and be accurate.

    Covers the realistic use of ``nested_compile_region``: the attention tail
    marked as a region and reused across layers, so the body is compiled once
    into an ``invoke_subgraph`` subgraph rather than inlined per layer.

    Reproduces the ``dxp_standalone`` failure from the Granite 3.3 2B
    whole-forward compile, where the o_proj batchmatmul lowered inside
    ``repeated_subgraph0`` comes out rank-4 and role-scrambled instead of a
    clean rank-3 ``[mb, out, in]``, so the backend scheduler finds more than
    one output-reuse (reduction) dimension and aborts with::

        error: sbf-ddc: DtException: out_reuse_dim.size() == 1
        error: sbf-run-scheduler-on-sdsc: failed on program 'sdsc_28'
    """

    @staticmethod
    def _inputs():
        """Build the inputs ONCE on CPU, for both paths to share.

        Generating them per-device would silently compare two different random
        problems: ``torch.randn`` does not reproduce values across devices or
        dtypes from the same seed.
        """
        torch.manual_seed(0)

        def randn(*shape):
            return torch.randn(*shape, dtype=torch.float16)

        hidden_states = randn(_BATCH, _SEQLEN, _HIDDEN)
        q = randn(_BATCH, _NUM_HEADS, _SEQLEN, _HEAD_DIM)
        key_cache = randn(_BATCH, _NUM_KVHEADS, _SEQLEN, _HEAD_DIM)
        value_cache = randn(_BATCH, _NUM_KVHEADS, _SEQLEN, _HEAD_DIM)
        # Causal additive mask, matching what generate() hands the block.
        mask = torch.full((_SEQLEN, _SEQLEN), float("-inf"), dtype=torch.float16)
        mask = torch.triu(mask, diagonal=1).view(1, 1, _SEQLEN, _SEQLEN)
        return hidden_states, q, key_cache, value_cache, mask

    def test_attention_tail_region_compiles_and_matches_cpu(self):
        torch.manual_seed(0)
        blocks = [_AttentionTail().eval() for _ in range(_NUM_LAYERS)]

        def outer(hidden_states, q, key_cache, value_cache, mask):
            h = hidden_states
            for block in blocks:
                h = _shared_attention_tail(block, h, q, key_cache, value_cache, mask)
            return h

        # Stacking is what keeps every call site's input shapes identical, so
        # Dynamo shares one subgraph instead of inlining per layer -- the HOP
        # assertion below is what catches a regression back to inlining.
        for block in blocks:
            block.to(device=DEVICE_NAME, dtype=torch.float16)

        seen_hops = []
        compiled = self._compile_counting_hops(outer, seen_hops)
        cpu_inputs = self._inputs()
        spyre_inputs = [t.to(DEVICE_NAME) for t in cpu_inputs]

        # .cpu() forces the launch, so a runtime (not just compile) failure surfaces.
        out = compiled(*spyre_inputs).cpu()

        self.assertEqual(tuple(out.shape), (_BATCH, _SEQLEN, _HIDDEN))
        self._assert_regions_not_inlined(seen_hops, expected=_NUM_LAYERS)
        self.assertTrue(
            torch.isfinite(out.to(torch.float32)).all(), "output has non-finite values"
        )

        # Same computation without the region, in eager CPU float32: an
        # inlined-but-wrong subgraph would still produce the right shape, so
        # numerics are the real assertion here.
        for block in blocks:
            block.to(device="cpu", dtype=torch.float32)

        def outer_cpu(hidden_states, q, key_cache, value_cache, mask):
            h = hidden_states
            for block in blocks:
                h = block(h, q, key_cache, value_cache, mask)
            return h

        compare_with_pytorch(
            None,
            outer_cpu,
            *[t.float() for t in cpu_inputs],
            atol=0.2,
            rtol=0.2,
            target=out.float(),
        )


_VOCAB = 128  # small: the embedding table's own size is irrelevant here


class TestInvokeSubgraphEmbeddingFedOperand(_RegionTestCase):
    """An embedding-fed region stack: every call site must get one operand layout.

    Same body as ``TestInvokeSubgraphAttention`` (SDPA -> o_proj), but the hidden
    state entering layer 0 comes from an ``nn.Embedding`` inside the graph instead
    of arriving as a graph input.

    Cost parity, not yet an optimization: the compiler inserts the SAME single
    copy the eager path open-coded -- one restickify of the embedding output, with
    every later call site already compliant. ``_subgraph_boundary_stl`` documents
    why one fixed boundary layout is the conservative first choice and how it
    could be sharpened later.

    Covers the OPERAND side of the boundary only. Subgraph RESULTS are still
    DECLARED to carry the generic layout by the ``MultiOutput`` branch of
    ``propagate_spyre_tensor_layouts`` rather than derived from the body; Granite
    3.3 2B happens to comply, so this test passes without exercising that half.
    A disagreement there no longer passes silently --
    ``_validate_subgraph_output_stls`` raises once the body has been propagated
    (see ``TestSubgraphOutputLayoutValidation``) -- but detection is not the same
    as support: a body that legitimately wants another orientation still cannot
    be compiled until the result layout is derived instead of declared.

    The asymmetry: layer 0's operand is the embedding output, committed as
    ``device_size=[1, 32, 512, 64]`` / ``stride_map=[-1, 64, 2048, 1]``; layers
    1..N-1 take the previous region's ``MultiOutput``, stamped with the generic
    layout ``[512, 32, 1, 64]`` / ``[2048, 64, -1, 1]``. The ``stride_map``
    contents agree and both put the same variable on the stick -- so
    ``stick_compatible`` calls them compatible and no ordinary restickify is
    planned -- but the 512 extent sits on a different device AXIS. Codegen derives
    device strides from ``device_size`` positionally
    (``_calculate_device_stride``), so the body, codegened once from the first
    call site, would address every later site's operand wrongly. Hence
    ``require_exact`` on the boundary edge in propagate_layouts.

    The body must MIX across the axis whose placement differs, or this test cannot
    see the bug: a pointwise body reads every element exactly once and writes each
    result back through the same addressing, so the permutation cancels and the
    output is bit-identical no matter which layout arrives. SDPA + o_proj contract
    over the sequence and hidden axes, so wrong addressing changes which values
    are combined -- which is exactly why the 40-layer model emitted wrong tokens
    (' Par' -> 'pec') rather than merely failing to compile.
    """

    def test_embedding_fed_attention_region_matches_cpu(self):
        torch.manual_seed(0)
        embed = nn.Embedding(_VOCAB, _HIDDEN).eval()
        blocks = [_AttentionTail().eval() for _ in range(_NUM_LAYERS)]

        def outer(ids, q, key_cache, value_cache, mask):
            # Mirrors hf_granite._run_backbone_forward's prologue MINUS the
            # transpose/contiguous round trip it used to need.
            h = embed(ids)
            for block in blocks:
                h = _shared_attention_tail(block, h, q, key_cache, value_cache, mask)
            return h

        embed.to(device=DEVICE_NAME, dtype=torch.float16)
        for block in blocks:
            block.to(device=DEVICE_NAME, dtype=torch.float16)

        seen_hops = []
        compiled = self._compile_counting_hops(outer, seen_hops)

        # Reuse the attention test's tensors, dropping its hidden_states (the
        # embedding produces the hidden state here) and adding token ids.
        _, q, key_cache, value_cache, mask = TestInvokeSubgraphAttention._inputs()
        ids = torch.randint(0, _VOCAB, (_BATCH, _SEQLEN))
        cpu_inputs = (ids, q, key_cache, value_cache, mask)
        spyre_inputs = [t.to(DEVICE_NAME) for t in cpu_inputs]

        # .cpu() forces the launch, so a runtime (not just compile) failure surfaces.
        out = compiled(*spyre_inputs).cpu()

        self.assertEqual(tuple(out.shape), (_BATCH, _SEQLEN, _HIDDEN))
        self._assert_regions_not_inlined(seen_hops, expected=_NUM_LAYERS)
        self.assertTrue(
            torch.isfinite(out.to(torch.float32)).all(), "output has non-finite values"
        )

        # The real gate: a body fed its operand through the wrong axis-order
        # addressing still produces the right shape and finite values, so only
        # value equality separates a correct boundary copy from a missing one.
        embed.to(device="cpu", dtype=torch.float32)
        for block in blocks:
            block.to(device="cpu", dtype=torch.float32)

        def outer_cpu(ids_cpu, q_, k_, v_, mask_):
            h = embed(ids_cpu)
            for block in blocks:
                h = block(h, q_, k_, v_, mask_)
            return h

        compare_with_pytorch(
            None,
            outer_cpu,
            ids,
            *[t.float() for t in cpu_inputs[1:]],
            atol=0.2,
            rtol=0.2,
            target=out.float(),
        )


class TestSubgraphOutputLayoutValidation(_RegionTestCase):
    """The parent's DECLARED subgraph-result layout must be checked against the body.

    ``propagate_spyre_tensor_layouts`` stamps ``generic_layout`` on every
    ``MultiOutput`` carrying an ``invoke_subgraph`` result, with
    ``AnyInNode.from_args()`` -- empty ``edge_costs``, zero cost, empty
    ``required_input_stls()``. So the layout is a declaration that nothing
    enforces. A body whose final op committed a different orientation would have
    its result addressed through the wrong device strides by the parent's
    consumers, with no error: the wrong-token class of failure, not a crash.

    ``_validate_subgraph_output_stls`` closes that by comparing the body's actual
    output layout against the parent's committed one once the body has been
    propagated. This test forces the disagreement the check exists to catch,
    because a check that cannot be shown to fire is worth nothing.

    The perturbation targets the LAST ``MultiOutput`` deliberately. In a stacked
    region, layer N's result is also layer N+1's operand, so perturbing any
    earlier one is caught first by the existing cross-site OPERAND check
    (``_subgraph_input_stls``) and would not exercise the output path at all. The
    last result feeds nothing downstream, so only the output check can see it.
    """

    def test_declared_output_layout_disagreeing_with_body_raises(self):
        from unittest.mock import patch

        from torch._inductor.ir import MultiOutput

        import torch_spyre._inductor.passes as _passes
        from torch_spyre._C import SpyreTensorLayout
        from torch_spyre._inductor.ir import FixedTiledLayout

        real_finalize = _passes.finalize_layouts
        perturbed = []

        def finalize_then_perturb(graph):
            real_finalize(graph)
            if getattr(graph, "parent", None) is not None:
                return  # only the parent declares result layouts
            mos = [op for op in graph.operations if isinstance(op, MultiOutput)]
            if not mos:
                return
            op = mos[-1]
            layout = op.maybe_get_layout()
            if not isinstance(layout, FixedTiledLayout):
                return
            dl = layout.device_layout
            dev_size = list(dl.device_size)
            stride_map = list(dl.stride_map)
            if len(dev_size) < 3:
                return
            # Swap the two outermost device axes: same extents, same stick, so
            # stick_compatible would accept it -- exactly the class of mismatch
            # that slips past ordinary compatibility checks.
            dev_size[0], dev_size[1] = dev_size[1], dev_size[0]
            stride_map[0], stride_map[1] = stride_map[1], stride_map[0]
            op.layout = FixedTiledLayout(
                layout.device,
                layout.dtype,
                layout.size,
                layout.stride,
                SpyreTensorLayout(
                    dev_size, stride_map, dl.device_dtype, dl.element_arrangement
                ),
                layout.offset,
            )
            perturbed.append(op.get_name())

        blocks = [_region(_Block()) for _ in range(2)]

        def outer(h):
            for b in blocks:
                h = b(h)
            return h

        h = torch.randn(8, 64, 128, dtype=torch.float16, device=DEVICE_NAME)

        with patch.object(_passes, "finalize_layouts", finalize_then_perturb):
            compiled = torch.compile(outer, dynamic=False, fullgraph=True)
            with self.assertRaises(Exception) as cm:
                compiled(h)

        self.assertTrue(
            perturbed,
            "no MultiOutput was perturbed, so the check was never given a "
            "disagreement to find",
        )
        message = str(cm.exception)
        self.assertIn("invoke_subgraph body", message)
        self.assertIn("produces output", message)

    def test_multi_output_body_compiles(self):
        """A body with SEVERAL outputs must not trip the check by mispairing them.

        The validator compares the body's outputs against the parent's declared
        result layouts, and those have to be paired BY POSITION. A region whose
        body returns more than one value -- which an autograd-partitioned region
        does, returning its result plus saved activations plus passthrough primals
        -- has outputs of unrelated shapes, so comparing all of them against all
        the parent's MultiOutputs reports a mismatch between a hidden state and a
        weight that no real disagreement produced.

        An embedding-fed stack is the shortest route to such a body: it makes the
        region take a produced hidden state, and the partitioner then gives the
        body three outputs. Before the fix this raised

            produces output ..._buf1 with device layout [64, 2, 8, 64]/...,
            but the parent declared [2, 128, 64]/... for buf4

        -- body output 0 (the hidden state) against the MultiOutput selecting
        index 2 (the weight).
        """
        embed = nn.Embedding(_VOCAB, 128).eval()
        embed.to(device=DEVICE_NAME, dtype=torch.float16)

        # The matmul must be INSIDE the region: that is what makes the partitioner
        # save activations, giving the body its extra outputs.
        @nested_compile_region
        def matmul_region(h, w):
            return torch.relu(h @ w)

        def outer(ids, w):
            h = embed(ids)
            for _ in range(3):
                h = matmul_region(h, w)
            return h

        ids = torch.randint(0, _VOCAB, (8, 64), device=DEVICE_NAME)
        w = torch.randn(128, 128, dtype=torch.float16, device=DEVICE_NAME)

        seen_hops = []
        compiled = self._compile_counting_hops(outer, seen_hops)
        out = compiled(ids, w).cpu()

        self.assertEqual(tuple(out.shape), (8, 64, 128))
        self._assert_regions_not_inlined(seen_hops, expected=2)


class TestSubgraphResultOperandRestickify(_RegionTestCase):
    """A subgraph RESULT feeding the next call site can be restickified.

    In a stacked region, layer N's result (a ``MultiOutput``) is layer N+1's
    operand. When its layout differs from the layout the body is compiled against,
    it has to be copied before the next call, like any intermediate operand. Two
    things had to work for that:

    - ``_subgraph_operand_args`` gives the operand an edge at all (it used to skip
      ``MultiOutput`` on the grounds that it always carried the generic layout);
    - ``_create_restickify_node`` can build a copy whose source is a
      ``MultiOutput``. The ``getitem`` node that selects a result is never in
      ``graph.env``, so the builder now makes the copy from the buffer itself.

    With the default boundary layout the results already comply and nothing is
    copied; the forced test swaps the boundary's two outer device axes so every
    chained result disagrees and must be copied.
    """

    @staticmethod
    def _stack_inputs():
        torch.manual_seed(0)
        embed = nn.Embedding(_VOCAB, 128).eval()
        w = torch.randn(128, 128, dtype=torch.float16) * 0.05
        ids = torch.randint(0, _VOCAB, (8, 64))
        return embed, w, ids

    def _compile_stack(self, embed, w, ids, boundary=None):
        from unittest.mock import patch

        import torch_spyre._inductor.insert_restickify as _ir
        import torch_spyre._inductor.propagate_layouts as _pl

        @nested_compile_region
        def region(h, weight):
            return torch.relu(h @ weight)

        def outer(tokens, weight):
            h = embed(tokens)
            for _ in range(3):
                h = region(h, weight)
            return h

        # (hop, restickified source, op-list positions of source/copy/hop)
        events = []
        real_insert = _ir.insert_restickify_on_subgraph_operands

        def recording_insert(op, resticks, operations):
            before = [i.maybe_get_name() for i in op.inputs]
            real_insert(op, resticks, operations)
            after = [i.maybe_get_name() for i in op.inputs]
            names = [o.get_name() for o in operations]
            # Each repointed slot names the source before and its copy after.
            for src, copy in zip(before, after):
                if src != copy:
                    events.append(
                        (
                            op.get_name(),
                            src,
                            names.index(src),
                            names.index(copy),
                            names.index(op.get_name()),
                        )
                    )

        patches = [
            patch.object(
                _ir, "insert_restickify_on_subgraph_operands", recording_insert
            )
        ]
        if boundary is not None:
            patches.append(patch.object(_pl, "_subgraph_boundary_stl", boundary))

        embed.to(device=DEVICE_NAME, dtype=torch.float16)
        seen_hops = []
        for p in patches:
            p.start()
        try:
            compiled = self._compile_counting_hops(outer, seen_hops)
            out = compiled(ids.to(DEVICE_NAME), w.to(DEVICE_NAME)).cpu().float()
        finally:
            for p in patches:
                p.stop()
        self._assert_regions_not_inlined(seen_hops, expected=3)
        return out, events

    @staticmethod
    def _reference(embed, w, ids):
        ref = nn.Embedding(_VOCAB, 128).eval()
        ref.load_state_dict({k: v.cpu().float() for k, v in embed.state_dict().items()})
        h = ref(ids)
        for _ in range(3):
            h = torch.relu(h @ w.float())
        return h

    def test_default_boundary_copies_no_result(self):
        embed, w, ids = self._stack_inputs()
        out, events = self._compile_stack(embed, w, ids)
        # Only the embedding (site 0) may need a copy; the chained results carry
        # the generic layout, which IS the default boundary.
        self.assertLessEqual(len(events), 1, events)
        torch.testing.assert_close(
            out, self._reference(embed, w, ids), atol=0.05, rtol=0.05
        )

    def test_forced_boundary_copies_every_chained_result(self):
        import torch_spyre._inductor.propagate_layouts as _pl
        from torch_spyre._C import SpyreTensorLayout

        real_boundary = _pl._subgraph_boundary_stl

        def swapped_boundary(buf):
            # Same extents, same stick (last axis untouched), different placement
            # of the outer coordinates -- stick_compatible accepts it, require_exact
            # does not, so every operand not already in it must be copied.
            stl = real_boundary(buf)
            dev_size = list(stl.device_size)
            stride_map = list(stl.stride_map)
            dev_size[0], dev_size[1] = dev_size[1], dev_size[0]
            stride_map[0], stride_map[1] = stride_map[1], stride_map[0]
            return SpyreTensorLayout(
                dev_size, stride_map, stl.device_dtype, stl.element_arrangement
            )

        embed, w, ids = self._stack_inputs()
        out, events = self._compile_stack(embed, w, ids, boundary=swapped_boundary)

        # Site 0 copies the embedding; sites 1 and 2 copy the previous site's
        # result. So three copies, and the last two have MultiOutput sources.
        self.assertEqual(len(events), 3, events)
        for hop, src, src_pos, copy_pos, hop_pos in events:
            self.assertLess(src_pos, copy_pos, f"{hop}: copy before its source")
            self.assertLess(copy_pos, hop_pos, f"{hop}: copy after its consumer")
        torch.testing.assert_close(
            out, self._reference(embed, w, ids), atol=0.05, rtol=0.05
        )


class TestSubgraphOutputConformance(_RegionTestCase):
    """A subgraph body returns the result layout its parent committed.

    The parent commits each result's MultiOutput -- and compiles its consumers --
    before the body is laid out, so the body has to meet that layout. Its output
    op's candidates are priced with an exit cost (ExitCostNode): the restickify
    from each candidate to the required layout. When the required layout is a free
    candidate the beam simply picks it; when the body cannot produce it, the body
    commits another layout, finalize_layouts plans a copy of the output, and
    insert_restickify makes it and has the body return the copy instead.

    Uses a mutating (auto-functionalized) region: its body output has two
    candidates, the same shape of problem a real model hits.
    """

    @staticmethod
    def _inputs():
        torch.manual_seed(0)
        h = torch.randn(8, 64, 128, dtype=torch.float16) * 0.1
        w = torch.randn(128, 128, dtype=torch.float16) * 0.05
        cache = torch.zeros(8, 64, 128, dtype=torch.float16)
        return h, w, cache

    @staticmethod
    def _reference(h, w, cache):
        c = cache.float().clone()
        out = h.float()
        for _ in range(2):
            c.add_(1.0)
            out = torch.relu(out * 2.0 + out @ w.float()) + c
        return out

    def _run(self, make_body_unable):
        from unittest.mock import patch

        import torch_spyre._inductor.insert_restickify as _ir
        import torch_spyre._inductor.passes as _passes
        from torch_spyre._inductor.propagate_layouts import (
            subgraph_required_output_layouts,
        )

        @nested_compile_region
        def region(cache, h, w):
            cache.add_(1.0)
            return torch.relu(h * 2.0 + h @ w) + cache

        def outer(h, w, cache):
            for _ in range(2):
                h = region(cache, h, w)
            return h

        removed = []
        real_propagate = _passes.propagate_spyre_tensor_layouts

        def propagate(graph):
            real_propagate(graph)
            if not make_body_unable or getattr(graph, "parent", None) is None:
                return
            # Take the required layout away from the body output's candidates, so
            # the body cannot produce it and a copy is the only way to comply.
            required = subgraph_required_output_layouts(graph)
            names = list(graph.get_output_names())
            for op in graph.operations:
                name = op.maybe_get_name()
                layouts = getattr(op, "layouts", None)
                if name not in names or not layouts:
                    continue
                want = {d.device_layout for _, d in required.get(names.index(name), [])}
                kept = [c for c in layouts if c not in want]
                if kept and len(kept) < len(layouts):
                    op.layouts[:] = kept
                    removed.append(name)

        copies = []
        real_create = _ir._create_restickify_node

        def create(info, op):
            result = real_create(info, op)
            copies.append(info.arg_name)
            return result

        body_outputs = []
        real_insert_outputs = _ir._insert_subgraph_output_restickifies

        def insert_outputs(graph, plan):
            before = len(copies)
            real_insert_outputs(graph, plan)
            if getattr(graph, "parent", None) is not None:
                body_outputs.append((list(graph.get_output_names()), copies[before:]))

        h, w, cache = self._inputs()
        seen_hops = []
        with (
            patch.object(_passes, "propagate_spyre_tensor_layouts", propagate),
            patch.object(_ir, "_create_restickify_node", create),
            patch.object(_ir, "_insert_subgraph_output_restickifies", insert_outputs),
        ):
            compiled = self._compile_counting_hops(outer, seen_hops)
            out = (
                compiled(h.to(DEVICE_NAME), w.to(DEVICE_NAME), cache.to(DEVICE_NAME))
                .cpu()
                .float()
            )
        self._assert_regions_not_inlined(seen_hops, expected=2)
        return out, removed, body_outputs, self._reference(h, w, cache)

    def test_required_layout_unavailable_body_returns_copy(self):
        out, removed, body_outputs, ref = self._run(make_body_unable=True)
        self.assertTrue(removed, "the required layout was never a candidate to remove")
        self.assertTrue(body_outputs, "no output restickify step ran on a body")
        names, copied = body_outputs[0]
        self.assertEqual(
            len(copied), 1, f"expected one output copy, got {copied} (outputs {names})"
        )
        self.assertNotIn(copied[0], names, "the body still returns the uncopied buffer")
        torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)

    def test_required_layout_free_body_makes_no_copy(self):
        out, removed, body_outputs, ref = self._run(make_body_unable=False)
        self.assertTrue(body_outputs, "no output restickify step ran on a body")
        _, copied = body_outputs[0]
        self.assertEqual(copied, [], "a copy was made although the layout was free")
        torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)


if __name__ == "__main__":
    unittest.main()
