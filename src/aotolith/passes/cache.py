"""Collapse a decode step's KV-cache roll into one operator.

The frontend has to spell the roll out in torch terms: take the layer's pair out
of the stacked cache, slice the oldest slot off the window, squeeze the two new
rows into a pair, concatenate it onto the window, and slice the pair apart again
so attention can read the key and the value separately. That is eleven operators
per layer, and because every one of them is a launch, a decode step spends about
a fifth of its time on twenty-four of these chains.

What the chain *means* is one statement: "slide the window down one slot, write
this step's key and value into the freed slot, and give me the window and the two
halves". That is :op:`kv_cache_append`, and this pass rewrites the eleven
operators into it.

The rewrite is deliberately narrow. It fires only when

* the concatenate builds a pair of length ``slots`` from a window of the same
  pair that is shorter by exactly one slot on the key axis,
* that window is a slice of the very pair being appended to, dropping slot zero,
* the pair is reached from the stacked cache by nothing but the layer slice, and
* the two halves are unpacked from the concatenate by a slice and a squeeze
  whose results have exactly the shapes the new operator produces.

Anything else - a window that drops more than one slot, a pair read by someone
other than the two halves, a value the chain produced that is exported or read
elsewhere, a cache that is not layer-stacked - is left alone, so a graph that
only looks similar keeps its general form.
"""

from __future__ import annotations

from ..ir import Graph, Operation
from .graph import _consumers, _producers, _replace_graph

#: The shape-changing operators a layer slice may hide behind. They only move
#: elements around, so the pair a decode step appends to can be reached through
#: them without changing what the write means.
_LAYOUT_OPS = ("squeeze", "reshape")


def _pair_source(
    graph: Graph, value_id: str, producers: dict[str, int]
) -> tuple[str, int, tuple[int, ...]] | None:
    """Return ``(cache, layer, ops_to_delete)`` when ``value_id`` is one layer."""

    current = value_id
    transparent: list[int] = []
    for _ in range(4):
        index = producers.get(current)
        if index is None:
            return None
        op = graph.ops[index]
        if op.op in _LAYOUT_OPS:
            transparent.append(index)
            current = op.inputs[0]
            continue
        if op.op != "slice":
            return None
        source_type = graph.value(op.inputs[0]).type
        rank = len(source_type.shape)
        if rank != 6 or int(op.attrs["dim"]) % rank != 0:
            return None
        start, end = op.attrs["start"], op.attrs["end"]
        if end - start != 1:
            return None
        if source_type.shape[1] != 2:
            return None
        transparent.append(index)
        return op.inputs[0], start, tuple(transparent)
    return None


def _stacked_row(
    graph: Graph,
    value_id: str,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
) -> tuple[str, str, tuple[int, ...]] | None:
    """Return ``(key, value, ops_to_delete)`` for the stacked pair of new rows."""

    index = producers.get(value_id)
    if index is None:
        return None
    stack = graph.ops[index]
    if stack.op != "cat" or len(stack.inputs) != 2:
        return None
    if int(stack.attrs["dim"]) != 0:
        return None
    rows: list[str] = []
    deletes: list[int] = [index]
    for operand in stack.inputs:
        operand_index = producers.get(operand)
        if operand_index is None:
            return None
        unsqueeze = graph.ops[operand_index]
        if unsqueeze.op != "unsqueeze" or int(unsqueeze.attrs["dim"]) != 0:
            return None
        if consumers.get(operand) != [index]:
            return None
        rows.append(unsqueeze.inputs[0])
        deletes.append(operand_index)
    return rows[0], rows[1], tuple(deletes)


def _projected_value(
    graph: Graph,
    value_id: str,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
    expected: tuple[int, ...],
) -> tuple[str, int, tuple[int, ...]] | None:
    """Return ``(source, offset, ops_to_delete)`` for a value read out of a row.

    The frontend spells the projection's value half as a slice of the packed
    row, a reshape into heads, and a transpose to put the head axis second.
    None of those three moves a value: they exist because the projection writes
    one row and attention wants one head layout. Reading the row directly from
    the cache operator's kernel removes all three.
    """

    transposed_index = producers.get(value_id)
    if transposed_index is None:
        return None
    transposed = graph.ops[transposed_index]
    if transposed.op != "transpose":
        return None
    reshaped_index = producers.get(transposed.inputs[0])
    if reshaped_index is None:
        return None
    reshaped = graph.ops[reshaped_index]
    if reshaped.op != "reshape":
        return None
    sliced_index = producers.get(reshaped.inputs[0])
    if sliced_index is None:
        return None
    sliced = graph.ops[sliced_index]
    if sliced.op != "slice":
        return None
    source_type = graph.value(sliced.inputs[0]).type
    if len(source_type.shape) != 3:
        return None
    rank = len(source_type.shape)
    if int(sliced.attrs["dim"]) % rank != 2:
        return None
    start = int(sliced.attrs["start"])
    batch, heads, head_dim = expected[1], expected[2], expected[4]
    # The frontend leaves the end open when the slice runs to the row's end, so
    # the width to check against is the one the row actually has.
    end = min(int(sliced.attrs["end"]), source_type.shape[2])
    if end - start != heads * head_dim:
        return None
    if tuple(int(dim) for dim in reshaped.attrs["shape"]) != (
        batch,
        1,
        heads,
        head_dim,
    ):
        return None
    if int(transposed.attrs["dim0"]) != 1 or int(transposed.attrs["dim1"]) != 2:
        return None
    return sliced.inputs[0], start, (transposed_index, reshaped_index, sliced_index)


def _half(
    graph: Graph,
    half: int,
    new_pair_id: str,
    producers: dict[str, int],
    consumers: dict[str, list[int]],
) -> tuple[str, tuple[int, ...]] | None:
    """Return ``(window, ops_to_delete)`` for ``new_pair[half]`` unpacked."""

    for sliced_index in consumers.get(new_pair_id, ()):
        sliced = graph.ops[sliced_index]
        if sliced.op != "slice" or sliced.inputs[0] != new_pair_id:
            continue
        rank = len(graph.value(new_pair_id).type.shape)
        if int(sliced.attrs["dim"]) % rank != 0:
            continue
        if (sliced.attrs["start"], sliced.attrs["end"]) != (half, half + 1):
            continue
        squeeze_indices = consumers.get(sliced.outputs[0], ())
        if len(squeeze_indices) != 1:
            continue
        squeeze_index = squeeze_indices[0]
        squeeze = graph.ops[squeeze_index]
        if squeeze.op != "squeeze":
            continue
        return squeeze.outputs[0], (sliced_index, squeeze_index)
    return None


def fuse_rope_into_kv_cache_update(graph: Graph) -> Graph:
    """Fold a layer's two ``rope`` nodes into its :op:`kv_cache_append`.

    A decode step rotates the query half and the key half of one projection row,
    and then the key goes straight into the cache. The frontend spells that as a
    slice, a reshape and a ``rope`` per half, which is two launches and two
    layout chains per layer for under a kilobyte of arithmetic - affordable at
    bf16, where a step is waiting on memory anyway, and the largest remaining
    fixed cost at four bits.

    The match is narrow. The key has to feed a ``kv_cache_append`` and nothing
    else, the query has to be read by one ``attention``, both rotations have to
    share one table pair, and both have to read the row the cache operator
    already reads. Anything else keeps its two rotations.
    """

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)

    def layout_root(
        value_id: str,
    ) -> tuple[str, int, tuple[int, ...]] | None:
        """Walk a view chain back to a projection row and its slice start."""

        chain: list[int] = []
        current = value_id
        for _ in range(6):
            index = producers.get(current)
            if index is None:
                return None
            op = graph.ops[index]
            if op.op in _LAYOUT_OPS or op.op == "transpose":
                chain.append(index)
                current = op.inputs[0]
                continue
            if op.op != "slice":
                return None
            source_type = graph.value(op.inputs[0]).type
            if len(source_type.shape) != 3:
                return None
            if int(op.attrs["dim"]) % 3 != 2:
                return None
            chain.append(index)
            return op.inputs[0], int(op.attrs["start"]), tuple(chain)
        return None

    #: Every rotation in the graph, keyed by the row it reads, so the query that
    #: belongs to a layer's key is a lookup rather than a rescan.
    rotations: dict[str, list[tuple[int, int, tuple[int, ...]]]] = {}
    for index, op in enumerate(graph.ops):
        if op.op != "rope" or len(op.inputs) != 3:
            continue
        root = layout_root(op.inputs[0])
        if root is None:
            continue
        rotations.setdefault(root[0], []).append((index, root[1], root[2]))

    rewrites: dict[int, Operation] = {}
    removable: set[int] = set()
    observed = set(graph.outputs)

    for index, op in enumerate(graph.ops):
        if op.op != "kv_cache_append":
            continue
        cache_id, key_id, row_id = op.inputs
        key_rope_index = producers.get(key_id)
        if key_rope_index is None:
            continue
        key_rope = graph.ops[key_rope_index]
        if key_rope.op != "rope" or consumers.get(key_id) != [index]:
            continue
        cosine_id, sine_id = key_rope.inputs[1], key_rope.inputs[2]
        key_root = layout_root(key_rope.inputs[0])
        if key_root is None or key_root[0] != row_id:
            continue
        row_type = graph.value(row_id).type
        if len(row_type.shape) != 3 or row_type.shape[1] != 1:
            continue
        cache_type = graph.value(cache_id).type
        if len(cache_type.shape) != 6:
            continue
        batch, _heads, head_dim = (
            cache_type.shape[2],
            cache_type.shape[3],
            cache_type.shape[5],
        )

        query: tuple[int, int, tuple[int, ...]] | None = None
        for candidate in rotations.get(row_id, ()):
            candidate_index, candidate_start, candidate_chain = candidate
            if candidate_index in (index, key_rope_index):
                continue
            rotation = graph.ops[candidate_index]
            if rotation.inputs[1] != cosine_id or rotation.inputs[2] != sine_id:
                continue
            readers = consumers.get(rotation.outputs[0], ())
            if len(readers) != 1 or graph.ops[readers[0]].op != "attention":
                continue
            if graph.value(rotation.outputs[0]).type.shape[-1] != head_dim:
                continue
            query = (candidate_index, candidate_start, candidate_chain)
            break
        if query is None:
            continue
        query_index, query_start, query_chain = query
        query_type = graph.value(graph.ops[query_index].outputs[0]).type
        if len(query_type.shape) != 4 or query_type.shape[0] != batch:
            continue
        if query_type.shape[2] != 1:
            continue

        deleted = {key_rope_index, query_index, *key_root[2], *query_chain}
        # None of the work this rewrite deletes may be read from outside it, or
        # a value would lose its producer. The two rotations' outputs are the
        # exception: the new operator writes the query, and the key only ever
        # fed the cache operator.
        rehomed = {graph.ops[query_index].outputs[0]}
        shared = any(
            consumer not in deleted and consumer != index
            for op_index in deleted
            for value_id in graph.ops[op_index].outputs
            if value_id not in rehomed
            for consumer in consumers.get(value_id, ())
        ) or any(
            value_id in observed
            for op_index in deleted
            for value_id in graph.ops[op_index].outputs
            if value_id not in rehomed
        )
        if shared:
            continue

        rewrites[index] = Operation(
            "rope_kv_cache_append",
            (cache_id, row_id, cosine_id, sine_id),
            (graph.ops[query_index].outputs[0], *op.outputs),
            {
                "layer": op.attrs["layer"],
                "query_start": query_start,
                "key_start": key_root[1],
                "value_start": op.attrs["value_start"],
                "query_heads": query_type.shape[1],
            },
        )
        removable.update(deleted)

    if not rewrites:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = rewrites.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index in removable:
            continue
        else:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)


def fuse_kv_cache_update(graph: Graph) -> Graph:
    """Replace the per-layer roll chain with one :op:`kv_cache_append`."""

    graph.validate()
    producers = _producers(graph)
    consumers = _consumers(graph)
    rewrites: dict[int, Operation] = {}
    removable: set[int] = set()

    for index, op in enumerate(graph.ops):
        if op.op != "cat" or len(op.inputs) != 2:
            continue
        new_pair_id = op.outputs[0]
        new_pair_type = graph.value(new_pair_id).type
        if len(new_pair_type.shape) != 5:
            continue
        if new_pair_type.shape[0] != 2 or new_pair_type.shape[3] < 2:
            continue
        key_axis = int(op.attrs["dim"]) % 5
        if key_axis != 3:
            continue

        window_id, row_id = op.inputs
        window_type = graph.value(window_id).type
        row_type = graph.value(row_id).type
        expected_window = list(new_pair_type.shape)
        expected_window[key_axis] -= 1
        expected_row = list(new_pair_type.shape)
        expected_row[key_axis] = 1
        if list(window_type.shape) != expected_window:
            continue
        if list(row_type.shape) != expected_row:
            continue

        # The window is the pair minus its first slot, slice-of-the-same-pair
        # and nothing else: that is what makes this a roll rather than an
        # append onto an arbitrary window.
        window_index = producers.get(window_id)
        if window_index is None:
            continue
        window_op = graph.ops[window_index]
        if window_op.op != "slice":
            continue
        if int(window_op.attrs["dim"]) % len(window_type.shape) != key_axis:
            continue
        slots = new_pair_type.shape[key_axis]
        if window_op.attrs["start"] != 1 or window_op.attrs["end"] < slots:
            continue
        if consumers.get(window_id) != [index]:
            continue
        pair_id = window_op.inputs[0]
        if graph.value(pair_id).type != new_pair_type:
            continue

        source = _pair_source(graph, pair_id, producers)
        if source is None:
            continue
        cache_id, layer, layer_ops = source

        row = _stacked_row(graph, row_id, producers, consumers)
        if row is None:
            continue
        key_id, value_id, row_ops = row
        projected = _projected_value(
            graph, value_id, producers, consumers, new_pair_type.shape
        )
        if projected is None:
            continue
        value_source, value_start, value_ops = projected

        expected_half = (
            new_pair_type.shape[1],
            new_pair_type.shape[2],
            new_pair_type.shape[3],
            new_pair_type.shape[4],
        )
        halves = [
            _half(graph, half, new_pair_id, producers, consumers) for half in (0, 1)
        ]
        if any(half is None for half in halves):
            continue
        windows = [half[0] for half in halves]
        half_ops = [op_index for half in halves for op_index in half[1]]
        if any(graph.value(window).type.shape != expected_half for window in windows):
            continue
        if graph.value(windows[0]).type.dtype != new_pair_type.dtype:
            continue
        if graph.value(key_id).type.dtype != new_pair_type.dtype:
            continue

        # Everything this rewrite deletes has to be read by the chain alone, or
        # a value would vanish from under another consumer. The three values the
        # new operator hands back are the exception: every reader of those keeps
        # reading them, now from the operator instead of from the chain.
        deleted = {
            window_index,
            *row_ops,
            *layer_ops,
            *half_ops,
            *value_ops,
        }
        rehomed = {new_pair_id, windows[0], windows[1]}
        # A graph output is a reader the consumer table cannot see, and a value
        # that is both deleted and observed would leave the graph without its
        # producer, so it counts as a share like any other.
        observed = set(graph.outputs)
        shared = any(
            consumer not in deleted and consumer != index
            for op_index in deleted
            for value_id in graph.ops[op_index].outputs
            if value_id not in rehomed
            for consumer in consumers.get(value_id, ())
        ) or any(
            value_id in observed
            for op_index in deleted
            for value_id in graph.ops[op_index].outputs
            if value_id not in rehomed
        )
        if shared:
            continue
        rewrites[index] = Operation(
            "kv_cache_append",
            (cache_id, key_id, value_source),
            (new_pair_id, windows[0], windows[1]),
            {"layer": layer, "value_start": value_start},
        )
        removable.update(deleted)

    if not rewrites:
        return graph

    rewritten: list[Operation] = []
    for index, op in enumerate(graph.ops):
        replacement = rewrites.get(index)
        if replacement is not None:
            rewritten.append(replacement)
        elif index in removable:
            continue
        else:
            rewritten.append(op)
    return _replace_graph(graph, tuple(rewritten), graph.outputs)
