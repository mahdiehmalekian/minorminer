# Copyright 2026 D-Wave
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minorminer.utils._clique.grid import Grid, Node

# offset patterns (dx, dy); internal couplers may change k, odd and external keep it.
_INTERNAL_OFFSETS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_V_SAME_K_OFFSETS = ((0, 2), (0, -2), (0, 4), (0, -4))  # odd + external
_H_SAME_K_OFFSETS = ((2, 0), (-2, 0), (4, 0), (-4, 0))

# deepest bridge implemented: a -- m1 -- m2 -- b
MAX_IMPLEMENTED_BRIDGE_NODES = 2


def neighbors(grid: Grid, ccoord: Node) -> list[Node]:
    """All neighbors of a node in a possibly faulty grid.

    A neighbor is a present node joined to ``ccoord`` by a working coupler of
    any type (internal, odd, or external). An absent ``ccoord`` has none.

    Args:
        grid: The grid to find the neighbors of the node in.
        ccoord: The node, represented with cartesian coordinates, to find the neighbors for.

    Returns:
        Neighbors of the node in the grid.
    """
    x, y, k = ccoord
    ccoord_to_lcoord = grid.cartesian_to_linear
    has_edge = grid.has_edge
    t = grid.t
    node_lcoord = ccoord_to_lcoord(ccoord)
    if node_lcoord is None:
        return []
    out = []
    if grid.is_vertical(ccoord):
        same_k_offsets = _V_SAME_K_OFFSETS
    else:
        same_k_offsets = _H_SAME_K_OFFSETS
    for dx, dy in _INTERNAL_OFFSETS:
        for kp in range(t):
            nbr_ccoord = (x + dx, y + dy, kp)
            nbr_lcoord = ccoord_to_lcoord(nbr_ccoord)
            if nbr_lcoord is not None and has_edge(node_lcoord, nbr_lcoord):
                out.append(nbr_ccoord)
    for dx, dy in same_k_offsets:
        nbr_ccoord = (x + dx, y + dy, k)
        nbr_lcoord = ccoord_to_lcoord(nbr_ccoord)
        if nbr_lcoord is not None and has_edge(node_lcoord, nbr_lcoord):
            out.append(nbr_ccoord)
    return out


def find_interchain_bridges(
    grid: Grid,
    chain_a: Collection[Node],
    chain_b: Collection[Node],
    used_nodes: set[Node],
    max_interchain_bridge_nodes: int = 2,
) -> list[tuple[Node, ...]]:
    """Find short interchain bridges between two chains.

    An interchain bridge between ``chain_a`` and ``chain_b`` is the sequence of internal nodes of
    a path in a possibly faulty grid that starts at a node of ``chain_a``, ends at a node of ``chain_b``,
    and whose internal nodes are all free, i.e. not in ``used_nodes``, ``chain_a``, or ``chain_b``.
    The endpoints stay in their chains; only the internal nodes form the bridge, listed in path
    order from the ``chain_a`` end to the ``chain_b`` end. The empty bridge ``()`` comes from a
    path with no internal nodes, i.e. a single edge joining the two chains.

    This function returns every distinct bridge with at most ``max_interchain_bridge_nodes``
    nodes. Paths that differ only in their endpoints give the same bridge, which is listed once.
    If the chains are already joined by an edge, only the empty bridge is returned.

    Note:
        Bridges with more than ``MAX_IMPLEMENTED_BRIDGE_NODES`` nodes are not implemented, so
        larger values of ``max_interchain_bridge_nodes`` are treated as
        ``MAX_IMPLEMENTED_BRIDGE_NODES``.

    Args:
        grid: The grid to find a bridge between the two chains in.
        chain_a: One of the chains.
        chain_b: The other chain.
        used_nodes: Nodes already used by other chains; a bridge cannot use them.
        max_interchain_bridge_nodes: Maximum number of middle nodes in a bridge,
            not counting the two endpoints. Values above 2 are treated as 2.
            With 0, only the direct-connection check is done: the result is
            ``[()]`` if the chains are directly connected and ``[]`` otherwise.

    Returns:
        A list of tuples of intermediate nodes, ordered by number of middle
        nodes (fewest first). Each tuple is one bridge:

        - ``()``: The two chains are already directly connected (no bridge
          needed). If present, it is the only element.
        - ``(m1,)``: One middle node.
        - ``(m1, m2)``: Two middle nodes.
    """
    A = set(chain_a)
    B = set(chain_b)

    def is_free(n: Node) -> bool:
        return n not in used_nodes and n not in A and n not in B

    # One pass over A's neighbors does two jobs:
    # 1) it detects a direct A--B edge (the result is then just the empty bridge),
    # 2) it collects A-side candidates.
    a_neigh: set[Node] = set()
    for a in A:
        for nbr in neighbors(grid, a):
            if nbr in B:
                return [()]
            if is_free(nbr):
                a_neigh.add(nbr)

    if max_interchain_bridge_nodes < 1:
        return []

    # B-side candidates; only membership is tested.
    b_neigh: set[Node] = {nbr for b in B for nbr in neighbors(grid, b) if is_free(nbr)}

    all_bridges: list[tuple[Node, ...]] = [(m,) for m in a_neigh if m in b_neigh]

    if max_interchain_bridge_nodes < 2:
        return all_bridges

    # Look for a -- m1 -- m2 -- b. m1 adjacent to A, m2 adjacent to B, m1--m2 edge.
    # Enumerate from A-side middles, hop once more, check adjacency to B.
    for bridge_node_1 in a_neigh:
        for bridge_node_2 in neighbors(grid, bridge_node_1):
            if bridge_node_2 in b_neigh:
                all_bridges.append((bridge_node_1, bridge_node_2))
    return all_bridges
