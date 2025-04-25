# Copyright 2025 D-Wave
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
#
# ================================================================================================


from __future__ import annotations

from functools import cached_property
from itertools import combinations, combinations_with_replacement, product
from typing import Iterable, Literal

from minorminer.utils.lattice_embedding.chain import ZephyrVHChain
from minorminer.utils.zephyr.node_edge import Edge

__all__ = ["ZVHChainGraph"]


class ZVHChainGraph:
    """Constructs a graph on an iterable of nodes, each of which is a ZephyrVHChain.

        Supports optional ``coupling_kind`` and ``loops``.
        Edges or arcs are generated based on
        - ``loops`` (if True, generates loops) and
        - whether pairs of nodes are considered coupled under the specified ``coupling_kind``.

    Args:
        nodes (Iterable[ZephyrVHChain]): The nodes of the graph.
        coupling_kind (Iterable[Literal[&quot;01&quot;, &quot;10&quot;]] | None, optional):
            The coupling_kind required between two ZephyrVHChain nodes of the graph for them to have an edge between.
            Defaults to None.
        loops (bool, optional): If True, the edges of the graph include loops; False otherwise.
            Defaults to False.

    Example (1):
    >>> from zephyr_utils.node_edge import ZNode
    >>> from burnaby.lattice_embedding.chain import ZephyrVHChain
    >>> from burnaby.lattice_embedding.chain_graph import ZVHChainGraph
    >>> z0, z1, z2, z3 = ZNode((0, 1)), ZNode((1, 0)), ZNode((2, 1)), ZNode((3, 0))
    >>> nodes = [ZephyrVHChain(z0, z1), ZephyrVHChain(z2, z3)]
    >>> G1 = ZVHChainGraph(nodes=nodes)
    >>> G2 = ZVHChainGraph(nodes=nodes, loops=True)
    >>> print(f"On the set of nodes {nodes}: ")
    >>> print(f"Edges without considering loops are {G1.edges}")
    >>> print(f"Edges including loops are {G2.edges}")
    On the set of nodes [ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))), ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0)))]:
    Edges without considering loops are {Edge(ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))), ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0))))}
    Edges including loops are {Edge(ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0))), ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0)))), Edge(ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))), ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0)))), Edge(ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))), ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))))}

    Example (2):
    >>> print(f"On the set of {nodes = }: ")
    >>> for coupling_kind in [None, ["01"], ["10"], ["01", "10"]]:
    ...     G = ZVHChainGraph(nodes=nodes, coupling_kind=coupling_kind)
    ...     if G.is_connected():
    ...         print(f"with {coupling_kind = }, the graph is connected.")
    ...     else:
    ...         print(f"with {coupling_kind = }, the graph is not connected.")
    On the set of nodes = [ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))), ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0)))]:
    with coupling_kind = None, the graph is connected.
    with coupling_kind = ['01'], the graph is connected.
    with coupling_kind = ['10'], the graph is not connected.
    with coupling_kind = ['01', '10'], the graph is not connected.

    Example (3):
    >>> print(f"On the set of {nodes = }: ")
    >>> for coupling_kind in [None, ["01"], ["10"], ["01", "10"]]:
    ...     G = ZVHChainGraph(nodes=nodes, coupling_kind=coupling_kind)
    ...     try:
    ...         print(f"with {coupling_kind = }, the graph has edges {G.edges}.")
    ...     except AttributeError:
    ...         print(f"with {coupling_kind = }, the graph has arcs {G.arcs}.")
    On the set of nodes = [ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))), ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0)))]:
    with coupling_kind = ['01'], the graph has arcs {(ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0))), ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))))}.
    with coupling_kind = ['01', '10'], the graph has edges set().
    """

    def __init__(
        self,
        nodes: Iterable[ZephyrVHChain],
        coupling_kind: Iterable[Literal["01", "10"]] | None = None,
        loops: bool = False,
    ) -> None:
        self._nodes = list(set(nodes))

        if coupling_kind:
            self._coupling_kind = set(coupling_kind)
        else:
            self._coupling_kind = None
        self._loops: bool = loops

    @property
    def nodes(self) -> list[ZephyrVHChain]:
        """Returns nodes of the graph."""
        return self._nodes

    @property
    def coupling_kind(self) -> set[Literal["01", "10"]] | None:
        """Returns the ``coupling_kind`` of the graph.

        Returns the coupling_kind required between two nodes of the graph for them to have an edge between.
        """
        return self._coupling_kind

    @property
    def loops(self) -> bool:
        """Returns True if the graph allows loops, False otherwise."""
        return self._loops

    @cached_property
    def _arcs_or_edges(self) -> Literal["arcs", "edges"]:
        """Indicates whether the graph has edges or arcs.

        Decides whether the graph has edges (in case ``coupling_kind`` implies symmetric connection between nodes),
        or arcs (in case ``coupling_kind`` implies asymmetric connection between nodes)."""
        if self._coupling_kind in ({"01"}, {"10"}):
            return "arcs"
        return "edges"

    @cached_property
    def edges(self) -> set[Edge[ZephyrVHChain]]:  # A set of Edges of two ZephyrVHChain objects
        """Returns the set of edges of the graph if available.

        Returns the set of edges (including loops if self.loops) of the graph if available.
        Raises AttributeError if the graph has arcs instead."""
        if self._arcs_or_edges == "arcs":
            raise AttributeError(
                "'edges' is only available when self.coupling_kind is one of None, {}, or {'01', '10'}. "
                "Use 'arcs' or 'edges_or_arcs' instead."
            )
        if self._loops:
            iterator = combinations_with_replacement(self.nodes, 2)
        else:
            iterator = combinations(self.nodes, 2)
        return {
            Edge(node, node_nbr)
            for node, node_nbr in iterator
            if node.is_coupled(node_nbr, coupling_kind=self._coupling_kind)
        }

    @cached_property
    def arcs(
        self,
    ) -> set[tuple[ZephyrVHChain]]:  # A set of tuples of exactly two ZephyrVHChain objects
        """Returns the set of arcs of the graph if available.

        Returns the set of arcs (including loops if self.loops) of the graph if available.
        Raises AttributeError if the graph has edges instead."""
        if self._arcs_or_edges == "edges":
            raise AttributeError(
                "'arcs' is only available when self.coupling_kind is one of {'01'} or {'10'}. "
                "Use 'edges' or 'edges_or_arcs' instead."
            )
        iterator = product(self.nodes, self.nodes)
        return {
            (node, node_nbr)
            for node, node_nbr in iterator
            if (self._loops or (node != node_nbr))
            if node.is_coupled(node_nbr, coupling_kind=self._coupling_kind)
        }

    @property
    def edges_or_arcs(self) -> set[Edge[ZephyrVHChain]] | set[tuple[ZephyrVHChain]]:
        """Returns the edges or the arcs of the graph.

        Returns the edges or the arcs of the graph, whichever available based on ``coupling_kind``.
        """
        return getattr(self, self._arcs_or_edges)

    def incident_edges(
        self,
        node: ZephyrVHChain,
    ) -> set[Edge[ZephyrVHChain]]:
        """Returns the edges incident with ``node`` if available.

        Returns the edges (including loops if self.loops) of the graph incident with ``node`` if available.
            Raises AttributeError if the graph has arcs instead.

        Args:
            node (ZephyrVHChain): The ZephyrVHChain node whose incident edges are to be retrieved.

        Raises:
            AttributeError: If the graph has arcs instead.

        Returns:
            set[Edge[ZephyrVHChain]]: The set of edges of the graph that are incident with ``node``.
        """
        try:
            return {e for e in self.edges if node in e}
        except AttributeError:
            raise AttributeError(
                "'incident_edges' is only available when self.coupling_kind is one of None, {}, or {'01', '10'}. "
                "Use 'incident_arcs' instead."
            )

    def incident_arcs(
        self,
        node: ZephyrVHChain,
        direction: Literal["in", "out"] | None = None,
    ) -> set[tuple[ZephyrVHChain]]:
        """Returns the ``direction`` arcs incident with ``node`` if available.

        Returns the ``direction`` arcs (including loops if self.loops) of the graph incident with ``node`` if available.
            Raises AttributeError if the graph has edges instead.

        Args:
            node (ZephyrVHChain): The ZephyrVHChain node whose incident ``direction`` arcs are to be retrieved.
            direction (Literal[&quot;in&quot;, &quot;out&quot;] | None, optional):
                The direction of incident arcs to be retrieved. If provided, must be "in" or "out" or None.
                - "in" to retrieve the arcs that end at ``node``.
                - "out" to retrieve the arcs that start at ``node``.
                - None to retrieve the arcs that start or end at ``node``.

        Raises:
            AttributeError:If the graph has edges instead.

        Returns:
            set[tuple[ZephyrVHChain]]:
                The set of arcs of the graph that are incident with ``node`` and have ``direction`` direction.
        """
        if direction == "out":
            index = 0
        elif direction == "in":
            index = 1
        else:
            index = None

        try:
            if index:
                return {e for e in self.arcs if e[index] == node}
            else:
                return {e for e in self.arcs if e[0] == node or e[1] == node}
        except AttributeError:
            raise AttributeError(
                "'incident_arcs' is only available when self.coupling_kind is one of {'01'} or {'10'}. "
                "Use 'incident_edges' instead."
            )

    def incident_edges_or_arcs(
        self,
        node: ZephyrVHChain,
        direction: Literal["in", "out"] | None = None,
    ) -> set[Edge[ZephyrVHChain]] | set[tuple[ZephyrVHChain]]:
        """Returns edges or the ``direction`` arcs incident with ``node``.

        Returns the set of edges or the ``direction`` arcs of the graph incident with ``node``,
            whichever available based on ``coupling_kind``.

        Args:
            node (ZephyrVHChain):
                The node whose incident edges or ``direction`` arcs are to be retrieved.
            direction (Literal[&quot;in&quot;, &quot;out&quot;] | None, optional):
                The direction of incident arcs to be retrieved. If provided, must be "in" or "out" or None.
                - "in" to retrieve the arcs that end at ``node``.
                - "out" to retrieve the arcs that start at ``node``.
                - None to retrieve the arcs that start or end at ``node``.
        Returns:
            set[Edge[ZephyrVHChain]] | set[tuple[ZephyrVHChain]]:
                The set of edges or the ``direction`` arcs of the graph incident with ``node``, whichever available based on ``coupling_kind``.
        """
        _edge_arc = self._arcs_or_edges
        attr = f"incident_{_edge_arc}"
        method = getattr(self, attr)
        if _edge_arc == "edges":
            return method(node)
        return method(node=node, direction=direction)

    def dfs(
        self,
        node: ZephyrVHChain,
        visited: list[ZephyrVHChain] | None = None,
    ) -> list[ZephyrVHChain]:
        """Performs DFS on the graph.

        Returns the list of all nodes reachable in the graph from the given ``node`` using a depth-first search (DFS).

        Args:
            node (ZephyrVHChain): The starting node for the DFS traversal.
            visited (list[ZephyrVHChain] | None, optional):
                The list of nodes used to track visited nodes during traversal. This is modified in-place.
                Defaults to None.

        Returns:
            list[ZephyrVHChain]: The list of all ZephyrVHChains reachable in the graph from ``node``.
        """
        if visited is None:
            visited = []
        if node in visited:
            return visited
        visited.append(node)
        if self._arcs_or_edges == "arcs":
            for e in self.incident_arcs(node=node, direction="out"):
                node_outnbr = e[1]
                if node_outnbr in visited:
                    continue
                self.dfs(node=node_outnbr, visited=visited)
        else:
            for e in self.incident_edges(node=node):
                node_nbr = next(iter(set(e) - {node}))
                self.dfs(node=node_nbr, visited=visited)
        return visited

    def is_connected(self) -> bool:
        """Returns True if the graph is connected; False otherwise."""
        node = self._nodes[0]
        return len(self.dfs(node=node)) == len(self._nodes)
