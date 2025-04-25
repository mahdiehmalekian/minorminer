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

from typing import Iterable, Literal

from minorminer.utils.zephyr.node_edge import Edge, NodeKind, ZNode
from minorminer.utils.zephyr.plane_shift import PlaneShift

__all__ = ["ZephyrVHChain"]


class ZephyrVHChain(Edge):
    """
    Iitializes a chain with two ``ZNode``s. The two nodes must be internal neighbors.

    Args:
        z0 (ZNode): Node of chain. Must have the same shape as ``z1``.
        z1 (ZNode): Node of chain. Must have the same shape as ``z0``.

    Raises:
        ValueError: If ``z0`` and ``z1`` do not have the same shape.
        ValueError: If ``z0`` and ``z1``, as ``ZNode``s, are not internal neighbors.
    Example (1):
    >>> from burnaby.lattice_embedding.chain import ZephyrVHChain
    >>> from zephyr_utils.node_edge import ZNode
    >>> zvh = ZephyrVHChain(ZNode((1, 0)), ZNode((0, 1)))
    >>> print(f"{zvh} consists of one vertical ZNode {zvh.vertical} and one horizontal ZNode {zvh.horizontal}")
    ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))) consists of one vertical ZNode ZNode((0, 1)) and one horizontal ZNode ZNode((1, 0))

    Example (2):
    >>> z0, z1 = ZNode((0, 1)), ZNode((5, 0))
    >>> zvh = ZephyrVHChain(z0, z1) # raises ValueError since z0, z1 are not internal neighbors
    ValueError: Expected z0, z1 to be internal neighbours, got (ZNode((0, 1)), ZNode((5, 0)))

    Example (3):
    >>> z0, z1, z2, z3 = ZNode((0, 1)), ZNode((1, 0)), ZNode((2, 1)), ZNode((3, 0))
    >>> zvh1 = ZephyrVHChain(z0, z1)
    >>> zvh2 = ZephyrVHChain(z2, z3)
    >>> for coupling_kind in [None, ["01"], ["10"], ["01", "10"]]:
    ...     are_coupled = zvh1.is_coupled(zvh2, coupling_kind=coupling_kind)
    ...     if are_coupled:
    ...         print(f"{zvh1} and {zvh2} are connected via {coupling_kind} connection.")
    ...     else:
    ...         print(f"{zvh1} and {zvh2} are not connected via {coupling_kind} connection.")
    ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))) and ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0))) are connected via None connection.
    ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))) and ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0))) are not connected via ['01'] connection.
    ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))) and ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0))) are connected via ['10'] connection.
    ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))) and ZephyrVHChain(ZNode((2, 1)), ZNode((3, 0))) are not connected via ['01', '10'] connection.
    """

    def __init__(
        self,
        z0: ZNode,
        z1: ZNode,
    ) -> None:
        if z0.shape != z1.shape:
            raise ValueError(f"Expected z0, z1 to have the same shape, got {z0.shape, z1.shape}")
        if not z0.is_internal_neighbor(z1):
            raise ValueError(f"Expected z0, z1 to be internal neighbours, got {z0, z1}")

        self._edge: tuple[ZNode] = self._set_edge(z0, z1)

    def _set_edge(self, z0: ZNode, z1: ZNode) -> None:
        """Returns an ordered tuple corresponding to the z0, z1.

        Returns the tuple corresponding to the z0, z1, where the 0-th index is the vertical ``ZNode``
        and the 1-st index is the horizontal ``ZNode``."""
        if z0.node_kind is NodeKind.VERTICAL:
            return (z0, z1)
        else:
            return (z1, z0)

    def is_coupled(
        self,
        other: ZephyrVHChain,
        coupling_kind: Iterable[Literal["01", "10"]] | None = None,
    ) -> bool:
        """Indicates whether the chain is connected to another instance via coupling_kind edge(s).

        Args:
            other (ZephyrVHChain): Another instance to compare self with.
            coupling_kind (Iterable[Literal[&quot;01&quot;, &quot;10&quot;]] | None, optional):
                The coupling kind between self, other. Must be an Iterable of '01', '10'.
                    - ["01"] to indicate the existence of an edge from self[0] to other[1].
                    - ["10"] to indicate the existence of an edge from self[1] to other[0].
                    - ["01", "10"] to indicate the existence of an edge from self[0] to other[1] and an edge from self[1] to other[0].
                    - None to indicate the existence of an edge from self to other.
                Defaults to None.

        Returns:
            bool: True if the chain is connected to another instance via coupling_kind edge(s); False otherwise.
        """
        if coupling_kind is not None:
            set_kind = set(coupling_kind)
        self_ver, self_hor = self
        other_ver, other_hor = other
        if coupling_kind is None or set_kind == {}:
            return (self_ver.is_internal_neighbor(other_hor)) or (
                self_hor.is_internal_neighbor(other_ver)
            )
        if set_kind == {"01"}:
            return self_ver.is_internal_neighbor(other_hor)
        if set_kind == {"10"}:
            return self_hor.is_internal_neighbor(other_ver)
        return (self_ver.is_internal_neighbor(other_hor)) and (
            self_hor.is_internal_neighbor(other_ver)
        )

    @property
    def vertical(self) -> ZNode:
        """Returns the vertical ``ZNode`` of the chain."""
        return self._edge[0]

    @property
    def horizontal(self) -> ZNode:
        """Returns the horizontal ``ZNode`` of the chain."""
        return self._edge[1]

    def intersect(self, other: ZephyrVHChain) -> bool:
        """Indicates whether the chain has a node in common with another instance.

        Args:
            other (ZephyrVHChain): Another instance to compare self with.

        Returns:
            bool: True if the chain has a node in common with another instance; False otherwise.
        """
        return self.vertical == other.vertical or self.horizontal == other.horizontal

    def __iter__(self) -> Iterable[ZNode]:
        return iter(self._edge)

    def __hash__(self) -> int:
        return hash(self._edge)

    def __lt__(self, other: ZephyrVHChain) -> bool:
        return self._edge < other._edge

    def __le__(self, other: ZephyrVHChain) -> bool:
        return self._edge <= other._edge

    def __eq__(self, other: ZephyrVHChain) -> bool:
        return self._edge == other._edge

    def __ne__(self, other: ZephyrVHChain) -> bool:
        return self._edge != other._edge

    def __gt__(self, other: ZephyrVHChain) -> bool:
        return self._edge > other._edge

    def __ge__(self, other: ZephyrVHChain) -> bool:
        return self._edge >= other._edge

    def __add__(self, ps: PlaneShift) -> ZephyrVHChain:
        return ZephyrVHChain(self[0] + ps, self[1] + ps)

    def __hash__(self) -> int:
        return hash(self._edge)

    def __str__(self) -> str:
        return f"{type(self).__name__}{self._edge}"

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._edge}"
