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


from functools import cached_property
from itertools import product

import networkx as nx
from dwave.system import DWaveSampler
from minorminer.utils.lattice_embedding.auxiliary_coordinates import UWJ, UWKJ, UWKJZ
from minorminer.utils.zephyr.node_edge import Edge, ZShape
from minorminer.utils.zephyr.survey import ZSE, ZSurvey

__all__ = ["LatticeSurvey"]


class LatticeSurvey(ZSurvey):
    """
    A subclass of ZSurvey that provides convenient representations and helpers
    for embedding algorithms of lattices on Zephyr topology.
    Enables calculating the stretch of external paths.
    Takes a Zephyr graph or DWaveSampler with Zephyr topology.

        Args:
            G (nx.Graph | DWaveSampler): A graph or DWaveSampler with Zephyr topology.

    Example:
    >>> from dwave_networkx import zephyr_graph
    >>> from burnaby.lattice_embedding.lattice_survey import LatticeSurvey
    >>> m = 3
    >>> G = zephyr_graph(m=m, t=4)
    >>> G.remove_node(1)
    >>> G.remove_node(20)
    >>> lsurvey = LatticeSurvey(G)
    >>> print(f"The number of missing nodes is {lsurvey.num_missing_nodes}. The missing nodes are {lsurvey.missing_nodes}")
    The number of missing nodes is 2. The missing nodes are {UWKJZ(u=0, w=0, k=3, j=0, z=2), UWKJZ(u=0, w=0, k=0, j=0, z=1)}
    >>> for uwj, uwj_dict in lsurvey_external_paths_stretch.items():
    ...     for k, z_stretches in uwj_dict.items():
    ...         if z_stretches != [ZSE(z_start=0, z_end=m - 1)]:
    ...             external_path = UWKJ(u=uwj.u, w=uwj.w, k=k, j=uwj.j)
    ...             print(f"The external path {external_path} does not extend across the full z-range.")
    The external path UWKJ(u=0, w=0, k=0, j=0) does not extend across the full z-range.
    The external path UWKJ(u=0, w=0, k=3, j=0) does not extend across the full z-range.
    """

    def __init__(
        self,
        G: nx.Graph | DWaveSampler,
    ) -> None:
        self._zsurvey = ZSurvey(G)

        self._nodes: set[UWKJZ] = {UWKJZ(*a.zcoord) for a in self._zsurvey.nodes}
        self._edges: set[tuple[UWKJZ]] = {
            (UWKJZ(*a.zcoord), UWKJZ(*b.zcoord)) for (a, b) in self._zsurvey.edges
        }

    def zsurvey(self) -> ZSurvey:
        """Returns the ZSurvey of ``G``."""
        return self._zsurvey

    @cached_property
    def missing_nodes(self) -> set[UWKJZ]:
        """Returns the missing nodes of the sampler or graph.

        Returns:
            set[UWKJZ]: The set of ``UWKJZ``s corresponding to the nodes of the sampler or graph
        which are missing compared to perfect yield Zephyr graph with the same shape.
        """
        return {UWKJZ(*a.zcoord) for a in self._zsurvey.missing_nodes}

    @property
    def shape(self) -> ZShape:
        """Returns the ``ZShape`` of G"""
        return self._zsurvey.shape

    @cached_property
    def missing_edges(self) -> set[Edge]:
        """Returns the missing edges of the sampler or graph.

        Returns:
            set[Edge]: The set of edges of the sampler or graph which are missing compared to
            perfect yield Zephyr graph on the same shape.
        """
        return {Edge(UWKJZ(*a.zcoord), UWKJZ(*b.zcoord)) for (a, b) in self._zsurvey.missing_edges}

    @property
    def extra_missing_edges(self) -> set[Edge]:
        """Returns the missing edges of the sampler or graph not incident with a missing node.

        Returns:
            set[Edge]: The Edges of the sampler or graph which are missing compared to
            perfect yield Zephyr graph on the same shape and are not incident with
            a missing node.
        """
        return {
            Edge(UWKJZ(*a.zcoord), UWKJZ(*b.zcoord)) for (a, b) in self._zsurvey.extra_missing_edges
        }

    def calculate_external_paths_stretch(self) -> dict[UWJ, dict[int, list[ZSE]]]:
        """Calculates the stretch of external paths of the graph/sampler.

        Returns:
        dict[UWJ, dict[int, list[ZSE]]]:
            A nested dictionary of the form:
                { uwj: { k: [ZSE(z_start, z_end), ...] } }

            - Outer keys are quotients of external paths, represented as ``UWJ``.
            - Each ``UWJ`` maps to a dictionary whose:
                - Keys are ``k`` values from external paths (``UWKJ``) such that ``UWKJ.uwj == uwj``.
                - Values are lists of ``ZSE`` objects representing the maximal connected z-segments
                within the corresponding external path.
        """
        m, t = self.shape
        zsurvey_ext = self._zsurvey.calculate_external_paths_stretch()
        convert = lambda a: UWKJ(u=a.u, w=a.w, k=a.k, j=a.j)
        zsurvey_ext = {convert(uwkj): dict_uwkj for uwkj, dict_uwkj in zsurvey_ext.items()}

        u_vals: set[int] = range(2)  # As in zephyr coordinates
        w_vals: list[int] = range(2 * m + 1)  # As in zephyr coordinates
        k_vals: list[int] = range(t)  # As in zephyr coordinates
        j_vals: list[int] = range(2)  # As in zephyr coordinates
        uwj_vals: list[UWJ] = [UWJ(u=u, w=w, j=j) for (u, w, j) in product(u_vals, w_vals, j_vals)]
        return {
            uwj: {k_idx: zsurvey_ext[UWKJ(u=uwj.u, w=uwj.w, k=k_idx, j=uwj.j)] for k_idx in k_vals}
            for uwj in uwj_vals
        }
