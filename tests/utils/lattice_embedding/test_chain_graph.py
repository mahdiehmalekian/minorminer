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


from unittest import TestCase

from minorminer.utils.lattice_embedding.chain import ZephyrVHChain
from minorminer.utils.lattice_embedding.chain_graph import ZVHChainGraph
from minorminer.utils.zephyr.node_edge import ZNode
from minorminer.utils.zephyr.plane_shift import PlaneShift


class TestZVHChainGraph(TestCase):
    def setUp(self) -> None:
        x, y = ZNode((0, 1)), ZNode((1, 0))
        self.ladder_chains = {
            ZephyrVHChain(x + k * PlaneShift(1, 1), y + k * PlaneShift(1, 1)) for k in range(4)
        }
        self.type0_chains = {
            ZephyrVHChain(x + PlaneShift(2 * i, 2 * j), y + PlaneShift(2 * i, 2 * j))
            for i in (0, 1)
            for j in (0, 1)
        }
        self.ladder_chains_with_overlap = self.ladder_chains | {
            ZephyrVHChain(ZNode((0, 1)), ZNode((1, 2)))
        }
        self.disc_chains = {
            ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0))),
            ZephyrVHChain(ZNode((2, 3)), ZNode((3, 2))),
        }
        self.symmetric_kinds = [None, {"01", "10"}, {}]
        self.asymmetric_kinds = [{"01"}, {"10"}]

    def test_runs(self):
        for kind in self.symmetric_kinds + self.asymmetric_kinds:
            ZVHChainGraph(nodes=self.ladder_chains, coupling_kind=kind)
            ZVHChainGraph(nodes=self.type0_chains, coupling_kind=kind)
            ZVHChainGraph(nodes=self.ladder_chains_with_overlap, coupling_kind=kind)
            ZVHChainGraph(nodes=self.disc_chains, coupling_kind=kind)

    def test_edges(self):
        for kind in self.symmetric_kinds:
            ladder_G = ZVHChainGraph(nodes=self.ladder_chains, coupling_kind=kind)
            type0_G = ZVHChainGraph(nodes=self.type0_chains, coupling_kind=kind)
            ladder_G_overlap = ZVHChainGraph(
                nodes=self.ladder_chains_with_overlap, coupling_kind=kind
            )
            disc_G = ZVHChainGraph(nodes=self.disc_chains, coupling_kind=kind)
            ladder_num_edges = len(ladder_G.edges)
            type0_num_edges = len(type0_G.edges)
            ladder_overlap_num_edges = len(ladder_G_overlap.edges)
            disc_num_edges = len(disc_G.edges)
            self.assertEqual(disc_num_edges, 0)
            if kind is None:
                self.assertEqual(ladder_num_edges, 3)
                self.assertEqual(type0_num_edges, 5)
                self.assertEqual(ladder_overlap_num_edges, 6)
            elif set(kind) == {"01", "10"}:
                self.assertEqual(ladder_num_edges, 3)
                self.assertEqual(type0_num_edges, 0)
                self.assertEqual(ladder_overlap_num_edges, 5)

    def test_arcs(self):
        for kind in self.asymmetric_kinds:
            ladder_G = ZVHChainGraph(nodes=self.ladder_chains, coupling_kind=kind)
            type0_G = ZVHChainGraph(nodes=self.type0_chains, coupling_kind=kind)
            disc_G = ZVHChainGraph(nodes=self.disc_chains, coupling_kind=kind)
            disc_num_arcs = len(disc_G.arcs)
            self.assertEqual(disc_num_arcs, 0)
            ladder_num_arcs = len(ladder_G.arcs)
            type0_num_arcs = len(type0_G.arcs)
            self.assertEqual(ladder_num_arcs, 6)
            self.assertEqual(type0_num_arcs, 5)

    def test_arcs_or_edges(self):
        for kind in self.symmetric_kinds + self.asymmetric_kinds:
            ladder_G = ZVHChainGraph(nodes=self.ladder_chains, coupling_kind=kind)
            ladder_G.edges_or_arcs
            type0_G = ZVHChainGraph(nodes=self.type0_chains, coupling_kind=kind)
            type0_G.edges_or_arcs
            if kind in self.symmetric_kinds:
                self.assertEqual(ladder_G.edges_or_arcs, ladder_G.edges)
                self.assertEqual(type0_G.edges_or_arcs, type0_G.edges)
            else:
                self.assertEqual(ladder_G.edges_or_arcs, ladder_G.arcs)
                self.assertEqual(type0_G.edges_or_arcs, type0_G.arcs)

    def test_connected(self):
        kind = {"01"}
        ladder_G = ZVHChainGraph(nodes=self.ladder_chains, coupling_kind=kind)
        self.assertTrue(ladder_G.is_connected())
        type0_G = ZVHChainGraph(nodes=self.type0_chains, coupling_kind=kind)
        self.assertTrue(type0_G.is_connected())
        vhc0 = ZephyrVHChain(ZNode((0, 1)), ZNode((1, 0)))
        vhc1 = ZephyrVHChain(ZNode((2, 3)), ZNode((3, 2)))
        disc_G = ZVHChainGraph(nodes=[vhc0, vhc1])
        self.assertFalse(disc_G.is_connected())
