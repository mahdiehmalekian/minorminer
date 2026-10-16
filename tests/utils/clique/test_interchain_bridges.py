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


import unittest

from dwave.graphs import zephyr_coordinates, zephyr_graph
from parameterized import parameterized

from minorminer.utils._clique.grid import Grid
from minorminer.utils._clique.interchain_bridges import find_interchain_bridges, neighbors

# faulty Z(3, 2): two missing nodes and one missing internal coupler
_M, _T = 3, 2
_MISSING_NODES = [(5, 2, 0), (6, 5, 1)]
_MISSING_EDGES = [((8, 3, 0), (9, 4, 1))]


def _faulty_graph_and_grid():
    graph = zephyr_graph(_M, _T, coordinates=True)
    to_zephyr = zephyr_coordinates.cartesian_to_zephyr
    graph.remove_nodes_from([to_zephyr(n) for n in _MISSING_NODES])
    graph.remove_edges_from([(to_zephyr(u), to_zephyr(v)) for u, v in _MISSING_EDGES])
    return graph, Grid(graph)


class TestNeighbors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph, cls.grid = _faulty_graph_and_grid()

    def test_interior_node(self):
        # dy = -4 would leave the grid (y = -1); the coupler to (9, 4, 1) is missing
        expected = [
            (7, 2, 0),
            (7, 2, 1),
            (7, 4, 0),
            (7, 4, 1),
            (9, 2, 0),
            (9, 2, 1),
            (9, 4, 0),
            (8, 1, 0),
            (8, 5, 0),
            (8, 7, 0),
        ]
        self.assertCountEqual(neighbors(self.grid, (8, 3, 0)), expected)

    def test_boundary_node(self):
        # x = 12 is the right edge: no dx = +1, and dy = +4 would give y = 13
        expected = [
            (11, 8, 0),
            (11, 8, 1),
            (11, 10, 0),
            (11, 10, 1),
            (12, 7, 1),
            (12, 11, 1),
            (12, 5, 1),
        ]
        self.assertCountEqual(neighbors(self.grid, (12, 9, 1)), expected)

    def test_missing_coupler(self):
        # 11 neighbors in the ideal grid; the missing coupler to (8, 3, 0) leaves 10
        # dx = +4 would give x = 13, off the grid
        expected = [
            (8, 3, 1),
            (8, 5, 0),
            (8, 5, 1),
            (10, 3, 0),
            (10, 3, 1),
            (10, 5, 0),
            (10, 5, 1),
            (7, 4, 1),
            (11, 4, 1),
            (5, 4, 1),
        ]
        result = neighbors(self.grid, (9, 4, 1))
        self.assertCountEqual(result, expected)
        self.assertNotIn((8, 3, 0), result)
        self.assertNotIn((9, 4, 1), result)

    @parameterized.expand([(node,) for node in _MISSING_NODES])
    def test_absent_node(self, node):
        self.assertEqual(neighbors(self.grid, node), [])

    def test_matches_graph(self):
        to_cartesian = zephyr_coordinates.zephyr_to_cartesian
        for zcoord in self.graph:
            ccoord = to_cartesian(zcoord)
            expected = [to_cartesian(n) for n in self.graph[zcoord]]
            with self.subTest(node=ccoord):
                self.assertCountEqual(neighbors(self.grid, ccoord), expected)


# bridges from chain A to _CHAIN_B
_CHAIN_B = [(10, 7, 1), (9, 6, 0)]
_DIRECT_CHAIN_A = [(3, 6, 0), (5, 6, 0)]  # (5, 6, 0) -- (9, 6, 0) is an external coupler
_CHAIN_A = [(3, 6, 0)]
_LONG_CHAIN_A = [(1, 6, 0), (3, 6, 0)]  # (1, 6, 0) is adjacent to (5, 6, 0) but no bridge node

# _CHAIN_A, nothing used
_ONE_NODE_BRIDGES = [((5, 6, 0),), ((7, 6, 0),)]
_TWO_NODE_BRIDGES = [
    ((1, 6, 0), (5, 6, 0)),
    ((4, 5, 0), (5, 6, 0)),
    ((4, 5, 1), (5, 6, 0)),
    ((4, 7, 0), (5, 6, 0)),
    ((4, 7, 1), (5, 6, 0)),
    ((5, 6, 0), (7, 6, 0)),
    ((7, 6, 0), (5, 6, 0)),
    ((7, 6, 0), (8, 5, 0)),
    ((7, 6, 0), (8, 5, 1)),
    ((7, 6, 0), (8, 7, 0)),
    ((7, 6, 0), (8, 7, 1)),
    ((7, 6, 0), (11, 6, 0)),
]
# _CHAIN_A, (5, 6, 0) used
_USED_TWO_NODE_BRIDGES = [
    ((7, 6, 0), (8, 5, 0)),
    ((7, 6, 0), (8, 5, 1)),
    ((7, 6, 0), (8, 7, 0)),
    ((7, 6, 0), (8, 7, 1)),
    ((7, 6, 0), (11, 6, 0)),
]
# _LONG_CHAIN_A, nothing used
_LONG_CHAIN_TWO_NODE_BRIDGES = [
    ((4, 5, 0), (5, 6, 0)),
    ((4, 5, 1), (5, 6, 0)),
    ((4, 7, 0), (5, 6, 0)),
    ((4, 7, 1), (5, 6, 0)),
    ((5, 6, 0), (7, 6, 0)),
    ((7, 6, 0), (5, 6, 0)),
    ((7, 6, 0), (8, 5, 0)),
    ((7, 6, 0), (8, 5, 1)),
    ((7, 6, 0), (8, 7, 0)),
    ((7, 6, 0), (8, 7, 1)),
    ((7, 6, 0), (11, 6, 0)),
]
_USED = {(5, 6, 0)}
_ALL_BRIDGES_USED = {(5, 6, 0), (7, 6, 0)}  # every route from A to B goes through these

# (name, chain_a, used_nodes, max_interchain_bridge_nodes, expected)
_BRIDGE_CASES = [
    ("directly_connected_cap_0", _DIRECT_CHAIN_A, set(), 0, [()]),
    ("directly_connected_cap_1", _DIRECT_CHAIN_A, set(), 1, [()]),
    ("directly_connected_cap_2", _DIRECT_CHAIN_A, set(), 2, [()]),
    ("nothing_used_cap_0", _CHAIN_A, set(), 0, []),
    ("nothing_used_cap_1", _CHAIN_A, set(), 1, _ONE_NODE_BRIDGES),
    ("nothing_used_cap_2", _CHAIN_A, set(), 2, _ONE_NODE_BRIDGES + _TWO_NODE_BRIDGES),
    ("nothing_used_cap_3", _CHAIN_A, set(), 3, _ONE_NODE_BRIDGES + _TWO_NODE_BRIDGES),
    ("used_node_cap_0", _CHAIN_A, _USED, 0, []),
    ("used_node_cap_1", _CHAIN_A, _USED, 1, [((7, 6, 0),)]),
    ("used_node_cap_2", _CHAIN_A, _USED, 2, [((7, 6, 0),)] + _USED_TWO_NODE_BRIDGES),
    ("chain_nodes_cap_0", _LONG_CHAIN_A, set(), 0, []),
    ("chain_nodes_cap_1", _LONG_CHAIN_A, set(), 1, _ONE_NODE_BRIDGES),
    (
        "chain_nodes_cap_2",
        _LONG_CHAIN_A,
        set(),
        2,
        _ONE_NODE_BRIDGES + _LONG_CHAIN_TWO_NODE_BRIDGES,
    ),
    ("all_bridges_used_cap_0", _CHAIN_A, _ALL_BRIDGES_USED, 0, []),
    ("all_bridges_used_cap_1", _CHAIN_A, _ALL_BRIDGES_USED, 1, []),
    ("all_bridges_used_cap_2", _CHAIN_A, _ALL_BRIDGES_USED, 2, []),
]


class TestFindInterchainBridges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.grid = _faulty_graph_and_grid()

    @parameterized.expand(_BRIDGE_CASES)
    def test_bridges(self, _name, chain_a, used_nodes, cap, expected):
        result = find_interchain_bridges(self.grid, chain_a, _CHAIN_B, used_nodes, cap)
        self.assertCountEqual(result, expected)
        lengths = [len(bridge) for bridge in result]
        self.assertEqual(lengths, sorted(lengths))  # fewest bridge nodes first

    def test_default_cap(self):
        result = find_interchain_bridges(self.grid, _CHAIN_A, _CHAIN_B, set())
        self.assertCountEqual(result, _ONE_NODE_BRIDGES + _TWO_NODE_BRIDGES)
