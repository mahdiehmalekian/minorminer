# Copyright 2025 D-Wave Systems Inc.
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


import unittest
from itertools import product

from zephyr_utils.node_edge import ZEdge, ZNode, ZShape
from zephyr_utils.plane_shift import PlaneShift
from burnaby.lattice_embedding.qtile import QuoTile


class TestQuoTile(unittest.TestCase):
    def test_quo_tile_runs(self) -> None:
        zn = ZNode((0, 1), ZShape(m=6))
        shifts0 = [(1, -1), (0, 2), (1, 1), (1, 3)]
        zns = [zn + PlaneShift(*x) for x in shifts0]
        QuoTile(zns=zns)

    def test_zns_define(self) -> None:
        seed0 = ZNode((0, 1), ZShape(m=6))
        shifts0 = [PlaneShift(1, 1), PlaneShift(0, 2)]
        zns0 = [seed0 + shift for shift in shifts0]
        seed1 = seed0 + PlaneShift(0, 2)
        shifts1 = [PlaneShift(1, -1), PlaneShift(0, 0)]
        zns1 = [seed1 + shift for shift in shifts1]
        zns2 = [ZNode((1, 2), ZShape(m=6)), ZNode((0, 0, 1, 0), ZShape(m=6))]
        qtile0 = QuoTile(zns=zns0)
        qtile1 = QuoTile(zns=zns1)
        qtile2 = QuoTile(zns=zns2)
        self.assertEqual(qtile0, qtile1)
        self.assertEqual(qtile0, qtile2)
        self.assertEqual(qtile2, qtile1)
        shifts = [PlaneShift(1, 1), PlaneShift(0, 2)]
        shifts_repeat = 2 * shifts
        for zn in [ZNode((0, 1), ZShape(m=6)), ZNode((5, 12))]:
            qtile0 = QuoTile(zns=[zn + shift for shift in shifts])
            qtile1 = QuoTile(zns=[zn + shift for shift in shifts_repeat])
            self.assertEqual(qtile0, qtile1)

    def test_edges(self) -> None:
        shifts = [PlaneShift(1, 1), PlaneShift(0, 2)]
        for zn in [ZNode((0, 1), ZShape(m=6)), ZNode((5, 12))]:
            edges_ = QuoTile([zn + shift for shift in shifts]).edges()
            for xy in edges_:
                self.assertTrue(isinstance(xy, ZEdge))
                self.assertTrue(isinstance(xy[0], ZNode))
                self.assertTrue(isinstance(xy[1], ZNode))
                self.assertTrue(xy[0].is_neighbor(xy[1]))
