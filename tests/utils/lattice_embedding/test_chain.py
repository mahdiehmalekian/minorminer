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
from minorminer.utils.zephyr.node_edge import ZNode, ZShape
from minorminer.utils.zephyr.plane_shift import PlaneShift


class TestZephyrVHChain(TestCase):
    def test_runs(self):
        x, y = ZNode((0, 1)), ZNode((1, 0))
        for k in range(4):
            ZephyrVHChain(x + k * PlaneShift(1, 1), y + k * PlaneShift(1, 1))

        q0 = ZNode((6, 9), shape=ZShape(m=4))
        ZephyrVHChain(q0, q0 + PlaneShift(1, -1))
        ZephyrVHChain(q0 + PlaneShift(-1, -1), q0 + PlaneShift(0, -2))
        ZephyrVHChain(q0 + PlaneShift(2, 0), q0 + PlaneShift(3, -1))

    def test_invlaid_input_raises_error(self):
        x, y = ZNode((0, 1)), ZNode((0, 3))
        for k in range(4):
            with self.assertRaises(ValueError):
                ZephyrVHChain(x + k * PlaneShift(1, 1), y + k * PlaneShift(1, 1))

        q0 = ZNode((2, 13), shape=ZShape(m=4))
        with self.assertRaises(ValueError):
            ZephyrVHChain(q0, q0 + PlaneShift(3, -1))
            ZephyrVHChain(q0, q0 + PlaneShift(2, 0))
            ZephyrVHChain(q0, q0 + PlaneShift(2, -2))
            ZephyrVHChain(q0)

    def test_is_coupled(self):
        x1, y1 = ZNode((0, 1)), ZNode((1, 0))
        x2, y2 = ZNode((1, 2)), ZNode((2, 1))

        c1 = ZephyrVHChain(x1, y1)
        c2 = ZephyrVHChain(x2, y2)
        for coupling_kind in [{"01"}, {"10"}, {"10", "01"}, None, {}]:
            self.assertTrue(c1.is_coupled(c2, coupling_kind=coupling_kind))
            self.assertTrue(c2.is_coupled(c1, coupling_kind=coupling_kind))
        q0 = ZNode((6, 9), shape=ZShape(m=4))
        zvh0 = ZephyrVHChain(q0, q0 + PlaneShift(1, -1))
        zvh1 = ZephyrVHChain(q0 + PlaneShift(-1, -1), q0 + PlaneShift(0, -2))
        zvh2 = ZephyrVHChain(q0 + PlaneShift(2, 0), q0 + PlaneShift(3, -1))
        for zvh in [zvh0, zvh1, zvh2]:
            self.assertTrue(zvh.is_coupled(zvh, coupling_kind=["01", "10"]))
