import unittest

from parameterized import parameterized

from minorminer.embedders.clique._zephyr.el_geometry import el_template, el_template_length, elbow

_M_CASES = [(m,) for m in (1, 2)]  # small grids


def vertical_blocks(m):
    """All vertical blocks (x even, y odd) with coords in [0, 4m]."""
    abs_max = 4 * m
    return [(x, y) for x in range(0, abs_max + 1, 2) for y in range(1, abs_max + 1, 2)]


def horizontal_blocks(m):
    """All horizontal blocks (x odd, y even) with coords in [0, 4m]."""
    abs_max = 4 * m
    return [(x, y) for x in range(1, abs_max + 1, 2) for y in range(0, abs_max + 1, 2)]


def block_pairs(m):
    """All (vertical block, horizontal block) pairs on a grid of size m."""
    for vx, vy in vertical_blocks(m):
        for hx, hy in horizontal_blocks(m):
            yield vx, vy, hx, hy


def diagonal_adjacent_family():
    """Diagonally-adjacent (``|vx - hx| = 1``, ``|vy - hy| = 1``) anchor cases: all four
    diagonal directions around a fixed interior horizontal block, at two grid
    sizes. The expected template comes from Zephyr topology (both arms collapse:
    ``vp_y = vy``, ``hp_x = hx``).

    Returns ``(m, vx, vy, hx, hy, expected_template)`` where
    ``expected_template = (vy, hx, (vx, vy % 4, vy, vy), (hx % 4, hy, hx, hx))``.
    """
    out = []
    for m, hx, hy in ((1, 1, 2), (12, 15, 10)):
        for vx in (hx - 1, hx + 1):
            for vy in (hy - 1, hy + 1):
                vquo = (vx, vy % 4, vy, vy)
                hquo = (hx % 4, hy, hx, hx)
                out.append((m, vx, vy, hx, hy, (vy, hx, vquo, hquo)))
    return out


# diagonally-adjacent family: (m, vx, vy, hx, hy, expected_template,),
# expected values from Zephyr topology.
KNOWN_ADJACENT = diagonal_adjacent_family()


class TestElbow(unittest.TestCase):
    # (m, vx, vy, hx, hy, expected): expected is the (vp_y, hp_x) elbow, or None
    # when the pair induces no template (crossing off-lattice).
    KNOWN_ELBOWS = [
        (3, 0, 3, 5, 8, (7, 1)),
        (6, 8, 3, 21, 12, (11, 9)),
        (6, 8, 3, 15, 12, (11, 7)),
        (1, 2, 3, 3, 2, (3, 3)),  # degenerate: single-block arms
        (7, 6, 17, 9, 24, (25, 5)),
        (4, 12, 11, 9, 12, (11, 13)),
        (1, 0, 1, 3, 0, None),  # hp_x = -1 off-lattice
        (6, 6, 17, 9, 24, None),  # vp_y = 25 > 4 * 6 off-lattice
        (3, 12, 11, 9, 12, None),  # hp_x = 13 > 4 * 3 off-lattice (+x side)
    ]

    @parameterized.expand(_M_CASES)
    def test_elbow_placement_invariants(self, m):
        """``hp_x`` sits one column off ``vx`` (``hp_x = vx +- 1``, hence odd), and
        ``vp_y``sits one row off ``hy`` (``vp_y = hy +- 1``, hence odd)."""
        for vx, vy, hx, hy in block_pairs(m):
            res = elbow(m, vx, vy, hx, hy)
            if res is None:
                continue
            vp_y, hp_x = res
            self.assertIn(hp_x - vx, (-1, 1))
            self.assertIn(vp_y - hy, (-1, 1))
            self.assertEqual(hp_x % 2, 1)
            self.assertEqual(vp_y % 2, 1)

    @parameterized.expand(KNOWN_ELBOWS)
    def test_known_elbows(self, m, vx, vy, hx, hy, expected):
        self.assertEqual(elbow(m, vx, vy, hx, hy), expected)


class TestElTemplate(unittest.TestCase):
    # (m, vx, vy, hx, hy, expected): full el_template output
    # (vp_y, hp_x, vquo, hquo), or None for a non-template pair.
    KNOWN_TEMPLATES = [
        (3, 0, 3, 5, 8, (7, 1, (0, 3, 3, 7), (1, 8, 1, 5))),
        (6, 8, 3, 21, 12, (11, 9, (8, 3, 3, 11), (1, 12, 9, 21))),
        (6, 8, 3, 15, 12, (11, 7, (8, 3, 3, 11), (3, 12, 7, 15))),
        (1, 2, 3, 3, 2, (3, 3, (2, 3, 3, 3), (3, 2, 3, 3))),
        (12, 2, 3, 3, 2, (3, 3, (2, 3, 3, 3), (3, 2, 3, 3))),
        (7, 6, 17, 9, 24, (25, 5, (6, 1, 17, 25), (1, 24, 5, 9))),
        (4, 12, 11, 9, 12, (11, 13, (12, 3, 11, 11), (1, 12, 9, 13))),
        (1, 0, 1, 3, 0, None),
        (5, 0, 1, 3, 0, None),
        (6, 6, 17, 9, 24, None),
        (3, 12, 11, 9, 12, None),
    ]

    @parameterized.expand(_M_CASES)
    def test_el_template_agrees_with_elbow(self, m):
        """el_template is None exactly when elbow is None, and when present its
        first two elements are exactly the elbow coordinates."""
        for vx, vy, hx, hy in block_pairs(m):
            e = elbow(m, vx, vy, hx, hy)
            t = el_template(m, vx, vy, hx, hy)
            self.assertEqual(e is None, t is None)
            if t is None:
                continue
            vp_y, hp_x, _vquo, _hquo = t
            self.assertEqual((vp_y, hp_x), e)

    @parameterized.expand(_M_CASES)
    def test_at_least_one_template_exists(self, m):
        """Sanity: the sweep is not vacuous -- some pair induces a template."""
        found = any(el_template(m, vx, vy, hx, hy) is not None for vx, vy, hx, hy in block_pairs(m))
        self.assertTrue(found, f"no template found on m={m}")

    @parameterized.expand(KNOWN_TEMPLATES)
    def test_known_templates(self, m, vx, vy, hx, hy, expected):
        self.assertEqual(el_template(m, vx, vy, hx, hy), expected)

    @parameterized.expand(KNOWN_ADJACENT)
    def test_known_adjacent(self, m, vx, vy, hx, hy, expected_template):
        """Diagonally-adjacent pair: full el_template output match the
        closed form (both arms collapsed to a single block)."""
        self.assertEqual(el_template(m, vx, vy, hx, hy), expected_template)


class TestElTemplateLength(unittest.TestCase):
    # (m, vquo, hquo, expected_length)
    KNOWN_ACCEPT_SPANS = [
        (3, (0, 3, 3, 7), (1, 8, 1, 5), 4),
        (6, (8, 3, 3, 11), (1, 12, 9, 21), 7),
        (6, (8, 3, 3, 11), (3, 12, 7, 15), 6),
        (1, (2, 3, 3, 3), (3, 2, 3, 3), 2),  # degenerate single-block arms
        (12, (2, 3, 3, 3), (3, 2, 3, 3), 2),
        (7, (6, 1, 17, 25), (1, 24, 5, 9), 5),
        (4, (12, 3, 11, 11), (1, 12, 9, 13), 3),
    ]

    # (m, vquo, hquo): span pairs el_template_length must reject (-> None), one
    # per rejection path.
    KNOWN_REJECT_SPANS = [
        # off-lattice crossing (elbow returns None, nothing matches)
        (1, (0, 1, 0, 1), (3, 0, -1, 3)),
        (5, (0, 1, 0, 1), (3, 0, -1, 3)),
        # same spans that are a real template at m=7, but off-lattice at m=6
        (6, (6, 1, 17, 25), (1, 24, 5, 9)),
        # same spans that are a real template at m=4, but off-lattice at m=3
        (3, (12, 3, 11, 11), (1, 12, 9, 13)),
        # residues + bounds all pass; only the elbow re-check rejects
        (7, (6, 1, 17, 25), (1, 24, 1, 9)),
        (4, (12, 3, 11, 11), (1, 12, 5, 9)),
        # residue mismatch: shift slot (1) disagrees with the endpoints (== 3)
        (1, (2, 1, 3, 3), (3, 2, 3, 3)),
    ]

    @parameterized.expand(_M_CASES)
    def test_roundtrip_constructor_is_a_real_template(self, m):
        """Every (vquo, hquo) produced by el_template must be accepted by
        el_template_length (positive int), never rejected as a non-template."""
        for vx, vy, hx, hy in block_pairs(m):
            t = el_template(m, vx, vy, hx, hy)
            if t is None:
                continue
            _vp_y, _hp_x, vquo, hquo = t
            length = el_template_length(m, vquo, hquo)
            self.assertIsNotNone(length)
            self.assertIsInstance(length, int)
            self.assertGreaterEqual(length, 2)  # >=1 block per arm

    @parameterized.expand(KNOWN_ACCEPT_SPANS)
    def test_known_accept_spans(self, m, vquo, hquo, expected):
        self.assertEqual(el_template_length(m, vquo, hquo), expected)

    @parameterized.expand(KNOWN_REJECT_SPANS)
    def test_known_reject_spans(self, m, vquo, hquo):
        self.assertIsNone(el_template_length(m, vquo, hquo))
