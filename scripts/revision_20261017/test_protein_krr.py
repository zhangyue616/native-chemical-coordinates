"""Finite synthetic checks of the fixed protein KRR contract."""
import unittest

import numpy as np

import protein_krr as k


class KernelContractTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(9917)
        self.data = {"dx": rng.normal(size=(9, 5)), "pa": rng.integers(0, 3, size=(9, 4)),
                     "pb": rng.integers(0, 3, size=(9, 4)), "dj": rng.normal(size=9)}

    def test_complete_second_order_explicit_features(self):
        categories = self.data["pa"]
        explicit = np.asarray([[float(row[a] == x and row[b] == y)
                                for a in range(4) for b in range(a + 1, 4)
                                for x in range(3) for y in range(3)] for row in categories]) / np.sqrt(6)
        np.testing.assert_allclose(k.second_order_site(categories, categories), explicit @ explicit.T, atol=1e-14)

    def test_blocks_PSD_and_nested_weights(self):
        blocks = k.raw_blocks(self.data)
        scales = k.block_scales(blocks, np.arange(7))
        for block in blocks.values():
            self.assertGreaterEqual(np.linalg.eigvalsh(block).min(), -1e-10)
        kernels = {m: k.sum_kernel(blocks, scales, m) for m in k.MODELS}
        np.testing.assert_allclose(kernels["S"] - kernels["X"], blocks["P"] / scales["P"], atol=1e-14)
        np.testing.assert_allclose(kernels["L"] - kernels["S"], blocks["Q"] / scales["Q"], atol=1e-14)
        for base, extended in (("X", "Xj"), ("S", "Sj"), ("L", "Lj")):
            np.testing.assert_allclose(kernels[extended] - kernels[base], blocks["j"] / scales["j"], atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(kernels[extended] - kernels[base]).min(), -1e-10)

    def test_pair_swap_and_probability_complement(self):
        swapped = {"dx": -self.data["dx"], "pa": self.data["pb"], "pb": self.data["pa"], "dj": -self.data["dj"]}
        blocks, swapped_blocks = k.raw_blocks(self.data), k.raw_blocks(swapped, self.data)
        for name in k.BLOCKS:
            np.testing.assert_allclose(swapped_blocks[name], -blocks[name], atol=1e-14)
        scores = np.asarray([-20, -1, -0.1, 0, 0.5, 1, 20])
        np.testing.assert_allclose(k.probability(-scores), 1 - k.probability(scores), atol=1e-14)

    def test_constant_j_zero_block_and_train_only_trace(self):
        data = dict(self.data, dj=np.zeros(9))
        blocks = k.raw_blocks(data)
        scales = k.block_scales(blocks, np.arange(7))
        self.assertEqual(scales["j"], 0)
        for base, extended in (("X", "Xj"), ("S", "Sj"), ("L", "Lj")):
            np.testing.assert_array_equal(k.sum_kernel(blocks, scales, base), k.sum_kernel(blocks, scales, extended))
        changed = {name: array.copy() for name, array in self.data.items()}
        changed["dx"][7:] *= 100000
        changed["dj"][7:] *= 100000
        original = k.block_scales(k.raw_blocks(self.data), np.arange(7))
        self.assertEqual(original, k.block_scales(k.raw_blocks(changed), np.arange(7)))

    def test_mean_loss_KRR_matches_primal(self):
        x = self.data["dx"]
        target = np.asarray([1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
        penalty = 0.1
        alpha, residual = k.fit_krr(x @ x.T, target, penalty)
        beta = np.linalg.solve(x.T @ x + len(x) * penalty * np.eye(x.shape[1]), x.T @ target)
        np.testing.assert_allclose(x @ x.T @ alpha, x @ beta, atol=1e-13)
        self.assertLess(residual, 1e-12)

    def test_center_exclusion_and_PAD(self):
        seq = "ACDEFGHIKLMNPQRSTVWY" * 4
        q = 20
        changed = seq[:q] + "Y" + seq[q + 1:]
        np.testing.assert_array_equal(k.control_features(seq, q), k.control_features(changed, q))
        np.testing.assert_array_equal(k.position_categories(seq, q), k.position_categories(changed, q))
        pos = k.position_categories(seq, 2)
        self.assertEqual(int((pos == 20).sum()), 13)
        self.assertEqual(pos.shape, (30,))

    def test_group_folds_are_order_invariant_and_no_split(self):
        groups = np.asarray(["a", "b", "c", "d", "e", "a", "f", "g", "h"])
        folds = k.fixed_folds(groups)
        self.assertEqual(folds[0], folds[5])
        order = np.asarray([8, 3, 1, 6, 0, 4, 2, 7, 5])
        np.testing.assert_array_equal(k.fixed_folds(groups[order]), folds[order])
        self.assertEqual(set(folds), {0, 1, 2, 3})

    def test_precision_aggregates_groups_and_does_not_cap_failure(self):
        groups = np.asarray(["a", "a", "b", "c", "d"])
        contributions = np.asarray([[1, 0, 0, 1], [-1, 0, 0, -1], [1, 1, 1, 0],
                                    [-1, -1, -1, 0], [0, 0, 0, 0]], dtype=float)
        plan = k.precision_plan(contributions, groups)
        _, grouped = k.group_average(contributions, groups)
        expected = int(np.ceil((2.4977 * 1.25 * grouped.std(axis=0, ddof=1).max() / 0.01)**2))
        self.assertEqual(plan["N_req"], expected)
        self.assertEqual(plan["group_count"], 4)
        self.assertEqual(plan["status"], "PRECISION_INFEASIBLE_WITHIN_CAP")
        self.assertIsNone(plan["authorized_target_within_cap"])
        small = k.precision_plan(contributions * 0.00001, groups)
        self.assertEqual(small["authorized_target_within_cap"], 512)

    def test_future_family_keeps_four_and_orientation(self):
        groups = np.asarray(["a", "b", "c", "d"])
        values = np.asarray([[0.1, 0.2, 0, -0.1], [0.2, 0.1, 0, 0.1],
                             [0.05, 0.3, 0, -0.25], [0.4, 0.15, 0, 0.25]])
        summary = k.future_primary_summary(values, groups)
        self.assertEqual(summary["family_size"], 4)
        self.assertEqual(summary["endpoints"][2]["p_for_Holm"], 1.0)
        self.assertEqual(summary["endpoints"][3]["direction"], "two-sided")
        values[0, 1] = np.nan
        summary = k.future_primary_summary(values, groups)
        self.assertEqual(summary["family_size"], 4)
        self.assertEqual(summary["endpoints"][1]["p_for_Holm"], 1.0)
        self.assertIsNone(summary["endpoints"][1]["mean"])


if __name__ == "__main__":
    unittest.main()
