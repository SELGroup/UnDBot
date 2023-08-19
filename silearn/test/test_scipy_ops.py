import os
import unittest

import numpy as np
import torch
from numpy.testing import assert_equal
from torch_geometric.seed import seed_everything

from silearn.backends.scipy_ops import *

# Sets the seed for generating random numbers in PyTorch, numpy and Python.
seed_everything(42)

class Testbackends(unittest.TestCase):
    def setUp(self):
        # self.adj =

        self.src = np.array([[1, 2], [5, 6], [3, 4], [7, 8], [9, 10], [11, 12]])
        self.idx = np.array([0, 1, 0, 1, 1, 3])
        self.cnt = np.array([2,3,0,1])
        self.sum = np.array([[4, 6], [21, 24], [0, 0], [11, 12]])
        self.max = np.array([[3, 4], [9, 10], [0, 0], [11, 12]])

        self.p = np.array([1, 0, 3, 1])
        self.q = np.array([1, 1/2, 1/4, 0])

    def test_nonzero_idx_dense(self):
        pass

    def test_vertex_reduce(self):
        pass

    def test_scatter_sum(self):
        assert_equal(scatter_sum(self.src, self.idx), self.sum)

    def test_scatter_cnt(self):
        assert_equal(scatter_cnt(self.idx), self.cnt)

    def test_entropy(self):
        ground_truth = np.array([0, 0, 6, np.log2(1e36)])
        assert_equal(entropy(self.p, self.q), ground_truth)

    def test_uncertainty(self):
        ground_truth = np.array([0, 1, 2, np.log2(1e36)])
        assert_equal(uncertainty(self.q), ground_truth)


if __name__ == '__main__':
    unittest.main()