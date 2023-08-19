import unittest

import torch
from torch_geometric.seed import seed_everything

from silearn.backends.torch_ops import *

# Sets the seed for generating random numbers in PyTorch, numpy and Python.
seed_everything(42)


class Testbackends(unittest.TestCase):

    def setUp(self):
        # self.adj =
        self.edges = torch.tensor([[0, 1], [2, 3], [0, 3], [1, 3], [0, 2],
                                   [2, 3]])
        self.weights = [
            torch.tensor([1, 1, 1, 1, 1, 1]),
            torch.tensor([2, 2, 2, 2, 1, 1])
        ]
        self.operation_ptrs = torch.tensor([0, 1, 3, 5])

        self.src = torch.tensor([[1, 2], [5, 6], [3, 4], [7, 8], [9, 10],
                                 [11, 12]])
        self.idx = torch.LongTensor([0, 1, 0, 1, 1, 3])
        self.cnt = torch.tensor([2, 3, 0, 1])
        self.sum = torch.tensor([[4, 6], [21, 24], [0, 0], [11, 12]])
        self.max = torch.tensor([[3, 4], [9, 10], [0, 0], [11, 12]])

        self.p = torch.tensor([1, 0, 3, 1])
        self.q = torch.tensor([1, 1 / 2, 1 / 4, 0])

    def test_nonzero(self):
        pass

    def test_scatter_sum(self):
        assert torch.all(scatter_sum(self.src, self.idx) == self.sum)

    def test_scatter_cnt(self):
        assert torch.all(scatter_cnt(self.idx) == self.cnt)

    def test_entropy(self):
        ground_truth = torch.tensor([0, 0, 6, torch.log2(torch.tensor(1e36))])
        assert torch.all(entropy(self.p, self.q) == ground_truth)

    def test_uncertainty(self):
        ground_truth = torch.tensor([0, 1, 2, torch.log2(torch.tensor(1e36))])
        assert torch.all(uncertainty(self.q) == ground_truth)

    def test_concat(self):
        x = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        ground_truth1 = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4],
                                      [5, 6, 7, 8, 5, 6, 7, 8]])
        ground_truth2 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4],
                                      [5, 6, 7, 8]])
        assert torch.all(concat([x, x]) == ground_truth1)
        assert torch.all(concat([x, x], 0) == ground_truth2)

    def test_edgeredu_get_edge_transform(self):
        ground_truth = torch.tensor([0, 4, 2, 3, 1, 4])
        assert torch.all(
            EdgeRedu._get_edge_transform(self.edges) == ground_truth)

    def test_edgeredu_reduction_edge(self):
        ground_truth = (torch.tensor([[0, 1], [0, 2], [0, 3], [1, 3],
                                      [2, 3]]), torch.tensor([1, 1, 1, 1, 2]),
                        torch.tensor([2, 1, 2, 2, 3]))
        edge_transform = EdgeRedu._get_edge_transform(self.edges)
        edge, weight1, weight2 = EdgeRedu._reduction_edge(
            self.edges, edge_transform, *self.weights)
        assert torch.all(edge == ground_truth[0])
        assert torch.all(weight1 == ground_truth[1])
        assert torch.all(weight2 == ground_truth[2])

    def test_sumup_duplicates(self):
        # todo： operation_ptrs condition
        ground_truth = (torch.tensor([[0, 1], [0, 2], [0, 3], [1, 3],
                                      [2, 3]]), torch.tensor([1, 1, 1, 1, 2]),
                        torch.tensor([2, 1, 2, 2, 3]))
        edge, weight1, weight2 = sumup_duplicates(self.edges, *self.weights)
        assert torch.all(edge == ground_truth[0])
        assert torch.all(weight1 == ground_truth[1])
        assert torch.all(weight2 == ground_truth[2])


if __name__ == '__main__':
    # unittest.main()
    edges = torch.tensor([[0, 1], [2, 3], [0, 3], [1, 3], [0, 2], [2, 3]])
    weights = [
        torch.tensor([1, 1, 1, 1, 1, 1]),
        torch.tensor([2, 2, 2, 2, 1, 1])
    ]
    operation_ptrs = torch.tensor([1, 2, 3, 5])
    print(sumup_duplicates(edges, *weights))