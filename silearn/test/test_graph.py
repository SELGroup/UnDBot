import unittest

import torch
from torch_geometric.seed import seed_everything
from silearn.graph import *

import silearn

# Sets the seed for generating random numbers in PyTorch, numpy and Python.
seed_everything(42)

class TestGraphEncoding(unittest.TestCase):
    def setUp(self):
        self.adj = torch.tensor([[0, 0, 1, 0],[0, 0, 4, 3], [1, 4, 0, 2], [0, 3, 2, 0]])
        self.adj = self.adj / self.adj.sum()
        self.edges = torch.nonzero(self.adj)
        self.p = self.adj[self.adj!=0]
        self.dist = torch.mm(torch.tensor([[1.0,1.0,1.0,1.0]]),torch.inverse((self.adj / self.adj.sum(dim=1, keepdim=True)) - torch.eye(4) + torch.ones((4,4)))).reshape(-1)
        self.graph = GraphSparse(self.edges, self.p, dist=self.dist)

    def test_parameters(self):
        # print(self.graph.)
        assert (hasattr(self.graph, 'device') and self.graph.device is not None)
        assert (hasattr(self.graph, 'backend') and self.graph.backend is not None)
        assert (hasattr(self.graph, 'directed') and self.graph.directed is not None)
        assert (hasattr(self.graph, 'num_vertices') and self.graph.num_vertices is not None)
        assert (hasattr(self.graph, 'num_edges') and self.graph.num_edges is not None)
        assert (hasattr(self.graph, 'stationary_dist') and self.graph.stationary_dist is not None)
        assert (hasattr(self.graph, 'edges') and self.graph.edges is not None)
        assert (hasattr(self.graph, 'vertex_weight_es') and self.graph.vertex_weight_es is not None)

    def test_device(self):
        assert self.graph.device == torch.device('cpu')

    def test_num_vertices(self):
        assert self.graph.num_vertices == 4

    def test_num_edges(self):
        assert self.graph.num_edges == 8

    def test_vertex_weight_es(self):
        assert torch.all(torch.isclose(self.graph.vertex_weight_es, torch.tensor([0.05, 0.35, 0.35, 0.25])))

    def test_edges(self):
        edge_index, p = self.graph.edges
        assert torch.all(edge_index == self.edges) and torch.all(p == self.p)

    def test_stationary_dist(self):
        assert torch.all(self.graph.stationary_dist==self.dist)

    def test_query_probability(self):
        assert self.graph.query_probability(1, 2) == 0.2



if __name__ == '__main__':
    unittest.main()