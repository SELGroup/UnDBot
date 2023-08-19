from abc import abstractmethod

import scipy
import torch
import torch_scatter

import silearn

import networkx



# Graph Model for Random Walk Process
#
class Graph(object):
    # adj : torch.Tensor
    backend = "torch"
    directed = True

    def __init__(self):
        pass


    @property
    def device(self):
        return "cpu"

    @property
    def num_vertices(self):
        return 0

    @property
    def num_edges(self):
        return 0

    # def vertex_reduce(self,  partition):
    #     silearn.vertex_reduce(self, partition)

    @abstractmethod
    def to_networkx(self, create_using = networkx.DiGraph()):
        raise NotImplementedError("Not Implemented")

    @property
    @abstractmethod
    def stationary_dist(self):
        raise NotImplementedError("Not Implemented")

    @property
    @abstractmethod
    def vertex_weight_es(self):
        raise NotImplementedError("Not Implemented")

    @property
    @abstractmethod
    def edges(self):
        raise NotImplementedError("Not Implemented")
    @abstractmethod
    def query_probability(self, es, et):
        raise NotImplementedError("Not Implemented")

    @abstractmethod
    def clone(self):
        raise NotImplementedError("Not Implemented")




class GraphSparse(Graph):

    """
    E x 2
    """
    _edges: None
    # E
    _p: None
    _dist = None
    n_vertices = 0

    tot_weights = 1

    def __init__(self, edges, p, dist = None, n_vertices = None):
        super().__init__()
        if n_vertices is None and dist is not None:
            self.n_vertices = dist.shape[-1]  #dist=di/vol(G)，当前self.n_vertices为实际节点数
        else:
            self.n_vertices = n_vertices
        self._edges, self._p, self._dist = edges, p, dist


    @property
    def device(self):
        return self._edges.device

    @property
    def num_vertices(self):
        return self.n_vertices #节点数

    @property
    def num_edges(self):
        return self._edges.shape[0]

    @property
    def vertex_weight_es(self):
        return silearn.scatter_sum(self._p, self.edges[0][:, 0]) #di


    @property
    def edges(self):
        return self._edges, self._p

    @property
    def stationary_dist(self):
        return self._dist  #dist=di/vol(G)

    def query_probability(self, es, et):
        edge_index = torch.tensor([es, et])
        if torch.where((self._edges==edge_index).all(dim=1))[0].shape[0] == 0:
            raise ValueError('Edge not found')
        return self._p[torch.where((self._edges==edge_index).all(dim=1))[0].shape[0]]

    def to_networkx(self, create_using = networkx.DiGraph()):
        edges = silearn.convert_backend(self._edges, "numpy")
        weights = silearn.convert_backend(self._p, "numpy")
        scipy.sparse.coo.coo_matrix((weights, (edges[:, 0], edges[:, 1])), (self.n_vertices, self.n_vertices))
        networkx.from_scipy_sparse_array(edges, create_using=create_using)

    # def query_weight(self, es, et):
    #
    def clone(self):
        return GraphSparse(silearn.clone(self._edges), silearn.clone(self._p),
                           silearn.clone(self._dist), self.n_vertices)





class GraphDense(Graph):
    adj: None
    dist: None


    def num_vertices(self):
        return self.adj.shape[0]

    def num_edges(self):
        return self.adj.shape[0] ** 2

    def stationary_dist(self):
        return self.dist

    def to_sparse(self):
        raise NotImplementedError("Not Implemented")



    def edges(self):
        edges = silearn.nonzero(self.adj)
        return edges, self.adj[edges[:, 0]][edges[:, 1]]

    def query_probability(self, es, et):
        return self.adj[es][et]



