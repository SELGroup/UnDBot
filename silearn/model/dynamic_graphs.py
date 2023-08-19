import silearn
from silearn.graph import Graph, GraphSparse, GraphDense
from encoding_tree import Partitioning
from .dynamic_nodes import UpdateMethod, DynamicNode, UpdateType


class IncreSparseGraph(GraphSparse, DynamicNode):
    r"""
    [Undirected] graph model with helper functions of modifying edge weights and edge sets.
    Weighted denotes the relative probability of random walking.
    """
    tot_weights = 0
    dup_edges = False

    def __init__(self, edges, weights, dist=None, n_vertices=None):
        super().__init__(edges, weights, dist, n_vertices)
        self.original_tot_weight = weights.sum()
        self.directed = False

    @staticmethod
    def as_incremental(g: GraphSparse):
        return IncreSparseGraph(g._edges,
                                g._p,
                                g._dist,
                                n_vertices=g.n_vertices)

    @UpdateMethod(UpdateType.incre_edge_insertion)
    def insert_edges(self, edges, weights):
        r"""
        edges[i][0], edges[i][1]: the start / target node-id of the i-th inserted edge
        """
        self.tot_weights += weights.sum()
        self._edges = silearn.concat((self._edges, edges), dim=0)
        self._p = silearn.concat(self._p, weights)

    @property
    def stationary_dist(self):
        return self._dist / self.tot_weights

    def refresh_stat_dist(self):
        self._dist = silearn.scatter_sum(self._p, self.edges[0])

    def clean_dup_edges(self):
        if not self.dup_edges:
            return False
        self.dup_edges = False
        self._edges, self._p = silearn.sumup_duplicates(self._edges, self._p)

    def clone(self):
        return IncreSparseGraph(silearn.clone(self._edges),
                                silearn.clone(self._p),
                                n_vertices=self.n_vertices)

    def combine_graph(self, g: GraphSparse):
        e, p = g.edges
        self.insert_edges(e, p)


def vertex_reduction(self, graph: Graph, encoding: Partitioning):
    r"""
    Vertex reduction maintaining the structural entropy
    """
    if isinstance(graph, GraphSparse):
        edges, p = graph.edges
        dist = graph.stationary_dist
        edges = encoding.node_id[edges]
        dist = silearn.scatter_sum(dist, encoding)
        edges, p = silearn.sumup_duplicates(edges, p)
        return GraphSparse(edges, p, dist, n_vertices=dist.shape[-1])
    else:
        # todo: implement GraphDense
        raise NotImplementedError("Not implemented yet")


# todo: implementation
# class IncrementalSparseDiGraph(IncrementalSparseGraph):
#