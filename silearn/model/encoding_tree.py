import networkx

import silearn
from silearn.graph import Graph
from silearn import *


class GraphEncoding:
    r"""
    The base Graph Encoding model
    """

    def __init__(self, g: Graph):
        self.graph = g

    def uncertainty(self, es, et, p):
        raise NotImplementedError("Not Implemented")

    def positioning_entropy(self):
        dist = self.graph.stationary_dist #dist=di/vol(G)
        return silearn.entropy(dist, dist) #-di/vol(G) log_2 di/vol(G)

    def entropy_rate(self, reduction="vertex", norm=False):
        edges, p = self.graph.edges
        es, et = edges[:, 0], edges[:, 1]
        nw = self.graph.vertex_weight_es[es] #第一列顶点的di E
        entropy = silearn.entropy(p, p / nw) # -p log_2 p/di

        if norm:
            dist = self.graph.stationary_dist[es]
            entropy = entropy / self.positioning_entropy()

        if reduction == "none":
            return entropy
        elif reduction == "vertex":
            return silearn.scatter_sum(entropy, et)  #按第二列顶点求和
        elif reduction == "sum":
            return entropy.sum()
        else:
            return entropy

    def structural_entropy(self, reduction="vertex", norm=False):
        edges, p = self.graph.edges
        es, et = edges[:, 0], edges[:, 1]
        # dist = self.graph.stationary_dist[es]
        dist = self.graph.stationary_dist[es] #di/vol(G)
        # tot = w.sum()
        entropy = p * self.uncertainty(es, et, p)

        if norm:
            entropy = entropy / silearn.entropy(dist, dist)
        if reduction == "none":
            return entropy
        elif reduction == "vertex":
            return silearn.scatter_sum(entropy, et)
        elif reduction == "sum":
            return entropy.sum()
        else:
            return entropy

    def to_networkx(self, create_using=networkx.DiGraph()):
        raise NotImplementedError()


class OneDim(GraphEncoding):

    def uncertainty(self, es, et, p):
        v1 = self.graph.stationary_dist[es]
        return uncertainty(v1)


class Partitioning(GraphEncoding):
    node_id = None  # :torch.LongTensor

    def __init__(self, g: Graph, init_parition):
        super().__init__(g)
        self.node_id = init_parition

    def uncertainty(self, es, et, p):
        v1e = self.graph.stationary_dist[es]  #第一列顶点的di/vol(G) E
        id_et = self.node_id[et] #第二列顶点所在社区 E
        id_es = self.node_id[es] #第一列顶点所在社区 E
        v2 = scatter_sum(self.graph.stationary_dist, self.node_id)  #按社区求di/vol(G),v2=vol(α)/vol(G),v2.sum=1
        v2e = v2[id_es] #第一列顶点所在社区的vol(α)/vol(G)
        flag = id_es != id_et #一条边连接两顶点是否在同一社区
        # print(v1e, v2, flag)
        return uncertainty(v1e / v2e) + flag * uncertainty(v2e / v2.sum())
        # 第一列顶点的 log_2 di/vol(α)  +两顶点不在一个社区 log_2 vol(α)/vol(G)

    def structural_entropy(self, reduction="vertex", norm=False):
        entropy = super(Partitioning, self).structural_entropy(reduction, norm)
        if reduction == "module":
            et = self.graph.edges[0][:, 1]
            return scatter_sum(entropy, self.node_id[et])
        return entropy

    def compound(self, hyper_partitioning):
        self.node_id = hyper_partitioning[self.node_id]

    # def to_graph(self):
    #     import numpy as np
    #     import torch
    #
    #     a = np.array([[0, 1.2, 0], [2, 3.1, 0], [0.5, 0, 0]])
    #     idx = a.nonzero()  # (row, col)
    #     data = a[idx]
    #
    #     # to torch tensor
    #     idx_t = torch.LongTensor(np.vstack(idx))
    #     data_t = torch.FloatTensor(data)
    #     coo_a = torch.sparse_coo_tensor(idx_t, data_t, a.shape)

    def to_networkx(self,
                    create_using=networkx.DiGraph(),
                    label_name="partition"):
        nx_graph = self.graph.to_networkx(create_using=create_using)
        label_np = silearn.convert_backend(self.node_id, "numpy")
        for i in range(label_np.shape[0]):
            nx_graph._node[i][label_name] = label_np[i]
        return nx_graph


class EncodingTree(GraphEncoding):
    parent_id: []

    def uncertainty(self, es, et, p):
        v1 = self.graph.stationary_dist[et]
        cur_ids = es
        cur_idt = et
        ret = 0
        for i in range(len(self.parent_id)):
            id_es = self.parent_id[i][cur_ids]
            id_et = self.parent_id[i][cur_idt]
            vp = scatter_sum(
                v1,
                id_et)[id_et] if i != len(self.parent_id) - 1 else v1.sum()
            if i == 0:
                ret += uncertainty(v1 / vp)
            else:
                flag = cur_ids != cur_idt
                ret += flag * uncertainty(v1 / vp)
            v1 = vp
            cur_ids, cur_idt = id_es, id_et
        return ret

    def structural_entropy(self, reduction="vertex", norm=False):
        entropy = super(EncodingTree, self).structural_entropy(reduction, norm)
        if reduction.startswith("level"):
            level = int(reduction[5:])
            level = min(-len(self.parent_id), level)
            level = max(len(self.parent_id) - 1, level)
            et = self.graph.edges[2]
            return scatter_sum(entropy, self.parent_id[level][et])
        return entropy

    """
    2-Dim Enc Tree: Level - -1, 0
    3-Dim Enc Tree: Level - -2, -1, 0, 1
    """

    def as_partition(self, level=-1):
        height = len(self.parent_id)
        assert -height <= level < height
        if level < 0:
            level = height + level
        if level != 0:
            trans = self.parent_id[level]
            for i in reversed(range(level)):
                trans = trans[self.parent_id[i]]
            return trans
        else:
            return self.parent_id
