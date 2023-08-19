from functools import partial

import torch
# import torch_sparse

import silearn
# from silearn.graph import GraphSparse, GraphDense, Graph

#
# def nonzero(g: GraphDense, return_weights=True):
#     r"""
#     Return the edge from GraphDense.adj.
#
#     Args:
#         g(GraphDense): the Dense Graph.
#         return_weights(Bool): return_weights control whether return weights of edges.
#
#     :rtype: :class:`Tensor`, :class:`Tensor`, :class:`Tensor`
#     """
#     es, et = torch.nonzero(g.adj, as_tuple=True)
#     if return_weights:
#         return es, et, g.adj[es][et]
#     else:
#         return es, et

# sumup_duplicates = torch_sparse.coalesce

# This implication is faster than torch_sparse.coalesce


class EdgeRedu:
    r"""
    Edge Reduction function class.
    """

    @staticmethod
    def _reduction_edge(edges, edge_transform, *weights):
        r"""
        Sum up the weights for the same edges.

        Return tuple (edges, weights)

        Args:
            edges(torch.LongTensor): Source edges with shape :obj:`(E,2)`.
            edge_transform(torch.Tensor): Edge id returned from `get_edge_transform`.
            *weights(list): (weight1, weight2, ...), weighti is torch.Tensor with shape :obj:`E`.
        
        :rtype: (:class:`Tensor`, :class:`Tensor`, ...)
        """
        cnt_e = edge_transform.max() + 1
        e1 = torch.zeros(size=(cnt_e, edges.shape[1]),
                         dtype=edges.dtype,
                         device=edges.device)
        edges = e1.scatter(0,
                           edge_transform.reshape(-1, 1).repeat((1, 2)), edges)
        ret = [edges
               ] + [silearn.scatter_sum(w, edge_transform) for w in weights]
        return tuple(ret)

    @staticmethod
    def _get_edge_transform(edges, identical_flag=False):
        r"""
        Assign an ID to each edge, where duplicate edges have the same ID.

        Return the assigned ID Tensor, (identical boolean Tensor)
        Args:
            edges(torch.LongTensor): Source edges with shape :obj:`(E,2)`.
            identical_flag(Bool): Identical_flag control whether returns a :class:`torch.bool` type Tensor indicating repeated edges.

        :rtype: :class:`Tensor` (, :class:`Tensor`)
        """
        max_id = int(edges[:, 1].max() + 1)
        bd = 1
        shift = 0
        while bd <= max_id:
            bd = bd << 1
            shift += 1
        # todo: hash if shift is too big
        edge_hash = (edges[:, 0] << shift) + edges[:, 1]

        if identical_flag:
            _, transform, counts = torch.unique(edge_hash,
                                                return_inverse=True,
                                                return_counts=True)
            flag = counts[transform] != 1
            return transform, flag
        else:
            _, transform = torch.unique(edge_hash, return_inverse=True)
            return transform


def sumup_duplicates(edges, *weights, operation_ptrs=None):
    r"""
    Sum up the weight of duplicate edge.

    Args:
        edges(torch.LongTensor): Edges with shape :obj:`(E,2)`.
        *weights(list): (weight1, weight2, ...), weighti is torch.Tensor with shape :obj:`E`.
    
    :rtype: (:class:`Tensor`, :class:`Tensor`, ...)
    """
    if operation_ptrs is not None:
        em = edges[operation_ptrs]
        wm = [w[operation_ptrs] for w in weights]
        trans = EdgeRedu._get_edge_transform(em)
        redu = EdgeRedu._reduction_edge(em, trans, *wm)
        cnt = redu[0].shape[0]
        operation_ptrs = operation_ptrs[:cnt]
        skip_ptrs = operation_ptrs[cnt:]
        edges[operation_ptrs] = redu[0]
        ret = [edges]
        for i in range(len(weights)):
            weights[i][operation_ptrs] = redu[i + 1]
            weights[i][skip_ptrs] = 0
            ret += weights[i]
        return ret
    else:
        trans = EdgeRedu._get_edge_transform(edges)
        return EdgeRedu._reduction_edge(edges, trans, *weights)


class ImageKNN():
    r"""
    Convert the image into a graph using K-NN method.
    """

    @staticmethod
    def get_sparse_conv(img: torch.Tensor, k=3, r=3, metric=None):
        r"""
        Construct the graph from the image :obj:`img`. If the metric is not specified, choose Euclidean distance as the default.
        
        Return tuple (weight, es, et) representing the weight, start node, and target node for edges.

        Args:
            img(torch.Tensor): The input image with shape :obj:`(H, W, C)`.
            k: The neighbor count of K-NN.
            r: The horizontal and vertical distance bound for linking pair of pixels.
        
        :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
        
        """
        if metric is None:
            metric = lambda f1, f2, s1, s2: (-(f1 - f2)**2).sum(dim=-1) * (
                s1**2 + s2**2)**0.5
        assert r % 2 == 1
        # img = img / 255
        if k >= r * r - 1:
            return ImageKNN._get_balanced_graph_conv(img, metric, r)
        img_pad = torch.nn.functional.pad(
            img, (0, 0, r // 2, r // 2, r // 2, r // 2), value=-1e10)
        imgH, imgW = img.shape[0], img.shape[1]
        neighbor = torch.full((imgH, imgW, r * r),
                              -1e10,
                              dtype=torch.float64,
                              device=img.device)

        for i in range(r):
            for j in range(r):
                if i == r // 2 and j == r // 2:
                    continue
                imgx = metric(img_pad[i:imgH + i, j:imgW + j], img,
                              (i - r // 2), (j - r // 2))
                neighbor[:, :, i * r + j] = imgx

        # if self.verbose: print(neighbor[0, 0])
        srt = torch.sort(neighbor, dim=2)
        pos = srt[1][:, :, -k:].reshape(-1, k)
        neighbor = srt[0][:, :, -k:].reshape(-1)
        del srt
        arr = torch.arange(imgW * imgH, dtype=torch.int64, device=img.device)

        if (r // 2 + 1)**2 <= k:
            hh = arr // imgW
            hh = hh.unsqueeze(1)
            ww = arr % imgW
            ww = ww.unsqueeze(1)
            rr = r // 2
            flag = (hh + pos // r - rr >= 0) & (hh + pos // r - rr < imgH) & \
                   (ww + pos % r - rr >= 0) & (ww + pos % r - rr < imgW) & (pos != r * r // 2)

            flag = flag.reshape(-1)
            pos = arr.unsqueeze(1) + (pos // r - r // 2) * imgW + (pos % r -
                                                                   r // 2)
            return neighbor[flag], pos.reshape(-1)[flag], arr.reshape(
                -1, 1).repeat(1, k).reshape(-1)[flag]
        pos = arr.unsqueeze(1) + (pos // r - r // 2) * imgW + (pos % r -
                                                               r // 2)

        return neighbor, pos.reshape(-1), arr.reshape(-1,
                                                      1).repeat(1,
                                                                k).reshape(-1)

    @staticmethod
    def _get_balanced_graph_conv(img: torch.Tensor, metric, r):
        r"""
        If k >= r * r - 1 , construct graph connecting all pairs of pixels with horizontal and vertical distance within r.
        """
        img_pad = torch.nn.functional.pad(
            img, (0, 0, r // 2, r // 2, r // 2, r // 2), value=-1e10)
        imgH, imgW = img.shape[0], img.shape[1]

        # neighbor = -10 * torch.ones(imgH, imgW,  r * r, dtype=torch.float64, device=img.device)

        arr = torch.arange(imgW * imgH, dtype=torch.int64,
                           device=img.device).reshape(imgH, -1)
        hh = arr // imgW
        ww = arr % imgW

        w, es, et = [], [], []
        for i in range(r):
            for j in range(r):
                if i == r // 2 and j == r // 2:
                    continue

                rr = r // 2
                pos_v = (i - rr) * imgW + (j - rr)
                imgx = metric(img_pad[i:imgH + i, j:imgW + j], img,
                              (i - r // 2), (j - r // 2))

                flag = (hh + i - rr >= 0) & (hh + i - rr < imgH) & \
                       (ww + j - rr >= 0) & (ww + j - rr < imgW)

                w.append(imgx[flag].reshape(-1))
                pos = arr[flag].reshape(-1)
                es.append(pos)
                et.append(pos + pos_v)

        return torch.cat(w), torch.cat(es), torch.cat(et)


def full_coo_graph(N, dev=None):
    r"""
    Get the edge vectors of full graph for coo representation.
    
    The method returns (es, et, weight).

    Args:
        N(int): Number of node.
        dev(torch.device): Control the returned tensor device.
    
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
    """
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    es = torch.arange(0, N, dtype=torch.int64, device=dev).repeat(N)
    et = torch.arange(0, N, dtype=torch.int64, device=dev).repeat_interleave(N)
    w = torch.zeros(N * N, device=dev)
    return es, et, w


def spatial_knn_graph(feature_map, k, r, metric):
    r"""
    Build a knn graph over the first 2 dim of feature_map of images.

    Call the method :obj:`ImageKNN.get_sparse_conv` and output the result.

    Args:
        feature_map(torch.tensor) : Feature with shape :obj:`(H, W, C)`.
        k: The neighbor count of K-NN.
        r: The horizontal and vertical distance bound for linking pair of pixels.
    
    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`)

    
    .. todo:: batched ?
    """
    return ImageKNN.get_sparse_conv(feature_map, k, r, metric)


from .matrix_ops import scatter_sum


def vertex_reduce(edges, partition, edge_fea, node_fea):
    if node_fea is not None:
        node_fea = scatter_sum(node_fea, partition)
    edges = partition[edges]
    e, ef = sumup_duplicates(edges, edge_fea)
    return (e, ef, node_fea) if node_fea is not None else (e, ef)
