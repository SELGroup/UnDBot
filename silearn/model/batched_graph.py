import torch


def combine_batch_graph(edges, w, n_vertices):
    r"""
    Combine the KNN graphs while avoiding duplicated edges between different batches. :py:mod:`BatchedGraphModuel`
    """
    bs = edges.shape[0]
    device = edges.device
    idx = (torch.arange(bs, device=device, dtype=torch.int64) * n_vertices)\
            .reshape(-1, 1, 1)
    return (edges + idx).reshape(-1, 2), w.reshape(-1)


class BatchedGraphModule(torch.nn.Module):
    r"""
    Combine the KNN graphs while avoiding duplicated edges between different batches.
    """

    def __init__(self, num_idx, bs, device):
        r"""
        Args: 
            num_idx(int): num_idx should be greater than the maximum node id in graph edges.
            bs(int): batch size.
            device(torch.device): device the graph stored.
        """
        super().__init__()
        self.register_buffer("idx", (torch.arange(bs, device=device, dtype=torch.int64) * num_idx)\
            .reshape(-1, 1, 1), persistent=False)

    def combine_batch(self, edges, w):
        r"""
        Combine the batched KNN graphs. Add different IDs to the edges of different batches so that the node id of the edges in different batches do not overlap.

        Args: 
            edges: graph edges :obj:`(B, N, 2)`.
            w: the weights of graph edges, shape :obj:`(B, N)`

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        return (edges + self.idx).reshape(-1, 2), w.reshape(-1)

    def reduce(self, labels):
        return


class ShiftGraph(torch.nn.Module):
    r"""
    Build knn graphs over the images with shape :obj:`(B, C, H, W)`.
    """

    def __init__(self, d, metric=None):
        r"""
        Args:
            d(int): Each pixel connect with 2d-1 other pixels.
            metric(func): This function controls the metric used for calculating the distance between pixels. Euclidean distances is default.
        """
        if metric is None:
            metric = lambda f1, f2, x, y: -(
                (f1 - f2)**2).sum(dim=1) * (x**2 + y**2)**0.5
        super(ShiftGraph, self).__init__()
        self.d = d
        self.metric = metric

    def forward(self, x: torch.Tensor):
        r"""
        Return the edges :obj:`(B, N, 2)` and weights :obj:`(B, N)` for batched knn graph.
        
        Args:
            x(torch.tensor): images with shape :obj:`(B, C, H, W)`.
        
        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """
        H, W = x.shape[-2], x.shape[-1]
        id0 = torch.arange(W * H, dtype=torch.int64,
                           device=x.device).reshape(H, -1)

        d = self.d
        img_pad = torch.nn.functional.pad(x, (d // 2, d // 2, d // 2, d // 2),
                                          value=-1e10)
        imgH, imgW = x.shape[-2], x.shape[-1]

        # neighbor = -10 * torch.ones(imgH, imgW,  r * r, dtype=torch.float64, device=img.device)

        w, edges = [], []
        for i in range(d):
            for j in range(d):
                if i == d // 2 and j == d // 2:
                    continue

                rr = d // 2
                pos_v = (i - rr) * imgW + (j - rr)
                imgx = self.metric(img_pad[:, :, i:imgH + i, j:imgW + j], x,
                                   (i - d // 2), (j - d // 2))

                imgx = imgx[:,
                            max(d // 2 - i, 0):imgH - max(i - d // 2, 0),
                            max(d // 2 - j, 0):imgW - max(j - d // 2, 0)]
                idx = id0[max(d // 2 - i, 0):imgH - max(i - d // 2, 0),
                          max(d // 2 - j, 0):imgW - max(j - d // 2, 0)]
                idx = idx.unsqueeze(0).repeat((imgx.shape[0], 1, 1))
                # flag = (self.H_idx + i - rr >= 0) & (self.H_idx + i - rr < imgH) & \
                #        (self.W_idx + j - rr >= 0) & (self.W_idx + j - rr < imgW)

                w.append(torch.flatten(imgx, start_dim=-2, end_dim=-1))
                pos = torch.flatten(idx, start_dim=-2, end_dim=-1)
                edges.append(torch.stack((pos, pos + pos_v), dim=-1))

        return torch.cat(edges, dim=-2), torch.cat(w, dim=-1)


if __name__ == '__main__':
    module = ShiftGraph(3)
    edges, ew = module(torch.arange(50).reshape(2, 1, 5, 5))
    print(edges.shape)
    model = BatchedGraphModule(25, 2, edges.device)
    edges, ew = model.combine_batch(edges, ew)
    print(edges.shape)