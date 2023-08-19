from collections import deque

import math

import cogdl.data
# import cv2
import numpy as np

import torch

import silearn.graph
import silearn.backends.torch_ops.graph_ops
from silearn import *
from silearn.optimizer.enc import *
from silearn.graph import GraphSparse
"""
Image Segmentation Model
"""


class KNNGraph(object):
    """
    KNN graph builder for image processing and feature map analysis
    """
    def get_sparse_conv(self, img: torch.Tensor, k=3, r=3):
        """
        Build the Graph from Image by KNN.

        @:param
        img: 3 x H x W.
        k: k edges per node.
        r: horizontal and vertical distance bound for linking pair of pixels.

        @:returns (w, es, et)
            w: edge weights representing distances.
            es: edge start nodes.
            et: edge target nodes.
            stored in np.ndarray
        """
        assert r % 2 == 1
        # img = img / 255
        if k >= r * r - 1:
            return self.__get_balanced_graph_conv__(img, r)
        img_pad = torch.nn.functional.pad(img, (0, 0, r // 2, r // 2, r // 2, r // 2), value=-1e10)
        imgH, imgW = img.shape[0], img.shape[1]
        neighbor = -10 * torch.ones(imgH, imgW, r * r, dtype=torch.float64, device=img.device)

        for i in range(r):
            for j in range(r):
                if i == r // 2 and j == r // 2:
                    continue
                imgx = -torch.sum((img_pad[i:imgH + i, j:imgW + j] - img) ** 2, dim=2)
                imgx *= (((i - r // 2) ** 2 + (j - r // 2) ** 2)) ** 0.5
                # if self.verbose: print((( - (i - r // 2) ** 2 - (j - r // 2) ** 2) ))
                neighbor[:, :, i * r + j] = imgx

        # if self.verbose: print(neighbor[0, 0])
        srt = torch.sort(neighbor, dim=2)
        pos = srt[1][:, :, -k:].reshape(-1, k)
        neighbor = srt[0][:, :, -k:].reshape(-1)
        del srt
        arr = torch.arange(imgW * imgH, dtype=torch.int64, device=img.device)

        if (r // 2 + 1) ** 2 <= k:
            hh = arr // imgW
            hh = hh.unsqueeze(1)
            ww = arr % imgW
            ww = ww.unsqueeze(1)
            rr = r // 2
            flag = (hh + pos // r - rr >= 0) & (hh + pos // r - rr < imgH) & \
                   (ww + pos % r - rr >= 0) & (ww + pos % r - rr < imgW) & (pos != r * r // 2)

            flag = flag.reshape(-1)
            pos = arr.unsqueeze(1) + (pos // r - r // 2) * imgW + (pos % r - r // 2)
            return neighbor[flag], pos.reshape(-1)[flag], arr.reshape(-1, 1).repeat(1, k).reshape(-1)[flag]
        pos = arr.unsqueeze(1) + (pos // r - r // 2) * imgW + (pos % r - r // 2)

        return neighbor, pos.reshape(-1), arr.reshape(-1, 1).repeat(1, k).reshape(-1)

    # if k >= r * r - 1 , construct graph connecting all pairs of
    #   pixels with horizontal and vertical distance within r.
    def __get_balanced_graph_conv__(self, img: torch.Tensor, r):
        img_pad = torch.nn.functional.pad(img, (0, 0, r // 2, r // 2, r // 2, r // 2), value=-1e10)
        imgH, imgW = img.shape[0], img.shape[1]

        # neighbor = -10 * torch.ones(imgH, imgW,  r * r, dtype=torch.float64, device=img.device)

        arr = torch.arange(imgW * imgH, dtype=torch.int64, device=img.device).reshape(imgH, -1)
        w, es, et = [], [], []
        for i in range(r):
            for j in range(r):
                if i == r // 2 and j == r // 2:
                    continue
                imgx = -torch.sum((img_pad[i:imgH + i, j:imgW + j] - img) ** 2, dim=2)

                rr = r // 2
                pos_v = (i - rr) * imgW + (j - rr)
                imgx *= (((i - r // 2) ** 2 + (j - r // 2) ** 2)) ** 0.5

                hh = arr // imgW
                ww = arr % imgW
                flag = (hh + i - rr >= 0) & (hh + i - rr < imgH) & \
                       (ww + j - rr >= 0) & (ww + j - rr < imgW)

                w.append(imgx[flag].reshape(-1))
                pos = arr[flag].reshape(-1)
                es.append(pos)
                et.append(pos + pos_v)

        return torch.cat(w), torch.cat(es), torch.cat(et)

    def get_full_graph(self, N, dev=None):
        """
        Get the edge vectors of full graph.

        @:param
        N: num of node.
        dev: torch device.
        @:returns es, et, w: torch tensor. Same as @get_sparse_conv
        """
        if dev is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        es = torch.arange(0, N, dtype=torch.int64, device=dev).repeat(N)
        et = torch.arange(0, N, dtype=torch.int64, device=dev).repeat_interleave(N)
        w = torch.zeros(N * N, device=dev)
        return es, et, w

    def pre_processing(self, image, use_lab=True):
        """
        Pre processing the input image (data & feature space conversion).
        @:param
        image: input (rgb) image (3 x H x W).
        use_lab: convert the rgb image into lab.
        @:returns (image, norm image).
            torch.tensor (3 x H x W).
            norm image: normalize the feature by standard deviation.
        """
        img = np.array(image)
        imgrgb = img.astype(np.float32) / 255
        if use_lab:
            imglab = cv2.cvtColor(imgrgb, cv2.COLOR_RGB2Lab).astype(np.float64)
            imglab /= np.array([255, 255, 255]).reshape((1, 1, -1))
            img = imglab
            # img = imglab / imglab.std() * 0.1

        else:
            img = img.astype(np.float64) / 255

        imgt = torch.tensor(img, dtype=torch.float64,
                            device="cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
        sqare = (np.var(img, axis=(0, 1)))  # /  (np.array([1.5, 1, 1]) )
        sqmean = np.mean(sqare)
        sstd = (sqare / sqmean) ** 0.5
        if self.verbose: print(sstd)
        self.__sstd__ = sstd
        return imgt, imgt / torch.tensor(sstd, device=imgt.device)


    def generate_weights(self, es, et, t, d):
        """
        Generate graph weights by distances and force the graph to be undirected.
        w = exp^(d / (mean(d) , t)).
        """
        es = es.reshape(-1)
        w = d.reshape(-1)
        e0, e1 = torch.cat([es, et]), torch.cat([et, es])
        del es, et
        w = torch.cat([w, w])
        norm = float(torch.mean(abs(w)))
        w /= norm * t
        w = torch.exp2(w)
        return e0, e1, w


    def hyper_distance(self, graph2, mean_node_feature, imgF, t, w_orig):
        """
        Update the weights by regional distances for full graph.

        @:param
        imgF: image feature num (3 for lab / rgb image).
        t: param of gen pixel-wise weights (e^(d / t)).
        w_orig: buffer indicating the equivalent edge num in the original graph.
        """
        mean = (mean_node_feature[graph2.edge_index[0]] - mean_node_feature[graph2.edge_index[1]]) ** 2
        mean = torch.sum(mean[:, :imgF], dim=1)
        mean *= self.r2 / t
        aggre = torch.clip(graph2.x[graph2.edge_index[0]][:, -1] * graph2.x[graph2.edge_index[1]][:, -1]
                           - w_orig[:, 1], min=0)
        ew = -mean * self.lambda_t_regional
        ew += torch.log2(torch.clip(graph2.edge_weight / graph2.edge_weight.sum(), min=2 ** -100)) * 0.1
        norm_ew = torch.exp2(ew) * aggre * self.lambda_weight_regional
        graph2.edge_weight += norm_ew
        w_orig[:, 1] += aggre

    def process(self, img_input, target_num=100, force_connectivity=True, use_lab=True,
                intercode=False, start_intercode_seg=math.inf,
                interseg=False, calc_se=False, fast_comp=True,
                relink_nearby=True, force_relink=3):
        """
        Segmentation process.
        @:param

        use_lab: convert the rgb input image into lab image (requires image to have 3 channels).
        target_num: segmentation target num.
        intercode: Also return a hierarchical encoding tree. Can be visualized by networkx.
        start_intercode_seg: target num of first level segmentation of hierarchical encoding tree.
            Finer segmentation is dismissed.
        interseg: return a hierarchical segmentation.
        calc_se: write the structural entropy map to self.dec_map.
        fast_comp: use parallel structural information algorithm.
        relink_nearby, force_relink: see method @recut_cc below.
            Disconnected segmentation in image may produced by long range edge construction.


        @:returns
            pixel label (np.ndarray, H x W) if interseg and intercode is False.
            [pixel label1, pixel label2, ..., pixel label n] if interseg is True.
                pixel labels are stored in torch tensor (cuda if available).
            pixel label, (es, et, c, d) if intercode and !interseg.
                pixel label: np.ndarray, H x W.
                et: vector of [es]'s parent node.
                c: mean color of corresponding region.
                d: size of corresponding region.
        """
        # torch.cuda.empty_cache()
        assert target_num > 0
        k, r, k2, r2 = self.k, self.r, self.k2, self.r2
        assert r > 0 and r2 > 0 and isinstance(r, int) and isinstance(r2, int)
        assert k > 0 and k2 > 0 and isinstance(k, int) and isinstance(k2, int)
        assert 0 < self.t_decay <= 1.0
        t = self.t
        assert t > 0

        # image tensors
        imgt, imgt_norm = self.pre_processing(img_input, use_lab)
        imgH = imgt.shape[0]
        imgW = imgt.shape[1]

        # sparse graph
        w, es, et = self.get_sparse_conv(imgt, k, r)
        e0, e1, w = self.generate_weights(es, et, t, w)
        g = silearn.graph.GraphSparse(torch.cat((e0, e1), dim=1), w)
        # g = cogdl.data.Graph(edge_index=(e0, e1), edge_weight=w)

        # initial partitioning
        # si = TwoDimSE(g)
        enc = silearn.graph.GraphEncoding(g)
        if fast_comp:
            dd = torch.abs((e0 - e1))
            pix_adjacency = (dd <= 1) + (dd == imgW) + (dd == imgW - 1) + (dd == imgW + 1)

            optim = OperatorPropagation(enc)
            optim.perform(adj_cover=pix_adjacency)
            # si.process_fast(adj_cover=pix_adjacency)
        else:
            # si.target_cluster = 10
            # si.global_var = 0
            dd = torch.abs((e0 - e1))
            pix_adjacency = (dd <= 1) + (dd == imgW) + (dd == imgW - 1) + (dd == imgW + 1)
            optim = OperatorMerge(enc)
            optim.perform(adj_cover=pix_adjacency)
            # si.process(adj_cover=pix_adjacency)
        # si.graph = None
        # del w, g, e0, e1, es, et

        # dense graph
        w, es, et = self.get_sparse_conv(imgt, k2, r2)
        norm = float(torch.mean(abs(w)))
        e0, e1, w = self.generate_weights(es, et, t, w)
        g = silearn.graph.GraphSparse(torch.cat((e0, e1), dim=1), w)

        # write node features
        imgF = imgt.shape[2]
        imgx = torch.arange(imgW).repeat(imgH).reshape(imgW, imgH, 1).swapaxes(0, 1).to(imgt.device)
        imgy = torch.arange(imgH).repeat(imgW).reshape(imgH, imgW, 1).to(imgt.device)
        imgxy = torch.cat([imgt_norm, imgx + 1, imgy + 1, torch.ones((imgH, imgW, 1)).to(imgt.device)], dim=2)
        del imgx, imgy

        si.graph.x = torch.tensor(imgxy, dtype=torch.float64, device=imgt.device).reshape(imgH * imgW, -1)
        com2 = si.community_result = si.community_result.to(imgt.device)

        # force connectivity
        idx = torch.arange(imgW * imgH, dtype=torch.int64, device=imgt.device).reshape(imgH, imgW)
        exs = idx[:, :-1].reshape(-1)
        ext = idx[:, 1:].reshape(-1)
        eys = idx[:-1, :].reshape(-1)
        eyt = idx[1:, :].reshape(-1)
        g

        es = torch.cat([si.graph.edge_index[0], exs, ext, eys, eyt])
        et = torch.cat([si.graph.edge_index[1], ext, exs, eyt, eys])
        ew = si.graph.edge_weight
        si.graph.edge_index = (es, et)
        emb = si.graph.x[:, :imgF] + 0
        # emb[:, 1:3] = emb[:, 1:3] *  2.55 ** 2
        wx = -torch.sum((emb[exs] - emb[ext]) ** 2, dim=1)
        wy = -torch.sum((emb[eys] - emb[eyt]) ** 2, dim=1)
        wx /= norm * t
        wy /= norm * t

        si.graph.edge_weight = torch.cat(
            [ew, self.lambda_connectivity_edges * 2 ** wx, self.lambda_connectivity_edges * 2 ** wx,
             self.lambda_connectivity_edges * 2 ** wy, self.lambda_connectivity_edges * 2 ** wy])
        si.community_result = com2 = com2.to(imgt.device)
        del exs, eys, ext, eyt

        # log parameters
        pos_inter_node = 0
        pos_inter_edge = 0
        color = None
        size = None
        last_com = None
        es_inter = None
        et_inter = None

        stopped_iter = 0

        intersegs = []
        if calc_se:
            g0 = si.graph

        size_scale = 1
        full_g_done = False
        com_sz_smmoth = com2.max() + 1
        w_orig = torch.ones_like(si.graph.edge_weight)
        w_edge = torch.ones_like(si.graph.edge_weight)
        w_orig[:ew.shape[0]] = 0
        w_edge[ew.shape[0]:] = 0
        w_orig = torch.stack((w_orig, w_edge), dim=1)

        full_g_weight_done = False
        g0 = si.graph.clone()
        com_full = None  # encoding from raw image to full graph

        # iterative merging
        while (com_sz_smmoth > target_num):

            if intercode and pos_inter_node == 0 and com2.max() < start_intercode_seg:
                pos_inter_node = si.community_result.max() + 1
                es_inter = torch.zeros(imgW * imgH * 2 + 1, dtype=torch.int64, device=imgt.device)
                et_inter = torch.zeros(imgW * imgH * 2 + 1, dtype=torch.int64, device=imgt.device)
                color = torch.zeros(imgW * imgH * 2 + 1, 3, dtype=torch.float64, device=imgt.device)
                size = torch.zeros(imgW * imgH * 2 + 1, dtype=torch.float64, device=imgt.device)
                last_com = torch.arange(pos_inter_node, dtype=torch.int64, device=imgt.device)

            graph2, w_orig = si.generate_hyper_graph(feature_aggregation='sum', erase_self_loop=False,
                                                     reset_community_id=False, edges_x=w_orig)

            if graph2.num_nodes < self.thresh_full_g and not full_g_done:
                es, et, w = self.get_full_graph(graph2.num_nodes, imgt.device)
                graph2 = cogdl.data.Graph(
                    edge_index=(torch.cat([graph2.edge_index[0], es]), torch.cat([graph2.edge_index[1], et])),
                    edge_weight=torch.cat([graph2.edge_weight, w]), x=graph2.x)

                w_orig = torch.stack((torch.cat([w_orig[:, 0], torch.zeros_like(w)]),
                                      torch.cat([w_orig[:, 1], torch.zeros_like(w)])), dim=1)
                si.graph = graph2
                si.force_coding(torch.arange(graph2.num_nodes, dtype=torch.int64, device=imgt.device))
                graph2, w_orig = si.generate_hyper_graph(reset_community_id=False, feature_aggregation='sum',
                                                         edges_x=w_orig)
                full_g_done = True

            si.graph = None

            meanx = graph2.x[:, :] / graph2.x[:, -1:]
            if intercode and pos_inter_node > 0:
                color[last_com] = meanx[:, :3] * self.__sstd__
                size[last_com] = graph2.x[:, -1]

            if graph2.num_edges == 0:
                break

            # ww = graph2.edge_weight.clone()
            if not full_g_weight_done and full_g_done:
                self.hyper_distance(graph2, meanx, imgF, norm * t, w_orig)
                if full_g_done:
                    full_g_weight_done = True
                    g0 = graph2.clone()
                    com_full = torch.arange(graph2.num_nodes)

            t /= self.t_decay
            sii = TwoDimSE(graph2)

            sii.target_cluster = target_num
            # es = w_orig[:, 1]
            # com2sz = graph2.num_nodes
            # edgesz = graph2.num_edges
            # interaction_size = edgesz / com2sz
            # receptive_field = interaction_size ** 2
            # conductance = torch.sum(es[graph2.edge_index[0] != graph2.edge_index[1]]) / torch.sum(w_orig)

            sii.global_var = size_scale  # * receptive_field

            sii.m_scale = size_scale
            # if self.verbose: print(size_scale)
            if fast_comp:
                sii.process_fast(adj_cover=(w_orig[:, 0] > 0), loop0=True)
            else:
                sii.process(adj_cover=(w_orig[:, 0] > 0))

            com2 = sii.community_result[com2]
            if com_full is not None:
                com_full = sii.community_result[com_full]
            cnt2 = scatter_cnt(com2)
            com_sz_smmoth = (cnt2 > 10).sum()

            si = sii
            if intercode and pos_inter_node > 0:
                cntcom = scatter_cnt(sii.community_result)
                lifted = cntcom[sii.community_result] > 1
                lifted_comm = cntcom > 1
                es0 = last_com[lifted]
                commx = sii.community_result.max()
                cnt_new_nodes = torch.sum(lifted_comm)
                remap = torch.zeros(commx + 1, dtype=torch.int64, device=imgt.device)
                remap[lifted_comm] = torch.arange(pos_inter_node, pos_inter_node + cnt_new_nodes, dtype=torch.int64,
                                                  device=imgt.device)
                et0 = remap[sii.community_result[lifted]]

                pos_inter_node += int(cnt_new_nodes)

                es_inter[pos_inter_edge:pos_inter_edge + es0.shape[0]] = es0
                et_inter[pos_inter_edge:pos_inter_edge + es0.shape[0]] = et0
                pos_inter_edge += int(es0.shape[0])

                pre_com = last_com
                last_com = torch.zeros(commx + 1, device=imgt.device, dtype=torch.int64)
                last_com[sii.community_result] = pre_com
                if self.verbose: print(torch.sum(lifted))
                if self.verbose: print(et0.shape)
                last_com[lifted_comm] = remap[lifted_comm]

            if sii.node_cnt == com2.max() + 1:
                stopped_iter += 1
            else:
                stopped_iter = 0

            if not full_g_done:
                size_scale *= 1.2
            elif graph2.num_nodes > 100:
                size_scale *= 1.2
            else:
                size_scale *= 1.01
            if size_scale >= self.scaling_bound:
                if self.verbose: print("scaling bound reached")
                break
            elif sii.node_cnt == (com2.max() + 1):
                if self.verbose: print("scaling...")

            else:
                if self.verbose: print("scaled")
                if interseg and full_g_done and com_sz_smmoth < 50:
                    six = TwoDimSE(g0)
                    six.force_coding(com_full)
                    si2 = six.calc_entropy(log_M=size_scale)
                    if self.verbose: print("log_M:{}, si2:{}, seg count{}".format(size_scale, si2, com_sz_smmoth))
                    intersegs.append((size_scale, com2, float(si2)))

            if stopped_iter > 200:
                if self.verbose: print("eary stopped")
                break

        if intercode:
            graph2 = si.generate_hyper_graph(feature_aggregation='sum', erase_self_loop=False, reset_community_id=False)
            meanx = graph2.x[:, :-1] / graph2.x[:, -1:]
            color[last_com] = meanx[:, :3] * self.__sstd__
            size[last_com] = graph2.x[:, -1]

            last_cmsz = last_com.shape[0]
            es_inter[pos_inter_edge:pos_inter_edge + last_cmsz] = last_com
            et_inter[pos_inter_edge:pos_inter_edge + last_cmsz] = pos_inter_node
            size[pos_inter_node] = imgH * imgW
            es_inter = es_inter[:pos_inter_edge + last_cmsz]
            et_inter = et_inter[:pos_inter_edge + last_cmsz]
            color = color[:pos_inter_node + 1]
            size = size[:pos_inter_node + 1]
        if calc_se:
            si = TwoDimSE(g0)
            si.force_coding(com2)
            self.dec_map = []
            self.dec_map.append(si.calc_e1(reduction=False))
            self.dec_map.append(si.calc_entropy(reduction=False))
            self.dec_map.append(si.calc_e1(reduction=False) - si.calc_entropy(reduction=False))
            self.dec_map.append(si.calc_entropy_rate(reduction=False))
            if self.verbose: print(si.calc_entropy() + 1 - si.calc_entropy_rate())

        if interseg:
            return intersegs

        if force_connectivity:
            ans = com2.cpu().numpy().reshape(imgH, imgW)
            ret = -np.ones_like(ans)
            ret = self.recut_cc(ans, ret, relink_nearby, force_relink)
            if intercode:
                return ret, (es_inter, et_inter, color, size)

            return ret
        if intercode:
            return com2.cpu().numpy().reshape(imgH, imgW), (es_inter, et_inter, color, size)
        return com2.cpu().numpy().reshape(imgH, imgW)

    def recut_cc(self, img_label0, img_label1, relink_nearby=False, force_relink=10):
        """
        Force the segmentation connected. (Each partition is a connected component.)

        image_label0: input buffer (np.ndarray).
        image_label1: output buffer (np.ndarray).
        return: image_label1.
        relink_nearby: 若找到不连通的划分，将该划分中较小的联通分量划归到某一个临近的划分。如果为False，则标记为新的划分。
        force_relink: 将大小低于该值的划分合并到临近的划分。
        算法：dfs+并查集。复杂度O(n log n)
        """
        h = img_label0.shape[0]
        w = img_label0.shape[1]
        pre_nearby = {}
        cur_usepos = {}
        pos = -1
        pre_pos = 0
        if relink_nearby:
            size_cc = np.zeros(img_label0.max() + 1)
            remap = np.arange(h * w)

        def get_remap(i):
            if remap[i] == i: return i
            remap[i] = get_remap(remap[i])
            return remap[i]

        for i in range(h):
            for j in range(w):
                if img_label1[i][j] != -1:
                    pre_pos = img_label1[i][j]
                    continue
                cur = img_label0[i][j]
                pos += 1
                use_pos = pos
                stack = deque()
                stack.append((i, j))
                size_tot = 0
                while len(stack) > 0:
                    m, n = stack.popleft()
                    if m < 0 or n < 0 or m >= h or n >= w or img_label0[m][n] != cur or img_label1[m][n] != -1:
                        continue
                    stack.append((m - 1, n))
                    stack.append((m, n - 1))
                    stack.append((m + 1, n))
                    stack.append((m, n + 1))
                    stack.append((m - 1, n - 1))
                    stack.append((m + 1, n + 1))
                    stack.append((m + 1, n - 1))
                    stack.append((m - 1, n + 1))
                    img_label1[m][n] = use_pos
                    size_tot += 1
                if relink_nearby:
                    if size_tot <= force_relink:
                        remap[use_pos] = get_remap(pre_pos)
                    elif size_tot > size_cc[cur]:
                        if size_cc[cur] != 0:
                            # remap pre cc to nearpre-pre cc
                            remap[cur_usepos[cur]] = get_remap(pre_nearby[cur])
                        pre_nearby[cur] = get_remap(pre_pos)
                        cur_usepos[cur] = use_pos
                        size_cc[cur] = size_tot
                    else:
                        remap[use_pos] = get_remap(pre_pos)
        if relink_nearby:
            for i in range(pos + 1):
                get_remap(i)
            pos1 = 0
            used = {}
            for i in range(pos + 1):
                if remap[i] in used.keys():
                    remap[i] = used[remap[i]]
                else:
                    used[remap[i]] = pos1
                    remap[i] = pos1
                    pos1 += 1
            img_label1 = remap[img_label1.astype(np.int)]
        return img_label1