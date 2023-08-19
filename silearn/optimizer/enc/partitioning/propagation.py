import time

import torch
import math

import silearn.model.encoding_tree
from silearn.optimizer.enc.operator import Operator
# import torch_sparse

import ctypes

import os
import numpy as np
import torch
import torch.nn.functional as F


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim


class OperatorPropagation(Operator):
    # partition: silearn.model.enc.encoding_tree.Partitioning
    adjacency_restriction: None

    @staticmethod
    def reduction_edge(edges, edge_transform, *weights):
        cnt_e = edge_transform.max() + 1#边处于社区位置的种类（社区内、割边）数，edge_transform为每条边的位置类型编号
        e1 = torch.zeros(size=(cnt_e, edges.shape[1]),
                         dtype=edges.dtype,
                         device=edges.device)
        # print(edge_transform.shape)
        # print(edges.shape)
        # print(e1.shape)

        edges = e1.scatter(0,
                           edge_transform.reshape(-1, 1).repeat((1, 2)), edges)#edges根据索引重新分配到e1中,dim=0,索引是 边位置编号edge_transform变为一列且重复一列(让edges一行一起分配),其实就是edges去重
        ret = [edges
               ] + [silearn.scatter_sum(w, edge_transform) for w in weights] #根据边处在社区位置的类型把边权加起来，同一个社区中的会double
        return tuple(ret)

    @staticmethod
    def get_edge_transform(edges, identical_flag=False):
        max_id = int(edges[:, 1].max() + 1) #此时的edges是社区下标，max_id是社区数
        bd = 1
        shift = 0
        while bd <= max_id:
            bd = bd << 1
            shift += 1  #社区数<2^bd
        # todo: hash if shift is too big
        edge_hash = (edges[:, 0] << shift) + edges[:, 1]  #边右顶点社区的下标+边左顶点社区下标*2^shift  从二进制看就是左部分为左顶点社区，有右部分为右顶点社区

        if identical_flag:
            _, transform, counts = torch.unique(edge_hash,
                                                return_inverse=True,
                                                return_counts=True)  #count为_中的每个元素在原始张量中的数量，transform为原始张量在_上索引
            flag = counts[transform] != 1
            return transform, flag
        else:
            _, transform = torch.unique(edge_hash, return_inverse=True)  #_为独立不重复元素组成张量，transform为原始张量在_上索引，边的左右顶点在的社区一样表示两条边当前在的社区位置相同（社区内、割边）
            return transform

    @staticmethod
    def sum_up_multi_edge(edges, *weights, operation_ptrs=None):
        if operation_ptrs is not None:
            em = edges[operation_ptrs]
            wm = [i[operation_ptrs] for i in weights]
            trans = OperatorPropagation.get_edge_transform(em)
            redu = OperatorPropagation.reduction_edge(em, trans, *wm)
            cnt = redu[0].shape[0]#边处于社区位置的种类数（社区内，不同社区间割边）
            # operation_ptrs = operation_ptrs[:cnt]
            # skip_ptrs = operation_ptrs[cnt:]
            # edges[operation_ptrs] = 0
            edges[operation_ptrs][:cnt] = redu[0]
            ret = [edges]
            for i in range(len(weights)):
                weights[i][operation_ptrs][:cnt] = redu[i + 1]
                weights[i][operation_ptrs][cnt:] = 0
                ret += [weights[i]]
            return ret
        else:
            trans = OperatorPropagation.get_edge_transform(edges)
            return OperatorPropagation.reduction_edge(edges, trans, *weights)#返回的是一个元组 边处于社区位置的种类 以及每个种类下的边权和

    @staticmethod
    def sum_up_multi_edge_ts(edges, *weights, operation_ptrs=None):
        if operation_ptrs is not None:
            em = edges[operation_ptrs]
            wm = [i[operation_ptrs] for i in weights]
            trans = OperatorPropagation.get_edge_transform(em)
            redu = OperatorPropagation.reduction_edge(em, trans, *wm)
            cnt = redu[0].shape[0]
            operation_ptrs = operation_ptrs[:cnt]
            skip_ptrs = operation_ptrs[cnt:]
            edges[operation_ptrs] = redu[0]
            for i in range(len(weights)):
                weights[i][operation_ptrs] = redu[i + 1]
                weights[i][skip_ptrs] = 0
            return skip_ptrs
        else:
            trans = OperatorPropagation.get_edge_transform(edges)
            return OperatorPropagation.reduction_edge(edges, trans, *weights)

    centers_proposal = torch.nn.AdaptiveAvgPool2d((15, 15))

    def perform_x(self, img):
        sim = pairwise_cos_sim(img[1].reshape(-1, 3), img.reshape(-1, 3))
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        sim.unsqueeze(dim=-1).sum(dim=2)
        mask.sum(dim=-1, keepdim=True)

    def perform(self,
                p=1.0,
                ter=0,
                contains_self_loops=None,
                adj_cover=None,
                min_com=1,
                max_com=math.inf,
                di_max=False,
                srt_M=False,
                f_hyper_edge=None,
                m_scale=None,
                re_compute=True):
        assert 1 <= min_com <= max_com
        assert 0.0 <= p <= 1.0
        edges, trans_prob = self.enc.graph.edges #edges,ew


        self._log2m = torch.log2(trans_prob.sum())  #trans_prob.sum()=vol(G) self._log2m=log2 vol(G)

        if m_scale != None:
            self._log2m += m_scale

        operated_cnt = math.inf
        transs = []
        merge_all = False
        if contains_self_loops is None:
            # noinspection PyUnresolvedReferences
            contains_self_loops = bool((edges[:, 0] == edges[:, 1]).any())  #连接自己的边

        # v1 = None
        vst = None
        dH0 = None

        # vst cached when unchanged
        cache = False
        # only detect altered_e
        # altered_e = None

        current_num_vertices = self.enc.graph.num_vertices  #节点数
        #print('current_num_vertices:{}'.format(current_num_vertices))

        t = time.time()

        if re_compute is not True:
            trans = self.enc.node_id
            if adj_cover is None:
                cache = False
                edges = trans[edges]
                edges, trans_prob = OperatorPropagation.sum_up_multi_edge(
                    edges, trans_prob)
                current_num_vertices = edges.max() + 1

            else:
                cache = False
                edges = trans[edges]
                edges, trans_prob, adj_cover = OperatorPropagation.sum_up_multi_edge(
                    edges, trans_prob, adj_cover)
                current_num_vertices = edges.max() + 1
            transs.append(trans)

        while operated_cnt > ter or not merge_all:
            max_operate_cnt = math.ceil((current_num_vertices - min_com) * p)#（当前顶点数-最小社区数）*1 最多需要进行的操作数
            #print('current_num_vertices:{},max_operate_cnt:{}'.format(current_num_vertices,max_operate_cnt))
            edge_s = edges[:, 0]#左一列顶点
            edge_t = edges[:, 1]#右一列顶点

            mx = torch.max(edges)#最大顶点下标，+1就是顶点/社区数了
            #print('mx:{},operated_cnt:{}'.format(mx,operated_cnt))
            if operated_cnt <= ter + 1:
                merge_all = True  #上一轮的merge操作数小于等于1
            # print(trans_prob.shape)
            # print(self.graph.edge_index[1].shape)
            # print("time-x:{}".format(time.time() - t))
            # t = time.time()
            # print(self.enc.graph.num_vertices)
            #print('merge_all:{},cache:{}'.format(merge_all,cache))
            if not cache:
                v1 = silearn.scatter_sum(trans_prob,
                                         edge_t,
                                         clip_length=mx + 1).reshape(-1) #按右一列顶点求di

                # print(self.enc.graph.stationary_dist)
                vst = v1[edges] #每条边的左右顶点/社区的di/vol(a)



            if contains_self_loops:
                if not cache:
                    g1 = silearn.scatter_sum(trans_prob * (edge_s != edge_t),
                                             edge_t,
                                             clip_length=mx + 1) #社区割边

                    gst = g1[edges]# 割边数变成edges形式
                    vx = vst.sum(dim=1) #每一条边连接的两个社区的vol(α)求和 ，每一行求和

                    # gx = gst.sum(dim = 1)

                    # g0s, g0t, g0x = vs, vt, vx
                    # if hyper_g:
                    #     if self.graph.x is None:
                    #         self.graph.x = g1.clone()
                    #     else:
                    #         g0 = self.graph.x
                    #         print(g0.shape)
                    #         g0s = g0[edges[:, 0]]
                    #         g0t = g0[edges[:, 1]]
                    #         g0x = g0s + g0t
                    vin = vst - gst #每一条边连接的两个社区的vol(α)-gα 即社区内部边权之和

                    dH1 = (vin[:, 0]) * torch.log2(
                        vst[:, 0]) + (vin[:, 1]) * torch.log2(vst[:, 1]) - (
                            vin[:, 0] + vin[:, 1]) * torch.log2(vx)  # 左社区的 (vol(α)-gα )*log2 vol(α) +右社区的(vol(β)-gβ )*log2 vol(β)-(vol(α)-gα+vol(β)-gβ )*log2 vol(α+β)
                    dH2 = 2 * trans_prob * ((self._log2m) - torch.log2(vx)) #2*(log2 vol(G)-log2 vol(α+β) )*边权
                    dH0 = dH1 + dH2 #Δ
                    op = (dH0 > 0)
                    # 使用M选择合并时
                    # if srt_M:
                    #     dH = - dH2 / dH1 # * (self._log2m)
                    # print(dH1.min())
                    # print(dH2.min())
                    # print(dH.min())
                    # dH = dH0
                    # else:
                    dH = dH0
                    dHM = dH

                else:
                    op = (dH0 > 0)
                    dH = dH0

                if not torch.any(op):
                    break
                if not merge_all:
                    op = (dH >= torch.median(dH[op]))
            else:
                dH = trans_prob * (self._log2m -
                                   torch.log2(vst[:, 0] + vst[:, 1])) #ew*(log2 vol(G)- log2(di+dj) )
                op = (dH > torch.median(dH[dH > 0])) #选大于0且大于等于中位数的那几条边

            cache = True

            if adj_cover is not None:
                op = torch.logical_and(op, adj_cover > 0)

            # print(f"t1:{ time.time() - t}")

            # print(operated_cnt)
            merge = op
            # rand_idx = torch.randint(0,  2**31,(1, self.enc.graph.num_vertices), device=self.device)[0]
            # noinspection PyTypeChecker
            # merge = torch.logical_and(merge, (vs < vt) + torch.logical_and((vs == vt) , rand_idx[edges[:, 0]] < rand_idx[edges[:, 1]]))
            # merge = torch.logical_and(merge, (vs < vt) + torch.logical_and((vs == vt) , rand_idx[edges[:, 0]] < rand_idx[edges[:, 1]]))
            hash_x = edge_s * 10007 // 1009
            hash_t = edge_t * 10007 // 1009

            merge = torch.logical_and(
                merge,   #dH大于中位数大于0 且满足下面条件（似乎是去掉重复边）
                torch.logical_or((vst[:, 0] < vst[:, 1]),  #左顶点di<右顶点di  or  左顶点di=右顶点di 且 左顶点hash<右顶点hash
                                 torch.logical_and((vst[:, 0] == vst[:, 1]),
                                                   hash_x < hash_t)))
            if not torch.any(merge):
                operated_cnt = 0 #没有一个要merge
                continue

            # t = time.time()
            id0 = edge_s
            id1 = edge_t
            if not merge_all:   #还没merge完
                dH = dH * (1.0 + 1e-6 * hash_x * 1e-6 * hash_t) #dH=dH*(1+hash_x/10^6 * hash_t/10^6)

            id0 = id0[merge]  #要merge的边的左顶点
            id1 = id1[merge]  #要merge的边的右顶点


            dH = dH[merge] #要merge的边的dH
            _, dH_amax = silearn.scatter_max(dH, id0)  #顶点有多条边要merge选个最好的，dH_amax放最终要merge的边在dH中索引，如id0[dH_amax[i]]和id1[dH_amax[i]]merge
            # dh_amax[i] = (argmax_j[dH[j]] for id0[j] = i) if \exist i in id0, else dH.shape + 1
            # dH_amax[i] is unique

            dH_amax = dH_amax[dH_amax <
                              dH.shape[0]]  # then dH_amax is a unique set
            operated_cnt = int(dH_amax.shape[0])  # cuda synchronized  要进行merge操作的次数

            if operated_cnt == 0:
                continue

            # print("max_op:"+str(max_operate_cnt))
            # print(max_operate_cnt)
            # print(operated_cnt)

            # 双向max
            if operated_cnt > max_operate_cnt and di_max:
                id0 = id0[dH_amax]
                id1 = id1[dH_amax]
                dH = dH[dH_amax]
                _, dH_amax = silearn.scatter_max(dH, id1)
                dH_amax = dH_amax[dH_amax < dH.shape[0]]
                operated_cnt = int(dH_amax.shape[0])

            if operated_cnt > max_operate_cnt:
                _, idx = torch.sort(dH[dH_amax], descending=True)
                # idx = torch.randperm(dH_amax.shape[0])
                dH_amax = dH_amax[idx[:max_operate_cnt]]
                operated_cnt = max_operate_cnt
                # p = 1.0
            current_num_vertices -= operated_cnt
            # operated_cnt = ddd_new

            ids = id0[dH_amax]
            idt = id1[dH_amax]  #要merge的左右顶点对
            trans = torch.arange(edges.max() + 1, device=self.enc.graph.device)#合并前的社区划分(一人一个）
            trans[ids] = trans[idt]
            # if operated_cnt < 10:
            #     altered = torch.zeros(self.enc.graph.num_vertices, device=self.enc.graph.device, dtype=torch.bool)
            #     altered[ids] = True
            #     altered[idt] = True
            #     altered_e = torch.nonzero(torch.logical_or(altered[edges[:, 0]],
            #                                  altered[edges[:, 1]]))  # edges
            #
            # else:
            #     altered_e = None
            # trans[id0] = id1
            # trans[i] = j: label node i to j
            # print(operated_cnt)
            lg_merge = math.log2(operated_cnt + 2)

            # todo: test speed: limit var
            # var = #trans != torch.arange(self.enc.graph.num_vertices, device=self.enc.graph.device)
            for i in range(int(lg_merge)):
                # ids[var] = trans[trans[var]]
                # old = trans[ids]
                trans[ids] = trans[trans[ids]]  #合并完的社区划分
                # ids = torch.nonzero(trans[ids] != old)

            # print(time.time() - t)

            # torch.tensor([id0])
            # torch.tensor([id1])

            # t = time.time()
            # failed = torch.logical_or(trans == torch.arange(self.enc.graph.num_vertices) , altered) == 0
            # print(ids)
            # print(idt)
            # print(torch.nonzero(failed))

            # print(torch.nonzero(altered))
            # print(torch.nonzero(trans != torch.arange(self.enc.graph.num_vertices)))

            # compress id allocation
            trans = torch.unique(trans, return_inverse=True)[1] #trans中每个在独立不重复张量中索引，即在第几个社区
            # _, trans, counts= torch.unique(trans, return_inverse=True, return_counts = True)

            # self.community_result = trans
            transs.append(trans)
            # print(trans)

            if self.enc.graph.num_vertices - operated_cnt == min_com:
                break
            # print(f"t2:{ time.time() - t}")

            if adj_cover is None:
                cache = False

                edges = trans[edges] #edges从顶点下标变为社区下标
                edges, trans_prob = OperatorPropagation.sum_up_multi_edge(
                    edges, trans_prob)

                # operated = torch.nonzero(torch.sum(counts[edges] > 1, dim = -1)).reshape(-1)
                # print(operated)
                # OperatorPropagation.sum_up_multi_edge(edges, trans_prob, operation_ptrs=torch.nonzero(merge).reshape(-1))

            else:
                cache = False
                edges = trans[edges]
                edges, trans_prob, adj_cover = OperatorPropagation.sum_up_multi_edge(
                    edges, trans_prob, adj_cover)
                # trans_prob = trans_prob + f_hyper_edge(self.graph.x,
                #                                                                edges[:, 0],
                #                                                                edges[:, 1],
                #                                                                trans_prob,
                #                                                                adj_cover)
            contains_self_loops = True
            # break

            # print(time.time() - t)
        if len(transs) != 0:
            trans = transs[-1]
            for i in reversed(range(len(transs) - 1)):
                trans = trans[transs[i]]

            self.enc.node_id = trans
        else:

            com0 = torch.arange(self.enc.graph.num_vertices,
                                device=self.enc.graph.device)
            self.enc.node_id = com0

    def iterative_merge(self,
                        verbose=False,
                        min_com=1,
                        max_iteration=30,
                        tau=0.1,
                        sample_ratio=0.5,
                        p=0.5,
                        m_scale=-1):
        prob_e = torch.ones(self.enc.graph.num_edges,
                            device=self.enc.graph.device)
        edges, _ = self.enc.graph.edges
        for i in range(max_iteration):

            rand = torch.rand(self.enc.graph.num_edges,
                              device=self.enc.graph.device) * prob_e
            bound = torch.msort(rand)[int(
                (prob_e.shape[0] - 1) * sample_ratio)]

            cover_adj = rand >= bound
            self.perform(adj_cover=cover_adj,
                         min_com=min_com,
                         p=p,
                         m_scale=m_scale)
            self.perform(min_com=min_com,
                         re_compute=False,
                         p=p,
                         m_scale=m_scale)
            c = torch.logical_not(cover_adj)
            operated = self.enc.node_id[edges[:,
                                              0]] == self.enc.node_id[edges[:,
                                                                            1]]
            # prob_e[torch.logical_and(cover_adj,torch.logical_not(operated))] *= 1 - tau
            prob_e[torch.logical_and(c, operated)] /= 1 - tau
            # print(prob_e)

            # print(self.enc.structural_entropy(reduction="sum"))

    # By Hujin

    # def iterative_merge_c(self, verbose = False, min_com = 1,
    #                     max_iteration = 30, tau = 0.1, sample_ratio = 0.5, p = 0.5, m_scale = 0):
    #     prob_e = torch.ones(self.enc.graph.num_edges, device=self.enc.graph.device)
    #     edges, _ = self.enc.graph.edges
    #     for i in range(max_iteration):
    #
    #         rand = torch.rand(self.enc.graph.num_edges, device=self.enc.graph.device) * prob_e
    #         bound = torch.msort(rand)[int((prob_e.shape[0] - 1) * sample_ratio)]
    #
    #         cover_adj = rand >= bound
    #         self.process_c(adj_cover = cover_adj)
    #         self.process_fast(min_com = min_com, re_compute=False, p = p, m_scale=m_scale)
    #         c = torch.logical_not(cover_adj)
    #         operated = self.enc.node_id[edges[:, 0]] == self.enc.node_id[edges[:, 1]]
    #         prob_e[torch.logical_and(c,torch.logical_not(operated))] *= 1 - tau
    #         prob_e[torch.logical_and(c,operated)] /= 1 - tau
    #         print(prob_e)
    #
    #         print(self.enc.structural_entropy(reduction="sum"))

    # def exchange_nodes(self, vin, vst, vx, trans_prob):
    #     dH1 = (vin[:, 0]) * torch.log2(vst[:, 0])
    #     dH2 = 2 * trans_prob * ((self._log2m) - torch.log2(vx))
    #
    #     dH3 = (vin[:, 0]) * torch.log2(vst[:, 0])
    #             vin[:, 0] + vin[:, 1]) * torch.log2(vx)
    #     dH4 = 2 * trans_prob * ((self._log2m) - torch.log2(vx))
    #
