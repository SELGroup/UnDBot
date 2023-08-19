import os

import numpy as np
import torch

import silearn.optimizer.enc.operator as op

import silearn
cpp_path = os.path.dirname(silearn.__file__)+ "/../../libsilearn/lib/"
dirs = os.listdir(cpp_path)
import ctypes
try:
    libSI = None
    import platform
    if platform.system() == 'Linux':
        exec=".so"
    else:
        exec=".dll"
    for x in dirs:
        if x.endswith(exec):
            libSI=ctypes.CDLL(cpp_path+x)
    native_si = libSI.si
    # native_si = libSI.si_hyper
except Exception as e:
    # pass
    print(e)
    raise Exception("SI lib is not correctly compiled")


class OperatorMerge(op.Operator):

    def perform(self, erase_loop=True, adj_cover=None, target_cluster = 0, m_scale =0):


        edges, trans_prob = self.enc.graph.edges
        edge_s = edges[:, 0].cpu().numpy().astype(np.uint32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint32))
        edge_t = edges[:, 1].cpu().numpy().astype(np.uint32).ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint32))
        edge_w = trans_prob.cpu().numpy().astype(np.double).ctypes.data_as(
            ctypes.POINTER(ctypes.c_double))


        # assert self.graph.edge_index[0].shape[0] == self.graph.num_edges and \
        #        self.graph.edge_index[1].shape[0] == self.graph.num_edges and \
        #        self.graph.edge_weight.shape[0] == self.graph.num_edges
        #
        # assert self.graph.edge_index[0].max() < self.node_cnt and self.graph.edge_index[1].max() < self.node_cnt
        # assert self.graph.edge_index[0].min() >= 0 and self.graph.edge_index[1].min() >= 0

        node_cnt = self.enc.graph.num_vertices
        edge_cnt = edges.shape[0]
        result = np.zeros(node_cnt, np.uint32)
        result_pt = result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

        if erase_loop:
            if target_cluster > 0:
                if adj_cover != None:
                    adj_cover = (adj_cover > 0).cpu().numpy().astype(np.uint32).ctypes.data_as(
                        ctypes.POINTER(ctypes.c_uint32))
                    libSI.si_tgt_adj(node_cnt, edge_cnt, edge_s, edge_t, edge_w, result_pt,
                                     ctypes.c_double(m_scale), adj_cover)
                else:
                    libSI.si_tgt(node_cnt, edge_cnt, edge_s, edge_t, edge_w, result_pt,
                                 ctypes.c_double(m_scale))


            else:
                libSI.si(node_cnt, edge_cnt, edge_s, edge_t, edge_w, result_pt)
        elif adj_cover is not None:
            adj_cover = (adj_cover > 0).cpu().numpy().astype(np.uint32).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint32))
            libSI.si_hyper_adj(node_cnt, edge_cnt, edge_s, edge_t, edge_w, adj_cover, result_pt)
        else:
            libSI.si_hyper(node_cnt, edge_cnt, edge_s, edge_t, edge_w, result_pt)

        self.enc.node_id = torch.LongTensor(result.astype(np.int64)).to(self.enc.graph.device)
