from inspect import isfunction

from silearn.graph import Graph

# (f, backend) --> f
__function_map__ = dict()


def get_dat_backend(dat):
    if isinstance(dat, torch.Tensor):
        return "torch"

def vertex_reduce(edges, partition, edges_fea, node_fea):
    backend = get_dat_backend(edges)
    if not __function_map__.__contains__((vertex_reduce, backend)):
        raise NotImplementedError(f"vertex_reduce() is not available for backend {backend}")
    return __function_map__[vertex_reduce, backend](edges, partition, edges_fea, node_fea)



def scatter_sum(src, idx, clip_length=0):
    backend = get_dat_backend(src)
    if not __function_map__.__contains__((scatter_sum, backend)):
        raise NotImplementedError(f"scatter_sum() is not available for backend {backend}")
    return __function_map__[scatter_sum, backend](src, idx, clip_length)


def scatter_cnt(idx, clip_length=0):
    backend = get_dat_backend(idx)
    if not __function_map__.__contains__((scatter_cnt, backend)):
        raise NotImplementedError(f"scatter_cnt() is not available for backend {backend}")
    return __function_map__[scatter_cnt, backend](idx, clip_length)

def scatter_max(src, idx):
    backend = get_dat_backend(src)
    if not __function_map__.__contains__((scatter_max, backend)):
        raise NotImplementedError(f"scatter_max() is not available for backend {backend}")
    return __function_map__[scatter_max, backend](src, idx)


def sumup_duplicates(edges, *weights, operation_ptrs = None):
    backend = get_dat_backend(edges)
    if not __function_map__.__contains__((sumup_duplicates, backend)):
        raise NotImplementedError(f"sumup_duplicates() is not available for backend {backend}")
    return __function_map__[sumup_duplicates, backend](edges, *weights,  operation_ptrs = operation_ptrs)



def uncertainty(p): # log p
    backend = get_dat_backend(p)
    if not __function_map__.__contains__((uncertainty, backend)):
        raise NotImplementedError(f"uncertainty() is not available for backend {backend}")
    return __function_map__[uncertainty, backend](p)



def entropy(p, q): # p log q
    backend = get_dat_backend(p)
    if not __function_map__.__contains__((entropy, backend)):
        raise NotImplementedError(f"entropy() is not available for backend {backend}")
    return __function_map__[entropy, backend](p, q)

def log2(x):
    backend = get_dat_backend(x)
    if not __function_map__.__contains__((log2, backend)):
        raise NotImplementedError(f"log2() is not available for backend {backend}")
    return __function_map__[log2, backend](x)

def log_e(x):
    backend = get_dat_backend(x)
    if not __function_map__.__contains__((log_e, backend)):
        raise NotImplementedError(f"log_e() is not available for backend {backend}")
    return __function_map__[log_e, backend](x)
def logical_and(x):
    backend = get_dat_backend(x)
    if not __function_map__.__contains__((logical_and, backend)):
        raise NotImplementedError(f"logical_and() is not available for backend {backend}")
    return __function_map__[logical_and, backend](x)

def logical_or(x):
    backend = get_dat_backend(x)
    if not __function_map__.__contains__((logical_or, backend)):
        raise NotImplementedError(f"logical_or() is not available for backend {backend}")
    return __function_map__[logical_or, backend](x)


def logical_not(x):
    backend = get_dat_backend(x)
    if not __function_map__.__contains__((logical_not, backend)):
        raise NotImplementedError(f"logical_not() is not available for backend {backend}")
    return __function_map__[logical_not, backend](x)

def nonzero(edges, return_weights  = False):
    backend = get_dat_backend(edges)
    if not __function_map__.__contains__((nonzero, backend)):
        raise NotImplementedError(f"nonzero() is not available for backend {backend}")
    return __function_map__[nonzero, backend](edges, return_weights)

def concat(list_of_tensors, dim = -1):
    backend = get_dat_backend(list_of_tensors[0])
    if not __function_map__.__contains__((concat, backend)):
        raise NotImplementedError(f"concat() is not available for backend {backend}")
    return __function_map__[concat, backend](list_of_tensors, dim)


def stack(list_of_tensors, dim = -1):
    backend = get_dat_backend(list_of_tensors[0])
    if not __function_map__.__contains__((stack, backend)):
        raise NotImplementedError(f"stack() is not available for backend {backend}")
    return __function_map__[stack, backend](list_of_tensors, dim)

def clone(t):
    backend = get_dat_backend(t)
    if not __function_map__.__contains__((clone, backend)):
        raise NotImplementedError(f"clone() is not available for backend {backend}")
    return __function_map__[clone, backend](t)



# TODO
def convert_backend(p, backend):
    """
    @:param
        backends ∈ {"torch", "numpy"}
    """
    backend_orig = get_dat_backend(p)
    if not __function_map__.__contains__((convert_backend, backend_orig)):
        raise NotImplementedError(f"convert_backend() is not available for original backend {backend_orig}")
    return __function_map__[convert_backend, backend_orig](p, backend)


def full_coo_graph(N, device, backend):
    """
    Get the edge vectors of full graph.

    @:param
    N: num of node.
    dev: torch device.
    @:returns es, et, w: torch tensor. Same as @get_sparse_conv
    """
    if not __function_map__.__contains__((full_coo_graph, backend)):
        raise NotImplementedError(f"full_graph() is not available for backend {backend}")
    return __function_map__[full_coo_graph, backend](N, device, backend)

def spatial_knn_graph(feature_map,  k, r, metric = None):
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
    """
    backend = get_dat_backend(feature_map)
    if not __function_map__.__contains__((spatial_knn_graph, backend)):
        raise NotImplementedError(f"spatial_knn_graph() is not available for backend {backend}")
    return __function_map__[spatial_knn_graph, backend](feature_map,  k, r, metric)





# def convert_backend(g: Graph, backend):
#     if not __function_map__.__contains__((convert_backend, g.backend)):
#         raise NotImplementedError(f"convert_backend() is not available for backend {g.backend}")
#     return __function_map__[convert_backend, g.backend](g, backend)



# noinspection PyUnresolvedReferences

__all__ = ["vertex_reduce",
           "convert_backend",
           "scatter_sum",
           "scatter_cnt",
           "scatter_max",
           "entropy",
           "uncertainty",
           "logical_and",
           "logical_or",
           "logical_not",
           "log2",
           "log_e",
           "concat",
           "stack",
           "clone",
           "convert_backend",
           "get_dat_backend",

           "sumup_duplicates",
           "full_coo_graph",
           "spatial_knn_graph"]

# from .functional import *

def __include_functions__(lib, name):
    for k, v in lib.__dict__.items():
        try:
            if isfunction(v) and isfunction(eval(k)):
                __function_map__[eval(k), name] = v
        except:
            pass


try:
    import torch
    import silearn.backends.torch_ops as torch_ops
    __include_functions__(torch_ops, "torch")

    import silearn.backends.scipy_ops as scipy_ops
    __include_functions__(scipy_ops, "numpy")
finally:
    pass

