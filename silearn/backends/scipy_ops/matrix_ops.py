import numpy as np
import scipy

from numpy.lib.shape_base import _make_along_axis_idx
import functools
from numpy.core import overrides
from numpy.core.multiarray import normalize_axis_index
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

def _add_along_axis_dispatcher(arr, indices, values, axis):
    return (arr, indices, values)

@array_function_dispatch(_add_along_axis_dispatcher)
def add_along_axis(arr, indices, values, axis):
    """
    numpy scatter_add operation

    @param:
    arr: target result array
    indices: indices to change along each 1d slice of arr.
    values: values to insert at those indices. Its shape and dimension are broadcast to match that of indices.
    axis: the axis to take 1d slices along.
    """
    # normalize inputs
    if axis is None:
        arr = arr.flat
        axis = 0
        arr_shape = (len(arr),)  # flatiter has no .shape
    else:
        axis = normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape
    # use the fancy index
    np.add.at(arr, _make_along_axis_idx(arr_shape, indices, axis), values)

def scatter_sum(src : np.ndarray, idx : np.ndarray, clip_length = 0):
    """
    @comment(xiang): only support cpu?
    """
    sz = np.max(idx) + 1 if clip_length == 0 else clip_length
    # arr = scipy.sparse.coo()
    # assert b.shape[0] >= sz
    if len(src.shape) == 1:
        result = np.zeros(sz, dtype=src.dtype)
        # return np.zeros(sz, dtype=b.dtype, device=b.device).scatter_add_(0, a, b)
        add_along_axis(result, idx, src, axis=0)
    else:
        sp = list(src.shape)
        sp[0] = sz
        # return torch.zeros(sp, dtype=b.dtype, device=b.device).scatter_add_(0, a.unsqueeze(-1).expand_as(b), b)
        result = np.zeros(sp, dtype=src.dtype)
        add_along_axis(result, np.broadcast_to(np.expand_dims(idx, -1), src.shape), src, axis=0)
    return result

def scatter_cnt(idx : np.ndarray, dtype = np.float, clip_length = 0):
    sz = np.max(idx) + 1 if clip_length == 0 else clip_length
    # return torch.zeros(sz, dtype = dtype, device=a.device).scatter_add_(0, a, torch.ones(a.shape[0], dtype = dtype, device=a.device))
    result = np.zeros(sz, dtype=idx.dtype)
    add_along_axis(result, idx, np.ones(idx.shape[0], dtype=dtype), axis=0)
    return result

def scatter_max(src : np.ndarray, idx : np.ndarray, clip_length = 0):
    raise NotImplementedError('Not Implemented')

eps_dtype = {
    np.float64: 1e-306,
    np.double: 1e-306,
    np.float32: 1e-36,
    np.float16: 1e-7,
}


# p * log_2 q
def entropy(p: np.ndarray, q:np.ndarray):
    dtype = p.dtype
    eps = eps_dtype[dtype] if eps_dtype.keys().__contains__(dtype) else 1e-36
    return -p * np.log2(np.clip(q, a_min = eps, a_max=None))


def uncertainty(q: np.ndarray):
    dtype = q.dtype
    eps = eps_dtype[dtype] if eps_dtype.keys().__contains__(dtype) else 1e-36
    return -np.log2(np.clip(q, a_min = eps, a_max=None))


# scatter_max = torch_scatter.scatter_max


logical_or = np.logical_or
logical_and = np.logical_or
logical_not = np.logical_not
log2 = np.log2
log_e = np.log

def concat(q, dim = -1):
    return torch.cat(q, dim)


import torch.utils.dlpack

try:
    import cupy
except:
    pass

def convert_backend(p : np.ndarray, backend: str):
    if backend == "numpy":
        # force
        return p.numpy(force = True)
    elif backend == "cupy":
        # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
        return cupy.asarray(p)
    elif backend == "dlpack":
        # noinspection PyUnresolvedReferences
        return torch.utils.dlpack.to_dlpack(p)
    else:
        raise NotImplementedError(f"convert_backend is not implemented for (np.ndarray, {str(backend)})")

