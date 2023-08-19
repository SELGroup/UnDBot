import numpy as np
import torch
import torch_scatter


def scatter_sum(src: torch.Tensor, idx: torch.LongTensor, clip_length=0):
    r"""
    Reduce sum all values from the :obj:`src` tensor at the indices specified in the :obj:`idx` tensor along axis
    :obj:`dim=0`.

    Args:
        src(torch.Tensor): The source tensor
        idx(torch.LongTensor): the indices of elements to scatter
        clip_length(int): if :obj:`clip_length` is not given, a minimal sized output tensor according to :obj:`idx.max()+1` is returned

    :rtype: :class:`Tensor`
    """
    sz = torch.max(idx) + 1 if clip_length == 0 else clip_length
    # assert b.shape[0] >= sz
    if len(src.shape) == 1:
        return torch.zeros(sz, dtype=src.dtype,
                           device=src.device).scatter_add_(0, idx, src)
    else:
        sp = list(src.shape)
        sp[0] = sz
        return torch.zeros(sp, dtype=src.dtype,
                           device=src.device).scatter_add_(
                               0,
                               idx.unsqueeze(-1).expand_as(src), src)


def scatter_cnt(idx: torch.LongTensor,clip_length=0, dtype=torch.float):
    r"""
    Count the occurrence of each element in the :obj:`idx`.

    Args:
        idx(torch.LongTensor): the indices of elements to scatter
        dtype(torch.dtype): specify the type of returned tensor
        clip_length(int): if :obj:`clip_length` is not given, a minimal sized output tensor according to :obj:`idx.max()+1` is returned

    :rtype: :class:`Tensor`
    """
    sz = torch.max(idx) + 1 if clip_length == 0 else clip_length
    return torch.zeros(sz, dtype=dtype, device=idx.device).scatter_add_(
        0, idx, torch.ones(idx.shape[0], dtype=dtype, device=idx.device))


eps_dtype = {
    torch.float64: 1e-306,
    torch.double: 1e-306,
    torch.float32: 1e-36,
    torch.bfloat16: 1e-36,
    torch.float16: 1e-7,
}


# p * log_2 q
def entropy(p: torch.Tensor, q: torch.Tensor):
    dtype = p.dtype
    eps = eps_dtype[dtype] if eps_dtype.keys().__contains__(dtype) else 1e-36
    return -p * torch.log2(torch.clip(q, min=eps))


def uncertainty(q: torch.Tensor):
    dtype = q.dtype
    eps = eps_dtype[dtype] if eps_dtype.keys().__contains__(dtype) else 1e-36
    return -torch.log2(torch.clip(q, min=eps))


scatter_max = torch_scatter.scatter_max
logical_or = torch.logical_or
logical_and = torch.logical_or
logical_not = torch.logical_not
log2 = torch.log2
log_e = torch.log
stack = torch.stack
clone = torch.clone


def concat(q, dim=-1):
    return torch.cat(q, dim)


import torch.utils.dlpack

cupy_available = False
try:
    import cupy
    cupy_available = True
except:
    pass


def convert_backend(p: torch.Tensor, backend: str):
    if backend == "numpy":
        # force
        return p.cpu().detach().numpy()
    elif backend == "cupy":
        if not cupy_available:
            raise Exception("Target backend cupy is not available")
        # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
        return cupy.asarray(p)
    elif backend == "dlpack":
        # noinspection PyUnresolvedReferences
        return torch.utils.dlpack.to_dlpack(p)
    else:
        raise NotImplementedError(
            f"convert_backend is not implemented for (torch.Tensor, {str(backend)})"
        )
