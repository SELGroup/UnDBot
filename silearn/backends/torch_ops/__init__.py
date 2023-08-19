import importlib.util

if importlib.util.find_spec("torch") is None:
    torch_available = False
else:
    torch_available = True
from .graph_ops import *
from .matrix_ops import *
