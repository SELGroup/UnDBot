
from silearn.model.encoding_tree import GraphEncoding


class Operator():
    """
    Map the encoding tree node to intermediate values in computing
    """
    cache_Vol: None
    cache_G: None
    cache_Eij: None


    def __init__(self, enc: GraphEncoding):
        self.enc = enc

    def perform(self, re_compute=True):
        """
        Perform Optimization on Graph
        @:returns the variance of structural information
        """
        pass



