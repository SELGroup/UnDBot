import silearn.backends

def from_networkx():

    pass

def graph_from_torch_sparse():
    pass

def enc_from_torch_sparse():
    pass

def from_cugraph():
    pass

def to_cugraph():
    pass


def create_cugraph(es, et, w):
    import cudf, cupy, cugraph
    df = cudf.DataFrame()
    df["es"]=cupy.array(es)
    df["et"]=cupy.array(et)
    df["w"]=cupy.array(w)
    return cugraph.from_cudf_edgelist(df, "es", "et", "w", create_using=cugraph.MultiGraph, renumber=False)

def create_cupy_partitioning(com):
    import cudf, cupy, cugraph
    df = cudf.DataFrame()
    df["vertex"] = cupy.arange(com.shape[0])
    df["cluster"] = cupy.array(com)
    return df