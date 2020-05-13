import pycuda.driver as cuda
from pycuda.gpuarray import to_gpu

def gpu_set_val( d_v, ind, val):
    v = d_v.get()
    d_v.gpudata.free()
    v[ind] = val
    d_new_v = to_gpu(v)
    return d_new_v


def free( device_data):
    for a in device_data:
        a.gpudata.free()
    return

def csr_to_gpu(ns, mat): 
    d_data = to_gpu(mat.data.astype(ns.type_fd))
    d_indptr = to_gpu(mat.indptr.astype(ns.type_int))
    d_indices = to_gpu(mat.indices.astype(ns.type_int))

    return d_data, d_indptr, d_indices


