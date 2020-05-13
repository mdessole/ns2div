import numpy 
from scipy.sparse import coo_matrix, csr_matrix
import pycuda.driver as cuda
from pycuda.gpuarray import to_gpu
from ctypes import c_ulonglong as c_ptr

#from pycuda_auxil import *
from ns_auxil import *
from pycuda_auxil import *

def pri_SR_by_key(ns, keys, vals, tp = 'sparse'):# (ns, chiave, valore):
    n = int(keys.size)

    
    in_keys_d = to_gpu(numpy.array(keys, dtype = ns.type_uint))
    in_vals_d = to_gpu(numpy.array(vals, dtype = ns.type_fd))
    out_keys_d = to_gpu(numpy.zeros(n,dtype = ns.type_uint))
    out_vals_d = to_gpu(numpy.zeros(n,dtype=ns.type_fd))
    ns.sort_by_key(c_ptr(in_keys_d.ptr),c_ptr(in_vals_d.ptr),n)

    ns.reduce_by_key(c_ptr(in_keys_d.ptr),c_ptr(in_vals_d.ptr),c_ptr(out_keys_d.ptr),c_ptr(out_vals_d.ptr),n)

    reduced_size = out_keys_d.get().nonzero()[0][-1]+1 
    offset =  out_keys_d.get()[0]==-1 
    out_keys , out_vals = out_keys_d.get(), out_vals_d.get()


    in_keys_d.gpudata.free()
    in_vals_d.gpudata.free()
    out_keys_d.gpudata.free()
    out_vals_d.gpudata.free()

    if tp == 'sparse':
        indices = numpy.nonzero(out_vals[offset:reduced_size]) # prendo indici elementi non zero
        out_vals = out_vals[offset:reduced_size][indices] # prendo solo quelli diversi da zero
        out_keys = out_keys[offset:reduced_size][indices]
    elif tp == 'dense':
        out_vals = out_vals[offset:reduced_size]
        out_keys = out_keys[offset:reduced_size]

    return out_keys, out_vals


def create_ij(ns, index, col):

        tot = index.shape[0]
        
        d_I = to_gpu( numpy.zeros( tot, dtype = ns.type_uint) )
        d_J = to_gpu( numpy.zeros( tot, dtype = ns.type_uint) )
        d_index = to_gpu( numpy.array( index, dtype = ns.type_uint) )
        
        ns.indice_kernel(d_index, d_I, d_J, #cuda.InOut(I), cuda.InOut(J),
                          ns.type_int(col), ns.type_uint(tot),
                          block = ns.block,
                          grid = ( int(numpy.ceil(tot/ns.th)),1) )
        
        I, J = d_I.get(), d_J.get()
        free([d_I, d_J, d_index])
        
        return I, J

def assembly(ns, mesh, scheme):
        
        ns.n  = mesh.get_n()
        ns.nn = mesh.get_nn()
        ns.nt = mesh.get_nt()
        ns.nbseg_x = mesh.get_nbseg_x()
        ns.nbseg_y = mesh.get_nbseg_y()
        ns.grid     = ( int(numpy.ceil(ns.nt/ns.th)),1)
        ns.grid_t   = ( int(numpy.ceil(ns.n/ns.th)),1)

           
        if (ns.renum == 1):
            M_P1_val = numpy.zeros(3 * 3 * ns.nt, dtype = ns.type_fd)
            M_P2_val = numpy.zeros(6 * 6 * ns.nt, dtype = ns.type_fd)
            M_P1_ind = numpy.zeros(3 * 3 * ns.nt, dtype = ns.type_uint)
            M_P2_ind = numpy.zeros(6 * 6 * ns.nt, dtype = ns.type_uint)

            tabella = numpy.array(mesh.get_tt(), dtype = numpy.int32 ).flatten()
            
            ns.assembly_mass_P2P1_kernel(cuda.In( numpy.array(mesh.xx, dtype = ns.type_fd) ), cuda.In(  numpy.array(mesh.yy, dtype = ns.type_fd) ), cuda.In(tabella),
                                      numpy.int32(ns.nt),numpy.int32(ns.n), ns.type_uint(ns.nn),
                                      cuda.Out(M_P1_val), cuda.Out(M_P1_ind),cuda.Out(M_P2_val), cuda.Out(M_P2_ind),
                                      block = ns.block, grid = ns.grid)
            
            r_M_P1_ind, r_M_P1_val   = pri_SR_by_key(ns, M_P1_ind, M_P1_val)
            I_M_P1, J_M_P1 = create_ij(ns, r_M_P1_ind, ns.nn)
            #ns.M_P1 = coo_matrix((r_M_P1_val, (I_M_P1, J_M_P1)), shape=(ns.n, ns.n), dtype = ns.type_fd).tocsr()
            
            r_M_P2_ind, r_M_P2_val   = pri_SR_by_key(ns, M_P2_ind, M_P2_val)
            I_M_P2, J_M_P2 = create_ij(ns, r_M_P2_ind, ns.nn)
            ns.M_P2 = coo_matrix((r_M_P2_val, (I_M_P2, J_M_P2)), shape=(ns.nn, ns.nn), dtype = ns.type_fd).tocsr()

            #ns.perm_P1, ns.perm_P1_inv = ns.symrcm(ns.M_P1, ns.n)
            ns.perm_P1 = numpy.arange(ns.n)
            ns.perm_P1_inv = numpy.arange(ns.n)
            ns.perm_P2, ns.perm_P2_inv = ns.symrcm(ns.M_P2, ns.nn)
            
            ns.tab_connectivity_P1 = mesh.get_t()
            ns.tab_connectivity = mesh.get_tt()
            for i in range(ns.nt):
                ns.tab_connectivity[i,:] = ns.perm_P2_inv[ns.tab_connectivity[i,:]]
                ns.tab_connectivity_P1[i,:] = ns.perm_P1_inv[ns.tab_connectivity_P1[i,:]]
            #endif

            ns.x  = mesh.get_x()
            ns.y  = mesh.get_y()
            ns.xx  = mesh.get_xx()
            ns.yy  = mesh.get_yy()
            
            ns.x = ns.x.copy()[ns.perm_P1]
            ns.y = ns.y.copy()[ns.perm_P1]
            ns.xx = ns.xx.copy()[ns.perm_P2]
            ns.yy = ns.yy.copy()[ns.perm_P2]
            
            ns.volume = mesh.get_volume()
            ns.volume = ns.volume.copy()[ns.perm_P1]
            ns.volume_barycenter = mesh.get_volume_barycenter()
            ns.volume_barycenter[0,:] = ns.volume_barycenter.copy()[0,ns.perm_P1]
            ns.volume_barycenter[1,:] = ns.volume_barycenter.copy()[1, ns.perm_P1]
            ns.horizontal_edges = mesh.get_horizontal_edges()
            #ns.horizontal_edges[0, ns.perm_P1] = ns.horizontal_edges.copy()[0, :]
            #ns.horizontal_edges[1, ns.perm_P1] = ns.horizontal_edges.copy()[1, :]
            ns.vertical_edges = mesh.get_vertical_edges()
            #ns.vertical_edges[0, ns.perm_P1]   = ns.vertical_edges.copy()[0, :]
            #ns.vertical_edges[1, ns.perm_P1]   = ns.vertical_edges.copy()[1, :]
            
        elif (ns.renum == 0):
            ns.tab_connectivity_P1 = mesh.get_t()
            ns.tab_connectivity = mesh.get_tt()
            ns.x  = mesh.get_x()
            ns.y  = mesh.get_y()
            ns.xx = mesh.get_xx()
            ns.yy = mesh.get_yy()

            ns.volume = mesh.get_volume()
            ns.volume_barycenter = mesh.get_volume_barycenter()
            ns.horizontal_edges = mesh.get_horizontal_edges()
            ns.vertical_edges = mesh.get_vertical_edges()
        #endif
        ns.x1 = numpy.amin(ns.xx)
        ns.x2 = numpy.amax(ns.xx)
        ns.y1 = numpy.amin(ns.yy)
        ns.y2 = numpy.amax(ns.yy)
        
        #finire di cambiare i d_... anche nel file init....
        ns.d_tab_connectivity_P1 = to_gpu( numpy.array( ns.tab_connectivity_P1, dtype = numpy.int32 ).flatten() )
        ns.d_tab_connectivity = to_gpu( numpy.array( ns.tab_connectivity, dtype = numpy.int32 ).flatten() )
        ns.d_x  = to_gpu( ns.x.astype(ns.type_fd) )
        ns.d_y  = to_gpu( ns.y.astype(ns.type_fd) )
        ns.d_xx = to_gpu( ns.xx.astype(ns.type_fd) )
        ns.d_yy = to_gpu( ns.yy.astype(ns.type_fd) )

        #Definisco le condizioni a contorno
        ns.N_diri1, ns.N_diri2, ns.N_free1, ns.N_free2 = identify_BC_velocity(ns, mesh)
        if not( ns.case == 1 or ns.case == 0):
            ns.N_dirirho = identify_BC_density(ns, mesh)
        #endif
        ns.rhs_bordo_init(mesh)

        #definisco le griglie per i kernel
        ns.nuf1 = ns.N_free1.shape[0]
        ns.nuf2 = ns.N_free2.shape[0]
        ns.npf  = ns.NPfree.shape[0]

        ns.grid_stk_dir   = (int(numpy.ceil( (ns.nn + ns.nn + ns.n)/ns.th) ) , 1)
        ns.grid_stk_dir_2 = (int(numpy.ceil(ns.n / ns.th)) , 1)
        ns.grid_stk_dir_3 = (int(numpy.ceil( (ns.nuf1 + ns.nuf2 + ns.npf) / ns.th) ) , 1)

        ns.grid_stk_proj   = (int(numpy.ceil( (ns.nn)/ns.th) ) , 1)
        ns.grid_stk_proj_2 = (int(numpy.ceil(ns.n / ns.th)) , 1)
        
        ns.grid_LU1 = ( int(numpy.ceil( ns.nuf1/ns.th)),1 )
        ns.grid_LU2 = ( int(numpy.ceil( ns.nuf2/ns.th)),1 )
            
        # creiamo i vari vettori che serviranno perla
        # memorizzazione delle matrici in formati coo
        # con un solo vettore per gli indici
        A_val = numpy.zeros(6 * 6 * ns.nt, dtype = ns.type_fd)
        B1_val = numpy.zeros(3 * 6 * ns.nt, dtype = ns.type_fd)
        B2_val = numpy.zeros(3 * 6 * ns.nt, dtype = ns.type_fd)
        # vettori con gli indici
        MA_ind = numpy.zeros(6 * 6 * ns.nt, dtype = ns.type_uint)
        B_ind = numpy.zeros(3 * 6 * ns.nt, dtype = ns.type_uint)


        # ATTENZIONE ACCESSO DIRETTO A MEMBRI "PRIVATI" DI MESH
        ns.assembly_lapl_div_P2P1_kernel(ns.d_xx, ns.d_yy, ns.d_tab_connectivity, ns.d_tab_connectivity_P1,
                                      numpy.int32(ns.nt),numpy.int32(ns.n), ns.type_uint(ns.nn),
                                      cuda.Out(A_val), cuda.Out(MA_ind),
                                      cuda.Out(B1_val), cuda.Out(B2_val), cuda.Out(B_ind),
                                      block = ns.block, grid = ns.grid)

        
        # Ordiniamo e facciamo riduzione
        r_A_ind, r_A_val   = pri_SR_by_key(ns, MA_ind, A_val) # da considerare se tenere r_A_ind
        
        ns.I_P2, ns.J_P2 = create_ij(ns, r_A_ind, ns.nn)
        ns.A = coo_matrix((r_A_val, (ns.I_P2, ns.J_P2)), shape=(ns.nn, ns.nn), dtype = ns.type_fd).tocsr()
        
        r_B1_ind, r_B1_val = pri_SR_by_key(ns, B_ind, B1_val)
        r_B2_ind, r_B2_val = pri_SR_by_key(ns, B_ind, B2_val)

        I_B1, J_B1 = create_ij(ns, r_B1_ind, ns.nn)
        I_B2, J_B2 = create_ij(ns, r_B2_ind, ns.nn)        
        ns.B1 = coo_matrix((r_B1_val, (I_B1, J_B1)), shape=(ns.n, ns.nn), dtype = ns.type_fd).tocsr()
        ns.B2 = coo_matrix((r_B2_val, (I_B2, J_B2)), shape=(ns.n, ns.nn), dtype = ns.type_fd).tocsr()
            
        ns.B1_tra = ns.B1.T.tocsr() # transpose(copy = True)
        ns.B2_tra = ns.B2.T.tocsr() #transpose(copy = True)


        M_val = numpy.zeros(6 * 6 * ns.nt, dtype = ns.type_fd)
        # vettori con gli indici
        M_ind = numpy.zeros(6 * 6 * ns.nt, dtype = ns.type_uint)


        # ATTENZIONE ACCESSO DIRETTO A MEMBRI "PRIVATI" DI MESH
        ns.assembly_mass_P2_kernel(ns.d_xx, ns.d_yy, ns.d_tab_connectivity, 
                                numpy.int32(ns.nt), ns.type_uint(ns.nn),
                                cuda.Out(M_val), cuda.Out(M_ind),
                                block = ns.block, grid = ns.grid)
        
        # Ordiniamo e facciamo riduzione
        r_M_ind, r_M_val   = pri_SR_by_key(ns, M_ind, M_val) # da considerare se tenere r_A_ind

        I_M, J_M = create_ij(ns, r_M_ind, ns.nn)
        ns.M_P2 = coo_matrix((r_M_val, (I_M, J_M)), shape=(ns.nn, ns.nn), dtype = ns.type_fd).tocsr()

        r_B1_ind, r_B1_val = pri_SR_by_key(ns, B_ind, B1_val)
        r_B2_ind, r_B2_val = pri_SR_by_key(ns, B_ind, B2_val)

        I_B1, J_B1 = create_ij(ns, r_B1_ind, ns.nn)
        I_B2, J_B2 = create_ij(ns, r_B2_ind, ns.nn)        
        ns.B1 = coo_matrix((r_B1_val, (I_B1, J_B1)), shape=(ns.n, ns.nn), dtype = ns.type_fd).tocsr()
        ns.B2 = coo_matrix((r_B2_val, (I_B2, J_B2)), shape=(ns.n, ns.nn), dtype = ns.type_fd).tocsr()
            
        ns.B1_tra = ns.B1.T.tocsr() # transpose(copy = True)
        ns.B2_tra = ns.B2.T.tocsr() #transpose(copy = True)

        if (scheme == 'proj2'):
            MG_val = numpy.zeros(3 * 3 * ns.nt, dtype = ns.type_fd)
            LAP_val = numpy.zeros(3 * 3 * ns.nt, dtype = ns.type_fd)
            MG_ind = numpy.zeros(3 * 3 * ns.nt, dtype = ns.type_uint)
            LAP_ind = numpy.zeros(3 * 3 * ns.nt, dtype = ns.type_uint)
            
            ns.assembly_mass_P1_kernel(ns.d_x, ns.d_y, ns.d_tab_connectivity_P1,
                                    numpy.int32(ns.nt),numpy.int32(ns.n), ns.type_uint(ns.nn),
                                    cuda.Out(MG_val), cuda.Out(MG_ind),
                                    block = ns.block, grid = ns.grid)
            
            r_MG_ind, r_MG_val   = pri_SR_by_key(ns, MG_ind, MG_val)
            ns.I_P1, ns.J_P1 = create_ij(ns, r_MG_ind, ns.nn)
            ns.MG = coo_matrix((r_MG_val, (ns.I_P1, ns.J_P1)), shape=(ns.n, ns.n), dtype = ns.type_fd).tocsr()
            
            ns.assembly_lapl_P1_kernel(ns.d_x, ns.d_y, ns.d_tab_connectivity_P1,
                                    numpy.int32(ns.nt),numpy.int32(ns.n), ns.type_uint(ns.nn),
                                    cuda.Out(LAP_val), cuda.Out(LAP_ind),
                                    block = ns.block, grid = ns.grid)

            r_LAP_ind, r_LAP_val   = pri_SR_by_key(ns, LAP_ind, LAP_val)
            I_LAP, J_LAP = create_ij(ns, r_LAP_ind, ns.nn)
            ns.LAP = coo_matrix((r_LAP_val, (I_LAP, J_LAP)), shape=(ns.n, ns.n), dtype = ns.type_fd).tocsr()
            ns.LAP = ns.LAP[ns.NPfree, :][:, ns.NPfree]            

            # mando a GPU
            #ns.d_B1_tra_data, ns.d_B1_tra_indptr, ns.d_B1_tra_indices = csr_to_gpu(ns, ns.B1_tra)
            #ns.d_B2_tra_data, ns.d_B2_tra_indptr, ns.d_B2_tra_indices = csr_to_gpu(ns, ns.B2_tra)
            ns.d_LAP_data, ns.d_LAP_indptr, ns.d_LAP_indices          = csr_to_gpu(ns, ns.LAP)
            ns.d_MG_data, ns.d_MG_indptr, ns.d_MG_indices             = csr_to_gpu(ns, ns.MG)
        #endif
        
        # mando a GPU
        ns.d_B1_data, ns.d_B1_indptr, ns.d_B1_indices             = csr_to_gpu(ns, ns.B1)
        ns.d_B2_data, ns.d_B2_indptr, ns.d_B2_indices             = csr_to_gpu(ns, ns.B2)
        ns.d_B1_tra_data, ns.d_B1_tra_indptr, ns.d_B1_tra_indices = csr_to_gpu(ns, ns.B1_tra)
        ns.d_B2_tra_data, ns.d_B2_tra_indptr, ns.d_B2_tra_indices = csr_to_gpu(ns, ns.B2_tra)


        
        #endif
        return        