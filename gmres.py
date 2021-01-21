from ctypes import c_ulonglong as c_ptr
import pycuda
import math
from pycuda_auxil import *
from scipy.sparse import csr_matrix
import numpy

def save_iteration_data(ns, i, iters, time, sys, axis):
        if (sys == 1):
            ns.iteration_nb1[i] = iters
            ns.iteration_exectime1[i] += time #elapsed_time
        elif (sys == 2):
            ns.iteration_nb2[i] = iters
            ns.iteration_exectime2[i] += time #elapsed_time
            #endif
        #endif
        return

def check_iteration(ns, it, i):
        #check if gmres reached maximum number of iterations without converging
        if (it >= 3000):
            ns.max_it_ns = i
            ns.GMRES_FAIL = 1


def buildcsrMatrix_cutDAG_GPU(ns, n, nnz, d_indptr, d_indices, axis = 1):

        d_nnz = to_gpu( numpy.array([nnz], dtype = ns.type_int) )
        if (axis == 1):
            ns.count_nnz_cutM(c_ptr(ns.d_LU1_data.ptr), c_ptr(d_indptr.ptr), c_ptr(d_indices.ptr),
                                int(ns.ncutL1), c_ptr(ns.d_cutL1.ptr), int(ns.ncutU1), c_ptr(ns.d_cutU1.ptr),  c_ptr(d_nnz.ptr))
            ns.nnzM1 = d_nnz.get()
            ns.d_M1_indptr = pycuda.gpuarray.zeros((n+1), ns.type_int)
            ns.d_M1_indices = pycuda.gpuarray.zeros(ns.nnzM1, ns.type_int)
            ns.d_M1_data = pycuda.gpuarray.zeros(ns.nnzM1, ns.type_fd)
            
            ns.cutColIndM(int(n), int(nnz), c_ptr(ns.d_LU1_data.ptr), c_ptr(d_indptr.ptr), c_ptr(d_indices.ptr), c_ptr(ns.d_M1_indptr.ptr), c_ptr(ns.d_M1_indices.ptr),
                            int(ns.ncutL1), c_ptr(ns.d_cutL1.ptr), int(ns.ncutU1), c_ptr(ns.d_cutU1.ptr))
        elif (axis == 2):
            ns.count_nnz_cutM(c_ptr(ns.d_LU2_data.ptr), c_ptr(d_indptr.ptr), c_ptr(d_indices.ptr),
                                int(ns.ncutL2), c_ptr(ns.d_cutL2.ptr), int(ns.ncutU2), c_ptr(ns.d_cutU2.ptr),  c_ptr(d_nnz.ptr))
            ns.nnzM2 = d_nnz.get()
            ns.d_M2_indptr = pycuda.gpuarray.zeros((n+1), ns.type_int)
            ns.d_M2_indices = pycuda.gpuarray.zeros(ns.nnzM2, ns.type_int)
            ns.d_M2_data = pycuda.gpuarray.zeros(ns.nnzM2, ns.type_fd)
            
            ns.cutColIndM(int(n), int(nnz), c_ptr(ns.d_LU2_data.ptr), c_ptr(d_indptr.ptr), c_ptr(d_indices.ptr), c_ptr(ns.d_M2_indptr.ptr), c_ptr(ns.d_M2_indices.ptr),
                            int(ns.ncutL2), c_ptr(ns.d_cutL2.ptr), int(ns.ncutU2), c_ptr(ns.d_cutU2.ptr))
        #endif
        free([d_nnz])
        return  
    
def def_cutDAG(ns, n, nnz, d_A_data, d_A_indptr, d_A_indices, axis = 1):

        if (ns.cut >= 1.0) or (ns.cut <= 0.0):
            print('Error in the choice of DAG percentage to delete. Setting cut = 1/2...')
            ns.cut = 1/2
        #endif
        
        d_levelInd = to_gpu( numpy.zeros(n, dtype = ns.type_int) )
        d_levelPtr = to_gpu( numpy.zeros(n, dtype = ns.type_int) )
        d_nlevels = to_gpu( numpy.array([2], dtype = ns.type_int) )
        
        ns.DAG_analysis(int(n), int(nnz), int(1), c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                          c_ptr(d_levelPtr.ptr), c_ptr(d_levelInd.ptr), c_ptr(d_nlevels.ptr))
        nlevels = d_nlevels.get()[0]
        levelInd = d_levelInd.get()
        levelPtr = d_levelPtr.get()
        ncut = int(min(math.ceil(nlevels*(1-ns.cut)), nlevels))
        cutL = numpy.array(numpy.sort(levelInd[levelPtr[nlevels-ncut]:levelPtr[nlevels]]), dtype = ns.type_int)
            
        ns.DAG_analysis(int(n), int(nnz), int(0), c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                          c_ptr(d_levelPtr.ptr), c_ptr(d_levelInd.ptr), c_ptr(d_nlevels.ptr))
        nlevels = d_nlevels.get()[0]
        levelInd = d_levelInd.get()
        levelPtr = d_levelPtr.get()
        ncut = int(min(math.ceil(nlevels*(1-ns.cut)), nlevels))
        cutL = numpy.array(numpy.sort(levelInd[levelPtr[nlevels-ncut]:levelPtr[nlevels]]), dtype = ns.type_int)
        cutU =  numpy.array(numpy.sort(levelInd[levelPtr[nlevels-ncut]:levelPtr[nlevels]]), dtype = ns.type_int)
            
        free([d_levelInd, d_nlevels, d_levelPtr])
        return cutL, cutU
            
def itaLU(ns, i, n, nnz, A, d_A_data, d_A_indptr, d_A_indices, axis = 1): # A, d_A_data, d_A_indptr, d_A_indices, 
        d_exectime = pycuda.gpuarray.zeros(1, ns.type_fd)
        d_diagU = pycuda.gpuarray.zeros(n, ns.type_fd)
            
        if (axis == 1):
            if (ns.init_italu1 == 0):
                #Initialize preconditioner
                
                if (ns.initialize_LU == 0):
                    #L,U = ILU(0)
                    if (ns.primo_italu1 == 1):
                        ns.primo_italu1 = 0
                    elif (ns.primo_italu1 == 0):
                        free([ns.d_LU1_data])
                    #endif
                    ns.d_LU1_data = d_A_data.copy()
                    ns.csrilu0(int(n), int(nnz),
                                 c_ptr(ns.d_LU1_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                                 c_ptr(d_exectime.ptr))

                elif (ns.initialize_LU == 1):
                    ns.d_LU1_data = d_A_data.copy() 
                #endif   
                if ((ns.cutDAG_LU == 1) and (ns.init_cutDAG_LU1 == 0)):
                    cutL, cutU = def_cutDAG(ns,n, nnz, ns.d_LU1_data, d_A_indptr, d_A_indices)
                    ns.d_cutL1 = to_gpu( numpy.array(cutL, dtype = ns.type_int) )
                    ns.ncutL1 = len(cutL)
                    ns.d_cutU1 = to_gpu( numpy.array(cutU, dtype = ns.type_int) )
                    ns.ncutU1 = len(cutU)
                    buildcsrMatrix_cutDAG_GPU(ns,n, nnz, d_A_indptr, d_A_indices, axis = 1)
                    ns.init_cutDAG_LU1 = 1
                #endif
                
                #set initialization FLAG
                ns.init_italu1 = 1
            else:
                #perform ITALU update
                ns.SITALU(int(ns.iters), int(n), int(nnz),
                                     c_ptr(ns.d_LU1_data.ptr), 
                                     c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                                     c_ptr(d_exectime.ptr))
            #endif         
            exectime = d_exectime.get()[0]
            ns.iteration_exectime1[i] = exectime 
        elif (axis == 2):
            if (ns.init_italu2 == 0):
                #initialize L,U
                if (ns.initialize_LU == 0):
                    #ILU0 init
                    if (ns.primo_italu2 == 1):
                        ns.primo_italu2 = 0
                    elif (ns.primo_italu2 == 0):
                        free([ns.d_LU2_data]) 
                    #endif
                    ns.d_LU2_data = d_A_data.copy() 
                    
                    ns.csrilu0(int(n), int(nnz), c_ptr(ns.d_LU2_data.ptr), 
                                 c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr), c_ptr(d_exectime.ptr))

                elif (ns.initialize_LU == 1):
                    ns.d_LU2_data = d_A_data.copy() 
                #endif
                
                if ((ns.cutDAG_LU == 1) and (ns.init_cutDAG_LU2 == 0)):
                    #define cutting level for L,U according to specified percentage
                    cutL, cutU = def_cutDAG(ns,n, nnz, ns.d_LU2_data , d_A_indptr, d_A_indices)
                    ns.d_cutL2 = to_gpu( numpy.array(cutL, dtype = ns.type_int) )
                    ns.ncutL2 = len(cutL)
                    ns.d_cutU2 = to_gpu( numpy.array(cutU, dtype = ns.type_int) )
                    ns.ncutU2 = len(cutU)
                    buildcsrMatrix_cutDAG_GPU(ns,n, nnz, d_A_indptr, d_A_indices, axis = 2)
                    ns.init_cutDAG_LU2 = 1
                #endif
                
                #set initialization FLAG
                ns.init_italu2 = 1
            else:  
                #perform ITALU update
                ns.SITALU(int(ns.iters), int(n), int(nnz),
                                 c_ptr(ns.d_LU2_data.ptr), 
                                 c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                                 c_ptr(d_exectime.ptr))
            #endif
            exectime = d_exectime.get()[0]
            ns.iteration_exectime2[i] = exectime
        #endif


        print('ITALU time = ', exectime, ' s')
        free([d_exectime, d_diagU])
        return
            

def block_diagonalize(ns, M, n):
        block_init = 0
        ii = 0

        while (ii < n):
            ii = block_init + 2
            while (ii < n) and (M.rows[ii].index(ii) >= 2): 
                ii = ii + 1
            block_end = min(ii,n-1) #-1

            for row in range(block_init+1, block_end):
                diag = M.rows[row].index(row)
                indices = range(len(M.rows[row]))       
                delete = {i for i in indices if ((M.rows[row][i] < block_init) or (M.rows[row][i] > block_end ))}
                M.data[row] = [M.data[row][i] for i in indices if i not in delete]
                M.rows[row] = [M.rows[row][i] for i in indices if i not in delete]
            #endfor

            block_init = block_end
        #endwhile
        
        return M

def csrMatrixBandwidth(ns, A, n, axis = 1):
        bandwidth = 0
        for i in range(n):
            for j in range(A.indptr[i], A.indptr[i+1]):
                if (abs(A.indices[j] - i) > bandwidth):
                    bandwidth = abs(A.indices[j] - i)
                #endif
            #endfor
        #endfor
        if (axis == 1):
            ns.bandwidth1 = bandwidth
        elif (axis == 2):
            ns.bandwidth2 = bandwidth
        #endif
        return
    
def matrix_blocking(ns, i, A, n, axis):
        if (axis == 1):
            M_data = ns.d_LU1_data.get()
        elif (axis == 2):
            M_data = ns.d_LU2_data.get()
        #endif

        M = csr_matrix((M_data, A.indices, A.indptr), shape = (n,n)).tolil()
        MM = M.copy()
        
        if (axis == 1) and (ns.init_blockprec1 == 0):
            csrMatrixBandwidth(ns, A, n, axis = 1)
            ns.bandwidth1 = int(math.ceil(ns.bandwidth1/8/2))
        elif (axis == 2) and (ns.init_blockprec2 == 0):
            csrMatrixBandwidth(ns, A, n, axis = 2)
            ns.bandwidth2 = int(math.ceil(ns.bandwidth2/8/2))

        if (axis == 1):
            bandwidth = ns.bandwidth1
        elif (axis == 2):
            bandwidth = ns.bandwidth2


        for row in range(n):
            for col in M.rows[row]:   
                if (abs(col - row) > bandwidth):
                    i = MM.rows[row].index(col)
                    del MM.rows[row][i]
                    del MM.data[row][i]
                #endif
            #endfor
        #endfor
        
        MM = block_diagonalize(ns, MM,  n)
        MM = csr_matrix(MM)
        
        
        if (axis == 1):
            if (ns.init_blockprec1 == 1):
                free([ns.d_M1_data, ns.d_M1_indptr, ns.d_M1_indices])
            else:
                ns.init_blockprec1 = 1
            ns.nnzM1 = MM.nnz
            ns.d_M1_data, ns.d_M1_indptr, ns.d_M1_indices = csr_to_gpu(ns, MM)
        elif (axis == 2):
            if (ns.init_blockprec2 == 1):
                free([ns.d_M2_data, ns.d_M2_indptr, ns.d_M2_indices])
            else:
                ns.init_blockprec2 = 1
            ns.nnzM2 = MM.nnz
            ns.d_M2_data, ns.d_M2_indptr, ns.d_M2_indices = csr_to_gpu(ns, MM)
        #endif
        
        return

def prec_cutDAG(ns, n, nnz, d_A_indptr, d_A_indices, axis):
        #this function processes the preconditioner according to parameters
        # in matrix U every element below ncutU is set zero
        # in matrix L every element below ncutL is set zero 
        # ncut refers to a certain level in DAG representing dependencies 
        # in the parallel solution of the corresponding triangular linear system
        if (axis == 1):
            ns.cutValM(int(n), int(nnz), c_ptr(ns.d_LU1_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                             int(ns.nnzM1), c_ptr(ns.d_M1_data.ptr) , c_ptr(ns.d_M1_indptr.ptr), c_ptr(ns.d_M1_indices.ptr),
                             int(ns.ncutL1), c_ptr(ns.d_cutL1.ptr), int(ns.ncutU1), c_ptr(ns.d_cutU1.ptr) )
        elif (axis == 2):
            ns.cutValM(int(n), int(nnz), c_ptr(ns.d_LU2_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                             int(ns.nnzM2), c_ptr(ns.d_M2_data.ptr) , c_ptr(ns.d_M2_indptr.ptr), c_ptr(ns.d_M2_indices.ptr),
                             int(ns.ncutL2), c_ptr(ns.d_cutL2.ptr), int(ns.ncutU2), c_ptr(ns.d_cutU2.ptr) )
        # 
        return
    
def gmres_italu(ns, i, n, nnz, A, d_A_data, d_A_indptr, d_A_indices, d_b, d_x, axis = 1, sys = 1):
        d_it = pycuda.gpuarray.zeros(1, ns.type_int)
        d_exectime = pycuda.gpuarray.zeros(1, ns.type_fd)

        if (ns.LU_block_jacobi == 1):
            matrix_blocking(ns, i, A, n, axis)
            if (axis == 1):
                ns.gmres_LU_BLOCK(int(ns.restart), ns.c_type_fd(ns.tol), int(n), int(ns.LUit_iters),
                              c_ptr(d_A_data.ptr), c_ptr(ns.d_LU1_data.ptr), 
                              c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr), int(nnz),
                              c_ptr(ns.d_M1_data.ptr) , 
                              c_ptr(ns.d_M1_indptr.ptr), c_ptr(ns.d_M1_indices.ptr), int(ns.nnzM1),
                              c_ptr(d_b.ptr), c_ptr(d_x.ptr), (c_ptr)(d_exectime.ptr), (c_ptr)(d_it.ptr))
            elif (axis == 2):
                ns.gmres_LU_BLOCK(int(ns.restart), ns.c_type_fd(ns.tol), int(n), int(ns.LUit_iters), 
                              c_ptr(d_A_data.ptr), c_ptr(ns.d_LU2_data.ptr), 
                              c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr), int(nnz),
                              c_ptr(ns.d_M2_data.ptr), 
                              c_ptr(ns.d_M2_indptr.ptr), c_ptr(ns.d_M2_indices.ptr), int(ns.nnzM2),
                              c_ptr(d_b.ptr), c_ptr(d_x.ptr), (c_ptr)(d_exectime.ptr), (c_ptr)(d_it.ptr))
            #endif     
        elif (ns.cutDAG_LU == 1):
            #GMRES with LU type preconditioner, 
            #L and U are stored in the same csr matrix and have a sparsity pattern different from that of A
            prec_cutDAG(ns, n, nnz, d_A_indptr, d_A_indices, axis)

            if (axis == 1):
                ns.gmres_LU3(int(ns.restart), ns.c_type_fd(ns.tol), int(n),
                               c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr), int(nnz), 
                               c_ptr(ns.d_M1_data.ptr), c_ptr(ns.d_M1_indptr.ptr), c_ptr(ns.d_M1_indices.ptr), int(ns.nnzM1), 
                               c_ptr(d_b.ptr), c_ptr(d_x.ptr), (c_ptr)(d_exectime.ptr), (c_ptr)(d_it.ptr))
            elif (axis == 2):
                ns.gmres_LU3(int(ns.restart), ns.c_type_fd(ns.tol), int(n),
                               c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr), int(nnz), 
                               c_ptr(ns.d_M2_data.ptr), c_ptr(ns.d_M2_indptr.ptr), c_ptr(ns.d_M2_indices.ptr), int(ns.nnzM2),
                               c_ptr(d_b.ptr), c_ptr(d_x.ptr), (c_ptr)(d_exectime.ptr), (c_ptr)(d_it.ptr))
            #endif
        else:
            #GMRES with LU type preconditioner, 
            #L and U are stored in the same csr matrix and have the same sparsity pattern from A
            if (axis == 1):
                ns.gmres_LU2(int(ns.restart), ns.c_type_fd(ns.tol),
                             int(ns.LU_scalar_jacobi), int(ns.LUit_iters), 
                             int(ns.approx_diag_LU), int(ns.diag),
                             int(n), int(nnz), c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                             c_ptr(ns.d_LU1_data.ptr), c_ptr(d_b.ptr), c_ptr(d_x.ptr), (c_ptr)(d_exectime.ptr), 
                             (c_ptr)(d_it.ptr))
            elif (axis == 2):
                ns.gmres_LU2(int(ns.restart), ns.c_type_fd(ns.tol),
                             int(ns.LU_scalar_jacobi), int(ns.LUit_iters), 
                             int(ns.approx_diag_LU), int(ns.diag),
                             int(n), int(nnz), c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                             c_ptr(ns.d_LU2_data.ptr), c_ptr(d_b.ptr), c_ptr(d_x.ptr), (c_ptr)(d_exectime.ptr), 
                             (c_ptr)(d_it.ptr))
            #endif
        #endif
        exectime = d_exectime.get()[0]
        it = d_it.get()[0]
        check_iteration(ns,it, i)
        free([d_it, d_exectime])
        print('Execution time of GMRES = ', exectime, ' s')

        save_iteration_data(ns,i, it, exectime, sys, axis)
        
        if (sys == 1):
            print('Execution time of GMRES + ITALU = ', ns.iteration_exectime1[i], ' s')
        elif (sys == 2):
            print('Execution time of GMRES + ITALU = ', ns.iteration_exectime2[i], ' s')
        #endif
        print()
        if (((ns.case == 4) or (ns.case == 5)) and (i <= 3)):
            ns.init_italu1 = 0
            ns.init_italu2 = 0
        #endif

        return

def gmres_prec(ns, i, n, nnz, d_A_data, d_A_indptr, d_A_indices, d_b, d_x, axis = 1, sys = 1):
        d_it = pycuda.gpuarray.zeros(1, ns.type_int)
        d_exectime = pycuda.gpuarray.zeros(1, ns.type_fd)

        ns.gmres_PREC(int(ns.restart), ns.c_type_fd(ns.tol), int(n),
                        int(ns.prec), int(nnz), c_ptr(d_A_data.ptr), c_ptr(d_A_indptr.ptr), c_ptr(d_A_indices.ptr),
                        c_ptr(d_b.ptr), c_ptr(d_x.ptr), (c_ptr)(d_exectime.ptr), (c_ptr)(d_it.ptr))

        it = d_it.get()[0]
        check_iteration(ns,it, i)
        exectime = d_exectime.get()[0]
        print('Execution time of GMRES = ', exectime, ' s')
        print()
        free([d_it, d_exectime])

        save_iteration_data(ns, i, it, exectime, sys, axis)
        return
    
    



