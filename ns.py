# Numerica
import numpy
import scipy
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix,diags, bmat, eye, triu, tril
from scipy.sparse.linalg import svds, norm, spsolve, spilu, splu, eigs
from scipy.linalg import inv, det, norm

# Cuda
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.gpuarray import to_gpu
from pycuda.gpuarray import to_gpu_async
from pycuda.curandom import rand as curand

# per importare funzioni in c/c++
import ctypes
from ctypes import c_ulonglong as c_ptr
from ctypes import c_float, c_double, c_int
from ctypes import POINTER, byref, pointer, cast
from compiler import *

# renumbering
from scipy.sparse.csgraph import reverse_cuthill_mckee

# Funzioni ausiliarie
from ns_auxil import *
from intg_gauss import *
from assembly import *
from pycuda_auxil import *
from gmres import *
from manual_setup import manual_setup

import time

import math
from scipy.io import loadmat



class NS:
    """
    Class representing Navier-Stokes problem. 
    Given a structured mesh and a time scheme,
    the mass and divergence term matrices are built accordingly.
    """
    def __init__(self,  deg_ls = 90, cambio_prec = 0):
        

        self.type_int = numpy.int32 #numpy.uint64
        self.type_uint = numpy.uint64
        self.type_fd = numpy.float64
        self.c_type_fd = c_double

            
        # mesh parameters
        self.n  = 1 # number of P1 mesh points
        self.nn = 1 # number of P2 mesh points
        self.nt = 1 # number of mesh triangles
        # pycuda parameters
        self.th = int(256) # threads per block, if mofified check *.cu files too
        self.block = (self.th, 1, 1)
        self.grid =(1,1)
        self.BLOCKDIM = int(64) # threads per block, if mofified check *.cu files too

        manual_setup(self)

        self.at_const  = 0.0 #multiplicative constant of additional term in EXAC case
        self.div       = 0.0
        self.res       = 0.0
        self.hmax      = 0.0
        self.Dt        = 0.0
        self.t         = 0.0
        self.Dt_FE     = 0.0
        self.Dt_FV     = 0.0
        self.Dto_FE    = 0.0
        self.Dto_FV    = 0.0
        self.max_it_ns = 0
        self.Dt_div    = 0
        self.GMRES_FAIL = 0

        #maximum error in time
        self.max_err_u   = None
        self.max_err_p   = None
        self.max_err_rho = None

        #flags
        # italu flags
        self.init_italu1  = 0
        self.primo_italu1 = 1
        self.init_italu2  = 0
        self.primo_italu2 = 1
        #DAG cutting flag 
        self.init_cutDAG_LU1 = 0
        self.init_cutDAG_LU2 = 0
        #Block Jacobi
        self.init_blockprec1 = 0
        self.init_blockprec2 = 0
        
        # CARICO FUNZIONI
        s_ker_assembly_rho = open('ker_assembly_rho.cu','r').read()
        mod_ker_assembly_rho = SourceModule(s_ker_assembly_rho, options = ['--prec-div=true'])
        self.assembly_lapl_div_P2P1_kernel = mod_ker_assembly_rho.get_function("assembly_lapl_div_P2P1")
        self.assembly_mass_nlt_P2P1_kernel = mod_ker_assembly_rho.get_function('assembly_mass_nlt_P2P1')
        self.assembly_rhs_kernel = mod_ker_assembly_rho.get_function('assembly_rhs')
        self.assembly_lapl_P1_kernel = mod_ker_assembly_rho.get_function('assembly_lapl_P1')
        self.assembly_mass_P1_kernel = mod_ker_assembly_rho.get_function('assembly_mass_P1')
        self.assembly_mass_P2_kernel = mod_ker_assembly_rho.get_function('assembly_mass_P2')
        self.assembly_mass_P2P1_kernel = mod_ker_assembly_rho.get_function('assembly_mass_P2P1')
        
        s_ker_muscl = open('ker_muscl_fv_squares.cu','r').read()
        mod_ker_muscl = SourceModule(s_ker_muscl, options = ['--prec-div=true'])
        self.update_rho_ker = mod_ker_muscl.get_function('update_muscl_fv_squares')
 
        s_ker_array = open('ker_array.cu','r').read()
        mod_ker_array = SourceModule(s_ker_array, options = ['--prec-div=true'])
        self.set_value_ker = mod_ker_array.get_function('set_value')
        self.axpy_gpu_ker = mod_ker_array.get_function('axpy_gpu')
        self.xpay_gpu_ker = mod_ker_array.get_function('xpay_gpu')
        self.xpy_ker = mod_ker_array.get_function('xpy')
        self.axpby_ker = mod_ker_array.get_function('axpby')

        s_ker_index = open('ker_index.cu','r').read()
        mod_ker_index = SourceModule(s_ker_index , options = ['--prec-div=true'])
        self.indice_kernel = mod_ker_index.get_function('ker_index')

        s_ker_norm = open('ker_norm.cu','r').read()
        mod_ker_norm = SourceModule(s_ker_norm , options = ['--prec-div=true'])
        self.norm_L2ex_1 = mod_ker_norm.get_function('norm_L2ex_1')
        self.norm_L2ex_2 = mod_ker_norm.get_function('norm_L2ex_2')
        self.norm_L2_1 = mod_ker_norm.get_function('norm_L2_1')
        self.norm_L2_2 = mod_ker_norm.get_function('norm_L2_2')
        self.rho_P0projection = mod_ker_norm.get_function('rho_P0projection')

       
        # italu
        cartella = './'
        nome = 'lib_italu'
        desinenza = '.cu'       
        check_so(cartella, nome, desinenza)
        lib_italu_so = ctypes.CDLL(cartella +'lib_italu' + '.so')
        #GMRES
        self.gmres_PREC = lib_italu_so.GMRES_PREC
        self.gmres_LU = lib_italu_so.GMRES_LU
        self.gmres_LU2 = lib_italu_so.GMRES_LU2
        self.gmres_LU3 = lib_italu_so.GMRES_LU3
        self.gmres_LU_BLOCK = lib_italu_so.GMRES_LU_BLOCK
        #Simplified ITALU
        self.SITALU = lib_italu_so.S_ITALU
        #cusparse
        self.cusparse_init = lib_italu_so.cusparse_init
        self.cusparse_end = lib_italu_so.cusparse_end
        self.cusparse_csrmv = lib_italu_so.csrmv
        self.csrilu0 = lib_italu_so.csrilu0
        #cublas
        self.cublas_init = lib_italu_so.cublas_init
        self.cublas_end = lib_italu_so.cublas_end
        self.cublas_axpby = lib_italu_so.axpby
        #DAGs corresponding to L,U triangular systems' solutions
        self.print_DAG_analysis = lib_italu_so.print_DAG_analysis
        self.DAG_analysis = lib_italu_so.DAG_analysis
        #cut DAG after a given level
        self.count_nnz_cutM = lib_italu_so.count_nnz_cutM
        self.cutColIndM = lib_italu_so.cutColIndM
        self.cutValM = lib_italu_so.cutValM

        #initialize cublas and cusparse context on device 
        self.cublas_init()
        self.cusparse_init()
        
        # trust
        cartella = './'
        nome = 'lib_thrust'
        desinenza = '.cu'
        
        check_so(cartella, nome, desinenza)
        lib_thrust_so = ctypes.CDLL(cartella +'lib_thrust' + '.so')

        self.sort_by_key = lib_thrust_so.sort_by_key
        self.reduce_by_key = lib_thrust_so.reduce_by_key
        self.reduction = lib_thrust_so.reduction
        self.reduction_int = lib_thrust_so.reduction_int
        self.inclusive_scan = lib_thrust_so.inclusive_scan

        return


    def benchmark_case(self, case):
        if (case == 'LDC'):
            self.case = 0
        elif (case == 'EXAC'):
            self.case = 1
        elif (case == 'DROP'):
            self.case = 4
        elif (case == 'RTIN'):
            self.case = 5
        else:
            print()
            print('Error: test case not implemented')
            print()
            self.case = -1
        #endif

        return

    def print_benchmark_case(self):

        if (self.case == 0):
            print('Benchmark case implemented = LDC (Lid-Driven Cavity), Re = ', self.Re)
        elif (self.case == 1):
            print('Benchmark case implemented = EXAC, Re = ', self.Re)
        elif (self.case == 4):
            print('Benchmark case implemented = DROP (falling droplet), At = ', self.RHO_MAX,' Re = ', self.Re)
        elif (self.case == 5):
            print('Benchmark case implemented = RTIN (Rayleigh-Taylor Instability), At = ', self.RHO_MAX,' Re = ', self.Re)
        else:
            print('ERROR: test case not implemented')
        #endif
        
        return


    def symrcm(self, A, n):
        p    = reverse_cuthill_mckee(A) #RCM row/column permutation
        pinv = numpy.zeros(n, dtype = numpy.int32) 
        pinv[p] = numpy.arange(n) #inverse permutation
            
        return p, pinv


    def scheme_rhs_cpu(self, wnl_x, wnl_y, scheme, wnlo_x = None, wnlo_y = None):
        # Nota sui nomi
        # wnl_x -> uo_x   wnl_y -> uo_y
        # wnlo_x -> uoo_x wnlo_y -> uoo_y

        rhsu_x = self.fu_x.copy()
        rhsu_y = self.fu_y.copy()
        rhsp = self.fp.copy()


        if scheme == 'EI+EE':
            rhsu_x += self.RG.dot(wnl_x)
            rhsu_y += self.RG.dot(wnl_y)            
        elif scheme == 'BDF2':
            # rhsu = rhsu + RG*(2*uo-0.5*uoo);
            rhsu_x += self.RG.dot( - self.b * wnl_x - self.c * wnlo_x)
            rhsu_y += self.RG.dot( - self.b * wnl_y - self.c * wnlo_y)
        elif scheme == 'proj2':
            # rhsu_x = rhsu_x + RG*(2*uo_x-0.5*uoo_x) + B1.T*(3po+4phio-phioo)/3; 
            # rhsu_y = rhsu_y + RG*(2*uo_y-0.5*uoo_y) + B2.T*(3po+4phio-phioo)/3; projection method of order 2
            rhsu_x += self.RG.dot( - self.b * wnl_x - self.c * wnlo_x) + self.B1_tra*( (3 * self.po + 4*self.phio - self.phioo)/3 )
            rhsu_y += self.RG.dot( - self.b * wnl_y - self.c * wnlo_y) + self.B2_tra*( (3 * self.po + 4*self.phio - self.phioo)/3 )
        elif scheme == 'proj1':
            # rhsu_x = rhsu_x + RG*(uo_x) + B1.T*(po+phio); 
            # rhsu_y = rhsu_y + RG*(uo_y) + B2.T*(po+phio)/; projection method of order 1
            rhsu_x += self.RG.dot( wnl_x ) + self.B1_tra*( self.po + self.phio )
            rhsu_y += self.RG.dot( wnl_y ) + self.B2_tra*( self.po + self.phio )
        #endif
        

        return rhsu_x, rhsu_y, rhsp


    def scheme_rhs_gpu(self, wnl_x, wnl_y, scheme, wnlo_x = None, wnlo_y = None):
        # wnl_x -> uo_x   wnl_y -> uo_y
        # wnlo_x -> uoo_x wnlo_y -> uoo_y

        d_rhsu_x = to_gpu( numpy.array(self.fu_x, dtype = self.type_fd) )
        d_rhsu_y = to_gpu( numpy.array(self.fu_y, dtype = self.type_fd) )
        rhsp     = self.fp.copy() 
        
        self.RG.data = numpy.array(self.RG.data, dtype = self.type_fd)

        self.d_RG_data, self.d_RG_indptr, self.d_RG_indices = csr_to_gpu(self, self.RG)

        d_temp1 = pycuda.gpuarray.zeros(self.nn, self.type_fd)
        d_temp2 = pycuda.gpuarray.zeros(self.nn, self.type_fd)

        if scheme == 'EI+EE':
            d_wnl_x = to_gpu( wnl_x )
            d_wnl_y = to_gpu( wnl_y )
            self.cusparse_csrmv(int(0), int(self.nn), int(self.nn), int(self.RG.nnz), 
                                c_ptr(self.d_RG_data.ptr), c_ptr(self.d_RG_indptr.ptr), c_ptr(self.d_RG_indices.ptr),
                                c_ptr(d_wnl_x.ptr), c_ptr(d_temp1.ptr))
            self.cusparse_csrmv(int(0), int(self.nn), int(self.nn), int(self.RG.nnz), 
                                c_ptr(self.d_RG_data.ptr), c_ptr(self.d_RG_indptr.ptr), c_ptr(self.d_RG_indices.ptr),
                                c_ptr(d_wnl_y.ptr), c_ptr(d_temp2.ptr))
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp1.ptr),
                              self.c_type_fd(1.0), c_ptr(d_rhsu_x.ptr))
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp2.ptr),
                              self.c_type_fd(1.0), c_ptr(d_rhsu_y.ptr))
            free([d_wnl_x, d_wnl_y])
        elif scheme == 'BDF2':
            d_wnl_x  = to_gpu( wnl_x )
            d_wnlo_x = to_gpu( wnlo_x )
            self.cublas_axpby(int(self.nn), self.c_type_fd(-self.b), c_ptr(d_wnl_x.ptr),
                              self.c_type_fd(-self.c), c_ptr(d_wnlo_x.ptr))

            d_wnl_y  = to_gpu( wnl_y )
            d_wnlo_y = to_gpu( wnlo_y )
            self.cublas_axpby(int(self.nn), self.c_type_fd(-self.b), c_ptr(d_wnl_y.ptr),
                              self.c_type_fd(-self.c), c_ptr(d_wnlo_y.ptr))
            
            self.cusparse_csrmv(int(0), int(self.nn), int(self.nn), int(self.RG.nnz), 
                                c_ptr(self.d_RG_data.ptr), c_ptr(self.d_RG_indptr.ptr), c_ptr(self.d_RG_indices.ptr),
                                c_ptr(d_wnlo_x.ptr), c_ptr(d_temp1.ptr))
            self.cusparse_csrmv(int(0), int(self.nn), int(self.nn), int(self.RG.nnz), 
                                c_ptr(self.d_RG_data.ptr), c_ptr(self.d_RG_indptr.ptr), c_ptr(self.d_RG_indices.ptr),
                                c_ptr(d_wnlo_y.ptr), c_ptr(d_temp2.ptr))

            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp1.ptr),self.c_type_fd(1.0),
                              c_ptr(d_rhsu_x.ptr))
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp2.ptr),self.c_type_fd(1.0),
                              c_ptr(d_rhsu_y.ptr))
            free([d_wnl_x, d_wnl_y, d_wnlo_x, d_wnlo_y])
            free([d_temp1, d_temp2])
        elif scheme == 'proj1':         
            d_temp3 = pycuda.gpuarray.zeros(self.nn, self.type_fd)
            d_temp4 = pycuda.gpuarray.zeros(self.nn, self.type_fd)
            
            d_wnl_x  = to_gpu( wnl_x ) #uo_x
            d_wnl_y  = to_gpu( wnl_y ) #uo_y

            self.cusparse_csrmv(int(0), int(self.nn), int(self.nn), int(self.RG.nnz), 
                                c_ptr(self.d_RG_data.ptr), c_ptr(self.d_RG_indptr.ptr), c_ptr(self.d_RG_indices.ptr),
                                c_ptr(d_wnl_x.ptr), c_ptr(d_temp1.ptr))
            self.cusparse_csrmv(int(0), int(self.nn), int(self.nn), int(self.RG.nnz),
                                c_ptr(self.d_RG_data.ptr), c_ptr(self.d_RG_indptr.ptr), c_ptr(self.d_RG_indices.ptr),
                                c_ptr(d_wnl_y.ptr), c_ptr(d_temp2.ptr))

            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp1.ptr),
                              self.c_type_fd(1.0), c_ptr(d_rhsu_x.ptr))
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp2.ptr),
                              self.c_type_fd(1.0), c_ptr(d_rhsu_y.ptr))

            
            d_po    = to_gpu( self.po )
            d_phio  = to_gpu( self.phio )
            
            self.cublas_axpby(int(self.n), self.c_type_fd(1.0), c_ptr(d_po.ptr),
                              self.c_type_fd(1.0), c_ptr(d_phio.ptr))
            
            self.cusparse_csrmv(int(1), int(self.n), int(self.nn), int(self.B1.nnz), 
                                c_ptr(self.d_B1_data.ptr), c_ptr(self.d_B1_indptr.ptr), c_ptr(self.d_B1_indices.ptr), 
                                c_ptr(d_phio.ptr), c_ptr(d_temp3.ptr))
            self.cusparse_csrmv(int(1), int(self.n), int(self.nn), int(self.B2.nnz), 
                                c_ptr(self.d_B2_data.ptr), c_ptr(self.d_B2_indptr.ptr), c_ptr(self.d_B2_indices.ptr), 
                                c_ptr(d_phio.ptr), c_ptr(d_temp4.ptr))
 
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp3.ptr),self.c_type_fd(1.0), c_ptr(d_rhsu_x.ptr))
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp4.ptr),self.c_type_fd(1.0), c_ptr(d_rhsu_y.ptr))

            free([d_wnl_x, d_wnl_y, d_po, d_phio])
            free([d_temp1, d_temp2, d_temp3, d_temp4])
            
        elif scheme == 'proj2':
            
            d_temp3 = pycuda.gpuarray.zeros(self.nn, self.type_fd)
            d_temp4 = pycuda.gpuarray.zeros(self.nn, self.type_fd)
            
            d_wnl_x  = to_gpu( wnl_x ) #uo_x
            d_wnlo_x = to_gpu( wnlo_x ) #uoo_x
            self.cublas_axpby(int(self.nn), self.c_type_fd(-self.b), c_ptr(d_wnl_x.ptr),
                              self.c_type_fd(-self.c), c_ptr(d_wnlo_x.ptr))

            d_wnl_y  = to_gpu( wnl_y ) #uo_y
            d_wnlo_y = to_gpu( wnlo_y ) #uoo_x
            self.cublas_axpby(int(self.nn), self.c_type_fd(-self.b), c_ptr(d_wnl_y.ptr),
                              self.c_type_fd(-self.c), c_ptr(d_wnlo_y.ptr))

            self.cusparse_csrmv(int(0), int(self.nn), int(self.nn), int(self.RG.nnz), 
                                c_ptr(self.d_RG_data.ptr), c_ptr(self.d_RG_indptr.ptr), c_ptr(self.d_RG_indices.ptr), 
                                c_ptr(d_wnlo_x.ptr), c_ptr(d_temp1.ptr))
            self.cusparse_csrmv(int(0), int(self.nn), int(self.nn), int(self.RG.nnz), 
                                c_ptr(self.d_RG_data.ptr), c_ptr(self.d_RG_indptr.ptr), c_ptr(self.d_RG_indices.ptr), 
                                c_ptr(d_wnlo_y.ptr), c_ptr(d_temp2.ptr))

            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp1.ptr),self.c_type_fd(1.0), c_ptr(d_rhsu_x.ptr))
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp2.ptr),self.c_type_fd(1.0), c_ptr(d_rhsu_y.ptr))

            
            d_po    = to_gpu( self.po )
            d_phio  = to_gpu( self.phio )
            d_phioo = to_gpu( self.phioo )
            
            self.cublas_axpby(int(self.n), self.c_type_fd(3.0), c_ptr(d_po.ptr),self.c_type_fd(4.0), c_ptr(d_phio.ptr))
            self.cublas_axpby(int(self.n), self.c_type_fd(1/3), c_ptr(d_phio.ptr),self.c_type_fd(-1/3), c_ptr(d_phioo.ptr))

            
            self.cusparse_csrmv(int(1), int(self.n), int(self.nn), int(self.B1.nnz), 
                                c_ptr(self.d_B1_data.ptr), c_ptr(self.d_B1_indptr.ptr), c_ptr(self.d_B1_indices.ptr), 
                                c_ptr(d_phioo.ptr), c_ptr(d_temp3.ptr))
            self.cusparse_csrmv(int(1), int(self.n), int(self.nn), int(self.B2.nnz), 
                                c_ptr(self.d_B2_data.ptr), c_ptr(self.d_B2_indptr.ptr), c_ptr(self.d_B2_indices.ptr), 
                                c_ptr(d_phioo.ptr), c_ptr(d_temp4.ptr))
 
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp3.ptr),self.c_type_fd(1.0), c_ptr(d_rhsu_x.ptr))
            self.cublas_axpby(int(self.nn), self.c_type_fd(1.0), c_ptr(d_temp4.ptr),self.c_type_fd(1.0), c_ptr(d_rhsu_y.ptr))

            free([d_wnl_x, d_wnl_y, d_wnlo_x, d_wnlo_y, d_po, d_phio, d_phioo])
            free([d_temp1, d_temp2, d_temp3, d_temp4])
            
        #endif
        
        rhsu_x = d_rhsu_x.get()
        rhsu_y = d_rhsu_y.get()
        free([d_rhsu_x, d_rhsu_y])
        free([self.d_RG_data, self.d_RG_indptr, self.d_RG_indices])
        
        return rhsu_x, rhsu_y, rhsp


    def scheme(self, mesh, scheme_type, it = None ):
        # it = numero di iterazione NS
            
        if scheme_type == 'stokes':
            self.LG = self.A / self.Re
            self.LG.data = self.LG.data.astype(self.type_fd)
        elif scheme_type == 'EI+EE':
            self.LG = (self.M / self.Dt_FE) + (self.A / self.Re)
            self.RG = self.M / self.Dt_FE + self.NL
            self.LG.data = self.LG.data.astype(self.type_fd)
            self.RG.data = self.RG.data.astype(self.type_fd)
        elif scheme_type == 'BDF2':
            self.tipo_prec = 2
            self.b = -1/self.Dto_FE - 1/self.Dt_FE
            self.c = self.Dt_FE/(self.Dto_FE*(self.Dt_FE+self.Dto_FE))
            self.LG = self.M*(-self.b -self.c) + self.A/self.Re + self.NL
            self.RG = self.M.copy()
            self.LG.data = self.LG.data.astype(self.type_fd)
            self.RG.data = self.RG.data.astype(self.type_fd)
        elif scheme_type == 'proj2':
            self.b = -1/self.Dto_FE - 1/self.Dt_FE
            self.c = self.Dt_FE/(self.Dto_FE*(self.Dt_FE+self.Dto_FE))
            self.LG = self.M*(-self.b -self.c) + self.A*1.0/self.Re + self.NL
            self.RG = self.M.copy()
            self.LG.data = self.LG.data.astype(self.type_fd)
            self.RG.data = self.RG.data.astype(self.type_fd)
        elif scheme_type == 'proj1':
            #self.b = -1/self.Dto_FE - 1/self.Dt_FE
            #self.c = self.Dt_FE/(self.Dto_FE*(self.Dt_FE+self.Dto_FE))
            self.LG = self.M*1.0/self.Dt_FE + self.A*1.0/self.Re + self.NL
            self.RG = self.M.copy()*1.0/self.Dt_FE
            self.LG.data = self.LG.data.astype(self.type_fd)
            self.RG.data = self.RG.data.astype(self.type_fd)
        #end if


        return 

    
    def GMRES(self, i, A_n, A_nnz, A, d_A_data, d_A_indptr, d_A_indices,
              d_f, d_u, update_prec = 1, sys = 1, axis = 1):
        if (self.iterLU == 1):
            #update the preconditioner
            if (update_prec == 1) and ((self.ITALU_update_param <= 1) or (i < 6) or ((i % self.ITALU_update_param) == 0)):
                itaLU(self, i, A_n, A_nnz, A, d_A_data, d_A_indptr, d_A_indices, axis = axis)
            #endif
            #solve linear system
            gmres_italu(self, i, A_n, A_nnz, A, d_A_data, d_A_indptr, d_A_indices, d_f, d_u, sys = sys, axis = axis)
        elif (self.iterLU == 0):
            gmres_prec(self, i, A_n, A_nnz, d_A_data, d_A_indptr, d_A_indices, d_f, d_u, sys = sys, axis = axis)
        else:
            printf('Unknown setting fron gmres')
        #endif

        return

        
    def stokes_proj2(self, mesh, i, rhsu_x = None, rhsu_y = None, rhsp = None):
        if rhsu_x is None:
            rhsu_x = self.fu_x
        if rhsu_y is None:
            rhsu_y = self.fu_y
        if rhsp is None:
            rhsp = self.fp

        #Boundary conditions for velocity
        wu_1   = numpy.zeros(self.nn, self.type_fd)
        wu_2   = numpy.zeros(self.nn, self.type_fd)

        wu_1[self.N_diri1] = self.u_x[self.N_diri1].copy()
        wu_2[self.N_diri2] = self.u_y[self.N_diri2].copy()
        d_wu_1  = to_gpu( wu_1.astype(self.type_fd) )
        d_wu_2  = to_gpu( wu_2.astype(self.type_fd) )
        
        d_fu_1  = to_gpu( rhsu_x.astype(self.type_fd) )
        d_fu_2  = to_gpu( rhsu_y.astype(self.type_fd) )
        
        d_tmp_1 = pycuda.gpuarray.zeros(self.nn, self.type_fd)
        d_tmp_2 = pycuda.gpuarray.zeros(self.nn, self.type_fd)
        d_uf_1  = pycuda.gpuarray.zeros(self.nuf1, self.type_fd)
        d_uf_2  = pycuda.gpuarray.zeros(self.nuf2, self.type_fd)

        d_LG_data, d_LG_indptr, d_LG_indices = csr_to_gpu(self, self.LG)
        LG_nnz = self.LG.nnz
        LG_shape = self.LG.shape


        # tmp = LG*wu
        self.cusparse_csrmv(int(0), int(LG_shape[0]), int(LG_shape[1]), int(LG_nnz), 
                            c_ptr(d_LG_data.ptr), c_ptr(d_LG_indptr.ptr), c_ptr(d_LG_indices.ptr), 
                            c_ptr(d_wu_1.ptr), c_ptr(d_tmp_1.ptr))
        self.cusparse_csrmv(int(0), int(LG_shape[0]), int(LG_shape[1]), int(LG_nnz), 
                            c_ptr(d_LG_data.ptr), c_ptr(d_LG_indptr.ptr), c_ptr(d_LG_indices.ptr), 
                            c_ptr(d_wu_2.ptr), c_ptr(d_tmp_2.ptr))
        
        #fu = fu - LG*wu
        self.axpby_ker(d_tmp_1, self.type_fd(-1.0), d_fu_1, self.type_fd(1.0), self.type_int(self.nn), 
                       block = self.block, grid = self.grid_stk_proj )
        self.axpby_ker(d_tmp_2, self.type_fd(-1.0), d_fu_2, self.type_fd(1.0), self.type_int(self.nn), 
                       block = self.block, grid = self.grid_stk_proj )

        
        #free memory
        free([d_LG_data, d_LG_indptr, d_LG_indices])

        #Solve Burgers equation on internal points u_t + LG u + NL(u) = Bt p
        A = self.LG[self.N_free1,:][:,self.N_free1]
        A_nnz = A.nnz
        A_n = A.shape[0]
        d_A_data, d_A_indptr, d_A_indices = csr_to_gpu(self, A)
        ffu_1   = d_fu_1.get()[self.N_free1]
        d_ffu_1 = to_gpu( ffu_1.astype(self.type_fd ) )

        
        #Solve system for x component of velocity
        if (self.gpu_solver == 1): #solve on GPU
            self.GMRES(i, A_n, A_nnz, A, d_A_data, d_A_indptr, d_A_indices, d_ffu_1, d_uf_1)
            uf_1 = d_uf_1.get()
        else: #solve on CPU
            uf_1 = spsolve(A, ffu_1)
        #endif

        #Solve system for y component of velocity
        if (self.case == 4) or (self.case == 5):
            free([d_A_data, d_A_indptr, d_A_indices])
            
            A = self.LG[self.N_free2,:][:,self.N_free2]
            A_nnz = A.nnz
            A_n = A.shape[0]
            d_A_data, d_A_indptr, d_A_indices = csr_to_gpu(self, A)
            ffu_2   = d_fu_2.get()[self.N_free2]
            d_ffu_2 = to_gpu(ffu_2.astype(self.type_fd) )
            

            if (self.gpu_solver == 1): #solve on GPU
                self.GMRES(i, A_n, A_nnz, A, d_A_data, d_A_indptr, d_A_indices, d_ffu_2, d_uf_2, sys = 2, axis = 2)
                uf_2 = d_uf_2.get()
            else: #solve on CPU
                uf_2 = spsolve(A, ffu_2)
            #endif
        else:
            ffu_2   = d_fu_2.get()[self.N_free2]
            d_ffu_2 = to_gpu( numpy.array( ffu_2, dtype = self.type_fd ) )
            if (self.gpu_solver == 1): #solve on GPU
                self.GMRES(i, A_n, A_nnz, A, d_A_data, d_A_indptr, d_A_indices, d_ffu_2, d_uf_2, 
                           update_prec = 0, sys = 2)
                uf_2 = d_uf_2.get()
            else: ##solve on CPU
                uf_2 = spsolve(A, ffu_2)
            #endif
        #endif

        self.u_x[self.N_free1] = uf_1.copy()
        self.u_y[self.N_free2] = uf_2.copy()
        
        #free gpu memory
        free([d_A_data, d_A_indptr, d_A_indices, d_tmp_1, d_tmp_2, d_uf_1, 
              d_uf_2, d_fu_1, d_fu_2, d_ffu_1, d_ffu_2, d_wu_1, d_wu_2])

        #Solve equations u_t + grad p = 0; div u = 0
        d_u_1    = to_gpu( self.u_x.astype(self.type_fd) )
        d_u_2    = to_gpu( self.u_y.astype(self.type_fd) )
        d_wfp    = to_gpu( rhsp.astype(self.type_fd) )
        d_tmp_1  = pycuda.gpuarray.zeros(self.n, self.type_fd)
        d_tmp_2  = pycuda.gpuarray.zeros(self.n, self.type_fd)
        d_phi    = pycuda.gpuarray.zeros(self.npf, self.type_fd)
        
        
        # B1*u(:,1)
        self.cusparse_csrmv(int(0), int(self.n), int(self.nn), int(self.B1.nnz), 
                            c_ptr(self.d_B1_data.ptr), c_ptr(self.d_B1_indptr.ptr), c_ptr(self.d_B1_indices.ptr), 
                            c_ptr(d_u_1.ptr), c_ptr(d_tmp_1.ptr))

        
        # B2*u(:,2)
        self.cusparse_csrmv(int(0), int(self.n), int(self.nn), int(self.B2.nnz),
                            c_ptr(self.d_B2_data.ptr), c_ptr(self.d_B2_indptr.ptr), c_ptr(self.d_B2_indices.ptr), 
                            c_ptr(d_u_2.ptr), c_ptr(d_tmp_2.ptr))

        # tmp_2 = (B1*u(:,1) + B2*u(:,2))
        self.xpy_ker(d_tmp_1, d_tmp_2, numpy.int32(self.n), block = self.block, grid = self.grid_stk_proj_2)
        
        # wfp = fp + (B1*u(:,1) + B2*u(:,2))*3.0/(2.0*self.Dt_FE)
        self.axpby_ker(d_tmp_2, self.type_fd(self.chi*3.0/(2.0*self.Dt_FE)), d_wfp, self.type_fd(1), 
                       self.type_int(self.n), block = self.block, grid = self.grid_stk_proj_2 )

        wffp = d_wfp.get() [self.NPfree]
        d_wffp = to_gpu( wffp.astype(self.type_fd) ) #internal points
            
        #self.phi = (-1.0)*spsolve( self.LAP, wffp ).copy() # - Lap * phi = wffp
        self.phi[self.NPfree] = (-1.0)*spsolve( self.LAP, wffp ).copy()
        self.phi[self.nfx] = 0.0 # -= self.phi[self.nfx]

        p_p_phi = self.p + self.phi
        d_p  = to_gpu( p_p_phi.astype(self.type_fd) )
        d_Rp = pycuda.gpuarray.zeros(self.n, self.type_fd)
        
        # Rp = MPres*(p+phi)
        self.cusparse_csrmv(int(0), int(self.n), int(self.n), int(self.MG.nnz),
                            c_ptr(self.d_MG_data.ptr), c_ptr(self.d_MG_indptr.ptr), c_ptr(self.d_MG_indices.ptr), 
                            c_ptr(d_p.ptr), c_ptr(d_Rp.ptr))

        # Rp = MPres*(p+phi) - (B1*u(:,1) + B2*u(:,2))/Re
        self.axpby_ker(d_tmp_2, self.type_fd(-1/self.Re), d_Rp, self.type_fd(1), self.type_int(self.n),
                       block = self.block, grid = self.grid_stk_proj_2 )
        
        self.p = spsolve(self.MG, d_Rp.get() ) # MPres * p = Rp



        gpu_dot_wp = pycuda.gpuarray.dot(d_tmp_2, d_tmp_2)
        dot_wp = gpu_dot_wp.get()
        self.div = numpy.sqrt(dot_wp)

        free([ d_u_1, d_u_2, d_wfp, d_tmp_1, d_tmp_2, d_phi, d_p, d_Rp, gpu_dot_wp]) # d_wffp,

        return

   

    def stokes_dir(self, mesh, i, rhsu_x = None, rhsu_y = None, rhsp = None):
        if rhsu_x is None:
            rhsu_x = self.fu_x
        if rhsu_y is None:
            rhsu_y = self.fu_y
        if rhsp is None:
            rhsp = self.fp


        #construction of the sparse matrix
        C = bmat([[self.LG, None , -self.B1_tra],
                  [None, self.LG, -self.B2_tra],
                  [-self.B1, -self.B2, None]])

        C = C.tocsr()

        d_Cf_data, d_Cf_indptr, d_Cf_indices = csr_to_gpu(self, C)
        Cf_nnz = C.nnz
        Cf_shape = C.shape

        #RHS
        wu_1 = numpy.zeros(self.nn, self.type_fd)
        wu_2 = numpy.zeros(self.nn, self.type_fd)
        wp   = numpy.zeros(self.n, self.type_fd)

        
        #self.u_x, self.u_y, p_ex, rho_ex = compute_exact_sol(self)
        wu_1[self.N_diri1] = self.u_x[self.N_diri1].copy() #u_ex_x[self.N_diri1]
        wu_2[self.N_diri2] = self.u_y[self.N_diri2].copy() #u_ex_y[self.N_diri2]
        if (self.case == 1):
            u_ex_x, u_ex_y, p_ex, rho_ex = compute_exact_sol(self, self.t)
            wp[self.nfx] = p_ex[self.nfx] # self.p[self.nfx]
        else:
            wp[self.nfx] = 0.0
        #endif
        
        d_U = to_gpu( numpy.array(numpy.bmat([wu_1, wu_2, wp]).flatten(), dtype = self.type_fd) )
        d_F = to_gpu( numpy.array(numpy.bmat([rhsu_x, rhsu_y, rhsp]).flatten(), dtype = self.type_fd) )
        d_tmp   = pycuda.gpuarray.zeros(self.nn + self.nn + self.n, self.type_fd)
        d_temp1 = pycuda.gpuarray.zeros(self.n, self.type_fd)
        d_temp2 = pycuda.gpuarray.zeros(self.n, self.type_fd)

        
        #C*U
        self.cusparse_csrmv(int(0), int(Cf_shape[0]), int(Cf_shape[1]), int(Cf_nnz),
                            c_ptr(d_Cf_data.ptr), c_ptr(d_Cf_indptr.ptr), c_ptr(d_Cf_indices.ptr), c_ptr(d_U.ptr), c_ptr(d_tmp.ptr))

        #F = F - C*U
        self.axpby_ker(d_tmp, self.type_fd(-1.0), d_F, self.type_fd(1.0), self.type_int(self.nn + self.nn + self.n), block = self.block, grid = self.grid_stk_dir )

        #free memory
        free([d_Cf_data, d_Cf_indptr, d_Cf_indices])
        
        F = d_F.get()
        FF = numpy.array( numpy.bmat([F[:,self.N_free1], F[:,self.nn + self.N_free2], F[:,2*self.nn + self.NPfree]]) ).flatten()
            
        
        d_UU = pycuda.gpuarray.zeros(self.nuf1 + self.nuf2 + self.npf, self.type_fd)
      
        CC = bmat([[self.LG[self.N_free1,:][:,self.N_free1],None,-self.B1_tra[self.N_free1, :][:, self.NPfree]],
                   [None,self.LG[self.N_free2,:][:,self.N_free2],-self.B2_tra[self.N_free2, :][:, self.NPfree]],
                   [-self.B1[self.NPfree, :][:, self.N_free1],-self.B2[self.NPfree, :][:, self.N_free2],None]])

        CC = CC.tocsr()
        
        d_FF = to_gpu( FF.astype(self.type_fd) )

        d_CCf_data, d_CCf_indptr, d_CCf_indices = csr_to_gpu(self, CC)
        CCf_nnz = CC.nnz
        CCf_shape = CC.shape

        
        if (self.gpu_solver == 1): #solve on GPU
            print('GPU linear system not implemented for this time scheme. Solving the linear system on CPU...')
            UU = spsolve(CC, FF)
            d_UU = to_gpu( UU.astype(self.type_fd) )
        else: #solve on CPU
            UU = spsolve(CC, FF)
            d_UU = to_gpu( UU.astype(self.type_fd) )
        #endif

         
        self.u_x[self.N_free1] = UU[range(self.nuf1)].copy()
        self.u_y[self.N_free2] = UU[range(self.nuf1, self.nuf1 + self.nuf2)].copy()
        self.p[self.NPfree]    = UU[range(self.nuf1 + self.nuf2, self.nuf1 + self.nuf2 + self.npf)].copy()
        self.p[self.nfx]       = 0.0
        
        d_u_x = to_gpu( self.u_x.astype(self.type_fd) )
        d_u_y = to_gpu( self.u_y.astype(self.type_fd) )

        # wp = B1*u(:,1) + B2*u(:,2) + rhs;
        # B1*u(:,1)
        self.cusparse_csrmv(int(0), int(self.n), int(self.nn), int(self.B1.nnz),
                            c_ptr(self.d_B1_data.ptr), c_ptr(self.d_B1_indptr.ptr), c_ptr(self.d_B1_indices.ptr),
                            c_ptr(d_u_x.ptr), c_ptr(d_temp1.ptr))
        # B2*u(:,2)
        self.cusparse_csrmv(int(0), int(self.n), int(self.nn), int(self.B2.nnz),
                            c_ptr(self.d_B2_data.ptr), c_ptr(self.d_B2_indptr.ptr), c_ptr(self.d_B2_indices.ptr),
                            c_ptr(d_u_y.ptr), c_ptr(d_temp2.ptr))
        

        d_wp = pycuda.gpuarray.zeros(self.n, self.type_fd)
        self.xpy_ker(d_temp1, d_wp, numpy.int32(self.n),
                     block = self.block, grid = self.grid_stk_dir_2)
        self.xpy_ker(d_temp2, d_wp, numpy.int32(self.n),
                     block = self.block, grid = self.grid_stk_dir_2)

        # div = norm(wp,2);
        gpu_dot_wp = pycuda.gpuarray.dot(d_wp, d_wp)
        dot_wp = gpu_dot_wp.get()
        self.div = numpy.sqrt(dot_wp)

        # residual = norm(rp,2), rp = CC*UU - FF
        d_rp = pycuda.gpuarray.zeros(self.nuf1 + self.nuf2 + self.npf, self.type_fd)
        self.cusparse_csrmv(int(0), int(CCf_shape[0]), int(CCf_shape[1]), int(CCf_nnz), 
                            c_ptr(d_CCf_data.ptr), c_ptr(d_CCf_indptr.ptr), c_ptr(d_CCf_indices.ptr),
                            c_ptr(d_UU.ptr), c_ptr(d_rp.ptr))
        self.axpby_ker(d_FF, self.type_fd(-1.0), d_rp, self.type_fd(1.0),
                       numpy.int32(self.nuf1 + self.nuf2 + self.npf), 
                       block = self.block, grid = self.grid_stk_dir_3 )

        gpu_dot_rp = pycuda.gpuarray.dot(d_rp, d_rp)
        dot_rp = gpu_dot_rp.get()
        self.res = numpy.sqrt(dot_rp)

        print('Residual = ',self.res)

        free([d_CCf_data, d_CCf_indptr, d_CCf_indices])
        free([gpu_dot_wp, d_U, d_F, d_tmp, d_temp1, d_temp2, d_UU, d_FF, d_wp, d_rp, d_u_x, d_u_y])

        return


    def pri_u_diri(self, mesh, t):
        # setto a 1 le entrate di u_x che hanno indice al bordo e y = y2
        if (self.case == 0): #LDC
            for i in self.N_diri1:
                if(self.yy[i] == self.y2) and (self.xx[i] != self.x2) and (self.xx[i] != self.x1):
                    self.u_x[i] = self.type_fd(1.0)
                #endif
            #endfor
            self.u_y[self.N_diri2] = self.type_fd(0.0)
        elif (self.case == 1): #EXAC
            u_ex_x, u_ex_y, p_ex, rho_ex = compute_exact_sol(self, t)
            self.u_x[self.N_diri1] = u_ex_x[self.N_diri1].copy()
            self.u_y[self.N_diri2] = u_ex_y[self.N_diri2].copy()
        elif (self.case == 4 or 5): #DROP or RTIN
            self.u_x[self.N_diri1] = self.type_fd(0.0)
            self.u_y[self.N_diri2] = self.type_fd(0.0)


    def rhs_bordo_init(self, mesh):
        #Function that intializes RHS, initial and boundary conditions
        self.nfx = pressure_point(self)
        
        self.NPfree = numpy.arange(self.nfx)
        self.NPfree = numpy.append(self.NPfree, numpy.arange(self.nfx+1, self.n))
        self.NPfree = self.NPfree.astype(self.type_int)

        self.u_x, self.u_y, self.p, self.rho = initialize(self, mesh)
        self.chi = 0.5*numpy.amin(self.rho) #for projection method

        self.fu_x = numpy.zeros(self.nn, dtype = self.type_fd)
        self.fu_y = numpy.zeros(self.nn, dtype = self.type_fd)
        self.fp   = numpy.zeros(self.n,  dtype = self.type_fd)

        #Dirichlet BC at t = 0
        self.pri_u_diri(mesh, 0)


    def resol_1it_ns(self, mesh, i, iter_ns, split_step, scheme):
        #Navier-Stokes
        NL_val = numpy.zeros(self.nt * 36, dtype = self.type_fd)
        NL_ind = numpy.zeros(self.nt * 36, dtype = self.type_uint)
        M_val = numpy.zeros(6 * 6 * self.nt, dtype = self.type_fd)
        M_ind = numpy.zeros(6 * 6 * self.nt, dtype = self.type_uint)

        if (i == 1):
            self.uo_x = self.type_fd(0.5)*self.u_x.copy()
            self.uo_y = self.type_fd(0.5)*self.u_y.copy()
            self.uoo_x = numpy.zeros(self.nn, dtype = self.type_fd)
            self.uoo_y = numpy.zeros(self.nn, dtype = self.type_fd)
        #endif

        
        self.assembly_mass_nlt_P2P1_kernel(self.d_xx, self.d_yy, self.d_tab_connectivity, self.d_tab_connectivity_P1,
                                      self.type_int(self.nt), self.type_uint(self.nn),
                                      cuda.In(self.rho.astype(self.type_fd)),
                                      cuda.In(self.uo_x.astype(self.type_fd)), cuda.In(self.uo_y.astype(self.type_fd)),
                                      cuda.In(self.uoo_x.astype(self.type_fd)), cuda.In(self.uoo_y.astype(self.type_fd)),
                                      cuda.InOut(M_val), cuda.InOut(M_ind),
                                      cuda.InOut(NL_val), cuda.InOut(NL_ind),
                                      block = self.block, grid = (int(numpy.ceil(self.nt / self.th)), 1) )
        
        r_M_ind, r_M_val   = pri_SR_by_key(self, M_ind, M_val)
        I_M, J_M = create_ij(self, r_M_ind, self.nn)
        self.M = coo_matrix((r_M_val, (I_M, J_M)), shape=(self.nn, self.nn), dtype = self.type_fd).tocsr()
        
        r_NL_ind, r_NL_val = pri_SR_by_key(self, NL_ind, NL_val)
        if (r_NL_ind.shape[0] == 0):
            self.NL = csr_matrix((self.nn, self.nn), dtype = self.type_fd)
        else:
            I_NL, J_NL = create_ij(self, r_NL_ind, self.nn)
            self.NL = coo_matrix((r_NL_val, (I_NL, J_NL)), shape=(self.nn, self.nn), dtype = self.type_fd).tocsr()
        #end if

        #define the linear system's matrix according to the scheme used
        if (scheme == 'proj2') and (i == 1): # first time step with a first order projection method
            self.scheme(mesh, scheme_type = 'proj1', it = i)
        else:
            self.scheme(mesh, scheme_type = scheme, it = i) 
        #endif

        if ((self.case == 4) or (self.case == 5)) and (i > 1):
            if (split_step == 2):
                density = 2.0*(self.rhoo) - self.rhooo
            else:
                density = 2.0*(self.rho) - self.rhoo
        else:
            density = self.rho
           
        if (self.case == 1) or (self.case == 4) or (self.case == 5):
            fu_x_k = numpy.zeros(self.nt * 6, dtype = self.type_fd)
            fu_y_k = numpy.zeros(self.nt * 6, dtype = self.type_fd)
            ind    = numpy.zeros(self.nt * 6, dtype = self.type_uint)
            self.assembly_rhs_kernel(self.d_xx, self.d_yy, self.d_tab_connectivity, self.d_tab_connectivity_P1,
                                numpy.int32(self.nt),
                                cuda.In(density.astype(self.type_fd)),
                                numpy.int32(self.case), numpy.int32(split_step),
                                self.type_fd(self.t), self.type_fd(self.Dt_FE), self.type_fd(self.at_const),
                                cuda.InOut(fu_x_k), cuda.InOut(fu_y_k),
                                cuda.InOut(ind),
                                block = self.block, grid = (int(numpy.ceil(self.nt / self.th)), 1) )
            indices, self.fu_x = pri_SR_by_key(self, ind, fu_x_k, tp = 'dense')
            indices, self.fu_y = pri_SR_by_key(self, ind, fu_y_k, tp = 'dense')
        #end if      

        if (scheme == 'proj2') and (i == 1): # first time step with a first order projection method
            rhsu_x, rhsu_y, rhsp = self.scheme_rhs_gpu(self.uo_x, self.uo_y, 'proj1', wnlo_x = self.uoo_x, wnlo_y = self.uoo_y)
        else:
            rhsu_x, rhsu_y, rhsp = self.scheme_rhs_gpu(self.uo_x, self.uo_y, scheme, wnlo_x = self.uoo_x, wnlo_y = self.uoo_y)

        
        #Resol stokes
        if (scheme == 'proj2'): 
            self.stokes_proj2(mesh, i, rhsu_x = rhsu_x, rhsu_y = rhsu_y, rhsp = rhsp)
        else:
            self.stokes_dir(mesh, i, rhsu_x = rhsu_x, rhsu_y = rhsu_y, rhsp = rhsp)
            
        if (self.case == 1): #zero fixed point for pressure in EXAC case
            self.p -= self.p[self.nfx]
            
        return


    def u_average(self):
        
        ubar_x = numpy.zeros((self.nbseg_x*(self.nbseg_y+1)), dtype = self.type_fd)
        ubar_y = numpy.zeros(((self.nbseg_x+1)*(self.nbseg_y)), dtype = self.type_fd)
        
        wrk_x = numpy.zeros(self.nt)
        wrk_y = numpy.zeros(self.nt)

        wrk_x[:]=(self.u_x[self.tab_connectivity[:,3]]+self.u_x[self.tab_connectivity[:,4]]+self.u_x[self.tab_connectivity[:,5]])/3.0
        wrk_y[:]=(self.u_y[self.tab_connectivity[:,3]]+self.u_y[self.tab_connectivity[:,4]]+self.u_y[self.tab_connectivity[:,5]])/3.0

        #check: quanto scritto sotto e' compatibile con una mesh riordinata???
        #bisogna riordinare anche i vertical e horizontal edges

        for i in range(self.nt):
            ubar_x[self.horizontal_edges[0][i]*(self.nbseg_y+1)+self.horizontal_edges[1][i]] += wrk_x[i]/2.0
            ubar_y[self.vertical_edges[0][i]*(self.nbseg_y)+self.vertical_edges[1][i]] += wrk_y[i]/2.0
        #end for
        for j in range(self.nbseg_x):
            ubar_x[j*(self.nbseg_y+1)] = ubar_x[j*(self.nbseg_y+1)]*2
            ubar_x[j*(self.nbseg_y+1)+self.nbseg_y] = ubar_x[j*(self.nbseg_y+1)+self.nbseg_y]*2
        #end for
        for i in range(self.nbseg_y):
            ubar_y[i] = ubar_y[i]*2
            ubar_y[(self.nbseg_x)*(self.nbseg_y)+i]=ubar_y[(self.nbseg_x)*(self.nbseg_y)+i]*2
        #end for
        return ubar_x, ubar_y

    def resol_1it_transport(self, step_split, transport_time, i):
 
        ubar_x, ubar_y = self.u_average()
        d_deltarho1 = pycuda.gpuarray.zeros(self.n, self.type_fd)
        d_deltarho2 = pycuda.gpuarray.zeros(self.n, self.type_fd)

        if(step_split == 1):
            t_u = self.type_fd(self.t - self.Dt_FE)
        elif(step_split == 2):
            t_u = self.type_fd(self.t)
        #endif
        t_r = self.type_fd(transport_time - self.Dt_FV)
        
        d_rho       = to_gpu( numpy.array(self.rhoo, dtype = self.type_fd) )
        d_rhostar = to_gpu( numpy.array(self.rhoo, dtype = self.type_fd) )

        self.update_rho_ker(numpy.int32(self.case),
                            cuda.In(numpy.array(ubar_x, dtype = self.type_fd)), cuda.In(numpy.array(ubar_y, dtype = self.type_fd)),
                            d_rhostar, 
                            self.d_x, self.d_y, 
                            cuda.In(numpy.array([self.x1, self.x2], dtype = self.type_fd)), cuda.In(numpy.array([self.y1, self.y2], dtype = self.type_fd)),
                            numpy.int32(self.nbseg_x), numpy.int32(self.nbseg_y), numpy.int32(self.n),
                            self.type_fd(self.Dt_FV), t_r, t_u, self.type_fd(self.epsilon), self.type_fd(self.beta), self.type_fd(self.at_const),
                            d_deltarho1,
                            block = self.block, grid = self.grid_t)

        #print 'max d_deltarho1 = ', numpy.amax(d_deltarho1.get()), 'min d_d_deltarho1 = ', numpy.amin(d_deltarho1.get())
        
        self.xpy_ker(d_deltarho1, d_rhostar, self.type_int(self.n), block = self.block, grid = self.grid_t) #d_rhostar = d_rhostar + d_deltarho1


        #Dirichlet BC
        if (self.case == 2) or (self.case == 3): #actually only on RRHO
            u_ex_x, u_ex_y, p_ex, rho_ex = compute_exact_sol(self, t_r)
            for i in self.N_dirirho:
                rhostar[i] = self.type_fd(rho_ex[i])
            #end for
        #end if

        self.update_rho_ker(numpy.int32(self.case),
                            cuda.In(numpy.array(ubar_x, dtype = self.type_fd)), cuda.In(numpy.array(ubar_y, dtype = self.type_fd)),
                            d_rhostar, 
                            self.d_x, self.d_y,
                            cuda.In(numpy.array([self.x1, self.x2], dtype = self.type_fd)), cuda.In(numpy.array([self.y1, self.y2], dtype = self.type_fd)),
                            numpy.int32(self.nbseg_x), numpy.int32(self.nbseg_y), numpy.int32(self.n),
                            self.type_fd(self.Dt_FV), self.type_fd(t_r + self.Dt_FV), t_u, self.type_fd(self.epsilon), self.type_fd(self.beta),  self.type_fd(self.at_const),
                            d_deltarho2, 
                            block = self.block, grid = self.grid_t)

        self.axpby_ker(d_deltarho1, self.type_fd(0.5), d_deltarho2, self.type_fd(0.5), self.type_int(self.n), block = self.block, grid = self.grid_t)
        self.xpy_ker(d_deltarho2, d_rho, self.type_int(self.n), block = self.block, grid = self.grid_t)
        rho_new = d_rho.get()

        #Dirichlet BC
        if (self.case == 2) or (self.case == 3): #actually only on RRHO
            u_ex_x, u_ex_y, p_ex, rho_ex = compute_exact_sol(self, t_r + Dt_FV)
            for i in self.N_dirirho:
                rho_new[i] = self.type_fd(rho_ex[i])
            #end for
        #end if

        free([d_deltarho1, d_deltarho2, d_rho, d_rhostar])
        
      
        return rho_new

    def headers_display(self):
        self.print_benchmark_case()
        
        if (self.gpu_solver == 0):
            print('Direct method on CPU')
        elif (self.gpu_solver == 1):
            if (self.iterLU == 1):
                if (self.LU_scalar_jacobi == 1):
                    methodLU = ' with Jacobi('+str(self.LUit_iters)+') iterative solve of LU systems'
                elif (self.approx_diag_LU == 1):
                    methodLU = ' with direct approximate solve of LU systems (diag = '+str(self.diag)+')'
                elif (self.cutDAG_LU == 1):
                    methodLU = ' with direct approximate solve of LU systems (DAG level cut = '+str(self.cut)+')'
                elif(self.LU_block_jacobi == 1):
                    methodLU = ' with block-Jacobi('+str(self.LUit_iters)+') iterative solve of LU systems'
                else:
                    methodLU = ' with direct extact solve of LU systems'
                #endif
                print('GMRES with SITALU2('+str(self.iters)+') preconditioner' + methodLU)
            #endif
            elif (self.prec == 0):
                print('GMRES without preconditioner')
            elif (self.prec == 1):
                print('GMRES with diagonal preconditioner')
            elif (self.prec == 2):
                if (self.LU_scalar_jacobi == 1):
                    print('GMRES with ILU(0) preconditioner with Jacobi('+
                          str(self.LUit_iters)+') iterative solve of LU systems')
                else:
                    print('GMRES with ILU(0) preconditioner')
            #endif
        #endif

        if (self.ITALU_update_param > 1):
            print('Update of L,U factors for preconditioning every k = ', self.ITALU_update_param, ' iterations')
            
        if (self.renum == 1):
            print('RCM renumbering')
        else:
            print('No renumbering')
        #endif
        print()
        
        return

    def solver(self, mesh, scheme = 'BDF2'):      
        self.hmax = numpy.maximum((self.x2-self.x1)/self.nbseg_x , (self.y2-self.y1)/self.nbseg_y)
        C = 1.0
        Alpha = 1.5 #1.5
        self.Dt = C*self.hmax**(Alpha)
        self.t  = 0.0
        self.max_it_ns = int(numpy.ceil(self.T / self.Dt))
        
        self.Dt_FE = self.Dt
        self.Dt_FV = self.Dt
        coef_CFL = 0.1*self.hmax #1.0    
        
        self.headers_display()
        
        self.div_arr       = numpy.zeros( (self.max_it_ns+2,) )
        self.err_rel_u     = numpy.zeros( (self.max_it_ns+2,) )
        self.err_rel_p     = numpy.zeros( (self.max_it_ns+2,) )
        self.err_rel_rho   = numpy.zeros( (self.max_it_ns+2,) )
        if (self.case == 1):
            self.err_u     = numpy.zeros( (self.max_it_ns+2,) )
            self.err_p     = numpy.zeros( (self.max_it_ns+2,) )
            self.err_rho   = numpy.zeros( (self.max_it_ns+2,) )
            self.atwood    = numpy.zeros( (self.max_it_ns+2,) )
        #endif
            
        self.iteration_nb1       = numpy.zeros( (self.max_it_ns+2,) )
        self.iteration_nb2       = numpy.zeros( (self.max_it_ns+2,) )
        self.iteration_exectime1 = numpy.zeros( (self.max_it_ns+2,) )
        self.iteration_exectime2 = numpy.zeros( (self.max_it_ns+2,) )

        
        #prime due iterazioni
        iter_ns = 0
        if (self.case == 1) or (self.case == 2) or (self.case == 3):
            self.uoo_x = self.u_x.copy()
            self.uoo_y = self.u_y.copy()
            self.poo   = self.p.copy()
            self.rhooo = self.rho.copy()
            
            iter_ns = 1
            deb_it_ns = iter_ns + 1;
            self.t = self.Dt*1
            self.uo_x, self.uo_y, self.po, self.rhoo = compute_exact_sol(self, self.t)
            
            iter_ns = 2
            deb_it_ns = iter_ns + 1;
            self.t = self.Dt*2
            self.u_x, self.u_y, self.p, self.rho = compute_exact_sol(self, self.t)
            
            self.phi   = numpy.zeros(self.n, self.type_fd)
            self.phio  = numpy.zeros(self.n, self.type_fd)
            self.phioo = numpy.zeros(self.n, self.type_fd)
        else:
            self.uo_x = self.u_x.copy()
            self.uo_y = self.u_y.copy()
            self.po   = self.p.copy()
            self.rhoo = self.rho.copy()
            
            self.uoo_x = self.u_x.copy()
            self.uoo_y = self.u_y.copy()
            self.poo   = self.p.copy()
            self.rhooo = self.rho.copy()
            
            self.phi   = numpy.zeros(self.n, self.type_fd)
            self.phio  = numpy.zeros(self.n, self.type_fd)
            self.phioo = numpy.zeros(self.n, self.type_fd)
        #endif

        i = iter_ns+1
        iter_ns = iter_ns+1

        while (i < self.max_it_ns) and (self.t < self.T):# and False: 
            
            #################################
            #First part: (A) trasport (B) ns#
            #################################
            step_split = 1
            self.step_split = 1
            print('Step Strang =', step_split)

            #self.Dto_FE = self.Dt_FE
            if (i == 1):
                self.Dt_FE = self.Dt**2
                self.Dto_FE = self.Dt_FE
            else:
                self.Dto_FE = self.Dt_FE 
                self.Dt_FE = self.Dt
            #endif

                

            
            if (self.t + self.Dt_FE > self.T):
                self.Dt_FE = self.T - self.t
            #endif
            self.t = self.t + self.Dt_FE

            #Aggiorno u e p
            self.uoo_x = self.uo_x.copy()
            self.uo_x  = self.u_x.copy()
            self.uoo_y = self.uo_y.copy()
            self.uo_y  = self.u_y.copy()
            self.po    = self.p.copy()
            #Aggiorno rho
            self.rhooo = self.rhoo.copy()
            self.rhoo  = self.rho.copy()
            #Aggiorno phi
            self.phioo = self.phio.copy()
            self.phio  = self.phi.copy()
            
            ###############################
            ###########FV scheme###########
            ###############################

            nrm = numpy.amax(numpy.sqrt(numpy.square(self.u_x)+numpy.square(self.u_y)))
            if (nrm > 0):
                min_DT = numpy.minimum(self.Dt_FE, (coef_CFL/nrm))
            else:
                min_DT = self.Dt_FE
            #endid

            self.Dto_FV = self.Dt_FV
            self.Dt_div = int(math.ceil(self.Dt_FE/min_DT))
            self.Dt_FV  = self.Dt_FE/self.Dt_div
            transport_t = self.t - self.Dt_FE + self.Dt_FV
            rhoo_tmp = self.rhoo.copy()
            for counter in range(self.Dt_div):
                self.rho = self.resol_1it_transport(step_split, transport_t, i)
                transport_t = transport_t + self.Dt_FV
                self.rhoo = self.rho.copy() #
            #end for
            self.rhoo = rhoo_tmp.copy()

            
            #Implemento condizioni Dirichlet
            if(self.case == 1): #EXAC
                self.pri_u_diri(mesh, self.t)
            #endif
            
            ###############################
            ###########FE scheme###########
            ###############################
            if (self.case == 3): #RRHO 3
                self.u_x, self.u_y, self.p, rho_ex = compute_exact_sol(self, self.t)
            else:
                self.resol_1it_ns(mesh, i, iter_ns, step_split, scheme)
                if (self.GMRES_FAIL == 1):
                    break
                #endif
            #endif

            #Errors and norms
            if (i < self.max_it_ns):
                self.iteration_post_processing(i, scheme)

            #Save rho
            if (self.test_plot == 1):
                self.save_rho(i, scheme)
            #endif       
                
            i = i+1
            iter_ns = iter_ns+1
            ##################################
            #Second part: (A) ns (B) trasport#
            ##################################   
            step_split = 2
            self.step_split = 2
            print('Step Strang =', step_split)

            #self.Dto_FE = self.Dt_FE
            self.t = self.t + self.Dt_FE

            #Aggiorno u e p
            self.uoo_x = self.uo_x.copy()
            self.uo_x  = self.u_x.copy()
            self.uoo_y = self.uo_y.copy()
            self.uo_y  = self.u_y.copy()
            self.po    = self.p.copy()
            #Aggiorno rho
            self.rhooo = self.rhoo.copy()
            self.rhoo  = self.rho.copy()
            #Aggiorno phi
            self.phioo = self.phio.copy()
            self.phio  = self.phi.copy()

            #Implemento condizioni Dirichlet
            if(self.case == 1): #EXAC
                self.pri_u_diri(mesh, self.t)
            #endif

            ###############################
            ###########FE scheme###########
            ###############################
            if(self.case == 3):  #RRHO
                self.u_x, self.u_y, self.p, rho_ex = compute_exact_sol(self, self.t)
            else:
                self.resol_1it_ns(mesh, i, iter_ns, step_split, scheme)
                if (self.GMRES_FAIL == 1):
                    break
                #endif
            #endif

            ###############################
            ###########FV scheme###########
            ###############################
            
            nrm = numpy.amax(numpy.sqrt(numpy.square(self.u_x)+numpy.square(self.u_y)))
            if (nrm > 0):
                min_DT = numpy.minimum(self.Dt_FE, (coef_CFL/nrm))
            else:
                min_DT = self.Dt_FE
                
            self.Dto_FV = self.Dt_FV
            self.Dt_div = int(math.ceil(self.Dt_FE/min_DT))
            self.Dt_FV  = self.Dt_FE/self.Dt_div
            transport_t = self.t - self.Dt_FE + self.Dt_FV
            rhoo_tmp = self.rhoo.copy()
            for counter in range(self.Dt_div):
                self.rho = self.resol_1it_transport(step_split, transport_t, i)
                transport_t = transport_t + self.Dt_FV
                self.rhoo = self.rho.copy() #
            #end for
            self.rhoo = rhoo_tmp.copy()

            #Errors and norms
            if (i < self.max_it_ns):
                self.iteration_post_processing(i, scheme)

            #Save rho
            if (self.test_plot == 1):
                self.save_rho(i, scheme)
            #endif

                        
            i = i + 1
            iter_ns = iter_ns+1
        #endwhile

        
        if (self.test_output == 1):
            self.save_outputs()
        #endif

            
        #Close cuBLAS/cuSPARSE environments
        if ((scheme == 'proj2') and (self.gpu_solver == 1)):
            if (self.init_italu1 == 1):
                free([self.d_LU1_data])
                if ((self.cutDAG_LU == 1) and (self.init_cutDAG_LU1 == 1)):
                    free([self.d_M1_data, self.d_M1_indptr, self.d_M1_indices,
                          self.d_cutL1, self.d_cutU1 ])
            #endif
            if (self.init_italu2 == 1):
                free([self.d_LU2_data])
                if ((self.cutDAG_LU == 1) and (self.init_cutDAG_LU2 == 1)):
                    free([self.d_M2_data, self.d_M2_indptr, self.d_M2_indices,
                          self.d_cutL2, self.d_cutU2 ])
                #endif
        #endif


        
        free([self.d_x, self.d_y, self.d_xx, self.d_yy,
                   self.d_tab_connectivity,  self.d_tab_connectivity_P1,
                   self.d_B1_data, self.d_B1_indptr, self.d_B1_indices,
                   self.d_B2_data, self.d_B2_indptr, self.d_B2_indices,
                   self.d_B1_tra_data, self.d_B1_tra_indptr, self.d_B1_tra_indices,
                   self.d_B2_tra_data, self.d_B2_tra_indptr, self.d_B2_tra_indices])
        if (scheme == 'proj2'):
            free([self.d_LAP_data, self.d_LAP_indptr, self.d_LAP_indices,
                       self.d_MG_data, self.d_MG_indptr, self.d_MG_indices])

        self.cublas_end()
        self.cusparse_end()
        print

    def save_outputs(self):
        #Save execution info
        
        #test
        if (self.case == 1):
            caso = 'EXAC_'
            if (self.at_const > 0.0):
                caso = 'EXAC'+str(self.at_const)+'_'
        elif (self.case == 4):
            caso = 'DROP'+str(self.RHO_MAX)+'_'
        elif (self.case == 5):
            caso = 'RTIN'+str(self.RHO_MAX)+'_'
        #endif
            
        #method
        if (self.gpu_solver == 1): #soluzione iterativa su GPU
            if (self.iterLU == 0):
                method = '_prec'+str(self.prec)
            elif(self.iterLU == 1):
                if (self.serarate_LU == 1):
                    method = '_olditalu'
                else:
                    method = '_italu'
                    if (self.iters >1):
                        method = method+str(self.iters)
                    #endif 
                #endif
            #endif
            print(method)
            if (self.renum == 1):
                method = '_renum' + method
            if (self.LU_scalar_jacobi == 1):
                method = method + '_LU_jacobi_scalar'+str(self.LUit_iters)
                #method = '_prec'+str(self.prec) + '_LU_jacobi_scalar'+str(self.LUit_iters)
            elif (self.LU_block_jacobi == 1):
                method = method + '_LU_jacobi_block'+str(self.LUit_iters)
            elif (self.approx_diag_LU == 1):
                method = method + '_LU_approx'+str(self.diag)
            elif (self.cutDAG_LU == 1):
                method = method + '_LU_cutDAG'+str(self.cut)
        else: #soluzione diretta su GPU
            method = '_directCPU'
        #endif


        if (self.ITALU_update_param > 1):
            method = method + '_ITALU_update_param'+str(self.ITALU_update_param)
            
        if (self.T != 4.0):
            method = '_T'+str(self.T)+method

        if (self.case == 1):
            numpy.savez(self.outputs_dir+'output'+caso+str(self.nt)+'_Re'+str(self.Re)+method,
                        iteration_nb1 = self.iteration_nb1, iteration_nb2 = self.iteration_nb2,
                        iteration_exectime1 = self.iteration_exectime1, iteration_exectime2 = self.iteration_exectime2,
                        div_arr = self.div_arr, err_rel_u = self.err_rel_u, 
                        err_rel_p = self.err_rel_p, err_rel_rho = self.err_rel_rho,
                        err_u = self.err_u, err_p = self.err_p, err_rho = self.err_rho, atwood = self.atwood)
        else:
            numpy.savez(self.outputs_dir+'output'+caso+str(self.nt)+'_Re'+str(self.Re)+method,
                        iteration_nb1 = self.iteration_nb1, iteration_nb2 = self.iteration_nb2,
                        iteration_exectime1 = self.iteration_exectime1, iteration_exectime2 = self.iteration_exectime2,
                        div_arr = self.div_arr, err_rel_u = self.err_rel_u, 
                        err_rel_p = self.err_rel_p, err_rel_rho = self.err_rel_rho)
        
        #endif
    
        return

   
    def save_rho(self, i, scheme):
        if (self.case == 4):
            time =  numpy.arange(0.0, self.T+0.25, 0.25)#[0.1, 0.2, 0.3, 0.5, 0.7, 0.95, 1.0, 1.125, 1.25, 1.3612]
            caso = 'DROP_'
        if (self.case == 5):
            time = numpy.arange(0.0, self.T+0.25, 0.25)
            caso = 'RTIN'+str(self.RHO_MAX)+'_'
  
        if (self.iterLU == 1) and (scheme == 'proj2'):
            for t in time:
                if (i ==  numpy.floor(t / self.Dt)):
                    numpy.save(self.outputs_dir+'PLOT/'+caso+'rho'+str(t)+
                               '_'+str(self.nt)+'_Re'+str(self.Re)+'_italu'+str(self.iters)+'_'+scheme, self.rho)
                #endif
            #endfor
        else:
            if (self.iterLU == 0) and (scheme == 'proj2'):
                for t in time:
                    if (i ==  numpy.floor(t / self.Dt)):  
                        numpy.save( self.outputs_dir+'PLOT/'+caso+'rho'+str(t)+'_'+str(self.nt)+
                                   '_Re'+str(self.Re)+'_prec'+str(self.prec)+'_'+scheme, self.rho)
                    #endif
                #endfor
            else:
                for t in time:
                    if (i ==  numpy.floor(t / self.Dt)):  
                        numpy.save( self.outputs_dir+'PLOT/'+caso+'rho'+str(t)+'_'+str(self.nt)+
                                   '_Re'+str(self.Re)+'_'+scheme, self.rho)
                    #endif
                #endfor
        #endif

    def errors (self, i):
        err_u       = numpy.sqrt( numpy.square(norm_L2_ex_parallel(self, 2, 1) ) 
                                 + numpy.square(norm_L2_ex_parallel(self, 2, 2) )  )
        err_p       =  norm_L2_ex_parallel(self, 1, 0)
        rho_proj_ex = rho_P0projection_parallel(self)
        rho_lin     = rho_recontruction(self)
        err_rho     = numpy.dot( numpy.absolute( rho_proj_ex - rho_lin), self.volume )
        
        if (self.max_err_u is None) or (self.max_err_u < err_u) or (i ==  numpy.floor(0.30 / self.Dt)):
            self.max_err_u = err_u
            #print 'max err u = ', i
        if (self.max_err_p is None) or (self.max_err_p < err_p) or (i ==  numpy.floor(0.30 / self.Dt)):
            self.max_err_p = err_p
            #print 'max err p = ', i
        if self.max_err_rho is None or (self.max_err_rho < err_rho) or (i ==  numpy.floor(0.30 / self.Dt)):
            self.max_err_rho = err_rho
            #print 'max err rho = ', i
        print('ERR EX U \t  ERR EX P \t ERR EX RHO ')
        print(err_u, '\t', err_p, '\t', err_rho)
        
        return
        
    def iteration_post_processing(self, i, scheme):
        nrmu_x   = norm_L2_parallel(self, 2, self.u_x)
        nrmu_y   = norm_L2_parallel(self, 2, self.u_y)
        nrmu     = numpy.sqrt( numpy.square( nrmu_x ) + numpy.square( nrmu_y ) ) 
        nrmp     = norm_L2_parallel(self, 1, self.p)
        nrmrho   = math.sqrt( numpy.dot(numpy.square(self.rho), self.volume) )
        nrml1rho = numpy.dot(numpy.absolute(self.rho), self.volume)


        nrmuo_x = norm_L2_parallel(self, 2, self.u_x - self.uo_x)
        nrmuo_y = norm_L2_parallel(self, 2, self.u_y - self.uo_y)
        nrmuo   = numpy.sqrt( numpy.square( nrmuo_x ) + numpy.square( nrmuo_y ) ) 
        nrmpo   = norm_L2_parallel(self, 1, self.p - self.po)
        nrmrhoo = math.sqrt( numpy.dot(numpy.square(self.rho - self.rhoo),  self.volume) )
        
        if (self.case == 1): #EXAC
            err_u       = numpy.sqrt( numpy.square(norm_L2_ex_parallel(self, 2, 1) ) 
                                     + numpy.square(norm_L2_ex_parallel(self, 2, 2) )  )
            err_p       = norm_L2_ex_parallel(self, 1, 0)
            rho_proj_ex = rho_P0projection_parallel(self)
            rho_lin     = rho_recontruction(self)
            err_rho     = numpy.dot( numpy.absolute( rho_proj_ex - rho_lin), self.volume )

            u_x_ex, u_y_ex, p_ex, rho_ex = compute_exact_sol(self, self.t)

            nrmu_x_ex   = norm_L2_parallel(self, 2, u_x_ex)
            nrmu_y_ex   = norm_L2_parallel(self, 2, u_y_ex)
            nrmu_ex     = numpy.sqrt( numpy.square( nrmu_x_ex ) + numpy.square( nrmu_y_ex ) ) 
            nrmp_ex     = norm_L2_parallel(self, 1, p_ex)
            nrmrho_ex   = math.sqrt( numpy.dot(numpy.square(rho_ex), self.volume) )
            
            #nrmu_ex     = numpy.sqrt( numpy.square(norm_L2_ex_parallel(self, 2, 1, nrm = 'ex') ) + numpy.square(norm_L2_ex_parallel(self, 2, 2,  nrm = 'ex') )  )
            #nrmp_ex     = norm_L2_ex_parallel(self, 1, 0,  nrm = 'ex')
            #nrmrho_ex   = numpy.dot( numpy.absolute( rho_proj_ex ), self.volume )
            
            if (nrmu > 0):
                eru  = err_u / nrmu_ex
            else: 
                eru  = err_u
            if (nrmp > 0):
                erp  = err_p / nrmp_ex
            else:
                erp  = err_p
            if (nrmrho > 0):
                errho = err_rho / nrmrho_ex
            else:
                errho = err_rho
            #endif

            if (self.test_convergence == 1):
                if (self.max_err_u is None) or (self.max_err_u < err_u):
                    self.max_err_u = err_u
                if (self.max_err_p is None) or (self.max_err_p < err_p):
                    self.max_err_p = err_p
                if self.max_err_rho is None or (self.max_err_rho < err_rho):
                    self.max_err_rho = err_rho
                #endif
            #endif
        else:
            if (nrmu_x > 0):
                eru_x = nrmuo_x / nrmu_x
            else:
                eru_x = nrmuo_x
            if (nrmu_y > 0):
                eru_y = nrmuo_y / nrmu_y
            else:
                eru_y = nrmuo_y
            if (nrmu > 0):
                eru  = nrmuo / nrmu
            else: 
                eru = nrmuo
            if (nrmp > 0):
                erp  = nrmpo / nrmp
            else:
                erp  = nrmpo
            if (nrmrho > 0):
                errho = nrmrhoo / nrmrho
            else:
                errho = nrmrhoo
            #endif
        #endif

        self.div_arr[i]     = self.div
        self.err_rel_u[i]   = eru
        self.err_rel_p[i]   = erp
        self.err_rel_rho[i] = errho
        if (self.case == 1):
            self.err_u[i]   = err_u
            self.err_p[i]   = err_p
            self.err_rho[i] = err_rho
            rho_max = numpy.amax(self.rho)
            rho_min = numpy.amin(self.rho)
            self.atwood[i]  = (rho_max - rho_min)/(rho_max+rho_min)
        #endif

        if (scheme == 'EI+EE') or (scheme == 'BDF2'):
    
            if (self.case == 3):
                print('IT NS \t TIME \t NORM L2 U \t  NORM L2 P \t NORM L2 P \t ERR REL U \t  ERR REL P \t ERR REL RHO ')
                print(i,'\t', self.t, '\t', nrmu_x,'\t',nrmp,'\t', nrmrho, '\t', eru, '\t', erp, '\t', errho)
            elif (self.case == 1):
                print('IT NS\t TIME\tRES\tDIV U\tNORM L2 U\tNORM L2 P\tNORM L1 RHO\tERR REL U\tERR REL P\tERR REL RHO\t ERR EX U\tERR EX P\tERR EX RHO')
                print(i,'\t',self.t, '\t',self.res,'\t', self.div,'\t', nrmu, '\t',nrmp,'\t', nrmrho, 
                      '\t', eru, '\t', erp, '\t', errho,'\t', err_u, '\t', err_p, '\t', err_rho)
            else:
                print('IT NS  IT FV     TIME            RES              DIV U              NORM L2 U        NORM L2 P      NORM L2 RHO      NORM L1 RHO       ERR REL U       ERR REL P       ERR REL RHO')
                print("%d \t %d \t %.6f \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e  "
                      %( i, self.Dt_div, self.t, self.res, self.div, nrmu, nrmp, nrmrho, nrml1rho, eru, erp, errho ))
         
        elif (scheme == 'proj2'):
            if (self.case == 3):
                print('IT NS\t TIME \t NORM L2 U \t  NORM L2 P \t NORM L2 P \t ERR REL U \t  ERR REL P \t ERR REL RHO ')
                print(i,'\t',self.t, '\t', nrmu,'\t',nrmp,'\t', nrmrho, '\t', eru, '\t', erp, '\t', errho)
            elif (self.case == 1):
                print('IT NS  IT FV      TIME            DIV U         NORM L2 U        NORM L2 P      NORM L2 RHO      NORM L1 RHO       ERR EX U       ERR EX P       ERR EX RHO   ERR REL U    ERR REL P       ERR REL RHO')
                print("%d \t %d \t %.6f \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e  " % ( i, self.Dt_div, self.t, self.div, nrmu, nrmp, nrmrho, nrml1rho, err_u, err_p, err_rho, eru, erp, errho ))

            else:
                print('IT NS  IT FV      TIME            DIV U         NORM L2 U        NORM L2 P      NORM L2 RHO      NORM L1 RHO       ERR REL U       ERR REL P       ERR REL RHO')
                print("%d \t %d \t %.6f \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e \t %.7e  " 
                      % ( i, self.Dt_div, self.t, self.div, nrmu, nrmp, nrmrho, nrml1rho, eru, erp, errho ))

            #endif
        #endif

        return

  
