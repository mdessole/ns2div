from __future__ import division
import os

def output_directory(file_path):

    if not os.path.isdir(file_path):
        print('Creating outputs directory: ' + file_path)
        os.makedirs(file_path)
    #endif

def manual_setup(ns):
    # Navier-Stokes parameters
    ns.Re = 1000 # Reynolds number
    ns.Dt = 0.003 #time step
    ns.T  = 4 #final time

    #Rayleigh-Taylor intability parameters (RTIN benchmark)
    ns.RHO_MAX = 3 #3
    ns.RHO_MIN = 1
    
    #FV scheme parameters
    ns.epsilon = ns.type_fd(1.000000e-6)
    ns.beta = ns.type_fd(1./3)

    #mesh renumbering
    ns.renum = 1 # 0 no renum # 1 rcm renum
    
    #Solution of linear system
    ns.gpu_solver = 1 # 0 spsolve on CPU; 1 GMRES (with preconditioner) on GPU
    
    #GMRES parameters
    ns.restart = 30 #restart
    ns.tol = 0.0000000000001 #tolerance
    ns.prec = 1 # 0 (no preconditioner), 1  (diagonal preconditioner), 2 (ILU(0) preconditioner)

    # ITALU parameters
    ns.iterLU = 1  # 0 no ITALU; 1 enable ITALU
    ns.initialize_LU = 0 # 0 initialize L,U con ILU(0); 1 inizialize L,U con tril,triu
    ns.iters = 1 # number of S_ITALU iterations
    ns.ITALU_restart = 0
    
    ns.ITALU_update_param = 1 # (if k > 1, the ITALU procedure is applied every k iterations)
       
    # parameters for L,U linear system solution in the application of the preconditioner
    ns.LU_scalar_jacobi = 0 # 0 direct solver; 1 Jacobi iterative solver
    ns.LU_block_jacobi = 0 # 0 direct solver; 1 Block Jacobi iterative solver
    ns.LUit_iters = 3 # number of Jacobi iterations

    #parameters for approximation by a band matrix
    ns.approx_diag_LU = 0 #1 enable approximation of L,U by a band matrix, 0 otherwise
    ns.diag = 3 #desired bandwidth

    #parameters for DAG cutting 
    ns.cutDAG_LU = 0
    ns.cut = 1/2 # percentage of levels to be deleted (between 0 and 1)
    # for instance, cut = 1/3 means that one third of levels will be deleted,
    # starting from the last level 

    # test on the order of convergence
    ns.test_convergence = 0

    #test output
    ns.test_output = 0 #1 enable output data storage at each time step, 0 output data storage disabled

    #cartella OUTPUT
    directory = './OUTPUTS/'
    ns.outputs_dir = directory
    
    #test plot
    ns.test_plot = 0 #1 enable plot making in RTIN or DROP test cases, 0 plot disabled

    #check output directory
    if (ns.test_convergence == 1) or (ns.test_output == 1) or (ns.test_plot == 1):
        output_directory(directory)
        ns.outputs_dir = directory
        if (ns.test_plot == 1):
            output_directory(directory+'PLOT')
    #endif


