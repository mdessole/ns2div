# File per i test
from rectangular_mesh import *
from assembly import *
from ns import *
from ns_auxil import *
from plot_RTIN_DROP import *
import sys
from numpy.linalg import norm
import time
import os


def test(nbseg_x = '8', nbseg_y = '8', scheme = 'BDF2',  case = 'EXAC'):
    # versione e' una lista con i nomi dei
    # casi:   versioni  da testare
    

    if ((int(nbseg_x) % 2) != 0) or ((int(nbseg_y) % 2) != 0):
        print('Error: the numer of discretization intervals per axis (nbseg_x and nbseg_y) must be even numbers')
        return
    
    version = str( (int(nbseg_x)+1)*(int(nbseg_y)+1)*2 )
        
    print("***********************************************")
    print("         Test start --- " + version +'---   ')
    print("***********************************************")

    if (case == 'LDC'):
        x1 = 0.0
        x2 = 1.0
        y1 = 0.0
        y2 = 1.0
    elif (case == 'EXAC'):
        x1 = -1.0
        x2 = 1.0
        y1 = -1.0
        y2 = 1.0
    elif(case == 'RTIN'):
        x1 = -0.5
        x2 = 0.5
        y1 = -2.0
        y2 = 2.0
    elif(case == 'DROP'):
        x1 = 0.0
        x2 = 1.0
        y1 = 0.0
        y2 = 2.0
    else:
        print('Error: test case not implemented')
        return
    #endif

        
    print("Generating mesh object...")
    grid = rect_mesh(x1 = x1, x2 = x2, y1 = y1, y2 = y2, nbseg_x = int(nbseg_x), nbseg_y = int(nbseg_y))
    print("Generating Navier-Stokes object...")
    prob = NS()

    prob.benchmark_case(case)
    if (prob.case == -1):
        return

    print("Assemblying FE matrices...")
    ti = time.clock()
    
    assembly(prob, grid, scheme)
    tf = time.clock()
    print("Assembly execution time = ", tf-ti)
    print()
    
    print('Navier-Stokes with ' + scheme + ' time scheme')
    tns_i = time.clock()
    prob.solver(grid, scheme)
    tns_f = time.clock()
    print('Navier-Stokes execution time  = ', tns_f - tns_i)
    print('Total elapsed time = ', tf - ti + tns_f - tns_i)

    if (prob.case == 0):
        print('rho max computed =', numpy.amax(prob.rho), 'rho min conputed=', numpy.amin(prob.rho))
    #endif
    if (prob.case == 1) or (prob.case == 3):
        u_ex_x, u_ex_y, p_ex, rho_ex = compute_exact_sol(prob, prob.T)
        print('rho=', rho_ex)
        print('rho computed = ', prob.rho)
        print('rho max =', numpy.amax(rho_ex), 'rho min =', numpy.amin(rho_ex))
        print('rho max computed =', numpy.amax(prob.rho), 'rho min conputed=', numpy.amin(prob.rho))
    #endif

        
    print()
    print('Stop simulation')
    
    if (prob.test_plot == 1):
        densitycontour(prob, scheme)

test(nbseg_x = sys.argv[1], nbseg_y = sys.argv[2], case = sys.argv[3], scheme = 'proj2' ) #(nbseg_x, nbseg_y, case)
