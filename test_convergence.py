from rectangular_mesh import *
from assembly import *
from ns import *
from ns_auxil import *
import numpy
import math
import sys
from matplotlib import pyplot

def test(scheme = 'proj2'):

    case = 'EXAC'
    
    lista = ['360','180','90','44','22'] #must be even 
    
    x1 = -1.0
    x2 = 1.0
    y1 = -1.0
    y2 = 1.0

    hmax    = numpy.zeros(len(lista))
    err_u   = numpy.zeros(len(lista))
    err_p   = numpy.zeros(len(lista))
    err_rho = numpy.zeros(len(lista))

    for re in [1]: 
        #da modificare, sarebbe carino se la cartella venisse data in input a NS       
        i = 0
        for versione in lista:
            print("Generating mesh object...")
            nbseg_x = versione
            nbseg_y = versione
            grid = rect_mesh(x1 = x1, x2 = x2, y1 = y1, y2 = y2, nbseg_x = int(nbseg_x), nbseg_y = int(nbseg_y))
            print('n =', grid.n, ', nn = ', grid.nn, ', nt = ', grid.nt, 
                  ', nbseg_x = ', grid.nbseg_x, ', nbseg_y = ', grid.nbseg_y)
            print("Generating Navier-Stokes object...")
            prob = NS()
            prob.T = 0.1
            prob.Re = re
            
            prob.iterLU = 1
            
            prob.benchmark_case(case)
            
            assembly(prob, grid, scheme)
            
            prob.test_convergence = 1
            prob.solver(grid, scheme)
            
            print('Stop simulation')
            err_u[i]   = prob.max_err_u
            err_p[i]   = prob.max_err_p
            err_rho[i] = prob.max_err_rho
            hmax[i]    = numpy.maximum((grid.x2-grid.x1)/grid.nbseg_x , (grid.y2-grid.y1)/grid.nbseg_y)
            
            i = i+1
        #endfor
        
        print('err u', err_u)
        print('err p', err_p)
        print('err rho', err_rho)
        print('hmax', hmax)

        K = 60
    
        pyplot.plot(hmax, err_u, 'ro-')
        pyplot.plot(hmax, err_p, 'g+-')
        pyplot.plot(hmax, err_rho, 'b^-')
        pyplot.plot(hmax, K*numpy.square(hmax), 'm-')
        pyplot.plot(hmax, K*numpy.power(hmax,3),'c--')
        
        numpy.savez(prob.outputs_dir+'PLOT/convergence' + '_' + scheme + '_Re' + str(re), 
                    hmax = hmax, err_u = err_u, err_p = err_p, err_rho = err_rho)
        
        pyplot.xscale('log')
        pyplot.yscale('log')
        
        pyplot.ylim([1e-08, 1e+04])

    
        pyplot.legend(['Err u','Err p','Err rho', 'Slope 2','Slope 3' ], loc ='upper left')
    
        file = prob.outputs_dir+'convergence' + '_' + scheme + '_Re' + str(re) + '.pdf'
        print('saving '+file)
        pyplot.savefig(file, format='pdf')
        #pyplot.show()
    #endfor


test(scheme = sys.argv[1])
