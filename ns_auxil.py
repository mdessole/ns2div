import numpy
import math
import pycuda.driver as cuda
import pycuda
from pycuda.gpuarray import to_gpu
from ctypes import c_ulonglong as c_ptr
from pycuda_auxil import *

def pressure_point(prob):
    x = prob.x.copy()
    y = prob.y.copy()
    n = prob.n
    
    if (prob.case == 4): #DROP
        nfx = int(round(prob.nbseg_x*1.0/2,0)) -1
    elif(prob.case == 1): #EXAC
        nfx = 0
        dist_min = x[0] ** 2 + y[0] ** 2
        for i in range(1,n):
            dist = x[i] ** 2 + y[i] ** 2
            if (dist < dist_min):
                nfx = i
                dist_min = dist
            #end if
        #end for
    else:
        nfx = int(round(n*1.0/2,0))-1
    #end if

    return nfx


def exact_function(prob, x, y, time):
    p1_x   = prob.x.copy()
    p1_y   = prob.y.copy()

    if (prob.case == 1): #EXAC
        u_ex_x = -y*math.cos(time)
        u_ex_y =  x*math.cos(time)
        rho_ex =  2.0 + math.cos(math.sin(time))*x+math.sin(math.sin(time))*y + prob.at_const*math.exp(-math.square(x) - math.square(y))
        nfx    =  prob.nfx
        p_ex   =  math.sin(time)*math.sin(x).T*math.sin(y)- math.sin(p1_x[nfx])*math.sin(p1_y[nfx])
    elif (prob.case == 2): #EXAC3
        u_ex_x =  math.square( math.square(x)-1 )*y*math.square( math.square(y)-1 )
        u_ex_y = -math.square( math.square(y)-1 )*x*math.square( math.square(x)-1 )
        rho_ex =  rho_ex + 1
        p_ex   =  0.5*math.square(x)*math.square(y) -1/18
    elif (prob.case == 3): #RRHO
        x0     = 0.0
        y0     = -0.5
        a      = 3*0.15*math.sqrt(2.0)/4
        phi0   = 1.0
        u_ex_x = -2*math.cos(time)*y
        u_ex_y = 2*math.cos(time)*x
        rho_ex = 1.0 +phi0*math.exp((-math.square(x*math.cos(2*math.sin(time))+y*math.sin(2*math.sin(time))-x0)-math.square(y*math.cos(2*math.sin(time))-x*math.sin(2*math.sin(time))-y0))/(a*a))
    else:
        u_ex_x = 0.0
        u_ex_y = 0.0
        p_ex   = 0.0
        rho_ex = 1.0 #non puo' essere zero
    #endif
    
    return u_ex_x, u_ex_y, p_ex, rho_ex

def compute_exact_sol(prob, time):
    x  = prob.x.copy()
    y  = prob.y.copy()
    xx = prob.xx.copy()
    yy = prob.yy.copy()
    n  = prob.n
    nn = prob.nn
    #time = prob.time
    
    rho_ex = numpy.zeros(n,  dtype = prob.type_fd)
    p_ex   = numpy.zeros(n,  dtype = prob.type_fd)
    u_ex_x = numpy.zeros(nn, dtype = prob.type_fd)
    u_ex_y = numpy.zeros(nn, dtype = prob.type_fd)
    
    if (prob.case == 1): #EXAC
        u_ex_x = -yy*math.cos(time)
        u_ex_y =  xx*math.cos(time)
        rho_ex =  2.0 + math.cos(math.sin(time))*x + math.sin(math.sin(time))*y + prob.at_const*numpy.exp(-numpy.square(x) - numpy.square(y))
        nfx    =  prob.nfx
        p_ex   =  math.sin(time)*numpy.multiply(numpy.sin(x), numpy.sin(y)) - numpy.sin(x[nfx])*numpy.sin(y[nfx])
    elif (prob.case == 2): #EXAC3
        u_ex_x =  numpy.square( numpy.square(xx)-1 )*yy*numpy.square( numpy.square(yy)-1 ) #probabilmente sbagliato
        u_ex_y = -numpy.square( numpy.square(yy)-1 )*xx*numpy.square( numpy.square(xx)-1 ) #probabilmente sbagliato
        rho_ex =  rho_ex + 1
        p_ex   =  0.5*numpy.square(x)*numpy.square(y) -1/18
    elif (prob.case == 3): #RRHO
        x0     = 0.0
        y0     = -0.5
        a      = 3*0.15*math.sqrt(2.0)/4
        phi0   = 1.0
        u_ex_x = -2*numpy.cos(time)*yy
        u_ex_y = 2*numpy.cos(time)*xx
        rho_ex = 1.0 +phi0*numpy.exp((-numpy.square(x*numpy.cos(2*numpy.sin(time))+y*numpy.sin(2*numpy.sin(time))-x0)-numpy.square(y*numpy.cos(2*numpy.sin(time))-x*numpy.sin(2*numpy.sin(time))-y0))/(a*a))
    #end if

    u_ex_x = numpy.array(u_ex_x, dtype = prob.type_fd)
    u_ex_y = numpy.array(u_ex_y, dtype = prob.type_fd)
    p_ex   = numpy.array(p_ex  , dtype = prob.type_fd)
    rho_ex = numpy.array(rho_ex, dtype = prob.type_fd)
    
    return u_ex_x, u_ex_y, p_ex, rho_ex

def initialize(prob, mesh):
    
    u_x = numpy.zeros(prob.nn)
    u_y = numpy.zeros(prob.nn)
    p   = numpy.zeros(prob.n)
    rho = numpy.zeros(prob.n)
    
    if (prob.case == 0):
        rho = numpy.ones(prob.n)
    elif (prob.case == 1) or (prob.case == 2) or (prob.case == 3):
        u_x, u_y, p, rho = compute_exact_sol(prob, 0)
    elif (prob.case == 4): #DROP
        x = prob.x.copy()
        y = prob.y.copy()
        x0 = 0.5
        y0 = 1.75
        a = 0.2
        yinter = 1.0
        valeur_r = numpy.sqrt(numpy.square(x-x0) + numpy.square(y-y0))
        for i in range(prob.n):
            if(valeur_r[i]<a or y[i]<yinter):
                rho[i] = prob.RHO_MAX
            else:
                rho[i] = prob.RHO_MIN
            #endif
        #endfor
    elif(prob.case == 5): #RTIN
        x = prob.x.copy()
        y = prob.y.copy()
        if (prob.RHO_MAX <= 3) or (prob.Re < 1.0):
            RTIN_eta = 0.1
        elif (prob.RHO_MAX > 3):
            RTIN_eta = 0.01
        #endif

        RTIN_d   = prob.x2 - prob.x1
        rho = (prob.RHO_MAX+prob.RHO_MIN)/2 + (prob.RHO_MAX-prob.RHO_MIN)/2*numpy.tanh((y + RTIN_eta*numpy.cos(2.0*math.pi*x/RTIN_d))*1.0/(0.01*RTIN_d))

    #endif

    u_x = numpy.array(u_x, dtype = prob.type_fd)
    u_y = numpy.array(u_y, dtype = prob.type_fd)
    p   = numpy.array(p  , dtype = prob.type_fd)
    rho = numpy.array(rho, dtype = prob.type_fd)
        
    return u_x, u_y, p, rho

def identify_BC_velocity(prob, mesh):
    nn = prob.nn
    xx = prob.xx.copy()
    yy = prob.yy.copy()

    size_diri1 = 0
    size_diri2 = 0
    size_free1 = 0
    size_free2 = 0

    if (prob.case == 0) or (prob.case == 1) or (prob.case == 2) or (prob.case == 3): #LDC EXAC
        for i in range(nn):
            if (xx[i]==prob.x2) or (yy[i]==prob.y2) or (xx[i]==prob.x1) or (yy[i]==prob.y1):
                if (size_diri1 == 0):
                    N_diri1 = numpy.array([i])
                    size_diri1 = size_diri1 +1
                else:
                    N_diri1 = numpy.append(N_diri1, numpy.array([i]))
                    size_diri1 = size_diri1 +1
                #endif
            else:
                if (size_free1 == 0):
                    N_free1 = numpy.array([i])
                    size_free1 = size_free1 +1
                else:
                    N_free1 = numpy.append(N_free1, numpy.array([i]))
                    size_free1 = size_free1 +1
                #endif
            #endif
        #endfor
        N_free2 = N_free1
        N_diri2 = N_diri1
    elif (prob.case == 4) or (prob.case == 5): #DROP RTIN
        for i in range(nn):
            if (yy[i]==prob.y2) or (yy[i]==prob.y1):
                if (size_diri1 == 0):
                    N_diri1 = numpy.array([i])
                    size_diri1 = size_diri1 +1
                else:
                    N_diri1 = numpy.append(N_diri1, numpy.array([i]))
                    size_diri1 = size_diri1 +1
                #end if
                if (size_diri2 == 0):
                    N_diri2 = numpy.array([i])
                    size_diri2 = size_diri2 +1
                else:
                    N_diri2 = numpy.append(N_diri2, numpy.array([i]))
                    size_diri2 = size_diri2 +1
                #end if
            elif (xx[i]==prob.x2) or (xx[i]==prob.x1):
                if (size_diri1 == 0):
                    N_diri1 = numpy.array([i])
                    size_diri1 = size_diri1 +1
                else:
                    N_diri1 = numpy.append(N_diri1, numpy.array([i]))
                    size_diri1 = size_diri1 +1
                #end if
                if (size_free2 == 0):
                    N_free2 = numpy.array([i])
                    size_free2 = size_free2 +1
                else:
                    N_free2 = numpy.append(N_free2, numpy.array([i]))
                    size_free2 = size_free2 +1
                #end if
            else:
                if (size_free1 == 0):
                    N_free1 = numpy.array([i])
                    size_free1 = size_free1 +1
                else:
                    N_free1 = numpy.append(N_free1, numpy.array([i]))
                    size_free1 = size_free1 +1
                #end if
                if (size_free2 == 0):
                    N_free2 = numpy.array([i])
                    size_free2 = size_free2 +1
                else:
                    N_free2 = numpy.append(N_free2, numpy.array([i]))
                    size_free2 = size_free2 +1
                #end if
            #end if
        #end for
    #end if

    N_diri1 = numpy.array(N_diri1, dtype = prob.type_int)
    N_diri2 = numpy.array(N_diri2, dtype = prob.type_int)
    N_free1 = numpy.array(N_free1, dtype = prob.type_int)
    N_free2 = numpy.array(N_free2, dtype = prob.type_int)

    return N_diri1, N_diri2, N_free1, N_free2

def identify_BC_density(prob, mesh):
    n = prob.n
    x = prob.x.copy()
    y = prob.y.copy()

    size = 0
    
    if (prob.case == 0) or (prob.case == 1) or (prob.case == 2) or (prob.case == 3) or (prob.case == 4) or (prob.case == 5): #LDC EXAC DROP RTIN
        for i in range(n):
            if (x[i] == prob.x1) or (x[i] == prob.x2) or (y[i] == prob.y1) or (y[i] == prob.y2):
                if (size == 0):
                    N_dirirho = numpy.array([i])
                    size = size + 1
                else:
                    N_dirirho = numpy.append(N_dirirho, numpy.array([i]))
                    size = size + 1
                #endif
            #end if
        #end for
    #end if                    

    return N_dirirho

def rho_P0projection(prob, mesh):
    n = prob.n
    x = prob.x.copy()
    y = prob.y.copy()

    proj_rho = numpy.zeros(n, dtype = prob.type_fd);

    coef = numpy.array([0.25, 0.25, 0.25, 0.25])
    coor = numpy.array([[0.311324865405187,  0.311324865405187],
                        [0.688675134594813,   0.311324865405187],
                        [0.688675134594813,   0.688675134594813],
                        [0.311324865405187,   0.688675134594813]])

    Dx = (prob.x2 - prob.x1)/prob.nbseg_x
    Dy = (prob.y2 - prob.y1)/prob.nbseg_y


    for i in range(prob.nbseg_x+1):
        for j in range(prob.nbseg_y+1):
            num  = j*(prob.nbseg_x+1)+i
            for k in range(4):
                xin = x[num] - Dx/2 + coor[k,0]*Dx
                yin = y[num] - Dy/2 + coor[k,1]*Dy
                u_ex_x, u_ex_y, p_ex, rho_ex = exact_function(prob, xin, yin, prob.t)
                proj_rho[num] =  proj_rho[num] + coef[k]*rho_ex
            #endfor
        #endfor
    #endfor
    return proj_rho

def rho_P0projection_parallel(prob):
    n = prob.n #mesh.get_nt()

    proj_rho = numpy.zeros(n, dtype = prob.type_fd);
    
    prob.rho_P0projection(prob.d_x, prob.d_y, numpy.int32(n),
                          cuda.In(numpy.array([prob.x1, prob.x2], dtype = prob.type_fd)), cuda.In(numpy.array([prob.y1, prob.y2], dtype = prob.type_fd)),
                          numpy.int32(prob.nbseg_x), numpy.int32(prob.nbseg_y), prob.type_fd(prob.t),  prob.type_fd(prob.at_const), cuda.InOut(proj_rho),
                          block = prob.block, grid = prob.grid_t)
    return proj_rho


def rho_recontruction(prob):
    n  = prob.n #mesh.get_nt()
    nt = prob.nt #mesh.get_nt()
    x  = prob.x.copy()
    y  = prob.y.copy()
    tt = prob.tab_connectivity_P1.copy()

    node_density_gradient = numpy.zeros((n,2))
    cell_gradient = numpy.zeros((2,nt))
    rholin = numpy.zeros(n)

    base1 = numpy.array([-1.0,1.0,0.0])
    base2 = numpy.array([-1.0,0.0,1.0])

    for i in range(nt):
        De = numpy.array( [[y[tt[i,2]] - y[tt[i, 0]], y[tt[i, 0]] - y[tt[i,1]]],
                           [x[tt[i,0]] - x[tt[i,2]],  x[tt[i,1]] - x[tt[i, 0]]]] )
        
        Je = De[0,0]*De[1,1] - De[0,1]*De[1,0]


        dxi = base1*De[0,0] + base2*De[0,1]
        dyi = base1*De[1,0] + base2*De[1,1]
        dxi = dxi/Je
        dyi = dyi/Je

        cell_gradient[0,i] = numpy.dot( dxi,prob.rho[tt[i,:]])
        cell_gradient[1,i] = numpy.dot( dyi,prob.rho[tt[i,:]])

        for j in range(2):
            node_density_gradient[tt[i,j],0] = node_density_gradient[tt[i,j],0] + (Je/(6*prob.volume[tt[i,j]]))*cell_gradient[0,i]
            node_density_gradient[tt[i,j],1] = node_density_gradient[tt[i,j],1] + (Je/(6*prob.volume[tt[i,j]]))*cell_gradient[1,i]
        #endfor
    #endfor

    for i in range(n):
        ddx = prob.volume_barycenter[0,i] - x[i]
        ddy = prob.volume_barycenter[1,i] - y[i]
        rholin[i] = prob.rho[i] + node_density_gradient[i,0]*ddx + node_density_gradient[i,1]*ddy
    #endfor

    return rholin
    
        
def norm_L2(mesh, deg, vec):
    nt = prob.nt
    
    if (deg == 1):
        nb_ddl = [0,1,2]
        tt = prob.tab_connectivity_P1.copy()
        pt_x  = prob.x.copy()
        pt_y  = prob.y.copy()
        coef = numpy.array( [-0.281250000000000,   0.260416666666667,   0.260416666666667,   0.260416666666667] )
        base = numpy.array([ [0.333333333333333,   0.333333333333333,   0.333333333333333],
                             [0.600000000000000,   0.200000000000000,   0.200000000000000],
                             [0.200000000000000,   0.600000000000000,   0.200000000000000],
                             [0.200000000000000,   0.200000000000000,   0.600000000000000]])
    elif (deg == 2):
        nb_ddl = [0,1,2,3,4,5]
        tt = prob.tab_connectivity
        pt_x  = prob.xx.copy()
        pt_y  = prob.yy.copy()
        coef = numpy.array( [0.1125000000000000, 0.0661970763942531, 0.0661970763942531, 0.0661970763942531, 0.0629695902724136, 0.0629695902724136, 0.0629695902724136])
        base = numpy.array([[-0.1111111111111112,  -0.1111111111111111,  -0.1111111111111111,   0.4444444444444444,   0.4444444444444444,   0.4444444444444444],
                            [-0.0525839011025455,  -0.0280749432230788,  -0.0280749432230788,   0.8841342417640726,   0.1122997728923152,   0.1122997728923152],
                            [-0.0280749432230789,  -0.0525839011025454,  -0.0280749432230788,   0.1122997728923152,   0.8841342417640726,   0.1122997728923152],
                            [-0.0280749432230789,  -0.0280749432230788,  -0.0525839011025454,   0.1122997728923152,   0.1122997728923152,   0.8841342417640726],
                            [0.4743526085855385,  -0.0807685941918872,  -0.0807685941918872,   0.0410358262631383,   0.3230743767675488,   0.3230743767675488],
                            [-0.0807685941918872,   0.4743526085855385,  -0.0807685941918872,   0.3230743767675487,   0.0410358262631383,   0.3230743767675488],
                            [-0.0807685941918872,  -0.0807685941918872,   0.4743526085855385,   0.3230743767675487,   0.3230743767675488,   0.0410358262631383]])
    #end if

    norm = 0.0
    for i in range(nt):
        De = numpy.array( [[pt_y[tt[i,2]] - pt_y[tt[i, 0]], pt_y[tt[i, 0]] - pt_y[tt[i,1]]],
                           [pt_x[tt[i,0]] - pt_x[tt[i,2]],  pt_x[tt[i,1]] - pt_x[tt[i, 0]]]] )
        
        Je = De[0,0]*De[1,1] - De[0,1]*De[1,0]

        tri = tt[i,nb_ddl]
        carre = numpy.square( numpy.dot(base, vec[tri]) )
        norm = norm + numpy.dot( coef, carre )*Je
        
    #endfor

    norm = numpy.sqrt(norm)

    return norm

def norm_L2_ex(prob, deg, mesh, comp):
    nt = prob.nt
    if (deg == 1):
        nb_ddl = [0,1,2]
        tt = prob.tab_connectivity_P1.copy()
        pt_x  = prob.x.copy()
        pt_y  = prob.y.copy()
        base = numpy.array([[0.8738219710169961,   0.0630890144915020,   0.0630890144915020],
                            [0.0630890144915020,   0.8738219710169960,   0.0630890144915020],
                            [0.0630890144915021,   0.0630890144915020,   0.8738219710169960],
                            [0.5014265096581800,   0.2492867451709100,   0.2492867451709100],
                            [0.2492867451709100,   0.5014265096581800,   0.2492867451709100],
                            [0.2492867451709100,   0.2492867451709100,   0.5014265096581800],
                            [0.6365024991213990,   0.0531450498448160,   0.3103524510337850],
                            [0.3103524510337850,   0.6365024991213990,   0.0531450498448160],
                            [0.3103524510337849,   0.0531450498448160,   0.6365024991213990],
                            [0.6365024991213990,   0.3103524510337850,   0.0531450498448160],
                            [0.0531450498448160,   0.6365024991213990,   0.3103524510337850],
                            [0.0531450498448161,   0.3103524510337850,   0.6365024991213990]])
    elif (deg == 2):
        tt = prob.tab_connectivity.copy()
        pt_x  = prob.xx.copy()
        pt_y  = prob.yy.copy()
        nb_ddl = [0,1,2,3,4,5]
        base = numpy.array([[0.65330770304705954,  -0.05512856699248411,  -0.05512856699248411,   0.01592089499803580,   0.22051426796993640,   0.22051426796993640],
                            [-0.05512856699248405,   0.65330770304705965,  -0.05512856699248411,   0.22051426796993642,   0.01592089499803579,   0.22051426796993612],
                            [-0.05512856699248414,  -0.05512856699248411,   0.65330770304705965,   0.22051426796993642,   0.22051426796993612,   0.01592089499803579],
                            [0.00143057951778969,  -0.12499898253509756,  -0.12499898253509756,   0.24857552527162491,   0.49999593014039018,   0.49999593014039018],
                            [-0.12499898253509753,   0.00143057951778980,  -0.12499898253509756,   0.49999593014039029,   0.24857552527162485,   0.49999593014039023],
                            [-0.12499898253509756,  -0.12499898253509756,   0.00143057951778980,   0.49999593014039029,   0.49999593014039023,   0.24857552527162485],
                            [0.17376836365417397,  -0.04749625719880005,  -0.11771516330842915,   0.06597478591860528,   0.79016044276582309,   0.13530782816862683],
                            [-0.11771516330842910,   0.17376836365417403,  -0.04749625719880005,   0.13530782816862683,   0.06597478591860527,   0.79016044276582331],
                            [-0.11771516330842935,  -0.04749625719880005,   0.17376836365417403,   0.13530782816862683,   0.79016044276582331,   0.06597478591860527],
                            [0.17376836365417400,  -0.11771516330842915,  -0.04749625719880005,   0.06597478591860528,   0.13530782816862683,   0.79016044276582309],
                            [-0.04749625719880010,   0.17376836365417403,  -0.11771516330842915,   0.79016044276582309,   0.06597478591860523,   0.13530782816862685],
                            [-0.04749625719880013,  -0.11771516330842915,   0.17376836365417403,   0.79016044276582309,   0.13530782816862685,   0.06597478591860523]])
    #endif

    coef = numpy.array([0.0254224531851030,   0.0254224531851030,   0.0254224531851030,   0.0583931378631890,   0.0583931378631890,   0.0583931378631890, 0.0414255378091870,
                        0.0414255378091870,   0.0414255378091870,   0.0414255378091870,   0.0414255378091870,   0.0414255378091870])
    coor = numpy.array([[0.0630890144915020,   0.0630890144915020],
                        [0.8738219710169960,   0.0630890144915020],
                        [0.0630890144915020,   0.8738219710169960],
                        [0.2492867451709100,   0.2492867451709100],
                        [0.5014265096581800,   0.2492867451709100],
                        [0.2492867451709100,   0.5014265096581800],
                        [0.0531450498448160,   0.3103524510337850],
                        [0.6365024991213990,   0.0531450498448160],
                        [0.0531450498448160,   0.6365024991213990],
                        [0.3103524510337850,   0.0531450498448160],
                        [0.6365024991213990,   0.3103524510337850],
                        [0.3103524510337850,   0.6365024991213990]])

    norm = 0.0
    
    for i in range(nt):
        De = numpy.array( [[pt_y[tt[i,2]] - pt_y[tt[i, 0]], pt_y[tt[i, 0]] - pt_y[tt[i,1]]],
                           [pt_x[tt[i,0]] - pt_x[tt[i,2]],  pt_x[tt[i,1]] - pt_x[tt[i, 0]]]] )
        
        Je = De[0,0]*De[1,1] - De[0,1]*De[1,0]

        i1 = tt[i,0]
        normloc = 0.0
        for k in range(12):
            xin = pt_x[i1] + coor[k,0]*De[1,1] - coor[k,1]*De[1,0]
            yin = pt_y[i1] - coor[k,0]*De[0,1] + coor[k,1]*De[0,0]
            u_ex_x, u_ex_y, p_ex, rho_ex =  exact_function(prob, xin, yin, prob.t)
            if (deg == 1): #pression
                sol     = numpy.dot( base[k,nb_ddl], prob.p[tt[i,nb_ddl]] )
                normloc = normloc + coef[k]*numpy.square( p_ex - sol )
            elif (deg == 2): #velocity
                if (comp == 1):
                    sol     = numpy.dot( base[k,nb_ddl], prob.u_x[tt[i,nb_ddl]] )
                    normloc = normloc + coef[k]*numpy.square( u_ex_x - sol )
                elif (comp == 2):
                    sol     = numpy.dot( base[k,nb_ddl], prob.u_y[tt[i,nb_ddl]] )
                    normloc = normloc + coef[k]*numpy.square( u_ex_y - sol )
                #endif
            #endif
        #endfor
        norm = norm + normloc*Je
        
    #endfor

    norm = numpy.sqrt( norm )

    return norm

def norm_L2_ex_parallel(prob, deg, comp):
    nt = prob.nt #mesh.get_nt()
    
    normloc = numpy.zeros(nt, dtype = prob.type_fd)
    d_normloc = to_gpu( normloc )
    
    if (deg == 1):
        prob.norm_L2ex_1(prob.d_x, prob.d_y, prob.d_tab_connectivity_P1, numpy.int32(nt), prob.type_fd(prob.t), numpy.int32(prob.nfx), cuda.In(prob.p),
                         d_normloc,
                         block = prob.block, grid = prob.grid)
    elif (deg == 2) and (comp == 1):
        prob.norm_L2ex_2(prob.d_xx, prob.d_yy, prob.d_tab_connectivity, numpy.int32(nt), prob.type_fd(prob.t), numpy.int32(prob.nfx), cuda.In(prob.u_x), numpy.int32(comp),
                         d_normloc,
                         block = prob.block, grid = prob.grid)
    elif (deg == 2) and (comp == 2):
        prob.norm_L2ex_2(prob.d_xx, prob.d_yy, prob.d_tab_connectivity, prob.type_int(nt), prob.type_fd(prob.t), prob.type_fd(prob.nfx), cuda.In(prob.u_y), prob.type_int(comp),
                         d_normloc,
                         block = prob.block, grid = prob.grid)
    #endif

    d_y = to_gpu( numpy.zeros(1, dtype = prob.type_fd) )
    prob.reduction( c_ptr(d_normloc.ptr), c_ptr(d_y.ptr), int(nt))
    norm = numpy.sqrt( d_y.get() )[0]

    free([d_y, d_normloc])
    
    return norm

def norm_L2_parallel(prob, deg, vec):
    nt = prob.nt

    normloc = numpy.zeros(nt, dtype = prob.type_fd)
    d_normloc = to_gpu( normloc )
    
    if (deg == 1):
        prob.norm_L2_1(prob.d_x, prob.d_y, prob.d_tab_connectivity_P1, prob.type_int(nt), cuda.In(prob.type_fd(vec)), d_normloc,
                       block = prob.block, grid = prob.grid)
    elif (deg == 2):

        prob.norm_L2_2(prob.d_xx, prob.d_yy, prob.d_tab_connectivity, prob.type_int(nt), cuda.In(prob.type_fd(vec)), d_normloc,
                       block = prob.block, grid = prob.grid)
    #endif

    d_y = to_gpu( numpy.zeros(1, dtype = prob.type_fd) )
    prob.reduction( c_ptr(d_normloc.ptr), c_ptr(d_y.ptr), int(nt))
    norm = numpy.sqrt( d_y.get() )[0]


    free([d_y, d_normloc])

    return norm



