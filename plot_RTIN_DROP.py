#plot
from __future__ import division
import matplotlib
import matplotlib.pyplot as plt

import numpy
import sys

def densitycontour(ns, scheme):

    outputs_dir = ns.outputs_dir+'PLOT/'
    
    nbseg_x = str(ns.nbseg_x)
    nbseg_y = str(ns.nbseg_y)
    caso = ns.case
    density_ratio = ns.RHO_MAX
    re = str(ns.Re)
    
    #mesh = Mesh()
    if (caso == 5):
        #cartella = "MESH_RT/" + versione
        time = [0.75, 2.0, 2.75, 3.0, 3.25, 3.5, 3.75]
        #[1.0, 1.5, 2.0, 2.5, 3.0, 3.25, 3.5, 3.75, 4.0] 
        #[1.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0]#
        x1 = -0.5
        x2 = 0.5
        y1 = -2
        y2 = 2
        caso = 'RTIN'+str(density_ratio)+'_'
    elif(caso == 4):
        #cartella = "MESH_DROP/" + versione
        time = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]#[0.2, 0.5, 1.125, 1.25] #, 1.3612]
        x1 = 0.0
        x2 = 1.0
        y1 = 0.0
        y2 = 2.0
        caso = 'DROP_'
    else:
        print('Error: plot not implemented for benchmark ', caso)
        return
    #endif

    versione = str(int(nbseg_x)*int(nbseg_y)*2)
    x,y = numpy.linspace( x1, x2, int(nbseg_x)+1), numpy.linspace( x1, x2, int(nbseg_y)+1)

    x,y = numpy.meshgrid(x,y)

    density_ratio = int(density_ratio)
    
    if (density_ratio == 3):
        levels = [1.40, 1.45, 1.50, 1.55, 1.60]
    elif (density_ratio == 7):
        levels = [2.0, 2.20, 2.60, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
    elif (density_ratio == 19):
        levels = [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
    else:
        levels = [(density_ratio/2-0.1*density_ratio), (density_ratio/2-0.05*density_ratio), (density_ratio/2), (density_ratio/2+0.05*density_ratio), (density_ratio/2+0.1*density_ratio)]
        

    cartella = outputs_dir

    f, axarr = plt.subplots(1, len(time), sharey = True, sharex = True)
    
    if (ns.iterLU == 1) and (scheme == 'proj2'):
        metodo = '_italu'+str(ns.iters)
    else:
        if (ns.iterLU == 0) and (scheme == 'proj2'):
            metodo = '_prec'+str(ns.prec)
        else:
            metodo = ''
        #endif
    #endif
    plt.rcParams.update({'font.size': 8})
    
    for i,t in enumerate(time):
        rho = numpy.load(cartella+caso+'rho'+str(t)+'_'+versione+'_Re'+re+metodo+'_'+scheme+'.npy')
        rho = rho.reshape(int(nbseg_y)+1, int(nbseg_x)+1)
        axarr[i].contourf(x,y, rho, levels)
        axarr[i].set_title('T = '+str(t))
        #plt.setp(axarr[i].tick_params(labelsize = 8))
    #endfor
                            
    plt.setp(axarr, xticks = [x1, x2])
    f.tight_layout(w_pad = 0.2)

    file = ns.outputs_dir+caso+'Re'+re+'_'+nbseg_x+'x'+nbseg_y+metodo+'_'+scheme+'.png'
    print('saving '+file)
    plt.savefig(file, format='png')


    return

