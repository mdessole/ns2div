import numpy
import scipy.io


class rect_mesh:
    def __init__(self, x1 = -1, x2 = 1, y1 = -1, y2 = 1, nbseg_x = 8, nbseg_y = 8):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.nbseg_x = nbseg_x
        self.nbseg_y = nbseg_y
        self.nt      = nbseg_x*nbseg_y*2
        self.n       = (nbseg_x + 1)*(nbseg_y +1)
        self.nn      = (2*nbseg_x + 1)*(2*nbseg_y +1)

        self.rect_2d()
        self.mesh_squares_fv()

        self.t = self.t.T
        self.tt = self.tt.T
        
        return

    def rect_2d(self):
        self.t = numpy.zeros((3,self.nt), dtype = numpy.int32)
        self.x = numpy.zeros(self.n)
        self.y = numpy.zeros(self.n)

        ie = -1

        for j in range(1, int(self.nbseg_y/2)+1):
            for i in range(1, int(self.nbseg_x/2)+1):
                #bottom left quadrant
                il = (2*j-2)*(self.nbseg_x+1) + 2*i - 1
                ie = ie + 1
                self.t[0,ie] = il
                self.t[1,ie] = il + self.nbseg_x + 1
                self.t[2,ie] = il - 1

                ie = ie + 1
                self.t[0,ie] = il + self.nbseg_x
                self.t[1,ie] = il - 1
                self.t[2,ie] = il + self.nbseg_x + 1

                #bottom right quadrant
                il = (2*j-2)*(self.nbseg_x+1) + 2*i
                ie = ie + 1
                self.t[0,ie] = il - 1
                self.t[1,ie] = il 
                self.t[2,ie] = il + self.nbseg_x 

                ie = ie + 1
                self.t[0,ie] = il + self.nbseg_x + 1
                self.t[1,ie] = il + self.nbseg_x 
                self.t[2,ie] = il

                #upper left quadrant
                il = (2*j-1)*(self.nbseg_x+1) + 2*i - 1
                ie = ie + 1
                self.t[0,ie] = il - 1
                self.t[1,ie] = il 
                self.t[2,ie] = il + self.nbseg_x 

                ie = ie + 1
                self.t[0,ie] = il + self.nbseg_x + 1
                self.t[1,ie] = il + self.nbseg_x 
                self.t[2,ie] = il

                #upper right quadrant
                il = (2*j-1)*(self.nbseg_x+1) + 2*i
                ie = ie + 1
                self.t[0,ie] = il 
                self.t[1,ie] = il + self.nbseg_x +1
                self.t[2,ie] = il - 1 

                ie = ie + 1
                self.t[0,ie] = il + self.nbseg_x 
                self.t[1,ie] = il - 1
                self.t[2,ie] = il + self.nbseg_x +1
            #endfor
        #endfor

        dx = (self.x2 - self.x1)*1.0/self.nbseg_x
        dy = (self.y2 - self.y1)*1.0/self.nbseg_y

        il = -1
        
        for j in range(self.nbseg_y + 1):
            for i in range(self.nbseg_x + 1):
                il = il+1
                self.x[il] = self.x1 + dx*i
                self.y[il] = self.y1 + dy*j
            #endfor
        #endfor


        self.edges()

        middle_x = (self.x[self.edges[0,:]] + self.x[self.edges[1,:]])*1.0/2
        middle_y = (self.y[self.edges[0,:]] + self.y[self.edges[1,:]])*1.0/2


        self.xx = numpy.append(self.x, middle_x)
        self.yy = numpy.append(self.y, middle_y)
        self.tt = numpy.append(self.t, self.triangle_edges + self.n, axis = 0)
        self.volume_barycenter = numpy.zeros((2, self.n))
        self.volume_barycenter[0,:] = self.x.copy()
        self.volume_barycenter[1,:] = self.y.copy()
        
        return

    def edges(self):

        work = numpy.zeros((2,self.nt+self.n),  dtype = numpy.int32)
        self.triangle_edges = numpy.zeros((3, self.nt),  dtype = numpy.int32)
        array_start = numpy.ones((self.n+1,), dtype = numpy.int32)*(-1)
        array = numpy.zeros((self.nt + self.n,), dtype = numpy.int32)

        nl = -1
        for ie in range(self.nt):
            for iv in range(3):
                ivp  = (iv+1) % 3
                ivpp = (ivp+1) % 3

                i = self.t[ivp,  ie]
                j = self.t[ivpp, ie]

                iinf = min(i,j)
                isup = max(i,j)

                il = array_start[iinf]
                
                exists = 0
                while (exists == 0) and (il != -1):
                    if (iinf == work[0,il]) and (isup == work[1,il]):
                        self.triangle_edges[iv, ie] = il
                        exists = 1
                        break
                    #endif
                    il = array[il]
                #endwhile

                if (exists == 0):
                    nl = nl+1
                    work[0,nl] = iinf
                    work[1,nl] = isup
                    array[nl] = array_start[iinf]
                    array_start[iinf] = nl
                    self.triangle_edges[iv, ie] = nl
                #end
            #end
        #end

        nb_edges = nl+1

        self.edges = work[:, range(nb_edges)]

        return      
                
                
    def mesh_squares_fv(self):

        self.horizontal_edges = numpy.zeros((2,self.nt), dtype = numpy.int32)
        self.vertical_edges   = numpy.zeros((2,self.nt), dtype = numpy.int32)

        ie = -1 

        for j in range(1, int(self.nbseg_y/2)+1):
            for i in range(1, int(self.nbseg_x/2)+1):
                #bottom left quadrant
                ie = ie+1
                self.horizontal_edges[0, ie] = 2*i-2;
                self.horizontal_edges[1, ie] = 2*j-2;
                self.vertical_edges[0, ie]   = 2*i-1;
                self.vertical_edges[1, ie]   = 2*j-2;
                
                ie = ie+1
                self.horizontal_edges[0, ie] = 2*i-2;
                self.horizontal_edges[1, ie] = 2*j-1;
                self.vertical_edges[0, ie]   = 2*i-2;
                self.vertical_edges[1, ie]   = 2*j-2;
                
                #bottom right quadrant
                ie = ie+1
                self.horizontal_edges[0, ie] = 2*i-1;
                self.horizontal_edges[1, ie] = 2*j-2;
                self.vertical_edges[0, ie]   = 2*i-1;
                self.vertical_edges[1, ie]   = 2*j-2;
                
                ie = ie+1
                self.horizontal_edges[0, ie] = 2*i-1;
                self.horizontal_edges[1, ie] = 2*j-1;
                self.vertical_edges[0, ie]   = 2*i;
                self.vertical_edges[1, ie]   = 2*j-2;
                
                #upper left quadrant
                ie = ie+1
                self.horizontal_edges[0, ie] = 2*i-2;
                self.horizontal_edges[1, ie] = 2*j-1;
                self.vertical_edges[0, ie]   = 2*i-2;
                self.vertical_edges[1, ie]   = 2*j-1;
                
                ie = ie+1
                self.horizontal_edges[0, ie] = 2*i-2;
                self.horizontal_edges[1, ie] = 2*j;
                self.vertical_edges[0, ie]   = 2*i-1;
                self.vertical_edges[1, ie]   = 2*j-1;
                
                #upper right quadrant
                ie = ie+1
                self.horizontal_edges[0, ie] = 2*i-1;
                self.horizontal_edges[1, ie] = 2*j-1;
                self.vertical_edges[0, ie]   = 2*i;
                self.vertical_edges[1, ie]   = 2*j-1;
                
                ie = ie+1
                self.horizontal_edges[0, ie] = 2*i-1;
                self.horizontal_edges[1, ie] = 2*j;
                self.vertical_edges[0, ie]   = 2*i-1;
                self.vertical_edges[1, ie]   = 2*j-1;
            #endfor
        #endfor

        
        dx = (self.x2 - self.x1)*1.0/self.nbseg_x
        dy = (self.y2 - self.y1)*1.0/self.nbseg_y

        self.volume = numpy.ones(self.n)*(dx*dy)
        
        #BOUNDARY VOLUMES
        for j in range(1, self.nbseg_y):
            K1 = (self.nbseg_x + 1)*j
            K2 = (self.nbseg_x + 1)*(j-1)
            self.volume[K1] = (dx*dy)*1.0/2
            self.volume[K2] = (dx*dy)*1.0/2
        #endfor
        for i in range(1, self.nbseg_x):
            K1 = i
            K2 = self.nbseg_y*(self.nbseg_x + 1) + i
            self.volume[K1] = (dx*dy)*1.0/2
            self.volume[K2] = (dx*dy)*1.0/2
        #endfor
        #CORNER VOLUMES
        self.volume[0] =  (dx*dy)*1.0/4
        self.volume[self.nbseg_x] =  (dx*dy)*1.0/4
        self.volume[(self.nbseg_x+1)*self.nbseg_y] =  (dx*dy)*1.0/4
        self.volume[(self.nbseg_x+1)*(self.nbseg_y+1)-1] =  (dx*dy)*1.0/4
        
        return

    
    def get_nbseg_x(self):
        #Metodo che restituisce x
        temp = numpy.copy(self.nbseg_x)
        return temp

    def get_nbseg_y(self):
        #Metodo che restituisce x
        temp = numpy.copy(self.nbseg_y)
        return temp
    
    def get_x(self):
        #Metodo che restituisce x
        temp = numpy.copy(self.x)
        return temp
    
    def get_y(self):
        #Metodo che restituisce y
        temp = numpy.copy(self.y)
        return temp
    
    def get_t(self):
        #Metodo che restituisce t
        temp = numpy.copy(self.t)
        return temp
    
    def get_xx(self):
        #Metodo che restituisce xx
        temp = numpy.copy(self.xx)
        return temp
    
    def get_yy(self):
        #Metodo che restituisce yy
        temp = numpy.copy(self.yy)
        return temp
    
    def get_tt(self):
        #Metodo che restituisce x
        temp = numpy.copy(self.tt)
        return temp

    def get_horizontal_edges(self):
        #Metodo che restituisce x
        temp = numpy.copy(self.horizontal_edges)
        return temp

    def get_vertical_edges(self):
        #Metodo che restituisce x
        temp = numpy.copy(self.vertical_edges)
        return temp
    
    def get_volume(self):
        #Metodo che restituisce x
        temp = numpy.copy(self.volume)
        return temp

    def get_volume_barycenter(self):
        #Metodo che restituisce x
        temp = numpy.copy(self.volume_barycenter)
        return temp
        
    def get_n(self):
        #Metodo che restituisce n
        temp = self.n
        return temp
    
    def get_nn(self):
        #Metodo che restituisce nn
        temp = self.nn
        return temp
    
    def get_nt(self):
        #Metodo che restituisce n
        temp = self.nt
        return temp
