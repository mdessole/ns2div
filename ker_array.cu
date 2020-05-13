/* 
   file che contiene kernel per la gestione
   degli array.
*/

typedef double               type_fd;
#include <stdio.h>

__global__ void set_value(type_fd* x, int i, type_fd val, int len){
  /* this kernel sets the i'th entry of x to val
     x[i] = val
     
     x = array
     i = index
     val = new value
     len = x's length
  */

  int id = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (id < len){
    if (id == i){
      x[i] = val;
    }
  }

}

__global__ void xpy(const type_fd* x, type_fd* y, int len){
  /* this kernel computes
     y = x + y
  */

  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < len){
    y[id] += x[id];
  }
}



__global__ void axpy(const type_fd* x, type_fd* y, type_fd alpha, int len){
  /* this kernel computes
     y = alpha * x + y
  */

  int id = blockIdx.x*blockDim.x + threadIdx.x;

  if (id < len){
    y[id] += (alpha * x[id]);
  }
}

__global__ void axpby(type_fd* x, type_fd alpha, type_fd* y, type_fd beta, int len){
  /* this kernel computes
     y = alpha * x + beta * y
  */

  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < len){
    y[id] = (alpha * x[id] + beta * y[id]);
  }
}

__global__ void axpy_gpu(const type_fd* x, type_fd* y, type_fd* alpha, int len){
  /* this kernel computes
     y = alpha * x + y
     where alpha is a lenth 1 gpu array 
  */

  int id = blockIdx.x*blockDim.x + threadIdx.x;

  if (id < len){
    y[id] += (alpha[0] * x[id]);
  }
}


__global__ void xpay_gpu(const type_fd* x, type_fd* y, type_fd* alpha, int len){
  /* this kernel computes
     y = x + alpha * y
     where alpha is a lenth 1 gpu array 
  */

  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if ( id < len){
    y[id] = x[id] + alpha[0] * y[id];
  }
}


__global__ void ax(type_fd* x, type_fd alpha, int len){
  /* this kernel computes
     x = alpha * x 
  */

  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < len){
    x[id] = (alpha * x[id]);
  }
}
