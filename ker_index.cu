/*
  Kernel per ricostruire i due vettori con gli
  indici a partire da quello condensato
  
  Data una matrice m * n l'indice memorizzato in
  indice[i] = riga[i] * n + colonna[i]

  => riga[i] = indice[i] / n
     colonna[i] = indice[i] % n
 */

typedef unsigned long long int    type_uint;

#include<stdio.h>

__global__ void ker_index(type_uint *indice, type_uint *riga, type_uint *colonna, int n, type_uint tot){

  int id = blockIdx.x*blockDim.x + threadIdx.x;
  
  //TO DO: mettere in shared una porzione di indice


  if ( id < tot){
    riga[id] = (indice[id] / n);
    colonna[id] = (indice[id] % n);
    // if (id == 0)
    //   printf("tot = %d \n", tot);
  }
  
  return;
}
