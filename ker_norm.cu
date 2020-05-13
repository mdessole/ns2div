typedef int                   intero;
typedef double                type_fd;
#define BLOCKDIM 256
#include <stdio.h>
#include <math.h>
//#include <thrust/gather.h>

__device__ void set_coef_rhoP0(type_fd *c){
  c[0]=0.25; c[1]=0.25; c[2]=0.25; c[3]=0.25;
  return;
}

__device__ void set_base_rhoP0(type_fd *b){
  b[0]=0.311324865405187;   b[1]=0.311324865405187;
  b[2]=0.688675134594813;   b[3]=0.311324865405187;
  b[4]=0.688675134594813;   b[5]=0.688675134594813;
  b[6]=0.311324865405187;   b[7]=0.688675134594813;
  return;
}

__device__ void set_coef_L2_1(type_fd *c){
  c[0]=-0.281250000000000;   c[1]=0.260416666666667;   c[2]=0.260416666666667;   c[3]=0.260416666666667;
  return;
}

__device__ void set_base_L2_1(type_fd *b){
  b[0]=0.333333333333333;   b[1]=0.333333333333333;   b[2]=0.333333333333333;
  b[3]=0.600000000000000;   b[4]=0.200000000000000;   b[5]=0.200000000000000;
  b[6]=0.200000000000000;   b[7]=0.600000000000000;   b[8]=0.200000000000000;
  b[9]=0.200000000000000;   b[10]=0.200000000000000;   b[11]=0.600000000000000;
  return;
}

__device__ void set_coef_L2_2(type_fd *c){
  c[0]=0.1125000000000000; c[1]=0.0661970763942531; c[2]=0.0661970763942531; c[3]=0.0661970763942531; c[4]=0.0629695902724136; c[5]=0.0629695902724136; c[6]=0.0629695902724136;
  return;
}

__device__ void set_base_L2_2(type_fd *b){

  b[0]=-0.1111111111111112;   b[1]=-0.1111111111111111;   b[2]=-0.1111111111111111;    b[3]=0.4444444444444444;    b[4]=0.4444444444444444;    b[5]=0.4444444444444444;
  b[6]=-0.0525839011025455;   b[7]= 0.0280749432230788;   b[8]=-0.0280749432230788;    b[9]=0.8841342417640726;    b[10]=0.1122997728923152;   b[11]=0.1122997728923152;
  b[12]=-0.0280749432230789;  b[13]=-0.0525839011025454;  b[14]=-0.0280749432230788;   b[15]=0.1122997728923152;   b[16]=0.8841342417640726;   b[17]=0.1122997728923152;
  b[18]=-0.0280749432230789;  b[19]=-0.0280749432230788;  b[20]=-0.0525839011025454;   b[21]=0.1122997728923152;   b[22]=0.1122997728923152;   b[23]=0.8841342417640726;
  b[24]= 0.4743526085855385;  b[25]=-0.0807685941918872;  b[26]=-0.0807685941918872;   b[27]=0.0410358262631383;   b[28]=0.3230743767675488;   b[29]=0.3230743767675488;
  b[30]=-0.0807685941918872;  b[31]= 0.4743526085855385;  b[32]=-0.0807685941918872;   b[33]=0.3230743767675487;   b[34]=0.0410358262631383;   b[35]=0.3230743767675488;
  b[36]=-0.0807685941918872;  b[37]=-0.0807685941918872;  b[38]=0.4743526085855385;    b[39]=0.3230743767675487;   b[40]=0.3230743767675488;   b[41]=0.0410358262631383;
  return;
}

__device__ void set_base_L2ex_1(type_fd *b){
  b[0]=0.8738219710169961;    b[1]=0.0630890144915020;    b[2]=0.0630890144915020;
  b[3]=0.0630890144915020;    b[4]=0.8738219710169960;    b[5]=0.0630890144915020;
  b[6]=0.0630890144915021;    b[7]=0.0630890144915020;    b[8]=0.8738219710169960;
  b[9]=0.5014265096581800;    b[10]=0.2492867451709100;   b[11]=0.2492867451709100;
  b[12]=0.2492867451709100;   b[13]=0.5014265096581800;   b[14]=0.2492867451709100;
  b[15]=0.2492867451709100;   b[16]=0.2492867451709100;   b[17]=0.5014265096581800;
  b[18]=0.6365024991213990;   b[19]=0.0531450498448160;   b[20]=0.3103524510337850;
  b[21]=0.3103524510337850;   b[22]=0.6365024991213990;   b[23]=0.0531450498448160;
  b[24]=0.3103524510337849;   b[25]=0.0531450498448160;   b[26]=0.6365024991213990;
  b[27]=0.6365024991213990;   b[28]=0.3103524510337850;   b[29]=0.0531450498448160;
  b[30]=0.0531450498448160;   b[31]=0.6365024991213990;   b[32]=0.3103524510337850;
  b[33]=0.0531450498448161;   b[34]=0.3103524510337850;   b[35]=0.6365024991213990;
  return;
}

__device__ void set_base_L2ex_2(type_fd *b){
  b[0] =0.65330770304705954;    b[1]=-0.05512856699248411;    b[2]=-0.05512856699248411;    b[3]=0.01592089499803580;    b[4]=0.22051426796993640;   b[5]=0.22051426796993640;
  b[6] =-0.05512856699248405;   b[7]=0.65330770304705965;     b[8]=-0.05512856699248411;    b[9]=0.22051426796993642;    b[10]=0.01592089499803579;   b[11]=0.22051426796993612;
  b[12]=-0.05512856699248414;   b[13]=-0.05512856699248411;   b[14]=0.65330770304705965;    b[15]=0.22051426796993642;   b[16]=0.22051426796993612;   b[17]=0.01592089499803579;
  b[18]=0.00143057951778969;    b[19]=-0.12499898253509756;   b[20]=-0.12499898253509756;   b[21]=0.24857552527162491;   b[22]=0.49999593014039018;   b[23]=0.49999593014039018;
  b[24]=-0.12499898253509753;   b[25]=0.00143057951778980;    b[26]=-0.12499898253509756;   b[27]=0.49999593014039029;   b[28]=0.24857552527162485;   b[29]=0.49999593014039023;
  b[30]=-0.12499898253509756;   b[31]=-0.12499898253509756;   b[32]=0.00143057951778980;    b[33]=0.49999593014039029;   b[34]=0.49999593014039023;   b[35]=0.24857552527162485;
  b[36]=0.17376836365417397;    b[37]=-0.04749625719880005;   b[38]=-0.11771516330842915;   b[39]=0.06597478591860528;   b[40]=0.79016044276582309;   b[41]=0.13530782816862683;
  b[42]=-0.11771516330842910;   b[43]=0.17376836365417403;    b[44]=-0.04749625719880005;   b[45]=0.13530782816862683;   b[46]=0.06597478591860527;   b[47]=0.79016044276582331;
  b[48]=-0.11771516330842935;   b[49]=-0.04749625719880005;   b[50]=0.17376836365417403;    b[51]=0.13530782816862683;   b[52]=0.79016044276582331;   b[53]=0.06597478591860527;
  b[54]=0.17376836365417400;    b[55]=-0.11771516330842915;   b[56]=-0.04749625719880005;   b[57]=0.06597478591860528;   b[58]=0.13530782816862683;   b[59]=0.79016044276582309;
  b[60]=-0.04749625719880010;   b[61]=0.17376836365417403;    b[62]=-0.11771516330842915;   b[63]=0.79016044276582309;   b[64]=0.06597478591860523;   b[65]=0.13530782816862685;
  b[66]=-0.04749625719880013;   b[67]=-0.11771516330842915;   b[68]=0.17376836365417403;    b[69]=0.79016044276582309;   b[70]=0.13530782816862685;   b[71]=0.06597478591860523;
  return;
}

__device__ void set_coef_L2ex(type_fd *c){
  c[0]=0.0254224531851030;   c[1]=0.0254224531851030;   c[2]=0.0254224531851030;   c[3]=0.0583931378631890;   c[4]=0.0583931378631890;   c[5]=0.0583931378631890; c[6]=0.0414255378091870; c[7]=0.0414255378091870;   c[8]=0.0414255378091870;   c[9]=0.0414255378091870;   c[10]=0.0414255378091870;   c[11]=0.0414255378091870;
  return;
}

__device__ void set_coor_L2ex(type_fd *c){
  c[0]=0.0630890144915020;    c[1]=0.0630890144915020;
  c[2]=0.8738219710169960;    c[3]=0.0630890144915020;
  c[4]=0.0630890144915020;    c[5]=0.8738219710169960;
  c[6]=0.2492867451709100;    c[7]=0.2492867451709100;
  c[8]=0.5014265096581800;    c[9]=0.2492867451709100;
  c[10]=0.2492867451709100;   c[11]=0.5014265096581800;
  c[12]=0.0531450498448160;   c[13]=0.3103524510337850;
  c[14]=0.6365024991213990;   c[15]=0.0531450498448160;
  c[16]=0.0531450498448160;   c[17]=0.6365024991213990;
  c[18]=0.3103524510337850;   c[19]=0.0531450498448160;
  c[20]=0.6365024991213990;   c[21]=0.3103524510337850;
  c[22]=0.3103524510337850;   c[23]=0.6365024991213990;
  return;
}

__device__ void jacobianoP2(type_fd* x, type_fd* y, int* tt, int th, type_fd* De){
  /*De e' lo jacobiamo della trsformazione che mi porta i t_id-esimo
    elemento nell'elemento di riferimento
           __                      __
          | y_3 - y_1      y_1 - y_2 |
    De =  |                          |
          | x_1 - x_3      x_2 - x_1 |
          |__                      __|
  */

  // 6 e' il numero di punti per 'riga'
  De[0] = y[tt[th * 6 + 2]] - y[tt[th * 6 + 0]]; // De[0][0]
  De[1] = y[tt[th * 6 + 0]] - y[tt[th * 6 + 1]]; // De[0][1]
  De[2] = x[tt[th * 6 + 0]] - x[tt[th * 6 + 2]]; // De[1][0]
  De[3] = x[tt[th * 6 + 1]] - x[tt[th * 6 + 0]]; // De[1][1]

  return;
}

__device__ void jacobianoP1(type_fd* x, type_fd* y, int* tt, int th, type_fd* De){
  /*De e' lo jacobiamo della trsformazione che mi porta i t_id-esimo
    elemento nell'elemento di riferimento
           __                      __
          | y_3 - y_1      y_1 - y_2 |
    De =  |                          |
          | x_1 - x_3      x_2 - x_1 |
          |__                      __|
  */

  // 6 e' il numero di punti per 'riga'
  De[0] = y[tt[th * 3 + 2]] - y[tt[th * 3 + 0]]; // De[0][0]
  De[1] = y[tt[th * 3 + 0]] - y[tt[th * 3 + 1]]; // De[0][1]
  De[2] = x[tt[th * 3 + 0]] - x[tt[th * 3 + 2]]; // De[1][0]
  De[3] = x[tt[th * 3 + 1]] - x[tt[th * 3 + 0]]; // De[1][1]

  return;
}


__device__ void load_tt(int*tt, int* tt_h, int tri, int th){
  
  int i;
  // porto in memori shared la tabella di connettivita'
  for(i = 0; i < 6; i++){
    tt[6 * th + i] = tt_h[6 * tri + i];
  } 
}

__device__ void load_t(int*tt, int* tt_h, int tri, int th){
  
  int i;
  // porto in memori shared la tabella di connettivita'
  for(i = 0; i < 3; i++){
    tt[3 * th + i] = tt_h[3 * tri + i];
  } 
}

__device__ void Prodotto_Matrici(type_fd* A, type_fd* B, type_fd* C, int r_A, int c_A, int r_B, int c_B) {
  /* funzione che esegue il prodotto matriciale A * B = C
     con le matrici memorizzate in formato array mettendo 
     in successione le righe

     Attenzione in caso i dimensioni errate non viene effettuato


     il prodotto e NON viene stampato nessun messaggio di errore
  */

  if ( c_A != r_B){
    return;
      }

  int i,j,k;
  type_fd somma, a, b;
  
  for(i = 0; i < r_A; ++i){
    for(j = 0; j < c_B; ++j){
      somma = 0;
      for(k = 0; k < c_A; ++k){
	a = A[i * c_A + k];
	b = B[k * c_B + j];
	somma += a * b;
      }
      C[i * c_B + j] = somma;
    }
  }
	return;
}


__device__ type_fd compute_exact_sol_u_x(type_fd x, type_fd y, type_fd t){
  type_fd uexact_x;
  uexact_x = -y*cos(t);
  return uexact_x;
}

__device__ type_fd compute_exact_sol_u_y(type_fd x, type_fd y, type_fd t){
  type_fd uexact_y;
  uexact_y = x*cos(t);
  return uexact_y;
}

__device__ type_fd compute_exact_sol_p(type_fd x, type_fd y, type_fd x_nfx, type_fd y_nfx, type_fd t){
  type_fd pexact;
  pexact = sin(t)*(sin(x)*sin(y) - sin(x_nfx)*sin(y_nfx));
  return pexact;
}

__device__ type_fd compute_exact_sol_rho(type_fd at_const, type_fd x, type_fd y, type_fd t){
  type_fd rhoexact;
  rhoexact = 2.0 + cos(sin(t))*x + sin(sin(t))*y + at_const*exp(-(x*x)-(y*y));
  return rhoexact;
}

__global__ void rho_P0projection(type_fd* x, type_fd* y, int n, type_fd* xm, type_fd* ym, int nbseg_x, int nbseg_y, type_fd time, type_fd at_const, type_fd* proj_rho){
  int num = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ type_fd base_rhoP0[4 * 2];
  __shared__ type_fd coef_rhoP0[4];

  if (num % blockDim.x == 0){
    set_base_rhoP0( base_rhoP0 );
    set_coef_rhoP0( coef_rhoP0 );
  }

  __syncthreads();

  if (num < n){
    type_fd Dx, Dy, xin, yin, rho_ex;
    int k;
    
    Dx = (xm[1]-xm[0])/nbseg_x;
    Dy = (ym[1]-ym[0])/nbseg_y;

    for (k = 0; k < 4; k++){
      xin = x[num] - Dx/2 + base_rhoP0[k*2 + 0]*Dx;
      yin = y[num] - Dy/2 + base_rhoP0[k*2 + 1]*Dy;
      rho_ex = compute_exact_sol_rho(at_const, xin, yin, time);
      proj_rho[num] =  proj_rho[num] + coef_rhoP0[k]*rho_ex;
    }
  }
  
}

__global__ void norm_L2ex_1(type_fd* x, type_fd* y, int* tt_h, int nt, type_fd time, int nfx, type_fd* vec,  type_fd* norm){
  
  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco

    

  __shared__ type_fd base_L2ex_1[12 * 3];
  __shared__ type_fd coef_L2ex[12];
  __shared__ type_fd coor_L2ex[2 * 12];
  __shared__ int  tt[3 * BLOCKDIM];

  if (tri % blockDim.x == 0){
    set_base_L2ex_1( base_L2ex_1 );
    set_coef_L2ex( coef_L2ex );
    set_coor_L2ex( coor_L2ex);
  }

  __syncthreads();

  if (tri < nt){
    type_fd De[2 * 2];
    type_fd Je;
    int k, tri1;
    type_fd xin, yin, p_ex, normloc = 0.0, sol = 0.0;

    // porto in memori shared la tabella di connettivita'
    load_t(tt, tt_h, tri, th);
    
    // calcolo il determinate di De
    
    jacobianoP1(x, y, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);

    tri1 = tt[th * 3 + 0];
    
    for (k = 0; k < 12; k++){
    xin  = x[tri1] + coor_L2ex[k*2 + 0]*De[1*2 + 1] - coor_L2ex[k*2 + 1]*De[1*2 + 0];
    yin  = y[tri1] - coor_L2ex[k*2 + 0]*De[0*2 + 1] + coor_L2ex[k*2 + 1]*De[0*2 + 0];
    p_ex = compute_exact_sol_p(xin, yin, x[nfx], y[nfx], time);
    sol  = base_L2ex_1[k*3 + 0] * vec[tt[th * 3 + 0]] + base_L2ex_1[k*3 + 1] * vec[tt[th * 3 + 1]] + base_L2ex_1[k*3 + 2] * vec[tt[th * 3 + 2]];
    normloc += coef_L2ex[k]*(p_ex - sol)*(p_ex - sol);
    }
    norm[tri] = normloc * Je;
  }

}

__global__ void norm_L2ex_2(type_fd* x, type_fd* y, int* tt_h, int nt, type_fd time, int nfx, type_fd* vec, int comp, type_fd* norm){
  
  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco

    

  __shared__ type_fd base_L2ex_2[12 * 6];
  __shared__ type_fd coef_L2ex[12];
  __shared__ type_fd coor_L2ex[2 * 12];
  __shared__ int  tt[6 * BLOCKDIM];

  if (tri % blockDim.x == 0){
    set_base_L2ex_2( base_L2ex_2 );
    set_coef_L2ex( coef_L2ex );
    set_coor_L2ex( coor_L2ex);
  }

  __syncthreads();

  if (tri < nt){
    type_fd De[2 * 2];
    type_fd Je;
    int k, tri1;
    type_fd xin, yin, u_ex, normloc = 0.0, sol = 0.0;

    // porto in memori shared la tabella di connettivita'
    load_tt(tt, tt_h, tri, th);
    
    // calcolo il determinate di De
    
    jacobianoP2(x, y, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);


    tri1 = tt[th * 6 + 0];
    
    for (k = 0; k < 12; k++){
    xin  = x[tri1] + coor_L2ex[k*2 + 0]*De[1*2 + 1] - coor_L2ex[k*2 + 1]*De[1*2 + 0];
    yin  = y[tri1] - coor_L2ex[k*2 + 0]*De[0*2 + 1] + coor_L2ex[k*2 + 1]*De[0*2 + 0];
    if (comp == 1)
      u_ex = compute_exact_sol_u_x(xin, yin, time);
    else if (comp == 2)
      u_ex = compute_exact_sol_u_y(xin, yin, time);
    
    sol  = base_L2ex_2[k*6 + 0] * vec[tt[th*6 + 0]] + base_L2ex_2[k*6 + 1] * vec[tt[th*6 + 1]] + base_L2ex_2[k*6 + 2] * vec[tt[th*6 + 2]] + base_L2ex_2[k*6 + 3] * vec[tt[th*6 + 3]] + base_L2ex_2[k*6 + 4] * vec[tt[th*6 + 4]] + base_L2ex_2[k*6 + 5] * vec[tt[th*6 + 5]];
    normloc += coef_L2ex[k]*(u_ex - sol)*(u_ex - sol);
    }
    
    norm[tri] = normloc * Je;
  }

}

__global__ void norm_L2_1(type_fd* x, type_fd* y, int* tt_h, int nt, type_fd* vec, type_fd* norm){
  
  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco

    

  __shared__ type_fd base_L2_1[4 * 3];
  __shared__ type_fd coef_L2_1[4];
  __shared__ int  tt[3 * BLOCKDIM];

  if (tri % blockDim.x == 0){
    set_coef_L2_1( coef_L2_1 );
    set_base_L2_1( base_L2_1 );

  }

  __syncthreads();

  if (tri < nt){
    type_fd De[2 * 2];
    type_fd Je;
    int  k;
    type_fd normloc = 0.0;
    type_fd carre[4] = {0.0};
    type_fd tmp[3] = {0.0}; 

    // porto in memori shared la tabella di connettivita'
    load_t(tt, tt_h, tri, th);

    for (k = 0; k < 3; k++)
      tmp[k] = vec[tt[th * 3 + k]];
    
    // calcolo il determinate di De
    
    jacobianoP1(x, y, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);
    
    Prodotto_Matrici(base_L2_1, tmp, carre, 4, 3, 3, 1);

    for (k = 0; k < 4; k++)
      normloc += coef_L2_1[k]*carre[k]*carre[k];
    
    norm[tri] = normloc * Je;
  }

}


__global__ void norm_L2_2(type_fd* x, type_fd* y, int* tt_h, int nt, type_fd* vec, type_fd* norm){
  
  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco

  __shared__ type_fd base_L2_2[7 * 6];
  __shared__ type_fd coef_L2_2[7];
  __shared__ int  tt[6 * BLOCKDIM];

  if (tri % blockDim.x == 0){
    set_coef_L2_2( coef_L2_2 );
    set_base_L2_2( base_L2_2 );
  }


  __syncthreads();

  if (tri < nt){
    type_fd De[2 * 2];
    type_fd Je;
    int  k;
    type_fd normloc = 0.0;
    type_fd carre[7] = {0.0};
    type_fd tmp[6] = {0.0}; 

    // porto in memori shared la tabella di connettivita'
    load_tt(tt, tt_h, tri, th);

    for (k = 0; k < 6; k++){
      tmp[k] = vec[tt[th * 6 + k]];
    }
    

    // if (tri == 0){
    //   printf("here \n");
    //   for (k = 0; k < 6*nt; k++)
    // 	printf("%d ", tt[k]);
    //   printf("\n"  );
    // }

    // printf("hello %d %d %d %d %d  \n", tri, normloc, tt_h[6 * tri + 0], tt_h[6 * tri + 1], tt_h[6 * tri + 2], tt_h[6 * tri + 3], tt_h[6 * tri + 4]);
    // calcolo il determinate di De
    
    jacobianoP2(x, y, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);

    
    Prodotto_Matrici(base_L2_2, tmp, carre, 7, 6, 6, 1);

    for (k = 0; k < 7; k++)
      normloc += coef_L2_2[k]*carre[k]*carre[k];

    // if (tri == 0){
    //   for (k = 0; k < 6; k++)
    // 	printf("%d %f \n", tt[th * 6 + k], tmp[k]);
    //   printf(" \n");
    // }
    
    // printf("tri %d %g \n", tri, normloc);
    normloc = normloc * Je;
    // printf("tri %d %g \n", tri, normloc);
    norm[tri] = normloc;
  }

  
}
