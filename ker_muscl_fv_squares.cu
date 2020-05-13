#include <stdio.h>
#include <math.h>
typedef double    type_fd;

__device__ int my_ceil(type_fd r){
  int n = ceil( (double) r);
  
  return n;
}


__device__ type_fd compute_exact_sol_u_x(int BENCHMARK_CASE, type_fd x, type_fd y, type_fd t){
  type_fd uexact_x;

  if (BENCHMARK_CASE == 1)/*EXAC*/
    uexact_x = -y*cos(t);
  else
    uexact_x = 0.0;

  return uexact_x;
}

__device__ type_fd compute_exact_sol_u_y(int BENCHMARK_CASE,type_fd x, type_fd y, type_fd t){
  type_fd uexact_y;

  if (BENCHMARK_CASE == 1)/*EXAC*/
    uexact_y = x*cos(t);
  else
    uexact_y = 0.0;
  
  return uexact_y;
}

__device__ type_fd compute_exact_sol_rho(int BENCHMARK_CASE, type_fd at_const, type_fd x, type_fd y, type_fd t){
  type_fd rhoexact;
 
  if (BENCHMARK_CASE == 1)/*EXAC*/
    rhoexact = 2.0 + cos(sin(t))*x + sin(sin(t))*y + at_const*exp(-(x*x) -(y*y));
  else
    rhoexact = 0.0;

  return rhoexact;
}


__device__ type_fd limiter_function(type_fd r){
  type_fd value;
  
  // /*Van Leer*/
  // if (r<0.0)
  //   value = 0.0;
  // else
  //   value = 2*r/(1.0+r);

  /*MinMod*/
  type_fd m = 1.0;
  value = 0.0; 
  if (r < m)
    m = r;
  // m = min(r, 1.0)
  if (m > 0.0)
    value = m;
  // value = max(0.0, m)
  
  // /*modified Van Leer*/
  // type_fd tau = 12./7.;
  // if (r <= 0.0){
  //   value = 0.0;
  // } else if (r <= 1.0){
  //   value = (r+(tau-1.0)*r)/(1.0+(tau-1.0)*r);
  // } else{
  //   value = (r+(tau-1.0)*r)/((tau-1.0)+r); 
  // }
  
  return value; 
}

__device__ type_fd phi(type_fd num, type_fd den, type_fd ep){
  type_fd frac;
  if(abs(den) > ep)
    frac = limiter_function(num/den);
  else
    frac = 0.0;
  
  return frac;
}

__global__ void update_muscl_fv_squares(int BENCHMARK_CASE, type_fd *ubar_x,type_fd *ubar_y, type_fd *rhon, type_fd *p1_x, type_fd *p1_y, type_fd *xm, type_fd *ym, int nbseg_x, int nbseg_y, int n, type_fd Dt, type_fd rho_t, type_fd u_t, type_fd epsilon, type_fd beta, type_fd at_const, type_fd* deltarho){
  
  const int ID = threadIdx.x+blockIdx.x*blockDim.x;

  int I, J, index, counter = 0;

  int neude = 10003;
  int print = 0;

  type_fd Dx = (xm[1]-xm[0])/nbseg_x;
  type_fd Dy = (ym[1]-ym[0])/nbseg_y;

  type_fd RHOGP = 0.,RHODM= 0., RHOHM= 0., RHOBP= 0.,VALPHI= 0.,VALPHIINV= 0.,FLUX= 0.,FLUXLOC = 0.;
  type_fd x, y, rho_out, solex_rho, solex_ub_x, solex_ug_y, solex_uh_x, solex_ud_y, solex_u_x, solex_u_y, uint_x, uint_y;

  if (ID < n){

    /****************************************/
    /*VERTICAL LEFT interfaces --- ID = NUMD*/
    /****************************************/

    I = (int) (ID)%(nbseg_x+1);
    J = my_ceil((type_fd) (ID+1)/(nbseg_x+1));

    if (I==0){
      if ( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2  || BENCHMARK_CASE == 3 ){
	/*_______BORDER LEFT VOLUMES_______*/
	if (2<=J && J<= nbseg_y){
	  /*non corner boundary cell*/
	  x = p1_x[ID];
	  y = p1_y[ID];
      
	  solex_rho = compute_exact_sol_rho(BENCHMARK_CASE, at_const, x,y,rho_t);
	  solex_u_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y,u_t);
	  solex_ub_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y-Dy/2.,u_t);
	  solex_uh_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y+Dy/2.,u_t);

	  uint_x= (solex_ub_x+solex_uh_x+solex_u_x)/3.;
      
	  if(uint_x<0){
	    FLUXLOC = Dy*uint_x*rhon[ID];
	    counter+=1;
	  }
	  else{
	    FLUXLOC = Dy*uint_x*solex_rho;
	    counter+=1;
	  }
	  FLUX-=2*FLUXLOC; /*considering half a cell*/      
	}else{
	  /*corner cells*/
	  x = p1_x[ID];
	  y = p1_y[ID];
	  solex_rho = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y,rho_t); 
	  solex_u_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y,u_t);
	  if(ID==0){ /*low left corner*/
	    solex_uh_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y+Dy/2.,u_t);
	    uint_x = (2*solex_uh_x+solex_u_x)/3.;
      
	    if(uint_x<0){
	      FLUXLOC = Dy*uint_x*rhon[ID];
	      counter+=1;
	    }
	    else{
	      FLUXLOC = Dy*uint_x*solex_rho;
	      counter+=1;
	    }      
	  }else{ /*up left corner*/
	    solex_ub_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y-Dy/2.,u_t);
	    uint_x = (2*solex_ub_x+solex_u_x)/3.;
      
	    if(uint_x<0){
	      FLUXLOC = Dy*uint_x*rhon[ID];
	      counter+=1;
	    }
	    else{
	      FLUXLOC = Dy*uint_x*solex_rho;
	      counter+=1;
	    }  
	  }
	  FLUX-=2*FLUXLOC; /*shouldnt it be *4?*/
	}
      }
    }
    else{
      /*_______INTERNAL EDGES_______*/
      index = (I-1)*(nbseg_y+1)+(J-1);
      if (I==1){
	/*BC on the left side of the rectangular domain*/
	/*here ID coincides with NUMD next to a boundary cell*/
	if(ubar_x[index]>0){
	  RHOGP = rhon[ID-1];
	  if(BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	    x = p1_x[ID-1];
	    y = p1_y[ID-1];
	    rho_out = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x-Dx,y,rho_t);
	    RHOGP = rhon[ID-1]+0.5*(beta*(rhon[ID-1]-rho_out)+(1-beta)*(rhon[ID]-rhon[ID-1]));
	  }
	  counter+=1;
	  FLUXLOC=Dy*ubar_x[index]*RHOGP;
	}else{
	  VALPHI = phi((rhon[ID+1]-rhon[ID]),(rhon[ID]-rhon[ID-1]),epsilon);
	  VALPHIINV = phi((rhon[ID]-rhon[ID-1]),(rhon[ID+1]-rhon[ID]),epsilon);
	  RHODM = rhon[ID]-0.5*(beta*VALPHIINV*(rhon[ID+1]-rhon[ID])+(1-beta)*VALPHI*(rhon[ID]-rhon[ID-1]));
	  FLUXLOC=Dy*ubar_x[index]*RHODM;
	  counter+=1;
	}    
      }else if(I == nbseg_x){
	if(ubar_x[index]>0){
	  VALPHI=phi((rhon[ID]-rhon[ID-1]),(rhon[ID-1]-rhon[ID-2]),epsilon);
	  VALPHIINV=phi((rhon[ID-1]-rhon[ID-2]),(rhon[ID]-rhon[ID-1]),epsilon);
	  RHOGP=rhon[ID-1]+0.5*(beta*VALPHI*(rhon[ID-1]-rhon[ID-2])+(1-beta)*VALPHIINV*(rhon[ID]-rhon[ID-1]));
	  FLUXLOC=Dy*ubar_x[index]*RHOGP;
	  counter+=1;
	}else{
	  RHODM = rhon[ID];
	  if( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	    x = p1_x[ID];
	    y = p1_y[ID];
	    rho_out = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x+Dx,y,rho_t);
	    RHODM = rhon[ID]-0.5*(beta*(rho_out-rhon[ID])+(1-beta)*(rhon[ID]-rhon[ID-1]));
	    //counter+=1;
	  }
	  counter+=1;
	  FLUXLOC=Dy*ubar_x[index]*RHODM;
	}
      }else{
	if(ubar_x[index]>0){
	  VALPHI=phi((rhon[ID]-rhon[ID-1]),(rhon[ID-1]-rhon[ID-2]),epsilon);
	  VALPHIINV=phi((rhon[ID-1]-rhon[ID-2]),(rhon[ID]-rhon[ID-1]),epsilon);
	  RHOGP=rhon[ID-1]+0.5*(beta*VALPHI*(rhon[ID-1]-rhon[ID-2])+(1-beta)*VALPHIINV*(rhon[ID]-rhon[ID-1]));
	  if (print == 1){
	    if (ID == neude)
	      printf("VALPHI = %18.16f, VALPHIINV = %18.16f, RHOGP = %18.16f, Dy = %18.16f \n", VALPHI, VALPHIINV, RHOGP, Dy);
	  }
	  FLUXLOC=Dy*ubar_x[index]*RHOGP;
	  counter+=1;
	}
	else{
	  VALPHI=phi((rhon[ID+1]-rhon[ID]),(rhon[ID]-rhon[ID-1]),epsilon);
	  VALPHIINV=phi((rhon[ID]-rhon[ID-1]),(rhon[ID+1]-rhon[ID]),epsilon);
	  RHODM = rhon[ID]-0.5*(beta*VALPHIINV*(rhon[ID+1]-rhon[ID])+(1-beta)*VALPHI*(rhon[ID]-rhon[ID-1]));
	  FLUXLOC=Dy*ubar_x[index]*RHODM;
	  counter+=1;
	}
      }
      if (I==nbseg_x)
	FLUX-=2*FLUXLOC; /*considering half a cell on the border*/
      else
	FLUX-=FLUXLOC;      
    }

    if (print == 1){
      if (ID == neude){
	printf("Vertical left interface:\n");
	printf("FLUXLOC = %18.16f\n", FLUXLOC);
	printf("UINTER = %18.16f\n", ubar_x[index]);
	printf("FLUX = %18.16f\n", FLUX);
	printf("counter = %d\n", counter);
      }
    }

    FLUXLOC= 0;
    /****************************************/
    /*VERTICAL RIGHT interfaces --- ID = NUMG*/
    /****************************************/

    I = (int) (ID+1)%(nbseg_x+1);
    J = my_ceil((type_fd) (ID+1)/(nbseg_x+1));

    if (I==0){
      if(BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	/*_______BORDER RIGHT VOLUMES_______*/
	if (2<=J && J<= nbseg_y){
	  /*non corner boundary cell*/
	  x = p1_x[ID];
	  y = p1_y[ID];
      
	  solex_rho = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y,rho_t);     
	  solex_ub_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y-Dy/2.,u_t);
	  solex_uh_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y+Dy/2.,u_t);
	  solex_u_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y,u_t);

	  uint_x = (solex_ub_x+solex_uh_x+solex_u_x)/3.;
      
	  if(uint_x>0){
	    FLUXLOC = Dy*uint_x*rhon[ID];
	    counter+=1;
	  }
	  else{
	    FLUXLOC = Dy*uint_x*solex_rho;
	    counter+=1;
	  }
      
	  FLUX+=2*FLUXLOC; /*considering half a cell*/      
	}else{
	  /*corner cell*/
	  x = p1_x[ID];
	  y = p1_y[ID];
	  solex_rho = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y,rho_t); 
	  solex_u_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y,u_t);
	  if(ID==nbseg_x){ /*right lower corner*/
	    solex_uh_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y+Dy/2.,u_t);
	    uint_x = (2*solex_uh_x+solex_u_x)/3.;
      
	    if(uint_x>0){
	      FLUXLOC = Dy*uint_x*rhon[ID];
	      counter+=1;
	    }
	    else{
	      FLUXLOC = Dy*uint_x*solex_rho;
	      counter+=1;
	    }
      
	  }else{ /*right upper corner*/
	    solex_ub_x = compute_exact_sol_u_x(BENCHMARK_CASE,x,y-Dy/2.,u_t);    
	    uint_x = (2*solex_ub_x+solex_u_x)/3.;
      
	    if(uint_x>0){
	      FLUXLOC = Dy*uint_x*rhon[ID];
	      counter+=1;
	    }
	    else{
	      FLUXLOC = Dy*uint_x*solex_rho;
	      counter+=1;
	    }
	  }
	  FLUX+=2*FLUXLOC; /*shouldnt it be *4?*/
	}
      }
    }else{
      /*_______INTERNAL EDGES_______*/
      index = (I-1)*(nbseg_y+1)+(J-1);
      if (I==1){
	/*BC on the left side of the rectangular domain*/
	/*here ID is boundary cell*/

	// if (ID == neude){
	//   printf("UINTER =  %18.16f\n", ubar_x[index]);
	// }
	if(ubar_x[index]>0){
	  RHOGP = rhon[ID];
	  if( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	    x = p1_x[ID];
	    y = p1_y[ID];
	    rho_out = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x-Dx,y,rho_t);
	    RHOGP = rhon[ID]+0.5*((beta*(rhon[ID]-rho_out))+((1.0-beta)*(rhon[ID+1]-rhon[ID])));
	    //counter+=1;
	  }
	  counter+=1;
	  FLUXLOC=Dy*ubar_x[index]*RHOGP;
	}else{
	  VALPHI = phi((rhon[ID+2]-rhon[ID+1]),(rhon[ID+1]-rhon[ID]),epsilon);
	  VALPHIINV = phi((rhon[ID+1]-rhon[ID]),(rhon[ID+2]-rhon[ID+1]),epsilon);
	  RHODM = rhon[ID+1]-0.5*(beta*VALPHIINV*(rhon[ID+2]-rhon[ID+1])+(1-beta)*VALPHI*(rhon[ID+1]-rhon[ID]));
	  FLUXLOC=Dy*ubar_x[index]*RHODM;
	  counter+=1;
	  // if (ID == neude){
	  //   printf("VALPHI =  %18.16f\n", VALPHI);
	  //   printf("VALPHIINV =  %18.16f\n", VALPHIINV);
	  //   printf("RHODM =  %18.16f\n", RHODM);
	  // }
	}    
      }else if(I==nbseg_x){
	if(ubar_x[index]>0){
	  VALPHI=phi((rhon[ID+1]-rhon[ID]),(rhon[ID]-rhon[ID-1]),epsilon);
	  VALPHIINV=phi((rhon[ID]-rhon[ID-1]),(rhon[ID+1]-rhon[ID]),epsilon);
	  RHOGP=rhon[ID]+0.5*(beta*VALPHI*(rhon[ID]-rhon[ID-1])+(1-beta)*VALPHIINV*(rhon[ID+1]-rhon[ID]));
	  FLUXLOC=Dy*ubar_x[index]*RHOGP;
	  counter+=1;
	}else{
	  RHODM = rhon[ID+1];
	  if( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	    x = p1_x[ID+1];
	    y = p1_y[ID+1];
	    rho_out = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x+Dx,y,rho_t);
	    RHODM = rhon[ID+1]-0.5*(beta*(rho_out-rhon[ID+1])+(1-beta)*(rhon[ID+1]-rhon[ID]));
	    //counter+=1;
	  }
	  counter+=1;
	  FLUXLOC=Dy*ubar_x[index]*RHODM;
	}
      }else{
	if(ubar_x[index]>0){
	  VALPHI=phi((rhon[ID+1]-rhon[ID]),(rhon[ID]-rhon[ID-1]),epsilon);
	  VALPHIINV=phi((rhon[ID]-rhon[ID-1]),(rhon[ID+1]-rhon[ID]),epsilon);
	  RHOGP=rhon[ID]+0.5*(beta*VALPHI*(rhon[ID]-rhon[ID-1])+(1-beta)*VALPHIINV*(rhon[ID+1]-rhon[ID]));
	  FLUXLOC=Dy*ubar_x[index]*RHOGP;
	  counter+=1;
	}
	else{
	  VALPHI=phi((rhon[ID+2]-rhon[ID+1]),(rhon[ID+1]-rhon[ID]),epsilon);
	  VALPHIINV=phi((rhon[ID+1]-rhon[ID]),(rhon[ID+2]-rhon[ID+1]),epsilon);
	  RHODM = rhon[ID+1]-0.5*(beta*VALPHIINV*(rhon[ID+2]-rhon[ID+1])+(1-beta)*VALPHI*(rhon[ID+1]-rhon[ID]));
	  FLUXLOC=Dy*ubar_x[index]*RHODM;
	  counter+=1;
	}
      }
      if (I==1)
	FLUX+=2*FLUXLOC; /*considering half a cell on the border*/
      else
	FLUX+=FLUXLOC;    
    }

    if (print == 1){    
      if (ID == neude){
	printf("Vertical right interface:\n");
	printf("FLUXLOC = %18.16f\n", FLUXLOC);
	printf("UINTER = %18.16f\n", ubar_x[index]);
	printf("FLUX = %18.16f\n", FLUX);
	printf("counter = %d\n", counter);
      }
    }

    FLUXLOC= 0;
    /*****************************************/
    /*HORIZOLTAL LOW interfaces --- ID = NUMH*/
    /*****************************************/

    I = (int) (ID)%(nbseg_x+1)+1;
    J = (my_ceil((type_fd) (ID+1)/(nbseg_x+1))-1)%(nbseg_y+1);

    if (J==0){
      if(BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	/*_______LOWER BORDER VOLUMES_______*/
	if (2<=I && I<= nbseg_x){
	  /*non corner boundary cell*/
	  x = p1_x[ID];
	  y = p1_y[ID];
      
	  solex_rho = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y,rho_t);     
	  solex_ug_y = compute_exact_sol_u_y(BENCHMARK_CASE,x-Dx/2.,y,u_t);
	  solex_ud_y = compute_exact_sol_u_y(BENCHMARK_CASE,x+Dx/2.,y,u_t);
	  solex_u_y = compute_exact_sol_u_y(BENCHMARK_CASE,x,y,u_t);

	  uint_y = (solex_ug_y+solex_ud_y+solex_u_y)/3.;
      
	  if(uint_y<0){
	    FLUXLOC = Dx*uint_y*rhon[ID];
	    counter+=1;
	  }
	  else{
	    FLUXLOC = Dx*uint_y*solex_rho;
	    counter+=1;
	  }
	  FLUX-=2*FLUXLOC; /*considering half a cell*/      
	}else{
	  /*corner cells*/
	  x = p1_x[ID];
	  y = p1_y[ID];
	  solex_rho = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y,rho_t); 
	  solex_u_y = compute_exact_sol_u_y(BENCHMARK_CASE,x,y,u_t);
	  if(ID==0){ /*left lower corner*/    
	    solex_ud_y = compute_exact_sol_u_y(BENCHMARK_CASE,x+Dx/2.,y,u_t);	
	    uint_y = (2*solex_ud_y+solex_u_y)/3.;
      
	    if(uint_y<0){
	      FLUXLOC = Dx*uint_y*rhon[ID];
	      counter+=1;
	    }else{
	      FLUXLOC = Dx*uint_y*solex_rho;
	      counter+=1;
	    }      
	  }else{ /*right lower corner*/    
	    solex_ug_y = compute_exact_sol_u_y(BENCHMARK_CASE,x-Dx/2.,y,u_t);	
	    uint_y = (2*solex_ug_y+solex_u_y)/3.;
      
	    if(uint_y<0){
	      FLUXLOC = Dx*uint_y*rhon[ID];
	      counter+=1;
	    }
	    else{
	      FLUXLOC = Dx*uint_y*solex_rho;
	      counter+=1;
	    }
	  }
	  FLUX-=2*FLUXLOC; /*shouldnt it be *4?*/
	}
      }
    }
    else{
      /*_______INTERNAL EDGES_______*/
      index = (I-1)*(nbseg_y)+(J-1);
      if (J == 1){
	if(ubar_y[index]>0){
	  RHOBP = rhon[ID-(nbseg_x+1)];
	  if( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	    x = p1_x[ID-(nbseg_x+1)];
	    y = p1_y[ID-(nbseg_x+1)];
	    rho_out = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y-Dy,rho_t);
	    RHOBP = rhon[ID-(nbseg_x+1)] + 0.5*(beta*(rhon[ID-(nbseg_x+1)]-rho_out)+(1-beta)*(rhon[ID]-rhon[ID-(nbseg_x+1)]));
	    //counter+=1;
	  }
	  counter+=1;
	  FLUXLOC = Dx*ubar_y[index]*RHOBP;
	}else{
	  VALPHI = phi((rhon[ID+(nbseg_x+1)]-rhon[ID]),(rhon[ID]-rhon[ID-(nbseg_x+1)]),epsilon);
	  VALPHIINV = phi((rhon[ID]-rhon[ID-(nbseg_x+1)]),(rhon[ID+(nbseg_x+1)]-rhon[ID]),epsilon);
	  RHOHM = rhon[ID]-0.5*(beta*VALPHIINV*(rhon[ID+(nbseg_x+1)]-rhon[ID])+(1-beta)*VALPHI*(rhon[ID]-rhon[ID-(nbseg_x+1)]));
	  FLUXLOC = Dx*ubar_y[index]*RHOHM;
	  counter+=1;
	}
      }else if(J == nbseg_y){
	if (ubar_y[index]>0){
	  /*BC on the upper side of the rectangular domain*/
	  /*here ID coincides with NUMH and is a boundary cell*/
	  VALPHI = phi((rhon[ID]-rhon[ID-(nbseg_x+1)]),(rhon[ID-(nbseg_x+1)]-rhon[ID-2*(nbseg_x+1)]),epsilon);
	  VALPHIINV = phi((rhon[ID-(nbseg_x+1)]-rhon[ID-2*(nbseg_x+1)]),(rhon[ID]-rhon[ID-(nbseg_x+1)]),epsilon);
	  RHOBP = rhon[ID-(nbseg_x+1)]+0.5*(beta*VALPHI*(rhon[ID-(nbseg_x+1)]-rhon[ID-2*(nbseg_x+1)])+(1-beta)*VALPHIINV*(rhon[ID]-rhon[ID-(nbseg_x+1)]));
	  FLUXLOC = Dx*ubar_y[index]*RHOBP;
	  counter+=1;
	}else{
	  /*BC on the upper side of the rectangular domain*/
	  /*here ID coincides with NUMH and is a boundary cell*/
	  x = p1_x[ID];
	  y = p1_y[ID];
	  RHOHM = rhon[ID];
	  if ( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	    rho_out = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y+Dy,rho_t);
	    RHOHM = rhon[ID] - 0.5*(beta*(rho_out-rhon[ID])+(1-beta)*(rhon[ID]-rhon[ID-(nbseg_x+1)]));
	    //counter+=1;
	  }
	  counter+=1;
	  FLUXLOC=Dx*ubar_y[index]*RHOHM;
	}
      }else{
	if(ubar_y[index]>0){
	  VALPHI = phi((rhon[ID]-rhon[ID-(nbseg_x+1)]),(rhon[ID-(nbseg_x+1)]-rhon[ID-2*(nbseg_x+1)]),epsilon);
	  VALPHIINV = phi((rhon[ID-(nbseg_x+1)]-rhon[ID-2*(nbseg_x+1)]),(rhon[ID]-rhon[ID-(nbseg_x+1)]),epsilon);
	  RHOBP = rhon[ID-(nbseg_x+1)]+0.5*(beta*VALPHI*(rhon[ID-(nbseg_x+1)]-rhon[ID-2*(nbseg_x+1)])+(1-beta)*VALPHIINV*(rhon[ID]-rhon[ID-(nbseg_x+1)]));
	  FLUXLOC = Dx*ubar_y[index]*RHOBP;
	  counter+=1;
	}else{
	  VALPHI = phi((rhon[ID+(nbseg_x+1)]-rhon[ID]),(rhon[ID]-rhon[ID-(nbseg_x+1)]),epsilon);
	  VALPHIINV = phi((rhon[ID]-rhon[ID-(nbseg_x+1)]),(rhon[ID+(nbseg_x+1)]-rhon[ID]),epsilon);
	  RHOHM = rhon[ID]-0.5*(beta*VALPHIINV*(rhon[ID+(nbseg_x+1)]-rhon[ID])+(1-beta)*VALPHI*(rhon[ID]-rhon[ID-(nbseg_x+1)]));
	  FLUXLOC = Dx*ubar_y[index]*RHOHM;
	  counter+=1;
	}
      }

      if (J == nbseg_y)
	FLUX-=2*FLUXLOC; 
      else
	FLUX-=FLUXLOC;    
    }

    if (print == 1){    
      if (ID == neude){
	printf("Horizontal low interface:\n");
	printf("FLUXLOC = %18.16f\n", FLUXLOC);
	printf("VINTER = %18.16f\n", ubar_y[index]);
	printf("FLUX = %18.16f\n", FLUX);
	printf("counter = %d\n", counter);
      }
    }
  
    FLUXLOC= 0;
    /****************************************/
    /*HORIZOLTAL UP interfaces --- ID = NUMB*/
    /****************************************/
  
    I = (int) (ID)%(nbseg_x+1)+1;
    J = (my_ceil((type_fd) (ID+1)/(nbseg_x+1)))%(nbseg_y+1);

    if (J==0){
      if( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	/*_______UPPER BORDER VOLUMES_______*/
	if (2<=I && I<= nbseg_x){
	  /*non corner boundary cell*/
	  x = p1_x[ID];
	  y = p1_y[ID];
      
	  solex_rho = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y,rho_t);     
	  solex_ug_y = compute_exact_sol_u_y(BENCHMARK_CASE,x-Dx/2.,y,u_t);
	  solex_ud_y = compute_exact_sol_u_y(BENCHMARK_CASE,x+Dx/2.,y,u_t);
	  solex_u_y = compute_exact_sol_u_y(BENCHMARK_CASE,x,y,u_t);

	  uint_y = (solex_ug_y+solex_ud_y+solex_u_y)/3.;
      
	  if(uint_y>0){
	    FLUXLOC = Dx*uint_y*rhon[ID];
	    counter+=1;
	  }else{
	    FLUXLOC = Dx*uint_y*solex_rho;
	    counter+=1;
	  }
      
	  FLUX+=2*FLUXLOC; /*considering half a cell*/      
	}else{
	  /*corner cells*/
	  x = p1_x[ID];
	  y = p1_y[ID];
	  solex_rho = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y,rho_t); 
	  solex_u_y = compute_exact_sol_u_y(BENCHMARK_CASE,x,y,u_t);
	  if(ID == ((nbseg_x+1)*nbseg_y)){ /*left upper corner*/    
	    solex_ud_y = compute_exact_sol_u_y(BENCHMARK_CASE,x+Dx/2.,y,u_t);	
	    uint_y = (2*solex_ud_y+solex_u_y)/3.;
      
	    if(uint_y>0){
	      FLUXLOC = Dx*uint_y*rhon[ID];
	      counter+=1;
	    }
	    else{
	      FLUXLOC = Dx*uint_y*solex_rho;
	      counter+=1;
	    }     
	  }else{ /*right upper corner*/    
	    solex_ug_y = compute_exact_sol_u_y(BENCHMARK_CASE,x-Dx/2.,y,u_t);	
	    uint_y = (2*solex_ug_y+solex_u_y)/3.;
      
	    if(uint_y>0){
	      FLUXLOC = Dx*uint_y*rhon[ID];
	      counter+=1;
	    }
	    else{
	      FLUXLOC = Dx*uint_y*solex_rho;
	      counter+=1;
	    }
	  }
	  FLUX+=2*FLUXLOC; /*shouldnt it be *4?*/
	}
      }
    }else{
      /*_______INTERNAL EDGES_______*/
      index = (I-1)*(nbseg_y)+(J-1);
      if (J == 1){
	// if (ID == neude)
	//   printf("VINTER = %18.16f \n", ubar_y[index]);
	if(ubar_y[index]>0){
	  RHOBP = rhon[ID];
	  if( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	    x = p1_x[ID];
	    y = p1_y[ID];
	    rho_out = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y-Dy,rho_t);
	    RHOBP = rhon[ID] + 0.5*(beta*(rhon[ID]-rho_out)+(1-beta)*(rhon[ID+(nbseg_x+1)]-rhon[ID]));
	    //counter+=1;
	  }
	  counter+=1;
	  FLUXLOC = Dx*ubar_y[index]*RHOBP;
	}else{
	  VALPHI = phi((rhon[ID+2*(nbseg_x+1)]-rhon[ID+(nbseg_x+1)]),(rhon[ID+(nbseg_x+1)]-rhon[ID]),epsilon);
	  VALPHIINV = phi((rhon[ID+(nbseg_x+1)]-rhon[ID]),(rhon[ID+2*(nbseg_x+1)]-rhon[ID+(nbseg_x+1)]),epsilon);
	  RHOHM = rhon[ID+(nbseg_x+1)]-0.5*(beta*VALPHIINV*(rhon[ID+2*(nbseg_x+1)]-rhon[ID+(nbseg_x+1)])+(1-beta)*VALPHI*(rhon[ID+(nbseg_x+1)]-rhon[ID]));
	  FLUXLOC = Dx*ubar_y[index]*RHOHM;
	  counter+=1;
	}
      }else if(J == nbseg_y){
	if (ubar_y[index]>0){
	  VALPHI = phi((rhon[ID+(nbseg_x+1)]-rhon[ID]),(rhon[ID]-rhon[ID-(nbseg_x+1)]),epsilon);
	  VALPHIINV = phi((rhon[ID]-rhon[ID-(nbseg_x+1)]),(rhon[ID+(nbseg_x+1)]-rhon[ID]),epsilon);
	  RHOBP = rhon[ID]+0.5*(beta*VALPHI*(rhon[ID]-rhon[ID-(nbseg_x+1)])+(1-beta)*VALPHIINV*(rhon[ID+(nbseg_x+1)]-rhon[ID]));
	  FLUXLOC = Dx*ubar_y[index]*RHOBP;
	  counter+=1;
	}else{
	  /*BC on the upper side of the rectangular domain*/
	  /*here ID coincides with NUMB next to a boundary cell*/
	  RHOHM = rhon[ID+(nbseg_x+1)];
	  if ( BENCHMARK_CASE == 1 || BENCHMARK_CASE == 2 || BENCHMARK_CASE == 3){
	    x = p1_x[ID+(nbseg_x+1)];
	    y = p1_y[ID+(nbseg_x+1)];
	    rho_out = compute_exact_sol_rho(BENCHMARK_CASE,at_const,x,y+Dy,rho_t);
	    RHOHM = rhon[ID+(nbseg_x+1)] - 0.5*(beta*(rho_out-rhon[ID+(nbseg_x+1)])+(1-beta)*(rhon[ID+(nbseg_x+1)]-rhon[ID]));
	    //counter+=1;
	  }
	  counter+=1;
	  FLUXLOC = Dx*ubar_y[index]*RHOHM;
	}
      }else{
	if(ubar_y[index]>0){
	  VALPHI = phi((rhon[ID+(nbseg_x+1)]-rhon[ID]),(rhon[ID]-rhon[ID-(nbseg_x+1)]),epsilon);
	  VALPHIINV = phi((rhon[ID]-rhon[ID-(nbseg_x+1)]),(rhon[ID+(nbseg_x+1)]-rhon[ID]),epsilon);
	  RHOBP = rhon[ID]+0.5*(beta*VALPHI*(rhon[ID]-rhon[ID-(nbseg_x+1)])+(1-beta)*VALPHIINV*(rhon[ID+(nbseg_x+1)]-rhon[ID]));
	  FLUXLOC = Dx*ubar_y[index]*RHOBP;
	  counter+=1;
	}else{
	  VALPHI = phi((rhon[ID+2*(nbseg_x+1)]-rhon[ID+(nbseg_x+1)]),(rhon[ID+(nbseg_x+1)]-rhon[ID]),epsilon);
	  VALPHIINV = phi((rhon[ID+(nbseg_x+1)]-rhon[ID]),(rhon[ID+2*(nbseg_x+1)]-rhon[ID+(nbseg_x+1)]),epsilon);
	  RHOHM = rhon[ID+(nbseg_x+1)]-0.5*(beta*VALPHIINV*(rhon[ID+2*(nbseg_x+1)]-rhon[ID+(nbseg_x+1)])+(1-beta)*VALPHI*(rhon[ID+(nbseg_x+1)]-rhon[ID]));
	  FLUXLOC = Dx*ubar_y[index]*RHOHM;
	  counter+=1;
	}
      }
    
      if (J == 1)
	FLUX+=2*FLUXLOC; 
      else
	FLUX+=FLUXLOC;    
    }

    if (print == 1){
      if (ID == neude){
	printf("Horizontal upper interface:\n");
	printf("FLUXLOC = %18.16f\n", FLUXLOC);
	printf("VINTER = %18.16f\n", ubar_y[index]);
	printf("FLUX = %18.16f\n", FLUX);
	printf("counter = %d\n", counter);
      }
    }
    
    deltarho[ID]=-(Dt/(Dx*Dy))*FLUX;  
  }
  return;
}
