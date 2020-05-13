#include <stdio.h>
#include <math.h>

typedef unsigned long long int    type_uint;
typedef double                    type_fd;

#define BLOCKDIM 256 


__device__ type_fd set_coef3(){

  return 1.0/6.0;
}

__device__ void set_coef4(type_fd* c){
  c[0] = -0.281250000000000; c[1] = 0.260416666666667; c[2] = 0.260416666666667; c[3] = 0.260416666666667;

  return;
}

__device__ void set_coef7(type_fd* c){
  c[0] = 0.1125;            c[1] = 0.066197076394253; c[2] = 0.066197076394253; c[3] = 0.066197076394253; 
  c[4] = 0.062969590272414; c[5] = 0.062969590272414; c[6] = 0.062969590272414;
  
  return;
}

__device__ void set_coor12(type_fd* c){
  c[0]  = 0.0630890144915020; c[1]  = 0.0630890144915020;
  c[2]  = 0.8738219710169960; c[3]  = 0.0630890144915020;
  c[4]  = 0.0630890144915020; c[5]  = 0.8738219710169960;
  c[6]  = 0.2492867451709100; c[7]  = 0.2492867451709100;
  c[8]  = 0.5014265096581800; c[9]  = 0.2492867451709100;
  c[10] = 0.2492867451709100; c[11] = 0.5014265096581800;
  c[12] = 0.0531450498448160; c[13] = 0.3103524510337850;
  c[14] = 0.6365024991213990; c[15] = 0.0531450498448160;
  c[16] = 0.0531450498448160; c[17] = 0.6365024991213990;
  c[18] = 0.3103524510337850; c[19] = 0.0531450498448160;
  c[20] = 0.6365024991213990; c[21] = 0.3103524510337850;
  c[22] = 0.3103524510337850; c[23] = 0.6365024991213990;
  
  return;
}

__device__ void set_coef12(type_fd* c){
  c[0] = 0.025422453185103; c[1] = 0.025422453185103; c[2] = 0.025422453185103; c[3] = 0.058393137863189;
  c[4] = 0.058393137863189; c[5] = 0.058393137863189; c[6] = 0.041425537809187; c[7] = 0.041425537809187;
  c[8] = 0.041425537809187; c[9] = 0.041425537809187; c[10]= 0.041425537809187; c[11] =  0.041425537809187;


  return;
}

__device__ void set_base1_4(type_fd* b){

  b[0] = 0.333333333333333; b[1]  = 0.333333333333333; b[2]  = 0.333333333333333;
  b[3] = 0.600000000000000; b[4]  = 0.200000000000000; b[5]  = 0.200000000000000;
  b[6] = 0.200000000000000; b[7]  = 0.600000000000000; b[8]  = 0.200000000000000;
  b[9] = 0.200000000000000; b[10] = 0.200000000000000; b[11] = 0.600000000000000;

  return;
}

__device__ void set_base2_4(type_fd* b){

  b[0]  = -0.111111111111111; b[1]  = -0.111111111111111; b[2]  = -0.111111111111111; b[3]  = 0.444444444444444; b[4]  = 0.444444444444444; b[5]  = 0.444444444444444;
  b[6]  =  0.120000000000000; b[7]  = -0.120000000000000; b[8]  = -0.120000000000000; b[9]  = 0.160000000000000; b[10] = 0.480000000000000; b[11] = 0.480000000000000;
  b[12] = -0.120000000000000; b[13] =  0.120000000000000; b[14] = -0.120000000000000; b[15] = 0.480000000000000; b[16] = 0.160000000000000; b[17] = 0.480000000000000;
  b[18] = -0.120000000000000; b[19] = -0.120000000000000; b[20] =  0.120000000000000; b[21] = 0.480000000000000; b[22] = 0.480000000000000; b[23] = 0.160000000000000;

  return;
}


__device__ void set_base1(type_fd* b){
  b[0] = 0;   b[1] = 0.5; b[2] =  0.5;                                                                                          
  b[3] = 0.5; b[4] = 0;   b[5] =  0.5;                                                                                         
  b[6] = 0.5; b[7] = 0.5; b[8] =  0;

  return;

}

__device__ void set_base1m(type_fd* b){
  b[0] = 0.3333333333333334; b[1] = 0.3333333333333333; b[2] =  0.3333333333333333;    
  b[3] = 0.0597158717897698; b[4] = 0.4701420641051151; b[5] =  0.4701420641051151;    
  b[6] = 0.4701420641051151; b[7] = 0.0597158717897698; b[8] =  0.4701420641051151;
  b[9] = 0.4701420641051151; b[10] = 0.4701420641051151; b[11] =  0.0597158717897698;
  b[12] = 0.7974269853530873; b[13] = 0.1012865073234563; b[14] =  0.1012865073234563;
  b[15] = 0.1012865073234563; b[16] = 0.7974269853530873; b[17] =  0.1012865073234563;
  b[18] = 0.1012865073234563; b[19] = 0.1012865073234563; b[20] =  0.7974269853530873;

  return;

}



__device__ void set_base21_2(type_fd* b){
  b[0]  =  1;    b[1] =  1;   b[2] = 0;    b[3] = 2;     b[4] = -2;    b[5] = -2;
  b[6]  = -1;    b[7] = -1;   b[8] = 0;    b[9] = 2;    b[10] = -2,    b[11] = 2;
  b[12] = -1;   b[13] =  1;  b[14] = 0;    b[15]= 0;     b[16] = 0;    b[17] = 0;

  return;
}



__device__ void set_base21_3(type_fd* b){

  b[0]   = 1;     b[1] = 0;      b[2] = 1;     b[3] = 2;    b[4] = -2;     b[5] = -2;
  b[6]  = -1;     b[7] = 0;      b[8] = 1;     b[9] = 0;    b[10] = 0;     b[11] = 0;
  b[12] = -1;    b[13] = 0;    b[14] = -1;    b[15] = 2;    b[16] = 2,    b[17] = -2;

  return;
}

__device__ void set_base2_2(type_fd* b){
  b[0]  = -1;   b[1] = 1;    b[2] = 0;   
  b[3]  = -1;   b[4] = 1;    b[5] = 0;  
  b[6]  = -1;   b[7] = 1;    b[8] = 0; 

  return;
}



__device__ void set_base2_3(type_fd* b){

  b[0] = -1;    b[1] = 0;     b[2] = 1;  
  b[3] = -1;    b[4] = 0;     b[5] = 1;  
  b[6] = -1;    b[7] = 0;     b[8] = 1; 

  return;
}


__device__ void set_base2m(type_fd* b){
  b[0] = -0.111111111111111;  b[1] = -0.111111111111111;  b[2] = -0.111111111111111;
  b[3] =  0.444444444444444;  b[4] =  0.444444444444444;  b[5] =  0.444444444444444;
  
  b[6] = -0.052583901102546;  b[7] = -0.028074943223079;  b[8] = -0.028074943223079;
  b[9] =  0.884134241764073;  b[10] = 0.112299772892315;  b[11] = 0.112299772892315;
  
  b[12] = -0.028074943223079; b[13] = -0.052583901102545; b[14] = -0.028074943223079; 
  b[15] =  0.112299772892315; b[16] =  0.884134241764073; b[17] =  0.112299772892315;
  
  b[18] = -0.028074943223079; b[19] = -0.028074943223079; b[20] = -0.052583901102545;
  b[21] =  0.112299772892315; b[22] =  0.112299772892315; b[23] =  0.884134241764073;
  
  b[24] = 0.474352608585539;  b[25] = -0.080768594191887; b[26] = -0.080768594191887; 
  b[27] = 0.041035826263138;  b[28] = 0.323074376767549;  b[29] =  0.323074376767549;
  
  b[30] = -0.080768594191887; b[31] =  0.474352608585539; b[32] = -0.080768594191887; 
  b[33] =  0.323074376767549; b[34] =  0.041035826263138; b[35] =  0.323074376767549;
  
  b[36] = -0.080768594191887; b[37] = -0.080768594191887; b[38] =  0.474352608585539; 
  b[39] =  0.323074376767549; b[40] =  0.323074376767549; b[41] =  0.041035826263138;

  return;
}

__device__ void set_base2m1_2(type_fd* b){
  b[0] = - 0.333333333333333;  b[1] =  0.333333333333333;   b[2] = 0.0;                  
  b[3] =   1.333333333333333;  b[4] = -1.333333333333333;   b[5] = 0.0;
  
  b[6] =   0.761136512840921;  b[7] =   0.880568256420460;  b[8] = 0.0;
  b[9] =   1.880568256420460;  b[10] = -1.880568256420460;  b[11] = -1.641704769261381;
  
  b[12] = -0.880568256420460;  b[13] = -0.761136512840921;  b[14] =  0.0;
  b[15] =  1.880568256420460;  b[16] = -1.880568256420460;  b[17] =  1.641704769261381;
  
  b[18] = -0.880568256420460;  b[19] =  0.880568256420460;  b[20] =  0.0;
  b[21] =  0.238863487159079;  b[22] = -0.238863487159079;  b[23] =  0.0;
  
  b[24] = -2.189707941412349;  b[25] = -0.594853970706175;  b[26] =  0.0;
  b[27] =  0.405146029293825;  b[28] = -0.405146029293825;  b[29] =  2.784561912118524;
  
  b[30] =  0.594853970706175;  b[31] =  2.189707941412349;  b[32] =  0.0;
  b[33] =  0.405146029293825;  b[34] = -0.405146029293825;  b[35] =  -2.784561912118524;
  
  b[36] =  0.594853970706175;  b[37] = -0.594853970706175;  b[38] =  0.0;
  b[39] =  3.189707941412349;  b[40] = -3.189707941412349;  b[41] =  0.0;

  return;
}

__device__ void set_base2m1_3(type_fd* b){
  b[0] = -0.333333333333333;   b[1] = 0.0;                  b[2] =  0.333333333333333;  
  b[3] =  1.333333333333333;   b[4] = 0.0,                  b[5] = -1.333333333333333;
  
  b[6] =  0.761136512840921;   b[7] = 0.0;                  b[8] =   0.880568256420460; 
  b[9] =  1.880568256420460;   b[10] = -1.641704769261381;  b[11] = -1.880568256420460;
  
  b[12] = -0.880568256420460;  b[13] = 0.0;                 b[14] =  0.880568256420460;
  b[15] =  0.238863487159079;  b[16] =  0.0;                b[17] = -0.238863487159079;
  
  b[18] = -0.880568256420460;  b[19] = 0.0;                 b[20] = -0.761136512840921;
  b[21] =  1.880568256420460;  b[22] =  1.641704769261381;  b[23] = -1.880568256420460;
  
  b[24] = -2.189707941412349;  b[25] = 0.0;                 b[26] = -0.594853970706175; 
  b[27] =  0.405146029293825;  b[28] =  2.784561912118524;  b[29] = -0.405146029293825;
  
  b[30] =  0.594853970706175;  b[31] = 0.0;                 b[32] = -0.594853970706175;
  b[33] =  3.189707941412349;  b[34] =  0.0;                b[35] = -3.189707941412349;
  
  b[36] =  0.594853970706175;  b[37] = 0.0;                 b[38] =  2.189707941412349;
  b[39] =  0.405146029293825;  b[40] = -2.784561912118524;  b[41] = -0.405146029293825;

  return;
}

__device__ void set_base1nl(type_fd* b){
  b[0]  =  0.8738219710169960; b[1]  =  0.0630890144915020; b[2]  =  0.0630890144915020;
  b[3]  =  0.0630890144915020; b[4]  =  0.8738219710169960; b[5]  =  0.0630890144915020;  
  b[6]  =  0.0630890144915020; b[7]  =  0.0630890144915020; b[8]  =  0.8738219710169960;
  
  b[9]  =  0.5014265096581800; b[10] =  0.2492867451709100; b[11] =  0.2492867451709100;
  b[12] =  0.2492867451709100; b[13] =  0.5014265096581800; b[14] =  0.2492867451709100;
  b[15] =  0.2492867451709100; b[16] =  0.2492867451709100; b[17] =  0.5014265096581800;

  b[18] =  0.6365024991213990; b[19] =  0.0531450498448160; b[20] =  0.3103524510337850;
  b[21] =  0.3103524510337850; b[22] =  0.6365024991213990; b[23] =  0.0531450498448160;  
  b[24] =  0.3103524510337850; b[25] =  0.0531450498448160; b[26] =  0.6365024991213990;
  
  b[27] =  0.6365024991213990; b[28] =  0.3103524510337850; b[29] =  0.0531450498448160;
  b[30] =  0.0531450498448160; b[31] =  0.6365024991213990; b[32] =  0.3103524510337850;
  b[33] =  0.0531450498448160; b[34] =  0.3103524510337850; b[35] =  0.6365024991213990;

  return;

}

__device__ void set_base2nl_1(type_fd* b) {
  
  b[0]  = -0.055128566992484; b[1]  = -0.055128566992484; b[2]  =  0.653307703047060;
  b[3]  =  0.220514267969936; b[4]  =  0.220514267969936; b[5]  =  0.015920894998036;
  
  b[6]  = -0.055128566992484; b[7]  =  0.653307703047060; b[8]  = -0.055128566992484;
  b[9]  =  0.220514267969936; b[10] =  0.015920894998036; b[11] =  0.220514267969936;

  b[12] =  0.653307703047060; b[13] = -0.055128566992484; b[14] = -0.055128566992484;
  b[15] =  0.015920894998036; b[16] =  0.220514267969936; b[17] =  0.220514267969936;

  b[18] = -0.124998982535098; b[19] = -0.124998982535098; b[20] =  0.001430579517790;
  b[21] =  0.499995930140390; b[22] =  0.499995930140390; b[23] =  0.248575525271625;
  
  b[24] = -0.124998982535098; b[25] =  0.001430579517790; b[26] = -0.124998982535098;
  b[27] =  0.499995930140390; b[28] =  0.248575525271625; b[29] =  0.499995930140390;

  b[30] =  0.001430579517790; b[31] = -0.124998982535098; b[32] = -0.124998982535098;
  b[33] =  0.248575525271625; b[34] =  0.499995930140390; b[35] =  0.499995930140390;

  b[36] = -0.117715163308429; b[37] = -0.047496257198800; b[38] =  0.173768363654174;
  b[39] =  0.135307828168627; b[40] =  0.790160442765823; b[41] =  0.065974785918605;

  b[42] = -0.117715163308429; b[43] =  0.173768363654174; b[44] = -0.047496257198800;
  b[45] =  0.135307828168627; b[46] =  0.065974785918605; b[47] =  0.790160442765823;

  b[48] =  0.173768363654174; b[49] = -0.047496257198800; b[50] = -0.117715163308429;
  b[51] =  0.065974785918605; b[52] =  0.790160442765823; b[53] =  0.135307828168627;

  b[54] =  0.173768363654174; b[55] = -0.117715163308429; b[56] = -0.047496257198800;
  b[57] =  0.065974785918605; b[58] =  0.135307828168627; b[59] =  0.790160442765823;

  b[60] = -0.047496257198800; b[61] = -0.117715163308429; b[62] =  0.173768363654174;
  b[63] =  0.790160442765823; b[64] =  0.135307828168627; b[65] =  0.065974785918605;

  b[66] = -0.047496257198800; b[67] =  0.173768363654174; b[68] = -0.117715163308429;
  b[69] =  0.790160442765823; b[70] =  0.065974785918605; b[71] =  0.135307828168627;
    
  return;
}



__device__ void set_base2nl_2(type_fd* b){
  
  b[0]  =  0.747643942033992; b[1]  = -0.747643942033992; b[2]  =  0.0;
  b[3]  =  3.495287884067984; b[4]  = -3.495287884067984; b[5]  =  0.0;

  b[6]  =  0.747643942033992; b[7]  =  2.495287884067984; b[8]  =  0.0;
  b[9]  =  0.252356057966008; b[10] = -0.252356057966008; b[11] = -3.242931826101976;

  b[12] = -2.495287884067984; b[13] = -0.747643942033992; b[14] =  0.0;
  b[15] =  0.252356057966008; b[16] = -0.252356057966008; b[17] =  3.242931826101976;

  b[18] =  0.002853019316360; b[19] = -0.002853019316360; b[20] =  0.0;
  b[21] =  2.005706038632720; b[22] = -2.005706038632720; b[23] = -0.000000000000000;

  b[24] =  0.002853019316360; b[25] =  1.005706038632720; b[26] =  0.0;
  b[27] =  0.997146980683640; b[28] = -0.997146980683640; b[29] = -1.008559057949080;

  b[30] = -1.005706038632720; b[31] = -0.002853019316360; b[32] =  0.0;
  b[33] =  0.997146980683640; b[34] = -0.997146980683640; b[35] =  1.008559057949080;

  b[36] = -0.241409804135140; b[37] = -0.787419800620736; b[38] =  0.0;
  b[39] =  2.546009996485596; b[40] = -2.546009996485596; b[41] =  1.028829604755876;

  b[42] = -0.241409804135140; b[43] =  1.546009996485596; b[44] =  0.0;
  b[45] =  0.212580199379264; b[46] = -0.212580199379264; b[47] = -1.304600192350456;

  b[48] = -1.546009996485596; b[49] = -0.787419800620736; b[50] =  0.0;
  b[51] =  1.241409804135140; b[52] = -1.241409804135140; b[53] =  2.333429797106332;

  b[54] = -1.546009996485596; b[55] =  0.241409804135140; b[56] =  0.0;
  b[57] =  0.212580199379264; b[58] = -0.212580199379264; b[59] =  1.304600192350456;

  b[60] =  0.787419800620736; b[61] =  0.241409804135140; b[62] =  0.0;
  b[63] =  2.546009996485596; b[64] = -2.546009996485596; b[65] = -1.028829604755876;

  b[66] =  0.787419800620736; b[67] =  1.546009996485596; b[69] =  0.0;
  b[69] =  1.241409804135140; b[70] = -1.241409804135140; b[71] = -2.333429797106332;

  return;
}




__device__ void set_base2nl_3(type_fd* b){
  
  b[0]  =  0.747643942033992; b[1]  =  0.0;                b[2]  =  2.495287884067984; 
  b[3]  =  0.252356057966008; b[4]  = -3.242931826101976; b[5]  = -0.252356057966008;
  
  b[6]  =  0.747643942033992; b[7]  =  0.0;                b[8]  = -0.747643942033992;
  b[9]  =  3.495287884067984; b[10] =  0,0;                b[11] = -3.495287884067984;
  
  b[12] = -2.495287884067984; b[13] =  0.0;                b[14] = -0.747643942033992;
  b[15] =  0.252356057966008; b[16] =  3.242931826101976;  b[17] = -0.252356057966008;

  b[18] =  0.002853019316360; b[19] =  0.0;                b[20] =  1.005706038632720;
  b[21] =  0.997146980683640; b[22] = -1.008559057949080;  b[23] = -0.997146980683640;

  b[24] =  0.002853019316360; b[25] =  0.0;                b[26] = -0.002853019316360;
  b[27] =  2.005706038632720; b[28] = -0.000000000000000;  b[29] = -2.005706038632720;

  b[30] = -1.005706038632720; b[31] =  0.0;                b[32] = -0.002853019316360;
  b[33] =  0.997146980683640; b[34] =  1.008559057949080;  b[35] = -0.997146980683640;

  b[36] = -0.241409804135140; b[37] =  0.0;                b[38] =  1.546009996485596;
  b[39] =  0.212580199379264; b[40] = -1.304600192350456;  b[41] = -0.212580199379264;

  b[42] = -0.241409804135140; b[43] =  0.0;                b[44] = -0.787419800620736;
  b[45] =  2.546009996485596; b[46] =   1.028829604755876; b[47] = -2.546009996485596;

  b[48] = -1.546009996485596; b[49] =  0.0;                b[50] =  0.241409804135140;
  b[51] =  0.212580199379264; b[52] =  1.304600192350456;  b[53] = -0.212580199379264;

  b[54] = -1.546009996485596; b[55] =  0.0;                b[56] = -0.787419800620736;
  b[57] =  1.241409804135140; b[59] =  2.333429797106332;  b[59] = -1.241409804135140;

  b[60] =  0.787419800620736; b[61] =  0.0;                b[62] =  1.546009996485596;
  b[63] =  1.241409804135140; b[64] = -2.333429797106332;  b[65] = -1.241409804135140;

  b[66] =  0.787419800620736; b[67] =  0.0;                b[68] =  0.241409804135140;
  b[69] =  2.546009996485596; b[70] = -1.028829604755876;  b[72] = -2.546009996485596;
  return;
}

__device__ void set_base2nl_1_1(type_fd* b){

  b[0]  =  0.65330770304705954;  b[1]  = -0.05512856699248411; b[2]  = -0.05512856699248411; b[3]  = 0.01592089499803580; b[4]  = 0.22051426796993640; b[5]  = 0.22051426796993640;
  b[6]  = -0.05512856699248405;  b[7]  =  0.65330770304705965; b[8]  = -0.05512856699248411; b[9]  = 0.22051426796993642; b[10] = 0.01592089499803579; b[11] = 0.22051426796993612;
  b[12] = -0.05512856699248414;  b[13] = -0.05512856699248411; b[14] =  0.65330770304705965; b[15] = 0.22051426796993642; b[16] = 0.22051426796993612; b[17] = 0.01592089499803579;
  b[18] =  0.00143057951778969;  b[19] = -0.12499898253509756; b[20] = -0.12499898253509756; b[21] = 0.24857552527162491; b[22] = 0.49999593014039018; b[23] = 0.49999593014039018;
  b[24] = -0.12499898253509753;  b[25] =  0.00143057951778980; b[26] = -0.12499898253509756; b[27] = 0.49999593014039029; b[28] = 0.24857552527162485; b[29] = 0.49999593014039023;
  b[30] = -0.12499898253509756;  b[31] = -0.12499898253509756; b[32] =  0.00143057951778980; b[33] = 0.49999593014039029; b[34] = 0.49999593014039023; b[35] = 0.24857552527162485;
  b[36] =  0.17376836365417397;  b[37] = -0.04749625719880005; b[38] = -0.11771516330842915; b[39] = 0.06597478591860528; b[40] = 0.79016044276582309; b[41] = 0.13530782816862683;
  b[42] = -0.11771516330842910;  b[43] =  0.17376836365417403; b[44] = -0.04749625719880005; b[45] = 0.13530782816862683; b[46] = 0.06597478591860527; b[47] = 0.79016044276582331;
  b[48] = -0.11771516330842935;  b[49] = -0.04749625719880005; b[50] =  0.17376836365417403; b[51] = 0.13530782816862683; b[52] = 0.79016044276582331; b[53] = 0.06597478591860527;
  b[54] =  0.17376836365417400;  b[55] = -0.11771516330842915; b[56] = -0.04749625719880005; b[57] = 0.06597478591860528; b[58] = 0.13530782816862683; b[59] = 0.79016044276582309;
  b[60] = -0.04749625719880010;  b[61] =  0.17376836365417403; b[62] = -0.11771516330842915; b[63] = 0.79016044276582309; b[64] = 0.06597478591860523; b[65] = 0.13530782816862685;
  b[66] = -0.04749625719880013;  b[67] = -0.11771516330842915; b[68] = 0.17376836365417403;  b[69] = 0.79016044276582309; b[70] = 0.13530782816862685; b[71] = 0.06597478591860523;

  return;
}

__device__ void set_base2nl_2_1(type_fd* b){

  b[0]  = -2.495287884067984;  b[1]  = -0.747643942033992; b[2]  = 0.000000000000000; b[3]  = 0.252356057966008; b[4]  = -0.252356057966008; b[5]  =  3.242931826101976;
  b[6]  =  0.747643942033992;  b[7]  =  2.495287884067984; b[8]  = 0.000000000000000; b[9]  = 0.252356057966008; b[10] = -0.252356057966008; b[11] = -3.242931826101976;
  b[12] =  0.747643942033992;  b[13] = -0.747643942033992; b[14] = 0.000000000000000; b[15] = 3.495287884067984; b[16] = -3.495287884067984; b[17] =  0.000000000000000;
  b[18] = -1.005706038632720;  b[19] = -0.002853019316360; b[20] = 0.000000000000000; b[21] = 0.997146980683640; b[22] = -0.997146980683640; b[23] =  1.008559057949080;
  b[24] =  0.002853019316360;  b[25] =  1.005706038632720; b[26] = 0.000000000000000; b[27] = 0.997146980683640; b[28] = -0.997146980683640; b[29] = -1.008559057949080;
  b[30] =  0.002853019316360;  b[31] = -0.002853019316360; b[32] = 0.000000000000000; b[33] = 2.005706038632720; b[34] = -2.005706038632720; b[35] = -0.000000000000000;
  b[36] = -1.546009996485596;  b[37] = -0.787419800620736; b[38] = 0.000000000000000; b[39] = 1.241409804135140; b[40] = -1.241409804135140; b[41] =  2.333429797106332;
  b[42] = -0.241409804135140;  b[43] =  1.546009996485596; b[44] = 0.000000000000000; b[45] = 0.212580199379264; b[46] = -0.212580199379264; b[47] = -1.304600192350456;
  b[48] = -0.241409804135140;  b[49] = -0.787419800620736; b[50] = 0.000000000000000; b[51] = 2.546009996485596; b[52] = -2.546009996485596; b[53] =  1.028829604755876;
  b[54] = -1.546009996485596;  b[55] =  0.241409804135140; b[56] = 0.000000000000000; b[57] = 0.212580199379264; b[58] = -0.212580199379264; b[59] =  1.304600192350456;
  b[60] =  0.787419800620736;  b[61] =  1.546009996485596; b[62] = 0.000000000000000; b[63] = 1.241409804135140; b[64] = -1.241409804135140; b[65] = -2.333429797106332;
  b[66] =  0.787419800620736;  b[67] =  0.241409804135140; b[68] = 0.000000000000000; b[69] = 2.546009996485596; b[70] = -2.546009996485596; b[71] = -1.028829604755876;

  return;
}

__device__ void set_base2nl_3_1(type_fd* b){

  b[0]  = -2.495287884067984;  b[1]  = 0.000000000000000; b[2]  = -0.747643942033992; b[3]  = 0.252356057966008; b[4]  =  3.242931826101976; b[5]  = -0.252356057966008;
  b[6]  =  0.747643942033992;  b[7]  = 0.000000000000000; b[8]  = -0.747643942033992; b[9]  = 3.495287884067984; b[10] =  0.000000000000000; b[11] = -3.495287884067984;
  b[12] =  0.747643942033992;  b[13] = 0.000000000000000; b[14] =  2.495287884067984; b[15] = 0.252356057966008; b[16] = -3.242931826101976; b[17] = -0.252356057966008;
  b[18] = -1.005706038632720;  b[19] = 0.000000000000000; b[20] = -0.002853019316360; b[21] = 0.997146980683640; b[22] =  1.008559057949080; b[23] = -0.997146980683640;
  b[24] =  0.002853019316360;  b[25] = 0.000000000000000; b[26] = -0.002853019316360; b[27] = 2.005706038632720; b[28] = -0.000000000000000; b[29] = -2.005706038632720;
  b[30] =  0.002853019316360;  b[31] = 0.000000000000000; b[32] =  1.005706038632720; b[33] = 0.997146980683640; b[34] = -1.008559057949080; b[35] = -0.997146980683640;
  b[36] = -1.546009996485596;  b[37] = 0.000000000000000; b[38] =  0.241409804135140; b[39] = 0.212580199379264; b[40] =  1.304600192350456; b[41] = -0.212580199379264;
  b[42] = -0.241409804135140;  b[43] = 0.000000000000000; b[44] = -0.787419800620736; b[45] = 2.546009996485596; b[46] =  1.028829604755876; b[47] = -2.546009996485596;
  b[48] = -0.241409804135140;  b[49] = 0.000000000000000; b[50] =  1.546009996485596; b[51] = 0.212580199379264; b[52] = -1.304600192350456; b[53] = -0.212580199379264;
  b[54] = -1.546009996485596;  b[55] = 0.000000000000000; b[56] = -0.787419800620736; b[57] = 1.241409804135140; b[58] =  2.333429797106332; b[59] = -1.241409804135140;
  b[60] =  0.787419800620736;  b[61] = 0.000000000000000; b[62] =  0.241409804135140; b[63] = 2.546009996485596; b[64] = -1.028829604755876; b[65] = -2.546009996485596;
  b[66] =  0.787419800620736;  b[67] = 0.000000000000000; b[68] =  1.546009996485596; b[69] = 1.241409804135140; b[70] = -2.333429797106332; b[71] = -1.241409804135140;

  return;
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



__device__ void jacobianoP2(type_fd* xx, type_fd* yy, int* tt, int th, type_fd* De){


  De[0] = yy[tt[th * 6 + 2]] - yy[tt[th * 6 + 0]]; // De[0][0]
  De[1] = yy[tt[th * 6 + 0]] - yy[tt[th * 6 + 1]]; // De[0][1]
  De[2] = xx[tt[th * 6 + 0]] - xx[tt[th * 6 + 2]]; // De[1][0]
  De[3] = xx[tt[th * 6 + 1]] - xx[tt[th * 6 + 0]]; // De[1][1]

  return;
}

__device__ void jacobianoP1(type_fd* x, type_fd* y, int* t, int th, type_fd* De){


  De[0] = y[t[th * 3 + 2]] - y[t[th * 3 + 0]]; // De[0][0]
  De[1] = y[t[th * 3 + 0]] - y[t[th * 3 + 1]]; // De[0][1]
  De[2] = x[t[th * 3 + 0]] - x[t[th * 3 + 2]]; // De[1][0]
  De[3] = x[t[th * 3 + 1]] - x[t[th * 3 + 0]]; // De[1][1]

  return;
}



__device__ void load_tt(int *tt, int *tt_h, int tri, int th){
  
  int i;
  // porto in memori shared la tabella di connettivita'
  for(i = 0; i < 6; i++){
    tt[6 * th + i] = tt_h[6 * tri + i];
  } 
 }

__device__ void load_t(int *t, int *t_h, int tri, int th){
  
  int i;
  // porto in memori shared la tabella di connettivita'
  for(i = 0; i < 3; i++){
    t[3 * th + i] = t_h[3 * tri + i];
  } 
 }


__global__ void assembly_lapl_div_P2P1 (type_fd* xx, type_fd* yy, 
                                        int* tt_h, int *t_h, int nt, 
                                        int n, type_uint nn,
                                        type_fd* A_val, type_uint* MA_ind,
                                        type_fd* B1_val, type_fd* B2_val, type_uint* B_ind){
  /* xx = x coordinate of P2  nodes
  
     yy = y coordinate of P2  nodes

     tt_h = connectivity P2 element table

     t_h = connectivity P1 element table

     nt = number of elements

     n = number of P1 nodes

     nn = number of P2 nodes

     A_val = device vector of dimension (6*6) * nt that contains the values of matrix A in COO format with a unique index 

     MA_ind = device vector containing indices of A

     B1_val = device vector of dimension (3*6) * nt that contains the values of matrix B1 in COO format with a unique index 

     B2_val = device vector of dimension (3*6) * nt that contains the values of matrix B2 in COO format with a unique index 
     
     B_ind = device vector containing indices of matrices B1/B2 */

    int tri = blockIdx.x*blockDim.x + threadIdx.x;
    int th = threadIdx.x; // numero del thread dentro il blocco

    
    __shared__ type_fd coef3;
    __shared__ type_fd base1[3 * 3];
    __shared__ type_fd base21_2[3 * 6];
    __shared__ type_fd base21_3[3 * 6];
    __shared__ int  tt[6 * BLOCKDIM];
    __shared__ int  t[3 * BLOCKDIM];

    if (tri % blockDim.x == 0){
      coef3 = set_coef3();
      set_base1( base1 );
      set_base21_2( base21_2 );
      set_base21_3( base21_3 );
    }
    __syncthreads();
    
    

  if (tri < nt){
     
    type_fd De[2 * 2];
    type_fd Je, inv_Je; // determinate di De

    type_fd dx[3 * 6] = {0.0};
    type_fd dy[3 * 6] = {0.0};
    type_fd temp_Ax[6 * 3] = {0.0}; // (diag(coef3)*dxi)'
    type_fd temp_Ay[6 * 3] = {0.0}; // (diag(coef3)*dyi)'
    type_fd temp_Ax_dx[6 * 6] = {0.0};
    type_fd temp_Ay_dy[6 * 6] = {0.0};
    type_fd temp_B[3 * 3] = {0.0}; // (diag(coef3)*bpi)'

    type_fd B1e[3 * 6] = {0.0};
    type_fd B2e[3 * 6] = {0.0};

    
    int i,j, indice, indice_tra; //, a, b, c, indice_tot;


    // porto in memori shared le tabelle di connettivita
    load_tt(tt, tt_h, tri, th);
    load_t(t, t_h, tri, th);

    // calcolo il determinate di De

    jacobianoP2(xx, yy, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);


    // riempio i vettori dx dy
    for(i = 0; i < 3 * 6; ++i){
      dx[i] = base21_2[i]*De[0] + base21_3[i]*De[1];
      dy[i] = base21_2[i]*De[2] + base21_3[i]*De[3];
    }
     // calcolo delle matrici ausiliarie
    for(i = 0; i < 3 ; ++i){
      for(j = 0; j < 6; ++j){
 	indice = i * 6 + j;
 	indice_tra = j * 3 + i;
 	temp_Ax[indice_tra] = coef3 * dx[indice];
 	temp_Ay[indice_tra] = coef3 * dy[indice];
      }
    }

    for(i = 0; i < 3*3; ++i){ // base1 e' simmetrica non serve fare la trasposta
 	temp_B[i] = coef3 * base1[i];
    }

    // calcolo B1e =  (diag(coef3)*bpi)' * dxj;
    Prodotto_Matrici(temp_B, dx, B1e, 3, 3, 3, 6);
    // calcolo B2e =  (diag(coef3)*bpi)' * dyj;
    Prodotto_Matrici(temp_B, dy, B2e, 3, 3, 3, 6);

    // calcolo temp_Ax_dx = (diag(coef3)*dxi)' * dxj
    Prodotto_Matrici(temp_Ax, dx, temp_Ax_dx, 6, 3, 3, 6);

    // calcolo temp_Ay_dy = (diag(coef3)*dyi)' * dyj
    Prodotto_Matrici(temp_Ay, dy, temp_Ay_dy, 6, 3, 3, 6);

    // if (tri == nt-1){
    //   printf("temp_Ax_dx = \n");
    //   for(i = 0; i < 6; ++i){
    // 	for (j = 0; j < 6; ++j)
    // 	  printf("%18.16f \t", temp_Ax_dx[i*6+j]);
    // 	printf("\n");
    //   }
    //   printf("\n");
    //   printf("temp_Ay_dy = \n");
    //   for(i = 0; i < 6; ++i){
    // 	for (j = 0; j < 6; ++j)
    // 	  printf("%18.16f \t", temp_Ay_dy[i*6+j]);
    // 	printf("\n");
    //   }
    //   printf("\n");
    // }

    
    inv_Je = 1.0/Je;

    // if (tri == 1794){
    //   for (i = 0; i < 6; ++i)
    // 	printf("%d %f, %f \n", tt[6 * th + i]);
    // }
    
    for(i = 0; i < 6; ++i){
      for(j = 0; j < 6; ++j){
     	indice = i * 6 + j;
	A_val[tri * 36 + indice] = (temp_Ax_dx[indice] + temp_Ay_dy[indice]) * inv_Je;
	MA_ind[tri * 36 + indice] = tt[th * 6 + i] * nn + tt[th * 6  + j];
      }
    }


    for(i = 0; i < 3; ++i){
      for(j = 0; j < 6; ++j){
 	indice = i * 6 + j;
 	B1_val[tri * 18 + indice] = B1e[indice];
 	B2_val[tri * 18 + indice] = B2e[indice];
	B_ind[tri * 18 + indice]  = t[th * 3 + i] * nn + tt[th * 6  + j];
	
      }
    }


     
  }

  return;
}



__global__ void assembly_mass_nlt_P2P1(type_fd *xx, type_fd *yy, int *tt_h, int *t_h,
                                      int nt, type_uint nn,
                                      type_fd *rho, type_fd *u_x, type_fd *u_y, type_fd *uo_x, type_fd *uo_y,
                                      type_fd *M_val, type_uint *M_ind,
                                      type_fd *NL_val, type_uint *NL_ind){

 
   /* xx = x coordinate of P2  nodes
  
     yy = y coordinate of P2  nodes

     tt_h = connectivity P2 element table

     t_h = connectivity P1 element table

     nt = number of elements

     nn = number of P2 nodes
     
     rho  = density

     u_x  = x component of velocity at time n

     u_y  = y component of velocity at time n

     uo_x  = x component of velocity at time n-1

     uo_y  = y component of velocity at time n-1

     M_val = device vector of dimension (6*6) * nt that contains the values of matrix M in COO format with a unique index 

     M_ind = device vector containing indices of M

     NL_val = device vector of dimension (6*6) * nt that contains the values of matrix NL in COO format with a unique index 
  
     NL_ind = device vector containing indices of matrix NL */

  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco
  
  
  __shared__ type_fd coef7[7];
  __shared__ type_fd coef12[12];
  __shared__ type_fd base1m[7 * 3];
  __shared__ type_fd base2m[7 * 6];
  __shared__ type_fd base1nl[ 12 * 3];
  __shared__ type_fd base2nl_1_1[12 * 6];
  __shared__ type_fd base2nl_2_1[12 * 6];
  __shared__ type_fd base2nl_3_1[12 * 6];
  __shared__ int  tt[6 * BLOCKDIM];
  __shared__ int  t[3 * BLOCKDIM];
  
  if (tri % blockDim.x == 0){
    set_coef7( coef7 );
    set_coef12( coef12 );
    set_base1m( base1m );
    set_base2m( base2m );
    set_base1nl( base1nl );
    set_base2nl_1_1( base2nl_1_1 );
    set_base2nl_2_1( base2nl_2_1 );
    set_base2nl_3_1( base2nl_3_1 );
  }
  __syncthreads();

  if (tri < nt){
    type_fd De[2 * 2];
    type_fd Je; // determinate di De    
    type_fd temp_M[6 * 7] = {0.0};
    type_fd prod[6 * 6] = {0.0};
    type_fd coef_rho[7*3] = {0.0};
    type_fd Me[6 * 6]  = {0.0};
    
    type_fd dx[12 * 6] = {0.0};
    type_fd dy[12 * 6] = {0.0};
    type_fd Mx[6 * 6] = {0.0};
    type_fd My[6 * 6] = {0.0};
    type_fd prod_cf [12 * 6] = {0.0};
    type_fd prod_brl [12 * 6] = {0.0};
    type_fd temp_x [6] = {0.0};
    type_fd temp_y [6] = {0.0};
    type_fd sol_x [6] = {0.0};
    type_fd sol_y [6] = {0.0};
    
    int i, j, k, m, indice;

    /******************CALCOLO MATRICE DI MASSA******************/
    
    // porto in memori shared la tabella di connettivita'
    load_tt(tt, tt_h, tri, th);
    load_t(t, t_h, tri, th);

    // calcolo il determinate di De

    jacobianoP2(xx, yy, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);


    // calcolo coeff_rho
    for (i = 0; i < 7; ++i){
      for(j = 0; j < 3; ++j){
	coef_rho[i*3+j] = coef7[i] * base1m[i * 3 +j];
      }
    }

    //loop on degrees of freedom of density
    for(k = 0; k < 3; ++k){

      // define temp_M = (diag(coef_rho(:,k))*bsi)'
      for(i = 0; i < 7; ++i){
	for(j = 0; j < 6; ++j){
	  temp_M[j * 7 + i ] = coef_rho[i*3+k] * base2m[i * 6 + j];
	}
      }

      //Calcolo  Me = (diag(coef7)*bli)' * blj
      //NOTA: rispetto al codice di Lille manca * Je * rho(tt(1,ke),1)
      Prodotto_Matrici(temp_M, base2m, prod, 6, 7, 7, 6);
      
      for(i = 0; i < 6; ++i){
	for(j = 0; j < 6; ++j){
	  indice = i * 6 + j;
	  Me[indice] += prod[indice] * Je * rho[t[th * 3 + k]]; // ho cambiato la tabella delle connettivita' da P2 a P1 per il calcolo di rho
	}
      }
      
    }

    for(i = 0; i < 6; ++i){
      for(j = 0; j < 6; ++j){
	indice = i * 6 + j;
	M_val[tri * 36 + indice] = Me[indice];
	M_ind[tri * 36 + indice] = tt[th * 6 + i] * nn + tt[th * 6  + j]; //il cast a type_uint e' reso da * nn 

      }
    }   
    
    
    /******************CALCOLO MATRICE TERMINE NON LINEARE******************/

    for(i = 0; i < 6; i++){
      sol_x[i] = 2*u_x[tt[th * 6 + i]] - uo_x[tt[th * 6 + i]];
      sol_y[i] = 2*u_y[tt[th * 6 + i]] - uo_y[tt[th * 6 + i]];
    }


    // dx, dy 12*6
    for( i = 0; i < 12 * 6; i++){
      dx[i] = base2nl_2_1[i] * De[0] + base2nl_3_1[i] * De[1];
      dy[i] = base2nl_2_1[i] * De[2] + base2nl_3_1[i] * De[3];
    }


    //calcolo prod_cf_bsi 12*6
    for (i = 0; i < 12; ++i){
      for(j = 0; j < 6; ++j){
        prod_cf[i*6+j] = coef12[i] * base2nl_1_1[i * 6 +j];
      }
    }


    //calcolo prod_brl_bsk 12*6
    for(k = 0; k < 3; k++){
      for(i = 0; i < 12; ++i){
	for(j = 0; j < 6; ++j){
	  prod_brl[i * 6 + j] += base1nl[i*3+k] * base2nl_1_1[i * 6 + j] * rho[t[th * 3 + k]]; // ho cambiato la tabella delle connettivita' da P2 a P1 per il calcolo di rho
	}
      }
    }


    //Loop on degrees of freedom for velocity
    for(i = 0; i < 6; i++){

      for(j = 0; j < 6 * 6; j++){
	// pulisco
	Mx[j] = My[j] = 0.0;
      }
      for(j = 0; j < 6; j++){
	// pulisco
	temp_x[j] = temp_y[j] = 0.0;
      }

      
      for(j = 0; j < 6; j++){
	for(k = 0; k < 6; k++){
	  for(m = 0; m < 12; m++){
	    Mx[j * 6 + k] +=  prod_cf[m * 6 + i] * dx[m * 6 + j] * prod_brl[m * 6 + k];
	    My[j * 6 + k] +=  prod_cf[m * 6 + i] * dy[m * 6 + j] * prod_brl[m * 6 + k];
	  }
	}
      }

      
      Prodotto_Matrici(Mx, sol_x, temp_x, 6, 6, 6, 1);
      Prodotto_Matrici(My, sol_y, temp_y, 6, 6, 6, 1);      
      
      for(j = 0; j < 6; j++){
	NL_val[tri * 36 + i * 6 + j ] = temp_x[j] + temp_y[j]; // non c'e bisogno di fare la trasposta
	NL_ind[tri * 36 + i * 6 + j ] = tt[th * 6 + i] * nn + tt[th * 6 + j];
      }

    }

    
  }
  return;
}

__global__ void assembly_mass_P2(type_fd *xx, type_fd *yy, int *tt_h, 
			    int nt, type_uint nn,
			    type_fd *M_val, type_uint *M_ind){

   /* xx = x coordinate of P2  nodes
  
     yy = y coordinate of P2  nodes

     tt_h = connectivity P2 element table

     nt = number of elements

     nn = number of P2 nodes

     M_val = device vector of dimension (6*6) * nt that contains the values of matrix M in COO format with a unique index 

     M_ind = device vector containing indices of M
*/

  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco
  
  
  __shared__ type_fd coef7[7];
  __shared__ type_fd coef12[12];
  __shared__ type_fd base1m[7 * 3];
  __shared__ type_fd base2m[7 * 6];
  __shared__ int  tt[6 * BLOCKDIM];
  
  if (tri % blockDim.x == 0){
    set_coef7( coef7 );
    set_coef12( coef12 );
    set_base1m( base1m );
    set_base2m( base2m );
  }
  __syncthreads();

  if (tri < nt){
    type_fd De[2 * 2];
    type_fd Je; // determinate di De    
    type_fd temp_M[6 * 7] = {0.0};
    type_fd prod[6 * 6] = {0.0};
    type_fd coef_rho[7*3] = {0.0};
    type_fd Me[6 * 6]  = {0.0};
    
    int i, j, k, m, indice;

    /******************CALCOLO MATRICE DI MASSA******************/
    
    // porto in memori shared la tabella di connettivita'
    load_tt(tt, tt_h, tri, th);


    // calcolo il determinate di De

    jacobianoP2(xx, yy, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);


    // calcolo coeff_rho
    for (i = 0; i < 7; ++i){
      for(j = 0; j < 3; ++j){
	coef_rho[i*3+j] = coef7[i] * base1m[i * 3 +j];
      }
    }

    //loop on degrees of freedom of density
    for(k = 0; k < 3; ++k){

      // define temp_M = (diag(coef_rho(:,k))*bsi)'
      for(i = 0; i < 7; ++i){
	for(j = 0; j < 6; ++j){
	  temp_M[j * 7 + i ] = coef_rho[i*3+k] * base2m[i * 6 + j];
	}
      }

      //Calcolo  Me = (diag(coef7)*bli)' * blj
      //NOTA: rispetto al codice di Lille manca * Je * rho(tt(1,ke),1)
      Prodotto_Matrici(temp_M, base2m, prod, 6, 7, 7, 6);
      
      for(i = 0; i < 6; ++i){
	for(j = 0; j < 6; ++j){
	  indice = i * 6 + j;
	  Me[indice] += prod[indice] * Je; // ho cambiato la tabella delle connettivita' da P2 a P1 per il calcolo di rho
	}
      }
      
    }

    for(i = 0; i < 6; ++i){
      for(j = 0; j < 6; ++j){
	indice = i * 6 + j;
	M_val[tri * 36 + indice] = Me[indice];
	M_ind[tri * 36 + indice] = tt[th * 6 + i] * nn + tt[th * 6  + j];
      }
    }   
    
  }
  return;
}
 


__global__ void assembly_rhs(type_fd *xx, type_fd *yy, int *tt_h, int *t_h, int nt, type_fd *rho, 
			int BENCHMARK_CASE, int split_step, type_fd time, type_fd Dt, type_fd at_const,
			type_fd *fu_x, type_fd *fu_y,
			type_uint *ind_x){
   /* xx = x coordinate of P2  nodes
  
     yy = y coordinate of P2  nodes

     tt_h = connectivity P2 element table

     t_h = connectivity P1 element table

     nt = number of elements

     nn = number of P2 nodes
     
     rho  = density
     
     BENCHMARK_CASE = 0 (EXAC), 4 (DROP), 5 (RTIN)
     
     split_step = Strang scheme split step
     
     time = simulation time 
     
     Dt = time step
     
     at_const = multiplicative constant of additional term in EXAC

     fu_x = x-component of source term is sparse format, vector of length 6*nt

     fu_y = t-component of source term is sparse format, vector of length 6*nt

     ind_x = indices of vector fu, vector of length 6*nt
 */

  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco
  type_uint cast = 1;
  
  __shared__ type_fd coef12[12];
  __shared__ type_fd coor12[12*2];
  __shared__ type_fd base1nl[12 * 3];
  __shared__ type_fd base2nl_1_1[12 * 6];
  __shared__ type_fd coef4[4];
  __shared__ type_fd base1_4[4 * 3];
  __shared__ type_fd base2_4[4 * 6];
  __shared__ int tt[6 * BLOCKDIM];
  __shared__ int t[3 * BLOCKDIM];
  
  if (tri % blockDim.x == 0){
    set_coef12( coef12 );
    set_coor12( coor12 );
    set_base1nl( base1nl );
    set_base2nl_1_1( base2nl_1_1 );
    set_coef4( coef4 );
    set_base1_4( base1_4 );
    set_base2_4( base2_4 );
  }
  
  __syncthreads();

  if (tri < nt){
    
    type_fd De[2*2] = {0.0};
    type_fd Je;
    type_fd xin, yin, rhoin, fu1in = 0.0, fu2in = 0.0;
    type_fd temps1, max, min, rhomed; 
    type_fd g[2] = {0.0,-1.0}; //9.81
    type_fd fe[3 * 6] = {0.0};
    type_fd feT[3*6] = {0.0};
    type_fd prov[3] = {0.0};
    type_fd me[6] = {0.0};
    type_fd temp[3 * 4];
    int  i, j, k;
    
    if (split_step == 1)
      temps1 = time;
    else
      temps1 = time - Dt;

    // porto in memori shared la tabella di connettivita'
    load_tt(tt, tt_h, tri, th);
    load_t(t, t_h, tri, th);

    // calcolo il determinate di De

    jacobianoP2(xx, yy, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);
      

    if(BENCHMARK_CASE == 1){ //EXAC
    
      for (i = 0; i < 6; i++){
	for(k = 0; k < 12; k++){
	  
	  xin = xx[tt[th * 6]] + coor12[k*2]*De[3] - coor12[k*2+1]*De[2];
	  yin = yy[tt[th * 6]] - coor12[k*2]*De[1] + coor12[k*2+1]*De[0];
	  
	  rhoin = 2.0 + xin * cos(sin(temps1)) + yin * sin(sin(temps1)) + at_const*exp(-(xin*xin)-(yin*yin));
	  
	  fu1in = +(yin*sin(time) - xin*cos(time)*cos(time))*rhoin + cos(xin)*sin(yin)*sin(time);
	  fu2in = -(xin*sin(time) + yin*cos(time)*cos(time))*rhoin + sin(xin)*cos(yin)*sin(time);
	  
	  fu_x[tri * 6 + i] += fu1in * base2nl_1_1[k * 6 + i] * coef12[k] * Je;
	  fu_y[tri * 6 + i] += fu2in * base2nl_1_1[k * 6 + i] * coef12[k] * Je;
	  ind_x[tri * 6 + i] = tt[th * 6 + i]*cast; 
	}
      }
    }else if (BENCHMARK_CASE == 4 || BENCHMARK_CASE == 5){ //DROP or RTIN

      for (i = 0; i < 4; ++i){
	for(j = 0; j < 3; ++j){
	  temp[j*4+i] = coef4[i] * base1_4[i * 3 +j];
	}
      }

      Prodotto_Matrici(temp, base2_4, fe, 3, 4, 4, 6);

      for(i=0; i< 3*6; i++)
	fe[i]=fe[i]*Je;

      min = rho[t[th * 3 + 0]];
      max = rho[t[th * 3 + 0]];
      for(i = 1; i < 3; ++i){
	if(rho[t[th * 3 + i]] < min)
	  min = rho[t[th * 3 + i]];
	if(rho[t[th * 3 + i]] > max)
	  max = rho[t[th * 3 + i]];
      }

      rhomed = (min + max)/2;

      for(k=0;k<3;k++)
	prov[k] = rhomed;

      // if (tri == 19675){
      // 	for(i = 0; i < 3; ++i)
      // 	  printf("%d \t", t[th * 3 + i]);
	
      // 	printf("\n min = %18.16f, max = %18.16f, rhomed = %18.16f \n", min, max, rhomed);
      // }

      for(i = 0; i < 6; ++i){
	  for(k = 0; k < 3; ++k){
	    feT[i*3+k] = fe[k*6+i];
	  }
      }

      Prodotto_Matrici(feT, prov, me, 6, 3, 3, 1);
    
      for(i = 0; i < 6; ++i){
	fu_x[tri * 6 + i] += g[0]*me[i];
	// if ((int(tt[th * 6 + i]) == 42757) && (tri == 19675))
	//   printf("\n Tri = %d, g[1]*me[i] =  %18.16f, g[1] = %18.16f, me[i] = %18.16f \n \n", tri, g[1]*me[i], g[1], me[i]);
	fu_y[tri * 6 + i] += g[1]*me[i];
	ind_x[tri * 6 + i] = tt[th * 6 + i]*cast; 	
      }
      
    }
    
  }
  
  __syncthreads();
  
  return;
}


__global__ void assembly_lapl_P1 (type_fd* x, type_fd* y, int* t_h, int nt, int n, type_uint nn,
			     type_fd *A_val, type_uint *MA_ind){


   /* x = x coordinate of P1  nodes
  
     y = y coordinate of P1  nodes
     

     t_h = connectivity P1 element table

     nt = number of elements

     n = number of P1 nodes
     
     nn = number of P2 nodes

     A_val = device vector of dimension (6*6) * nt that contains the values of matrix A in COO format with a unique index 

     MA_ind = device vector containing indices of M */


    int tri = blockIdx.x*blockDim.x + threadIdx.x;
    int th = threadIdx.x; // numero del thread dentro il blocco

    
    __shared__ type_fd coef3;
    __shared__ type_fd base2_2[3 * 3];
    __shared__ type_fd base2_3[3 * 3];
    __shared__ int  t[3 * BLOCKDIM];

    if (tri % blockDim.x == 0){
      coef3 = set_coef3();
      set_base2_2( base2_2 );
      set_base2_3( base2_3 );
    }
    __syncthreads();
    
    

  if (tri < nt){
     
    type_fd De[2 * 2];
    type_fd Je, inv_Je; // determinate di De

    type_fd dx[3 * 3] = {0.0};
    type_fd dy[3 * 3] = {0.0};
    type_fd temp_Ax[3 * 3] = {0.0}; // (diag(coef3)*dxi)'
    type_fd temp_Ay[3 * 3] = {0.0}; // (diag(coef3)*dyi)'
    type_fd temp_Ax_dx[3 * 3] = {0.0};
    type_fd temp_Ay_dy[3 * 3] = {0.0};

    
    int i,j, indice, indice_tra;


    // porto in memori shared la tabella di connettivita'
    load_t(t, t_h, tri, th);

    // calcolo il determinate di De

    jacobianoP1(x, y, t, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);
    

    // riempio i vettori dx dy
    for(i = 0; i < 3 * 3; ++i){
      dx[i] = base2_2[i]*De[0] + base2_3[i]*De[1];
      dy[i] = base2_2[i]*De[2] + base2_3[i]*De[3];
    }
    
     // calcolo delle matrici ausiliarie
    for(i = 0; i < 3 ; ++i){
      for(j = 0; j < 3; ++j){
 	indice = i * 3 + j;
 	indice_tra = j * 3 + i;
 	temp_Ax[indice_tra] = coef3 * dx[indice];
 	temp_Ay[indice_tra] = coef3 * dy[indice];
      }
    }


    // calcolo temp_Ax_dx = (diag(coef3)*dxi)' * dxj
    Prodotto_Matrici(temp_Ax, dx, temp_Ax_dx, 3, 3, 3, 3);

    // calcolo temp_Ay_dy = (diag(coef3)*dyi)' * dyj
    Prodotto_Matrici(temp_Ay, dy, temp_Ay_dy, 3, 3, 3, 3);

    
    
    inv_Je = 1.0/Je;  
    for(i = 0; i < 3; ++i){
      for(j = 0; j < 3; ++j){
     	indice = i * 3 + j;
	A_val[tri * 9 + indice] = (temp_Ax_dx[indice] + temp_Ay_dy[indice]) * inv_Je;
	MA_ind[tri * 9 + indice] = t[th * 3 + i] * nn + t[th * 3  + j];
      }
    }
     
  }
  return;
}


__global__ void assembly_mass_P1 (type_fd *x, type_fd *y, int *t_h, int nt, int n, type_uint nn,
			     type_fd *M_val, type_uint *MA_ind){

   /* x = x coordinate of P1  nodes
  
     y = y coordinate of P1  nodes


     t_h = connectivity P1 element table

     nt = number of elements

     n = number of P1 elements

     nn = number of P2 nodes

     M_val = device vector of dimension (6*6) * nt that contains the values of matrix M in COO format with a unique index 

     MA_ind = device vector containing indices of M */


  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco

    
  __shared__ type_fd coef3;
  __shared__ type_fd base1[3 * 3];
  __shared__ int  t[3 * BLOCKDIM];

  if (tri % blockDim.x == 0){
    coef3 = set_coef3();
    set_base1( base1 );
  }
  __syncthreads();
    
    

  if (tri < nt){
     
    type_fd De[2 * 2];
    type_fd Je; // determinate di De

    type_fd temp_M[3 * 3] = {0.0}; // (diag(coef3)*bli)'
    type_fd Me[3 * 3] = {0.0};

    
    int i,j, indice, indice_tra;


    // porto in memori shared la tabella di connettivita'
    load_t(t, t_h, tri, th);

    // calcolo il determinate di De

    jacobianoP1(x, y, t, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);

    // calcolo della matrice ausiliaria
    for(i = 0; i < 3 ; ++i){
      for(j = 0; j < 3; ++j){
 	indice = i * 3 + j;
 	indice_tra = j * 3 + i;
 	temp_M[indice_tra] = coef3 * base1[indice];
      }
    }


    // calcolo Me =  (diag(coef3)*bli)' * blj;
    Prodotto_Matrici(temp_M, base1, Me, 3, 3, 3, 3);


    for(i = 0; i < 3; ++i){
      for(j = 0; j < 3; ++j){
 	indice = i * 3 + j;
 	M_val[tri * 9 + indice] = Me[indice] * Je;
	MA_ind[tri * 9 + indice] = t[th * 3 + i] * nn + t[th * 3  + j];
      }
    }
     
  }
  return;
}


__global__ void assembly_mass_P2P1 (type_fd *xx, type_fd *yy, int *tt_h, int nt, int n, type_uint nn,
			       type_fd *M_P1_val, type_uint *M_P1_ind, type_fd *M_P2_val, type_uint *M_P2_ind){

   /* xx = x coordinate of P2  nodes
  
     yy = y coordinate of P2  nodes

     tt_h = connectivity P2 element table

     nt = number of elements

     n = number of P1 nodes

     nn = number of P2 nodes

     M_P1_val = device vector of dimension (6*6) * nt that contains the values of matrix M_P1 in COO format with a unique index 

     M_P1_ind = device vector containing indices of M_P1

     M_P2_val = device vector of dimension (6*6) * nt that contains the values of matrix M_P2 in COO format with a unique index 
  
     M_P2_ind = device vector containing indices of matrix M_P2 */


  int tri = blockIdx.x*blockDim.x + threadIdx.x;
  int th = threadIdx.x; // numero del thread dentro il blocco

    
  __shared__ type_fd coef7[7];
  __shared__ type_fd base1[7 * 3];
  __shared__ type_fd base2[7 * 6];
  __shared__ int  tt[6 * BLOCKDIM];

  if (tri % blockDim.x == 0){
    set_coef7( coef7 );
    set_base1m( base1 );
    set_base2m( base2 );
  }
  __syncthreads();
    
    

  if (tri < nt){
     
    type_fd De[2 * 2];
    type_fd Je; // determinate di De

    type_fd temp_M_P1[3 * 3] = {0.0}; // (diag(coef3)*bli)'
    type_fd Me_P1[3 * 7] = {0.0};
    type_fd temp_M_P2[6 * 7] = {0.0}; // (diag(coef7)*bli)'
    type_fd Me_P2[6 * 6] = {0.0};

    
    int i,j, indice, indice_tra;


    // porto in memori shared la tabella di connettivita'
    load_tt(tt, tt_h, tri, th);

    // calcolo il determinate di De

    jacobianoP2(xx, yy, tt, th, De);
    Je = (De[0] * De[3]) - (De[1] * De[2]);

    // P1-Mass
    // calcolo della matrice ausiliaria
    for(i = 0; i < 7 ; ++i){
      for(j = 0; j < 3; ++j){
 	indice = i * 7 + j;
 	indice_tra = j * 3 + i;
 	temp_M_P1[indice_tra] = coef7[i] * base1[indice];
      }
    }

    // calcolo Me =  (diag(coef3)*bli)' * blj;
    Prodotto_Matrici(temp_M_P1, base1, Me_P1, 3, 7, 7, 3);
    
    for(i = 0; i < 3; ++i){
      for(j = 0; j < 3; ++j){
 	indice = i * 3 + j;
 	M_P1_val[tri * 9 + indice] = Me_P1[indice] * Je;
	M_P1_ind[tri * 9 + indice] = tt[th * 6 + i] * nn + tt[th * 6  + j];
      }
    }

    // P2-Mass
    // calcolo della matrice ausiliaria
    for(i = 0; i < 7 ; ++i){
      for(j = 0; j < 6; ++j){
 	indice = i * 6 + j;
 	indice_tra = j * 7 + i;
	temp_M_P2[indice_tra] = coef7[i] * base2[indice];
      }
    }

    // calcolo Me =  (diag(coef3)*bli)' * blj;
    Prodotto_Matrici(temp_M_P2, base2, Me_P2, 6, 7, 7, 6);
    
    for(i = 0; i < 6; ++i){
      for(j = 0; j < 6; ++j){
 	indice = i * 6 + j;
 	M_P2_val[tri * 36 + indice] = Me_P2[indice] * Je;
	M_P2_ind[tri * 36 + indice] = tt[th * 6 + i] * nn + tt[th * 6  + j];
      }
    }
     
  }
  return;
}


