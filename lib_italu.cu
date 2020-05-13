#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>

typedef double    type_fd;
#define DOUBLE_PRECISION

#define TILE_DIM 16
#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define maxit 3000
#define BLOCK 256
#define WARP_SIZE 32
#define NNZ 32

#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
    {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

cusparseHandle_t cusparseHandle;
cublasHandle_t cublasHandle;

cudaError_t cudaStat1 = cudaSuccess, cudaStat2 = cudaSuccess,
  cudaStat3 = cudaSuccess, cudaStat4 = cudaSuccess, cudaStat5 = cudaSuccess, cudaStat6 = cudaSuccess, cudaStat7 = cudaSuccess, cudaStat8 = cudaSuccess;
cusparseStatus_t cusparseStatus1 = CUSPARSE_STATUS_SUCCESS, cusparseStatus2 = CUSPARSE_STATUS_SUCCESS, cusparseStatus3 = CUSPARSE_STATUS_SUCCESS;

type_fd one = 1.0, oneopp = -1.0, zero = 0.0;

cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
cusparseOperation_t trans_t = CUSPARSE_OPERATION_TRANSPOSE;

int imin(int a, int b){
  if (a<b)
    return a;
  else
    return b;  
}

struct is_nonzero
{

  __host__ __device__
  bool operator()(const type_fd x)
  {
    return (x != 0.0);
  }
  
};

struct is_zero
{

  __host__ __device__
  bool operator()(const type_fd x)
  {
    return (x == 0.0);
  }
  
};
  
struct is_nonneg
{
  
  __host__ __device__
  bool operator()(const int x)
  {
    return (x >= 0);
  }

};

extern "C" void cusparse_init(){
  cusparseStatus_t cusparseStatus;
  
  cusparseStatus= cusparseCreate(&cusparseHandle);
  if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    printf("CUSPARSE Library initialization failed\n");
    return;
  }

  cusparseSetPointerMode(cusparseHandle, CUSPARSE_POINTER_MODE_HOST);
  
  return;  
}

extern "C" void cusparse_end(){ 
  cusparseDestroy(cusparseHandle);
  return;
}

extern "C" void cublas_init(){
  cublasStatus_t cublasStatus;
  
  cublasStatus= cublasCreate(&cublasHandle);
  if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS Library initialization failed\n");
    return;
  }
  
  return;  
}

extern "C" void cublas_end(){ 
  cublasDestroy(cublasHandle);
  return;
}



extern "C" cusparseMatDescr_t createMatDescrA(){
  cusparseMatDescr_t descrA;
  cusparseStatus_t cusparseStatus;
  
  cusparseStatus = cusparseCreateMatDescr(&descrA);
  if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    printf("createMatDescrA failed\n");
    return descrA;
  }
  
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  
  return descrA;
}


extern "C" cusparseMatDescr_t createMatDescrL(){
  cusparseMatDescr_t descrL;
  cusparseStatus_t cusparseStatus;

  cusparseStatus= cusparseCreateMatDescr(&descrL);
  if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    printf("createMatDescrL failed\n");
    return descrL;
  }
  cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
  cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_UNIT);

  return descrL;
}

extern "C" cusparseMatDescr_t createMatDescrU(){
  cusparseMatDescr_t descrU;
  cusparseStatus_t cusparseStatus;

  cusparseStatus= cusparseCreateMatDescr(&descrU);
  if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
    printf("createMatDescrU failed\n");
    return descrU;
  }

  cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER);
  cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT);

  return descrU;
}


__device__ int my_ceil(type_fd r){
  int n = ceil( (double) r);
  
  return n;
}

__global__ void arange(int *array, int n){
  
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id<n)
    array[id] = id;
  
  return;
}

extern "C" void axpby(int n, type_fd alpha, type_fd *x, type_fd beta, type_fd *y){
#ifdef DOUBLE_PRECISION
  cublasDscal(cublasHandle, n, &beta, y, 1);
  cublasDaxpy(cublasHandle, n, &alpha, x, 1, y, 1);
#else
  cublasSscal(cublasHandle, n, &beta, y, 1);
  cublasSaxpy(cublasHandle, n, &alpha, x, 1, y, 1);
#endif
  return;
}

extern "C" void csrilu0(int n, int nnz, type_fd *csrVal, int *csrRowPtr, int *csrColInd, type_fd *exec_time){
  cusparseSolveAnalysisInfo_t info;
  cusparseMatDescr_t descrA;

  cusparseCreateSolveAnalysisInfo(&info);

  descrA = createMatDescrA();

#ifdef DOUBLE_PRECISION
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info);
#else
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info);
#endif

  clock_t begin = clock();
#ifdef DOUBLE_PRECISION
  cusparseDcsrilu0(cusparseHandle, trans, n, descrA, csrVal, csrRowPtr, csrColInd, info);
#else
  cusparseScsrilu0(cusparseHandle, trans, n, descrA, csrVal, csrRowPtr, csrColInd, info);
#endif
  clock_t end = clock();
  type_fd h_exec_time = ((type_fd) (end - begin)) / CLOCKS_PER_SEC;
  
  cudaStat1 = cudaMemcpy(exec_time, &h_exec_time,  (size_t)(sizeof(exec_time[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy ERROR\n");
  
  cusparseDestroyMatDescr(descrA);
  cusparseDestroySolveAnalysisInfo(info);
  return;
}



__global__  void spmv_csr_vector_kernel(const  int lower_tri,
					const  int  num_rows,
					const  int    * ptr,
					const  int    * indices,
					const  type_fd * data ,
					const  type_fd * x,
					type_fd * y){
  extern volatile __shared__  type_fd vals [];
  type_fd value;
  int  thread_id = blockDim.x * blockIdx.x + threadIdx.x;//  global  thread  index
  int  warp_id    = thread_id / 32;  //  global  warp  index
  int  lane       = thread_id & (32 - 1);//  thread  index  within  the  warp
  //  one  warp  per  row
  int  ind, row   = warp_id;

  if (row < num_rows ){
    int  row_start = ptr[row];
    int  row_end   = ptr[row +1]; //  compute  running  sum  per  thread
    vals[threadIdx.x] = 0;
    for(int jj = row_start + lane; jj < row_end; jj += 32){
      ind = indices[jj];
      if ((lower_tri == 1) and (ind == row))
      	value  = 1.0;
      else if ((lower_tri == 0) && (ind < row)) // la matrice e' triangolare superiore
      	value = 0.0;
      else if ((lower_tri == 1) && (ind > row)) // la matrice e' triangolare inferiore
      	value = 0.0;
      else
      	value = data[jj];

      vals[threadIdx.x] += value * x[ind];
    }
    
    //  parallel  reduction  in  shared  memory
    if (lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16];
    if (lane <  8) vals[threadIdx.x] += vals[threadIdx.x +  8];
    if (lane <  4) vals[threadIdx.x] += vals[threadIdx.x +  4];
    if (lane <  2) vals[threadIdx.x] += vals[threadIdx.x +  2]; 
    if (lane <  1) vals[threadIdx.x] += vals[threadIdx.x +  1];
    //  first  thread  writes  the  result
    if (lane == 0)
      y[row] = vals[threadIdx.x];
  }
  return;
}

extern "C" void csrmv(const int op_trans, int m,  int n, int nnz, type_fd *csrVal, int *csrRowPtr, int *csrColInd, type_fd *x, type_fd *y){
  cusparseMatDescr_t descrA;
  
  descrA = createMatDescrA();
  
  if (op_trans == 0){
#ifdef DOUBLE_PRECISION
    cusparseDcsrmv(cusparseHandle, trans, m, n, nnz, &one, descrA, csrVal, csrRowPtr, csrColInd, x, &zero, y);
#else
    cusparseScsrmv(cusparseHandle, trans, m, n, nnz, &one, descrA, csrVal, csrRowPtr, csrColInd, x, &zero, y);
#endif
  }else if(op_trans == 1){
#ifdef DOUBLE_PRECISION
    cusparseDcsrmv(cusparseHandle, trans_t, m, n, nnz, &one, descrA, csrVal, csrRowPtr, csrColInd, x, &zero, y);
#else
    cusparseScsrmv(cusparseHandle, trans_t, m, n, nnz, &one, descrA, csrVal, csrRowPtr, csrColInd, x, &zero, y);
#endif
  }else{
    printf("Error cusparse<t>csrmv: wrong trans value. Trans must be 0 (non transpose) or 1 (transpose)\n");
    return;
  }

  cusparseDestroyMatDescr(descrA);
  return;
}


__global__ void approximate_vals_LU(type_fd *csrVal, int *csrRowPtr, int *csrColInd, int k, int n){
  // this kernel deletes matrix elements outside the k-th sub/super diaginals

  int id = threadIdx.x + blockIdx.x*blockDim.x;
  int beg, end; // j
  
  if (id<n){
    beg = csrRowPtr[id];
    end = csrRowPtr[id+1];

    for (int i = beg; i<end; i++){
      if (abs(csrColInd[i]-id) > k)
	csrVal[i] = 0.0;
    }
    
  }
  
  return;
}

__global__ void count_nnz_cutM_kernel(int lower_tri, int *csrRowPtrA, int *csrColIndA, int *cut, int *nnz_cut, int ncut){
  // this kernels counts how many elements are going to be deleted in L (if lower_tri = 1) or U (lower_tri = 0)
  // when the DAG is being cut
  
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  int j, beg, end, nnzloc, row;

  if (id < ncut){
    row = cut[id];

    beg = csrRowPtrA[row];
    end = csrRowPtrA[row+1];

    nnzloc = 0;
    
    for(j = beg; j < end; j++){
      if (lower_tri == 1){
	if (csrColIndA[j] < row){
	  nnzloc += 1;
	}
      }else if (lower_tri == 0){
	if (csrColIndA[j] > row){
	  nnzloc += 1;
	}
      }
    }

    nnz_cut[id] = nnzloc;
  }

  return;
}


extern "C" void count_nnz_cutM(type_fd* csrValM, int *csrRowPtrA, int *csrColIndA, int ncutL, int *cutL, int ncutU, int *cutU, int *nnz){
  /* Matrix cointaining LU factors: csrValM, csrRowPtrA, csrColIndA
     cutL/cutU: vectors containing the indices of rows to be diagonalised
     ncutL/ncutU: length of cutL/cutU
     nnz: in input it is the number of non zeros LU before the procedure,in output it is the new number of 
          non zeros of LU
 */
  int *nnzL, *nnzU, tot_nnzL, tot_nnzU, nnzo;
  
  cudaMalloc((void**)&nnzL,  (ncutL)*sizeof(nnzL[0])); //nnz da tagliare dalle righe cutL di M
  cudaMalloc((void**)&nnzU,  (ncutU)*sizeof(nnzU[0])); //nnz da tagliare dalle righe cutU di M

  cudaMemcpy(&nnzo, nnz,  (size_t)(sizeof(nnzo)), cudaMemcpyDeviceToHost); //nnz = numero di nnz di M prima del taglio

  count_nnz_cutM_kernel<<< (int) (ncutL/BLOCK +1), BLOCK>>>(1, csrRowPtrA, csrColIndA, cutL, nnzL, ncutL);

  count_nnz_cutM_kernel<<< (int) (ncutU/BLOCK +1), BLOCK>>>(0, csrRowPtrA, csrColIndA, cutU, nnzU, ncutU);

  thrust::device_ptr<int> nnzL_ptr(nnzL);
  thrust::device_ptr<int> nnzU_ptr(nnzU);

  tot_nnzL = thrust::reduce(nnzL_ptr, nnzL_ptr + ncutL); // numero totale di nnz tagliati da L
  tot_nnzU = thrust::reduce(nnzU_ptr, nnzU_ptr + ncutU); // numero totale di nnz tagliati da L
  
  nnzo = nnzo - tot_nnzL - tot_nnzU; 

  cudaMemcpy(nnz, &nnzo,  (size_t)(sizeof(nnz[0])), cudaMemcpyHostToDevice); //nnz = numero di nnz di M dopo del taglio

  cudaFree(nnzL);
  cudaFree(nnzU);
  return;
}



__global__ void nnzXrow_kernel(int *csrRowPtr, int *nnz, int n){
  // This function count the number of non zeros elements per row of a CSR matrix
  
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id < n){
    nnz[id] = csrRowPtr[id+1]- csrRowPtr[id];
  }

  return;
}

__global__ void sistemaPtrCut_kernel(int *nnzXrow,  int *cut, int *nnzcut, int ncut){
  // this function generates row pointer vector of CSR Matrix after DAG cutting
  
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id < ncut){
    int row = cut[id];
    nnzXrow[row] =  nnzXrow[row] - nnzcut[id];
  }

  return;
}

__global__ void sistemaPtr_kernel(int *nnzXrow, int *zeros, int n){
  // this function generates row pointer vector of CSR Matrix after DAG cutting
  

  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id < n){
    nnzXrow[id] =  nnzXrow[id] - zeros[id];
  }

  return;
}

__global__ void cutM_colInd_kernel(int lower_tri, int *csrRowPtrA, int *csrColIndA, int *cut, int *nnz_cut, int ncut){
  // this kernels sets to -1 the column indices of elements to be deleted of rows within cut
  // if lower_tri = 1, it processes lower part, otherwise the upper part if lower_tri = 0

  int id = threadIdx.x + blockIdx.x*blockDim.x;
  int j, beg, end, nnzloc, row;

  if (id < ncut){
    row = cut[id];

    beg = csrRowPtrA[row];
    end = csrRowPtrA[row+1];

    nnzloc = 0;
    
    for(j = beg; j < end; j++){
      if (lower_tri == 1){
	if (csrColIndA[j] < row){
	  csrColIndA[j] = -1;
	  nnzloc += 1;
	}
      }else if (lower_tri == 0){
	if (csrColIndA[j] > row){
	  csrColIndA[j] = -1;
	  nnzloc +=1;
	}
      }
    }
    nnz_cut[id] = nnzloc;
    
  }

  return;
}

extern "C" void cutColIndM(int n, int nnzA, type_fd* csrValM, int *csrRowPtrA, int *csrColIndA, int *csrRowPtrM, int *csrColIndM, int ncutL, int *cutL, int ncutU, int *cutU){
  /* this function computes the sparsity pattern of  L,U matrices after DAG cut 
     Input L,U csr matrices saved together in M: csrValM, csrRowPtrA, csrColIndA
     CSR pattern after DAG cut: csrRowPtrM, csrColIndM
     cutL/cutU: vectors containing rows indices of L/U of rows to be diagonalized
     ncutL/ncutU: length of cutL/cutU
  */
  int *csrColIndcpy, *zerosL, *zerosU;

  cudaMalloc((void**)&zerosL,  (ncutL)*sizeof(zerosL[0])); 
  cudaMalloc((void**)&zerosU,  (ncutU)*sizeof(zerosU[0]));
  
  cudaMalloc((void**)&csrColIndcpy,  (nnzA)*sizeof(csrColIndcpy[0]));
  cudaMemcpy(csrColIndcpy, csrColIndA,  (size_t)(sizeof(csrColIndcpy[0])*nnzA), cudaMemcpyDeviceToDevice);

  cutM_colInd_kernel<<< (int) (ncutL/BLOCK +1), BLOCK>>>(1, csrRowPtrA, csrColIndcpy, cutL, zerosL, ncutL);
  cutM_colInd_kernel<<< (int) (ncutU/BLOCK +1), BLOCK>>>(0, csrRowPtrA, csrColIndcpy, cutU, zerosU, ncutU);


  thrust::device_ptr<int> csrColIndcpy_ptr(csrColIndcpy);
  thrust::device_ptr<int> csrColIndM_ptr(csrColIndM);
  thrust::device_ptr<int> csrRowPtrM_ptr(csrRowPtrM);

  thrust::copy_if(csrColIndcpy_ptr, csrColIndcpy_ptr + nnzA, csrColIndM_ptr, is_nonneg());

  nnzXrow_kernel<<< (int) (n/BLOCK +1), BLOCK>>>(csrRowPtrA, csrRowPtrM, n); 

  sistemaPtrCut_kernel<<< (int) (ncutL/BLOCK +1), BLOCK>>>(csrRowPtrM, cutL, zerosL, ncutL);
  sistemaPtrCut_kernel<<< (int) (ncutU/BLOCK +1), BLOCK>>>(csrRowPtrM, cutU, zerosU, ncutU);
  thrust::exclusive_scan(csrRowPtrM_ptr, csrRowPtrM_ptr + (n+1), csrRowPtrM_ptr);

  cudaFree(csrColIndcpy);
  cudaFree(zerosL);
  cudaFree(zerosU);
  return;
}

__global__ void cutM_vals_kernel(int lower_tri, type_fd* csrValM, int *csrRowPtrA, int *csrColIndA, int *cut, int ncut){
  //this kernel zeros out the values of rows of L (lower_tri = 1) or U (lower_tri = 0) with indices in ncut after DAG cut
  
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  int j, beg, end, row;

  if (id < ncut){
    row = cut[id];

    beg = csrRowPtrA[row];
    end = csrRowPtrA[row+1];

    for(j = beg; j < end; j++){
      if (lower_tri == 1){
	if (csrColIndA[j] < row){
	  csrValM[j] = 0.0;
	}
      }else if (lower_tri == 0){
	if (csrColIndA[j] > row){
	  csrValM[j] = 0.0;
	}
      }
    }

  }

  return;
}

extern "C" void cutValM(int n, int nnzA, type_fd* csrValM, int *csrRowPtrA, int *csrColIndA,
			int nnzM, type_fd* csrValMcut, int *csrRowPtrM, int *csrColIndM,
			int ncutL, int *cutL, int ncutU, int *cutU){
 /* this function computes the sparsity pattern of  L,U matrices after DAG cut 
     Input L,U csr matrices saved together in M: csrValM, csrRowPtrA, csrColIndA
     CSR pattern after DAG cut: csrRowPtrM, csrColIndM (previously computed)
     CSR values after DAG cut: csrValMcut
     cutL/cutU: vectors containing rows indices of L/U of rows to be diagonalized
     ncutL/ncutU: length of cutL/cutU
  */

  type_fd *csrValMcpy;
  
  cudaMalloc((void**)&csrValMcpy,  (nnzA)*sizeof(csrValMcpy[0]));

#ifdef DOUBLE_PRECISION
  cublasDcopy(cublasHandle, nnzA, csrValM, 1, csrValMcpy, 1);
#else
  cublasScopy(cublasHandle, nnzA, csrValM, 1, csrValMcpy, 1);
#endif

  cutM_vals_kernel<<< (int) (ncutL/BLOCK +1), BLOCK >>>(1, csrValMcpy, csrRowPtrA, csrColIndA, cutL, ncutL);
  cutM_vals_kernel<<< (int) (ncutU/BLOCK +1), BLOCK >>>(0, csrValMcpy, csrRowPtrA, csrColIndA, cutU, ncutU);

  thrust::device_ptr<type_fd> csrValMcpy_ptr(csrValMcpy);
  thrust::remove_if(csrValMcpy_ptr, csrValMcpy_ptr + nnzA, is_zero());

#ifdef DOUBLE_PRECISION
  cublasDcopy(cublasHandle, nnzM, csrValMcpy, 1, csrValMcut, 1);
#else
  cublasScopy(cublasHandle, nnzM, csrValMcpy, 1, csrValMcut, 1);
#endif

  cudaFree(csrValMcpy);
  return;
}


__global__ void count_nnz_ILUT_kernel(type_fd *csrVal, int *csrRowPtr, int *csrColInd, int *nnz_vec, type_fd t, int n){
  // Kernel che elimina gli elementi di una matrice CSR al di fuori della diagonale k-esima

  int id = threadIdx.x + blockIdx.x*blockDim.x;
  int i, beg, end, nnz_loc;
  
  if (id<n){
    nnz_loc = 0;
    beg = csrRowPtr[id];
    end = csrRowPtr[id+1];
    
    for (i = beg; i < end; i++){
      if ((abs(csrVal[i])<t) && (csrColInd[i] != id))
	nnz_loc += 1;
    }
    nnz_vec[id] = nnz_loc;
  }
  return;
}


extern "C" void count_nnz_ILUT(int n, int *nnzA, type_fd* csrValM, int *csrRowPtrA, int *csrColIndA, int *csrRowPtrM, type_fd t){

  int *zeros_r, nnz, nnzo, zeros;

  cudaMalloc((void**)&zeros_r, n*sizeof(zeros_r[0]));
  cudaMemcpy(&nnzo, nnzA,  (size_t)(sizeof(nnzo)), cudaMemcpyDeviceToHost);

  count_nnz_ILUT_kernel<<< (int) (n/BLOCK +1), BLOCK>>>(csrValM, csrRowPtrA, csrColIndA, zeros_r, t, n);

  thrust::device_ptr<int> zeros_r_ptr(zeros_r);
  thrust::device_ptr<int> csrRowPtrM_ptr(csrRowPtrM);

  zeros = thrust::reduce(zeros_r_ptr, zeros_r_ptr + n);

  nnz = nnzo-zeros;
  cudaMemcpy(nnzA, &nnz,  (size_t)(sizeof(nnzA[0])), cudaMemcpyHostToDevice);
  
  nnzXrow_kernel<<< (int) (n/BLOCK +1), BLOCK>>>(csrRowPtrA, csrRowPtrM, n); 
  sistemaPtr_kernel<<< (int) (n/BLOCK +1), BLOCK>>>(csrRowPtrM, zeros_r, n);
  thrust::exclusive_scan(csrRowPtrM_ptr, csrRowPtrM_ptr + (n+1), csrRowPtrM_ptr);

  cudaFree(zeros_r);
  return;
}


void ApplyPlaneRotation(type_fd *dx, type_fd *dy, type_fd cs, type_fd sn) {
  type_fd temp = cs * (*dx) + sn * (*dy);
  *dy = -sn*(*dx)+cs*(*dy);
  *dx = temp;
  
  return;
}

void GeneratePlaneRotation(type_fd dx, type_fd dy, type_fd *cs, type_fd *sn) {
  //matlab rotmat
  
  if(dy == 0.0){
    *cs = 1.0;
    *sn = 0.0;
  }else if ( abs(dy) > abs(dx) ) {
    type_fd tmp = dx / dy;
    *sn = 1.0 / sqrt(1.0 + tmp*tmp);
    *cs = tmp*(*sn);            
  }else {
    type_fd tmp = dy / dx;
    *cs = 1.0 / sqrt(1.0 + tmp*tmp);
    *sn = tmp*(*cs);
  }

  return;
}

void PlaneRotation(type_fd *H, int nbcol, type_fd *cs, type_fd *sn, type_fd *s, int i){
  //int ii, jj;
  
  for (int k = 0; k < i; k++){
    ApplyPlaneRotation(&H[i + nbcol*k], &H[i + nbcol*(k+1)], cs[k], sn[k]);
  }
  
  GeneratePlaneRotation(H[i + nbcol*i], H[i + nbcol*(i+1)], &cs[i], &sn[i]);
  
  ApplyPlaneRotation(&H[i*nbcol + i], &H[i + (i+1)*nbcol], cs[i], sn[i]);
  ApplyPlaneRotation(&s[i], &s[i+1], cs[i], sn[i]);

  return;
}

__global__ void solve_diag_system(type_fd *D, type_fd *b, type_fd *x, int n){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id<n){
    x[id] = b[id]/D[id];
  }
  return;
}

__global__ void csrExtractDiag(type_fd *csrVal, int *csrRowPtr, int *csrColInd, type_fd *diag, int n){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id<n){
    for(int j = csrRowPtr[id]; j < csrRowPtr[id+1]; j++){
      if(csrColInd[j] == id){
        diag[id] = csrVal[j];
	break;
      }
    }
  }
  
  return;
}


__global__ void csrSetDiag(type_fd *csrVal, int *csrRowPtr, int *csrColInd, type_fd *diag, int n){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id<n){
    for(int j = csrRowPtr[id]; j < csrRowPtr[id+1]; j++){
      if(csrColInd[j] == id){
        csrVal[j] = diag[id];
      }
    }
  }
  
  return;
}

__global__ void csrExtractInvDiag(type_fd *csrVal, int *csrRowPtr, int *csrColInd, type_fd *diag, int n){
  // this kernel extracts diagonal elements of a CSR matrix and computes their inverses
  // NB: column indices must be sorted in ascending order

  int id = threadIdx.x + blockIdx.x*blockDim.x;
  int j;

  if (id<n){
    j = csrRowPtr[id];
    while (id > csrColInd[j]) j++;
    diag[id] = 1.0/csrVal[j];
  }
   
  return;
}

__global__ void csrScalLowerTri(type_fd *csrValL, type_fd *values, int *csrRowPtr, int *csrColInd, int n){
  // NB: column indices must be sorted in ascending order
  //L = L*D post multiplication of matrix L times a diagonal matrix D = diag(values)
  
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  int j;
  
  if (id<n){
    for (j = csrRowPtr[id]; j < csrRowPtr[id+1]; j++){
      if (csrColInd[j] < id)
	csrValL[j] = csrValL[j]*values[j];
    }
  }
  
  return;
  
}

__global__ void multiply(type_fd *x, type_fd *y, int n){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id<n){
    y[id] = y[id]*x[id];
  }  
  return;
}

__global__ void sum(type_fd *x, type_fd *y, int n){
  int id = threadIdx.x + blockIdx.x*blockDim.x;

  if (id<n){
    y[id] = y[id]+x[id];
  }  
  return;
}

extern "C" void wrap_extract_diag(type_fd *csrVal, int *csrRowPtr, int* csrColInd, int n, type_fd *diag){

  csrExtractDiag<<< (int) (n/BLOCK +1), BLOCK >>>(csrVal, csrRowPtr, csrColInd, diag, n);
  
  return;
}


extern "C" void scalar_jacobi_relaxation(int iters, int n, int lower_tri, type_fd *csrVal, int *csrRowPtr, int *csrColInd, type_fd *diagUinv, type_fd *x, type_fd *x0, type_fd *rhs){
  int i;


  if (lower_tri == 0){ //Only for U factor
    // rhs = Dinv*rhs
    multiply<<< (int) (n/BLOCK +1), BLOCK >>>(diagUinv, rhs, n); 
  }
  // x0 = rhs
#ifdef DOUBLE_PRECISION
  cublasDcopy(cublasHandle, n, rhs, 1, x0, 1);
#else
  cublasScopy(cublasHandle, n, rhs, 1, x0, 1);
#endif


  for (i = 0; i < iters; i++){
#ifdef DOUBLE_PRECISION
    //x = M*x0 
    spmv_csr_vector_kernel<<< (int) (32*n/BLOCK +1), BLOCK,  BLOCK*sizeof(type_fd) >>>(lower_tri, n, csrRowPtr, csrColInd, csrVal , x0,  x);    
    if (lower_tri == 0){ //Only for U factor
      // x = Dinv*x
      multiply<<< (int) (n/BLOCK +1), BLOCK >>>(diagUinv, x, n); 
    }
    //x0 = -x + x0
    cublasDaxpy(cublasHandle, n, &oneopp, x, 1, x0, 1);
    //x0 = x0 + b
    cublasDaxpy(cublasHandle, n, &one, rhs, 1, x0, 1);
#else
    //x = M*x0 
    spmv_csr_vector_kernel<<< (int) (32*n/BLOCK +1), BLOCK,  BLOCK*sizeof(type_fd) >>>(lower_tri, n, csrRowPtr, csrColInd, csrVal , x0,  x);
    if (lower_tri == 0){ //Only for U factor
      // x = Dinv*x
      multiply<<< (int) (n/BLOCK +1), BLOCK >>>(diagUinv, x, n); 
    }    
    //x0 = -x + x0
    cublasSaxpy(cublasHandle, n, &oneopp, x, 1, x0, 1);
    //x0 = x0 + b
    cublasSaxpy(cublasHandle, n, &one, rhs, 1, x0, 1);
#endif
  }
  
  return;
}

void scalar_jacobi(int iters, int n, type_fd *csrVal, int *csrRowPtr, int *csrColInd, type_fd *diagUinv, type_fd *w, type_fd *y, type_fd *r0){
  //Ly = r0
  scalar_jacobi_relaxation(iters, n, 1, csrVal, csrRowPtr, csrColInd, diagUinv, w, y, r0);
  //Uw = y
  scalar_jacobi_relaxation(iters, n, 0, csrVal, csrRowPtr, csrColInd, diagUinv, r0, w, y);
  return;
}

void block_jacobi_relaxation(int iters, int n, int lower_tri, cusparseMatDescr_t descr, cusparseSolveAnalysisInfo_t info,
			     type_fd *csrValT, int *csrRowPtrT, int *csrColIndT,
			     type_fd *csrValD, int *csrRowPtrD, int *csrColIndD,
			     type_fd *y, type_fd *z, type_fd *b, type_fd *x){
  int i;

  //Solve x = D^{-1}b and set b = x
#ifdef DOUBLE_PRECISION
  cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descr, csrValD, csrRowPtrD, csrColIndD, info, b, x);
  cublasDcopy(cublasHandle, n, x, 1, b, 1);
#else
  cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descr, csrValD, csrRowPtrD, csrColIndD, info, b, x);
  cublasScopy(cublasHandle, n, x, 1, b, 1);
#endif

  for (i = 0; i < iters; i++){
#ifdef DOUBLE_PRECISION
    //y = T*x
    spmv_csr_vector_kernel<<< (int) (32*n/BLOCK +1), BLOCK,  BLOCK*sizeof(type_fd) >>>(lower_tri, n, csrRowPtrT, csrColIndT, csrValT, x,  y);
    // Solve z = D^{-1}*y
    cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descr, csrValD, csrRowPtrD, csrColIndD, info, y, z);
    //x = -z + x
    cublasDaxpy(cublasHandle, n, &oneopp, z, 1, x, 1);
    //x = b + x
    cublasDaxpy(cublasHandle, n, &one, b, 1, x, 1);
#else
    //y = T*x
    spmv_csr_vector_kernel<<< (int) (32*n/BLOCK +1), BLOCK,  BLOCK*sizeof(type_fd) >>>(lower_tri, n, csrRowPtrT, csrColIndT, csrValT, x,  y);
    // Solve z = D^{-1}*y
    cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descr, csrValD, csrRowPtrD, csrColIndD, info, y, z);
    //x = -z + x
    cublasSaxpy(cublasHandle, n, &oneopp, z, 1, x, 1);
    //x = b + x
    cublasSaxpy(cublasHandle, n, &one, b, 1, x, 1);
#endif
  }
  
  return;
}

void block_jacobi(int iters, int n,
		  type_fd *csrValM_full, int *csrRowPtrA, int *csrColIndA,
		  type_fd *csrValM, int *csrRowPtrM, int *csrColIndM,
		  cusparseSolveAnalysisInfo_t infoL,  cusparseSolveAnalysisInfo_t infoU,
		  cusparseMatDescr_t descrL, cusparseMatDescr_t descrU,
		  type_fd *xx, type_fd *w, type_fd *r0, type_fd *y){
  
  //Block Jacobi L*y = r0
  block_jacobi_relaxation(iters, n, 1, descrL, infoL,
			  csrValM_full, csrRowPtrA, csrColIndA,
			  csrValM, csrRowPtrM, csrColIndM,
			  xx, w, r0, y); //type_fd *y, type_fd *z, type_fd *b, type_fd *x)
  //Block Jacobi U*w = y
  block_jacobi_relaxation(iters, n, 0, descrU, infoU,
			  csrValM_full, csrRowPtrA, csrColIndA,
			  csrValM, csrRowPtrM, csrColIndM,
			  xx, r0, y, w);
  
  return;
}


extern "C" void GMRES_PREC(int m, type_fd tol,  int n,
			   const int prec, int nnzA, type_fd *csrValA, int *csrRowPtrA, int *csrColIndA,
			   type_fd *b, type_fd *x0, type_fd *exec_time, int *it) {
  //GMRES with no preconditioner / diagonal preconditioner / ILU(0) preconditioner
  
  //GMRES(m) variables
  int l, i, j, k; //, ii;
  type_fd *w, *r0, *V, *y; //on device
  type_fd *s, *cs, *sn, *H; //on host
  type_fd *diagM, *csrValLU, *diaglM;
  type_fd beta, bnrm2, betainv, inv, proj, res = 1.0; // one = 1.0, oneopp = -1.0, zero = 0.0,
  cusparseSolveAnalysisInfo_t info, infoL, infoU;
  cusparseMatDescr_t descrL, descrU, descrA;

  
  if (prec == 1){
    cusparseStatus1 = cusparseCreateSolveAnalysisInfo(&info);
    if (cusparseStatus1 != CUSPARSE_STATUS_SUCCESS)
      printf("cusparseCreateSolveAnalysisInfo failed in GMRES_PREC\n");

    cudaStat1 = cudaMalloc((void**)&diaglM,  (n)*sizeof(diaglM[0]));
    if (cudaStat1 != cudaSuccess) {
      printf("Device malloc failed in GMRES_PREC\n");
      return;
    }
  }
  if (prec == 2){
    cusparseStatus1 = cusparseCreateSolveAnalysisInfo(&info);
    cusparseStatus2 = cusparseCreateSolveAnalysisInfo(&infoU);
    cusparseStatus3 = cusparseCreateSolveAnalysisInfo(&infoL);
    if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
	(cusparseStatus2 != CUSPARSE_STATUS_SUCCESS) ||
	(cusparseStatus3 != CUSPARSE_STATUS_SUCCESS))
      printf("cusparseCreateSolveAnalysisInfo failed in GMRES_PREC\n");
    descrL = createMatDescrL();
    descrU = createMatDescrU();
  }

  descrA = createMatDescrA();
  
  s  = (type_fd *) calloc((m+1), sizeof(s[0]));
  cs = (type_fd *) calloc(m, sizeof(cs[0]));
  sn = (type_fd *) calloc(m, sizeof(sn[0]));
  H  = (type_fd *) calloc((m+1)*m, sizeof(H[0]));

  cudaStat1 = cudaMalloc((void**)&r0,     n*sizeof(r0[0]));
  cudaStat2 = cudaMalloc((void**)&w,      n*sizeof(w[0]));
  cudaStat3 = cudaMalloc((void**)&V,      (n*(m+1))*sizeof(V[0])); // m+1 vector basis of the Krylov approximation space
  cudaStat4 = cudaMalloc((void**)&y,      n*sizeof(y[0]));
  if (prec == 1)
    cudaStat5 = cudaMalloc((void**)&diagM,  n*sizeof(diagM[0]));
  if (prec == 2)
    cudaStat6 = cudaMalloc((void**)&csrValLU,  nnzA*sizeof(csrValLU[0]));

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ||
      (cudaStat5 != cudaSuccess)) {
    printf("Device malloc failed in GMRES_PREC\n");
    return;
  }

  clock_t begin = clock();
  if (prec == 1){
    csrExtractDiag<<< (int) (n/BLOCK +1), BLOCK >>>(csrValA, csrRowPtrA, csrColIndA, diagM, n);
    cudaMemset(diaglM, 0, n*sizeof(diaglM[0]));
  }
  if (prec == 2){
#ifdef DOUBLE_PRECISION   
    cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info);
    cublasDcopy(cublasHandle, nnzA, csrValA, 1, csrValLU, 1);
    cusparseDcsrilu0(cusparseHandle, trans, n, descrA, csrValLU, csrRowPtrA, csrColIndA, info);
    cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzA, descrL, csrValLU, csrRowPtrA, csrColIndA, infoL);
    cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzA, descrU, csrValLU, csrRowPtrA, csrColIndA, infoU);
#else
    cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, info);
    cublasScopy(cublasHandle, nnzA, csrValA, 1, csrValLU, 1);
    cusparseScsrilu0(cusparseHandle, trans, n, descrA, csrValLU, csrRowPtrA, csrColIndA, info);
    cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzA, descrL, csrValLU, csrRowPtrA, csrColIndA, infoL);
    cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzA, descrU, csrValLU, csrRowPtrA, csrColIndA, infoU);
#endif
  }
  
  l = 0;
#ifdef DOUBLE_PRECISION
  cublasDnrm2(cublasHandle, n, b, 1, &bnrm2);
#else
  cublasSnrm2(cublasHandle, n, b, 1, &bnrm2);
#endif
  if (bnrm2 == 0.0)
    bnrm2 = 1.0;

  while (res/bnrm2 > tol && l < maxit){
    
#ifdef DOUBLE_PRECISION
    //r0 = Ax0
    cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasDaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasDscal(cublasHandle, n, &oneopp, r0, 1);
#else
    //r0 = Ax0
    cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasSaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasSscal(cublasHandle, n, &oneopp, r0, 1);
#endif


    if (prec == 0){
      //w = r0
#ifdef DOUBLE_PRECISION
      cublasDcopy(cublasHandle, n, r0, 1, w, 1);
#else
      cublasScopy(cublasHandle, n, r0, 1, w, 1);
#endif  
    }
    // Solve Mw = r0
    else if (prec == 1){
      //solve_diag_system<<< (int) (n/BLOCK +1), BLOCK >>>(diagM, r0, w, n);
 #ifdef DOUBLE_PRECISION
      cublasDcopy(cublasHandle, n, r0, 1, w, 1);
      cusparseDgtsv_nopivot(cusparseHandle, n, 1, diaglM, diagM, diaglM, w, n);
#else
      cublasScopy(cublasHandle, n, r0, 1, w, 1);
      cusparseSgtsv_nopivot(cusparseHandle, n, 1, diaglM, diagM, diaglM, w, n);
#endif
    }
    else if (prec == 2){
#ifdef DOUBLE_PRECISION
      cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValLU, csrRowPtrA, csrColIndA, infoL, r0, y);
      cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValLU, csrRowPtrA, csrColIndA, infoU, y,  w);
#else
      cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValLU, csrRowPtrA, csrColIndA, infoL, r0, y);
      cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValLU, csrRowPtrA, csrColIndA, infoU, y,  w);
#endif
    }

#ifdef DOUBLE_PRECISION
    //beta = nrm2(w)
    cublasDnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasDscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasDcopy(cublasHandle, n, w, 1, V, m+1);
#else
    //beta = nrm2(w)
    cublasSnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasSscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasScopy(cublasHandle, n, w, 1, V, m+1);
#endif

    //p = beta*e1
    for (i = 1; i < m+1; i++)
      s[i] = 0.0;
    
    s[0] = beta;
    res = abs(s[0]);
    
    if (res/bnrm2 < tol){
      break;
    }

    
    i = -1;
    while (i+1 < m && l < maxit){
      ++i;
      ++l;     
      //r0 = Aw (w = Vi)
#ifdef DOUBLE_PRECISION
      cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#else
      cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#endif
      
      if (prec == 0){
	//w = r0
#ifdef DOUBLE_PRECISION
	cublasDcopy(cublasHandle, n, r0, 1, w, 1);
#else
	cublasScopy(cublasHandle, n, r0, 1, w, 1);
#endif
      }//Solve Mw = r0
      else if (prec == 1){
	//solve_diag_system<<< (int) (n/BLOCK +1), BLOCK >>>(diagM, r0, w, n);
#ifdef DOUBLE_PRECISION
      cublasDcopy(cublasHandle, n, r0, 1, w, 1);
      cusparseDgtsv_nopivot(cusparseHandle, n, 1, diaglM, diagM, diaglM, w, n);
#else
      cublasScopy(cublasHandle, n, r0, 1, w, 1);
      cusparseSgtsv_nopivot(cusparseHandle, n, 1, diaglM, diagM, diaglM, w, n);
#endif
      }
      else if (prec == 2){
#ifdef DOUBLE_PRECISION
	cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValLU, csrRowPtrA, csrColIndA, infoL, r0, y);
	cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValLU, csrRowPtrA, csrColIndA, infoU, y,  w);
#else
	cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValLU, csrRowPtrA, csrColIndA, infoL, r0, y);
	cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValLU, csrRowPtrA, csrColIndA, infoU, y,  w);
#endif
      }
      
      for (k = 0; k <= i; k++){
#ifdef DOUBLE_PRECISION
  	//<w,Vk>
  	cublasDdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasDaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#else
  	//<w,Vk>
  	cublasSdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasSaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#endif
      }
#ifdef DOUBLE_PRECISION
      //H[i+1,i] = nrm2(w)
      cublasDnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasDscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasDcopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#else
      //H[i+1,i] = nrm2(w)
      cublasSnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasSscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasScopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#endif
      PlaneRotation(H, m, cs, sn, s, i);
      res = abs(s[i+1]);
      
      if (res/bnrm2 <= tol){
  	break;
      }
    }
    // solve upper triangular system in place Hy=p
    for (j = i; j >= 0; j--){
      s[j] = s[j]*1.0/H[j + m*j];
      //p(0:i) = p(0:i) - p[i] H(0:i,i)
      for (k = j-1; k >= 0; k--){
  	s[k] = s[k] - H[j + m*k] * s[j];
      }
    }
    
    for (j = 0; j <= i; j++)
#ifdef DOUBLE_PRECISION
      cublasDaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#else
      cublasSaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#endif
  }
  cudaDeviceSynchronize();
  clock_t end = clock();
  type_fd h_exec_time = ((type_fd) (end - begin)) / CLOCKS_PER_SEC;

  cudaStat1 = cudaMemcpy(exec_time, &h_exec_time,  (size_t)(sizeof(exec_time[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_PREC\n");

  if (l >= maxit || res/bnrm2 > tol){
    printf("Solver reached iteration limit %d before converging. Final residual = %18.16f\n", l, res);
  }else{
    printf("Solver converged to %18.16f relative tolerance after %d iterations. Final residual = %18.16f \n", tol, l, res);
  }


  cudaStat1 = cudaMemcpy(it, &l,  (size_t)(sizeof(it[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_PREC\n");



  cudaStat1 = cudaFree(r0);
  cudaStat2 = cudaFree(y);
  cudaStat3 = cudaFree(w);
  cudaStat4 = cudaFree(V);
  cusparseStatus2 = cusparseDestroyMatDescr(descrA);
  if (prec == 1){
    cudaStat5 = cudaFree(diagM);
    cudaStat6 = cudaFree(diaglM);
    //cusparseStatus1 = cusparseDestroyMatDescr(descrM);
    if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS))
      printf("cusparseDestroyMatDescr failed in GMRES_PREC\n");
  }
  if (prec == 2){
    cudaStat7 = cudaFree(csrValLU);
    cusparseDestroySolveAnalysisInfo(info);
    cusparseDestroySolveAnalysisInfo(infoL);
    cusparseDestroySolveAnalysisInfo(infoU);
    cusparseStatus1 = cusparseDestroyMatDescr(descrL);
    cusparseStatus2 = cusparseDestroyMatDescr(descrU);
    if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
	(cusparseStatus2 != CUSPARSE_STATUS_SUCCESS))
      printf("cusparseDestroyMatDescr failed in GMRES_PREC\n");
  }

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ||
      (cudaStat5 != cudaSuccess) ||
      (cudaStat6 != cudaSuccess) ||
      (cudaStat7 != cudaSuccess)) {
    printf("Device free failed in GMRES_PREC\n");
  }

  return;
}

extern "C" void GMRES_LU(int m, type_fd tol,  int n,
			 int nnzA, type_fd *csrValA, int *csrRowPtrA, int *csrColIndA,
			 type_fd *csrValU, int *csrRowPtrU, int *csrColIndU,
			 type_fd *csrValL, int *csrRowPtrL, int *csrColIndL,
			 type_fd *b, type_fd *x0, type_fd *exec_time, int *it) {
  //L and U are stored in different csr matrices
  
  //GMRES(m) variables
  int l, i, j, k, nnz; //, ii;
  type_fd *w, *r0, *V, *y; //on device
  type_fd *s, *cs, *sn, *H; //on host
  //type_fd *h_r0, *h_w, *h_V; //on host
  type_fd beta, bnrm2, betainv, inv, proj, res = 1.0; //one = 1.0, oneopp = -1.0, zero = 0.0,

  
  cusparseSolveAnalysisInfo_t infoL, infoU;
  cusparseMatDescr_t descrL, descrU, descrA;
  descrA = createMatDescrA();
  descrL = createMatDescrL();
  descrU = createMatDescrU();
  
  cusparseStatus1 = cusparseCreateSolveAnalysisInfo(&infoL);
  cusparseStatus2 = cusparseCreateSolveAnalysisInfo(&infoU);
  if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS))
    printf("cusparseCreateSolveAnalysisInfo failed in GMRES_LU\n");
  

  // Analysis phase for U
  cudaStat1 = cudaMemcpy(&nnz, csrRowPtrU+(n),  (size_t)(sizeof(nnz)), cudaMemcpyDeviceToHost);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_LU\n");
#ifdef DOUBLE_PRECISION
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnz , descrU, csrValU, csrRowPtrU, csrColIndU, infoU);
#else
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnz , descrU, csrValU, csrRowPtrU, csrColIndU, infoU);
#endif

   // Analysis phase for L
  cudaStat1 = cudaMemcpy(&nnz, csrRowPtrL+(n),  (size_t)(sizeof(nnz)), cudaMemcpyDeviceToHost);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_LU\n");
#ifdef DOUBLE_PRECISION
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnz , descrL, csrValL, csrRowPtrL, csrColIndL, infoL);
#else
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnz , descrL, csrValL, csrRowPtrL, csrColIndL, infoL);
#endif
  
  s  = (type_fd *) calloc((m+1), sizeof(s[0]));
  cs = (type_fd *) calloc(m, sizeof(cs[0]));
  sn = (type_fd *) calloc(m, sizeof(sn[0]));
  H  = (type_fd *) calloc((m+1)*m, sizeof(H[0]));

  cudaStat1 = cudaMalloc((void**)&r0, n*sizeof(r0[0]));
  cudaStat2 = cudaMalloc((void**)&w,  n*sizeof(w[0]));
  cudaStat3 = cudaMalloc((void**)&V,  (n*(m+1))*sizeof(V[0])); // m+1 vector basis of the Krylov approximation space
  cudaStat4 = cudaMalloc((void**)&y,  n*sizeof(y[0]));

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("Device malloc failed in GMRES_LU\n");
    return;
  }
  
  clock_t begin = clock();
  l = 0;
#ifdef DOUBLE_PRECISION
  cublasDnrm2(cublasHandle, n, b, 1, &bnrm2);
#else
  cublasSnrm2(cublasHandle, n, b, 1, &bnrm2);
#endif
  if (bnrm2 == 0.0)
    bnrm2 = 1.0;

  while (res/bnrm2 > tol && l < maxit){
    
#ifdef DOUBLE_PRECISION
    //r0 = Ax0
    cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasDaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasDscal(cublasHandle, n, &oneopp, r0, 1);
#else
    //r0 = Ax0
    cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasSaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasSscal(cublasHandle, n, &oneopp, r0, 1);
#endif


    // Solve Mw = r0
#ifdef DOUBLE_PRECISION
      cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValL, csrRowPtrL, csrColIndL, infoL, r0, y);
      cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValU, csrRowPtrU, csrColIndU, infoU, y,  w);
#else
      cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValL, csrRowPtrL, csrColIndL, infoL, r0, y);
      cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValU, csrRowPtrU, csrColIndU, infoU, y,  w);
#endif

#ifdef DOUBLE_PRECISION
    //beta = nrm2(w)
    cublasDnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasDscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasDcopy(cublasHandle, n, w, 1, V, m+1);
#else
    //beta = nrm2(w)
    cublasSnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasSscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasScopy(cublasHandle, n, w, 1, V, m+1);
#endif

    //p = beta*e1
    for (i = 1; i < m+1; i++)
      s[i] = 0.0;
    
    s[0] = beta;
    res = abs(s[0]);
    
    if (res/bnrm2 < tol){
      break;
    }

    
    i = -1;
    while (i+1 < m && l < maxit){
      ++i;
      ++l;     
      //r0 = Aw (w = Vi)
#ifdef DOUBLE_PRECISION
      cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#else
      cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#endif
      
      //Solve Mw = r0
#ifdef DOUBLE_PRECISION
	cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValL, csrRowPtrL, csrColIndL, infoL, r0, y);
	cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValU, csrRowPtrU, csrColIndU, infoU, y,  w);
#else
	cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValL, csrRowPtrL, csrColIndL, infoL, r0, y);
	cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValU, csrRowPtrU, csrColIndU, infoU, y,  w);
#endif
      

      for (k = 0; k <= i; k++){
#ifdef DOUBLE_PRECISION
  	//<w,Vk>
  	cublasDdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasDaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#else
  	//<w,Vk>
  	cublasSdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasSaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#endif
      }
#ifdef DOUBLE_PRECISION
      //H[i+1,i] = nrm2(w)
      cublasDnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasDscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasDcopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#else
      //H[i+1,i] = nrm2(w)
      cublasSnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasSscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasScopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#endif
      PlaneRotation(H, m, cs, sn, s, i);
      res = abs(s[i+1]);
      
      if (res/bnrm2 <= tol){
  	break;
      }
    }

    // solve upper triangular system in place Hy=p
    for (j = i; j >= 0; j--){
      s[j] = s[j]*1.0/H[j + m*j];
      //p(0:i) = p(0:i) - p[i] H(0:i,i)
      for (k = j-1; k >= 0; k--){
  	s[k] = s[k] - H[j + m*k] * s[j];
      }
    }
    
    for (j = 0; j <= i; j++)
#ifdef DOUBLE_PRECISION
      cublasDaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#else
      cublasSaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#endif
  }
  cudaDeviceSynchronize();
  clock_t end = clock();
  type_fd h_exec_time = ((type_fd) (end - begin)) / CLOCKS_PER_SEC;
  //printf("h_exec_time = %f \n", h_exec_time);
  cudaStat1 = cudaMemcpy(exec_time, &h_exec_time,  (size_t)(sizeof(exec_time[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy in GMRES_LU\n");
  
  if (l >= maxit || res/bnrm2 > tol){
    printf("Solver reached iteration limit %d before converging. Final residual = %18.16f\n", l, res);
  }else{
    printf("Solver converged to %18.16f relative tolerance after %d iterations. Final residual = %18.16f \n", tol, l, res);
  }


  cudaStat1 = cudaMemcpy(it, &l,  (size_t)(sizeof(it[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy in GMRES_LU\n");

  cusparseStatus1 = cusparseDestroyMatDescr(descrL);
  cusparseStatus2 = cusparseDestroyMatDescr(descrU);
  cusparseStatus3 = cusparseDestroyMatDescr(descrA);
  if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus3 != CUSPARSE_STATUS_SUCCESS))
    printf("cusparseDestroyMatDescr failed in GMRES_LU\n");
  

  cudaStat1 = cudaFree(r0);
  cudaStat2 = cudaFree(y);
  cudaStat3 = cudaFree(w);
  cudaStat4 = cudaFree(V);
  cusparseDestroySolveAnalysisInfo(infoL);
  cusparseDestroySolveAnalysisInfo(infoU);



  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("Device free failed in GMRES_LU\n");
  }


  return;
}

extern "C" void GMRES_LU2(int m, type_fd tol,
			  int jacobi, int iters, int approx, int diag, 
			  int n, int nnzA, type_fd *csrValA, int *csrRowPtrA, int *csrColIndA,
			  type_fd *csrValM, type_fd *b, type_fd *x0, type_fd *exec_time, int *it) {
  //L and U are stored in the same csr matrix and have the same sparsity pattern from A
  
  //GMRES(m) variables
  int l, i, j, k; //, ii;
  type_fd *w, *r0, *V, *y; //on device
  type_fd *s, *cs, *sn, *H; //on host
  type_fd beta, bnrm2, betainv, inv, proj, res = 1.0; 
  
  cusparseMatDescr_t descrL, descrU, descrA;
  cusparseSolveAnalysisInfo_t infoL, infoU;
  type_fd *diagUinv, *csrValMcpy;
  
  descrA = createMatDescrA();
  descrL = createMatDescrL();
  descrU = createMatDescrU();

  
  s  = (type_fd *) calloc((m+1), sizeof(s[0]));
  cs = (type_fd *) calloc(m, sizeof(cs[0]));
  sn = (type_fd *) calloc(m, sizeof(sn[0]));
  H  = (type_fd *) calloc((m+1)*m, sizeof(H[0]));

  cudaStat1 = cudaMalloc((void**)&r0, n*sizeof(r0[0]));
  cudaStat2 = cudaMalloc((void**)&w,  n*sizeof(w[0]));
  cudaStat3 = cudaMalloc((void**)&V,  (n*(m+1))*sizeof(V[0])); // m+1 vector basis of the Krylov approximation space
  cudaStat4 = cudaMalloc((void**)&y,  n*sizeof(y[0]));

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("Device malloc failed in GMRES_LU2\n");
    return;
  }
  
  if (jacobi == 1){
    // Scalar Jacobi
    cudaStat1 = cudaMalloc((void**)&diagUinv, n*sizeof(diagUinv[0]));
    csrExtractInvDiag<<< (int) (n/BLOCK +1), BLOCK >>>(csrValM, csrRowPtrA, csrColIndA, diagUinv, n);
  }else{
    cusparseStatus1 = cusparseCreateSolveAnalysisInfo(&infoL);
    cusparseStatus2 = cusparseCreateSolveAnalysisInfo(&infoU);
    if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
	(cusparseStatus2 != CUSPARSE_STATUS_SUCCESS))
      printf("cusparseCreateSolveAnalysisInfo failed in GMRES_LU2\n");
    
    if (approx == 1){
      //elimina gli elementi fuori dalla diagonale k
      cudaMalloc((void**)&csrValMcpy, nnzA*sizeof(csrValMcpy[0]));
#ifdef DOUBLE_PRECISION
      cublasDcopy(cublasHandle, nnzA, csrValM, 1, csrValMcpy, 1);
#else
      cublasScopy(cublasHandle, nnzA, csrValM, 1, csrValMcpy, 1);
#endif
      cudaDeviceSynchronize();
      approximate_vals_LU<<< (int) (n/BLOCK +1), BLOCK >>>(csrValM, csrRowPtrA, csrColIndA, diag, n);
      cudaDeviceSynchronize();
    }
    
#ifdef DOUBLE_PRECISION
    cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzA, descrU, csrValM, csrRowPtrA, csrColIndA, infoU);
    cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzA, descrL, csrValM, csrRowPtrA, csrColIndA, infoL);
#else
    cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzA, descrU, csrValM, csrRowPtrA, csrColIndA, infoU);
    cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzA, descrL, csrValM, csrRowPtrA, csrColIndA, infoL);
#endif
  }
  

  cudaProfilerStart();
  clock_t Begin = clock();
  l = 0;
#ifdef DOUBLE_PRECISION
  cublasDnrm2(cublasHandle, n, b, 1, &bnrm2);
#else
  cublasSnrm2(cublasHandle, n, b, 1, &bnrm2);
#endif
  if (bnrm2 == 0.0)
    bnrm2 = 1.0;

  while (res/bnrm2 > tol && l < maxit){
    
#ifdef DOUBLE_PRECISION
    //r0 = Ax0
    cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasDaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasDscal(cublasHandle, n, &oneopp, r0, 1);
#else
    //r0 = Ax0
    cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasSaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasSscal(cublasHandle, n, &oneopp, r0, 1);
#endif

    // Solve Mw = r0
    if (jacobi == 1){
      //Ly = r0
      scalar_jacobi_relaxation(iters, n, 1, csrValM, csrRowPtrA, csrColIndA, diagUinv, w, y, r0);
      //Uw = y
      scalar_jacobi_relaxation(iters, n, 0, csrValM, csrRowPtrA, csrColIndA, diagUinv, r0, w, y);
    }else{
#ifdef DOUBLE_PRECISION
      cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValM, csrRowPtrA, csrColIndA, infoL, r0, y);
      cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValM, csrRowPtrA, csrColIndA, infoU, y,  w);
#else
      cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValM, csrRowPtrA, csrColIndA, infoL, r0, y);
      cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValM, csrRowPtrA, csrColIndA, infoU, y,  w);
#endif
    }


#ifdef DOUBLE_PRECISION
    //beta = nrm2(w)
    cublasDnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasDscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasDcopy(cublasHandle, n, w, 1, V, m+1);
#else
    //beta = nrm2(w)
    cublasSnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasSscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasScopy(cublasHandle, n, w, 1, V, m+1);
#endif

    //p = beta*e1
    for (i = 1; i < m+1; i++)
      s[i] = 0.0;
    
    s[0] = beta;
    res = abs(s[0]);
    
    if (res/bnrm2 < tol){
      break;
    }

    
    i = -1;
    while (i+1 < m && l < maxit){
      ++i;
      ++l;     
      //r0 = Aw (w = Vi)
#ifdef DOUBLE_PRECISION
      cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#else
      cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#endif

      //Solve Mw = r0
      if (jacobi == 1){
	//Ly = r0
	scalar_jacobi_relaxation(iters, n, 1, csrValM, csrRowPtrA, csrColIndA, diagUinv, w, y, r0);
	//Uw = y
	scalar_jacobi_relaxation(iters, n, 0, csrValM, csrRowPtrA, csrColIndA, diagUinv, r0, w, y);
      }else{
#ifdef DOUBLE_PRECISION
	cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValM, csrRowPtrA, csrColIndA, infoL, r0, y);
	cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValM, csrRowPtrA, csrColIndA, infoU, y,  w);
#else
	cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValM, csrRowPtrA, csrColIndA, infoL, r0, y);
	cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValM, csrRowPtrA, csrColIndA, infoU, y,  w);
#endif
      }
      
      for (k = 0; k <= i; k++){
#ifdef DOUBLE_PRECISION
  	//<w,Vk>
  	cublasDdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasDaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#else
  	//<w,Vk>
  	cublasSdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasSaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#endif
      }
#ifdef DOUBLE_PRECISION
      //H[i+1,i] = nrm2(w)
      cublasDnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasDscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasDcopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#else
      //H[i+1,i] = nrm2(w)
      cublasSnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasSscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasScopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#endif
      PlaneRotation(H, m, cs, sn, s, i);
      res = abs(s[i+1]);
      
      if (res/bnrm2 <= tol){
  	break;
      }

    }

    // solve upper triangular system in place Hy=p
    for (j = i; j >= 0; j--){
      s[j] = s[j]*1.0/H[j + m*j];
      //p(0:i) = p(0:i) - p[i] H(0:i,i)
      for (k = j-1; k >= 0; k--){
  	s[k] = s[k] - H[j + m*k] * s[j];
      }
    }
    
    for (j = 0; j <= i; j++)
#ifdef DOUBLE_PRECISION
      cublasDaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#else
      cublasSaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#endif
  }
  cudaDeviceSynchronize();
  clock_t End = clock();
  cudaProfilerStop();
  type_fd h_exec_time = ((type_fd) (End - Begin)) / CLOCKS_PER_SEC;
  //printf("Preconditioning time Mr = w = %f \n", el_t);
  cudaStat1 = cudaMemcpy(exec_time, &h_exec_time,  (size_t)(sizeof(exec_time[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in in GMRES_LU2\n");

  if (l >= maxit || res/bnrm2 > tol){
    printf("Solver reached iteration limit %d before converging. Final residual = %18.16f\n", l, res);
  }else{
    printf("Solver converged to %18.16f relative tolerance after %d iterations. Final residual = %18.16f \n", tol, l, res);
  }

  cudaStat1 = cudaMemcpy(it, &l,  (size_t)(sizeof(it[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_LU2\n");

  cudaStat1 = cudaFree(r0);
  cudaStat2 = cudaFree(y);
  cudaStat3 = cudaFree(w);
  cudaStat4 = cudaFree(V);

  cusparseDestroyMatDescr(descrA);
  cusparseDestroyMatDescr(descrL);
  cusparseDestroyMatDescr(descrU);


  if (jacobi == 1){
    cudaFree(diagUinv);
  }else{
    cusparseDestroySolveAnalysisInfo(infoL);
    cusparseDestroySolveAnalysisInfo(infoU);
    if (approx == 1){
#ifdef DOUBLE_PRECISION
      cublasDcopy(cublasHandle, nnzA, csrValMcpy, 1, csrValM, 1);
#else
      cublasScopy(cublasHandle, nnzA, csrValMcpy, 1, csrValM, 1);
#endif
      cudaFree(csrValMcpy);
    }
  }

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("Device free failed in GMRES_LU2\n");
  }

  return;
}

extern "C" void GMRES_LU3(int m, type_fd tol, int n,
			  type_fd *csrValA, int *csrRowPtrA, int *csrColIndA, int nnzA, 
			  type_fd *csrValM, int *csrRowPtrM, int *csrColIndM, int nnzM, 
			  type_fd *b, type_fd *x0, type_fd *exec_time, int *it){
  //L and U are stored in the same csr matrix and have a sparsity pattern different from that of A
  
  //GMRES(m) variables
  int l, i, j, k; //, ii;
  type_fd *w, *r0, *V, *y; //on device
  type_fd *s, *cs, *sn, *H; //on host
  //type_fd *h_r0, *h_w, *h_V; //on host
  type_fd beta, bnrm2, betainv, inv, proj,  res = 1.0; // alpha = 1.0, alphaopp = -1.0, zero = 0.0,

  cusparseSolveAnalysisInfo_t infoL, infoU;
  cusparseMatDescr_t descrL, descrU, descrA;
  
  descrL = createMatDescrL();
  descrU = createMatDescrU();
  descrA = createMatDescrA();
  
  cusparseStatus1 = cusparseCreateSolveAnalysisInfo(&infoL);
  cusparseStatus2 = cusparseCreateSolveAnalysisInfo(&infoU);
  if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS))
    printf("cusparseCreateSolveAnalysisInfo failed in GMRES_LU3\n");
  

  // Analysis phase for U and L
#ifdef DOUBLE_PRECISION
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzM , descrU, csrValM, csrRowPtrM, csrColIndM, infoU);
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzM , descrL, csrValM, csrRowPtrM, csrColIndM, infoL);
#else
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzM , descrU, csrValM, csrRowPtrM, csrColIndM, infoU);
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzM , descrL, csrValM, csrRowPtrM, csrColIndM, infoL);
#endif

  
  s  = (type_fd *) calloc((m+1), sizeof(s[0]));
  cs = (type_fd *) calloc(m, sizeof(cs[0]));
  sn = (type_fd *) calloc(m, sizeof(sn[0]));
  H  = (type_fd *) calloc((m+1)*m, sizeof(H[0]));

  cudaStat1 = cudaMalloc((void**)&r0, n*sizeof(r0[0]));
  cudaStat2 = cudaMalloc((void**)&w,  n*sizeof(w[0]));
  cudaStat3 = cudaMalloc((void**)&V,  (n*(m+1))*sizeof(V[0])); // m+1 vector basis of the Krylov approximation space
  cudaStat4 = cudaMalloc((void**)&y,  n*sizeof(y[0]));

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("Device malloc failed in GMRES_LU3\n");
    return;
  }

  
  clock_t begin = clock();
  l = 0;
#ifdef DOUBLE_PRECISION
  cublasDnrm2(cublasHandle, n, b, 1, &bnrm2);
#else
  cublasSnrm2(cublasHandle, n, b, 1, &bnrm2);
#endif
  if (bnrm2 == 0.0)
    bnrm2 = 1.0;
  
  while (res/bnrm2 > tol && l < maxit){
    
#ifdef DOUBLE_PRECISION
    //r0 = Ax0
    cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasDaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasDscal(cublasHandle, n, &oneopp, r0, 1);
#else
    //r0 = Ax0
    cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasSaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasSscal(cublasHandle, n, &oneopp, r0, 1);
#endif


    // Solve Mw = r0
#ifdef DOUBLE_PRECISION
    cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValM, csrRowPtrM, csrColIndM, infoL, r0, y);
    cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValM, csrRowPtrM, csrColIndM, infoU, y,  w);
#else
    cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValM, csrRowPtrM, csrColIndM, infoL, r0, y);
    cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValM, csrRowPtrM, csrColIndM, infoU, y,  w);
#endif
    

#ifdef DOUBLE_PRECISION
    //beta = nrm2(w)
    cublasDnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasDscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasDcopy(cublasHandle, n, w, 1, V, m+1);
#else
    //beta = nrm2(w)
    cublasSnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasSscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasScopy(cublasHandle, n, w, 1, V, m+1);
#endif

    //p = beta*e1
    for (i = 1; i < m+1; i++)
      s[i] = 0.0;
    
    s[0] = beta;
    res = abs(s[0]);
    
    if (res/bnrm2 < tol){
      break;
    }

    
    i = -1;
    while (i+1 < m && l < maxit){
      ++i;
      ++l;     
      //r0 = Aw (w = Vi)
#ifdef DOUBLE_PRECISION
      cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#else
      cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#endif
      
      //Solve Mw = r0
#ifdef DOUBLE_PRECISION
      cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValM, csrRowPtrM, csrColIndM, infoL, r0, y);
      cusparseDcsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValM, csrRowPtrM, csrColIndM, infoU, y,  w);
#else
      cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrL, csrValM, csrRowPtrM, csrColIndM, infoL, r0, y);
      cusparseScsrsv_solve(cusparseHandle, trans, n, &one, descrU, csrValM, csrRowPtrM, csrColIndM, infoU, y,  w);
#endif

      for (k = 0; k <= i; k++){
#ifdef DOUBLE_PRECISION
  	//<w,Vk>
  	cublasDdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasDaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#else
  	//<w,Vk>
  	cublasSdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasSaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#endif
      }
#ifdef DOUBLE_PRECISION
      //H[i+1,i] = nrm2(w)
      cublasDnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasDscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasDcopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#else
      //H[i+1,i] = nrm2(w)
      cublasSnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasSscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasScopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#endif
      PlaneRotation(H, m, cs, sn, s, i);
      res = abs(s[i+1]);
      
      if (res/bnrm2 <= tol){
  	break;
      }
    }

    // solve upper triangular system in place Hy=p
    for (j = i; j >= 0; j--){
      s[j] = s[j]*1.0/H[j + m*j];
      //p(0:i) = p(0:i) - p[i] H(0:i,i)
      for (k = j-1; k >= 0; k--){
  	s[k] = s[k] - H[j + m*k] * s[j];
      }
    }
    
    for (j = 0; j <= i; j++)
#ifdef DOUBLE_PRECISION
      cublasDaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#else
      cublasSaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#endif
  }
  cudaDeviceSynchronize();
  clock_t end = clock();
  type_fd h_exec_time = ((type_fd) (end - begin)) / CLOCKS_PER_SEC;
  //printf("h_exec_time = %f \n", h_exec_time);
  cudaStat1 = cudaMemcpy(exec_time, &h_exec_time,  (size_t)(sizeof(exec_time[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_LU3\n");
  
  if (l >= maxit || res/bnrm2 > tol){
    printf("Solver reached iteration limit %d before converging. Final residual = %18.16f\n", l, res);
  }else{
    printf("Solver converged to %18.16f relative tolerance after %d iterations. Final residual = %18.16f \n", tol, l, res);
  }


  cudaStat1 = cudaMemcpy(it, &l,  (size_t)(sizeof(it[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_LU3\n");

  cusparseStatus1 = cusparseDestroyMatDescr(descrL);
  cusparseStatus2 = cusparseDestroyMatDescr(descrU);
  cusparseStatus3 = cusparseDestroyMatDescr(descrA);
  if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus3 != CUSPARSE_STATUS_SUCCESS))
    printf("cusparseDestroyMatDescr failed in GMRES_LU3\n");
  
  cudaStat1 = cudaFree(r0);
  cudaStat2 = cudaFree(y);
  cudaStat3 = cudaFree(w);
  cudaStat4 = cudaFree(V);
  cusparseDestroySolveAnalysisInfo(infoL);
  cusparseDestroySolveAnalysisInfo(infoU);


  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("Device free failed in GMRES_LU3\n");
  }


  return;
}

__global__ void csrScalUpperTri(type_fd *csrValU, type_fd *diag, int *csrRowPtr, int *csrColInd, int n){
  //NB: gli inglici colonna devono essere orsinato in maniera crescente!!
  //U = D*U
  int id = threadIdx.x + blockIdx.x*blockDim.x;
  int j;
  
  if (id<n){
    for (j = csrRowPtr[id]; j < csrRowPtr[id+1]; j++){
      if (csrColInd[j] >= id) //
	csrValU[j] = csrValU[j]*diag[id]; // valutare se mettere la diagonale a 1.0
    }
  }
  
  return;
}

extern "C" void GMRES_LU_BLOCK(int m, type_fd tol, int n, int LU_iters, 
			       type_fd *csrValA, type_fd *csrValM_full, int *csrRowPtrA, int *csrColIndA, int nnzA, 
			       type_fd *csrValM, int *csrRowPtrM, int *csrColIndM, int nnzM, 
			       type_fd *b, type_fd *x0, type_fd *exec_time, int *it){
  // L and U are stored together in M_full (same sp pattern of A) 
  // and M contains their dropped block diagonal version

  printf("GMRES block Jacobi \n");
  
  //GMRES(m) variables
  int l, i, j, k; //, ii;
  type_fd *w, *r0, *V, *y; //on device
  type_fd *s, *cs, *sn, *H; //on host  //type_fd *h_r0, *h_w, *h_V; //on host
  type_fd beta, bnrm2, betainv, inv, proj, res = 1.0; //alpha = 1.0, alphaopp = -1.0, zero = 0.0, 
  
  //Variabili aggiuntive per il metodo di jacobi
  type_fd *xx;

  //strutture di cusparse per descivere le matrici
  cusparseMatDescr_t descrL, descrU, descrA;
  cusparseSolveAnalysisInfo_t infoL, infoU;
  
  descrA = createMatDescrA();
  descrL = createMatDescrL();
  descrU = createMatDescrU();

  cusparseStatus1 = cusparseCreateSolveAnalysisInfo(&infoL);
  cusparseStatus2 = cusparseCreateSolveAnalysisInfo(&infoU);
  if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS))
    printf("cusparseCreateSolveAnalysisInfo failed in GMRES_LU_BLOCK\n");


  //Allocate assitional memory for Block-Jacobi
  cudaStat2 = cudaMalloc((void**)&xx, n*sizeof(xx[0]));
  

  // Analysis phase for Block U
#ifdef DOUBLE_PRECISION
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzM, descrU, csrValM, csrRowPtrM, csrColIndM, infoU);
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnzM, descrL, csrValM, csrRowPtrM, csrColIndM, infoL);
#else
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzM, descrU, csrValM, csrRowPtrM, csrColIndM, infoU);
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnzM, descrL, csrValM, csrRowPtrM, csrColIndM, infoL);
#endif


  s  = (type_fd *) calloc((m+1), sizeof(s[0]));
  cs = (type_fd *) calloc(m, sizeof(cs[0]));
  sn = (type_fd *) calloc(m, sizeof(sn[0]));
  H  = (type_fd *) calloc((m+1)*m, sizeof(H[0]));

  cudaStat1 = cudaMalloc((void**)&r0, n*sizeof(r0[0]));
  cudaStat2 = cudaMalloc((void**)&w,  n*sizeof(w[0]));
  cudaStat3 = cudaMalloc((void**)&V,  (n*(m+1))*sizeof(V[0])); // m+1 vector basis of the Krylov approximation space
  cudaStat4 = cudaMalloc((void**)&y,  n*sizeof(y[0]));

  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("Device malloc failed in GMRES_LU_BLOCK\n");
    return;
  }

  
  clock_t begin = clock();
  l = 0;
#ifdef DOUBLE_PRECISION
  cublasDnrm2(cublasHandle, n, b, 1, &bnrm2);
#else
  cublasSnrm2(cublasHandle, n, b, 1, &bnrm2);
#endif
  if (bnrm2 == 0.0)
    bnrm2 = 1.0;
  
  while (res/bnrm2 > tol && l < maxit){
    
#ifdef DOUBLE_PRECISION
    //r0 = Ax0
    cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasDaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasDscal(cublasHandle, n, &oneopp, r0, 1);
#else
    //r0 = Ax0
    cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &alpha, one, csrValA, csrRowPtrA, csrColIndA, x0, &zero, r0);
    //r0 = b - r0
    cublasSaxpy(cublasHandle, n, &oneopp, b, 1, r0, 1);
    cublasSscal(cublasHandle, n, &oneopp, r0, 1);
#endif


    // Solve Mw = r0
    block_jacobi(LU_iters, n,
		   csrValM_full, csrRowPtrA, csrColIndA,
		   csrValM, csrRowPtrM, csrColIndM,
		   infoL, infoU,
		   descrL, descrU,
		   xx, w, r0, y);

#ifdef DOUBLE_PRECISION
    //beta = nrm2(w)
    cublasDnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasDscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasDcopy(cublasHandle, n, w, 1, V, m+1);
#else
    //beta = nrm2(w)
    cublasSnrm2(cublasHandle, n, w, 1, &beta);
    
    //(v1 =) w = w/beta
    betainv = (1.0)/beta;
    cublasSscal(cublasHandle, n, &betainv, w, 1);
    //save w (v1) in V
    cublasScopy(cublasHandle, n, w, 1, V, m+1);
#endif

    //p = beta*e1
    for (i = 1; i < m+1; i++)
      s[i] = 0.0;
    
    s[0] = beta;
    res = abs(s[0]);
    
    if (res/bnrm2 < tol){
      break;
    }

    
    i = -1;
    while (i+1 < m && l < maxit){
      ++i;
      ++l;     
      //r0 = Aw (w = Vi)
#ifdef DOUBLE_PRECISION
      cusparseDcsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#else
      cusparseScsrmv(cusparseHandle, trans, n, n, nnzA, &one, descrA, csrValA, csrRowPtrA, csrColIndA, w, &zero, r0);
#endif
      
      //Solve Mw = r0
      block_jacobi(LU_iters, n,
		     csrValM_full, csrRowPtrA, csrColIndA,
		     csrValM, csrRowPtrM, csrColIndM,
		     infoL, infoU,
		     descrL, descrU,
		     xx, w, r0, y);

      for (k = 0; k <= i; k++){
#ifdef DOUBLE_PRECISION
  	//<w,Vk>
  	cublasDdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasDaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#else
  	//<w,Vk>
  	cublasSdot(cublasHandle, n, w, 1, V+k, m+1, &H[i + m*k]);
  	proj = -1.0*H[i + m*k];
  	//w=w-H[i,k]*Vi
  	cublasSaxpy(cublasHandle, n, &proj, V+k, m+1, w, 1);
#endif
      }
#ifdef DOUBLE_PRECISION
      //H[i+1,i] = nrm2(w)
      cublasDnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasDscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasDcopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#else
      //H[i+1,i] = nrm2(w)
      cublasSnrm2(cublasHandle, n, w, 1, &H[i + m*(i+1)]);
      //w = w/H[i+1,i]
      inv = (1.0)/H[i + m*(i+1)];
      cublasSscal(cublasHandle, n, &inv, w, 1);      
      //save w (Vi+1) in V
      cublasScopy(cublasHandle, n, w, 1, V+(i+1), m+1);
#endif
      PlaneRotation(H, m, cs, sn, s, i);
      res = abs(s[i+1]);
      
      if (res/bnrm2 <= tol){
  	break;
      }
    }

    // solve upper triangular system in place Hy=p
    for (j = i; j >= 0; j--){
      s[j] = s[j]*1.0/H[j + m*j];
      //p(0:i) = p(0:i) - p[i] H(0:i,i)
      for (k = j-1; k >= 0; k--){
  	s[k] = s[k] - H[j + m*k] * s[j];
      }
    }
    
    for (j = 0; j <= i; j++)
#ifdef DOUBLE_PRECISION
      cublasDaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#else
      cublasSaxpy(cublasHandle, n, &s[j], V+j, m+1, x0, 1);
#endif
  }
  cudaDeviceSynchronize();
  clock_t end = clock();
  type_fd h_exec_time = ((type_fd) (end - begin)) / CLOCKS_PER_SEC;

  cudaStat1 = cudaMemcpy(exec_time, &h_exec_time,  (size_t)(sizeof(exec_time[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_LU_BLOCK\n");
  
  if (l >= maxit || res/bnrm2 > tol){
    printf("Solver reached iteration limit %d before converging. Final residual = %18.16f\n", l, res);
  }else{
    printf("Solver converged to %18.16f relative tolerance after %d iterations. Final residual = %18.16f \n", tol, l, res);
  }


  cudaStat1 = cudaMemcpy(it, &l,  (size_t)(sizeof(it[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy failed in GMRES_LU_BLOCK\n");

  cusparseStatus1 = cusparseDestroyMatDescr(descrL);
  cusparseStatus2 = cusparseDestroyMatDescr(descrU);
  cusparseStatus3 = cusparseDestroyMatDescr(descrA);
  if ((cusparseStatus1 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus2 != CUSPARSE_STATUS_SUCCESS) ||
      (cusparseStatus3 != CUSPARSE_STATUS_SUCCESS))
    printf("cusparseDestroyMatDescr failed in GMRES_LU_BLOCK\n");
  
  
  cudaStat1 = cudaFree(r0);
  cudaStat2 = cudaFree(y);
  cudaStat3 = cudaFree(w);
  cudaStat4 = cudaFree(V);
  cudaFree(xx);
  cusparseDestroySolveAnalysisInfo(infoL);
  cusparseDestroySolveAnalysisInfo(infoU);


  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("GMRES_LU_BLOCK: Device free failed\n");
  }


  return;
}




void print_analysis(cusparseSolveAnalysisInfo_t info, int nrows){
  int i, j, *nlevels, *levelPtr_d, *levelInd_d, *levelPtr_h, *levelInd_h;

  // can be called after csrsv_analysis call

  nlevels    = (int *)malloc(1*sizeof(int));
  levelPtr_h = (int *)malloc(nrows*sizeof(int));
  levelInd_h = (int *)malloc(nrows*sizeof(int));
  
  cudaMalloc((void**)&levelPtr_d, sizeof(int)*nrows);
  cudaMalloc((void**)&levelInd_d, sizeof(int)*nrows);

  cusparseGetLevelInfo(cusparseHandle, info, nlevels, &levelPtr_d, &levelInd_d);
  cudaDeviceSynchronize();

  cudaMemcpy(levelPtr_h, levelPtr_d, sizeof(int)*nlevels[0]+1, cudaMemcpyDeviceToHost);
  cudaMemcpy(levelInd_h, levelInd_d, sizeof(int)*nrows, cudaMemcpyDeviceToHost);

  printf("nlevels = %d\n", nlevels[0]);

  levelPtr_h[nlevels[0]] = nrows;
  
  printf("levelPtr = ");
  for(i=0; i<nlevels[0]+1; i++)
    printf("\t %d ", levelPtr_h[i]);
  printf("\n");

  printf("levelInd = \n");
  for(i=0; i<nlevels[0]; i++) {
    for (j=levelPtr_h[i]; j<levelPtr_h[i+1]; j++){
      printf("\t %d ", levelInd_h[j]);
    }
    printf("\n");
  }
  return;  
}

extern "C" void print_DAG_analysis(int n, int nnz, type_fd *csrVal, int *csrRowPtr, int *csrColInd, int lower_tri){
  cusparseSolveAnalysisInfo_t info;
  cusparseMatDescr_t descr;
  int i, j, *nlevels, *levelPtr_d, *levelInd_d, *levelPtr_h, *levelInd_h;

  if (lower_tri == 1)
    descr = createMatDescrL();
  else if (lower_tri == 0)
    descr = createMatDescrU();
  else
    descr = createMatDescrA();

  cusparseCreateSolveAnalysisInfo(&info);
#ifdef DOUBLE_PRECISION
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnz , descr, csrVal, csrRowPtr, csrColInd, info);
#else
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnz , descr, csrVal, csrRowPtr, csrColInd, info);
#endif
  
  nlevels    = (int *)malloc(1*sizeof(int));
  levelPtr_h = (int *)malloc(n*sizeof(int));
  levelInd_h = (int *)malloc(n*sizeof(int));
  
  cudaMalloc((void**)&levelPtr_d, sizeof(int)*n);
  cudaMalloc((void**)&levelInd_d, sizeof(int)*n);
  
  cusparseGetLevelInfo(cusparseHandle, info, nlevels, &levelPtr_d, &levelInd_d);
  cudaDeviceSynchronize();
  
  cudaMemcpy(levelPtr_h, levelPtr_d, sizeof(int)*nlevels[0]+1, cudaMemcpyDeviceToHost);
  cudaMemcpy(levelInd_h, levelInd_d, sizeof(int)*n, cudaMemcpyDeviceToHost);
  
  printf("nlevels = %d\n", nlevels[0]);
  
  levelPtr_h[nlevels[0]] = n;
  
  printf("levelPtr = ");
  for(i=0; i<nlevels[0]+1; i++)
    printf("\t %d ", levelPtr_h[i]);
  printf("\n");
  
  printf("levelInd = \n");
  for(i=0; i<nlevels[0]; i++) {
    for (j=levelPtr_h[i]; j<levelPtr_h[i+1]; j++)
      printf("\t %d ", levelInd_h[j]);
    printf("\n");
  }

  free(levelPtr_h);
  free(levelInd_h);
  cudaFree(levelPtr_d);
  cudaFree(levelInd_d);
  cusparseDestroyMatDescr(descr);
  cusparseDestroySolveAnalysisInfo(info);
  
  return;
}


extern "C" void DAG_analysis(int n, int nnz, int lower_tri, type_fd *csrVal, int *csrRowPtr, int *csrColInd, int *levelPtr, int *levelInd, int *nlevels){
  cusparseSolveAnalysisInfo_t info;
  cusparseMatDescr_t descr;
  int *nlevels_h, *levelPtr_d, *levelInd_d;

  if (lower_tri == 1)
    descr = createMatDescrL();
  else if (lower_tri == 0)
    descr = createMatDescrU();
  else
    descr = createMatDescrA();

  cusparseCreateSolveAnalysisInfo(&info);
#ifdef DOUBLE_PRECISION
  cusparseDcsrsv_analysis(cusparseHandle, trans, n, nnz , descr, csrVal, csrRowPtr, csrColInd, info);
#else
  cusparseScsrsv_analysis(cusparseHandle, trans, n, nnz , descr, csrVal, csrRowPtr, csrColInd, info);
#endif
  
  nlevels_h  = (int *)malloc(1*sizeof(int));

  cusparseGetLevelInfo(cusparseHandle, info, nlevels_h, &levelPtr_d, &levelInd_d);
  cudaDeviceSynchronize();

  cudaMemcpy(nlevels, nlevels_h, sizeof(nlevels[0]), cudaMemcpyHostToDevice);
  cudaMemcpy(levelPtr, levelPtr_d, sizeof(levelPtr[0])*(nlevels_h[0]+1), cudaMemcpyDeviceToDevice);  
  cudaMemcpy(levelInd, levelInd_d, sizeof(levelInd[0])*n, cudaMemcpyDeviceToDevice); 
  

  free(nlevels_h);
  cusparseDestroyMatDescr(descr);
  cusparseDestroySolveAnalysisInfo(info);
  
  return;
}


__global__  void spApLU_csr_kernel(int n, int nnz, type_fd *csrValM, type_fd *csrValA, type_fd *csrValAout, int *csrRowPtrA, int *csrColIndA){
  /* Kernel that computes the entries (A - L*U)_i such that A_i is non zero. 
  L,U have the same sparsity pattern of A
  NB: the maximum nb of non zero elements of each row of A is 32 */
  
  int row, row_loc, row_start, row_end, rowU_start, it_row_U, rowU_end;
  int col, colU, index, ind, ind_loc, c_rowU, end, k, i;
  type_fd valL, valU, c_valL;

  int warpId = threadIdx.x / WARP_SIZE; 
  int laneId = threadIdx.x % WARP_SIZE; 
  
  volatile __shared__ type_fd   ValA[NNZ*BLOCK/WARP_SIZE];
  __shared__ type_fd   ValM[NNZ*BLOCK/WARP_SIZE];
  __shared__ int  ColInd[NNZ*BLOCK/WARP_SIZE];
  volatile __shared__ int  s_rowU[BLOCK/WARP_SIZE]; //s_ = shared
  volatile __shared__ type_fd s_valL[BLOCK/WARP_SIZE];

  row = blockIdx.x*(BLOCK/WARP_SIZE) + warpId; // warp warpId processes row
  row_loc = warpId;

  if (row < n){
    
    row_start = csrRowPtrA[row];
    row_end   = csrRowPtrA[row + 1];

    ind = row_start + laneId;
    ind_loc = row_loc*NNZ + laneId;
    
    if (ind < row_end){  
      ValM[ind_loc]   = csrValM[ind];
      ValA[ind_loc]   = csrValA[ind];
      ColInd[ind_loc] = csrColIndA[ind];

      col  = ColInd[ind_loc];
      valL = ValM[ind_loc];
      
      valU = valL*(col >= row);
      valL = valL*(col < row); 
      
    }else{     
      ValM[ind_loc] = 0.0;
      ValA[ind_loc] = 0.0;
      ColInd[ind_loc] = -1;   
      col = -1;
      valL = valU = 0.0;
    }
    
    //processing the row with index rowU = row
    if (col>=row) ValA[ind_loc] -= ValM[ind_loc]; 

    //processing the other rows of U with index corresponding to the non diagonal non zero elements of columns of L 
    for (k = 0, end = __popc(__ballot_sync( __activemask(), ((col<row)&&(col>=0)) )); k < end; k++){
      if (laneId == k){ // k-th thread k sets the row of U to read the value of L that scales it
	s_rowU[warpId] = col;
	s_valL[warpId] = valL;
      }
      c_rowU = s_rowU[warpId];
      c_valL = s_valL[warpId];


      if ((c_rowU < 0) || (c_valL == 0.0))
	continue; // continue if values are invalid
      else {
	rowU_start = csrRowPtrA[c_rowU];
	rowU_end   = csrRowPtrA[c_rowU + 1];
	
	for (it_row_U = rowU_start + laneId; __any_sync( __activemask(), (it_row_U < rowU_end) ); it_row_U += WARP_SIZE){
	  if (it_row_U < rowU_end){
	    colU = csrColIndA[it_row_U];
	    valU = csrValM[it_row_U];
	  }else{
	    colU = -1;
	    valU = 0.0;
	  }
	  
	  colU = (colU >= c_rowU) ? colU : -1;
	  valU = valU*(colU >= c_rowU);
	  index = -1;

	  for (i = 0; i < NNZ; i++){

	    
	    if ( ColInd[row_loc*NNZ + (laneId+i)%NNZ] == colU ) index = row_loc*NNZ + (laneId+i)%NNZ;
	  }

	  if (index >= 0) ValA[index] -= c_valL*valU;
	}
      }
    }

    
    //copy back in global memory
    if (ind < row_end){
      csrValAout[ind] = ValA[ind_loc];
    }

  }
  return;
}

extern "C" void spApLU_csr(int n, int nnz, type_fd *csrValM, type_fd *csrValA, type_fd *csrValAout, int *csrRowPtr, int *csrColInd){//,  type_fd *prec_time){

  spApLU_csr_kernel<<< (int) (32*n/BLOCK +1), BLOCK >>>(n, nnz, csrValM, csrValA, csrValAout, csrRowPtr, csrColInd);

  return;
}

extern "C" void S_ITALU(int iters, int n, int nnz, type_fd *csrValM, type_fd *csrValA, int *csrRowPtrA, int *csrColIndA, type_fd *prec_time){

  /*Simplified ITALU algorithm
  M contains the L,U factors of an LU type preconditioner for matrix A to be updated.
  Here L,U are stored together (the diagonal of L is made of ones and it is omitted).
  Input:
  iters = nb of iterations of ITALU
  n = size of A,M matrices (n = nb of rows = nb of columns)
  nnz = non zeros elements of A and M
  csrValA, csrRowPtrA, csrColIndA = CSR storage of matrix A (values, row pointer, column indices)
  csrValM, csrRowPtrA, csrColIndA = CSR storage of matrix M (values, row pointer, column indices)
  prec_time = execution time of ITALU procedure
  */

  type_fd *csrValR, *diaglong, *values; // *diagUinv, *h_diagUinv, *h_csrValR,*h_diaglong,

  int k;
  
  cudaStat2 = cudaMalloc((void**)&csrValR,  (nnz)*sizeof(csrValR[0]));
  cudaStat3 = cudaMalloc((void**)&diaglong, (nnz)*sizeof(diaglong[0]));
  cudaStat4 = cudaMalloc((void**)&values,   (nnz)*sizeof(values[0]));


  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("device malloc failed\n");
    return;
  }

  thrust::device_ptr<type_fd> diaglong_ptr(diaglong);
  thrust::device_ptr<type_fd> values_ptr(values);
  thrust::device_ptr<int>  csrColIndA_ptr(csrColIndA);

  cudaProfilerStart();
  clock_t begin = clock();
  for (k=0; k<iters; k++){
    // R = A - L*U
    spApLU_csr_kernel<<< (int) (32*n/BLOCK +1), BLOCK >>>(n, nnz, csrValM, csrValA, csrValR, csrRowPtrA, csrColIndA);
    
    // Extract diag(U)^-1
    csrExtractInvDiag<<< (int) (n/BLOCK +1), BLOCK >>>(csrValM, csrRowPtrA, csrColIndA, diaglong, n);
    
    // tril(R) = tril(R*diag(U)^-1)
    thrust::gather(csrColIndA_ptr, csrColIndA_ptr + nnz, diaglong_ptr, values_ptr);
    // values contiene i valori di diag(U)^-1 che andranno a scalare la parte triangolare inferiore della matrice R
    csrScalLowerTri<<< (int) (n/BLOCK +1), BLOCK >>>(csrValR, values, csrRowPtrA, csrColIndA, n);
    
    // LU = LU + R
    axpby(nnz, one, csrValR, one, csrValM);

  }
  cudaDeviceSynchronize();
  clock_t end = clock();
  type_fd h_prec_time = ((type_fd) (end - begin)) / CLOCKS_PER_SEC;

  cudaStat1 = cudaMemcpy(prec_time, &h_prec_time,  (size_t)(sizeof(prec_time[0])), cudaMemcpyHostToDevice);
  if (cudaStat1 != cudaSuccess)
    printf("cudaMemcpy ERROR\n");

  cudaStat1 = cudaFree(values);
  cudaStat2 = cudaFree(diaglong);
  cudaStat3 = cudaFree(csrValR);
  if ((cudaStat1 != cudaSuccess) ||
      (cudaStat2 != cudaSuccess) ||
      (cudaStat3 != cudaSuccess) ||
      (cudaStat4 != cudaSuccess) ) {
    printf("Device free failed\n");
    return;
  }
  
  cudaProfilerStop();
  
  return;
}



__global__ void diag_pre_mult(type_fd *diag, type_fd *mat, int n, int m){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int idy = threadIdx.y + blockIdx.y*blockDim.y;

  if ((idx < m) && (idy < n))
    mat[IDX2C(idx,idy,n)] = mat[IDX2C(idx,idy,n)]*diag[idy];
  
  return;
} 
__global__ void diag_post_mult(type_fd *diag, type_fd *mat, int n, int m){
  int idx = threadIdx.x + blockIdx.x*blockDim.x;
  int idy = threadIdx.y + blockIdx.y*blockDim.y;

  if ((idx < m) && (idy < n))
    mat[IDX2C(idx,idy,n)] = mat[IDX2C(idx,idy,n)]*diag[idx];
  
  return;
}

__global__ void italu_UpX(int n, int nnz, type_fd *csrVal,int *csrRowPtr, int *csrColInd, type_fd *mat){
 int id = threadIdx.x + blockIdx.x*blockDim.x;
 int i, j, end;
 
 if (id<n){
   j = csrRowPtr[id];
   end = csrRowPtr[id+1];
   while (id > csrColInd[j]) j++;
   for (i = j; i<end; i++)
     csrVal[i] = csrVal[i] + mat[IDX2C(id, csrColInd[i], n)];
   
 }
 
 return;
}


__global__ void italu_LpX(int n, int nnz, type_fd *csrVal,int *csrRowPtr, int *csrColInd, type_fd *mat){
 int id = threadIdx.x + blockIdx.x*blockDim.x;
 int i, j, start;
 
 if (id<n){
   j = csrRowPtr[id];
   start = j;
   while (id > csrColInd[j]) j++;
   for (i = start; i<j; i++)
     csrVal[i] = csrVal[i] + mat[IDX2C(csrColInd[i], id, n)];
   
 }
 
 return;
}


void jacobi_upper_tri(int iters, int n, int nnz, type_fd *diag, type_fd *csrValM, type_fd *csrValR, int *csrRowPtrA, int *csrColIndA, type_fd *mat){
  int k;
  type_fd *tmp, *rhs; 
  cusparseMatDescr_t descrU, descrA;

  dim3 blockSize(TILE_DIM,TILE_DIM);
  int bx = (n + blockSize.x -1)/blockSize.x;
  int by = (n + blockSize.y -1)/blockSize.y;
  dim3 gridSize = dim3 (bx, by);
  
  descrU = createMatDescrU();
  descrA = createMatDescrA();

  cudaStat1 = cudaMalloc((void**)&tmp, (n*n)*sizeof(tmp[0]));
  cudaStat2 = cudaMalloc((void**)&rhs, (n*n)*sizeof(rhs[0]));

  
#ifdef DOUBLE_PRECISION
  cusparseDcsc2dense(cusparseHandle, n,n, descrA, csrValR, csrColIndA, csrRowPtrA, rhs, n); // includes transposition
  cublasDcopy(cublasHandle, n*n, rhs, 1, tmp, 1);
  cublasDcopy(cublasHandle, n*n, rhs, 1, mat, 1);
#else
  cusparseScsc2dense(cusparseHandle, n,n, descrA, csrValR, csrColIndA, csrRowPtrA, rhs, n);
  cublasScopy(cublasHandle, n*n, mat, 1, tmp, 1);
  cublasScopy(cublasHandle, n*n, rhs, 1, mat, 1);
#endif
  
  for (k=0; k<iters;k++){
  
#ifdef DOUBLE_PRECISION
    cusparseDcsrmm(cusparseHandle, trans_t, n, n, n, nnz, &oneopp, descrU, csrValM, csrRowPtrA, csrColIndA, mat, n, &zero, tmp, n);
    diag_pre_mult<<< gridSize, blockSize >>>(diag, tmp, n, n);
    cublasDaxpy(cublasHandle, (n*n), &one, mat, 1, tmp, 1);
    cublasDaxpy(cublasHandle, (n*n), &one, rhs, 1, tmp, 1);
    cublasDcopy(cublasHandle, (n*n), tmp, 1, mat, 1);
#else
    cusparseScsrmm(cusparseHandle, trans_t, n, n, n, nnz, &oneopp, descrU, csrValM, csrRowPtrA, csrColIndA, mat, n, &zero, tmp, n);
    diag_pre_mult<<< gridSize, blockSize >>>(diag, tmp, n, n);
    cublasSaxpy(cublasHandle, (n*n), &one, mat, 1, tmp, 1);
    cublasSaxpy(cublasHandle, (n*n), &one, rhs, 1, tmp, 1);
    cublasScopy(cublasHandle, (n*n), tmp, 1, mat, 1);
#endif
  }

  cudaFree(rhs);
  cudaFree(tmp);
  cusparseStatus1 = cusparseDestroyMatDescr(descrU);
  cusparseStatus1 = cusparseDestroyMatDescr(descrA);
  
  return;
}

void jacobi_lower_tri(int iters, int n, int nnz, type_fd *csrValM, type_fd *csrValR, int *csrRowPtrA, int *csrColIndA, type_fd *mat){
  int k;
  type_fd *tmp, *rhs; 
  cusparseMatDescr_t descrL, descrA;
  
  descrL = createMatDescrL();
  descrA = createMatDescrA();

  
  cudaStat1 = cudaMalloc((void**)&tmp, (n*n)*sizeof(tmp[0]));
  cudaStat2 = cudaMalloc((void**)&rhs, (n*n)*sizeof(rhs[0]));

  
#ifdef DOUBLE_PRECISION
  cusparseDcsr2dense(cusparseHandle, n,n, descrA, csrValR, csrRowPtrA, csrColIndA, rhs, n);
  cublasDcopy(cublasHandle, n*n, rhs, 1, tmp, 1);
  cublasDcopy(cublasHandle, n*n, rhs, 1, mat, 1);
#else
  cusparseScsr2dense(cusparseHandle, n,n, descrA, csrValR, csrRowPtrA, csrColIndA, rhs, n);
  cublasScopy(cublasHandle, n*n, rhs, 1, tmp, 1);
  cublasScopy(cublasHandle, n*n, rhs, 1, mat, 1);
#endif
  
  for (k=0; k<iters;k++){
  
#ifdef DOUBLE_PRECISION
    cusparseDcsrmm(cusparseHandle, trans, n, n, n, nnz, &oneopp, descrL, csrValM, csrRowPtrA, csrColIndA, mat, n, &one, tmp, n);
    cublasDaxpy(cublasHandle, n*n, &one, rhs, 1, tmp, 1);
    cublasDcopy(cublasHandle, n*n, tmp, 1, mat, 1);
#else
    cusparseScsrmm(cusparseHandle, trans, n, n, n, nnz, &oneopp, descrL, csrValM, csrRowPtrA, csrColIndA, mat, n, &one, tmp, n);
    cublasSaxpy(cublasHandle, n*n, &one, rhs, 1, tmp, 1);
    cublasScopy(cublasHandle, n*n, tmp, 1, mat, 1);
#endif
  }

  cudaFree(rhs);
  cudaFree(tmp);
  cusparseStatus1 = cusparseDestroyMatDescr(descrL);
  cusparseStatus2 = cusparseDestroyMatDescr(descrA);
  
  return;
}
