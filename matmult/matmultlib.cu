#include <stdio.h>
#include <stdlib.h>
//#include <sunperf.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <sys/time.h>
#include <cublas.h>


#define min(a,b)(((a)<(b))?(a):(b))
#define SIZEXBLOCK 32 
#define SIZEYBLOCK 32

#ifndef MATMULT_LIB_H
#define MATMULT_LIB_H
extern "C" {
#include <cblas.h>

__global__ void gpu5(int m, int n, int k, double *a, double *b, double *c){
	double sum=0;
	int globalThreadIdx=blockIdx.x*blockDim.x+threadIdx.x;
	int globalThreadIdy=blockIdx.y*blockDim.y+threadIdx.y;
	int i,j;
	__shared__ double A_s[SIZEXBLOCK][SIZEXBLOCK];
	__shared__ double B_s[SIZEXBLOCK][SIZEXBLOCK];
	if (globalThreadIdx < m && globalThreadIdy < n) {
		for(j=0;j<k; j+=SIZEXBLOCK){
			A_s[threadIdx.y][threadIdx.x]=a[globalThreadIdy+(j+threadIdx.x)*k];
			B_s[threadIdx.x][threadIdx.y]=b[globalThreadIdx*n+j+threadIdx.y];
			__syncthreads();
			for (i =0; i< SIZEXBLOCK; i++){
				if (j+i<k){
				//	if(m%SIZEXBLOCK==0 || n/SIZEXBLOCK == 0 || j<k-SIZEXBLOCK){
						sum+=A_s[threadIdx.y][i]*B_s[threadIdx.x][i];
//					} else if( j<k-SIZEXBLOCK ){
//						sum+=A_s[threadIdx.y][i]*b[globalThreadIdx*n+i+j];
				//	}else if ((threadIdx.y < n%SIZEXBLOCK && j>k-SIZEXBLOCK)){
				//		sum+=A_s[threadIdx.y][i]*b[globalThreadIdx*n+i+j];
					//	printf("I actually do something \n");
		//				sum+=a[globalThreadIdx*k+j+i]*b[globalThreadIdy+(i+j)*n];
					//}//else {printf("oh, hai, I'm skipping around! \n");}
				}
			}
			__syncthreads();
		}
		c[globalThreadIdy+globalThreadIdx*n]=sum;	
	}
}


void matmult_gpu5(int m, int n, int k, double **A, double **B, double **C){
	int sizeXBlock = SIZEXBLOCK;
	int sizeXGrid = (m+sizeXBlock-1)/sizeXBlock;
	int sizeYBlock = SIZEYBLOCK;
	int sizeYGrid =  (n+sizeYBlock-1)/sizeYBlock;
	double *a_d;
	double *b_d;
	double *c_d;

	dim3 DimGrid(sizeXGrid,sizeYGrid);
	dim3 DimBlock(sizeXBlock, sizeYBlock);

	checkCudaErrors(cudaMalloc((void**)&a_d,m*k*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&b_d,k*n*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&c_d,m*n*sizeof(double)));
	checkCudaErrors(cudaMemcpy(a_d,A[0], m*k*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_d,B[0], k*n*sizeof(double),cudaMemcpyHostToDevice));

	gpu5<<< DimGrid, DimBlock>>>(m,n,k,a_d,b_d,c_d);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(C[0],c_d, m*n*sizeof(double),cudaMemcpyDeviceToHost));

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	checkCudaErrors(cudaGetLastError());
}

__global__ void gpu4(int m, int n, int k, double *a, double *b, double *c){
	double sum=0;
	int globalThreadIdx=blockIdx.x*blockDim.x+threadIdx.x;
	int globalThreadIdy=blockIdx.y*blockDim.y+threadIdx.y;
	int i,j;
	__shared__ double A_s[SIZEXBLOCK][SIZEXBLOCK];
	if (globalThreadIdx < m && globalThreadIdy < n) {
		for(j=0;j<k; j+=SIZEXBLOCK){
			A_s[threadIdx.y][threadIdx.x]=a[globalThreadIdy+(j+threadIdx.x)*k];
			__syncthreads();
			for (i =0; i< SIZEXBLOCK; i++){
				if (j+i<k){
					sum+=A_s[threadIdx.y][i]*b[globalThreadIdx*n+i+j];
				}
			}
			__syncthreads();
		}
		c[globalThreadIdy+globalThreadIdx*n]=sum;	
	}
}


void matmult_gpu4(int m, int n, int k, double **A, double **B, double **C){
	int sizeXBlock = SIZEXBLOCK;
	int sizeXGrid = (m+sizeXBlock-1)/sizeXBlock;
	int sizeYBlock = SIZEYBLOCK;
	int sizeYGrid =  (n+sizeYBlock-1)/sizeYBlock;
	double *a_d;
	double *b_d;
	double *c_d;

	dim3 DimGrid(sizeXGrid,sizeYGrid);
	dim3 DimBlock(sizeXBlock, sizeYBlock);

	checkCudaErrors(cudaMalloc((void**)&a_d,m*k*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&b_d,k*n*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&c_d,m*n*sizeof(double)));
	checkCudaErrors(cudaMemcpy(a_d,A[0], m*k*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_d,B[0], k*n*sizeof(double),cudaMemcpyHostToDevice));

	gpu4<<< DimGrid, DimBlock>>>(m,n,k,a_d,b_d,c_d);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(C[0],c_d, m*n*sizeof(double),cudaMemcpyDeviceToHost));

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
	checkCudaErrors(cudaGetLastError());
}

__global__ void gpu3(int m, int n, int k, double *a, double *b, double *c)
{
	int j,nlim=n/4;
	int Idx=blockIdx.x*blockDim.x+threadIdx.x;
	int Idy=blockIdx.y*blockDim.y+threadIdx.y;
	double sum0=0, sum1=0, sum2=0, sum3=0;
        if (Idx < n && Idy < nlim) {
		for(j=0;j<k;j++){
			sum0+=a[Idy*k+j]*b[Idx+j*n];
			sum1+=a[Idy*k+j]*b[Idx+nlim+j*n];
			sum2+=a[Idy*k+j]*b[Idx+2*nlim+j*n];
			sum3+=a[Idy*k+j]*b[Idx+3*nlim+j*n];
		}
		c[(Idx*n)+(Idy)]=sum0;
		c[(Idx*n)+(Idy+nlim)]=sum1;
		c[(Idx*n)+(Idy+2*nlim)]=sum2;
		c[(Idx*n)+(Idy+3*nlim)]=sum3;
	}
}


void matmult_gpu3(int m, int n, int k, double **A, double **B, double **C)
{
	int sizeXBlock = SIZEXBLOCK/4;
	int sizeXGrid = (m+sizeXBlock-1)/sizeXBlock;
	int sizeYBlock = SIZEYBLOCK;
	int sizeYGrid =  (n+sizeYBlock-1)/sizeYBlock;
	double *a_d;
	double *b_d;
	double *c_d;

	dim3 DimGrid(sizeXGrid,sizeYGrid);
	dim3 DimBlock(sizeXBlock, sizeYBlock);
	checkCudaErrors(cudaMalloc((void**)&a_d,m*k*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&b_d,k*n*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&c_d,m*n*sizeof(double)));
	checkCudaErrors(cudaMemcpy(a_d,A[0], m*k*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_d,B[0], k*n*sizeof(double),cudaMemcpyHostToDevice));	
	checkCudaErrors(cudaGetLastError());
	gpu3<<< DimGrid, DimBlock >>>(m,n,k,a_d,b_d,c_d);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(C[0],c_d, m*n*sizeof(double),cudaMemcpyDeviceToHost));

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
}

__global__ void gpu2(int m, int n, int k, double *a, double *b, double *c)
{
	int j;
	double sum1=0;
	double sum2=0;
	int globalThreadIdx=blockIdx.x*blockDim.x+threadIdx.x;
	int globalThreadIdy=blockIdx.y*blockDim.y+threadIdx.y;	
	if (globalThreadIdx < m && globalThreadIdy < n/2) {
		for(j=0;j<k;j++){
			sum1+=a[globalThreadIdy*k+j]*b[globalThreadIdx+j*n];
			sum2+=a[globalThreadIdy*k+j]*b[(globalThreadIdx+n/2)+j*n];
		}
		c[globalThreadIdy+globalThreadIdx*n]=sum1;
		c[globalThreadIdy+n/2+globalThreadIdx*n]=sum2;		
	}
}



void matmult_gpu2(int m, int n, int k, double **A, double **B, double **C)
{
int sizeXBlock = 32;
	int sizeXGrid = (m+sizeXBlock-1)/sizeXBlock;
	int sizeYBlock = 32;
	int sizeYGrid =  (n+sizeYBlock-1)/sizeYBlock;
	double *a_d;
	double *b_d;
	double *c_d;

	dim3 DimGrid(sizeXGrid,sizeYGrid);
	dim3 DimBlock(sizeXBlock, sizeYBlock);

	checkCudaErrors(cudaMalloc((void**)&a_d,m*k*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&b_d,k*n*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&c_d,m*n*sizeof(double)));
	checkCudaErrors(cudaMemcpy(a_d,A[0], m*k*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_d,B[0], k*n*sizeof(double),cudaMemcpyHostToDevice));

	gpu2<<< DimGrid, DimBlock >>>(m,n,k,a_d,b_d,c_d);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(C[0],c_d, m*n*sizeof(double),cudaMemcpyDeviceToHost));

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
}





__global__ void gpu1(int m, int n, int k, double *a, double *b, double *c)
{
	int j;
	double sum=0;
	int globalThreadIdx=blockIdx.x*blockDim.x+threadIdx.x;
	int globalThreadIdy=blockIdx.y*blockDim.y+threadIdx.y;	
	if (globalThreadIdx < m && globalThreadIdy < n) {
		for(j=0;j<k;j++){
			sum+=a[globalThreadIdy*k+j]*b[globalThreadIdx+j*n];
		}
		c[globalThreadIdx+globalThreadIdy*n]=sum;	
	}
}



void matmult_gpu1(int m, int n, int k, double **A, double **B, double **C)
{
	int sizeXBlock = 32;
	int sizeXGrid = (m+sizeXBlock-1)/sizeXBlock;
	int sizeYBlock = 32;
	int sizeYGrid =  (n+sizeYBlock-1)/sizeYBlock;

	double *a_d;
	double *b_d;
	double *c_d;

	dim3 DimGrid(sizeXGrid,sizeYGrid);
	dim3 DimBlock(sizeXBlock, sizeYBlock);

	checkCudaErrors(cudaMalloc((void**)&a_d,m*k*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&b_d,k*n*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&c_d,m*n*sizeof(double)));
	checkCudaErrors(cudaMemcpy(a_d,A[0], m*k*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(b_d,B[0], k*n*sizeof(double),cudaMemcpyHostToDevice));

	gpu1<<< DimGrid, DimBlock >>>(m,n,k,a_d,b_d,c_d);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(C[0],c_d, m*n*sizeof(double),cudaMemcpyDeviceToHost));

	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);
}

void matmult_lib(int m, int n, int k, double **A, double **B, double **C){
	double alpha, beta;
	alpha = 1.0; beta = 0.0;
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans, n, m, k, alpha, B[0], n, A[0], k, beta, C[0], n);		
}

void matmult_gpulib(int m, int n, int k, double **A, double **B, double **C){
	double alpha, beta,*A_d, *B_d, *C_d;
	alpha = 1.0; beta = 0.0;
	checkCudaErrors(cudaMalloc((void**)&A_d, (m*k*sizeof(double*))));
	checkCudaErrors(cudaMalloc((void**)&B_d, (k*n*sizeof(double*))));
	checkCudaErrors(cudaMalloc((void**)&C_d, (m*n*sizeof(double*)))); 
	checkCudaErrors(cudaMemcpy(A_d, A[0], (m*k*sizeof(double)), cudaMemcpyHostToDevice));	
	checkCudaErrors(cudaMemcpy(B_d, B[0], (k*n*sizeof(double)), cudaMemcpyHostToDevice));	
	
	cublasDgemm('n', 'n', n, m, k, alpha, B_d, n, A_d, k, beta, C_d, n);
		
	checkCudaErrors(cudaMemcpy(C[0],C_d, (m*n*sizeof(double)), cudaMemcpyDeviceToHost));	
	checkCudaErrors(cudaFree(A_d));
	checkCudaErrors(cudaFree(B_d));
	checkCudaErrors(cudaFree(C_d));
}

void matmult_nat(int m, int n, int k, double **A, double **B, double **C){
	int i,j,t;
	for ( i = 0; i < m ; i++){
		for ( j = 0; j < n; j++){
			C[i][j]=0;
			for ( t = 0; t < k ; t++){
				C[i][j] = C[i][j]+ A[i][t]*B[t][j];
			}	
		}
	}
}

void matmult_mnk(int m, int n, int k, double **A, double **B, double **C){
	int i,j,t;
	for ( i = 0; i < m ; i++){
		for ( j = 0; j < n; j++){
			C[i][j]=0;
			for ( t = 0; t < k ; t++){
				C[i][j] += A[i][t]*B[t][j];
			}	
		}
	}
}

void matmult_mkn(int m, int n, int k, double **A, double **B, double **C){
	int i,j,t;
	for ( i = 0; i < m ; i++){
		for ( j = 0; j < n; j++){
			C[i][j]=0;
		}
	}
	for ( i = 0; i < m ; i++){
		for ( t = 0; t < k ; t++){
			for ( j = 0; j < n; j++){
				C[i][j] += A[i][t]*B[t][j];
			}	
		}
	}
}

void matmult_nmk(int m, int n, int k, double **A, double **B, double **C){
	int i,j,t;
	for ( i = 0; i < m ; i++){
		for ( j = 0; j < n; j++){
			C[i][j]=0;
		}
	}for ( j = 0; j < n; j++){
		for ( i = 0; i < m ; i++){
			for ( t = 0; t < k ; t++){
				C[i][j] += A[i][t]*B[t][j];
			}	
		}
	}
}

void matmult_nkm(int m, int n, int k, double **A, double **B, double **C){
	int i,j,t;
	for ( i = 0; i < m ; i++){
		for ( j = 0; j < n; j++){
			C[i][j]=0;
		}
	}for ( j = 0; j < n; j++){
		for ( t = 0; t < k ; t++){
			for ( i = 0; i < m ; i++){
				C[i][j] += A[i][t]*B[t][j];
			}	
		}
	}
}

void matmult_knm(int m, int n, int k, double **A, double **B, double **C){
	int i,j,t;
	for ( i = 0; i < m ; i++){
		for ( j = 0; j < n; j++){
			C[i][j]=0;
		}
	}for ( t = 0; t < k ; t++){
		for ( j = 0; j < n; j++){
			for ( i = 0; i < m ; i++){
				C[i][j] += A[i][t]*B[t][j];
			}	
		}
	}
}

void matmult_kmn(int m, int n, int k, double **A, double **B, double **C){
	int i,j,t;
	for ( i = 0; i < m ; i++){
		for ( j = 0; j < n; j++){
			C[i][j]=0;
		}
	}for ( t = 0; t < k ; t++){
		for ( i = 0; i < m ; i++){
			for ( j = 0; j < n; j++){
				C[i][j] += A[i][t]*B[t][j];
			}	
		}
	}
}

void matmult_blk(int m, int n, int l, double **A, double **B, double **C, int bb){
	int i,j,k,jj,kk;
	for ( i = 0; i < m ; i++){
		for ( j = 0; j < n; j++){
			C[i][j]=0;
		}
	}
	double r;
	//size of block from number of elements
	bb=sqrt(bb);
	//blocked multiplication
	for(kk=0;kk<l;kk+=bb){
		for(jj=0;jj<n;jj+=bb){
			for(i=0;i<m;i++){
				for(k=kk;k<min(kk+bb,l);k++){
					r=A[i][k];
					for(j=jj;j<min(jj+bb,n);j++){
						C[i][j]+=r*B[k][j];
					}
				}
			}
		}
	}
}

}



#endif
