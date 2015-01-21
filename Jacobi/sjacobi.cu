#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> 
#include <sys/time.h>
#include <time.h>

#define MAX_ITER 10000
#define NODES 8  // boundaries included



__global__ void kernel_jacobi(double *u, double *u_old, double *f, int N, int max_iter){

	
	int i,j,k;
	double h, *temp;

	h = 2.0/((double)N-1);
	k = 1;

	j = blockIdx.x * blockDim.x + threadIdx.x+1;

	// computing solution
	
	while(k < max_iter){
			
		temp = u;
		u = u_old;
		u_old = temp;	

		// update solution
		//for(j=1; j<N-1; j++){
			for(i=1; i< N-1; i++){
				u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N] + \
						    u_old[i+(j-1)*N] + u_old[i+(j+1)*N] + \
					            h*h*f[i+j*N] );	
			}
		//}

		k++;
		
	} /* end while */
}

int main(int argc, char *argv[])
{

	int N = NODES;
	int sizeXGrid = 1;
	int sizeXBlock = N-2;

	int i, j, k, max_iter;
	double *u_h, *u_old_h, *f_h;
	double *u_d, *u_old_d, *f_d;
	double conv;

	dim3 DimGrid = sizeXGrid;
	dim3 DimBlock = sizeXBlock;

	max_iter = MAX_ITER;
	

	// allocating solution and forcing term in the host 
	f_h     = (double *)malloc(N*N * sizeof(double));
	u_h     = (double *)malloc(N*N * sizeof(double));
	u_old_h = (double *)malloc(N*N * sizeof(double));	

	
	//initialinzing solution and forcing term
	conv = 2.0/((double)N-1);
	for(j=0; j<N; j++){
		for(i=0; i<N; i++)
		{		
			if( conv*i-1.0 > 0.0 && conv*i-1.0 < 1.0/3.0 &&\
		   conv*j-1.0 > -2.0/3.0 && conv*j-1.0 < -1.0/3.0){			
				f_h[i+j*N]=200;
			}else {f_h[i+j*N]=0;}

			u_h[i+j*N] = 0;
			u_old_h[i+j*N] = 0;
			if(i == 0 || j == N-1 || i == N-1){ // boundary condition u(1,y) and u(-1,y)
				u_h[i+j*N] = 20;
				u_old_h[i+j*N] = 20;
			}
		} 
	}
	
	// allocating solution and forcing term in the device
	cudaMalloc((void**)&f_d,N*N * sizeof(double));
	cudaMalloc((void**)&u_d,N*N * sizeof(double));
	cudaMalloc((void**)&u_old_d,N*N * sizeof(double));

	// copying from device to host
	cudaMemcpy(u_d,u_h, N*N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(u_old_d,u_old_h, N*N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(f_d,f_h, N*N*sizeof(double),cudaMemcpyHostToDevice);

	// calling kernel	
	kernel_jacobi<<< DimGrid, DimBlock >>>(u_d, u_old_d, f_d, N, max_iter);

	// copying from device to host
	cudaMemcpy(u_h,u_d, N*N *sizeof(double),cudaMemcpyDeviceToHost);

	// print solution
	FILE * fp;

   	fp = fopen ("solution.txt", "w+");

	for(j=0; j<N; j++){
		for(i=0; i< N; i++){
			fprintf(fp, "%lf ",u_h[i+j*N]);
		}
		fprintf(fp, "\n");
	}
   
   	fclose(fp);	


	// freeing memory	
	free(u_old_h);
	free(u_h);
	free(f_h);
	
	return 0;

}
