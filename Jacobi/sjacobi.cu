#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> 
#include <sys/time.h>

#define MAX_ITER 1000
#define NODES 10 // boundaries included



__global__ void kernel_jacobi(double *u, double *u_old, double *f, int N){

	
	int i,j;
	double h;

	h = 2.0/((double)N-1);

	i = blockIdx.y * blockDim.y + threadIdx.y+1;
	j = blockIdx.x * blockDim.x + threadIdx.x+1;

	// computing solution
	
	if (j < N-1 && i < N-1){
		
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N] + \
				    u_old[i+(j-1)*N] + u_old[i+(j+1)*N] + \
			            h*h*f[i+j*N] );	
	}
	
}

int main(int argc, char *argv[])
{

	int N = NODES;
	int sizeXGrid = 1;
	int sizeXBlock = N-2; 

	if (argc == 2)
	{	
		N = atoi(argv[1]);
		if (N <= 24){ 
			sizeXBlock = N-2;
			sizeXGrid = 1;
		} else {	
			sizeXBlock = 22; 
			sizeXGrid = ((N-2)+sizeXBlock-1)/sizeXBlock;
		}
	}

	// variables declaration
	int i, j, k, max_iter;
	double *u_h, *u_old_h, *f_h;
	double *u_d, *u_old_d, *f_d;
	double conv, *temp;

	dim3 DimGrid(sizeXGrid,sizeXGrid);
	dim3 DimBlock(sizeXBlock,sizeXBlock); // 484 threads per block 

	max_iter = MAX_ITER;
	
	// initializing stopwatches
	StopWatchInterface *timeKer;
	sdkCreateTimer(&timeKer);

	// allocating solution and forcing term in the host 
	f_h     = (double *)malloc(N*N * sizeof(double));
	u_h     = (double *)malloc(N*N * sizeof(double));
	u_old_h = (double *)malloc(N*N * sizeof(double));	

	
	//initialinzing solution and forcing term
	conv = 2.0/((double)N);
	for(j=0; j<N; j++){
		for(i=0; i<N; i++)
		{		
			if( conv*i-1.0 > 0.0 && conv*i-1.0 < 1.0/3.0 &&\
		   conv*j-1.0 > -2.0/3.0 && conv*j-1.0 < -1.0/3.0){			
				f_h[i+j*N]=200;
			}else {f_h[i+j*N]=0;}

			u_h[i+j*N] = 0;
			u_old_h[i+j*N] = 0;
			// boundary conditions
			if(i == 0 || j == N-1 || i == N-1){ 
				u_h[i+j*N] = 20;
				u_old_h[i+j*N] = 20;
			}
		} 
	}
	
	// allocating solution and forcing term in the device
	cudaMalloc((void**)&f_d,N*N * sizeof(double));
	cudaMalloc((void**)&u_d,N*N * sizeof(double));
	cudaMalloc((void**)&u_old_d,N*N * sizeof(double));

	// copying from host to device
	cudaMemcpy(u_d,u_h, N*N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(u_old_d,u_old_h, N*N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(f_d,f_h, N*N*sizeof(double),cudaMemcpyHostToDevice);

	// calling kernel and taking time	
	sdkStartTimer(&timeKer);
	k = 1;
	while (k < max_iter){
		temp = u_d;
		u_d = u_old_d;
		u_old_d = temp;
		kernel_jacobi<<< DimGrid, DimBlock >>>(u_d, u_old_d, f_d, N);
		cudaDeviceSynchronize();
		k++;
	}
	sdkStopTimer(&timeKer);

	// copying from device to host
	cudaMemcpy(u_h,u_d, N*N *sizeof(double),cudaMemcpyDeviceToHost);

	// print solution
	FILE * fp;

   	fp = fopen ("solution2.txt", "w+");

	for(j=0; j<N; j++){
		for(i=0; i< N; i++){
			fprintf(fp, "%lf ",u_h[i+j*N]);
		}
		fprintf(fp, "\n");
	}
   
   	fclose(fp);	

	// print time
	double tK = sdkGetTimerValue(&timeKer);
	printf("Kernel time: %f \n", tK/1e3);
	printf("Block size: %i x %i \n", sizeXBlock,sizeXBlock);
	printf("Grid size: %i x %i \n", sizeXGrid, sizeXGrid);

	// freeing memory	
	free(u_old_h);
	free(u_h);
	free(f_h);
	cudaFree(u_old_d);
	cudaFree(u_d);
	cudaFree(f_d);
	
	return 0;

}
