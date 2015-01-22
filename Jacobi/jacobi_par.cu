#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> 
#include <sys/time.h>

#define MAX_ITER 1
#define NODES 10 // boundaries included


__global__ void kernel_jacobi0(double *u, double *u_old, double *u_d1, double *f, int N){

	
	int i,j;
	double h;

	h = 2.0/((double)N-1);

	j = blockIdx.y * blockDim.y + threadIdx.y+1; //row
	i = blockIdx.x * blockDim.x + threadIdx.x+1; //column

	// computing solution
	if (j == 1 && i < N-1){ printf("boundary %5.0lf in thread %i, %i \n", u[i+(j-1)*N], j-1, i);}	
	if (j < N/2-1 && i < N-1){ printf("dev 0 %5.0lf in thread %i, %i \n", u[i+j*N], j, i);}
	if (j == N/2-1 && i < N-1){ printf(" last row %5.0lf in thread %i, %i \n", u[i+j*N], j, i);}
	
	/*if (j < N/2-1 && i < N-1){
		
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N] + \
				    u_old[i+(j-1)*N] + u_old[i+(j+1)*N] + \
			            h*h*f[i+j*N] );	
	}else if (j == N/2-1 && i < N-1){
		
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N] + \
				    u_old[i+(j-1)*N] + u_d1[i] + \
			            h*h*f[i+j*N] );	
	}*/
	
}

__global__ void kernel_jacobi1(double *u, double *u_old, double *u_d0, double *f, int N){

	
	int i,j;
	double h;

	h = 2.0/((double)N-1);

	j = blockIdx.y * blockDim.y + threadIdx.y;   //row
	i = blockIdx.x * blockDim.x + threadIdx.x+1; //column

	// computing solution
	if (j > 0 && j < N/2-1 && i < N-1){ printf("dev 1 %5.0lf in thread %i, %i \n", u[i+j*N], j, i);}
	if (j == 0 && i < N-1){ printf(" first row row %5.0lf in thread %i, %i \n", u[i+j*N], j, i);}
	if (j == N/2 && i < N-1){ printf(" boundary %5.0lf in thread %i, %i \n", u[i+j*N], j, i);}
	
	//printf("%lf \n", u[i+j*N]);

	/*if (j > 0 && j < N/2-1 && i < N-1){
		
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N] + \
				    u_old[i+(j-1)*N] + u_old[i+(j+1)*N] + \
			            h*h*f[i+j*N] );	
	}else if (j == 0 && i < N-1){
		
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N]  + \
				    u_d0[i+(N/2-1)*N] + u_old[i+(j+1)*N] + \
			            h*h*f[i+j*N] );	
	}*/
	
}


int main(int argc, char *argv[])
{

	int N;// = NODES;
	int sizeXGrid;// = 1;
	int sizeXBlock;// = N-2; 

	//if (argc == 2)
	//{	
		N = NODES;//atoi(argv[1]);
		if (N <= 24){ 
			sizeXBlock = N-2;
			sizeXGrid = 1;
		} else {	
			sizeXBlock = 16; 
			sizeXGrid = ((N-2)+sizeXBlock-1)/sizeXBlock;
		}
	//}

	// variables declaration
	int i, j, k, max_iter;
	double *u_h, *u_old_h, *f_h;
	double *u_d0, *u_old_d0, *f_d0;
	double *u_d1, *u_old_d1, *f_d1;
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
			if(i == 1 && j==5) {u_h[i+j*N] = 365;}
			if(i == 8 && j==8) {u_h[i+j*N] = 234;}
			printf("%5.0lf ",u_h[i+j*N]);
		} 
		printf("\n");
	}
	
	// set device 0 as current and allocating solution and forcing term
	cudaSetDevice(0);
	cudaMalloc((void**)&f_d0,(N*N/2) * sizeof(double));
	cudaMalloc((void**)&u_d0,(N*N/2) * sizeof(double));
	cudaMalloc((void**)&u_old_d0,(N*N/2) * sizeof(double));

	// copying from host to device 0
	cudaMemcpy(u_d0, u_h, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(u_old_d0,u_old_h, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(f_d0,f_h, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);

	// enable peer to peer access with device 1
	cudaDeviceEnablePeerAccess(1,0);

	// set device 1 as current and allocating solution and forcing term
	cudaSetDevice(1);
	cudaMalloc((void**)&f_d1,(N*N/2) * sizeof(double));
	cudaMalloc((void**)&u_d1,(N*N/2) * sizeof(double));
	cudaMalloc((void**)&u_old_d1,(N*N/2) * sizeof(double));

	// copying from host to device 1
	cudaMemcpy(u_d1, u_h+N*N/2, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(u_old_d1, u_old_h+(N*N/2), (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(f_d1, f_h+N*N/2, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);

	// enable peer to peer access with device 0
	cudaDeviceEnablePeerAccess(0,0);

	// calling kernel and taking time on device 0	
	sdkStartTimer(&timeKer);
	k = 0;
	while (k < max_iter){
		cudaSetDevice(0);
		//temp = u_d0;
		//u_d0 = u_old_d0;
		//u_old_d0 = temp;
		kernel_jacobi0<<< DimGrid, DimBlock >>>(u_d0, u_old_d0, u_d1, f_d0, N);

		cudaSetDevice(1);
		//temp = u_d1;
		//u_d1 = u_old_d1;
		//u_old_d1 = temp;
		kernel_jacobi1<<< DimGrid, DimBlock >>>(u_d1, u_old_d1, u_d0, f_d1, N);
		// synchronize devices
		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaSetDevice(1);
		cudaDeviceSynchronize();
		k++;
	}
	sdkStopTimer(&timeKer);

/*
	// synchronize devices
	cudaSetDevice(0);
	cudaCheckErrors(cudaDeviceSynchronize());
	cudaSetDevice(1);
	cudaCheckErrors(cudaDeviceSynchronize());*/

	// copying from device 0 to host
	cudaSetDevice(0);
	cudaMemcpy(u_h,u_d0, N*N/2 *sizeof(double),cudaMemcpyDeviceToHost);

	// copying from device 1 to host
	cudaSetDevice(0);
	cudaMemcpy(u_h+N*N/2,u_d0, N*N/2 *sizeof(double),cudaMemcpyDeviceToHost);

	// print solution
	FILE * fp;

   	fp = fopen ("solution_par.txt", "w+");

	for(j=0; j<N; j++){
		for(i=0; i< N; i++){
			fprintf(fp, "%5.0lf ",u_h[i+j*N]);
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
	cudaFree(u_old_d0);
	cudaFree(u_d0);
	cudaFree(f_d0);
	
	return 0;

}
