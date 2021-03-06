#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> 
#include <sys/time.h>

#define MAX_ITER 10000
#define NODES 500 // boundaries included


__global__ void kernel_jacobi0(double *u, double *u_old, double *u_d1, double *f, int N){

	
	int i,j;
	double h;

	h = 2.0/((double)N-1);

	j = blockIdx.y * blockDim.y + threadIdx.y+1; //row
	i = blockIdx.x * blockDim.x + threadIdx.x+1; //column

	// computing solution
	
	if (j < N/2-1 && i < N-1){		
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N] + \
				    u_old[i+(j-1)*N] + u_old[i+(j+1)*N] + \
			            h*h*f[i+j*N] );
	
	}else if (j == N/2-1 && i < N-1){
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N] + \
				    u_old[i+(j-1)*N] + u_d1[i] + \
			            h*h*f[i+j*N] );	
	}
	
}

__global__ void kernel_jacobi1(double *u, double *u_old, double *u_d0, double *f, int N){

	
	int i,j;
	double h;

	h = 2.0/((double)N-1);

	j = blockIdx.y * blockDim.y + threadIdx.y;   //row
	i = blockIdx.x * blockDim.x + threadIdx.x+1; //column

	// computing solution
	
	if (j > 0 && j < N/2-1 && i < N-1){
		
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N] + \
				    u_old[i+(j-1)*N] + u_old[i+(j+1)*N] + \
			            h*h*f[i+j*N] );	
	}else if (j == 0 && i < N-1){
		
		u[i+j*N] = 0.25 * ( u_old[(i-1)+j*N] + u_old[(i+1)+j*N]  + \
				    u_d0[i+(N/2-1)*N] + u_old[i+(j+1)*N] + \
			            h*h*f[i+j*N] );	
	}
	
}


int main(int argc, char *argv[])
{

	struct timeval t1, t2;
	gettimeofday(&t1, NULL);

	int N = NODES;
	int sizeXGridx = 1;
	int sizeXGridy = 1;
	int sizeXBlockx = N-2;
	int sizeXBlocky = N-2; 

	if (argc == 2)
	{	
		N = atoi(argv[1]);
		if (N <= 34){ 
			sizeXBlockx = N-2;
			sizeXBlocky = (sizeXBlockx/2)+1;
			sizeXGridx = 1;
			sizeXGridy = sizeXGridx;
		} else {	
			sizeXBlockx = 32;
			sizeXBlocky = sizeXBlockx; 
			sizeXGridx = (32+sizeXBlockx-1)/sizeXBlockx;
			sizeXGridy = (sizeXGridx/2)+1;
		}
	}

	// variables declaration
	int i, j, k, max_iter;
	double *u_h, *u_old_h, *f_h;
	double *u_d0, *u_old_d0, *f_d0;
	double *u_d1, *u_old_d1, *f_d1;
	double conv, *temp;

	dim3 DimGrid(sizeXGridx,sizeXGridy);
	dim3 DimBlock(sizeXBlockx,sizeXBlocky); // 484 threads per block 

	max_iter = MAX_ITER;
	
	// initializing stopwatches
	StopWatchInterface *timeKer;
	sdkCreateTimer(&timeKer);

	StopWatchInterface *timeMem10; //time to allocate to host
	StopWatchInterface *timeMem11;	
	sdkCreateTimer(&timeMem10);
	sdkCreateTimer(&timeMem11);

	StopWatchInterface *timeMem20; //time to copy to device
	StopWatchInterface *timeMem21;	
	sdkCreateTimer(&timeMem20);
	sdkCreateTimer(&timeMem21);

	StopWatchInterface *timeMem30; //time to copy to host
	StopWatchInterface *timeMem31;	
	sdkCreateTimer(&timeMem30);
	sdkCreateTimer(&timeMem31);

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
	
	
	// set device 0 as current and allocating solution and forcing term
	cudaSetDevice(0);
	sdkStartTimer(&timeMem10);	
	cudaMalloc((void**)&f_d0,(N*N/2) * sizeof(double));
	cudaMalloc((void**)&u_d0,(N*N/2) * sizeof(double));
	cudaMalloc((void**)&u_old_d0,(N*N/2) * sizeof(double));
	sdkStopTimer(&timeMem10);

	// copying from host to device 0
	sdkStartTimer(&timeMem20);
	cudaMemcpy(u_d0, u_h, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(u_old_d0,u_old_h, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(f_d0,f_h, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	sdkStopTimer(&timeMem20);

	// enable peer to peer access with device 1
	cudaDeviceEnablePeerAccess(1,0);

	// set device 1 as current and allocating solution and forcing term
	cudaSetDevice(1);
	sdkStartTimer(&timeMem11);
	cudaMalloc((void**)&f_d1,(N*N/2) * sizeof(double));
	cudaMalloc((void**)&u_d1,(N*N/2) * sizeof(double));
	cudaMalloc((void**)&u_old_d1,(N*N/2) * sizeof(double));
	sdkStopTimer(&timeMem11);

	// copying from host to device 1
	sdkStartTimer(&timeMem21);
	cudaMemcpy(u_d1, u_h+(N*N/2), (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(u_old_d1, u_old_h+(N*N/2), (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(f_d1, f_h+N*N/2, (N*N/2)*sizeof(double),cudaMemcpyHostToDevice);
	sdkStopTimer(&timeMem21);

	// enable peer to peer access with device 0
	cudaDeviceEnablePeerAccess(0,0);

	// calling kernel and taking time on device 0	
	sdkStartTimer(&timeKer);
	k = 1;
	while (k < max_iter){
		cudaSetDevice(0);
		temp = u_d0;
		u_d0 = u_old_d0;
		u_old_d0 = temp;
		kernel_jacobi0<<< DimGrid, DimBlock >>>(u_d0, u_old_d0, u_d1, f_d0, N);

		cudaSetDevice(1);
		temp = u_d1;
		u_d1 = u_old_d1;
		u_old_d1 = temp;
		kernel_jacobi1<<< DimGrid, DimBlock >>>(u_d1, u_old_d1, u_d0, f_d1, N);
		// synchronize devices
		cudaSetDevice(0);
		cudaDeviceSynchronize();
		cudaSetDevice(1);
		cudaDeviceSynchronize();
		k++;
	}
	sdkStopTimer(&timeKer);

	// copying from device 0 to host
	cudaSetDevice(0);
	sdkStartTimer(&timeMem30);
	cudaMemcpy(u_h,u_d0, N*N/2 *sizeof(double),cudaMemcpyDeviceToHost);
	sdkStopTimer(&timeMem30);

	// copying from device 1 to host
	cudaSetDevice(1);
	sdkStartTimer(&timeMem31);
	cudaMemcpy(u_h+N*N/2,u_d1, N*N/2 *sizeof(double),cudaMemcpyDeviceToHost);
	sdkStopTimer(&timeMem31);

	// print solution

	FILE * fp;

   	fp = fopen ("solution_par.txt", "w+");

	for(j=0; j<N; j++){
		for(i=0; i< N; i++){
			fprintf(fp, "%lf ",u_h[i+j*N]);
		}
		fprintf(fp, "\n");
	}
   
   	fclose(fp);	

	// print time
	//printf("Block size: %i x %i \n", sizeXBlockx,sizeXBlocky);
	//printf("Grid size: %i x %i \n", sizeXGridx, sizeXGridy);

	// freeing memory	
	free(u_old_h);
	free(u_h);
	free(f_h);
	cudaFree(u_old_d0);
	cudaFree(u_d0);
	cudaFree(f_d0);

	double tK = sdkGetTimerValue(&timeKer);

	double tM10 = sdkGetTimerValue(&timeMem10);
	double tM20 = sdkGetTimerValue(&timeMem20);
	double tM30 = sdkGetTimerValue(&timeMem30);

	double tM11 = sdkGetTimerValue(&timeMem11);
	double tM21 = sdkGetTimerValue(&timeMem21);
	double tM31 = sdkGetTimerValue(&timeMem31);


	printf("%lf \t", tK/1e3);
	printf("%lf \t", (tM10 + tM20 + tM30 + tM11 + tM21 + tM31)/1e3);

	//printf("Block size: %i x %i \n", sizeXBlock,sizeXBlock);
	//printf("Grid size: %i x %i \n", sizeXGrid, sizeXGrid);

	double gputime = (tK + tM10 + tM20 + tM30 + tM11 + tM21 + tM31)/1e3;

	gettimeofday(&t2, NULL);
	double  walltime = t2.tv_sec - t1.tv_sec + (t2.tv_usec -t1.tv_usec) / 1.e6;
	double cputime = walltime - gputime;
	printf("%lf \t", walltime);
	printf("%lf \t", cputime);

	double kerneltime = (tK/2)/max_iter;
	double flops = ((N*N*5)/1e9)/(kerneltime/1e3);
	printf("%lf \t", flops);
	
	return 0;

}
