#include <stdio.h>
#include <stdlib.h>
//#include <sunperf.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define min(a,b)(((a)<(b))?(a):(b))

#ifndef MATMULT_LIB_H
#define MATMULT_LIB_H

void matmult_lib(int m, int n, int k, double **A, double **B, double **C){
	double alpha, beta;
	char transa, transb;
	int i,j;
	alpha = 1.0; beta = 0.0;
	transa = 'n', transb='n';
  
	dgemm(transa, transb, n, m, k, alpha, B[0], n, A[0], k, beta, C[0], n);
		
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
	int i,j,k,ii,jj,kk;
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
#endif
