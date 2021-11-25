#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "rdtsc.h"
#include <immintrin.h>
#include <chrono>
void kernel( int sizeKernel, int sizeMatrix, int sizeResult, double*  matrix, double*  result, double*  filter ) {
  /*
   * Assumptions:
   * matrix - stored row-wise
   * res - stored row-wise
   * filter - stored row-wise
  */

  for (int i = 0; i < sizeResult; i++) {
    for (int j = 0; j < sizeResult; j++) {
      for (int kx = 0; kx < sizeKernel; kx++) {
        for (int ky = 0; ky < sizeKernel; ky++) {
          result[i * sizeResult + j] += matrix[(i * sizeMatrix)+ j + kx + ky] * filter[kx * sizeKernel + ky];
        }
      }
    }
  }
}


int main(int argc, char **argv){
  int sizeMatrix = 28;//atoi(argv[1]);
  int runs = 1;

  double *matrix ;
  double *filter ;
  double *result ;

  int sizeKernel = 5;
  int padding = 0;
  int strides = 1;
  long long sum1 = 0;
  tsc_counter t0, t1;

  int sizeResult = (((sizeMatrix - sizeKernel + 2 * padding) / strides) + 1);
  printf("size Result: %d\n", sizeResult);

  matrix =(double *)_aligned_malloc(sizeMatrix * sizeMatrix * sizeof(double), 64 );
  filter =(double *)_aligned_malloc(sizeKernel * sizeKernel * sizeof(double),64);
  result=(double *)_aligned_malloc( sizeResult * sizeResult * sizeof(double),64);

  for (int i = 0; i != 16; ++i) {
    filter[i] = 2;
  }

  for(int i = 0; i<sizeMatrix*sizeMatrix;++i) {
    matrix[i] = 1.0;
  }

  for(int i = 0; i<sizeResult*sizeResult;++i) {
    result[i] = 0.0;
  }

  auto start =  std::chrono::system_clock::now();
      kernel(sizeKernel,sizeMatrix,sizeResult,matrix,result,filter);
  auto end =  std::chrono::system_clock::now();
  auto cost = std::chrono::duration<double, std::micro>(end - start).count();
  printf("Average time: %lf \n", (double) cost );


/**
 * To check correctness uncomment the below code
 */

//     for(int i = 0; i<sizeResult;i++){
//         for(int j= 0; j<sizeResult;j++){
//            printf("%f           ",result[i*sizeResult + j]);
//         }
//         printf("\n");
//     }

      _aligned_free(matrix);
      _aligned_free(result);
      _aligned_free(filter);
    }