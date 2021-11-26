#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <chrono>
static void convolve_2d(float *input_img, float *kernel,float *output_img, const int& ht, const int& wd, const int& f, const int& stride = 1) {
  int n_wd = (wd - f) / stride + 1;
  int n_ht = (ht - f) / stride + 1;
  __m256 p_res1 = _mm256_setzero_ps();
  __m256 p_res2 = _mm256_setzero_ps();
  __m256 p_res3 = _mm256_setzero_ps();
  __m256 p_res4 = _mm256_setzero_ps();
  __m256 p_res5 = _mm256_setzero_ps();
  __m256 p_res6 = _mm256_setzero_ps();
  __m256 p_res7 = _mm256_setzero_ps();
  __m256 p_res8 = _mm256_setzero_ps();

  __m256 brod;

  for (int i = 0; i < n_ht; i++) {
    int j = 0;
    for (j = 0; j <= n_wd - 64; j += 64) {
      p_res1 = _mm256_loadu_ps(&output_img[i * n_wd + j]);
      p_res2 = _mm256_loadu_ps(&output_img[i * n_wd + j + 8]);
      p_res3 = _mm256_loadu_ps(&output_img[i * n_wd + j + 16]);
      p_res4 = _mm256_loadu_ps(&output_img[i * n_wd + j + 24]);
      p_res5 = _mm256_loadu_ps(&output_img[i * n_wd + j + 32]);
      p_res6 = _mm256_loadu_ps(&output_img[i * n_wd + j + 40]);
      p_res7 = _mm256_loadu_ps(&output_img[i * n_wd + j + 48]);
      p_res8 = _mm256_loadu_ps(&output_img[i * n_wd + j + 56]);
      for (int fy = 0; fy < f; fy++)
        for (int fx = 0; fx < f; fx++) {
          brod = _mm256_set1_ps(kernel[fy * f + fx]);
          p_res1 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), brod, p_res1);
          p_res2 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 8 + fx]), brod, p_res2);
          p_res3 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 16 + fx]), brod, p_res3);
          p_res4 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 24 + fx]), brod, p_res4);
          p_res5 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 32 + fx]), brod, p_res5);
          p_res6 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 40 + fx]), brod, p_res6);
          p_res7 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 48 + fx]), brod, p_res7);
          p_res8 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 56 + fx]), brod, p_res8);
        }
      _mm256_storeu_ps(&output_img[i * n_wd + j], p_res1);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 8], p_res2);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 16], p_res3);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 24], p_res4);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 32], p_res5);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 40], p_res6);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 48], p_res7);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 56], p_res8);
    }

    for (; j <= n_wd - 32; j += 32) {
      p_res1 = _mm256_loadu_ps(&output_img[i * n_wd + j]);
      p_res2 = _mm256_loadu_ps(&output_img[i * n_wd + j + 8]);
      p_res3 = _mm256_loadu_ps(&output_img[i * n_wd + j + 16]);
      p_res4 = _mm256_loadu_ps(&output_img[i * n_wd + j + 24]);
      for (int fy = 0; fy < f; fy++)
        for (int fx = 0; fx < f; fx++) {
          brod = _mm256_set1_ps(kernel[fy * f + fx]);
          p_res1 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), brod, p_res1);
          p_res2 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 8 + fx]), brod, p_res2);
          p_res3 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 16 + fx]), brod, p_res3);
          p_res4 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 24 + fx]), brod, p_res4);
        }
      _mm256_storeu_ps(&output_img[i * n_wd + j], p_res1);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 8], p_res2);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 16], p_res3);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 24], p_res4);
    }

    for (; j <= n_wd - 16; j += 16) {
      p_res1 = _mm256_loadu_ps(&output_img[i * n_wd + j]);
      p_res2 = _mm256_loadu_ps(&output_img[i * n_wd + j + 8]);
      for (int fy = 0; fy < f; fy++)
        for (int fx = 0; fx < f; fx++) {
          brod = _mm256_set1_ps(kernel[fy * f + fx]);
          p_res1 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), brod, p_res1);
          p_res2 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + 8 + fx]), brod, p_res2);
        }
      _mm256_storeu_ps(&output_img[i * n_wd + j], p_res1);
      _mm256_storeu_ps(&output_img[i * n_wd + j + 8], p_res2);
    }

    for (; j <= n_wd - 8; j += 8) {
      p_res1 = _mm256_loadu_ps(&output_img[i * n_wd + j]);
      for (int fy = 0; fy < f; fy++)
        for (int fx = 0; fx < f; fx++)
          p_res1 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), _mm256_set1_ps(kernel[fy * f + fx]), p_res1);

      _mm256_storeu_ps(&output_img[i * n_wd + j], p_res1);
    }

    if (j < n_wd) {
      p_res1 = _mm256_setzero_ps();
      for (int rmd = j, pi = 0; rmd < n_wd; rmd++)
        p_res1[pi++] = output_img[i * n_wd + rmd];
      for (int fy = 0; fy < f; fy++)
        for (int fx = 0; fx < f; fx++)
          p_res1 = _mm256_fmadd_ps(_mm256_loadu_ps(&input_img[(i * stride + fy) * wd + j * stride + fx]), _mm256_set1_ps(kernel[fy * f + fx]), p_res1);

      for (int pi = 0; j < n_wd; j++)
        output_img[i * n_wd + j] = p_res1[pi++];
    }


  }
}
#include <immintrin.h>
#include <vector>
#include <assert.h>



float dot(std::int32_t n, float x[], float y[])
{

  float sum=0;
  int i=0;
  __m256 temp256 = _mm256_setzero_ps();
  for (; i <= n - 8; i += 8) {
      __m256 vx = _mm256_loadu_ps(&x[i]);
      __m256 vy = _mm256_loadu_ps(&y[i]);
    temp256 = _mm256_add_ps(_mm256_mul_ps(vx, vy), temp256);
  }
  sum += temp256[0];
  sum += temp256[1];
  sum += temp256[2];
  sum += temp256[3];
  sum += temp256[4];
  sum += temp256[5];
  sum += temp256[6];
  sum += temp256[7];



//  float* f = (float*)&temp;
//  for (int i=0;i<8;i++)
//    printf("%f,",f[i]);

  for (int j=0;j<n-i;j++)
    sum+=x[j]*y[j];

  return sum;
}
float dot1(std::int32_t n, float x[], float y[])
{
  float res=0;
  for (int j=0;j<n;j++)
    res+=x[j]*y[j];

  return res;
}




int main(int argc, char **argv)
{
  float *matrix ;
  float *filter ;
  float *result ;

  int sizeMatrix = 28;
  int sizeKernel = 5;
  int padding = 0;
  int strides = 1;

  long long sum1 = 0;
  int runs = 1;

//  tsc_counter t0, t1;

  int sizeResult = (((sizeMatrix - sizeKernel + 2 * padding) / strides) + 1);
  printf("size Result: %d\n", sizeResult);
  matrix =(float *)_aligned_malloc(sizeMatrix * sizeMatrix * sizeof(float), 32 );
  filter =(float *)_aligned_malloc(sizeKernel * sizeKernel * sizeof(float),32);
  result=(float *)_aligned_malloc( sizeResult * sizeResult * sizeof(float),32);

//  matrix =(float *)malloc(sizeMatrix * sizeMatrix * sizeof(float) );
//  filter =(float *)malloc(sizeKernel * sizeKernel * sizeof(float));
//  result=(float *)malloc( sizeResult * sizeResult * sizeof(float));

  for (int i = 0; i != sizeKernel*sizeKernel; ++i) {
    filter[i] = 2;
  }

  for(int i = 0; i<sizeMatrix*sizeMatrix;++i) {
    matrix[i] = 1.0;
  }

  for(int i = 0; i<sizeResult*sizeResult;++i) {
    result[i] = 0.0;
  }
  auto start =  std::chrono::system_clock::now();
  for (int i=0;i<100;i++)
  convolve_2d(matrix, filter, result, sizeMatrix, sizeMatrix,sizeKernel, strides) ;
  auto end =  std::chrono::system_clock::now();
  auto cost = std::chrono::duration<double, std::micro>(end - start).count();
  printf("Average time: %lf \n", (double) cost );
//     for(int i = 0; i<sizeResult;i++){
//         for(int j= 0; j<sizeResult;j++){
//            printf("%f           ",result[i*sizeResult + j]);
//         }
//         printf("\n");
//     }
//free(matrix);
//free(result);
//free(filter);

  float sum=dot(20,filter,matrix);
  printf("float sum %lf\n",sum);
  sum=dot1(20,filter,matrix);
  printf("float sum %lf\n",sum);
  _aligned_free(matrix);
  _aligned_free(result);
  _aligned_free(filter);

}