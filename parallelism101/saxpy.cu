#include <stdio.h>
#include <sys/time.h>

__global__ void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

int main(void)
{
  int N = 1 << 20;
  float *x, *y, *d_x, *d_y;
  x = (float *)malloc(N * sizeof(float));
  y = (float *)malloc(N * sizeof(float));

  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);

  clock_gettime(CLOCK_MONOTONIC, &end);
  double time_taken;
  time_taken = (end.tv_sec - start.tv_sec) * 1e9;
  time_taken = (time_taken + (end.tv_nsec - start.tv_nsec)) * 1e-9;
  printf("Time: %f\n", time_taken);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i] - 4.0f));
  printf("Max error: %f\n", maxError);

  free(x);
  free(y);
}