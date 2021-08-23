#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

#pragma omp parallel for
  for (int i = 0; i < N; i++)
  {
    y[i] = 2.0f * x[i] + y[i];
  }

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, abs(y[i] - 4.0f));
  printf("Max error: %f\n", maxError);

  free(x);
  free(y);
}