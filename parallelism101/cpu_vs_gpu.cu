#include <stdio.h>
#include <sys/time.h>

__global__
void log1p(double p, double q, double * y)
{
  y[0] = q + log1p(exp(p - q));
}

__global__
void log_1p(double p, double q, double * y)
{
  y[0] = q + log(1 + exp(p - q));
}

int main(void)
{
  double a, b;
  double * y;

  cudaMallocManaged(&y, sizeof(*y));
  y[0] = 100.0;

  printf("Case1:\n");
  a = -7869.9955677831958382739685475826263427734375;
  b = -7869.5160871966154445544816553592681884765625;
  log1p<<<1, 1>>>(a, b, y);
  printf("log1p      CPU: %.60f\n", b + log1p(exp(a - b)));
  cudaDeviceSynchronize();
  printf("log1p      GPU: %.60f\n", y[0]);
  log_1p<<<1, 1>>>(a, b, y);
  printf("log(1 + x) CPU: %.60f\n", b + log(1 + exp(a - b)));
  cudaDeviceSynchronize();
  printf("log(1 + x) GPU: %.60f\n", y[0]);

  printf("Case2:\n");
  a = -39983.496316437478526495397090911865234375;
  b = -39983.274149101882358081638813018798828125;
  log1p<<<1, 1>>>(a, b, y);
  printf("log1p      CPU: %.60f\n", b + log1p(exp(a - b)));
  cudaDeviceSynchronize();
  printf("log1p      GPU: %.60f\n", y[0]);
  log_1p<<<1, 1>>>(a, b, y);
  printf("log(1 + x) CPU: %.60f\n", b + log(1 + exp(a - b)));
  cudaDeviceSynchronize();
  printf("log(1 + x) GPU: %.60f\n", y[0]);

  printf("Case3:\n");
  a = -2639.88414462528953663422726094722747802734375;
  b = -2633.387596741364177432842552661895751953125;
  log1p<<<1, 1>>>(a, b, y);
  printf("log1p      CPU: %.60f\n", b + log1p(exp(a - b)));
  cudaDeviceSynchronize();
  printf("log1p      GPU: %.60f\n", y[0]);
  log_1p<<<1, 1>>>(a, b, y);
  printf("log(1 + x) CPU: %.60f\n", b + log(1 + exp(a - b)));
  cudaDeviceSynchronize();
  printf("log(1 + x) GPU: %.60f\n", y[0]);

  cudaFree(y);
}