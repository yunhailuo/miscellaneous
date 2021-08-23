#include <math.h>
#include <stdio.h>
#include <string.h>

int main()
{
    int m = 512, n = 512;
    int iter = 0, iter_max = 1000;
    float err = 1.0, tol = 1.0e-5;
    const float pi  = 2.0 * asinf(1.0);
    float y0[n];
    float A[n][m], Anew[n][m];

    // initialization
    memset(A, 0, n * m * sizeof(float));

    // set boundary conditions
#pragma omp parallel for
    for (int i = 0; i < m; i++)
    {
        A[0][i]   = 0.0;
        A[n-1][i] = 0.0;
    }
#pragma omp parallel for
    for (int j = 0; j < n; j++)
    {
        y0[j] = sinf(pi * j / (n-1));
        A[j][0] = y0[j];
        A[j][m-1] = y0[j]*expf(-pi);
    }

    while ( err > tol && iter < iter_max ) {
        err = 0.0;
#pragma omp parallel for collapse(2) reduction (max:err)
        for( int j = 1; j < n-1; j++) {
            for(int i = 1; i < m-1; i++) {
                Anew[j][i] = 0.25 * (A[j][i+1] + A[j][i-1] +
                A[j-1][i] + A[j+1][i]);
                err = fmax(err, fabs(Anew[j][i] - A[j][i]));
            }
        }
#pragma omp parallel for collapse(2)
        for( int j = 1; j < n-1; j++) {
            for( int i = 1; i < m-1; i++ ) {
                A[j][i] = Anew[j][i];
            }
        }
        iter++;
    }
    printf("%e\n", err);
    printf("%d\n", iter);
}
