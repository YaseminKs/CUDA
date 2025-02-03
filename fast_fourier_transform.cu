#include <stdio.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <math.h>

#define N 8  // Must be a power of 2

__global__ void fftKernel( cuComplex *X, int n, int step ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if( tid < n / 2 ){
        int pos = tid * step * 2;
        cuComplex even = X[pos];
        cuComplex odd = X[pos + step];

        float angle = -2.0f * M_PI * tid / n;
        cuComplex twiddle = make_cuComplex( cosf( angle ), sinf( angle ) );
        cuComplex temp = cuCmulf( twiddle, odd );

        X[pos] = cuCaddf( even, temp );
        X[pos + step] = cuCsubf( even, temp );
    }
}

void cudaFFT( cuComplex *h_X ){
    cuComplex *d_X;
    cudaMalloc( ( void** )&d_X, N * sizeof( cuComplex ) );
    cudaMemcpy( d_X, h_X, N * sizeof( cuComplex ), cudaMemcpyHostToDevice );

    for( int step = 1 ; step < N ; step *= 2 ){
        fftKernel<<<1, N / 2>>>( d_X, N, step );
        cudaDeviceSynchronize();
    }

    cudaMemcpy( h_X, d_X, N * sizeof( cuComplex ), cudaMemcpyDeviceToHost );
    cudaFree( d_X );
}

int main(){
    cuComplex h_X[N];
    for( int i = 0 ; i < N ; i++ ){
        h_X[i] = make_cuComplex( i, 0 ); // Example: Real input, imaginary part = 0
    }

    printf( "Input:\n" );
    for( int i = 0 ; i < N ; i++ ){
        printf( "(%f, %f) ", cuCrealf( h_X[i] ), cuCimagf( h_X[i] ) );
    }
    printf( "\n" );

    cudaFFT( h_X );

    printf( "FFT Output:\n" );
    for( int i = 0 ; i < N ; i++ ){
        printf( "(%f, %f) ", cuCrealf( h_X[i] ), cuCimagf( h_X[i] ) );
    }
    printf( "\n" );

    return 0;
}
