#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>

#define N 1000000

__global__ void monteCarloKernel( int *d_count, int n ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curandState state;
    curand_init( 1234, tid, 0, &state );

    if( tid < n ){
        float x = curand_uniform( &state );
        float y = curand_uniform( &state );
        if( x * x + y * y <= 1.0f )
            atomicAdd( d_count, 1 );
    }
}

void cudaMonteCarlo(){
    int h_count = 0, *d_count;
    cudaMalloc( ( void** )&d_count, sizeof( int ) );
    cudaMemcpy( d_count, &h_count, sizeof( int ), cudaMemcpyHostToDevice );

    monteCarloKernel<<<( N + 255 ) / 256, 256>>>( d_count, N );
    cudaMemcpy( &h_count, d_count, sizeof( int ), cudaMemcpyDeviceToHost );

    cudaFree( d_count );
    printf( "Estimated Pi: %f\n", ( 4.0f * h_count ) / N );
}

int main(){
    cudaMonteCarlo();
    return 0;
}
