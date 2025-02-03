#include <stdio.h>
#include <cuda.h>

#define N 16
#define NUM_BITS 8  // Assumes 8-bit integers for simplicity

__global__ void countSortKernel( int *d_arr, int *d_out, int n, int bit ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if( tid < n ){
        int mask = ( 1 << bit );
        int count = 0;

        for( int i = 0 ; i < n ; i++ ){
            if( ( d_arr[i] & mask ) == 0 )
                count++;
        }

        int new_pos = ( d_arr[tid] & mask ) ? count + __syncthreads_count( d_arr[tid] & mask ) 
                              - 1 : tid - __syncthreads_count( d_arr[tid] & mask );
        d_out[new_pos] = d_arr[tid];
    }
}

void cudaRadixSort( int *arr ){
    int *d_arr, *d_out;
    cudaMalloc( ( void** )&d_arr, N * sizeof( int ) );
    cudaMalloc( ( void** )&d_out, N * sizeof( int ) );
    cudaMemcpy( d_arr, arr, N * sizeof( int ), cudaMemcpyHostToDevice );

    for( int bit = 0 ; bit < NUM_BITS ; bit++ ){
        countSortKernel<<<1, N>>>( d_arr, d_out, N, bit );
        cudaMemcpy( d_arr, d_out, N * sizeof( int ), cudaMemcpyDeviceToDevice );
    }

    cudaMemcpy( arr, d_arr, N * sizeof( int ), cudaMemcpyDeviceToHost );
    cudaFree( d_arr );
    cudaFree( d_out );
}

int main(){
    int arr[N] = { 170, 45, 75, 90, 802, 24, 2, 66, 15, 9, 98, 32, 49, 100, 150, 1 };

    printf( "Unsorted array:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", arr[i] );
    printf( "\n" );

    cudaRadixSort( arr );

    printf( "Sorted array:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", arr[i] );
    printf( "\n" );

    return 0;
}
