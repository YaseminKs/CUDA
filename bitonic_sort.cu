#include <stdio.h>
#include <cuda.h>

#define N 16  // Must be a power of 2

// CUDA Bitonic Merge
__global__ void bitonicMerge( int *arr, int j, int k ){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int ixj = tid ^ j;

    if( ixj > tid ){
        if( ( tid & k ) == 0 ){
            if( arr[tid] > arr[ixj] ){
                int temp = arr[tid];
                arr[tid] = arr[ixj];
                arr[ixj] = temp;
            }
        }else{
            if( arr[tid] < arr[ixj] ){
                int temp = arr[tid];
                arr[tid] = arr[ixj];
                arr[ixj] = temp;
            }
        }
    }
}

// CUDA Bitonic Sort
__global__ void bitonicSort( int *arr ){
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for( int k = 2 ; k <= N ; k <<= 1 ){        // Bitonic sequence size
        for( int j = k >> 1 ; j > 0 ; j >>= 1 )  // Sorting step
            bitonicMerge<<<1, N>>>( arr, j, k );
    }
}

// Host function
void cudaBitonicSort( int *arr ){
    int *d_arr;
    
    cudaMalloc( ( void** )&d_arr, N * sizeof( int ) );
    cudaMemcpy( d_arr, arr, N * sizeof( int ), cudaMemcpyHostToDevice );

    bitonicSort<<<1, N>>>( d_arr );
    cudaMemcpy( arr, d_arr, N * sizeof( int ), cudaMemcpyDeviceToHost );

    cudaFree( d_arr );
}

// Main Function
int main(){
    int arr[N] = { 12, 34, 7, 3, 16, 8, 24, 1, 9, 15, 18, 2, 10, 14, 4, 6 };

    printf( "Unsorted array:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", arr[i] );
    printf( "\n" );

    cudaBitonicSort( arr );

    printf( "Sorted array:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", arr[i] );
    printf( "\n" );

    return 0;
}
