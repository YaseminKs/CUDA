#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 16  // Size of the array (change as needed)
#define THREADS_PER_BLOCK 8  // Define threads per block

// Swap function for merging
__device__ void swap( int *a, int *b ){
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Parallel Merge Function (Runs on GPU)
__global__ void mergeSort( int *arr, int *temp, int left, int right ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( tid >= right ) return;

    // Single element is already sorted
    if( left < right ){
        int mid = left + (right - left) / 2;

        // Recursively sort first and second halves
        mergeSort<<<1, 1>>>( arr, temp, left, mid );
        mergeSort<<<1, 1>>>( arr, temp, mid + 1, right );
        cudaDeviceSynchronize();

        // Merge sorted halves
        int i = left, j = mid + 1, k = left;

        while( i <= mid && j <= right ){
            if( arr[i] <= arr[j] )
                temp[k++] = arr[i++];
            else
                temp[k++] = arr[j++];
        }

        while( i <= mid )
            temp[k++] = arr[i++];
        while( j <= right )
            temp[k++] = arr[j++];

        for( int i = left ; i <= right ; i++ )
            arr[i] = temp[i];
    }
}

// Host function to call CUDA kernel
void cudaMergeSort( int *arr, int size ){
    int *d_arr, *d_temp;

    // Allocate memory on GPU
    cudaMalloc( ( void** )&d_arr, size * sizeof( int ) );
    cudaMalloc( ( void** )&d_temp, size * sizeof( int ) );

    // Copy array from host to device
    cudaMemcpy( d_arr, arr, size * sizeof( int ), cudaMemcpyHostToDevice );

    // Launch kernel
    mergeSort<<<1, THREADS_PER_BLOCK>>>( d_arr, d_temp, 0, size - 1 );

    // Copy result back to host
    cudaMemcpy( arr, d_arr, size * sizeof( int ), cudaMemcpyDeviceToHost );

    // Free memory
    cudaFree( d_arr );
    cudaFree( d_temp );
}

// Main function
int main(){
    int arr[N] = { 12, 11, 13, 5, 6, 7, 3, 1, 9, 15, 8, 2, 10, 14, 4, 16 };
    
    printf( "Unsorted array:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", arr[i] );
    printf( "\n" );

    // Call CUDA Merge Sort
    cudaMergeSort( arr, N );

    printf( "Sorted array:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", arr[i] );
    printf( "\n" );

    return 0;
}
