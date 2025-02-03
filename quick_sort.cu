#include <stdio.h>
#include <cuda.h>

#define N 16  // Change the size of the array as needed

// CUDA Partition Function
__device__ int partition( int *arr, int low, int high ){
    int pivot = arr[high];  // Last element as pivot
    int i = low - 1;        // Small index

    for( int j = low ; j < high ; j++ ){
        if( arr[j] <= pivot ){
            i++;
            // Swap arr[i] and arr[j]
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
    }
    // Swap arr[i+1] and pivot
    int temp = arr[i + 1];
    arr[i + 1] = arr[high];
    arr[high] = temp;

    return (i + 1);
}

// CUDA Quick Sort Kernel
__global__ void quickSortKernel( int *arr, int low, int high ){
    if( low < high ){
        int pi = partition( arr, low, high );

        // Launch parallel recursive calls
        quickSortKernel<<<1, 1>>>( arr, low, pi - 1 );
        quickSortKernel<<<1, 1>>>( arr, pi + 1, high );
        cudaDeviceSynchronize();
    }
}

// Host function
void cudaQuickSort( int *arr, int size ){
    int *d_arr;
    
    cudaMalloc( ( void** )&d_arr, size * sizeof( int ) );
    cudaMemcpy( d_arr, arr, size * sizeof( int ), cudaMemcpyHostToDevice );

    quickSortKernel<<<1, 1>>>( d_arr, 0, size - 1 );
    cudaMemcpy( arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree( d_arr );
}

// Main Function
int main() {
    int arr[N] = { 12, 34, 7, 3, 16, 8, 24, 1, 9, 15, 18, 2, 10, 14, 4, 6 };

    printf( "Unsorted array:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", arr[i] );
    printf( "\n" );

    cudaQuickSort( arr, N );

    printf( "Sorted array:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", arr[i] );
    printf( "\n" );

    return 0;
}
