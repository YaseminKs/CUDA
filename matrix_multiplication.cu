#include <iostream>
#include <cuda_runtime.h>

#define N 16  // Matrix size (N x N)

// CUDA Kernel for Matrix Multiplication
__global__ void matrixMulCUDA( float *A, float *B, float *C, int width ){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( row < width && col < width ){
        float sum = 0.0f;
        for( int i = 0 ; i < width ; i++ ){
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Function to initialize matrices with random values
void initializeMatrix( float *mat, int size ){
    for( int i = 0 ; i < size ; i++ ){
        mat[i] = static_cast<float>( rand() % 10 );
    }
}

// Function to print matrices
void printMatrix( float *mat, int width ){
    for( int i = 0 ; i < width ; i++ ){
        for( int j = 0 ; j < width ; j++ ){
            std::cout << mat[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(){
    int size = N * N * sizeof( float );
    
    // Host matrices
    float *h_A = ( float* )malloc( size );
    float *h_B = ( float* )malloc( size );
    float *h_C = ( float* )malloc( size );
    
    // Initialize input matrices
    initializeMatrix( h_A, N * N );
    initializeMatrix( h_B, N * N );
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc( ( void** )&d_A, size );
    cudaMalloc( ( void** )&d_B, size );
    cudaMalloc( ( void** )&d_C, size );
    
    // Copy matrices from host to device
    cudaMemcpy( d_A, h_A, size, cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, h_B, size, cudaMemcpyHostToDevice );
    
    // Define grid and block sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    // Launch CUDA kernel
    matrixMulCUDA<<<gridDim, blockDim>>>( d_A, d_B, d_C, N );
    
    // Copy result back to host
    cudaMemcpy( h_C, d_C, size, cudaMemcpyDeviceToHost );
    
    // Print results
    std::cout << "Matrix A:" << std::endl;
    printMatrix( h_A, N );
    
    std::cout << "Matrix B:" << std::endl;
    printMatrix( h_B, N );
    
    std::cout << "Result Matrix C (A x B):" << std::endl;
    printMatrix( h_C, N );
    
    // Free memory
    free( h_A );
    free( h_B );
    free( h_C );
    cudaFree( d_A );
    cudaFree( d_B );
    cudaFree( d_C );
    
    return 0;
}
