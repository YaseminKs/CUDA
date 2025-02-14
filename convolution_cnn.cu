// Convolutional operations are fundamental in many areas, such as image processing, 
// signal processing, and deep learning (e.g., Convolutional Neural Networks - CNNs). 
// They involve applying a kernel (or filter) to an input matrix (like an image) 
// to produce a transformed output.

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 16

// CUDA kernel for 2D convolution
__global__ void convolution2D( float* input, float* output, float* kernel, int width, int height, int kernel_size ){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_k = kernel_size / 2;

    if( x < width && y < height ){
        float sum = 0.0f;
        for( int i = -half_k ; i <= half_k ; i++ ){
            for( int j = -half_k ; j <= half_k ; j++ ){
                int img_x = min( max( x + i, 0 ), width - 1 );
                int img_y = min( max( y + j, 0 ), height - 1 );
                float pixel = input[img_y * width + img_x];
                float coeff = kernel[( i + half_k ) * kernel_size + ( j + half_k )];
                sum += pixel * coeff;
            }
        }
        output[y * width + x] = sum;
    }
}

// Generate a Gaussian kernel
void generateGaussianKernel( float* kernel, int size, float sigma = 1.0f ){
    int center = size / 2;
    float sum = 0.0f;
    for( int i = 0 ; i < size ; i++ ){
        for( int j = 0 ; j < size ; j++ ){
            int x = i - center;
            int y = j - center;
            kernel[i * size + j] = exp( -( x * x + y * y )/( 2 * sigma * sigma ) );
            sum += kernel[i * size + j];
        }
    }
    for( int i = 0 ; i < size * size ; i++ ){
        kernel[i] /= sum;
    }
}

int main(){
    int width = 1024;
    int height = 1024;
    int kernel_size = 3;

    // Allocate host memory
    std::vector<float> h_input( width * height, 1.0f );  // Example: uniform image
    std::vector<float> h_output( width * height, 0.0f );
    std::vector<float> h_kernel( kernel_size * kernel_size );

    // Generate Gaussian kernel
    generateGaussianKernel( h_kernel.data(), kernel_size );

    // Allocate device memory
    float *d_input, *d_output, *d_kernel;
    cudaMalloc( &d_input, width * height * sizeof( float ) );
    cudaMalloc( &d_output, width * height * sizeof( float ) );
    cudaMalloc( &d_kernel, kernel_size * kernel_size * sizeof( float ) );

    // Copy data to device
    cudaMemcpy( d_input, h_input.data(), width * height * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_kernel, h_kernel.data(), kernel_size * kernel_size * sizeof( float ), cudaMemcpyHostToDevice );

    // Define grid and block dimensions
    dim3 blockSize( BLOCK_SIZE, BLOCK_SIZE );
    dim3 gridSize( ( width + BLOCK_SIZE - 1 ) / BLOCK_SIZE, ( height + BLOCK_SIZE - 1 ) / BLOCK_SIZE );

    // Launch convolution kernel
    convolution2D<<<gridSize, blockSize>>>( d_input, d_output, d_kernel, width, height, kernel_size );
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy( h_output.data(), d_output, width * height * sizeof( float ), cudaMemcpyDeviceToHost );

    // Clean up
    cudaFree( d_input );
    cudaFree( d_output );
    cudaFree( d_kernel );

    // Print part of the output
    std::cout << "Sample output: ";
    for( int i = 0 ; i < 10 ; i++ ){
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
