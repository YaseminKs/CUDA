#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define INPUT_NODES 3
#define HIDDEN_NODES 4
#define OUTPUT_NODES 2
#define LEARNING_RATE 0.1

__device__ float sigmoid( float x ){
    return 1.0 / ( 1.0 + expf( -x ) );
}

__global__ void forwardPass( float *inputs, float *weightsIH, float *hidden, float *weightsHO, float *outputs ){
    int tid = threadIdx.x;

    if( tid < HIDDEN_NODES ){
        hidden[tid] = 0;
        for( int i = 0 ; i < INPUT_NODES ; i++ )
            hidden[tid] += inputs[i] * weightsIH[i * HIDDEN_NODES + tid];
        hidden[tid] = sigmoid(hidden[tid] );
    }

    __syncthreads();  

    if( tid < OUTPUT_NODES ){
        outputs[tid] = 0;
        for( int i = 0 ; i < HIDDEN_NODES ; i++ )
            outputs[tid] += hidden[i] * weightsHO[i * OUTPUT_NODES + tid];
        outputs[tid] = sigmoid( outputs[tid] );
    }
}

__global__ void backwardPass( float *inputs, float *hidden, float *outputs, float *weightsIH, float *weightsHO, float *targets ){
    int tid = threadIdx.x;

    __shared__ float outputDeltas[OUTPUT_NODES];
    if( tid < OUTPUT_NODES )
        outputDeltas[tid] = ( targets[tid] - outputs[tid] ) * outputs[tid] * ( 1 - outputs[tid] );

    __syncthreads();

    if( tid < HIDDEN_NODES ){
        float hiddenDelta = 0;
        for( int j = 0 ; j < OUTPUT_NODES ; j++ )
            hiddenDelta += outputDeltas[j] * weightsHO[tid * OUTPUT_NODES + j];

        hiddenDelta *= hidden[tid] * (1 - hidden[tid]);

        for( int i = 0 ; i < INPUT_NODES ; i++ )
            weightsIH[i * HIDDEN_NODES + tid] += LEARNING_RATE * hiddenDelta * inputs[i];
    }

    __syncthreads();

    if( tid < OUTPUT_NODES ){
        for( int i = 0 ; i < HIDDEN_NODES ; i++ )
            weightsHO[i * OUTPUT_NODES + tid] += LEARNING_RATE * outputDeltas[tid] * hidden[i];
    }
}

void cudaBackpropagation(){
    float h_inputs[INPUT_NODES] = { 1.0, 0.5, -1.5 };
    float h_targets[OUTPUT_NODES] = { 0.1, 0.9 };
    
    float *d_inputs, *d_hidden, *d_outputs, *d_targets, *d_weightsIH, *d_weightsHO;
    cudaMalloc( ( void** )&d_inputs, INPUT_NODES * sizeof( float ) );
    cudaMalloc( ( void** )&d_hidden, HIDDEN_NODES * sizeof( float ) );
    cudaMalloc( ( void** )&d_outputs, OUTPUT_NODES * sizeof( float ) );
    cudaMalloc( ( void** )&d_targets, OUTPUT_NODES * sizeof( float ) );
    cudaMalloc( ( void** )&d_weightsIH, INPUT_NODES * HIDDEN_NODES * sizeof( float ) );
    cudaMalloc( ( void** )&d_weightsHO, HIDDEN_NODES * OUTPUT_NODES * sizeof( float ) );

    cudaMemcpy( d_inputs, h_inputs, INPUT_NODES * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_targets, h_targets, OUTPUT_NODES * sizeof( float ), cudaMemcpyHostToDevice );

    for( int epoch = 0 ; epoch < 1000 ; epoch++ ){
        forwardPass<<<1, max( HIDDEN_NODES, OUTPUT_NODES )>>>( d_inputs, d_weightsIH, d_hidden, d_weightsHO, d_outputs );
        cudaDeviceSynchronize();
        backwardPass<<<1, max( HIDDEN_NODES, OUTPUT_NODES )>>>( d_inputs, d_hidden, d_outputs, d_weightsIH, d_weightsHO, d_targets );
        cudaDeviceSynchronize();
    }

    cudaFree( d_inputs );
    cudaFree( d_hidden );
    cudaFree( d_outputs );
    cudaFree( d_targets );
    cudaFree( d_weightsIH );
    cudaFree( d_weightsHO );
    printf( "Backpropagation Completed\n" );
}

int main(){
    cudaBackpropagation();
    return 0;
}
