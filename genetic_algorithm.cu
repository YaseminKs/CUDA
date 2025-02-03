#include <stdio.h>
#include <cuda.h>
#include <curand_kernel.h>

#define POP_SIZE 1024
#define CHROMO_LENGTH 16
#define MUTATION_RATE 0.01

__device__ int fitness( int chromosome ){
    int count = 0;
    for( int i = 0 ; i < CHROMO_LENGTH ; i++ )
        if( chromosome & ( 1 << i ) ) count++;
    return count;  // Count the number of 1s (maximize ones)
}

__global__ void initPopulation( int *population, curandState *states ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init( 1234, tid, 0, &states[tid] );
    population[tid] = curand( &states[tid] ) % ( 1 << CHROMO_LENGTH );
}

__global__ void mutate( int *population, curandState *states ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( curand_uniform( &states[tid] ) < MUTATION_RATE )
        population[tid] ^= ( 1 << ( curand( &states[tid] ) % CHROMO_LENGTH ) );
}

__global__ void crossover( int *population, curandState *states ){
    int tid = threadIdx.x * 2;
    if( tid < POP_SIZE ){
        int crossover_point = curand( &states[tid] ) % CHROMO_LENGTH;
        int mask = ( 1 << crossover_point ) - 1;
        int temp = ( population[tid] & mask ) | ( population[tid + 1] & ~mask );
        population[tid + 1] = ( population[tid + 1] & mask ) | ( population[tid] & ~mask );
        population[tid] = temp;
    }
}

void cudaGeneticAlgorithm(){
    int *d_population;
    curandState *d_states;

    cudaMalloc( ( void** )&d_population, POP_SIZE * sizeof( int ) );
    cudaMalloc( ( void** )&d_states, POP_SIZE * sizeof( curandState ) );

    initPopulation<<<POP_SIZE / 256, 256>>>( d_population, d_states );
    for( int gen = 0 ; gen < 100 ; gen++ ){
        mutate<<<POP_SIZE / 256, 256>>>( d_population, d_states );
        crossover<<<POP_SIZE / 512, 256>>>( d_population, d_states );
    }

    cudaFree( d_population );
    cudaFree( d_states );
    printf( "Genetic Algorithm Completed\n" );
}

int main() {
    cudaGeneticAlgorithm();
    return 0;
}
