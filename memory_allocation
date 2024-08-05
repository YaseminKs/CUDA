#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

const int allocation_size = 2 * 1024 * 1024 * 1024 ;

void* cpu_p;
void* gpu_p;

void cpu_alloc(){
    cpu_p = malloc( allocation_size );
}

void gpu_alloc(){
    cudaError_t result = cudaMalloc( &gpu_p, allocation_size );
    printf( "The result is %d ", result );
}

int main(){
    cpu_alloc();
    gpu_alloc();

}
