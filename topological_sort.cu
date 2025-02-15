// CUDA implementation of Topological Sort

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void topologicalSortKernel( int *adj, int *inDegree, int *result, int V ){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx < V && inDegree[idx] == 0 ){
        result[idx] = idx;
        inDegree[idx] = -1;
        for( int i = 0 ; i < V ; i++ ){
            if( adj[idx * V + i] ){
                atomicSub( &inDegree[i], 1 );
            }
        }
    }
}

int main(){
    const int V = 6;
    int adj[V][V] = { { 0, 0, 0, 0, 0, 0 },
                     { 0, 0, 0, 0, 0, 0 },
                     { 0, 0, 0, 1, 0, 0 },
                     { 0, 0, 0, 0, 0, 0 },
                     { 0, 1, 0, 0, 0, 0 },
                     { 0, 0, 1, 0, 0, 0 } };

    int inDegree[V] = { 0, 0, 0, 0, 0, 0 };
    for( int i = 0 ; i < V ; i++ )
        for( int j = 0 ; j < V ; j++ )
            inDegree[j] += adj[i][j];

    int *d_adj, *d_inDegree, *d_result;
    cudaMalloc( &d_adj, V * V * sizeof( int ) );
    cudaMalloc( &d_inDegree, V * sizeof( int) );
    cudaMalloc( &d_result, V * sizeof( int ) );

    cudaMemcpy( d_adj, adj, V * V * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_inDegree, inDegree, V * sizeof( int ), cudaMemcpyHostToDevice );

    topologicalSortKernel<<<1, V>>>( d_adj, d_inDegree, d_result, V );

    int result[V];
    cudaMemcpy( result, d_result, V * sizeof( int ), cudaMemcpyDeviceToHost );

    printf( "Topological Sort:\n" );
    for( int i = 0 ; i < V ; i++ )
        printf( "%d ", result[i] );

    cudaFree( d_adj );
    cudaFree( d_inDegree );
    cudaFree( d_result );
    
  return 0;
}
