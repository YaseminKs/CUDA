#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define N 6
#define INF 9999

__global__ void bfsKernel( int *d_adj, int *d_dist, int *d_frontier, int n ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if( tid < n && d_frontier[tid] ){
        d_frontier[tid] = 0;
        for( int i = 0 ; i < n ; i++ ){
            if( d_adj[tid * n + i] == 1 && d_dist[i] == INF ){
                d_dist[i] = d_dist[tid] + 1;
                d_frontier[i] = 1;
            }
        }
    }
}

void cudaBFS( int *adj, int start ){
    int *d_adj, *d_dist, *d_frontier;
    int dist[N], frontier[N];

    for( int i = 0 ; i < N ; i++ ){
        dist[i] = INF;
        frontier[i] = 0;
    }
    dist[start] = 0;
    frontier[start] = 1;

    cudaMalloc( ( void** )&d_adj, N * N * sizeof( int ) );
    cudaMalloc( ( void** )&d_dist, N * sizeof( int ) );
    cudaMalloc( ( void** )&d_frontier, N * sizeof( int ) );

    cudaMemcpy( d_adj, adj, N * N * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_dist, dist, N * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_frontier, frontier, N * sizeof( int ), cudaMemcpyHostToDevice );

    for( int level = 0; level < N ; level++ ){
        bfsKernel<<<1, N>>>( d_adj, d_dist, d_frontier, N );
        cudaDeviceSynchronize();
    }

    cudaMemcpy( dist, d_dist, N * sizeof( int ), cudaMemcpyDeviceToHost );
    cudaFree( d_adj );
    cudaFree( d_dist );
    cudaFree( d_frontier );

    printf( "BFS Distances:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", dist[i] );
    printf( "\n" );
}

int main(){
    int adj[N][N] = { { 0, 1, 1, 0, 0, 0 },
                     { 1, 0, 0, 1, 0, 0 },
                     { 1, 0, 0, 1, 1, 0 },
                     { 0, 1, 1, 0, 1, 1 },
                     { 0, 0, 1, 1, 0, 1 },
                     { 0, 0, 0, 1, 1, 0 } };

    cudaBFS( ( int* )adj, 0 );
    return 0;
}
