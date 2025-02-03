#include <stdio.h>
#include <cuda.h>
#include <limits.h>

#define N 6
#define INF INT_MAX

__global__ void minDistanceKernel( int *d_dist, int *d_visited, int *d_minIndex ){
    int tid = threadIdx.x;
    __shared__ int minVal;
    __shared__ int minIdx;

    if( tid == 0 ){
        minVal = INF;
        minIdx = -1;
    }
    __syncthreads();

    if( d_visited[tid] == 0 && d_dist[tid] < minVal ){
        minVal = d_dist[tid];
        minIdx = tid;
    }
    __syncthreads();

    if( tid == 0 )
        *d_minIndex = minIdx;
}

__global__ void updateDistances( int *d_adj, int *d_dist, int *d_visited, int *d_minIndex ){
    int tid = threadIdx.x;
    int u = *d_minIndex;

    if( u != -1 && d_adj[u * N + tid] && d_dist[u] + d_adj[u * N + tid] < d_dist[tid] ){
        d_dist[tid] = d_dist[u] + d_adj[u * N + tid];
    }
}

void cudaDijkstra( int *adj, int start ){
    int *d_adj, *d_dist, *d_visited, *d_minIndex;
    int dist[N], visited[N];

    for( int i = 0 ; i < N ; i++ ){
        dist[i] = INF;
        visited[i] = 0;
    }
    dist[start] = 0;

    cudaMalloc( ( void** )&d_adj, N * N * sizeof( int ) );
    cudaMalloc( ( void** )&d_dist, N * sizeof( int ) );
    cudaMalloc( ( void** )&d_visited, N * sizeof( int ) );
    cudaMalloc( ( void** )&d_minIndex, sizeof( int ) );

    cudaMemcpy( d_adj, adj, N * N * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_dist, dist, N * sizeof( int ), cudaMemcpyHostToDevice );
    cudaMemcpy( d_visited, visited, N * sizeof( int ), cudaMemcpyHostToDevice );

    for( int i = 0 ; i < N - 1 ; i++ ){
        minDistanceKernel<<<1, N>>>( d_dist, d_visited, d_minIndex );
        cudaDeviceSynchronize();
        updateDistances<<<1, N>>>( d_adj, d_dist, d_visited, d_minIndex );
        cudaDeviceSynchronize();
    }

    cudaMemcpy( dist, d_dist, N * sizeof( int ), cudaMemcpyDeviceToHost );
    cudaFree( d_adj );
    cudaFree( d_dist );
    cudaFree( d_visited );
    cudaFree( d_minIndex );

    printf( "Dijkstra Distances:\n" );
    for( int i = 0 ; i < N ; i++ )
        printf( "%d ", dist[i] );
    printf( "\n" );
}

int main() {
    int adj[N][N] = { { 0, 10, 20, 0, 0, 0 },
                     { 10, 0, 5, 1, 0, 0 },
                     { 20, 5, 0, 8, 2, 0 },
                     { 0, 1, 8, 0, 6, 3 },
                     { 0, 0, 2, 6, 0, 7 },
                     { 0, 0, 0, 3, 7, 0 } };

    cudaDijkstra( ( int* )adj, 0 );
    return 0;
}
