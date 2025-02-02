#include<stdio.h>
#include<stdlib.h>

#define N 10  // Number of nodes in the linked list

// Define a node structure
struct Node{
    int data;
    struct Node* next;
};

// CUDA kernel to process the linked list elements in parallel
__global__ void process_list( int* d_data, int size ){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if( idx < size){
        d_data[ idx ] *= 2;  // Example: Multiply each element by 2
    }
}

// Function to create a new node
Node* create_node( int data ){
    Node* new_node = ( Node* )malloc( sizeof( Node ) );
    new_node -> data = data;
    new_node -> next = NULL;
    return new_node;
}

// Function to append a node to the end of the list
void append( Node** head, int data ){
    Node* new_node = create_node( data );
    if( *head == NULL ){
        *head = new_node;
        return;
    }
    Node* temp = *head;
    while( temp -> next != NULL ){
        temp = temp -> next;
    }
    temp -> next = new_node;
}

// Function to convert linked list to array
void list_to_array( Node* head, int* arr, int size ){
    Node* temp = head;
    for( int i = 0 ; i < size && temp != NULL ; i++ ){
        arr[ i ] = temp -> data;
        temp = temp -> next;
    }
}

// Function to update the linked list from an array
void array_to_list( Node* head, int* arr, int size ){
    Node* temp = head;
    for( int i = 0 ; i < size && temp != NULL ; i++ ){
        temp -> data = arr[ i ];
        temp = temp -> next;
    }
}

// Function to print the linked list
void print_list( Node* head ){
    Node* temp = head;
    while( temp != NULL ){
        printf( "%d -> ", temp -> data );
        temp = temp -> next;
    }
    printf( "NULL\n" );
}

// Function to free the linked list
void free_list( Node* head ){
    Node* temp;
    while( head != NULL ){
        temp = head;
        head = head -> next;
        free( temp );
    }
}

int main(){
    Node* head = NULL;

    // Create a linked list with values from 1 to N
    for( int i = 1 ; i <= N ; i++ ){
        append( &head, i );
    }

    printf( "Original List:\n" );
    print_list( head );

    // Convert linked list to an array
    int h_data[ N ];
    list_to_array( head, h_data, N );

    // Allocate memory on the GPU
    int* d_data;
    cudaMalloc( ( void** )&d_data, N * sizeof( int ) );
    cudaMemcpy( d_data, h_data, N * sizeof( int ), cudaMemcpyHostToDevice );

    // Launch CUDA kernel with enough threads
    int blockSize = 256;
    int gridSize = ( N + blockSize - 1 )/blockSize;
    process_list<<< gridSize, blockSize >>>( d_data, N );
    cudaDeviceSynchronize();

    // Copy results back to the host
    cudaMemcpy( h_data, d_data, N * sizeof( int ), cudaMemcpyDeviceToHost );

    // Update the linked list with processed data
    array_to_list( head, h_data, N );

    printf( "Processed List (Each value multiplied by 2):\n" );
    print_list( head );

    // Free allocated memory
    cudaFree( d_data );
    free_list( head );

    return 0;
}
