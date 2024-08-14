#include<cuda.h>
#include<cuda_device_runtime_api.h>
#include<cublas.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <sys/utime.h>

using namespace std::chrono;
using namespace std;

struct Mat{
	Mat() : data( NULL ), w( 0 ), h( 0 ){}
	~Mat() { if( data ) delete[] data; }
	float *data;
	int w;
	int h;
};

inline static float getValue( Mat *mat, int x, int y ){
	if( x > mat -> w || y > mat -> h ){
		throw runtime_error( "invalid access" );
	}
	return mat -> data[ y * mat -> w + x ];
}

inline static void setValue( Mat *mat, int x, int y, float val ){
	if( x > mat -> w || y > mat -> h ){
		throw runtime_error( "invalid access" );
	}
	mat -> data[ y * mat -> w + x ] = val;
}

static void initMat( Mat *mat, int height, int width ){
	//std::cout << "make matrix ( w/h ): " << width << "/" << height << std::endl;
	mat -> data = new float[ height * width ];
	mat -> w = width;
	mat -> h = height;
	for( int i = 0 ; i < mat -> w ; i++ ){
		for( int j = 0 ; j < mat -> h ; j++ ){
			setValue( mat, i, j, 0.0f );
		}
	}
}

static void printMat( Mat &mat, bool force = false ){
	std::cout << "Dim: " << mat.h << ", " << mat.w << "\n";
	if( ( mat.w < 10 && mat.h < 10 ) || force )	{
		for( int j = 0 ; j < mat.h ; j++ ){
			for( int i = 0 ; i < mat.w ; i++ ){
				std::cout << getValue( &mat, i, j ) << "\t";
			}
			std::cout << "\n";
		}
	}
	std::cout << std::endl;
}

static bool read_csv( string file, Mat *xs, Mat *ys ){
	ifstream s( file );
	if( !s.is_open() ){
		throw runtime_error( file + " doesn't exist" );
	}
	int rows = 0;
	int cols = 0;
	string line;
	while( getline( s, line ) ){
		// if we read first line, check how many columns
		if( rows++ == 0 ){
			stringstream ss( line );
			while( ss.good() ){
				string substr;
				getline( ss, substr, ',' );
				cols++;
			}
		}
	}
	std::cout << "found " << rows << " rows with " << cols << " columns." << std::endl;
	s.clear() ;
	s.seekg( 0, ios::beg );

	initMat( xs, rows - 1, cols - 2 );
	initMat( ys, rows - 1, 1 );

	// go to second line
	getline( s, line );
	int y = 0;
	while( getline( s, line ) ){
		stringstream ss( line );
		int x = 0;
		while( ss.good() ){
			string substr;
			getline( ss, substr, ',' );

			// first column is uninteresting
			// second column is target values
			if( x == 1 ){
				float val = atof( substr.c_str() );
				setValue( ys, 0, y, val );
			}else if( x > 1 ){
				float val = atof( substr.c_str() );
				setValue( xs, ( x - 2 ), y, val );
			}
			x++;
		}
		y++;
	}

	return true;
}

int main( int argc, char **argv ){
	float time;
	cudaEvent_t start, stop;

	HANDLE_ERROR( cudaEventCreate(&start) );
	HANDLE_ERROR( cudaEventCreate(&stop) );
	HANDLE_ERROR( cudaEventRecord(start, 0) );

	string csv_file("./deneme2017.csv");

	Mat xs;
	Mat ys;
	Mat params;
	Mat trainedParams;

	read_csv(csv_file, &xs, &ys);

	printMat(xs, true);
	printMat(ys);

	HANDLE_ERROR( cudaEventRecord(stop, 0) );
	HANDLE_ERROR( cudaEventSynchronize(stop) );
	HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );

	printf("Time to generate:  %3.1f ms \n", time);
	
	int ch = getchar();

	return 0;
}
