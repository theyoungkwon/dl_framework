/** @file MathModule.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module MathModule
 *  @version 1.0
 *  @brief gather common setup includes and functions.
 *
 *  Language: C++
 *
 *  This contains the prototypes for the console
 *  driver and eventually any macros, constants,
 *  or global variables you will need.
 *
 *  License: GNU Public License
 *
 *  @author Youngdae Kwon (young231)
 *  E-Mail: kydchonje@gmail.com
 *
 *  @bug No known bugs.
 *
 *  Database Tables Used: No
 *
 *  Thread Safe: No
 *
 *  Extendable: No
 *
 *  Platform Dependencies: None
 *
 *  Compiler Options: -std=c++11
 */

#include "MathModule.h"

MathModule::MathModule(){ }

MathModule::~MathModule(){ }


MatrixDoubleType MathModule::makeOneHotVec( int rLabel , int rDim){
	MatrixDoubleType t(rDim, 1);
    t.aMatrix[rLabel][0] = 1.0;
    return t;
}

int MathModule::sampling( char ch ){

}

MatrixDoubleType MathModule::mysoftmax( MatrixDoubleType& rZ2 ){
	MatrixDoubleType probs(rZ2.aRowSize, rZ2.aColSize);
	MatrixDoubleType temp_matrix(rZ2.aRowSize, 1);
	int i , j;
	double temp_double = 0.0;
	// get max
	for( i = 0; i < probs.aRowSize; i++ ){
		for( j = 0; j < probs.aColSize ; j++ ){
			if (  temp_double < rZ2.aMatrix[i][j] ) {
				temp_double = rZ2.aMatrix[i][j];
			}
		}
		temp_matrix.aMatrix[i][0] = temp_double;
		temp_double = 0.0;
	}
	probs = rZ2 - temp_matrix;
	for ( i = 0 ; i < probs.aRowSize; i++ ){
		for( j = 0 ; j < probs.aColSize; j++ ){
			probs.aMatrix[i][j] = exp( probs.aMatrix[i][j] );
		}
	}
	for( i = 0; i < probs.aRowSize; i++ ){
		// get sum
		for( j = 0; j < probs.aColSize ; j++ ){
			temp_double += probs.aMatrix[i][j];
		}
		// divide
		for( j = 0; j < probs.aColSize ; j++ ){
			probs.aMatrix[i][j] = probs.aMatrix[i][j] / temp_double;
		}
		temp_double = 0.0;
	}

	return probs;
}


MatrixIntType MathModule::argmax( MatrixDoubleType& rProbs, int rAxis = 1 ){
	MatrixIntType mat;
	int i , j, idx;
	double temp_max = 0.0;
	if ( rAxis == 0 ){
		mat.init( 1, rProbs.aColSize );
		for( j = 0; j < rProbs.aColSize; j++ ){
			for( i = 0; i < rProbs.aRowSize; i++ ){
				if ( temp_max < rProbs.aMatrix[i][j] ){
					idx = i;
					temp_max = rProbs.aMatrix[i][j];
				}
			}
			mat.aMatrix[0][j] = idx;
			temp_max = 0.0;
		}
	} // column-wise calculation
	else {
		mat.init( rProbs.aRowSize, 1 );
		for( i = 0; i < rProbs.aRowSize; i++ ){
			for( j = 0; j < rProbs.aColSize; j++ ){
				if ( temp_max < rProbs.aMatrix[i][j] ){
					idx = j;
					temp_max = rProbs.aMatrix[i][j];
				}
			}
			mat.aMatrix[i][0] = idx;
			temp_max = 0.0;
		}
	} // row-wise calculation

	return mat;
}

double MathModule::sum( const MatrixDoubleType& rParams ){
	int i, j;
	double total_sum = 0.0;

	for( i = 0; i < rParams.aRowSize; i++ ){
		for( j = 0 ; j < rParams.aColSize; j++ ){
			total_sum += rParams.aMatrix[i][j];
		}
	}

	return total_sum;
}

double MathModule::length( const MatrixDoubleType& rParams ) const{
	double length = 0.0;
	int i, j;
	for( i = 0; i < rParams.aRowSize; i++ ){
		for( j = 0; j < rParams.aColSize; j++ ){
			length += rParams.aMatrix[i][j]*rParams.aMatrix[i][j];
		}
	}

	return sqrt(length);
}

double MathModule::mysecond(void) {
        struct timeval tp;
        struct timezone tzp;
        int i;

        i = gettimeofday(&tp,&tzp);
        return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}