/** @file NeuronUnits.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module NeuronUnits
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

#include "NeuronUnits.h"

NeuronUnits::NeuronUnits(){ }

NeuronUnits::~NeuronUnits(){ }

MatrixDoubleType NeuronUnits::sigmoid( const MatrixDoubleType& m ){
	int row_size = m.aRowSize;
	int col_size = m.aColSize;
	MatrixDoubleType mat(row_size, col_size);
	int i, j;
	for( i = 0; i < row_size; i++ ){
		for( j = 0; j < col_size; j++ ){
			mat.aMatrix[i][j] = 1.0/( 1.0 + exp(-m.aMatrix[i][j]) );
		}
	}
	return mat;
}

MatrixDoubleType NeuronUnits::tanh( const MatrixDoubleType& m ){
	int row_size = m.aRowSize;
	int col_size = m.aColSize;
	MatrixDoubleType mat(row_size, col_size);
	int i, j;
	for( i = 0; i < row_size; i++ ){
		for( j = 0; j < col_size; j++ ){
			mat.aMatrix[i][j] = 2.0/( 1.0 + exp(-2.0*m.aMatrix[i][j]) ) - 1.0;
		}
	}
	return mat;
}

MatrixDoubleType NeuronUnits::relu( const MatrixDoubleType& m ){
	int row_size = m.aRowSize;
	int col_size = m.aColSize;
	MatrixDoubleType mat(row_size, col_size);
	int i, j;
	for( i = 0; i < row_size; i++ ){
		for( j = 0; j < col_size; j++ ){
			mat.aMatrix[i][j] = MAX(m.aMatrix[i][j], 0.0);
		}
	}
	return mat;
}
