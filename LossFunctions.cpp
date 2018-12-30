/** @file LossFunctions.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module LossFunctions
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

#include "LossFunctions.h"

LossFunctions::LossFunctions(){ }

LossFunctions::~LossFunctions(){ }

double LossFunctions::negLogLikelihood( const MatrixDoubleType& probs, const MatrixIntType& labels ){
	int row_size = probs.aRowSize;
	double loss = 0.0;
	int i;
	for( i = 0; i < row_size; i++ ){
		loss += -log( probs.aMatrix[i][labels.aMatrix[0][i]] );
	}
	return loss;
}

double LossFunctions::squareError( const MatrixDoubleType& probs, const MatrixIntType& labels ){
	int row_size = probs.aRowSize;
	int col_size = probs.aColSize;
	MatrixDoubleType mat(row_size, col_size);
	double loss = 0.0;
	int i, j;
	for( i = 0; i < row_size; i++ ){
		mat.aMatrix[i][labels.aMatrix[0][i]] = 1.0;
	}
	mat -= probs;
	mat *= mat;
	for( i = 0; i < row_size; i++ ){
		for( j = 0; j < col_size; j++ ){
			loss += mat.aMatrix[i][j];
		}
	}
	return loss;
}

double LossFunctions::regLoss( double lambda, const MatrixDoubleType& paramsW, const MatrixDoubleType& paramsU ){
	int row_size_w = paramsW.aRowSize;
	int col_size_w = paramsW.aColSize;
	int row_size_u = paramsU.aRowSize;
	int col_size_u = paramsU.aColSize;
	double reg_loss = 0.0;
	int i, j;
	for( i = 0; i < row_size_w; i++ ){
		for( j = 0; j < col_size_w; j++){
			reg_loss += paramsW.aMatrix[i][j] * paramsW.aMatrix[i][j];
		}
	}
	for( i = 0; i < row_size_u; i++ ){
		for( j = 0; j < col_size_u; j++){
			reg_loss += paramsU.aMatrix[i][j] * paramsU.aMatrix[i][j];
		}
	}
	reg_loss = 0.5 * lambda * reg_loss;
	return reg_loss;
}
