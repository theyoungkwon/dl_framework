/** @file RandomModule.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module RandomModule
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

#include "RandomModule.h"

RandomModule::RandomModule(){ }

RandomModule::~RandomModule(){ }

void RandomModule::xavierDoubleInit( MatrixDoubleType& m ){

	random_device rd;
    mt19937 gen(rd());
    // mt19937 gen(10);
    uniform_real_distribution<double> dis(0, 1);
    int i, j;
	double epsilon = sqrt(6) / sqrt( m.aRowSize + m.aColSize );
	for( i = 0; i < m.aRowSize; i++ ){
		for( j = 0; j < m.aColSize; j++ ) {
			m.aMatrix[i][j] = (2.0*epsilon)*dis(gen) - epsilon;
		}
	}
}

MatrixDoubleType RandomModule::xavierDoubleInit( int rRowSize, int rColSize ){

	random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(0, 1);
    int i, j;
	MatrixDoubleType mat(rRowSize, rColSize);
	double epsilon = sqrt(6) / sqrt( rRowSize + rColSize );
	for( i = 0; i < rRowSize; i++ ){
		for( j = 0; j < rColSize; j++ ) {
			mat.aMatrix[i][j] = (2.0*epsilon)*dis(gen) - epsilon;
		}
	}

	return mat;
}

void RandomModule::xavierIntInit( MatrixIntType& m ){ }

MatrixIntType RandomModule::xavierIntInit( int rRowSize, int rColSize ){ }

void RandomModule::randNormalDoubleInit( MatrixDoubleType& m ){
	random_device rd;
    mt19937 gen(rd());
 	int i , j;
    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    normal_distribution<double> d(0,1);

    for( i = 0; i < m.aRowSize; i++ ){
		for( j = 0; j < m.aColSize; j++ ) {
			m.aMatrix[i][j] = d(gen);
		}
	}
}

MatrixDoubleType RandomModule::randNormalDoubleInit( int rRowSize, int rColSize ){

	random_device rd;
    mt19937 gen(rd());
 	int i , j;
	MatrixDoubleType mat(rRowSize, rColSize);
    // values near the mean are the most likely
    // standard deviation affects the dispersion of generated values from the mean
    normal_distribution<double> d(0,1);

    for( i = 0; i < rRowSize; i++ ){
		for( j = 0; j < rColSize; j++ ) {
			mat.aMatrix[i][j] = d(gen);
		}
	}

	return mat;
}

void RandomModule::randNormalIntInit( MatrixIntType& m ){}

MatrixIntType RandomModule::randNormalIntInit( int rRowSize, int rColSize ){}
