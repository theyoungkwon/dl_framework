/** @file setup.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module setup
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

#include "setup.h"

MatrixDoubleType::MatrixDoubleType(){}

MatrixDoubleType::MatrixDoubleType(int n){
	aMatrix.resize(1);
	aMatrix[0].resize(n);
	aRowSize = 1;
	aColSize = n;
}

MatrixDoubleType::MatrixDoubleType(int m, int n){
	aMatrix.resize(m);
	for (int i = 0; i<m; i++){
		aMatrix[i].resize(n);
	}
	aRowSize = m;
	aColSize = n;
}

MatrixDoubleType::MatrixDoubleType( const MatrixDoubleType& m ){
	aMatrix.resize(m.aRowSize);
	for (int i = 0; i<m.aRowSize; i++){
		aMatrix[i].resize(m.aColSize);
	}
	aRowSize = m.aRowSize;
	aColSize = m.aColSize;
}

MatrixDoubleType::~MatrixDoubleType(){}

MatrixDoubleType MatrixDoubleType::dot( const MatrixDoubleType& m ) const {
	int row_size = aMatrix.size();
	int col_size = aMatrix[0].size();
	int row_size_m = m.aMatrix.size();
	int col_size_m = m.aMatrix[0].size();
	MatrixDoubleType mat(row_size, col_size_m);
	int i, j, k;
	if ( col_size != row_size_m ){
		printf("matrix row - column size different \n");
	}
	for( i = 0; i < row_size; i++ ){
		for( j = 0; j < col_size_m; j++ ){
			for ( k = 0; k < col_size; k++ ){
				mat.aMatrix[i][j] += aMatrix[i][k] * m.aMatrix[k][j];
			}
		}
	}
	return mat;
}

MatrixDoubleType MatrixDoubleType::transpose() const {
	MatrixDoubleType mat(aColSize, aRowSize);
	int i, j;
	for( i = 0; i < aRowSize; i++){
		for( j = 0; j < aColSize; j++){
			mat.aMatrix[j][i] = aMatrix[i][j];
		}
	}
	return mat;
}

double MatrixDoubleType::length() const{
	double length = 0.0;
	int i, j;
	for( i = 0; i < aRowSize; i++ ){
		for( j = 0; j < aColSize; j++ ){
			length += aMatrix[i][j]*aMatrix[i][j];
		}
	}

	return sqrt(length);
}

void MatrixDoubleType::display() const{
	int i, j;
	for( i = 0; i < aRowSize; i++){
		for( j = 0; j < aColSize; j++){
			printf("%g ", aMatrix[i][j]);
		}
		printf("\n");
	}
}

void MatrixDoubleType::shape() const{
	printf("(%d, %d)\n", aRowSize, aColSize);
}

void MatrixDoubleType::reset() {
	int i;
	for( i = 0; i < aRowSize; i++ ){
		aMatrix[i].clear();
		aMatrix[i].resize(aColSize);
	}
}

void MatrixDoubleType::insert1dTo1d( MatrixDoubleType& m, int rIdx ){
	int i;
	if ( aColSize > aRowSize ){
		if ( m.aColSize > m.aRowSize ){
			for( i = 0; i < m.aColSize; i++ ){
				aMatrix[0][i+rIdx] = m.aMatrix[0][i];
			}
		} // col <= col
		else {
			for( i = 0; i < m.aRowSize; i++ ){
				aMatrix[0][i+rIdx] = m.aMatrix[i][0];
			}
		} // col <= row
	}
	else {
		if ( m.aColSize > m.aRowSize ){
			for( i = 0; i < m.aColSize; i++ ){
				aMatrix[i+rIdx][0] = m.aMatrix[0][i];
			}
		} // row <= col
		else {
			for( i = 0; i < m.aRowSize; i++ ){
				aMatrix[i+rIdx][0] = m.aMatrix[i][0];
			}
		} // row <= row
	}
}

void MatrixDoubleType::insert1dTo2d( MatrixDoubleType& m, int rIdx , int rColIdx){
	int i;
	for( i = 0; i < m.aColSize; i++ ){
		aMatrix[i+rIdx][rColIdx] = m.aMatrix[0][i];
	}
}

MatrixIntType::MatrixIntType(){}

MatrixIntType::MatrixIntType(int n){
	aMatrix.resize(1);
	aMatrix[0].resize(n);
	aRowSize = 1;
	aColSize = n;
}

MatrixIntType::MatrixIntType(int m, int n){
	aMatrix.resize(m);
	for (int i = 0; i<m; i++){
		aMatrix[i].resize(n);
	}
	aRowSize = m;
	aColSize = n;
}

MatrixIntType::MatrixIntType( const MatrixIntType& m ){
	aMatrix.resize(m.aRowSize);
	for (int i = 0; i<m.aRowSize; i++){
		aMatrix[i].resize(m.aColSize);
	}
	aRowSize = m.aRowSize;
	aColSize = m.aColSize;
}

MatrixIntType::~MatrixIntType(){}

MatrixIntType MatrixIntType::dot( const MatrixIntType& m ) const {
	int row_size = aMatrix.size();
	int col_size = aMatrix[0].size();
	int row_size_m = m.aMatrix.size();
	int col_size_m = m.aMatrix[0].size();
	MatrixIntType mat(row_size, col_size_m);
	int i, j, k;
	if ( col_size != row_size_m ){
		printf("matrix row - column size different \n");
	}
	for( i = 0; i < row_size; i++ ){
		for( j = 0; j < col_size_m; j++ ){
			for ( k = 0; k < col_size; k++ ){
				mat.aMatrix[i][j] += aMatrix[i][k] * m.aMatrix[k][j];
			}
		}
	}
	return mat;
}

MatrixIntType MatrixIntType::transpose() const {
	MatrixIntType mat(aColSize, aRowSize);
	int i, j;
	for( i = 0; i < aRowSize; i++){
		for( j = 0; j < aColSize; j++){
			mat.aMatrix[j][i] = aMatrix[i][j];
		}
	}
	return mat;
}

double MatrixIntType::length() const{
	int length = 0;
	int i, j;
	for( i = 0; i < aRowSize; i++ ){
		for( j = 0; j < aColSize; j++ ){
			length += aMatrix[i][j]*aMatrix[i][j];
		}
	}

	return sqrt((double)length);
}

void MatrixIntType::display() const{
	int i, j;
	for( i = 0; i < aRowSize; i++){
		for( j = 0; j < aColSize; j++){
			printf("%d ", aMatrix[i][j]);
		}
		printf("\n");
	}
}

void MatrixIntType::shape() const{
	printf("(%d, %d)\n", aRowSize, aColSize);
}

void MatrixIntType::reset() {
	int i;
	for( i = 0; i < aRowSize; i++ ){
		aMatrix[i].clear();
		aMatrix[i].resize(aColSize);
	}
}

void MatrixIntType::insert1dTo1d( MatrixIntType& m, int idx ){
	int i;
	if ( aColSize > aRowSize ){
		if ( m.aColSize > m.aRowSize ){
			for( i = 0; i < m.aColSize; i++ ){
				aMatrix[0][i+idx] = m.aMatrix[0][i];
			}
		} // col <= col
		else {
			for( i = 0; i < m.aRowSize; i++ ){
				aMatrix[0][i+idx] = m.aMatrix[i][0];
			}
		} // col <= row
	}
	else {
		if ( m.aColSize > m.aRowSize ){
			for( i = 0; i < m.aColSize; i++ ){
				aMatrix[i+idx][0] = m.aMatrix[0][i];
			}
		} // row <= col
		else {
			for( i = 0; i < m.aRowSize; i++ ){
				aMatrix[i+idx][0] = m.aMatrix[i][0];
			}
		} // row <= row
	}
}

void MatrixIntType::insert1dTo2d( MatrixIntType& m, int rIdx , int rColIdx){
	int i;
	for( i = 0; i < m.aColSize; i++ ){
		aMatrix[i+rIdx][rColIdx] = m.aMatrix[0][i];
	}
}