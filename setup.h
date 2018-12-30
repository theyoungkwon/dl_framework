/** @file setup.h
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

#ifndef setup_h
#define setup_h

// C standard
#include <stdio.h>
#include <stdlib.h>		// rand, srand
#include <string.h>		// strcpy, strlen, strcat
#include <time.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>

// C++ Library
#include <iostream>		// std::cout
#include <sstream>		// std::stringstream
#include <random>
#include <fstream>		// std::open, std::close
// #include <ctime>        // std::time
// #include <cstdlib>      // std::rand, std::srand
// #include <cstring> 		// strcpy, strlen, strcat

// STL
#include <algorithm>    // std::random_shuffle
#include <array> 		// std::array
#include <map>			// std::map
#include <set>
#include <string> 		// string concatenation
#include <vector>       // std::vector
using namespace std;

// common macros
#ifndef MAX
	#define MAX(a,b) ((a)>(b)?(a):(b))
#endif
#ifndef MIN
	#define MIN(a,b) ((a)<(b)?(a):(b))
#endif

struct WordLabelType
{
	string aWord;
	string aLabel;
};

class MatrixDoubleType
{
public:

	vector< vector<double> > aMatrix;
	int aRowSize;
	int aColSize;
	/** @brief Constructor of the MatrixDoubleType class.
	 *
	 *  Default constructor method of the MatrixDoubleType class.
	 *  If you want to use the method declared in the MatrixDoubleType
	 *  class, you should first initialize the MatrixDoubleType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	MatrixDoubleType();

	/** @brief Constructor of 1 Dimensional Matrix class.
	 *
	 *  Default constructor method of the MatrixDoubleType class.
	 *  If you want to use the method declared in the MatrixDoubleType
	 *  class, you should first initialize the MatrixDoubleType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param n the number of dimension
	 */
	MatrixDoubleType(int n);

	/** @brief Constructor of 2 Dimensional Matrix class.
	 *
	 *  Default constructor method of the MatrixDoubleType class.
	 *  If you want to use the method declared in the MatrixDoubleType
	 *  class, you should first initialize the MatrixDoubleType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param m the number of first dimension
	 *  @param n the number of second dimension
	 */
	MatrixDoubleType(int m, int n);

	/** @brief Constructor of 2 Dimensional Matrix class.
	 *
	 *  Default constructor method of the MatrixDoubleType class.
	 *  If you want to use the method declared in the MatrixDoubleType
	 *  class, you should first initialize the MatrixDoubleType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param m matrix with the same dimension
	 */
	MatrixDoubleType( const MatrixDoubleType& m );

	/** @brief Destructor of the MatrixDoubleType class.
	 *
	 *  Default destructor method of the MatrixDoubleType class.
	 *  If there is no need for you to use the method declared in
	 *  the MatrixDoubleType class, you can invoke the destructor method
	 *  of the MatrixDoubleType class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~MatrixDoubleType();

	/** @brief Initialize 1D matrix
	 *
	 *  Default constructor method of the MatrixDoubleType class.
	 *  If you want to use the method declared in the MatrixDoubleType
	 *  class, you should first initialize the MatrixDoubleType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param n the number of first dimension
	 */
	inline void init( int n) {
		aMatrix.resize(1);
		aMatrix[0].resize(n);
		aRowSize = 1;
		aColSize = n;
	}

	/** @brief Initialize 2D matrix
	 *
	 *  Default constructor method of the MatrixDoubleType class.
	 *  If you want to use the method declared in the MatrixDoubleType
	 *  class, you should first initialize the MatrixDoubleType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param m the number of first dimension
	 *  @param n the number of second dimension
	 */
	inline void init(int m, int n){
		aMatrix.resize(m);
		for (int i = 0; i<m; i++){
			aMatrix[i].resize(n);
		}
		aRowSize = m;
		aColSize = n;
	}

	/** @brief Initialize 2D matrix
	 *
	 *  Default constructor method of the MatrixDoubleType class.
	 *  If you want to use the method declared in the MatrixDoubleType
	 *  class, you should first initialize the MatrixDoubleType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param m the number of first dimension
	 *  @param n the number of second dimension
	 */
	inline void init( const MatrixDoubleType& m ){
		aMatrix.resize(m.aRowSize);
		for (int i = 0; i<m.aRowSize; i++){
			aMatrix[i].resize(m.aColSize);
		}
		aRowSize = m.aRowSize;
		aColSize = m.aColSize;
	}

	/** @brief access operator for first dimetion of Matrix
	 *
	 *  @param i index that you want to access to
	 *  @return value of Matrix at index i
	 */
	inline double& operator[]( unsigned i ){ return aMatrix[0][i]; }

	/** @brief access operator for first dimetion of Matrix
	 *
	 *  @param i index that you want to access to
	 *  @return value of Matrix at index i
	 */
	inline double& operator[]( int i ){ return aMatrix[0][i]; }

	/** @brief access operator for first dimetion of Matrix
	 *
	 *  @param i index that you want to access to
	 *  @return value of Matrix at index i
	 */
	inline const double& operator[]( unsigned i ) const{ return aMatrix[0][i]; }

	/** @brief access operator for first dimetion of Matrix
	 *
	 *  @param i index that you want to access to
	 *  @return value of Matrix at index i
	 */
	inline const double& operator[]( int i ) const{ return aMatrix[0][i]; }

	/** @brief Addition operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 *  @return Another Matrix added
	 */
	inline MatrixDoubleType operator+( double num ) const
	{
		MatrixDoubleType mat(aRowSize, aColSize);
		int i, j;
		for( i = 0; i < aRowSize; i++ ){
			for( j = 0; j < aColSize; j++ ){
				mat.aMatrix[i][j] = aMatrix[i][j] + num;
			}
		}
		return mat;
	}

	/** @brief Addition operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 *  @return Another Matrix added
	 */
	inline MatrixDoubleType operator+( const MatrixDoubleType& m ) const
	{
		int row_size = m.aRowSize;
		int col_size = m.aColSize;
		int i, j;
		MatrixDoubleType mat(aRowSize, aColSize);

		if ( col_size == 1 ){
			for( i = 0; i < row_size; i++ ){
				for( j = 0; j < aColSize; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] + m.aMatrix[i][0];
				}
			}
		} // broadcast to axis = 1 ( to right )
		else if ( row_size == 1 ) {
			for( i = 0; i < aRowSize; i++ ){
				for( j = 0; j < col_size; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] + m.aMatrix[0][j];
				}
			}
		} // broadcast to axis = 0 ( to down )
		else {
			for( i = 0; i < row_size; i++ ){
				for( j = 0; j < col_size; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] + m.aMatrix[i][j];
				}
			}
		}

		return mat;
	}

	/** @brief Subtraction operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 *  @return Another Matrix subtracted
	 */
	inline MatrixDoubleType operator-( double num ) const{
		MatrixDoubleType mat(aRowSize, aColSize);
		int i, j;
		for( i = 0; i < aRowSize; i++ ){
			for( j = 0; j < aColSize; j++ ){
				mat.aMatrix[i][j] = aMatrix[i][j] - num;
			}
		}
		return mat;
	}

	/** @brief Subtraction operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 *  @return Another Matrix subtracted
	 */
	inline MatrixDoubleType operator-( const MatrixDoubleType& m ) const
	{
		int row_size = m.aRowSize;
		int col_size = m.aColSize;
		int i, j;
		MatrixDoubleType mat(aRowSize, aColSize);

		if ( col_size == 1 ){
			for( i = 0; i < row_size; i++ ){
				for( j = 0; j < aColSize; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] - m.aMatrix[i][0];
				}
			}
		} // broadcast to axis = 1 ( to right )
		else if ( row_size == 1 ) {
			for( i = 0; i < aRowSize; i++ ){
				for( j = 0; j < col_size; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] - m.aMatrix[0][j];
				}
			}
		} // broadcast to axis = 0 ( to down )
		else {
			for( i = 0; i < row_size; i++ ){
				for( j = 0; j < col_size; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] - m.aMatrix[i][j];
				}
			}
		}
		return mat;
	}

	/** @brief Addition-Assignment operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 */
	inline MatrixDoubleType& operator+=( double num )
	{
			return *this=operator+(num);
	}

	/** @brief Addition-Assignment operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 */
	inline MatrixDoubleType& operator+=( const MatrixDoubleType& m )
	{ return *this=operator+(m); }

	/** @brief Subtraction-Assginment operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 */
	inline MatrixDoubleType& operator-=( double num ){
		return *this=operator-(num);
	}

	/** @brief Subtraction-Assginment operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 */
	inline MatrixDoubleType& operator-=( const MatrixDoubleType& m )
	{ return *this=operator-(m); }

	/** @brief Matrix-Scalar multiplication operator
	*
	*  @param f Matrix-Scalar to multiply
	*  @return Another Matrix multiplied
	*/
	inline MatrixDoubleType operator*( double f ) const
	{
		// int row_size = aMatrix.size();
		// int col_size = aMatrix[0].size();
		MatrixDoubleType mat(aRowSize, aColSize);
		int i, j;
		for( i = 0; i < aRowSize; i++ ){
			for( j = 0; j < aColSize; j++ ){
				mat.aMatrix[i][j] = aMatrix[i][j] * f;
			}
		}
		return mat;
	}

	/** @brief Matrix-Matrix multiplication operator
	 *
	 *  @TODO if we need broadcasting function in multiply
	 *  add broadcasting function here.
	 *  @param m Matrix to multiply
	 *  @return Another Matrix multiplied
	 */
	inline MatrixDoubleType operator*( const MatrixDoubleType& m ) const
	{
		// int row_size = aMatrix.size();
		// int col_size = aMatrix[0].size();
		MatrixDoubleType mat(aRowSize, aColSize);
		int i, j;
		for( i = 0; i < aRowSize; i++ ){
			for( j = 0; j < aColSize; j++ ){
				mat.aMatrix[i][j] = aMatrix[i][j] * m.aMatrix[i][j];
			}
		}
		return mat;
	}

	/** @brief Matrix-Matrix multiplication Assignment operator
	 *
	 *  @param m Matrix to multiply
	 */
	inline MatrixDoubleType& operator*=( double num )
	{ return *this=operator*(num); }

	/** @brief Matrix-Matrix multiplication Assignment operator
	 *
	 *  @param m Matrix to multiply
	 */
	inline MatrixDoubleType& operator*=( const MatrixDoubleType& m )
	{ return *this=operator*(m); }

	/** @brief Dot product operator
	*
	*  @param m Matrix to perform dot product
	*  @return Matrix dot producted
	*/
	MatrixDoubleType dot( const MatrixDoubleType& m ) const;

	/** @brief Transpose function
	*
	*  @return Matrix transposed
	*/
	MatrixDoubleType transpose() const;

	/** @brief calculate length of matrix
	*
	*  calculate length
	*
	*  @return normalized length
	*/
	double length() const;

	/** @brief Display matrix function
	*
	*  Print entire matrix
	*
	*  @return void
	*/
	void display() const;

	/** @brief Display row and column dimension of the matrix
	*
	*  Print dimension of the matrix
	*
	*  @return void
	*/
	void shape() const;

	/** @brief Reset Matrix
	*
	*  clear the contents and then resize it with 0
	*
	*  @return void
	*/
	void reset();

	/** @brief insert 1d matrix into 1d matrix starting from input index
	*
	*  insert matrix into this matrix
	*
	*  @return void
	*/
	void insert1dTo1d( MatrixDoubleType& m, int rIdx );

	/** @brief insert 1d matrix into 2d matrix starting from input index
	*
	*  insert matrix into this matrix
	*
	*  @return void
	*/
	void insert1dTo2d( MatrixDoubleType& m, int rIdx, int rColIdx );

protected:

private:

};


class MatrixIntType
{
public:

	vector< vector<int> > aMatrix;
	int aRowSize;
	int aColSize;
	/** @brief Constructor of the MatrixIntType class.
	 *
	 *  Default constructor method of the MatrixIntType class.
	 *  If you want to use the method declared in the MatrixIntType
	 *  class, you should first initialize the MatrixIntType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	MatrixIntType();

	/** @brief Constructor of 1 Dimensional Matrix class.
	 *
	 *  Default constructor method of the MatrixIntType class.
	 *  If you want to use the method declared in the MatrixIntType
	 *  class, you should first initialize the MatrixIntType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param n the number of dimension
	 */
	MatrixIntType(int n);

	/** @brief Constructor of 2 Dimensional Matrix class.
	 *
	 *  Default constructor method of the MatrixIntType class.
	 *  If you want to use the method declared in the MatrixIntType
	 *  class, you should first initialize the MatrixIntType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param m the number of first dimension
	 *  @param n the number of second dimension
	 */
	MatrixIntType(int m, int n);

	/** @brief Constructor of 2 Dimensional Matrix class.
	 *
	 *  Default constructor method of the MatrixIntType class.
	 *  If you want to use the method declared in the MatrixIntType
	 *  class, you should first initialize the MatrixIntType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param m matrix with the same dimension
	 */
	MatrixIntType( const MatrixIntType& m );

	/** @brief Destructor of the MatrixIntType class.
	 *
	 *  Default destructor method of the MatrixIntType class.
	 *  If there is no need for you to use the method declared in
	 *  the MatrixIntType class, you can invoke the destructor method
	 *  of the MatrixIntType class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~MatrixIntType();

	/** @brief Initialize 1D matrix
	 *
	 *  Default constructor method of the MatrixIntType class.
	 *  If you want to use the method declared in the MatrixIntType
	 *  class, you should first initialize the MatrixIntType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param n the number of first dimension
	 */
	inline void init( int n) {
		aMatrix.resize(1);
		aMatrix[0].resize(n);
		aRowSize = 1;
		aColSize = n;
	}

	/** @brief Initialize 2D matrix
	 *
	 *  Default constructor method of the MatrixIntType class.
	 *  If you want to use the method declared in the MatrixIntType
	 *  class, you should first initialize the MatrixIntType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param m the number of first dimension
	 *  @param n the number of second dimension
	 */
	inline void init(int m, int n){
		aMatrix.resize(m);
		for (int i = 0; i<m; i++){
			aMatrix[i].resize(n);
		}
		aRowSize = m;
		aColSize = n;
	}

	/** @brief Initialize 2D matrix
	 *
	 *  Default constructor method of the MatrixIntType class.
	 *  If you want to use the method declared in the MatrixIntType
	 *  class, you should first initialize the MatrixIntType class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *
	 *  @param m the number of first dimension
	 *  @param n the number of second dimension
	 */
	inline void init( const MatrixIntType& m ){
		aMatrix.resize(m.aRowSize);
		for (int i = 0; i<m.aRowSize; i++){
			aMatrix[i].resize(m.aColSize);
		}
		aRowSize = m.aRowSize;
		aColSize = m.aColSize;
	}

	/** @brief access operator for first dimetion of Matrix
	 *
	 *  @param i index that you want to access to
	 *  @return value of Matrix at index i
	 */
	inline int& operator[]( unsigned i ){ return aMatrix[0][i]; }

	/** @brief access operator for first dimetion of Matrix
	 *
	 *  @param i index that you want to access to
	 *  @return value of Matrix at index i
	 */
	inline int& operator[]( int i ){ return aMatrix[0][i]; }

	/** @brief access operator for first dimetion of Matrix
	 *
	 *  @param i index that you want to access to
	 *  @return value of Matrix at index i
	 */
	inline const int& operator[]( unsigned i ) const{ return aMatrix[0][i]; }

	/** @brief access operator for first dimetion of Matrix
	 *
	 *  @param i index that you want to access to
	 *  @return value of Matrix at index i
	 */
	inline const int& operator[]( int i ) const{ return aMatrix[0][i]; }

	/** @brief Addition operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 *  @return Another Matrix added
	 */
	inline MatrixIntType operator+( int num ) const
	{
		MatrixIntType mat(aRowSize, aColSize);
		int i, j;
		for( i = 0; i < aRowSize; i++ ){
			for( j = 0; j < aColSize; j++ ){
				mat.aMatrix[i][j] = aMatrix[i][j] + num;
			}
		}
		return mat;
	}

	/** @brief Addition operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 *  @return Another Matrix added
	 */
	inline MatrixIntType operator+( const MatrixIntType& m ) const
	{
		int row_size = m.aRowSize;
		int col_size = m.aColSize;
		int i, j;
		MatrixIntType mat(aRowSize, aColSize);

		if ( col_size == 1 ){
			for( i = 0; i < row_size; i++ ){
				for( j = 0; j < aColSize; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] + m.aMatrix[i][0];
				}
			}
		} // broadcast to axis = 1 ( to right )
		else if ( row_size == 1 ) {
			for( i = 0; i < aRowSize; i++ ){
				for( j = 0; j < col_size; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] + m.aMatrix[0][j];
				}
			}
		} // broadcast to axis = 0 ( to down )
		else {
			for( i = 0; i < row_size; i++ ){
				for( j = 0; j < col_size; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] + m.aMatrix[i][j];
				}
			}
		}

		return mat;
	}

	/** @brief Subtraction operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 *  @return Another Matrix subtracted
	 */
	inline MatrixIntType operator-( int num ) const{
		MatrixIntType mat(aRowSize, aColSize);
		int i, j;
		for( i = 0; i < aRowSize; i++ ){
			for( j = 0; j < aColSize; j++ ){
				mat.aMatrix[i][j] = aMatrix[i][j] - num;
			}
		}
		return mat;
	}

	/** @brief Subtraction operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 *  @return Another Matrix subtracted
	 */
	inline MatrixIntType operator-( const MatrixIntType& m ) const
	{
		int row_size = m.aRowSize;
		int col_size = m.aColSize;
		int i, j;
		MatrixIntType mat(aRowSize, aColSize);

		if ( col_size == 1 ){
			for( i = 0; i < row_size; i++ ){
				for( j = 0; j < aColSize; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] - m.aMatrix[i][0];
				}
			}
		} // broadcast to axis = 1 ( to right )
		else if ( row_size == 1 ) {
			for( i = 0; i < aRowSize; i++ ){
				for( j = 0; j < col_size; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] - m.aMatrix[0][j];
				}
			}
		} // broadcast to axis = 0 ( to down )
		else {
			for( i = 0; i < row_size; i++ ){
				for( j = 0; j < col_size; j++ ){
					mat.aMatrix[i][j] = aMatrix[i][j] - m.aMatrix[i][j];
				}
			}
		}
		return mat;
	}

	/** @brief Addition-Assignment operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 */
	inline MatrixIntType& operator+=( int num )
	{
			return *this=operator+(num);
	}

	/** @brief Addition-Assignment operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 */
	inline MatrixIntType& operator+=( const MatrixIntType& m )
	{ return *this=operator+(m); }

	/** @brief Subtraction-Assginment operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 */
	inline MatrixIntType& operator-=( int num ){
		return *this=operator-(num);
	}

	/** @brief Subtraction-Assginment operator of Matrix
	 *
	 *  @param m const matrix with same dimensions
	 */
	inline MatrixIntType& operator-=( const MatrixIntType& m )
	{ return *this=operator-(m); }

	/** @brief Matrix-Scalar multiplication operator
	*
	*  @param f Matrix-Scalar to multiply
	*  @return Another Matrix multiplied
	*/
	inline MatrixIntType operator*( int f ) const
	{
		// int row_size = aMatrix.size();
		// int col_size = aMatrix[0].size();
		MatrixIntType mat(aRowSize, aColSize);
		int i, j;
		for( i = 0; i < aRowSize; i++ ){
			for( j = 0; j < aColSize; j++ ){
				mat.aMatrix[i][j] = aMatrix[i][j] * f;
			}
		}
		return mat;
	}

	/** @brief Matrix-Matrix multiplication operator
	 *
	 *  @param m Matrix to multiply
	 *  @return Another Matrix multiplied
	 */
	inline MatrixIntType operator*( const MatrixIntType& m ) const
	{
		// int row_size = aMatrix.size();
		// int col_size = aMatrix[0].size();
		MatrixIntType mat(aRowSize, aColSize);
		int i, j;
		for( i = 0; i < aRowSize; i++ ){
			for( j = 0; j < aColSize; j++ ){
				mat.aMatrix[i][j] = aMatrix[i][j] * m.aMatrix[i][j];
			}
		}
		return mat;
	}

	/** @brief Matrix-Matrix multiplication Assignment operator
	 *
	 *  @param m Matrix to multiply
	 */
	inline MatrixIntType& operator*=( const MatrixIntType& m )
	{ return *this=operator*(m); }

	/** @brief Dot product operator
	*
	*  @param m Matrix to perform dot product
	*  @return Matrix dot producted
	*/
	MatrixIntType dot( const MatrixIntType& m ) const;

	/** @brief Transpose function
	*
	*  @return Matrix transposed
	*/
	MatrixIntType transpose() const;

	/** @brief calculate length of matrix
	*
	*  calculate length
	*
	*  @return normalized length
	*/
	double length() const;

	/** @brief Display matrix function
	*
	*  Print entire matrix
	*
	*  @return void
	*/
	void display() const;

	/** @brief Display row and column dimension of the matrix
	*
	*  Print dimension of the matrix
	*
	*  @return void
	*/
	void shape() const;

	/** @brief Reset Matrix
	*
	*  clear the contents and then resize it with 0
	*
	*  @return void
	*/
	void reset();

	/** @brief insert matrix into matrix starting from input index
	*
	*  insert matrix into this matrix
	*
	*  @return void
	*/
	void insert1dTo1d( MatrixIntType& m, int idx );

	/** @brief insert 1d matrix into 2d matrix starting from input index
	*
	*  insert matrix into this matrix
	*
	*  @return void
	*/
	void insert1dTo2d( MatrixIntType& m, int rIdx, int rColIdx );

protected:

private:

};

#endif
