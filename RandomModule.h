/** @file RandomModule.h
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

#ifndef RandomModule_h
#define RandomModule_h

#include "setup.h"
using namespace std;

class RandomModule
{
public:

	/** @brief Constructor of the RandomModule class.
	 *
	 *  Default constructor method of the RandomModule class.
	 *  If you want to use the method declared in the RandomModule
	 *  class, you should first initialize the RandomModule class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	RandomModule();

	/** @brief Destructor of the RandomModule class.
	 *
	 *  Default destructor method of the RandomModule class.
	 *  If there is no need for you to use the method declared in
	 *  the RandomModule class, you can invoke the destructor method
	 *  of the RandomModule class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~RandomModule();

	/** @brief Initialize parameters of input matrix using Xavier
	 *		   Initialization Scheme.
	 *
	 *  initialize weights based on Xavier Initialization scheme.
	 *  this scheme is known for making the model better in terms of
	 *  predicing performance
	 *
	 *  @param m matrix to be initialized
	 *  @return void
	 */
	void xavierDoubleInit( MatrixDoubleType& m );

	/** @brief Initialize parameters of input matrix using Xavier
	 *		   Initialization Scheme.
	 *
	 *  initialize weights based on Xavier Initialization scheme.
	 *  this scheme is known for making the model better in terms of
	 *  predicing performance
	 *
	 *  @param m row size
	 *  @param n column size
	 *  @return double matrix
	 */
	MatrixDoubleType xavierDoubleInit( int rRowSize, int rColSize );

	/** @brief Initialize parameters of input matrix using Xavier
	 *		   Initialization Scheme.
	 *
	 *  initialize weights based on Xavier Initialization scheme.
	 *  this scheme is known for making the model better in terms of
	 *  predicing performance
	 *
	 *  @param m matrix to be initialized
	 *  @return void
	 */
	void xavierIntInit( MatrixIntType& m );

	/** @brief Initialize parameters of input matrix using Xavier
	 *		   Initialization Scheme.
	 *
	 *  initialize weights based on Xavier Initialization scheme.
	 *  this scheme is known for making the model better in terms of
	 *  predicing performance
	 *
	 *  @param m row size
	 *  @param n column size
	 *  @return int matrix
	 */
	MatrixIntType xavierIntInit( int rRowSize, int rColSize );

	/** @brief Initialize parameters of input matrix using rand
	 *		   function.
	 *
	 *  initialize weights based on Gaussian distribution
	 *
	 *  @param m matrix to be initialized
	 *  @return void
	 */
	void randNormalDoubleInit( MatrixDoubleType& m );

	/** @brief Initialize parameters of input matrix using rand
	 *		   function.
	 *
	 *  initialize weights based on Gaussian distribution
	 *
	 *  @param m row size
	 *  @param n column size
	 *  @return double matrix
	 */
	MatrixDoubleType randNormalDoubleInit( int rRowSize, int rColSize );

	/** @brief Initialize parameters of input matrix using rand
	 *		   function.
	 *
	 *  initialize weights based on Gaussian distribution
	 *
	 *  @param m matrix to be initialized
	 *  @return void
	 */
	void randNormalIntInit( MatrixIntType& m );

	/** @brief Initialize parameters of input matrix using rand
	 *		   function.
	 *
	 *  initialize weights based on Gaussian distribution
	 *
	 *  @param m row size
	 *  @param n column size
	 *  @return int matrix
	 */
	MatrixIntType randNormalIntInit( int rRowSize, int rColSize );

protected:

private:

};


#endif
