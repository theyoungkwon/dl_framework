/** @file MathModule.h
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

#ifndef MathModule_h
#define MathModule_h

#include "setup.h"
using namespace std;

class MathModule
{
public:

	/** @brief Constructor of the MathModule class.
	 *
	 *  Default constructor method of the MathModule class.
	 *  If you want to use the method declared in the MathModule
	 *  class, you should first initialize the MathModule class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	MathModule();

	/** @brief Destructor of the MathModule class.
	 *
	 *  Default destructor method of the MathModule class.
	 *  If there is no need for you to use the method declared in
	 *  the MathModule class, you can invoke the destructor method
	 *  of the MathModule class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~MathModule();

	/** @brief Make one hot vector which is used for comparing
	 *		   model outputs(predictions) to correct outputs (labels).
	 *
	 *  @param label [0, 1, 2, 3, 4]
	 *  @param Dim   Dimension of 1d matrix
	 *  @return double matrix with only one element value, 1.0
	 */
	MatrixDoubleType makeOneHotVec( int rLabel , int rDim );

	/** @brief sampling one of the word indices from vocabulary.
     *
	 *
	 *  @param ch the label which is given for a right answer
	 *  @return Loss of the model
	 */
	int sampling( char ch );

	/** @brief Softmax function (compute probablities of output layer)
	 *
	 *  @param z2 the array calculated from the model which
	 *  	   will be computed to get probabilities
	 *  @return probs of the model
	 */
	MatrixDoubleType mysoftmax( MatrixDoubleType& rZ2 );

	/** @brief argmax function ( return index with highest value )
	 *
	 *  @param rProbs probabilities of the predictions
	 *  @param rAxis column wise calculation = 0
	 *  			 row wise calculation = 1
	 *  @return index with highset value
	 */
	MatrixIntType argmax( MatrixDoubleType& rProbs, int rAxis );

	/** @brief sum function (compute total sum of the matrix )
	 *
	 *  @param rParams matrix to be computed
	 *  @return double total sum of the matrix
	 */
	double sum( const MatrixDoubleType& rParams );

	/** @brief calculate length of matrix
	*
	*  calculate length
	*
	*  @return normalized length
	*/
	double length( const MatrixDoubleType& rParams ) const;

	/** @brief Give you the current time
	 *
	 *  @param void
	 *  @return current time
	 */
	double mysecond();

protected:

private:

};


#endif
