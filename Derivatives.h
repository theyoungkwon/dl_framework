/** @file Derivatives.h
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module Derivatives
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

#ifndef Derivatives_h
#define Derivatives_h

#include "setup.h"
using namespace std;

class Derivatives
{
public:

	/** @brief Constructor of the Derivatives class.
	 *
	 *  Default constructor method of the Derivatives class.
	 *  If you want to use the method declared in the Derivatives
	 *  class, you should first initialize the Derivatives class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	Derivatives();

	/** @brief Destructor of the Derivatives class.
	 *
	 *  Default destructor method of the Derivatives class.
	 *  If there is no need for you to use the method declared in
	 *  the Derivatives class, you can invoke the destructor method
	 *  of the Derivatives class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~Derivatives();

	/** @brief Sigmoid gradient function (non-linearity neuron units)
	 *
	 *  sigmoidGradient(z) = sigmoid(z)/(1 - sigmoid(z))
	 *
	 *  @param m matrix to compute sigmoidGradient function
	 *  @return matrix after sigmoidGradient function
	 */
	MatrixDoubleType sigmoidGradient( const MatrixDoubleType& m );

	/** @brief Hyperbolic Tangent gradient function (non-linearity neuron
	 * 		   units)
	 *
	 *  tanhGradient(z) = 1 - square( tanh(z) )
	 *
	 *  @param m matrix to compute tanhGradient function
	 *  @return matrix after tanhGradient function
	 */
	MatrixDoubleType tanhGradient( const MatrixDoubleType& m );

	/** @brief Relu gradient function (non-linearity neuron units)
	 *
	 *  reluGradient(z) = 1 : z > 0
	 *					= 0 : otherwise
	 *
	 *  @param m matrix to compute reluGradient function
	 *  @return matrix after reluGradient function
	 */
	MatrixDoubleType reluGradient( const MatrixDoubleType& m );

protected:

private:

};

#endif
