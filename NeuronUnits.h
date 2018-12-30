/** @file NeuronUnits.h
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

#ifndef NeuronUnits_h
#define NeuronUnits_h

#include "setup.h"
using namespace std;

class NeuronUnits
{
public:

	/** @brief Constructor of the NeuronUnits class.
	 *
	 *  Default constructor method of the NeuronUnits class.
	 *  If you want to use the method declared in the NeuronUnits
	 *  class, you should first initialize the NeuronUnits class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	NeuronUnits();

	/** @brief Destructor of the NeuronUnits class.
	 *
	 *  Default destructor method of the NeuronUnits class.
	 *  If there is no need for you to use the method declared in
	 *  the NeuronUnits class, you can invoke the destructor method
	 *  of the NeuronUnits class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~NeuronUnits();

	/** @brief Sigmoid function (non-linearity neuron units)
	 *
	 *  sigmoid(z) = 1 / 1 + exp(-z) where 0 <= sigmoid(z) <= 1
	 *
	 *  @param m matrix to compute sigmoid function
	 *  @return matrix after sigmoid function
	 */
	MatrixDoubleType sigmoid( const MatrixDoubleType& m );

	/** @brief Hyperbolic Tangent function (non-linearity neuron
	 * 		   units)
	 *
	 *  tanh(z) = (2 / 1 + exp(-2z)) - 1 where -1 <= tanh(z) <= 1
	 *
	 *  @param m matrix to compute tanh function
	 *  @return matrix after tanh function
	 */
	MatrixDoubleType tanh( const MatrixDoubleType& m );

	/** @brief Relu function (non-linearity neuron units)
	 *
	 *  relu(z) = max(z,0) where 0 <= relu(z)
	 *
	 *  @param m matrix to compute relu function
	 *  @return matrix after relu function
	 */
	MatrixDoubleType relu( const MatrixDoubleType& m );

protected:

private:

};

#endif
