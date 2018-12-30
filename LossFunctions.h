/** @file LossFunctions.h
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

#ifndef LossFunctions_h
#define LossFunctions_h

#include "setup.h"
using namespace std;

class LossFunctions
{
public:

	/** @brief Constructor of the LossFunctions class.
	 *
	 *  Default constructor method of the LossFunctions class.
	 *  If you want to use the method declared in the LossFunctions
	 *  class, you should first initialize the LossFunctions class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	LossFunctions();

	/** @brief Destructor of the LossFunctions class.
	 *
	 *  Default destructor method of the LossFunctions class.
	 *  If there is no need for you to use the method declared in
	 *  the LossFunctions class, you can invoke the destructor method
	 *  of the LossFunctions class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~LossFunctions();

	/** @brief Calculate Loss of the model using Negative Log
	 * 		   Likelihood (NLL).
	 *
	 *  NLL += -log(y[i, labels[i]])
	 *
	 *  @param probs the array calculated from the model which
	 *  	   contains predictions
	 *  @param labels the label which is given for a right answer
	 *  @return Loss of the model
	 */
	double negLogLikelihood( const MatrixDoubleType& probs, const MatrixIntType& labels );

	/** @brief Calculate Loss of the model using Square Loot
	 * 		   Error.
	 *
	 *  Square Error = sum( (labels - predictions)^2 )
	 *
	 *  @param probs the array calculated from the model which
	 *  	   contains predictions
	 *  @param labels the label which is given for a right answer
	 *  @return Loss of the model
	 */
	double squareError( const MatrixDoubleType& probs, const MatrixIntType& labels );

	/** @brief Calculate Loss of the model using regression.
	 *
	 *  Regularization Loss = 0.5*lambda*(W^2 + U^2)
	 *
	 *  @param probs the array calculated from the model which
	 *  	   contains predictions
	 *  @param labels the label which is given for a right answer
	 *  @return Loss of the model
	 */
	double regLoss( double lambda, const MatrixDoubleType& paramsW, const MatrixDoubleType& paramsU );

protected:

private:

};

#endif