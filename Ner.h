/** @file Ner.h
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module Ner
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

#ifndef Ner_h
#define Ner_h

#include "setup.h"
#include "RandomModule.h"
#include "NeuronUnits.h"
#include "Derivatives.h"
#include "LossFunctions.h"
#include "MathModule.h"

using namespace std;

class Ner
{
public:

	double aLambda;
	double aAlpha;
	map< int, MatrixDoubleType > aSparamsL;
	map< int, MatrixDoubleType > aSgradL;
	map< int, int > aUpdatedSgradLKeys;
	MatrixDoubleType aParamsW;
	MatrixDoubleType aParamsU;
	MatrixDoubleType aParamsB1;
	MatrixDoubleType aParamsB2;
	MatrixDoubleType aGradW;
	MatrixDoubleType aGradU;
	MatrixDoubleType aGradB1;
	MatrixDoubleType aGradB2;

	int aWindowSize;
	vector< int > aDims;
	int aSeed;
	int aVocaSize;

	// library classes
	RandomModule 	YdRm;
	NeuronUnits 	YdNu;
	Derivatives 	YdDeri;
	LossFunctions 	YdLf;
	MathModule 		YdMm;

	/** @brief Constructor of the Ner class.
	 *
	 *  Default constructor method of the Ner class.
	 *  If you want to use the method declared in the Ner
	 *  class, you should first initialize the Ner class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	Ner();

	/** @brief Constructor of the Ner class.
	 *
	 *  Default constructor method of the Ner class.
	 *  If you want to use the method declared in the Ner
	 *  class, you should first initialize the Ner class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	Ner( map< int, MatrixDoubleType >& rWordVec, int rWindowSize, int rDims[],
		 double rLambda, double rAlpha, int rSeed);

	/** @brief Destructor of the Ner class.
	 *
	 *  Default destructor method of the Ner class.
	 *  If there is no need for you to use the method declared in
	 *  the Ner class, you can invoke the destructor method
	 *  of the Ner class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~Ner();

	/** @brief Initialize the Ner class.
	 *
	 *  Default constructor method of the Ner class.
	 *  If you want to use the method declared in the Ner
	 *  class, you should first initialize the Ner class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 *  @return void
	 */
	void init( map< int, MatrixDoubleType >& rWordVec, int rWindowSize, int rDims[],
		 double rLambda, double rAlpha, int rSeed);

	/** @brief reset accumulated gradients
	 *
	 *  this function is needed to accumulate new gradients
	 *  for the next iteration windows
	 *  @return void
	 */
	void resetAccGrads();

	/** @brief Accumulate greadients
	 *
	 *  this function is needed if the model is trained
	 *  with minibatch. By using accumulating the gradients of
	 *  each minibatch, we can properly train the model
	 *
	 *  @param window [w_i-1, x_i, x_i+1]
	 *  @param label single int among [0, 1, 2, 3, 4]
	 *  @return void
	 */
	void accGrads( MatrixIntType& rWindow, int label );

	/** @brief Apply accumulated gradients
	 *
	 *  update gradients (= training )
	 *
	 *  @return void
	 */
	void applyAccGrads( double rAlpha );

	/** @brief Predict class probabilities
	 *
	 *  @param rWindows matrix ( n x WindowSize )
	 *  	   each row is a window of indices
	 *  @return void
	 */
	MatrixDoubleType predictProbs( MatrixIntType& rWindows );

	/** @brief Predict Labels
	 *
	 *  @param rWindows matrix ( n x WindowSize )
	 *  	   each row is a window of indices
	 *  @return predicted classes
	 */
	MatrixIntType predictLabels( MatrixIntType& rWindows );

	/** @brief Compute loss of a dataset
	 *
	 *  @param rWindows matrix ( n x WindowSize )
	 *  	   each row is a window of indices*
	 *  @param rLabels matrix ( 1 x n )
	 *  	   each column is a label
	 *  @return loss
	 */
	double computeLoss( MatrixIntType& rWindows, MatrixIntType& rLabels  );

	/** @brief Compute Mean loss of a dataset
	 *
	 *  @param rWindows matrix ( n x WindowSize )
	 *  	   each row is a window of indices*
	 *  @param rLabels matrix ( 1 x n )
	 *  	   each column is a label
	 *  @return mean loss
	 */
	double computeMeanLoss( MatrixIntType& rWindows, MatrixIntType& rLabels );

	/** @brief Train the model using Stochastic
	 *  	   Gradient Descent (SGD)
	 *
	 *  @param rXs matrix ( n x WindowSize )
	 *  	   each row is a window of indices
	 *  @param rYs matrix ( 1 x n )
	 *  	   each column is a label
	 *  @param rIdxIter matrix ( iterSize x batchSize )
	 *  @param rAlpha learning rate ( 1 x iterSize )
	 *  @param rPrintEvery print how iteration is computed on every sth counts
	 *  @param rCostEvery print loss on every sth counts
	 *  @param rDevIter matrix ( iterSize x batchSize )
	 *  @return Probs matrix ( n x LabelSize )
	 */
	vector< double > trainSgd( MatrixIntType& rXs, MatrixIntType& rYs,
					 MatrixIntType& rIdxIter, MatrixDoubleType& rAlphaIter,
					 int rPrintEvery, int rLossEvery,
					 MatrixIntType& rDevIter );

	/** @brief Train single point data
	 *
	 *  @param rWindows matrix ( n x WindowSize )
	 *  	   each row is a window of indices
	 *  @param rY  single label
	 *  	   each column is a label
	 *  @param rAlpha learning rate
	 *  @return void
	 */
	void trainPointSgd( MatrixIntType& rX, int rY, double rAlpha );

	/** @brief Train minibatch data
	 *
	 *  @param rWindows matrix ( n x WindowSize )
	 *  	   each row is a window of indices
	 *  @param rLabels matrix ( 1 x n )
	 *  	   each column is a label
	 *  @param rAlpha learning rate
	 *  @return void
	 */
	void trainMinibatchSgd( MatrixIntType& rXs, MatrixIntType& rYs, double rAlpha );

	/** @brief Check gradients
	 *
	 *  @param rXs matrix ( n x WindowSize )
	 *  	   each row is a window of indices
	 *  @param rYs matrix ( 1 x n )
	 *  	   each column is a label
	 *  @return void
	 */
	void gradCheck( MatrixIntType& rXs, MatrixIntType& rYs, double rEps,
		 			double rThreshold, bool rVerbos );

protected:

private:

};


#endif

