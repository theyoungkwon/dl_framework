/** @file Ner.cpp
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

#include "Ner.h"

Ner::Ner(){}

Ner::Ner( map< int, MatrixDoubleType >& rWordVec, int rWindowSize, int rDims[],
		 double rLambda, double rAlpha, int rSeed){
	aLambda = rLambda;
	aAlpha = rAlpha;
	aWindowSize = rWindowSize;
	aSeed = rSeed;
	aVocaSize = rWordVec.size();
	aDims.push_back(rDims[0]);
	aDims.push_back(rDims[1]);
	aDims.push_back(rDims[2]);

	// aDims[1] x /aWindowSize*aDims[0]/
	aParamsW.init(aDims[1], aWindowSize*aDims[0]);
	aGradW.init(aDims[1], aWindowSize*aDims[0]);
	// aDims[1] x 1
	aGradB1.init(aDims[1], 1);

	// aDims[2] x aDims[1]
	aParamsU.init(aDims[2], aDims[1]);
	aGradU.init(aDims[2], aDims[1]);
	// aDims[2] x 1
	aGradB2.init(aDims[2], 1);

	// vocaSize x rdims[0]
	// aSparamsL = rWordVec;
	MatrixDoubleType temp(1, aDims[0]);
	for( int i = 0; i < aVocaSize; i++){
		aSparamsL[i] = rWordVec[i];
		aSgradL[i] = temp;
	}
	// initialize random weights
	YdRm.xavierDoubleInit(aParamsW);
	YdRm.xavierDoubleInit(aParamsU);
}

Ner::~Ner(){ }

void Ner::init( map< int, MatrixDoubleType >& rWordVec, int rWindowSize, int rDims[],
		 double rLambda, double rAlpha, int rSeed){
	aLambda = rLambda;
	aAlpha = rAlpha;
	aWindowSize = rWindowSize;
	aSeed = rSeed;
	aVocaSize = rWordVec.size();
	aDims.push_back(rDims[0]);
	aDims.push_back(rDims[1]);
	aDims.push_back(rDims[2]);

	// aDims[1] x /aWindowSize*aDims[0]/
	aParamsW.init(aDims[1], aWindowSize*aDims[0]);
	aGradW.init(aDims[1], aWindowSize*aDims[0]);
	// aDims[1] x 1
	aParamsB1.init(aDims[1], 1);
	aGradB1.init(aDims[1], 1);

	// aDims[2] x aDims[1]
	aParamsU.init(aDims[2], aDims[1]);
	aGradU.init(aDims[2], aDims[1]);
	// aDims[2] x 1
	aParamsB2.init(aDims[2], 1);
	aGradB2.init(aDims[2], 1);

	// vocaSize x rdims[0]
	// aSparamsL = rWordVec;
	MatrixDoubleType temp(1, aDims[0]);
	for( int i = 0; i < aVocaSize; i++){
		aSparamsL[i] = rWordVec[i];
		aSgradL[i] = temp;
		// aSparamsL[i].display();
		// aSgradL[i].display();
	}

	// initialize random weights
	// aParamsW += 0.5;
	// aParamsU += 0.2;

	YdRm.xavierDoubleInit(aParamsW);
	YdRm.xavierDoubleInit(aParamsU);
}

void Ner::resetAccGrads(){
	// grads reset
	aGradW.reset();
	aGradU.reset();
	aGradB1.reset();
	aGradB2.reset();
	// sgrads reset
	int aSgradL_key;
	for (map<int, int>::iterator it=aUpdatedSgradLKeys.begin(); it!=aUpdatedSgradLKeys.end(); ++it){
		aSgradL_key = it->first;
		aSgradL[aSgradL_key].reset();
	}
	aUpdatedSgradLKeys.clear();
}

void Ner::accGrads( MatrixIntType& rWindow, int label ){

	// x_temp = /window_size/ x /word_dim/
	MatrixDoubleType x_temp(1, aDims[0]);
	// x = /window_size*word_dim/ x /1/
	MatrixDoubleType x(aWindowSize*aDims[0], 1);
	int i, j;

	// make one long input vector x
	for( i = 0; i < aWindowSize; i++ ){
		x_temp = aSparamsL[rWindow.aMatrix[0][i]];
		// x_temp.display();
		x.insert1dTo1d(x_temp, i*x_temp.aColSize);
	}

	// printf("accGrads before forward\n");
	// initialize needed double matrix for forward propagation
    MatrixDoubleType z1(aDims[1], 1);
    MatrixDoubleType h(aDims[1], 1);
    MatrixDoubleType z2(aDims[2], 1);
    MatrixDoubleType y(1, aDims[2]);
    // y.shape();
    // Forward propagation
    z1 = aParamsW.dot(x) + aParamsB1;
    // printf("accGrads after W.dot( x ) \n");
    h = YdNu.tanh(z1);
    z2 = aParamsU.dot(h) + aParamsB2;
    // printf("accGrads after U.dot( h ) \n");
    // z2.shape();
    y = z2.transpose();
    // y.shape();
    // z2.shape();
    y = YdMm.mysoftmax( y );
    // y.shape();
    // printf("accGrads after forward\n");
    // make one hot vector t="target"
    MatrixDoubleType t(aDims[2], 1);
    t.aMatrix[label][0] = 1.0;

    // initialize needed double matrix for back propagation
    MatrixDoubleType delta_s(aDims[2], 1);
    MatrixDoubleType delta_h(aDims[1], 1);
    MatrixDoubleType delta_x(aWindowSize*aDims[0], 1);
    MatrixDoubleType aParamsU_T(aDims[1], aDims[2]);
    MatrixDoubleType aParamsW_T(aWindowSize*aDims[0], aDims[1]);
    MatrixDoubleType h_T(1, aDims[1]);

    // printf("accGrads before backward\n");
    // Back propagation
    // delta_s.shape();
    // delta_h.shape();
    // y.shape();
    // t.shape();
    delta_s = y.transpose();
    delta_s -= t;
    aParamsU_T = aParamsU.transpose();
    // aParamsU.shape();
    // aParamsU_T.shape();
    delta_h = ( aParamsU_T.dot(delta_s) ) * ( YdDeri.tanhGradient(h) );
    // printf("accGrads before accumulate gradients\n");
    // accumulate dense gradients ( weights )
    h_T = h.transpose();
    aGradU  += ( delta_s.dot(h_T) );
    aGradU  += ( aParamsU * aLambda );
    aGradB2 += delta_s;
    aGradW  += ( delta_h.dot(x.transpose()) );
    aGradW  += ( aParamsW * aLambda );
    aGradB1 += delta_h;
    // aGradU.shape();
    // aGradB2.shape();
    // aGradW.shape();
    // aGradB1.shape();
    // printf("accGrads after backward\n");
	// accumulate sparse gradients ( words )
	aParamsW_T = aParamsW.transpose();
	delta_x = aParamsW_T.dot(delta_h);
	// delta_x.shape();
	// printf("%d rWindow", rWindow.aMatrix[0][i]);
	for ( i = 0; i < aWindowSize; i++ ){
		for ( j = 0; j < aDims[0]; j++){
			x_temp.aMatrix[0][j] = delta_x.aMatrix[i*aDims[0] + j][0];
		}
		// printf("%d rWindow\n", rWindow.aMatrix[0][i]);
		aSgradL[ rWindow.aMatrix[0][i] ] += x_temp;
		aUpdatedSgradLKeys[ rWindow.aMatrix[0][i] ] = rWindow.aMatrix[0][i];
	}
}

void Ner::applyAccGrads( double rAlpha ){
	int aSgradL_key;
	if ( 0.0 > rAlpha ){
		// Dense updates
		aParamsW -= aGradW * aAlpha;
		aParamsU -= aGradU * aAlpha;
		aParamsB1 -= aGradB1 * aAlpha;
		aParamsB2 -= aGradB2 * aAlpha;
	    // Sparse updates
		for (map<int, int>::iterator it=aUpdatedSgradLKeys.begin(); it!=aUpdatedSgradLKeys.end(); ++it){
			aSgradL_key = it->first;
			aSparamsL[aSgradL_key] -= aSgradL[aSgradL_key] * aAlpha;
		}
	} // when there is no input rAlpha ( use default aAlpha )
	else {
		// Dense updates
		aParamsW -= aGradW * rAlpha;
		aParamsU -= aGradU * rAlpha;
		aParamsB1 -= aGradB1 * rAlpha;
		aParamsB2 -= aGradB2 * rAlpha;
	    // Sparse updates
		for (map<int, int>::iterator it=aUpdatedSgradLKeys.begin(); it!=aUpdatedSgradLKeys.end(); ++it){
			aSgradL_key = it->first;
			aSparamsL[aSgradL_key] -= aSgradL[aSgradL_key] * rAlpha;
		}
	} // when there are input rAlpha
}

MatrixDoubleType Ner::predictProbs( MatrixIntType& rWindows ){

	int total_size = rWindows.aRowSize;
	// x_temp = /window_size/ x /word_dim/
	MatrixDoubleType x_temp(1, aDims[0]);
	// x = /window_size*word_dim/ x /N/
	MatrixDoubleType x(aWindowSize*aDims[0], total_size);
	int i, j;
	// make one long input vector x
	for ( i = 0; i< total_size; i++){
		for( j = 0; j < aWindowSize; j++ ){
			x_temp = aSparamsL[rWindows.aMatrix[i][j]];
			x.insert1dTo2d(x_temp, j*x_temp.aColSize, i);
		}
	}

	// initialize needed double matrix for forward propagation
    MatrixDoubleType z1(aDims[1], total_size);
    MatrixDoubleType h(aDims[1], total_size);
    MatrixDoubleType z2(aDims[2], total_size);
    MatrixDoubleType y(total_size, aDims[2]);

    // Forward propagation
    // aParamsB1 is broadcasted to /W/ x /x/
    z1 = aParamsW.dot(x) + aParamsB1;
    h = YdNu.tanh(z1);
    // aParamsB2 is broadcasted to /U/ x /h/
    z2 = aParamsU.dot(h) + aParamsB2;
    y = z2.transpose();
    y = YdMm.mysoftmax( y );

    return y;
}

MatrixIntType Ner::predictLabels( MatrixIntType& rWindows ){

	// initialize Matrix needed
	MatrixDoubleType y(rWindows.aRowSize, aDims[2]);
	MatrixIntType    c(rWindows.aRowSize, 1);

	// perform predictions
	y = predictProbs(rWindows);
	// row-wise calculation
	c = YdMm.argmax(y, 1);

	return c;
}

double Ner::computeLoss( MatrixIntType& rWindows, MatrixIntType& rLabels ){

	// initialize needed matrix
	int total_size = rWindows.aRowSize;
	MatrixDoubleType y( total_size, aDims[2] );
	double loss = 0.0;
	double reg_loss = 0.0;
	int i;

	y = predictProbs(rWindows);
	// compute Loss of the model without regularization
	for( i = 0; i < total_size; i++ ){
		loss += -log( y.aMatrix[i][ rLabels.aMatrix[0][i] ]);
	}

  	// compute Loss of regularization
	reg_loss = 0.5*aLambda*( YdMm.sum(aParamsW * aParamsW) + YdMm.sum(aParamsU * aParamsU) );

	// compute total loss of the model
	loss += reg_loss;

	return loss;
}

double Ner::computeMeanLoss( MatrixIntType& rWindows, MatrixIntType& rLabels ){
	int length = rLabels.aColSize;
	return computeLoss(rWindows, rLabels) / (double) length;
}

vector< double > Ner::trainSgd(MatrixIntType& rXs, MatrixIntType& rYs,
					 MatrixIntType& rIdxIter, MatrixDoubleType& rAlphaIter,
					 int rPrintEvery, int rLossEvery,
					 MatrixIntType& rDevIter){

    vector< double > losses;
    double loss = 0.0;
    int idx;
    int alpha;
    int total_size = rIdxIter.aRowSize;
    int batch_size = rIdxIter.aColSize;
    int i, j, counter, q;
    double t = YdMm.mysecond();
    double t_total = 0.0;
    printf( "Begin SGD... \n");
    MatrixIntType x_dev;
    MatrixIntType y_dev;
    if( true != rDevIter.aMatrix.empty()  ){
    	x_dev.init(rDevIter.aRowSize, aWindowSize);
    	y_dev.init(1, rDevIter.aRowSize);

    	for( i = 0 ; i < rDevIter.aRowSize ; i++ ){
			for(j = 0; j < aWindowSize; j++ ){
				x_dev.aMatrix[i][j] = rXs.aMatrix[ rDevIter.aMatrix[i][0] ][j];
			}
			y_dev.aMatrix[0][i] = rYs.aMatrix[0][ rDevIter.aMatrix[i][0] ];
		}
    }

	    if( 1 < batch_size ){
	    	MatrixIntType x(batch_size, aWindowSize);
	    	MatrixIntType y(1, batch_size);

	    	for ( counter = 0; counter < total_size ; counter++ ){
		    	if ( 0 == (counter % rPrintEvery) ){
		    		t = YdMm.mysecond() - t;
		    		t_total += t;
		    		t = YdMm.mysecond();
		    		printf("Batch : Computed %d in time %g s\n", counter, t_total);
		    	}
		    	if ( 0 == (counter % rLossEvery) ){
		    		if ( true != rDevIter.aMatrix.empty() ) {
				    	loss = computeMeanLoss(x_dev, y_dev);
				    } // there exist dev set
				    else {
				    	loss = computeMeanLoss(rXs, rYs);
				    } // when there are no dev set
				    losses.push_back( loss );
	                printf("Batch :  [%d]: mean loss %g\n" , counter, loss );
		    	}
		    	for( i = 0 ; i < batch_size ; i++ ){
		    		for(j = 0; j < aWindowSize; j++ ){
		    			x.aMatrix[i][j] = rXs.aMatrix[ rIdxIter.aMatrix[counter][i] ][j];
		    		}
		    		y.aMatrix[0][i] = rYs.aMatrix[0][ rIdxIter.aMatrix[counter][i] ];
		    	}
		    	trainMinibatchSgd(x, y, rAlphaIter.aMatrix[0][counter] );
		    }
	    } // minibatch sgd
	    else {
	    	MatrixIntType x(batch_size, aWindowSize);
	    	int y;

	    	for ( counter = 0; counter < total_size ; counter++ ){
		    	if ( 0 == (counter % rPrintEvery) ){
		    		t = YdMm.mysecond() - t;
		    		t_total += t;
		    		t = YdMm.mysecond();
		    		printf("Single : Computed %d in time %g s\n", counter, t_total);
		    	}
		    	if ( 0 == (counter % rLossEvery) ){
		    		if ( true != rDevIter.aMatrix.empty() ) {
				    	loss = computeMeanLoss(x_dev, y_dev);
				    } // there exist dev set
				    else {
				    	loss = computeMeanLoss(rXs, rYs);
				    } // when there are no dev set
				    losses.push_back( loss );
	                printf("Single :  [%d]: mean loss %g\n" , counter, loss );
		    	}
		    	for(j = 0; j < aWindowSize; j++ ){
	    			x.aMatrix[0][j] = rXs.aMatrix[ rIdxIter.aMatrix[counter][0] ][j];
	    		}
	    		y = rYs.aMatrix[0][ rIdxIter.aMatrix[counter][0] ];
		    	trainPointSgd(x, y, rAlphaIter.aMatrix[0][counter] );
		    }
	    } // single point sgd

    // compute loss after training
    if ( true != rDevIter.aMatrix.empty() ) {
    	loss = computeMeanLoss(x_dev, y_dev);
    } // there exist dev set
    else {
    	loss = computeMeanLoss(rXs, rYs);
    } // when there are no dev set
    losses.push_back(loss);
    t = YdMm.mysecond() - t;
	t_total += t;
    printf("SGD complete: %d examples in %g seconds.\n", counter, t_total );
	printf("  [%d]: mean loss %g\n", counter, loss );

    return losses;
}

void Ner::trainPointSgd( MatrixIntType& rX, int rY, double rAlpha ){
	resetAccGrads();
	accGrads(rX, rY);
	applyAccGrads(rAlpha);
}

void Ner::trainMinibatchSgd( MatrixIntType& rXs, MatrixIntType& rYs, double rAlpha ){

	resetAccGrads();
	int i, j;
	MatrixIntType x(1, rXs.aColSize);
	for( i = 0 ; i < rYs.aColSize; i++){
		for( j = 0; j < rXs.aColSize; j++ ){
			x.aMatrix[0][j] = rXs.aMatrix[i][j];
		}
		accGrads(x, rYs.aMatrix[0][i]);
	}
	applyAccGrads(rAlpha);
}

void Ner::gradCheck(MatrixIntType& rXs, MatrixIntType& rYs, double rEps = 0.0001,
		 			double rThreshold = 0.000001, bool rVerbose = false ){

	resetAccGrads();
	int i, j;
	MatrixIntType x(1, rXs.aColSize);
	for( i = 0 ; i < rYs.aColSize; i++){
		for( j = 0; j < rXs.aColSize; j++ ){
			x.aMatrix[0][j] = rXs.aMatrix[i][j];
		}
		accGrads(x, rYs.aMatrix[0][i]);
	}

	/////////////////////////////////////////////////////////////////////////
	/////////// check dense parameters' gradients ( W, U, B1, B2 ) //////////
	///// check W's gradients /////
	double loss_plus;
	double loss_minus;
	double temp;
	double grad_delta;
	MatrixDoubleType grad_computed_W(aParamsW.aRowSize, aParamsW.aColSize);
	MatrixDoubleType grad_approx_W(aParamsW.aRowSize, aParamsW.aColSize);
	grad_computed_W = aGradW;
	for(i = 0; i < aParamsW.aRowSize; i++ ){
		for( j = 0; j < aParamsW.aColSize; j++ ){
			temp = aParamsW.aMatrix[i][j];
			aParamsW.aMatrix[i][j] = temp + rEps;
			loss_plus = computeLoss(rXs, rYs);
			aParamsW.aMatrix[i][j] = temp - rEps;
			loss_minus = computeLoss(rXs, rYs);
			aParamsW.aMatrix[i][j] = temp;
			grad_approx_W.aMatrix[i][j] = (loss_plus - loss_minus) / (2*rEps);
		}
	}

	// print gradient check result ( [ok] or [Error])
	grad_delta = YdMm.length(grad_approx_W - grad_computed_W);
	printf("grad_check: dJ/d%s error norm = %.04g \n", "aParamsW" , grad_delta);
	if ( rThreshold > grad_delta ) {
		printf("[ok]\n");
	} else {
		printf("** ERROR **");
	}

	// when you set rVerbose 'true', you can see detailed information of which element
	// generates gradient's error and its value.
    if ( ( false != rVerbose ) && (grad_delta > rThreshold) ){
    	for(i = 0; i < aParamsW.aRowSize; i++ ){
			for( j = 0; j < aParamsW.aColSize; j++ ){
		 		if( rThreshold < (grad_approx_W.aMatrix[i][j] - grad_computed_W.aMatrix[i][j]) ){
			        printf("grad_approx_W: %g\n", grad_approx_W.aMatrix[i][j] );
			        printf("grad_computed_W:  %g\n", grad_computed_W.aMatrix[i][j] );
			    }
		    }
		}
    }

    ////// check U's gradients /////
    MatrixDoubleType grad_computed_U(aParamsU.aRowSize, aParamsU.aColSize);
	MatrixDoubleType grad_approx_U(aParamsU.aRowSize, aParamsU.aColSize);
	grad_computed_U = aGradU;
	for(i = 0; i < aParamsU.aRowSize; i++ ){
		for( j = 0; j < aParamsU.aColSize; j++ ){
			temp = aParamsU.aMatrix[i][j];
			aParamsU.aMatrix[i][j] = temp + rEps;
			loss_plus = computeLoss(rXs, rYs);
			aParamsU.aMatrix[i][j] = temp - rEps;
			loss_minus = computeLoss(rXs, rYs);
			aParamsU.aMatrix[i][j] = temp;
			grad_approx_U.aMatrix[i][j] = (loss_plus - loss_minus) / (2*rEps);
		}
	}

	// print gradient check result ( [ok] or [Error])
	grad_delta = YdMm.length(grad_approx_U - grad_computed_U);
	printf("grad_check: dJ/d%s error norm = %.04g \n", "aParamsW" , grad_delta);
	if ( rThreshold > grad_delta ) {
		printf("[ok]\n");
	} else {
		printf("** ERROR **");
	}

	// when you set rVerbose 'true', you can see detailed information of which element
	// generates gradient's error and its value.
    if ( ( false != rVerbose ) && (grad_delta > rThreshold) ){
    	for(i = 0; i < aParamsU.aRowSize; i++ ){
			for( j = 0; j < aParamsU.aColSize; j++ ){
		 		if( rThreshold < (grad_approx_U.aMatrix[i][j] - grad_computed_U.aMatrix[i][j]) ){
			        printf("grad_approx_U: %g\n", grad_approx_U.aMatrix[i][j] );
			        printf("grad_computed_U:  %g\n", grad_computed_U.aMatrix[i][j] );
			    }
		    }
		}
    }

    ///// check Bias1's gradients /////
    MatrixDoubleType grad_computed_B1(aParamsB1.aRowSize, aParamsB1.aColSize);
	MatrixDoubleType grad_approx_B1(aParamsB1.aRowSize, aParamsB1.aColSize);
	grad_computed_B1 = aGradB1;
	for(i = 0; i < aParamsB1.aRowSize; i++ ){
		for( j = 0; j < aParamsB1.aColSize; j++ ){
			temp = aParamsB1.aMatrix[i][j];
			aParamsB1.aMatrix[i][j] = temp + rEps;
			loss_plus = computeLoss(rXs, rYs);
			aParamsB1.aMatrix[i][j] = temp - rEps;
			loss_minus = computeLoss(rXs, rYs);
			aParamsB1.aMatrix[i][j] = temp;
			grad_approx_B1.aMatrix[i][j] = (loss_plus - loss_minus) / (2*rEps);
		}
	}

	// print gradient check result ( [ok] or [Error])
	grad_delta = YdMm.length(grad_approx_B1 - grad_computed_B1);
	printf("grad_check: dJ/d%s error norm = %.04g \n", "aParamsB1" , grad_delta);
	if ( rThreshold > grad_delta ) {
		printf("[ok]\n");
	} else {
		printf("** ERROR **");
	}

	// when you set rVerbose 'true', you can see detailed information of which element
	// generates gradient's error and its value.
    if ( ( false != rVerbose ) && (grad_delta > rThreshold) ){
    	for(i = 0; i < aParamsB1.aRowSize; i++ ){
			for( j = 0; j < aParamsB1.aColSize; j++ ){
		 		if( rThreshold < (grad_approx_B1.aMatrix[i][j] - grad_computed_B1.aMatrix[i][j]) ){
			        printf("grad_approx_B1: %g\n", grad_approx_B1.aMatrix[i][j] );
			        printf("grad_computed_B1:  %g\n", grad_computed_B1.aMatrix[i][j] );
			    }
		    }
		}
    }

    ///// check Bias2's gradients /////
    MatrixDoubleType grad_computed_B2(aParamsB2.aRowSize, aParamsB2.aColSize);
	MatrixDoubleType grad_approx_B2(aParamsB2.aRowSize, aParamsB2.aColSize);
	grad_computed_B2 = aGradB2;
	for(i = 0; i < aParamsB2.aRowSize; i++ ){
		for( j = 0; j < aParamsB2.aColSize; j++ ){
			temp = aParamsB2.aMatrix[i][j];
			aParamsB2.aMatrix[i][j] = temp + rEps;
			loss_plus = computeLoss(rXs, rYs);
			aParamsB2.aMatrix[i][j] = temp - rEps;
			loss_minus = computeLoss(rXs, rYs);
			aParamsB2.aMatrix[i][j] = temp;
			grad_approx_B2.aMatrix[i][j] = (loss_plus - loss_minus) / (2*rEps);
		}
	}

	// print gradient check result ( [ok] or [Error])
	grad_delta = YdMm.length(grad_approx_B2 - grad_computed_B2);
	printf("grad_check: dJ/d%s error norm = %.04g \n", "aParamsB2" , grad_delta);
	if ( rThreshold > grad_delta ) {
		printf("[ok]\n");
	} else {
		printf("** ERROR **");
	}

	// when you set rVerbose 'true', you can see detailed information of which element
	// generates gradient's error and its value.
    if ( ( false != rVerbose ) && (grad_delta > rThreshold) ){
    	for(i = 0; i < aParamsB2.aRowSize; i++ ){
			for( j = 0; j < aParamsB2.aColSize; j++ ){
		 		if( rThreshold < (grad_approx_B2.aMatrix[i][j] - grad_computed_B2.aMatrix[i][j]) ){
			        printf("grad_approx_B2: %g\n", grad_approx_B2.aMatrix[i][j] );
			        printf("grad_computed_B2:  %g\n", grad_computed_B2.aMatrix[i][j] );
			    }
		    }
		}
    }

    //////////////////////////////////////////////////////////////////////////////////
    ///////////// check Sparse parameter's gradients ( Word Vectors ) ////////////////
    // find updated key (= words)
	int sgradL_key;
	int row_size = aUpdatedSgradLKeys.size();
	int col_size = aDims[0];
	MatrixDoubleType grad_computed_L(row_size, col_size);
	MatrixDoubleType grad_approx_L(row_size, col_size);
	MatrixDoubleType grad_temp_L(row_size, col_size);
	vector< int > vTempKey;
	// make temp grad matrix which contains only updated words (cf. total amount words is too big )
	i = 0;
	for (map<int, int>::iterator it=aUpdatedSgradLKeys.begin(); it!=aUpdatedSgradLKeys.end(); ++it){
		sgradL_key = it->first;
		for( j = 0; j < col_size; j++ ){
			grad_temp_L.aMatrix[i][j] = aSparamsL[sgradL_key].aMatrix[0][j];
			grad_computed_L.aMatrix[i][j] = aSgradL[sgradL_key].aMatrix[0][j];
		}
		vTempKey.push_back(sgradL_key);
		i++;
	}

	// check gradients for these key (= words)
	for(i = 0; i < row_size; i++ ){
		for( j = 0; j < col_size; j++ ){
			temp = grad_temp_L.aMatrix[i][j];
			aSparamsL[vTempKey[i]].aMatrix[0][j] = temp + rEps;
			loss_plus = computeLoss(rXs, rYs);
			aSparamsL[vTempKey[i]].aMatrix[0][j] = temp - rEps;
			loss_minus = computeLoss(rXs, rYs);
			aSparamsL[vTempKey[i]].aMatrix[0][j] = temp;
			grad_approx_L.aMatrix[i][j] = (loss_plus - loss_minus) / (2*rEps);
		}
	}

	// print gradient check result ( [ok] or [Error])
	grad_delta = YdMm.length(grad_approx_L - grad_computed_L);
	printf("grad_check: dJ/d%s error norm = %.04g \n", "aSparamsL" , grad_delta);
	if ( rThreshold > grad_delta ) {
		printf("[ok]\n");
	} else {
		printf("** ERROR **");
	}

	// when you set rVerbose 'true', you can see detailed information of which element
	// generates gradient's error and its value.
    if ( ( false != rVerbose ) && (grad_delta > rThreshold) ){
    	for(i = 0; i < row_size; i++ ){
			for( j = 0; j < col_size; j++ ){
		 		if( rThreshold < (grad_approx_L.aMatrix[i][j] - grad_computed_L.aMatrix[i][j]) ){
			        printf("grad_approx_L: %g\n", grad_approx_L.aMatrix[i][j] );
			        printf("grad_computed_L:  %g\n", grad_computed_L.aMatrix[i][j] );
			    }
		    }
		}
    }

	resetAccGrads();
}