/** @file setup_test.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module setup_test
 *  @version 1.0
 *  @brief gather common setup_test includes and functions.
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
#include "NeuronUnits.h"
#include "Derivatives.h"
#include "LossFunctions.h"
#include "MathModule.h"
#include "Ner.h"
#include "RandomModule.h"

int main(int argc, char *argv[]){
	MatrixDoubleType mat1_3(3);
	MatrixDoubleType mat3_1(3, 1);
	MatrixDoubleType mat1_4(4);
	MatrixDoubleType mat4_1(4, 1);
	MatrixDoubleType mat1_5(5);
	MatrixDoubleType mat5_1(5, 1);
	MatrixDoubleType mat3(3, 3);
	MatrixDoubleType mat33(3, 3);

	MatrixDoubleType mat4(4, 4);
	MatrixDoubleType mat44(4, 4);
	MatrixDoubleType mat444(4, 4);

	MatrixDoubleType mat45(4, 5);
	MatrixDoubleType mat54(5, 4);

	printf("\n\n/////////////// TEST +, += //////////////////\n");
	mat1_3 += 1.0;
	mat1_4 += 4.0;
	mat1_5 += 5.0;


	mat1_3.display();
	mat1_3.shape();
	mat1_4.display();
	mat1_4.shape();
	mat1_5.display();
	mat1_5.shape();


	mat4 = mat4 + 1.0;
	mat4.display();
	mat4.shape();

	mat44 = mat44 + 1.0;
	mat44.display();
	mat44.shape();

	mat444 = mat4+mat44;
	mat444.display();
	mat444.shape();

	mat444 += mat44;
	mat444.display();
	mat444.shape();

	printf("\n\n/////////////// TEST -, -= //////////////////\n");

	mat1_3 -= 0.5;
	mat1_4 -= 2.0;
	mat1_5 -= 2.5;


	mat1_3.display();
	mat1_3.shape();
	mat1_4.display();
	mat1_4.shape();
	mat1_5.display();
	mat1_5.shape();


	mat4 = mat4 - 0.5;
	mat4.display();
	mat4.shape();

	mat44 = mat44 - 0.5;
	mat44.display();
	mat44.shape();

	mat444 = mat4-mat44;
	mat444.display();
	mat444.shape();

	mat444 -= mat44;
	mat444.display();
	mat444.shape();

	printf("\n\n/////////////// TEST *, *= //////////////////\n");

	mat1_3 *= 1.0;
	mat1_4 *= 4.0;
	mat1_5 *= 5.0;


	mat1_3.display();
	mat1_3.shape();
	mat1_4.display();
	mat1_4.shape();
	mat1_5.display();
	mat1_5.shape();


	mat4 = mat4 * 0.5;
	mat4.display();
	mat4.shape();

	mat44 = mat44 * 0.5;
	mat44.display();
	mat44.shape();

	mat444 = mat4*mat44;
	mat444.display();
	mat444.shape();

	mat444 *= mat44;
	mat444.display();
	mat444.shape();

	printf("\n\n/////////////// TEST transpose, dot //////////////////\n");

	mat3_1 = mat1_3.transpose();
	mat3_1.display();
	mat3_1.shape();

	mat4_1 = mat1_4.transpose();
	mat4_1.display();
	mat4_1.shape();

	mat45 += 2.0;
	MatrixDoubleType matdot = mat4.dot(mat45);
	matdot.display();
	matdot.shape();

	mat54 = mat45.transpose();
	mat54.display();
	mat54.shape();


	printf("\n\n/////////////// TEST sigmoid, gradient //////////////////\n");
	NeuronUnits YdNu;
	Derivatives YdDeri;

	MatrixDoubleType h1(mat4);
	MatrixDoubleType h1_grad(mat4);
	h1 += 0.5;
	h1 = YdNu.sigmoid(h1);
	h1.display();
	h1.shape();
	h1_grad = YdDeri.sigmoidGradient(h1);
	h1_grad.display();
	h1_grad.shape();

	printf("\n\n/////////////// TEST tanh, gradient //////////////////\n");

	MatrixDoubleType h2(mat4);
	MatrixDoubleType h2_grad(mat4);
	h2 += 0.5;
	h2 = YdNu.tanh(h2);
	h2.display();
	h2.shape();
	h2_grad = YdDeri.tanhGradient(h2);
	h2_grad.display();
	h2_grad.shape();

	printf("\n\n/////////////// TEST relu, gradient //////////////////\n");

	MatrixDoubleType h3(mat4);
	MatrixDoubleType h3_grad(mat4);
	h3 += 0.5;
	h3 = YdNu.relu(h3);
	h3.display();
	h3.shape();
	h3_grad = YdDeri.reluGradient(h3);
	h3_grad.display();
	h3_grad.shape();


	printf("\n\n/////////////// TEST negative log likelihood //////////////////\n");
	LossFunctions YdLf;

	double loss = 0.0;
	double loss_log = 0.0;
	MatrixDoubleType y(1,5);
	y.aMatrix[0][0] = 0.5;
	y.aMatrix[0][1] = 0.1;
	y.aMatrix[0][2] = 0.1;
	y.aMatrix[0][3] = 0.1;
	y.aMatrix[0][4] = 0.2;
	int label = 0;
	for( int i = 0; i < 1; i++ ){
		loss += -log( y.aMatrix[i][ label ] );
	}
	loss_log = -log(0.5);
	printf("%lf\n", loss);
	printf("%lf\n", loss_log);


	printf("\n\n/////////////// TEST square error //////////////////\n");
	MatrixIntType labels(1,1);

	loss = YdLf.squareError(y, labels);
	printf("%lf\n", loss);


	printf("\n\n/////////////// TEST regularization loss //////////////////\n");
	MathModule YdMm;
	double reg_loss = 0.0;
	double aLambda = 0.001;

	reg_loss = 0.5*aLambda*( YdMm.sum(mat1_5 * mat1_5) + YdMm.sum(mat1_5 * mat1_5) );
	printf("%lf\n", reg_loss);

	printf("\n\n/////////////// TEST softmax //////////////////\n");

	MatrixDoubleType matProbs(1,5);
	// y += 0.1;
	matProbs = YdMm.mysoftmax(y);
	matProbs.display();
	matProbs.shape();

	printf("\n\n/////////////// TEST argmax //////////////////\n");

	MatrixIntType matLabel(1,1);
	matLabel = YdMm.argmax(y, 1);
	matLabel.display();
	matLabel.shape();

	MatrixDoubleType matArgmax(2,4);
	MatrixIntType matArgmaxLabel(2,1);
	matArgmax += 0.2;
	matArgmax.aMatrix[0][1] += 0.2;
	matArgmax.aMatrix[1][1] += 0.2;
	matArgmax.aMatrix[0][2] += 0.4;
	matArgmax.aMatrix[1][2] += 0.4;
	matArgmaxLabel = YdMm.argmax(matArgmax, 1);
	matArgmaxLabel.display();
	matArgmaxLabel.shape();

	printf("\n\n/////////////// TEST sum //////////////////\n");

	double sum = 0.0;
	sum = YdMm.sum(y);
	printf("%lf\n", sum);

	sum = YdMm.sum(mat4);
	printf("%lf\n", sum);

	printf("\n\n/////////////// TEST length //////////////////\n");

	double length = 0.0;
	length = YdMm.length(y);
	printf("%lf\n", length);

	length = YdMm.length(mat4);
	printf("%lf\n", length);

	MatrixDoubleType mat22(2,2);
	mat22 += 2.0;
	length = YdMm.length(mat22);
	printf("%lf\n", length);


	//////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////
	printf("\n\n/////////////// TEST NER Model //////////////////////\n");
	map< int, MatrixDoubleType > wordvec;
	int window_size = 3;
	int dims[] = {50, 100, 5};
	double lambda = 0.001;
	double alpha = 0.01;
	int seed = 20;

	MatrixDoubleType matTemp(1,50);
	wordvec[0] = matTemp + 0.2;
	wordvec[1] = matTemp + 0.4;
	wordvec[2] = matTemp + 0.5;
	wordvec[3] = matTemp + 0.15;
	wordvec[4] = matTemp + 0.25;

	// wordvec[1] = aNer.YdRm.xavierDoubleInit(1,50);
	// wordvec[2] = aNer.YdRm.xavierDoubleInit(1,50);
	// wordvec[0].display();
	Ner aNer;
	aNer.init( wordvec, window_size, dims,
		 	lambda, alpha, seed );
	MatrixIntType X(1, 3);
	MatrixIntType Y(1,1);
	X.aMatrix[0][0] = 0;
	X.aMatrix[0][1] = 1;
	X.aMatrix[0][2] = 2;
	Y.aMatrix[0][0] = 3;

	MatrixDoubleType ner_probs(1,5);
	double ner_loss = 0.0;
	MatrixIntType ner_label(1,1);

	printf("\n\n/////////////// TEST predictProbs ///////////////////\n");
	ner_probs = aNer.predictProbs(X);
	ner_probs.display();
	ner_probs.shape();

	printf("\n\n/////////////// TEST predictLabels //////////////////\n");
	ner_label = aNer.predictLabels(X);
	ner_label.display();
	ner_label.shape();

	printf("\n\n/////////////// TEST computeLoss ///////////////////\n");
	ner_loss = aNer.computeLoss(X, Y);
	printf("loss = %lf\n", ner_loss);

	printf("\n\n/////////////// TEST gradCheck /////////////////////\n");
	aNer.gradCheck(X, Y , 0.0001,
		 			0.000001, true);


	return 0;
}