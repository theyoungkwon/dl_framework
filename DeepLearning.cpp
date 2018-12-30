/** @file DeepLearning.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module DeepLearning
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

#include "DeepLearning.h"

int main(int argc, char* argv[]){

	// deep_learning.gradCheck();
	DeepLearning deep_learning1;
	deep_learning1.run1();

	DeepLearning deep_learning2;
	deep_learning2.run2();

	DeepLearning deep_learning3;
	deep_learning3.run3();

	DeepLearning deep_learning4;
	deep_learning4.run_100_01_5();

	DeepLearning deep_learning5;
	deep_learning5.run_100_1_5();

	DeepLearning deep_learning6;
	deep_learning6.run_150_01();

	return 0;
}

DeepLearning::DeepLearning(){ }

DeepLearning::~DeepLearning(){ }

void DeepLearning::run1(){

	int dims[] = {50, 100, 5};
	double lambda = 0.001;
	double alpha = 0.01;
	int seed = 20;
	int window_size = 3;

	DataUtils YdDu;
	map< int, MatrixDoubleType > word_vec;
	map< int, string > num_to_word;
	map< string, int > word_to_num;
	map< int, string > num_to_tag;
	map< string, int > tag_to_num;
	vector< vector< vector< WordLabelType > > > docs;

	MatrixIntType X_train;
	MatrixIntType X_dev;
	MatrixIntType X_test;
	MatrixIntType Y_train;
	MatrixIntType Y_dev;
	MatrixIntType Y_test;

	string word_vec_file_name = "data/wordVectors.txt";
	string voca_file_name = "data/vocab.txt";
	string train_data_file_name = "data/train";
	string dev_data_file_name = "data/dev";
	string test_data_file_name = "data/test.masked";


	YdDu.loadWordVec( word_vec_file_name, word_vec, dims[0] );

	YdDu.loadVoca( voca_file_name, num_to_word, word_to_num );
	YdDu.setTags( num_to_tag, tag_to_num );

	YdDu.loadDataset( train_data_file_name, docs );
	YdDu.makeDocsToWindows( X_train, Y_train, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( dev_data_file_name, docs );
	YdDu.makeDocsToWindows( X_dev, Y_dev, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( test_data_file_name, docs );
	YdDu.makeDocsToWindows( X_test, Y_test, docs, word_to_num,
			tag_to_num, window_size );

	aNer.init( word_vec, window_size, dims,
		 	lambda, alpha, seed );


	printf("\n===== Training Started... =====\n");
	vector< double > losses;
	int print_every = 50000;
	int loss_every = 10000;

	int train_total_size = Y_train.aColSize;
	int dev_total_size = Y_dev.aColSize;
	int epoch = 1;
	int batch_size = 1;
	int i , j;
	// plain iteration
	MatrixIntType idx_iter(epoch*train_total_size, batch_size);
	MatrixDoubleType alpha_iter(1, epoch*train_total_size);
	// MatrixIntType dev_iter(epoch*dev_total_size, batch_size);
	MatrixIntType dev_iter;

	// train iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < train_total_size ; j++ ){
			idx_iter.aMatrix[i*train_total_size + j][0] = (i*train_total_size + j)%train_total_size;
			alpha_iter.aMatrix[0][i*train_total_size + j] = 0.1;
		}
	}

	// dev iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < dev_total_size ; j++ ){
			dev_iter.aMatrix[i*dev_total_size + j][0] = (i*dev_total_size + j)%dev_total_size;
		}
	}

	// train the model using sgd
	losses = aNer.trainSgd( X_train, Y_train, idx_iter, alpha_iter,
		 print_every, loss_every, dev_iter );

	printf("\n===== End of Training =====\n");
	for ( i = 0; i < losses.size(); i++ ){
		printf("%lf\n", losses[i] );
	}

	// predict labels
	printf("before init predictlab\n");
	MatrixIntType predicted_labels(dev_total_size, 1);
	printf("before predictLabels \n");
	predicted_labels = aNer.predictLabels(X_dev);
	// print performance
	printf("before evalPerfom\n");
	YdDu.evalPerformance(Y_dev, predicted_labels, num_to_tag);

}


void DeepLearning::run2(){

	int dims[] = {50, 100, 5};
	double lambda = 0.001;
	double alpha = 0.01;
	int seed = 20;
	int window_size = 3;

	DataUtils YdDu;
	map< int, MatrixDoubleType > word_vec;
	map< int, string > num_to_word;
	map< string, int > word_to_num;
	map< int, string > num_to_tag;
	map< string, int > tag_to_num;
	vector< vector< vector< WordLabelType > > > docs;

	MatrixIntType X_train;
	MatrixIntType X_dev;
	MatrixIntType X_test;
	MatrixIntType Y_train;
	MatrixIntType Y_dev;
	MatrixIntType Y_test;

	string word_vec_file_name = "data/wordVectors.txt";
	string voca_file_name = "data/vocab.txt";
	string train_data_file_name = "data/train";
	string dev_data_file_name = "data/dev";
	string test_data_file_name = "data/test.masked";


	YdDu.loadWordVec( word_vec_file_name, word_vec, dims[0] );

	YdDu.loadVoca( voca_file_name, num_to_word, word_to_num );
	YdDu.setTags( num_to_tag, tag_to_num );

	YdDu.loadDataset( train_data_file_name, docs );
	YdDu.makeDocsToWindows( X_train, Y_train, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( dev_data_file_name, docs );
	YdDu.makeDocsToWindows( X_dev, Y_dev, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( test_data_file_name, docs );
	YdDu.makeDocsToWindows( X_test, Y_test, docs, word_to_num,
			tag_to_num, window_size );

	aNer.init( word_vec, window_size, dims,
		 	lambda, alpha, seed );


	printf("\n===== Training Started... =====\n");
	vector< double > losses;
	int print_every = 50000;
	int loss_every = 10000;

	int train_total_size = Y_train.aColSize;
	int dev_total_size = Y_dev.aColSize;
	int epoch = 1;
	int batch_size = 1;
	int i , j;
	// plain iteration
	MatrixIntType idx_iter(epoch*train_total_size, batch_size);
	MatrixDoubleType alpha_iter(1, epoch*train_total_size);
	// MatrixIntType dev_iter(epoch*dev_total_size, batch_size);
	MatrixIntType dev_iter;

	// train iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < train_total_size ; j++ ){
			idx_iter.aMatrix[i*train_total_size + j][0] = (i*train_total_size + j)%train_total_size;
			alpha_iter.aMatrix[0][i*train_total_size + j] = 0.03;
		}
	}

	// dev iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < dev_total_size ; j++ ){
			dev_iter.aMatrix[i*dev_total_size + j][0] = (i*dev_total_size + j)%dev_total_size;
		}
	}

	// train the model using sgd
	losses = aNer.trainSgd( X_train, Y_train, idx_iter, alpha_iter,
		 print_every, loss_every, dev_iter );

	printf("\n===== End of Training =====\n");
	for ( i = 0; i < losses.size(); i++ ){
		printf("%lf\n", losses[i] );
	}

	// predict labels
	MatrixIntType predicted_labels(dev_total_size, 1);

	predicted_labels = aNer.predictLabels(X_dev);
	// print performance
	YdDu.evalPerformance(Y_dev, predicted_labels, num_to_tag);

}

void DeepLearning::run3(){

	int dims[] = {50, 100, 5};
	double lambda = 0.001;
	double alpha = 0.01;
	int seed = 20;
	int window_size = 3;

	DataUtils YdDu;
	map< int, MatrixDoubleType > word_vec;
	map< int, string > num_to_word;
	map< string, int > word_to_num;
	map< int, string > num_to_tag;
	map< string, int > tag_to_num;
	vector< vector< vector< WordLabelType > > > docs;

	MatrixIntType X_train;
	MatrixIntType X_dev;
	MatrixIntType X_test;
	MatrixIntType Y_train;
	MatrixIntType Y_dev;
	MatrixIntType Y_test;

	string word_vec_file_name = "data/wordVectors.txt";
	string voca_file_name = "data/vocab.txt";
	string train_data_file_name = "data/train";
	string dev_data_file_name = "data/dev";
	string test_data_file_name = "data/test.masked";


	YdDu.loadWordVec( word_vec_file_name, word_vec, dims[0] );

	YdDu.loadVoca( voca_file_name, num_to_word, word_to_num );
	YdDu.setTags( num_to_tag, tag_to_num );

	YdDu.loadDataset( train_data_file_name, docs );
	YdDu.makeDocsToWindows( X_train, Y_train, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( dev_data_file_name, docs );
	YdDu.makeDocsToWindows( X_dev, Y_dev, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( test_data_file_name, docs );
	YdDu.makeDocsToWindows( X_test, Y_test, docs, word_to_num,
			tag_to_num, window_size );

	aNer.init( word_vec, window_size, dims,
		 	lambda, alpha, seed );


	printf("\n===== Training Started... =====\n");
	vector< double > losses;
	int print_every = 50000;
	int loss_every = 10000;

	int train_total_size = Y_train.aColSize;
	int dev_total_size = Y_dev.aColSize;
	int epoch = 1;
	int batch_size = 1;
	int i , j;
	// plain iteration
	MatrixIntType idx_iter(epoch*train_total_size, batch_size);
	MatrixDoubleType alpha_iter(1, epoch*train_total_size);
	// MatrixIntType dev_iter(epoch*dev_total_size, batch_size);
	MatrixIntType dev_iter;

	// train iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < train_total_size ; j++ ){
			idx_iter.aMatrix[i*train_total_size + j][0] = (i*train_total_size + j)%train_total_size;
			alpha_iter.aMatrix[0][i*train_total_size + j] = 0.01;
		}
	}

	// dev iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < dev_total_size ; j++ ){
			dev_iter.aMatrix[i*dev_total_size + j][0] = (i*dev_total_size + j)%dev_total_size;
		}
	}

	// train the model using sgd
	losses = aNer.trainSgd( X_train, Y_train, idx_iter, alpha_iter,
		 print_every, loss_every, dev_iter );

	printf("\n===== End of Training =====\n");
	for ( i = 0; i < losses.size(); i++ ){
		printf("%lf\n", losses[i] );
	}

	// predict labels
	MatrixIntType predicted_labels(dev_total_size, 1);

	predicted_labels = aNer.predictLabels(X_dev);
	// print performance
	YdDu.evalPerformance(Y_dev, predicted_labels, num_to_tag);

}

void DeepLearning::run_100_01_5(){

	int dims[] = {50, 100, 5};
	double lambda = 0.001;
	double alpha = 0.01;
	int seed = 20;
	int window_size = 3;

	DataUtils YdDu;
	map< int, MatrixDoubleType > word_vec;
	map< int, string > num_to_word;
	map< string, int > word_to_num;
	map< int, string > num_to_tag;
	map< string, int > tag_to_num;
	vector< vector< vector< WordLabelType > > > docs;

	MatrixIntType X_train;
	MatrixIntType X_dev;
	MatrixIntType X_test;
	MatrixIntType Y_train;
	MatrixIntType Y_dev;
	MatrixIntType Y_test;

	string word_vec_file_name = "data/wordVectors.txt";
	string voca_file_name = "data/vocab.txt";
	string train_data_file_name = "data/train";
	string dev_data_file_name = "data/dev";
	string test_data_file_name = "data/test.masked";


	YdDu.loadWordVec( word_vec_file_name, word_vec, dims[0] );

	YdDu.loadVoca( voca_file_name, num_to_word, word_to_num );
	YdDu.setTags( num_to_tag, tag_to_num );

	YdDu.loadDataset( train_data_file_name, docs );
	YdDu.makeDocsToWindows( X_train, Y_train, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( dev_data_file_name, docs );
	YdDu.makeDocsToWindows( X_dev, Y_dev, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( test_data_file_name, docs );
	YdDu.makeDocsToWindows( X_test, Y_test, docs, word_to_num,
			tag_to_num, window_size );

	aNer.init( word_vec, window_size, dims,
		 	lambda, alpha, seed );


	printf("\n===== Training Started... =====\n");
	vector< double > losses;
	int print_every = 50000;
	int loss_every = 10000;

	int train_total_size = Y_train.aColSize;
	int dev_total_size = Y_dev.aColSize;
	int epoch = 1;
	int batch_size = 5;
	int i , j;
	// plain iteration
	MatrixIntType idx_iter(epoch*train_total_size, batch_size);
	MatrixDoubleType alpha_iter(1, epoch*train_total_size);
	// MatrixIntType dev_iter(epoch*dev_total_size, batch_size);
	MatrixIntType dev_iter;

	// train iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < train_total_size ; j++ ){
			idx_iter.aMatrix[i*train_total_size + j][0] = (i*train_total_size + j)%train_total_size;
			alpha_iter.aMatrix[0][i*train_total_size + j] = 0.01;
		}
	}

	// dev iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < dev_total_size ; j++ ){
			dev_iter.aMatrix[i*dev_total_size + j][0] = (i*dev_total_size + j)%dev_total_size;
		}
	}

	// train the model using sgd
	losses = aNer.trainSgd( X_train, Y_train, idx_iter, alpha_iter,
		 print_every, loss_every, dev_iter );

	printf("\n===== End of Training =====\n");
	for ( i = 0; i < losses.size(); i++ ){
		printf("%lf\n", losses[i] );
	}

	// predict labels
	MatrixIntType predicted_labels(dev_total_size, 1);

	predicted_labels = aNer.predictLabels(X_dev);
	// print performance
	YdDu.evalPerformance(Y_dev, predicted_labels, num_to_tag);

}

void DeepLearning::run_100_1_5(){

	int dims[] = {50, 100, 5};
	double lambda = 0.001;
	double alpha = 0.1;
	int seed = 20;
	int window_size = 3;

	DataUtils YdDu;
	map< int, MatrixDoubleType > word_vec;
	map< int, string > num_to_word;
	map< string, int > word_to_num;
	map< int, string > num_to_tag;
	map< string, int > tag_to_num;
	vector< vector< vector< WordLabelType > > > docs;

	MatrixIntType X_train;
	MatrixIntType X_dev;
	MatrixIntType X_test;
	MatrixIntType Y_train;
	MatrixIntType Y_dev;
	MatrixIntType Y_test;

	string word_vec_file_name = "data/wordVectors.txt";
	string voca_file_name = "data/vocab.txt";
	string train_data_file_name = "data/train";
	string dev_data_file_name = "data/dev";
	string test_data_file_name = "data/test.masked";


	YdDu.loadWordVec( word_vec_file_name, word_vec, dims[0] );

	YdDu.loadVoca( voca_file_name, num_to_word, word_to_num );
	YdDu.setTags( num_to_tag, tag_to_num );

	YdDu.loadDataset( train_data_file_name, docs );
	YdDu.makeDocsToWindows( X_train, Y_train, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( dev_data_file_name, docs );
	YdDu.makeDocsToWindows( X_dev, Y_dev, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( test_data_file_name, docs );
	YdDu.makeDocsToWindows( X_test, Y_test, docs, word_to_num,
			tag_to_num, window_size );

	aNer.init( word_vec, window_size, dims,
		 	lambda, alpha, seed );


	printf("\n===== Training Started... =====\n");
	vector< double > losses;
	int print_every = 50000;
	int loss_every = 10000;

	int train_total_size = Y_train.aColSize;
	int dev_total_size = Y_dev.aColSize;
	int epoch = 1;
	int batch_size = 5;
	int i , j;
	// plain iteration
	MatrixIntType idx_iter(epoch*train_total_size, batch_size);
	MatrixDoubleType alpha_iter(1, epoch*train_total_size);
	// MatrixIntType dev_iter(epoch*dev_total_size, batch_size);
	MatrixIntType dev_iter;

	// train iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < train_total_size ; j++ ){
			idx_iter.aMatrix[i*train_total_size + j][0] = (i*train_total_size + j)%train_total_size;
			alpha_iter.aMatrix[0][i*train_total_size + j] = 0.1;
		}
	}

	// dev iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < dev_total_size ; j++ ){
			dev_iter.aMatrix[i*dev_total_size + j][0] = (i*dev_total_size + j)%dev_total_size;
		}
	}

	// train the model using sgd
	losses = aNer.trainSgd( X_train, Y_train, idx_iter, alpha_iter,
		 print_every, loss_every, dev_iter );

	printf("\n===== End of Training =====\n");
	for ( i = 0; i < losses.size(); i++ ){
		printf("%lf\n", losses[i] );
	}

	// predict labels
	MatrixIntType predicted_labels(dev_total_size, 1);

	predicted_labels = aNer.predictLabels(X_dev);
	// print performance
	YdDu.evalPerformance(Y_dev, predicted_labels, num_to_tag);

}

void DeepLearning::run_150_01(){

	int dims[] = {50, 150, 5};
	double lambda = 0.001;
	double alpha = 0.01;
	int seed = 20;
	int window_size = 3;

	DataUtils YdDu;
	map< int, MatrixDoubleType > word_vec;
	map< int, string > num_to_word;
	map< string, int > word_to_num;
	map< int, string > num_to_tag;
	map< string, int > tag_to_num;
	vector< vector< vector< WordLabelType > > > docs;

	MatrixIntType X_train;
	MatrixIntType X_dev;
	MatrixIntType X_test;
	MatrixIntType Y_train;
	MatrixIntType Y_dev;
	MatrixIntType Y_test;

	string word_vec_file_name = "data/wordVectors.txt";
	string voca_file_name = "data/vocab.txt";
	string train_data_file_name = "data/train";
	string dev_data_file_name = "data/dev";
	string test_data_file_name = "data/test.masked";


	YdDu.loadWordVec( word_vec_file_name, word_vec, dims[0] );

	YdDu.loadVoca( voca_file_name, num_to_word, word_to_num );
	YdDu.setTags( num_to_tag, tag_to_num );

	YdDu.loadDataset( train_data_file_name, docs );
	YdDu.makeDocsToWindows( X_train, Y_train, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( dev_data_file_name, docs );
	YdDu.makeDocsToWindows( X_dev, Y_dev, docs, word_to_num,
			tag_to_num, window_size );

	YdDu.loadDataset( test_data_file_name, docs );
	YdDu.makeDocsToWindows( X_test, Y_test, docs, word_to_num,
			tag_to_num, window_size );

	aNer.init( word_vec, window_size, dims,
		 	lambda, alpha, seed );


	printf("\n===== Training Started... =====\n");
	vector< double > losses;
	int print_every = 50000;
	int loss_every = 10000;

	int train_total_size = Y_train.aColSize;
	int dev_total_size = Y_dev.aColSize;
	int epoch = 1;
	int batch_size = 5;
	int i , j;
	// plain iteration
	MatrixIntType idx_iter(epoch*train_total_size, batch_size);
	MatrixDoubleType alpha_iter(1, epoch*train_total_size);
	// MatrixIntType dev_iter(epoch*dev_total_size, batch_size);
	MatrixIntType dev_iter;

	// train iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < train_total_size ; j++ ){
			idx_iter.aMatrix[i*train_total_size + j][0] = (i*train_total_size + j)%train_total_size;
			alpha_iter.aMatrix[0][i*train_total_size + j] = 0.01;
		}
	}

	// dev iteration init
	for ( i = 0; i < epoch; i++ ){
		for ( j = 0; j < dev_total_size ; j++ ){
			dev_iter.aMatrix[i*dev_total_size + j][0] = (i*dev_total_size + j)%dev_total_size;
		}
	}

	// train the model using sgd
	losses = aNer.trainSgd( X_train, Y_train, idx_iter, alpha_iter,
		 print_every, loss_every, dev_iter );

	printf("\n===== End of Training =====\n");
	for ( i = 0; i < losses.size(); i++ ){
		printf("%lf\n", losses[i] );
	}

	// predict labels
	MatrixIntType predicted_labels(dev_total_size, 1);

	predicted_labels = aNer.predictLabels(X_dev);
	// print performance
	YdDu.evalPerformance(Y_dev, predicted_labels, num_to_tag);

}

void DeepLearning::gradCheck(){

	int dims[] = {50, 100, 5};
	double lambda = 0.001;
	double alpha = 0.01;
	int seed = 20;

	DataUtils YdDu;
	map< int, MatrixDoubleType > word_vec;
	map< int, string > num_to_word;
	map< string, int > word_to_num;
	map< int, string > num_to_tag;
	map< string, int > tag_to_num;
	vector< vector< vector< WordLabelType > > > docs;

	MatrixIntType X_train;
	MatrixIntType X_dev;
	MatrixIntType X_test;
	MatrixIntType Y_train;
	MatrixIntType Y_dev;
	MatrixIntType Y_test;

	string word_vec_file_name = "data/wordVectors.txt";
	string voca_file_name = "data/vocab.txt";
	string train_data_file_name = "data/train";
	string dev_data_file_name = "data/dev";
	string test_data_file_name = "data/test.masked";

	int window_size = 3;
	int i, j;
	string str;

	printf("\n\n/////////////// TEST loadWordVec ///////////////////\n");
	YdDu.loadWordVec( word_vec_file_name, word_vec, dims[0] );

	printf("\n\n/////////////// TEST invertDict //////////////////\n");
	YdDu.loadVoca( voca_file_name, num_to_word, word_to_num );
	YdDu.setTags( num_to_tag, tag_to_num );

	printf("\n\n/////////////// TEST loadDataset train ///////////////////\n");
	YdDu.loadDataset( train_data_file_name, docs );
	YdDu.makeDocsToWindows( X_train, Y_train, docs, word_to_num,
			tag_to_num, window_size );

	printf("\n\n/////////////// TEST loadDataset dev ///////////////////\n");
	YdDu.loadDataset( dev_data_file_name, docs );
	YdDu.makeDocsToWindows( X_dev, Y_dev, docs, word_to_num,
			tag_to_num, window_size );

	printf("\n\n/////////////// TEST loadDataset test ///////////////////\n");
	YdDu.loadDataset( test_data_file_name, docs );
	YdDu.makeDocsToWindows( X_test, Y_test, docs, word_to_num,
			tag_to_num, window_size );

	aNer.init( word_vec, window_size, dims,
		 	lambda, alpha, seed );

	MatrixIntType X(1, 3);
	MatrixIntType Y(1,1);

	X.aMatrix[0][0] = 30;
	X.aMatrix[0][1] = 6659;
	X.aMatrix[0][2] = 12637;
	Y.aMatrix[0][0] = 3;

	aNer.gradCheck(X, Y , 0.0001,
		 			0.000001, true);

}

int DeepLearning::printPerformance( char ch ){

}
