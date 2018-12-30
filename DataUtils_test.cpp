/** @file DataUtils_test.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module DataUtils_test
 *  @version 1.0
 *  @brief gather common DataUtils_test includes and functions.
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
#include "DataUtils.h"

int main(int argc, char *argv[]){

	printf("\n\n/////////////// TEST DataUtils //////////////////////\n");

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
	YdDu.loadWordVec( word_vec_file_name, word_vec, 50);

	printf("word_vec size = %ld\n", word_vec.size() );
	for( i = 0; i < 3; i++ ){
		word_vec[i].display();
		word_vec[i].shape();
	}

	printf("\n\n/////////////// TEST invertDict //////////////////\n");
	YdDu.loadVoca( voca_file_name, num_to_word, word_to_num );

	printf("num_to_word size = %ld\n", num_to_word.size() );
	printf("word_to_num size = %ld\n", word_to_num.size() );

	for( i = 0; i < 10; i++ ){
		cout << "num_to_word[" << i << "] = " << num_to_word[i] << endl;
	}

	j = 0;
	for ( map< int, string >::iterator it=num_to_word.begin() ; it!=num_to_word.end(); ++it){
		cout << "num_to_word[" << it->first << "] = " << it->second << endl;
		cout << "word_to_num[" << it->second << "] = " << word_to_num[it->second] << endl;
		j += 1;
		if ( 10 == j) break;
	}

	j = 0;
	for ( map< string, int >::iterator it=word_to_num.begin() ; it!=word_to_num.end(); ++it){
		cout << "word_to_num[" << it->first << "] = " << it->second << endl;
		j += 1;
		if ( 10 == j) break;
	}

	printf("\n\n/////////////// TEST loadDataset train ///////////////////\n");
	YdDu.setTags( num_to_tag, tag_to_num );
	printf("After set tags \n");

	YdDu.loadDataset( train_data_file_name, docs );
	for( i = 0; i < 10; i++ ){
		for( j = 0; j < 2; j++ ){
			cout << "docs[" << i << "][" << j << "] = " << docs[i][j][0].aWord << "   " << docs[i][j][0].aLabel << endl;
		}
	}

	YdDu.makeDocsToWindows( X_train, Y_train, docs, word_to_num,
			tag_to_num, window_size );

	for ( i = 0; i < 10 ; i++ ){
		for ( j = 0 ; j < window_size ; j++ ){
			printf("%d ", X_train.aMatrix[i][j]);
		}
		printf("\n label = %d \n", Y_train.aMatrix[0][i] );
	}

	X_train.shape();
	Y_train.shape();
	for ( j = 0 ; j < window_size ; j++ ){
		printf("%d ", X_train.aMatrix[55][j]);
	}
	for ( j = 0 ; j < window_size ; j++ ){
		printf("%d ", X_train.aMatrix[555][j]);
	}
	for ( j = 0 ; j < window_size ; j++ ){
		printf("%d ", X_train.aMatrix[5555][j]);
	}
	for ( j = 0 ; j < window_size ; j++ ){
		printf("%d ", X_train.aMatrix[55555][j]);
	}
	for ( j = 0 ; j < window_size ; j++ ){
		printf("%d ", X_train.aMatrix[200000][j]);
	}
	for ( j = 0 ; j < window_size ; j++ ){
		printf("%d ", X_train.aMatrix[203620][j]);
	}

	printf("\n\n/////////////// TEST loadDataset dev ///////////////////\n");
	YdDu.loadDataset( dev_data_file_name, docs );
	for( i = 0; i < 10; i++ ){
		for( j = 0; j < 2; j++ ){
			cout << "docs[" << i << "][" << j << "] = " << docs[i][j][0].aWord << "   " << docs[i][j][1].aLabel << endl;
		}
	}

	YdDu.makeDocsToWindows( X_dev, Y_dev, docs, word_to_num,
			tag_to_num, window_size );

	for ( i = 0; i < 10 ; i++ ){
		for ( j = 0 ; j < window_size ; j++ ){
			printf("%d ", X_dev.aMatrix[i][j]);
		}
		printf("\n label = %d \n", Y_dev.aMatrix[0][i] );
	}

	X_dev.shape();
	Y_dev.shape();


	printf("\n\n/////////////// TEST loadDataset test ///////////////////\n");
	YdDu.loadDataset( test_data_file_name, docs );
	for( i = 0; i < 10; i++ ){
		for( j = 0; j < 2; j++ ){
			cout << "docs[" << i << "][" << j << "] = " << docs[i][j][0].aWord << "   " << docs[i][j][1].aLabel << endl;
		}
	}

	YdDu.makeDocsToWindows( X_test, Y_test, docs, word_to_num,
			tag_to_num, window_size );

	for ( i = 0; i < 10 ; i++ ){
		for ( j = 0 ; j < window_size ; j++ ){
			printf("%d ", X_test.aMatrix[i][j]);
		}
		printf("\n label = %d \n", Y_test.aMatrix[0][i] );
	}

	X_test.shape();

	printf("\n\n/////////////// TEST makeDocsToWindows /////////////////////\n");


	return 0;
}