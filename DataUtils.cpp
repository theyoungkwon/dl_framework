/** @file DataUtils.cpp
 *
 *  System: 		Deep Learning for Natural Language Processing
 *  Component Name: Module DataUtils
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

#include "DataUtils.h"

DataUtils::DataUtils(){ }

DataUtils::~DataUtils(){ }

void DataUtils::loadWordVec( string rWordVecFileName,
				map< int, MatrixDoubleType>& rWordVec,
				int rDimSize = 50 ){

	fstream ifs;
	ifs.open ( rWordVecFileName, fstream::in );
	cout << rWordVecFileName << endl;
	int i = 0;
	int j = 0;
	double temp = 0.0;
	MatrixDoubleType matTemp(1, rDimSize);
	string line;
	string buf;
	string::size_type sz; // alias of size_t
	while ( getline(ifs, line) ) {
	    stringstream ss(line);
	    j = 0;
	    while (ss >> buf){
	    	temp = stod( buf, &sz );
	    	matTemp.aMatrix[0][j] = temp;
	        j += 1;
	    }
	    rWordVec[i] = matTemp;
	    i += 1;
	}

	ifs.close();

}

void DataUtils::loadVoca( string rVocaFileName,
	map< int, string >& rNumToWord, map< string, int >& rWordToNum ){

	fstream ifs;
	ifs.open ( rVocaFileName, fstream::in );

	int i = 0;
	int j = 0;
	string line;
	string buf;
	while ( getline(ifs, line) ) {
	    stringstream ss(line);
	    while (ss >> buf){
	    	rNumToWord[i] = buf;
	    }
	    i += 1;
	}

	ifs.close();

	invertDictFromIntToString( rNumToWord, rWordToNum );

}

void DataUtils::invertDictFromStringToInt( map< int, string >& rNumToWord,
					map< string, int >& rWordToNum ){

	for ( map< string, int >::iterator it=rWordToNum.begin() ; it!=rWordToNum.end(); ++it){
		rNumToWord[it->second] = it->first;
	}

}

void DataUtils::invertDictFromIntToString( map< int, string >& rNumToWord,
					map< string, int >& rWordToNum ){

	for ( map< int, string >::iterator it=rNumToWord.begin() ; it!=rNumToWord.end(); ++it){
		rWordToNum[it->second] = it->first;
	}

}

void DataUtils::setTags( map< int, string >& rNumToTag,
					map< string, int >& rTagToNum ){

	// set num2tag
	rNumToTag[0] = "O";
	rNumToTag[1] = "LOC";
	rNumToTag[2] = "MISC";
	rNumToTag[3] = "ORG";
	rNumToTag[4] = "PER";

	// set tag2num
	rTagToNum["O"] = 0;
	rTagToNum["LOC"] = 1;
	rTagToNum["MISC"] = 2;
	rTagToNum["ORG"] = 3;
	rTagToNum["PER"] = 4;

}

void DataUtils::loadDataset( string rDataFileName,
		vector< vector< vector< WordLabelType > > >& rDocs ){

	rDocs.clear();
	fstream ifs;
	ifs.open ( rDataFileName, fstream::in );
	int i = -1;
	int j = 0;
	int k = 0;
	string line;
	string buf;
	vector< string > vStringBuf;
	string string_buf[2];
	WordLabelType word_label;
	printf("before call getline \n");
	while ( getline(ifs, line) ) {
	    if ( 0 < line.size() ){
	    	k = 0;
		    stringstream ss(line);
	    	while (ss >> buf){
	    		if ( 0 == k) {
	    			word_label.aWord = buf;
	    		} else {
	    			word_label.aLabel = buf;
	    		}
	    		k += 1;
	    	}

	    	if ( (0 == word_label.aWord.compare("-DOCSTART-")) ) {
	    		i += 1;
	    		j = -1;
	    		rDocs.push_back( vector< vector< WordLabelType > >() );
	    	} // when meeting "-DOCSTART-", plus doc idx
	    	else {
	    		rDocs[i][j].push_back( word_label );
	    	} // when it is in doc idx (i), sentence idx (j)

	    } // when line size is over 0
	    else {
    		j += 1;
    		rDocs[i].push_back( vector< WordLabelType >() );
	    }
	}
	printf("after call getline\n");
	ifs.close();

}

void DataUtils::makeDocsToWindows( MatrixIntType& rXs,
		MatrixIntType& rYs,
		vector< vector< vector< WordLabelType > > >& rDocs,
		map< string, int >& rWordToNum,
		map< string, int >& rTagToNum,
		int rWindowSize ){

	int pad_size = (rWindowSize - 1) / 2;
	int i, j, k, q;
	int docs_size = rDocs.size();
	int total_size = 0;
	int X_col_size = 3;
	WordLabelType front_pad;
	WordLabelType back_pad;
	front_pad.aWord = "<s>";
	front_pad.aLabel = "";
	back_pad.aWord = "</s>";
	back_pad.aLabel = "";

	string temp;
	// make every word to lowercase
	for( i = 0 ; i < docs_size; i++ ){
		// iterate for sentence
		for ( j = 0; j < rDocs[i].size(); j++ ){
			// itertate for words + tags
			for ( k = 0; k < rDocs[i][j].size(); k++ ){
				temp = rDocs[i][j][k].aWord;
				for ( q = 0; q < temp.size(); q++ ){
					if ( (65 <= temp[q]) && (90 >= temp[q]) ){
						temp[q] += 32;
					}
				}
				rDocs[i][j][k].aWord = temp;
			}
		}
	}

	printf("before adding padding to sentences \n");
	// insert pad on both sides of every sentence
	// iterate for docs
	for( i = 0 ; i < docs_size; i++ ){
		// iterate for sentence
		for ( j = 0; j < rDocs[i].size(); j++ ){
			// insert pad at the front of sentence
			for ( q = 0; q < pad_size; q++ ){
				rDocs[i][j].insert( rDocs[i][j].begin(), front_pad );
			}

			// insert pad at the back of sentence
			for ( q = 0; q < pad_size; q++ ){
				rDocs[i][j].push_back( back_pad );
			}
		}
	}

	printf("before making X, Y \n");
	// rXs.aMatrix.push_back( vector< int >() );
	rYs.aMatrix.push_back( vector< int >() );
	printf("before going into 4 for loop\n");
	// iterate for docs
	for( i = 0 ; i < docs_size; i++ ){
		// iterate for sentence
		for ( j = 0; j < rDocs[i].size(); j++ ){
			// itertate for words + tags
			for ( k = 0; k < rDocs[i][j].size(); k++ ){
				// skip if we encounter the pad
				if ( (0 == rDocs[i][j][k].aWord.compare("<s>")) ||
					(0 == rDocs[i][j][k].aWord.compare("</s>")) ){
					continue;
				}

				// insert word and tag index to rXs and rYs
				rXs.aMatrix.push_back( vector< int >() );
				rYs.aMatrix[0].push_back( rTagToNum[rDocs[i][j][k].aLabel] );
				for( q = k- pad_size; q <= k + pad_size; q++ ){
					rXs.aMatrix[total_size].push_back( rWordToNum[rDocs[i][j][q].aWord] );
				}
				total_size += 1;
			}
		}
	}

	rXs.aRowSize = total_size;
	rXs.aColSize = rWindowSize;
	rYs.aRowSize = 1;
	rYs.aColSize = total_size;

}

void DataUtils::evalPerformance( MatrixIntType& rYs, MatrixIntType& rPredictedLabels,
		map< int, string >& rNumToTag ){
	// precision = true_pos / ( total_predicted )
	// recall    = true_pos / ( total_original  )
	// f1_score  = ( 2* precision * recall ) / ( precision + recall )

	double precision_0 = 0.0;
	double recall_0 = 0.0;
	double f1_score_0 = 0.0;

	double precision_1 = 0.0;
	double recall_1 = 0.0;
	double f1_score_1 = 0.0;

	double precision_2 = 0.0;
	double recall_2 = 0.0;
	double f1_score_2 = 0.0;

	double precision_3 = 0.0;
	double recall_3 = 0.0;
	double f1_score_3 = 0.0;

	double precision_4 = 0.0;
	double recall_4 = 0.0;
	double f1_score_4 = 0.0;

	int true_pos_0 = 0;
	int total_original_0 = 0;
	int total_predicted_0 = 0;

	int true_pos_1 = 0;
	int total_original_1 = 0;
	int total_predicted_1 = 0;

	int true_pos_2 = 0;
	int total_original_2 = 0;
	int total_predicted_2 = 0;

	int true_pos_3 = 0;
	int total_original_3 = 0;
	int total_predicted_3 = 0;

	int true_pos_4 = 0;
	int total_original_4 = 0;
	int total_predicted_4 = 0;

	int i, j, k;
	int total_size = rYs.aColSize;
	int origin_label;
	int predicted_label;
	for ( i = 0; i < total_size ; i++ ){
		predicted_label = rPredictedLabels.aMatrix[i][0];
		// for ( j = 0; j < total_size; j++ ){
			origin_label = rYs.aMatrix[0][i];
			switch( predicted_label ){
				case 0 :
					total_predicted_0 += 1;
					switch( origin_label ){
						case 0: // got it
							true_pos_0 += 1;
							total_original_0 += 1;
						break;
						case 1:
							total_original_1 += 1;
						break;
						case 2:
							total_original_2 += 1;
						break;
						case 3:
							total_original_3 += 1;
						break;
						case 4:
							total_original_4 += 1;
						break;
					}
				break;
				case 1 :
					total_predicted_1 += 1;
					switch( origin_label ){
						case 0: // got it
							total_original_0 += 1;
						break;
						case 1:
							true_pos_1 += 1;
							total_original_1 += 1;
						break;
						case 2:
							total_original_2 += 1;
						break;
						case 3:
							total_original_3 += 1;
						break;
						case 4:
							total_original_4 += 1;
						break;
					}
				break;
				case 2 :
					total_predicted_2 += 1;
					switch( origin_label ){
						case 0: // got it
							total_original_0 += 1;
						break;
						case 1:
							total_original_1 += 1;
						break;
						case 2:
							true_pos_2 += 1;
							total_original_2 += 1;
						break;
						case 3:
							total_original_3 += 1;
						break;
						case 4:
							total_original_4 += 1;
						break;
					}
				break;
				case 3 :
					total_predicted_3 += 1;
					switch( origin_label ){
						case 0: // got it
							total_original_0 += 1;
						break;
						case 1:
							total_original_1 += 1;
						break;
						case 2:
							total_original_2 += 1;
						break;
						case 3:
							true_pos_3 += 1;
							total_original_3 += 1;
						break;
						case 4:
							total_original_4 += 1;
						break;
					}
				break;
				case 4 :
					total_predicted_4 += 1;
					switch( origin_label ){
						case 0: // got it
							total_original_0 += 1;
						break;
						case 1:
							total_original_1 += 1;
						break;
						case 2:
							total_original_2 += 1;
						break;
						case 3:
							total_original_3 += 1;
						break;
						case 4:
							true_pos_4 += 1;
							total_original_4 += 1;
						break;
					}
				break;
			}
		// }
	}
	// precision = true_pos / ( total_predicted )
	// recall    = true_pos / ( total_original  )
	// f1_score  = ( 2* precision * recall ) / ( precision + recall )

	// type 0

	precision_0 = (double)(true_pos_0) / (double)(total_predicted_0);
	precision_1 = (double)(true_pos_1) / (double)(total_predicted_1);
	precision_2 = (double)(true_pos_2) / (double)(total_predicted_2);
	precision_3 = (double)(true_pos_3) / (double)(total_predicted_3);
	precision_4 = (double)(true_pos_4) / (double)(total_predicted_4);

	recall_0 = (double)(true_pos_0) / (double)(total_original_0);
	recall_1 = (double)(true_pos_1) / (double)(total_original_1);
	recall_2 = (double)(true_pos_2) / (double)(total_original_2);
	recall_3 = (double)(true_pos_3) / (double)(total_original_3);
	recall_4 = (double)(true_pos_4) / (double)(total_original_4);

	f1_score_0 = (2 * precision_0 * recall_0) / (precision_0 + recall_0);
	f1_score_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1);
	f1_score_2 = (2 * precision_2 * recall_2) / (precision_2 + recall_2);
	f1_score_3 = (2 * precision_3 * recall_3) / (precision_3 + recall_3);
	f1_score_4 = (2 * precision_4 * recall_4) / (precision_4 + recall_4);

	cout << "                    ";
	for( i = 0; i < 5; i++ ){
		cout << rNumToTag[i] << "    ";
	}
	printf("\n");
	printf("Predicted Cnt    : %d %d %d %d %d \n", total_predicted_0, total_predicted_1, total_predicted_2, total_predicted_3, total_predicted_4);
	printf("Origin Label Cnt : %d %d %d %d %d \n", total_original_0, total_original_1, total_original_2, total_original_3, total_original_4);
	printf("Precision        :  %lf  %lf  %lf  %lf  %lf\n", precision_0, precision_1, precision_2, precision_3, precision_4);
	printf("Recall           :  %lf  %lf  %lf  %lf  %lf\n", recall_0, recall_1, recall_2, recall_3, recall_4);
	printf("F1-Score         :  %lf  %lf  %lf  %lf  %lf\n", f1_score_0, f1_score_1, f1_score_2, f1_score_3, f1_score_4);

	double mean_precision = 0.0;
	double mean_recall = 0.0;
	double mean_f1_score = 0.0;
	printf("==== Performance without (O) \n");
	mean_precision = double(true_pos_1+true_pos_2+true_pos_3+true_pos_4)/double(total_predicted_1+total_predicted_2+total_predicted_3+total_predicted_4);
	mean_recall = double(true_pos_1+true_pos_2+true_pos_3+true_pos_4)/double(total_original_1+total_original_2+total_original_3+total_original_4);
	mean_f1_score = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall);
	printf("mean_Precision : %lf\n", mean_precision);
	printf("mean_Recall : %lf\n", mean_recall);
	printf("mean_F1_Score : %lf\n", mean_f1_score);
}

int DataUtils::savePredictions( char ch ){

}

int DataUtils::loadPredictions( char ch ){

}
