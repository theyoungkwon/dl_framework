/** @file DataUtils.h
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

#ifndef DataUtils_h
#define DataUtils_h

#include "setup.h"
using namespace std;

class DataUtils
{
public:

	/** @brief Constructor of the DataUtils class.
	 *
	 *  Default constructor method of the DataUtils class.
	 *  If you want to use the method declared in the DataUtils
	 *  class, you should first initialize the DataUtils class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	DataUtils();

	/** @brief Destructor of the DataUtils class.
	 *
	 *  Default destructor method of the DataUtils class.
	 *  If there is no need for you to use the method declared in
	 *  the DataUtils class, you can invoke the destructor method
	 *  of the DataUtils class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~DataUtils();

	/** @brief Load word vectors from pre-computed files
	 *
	 *  @param rWordVecFileName File name which contains word vectors
	 *  @param rWordVec data structure which will
	 *  	   contains word vectors
	 *  @param rDimSize dimension size of word vector
	 *  @return void
	 */
	void loadWordVec( string rWordVecFileName,
				map< int, MatrixDoubleType>& rWordVec,
				int rDimSize );

	/** @brief Load vocabulary from file.
	 *
	 *  By loading vocabulary from a file, produce num_to_word map data structure
	 *  and word_to_num map data structure.
	 *
	 *  @param rVocaFileName File name which contains word (vocabulary)
	 *  @param rNumToWord map data structure converting from int
	 * 		   index to word string
	 *  @param rWordToNum map data structure converting from word
	 * 		   string to int index
	 *  @return void
	 */
	void loadVoca( string rVocaFileName,
	map< int, string >& rNumToWord, map< string, int >& rWordToNum );

	/** @brief Invert dictionary from word2num to num2word
	 *
	 *  @param rNumToWord Map structure converting from index to string
	 *  @param rWordToNum Map structure converting from string to index
	 *  @return void
	 */
	void invertDictFromStringToInt( map< int, string >& rNumToWord,
					map< string, int >& rWordToNum );

	/** @brief Invert dictionary from num2word to word2num
	 *
	 *  @param rNumToWord Map structure converting from index to string
	 *  @param rWordToNum Map structure converting from string to index
	 *  @return void
	 */
	void invertDictFromIntToString( map< int, string >& rNumToWord,
					map< string, int >& rWordToNum );

	/** @brief Set Dictionary-typed num2tag and tag2num
	 *
	 *	There exist 5 kinds of tags such as, "O", "LOC", "MISC", "ORG", "PER"
	 *  "LOC" means Location, "MISC" means Miscellaneous, "ORG" means Organization
	 *  "PER" means Person, "O" means none of the above categories.
	 *
	 *  @param rNumToTag Map structure converting from index to string
	 *  @param rTagToNum Map structure converting from string to index
	 *  @return void
	 */
	void setTags( map< int, string >& rNumToTag,
					map< string, int >& rTagToNum );

	/** @brief Load training, dev, test dataset.
	 *
	 *  @param rDataFileName File name (= training, dev, test)
	 *  @param rDocs vector data structure containing string of
	 *  	   single word of dataset.
	 *  @return void
	 */
	void loadDataset( string rDataFileName,
		vector< vector< vector< WordLabelType > > >& rDocs );

	/** @brief Produce input data to NER model.
	 *
	 *  @param rXs Input to the model
	 *  @param rYs Output labels of the model
	 *  @param rDocs vector structure containing string of single word of dataset
	 *  @param rWordToNum Map structure converting from string to index
	 *  @param rTagToNum Map structure converting from string to index
	 *  @param rWindowSize Window size of input to the model
	 *  @return void
	 */
	void makeDocsToWindows( MatrixIntType& rXs,
		MatrixIntType& rYs,
		vector< vector< vector< WordLabelType > > >& rDocs,
		map< string, int >& rWordToNum,
		map< string, int >& rTagToNum,
		int rWindowSize );

	/** @brief Evaluate Performance of the model.
	 *
	 *  @param rYs Original labels
	 *  @param rPredictedLabels output of the model
	 *  @return void
	 */
	void evalPerformance( MatrixIntType& rYs, MatrixIntType& rPredictedLabels,
			map< int, string >& num_to_tag );

	/** @brief Save predictions of the model.
	 *
	 *
	 *
	 *  @param ch the array calculated from the model which
	 *  	   contains predictions
	 *  @param ch the label which is given for a right answer
	 *  @return Loss of the model
	 */
	int savePredictions( char ch );

	/** @brief Load the predictions from a file.
	 *
	 *
	 *
	 *  @param ch the array calculated from the model which
	 *  	   contains predictions
	 *  @param ch the label which is given for a right answer
	 *  @return Loss of the model
	 */
	int loadPredictions( char ch );

protected:

private:

};

#endif
