/** @file DeepLearning.h
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

#ifndef DeepLearning_h
#define DeepLearning_h

#include "setup.h"
#include "Ner.h"
#include "DataUtils.h"

using namespace std;

class DeepLearning
{
public:

	Ner aNer;
	// Library Classes
	DataUtils 	 YdDu;
	/** @brief Constructor of the DeepLearning class.
	 *
	 *  Default constructor method of the DeepLearning class.
	 *  If you want to use the method declared in the DeepLearning
	 *  class, you should first initialize the DeepLearning class
	 *  with constructor method. After initializing the class, one
	 *  can make use of this class.
	 */
	DeepLearning();

	/** @brief Destructor of the DeepLearning class.
	 *
	 *  Default destructor method of the DeepLearning class.
	 *  If there is no need for you to use the method declared in
	 *  the DeepLearning class, you can invoke the destructor method
	 *  of the DeepLearning class. By using destructor method, you
	 *  can safely remove allocated memory used in this class if there
	 *  is any.
	 */
	~DeepLearning();

	/** @brief Print Performance of the model
	 *
	 *  @param ch the array calculated from the model which
	 *  	   contains predictions
	 *  @param ch the label which is given for a right answer
	 *  @return Loss of the model
	 */
	int printPerformance( char ch );

	/** @brief Test case 1
	 *
	 *  Dim = 100, alpha = 0.1, lambda = 0.001, batchsize = 1, epoch = 1
	 *  @return void
	 */
	void run1();

	/** @brief Test case 2
	 *
	 *  Dim = 100, alpha = 0.03, lambda = 0.001, batchsize = 1, epoch = 1
	 *  @return void
	 */
	void run2();

	/** @brief Test case 3
	 *
	 *  Dim = 150, alpha = 0.01, lambda = 0.001, batchsize = 1, epoch = 1
	 *  @return void
	 */
	void run3();

	/** @brief Test case 4
	 *
	 *  Dim = 100, alpha = 0.01, lambda = 0.001, batchsize = 5, epoch = 1
	 *  @return void
	 */
	void run_100_01_5();

	/** @brief Test case 5
	 *
	 *  Dim = 100, alpha = 0.01, lambda = 0.001, batchsize = 5, epoch = 1
	 *  @return void
	 */
	void run_100_1_5();

	/** @brief Test case 6
	 *
	 *  Dim = 150, alpha = 0.1, lambda = 0.001, batchsize = 1, epoch = 1
	 *  @return void
	 */
	void run_150_01();

	/** @brief Check gradient of the model
	 *
	 *  @return void
	 */
	void gradCheck();

protected:

private:

};


#endif
