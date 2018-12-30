## Deep Learning Framework for NLP from scratch


#### Description
Implemented C++ based framework for deep learning research from scratch.
Implemented function codes for derivatives, neuron units (e.g., sigmoid, tanh, relu), loss functions (e.g., negative log likelihood), matrix operators, parameter initialization (e.g., Xavier, Randomized), dealing with datasets, forward and backward propagations.
Experimented my framework with well-known NLP task, Named Entity Recognition.


Since C++ language does not support Python's "numpy" or other library,
I developed the double and integer matrix for NLP task, which are MatrixDoubleType
and MatrixIntType respectively, in order to support element-wise matrix operations,
dot product, transposing matrix and so on.


These matrix type is implemented internally as vectors of vectors so that matrix
can have any 2 Dimensional size. In other words, these matrix type is (m x n) matrix.
- For example

std::vector< std::vector< double > > aMatrix;
std::vector< std::vector< int > > aMatrix;

