# define the C compiler to use
CC = gcc
CXX = g++
# define any compile-time flags
CFLAGS = -Wall -g
CXXFLAGS = -std=c++11
RM = rm -f

# define the C++ source files

# SRCS = DeepLearning.cpp setup.cpp Ner.cpp RandomModule.cpp MathModule.cpp Derivatives.cpp NeuronUnits.cpp LossFunctions.cpp
SRCS = DeepLearning.cpp setup.cpp Ner.cpp DataUtils.cpp RandomModule.cpp MathModule.cpp Derivatives.cpp NeuronUnits.cpp LossFunctions.cpp

# define the C++ object files
OBJS = $(SRCS:.cpp=.o)
HEADERS = $(SRCS:.cpp=.h)
# define the executable file
TARGET = DeepLearning

all:    $(TARGET)
	@echo  Simple compiler named main has been compiled

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRCS) $(HEADERS) -lm

# $(TARGET): $(OBJS)
# 	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) -lm

# DeepLearning.o : DeepLearning.cpp $(HEADERS)
# 	$(CXX) $(CXXFLAGS) -c DeepLearning.cpp $(HEADERS)

# setup.o : setup.cpp setup.h
# 	$(CXX) $(CXXFLAGS) -c setup.cpp setup.h

# Ner.o : Ner.cpp Ner.h
# 	$(CXX) $(CXXFLAGS) -c Ner.cpp Ner.h

# DataUtils.o : DataUtils.cpp DataUtils.h
# 	$(CXX) $(CXXFLAGS) -c DataUtils.cpp DataUtils.h

# RandomModule.o : RandomModule.cpp RandomModule.h
# 	$(CXX) $(CXXFLAGS) -c RandomModule.cpp RandomModule.h

# MathModule.o : MathModule.cpp MathModule.h
# 	$(CXX) $(CXXFLAGS) -c MathModule.cpp MathModule.h

# Derivatives.o : Derivatives.cpp Derivatives.h
# 	$(CXX) $(CXXFLAGS) -c Derivatives.cpp Derivatives.h

# NeuronUnits.o : NeuronUnits.cpp NeuronUnits.h
# 	$(CXX) $(CXXFLAGS) -c NeuronUnits.cpp NeuronUnits.h

# LossFunctions.o : LossFunctions.cpp LossFunctions.h
# 	$(CXX) $(CXXFLAGS) -c LossFunctions.cpp LossFunctions.h

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file)
# (see the gnu make manual section about automatic variables)
clean:
	$(RM) *.o *~ $(TARGET)
	$(RM) *.gch