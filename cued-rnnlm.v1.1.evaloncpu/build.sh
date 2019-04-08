# make clean
rm -f *.o rnnlm.cued.v1.1.eval
g++ -g -std=c++0x -O2 main.cpp rnnlm.cpp  Mathops.cpp fileops.cpp helper.cpp layer.cpp  -o rnnlm.cued.v1.1.eval -lrt -fopenmp
