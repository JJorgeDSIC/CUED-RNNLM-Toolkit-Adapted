# make clean
rm -f *.o rnnlm.cued.v1.1

export PATH=/path/to/cuda/bin:$PATH
export LD_LIBRARY_PATH=/path/to/cuda/lib64:/path/to/cuda/lib:$LD_LIBRARY_PATH

nvcc -m64  -c cudaops.cu
g++ -g -std=c++0x -O2 main.cpp rnnlm.cpp cudamatrix.cpp Mathops.cpp fileops.cpp helper.cpp layer.cpp  -o rnnlm.cued.v1.1 cudaops.o -lcudart -lcublas -lcuda -lrt -L/path/to/cuda/lib64 -fopenmp
