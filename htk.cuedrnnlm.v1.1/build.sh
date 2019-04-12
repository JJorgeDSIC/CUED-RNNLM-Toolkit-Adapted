#export LD_LIBRARY_PATH=/library/data4/tools/gcc-4.8.5/usr/local/lib:/library/data4/tools/glibc-2.16/lib64
#export PATH=/library/data4/tools/gcc-4.8.5/usr/local/bin:$PATH

. env/exp.linux
cd HTKLib
make clean; make
cd ../HTKTools
make clean; make HLRescore
