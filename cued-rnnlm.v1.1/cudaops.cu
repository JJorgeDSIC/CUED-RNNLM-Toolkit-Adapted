#ifndef _CUDAOPS_H__
#define _CUDAOPS_H__

#include <cuda_runtime_api.h>           // for CUDA API
 #include <cublas_v2.h>                  // cublas library
#include <cuda.h>                       // for device API
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
// #include <cublas.h>
#include "DataType.h"
#include "head.h"
using namespace std;

typedef int CUdevice;
static cublasHandle_t handle;
static curandState_t *states;


extern "C" void initcuHandle ()
{
    cublasCreate(&handle);
    // cannot set this flag??
    // cudaSetDeviceFlags (cudaDeviceScheduleBlockingSync);
    cudaDeviceSynchronize();
}

__global__ void init (unsigned int seed, curandState_t* states, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    int idx = i+j*nrows;
    curand_init(seed, idx, 0, &states[idx]);
}

extern "C" void initcuRandstates ()
{
    int nrows = 2000;
    int ncols = 300;
    dim3 t (32,16);
    dim3 b ((nrows+t.x-1)/t.x,(ncols+t.y-1)/t.y);
    cudaMalloc((void**) &states, nrows * ncols * sizeof(curandState_t));
    init<<<b, t>>>(time(0), states, nrows, ncols);
    cudaDeviceSynchronize();
}

// log (exp(x)+exp(y))
__device__ real culogadd (real x, real y)
{
    if (x > y)
    {
        return (x+logf(1+expf(y-x)));
    }
    else
    {
        return (y+logf(1+expf(x-y)));
    }
}

static void operator ||  (CUresult rc, const char *msg)
{
    if (rc != CUDA_SUCCESS)
    {
        char buf[1024];
        sprintf (buf, "%s: cuda API error %d", msg, rc);
        exit (0);
    }
}

extern "C" size_t numdevices ()
{
    int deviceNum = 0;
    // cudaGetDeviceCount (&deviceNum) || "cudaGetDeviceCount failed";
    if (cudaGetDeviceCount (&deviceNum) != cudaSuccess)
    {
        printf ("cudaGetDeviceCount failed\n");
        exit (0);
    }
    return (size_t) deviceNum;
}

extern "C" void destroycuHandle()
{
    cublasDestroy (handle);
}

extern "C" void dumpdeviceInfo ()
{
    cuInit (0) || "cuInit failed";
    size_t n = numdevices ();
    for (size_t i=0; i<n; i++)
    {
        CUdevice cuDevice;
        cuDeviceGet (&cuDevice, (int) i) || "cuDeviceGet failed";
        char name[1024] = {0};
        cuDeviceGetName (&name[0], 1023, cuDevice) || "cuDeviceGetName failed";
        fprintf (stdout, "CUDA device %d: %s\n", (int)i, name);
    }
}

extern "C" void setDeviceid (size_t i)
{
    // cudaSetDevice(i) || "cudaSetDevice failed";
    if (cudaSetDevice(i) != cudaSuccess)
    {
        printf ("cudaSetDevice failed\n");
        exit (0);
    }
}

__global__ void sigmoidij (real* us, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    real exponent = us[j*nrows + i];
    if (exponent < -30.0f)
        us[j*nrows + i] = 0.0f;
    else
        us[j*nrows + i] = 1.0f / (1.0f + expf (-exponent));

    __syncthreads (); // need to sync here
}
extern "C" void cusigmoid (real *us, int nrows, int ncols)
{
    dim3 t (32,16);
    dim3 b ((nrows+t.x-1)/t.x,(ncols+t.y-1)/t.y);
    sigmoidij<<<b, t>>> (us, nrows, ncols);
    cudaDeviceSynchronize();
}

__global__ void tanhij (real* us, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    real exponent = us[j*nrows + i];
    if (exponent < -30.0f)
        us[j*nrows + i] = -1.0f;
    else if (exponent > 30.0f)
        us[j*nrows + i] = 1.0f;
    else
    {
        exponent = expf(2*exponent);
        us[j*nrows + i] = (exponent-1.0f) / (exponent+1.0f);
    }

    __syncthreads (); // need to sync here
}
extern "C" void cutanh (real *us, int nrows, int ncols)
{
    dim3 t (32,16);
    dim3 b ((nrows+t.x-1)/t.x,(ncols+t.y-1)/t.y);
    tanhij<<<b, t>>> (us, nrows, ncols);
    cudaDeviceSynchronize();
}

__global__ void cudropoutij (float *us, float *dropoutmask, curandState_t *states,  float dropoutrate, bool evalmode, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    if (evalmode)
    {
#ifdef DROPOUT_V2
        us[i+j*nrows] *= 1;
#else
        us[i+j*nrows] *= (1-dropoutrate);
#endif
    }
    else
    {
        us[i+j*nrows] *= dropoutmask[i+j*nrows];
    }
}

extern "C" void cudropout (float *us, float *dropoutmask, float dropoutrate, bool evalmode, int nrows, int ncols)
{
    dim3 t (32,16);
    dim3 b ((nrows+t.x-1)/t.x,(ncols+t.y-1)/t.y);
    cudropoutij<<<b, t>>> (us, dropoutmask, states, dropoutrate, evalmode, nrows, ncols);
    cudaDeviceSynchronize ();
}

__global__ void cugendropoutmaskij (float *dropoutmask, curandState_t *states,  float dropoutrate, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    int idx = i+j*nrows;
    if (idx > 20000)  idx = idx%20000;
    float randv = curand_uniform(&states[idx]);
    if (randv < dropoutrate)
    {
        dropoutmask[i+j*nrows] = 0;
    }
    else
    {
#ifdef DROPOUT_V2
        dropoutmask[i+j*nrows] = 1 / (1-dropoutrate);
#else
        dropoutmask[i+j*nrows] = 1;
#endif
    }
}

extern "C" void cugendropoutmask (float *us, float dropoutrate, int nrows, int ncols)
{
    dim3 t (32,16);
    dim3 b ((nrows+t.x-1)/t.x,(ncols+t.y-1)/t.y);
    cugendropoutmaskij<<<b, t>>> (us, states, dropoutrate, nrows, ncols);
    cudaDeviceSynchronize ();
}


__global__ void cugenEmbeddropoutmaskij (float *dropoutmask, curandState_t *states,  float dropoutrate, int nrows, int ncols)
{
    int i = 0;
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (j >= ncols)
    {
        return;
    }
    int idx = i+j*nrows;
    if (idx > 20000)  idx = idx%20000;
    float randv = curand_uniform(&states[idx]);
    if (randv < dropoutrate)
    {
        for (i=0; i<nrows; i++)
        {
            dropoutmask[i+j*nrows] = 0;
        }
    }
    else
    {
        for (i=0; i<nrows; i++)
        {
#ifdef DROPOUT_V2
            dropoutmask[i+j*nrows] = 1 / (1-dropoutrate);
#else
            dropoutmask[i+j*nrows] = 1;
#endif
        }
    }
}
extern "C" void cugenEmbeddropoutmask (float *us, float dropoutrate, int nrows, int ncols)
{
    dim3 t (1,16);
    dim3 b (1,(ncols+t.y-1)/t.y);
    cugenEmbeddropoutmaskij<<<b, t>>> (us, states, dropoutrate, nrows, ncols);
    cudaDeviceSynchronize ();
}

__global__ void cugenvardropoutmaskij (float *dropoutmask, curandState_t *states, int mbidx, float dropoutrate, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= nrows)
    {
        return;
    }
    const size_t j = mbidx;
    int idx = i+j*nrows;
    if (idx > 20000)  idx = idx%20000;
    float randv = curand_uniform(&states[idx]);
    if (randv < dropoutrate)
    {
        dropoutmask[i+j*nrows] = 0;
    }
    else
    {
#ifdef DROPOUT_V2
        dropoutmask[i+j*nrows] = 1 / (1-dropoutrate);
#else
        dropoutmask[i+j*nrows] = 1;
#endif
    }
}

extern "C" void cugenvardropoutmask (float *us, int mbidx, float dropoutrate, int nrows, int ncols)
{
    dim3 t (32,1);
    dim3 b ((nrows+t.x-1)/t.x, 1);
    cugenvardropoutmaskij<<<b, t>>> (us, states, mbidx, dropoutrate, nrows, ncols);
    cudaDeviceSynchronize ();
}

__global__ void reluij (real* us, int nrows, int ncols, float ratio)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    if (us[j*nrows+i] > 0)
    {
#if 0
        if (us[i+j*nrows] > 20)
        {
            us[j*nrows+i] = 20;
        }
        else
#endif
        {
            us[j*nrows+i] = ratio*us[j*nrows+i];
        }
    }
    else
    {
        us[j*nrows+i] = 0;
    }
    __syncthreads (); // need to sync here
}

extern "C" void curelu (real *us, int nrows, int ncols, float ratio)
{
    dim3 t (32,16);
    dim3 b ((nrows+t.x-1)/t.x,(ncols+t.y-1)/t.y);
    reluij<<<b, t>>> (us, nrows, ncols, ratio);
    cudaDeviceSynchronize();
}

__global__ void cugradcutoffij (real *us, int nrows, int ncols, real gradcutoff)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    if (us[j*nrows+i] > gradcutoff)
    {
        us[j*nrows+i] = gradcutoff;
    }
    else if (us[j*nrows+i] < -gradcutoff)
    {
        us[j*nrows+i] = -gradcutoff;
    }
}
extern "C" void cugradcutoff (real *us, int nrows, int ncols, real gradcutoff)
{
    dim3 t (32, 16);
    dim3 b ((nrows+t.x-1)/t.x, (ncols+t.y-1)/t.y);
    cugradcutoffij<<<b, t>>> (us, nrows, ncols, gradcutoff);
    cudaDeviceSynchronize ();
}

extern "C"  void cucpyfromGPU (void *dst, void *src, size_t sz)
{
    cudaMemcpy (dst, src, sz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

extern "C"  void * cucalloc (size_t sz)
{
    void * p;
    // cudaMalloc (&p, sz) || "cudaMalloc failed";
    if (cudaMalloc(&p, sz) != cudaSuccess)
    {
        printf ("cudaMalloc failed\n");
        exit (0);
    }
    cudaDeviceSynchronize();
    return p;
}

extern "C" void cucpytoGpu (void *dst, void *src, size_t sz)
{
    cudaMemcpy (dst, src, sz, cudaMemcpyHostToDevice);
     cudaDeviceSynchronize();
}

__global__ void cucpyInGpu_v1 (char *dst, char *src, size_t sz)
{

    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= sz)
    {
        return;
    }
    dst[i] = src[i];
    __syncthreads();   // need to sync here.
}


__global__ void cucpyInGpu_v2 (char *dst, char *src, size_t sz)
{

    size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i>=sz) return;
    while (i < sz)
    {
        dst[i] = src[i];
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();   // need to sync here.
}
// copy data from GPU to GPU, don't transfer through CPU.
extern "C" void cucpyInGpu (void *dst, void *src, size_t sz)
{
    // This maybe too large, need to make it run in a fixed configuration
    dim3 t (256);
    dim3 b (32);
    // cucpyInGpu_v1<<<dim3 ((((unsigned int)sz)+31)/32), 32>>> ((char*)dst, (char *)src, sz);
    cucpyInGpu_v2<<<b, t>>> ((char*)dst, (char *)src, sz);
    cudaDeviceSynchronize();
}
extern "C" void cucpyrealInGpu (real *dst, real *src, size_t sz)
{
    if (isdouble(real))
        cublasDcopy (handle, sz, (double *)src, 1, (double *)dst, 1);
    else
        cublasScopy (handle, sz, (float *)src, 1, (float *)dst, 1);
}

extern "C" void cufree (void *ptr)
{
    cudaFree (ptr);
     cudaDeviceSynchronize();
}

extern "C" void cumemset (void *ptr, int value, size_t sz)
{
    cudaMemset (ptr, value, sz);
    // cudaDeviceSynchronize();
}

__global__ void assign (real value, real *ptr)
{
    *ptr = value;
    __syncthreads (); // need to sync here
}
extern "C" void cuassign (real*ptr, real value)
{
    assign<<<1,1>>> (value, ptr);
    cudaDeviceSynchronize();
}


__global__ void cuassignmatvalue_v1 (real *ptr, size_t nrows, size_t ncols, real value)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows)
        return;
    if (j >= ncols)
        return;
    ptr[i+j*nrows] = value;

}

extern "C" void cuassignmatvalue (real *ptr, size_t nrows, size_t ncols, real value)
{
    dim3 t(32, 32);
    dim3 b((nrows+31)/32, (ncols + 31)/32);
    cuassignmatvalue_v1<<<t, b>>>(ptr, nrows, ncols, value);
}

__global__ void cuassigncolumn_v1 (real *ptr, int icol, real value, size_t nrows)
{
    size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= nrows) return;
    while (i < nrows)
    {
        ptr[i+icol*nrows] = value;
        i += blockDim.x;
    }
}
extern "C" void cuassigncolumn (real *ptr, int icol, real value, size_t nrows)
{
    dim3 t(1024);
    dim3 b((nrows+t.x-1)/t.x);
    cuassigncolumn_v1<<<b, t>>> (ptr, icol, value, nrows);
}

__global__ void cuassignneu0ac_v1 (real *dev_data, int *prevwords, size_t nrow, size_t mb, real v)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= mb)
    {
        return;
    }
    if (prevwords[i] != INVALID_INT)
    {
        dev_data[i*nrow + prevwords[i]] = v;
    }
}
__global__ void cuassignneu0ac_v2 (real *dev_data, int *prevwords, size_t nrow, size_t mb, real v)
{
    size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    while (i < mb)
    {
        if (prevwords[i] != INVALID_INT)
        {
            dev_data[i*nrow+prevwords[i]] = v;
        }
        i += blockDim.x;
    }
}
extern "C" void cuassignneu0ac (real *dev_data, int *prevwords, size_t nrow, size_t mb, real v)
{
#if 0
    dim3 t(mb);
    dim3 b(1);
    cuassignneu0ac_v1<<<b, t>>> (dev_data, prevwords, nrow, mb, v);
#else
    dim3 t(512);
    dim3 b((mb+t.x-1)/t.x);
    cuassignneu0ac_v2<<<b, t>>> (dev_data, prevwords, nrow, mb, v);

#endif
}

__global__ void cufetchwordprobs_v1 (real *data, size_t nrow, size_t ncol, int *curwords, real *wordprobs, int fulldict_size)
{
    size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    while (i < ncol)
    {
        if (curwords[i] == nrow - 1)   // OOS node
        {
             wordprobs[i] = data[i*nrow + curwords[i]] / (fulldict_size - nrow + 1);
             // wordprobs[i] = data[i*nrow + curwords[i]];
        }
        else
        {
            wordprobs[i] = data[i*nrow + curwords[i] ] ;
        }
        i += blockDim.x;
    }
}

// fetch the word probs vector from the output layer
extern "C" void cufetchwordprobs (real *dev_data, size_t nrow, size_t ncol, int *dev_curwords, real *dev_wordprobs, int fulldict_size)
{
    dim3 t(512);
    dim3 b((ncol+t.x-1)/t.x);
    cufetchwordprobs_v1<<<b, t>>> (dev_data, nrow, ncol, dev_curwords, dev_wordprobs, fulldict_size);
}


// destmat: m x n (output)
// srcmat:  m x k (input)
// syn:     k x n (input)
// C = alpha * A X B + beta * C
extern "C" void matXmat (bool transA, bool transB, size_t m, size_t n, size_t k, real *dataA, size_t lda, real *dataB, size_t ldb, real *dataC, size_t ldc, real alpha /* = 1.0 */, real beta /* = 0.0*/)
{
    cublasStatus_t ret;
    if (isdouble(real))
        ret = cublasDgemm(handle, transA ? CUBLAS_OP_T : CUBLAS_OP_N, transB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, (double *)&alpha, (double *)dataA, lda, (double *)dataB, ldb, (double *)&beta, (double *)dataC, ldc);
    else
        ret = cublasSgemm(handle, transA ? CUBLAS_OP_T : CUBLAS_OP_N, transB ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, (float *)&alpha, (float *)dataA, lda, (float *)dataB, ldb, (float *)&beta, (float *)dataC, ldc);

    if (ret != CUBLAS_STATUS_SUCCESS)
    {
        cout<<"*** ERROR *** matXmat returned error code "<<ret<<", line "<<__LINE__<<endl;
        exit(EXIT_FAILURE);
    }
     cudaDeviceSynchronize();
}

__global__ void cuneu1addneu0words (int *prevwords, real *ac1, real *layer0, size_t l0, size_t l1, size_t mb)
{
    size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= l1) return;
#if 0
    while (i < l1)
    {
        for (int t = 0; t < mb; t ++)
        {
            ac1[t*l1 + i] += layer0[prevwords[t] + i*l0];
        }
        i += blockDim.x;
    }
#else
    for (int t = 0; t < mb; t ++)
    {
        if (prevwords[t] == INVALID_INT)
        {
            ac1[t*l1 + i] += 0.0;
        }
        else if (prevwords[t] == INPUT_PAD_INT)
        {
            ac1[t*l1 + i] += 0.0;
        }
        else
        {
            ac1[t*l1 + i] += layer0[prevwords[t] + i*l0];
        }
    }
#endif
    __syncthreads();   // need to sync here.

}
// TODO: need to be modified when layer1_size is greater than 1024
extern "C" void neu1addneu0words (int *prevwords, real *ac1, real *layer0, size_t layer0_size, size_t layer1_size, size_t mb)
{
    dim3 t(1024);
    // dim3 b(1);
    dim3 b((layer1_size+t.x-1)/t.x);
    cuneu1addneu0words<<<b, t>>> (prevwords, ac1,
                                layer0, layer0_size, layer1_size, mb);
    cudaDeviceSynchronize();
}


#if 0
__global__ void softmaxj_v2 (real *us, size_t nrows, size_t mb) // thread = one per column
{
    // __shared__ real cache[blockDim.x*blockDim.y];
    __shared__ real cache[1024];
    //// const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    //// const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    const size_t i = threadIdx.x + (threadIdx.y * blockDim.x);
    const size_t j = blockIdx.x + (blockIdx.y * gridDim.x);
    if (j >= mb)
    {
        return;
    }
    real temp = 0.0;

    // find max
    real colmax = 0.0f;
#if 0
    for (size_t i = 0 ; i < nrows; i++)
    {
        real usij = us[j*nrows + i];
        if (usij > colmax)
            colmax = usij;
    }
#else
    int idx = i;
    cache[i] = us[j*nrows+i];
    while (idx < nrows)
    {
        real usij = us[j*nrows+idx];
        if (usij > cache[i])
            cache[i] = usij;
        idx += blockDim.x*blockDim.y;
    }
    __syncthreads();
    idx = blockDim.x * blockDim.y / 2;
    while (idx != 0)
    {
        if (i < idx)
            if (cache[i] < cache[i+idx])
                cache[i] = cache[i+idx];
        __syncthreads();
        idx = idx/2;
    }
    colmax = cache[0];
#endif

    // sum(exp(i))
    int index = i;
    while (index < nrows)
    {
        us[j*nrows+index] = expf (us[j*nrows+index]-colmax);
        temp += us[j*nrows+index];
        index += blockDim.x*blockDim.y;
    }
    cache[i] = temp;
    __syncthreads ();
    index = blockDim.x * blockDim.y / 2;
    while (index != 0)
    {
        if (i < index)
            cache[i] += cache[i+index];
        __syncthreads();
        index /= 2;
    }
    // exp(i) / sum
    // cache[0] stores the sum of exp(i) for each sample.
    index = i;
    while (index < nrows)
    {
        us[j*nrows+index] = us[j*nrows+index] / cache[0];
        index += blockDim.x*blockDim.y;
    }
    __syncthreads();
}



extern "C" void cusoftmax(real *us, size_t nrows, size_t mb)
{
    dim3 t(32, 32);
    dim3 b((mb+31)/32, 32);
    softmaxj_v2 <<<b, t>>> (us, nrows, mb);
    cudaDeviceSynchronize();
}

#endif

__global__ void softmaxj_v2 (real *us, size_t nrows, size_t mb, real *lognorms) // thread = one per column
{
    // __shared__ real cache[blockDim.x*blockDim.y];
    __shared__ real cache[1024];
    //// const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    //// const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    const size_t i = threadIdx.x + (threadIdx.y * blockDim.x);
    const size_t j = blockIdx.x + (blockIdx.y * gridDim.x);
    if (j >= mb)
    {
        return;
    }
    if (i >= nrows)
    {
        return;
    }
    real temp = 0.0f;

    // find max
    real colmax = 0.0f;
#if 0
    for (size_t i = 0 ; i < nrows; i++)
    {
        real usij = us[j*nrows + i];
        if (usij > colmax)
            colmax = usij;
    }
    if (i ==0)
    {
        lognorms[j] = colmax;
    }
#else
    int idx = i;
    cache[i] = us[j*nrows+i];
    while (idx < nrows)
    {
        real usij = us[j*nrows+idx];
        if (usij > cache[i])
            cache[i] = usij;
        idx += blockDim.x*blockDim.y;
    }
    __syncthreads();
    idx = blockDim.x * blockDim.y / 2;
    while (idx != 0)
    {
        if (i < idx)
            // if (cache[i] < cache[i+idx])
            if (cache[i] < cache[i+idx] && (i+idx) < nrows)
                cache[i] = cache[i+idx];
        __syncthreads();
        idx = idx/2;
    }
    colmax = cache[0];
    __syncthreads();
#endif

    // sum(exp(i))
    int index = i;
    while (index < nrows)
    {
        us[j*nrows+index] = expf (us[j*nrows+index]-colmax);
        temp += us[j*nrows+index];
        index += blockDim.x*blockDim.y;
    }
    cache[i] = temp;
    __syncthreads ();
    index = blockDim.x * blockDim.y / 2;
    while (index != 0)
    {
        // if (i < index)
        if (i < index && (i+index) < nrows)
            cache[i] += cache[i+index];
        __syncthreads();
        index /= 2;
    }
    // exp(i) / sum
    // cache[0] stores the sum of exp(i) for each sample.
    index = i;
    while (index < nrows)
    {
        us[j*nrows+index] = us[j*nrows+index] / cache[0];
        index += blockDim.x*blockDim.y;
    }
    __syncthreads();
    if (i == 0)
    {
        lognorms[j] = logf(cache[0]) + colmax;
    }
}

extern "C" void cusoftmax(real *us, size_t nrows, size_t mb, float *lognorms)
{
    dim3 t(32, 32);
    dim3 b((mb+31)/32, 32);
    softmaxj_v2 <<<b, t>>> (us, nrows, mb, lognorms);
    cudaDeviceSynchronize();
}

extern "C" void assignsubmat (size_t i0, size_t j0, size_t nr, size_t nc, size_t width, real *src, real *tgt)
{
    initcuHandle();
    cublasSetMatrix (nr, nc, sizeof(real), &src[i0+j0*width], width, tgt, nr);
    cudaDeviceSynchronize();
}
__global__ void cumulsigmoid (real *er, real *ac, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    er [i+j*nrows] = er[i+j*nrows] * ac[i+j*nrows] * (1-ac[i+j*nrows]);
    __syncthreads();   // need to sync here.
}
extern "C" void multiplyacsigmoid (real *er, real *ac, int nrow, int ncol)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (ncol+t.y-1)/t.y);
    cumulsigmoid<<<b, t>>>(er, ac, nrow, ncol);
    cudaDeviceSynchronize();
}

__global__ void cumultanh (real *er, real *ac, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    er [i+j*nrows] = er[i+j*nrows] * (1-ac[i+j*nrows]*ac[i+j*nrows]);
    __syncthreads();   // need to sync here.
}
extern "C" void multiplyactanh (real *er, real *ac, int nrow, int ncol)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (ncol+t.y-1)/t.y);
    cumultanh<<<b, t>>>(er, ac, nrow, ncol);
    cudaDeviceSynchronize();
}


__global__ void cudotMultiplyij (real *us, real *other, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    us [i+j*nrows] = us[i+j*nrows] * other[i+j*nrows];
    __syncthreads();   // need to sync here.
}
extern "C" void cudotMultiply (real *us, real *other, int nrow, int ncol)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (ncol+t.y-1)/t.y);
    cudotMultiplyij<<<b, t>>> (us, other, nrow, ncol);
    cudaDeviceSynchronize ();
}

__global__ void cucalHiddenacGRUij (real *us, real *x, real *h_, real *z, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    // us = z*h_ + (1-z)*x
    us [i+j*nrows] = z[i+j*nrows]*h_[i+j*nrows] + (1-z[i+j*nrows])*x[i+j*nrows];
    __syncthreads();   // need to sync here.
}
extern "C" void cucalHiddenacGRU (real *us, real *x, real *h, real *z, int nrow, int ncol)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (ncol+t.y-1)/t.y);
    cucalHiddenacGRUij<<<b, t>>> (us, x, h, z, nrow, ncol);
    cudaDeviceSynchronize ();
}


__global__ void cumultiplyScalarij (real *us, float v, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    us[i+j*nrows] *= v;

}
extern "C" void cumultiplyScalar (real *us, float v, real nrow, int ncol)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (ncol+t.y-1)/t.y);
    cumultiplyScalarij<<<b, t>>> (us, v, nrow, ncol);
    cudaDeviceSynchronize ();
}

__global__ void cuaddScalarij (real *us, float v, int nrows, int ncols)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    us[i+j*nrows] += v;

}
extern "C" void cuaddScalar (real *us, float v, real nrow, int ncol)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (ncol+t.y-1)/t.y);
    cuaddScalarij<<<b, t>>> (us, v, nrow, ncol);
    cudaDeviceSynchronize ();
}

__global__ void cumulrelu (real *er, real *ac, int nrows, int ncols, float ratio)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nrows || j >= ncols)
    {
        return;
    }
    if (ac[i+j*nrows] > 0)
    {
        er[i+j*nrows] = ratio*er[i+j*nrows];
    }
    else
    {
        er[i+j*nrows] = 0;
    }
    __syncthreads();   // need to sync here.
}
extern "C" void multiplyacrelu (real *er, real *ac, int nrow, int ncol, float ratio)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (ncol+t.y-1)/t.y);
    cumulrelu<<<b, t>>>(er, ac, nrow, ncol, ratio);
    cudaDeviceSynchronize();
}

__global__ void cuupdatelayer0_wordweight (real *layer0, real *er, int *words, int nrows, int ncols, int mb, real alpha, real beta)
{
    size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= ncols)  return;
	for (size_t t = 0; t < mb; t ++)
	{
		int word = words[t];
        if (word != INVALID_INT && word != INPUT_PAD_INT)
        {
            layer0[word+i*nrows] += alpha * er[i+ncols*t] + beta * layer0[word+i*nrows];
        }
	}
    __syncthreads();   // need to sync here.
}
// TODO: need to be modified if layer1_size is greater than 1024
extern "C" void updatelayer0_wordweight (real *layer0, real *er, int *words, int nrows, int ncols, int mb, real alpha, real beta)
{
    dim3 t(1024);
    // dim3 b(1);
    dim3 b((ncols+t.x-1)/t.x);
    cuupdatelayer0_wordweight <<<b, t>>> (layer0, er, words, nrows, ncols, mb, alpha, beta);
    cudaDeviceSynchronize();
}

__global__ void cuaddmatrix (real *us, real *other, size_t nelem)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= nelem)
        return;
    us[i] += other[i];
    __syncthreads();   // need to sync here.
}
extern "C" void addmatrix (real *us, real *other, size_t nelem)
{
    cuaddmatrix<<<dim3((((unsigned int)nelem)+31)/32), 32>>>(us, other, nelem);
    cudaDeviceSynchronize();
}

__global__ void cuaddproductmat (real *us, real *other1, real *other2, size_t nelem)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= nelem)
        return;
    us[i] += other1[i]*other2[i];
    __syncthreads();   // need to sync here.
}
extern "C" void addProductMat (real *us, real *other1, real *other2, size_t nelem)
{
    cuaddproductmat<<<dim3((((unsigned int)nelem)+31)/32), 32>>>(us, other1, other2, nelem);
    cudaDeviceSynchronize();
}


__global__ void cuadddotMultiplyMat (real *us, real *peelhole, real *c, size_t nr, size_t nc)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nr || j >= nc)
    {
        return;
    }
    us[i+j*nr] += peelhole[i]*c[i+j*nr];
    __syncthreads();   // need to sync here.
}
extern "C" void adddotMultiplyMat (real *us, real *peelhole, real *c, size_t nr, size_t nc)
{
    dim3 t(32, 16);
    dim3 b((nr+t.x-1)/t.x, (nc+t.y-1)/t.y);
    cuadddotMultiplyMat<<<b, t>>> (us, peelhole, c, nr, nc);
}

__global__ void cuaddOneMinusMatrix (real *us, real *other, size_t nelem)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= nelem)
        return;
    us[i] += 1 - other[i];
    __syncthreads();   // need to sync here.
}
extern "C" void addOneMinusMatrix (real *us, real *other, size_t nelem)
{
    cuaddOneMinusMatrix<<<dim3((((unsigned int)nelem)+31)/32), 32>>>(us, other, nelem);
    cudaDeviceSynchronize();
}


__global__ void cusubtractmatrix (real *us, real *other, size_t nelem)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= nelem)
        return;
    us[i] -= other[i];
    __syncthreads();   // need to sync here.
}
extern "C" void subtractmatrix (real *us, real *other, size_t nelem)
{
    cusubtractmatrix<<<dim3((((unsigned int)nelem)+31)/32), 32>>>(us, other, nelem);
    cudaDeviceSynchronize();
}

__global__ void cuaddadagradij (real *us, real *dU, real *accsdU, float alpha, float l2reg, size_t nr, size_t nc)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nr || j >= nc)
    {
        return;
    }
    us[i+j*nr] = (1 - l2reg) * us[i+j*nr] + alpha * dU[i+j*nr] / (sqrtf(accsdU[i+j*nr])+ 1e-3);
    __syncthreads();   // need to sync here.
}

extern "C" void cuaddadagrad (real *us, real *dU, real *accsdU, float alpha, float l2reg, size_t nr, size_t nc)
{
    dim3 t(32, 16);
    dim3 b((nr+t.x-1)/t.x, (nc+t.y-1)/t.y);
    cuaddadagradij<<<b, t>>> (us, dU, accsdU, alpha, l2reg, nr, nc);
}

__global__ void cuaddgradij (real *us, real *other, float alpha, float l2reg, size_t nr, size_t nc)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nr || j >= nc)
    {
        return;
    }
    us[i+j*nr] = (1-l2reg)*us[i+j*nr] + alpha*other[i+j*nr];
    __syncthreads();   // need to sync here.
}

extern "C" void cuaddgrad (real *us, real *other, float alpha, float l2reg, size_t nr, size_t nc)
{
    dim3 t(32, 16);
    dim3 b((nr+t.x-1)/t.x, (nc+t.y-1)/t.y);
    cuaddgradij <<<b, t>>> (us, other, alpha, l2reg, nr, nc);
}

__global__ void cuaddsquaregradij (real *us, real *other, float gamma, float beta, size_t nr, size_t nc)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nr || j >= nc)
    {
        return;
    }
    us[i+j*nr] = gamma * us[i+j*nr] + beta * other[i+j*nr] * other[i+j*nr];
    __syncthreads();   // need to sync here.
}

extern "C" void cuaddsquaregrad (real *us, real *other, float gamma, float beta, size_t nr, size_t nc)
{
    dim3 t(32, 16);
    dim3 b((nr+t.x-1)/t.x, (nc+t.y-1)/t.y);
    cuaddsquaregradij<<<b, t>>> (us, other, gamma, beta, nr, nc);
}

__global__ void cuaddpeepholegradij (real *us, real *other, float alpha, float l2reg, size_t nr, size_t nc)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= nr)
        return;
    for (int j=0; j<nc; j++)
    {
        us[i] = (1-l2reg)*us[i] + alpha*other[i+j*nr];
    }
    __syncthreads();   // need to sync here.
}
extern "C" void cuaddpeepholegrad (real *us, real *other, float alpha, float l2reg, size_t nr, size_t nc)
{
    cuaddpeepholegradij<<<dim3((((unsigned int)nr)+31)/32), 32>>> (us, other, alpha, l2reg, nr, nc);
}

__global__ void cuaddgrad_word (real *layer0_word, real *thisgrad, int *prevwords, size_t nr, size_t nc, size_t mb)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    if (i >= nc)
    {
        return;
    }
    for (int t=0; t<mb; t++)
    {
        int word = prevwords[t];
        if (word != INVALID_INT)
            layer0_word[word+nr*i] += thisgrad[word+nr*i];
    }
    for (int t=0; t<mb; t++)
    {
        int word = prevwords[t];
        if (word != INVALID_INT)
            thisgrad[word+nr*i] = 0;
    }
    __syncthreads();   // need to sync here.
}

extern "C" void addgrad_word_v1 (real *layer0_word, real *thisgrad, int *prevwords, size_t nr, size_t nc, size_t mb)
{
    dim3 t(1024);
    dim3 b((nc+t.x-1)/t.x);
    cuaddgrad_word<<<b, t>>> (layer0_word, thisgrad, prevwords, nr, nc, mb);
}

__global__ void cuaddsubmatrix (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t colwidth_us, size_t colwidth_other,real alpha, real beta)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nr || j >= nc)
    {
        return;
    }
    us[i+i0+(j+j0)*colwidth_us] = alpha * other[i+j*colwidth_other] + beta * us[i+i0+(j+j0)*colwidth_us];
    __syncthreads();   // need to sync here.
}
extern "C" void addsubmatrix_v1 (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t colwidth_us, size_t colwidth_other,real alpha, real beta)
{
    dim3 t(32, 16);
    dim3 b((nr+t.x-1)/t.x, (nc+t.y-1)/t.y);
    cuaddsubmatrix<<<b, t>>>(us, other, i0, j0, nr, nc, colwidth_us, colwidth_other, alpha, beta);
    cudaDeviceSynchronize();
}


__global__ void cuassignsubmatrix (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t colwidth_us, size_t colwidth_other)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nr || j >= nc)
    {
        return;
    }
    us[i+i0+(j+j0)*colwidth_us] = other[i+j*colwidth_other];
}
extern "C" void assignsubmatrix_v1 (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t colwidth_us, size_t colwidth_other)
{
    dim3 t(32, 16);
    dim3 b((nr+t.x-1)/t.x, (nc+t.y-1)/t.y);
    cuassignsubmatrix<<<b, t>>> (us, other, i0, j0, nr, nc, colwidth_us, colwidth_other);
    cudaDeviceSynchronize();
}


__global__ void cugetsubmatrix_v1 (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t rowwidth_other, size_t colwidth_other)
{
    const size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    if (i >= nr || j >= nc)
    {
        return;
    }
    us[i + j*nr] = other[i+i0 + (j+j0)*rowwidth_other];
}
extern "C" void cugetsubmatrix (real *us, real *other, size_t i0, size_t j0, size_t nr, size_t nc, size_t rowwidth_other, size_t colwidth_other)
{
    dim3 t(32, 16);
    dim3 b((nr+t.x-1)/t.x, (nc+t.y-1)/t.y);
    cugetsubmatrix_v1<<<b,t>>> (us, other, i0, j0, nr, nc, rowwidth_other, colwidth_other);
}

extern "C" void GPUsynchronizewithCPU ()
{
    cudaDeviceSynchronize();
}

__global__ void cucalerronoutputlayer_v1 (real *us, real *ac, int *words, size_t rowwidth, size_t colwidth)
{
    size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    int curword = words[j];
    if (j >= colwidth)
        return;
    if (curword == INVALID_INT)
    {
        while (i < rowwidth)
        {
            us[i+j*rowwidth] = 0.0;
            i += blockDim.x * gridDim.x;
        }
    }
    else
    {
        while (i < rowwidth)
        {
            if (i != curword)
            {
                us[i + j*rowwidth] = 0.0 - ac[i + j*rowwidth];
            }
            else
            {
                us[i + j*rowwidth] = 1.0 - ac[i + j*rowwidth];
            }
            i += blockDim.x * gridDim.x;
        }
    }
}
extern "C" void cucalerronoutputlayer (real *us, real *ac, int *words, size_t rowwidth, size_t colwidth)
{
    dim3 t(32, 16);
    dim3 b(32, (colwidth+t.y-1)/16);
    cucalerronoutputlayer_v1<<<b, t>>> (us, ac, words, rowwidth, colwidth);
    cudaDeviceSynchronize();
}

__global__ void cucalerronoutputlayer_vr_v1 (real *us, real *ac, int *words, size_t rowwidth, size_t colwidth, real *lognorms, real vrpenalty)
{
    size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t j = threadIdx.y + (blockIdx.y * blockDim.y);
    int curword = words[j];
    if (j >= colwidth)
        return;
    if (curword == INVALID_INT)
    {
        while (i < rowwidth)
        {
            us[i+j*rowwidth] = 0.0;
            i += blockDim.x * gridDim.x;
        }
    }
    else
    {
        while (i < rowwidth)
        {
            if (i != curword)
            {
                us[i + j*rowwidth] = 0.0 - ac[i + j*rowwidth] - vrpenalty*(lognorms[j]) * ac[i+j*rowwidth];
            }
            else
            {
                us[i + j*rowwidth] = 1.0 - ac[i + j*rowwidth] - vrpenalty*(lognorms[j])*ac[i+j*rowwidth];
            }
            i += blockDim.x * gridDim.x;
        }
    }
}

extern "C" void cucalerronoutputlayer_vr (real *us, real *ac, int *words, size_t rowwidth, size_t colwidth, real *lognorms, real vrpenalty)
{
    dim3 t(32, 16);
    dim3 b(32, (colwidth+t.y-1)/16);
    cucalerronoutputlayer_vr_v1<<<b, t>>> (us, ac, words, rowwidth, colwidth, lognorms, vrpenalty);
    cudaDeviceSynchronize();
}

__global__ void cusample (float *data, int nr, int nc, int *sample, float *randv)
{
    // const size_t mbid = threadIdx.x;
    // size_t i = threadIdx.x + (blockIdx.x * blockDim.x);
    const size_t mbid = threadIdx.y + (blockIdx.y * blockDim.y);
    if (mbid > nc) return;
    float f = randv[mbid];
    float g = 0;
    int i = 0;
    while ((g < f) && (i < nr))
    {
        g += data[i+mbid*nr];
        i++;
    }
    sample[mbid] = i-1;
}

extern "C" void sample_v1 (float *data, int nr, int nc, int *sample, float *randv)
{
    dim3 t(1, 32);
    dim3 b(1, (nc+t.y-1)/32);
    cusample<<<b, t>>> (data, nr, nc, sample, randv);
}


__global__ void cuforwardWordlayer_v1 (float *weights, float *srcac, float *tgtac, int *curclass, int *classinfo, int l1, int l2, int mbsize)
{
    const size_t mbid = blockIdx.x;
    if (mbid >= mbsize) return;
    size_t clsid = curclass[mbid];
    size_t swordid = classinfo[clsid*3];
    size_t ewordid = classinfo[clsid*3+1];
    size_t nword   = classinfo[clsid*3+2];
    if (threadIdx.x >= nword) return;
    size_t wordid = swordid+threadIdx.x;
    float v=0.0;
    while (wordid <= ewordid)
    {
        for (int i=0; i<l1; i++)
        {
            v += srcac[i+l1*mbid]*weights[i+wordid*l1];
        }
        tgtac[wordid+l2*mbid] = v;
        wordid += gridDim.x;
    }
}

__global__ void cuforwardWordlayer_v2 (float *weights, float *srcac, float *tgtac, int *curclass, int *classinfo, int l1, int l2, int mbsize)
{
    __shared__ float cache[1024];
    const size_t i = threadIdx.x + (threadIdx.y*blockDim.x);
    if (i>=l1) return;
    const size_t mbid = blockIdx.x;
    if (mbid >= mbsize) return;
    size_t clsid = curclass[mbid];
    size_t swordid = classinfo[clsid*3];
    size_t ewordid = classinfo[clsid*3+1];
    size_t nword   = classinfo[clsid*3+2];
    if (blockIdx.y >= nword) return;
    size_t wordid = swordid+blockIdx.y;
    size_t colwidth = blockDim.x*blockDim.y;
    cache[i] = 0;
    while (wordid <= ewordid)
    {
        int index = i;
        while (index < l1)
        {
            cache[index] += srcac[index+l1*mbid]*weights[index+wordid*l1];
            index += colwidth;
        }
        __syncthreads();

        index = colwidth/2;
        while (index != 0)
        {
            if (i<index && (i+index)<l1)
            {
                cache[i] += cache[i+index];
            }
            __syncthreads();
            index /= 2;
        }
        if (i==0) tgtac[wordid+l2*mbid] = cache[0];
        wordid += gridDim.y;
    }
}

extern "C" void cuforwardWordlayer (float *weights, float *srcac, float *tgtac, int *curclass, int *classinfo, int l1, int l2, int mbsize)
{
#if 0
    dim3 t(512);
    dim3 b(mbsize);
    cuforwardWordlayer_v1<<<b, t>>>(weights, srcac, tgtac, curclass, classinfo, l1, l2, mbsize);
#else
    dim3 t(32, 32);
    dim3 b(mbsize, 512);
    cuforwardWordlayer_v2<<<b, t>>>(weights, srcac, tgtac, curclass, classinfo, l1, l2, mbsize);
#endif
}


__global__ void cusoftmaxWordlayer_v1 (float *ac, int *curclass, int *classinfo, int l2, int mbsize)
{
    size_t mbid = blockIdx.x;
    size_t clsid = curclass[mbid];
    size_t swordid = classinfo[clsid*3];
    size_t ewordid = classinfo[clsid*3+1];
    size_t nword = classinfo[clsid*3+2];
    float maxv = -1, v;
    float sum = 0.0;
    for (int i=swordid; i<=ewordid; i++)
    {
        v = ac[i+l2*mbid];
        if (maxv < v)  maxv = v;
    }
    v = 0;
    for (int i=swordid; i<=ewordid; i++)
    {
        v = ac[i+l2*mbid]-maxv;
        ac[i+l2*mbid] = exp (v);
        sum += ac[i+l2*mbid];
    }
    for (int i=swordid; i<=ewordid; i++)
    {
        v = ac[i+l2*mbid];
        ac[i+l2*mbid] = v/sum;
    }
}

__global__ void cusoftmaxWordlayer_v2 (float *ac, int *curclass, int *classinfo, int nrows, int mbsize)
{
    __shared__ float cache[1024];
    const size_t i = threadIdx.x + (threadIdx.y * blockDim.x);
    size_t mbid = blockIdx.x + (blockIdx.y * gridDim.x);
    if (mbid >= mbsize)  return;
    size_t clsid = curclass[mbid];
    size_t swordid = classinfo[clsid*3];
    size_t ewordid = classinfo[clsid*3+1];
    size_t nword = classinfo[clsid*3+2];
    if (i >= nword) return;
    float temp = 0.0;
    float colmax = -1e5;
    size_t colwidth = blockDim.x*blockDim.y;

    // compute the max value in each column
    size_t wordid = swordid + i;
    cache[i] = ac[wordid+mbid*nrows];
    int idx = wordid + colwidth;
    while (idx <= ewordid)
    {
        float usij = ac[mbid*nrows+idx];
        if (usij > cache[i])
        {
            cache[i] = usij;
        }
        idx += colwidth;
    }
    __syncthreads();
    idx = colwidth / 2;
    while (idx != 0)
    {
        if (i<idx && (i+idx)<nword)
        {
            if (cache[i] < cache[i+idx])
            {
                cache[i] = cache[i+idx];
            }
        }
        __syncthreads();
        idx = idx/2;
    }
    colmax = cache[0];
    __syncthreads();

    // compute sum(exp(i))
    int index = i+swordid;
    while (index <= ewordid)
    {
        ac[mbid*nrows+index] = expf(ac[mbid*nrows+index]-colmax);
        temp += ac[mbid*nrows+index];
        index += colwidth;
    }
    cache[i] = temp;
    __syncthreads();
    index = colwidth/2;
    while (index != 0)
    {
        if (i < index && (i+index)<nword)
        {
            cache[i] += cache[i+index];
        }
        __syncthreads();
        index /= 2;
    }
    // compute  exp(i)/sum
    index = i+swordid;
    while (index <= ewordid)
    {
        ac[mbid*nrows+index] = ac[mbid*nrows+index] / cache[0];
        index += colwidth;
    }
    __syncthreads();
}

extern "C" void cusoftmaxWordlayer (float *ac, int *curclass, int *classinfo, int l2, int mbsize)
{
#if 0
    dim3 t(1);
    dim3 b(mbsize);
    cusoftmaxWordlayer_v1<<<b, t>>> (ac, curclass, classinfo, l2, mbsize);
#else
    dim3 t(32, 32);
    dim3 b((mbsize+31)/32, 32);
    cusoftmaxWordlayer_v2<<<b, t>>> (ac, curclass, classinfo, l2, mbsize);
#endif
    cudaDeviceSynchronize();
}

__global__ void cucalerronWordlayer_v1 (float *er, float *ac, int *curclass, int *curwords, int *classinfo, int l2, int mbsize)
{
    size_t tid = threadIdx.x;
    size_t mbid = blockIdx.x;
    size_t clsid = curclass[mbid];
    size_t swordid = classinfo[clsid*3];
    size_t ewordid = classinfo[clsid*3+1];
    size_t nword   = classinfo[clsid*3+2];
    if (tid >= nword) return;
    size_t wordid = tid + swordid;
    int tgtwordid = curwords[mbid];
    while (wordid <= ewordid)
    {
        if (wordid == tgtwordid)
        {
            er[wordid+mbid*l2] = 1 - ac[wordid+mbid*l2];
        }
        else
        {
            er[wordid+mbid*l2] = -ac[wordid+mbid*l2];
        }
        wordid += blockDim.x;
    }
}

extern "C" void cucalerronWordlayer (float *er, float *ac, int *curclass, int *curwords, int *classinfo, int l2, int mbsize)
{
    dim3 t(1024);
    dim3 b(mbsize);
    cucalerronWordlayer_v1<<<b, t>>> (er, ac, curclass, curwords, classinfo, l2, mbsize);
}

__global__ void cubperWordlayer_v1 (float *weights, float *srcer, float *tgter, int *curclass, int *classinfo, int l1, int l2, int mbsize, float alpha, float beta)
{
    size_t tid = threadIdx.x;
    size_t mbid = blockIdx.x;
    if (tid >= l1) return;
    if (mbid >= mbsize) return;
    size_t clsid = curclass[mbid];
    size_t swordid = classinfo[clsid*3];
    size_t ewordid = classinfo[clsid*3+1];
    while (tid < l1)
    {
        for (int wordid=swordid; wordid<=ewordid; wordid++)
        {
            tgter[tid+l1*mbid] = alpha*srcer[wordid+l2*mbid]*weights[tid+l1*wordid] + beta*tgter[tid+l1*mbid];
        }
        tid += blockDim.x;
    }
}

__global__ void cubperWordlayer_v2 (float *weights, float *srcer, float *tgter, int *curclass, int *classinfo, int l1, int l2, int mbsize, float alpha, float beta)
{
    __shared__ float cache[1024];
    const size_t i = threadIdx.x + (threadIdx.y * blockDim.x);
    size_t tid  = blockIdx.y;
    size_t mbid = blockIdx.x;
    if (tid >= l1) return;
    if (mbid >= mbsize) return;
    size_t clsid = curclass[mbid];
    size_t swordid = classinfo[clsid*3];
    size_t ewordid = classinfo[clsid*3+1];
    size_t nword = classinfo[clsid*3+2];
    if (i >= nword) return;
    size_t colwidth = blockDim.x*blockDim.y;
    size_t wordid;
    cache[i] = 0;
    while (tid < l1)
    {
        int index = i;
        while (index < nword)
        {
            wordid = swordid+index;
            cache[index] += srcer[wordid+l2*mbid]*weights[tid+l1*wordid];
            index += colwidth;
        }
        __syncthreads();

        index = colwidth/2;
        while (index != 0)
        {
            if (i<index && (i+index)<nword)
            {
                cache[i] += cache[i+index];
            }
            __syncthreads();
            index /= 2;
        }
        if (i==0) tgter[tid+l1*mbid] = alpha*cache[0] + beta*tgter[tid+l1*mbid];
        tid += gridDim.y;
    }
}

extern "C" void cubperWordlayer (float *weights, float *srcer, float *tgter, int *curclass, int *classinfo, int l1, int l2, int mbsize, float alpha, float beta)
{
#if 0
    dim3 t(1024);
    dim3 b(mbsize);
    cubperWordlayer_v1<<<b, t>>> (weights, srcer, tgter, curclass, classinfo, l1, l2, mbsize, alpha, beta);
#else
    dim3 t(32, 32);
    dim3 b(mbsize, 1024);
    cubperWordlayer_v2<<<b, t>>> (weights, srcer, tgter, curclass, classinfo, l1, l2, mbsize, alpha, beta);
#endif
}

__global__ void cubpupdateWordlayer_v1 (float *ac, float *er, float *weights, int *curclass, int *classinfo, int l1, int l2, int mbsize, float alpha, float beta)
{
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    if (tid >= l1) return;
    while (tid <= l1)
    {
        for (int mbid=0; mbid<mbsize; mbid++)
        {
            size_t clsid = curclass[mbid];
            size_t swordid = classinfo[clsid*3];
            size_t ewordid = classinfo[clsid*3+1];
            size_t wordid = swordid + bid;
            while (wordid <= ewordid)
            {
                weights[tid+wordid*l1] = alpha*ac[tid+mbid*l1]*er[wordid+mbid*l2] + beta*weights[tid+wordid*l1];
                wordid += gridDim.x;
            }
        }
        tid += blockDim.x;
    }
}

__global__ void cubpupdateWordlayer_v2 (float *ac, float *er, float *weights, int *curclass, int *classinfo, int l1, int l2, int mbsize, float alpha, float beta)
{
    size_t tid = threadIdx.x;
    size_t bid = blockIdx.x;
    if (tid >= l1) return;
    while (tid <= l1)
    {
        for (int mbid=0; mbid<mbsize; mbid++)
        {
            size_t clsid = curclass[mbid];
            size_t swordid = classinfo[clsid*3];
            size_t ewordid = classinfo[clsid*3+1];
            size_t wordid = swordid + bid;
            while (wordid <= ewordid)
            {
                weights[tid+wordid*l1] = alpha*ac[tid+mbid*l1]*er[wordid+mbid*l2] + beta*weights[tid+wordid*l1];
                wordid += gridDim.x;
            }
        }
        tid += blockDim.x;
    }
}

extern "C" void cubpupdateWordlayer (float *ac, float *er, float *weights, int *curclass, int *classinfo, int l1, int l2, int mbsize, float alpha, float beta)
{
#if 0
    dim3 t(1024);
    dim3 b(2000);
    cubpupdateWordlayer_v1<<<b, t>>> (ac, er, weights, curclass, classinfo, l1, l2, mbsize, alpha, beta);
#else
    dim3 t(1024);
    dim3 b(2000);
    cubpupdateWordlayer_v2<<<b, t>>> (ac, er, weights, curclass, classinfo, l1, l2, mbsize, alpha, beta);
#endif
}

__global__ void cucopyOutputWgtsforNCE_v1 (float *srcwgt, float *dstwgt, int *sample, int nsample, int nrow, int spos)
{
    const size_t rowid = threadIdx.x + blockIdx.x*blockDim.x;
    const size_t dstcolid = threadIdx.y + blockIdx.y*blockDim.y;
    if (rowid >= nrow || dstcolid >= nsample)
    {
        return;
    }
    {
        int srccolid = sample[dstcolid];
        if (srccolid == INVALID_INT)        // NULL node
        {
            dstwgt[rowid+dstcolid*nrow + spos*nrow] = 0;
        }
        else
        {
            dstwgt[rowid+dstcolid*nrow + spos*nrow] = srcwgt[rowid+srccolid*nrow];
        }
    }
}

extern "C" void cucopyOutputWgtsforNCE(float *srcwgt, float *dstwgt, int *sample, int nsample, int nrow, int spos)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (nsample+t.y-1)/t.y);
    cucopyOutputWgtsforNCE_v1<<<b, t>>> (srcwgt, dstwgt, sample, nsample, nrow, spos);
}

__global__ void cucalerronOutputLayer_v1 (float *er, float *ac, float *log_noise, int *curwords, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample, int nrow, int mbsize)
{
    const size_t rowid = threadIdx.x + blockIdx.x*blockDim.x;
    const size_t colid = threadIdx.y + blockIdx.y*blockDim.y;
    if (rowid >= nrow || colid >= mbsize)
    {
        return;
    }
    if (curwords[colid] == INVALID_INT)
    {
        er[rowid+colid*nrow] = 0;
        return;
    }
    if (rowid < ntargetsample)      // target
    {
        int wordid = targetsample[rowid];
        if (wordid == INVALID_INT)      // NULL NODE
        {
            er[rowid+colid*nrow] = 0;
        }
        else
        {
            if (mbid2arrid[colid] == rowid)
            {
                float v = ac[rowid+colid*nrow];
                float log_er = v - culogadd(v, log_noise[wordid]);
                v = 1 - exp (log_er);
                er[rowid+colid*nrow] = 1 - exp (log_er);
            }
            else
            {
                er[rowid+colid*nrow] = 0;
            }
        }
    }
    else        // noise sample
    {
        int wordid = ncesample[rowid-ntargetsample];
        float v = ac[rowid+colid*nrow];
        float log_er = v - culogadd(v, log_noise[wordid]);
        v = -exp (log_er);
        er[rowid+colid*nrow] = v * ncesamplecnt[rowid-ntargetsample];
    }
}

extern "C" void cucalerronOutputLayer (float *er, float *ac, float *log_noise, int *curwords, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample, int nrow, int mbsize)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (mbsize+t.y-1)/t.y);
    cucalerronOutputLayer_v1<<<b, t>>> (er, ac, log_noise, curwords, targetsample, ncesample, ncesamplecnt, mbid2arrid, ntargetsample, nncesample, nrow, mbsize);
}


__global__ void cucalerronOutputLayer_oldversion_v1 (float *er, float *ac, float *er_mask, float *log_noise, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample, int nrow, int mbsize)
{
    const size_t rowid = threadIdx.x + blockIdx.x*blockDim.x;
    const size_t colid = threadIdx.y + blockIdx.y*blockDim.y;
    if (rowid >= nrow || colid >= mbsize)
    {
        return;
    }
    if (rowid < ntargetsample)      // target
    {
        int wordid = targetsample[rowid];
        if (mbid2arrid[colid] == rowid)
        {
            float v = ac[rowid+colid*nrow];
            float log_er = v - culogadd(v, log_noise[wordid]);
            v = 1 - exp (log_er);
            er[rowid+colid*nrow] = v;
        }
        else
        {
            er[rowid+colid*nrow] = 0;
        }
    }
    else        // noise sample
    {
        int wordid = ncesample[rowid-ntargetsample];
        float v = ac[rowid+colid*nrow];
        float log_er = v - culogadd(v, log_noise[wordid]);
        v = -exp (log_er);
#ifdef NCE_SHARESAMPLE
        er[rowid+colid*nrow] = v * ncesamplecnt[rowid-ntargetsample];
#else
        er[rowid+colid*nrow] = v * er_mask[rowid-ntargetsample+colid*nrow];
#endif
    }
}

extern "C" void cucalerronOutputLayer_oldversion (float *er, float *ac, float *er_mask, float *log_noise, int *targetsample, int *ncesample, int *ncesamplecnt, int *mbid2arrid, int ntargetsample, int nncesample, int nrow, int mbsize)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (mbsize+t.y-1)/t.y);
    cucalerronOutputLayer_oldversion_v1<<<b, t>>> (er, ac, er_mask, log_noise, targetsample, ncesample, ncesamplecnt, mbid2arrid, ntargetsample, nncesample, nrow, mbsize);
}

__global__ void cumulScalarij (float *us, int nrow, int ncol, float gradient_cutoff)
{
    const size_t rowid = threadIdx.x + blockIdx.x*blockDim.x;
    const size_t colid = threadIdx.y + blockIdx.y*blockDim.y;
    if (rowid >= nrow || colid >= ncol)
    {
        return;
    }
    us[rowid+nrow*colid] *= gradient_cutoff;
#if 0
        // need to optimize later
        if (us[rowid+nrow*colid] > gradient_cutoff)
        {
            us[rowid+nrow*colid] = gradient_cutoff;
        }
#endif

}

extern "C" void cumulScalar (float *us, int nrow, int ncol, float gradient_cutoff)
{
    dim3 t(32, 16);
    dim3 b((nrow+t.x-1)/t.x, (ncol+t.y-1)/t.y);
    cumulScalarij <<<b, t>>> (us, nrow, ncol, gradient_cutoff);
}


__global__ void cucalnorm2_v1 (float *us, int num, int minibatch, float *norm2)
{
    norm2[0] = 0.0;
    __shared__ float cache[1024];
    const size_t i = threadIdx.x + (threadIdx.y * blockDim.x);
    if (i >= num)
    {
        return;
    }
    int idx = i;
    cache[i] = 0;
    while (idx < num)
    {
        float usi = us[idx]/minibatch;
        // float usi = us[idx];
        cache[i] += usi * usi;
        idx += blockDim.x*blockDim.y;
    }
    __syncthreads();
    idx = blockDim.x*blockDim.y/2;
    while (idx != 0)
    {
        if (i < idx)
        {
            cache[i] += cache[i+idx];
        }
        __syncthreads();
        idx = idx/2;
    }
    norm2[0] = sqrtf(cache[0]);
    __syncthreads();
}

extern "C" void cucalnorm2 (float *dev_data, int num, int minibatch, float *norm2)
{
    dim3 t(32, 32);
    dim3 b(1, 1);
    cucalnorm2_v1 <<<b, t>>> (dev_data, num, minibatch, norm2);
}

__global__ void cuaddgrad_NCE_v1 (float *us, float *gradwgt, int *sample, int nsample, float alpha, int nrow, int ncol, int spos)
{
    const size_t rowid = threadIdx.x + blockIdx.x*blockDim.x;
    size_t colid = threadIdx.y + blockIdx.y*blockDim.y;
    if (rowid >= nrow || colid >= nsample)
    {
        return;
    }
    int wordid = sample[colid];
    if (wordid != INVALID_INT)      // only add for NON-NULL token
    {
        colid = colid + spos;
        us[rowid+nrow*wordid] += alpha*gradwgt[rowid+nrow*colid];
    }
}

extern "C" void cuaddgrad_NCE (float *us, float *gradwgt, int *targetsample, int ntargetsample, int *ncesample, int nncesample, float alpha, int nrow, int ncol)
{
    dim3 t(32, 16);
    dim3 b1((nrow+t.x-1)/t.x, (ntargetsample+t.y-1)/t.y);
    cuaddgrad_NCE_v1<<<b1, t>>> (us, gradwgt, targetsample, ntargetsample, alpha, nrow, ncol, 0);
    dim3 b2((nrow+t.x-1)/t.x, (nncesample+t.y-1)/t.y);
    cuaddgrad_NCE_v1<<<b2, t>>> (us, gradwgt, ncesample, nncesample, alpha, nrow, ncol, ntargetsample);
}

#endif
