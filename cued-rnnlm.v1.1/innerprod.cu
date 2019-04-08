__global__ void innerprod_sum_ij (real *us, size_t nrows, size_t ncols, float *sum)
{

    __shared__ real cache[1024];
    const size_t i = threadIdx.x + (threadIdx.y * blockDim.x);
    const size_t j = blockIdx.x + (blockIdx.y * gridDim.x);
    if (i >= nrows)
    {
        return;
    }
    if (j >= ncols)
    {
        return;
    }
    real temp = 0.0f;
    int index = i;
    while (index < nrows)
    {
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
    sum[j] = cache[0];
}

extern "C" void innerprod_sum(real *us, size_t nrows, size_t ncols, float *sum)
{
    dim3 t(32, 32);
    dim3 b((mb+31)/32, 32);
    innerprod_sum_ij <<<b, t>>> (us, nrows, ncols, sum);
    cudaDeviceSynchronize();
}
