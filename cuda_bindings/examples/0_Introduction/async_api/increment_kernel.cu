// #include <thrust/device_ptr.h>
// #include <thrust/count.h>
// #include <thrust/execution_policy.h>

// #include <iostream>
#include <cstdio>

extern "C" __global__ void increment_kernel(int *g_data, int inc_value)
{
    int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    g_data[idx] = g_data[idx] + inc_value;
}