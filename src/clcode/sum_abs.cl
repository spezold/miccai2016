
// #define WGSIZE 32  // must be a power of 2

/*
Note: Be sure to read the requirement for <idata_size> w.r.t. <WGSIZE> below!

Partially reduces idata as

    odata <- sum(abs(idata)),
    
implementation based on [3], followed up to reduction #4 (i.e., no unrolling).
Uses pairwise summation [2] to mitigate the numerical error inherent in
floating point arithmetic.

Assumes 1D arrays for <idata> and <odata>, assumes to be called with a global
size >= (idata_size / 2.).

References:
[2] http://en.wikipedia.org/wiki/Pairwise_summation (20150219)
[3] M. Harris, “Optimizing Parallel Reduction in CUDA,”
    http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf (20150219)

@param idata_size Must be >= WGSIZE
*/
__kernel 
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
void sum_abs(__global const float *idata,
             __global float *odata,
             const int idata_size)
{
    const size_t i_global = get_global_id(0);
    const size_t i_local = get_local_id(0);
    
    __local float scratch[WGSIZE];
    
    // Load elements from global to shared memory, take absolute value,
    // calculate first reduction (we sometimes jump beyond the array
    // boundary here, as we chose global_size only roughly equal to half of the
    // idata_size)
    const int i_partner = i_global + get_global_size(0);
    scratch[i_local] = fabs(idata[i_global]) + (i_partner < idata_size ? fabs(idata[i_partner]) : 0.0f);
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate reduction in the shared memory
    for (uint s=WGSIZE>>1; s>0; s>>=1)
    {
        if (i_local < s)
        {
            scratch[i_local] += scratch[i_local + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write workgroup result to global memory
    if (i_local == 0)
        odata[get_group_id(0)] = scratch[0];
}
