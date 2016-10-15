#include "basedefs.h"

// #define WGSIZE 32  // must be a power of 2

/**
Partially reduces <idata> as odata <- max(idata).

More specifically, reduces a number of 2 * <WGSIZE> elements of <iData> to one
element in <oData> finding their maximum value, and fills up <oData> from its
beginning with the results in consecutive order. Implementation based on [1],
followed up to reduction #4 (i.e., no unrolling).

Assumes 1D arrays, assumes to be called with a global size g that adheres to
(iDataSize / 2.) <= g <= iDataSize.

References
[1] M. Harris, “Optimizing Parallel Reduction in CUDA,”
    http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf (20150219)

@param[in]  iData the data to be reduced
@param[out] oData the reduction result
@param[in]  iDataSize the number of actual elements in <iData>; must be >=
            <WGSIZE>
*/
__kernel 
__attribute__((reqd_work_group_size(WGSIZE, 1, 1)))
void redMax(__global const ftype *iData,
            __global ftype *oData,
            const itype iDataSize)
{
    const size_t iGlobal = get_global_id(0);
    const size_t iLocal = get_local_id(0);
    
    __local ftype scratch[WGSIZE];
    
    // Load elements from global to shared memory, calculating first reduction
    // on the fly. We sometimes might jump beyond the array boundary here, as
    // we chose the global size only roughly equal to half of the iDataSize.
    // In this case, NAN will behave as the neutral element.
    const itype iPartner = iGlobal + get_global_size(0);
    scratch[iLocal] = fmax(iData[iGlobal],
                           (iPartner < iDataSize) ? iData[iPartner] : NAN);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Calculate reduction in the shared memory
    for (uint s=WGSIZE>>1; s>0; s>>=1)
    {
        if (iLocal < s)
        {
            scratch[iLocal] = fmax(scratch[iLocal], scratch[iLocal + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write workgroup result to global memory
    if (iLocal == 0)
        oData[get_group_id(0)] = scratch[0];
}
