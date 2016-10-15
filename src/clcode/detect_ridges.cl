#include "basedefs.h"
#include "quaternion_math.h"
#include "matrix_math.h"

/**
Detect ridges in Frangi vesselness response.
*/

// #define THRESHOLD
// ^ Absolute value below which values in iResponse will not be considered as
// ridge candidates -- for noise robustness

/**
From the given Frangi response and its respective directions, detect the ridges
in the response by determining all local maxima perpendicular to the current
point's main direction.

Assumes 3D arrays of identical size (for <iEigs>, see below). Assumes to be
called with the reversed image shape.

@param[out] oRidge the ridge detection result, ridge points labeled with value
            1.0, other points labeled with 0.0; may *not* be the same as
            <iResponse>
@param[in]  iResponse Frangi filter response; may *not* be the same as <oRidge>
@param[in]  iEigs the eigenvectors that produced the Frangi filter response.
            The eigenvectors are supposed to be ordered in ascending order of
            their eigenvalues' absolute value, placed in the columns of a
            matrix, converted to the respective unit quaternion, and then
            "squeezed" to three values via the qsqueeze() function. The three
            resulting values for each point are supposed to be given in
            consecutive order; thus <iEigs> is supposed to have exactly three
            times the size of <iResponse> and <oRidge> 
*/
__kernel void detectRidges(__global ftype *oRidge,
                           __global const ftype *iResponse,
                           __global const ftype *iEigs)
{
    const size_t i0 = get_global_id(2);
    const size_t i1 = get_global_id(1);
    const size_t i2 = get_global_id(0);
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);
    
    const itype i = i2 + dim2 * (i1 + dim1 * i0);  // current index
    
    // Get the current directions
    vec vDir = {iEigs[3 * i], iEigs[3 * i + 1], iEigs[3 * i + 2]};
    qat qDir;
    qexpand(vDir, qDir);
    float3 dir1, dir2;
    quat2matFloat3(qDir, (float3*)0, &dir1, &dir2);  // Don't need direction 0 here
    
    float3 pos = (float3)(i0, i1, i2);

    ftype maxVal = -INFINITY;
    for(itype j = 0; j < 8; ++j)
    {
        // Get current neighboring coordinate (this cycles over all neighbors,
        // i.e. NW, N, NE, W, E, SW, S, SE -- but in a different order)
        float3 pos_j = pos + ((-1) * (j < 3) + (j > 4)) * dir1 +
                             ((j % 3) - 1 + (j == 4))   * dir2;
        
        int3 origin_j = convert_int3(pos_j);
        
        // Make it relative to its enclosing unit cube
        pos_j = pos_j - convert_float3(origin_j);
        
        // Get its value by trilinear interpolation [*]
        //
        // References
        // [*] http://paulbourke.net/miscellaneous/interpolation/ (20150917)
        ftype val_j = 0;
        for(itype k = 0; k < 8; ++k)
        {
            
            // Get current unit cube corner (this cycles over [0,0,0], ...,
            // [1,1,1], converting the bits of k to 0 and 1, respectively)
            int3 crnRel_k = (int3)(k >> 2, (k >> 1) & 1, k & 1);
            // ^ No need for &1 in the k>>2 case, as higher bits don't get set;
            // need int3 here for use in "?:" with float below (according to
            // OpenCL spec, types must have same number of bits)
            int3 crnAbs_k = origin_j + crnRel_k;
            
            // Add current corner's value, inversely weighted by the distance
            // to the current position, which, when summed up for all corners,
            // results in trilinear interpolation
            float3 weight_k = (crnRel_k == 1 ? pos_j : 1 - pos_j);
            val_j += (any(crnAbs_k < 0) || any(crnAbs_k >= (int3)(dim0, dim1, dim2)) ? -1 :
                      iResponse[crnAbs_k.s2 + dim2 * (crnAbs_k.s1 + dim1 * crnAbs_k.s0)]) *
                      weight_k.s0 * weight_k.s1 * weight_k.s2;
        }
        
        // Remember the maximum value for all neighboring coordinates
        maxVal = val_j > maxVal ? val_j : maxVal;
    }
    
    // Mark current position as a ridge point if its value is larger than the
    // largest of its neighbors and larger than the specified threshold
    oRidge[i] = iResponse[i] > THRESHOLD && iResponse[i] > maxVal;
}

