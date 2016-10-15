#include "basedefs.h"

#define TINY (FLT_MIN)

/**
Calculate the current scale's Frangi vesselness response [1] from the given
eigenvalues, update the current maximum response if necessary.

Assumes 1D arrays, assumes to be called with the number of voxels in the image
that is analyzed. Assumed array sizes: image size for ioValue, 3 * image size
for ioQuat, 6 * image size for inEigs. Assumes that ioValue has been
initialized with -INFINITY before the kernel is called for the first scale.

References
[1] A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever,
    “Multiscale vessel enhancement filtering,” in Medical Image Computing and
    Computer-Assisted Interventation — MICCAI’98, W. M. Wells, A. Colchester,
    and S. Delp, Eds. Springer Berlin Heidelberg, 1998, pp. 130–137.

@param[in,out] ioValue on output, for each voxel, the value of max{v, v*} where
               <v> is the vesselness response for the current scale and <v*> is
               the currently stored value, which should represent the current
               maximum vesselness response
@param[in,out] ioQuat on output, for each voxel: if v = max{v, v*}, the 3
               values of the current eigendecomposition's "squeezed"
               quaternion; else, the 3 currently stored values, which should
               represent the current maximum vesselness response's quaternion
               (cf. <ioValue> and <inEigs>)
@param[in]     inEigs for each voxel, the 6 values of its Hessian matrix's
               eigendecomposition, given in consecutive order: The first 3
               elements hold the sorted eigenvalues in ascending order of their
               absolute values. The last 3 elements hold a "squeezed" version
               of the unit quaternion that represents the matrix of sorted
               eigenvectors, as produced by the qsqueeze() function.
@param[in]     alpha weighting for the \lambda_2 / \lambda_3 ratio; see eqs.
               (11) and (13) in [1]
@param[in]     beta weighting for the \lambda_1 / sqrt(\lambda_2 * \lambda_3)
               ratio; see eqs. (10) and (13) in [1]
@param[in]     c weighting for the "second order structureness" term; see eqs.
               (12) and (13) in [1]
@param[in]     brightObjects if true, filter for bright structures on dark
               background; if false, filter for dark structures on bright
               background (note that we use an integer type here, as the
               OpenCL API does not specify a bool data type)
*/               
__kernel void updateVesselness(__global ftype *ioValue,
                               __global ftype *ioQuat,
                               __global const ftype *inEigs,
                               const ftype alpha,
                               const ftype beta,
                               const ftype c,
                               const char brightObjects)
{
    const size_t i = get_global_id(0);
    
    // Get the eigenvalues
    inEigs += 6 * i;
    const ftype lambda1 = fabs(*inEigs++);
    ftype lambda2 = *inEigs++;
    ftype lambda3 = *inEigs++;
    
    // Check if we are in the foreground    
    lambda2 = brightObjects ? lambda2 : -lambda2;
    lambda3 = brightObjects ? lambda3 : -lambda3;
    const bool validPos = lambda2 < 0 && lambda3 < 0;
    
    // For the vesselness measures, we need absolute values of the eigenvalues
    lambda2 = fabs(lambda2);
    lambda3 = fabs(lambda3);
    
    // Calculate "blobness", i.e. eq. (10) in [1]
    const ftype rB = lambda1 / (sqrt(lambda2 * lambda3) + TINY);
    // Calculate "plateness vs. tubeness", i.e. eq. (11) in [1]
    const ftype rA = lambda2 / (lambda3 + TINY);
    // Calculate "second-order structureness", i.e. eq. (12) in [1]
    const ftype sSquare = square(lambda1) + square(lambda2) + square(lambda3);
    
    // Calculate the combined vesselness measure: equation (13) in [1]
    const ftype v = validPos * ((1.0f - exp(-0.5f * square(rA / alpha))) *
                                        exp(-0.5f * square(rB / beta)) *
                                (1.0f - exp(-0.5f * sSquare / square(c))));
    ioValue += i;
    // Update the i/o buffers if necessary
    if (v > *ioValue)
    {
        ioQuat += 3 * i;
        *ioValue = v;
        *ioQuat++ = *inEigs++;
        *ioQuat++ = *inEigs++;
        *ioQuat   = *inEigs;
    }
}

/**
For each given eigendecomposition, calculate the norm as the square root of the
sum of squares of the eigenvalues.

Assumes 1D arrays, assumes to be called with the number of voxels in the image
that is analyzed. Assumed array sizes: image size for oNorm, 6 * image size for
inEigs.

@param[out] oNorm the norm for the current element
@param[in]  inEigs for each voxel, the 6 values of its Hessian matrix's
            eigendecomposition, given in consecutive order: The first 3
            elements hold the eigenvalues, the last 3 elements are ignored
*/
__kernel void calculateNorm(__global ftype *oNorm,
                            __global const ftype *inEigs)
{
    const size_t i = get_global_id(0);
    const size_t six_i = 6 * i;
    oNorm[i] = sqrt(square(inEigs[six_i]) +
                    square(inEigs[six_i + 1]) +
                    square(inEigs[six_i + 2]));
}
