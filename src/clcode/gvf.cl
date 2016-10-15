#include "basedefs.h"

/**
Diffuse a vector field using Xu and Prince's classical gradient vector flow 
(GVF) approach [1,2,3].

References
[1] C. Xu and J. L. Prince, “Gradient Vector Flow: A New External Force for
    Snakes,” in CVPR, 1997, pp. 66–71.
[2] C. Xu and J. L. Prince, “Snakes, Shapes, and Gradient Vector Flow,” IEEE
    Transactions on image processing, vol. 7, no. 3, pp. 359–369, 1998.
[3] C. Xu and J. L. Prince, “Gradient Vector Flow,” in Computer Vision, K.
    Ikeuchi, Ed. Springer US, 2014, pp. 349–354.
*/

// #define MU    // Regularization parameter (the higher, the smoother)
// #define DT    // Time step
// #define RSS0  // Squared reciprocal of the voxel scaling along dimension 0 [voxel^2/mm^2]
// #define RSS1  // Squared reciprocal of the voxel scaling along dimension 1 [voxel^2/mm^2]
// #define RSS2  // Squared reciprocal of the voxel scaling along dimension 2 [voxel^2/mm^2]

#define sqroot(x) native_sqrt(x)

/**
Calculate the value for the GVF field update for one interation. The result
must be multiplied with the time step and then subtracted from the current GVF
field value (see the <gvfUpdate()> kernel).

Assumes 3D arrays of identical size. Assumes to be called with the reversed
image shape.

@param[out] oDelta0 value for the GVF field update, vector component of axis 0
@param[out] oDelta1 value for the GVF field update, vector component of axis 1
@param[out] oDelta2 value for the GVF field update, vector component of axis 2
@param[in]  iGvf0 GVF field's current state, vector component of axis 0
@param[in]  iGvf1 GVF field's current state, vector component of axis 1
@param[in]  iGvf2 GVF field's current state, vector component of axis 2
@param[in]  iVec0 original vector field, vector component of axis 0
@param[in]  iVec1 original vector field, vector component of axis 1
@param[in]  iVec2 original vector field, vector component of axis 2
@param[in]  iResponse weight of the GVF fidelity term (squared vector field
            magnitude in [1])
*/
__kernel void gvfDelta(__global ftype *oDelta0,
                       __global ftype *oDelta1,
                       __global ftype *oDelta2,
                       __global const ftype *iGvf0,
                       __global const ftype *iGvf1,
                       __global const ftype *iGvf2,
                       __global const ftype *iVec0,
                       __global const ftype *iVec1,
                       __global const ftype *iVec2,
                       __global const ftype *iResponse)
{
    const size_t i0 = get_global_id(2);
    const size_t i1 = get_global_id(1);
    const size_t i2 = get_global_id(0);
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);
    
    const itype i = i2 + dim2 * (i1 + dim1 * i0);  // current index
    
    const ftype iGvf0_i = iGvf0[i];
    const ftype iGvf1_i = iGvf1[i];
    const ftype iGvf2_i = iGvf2[i];

    // Calculate Laplacian of the current GVF field state, separately for all
    // GVF vector components, using mirror boundaries
    ftype lapGvf0;
    ftype lapGvf1;
    ftype lapGvf2;
    itype i_bef;
    itype i_aft;
    
    i_bef = i + (i0 > 0        ? -1 :  1) * dim2 * dim1;
    i_aft = i + (i0 < dim0 - 1 ?  1 : -1) * dim2 * dim1;
    lapGvf0 = (iGvf0[i_bef] + iGvf0[i_aft] - 2 * iGvf0_i) * RSS0;
    lapGvf1 = (iGvf1[i_bef] + iGvf1[i_aft] - 2 * iGvf1_i) * RSS0;
    lapGvf2 = (iGvf2[i_bef] + iGvf2[i_aft] - 2 * iGvf2_i) * RSS0;
    
    i_bef = i + (i1 > 0        ? -1 :  1) * dim2;
    i_aft = i + (i1 < dim1 - 1 ?  1 : -1) * dim2;
    lapGvf0 += (iGvf0[i_bef] + iGvf0[i_aft] - 2 * iGvf0_i) * RSS1;
    lapGvf1 += (iGvf1[i_bef] + iGvf1[i_aft] - 2 * iGvf1_i) * RSS1;
    lapGvf2 += (iGvf2[i_bef] + iGvf2[i_aft] - 2 * iGvf2_i) * RSS1;
    
    i_bef = i + (i2 > 0        ? -1 :  1);
    i_aft = i + (i2 < dim2 - 1 ?  1 : -1);
    lapGvf0 += (iGvf0[i_bef] + iGvf0[i_aft] - 2 * iGvf0_i) * RSS2;
    lapGvf1 += (iGvf1[i_bef] + iGvf1[i_aft] - 2 * iGvf1_i) * RSS2;
    lapGvf2 += (iGvf2[i_bef] + iGvf2[i_aft] - 2 * iGvf2_i) * RSS2;
    
    // Calculate GVF update
    oDelta0[i] = iResponse[i] * (iGvf0_i - iVec0[i]) - MU * lapGvf0;
    oDelta1[i] = iResponse[i] * (iGvf1_i - iVec1[i]) - MU * lapGvf1;
    oDelta2[i] = iResponse[i] * (iGvf2_i - iVec2[i]) - MU * lapGvf2;    
}


/**
Same as <gvfDelta()>, but only for one vector component at a time. Seems to
make no difference, speed-wise -- but may make a tiny difference calculation-
wise if compiler optimizations are enabled.
*/
__kernel void gvfDeltaI(__global ftype *oDeltaI,
                        __global const ftype *iGvfI,
                        __global const ftype *iVecI,
                        __global const ftype *iResponse)
{
    const size_t i0 = get_global_id(2);
    const size_t i1 = get_global_id(1);
    const size_t i2 = get_global_id(0);
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);
    
    const itype i = i2 + dim2 * (i1 + dim1 * i0);  // current index
    
    const ftype iGvfI_i = iGvfI[i];

    // Calculate Laplacian of the current GVF field state, separately for all
    // GVF vector components, using mirror boundaries
    ftype lapGvfI;
    itype i_bef;
    itype i_aft;
    
    i_bef = i + (i0 > 0        ? -1 :  1) * dim2 * dim1;
    i_aft = i + (i0 < dim0 - 1 ?  1 : -1) * dim2 * dim1;
    lapGvfI = (iGvfI[i_bef] + iGvfI[i_aft] - 2 * iGvfI_i) * RSS0;
    
    i_bef = i + (i1 > 0        ? -1 :  1) * dim2;
    i_aft = i + (i1 < dim1 - 1 ?  1 : -1) * dim2;
    lapGvfI += (iGvfI[i_bef] + iGvfI[i_aft] - 2 * iGvfI_i) * RSS1;
    
    i_bef = i + (i2 > 0        ? -1 :  1);
    i_aft = i + (i2 < dim2 - 1 ?  1 : -1);
    lapGvfI += (iGvfI[i_bef] + iGvfI[i_aft] - 2 * iGvfI_i) * RSS2;
    
    // Calculate GVF update
    oDeltaI[i] = iResponse[i] * (iGvfI_i - iVecI[i]) - MU * lapGvfI;
}


/**
Update the GVF field value with the appropriately scaled delta.

Assumes 1D arrays of identical size.

@param[in,out] ioGvf0 GVF field's current state, vector component of axis 0
@param[in,out] ioGvf1 GVF field's current state, vector component of axis 1
@param[in,out] ioGvf2 GVF field's current state, vector component of axis 2
@param[in]     iDelta0 value for the GVF field update as produced by the
               <gvfDelta()> kernel, vector component of axis 0
@param[in]     iDelta1 value for the GVF field update as produced by the
               <gvfDelta()> kernel, vector component of axis 1
@param[in]     iDelta2 value for the GVF field update as produced by the
               <gvfDelta()> kernel, vector component of axis 2
*/
__kernel void gvfUpdate(__global ftype *ioGvf0,
                        __global ftype *ioGvf1,
                        __global ftype *ioGvf2,
                        __global const ftype *iDelta0,
                        __global const ftype *iDelta1,
                        __global const ftype *iDelta2)
{
    const size_t i = get_global_id(0);
    ioGvf0[i] = ioGvf0[i] - DT * iDelta0[i];
    ioGvf1[i] = ioGvf1[i] - DT * iDelta1[i];
    ioGvf2[i] = ioGvf2[i] - DT * iDelta2[i];
}


/**
Calculate the magnitude of the GVF field update as convergence criterion.

Assumes 1D arrays of identical size.

@param[out] oMagDelta the magnitude of the GVF field update; may be the same as
            either of the <iDelta*> buffers
@param[in]  iDelta0 value for the GVF field update as produced by the
            <gvfDelta()> kernel, vector component of axis 0; may be the same as
            <oMagDelta>
@param[in]  iDelta1 value for the GVF field update as produced by the
            <gvfDelta()> kernel, vector component of axis 1; may be the same as
            <oMagDelta>
@param[in]  iDelta2 value for the GVF field update as produced by the
            <gvfDelta()> kernel, vector component of axis 2; may be the same as
            <oMagDelta>
*/
__kernel void magDelta(__global ftype *oMagDelta,
                       __global const ftype *iDelta0,
                       __global const ftype *iDelta1,
                       __global const ftype *iDelta2)
{
    const size_t i = get_global_id(0);
    oMagDelta[i] = sqroot(square(iDelta0[i]) +
                          square(iDelta1[i]) +
                          square(iDelta2[i]));
}


/**
Normalize the current GVF field, so that each vector has unit length (zero-
length vectors will remain zero-length).

Assumes 1D arrays of identical size.

@param[in,out] ioGvf0 GVF field's current state, vector component of axis 0
@param[in,out] ioGvf1 GVF field's current state, vector component of axis 1
@param[in,out] ioGvf2 GVF field's current state, vector component of axis 2
*/
__kernel void gvfNormalize(__global ftype *ioGvf0,
                           __global ftype *ioGvf1,
                           __global ftype *ioGvf2)
{
    const size_t i = get_global_id(0);
    const ftype gvf0 = ioGvf0[i];
    const ftype gvf1 = ioGvf1[i];
    const ftype gvf2 = ioGvf2[i];
    ftype nrm = sqroot(square(gvf0) + square(gvf1) + square(gvf2));
    if (nrm != 0)
    {
        // Normalize by taking the reciprocal and multiplying with it
        nrm = 1 / nrm;
        ioGvf0[i] = nrm * gvf0;
        ioGvf1[i] = nrm * gvf1;
        ioGvf2[i] = nrm * gvf2;
    }
}


/**
Calculate Xu and Prince's original weight for the GVF fidelity term, namely the
squared magnitude of the given vector field's vectors.

Assumes 1D arrays of identical size.

@param[out] oResponse calculated weight
@param[in]  iVec0 vector field, vector component of axis 0
@param[in]  iVec1 vector field, vector component of axis 1
@param[in]  iVec2 vector field, vector component of axis 2
*/
__kernel void response(__global ftype *oResponse,
                       __global const ftype *iVec0,
                       __global const ftype *iVec1,
                       __global const ftype *iVec2)
{
    const size_t i = get_global_id(0);
    oResponse[i] = square(iVec0[i]) + square(iVec1[i]) + square(iVec2[i]);
}

