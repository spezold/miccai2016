#include "basedefs.h"
#include "quaternion_math.h"
#include "matrix_math.h"

/**
Several functions necessary to make the vesselness main directions point the
same way (rather than the opposite direction, possibly) to make them suitable
for diffusion with gradient vector flow.
*/

// #define SQUARED_DISTANCE
// ^ If this macro is defined, calculate the squared Euclidean distance instead
// of the Euclidean distance in the distanceTransform kernel

/**
From the given unit quaternions, extract the directions of the respective
transformation matrices' first columns.

Assumes 1D arrays of identical size, assumes to be called with the number of
given quaternions.

@param[out] oDirs the resulting directions, giving the three vector values in
            consecutive order; may be the same as <iEigs>
@param[in]  iEigs the quaternions to be processed; supposed to be "squeezed" to
            three values via the qsqueeze() function. The three resulting
            values for each point are supposed to be given in consecutive
            order; may be the same <oDirs> 
*/
__kernel void extractMainDirections(__global ftype *oDirs,
                                    __global const ftype *iEigs)
{
    const size_t i_start = 3 * get_global_id(0);

    vec vSqueezed = {iEigs[i_start], iEigs[i_start + 1], iEigs[i_start + 2]};
    qat qExpanded;
    qexpand(vSqueezed, qExpanded);
    float3 dir0;
    quat2matFloat3(qExpanded, &dir0, (float3*)0, (float3*)0); // Only need first column
    
    oDirs[i_start]     = dir0.s0;
    oDirs[i_start + 1] = dir0.s1;
    oDirs[i_start + 2] = dir0.s2;
}

/**
Recomplete vectors to coordinate systems.

Assuming that the given vectors are the first columns of transformation
matrices, complete each of them to a full transformation matrix by (1)
normalizing it, (2) finding two other arbitrary unit vectors that are
orthogonal to the given one and that together result in a matrix with a
determinant of value 1, (3) converting the matrix to an equivalent unit
quaternion and "squeezing" it to three values using the qsqueeze() function.

With a little cautiousness, this kernel can thus be seen as the reverse
operation of the extractMainDirections() kernel. Being cautious is necessary as
columns 2 and 3 of the result are more or less arbitrary.

Assumes 1D arrays of identical size, assumes to be called with the number of
given vectors.

@param[out] oQuats the resulting quaternions, giving the three "squeezed"
            values in consecutive order; may be the same as <iDirs>
@param[in]  iDirs the directions to be completed. The values for each vector
            are supposed to be given in consecutive order; may be the same as
            <oQuats>
*/
__kernel void completeTransformations(__global ftype *oQuats,
                                      __global const ftype *iDirs)
{
    // Get axis 0
    const float3 evec0 = normalize(vload3(get_global_id(0), iDirs));
    
    // Find axis 1: Determine cross product with either (1, 0, 0) or (0, 1, 0),
    // depending on which forms the larger angle with axis 0. We can calculate 
    // this cross product explicitly very easily: (1) Choose as follows: "If
    // .s0 > .s1, calculate with .s0, otherwise with .s1"; (2) set the entry of
    // the position of the 1 to 0, swap the two remaining elements of evec0,
    // negate one of them
    const float3 evec1 = normalize(fabs(evec0.s0) > fabs(evec0.s1) ?
                                   (float3)(-evec0.s2, 0.0f, evec0.s0) :
                                   (float3)(0.0f, evec0.s2, -evec0.s1));
    
    // Find axis 2
    float3 evec2 = cross(evec0, evec1);
    
    // Build the matrix (given the order of evec0 and evec1 in the last cross
    // product, we can be sure that the determinant is +1 and not -1)
    mat m = {{evec0.s0, evec1.s0, evec2.s0},
             {evec0.s1, evec1.s1, evec2.s1},
             {evec0.s2, evec1.s2, evec2.s2}};
    
    // Calculate the respective quaternion, squeeze and return it
    qat q;
    mat2quat(m, q);
    vec squeezed;
    qsqueeze(q, squeezed);
    
    vstore3((float3)(squeezed[0], squeezed[1], squeezed[2]), get_global_id(0), oQuats);
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                          CLK_ADDRESS_NONE |
                          CLK_FILTER_NEAREST;
// ^ Absolute coords, no handling of outside values, no interpolation

/**
From the given ushort4 2D image, extract the uint4(!) tuple at the given 1d
index. The approach largely follows [*].

The given image is assumed to have a fixed axis 1 dimension of 8192 elements
(which is the guaranteed minimum width according to the OpenCL specification).
It is up to the caller that the given 1D index maps to a valid position in the
2D image, otherwise the return value is undefined. For example, trying to
access the index 9000 on an image of axis 0 dimension == 1 would be invalid, as
the image is only capable of holding 8192 elements.

References
[*] http://www.cmsoft.com.br/opencl-tutorial/opencl-image2d-variables/ (20151126)

@return    The tuple at the given 1d index, as uint4(!)
@param[in] index the (1D) index to be accessed; must map to a valid (2D) image
           index
@param[in] values the image that holds the ushort4 values to be accessed; must
           have an axis 1 dimension of 8192 elements (axis 1 dimension may be
           in the interval [m, 8192] if the maximum 1D index is m < 8192)
*/
uint4 valueForIndexUs(int index, __read_only image2d_t values)
{
    // Calculate the 2D index: We exploit the fact that, if the image width is
    // 2 ^ n with n being an integer, the 2D index can be calculated via bit
    // operations rather than more expensive division and modulo. Furthermore,
    // we can hard-code n as 13, as 2 ^ 13 = 8192 is our required image width
    const int2 i2d;
    i2d.s0 = index >> 13;   // i.e. index >> n;          usually: index / width
    i2d.s1 = index & 8191;  // i.e. index & (width - 1); usually: index % width
    
    return read_imageui(values, sampler, i2d);
}

/**
Analogous to <valueForIndexUs()>, but extract float4 tuples from float4 images.
*/
float4 valueForIndexF(int index, __read_only image2d_t values)
{
    const int2 i2d;
    i2d.s0 = index >> 13;
    i2d.s1 = index & 8191;
    
    return read_imagef(values, sampler, i2d);
}


/**
For each image coordinate i_i: (1) Find the closest ridge coordinate i_r and
(2) adjust the sign of the directional vector at i_i such that the dot product
with the directional vector at i_r is maximized.

Assumes ioDirs to be a 3D array (for iRidgeCoords, see below). Assumes to be
called with the reversed image shape.

@param[in,out] ioDirs the 3D directions to be adusted. The 3 values for each
               vector are supposed to be given in consecutive order
@param[in]     iRidgeCoords the ridge voxel coordinates as ushort values (R:
               axis 0, G: axis 1, B: axis 2), stored in a 2D RGBA image with a
               fixed axis 1 dimension of 8192 elements (axis 1 dimension may be
               in the interval [m, 8192] if the number of ridge voxel
               coordinates is 8192 or smaller and thus the maximum index is m <
               8192)
@param[in]     iRidgeDirs for each ridge voxel coordinate the respective
               directional vector as float values (R: axis 0, G: axis 1, B:
               axis 2), stored in a 2D RGBA image with a fixed axis 1 dimension
               of 8192 elements (axis 1 dimension may be in the interval [m,
               8192] if the number of ridge voxel coordinates is 8192 or
               smaller and thus the maximum index is m < 8192); should match
               the corresponding values in ioDirs. In any case, ridge voxels in
               ioDirs will be treated like non-ridge voxels and the directions
               of iRidgeDirs will determine the need for swapping.
@param[in]     scaling the voxel size along axis 0, 1, and 2 [mm/voxel]
@param[in]     numRidgeCoords the number of ridge voxel coordinates
*/
__kernel void adjustNonRidgeSigns(__global float *ioDirs,
                                  __read_only image2d_t iRidgeCoords,
                                  __read_only image2d_t iRidgeDirs,
                                  const float3 scaling,
                                  const int numRidgeCoords)
{
    // We basically implement a brute-force nearest neighbor search here, as
    // suggested in [*] to be a feasible implementation on the GPU.
    //
    // References
    // [*] V. Garcia, E. Debreuve, and M. Barlaud, “Fast k nearest neighbor
    //     search using GPU,” in IEEE Computer Society Conference on Computer
    //     Vision and Pattern Recognition Workshops, 2008. CVPRW ’08, 2008, pp.
    //     1–6.
    
    const uint3 i3d = (uint3)(get_global_id(2),
                              get_global_id(1),
                              get_global_id(0));

    // Find the closest ridge point (for ridge points, this should be the point
    // itself)

    float minDistSq = INFINITY;
    int minJ;  // Min. index as an index of iRidgeCoords and iRidgeDirs
    
    for (int j = 0; j < numRidgeCoords; ++j)
    {
        const uint3 coordJ = as_uint3(valueForIndexUs(j, iRidgeCoords));
        // ^ We can make a bitwise reinterpretation here (uint4 -> uint3;
        // see Section 6.2.4.2 of the OpenCL specification)
        const float3 diffIJ = convert_float3(abs_diff(i3d, coordJ)) * scaling;
        // ^ For whatever reason, we have to explicitly convert here (uint3 ->
        // float3) in order to make the uint3-float3 multiplication work. More
        // importantly, we have to take abs_diff() here to avoid uint underflow
        // (which is ok, as we are only interested in the scalar distance
        // rather than the distance vector)
        const float distIJSq = dot(diffIJ, diffIJ);
        const bool currentSmaller = distIJSq < minDistSq;
        
        minDistSq = currentSmaller ? distIJSq : minDistSq;
        minJ      = currentSmaller ? j        : minJ;
    }
    
    
    // Adjust the sign of the current vector accordingly by maximizing the dot
    // product of the directional vector pair (note that we do not have to
    // handle the case of the current point being a ridge point explicitly, as
    // its sign already maximizes the dot product with itself, i.e. it won't
    // change)
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);
    const int i = i3d.s2 + dim2 * (i3d.s1 + dim1 * i3d.s0);
    // ^ vload3/vstore3 take care of the length 3, so do not multiply by 3 here

    float3       dirI = vload3(i, ioDirs);
    const float3 dirJ = as_float3(valueForIndexF(minJ, iRidgeDirs));
    
    if (dot(dirI, dirJ) < 0)
    {
        dirI = -dirI;
        vstore3(dirI, i, ioDirs);
    }
}


/**
For each image coordinate, store the unsigned Euclidean distance to the closest
of the given object coordinates. If the macro SQUARED_DISTANCE is defined,
calculate the squared Euclidean distance instead.

Assumes oDists to be a 3D array of image size. Assumes to be called with the
reversed image shape.

@param[out] oDists the distances for all coordinates, respecting the given
            scaling ([mm/voxel] or something the like)
@param[in]  iObjectCoords the object voxel coordinates as ushort values (R:
            axis 0, G: axis 1, B: axis 2), stored in a 2D RGBA image with a
            fixed axis 1 dimension of 8192 elements (axis 1 dimension may be in
            the interval [m, 8192] if the number of ridge voxel coordinates is
            8192 or smaller and thus the maximum index is m < 8192)
@param[in]  numObjectCoords the number of object voxel coordinates
*/
__kernel void distanceTransform(__global float *oDists,
                                __read_only image2d_t iObjectCoords,
                                const float3 scaling,
                                const int numObjectCoords)
{
    // Similar to <adjustNonRidgeSigns()>: brute-force nearest neighbor search

    const uint3 i3d = (uint3)(get_global_id(2),
                              get_global_id(1),
                              get_global_id(0));
                              
    // Find the closest object point (for object points, this should be the
    // point itself)
    
    ftype minDist = INFINITY;
    
    for (int j = 0; j < numObjectCoords; ++j)
    {
        const uint3 coordJ = as_uint3(valueForIndexUs(j, iObjectCoords));
        // ^ We can make a bitwise reinterpretation here (uint4 -> uint3;
        // see Section 6.2.4.2 of the OpenCL specification)
        const float3 diffIJ = convert_float3(abs_diff(i3d, coordJ)) * scaling;
        // ^ Take abs_diff() to avoid uint underflow (which is ok, as we are
        // only interested in the scalar distance rather than the distance
        // vector)
        
        #ifdef SQUARED_DISTANCE
        const float distIJ = dot(diffIJ, diffIJ);
        #else
        const float distIJ = length(diffIJ);
        #endif  // SQUARED_DISTANCE
        
        minDist = (distIJ < minDist) ? distIJ : minDist;
    }
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);
    const int i = i3d.s2 + dim2 * (i3d.s1 + dim1 * i3d.s0);
    
    oDists[i] = minDist;
}


__kernel void testSumImage(__global ftype *result,
                           __read_only image2d_t imagef,
                           const int numValues)
{
    if (get_global_id(0) == 0)
    {
        int3 sum = (int3)(0, 0, 0);
        for (int i = 0; i < numValues; ++i)
        {
            sum += as_int3(valueForIndexUs(i, imagef));
        }
        result[0] = sum.s0;
        result[1] = sum.s1;
        result[2] = sum.s2;
    }
}

