#include "basedefs.h"

/**
The most naive way to calculate a 3D convolution with separable kernels (always
assuming odd kernel sizes) and mirror boundary conditions; does not have any
constraints on the image size, except for having at least 2 elements in the
dimension along which the image is to be convolved. Specifically, the image
may be smaller than the kernel it is convolved with.
*/

#define kernelSizeFor(kernelRadius) (2 * kernelRadius + 1)

/**
Convolve along axis 0 of the given image data with the given kernel, using
mirror boundary conditions.

Assumes 3D arrays of identical size, assumes to be called with the reversed
image data shape.

@param[out] oData the convolution result; may *not* be the same as <iData>
@param[in]  iData the data to be convolved; may *not* be the same as <oData>
@param[in]  axis0Kernel the convolution kernel; assumed to have an odd size
@param[in]  kernelRadius size of one kernel "arm", i.e. (kernel size - 1) / 2
*/
__kernel void axis0(__global ftype *oData,
                    __global const ftype *iData,
                    __constant ftype *axis0Kernel,
                    const itype kernelRadius)
{
    const int i0 = get_global_id(2);  // Might result in negative values
    const size_t i1 = get_global_id(1);
    const size_t i2 = get_global_id(0);
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);

    // Get the number of elements in axis 0, assuming that the image would be
    // mirror completed once, then enabling tiling by simple repetition (e.g.
    // [0, 1, 2, 3] -> [0, 1, 2, 3, 2, 1])
    const int imagePeriod = 2 * (dim0 - 1);
    
    size_t imageIndex;
    ftype result = 0;
    
    // Iterate over kernel once backward, over mirror-completed image
    // periodically forward
    int tileIndex = (i0 - kernelRadius - 1) % imagePeriod;
    tileIndex += (tileIndex < 0 ? imagePeriod : 0);  // Ensure positive result
    for(int j = 0; j < kernelSizeFor(kernelRadius); ++j)
    {
        // Calculate image index: (1) account for periodicity ...
        tileIndex++;
        tileIndex = (tileIndex == imagePeriod ? 0 : tileIndex);
        // ... (2) account for mirror completion in the assumed tile ...
        imageIndex = (tileIndex < dim0) ? tileIndex : (imagePeriod - tileIndex);
        // ... (3) calculate full 3D index
        imageIndex = i2 + dim2 * (i1 + dim1 * imageIndex);
        
        result += axis0Kernel[kernelSizeFor(kernelRadius) - j - 1] * iData[imageIndex];
    }
    oData[i2 + dim2 * (i1 + dim1 * i0)] = result;
}

/**
Convolve along axis 1 of the given image data with the given kernel, using
mirror boundary conditions.

Assumes 3D arrays of identical size, assumes to be called with the reversed
image data shape.

@param[out] oData the convolution result; may *not* be the same as <iData>
@param[in]  iData the data to be convolved; may *not* be the same as <oData>
@param[in]  axis1Kernel the convolution kernel; assumed to have an odd size
@param[in]  kernelRadius size of one kernel "arm", i.e. (kernel size - 1) / 2
*/
__kernel void axis1(__global ftype *oData,
                    __global const ftype *iData,
                    __constant ftype *axis1Kernel,
                    const itype kernelRadius)
{
    const size_t i0 = get_global_id(2);
    const int i1 = get_global_id(1);  // Might result in negative values
    const size_t i2 = get_global_id(0);
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);

    // Get the number of elements in axis 1, assuming that the image would be
    // mirror completed once, then enabling tiling by simple repetition (e.g.
    // [0, 1, 2, 3] -> [0, 1, 2, 3, 2, 1])
    const int imagePeriod = 2 * (dim1 - 1);
    
    size_t imageIndex;
    ftype result = 0;
    
    // Iterate over kernel once backward, over mirror-completed image
    // periodically forward
    int tileIndex = (i1 - kernelRadius - 1) % imagePeriod;
    tileIndex += (tileIndex < 0 ? imagePeriod : 0);  // Ensure positive result
    for(int j = 0; j < kernelSizeFor(kernelRadius); ++j)
    {
        // Calculate image index: (1) account for periodicity ...
        tileIndex++;
        tileIndex = (tileIndex == imagePeriod ? 0 : tileIndex);
        // ... (2) account for mirror completion in the assumed tile ...
        imageIndex = (tileIndex < dim1) ? tileIndex : (imagePeriod - tileIndex);
        // ... (3) calculate full 3D index
        imageIndex = i2 + dim2 * (imageIndex + dim1 * i0);
        
        result += axis1Kernel[kernelSizeFor(kernelRadius) - j - 1] * iData[imageIndex];
    }
    oData[i2 + dim2 * (i1 + dim1 * i0)] = result;
}

/**
Convolve along axis 2 of the given image data with the given kernel, using
mirror boundary conditions.

Assumes 3D arrays of identical size, assumes to be called with the reversed
image data shape.

@param[out] oData the convolution result; may *not* be the same as <iData>
@param[in]  iData the data to be convolved; may *not* be the same as <oData>
@param[in]  axis2Kernel the convolution kernel; assumed to have an odd size
@param[in]  kernelRadius size of one kernel "arm", i.e. (kernel size - 1) / 2
*/
__kernel void axis2(__global ftype *oData,
                    __global const ftype *iData,
                    __constant ftype *axis2Kernel,
                    const itype kernelRadius)
{
    const size_t i0 = get_global_id(2);
    const size_t i1 = get_global_id(1);
    const int i2 = get_global_id(0);  // Might result in negative values
    
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);

    // Get the number of elements in axis 2, assuming that the image would be
    // mirror completed once, then enabling tiling by simple repetition (e.g.
    // [0, 1, 2, 3] -> [0, 1, 2, 3, 2, 1])
    const int imagePeriod = 2 * (dim2 - 1);
    
    size_t imageIndex;
    ftype result = 0;
    
    // Iterate over kernel once backward, over mirror-completed image
    // periodically forward
    int tileIndex = (i2 - kernelRadius - 1) % imagePeriod;
    tileIndex += (tileIndex < 0 ? imagePeriod : 0);  // Ensure positive result
    for(int j = 0; j < kernelSizeFor(kernelRadius); ++j)
    {
        // Calculate image index: (1) account for periodicity ...
        tileIndex++;
        tileIndex = (tileIndex == imagePeriod ? 0 : tileIndex);
        // ... (2) account for mirror completion in the assumed tile ...
        imageIndex = (tileIndex < dim2) ? tileIndex : (imagePeriod - tileIndex);
        // ... (3) calculate full 3D index
        imageIndex = imageIndex + dim2 * (i1 + dim1 * i0);
        
        result += axis2Kernel[kernelSizeFor(kernelRadius) - j - 1] * iData[imageIndex];
    }
    oData[i2 + dim2 * (i1 + dim1 * i0)] = result;
}
