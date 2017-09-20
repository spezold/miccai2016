#!/usr/bin/env python
# coding: utf-8

"""
Implementation of the "vesselness" feature filter as described by Frangi et al.
in [1], porting all essential number crunching to the GPU, using PyOpenCL.

The quaternions that are created represent the matrix of eigenvectors where
each column holds an eigenvector.

References
[1] A. F. Frangi, W. J. Niessen, K. L. Vincken, and M. A. Viergever,
    “Multiscale vessel enhancement filtering,” in Medical Image Computing and
    Computer-Assisted Interventation — MICCAI’98, W. M. Wells, A. Colchester,
    and S. Delp, Eds. Springer Berlin Heidelberg, 1998, pp. 130–137.
"""

from __future__ import division

import numpy as np
import os
import pyopencl as cl

from helpers.cl_helpers import cl_workgroup_size_3d_for, ClProgramLoader, Reducer
import helpers.parameters
from processing import slicing
from collections import namedtuple


class GaussianFilter3D(object):
    """
    3D Gaussian filter -- reimplementation of
    <scipy.ndimage.filters.gaussian_filter()>.
    """

    def __init__(self, loader, n_sigmas=4, round_up=True):
        """
        <loader>
            A <ClProgramLoader> instance.
        <n_sigmas>
            Number of standard deviations covered by each 'arm' of each
            Gaussian filter used for filtering.
        <round_up>
            Rounding mode for Gaussian kernel 'arm' size s = n_sigmas * sigma,
            in case <s> is not an integer. If <round_up> is True, round as
            ceil(s), which means that the minimum number of covered standard
            deviations is guaranteed. If <round_up> is False, round as int(s +
            0.5), which mimics the behavior of Scipy's <gaussian_filter()>.
        """
        self.__n_sigmas = n_sigmas
        self.__round_up = round_up
        
        self._shape = None
        self._loader = loader
        
        self._dtype = np.float32
        self._lsize_3d = None
        self._conv_clkernels = self._init_cl_programs()
    
    def _init_cl_programs(self):
        
        SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        PROGRAM_PATH = os.path.join(SRC_DIR, "clcode", "convolve_naive.cl")
        INCLUDE_PATH = os.path.join(SRC_DIR, "clcode")
        
        prg = self._loader.program_from_file(PROGRAM_PATH, includes=[INCLUDE_PATH])
        clkernels = [prg.axis0, prg.axis1, prg.axis2]
        for clkernel in clkernels:
            clkernel.set_scalar_arg_dtypes([None, None, None, np.int32])
        return clkernels

    @property    
    def n_sigmas(self):
        """
        Number of standard deviations covered by each 'arm' of each Gaussian
        filter used for filtering.
        """
        return self.__n_sigmas
    
    @property
    def round_up(self):
        """
        Rounding mode for Gaussian kernel 'arm' size s = n_sigmas * sigma, in
        case <s> is not an integer. If <round_up> is True, round as ceil(s),
        which means that the minimum number of covered standard deviations is
        guaranteed. If <round_up> is False, round as int(s + 0.5), which mimics
        the behavior of Scipy's <gaussian_filter()>.
        """
        return self.__round_up
        
    def execute(self, in_buffer, swap_buffer, shape, sigma, order, scale_space_scaling=True, scale_space_gamma=1):
        """
        <in_buffer>
            Buffer containing the image data to be filtered.
        <swap_buffer>
            Buffer for intermediate results.
        <shape>
            The 3D shape of the image data to be processed; given as a three-
            element sequence.
        <sigma>
            Standard deviation(s) [voxel] for Gaussian kernel; given as a
            sequence in the order of the image axes or as a single number, in
            which case it is equal for all axes.
        <order>
            Order(s) for the filter along each axis (in {0, 1, 2, 3}),
            corresponding to convolution with the derivative of respective
            order of the Gaussian kernel; given as a sequence in the order of
            the image axes or as a single number, in which case it is equal for
            all axes.
        <scale_space_scaling>
            If True, scale filtering results with <sigma> **
            <scale_space_gamma>, as required by equation (2) in [1].
        <scale_space_gamma>
            Relative weighting for the scales (cf. [1,p.2]), defaults to 1,
            meaning no preference for a certain scale. Will only be evaluated
            if <scale_space_scaling> is True.
            
        Return
            Two-tuple (result_buffer, other_buffer) for convenience, where
            <result_buffer> is the buffer that contains the result (which is
            the same as <swap_buffer>) and <other_buffer> is the remaining
            buffer (which is the same as <in_buffer>, but with altered
            content).
        """
        assert len(shape) == 3
        data_bytes = np.prod(shape) * self._dtype().nbytes
        assert in_buffer.size >= data_bytes
        assert swap_buffer.size >= data_bytes
        
        shape = np.array(shape)
        shape_rev = shape[::-1]
        if np.any(shape != self._shape):
            self._shape = shape
            self._lsize_3d = cl_workgroup_size_3d_for(shape, reverse=True)
        
        sigma = np.multiply(sigma, np.ones(3))
        order = np.multiply(order, np.ones(3, dtype=np.int), dtype=np.int)
        scaling_factor = (lambda s, o :  s ** (o * scale_space_gamma) if
                          scale_space_scaling else 1)
        gauss_kernels = [self._gauss_kernel_with(s, o, scaling_factor(s, o))
                         for s, o in zip(sigma, order)]
        kernel_radii = [int(len(gk) // 2) for gk in gauss_kernels]
        
        gauss_kernel_buffers = [self._buffer_for(gk) for gk in gauss_kernels]
        i_buffer = in_buffer
        o_buffer = swap_buffer
        
        for i in range(3):
            clkernel = self._conv_clkernels[i]
            gauss_kernel_buffer = gauss_kernel_buffers[i]
            clkernel(self._loader.queue, shape_rev, self._lsize_3d,
                     o_buffer, i_buffer, gauss_kernel_buffer, kernel_radii[i])
            i_buffer, o_buffer = o_buffer, i_buffer  # Swap the buffers
            
        return i_buffer, o_buffer  # Return i_buffer first, as we already swapped
        
        
    def _gauss_kernel_with(self, sigma, order, scaling_factor=1):
        """
        Create an odd-sized gauss kernel of the given derivative <order> (in
        {0, 1, 2, 3}) with the given standard deviation <sigma> [voxel],
        capturing at least <self.n_sigmas> standard deviations in each 'kernel
        arm'. If a <scaling_factor> is given, scale the kernel values with it.
        """
        sigma_sq = sigma ** 2
        
        def gaussian(x):
            """Standard Gaussian, normalized for truncated x range."""
            result = np.exp(-.5 * x ** 2 / sigma_sq)
            return result / np.sum(result)
            
        order0 = lambda x : gaussian(x)
        order1 = lambda x : gaussian(x) * (-x)                        / (sigma_sq)
        order2 = lambda x : gaussian(x) * (x ** 2 - sigma_sq)         / (sigma_sq ** 2)
        order3 = lambda x : gaussian(x) * x * (3 * sigma_sq - x ** 2) / (sigma_sq ** 3)
        
        orders = [order0, order1, order2, order3]
        x_max = (int(np.ceil(self.n_sigmas * sigma)) if self.round_up else
                 int(self.n_sigmas * sigma + 0.5))
        x_s = np.arange(-x_max, x_max + 1)
        
        kernel = np.require(scaling_factor * orders[order](x_s), self._dtype, ["C", "A"])
        # ^ <np.require()> for being digestible with OpenCL
        
        return kernel
    
    def _buffer_for(self, gauss_kernel):
        
        buff = cl.Buffer(self._loader.ctx,
                         cl.mem_flags.USE_HOST_PTR | cl.mem_flags.READ_ONLY,
                         hostbuf=gauss_kernel)
        return buff
    
    
class Hessian(object):
    """
    Calculate Hessian matrices via Gaussian derivatives.
    """
    
    def __init__(self, gaussian_filter):
        """
        <gaussian_filter>
            A <GaussianFilter3D> instance.
        """
        self._gaussian_filter = gaussian_filter
        self._dtype = np.float32
        
    def execute(self, in_buffer, swap_buffer1, swap_buffer2, shape, result_buffer, scales):
        """
        Calculate Hessian matrix (i.e. second-order partial derivatives) for
        the given 3D image data.
        
        <in_buffer>
            Buffer containing the image data to be filtered. Content guaranteed
            to remain unchanged.
        <swap_buffer*>
            Buffer for intermediate results.
        <shape>
            The 3D shape of the image data to be processed; given as a three-
            element sequence.
        <result_buffer>
            Buffer for the final result, i.e. the Hessian matrices, holding the
            6 unique values of the i-th matrix in consecutive order according
            to the following scheme:
                [i + 0] -> [0][0], [i + 1] -> [1][1], [i + 2] -> [2][2],
                [i + 3] -> [0][1], [i + 4] -> [0][2], [i + 5] -> [1][2].
        <scales>
            Derivative scale(s), i.e. standard deviation(s) [voxel] for
            Gaussian derivatives; given as a sequence in the order of the image
            axes or as a single number, in which case it is equal for all axes.
            
        Return
            <result_buffer> for convenience.
        """
        ndim = len(shape)
        data_num = np.prod(shape)
        data_bytes = data_num * self._dtype().nbytes
        
        assert ndim == 3
        assert in_buffer.size >= data_bytes
        assert swap_buffer1.size >= data_bytes
        assert swap_buffer2.size >= data_bytes
        assert result_buffer.size >= 6 * data_bytes

        loader = self._gaussian_filter._loader
        
        consecutive_index_for = lambda i, j: i if (i == j) else (2 + i + j)
        result_array = np.empty((6, data_num), dtype=self._dtype)
        # ^ 6xN, i.e. equivalent Hessian entries in consecutive order
        
        for i in range(ndim):
            for j in range(i + 1):
                # ^ We only need to calculate one half of the Hessian matrices
                # (including the diagonal), as they are symmetric
                
                # Create a sequence determining the derivative order for
                # positions [i, j] and [j, i] in the Hessian matrices
                order_ij = np.zeros(ndim, dtype=np.int)
                order_ij[i] = 1
                order_ij[j] += 1
                
                # Get original image data
                cl.enqueue_copy(loader.queue, swap_buffer1, in_buffer, byte_count=data_bytes)
                result_buffer_ij, unused = self._gaussian_filter.execute(swap_buffer1, swap_buffer2, shape, scales, order_ij)
                cl.enqueue_copy(loader.queue, result_array[consecutive_index_for(i, j)], result_buffer_ij)
                
        # Copy complete result to <result_buffer>
        result_array = np.require(result_array.T, self._dtype, ["C", "A"])
        # ^ Nx6, i.e. complete Hessian matrices in consecutive order
        cl.enqueue_copy(loader.queue, result_buffer, result_array)
        return result_buffer
    

class Eigendecomposition(object):
    """
    Decompose a Hessian matrix into its eigenvalues and a unit quaternion that
    represents the corresponding eigenvectors.
    """
    def __init__(self, loader, maxiter=50):
        """
        <loader>
            A <ClProgramLoader> instance.
        <maxiter>
            Maximum number of Jacobi sweeps (i.e. iterations) for the
            eigendecomposition.
        """
        self._dtype = np.float32
        self._maxiter = maxiter
        
        self._loader = loader
        self._qeigs_kernel = self._create_qeigs_kernel()
        
    def _create_qeigs_kernel(self):
        
        SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        PROGRAM_PATH = os.path.join(SRC_DIR, "clcode", "qeigs.cl")
        INCLUDE_PATH = os.path.join(SRC_DIR, "clcode")

        
        defs = {"eigcmp(current, extreme)" : "(fabs(current) < fabs(extreme))",
        # ^ Redefine eigenvalue order for [1]: ascending absolute value
                "MAXITER" : self._maxiter}
        
        k = self._loader.program_from_file(PROGRAM_PATH, defs=defs,
                                           includes=[INCLUDE_PATH]).eigs
        return k
    
    def execute(self, num_voxels, in_buffer, result_buffer=None):
        """
        Calculate eigendecompositions for the given Hessian matrices.
        
        <in_buffer>
            Buffer containing the Hessian matrices to be decomposed. Expected
            to contain the 6 unique values of the i-th matrix in consecutive
            order according to the following scheme:
                [i + 0] -> [0][0], [i + 1] -> [1][1], [i + 2] -> [2][2],
                [i + 3] -> [0][1], [i + 4] -> [0][2], [i + 5] -> [1][2].
        <result_buffer>
            On output: The eigendecompositions of the Hessian matrices; in
            consecutive order: the first 3 elements hold the sorted eigenvalues
            in ascending order of their absolute values; the last 3 elements
            hold a "squeezed" version of the unit quaternion that represents
            the matrix of sorted eigenvectors, as produced by the qsqueeze()
            function. If None, results will be written to <in_buffer> instead.
            
        Return
            The buffer that contains the result, for convenience.
        """
        result_buffer = result_buffer if result_buffer is not None else in_buffer
        self._qeigs_kernel(self._loader.queue, (num_voxels, ), None,
                           in_buffer, result_buffer)
        return result_buffer


class VesselnessUpdate(object):
    """
    Update the accumulated vesselness response with the vesselness response
    for the current scale.
    """
    def __init__(self, loader):
        """
        <loader>
            A <ClProgramLoader> instance.
        """
        self._dtype = np.float32
        
        self._loader = loader
        self._update_vesselness_kernel = None
        self._calculate_norm_kernel = None
        self._max_reducer = self._create_reducer()
        
        self._init_cl_programs()
        
    def _create_reducer(self):
        
        SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        PROGRAM_PATH = os.path.join(SRC_DIR, "clcode", "max.cl")
        INCLUDE_PATH = os.path.join(SRC_DIR, "clcode")
        
        wgsize = 1024
        defs = {"WGSIZE" : wgsize}  # Somewhat arbitrary (must be 2 ** N)
        k = self._loader.program_from_file(PROGRAM_PATH, defs=defs,
                                           includes=[INCLUDE_PATH]).redMax
        k.set_scalar_arg_dtypes([None, None, np.int32])
        prg_cpu = lambda a : np.max(a)  # For the final reduction
        
        reducer = Reducer(k, prg_cpu, wgsize, self._loader.queue)
        return reducer
        
    def _init_cl_programs(self):
        
        SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        PROGRAM_PATH = os.path.join(SRC_DIR, "clcode", "frangi.cl")
        INCLUDE_PATH = os.path.join(SRC_DIR, "clcode")
        
        prg = self._loader.program_from_file(PROGRAM_PATH, includes=[INCLUDE_PATH])
        k = prg.updateVesselness
        k.set_scalar_arg_dtypes([None, None, None,
                                 np.float32, np.float32, np.float32, np.bool8])
        self._update_vesselness_kernel = k
        self._calculate_norm_kernel = prg.calculateNorm
    
    def execute(self, num_voxels, val_buffer, quat_buffer, eigs_buffer,
                bright_objects, alpha, beta, c=None, c_buffer1=None,
                c_buffer2=None):
        """
        Update the accumulated vesselness response, as stored in <val_buffer>
        and <quat_buffer>, with the vesselness response for the current scale,
        determined from the content of <eigs_buffer>.
        
        <num_voxels>
            The number of voxels in the image data to be processed.
        <val_buffer>
            The current maximum vesselness responses, which will be updated
            (size should be >= <num_voxels>)
        <quat_buffer>
            The "squeezed" unit quaternions corresponding to the eigenvector
            matrices of the current maximum vesselness responses; will be
            updated (size should be >= 3 * <num_voxels>)
        <eigs_buffer>
            The eigendecompositions of the current scale's Hessian matrices; in
            consecutive order: the first 3 elements hold the sorted eigenvalues
            in ascending order of their absolute values; the last 3 elements
            hold a "squeezed" version of the unit quaternion that represents
            the matrix of sorted eigenvectors, as produced by the qsqueeze()
            function; will remain unaltered
        <bright_objects>
            if True, filter for bright structures on dark background; if False,
            filter for dark structures on bright background
        <alpha>
            weighting for the \lambda_2 / \lambda_3 ratio; see eqs. (11) and
            (13) in [1]
        <beta>
            weighting for the \lambda_1 / sqrt(\lambda_2 * \lambda_3) ratio;
            see eqs. (10) and (13) in [1]
        <c>
            weighting for the "second order structureness" term; see eqs. (12)
            and (13) in [1]; if None, determine c as half the value of the
            maximum "Hessian norm" (i.e. the square root of the sum of squares
            of the Hessian matrix's eigenvalues for each Hessian matrix), as
            suggested in [1]
        <c_buffer*>
            buffer capable of holding the image; used for intermediate results
            when determining <c>; must only be given if <c> is None
        """
        global_size = (num_voxels, )
        queue = self._loader.queue
        
        if c is None:
            assert c_buffer1 is not None and c_buffer2 is not None
            # Calculate the matrix norms, find their maximum to determine c
            self._calculate_norm_kernel(queue, global_size, None, c_buffer1, eigs_buffer)
            c = 0.5 * self._max_reducer.reduce(c_buffer1, num_voxels, c_buffer2, dtype=self._dtype)
            
        # Calculate actual vesselness response for current scale, update
        # accumulated results (i.e. vesselness value and corresponding eigenvectors)
        self._update_vesselness_kernel(queue, global_size, None,
                                       val_buffer, quat_buffer, eigs_buffer,
                                       alpha, beta, c, bright_objects)
        

class VesselnessCalculation(object):
    """
    Calculate and return the Frangi filter [1] response for the given image.
    """
    class Parameters(helpers.parameters.Parameters):
        
        def __init__(self):
            
            self.alpha = 0.5
            # ^ Weighting for the \lambda_2 / \lambda_3 ratio (see equations
            # (11) and (13) in [1])
            self.beta = 0.5
            # ^ Weighting for the \lambda_1 / sqrt(\lambda_2 * \lambda_3) ratio
            # (see equations (10) and (13) in [1])
            self.c = None
            # ^ Weighting for the "second order structureness" term (see
            # equations (12) and (13) in [1]). If None, use half the value of
            # the maximum "Hessian norm" (i.e. the square root of the sum of
            # squares of the Hessian matrix's eigenvalues for each Hessian
            # matrix), as suggested in [1]
            self.bright_objects = None
            # ^ If True, filter for bright structures on dark background; if
            # False, filter for dark structures on bright background
            self.scales = None
            # ^ Three-tuple (start [mm], stop [mm], number) defining the scales
            # for the Frangi filter. Should be about half of the diameter of
            # the vessel structures to be detected
    
    def __init__(self, loader):
        """
        <loader>
            A <ClProgramLoader> instance.
        """
        self._dtype = np.float32
        self._dbytes = self._dtype().nbytes
        
        self._loader = loader
        self._to_value_kernel = self._create_to_value_kernel()
        
        self._params = None
        self._shape = None
        self._nvoxels = None
        self._nbytes = None
        self._voxel_size = None
        self._scales_vx = None  # Mx3 for M scales, [voxel]
        
        # OpenCL buffers
        self._img_buffer = None
        self._swap_buffer1 = None
        self._swap_buffer2 = None
        self._hessian_buffer = None
        self._val_buffer = None
        self._quat_buffer = None
        
        # Workers
        self._hessian = None
        self._eigendecomposition = None
        self._vesselness_update = None
        
    def _create_to_value_kernel(self):
        
        SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        PROGRAM_PATH = os.path.join(SRC_DIR, "clcode", "helpers.cl")
        
        k = self._loader.program_from_file(PROGRAM_PATH).to_value
        k.set_scalar_arg_dtypes([None, self._dtype])
        return k
        
        
    def execute(self, params, shape, voxel_size, img_buffer, swap_buffer1=None,
                swap_buffer2=None, hessian_buffer=None, val_buffer=None,
                quat_buffer=None, gaussian_filter_3d=None,
                eigendecomposition=None, vesselness_update=None):
        """
        Calculate the Frangi filter [1] response for the given image data,
        using the given parameterization. Apart from <img_buffer>, the
        <*_buffer*> instances will be created if None (or used if given). The
        same applies to <gaussian_filter_3d>, <eigendecomposition>, and
        <vesselness_update>.
        
        <params>
            A <VesselnessCalculation.Parameters> instance.
        <shape>
            A three-tuple or similar, giving the 3D shape of the given image
            data.
        <voxel_size>
            A scalar or three-tuple (or similar), giving the voxel size of the
            given image data as [mm / voxel], for correct interpretation of the
            feature scales.
        <img_buffer>
            The buffer that contains the image data to be analyzed. Guaranteed
            to remain unchanged.
        <swap_buffer*>
            Buffers for intermediate results; size >= image size.
        <hessian_buffer>
            Buffer for voxel-wise hessian matrices and their decompositions;
            size >= 6 * image size.
        <val_buffer>
            On output: contains the values of the feature response; size >=
            image size.
        <quat_buffer>
            On output: contains the respective eigenvectors of the scale that
            produced the feature response, represented as "squeezed"
            quaternions, as produced by the <qsqueeze()> function.
        <gaussian_filter_3d>
            A <GaussianFilter3D> instance.
        <eigendecomposition>
            An <Eigendecomposition> instance.
        <vesselness_update>
            A <VesselnessUpdate> instance.
        
        Return
            The two-tuple (val_buffer, quat_buffer) containing the results.
            Will be the input buffers if <val_buffer> and <quat_buffer> were
            given, will be new buffers if they were None.
        """
        self._init_fields(params, shape, voxel_size)
        self._init_buffers(img_buffer, swap_buffer1, swap_buffer2,
                           hessian_buffer, val_buffer, quat_buffer)
        self._init_workers(gaussian_filter_3d, eigendecomposition, vesselness_update)
        
        for scale in self._scales_vx:
            
            self._hessian.execute(img_buffer, self._swap_buffer1,
                                  self._swap_buffer2, shape,
                                  self._hessian_buffer, scale)
            self._eigendecomposition.execute(self._nvoxels,
                                             self._hessian_buffer)
            self._vesselness_update.execute(self._nvoxels, self._val_buffer,
                                            self._quat_buffer,
                                            self._hessian_buffer,
                                            self._params.bright_objects,
                                            self._params.alpha,
                                            self._params.beta,
                                            self._params.c,
                                            self._swap_buffer1,
                                            self._swap_buffer2)
            
        return self._val_buffer, self._quat_buffer
        
    def _init_fields(self, params, shape, voxel_size):
        
        self._params = params
        self._shape = shape
        self._nvoxels = np.prod(shape)
        self._nbytes = self._nvoxels * self._dbytes
        self._voxel_size = voxel_size
        
        scales_mm = np.linspace(*params.scales).reshape(-1, 1)  # Mx1, [mm]
        scaling = np.divide(1., voxel_size).reshape(1, -1)  # 1x3, [voxel/mm]
        self._scales_vx = np.dot(scales_mm, scaling)  # Mx3, [voxel]
        
    def _init_buffers(self, img_buffer, swap_buffer1, swap_buffer2,
                      hessian_buffer, val_buffer, quat_buffer):
        
        def init(given, multiplier):
            if given is not None:
                assert given.size >= self._nbytes * multiplier
                result = given
            else:
                result = cl.Buffer(self._loader.ctx, cl.mem_flags.READ_WRITE,
                                   self._nbytes * multiplier)
            return result
        
        assert img_buffer is not None
        self._img_buffer     = init(img_buffer,     multiplier=1)
        self._swap_buffer1   = init(swap_buffer1,   multiplier=1)
        self._swap_buffer2   = init(swap_buffer2,   multiplier=1)
        self._hessian_buffer = init(hessian_buffer, multiplier=6)
        self._val_buffer     = init(val_buffer,     multiplier=1)
        self._quat_buffer    = init(quat_buffer,    multiplier=3)
        
        # Initialize <self._val_buffer> with -inf, for the maximum vesselness
        # response to deliver the correct comparison result in the first scale
        self._to_value_kernel(self._loader.queue, (self._nvoxels, ), None,
                              self._val_buffer, -np.inf)
        
    def _init_workers(self, gaussian_filter_3d, eigendecomposition, vesselness_update):
        
        if gaussian_filter_3d is None:
            gaussian_filter_3d = GaussianFilter3D(self._loader)
        self._hessian = Hessian(gaussian_filter_3d)
        
        if eigendecomposition is None:
            eigendecomposition = Eigendecomposition(self._loader)
        self._eigendecomposition = eigendecomposition
        
        if vesselness_update is None:
            vesselness_update = VesselnessUpdate(self._loader)
        self._vesselness_update = vesselness_update
        

class FeatureFilter(helpers.parameters.TreePrintable):
    """
    Calculate Frangi's vesselness feature [1] for given 3D images in their
    given ROIs at the present scales.
    """

    DummyBuffer = namedtuple("DummyBuffer", ["size"])
    
    def __init__(self, vesselness_params, loader=None, verbose=True):
        """
        <vesselness_params>
            A <VesselnessCalculation.Parameters> instance.
        <loader>
            A <ClProgramLoader> instance; will be used if given, will be
            created if None.
        <verbose>
            Verbose console output on progress (True) or not (False) --
            currently not in use.
        """
        self._dtype = np.float32
        self._dbytes = self._dtype().nbytes
        
        self._loader = ClProgramLoader if loader is None else loader
        
        self.vesselness_params = vesselness_params
        
        # OpenCL buffers; initialize them with a dummy buffer of size -1, so
        # that they are created correctly on the first call of
        # <self._init_buffers()>
        self._reset_buffers()
        
        # Workers ("round_up=False" for better compatibility with Scipy)
        self._gaussian_filter_3d = GaussianFilter3D(self._loader, round_up=False)
        self._eigendecomposition = Eigendecomposition(self._loader)
        self._vesselness_calculation = VesselnessCalculation(self._loader)
        
    def _reset_buffers(self):
        
        self._img_buffer     = self.DummyBuffer(size=-1)
        self._swap_buffer1   = self.DummyBuffer(size=-1)
        self._swap_buffer2   = self.DummyBuffer(size=-1)
        self._hessian_buffer = self.DummyBuffer(size=-1)
        self._val_buffer     = self.DummyBuffer(size=-1)
        self._quat_buffer    = self.DummyBuffer(size=-1)
        
    def _init_buffers(self, nbytes):
        
        def init(buff, multiplier):
            """Create new buffer if required number of bytes changed."""
            required_size = nbytes * multiplier
            if buff.size != required_size:
                buff = cl.Buffer(self._loader.ctx, cl.mem_flags.READ_WRITE,
                                 required_size)
            return buff
        
        self._img_buffer     = init(self._img_buffer,     multiplier=1)
        self._swap_buffer1   = init(self._swap_buffer1,   multiplier=1)
        self._swap_buffer2   = init(self._swap_buffer2,   multiplier=1)
        self._hessian_buffer = init(self._hessian_buffer, multiplier=6)
        self._val_buffer     = init(self._val_buffer,     multiplier=1)
        self._quat_buffer    = init(self._quat_buffer,    multiplier=3)
        
    def execute_for(self, img, roi, voxel_size=None):
        """
        Calculate the vesselness feature for the given <img> at all positions
        in the given <roi>.
        
        <img>
            Three-dimensional Numpy array.
        <roi>
            Three options: (1) it is a Numpy array of <img>'s shape, marking
            the boundaries of the region of interest with non-zeros and
            containing zeros otherwise; (2) it explicitly contains the slicing
            information for the region of interest as a 3-tuple of <slice>
            instances; (3) it is None, which means that the complete image
            should be processed.
        <voxel_size>
            The voxel size for <img> (three-tuple, [mm/voxel]), assuming unit
            voxel size if None.
        
        Return
            Tuple (vesselness, vesselness_dirs, slc):
            <vesselness>
                The vesselness response for the given roi (with possible
                positive margin).
            <vesselness_dirs>
                The respective ordered eigenvectors represented by "squeezed"
                unit quaternions (4D, "eigenvectors in columns") (cf.
                <quaternion_math.qsqueeze()> and <quaternion_math.qexpand()>).
            <slc>
                The respective slicing information (with possible positive
                margins).
        """
        voxels = img
        voxel_size = np.ones_like(img.shape) if voxel_size is None else voxel_size
        cropped_voxels, slc = self._determine_slicing_and_crop_voxels(voxels, roi, voxel_size)
        cropped_voxels = np.require(cropped_voxels, self._dtype, ["C", "A"])
        shape = cropped_voxels.shape

        # (Re-)initialize the OpenCL buffers, then copy image data to GPU
        self._init_buffers(cropped_voxels.nbytes)
        cl.enqueue_copy(self._loader.queue, self._img_buffer, cropped_voxels)
        
        # Actual feature calculation
        self._vesselness_calculation.execute(self.vesselness_params, shape,
                                             voxel_size, self._img_buffer,
                                             self._swap_buffer1,
                                             self._swap_buffer2,
                                             self._hessian_buffer,
                                             self._val_buffer,
                                             self._quat_buffer,
                                             self._gaussian_filter_3d,
                                             self._eigendecomposition,
                                             vesselness_update=None)
        vesselness = np.empty_like(cropped_voxels)
        vesselness_dirs = np.empty(np.r_[shape, 3], dtype=self._dtype)
        cl.enqueue_copy(self._loader.queue, vesselness, self._val_buffer)
        cl.enqueue_copy(self._loader.queue, vesselness_dirs, self._quat_buffer)
        
        return vesselness, vesselness_dirs, slc
    
    def _determine_slicing_and_crop_voxels(self, voxels, roi, voxel_size):
        """
        Calculate optimal margins for feature calculation: it is determined by
        the Gaussian convolution for the Hessian matrices: for the
        <GaussianFilter3D> instance we use, the size of a convolution kernel
        'arm' is n_sigmas * sigma, where <sigma> is the kernel spread and
        <n_sigmas> is a <GaussianFilter3D> parameter. We have to convert from
        [mm] to voxels first, thus the <voxel_size> is necessary.
        """
        if roi is not None:
            max_scales = (self.vesselness_params.scales[1] / 
                          np.asarray(voxel_size, dtype=np.float))
            
            n_sigmas = self._gaussian_filter_3d.n_sigmas
            round_up = self._gaussian_filter_3d.round_up
            
            margin = (np.ceil(n_sigmas * max_scales) if round_up else
                      (n_sigmas * max_scales + 0.5).astype(np.int))

            if roi is not None:
                if isinstance(roi, np.ndarray):
                    roi = slicing.bounding_box(roi, margin=0)
                slc = slicing.add_margin_to_slc(margin, roi)
            cropped_voxels = voxels[slc]
        else:
            slc = None
            cropped_voxels = voxels
            
        return cropped_voxels, slc
