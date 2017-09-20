#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import numpy as np
import os
import pyopencl as cl
import pyopencl.tools as cltools

from helpers.cl_helpers import cl_workgroup_size_3d_for, ClProgramLoader, Reducer
from helpers.parameters import Parameters

DIR_CLCODE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "clcode")

"""
Implement Xu and Prince's gradient vector flow (GVF) [1].

References
[1] C. Xu and J. L. Prince, “Snakes, Shapes, and Gradient Vector Flow,” IEEE
    Transactions on image processing, vol. 7, no. 3, pp. 359–369, 1998.
"""


class GvfParameters(Parameters):
    """
    <self.mu>
        The regularization parameter (the higher, the smoother)
    <self.tol>
        The tolerance for converging
    <self.maxiter>
        The maximum number of GVF iterations
    <self.callback>
        The callback to be used for showing iteration progress (may be None).
        Gets the following input parameters: (1) the current vector field
        components (n+1-dimensional Numpy array, vectors along last dimension);
        (2) the current iteration count; (3) the current value of the
        convergence criterion
    <self.update>
        The number of iterations that will pass before the convergence
        criterion is evaluated again and the callback is called
    <self.normalize>
        If True, normalize the vectors of the resulting vector field to unit
        length before returning
    """
    
    def __init__(self):

        self.mu = None
        self.tol = None
        self.maxiter = None
        self.callback = None
        self.update = None
        self.normalize = None

        
def time_step_for(voxel_size, mu, ndim):
    """
    Determine the time step for one GVF iteration.
    
    This is a trial-and-error approach based on [1,p.363], which I guess is not
    quite correct.
    """
    return np.minimum(0.99, (np.min(voxel_size) ** ndim) / (2.01 ** ndim * mu))


class GvfGpu(object):
    """
    CAUTION: Works for 3D only.
    """
    def __init__(self, vfield, params, voxel_size=None, response=None, cl_program_loader=None):
        
        self.ftype = np.float32
        
        assert vfield.ndim == 4 and vfield.shape[-1] == 3
        self.vfield0 = np.require(vfield[..., 0], dtype=self.ftype, requirements=["C", "A"])
        self.vfield1 = np.require(vfield[..., 1], dtype=self.ftype, requirements=["C", "A"])
        self.vfield2 = np.require(vfield[..., 2], dtype=self.ftype, requirements=["C", "A"])
        self.n_voxels = np.prod(self.vfield0.shape)
        self.params = params
        
        self.voxel_size = (np.ones(3) if voxel_size is None else
                           np.multiply(voxel_size, np.ones(3)))
        
        # Determine the time step
        self.dt = time_step_for(self.voxel_size, params.mu, ndim=3)
        
        # Load the OpenCL kernels
        self.loader = cl_program_loader if cl_program_loader is not None else ClProgramLoader()
        self.reducer = self._create_reducer()
        
        self.lsize_1d = None
        self.lsize_3d = cl_workgroup_size_3d_for(self.vfield0.shape, reverse=True)
        
        self.scalar_shape_rev = self.vfield0.shape[::-1]
        self.scalar_shape_1d = (self.vfield0.size, )
        self.scalar_bytes = self.vfield0.nbytes
        
        self._init_clprograms()
        self._init_buffers()
        
        if response is not None:
            assert response.shape == self.vfield0.shape
            response = np.require(response, dtype=self.ftype, requirements=["C", "A"])
            cl.enqueue_copy(self.loader.queue, self.buf_response, response)
        else:
            self._init_response()
    
    def _create_reducer(self):
        
        path = os.path.join(DIR_CLCODE, "sum_abs.cl")
        wgsize = 1024  # Somewhat arbitrary (must be 2**N)
        defs = {"WGSIZE" : wgsize}
        
        prg_gpu = self.loader.program_from_file(path, defs=defs).sum_abs
        prg_gpu.set_scalar_arg_dtypes([None, None, np.int32])
        prg_cpu = lambda a : np.sum(np.abs(a))  # For the final reduction
        
        reducer = Reducer(prg_gpu, prg_cpu, wgsize, self.loader.queue)
        return reducer
    
    def _init_clprograms(self):
        
        path = os.path.join(DIR_CLCODE, "helpers.cl")
        helpers = self.loader.program_from_file(path)
        self.to_zeros_kernel = helpers.to_zeros
        
        path = os.path.join(DIR_CLCODE, "gvf.cl")
        rss = 1. / (self.voxel_size ** 2)
        defs = {"MU" : self.params.mu,
                "DT" : self.dt,
                "RSS0" : rss[0],
                "RSS1" : rss[1],
                "RSS2" : rss[2],}
        gvf_program = self.loader.program_from_file(path, defs=defs, includes=[DIR_CLCODE])
        self.gvf_delta_kernel = gvf_program.gvfDelta
        self.gvf_delta_i_kernel = gvf_program.gvfDeltaI
        self.gvf_update_kernel = gvf_program.gvfUpdate
        self.mag_delta_kernel = gvf_program.magDelta
        
        self.normalize_kernel = gvf_program.gvfNormalize
        self.response_kernel = gvf_program.response
        
    def _init_buffers(self):
        
        ctx = self.loader.ctx
        flags = cl.mem_flags
        
        self.buf_vfield0   = cl.Buffer(ctx, flags.USE_HOST_PTR | flags.READ_ONLY, hostbuf=self.vfield0)
        self.buf_vfield1   = cl.Buffer(ctx, flags.USE_HOST_PTR | flags.READ_ONLY, hostbuf=self.vfield1)
        self.buf_vfield2   = cl.Buffer(ctx, flags.USE_HOST_PTR | flags.READ_ONLY, hostbuf=self.vfield2)
        
        self.buf_response  = cl.Buffer(ctx, flags.READ_WRITE, self.scalar_bytes)

        # For convergence check
        self.buf_delta0 = cl.Buffer(ctx, flags.READ_WRITE, self.scalar_bytes)
        self.buf_delta1 = cl.Buffer(ctx, flags.READ_WRITE, self.scalar_bytes)
        self.buf_delta2 = cl.Buffer(ctx, flags.READ_WRITE, self.scalar_bytes)
        
        self.buf_gvf0      = cl.Buffer(ctx, flags.READ_WRITE, self.scalar_bytes)
        self.buf_gvf1      = cl.Buffer(ctx, flags.READ_WRITE, self.scalar_bytes)
        self.buf_gvf2      = cl.Buffer(ctx, flags.READ_WRITE, self.scalar_bytes)
        
    def _init_response(self):
        
        self.response_kernel(self.loader.queue, self.scalar_shape_1d, self.lsize_1d,
                             self.buf_response, self.buf_vfield0, self.buf_vfield1, self.buf_vfield2)
        
        
    def execute(self):
        
        try:
            return self._execute()
        except Exception as e:
            cltools.clear_first_arg_caches()
            raise e
        
    def _execute(self):
        
        params = self.params
        self._gvf_to_zeros()
        
        for i in xrange(int(params.maxiter)):
            
            # Perform one GVF iteration: (1) Calculate the update ...
            self.gvf_delta_kernel(self.loader.queue, self.scalar_shape_rev, self.lsize_3d,
                                  self.buf_delta0, self.buf_delta1, self.buf_delta2,
                                  self.buf_gvf0, self.buf_gvf1, self.buf_gvf2,
                                  self.buf_vfield0, self.buf_vfield1, self.buf_vfield2,
                                  self.buf_response)
             
#             self.gvf_delta_i_kernel(self.loader.queue, self.scalar_shape_rev, self.lsize_3d,
#                                     self.buf_delta0, self.buf_gvf0, self.buf_vfield0, self.buf_response)
#             self.gvf_delta_i_kernel(self.loader.queue, self.scalar_shape_rev, self.lsize_3d,
#                                     self.buf_delta1, self.buf_gvf1, self.buf_vfield1, self.buf_response)
#             self.gvf_delta_i_kernel(self.loader.queue, self.scalar_shape_rev, self.lsize_3d,
#                                     self.buf_delta2, self.buf_gvf2, self.buf_vfield2, self.buf_response)

            # (2) ... then apply the update to the current GVF field state
            self.gvf_update_kernel(self.loader.queue, self.scalar_shape_1d, self.lsize_1d,
                                   self.buf_gvf0, self.buf_gvf1, self.buf_gvf2,
                                   self.buf_delta0, self.buf_delta1, self.buf_delta2)
            
            if not i % params.update:
                
                # Calculate the GVF update magnitude for convergence checking
                # (misuse buf_delta0 for catching the output, buf_delta1 for
                # swapping in the reduction)
                buf_mag_delta = self.buf_delta0
                buf_swap = self.buf_delta1
                n_voxels = self.scalar_shape_1d[0]

                self.mag_delta_kernel(self.loader.queue, self.scalar_shape_1d, self.lsize_1d,
                                      buf_mag_delta,
                                      self.buf_delta0, self.buf_delta1, self.buf_delta2)
                
                mag_delta = self.reducer.reduce(buf_mag_delta, n_voxels, buf_swap, self.ftype) / n_voxels
                if params.callback is not None:
                    gvf = self._gvf_from_buffers()
                    params.callback(gvf, i, mag_delta)
                    
                if mag_delta < params.tol:
                    break
                
        if params.normalize:
            # Normalize result vectors to unit length if desired
            self.normalize_kernel(self.loader.queue, self.scalar_shape_1d, self.lsize_1d,
                                  self.buf_gvf0, self.buf_gvf1, self.buf_gvf2)
        
        gvf = self._gvf_from_buffers()
        return gvf
        
                    
    def _gvf_to_zeros(self):
        """Zero out the GVF components."""
        self.to_zeros_kernel(self.loader.queue, self.scalar_shape_1d, self.lsize_1d, self.buf_gvf0)
        self.to_zeros_kernel(self.loader.queue, self.scalar_shape_1d, self.lsize_1d, self.buf_gvf1)
        self.to_zeros_kernel(self.loader.queue, self.scalar_shape_1d, self.lsize_1d, self.buf_gvf2)
        
    def _gvf_from_buffers(self):
        """
        Return the current state of the GVF vector field in one array with the
        vectors along its last dimension (i.e. in the same shape as the input
        vector field).
        """
        queue = self.loader.queue
        gvf = np.empty(np.r_[3, self.vfield0.shape], dtype=self.ftype, order="C")
        cl.enqueue_copy(queue, gvf[0], self.buf_gvf0)
        cl.enqueue_copy(queue, gvf[1], self.buf_gvf1)
        cl.enqueue_copy(queue, gvf[2], self.buf_gvf2)
        gvf = np.rollaxis(gvf, 0, gvf.ndim)
        return gvf
