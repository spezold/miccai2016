#!/usr/bin/env python
# coding: utf-8

"""
A continuous max flow approach with anisotropic total variation (aTV)
regularization [1], with a solution scheme similar to [2, 3].

References
[1] M. Grasmair and F. Lenzen, “Anisotropic Total Variation Filtering,” Appl
    Math Optim, vol. 62, no. 3, pp. 323–339, Dec. 2010.
[2] J. Yuan, E. Bae, and X.-C. Tai, “A study on continuous max-flow and min-cut
    approaches,” in 2010 IEEE Conference on Computer Vision and Pattern
    Recognition (CVPR), 2010, pp. 2217–2224.
[3] J. Yuan, E. Bae, X.-C. Tai, and Y. Boykov, “A study on continuous max-flow
    and min-cut approaches,” UCLA, CAM, UCLA, technical report CAM 10-61, 2010.
"""

from __future__ import division

import os
import numpy as np
import pyopencl as cl
import pyopencl.tools as cltools

from helpers.cl_helpers import ClProgramLoader, Reducer
from helpers.parameters import Parameters, TreePrintable
from helpers.status import Status


DIR_CLCODE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "clcode")
DIR_CLCODE_ANISOTROPIC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "clcode-anisotropic")


class CmfCutQAnisotropicGpu(TreePrintable):
    
    class CutParameters(Parameters):
        
        def __init__(self):
            
            self.cc = 0.2  # step size for the augmented lagrangian
            self.error_bound = 1e-4
            self.num_iter = 1000
            self.update = 1
            self.steps = 0.11  # step size for the gradient ascent
            
            self.lsize_3d = (64, 2, 1)  # For the "naive" kernel implementations
            self.lsize_1d = None
            
    def __init__(self, params, cl_program_loader=None):
        
        self.float_type = np.float32
        self.float_bytes = int(self.float_type().itemsize)
        
        self.params = params
        self.loader = cl_program_loader if cl_program_loader is not None else ClProgramLoader()
        self.reducer = self._create_reducer()
        
        # Parameters of the current OpenCL kernels:
        self.scaling = None      # Current image scaling
        self.const_sqevals = None
        # ^ Do we use constant or pointwise eigenvalues? If pointwise, set to
        # False; if constant give their square roots in desired order here)
        self.supervised = None   # Do we use supervised or unsupervised cut?
        
    def _create_reducer(self):
        
        path = os.path.join(DIR_CLCODE, "sum_abs.cl")
        wgsize = 1024  # Somewhat arbitrary (must be 2**N)
        defs = {"WGSIZE" : wgsize}
        
        prg_gpu = self.loader.program_from_file(path, defs=defs).sum_abs
        prg_gpu.set_scalar_arg_dtypes([None, None, np.int32])
        prg_cpu = lambda a : np.sum(np.abs(a))  # For the final reduction
        
        reducer = Reducer(prg_gpu, prg_cpu, wgsize, self.loader.queue)
        return reducer
    
    def _reinit_clprograms(self, scaling, const_sqevals, supervised):
        """
        Reinitializes the OpenCL kernels for the given new parameters, if
        necessary.
        
        <scaling>
            image scaling: three-tuple or similar
        <const_sqevals>
            False if pointwise eigenvalues are to be used; a three-tuple or
            similar of the square roots of the constant eigenvalues in the
            desired order otherwise
        <supervised>
            True if the supervised version of the cut shall be used, False
            otherwise
        """
        if (np.array_equal(self.scaling, scaling) and
            np.array_equal(self.const_sqevals, const_sqevals) and
            self.supervised == supervised):
            return
        
        self.scaling = scaling
        self.const_sqevals = const_sqevals
        self.supervised = supervised
        
        # Calculate step sizes and adjust them for numerical stability (note to
        # self: cf. notes from 19.01.2015 regarding how the operator norm is
        # calculated for arbitrary scalings -- at least I think that's what
        # relevant here)
        rs = np.divide(1., scaling)
        min_scaling = np.min(scaling)
        ss = np.divide(self.params.steps, scaling) * min_scaling ** 2
        self.rcc = rcc = 1. / (self.params.cc * min_scaling ** 2)
        
        defs = {"RCC" : rcc,
                "RS0" : rs[0],
                "RS1" : rs[1],
                "RS2" : rs[2],
                "SS0" : ss[0],
                "SS1" : ss[1],
                "SS2" : ss[2],
                }
        
        if not np.array_equal(const_sqevals, False):
            defs["CONST_EVALS"] = None
            defs["SQEVAL0"] = const_sqevals[0]  
            defs["SQEVAL1"] = const_sqevals[1]  
            defs["SQEVAL2"] = const_sqevals[2]  
            
        if supervised:
            defs["SUPERVISED"] = None
        
        path = os.path.join(DIR_CLCODE_ANISOTROPIC, "update_qp.cl")
        update = self.loader.program_from_file(path, defs=defs, includes=[DIR_CLCODE])
        self.update_p   = update.update_p
        self.update_u   = update.update_u
        self.update_tmp = update.update_tmp
        
        path = os.path.join(DIR_CLCODE, "helpers.cl")
        helpers = self.loader.program_from_file(path)
        self.to_zeros = helpers.to_zeros
        self.el_min = helpers.el_min
        
        
    def execute(self, img, quats, eigs, alpha, cs, ct, labels=None, scaling=None, verbose=True):
        """
        <img>
            The three-dimensional image to be processed, values \in [0,1]. Used
            for initialization and for shape information (three-dimensional
            Numpy array expected).
        <quats>
            The quaternions to be used for anisotropic TV (aTV) regularization.
            In particular, aTV uses the regularization term
            
                (grad(u).T A grad(u)) ** (1/2) .
                
            Here, we require for each pixel the quaternion that represents the
            eigenvector matrix R that results from an eigendecomposition of the
            strongly positive definite matrix A in the form
            
                A = R L R.T,
                
            where L is the diagonal matrix of non-negative eigenvalues. The
            quaternions are expected to be given as a four-dimensional array,
            where the last dimension holds 4 elements or the 3 elements as
            produced by the qsqueeze() function.
        <eigs>
            The non-negative eigenvalues of A in the same order as the
            eigenvector matrix (see <quats> above).
                If pointwise eigenvalues are to be used:
                    expected to be given as a four-dimensional array, where the
                    last dimension holds the 3 ordered eigenvalues
                If constant eigenvalues are to be used:
                    expected to be given as a three-tuple or similar, holding
                    the 3 ordered eigenvalues
        <alpha>
            Weighting for the aTV term. May be defined pointwise or as a
            scalar.
        <cs>
            The pointwise foreground weights (three-dimensional Numpy array
            expected).
        <ct>
            The pointwise background weights (three-dimensional Numpy array
            expected).
        <labels>
            Foreground labels for the supervised version of the algorithm
            (three-dimensional Numpy array expected, foreground seeds labeled
            with 1, background seeds labeled with 2, remaining voxels 0). If
            None, the unsupervised version of the algorithm is used.
        <scaling>
            The voxel size / voxel distance [mm/voxel] (three-tuple or similar
            expected). If None, assume 1 mm/voxel isotropic.
        """
        try:
            return self._execute(img, quats, eigs, alpha, cs, ct, labels, scaling, verbose)
        except Exception as e:
            cltools.clear_first_arg_caches()
            raise e
        
    def _execute(self, img, quats, eigs, alpha, cs, ct, labels, scaling, verbose):
        
        loader = self.loader
        ctx = loader.ctx
        queue = loader.queue
        flags = cl.mem_flags
        
        # FIXME: Make sure that global size is a multiple of local size (for
        # each dimension) always
        lsize_1d = self.params.lsize_1d
        lsize_3d = self.params.lsize_3d
        
        float_type = self.float_type
        
        img = np.require(img, dtype=float_type, requirements=["C", "A"])
        
        scaling = (1, 1, 1) if scaling is None else scaling

        # In any case, we need the square roots of the eigenvalues
        sqevals = np.sqrt(np.asanyarray(eigs))
        if sqevals.ndim == 1:
            const_sqevals = sqevals
        else:
            const_sqevals = False
        
        supervised = True if labels is not None else False

        # Reinitialize the OpenCL kernels for the current settings
        self._reinit_clprograms(scaling, const_sqevals, supervised)
        
        img_shape_rev = img.shape[::-1]
        img_shape_1d = (img.size, )
        img_bytes = img.nbytes
        
        if not isinstance(alpha, np.ndarray):
            alpha = np.multiply(alpha, np.ones(img.shape), dtype=float_type)
        alpha = np.require(alpha, dtype=float_type, requirements=["C", "A"])
        
        cs = np.require(cs, dtype=float_type, requirements=["C", "A"])
        ct = np.require(ct, dtype=float_type, requirements=["C", "A"])
        
        if supervised:
            labels = np.require(labels, dtype=np.ubyte, requirements=["C", "A"])
        
        # Arrange the values for the eigs buffer: if constant eigenvalues are
        # to be used, it will only contain the squeezed quaternions; if
        # pointwise eigenvalues are to be used, the square roots of the
        # eigenvalues go first and the squeezed quaternions go last
        if np.array_equal(const_sqevals, False):
            eigs = np.empty(np.r_[img.shape, 6], dtype=float_type)
            eigs[..., :3] = sqevals
            eigs[..., 3:] = quats
        else:  # const_sqevals is False
            eigs = np.require(quats, dtype=float_type, requirements=["C", "A"])
        
        
        # Create necessary buffers
        buf_hp0    = cl.Buffer(ctx, flags.READ_WRITE, img_bytes)
        buf_hp1    = cl.Buffer(ctx, flags.READ_WRITE, img_bytes)
        buf_hp2    = cl.Buffer(ctx, flags.READ_WRITE, img_bytes)

        buf_pt     = cl.Buffer(ctx, flags.READ_WRITE, img_bytes)
        buf_tmp    = cl.Buffer(ctx, flags.READ_WRITE, img_bytes)
        buf_u      = cl.Buffer(ctx, flags.READ_WRITE, img_bytes)
        
        buf_eigs   = cl.Buffer(ctx, flags.USE_HOST_PTR | flags.READ_ONLY, hostbuf=eigs)
        buf_alpha  = cl.Buffer(ctx, flags.USE_HOST_PTR | flags.READ_ONLY, hostbuf=alpha)
        
        buf_cs     = cl.Buffer(ctx, flags.USE_HOST_PTR | flags.READ_ONLY, hostbuf=cs)
        buf_ct     = cl.Buffer(ctx, flags.USE_HOST_PTR | flags.READ_ONLY, hostbuf=ct)
        if supervised:
            buf_labels = cl.Buffer(ctx, flags.USE_HOST_PTR | flags.READ_ONLY, hostbuf=labels)
            
        buf_reduce = cl.Buffer(ctx, flags.READ_WRITE,
                               self.reducer.swap_buffer_bytes_for(img.size, float_type))
         
        
        # Initialize buffers
        self.to_zeros(queue, img_shape_1d, lsize_1d, buf_hp0)        
        self.to_zeros(queue, img_shape_1d, lsize_1d, buf_hp1)        
        self.to_zeros(queue, img_shape_1d, lsize_1d, buf_hp2)        

        self.el_min(queue, img_shape_1d, lsize_1d, buf_cs, buf_ct, buf_pt)

        u = np.divide(img, np.max(img) + np.finfo(img.dtype).tiny)  # Normalized image
        u = np.require(u, dtype=float_type, requirements=["C", "A"])
        cl.enqueue_copy(queue, buf_u, u)

        # Initialize tmp:
        # tmp <- divhp - ps + pt - u/c == 0 - u/c (assuming that ps == pt)
        self.to_zeros(queue, img_shape_1d, lsize_1d, buf_tmp)
        self.update_tmp(queue, img_shape_1d, lsize_1d, buf_tmp, buf_u)

        status = Status("Calculate the cut", verbose=verbose)
        for i in range(self.params.num_iter):
            
            # Optimize flows p then recalculate the divergences
            
            # p = p + steps * H.T grad(div H p - p_s + p_t - u / cc)
            # p = project(p; alpha)
            # <=>
            # p = H^-1 hp + steps * H.T grad(tmp) with tmp == div H p - ...
            # p = project(...)
            # hp = H p
            self.update_p(queue, img_shape_rev, lsize_3d,
                          buf_hp0, buf_hp1, buf_hp2,
                          buf_tmp, buf_eigs, buf_alpha)
            
            # divhp = div(hp)
            #
            # ps = project((1 - u) / cc + divhp + pt; cs) (or resp. supervised equation)
            # pt = project(u / cc - divhp + ps; ct) (or resp. supervised equation)
            #
            # tmp = divhp - ps + pt
            # u = u - cc * tmp
            args = [queue, img_shape_rev, lsize_3d,
                    buf_hp0, buf_hp1, buf_hp2,
                    buf_tmp, buf_cs, buf_ct,
                    buf_pt, buf_u]
            if supervised:
                args.append(buf_labels)
            self.update_u(*args)

            if not i % self.params.update:
                # Calculate the convergence criterion as
                #
                #     err_iter = cc * sum(abs(tmp)) / img_size
                err_iter = self.reducer.reduce(buf_tmp, img.size, buf_reduce,
                                               float_type) / (img.size * self.rcc)
                if verbose:
                    print i + 1, err_iter
                
                if np.isnan(err_iter):
                    if verbose:
                        print "Convergence criterion is NaN. Stop iterating."
                    break
                elif err_iter < self.params.error_bound:
                    if verbose:
                        print "Converged."
                    break

            # tmp = tmp - u / cc == divhp - ps + pt - u / cc
            self.update_tmp(queue, img_shape_1d, lsize_1d, buf_tmp, buf_u)
        
        del status
        cl.enqueue_copy(queue, u, buf_u)    
        return u
