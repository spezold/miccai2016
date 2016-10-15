#!/usr/bin/env python
# coding: utf-8

"""
Two approaches for adjusting directions of vesselness/tubularity directions,
making sure that neighboring vectors point in more or less the same direction
(rather than the opposite direction); which, in turn, makes it possible to
apply gradient vector flow to the result.
"""
from __future__ import division

import numpy as np
import os
import pyopencl as cl
from scipy.ndimage import label
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

from helpers.cl_helpers import cl_workgroup_size_3d_for, to_cl_image_8192, ClProgramLoader
from helpers.parameters import TreePrintable
from helpers.status import Status
from processing.ridge_detection import RawRidgeDetector, hysteresis_threshold


class AdvancedAdjuster(TreePrintable):
    """
    Detect the ridge coordinates in the tubularity response, propagate a
    consistent direction along their minimum spanning tree, then adjust the
    direction vector of each non-ridge coordinate w.r.t. the closest ridge
    coordinate.
    
    May also be misused for pure ridge detection. In either case, the detected
    ridge may be accessed via <self.ridge> after execution.
    """
    
    def __init__(self, threshold, qeigs, response, voxel_size=1,
                 ridge_detection_only=False, clean_ridge=True, loader=None):
        """
        Instance can be reused by adjusting the respective fields directly.
        
        <threshold>
            If <threshold> is a scalar, then values in the tubularity response
            below the given value are not considered as potential ridge
            coordinates (for noise robustness). If <threshold> is a two-tuple,
            perform hysteresis thresholding for the ridge: consider coordinates
            with tubularity response values between the lower and the upper
            threshold as ridge candidates and retain those that are connected
            to sure ridge points (i.e. coordinates with a tubularity response
            value above the upper threshold) -- similar to the Canny edge
            detector; lower and upper threshold will be sorted for convenience.
            May be None in order to be set later.
        <qeigs>
            The full directional matrices of the tubularity response, each
            represented as a "squeezed" unit quaternion (4-dimensional Numpy
            array expected, with the three quaternion values along the last
            dimension). The direction of the first matrix column will be
            adjusted and returned. May be None in order to be set later.
        <response>
            The values of the tubularity response (3-dimensional Numpy array
            expected). May be None in order to be set later.
        <voxel_size>
            The voxel size of the underlying image, used for determining
            distances for minimum spanning tree and distance from it. May be
            a scalar or a three-tuple (or similar). May be None in order to be
            set later.
        <ridge_detection_only>
            If True, only detect the ridge of the tubularity response, don't
            adjust the directions.
        <clean_ridge>
            If True, only keep the longest straight ridge segments after
            thresholding.
        <loader>
            A <ClProgramLoader> instance. Will be used if given, will be
            created if None.
        """
        self.__threshold = None
        self.__qeigs = None
        self.__response = None
        self.__voxel_size = None
        self.__ridge_detection_only = None
        self.__clean_ridge = None
        self.__ridge = None
        self.__main_dirs_unadjusted = None
        
        self._ridge_detector = None
        self._hysteresis = None

        self._dtype = np.float32  
        self._loader = loader if loader is not None else ClProgramLoader()
        self._lsize_3d = None
        
        if threshold is not None:
            self.threshold = threshold # Also creates the ridge detector
        if qeigs is not None:
            self.qeigs = qeigs
        if response is not None:
            self.response = response
        if voxel_size is not None:
            self.voxel_size = voxel_size
        self.ridge_detection_only = ridge_detection_only
        self.clean_ridge = clean_ridge
        
        self._init_cl_programs()
    
    @property    
    def threshold(self):
        return self.__threshold
    
    @threshold.setter
    def threshold(self, value):
        
        try:
            len(value)  # Distinguish list-like from scalar
            value = tuple(sorted(value))
            self._hysteresis = True
            lower_thresh = value[0]
        except TypeError:
            self._hysteresis = False
            lower_thresh = value
            
        if value != self.__threshold:
            self.__threshold = value
            # If the threshold value changes, a re-initialization of the
            # ridge detector is necessary
            self._init_ridge_detector_with_threshold(lower_thresh)
            
    @property
    def qeigs(self):
        return self.__qeigs
    
    @qeigs.setter
    def qeigs(self, value):
        self._lsize_3d = cl_workgroup_size_3d_for(value.shape[:3], reverse=True)
        recreate_buffers = (self.__qeigs is None) or np.prod(self.__qeigs.shape) != np.prod(value.shape)
        self.__qeigs = np.require(value, dtype=self._dtype, requirements=["C", "A"])
        if recreate_buffers:
            # If the number of values in the given array changes (or if it is
            # set for the first time), the respective OpenCL buffers have to be
            # recreated
            self._init_buffers_for_scalar_bytes(self.__qeigs.nbytes // 3)
            
    @property
    def response(self):
        return self.__response
    
    @response.setter
    def response(self, value):
        self.__response = np.require(value, dtype=self._dtype, requirements=["C", "A"])
        
    @property
    def voxel_size(self):
        return self.__voxel_size
    
    @voxel_size.setter
    def voxel_size(self, value):
        self.__voxel_size = tuple(np.multiply(np.ones(3), value))
        
    @property
    def ridge_detection_only(self):
        return self.__ridge_detection_only
    
    @ridge_detection_only.setter
    def ridge_detection_only(self, value):
        self.__ridge_detection_only = value
        
    @property
    def clean_ridge(self):
        return self.__clean_ridge
        
    @clean_ridge.setter
    def clean_ridge(self, value):
        self.__clean_ridge = value
    
    @property
    def ridge(self):
        """
        After adjustment, the detected ridge positions for convenience.
        """
        return self.__ridge
    
    @property
    def main_dirs_unadjusted(self):
        """
        After adjustment, the extracted *unadjusted* main directions.
        """
        return self.__main_dirs_unadjusted
           
    def _init_cl_programs(self):
        
        SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        PROGRAM_PATH = os.path.join(SRC_DIR, "clcode", "adjust_directions.cl")
        INCLUDE_PATH = os.path.join(SRC_DIR, "clcode")
        
        prg = self._loader.program_from_file(PROGRAM_PATH,
                                             includes=[INCLUDE_PATH])
        self.extract_main_directions_kernel = prg.extractMainDirections
        self.adjust_non_ridge_signs_kernel = prg.adjustNonRidgeSigns
        self.adjust_non_ridge_signs_kernel.set_scalar_arg_dtypes([None, None, None,
                                                                  None, np.int32])
        
    def _init_ridge_detector_with_threshold(self, threshold):
        self._ridge_detector = RawRidgeDetector(self._loader, threshold)
    
    def _init_buffers_for_scalar_bytes(self, nbytes):
        
        flags = cl.mem_flags
        ctx = self._loader.ctx
        
        self._buf_qeigs = cl.Buffer(ctx, flags.READ_WRITE, nbytes * 3)
        self._buf_response = cl.Buffer(ctx, flags.READ_WRITE, nbytes)
        self._buf_ridge = cl.Buffer(ctx, flags.READ_WRITE, nbytes)
        
    def execute(self, verbose=True):
        """
        Adjust main directions as described above; return the adjustment result
        as a four-dimensional Numpy array that contains the vector data along
        its last dimension.
        
        If <self.ridge_detection_only> is True, return None.
        """
        
        assert self.response.shape == self.qeigs.shape[:3] and self.qeigs.shape[3] == 3

        status = Status("Detect ridge", verbose=verbose)
        labeled_ridge = self._label_ridge()
        adjusted_dirs = None
        
        if (not self.ridge_detection_only) or self.clean_ridge:
            
            # In both cases, actual direction adjustment and clean ridge, we
            # need the adjusted directions on the (uncleaned) ridge
            status.update("Extract main directions")
            main_dirs = self._qeigs_to_main_dirs()
            status.update("Adjust ridge directions")
            ridge_coords, ridge_dirs = self._adjust_ridge_directions(labeled_ridge, main_dirs)
            
            if self.clean_ridge:
                
                status.update("Clean ridge")
                labeled_ridge = self._do_clean_ridge(labeled_ridge, ridge_coords, ridge_dirs)
                
                if not self.ridge_detection_only:
                
                    # If we actually want to adjust the directions (i.e.
                    # ridge_detection_only == False), adjust the directions on
                    # the ridge again, now on the cleaned version
                    status.update("Adjust ridge directions again")
                    ridge_coords, ridge_dirs = self._adjust_ridge_directions(labeled_ridge, main_dirs)
            
            if not self.ridge_detection_only:
                
                status.update("Adjust all directions")
                adjusted_dirs = self._adjust_all_directions(ridge_coords, ridge_dirs)
            
        del status
        return adjusted_dirs
        
    def _label_ridge(self):
        """
        Label the ridge voxels, i.e. mark ridge voxels with one, non-ridge
        voxels with zero.
        """
        
        queue = self._loader.queue
        ridge = np.empty_like(self.response)
        
        cl.enqueue_copy(queue, self._buf_qeigs, self.qeigs)
        cl.enqueue_copy(queue, self._buf_response, self.response)
        self._ridge_detector.execute(self._buf_ridge, self._buf_response,
                                     self._buf_qeigs, self.response.shape)
        cl.enqueue_copy(queue, ridge, self._buf_ridge)
        
        if self._hysteresis:
            
            # In the case of hysteresis thresholding, <ridge> contains all
            # candidates, i.e. all voxels that are above the lower threshold
            ridge[:] = self._hysteresis_threshold(ridge)
        
        self.__ridge = ridge
        return ridge
    
    def _hysteresis_threshold(self, ridge):
        
        sure = np.logical_and(ridge, self.response >= self.threshold[1])
        # ^ <sure>: ridge points with responses >= upper threshold
        cand = ridge
        
        return hysteresis_threshold(cand, sure)
        
    def _do_clean_ridge(self, labeled_ridge, ridge_coords, ridge_dirs):
        """
        Clean up the ridge line response: only keep the longest straight ridge
        segments.
        
        In particular, keep those ridge segments whose summed up directional
        vectors are the longest. The number of kept segments is determined as
        
            ceil(3 * ln(e + n)),
            
        where <n> is the number of connected ridge segments in the
        <labeled_ridge>. In this way, the number of retained segments is
        limited to stay around 10, allowing for slightly more retained segments
        if the number of segments is considerably greater.
        
        <labeled_ridge>
            The detected ridge (three-dimensional Numpy array).
        <ridge_coords>
            The ridge coordinates, which we already know anyway, as Nx3 Numpy
            array where <N> is the number of ridge points
        <ridge_dirs>
            The respective adjusted directional vectors (i.e. it has been
            ensured that they point in approximately the same rather than
            opposite directions); same size as <ridge_coords>
        Return
            The cleaned-up ridge response (three dimensional Numpy array of the
            same shape as <ridge>).
        """
        # Account for voxel size in directions
        scaled_dirs = ridge_dirs * self.__voxel_size
        
        dirs_dict = {tuple(c) : d for (c, d) in zip(ridge_coords, scaled_dirs)}
        components = label(labeled_ridge, structure=np.ones((3,3,3), dtype=np.int))[0]
        labels = np.unique(components)[1:]  # "1:" to exclude 0
        
        # For each label, sum up its directional vectors and remember the length
        lengths = {}
        for l in labels:
            dirs = [dirs_dict[tuple(i)] for i in np.array(np.where(components == l)).T]
            lengths[l] = np.linalg.norm(np.sum(dirs, axis=0))
    
        # Keep the longest results
        sorted_labels = sorted(lengths, key=lambda k : lengths[k], reverse=True)
        boundary = int(np.ceil(3 * np.log(np.e + len(sorted_labels))))
        print "Keeping %s of %s labels." % (boundary, len(sorted_labels))
        kept_labels = sorted_labels[:boundary]
        cleaned_ridge = np.in1d(components, kept_labels).astype(labeled_ridge.dtype).reshape(components.shape)
        
        self.__ridge = cleaned_ridge
        return cleaned_ridge
    
    def _qeigs_to_main_dirs(self):
        """
        From the current quaternions that represent the full directional
        matrices of the tubularity response, extract the main directions, i.e.
        the ones belonging to the zeroth eigenvalue; return them in a new
        four-dimensional Numpy array with the main directions along the last
        dimension.
        
        Also adjust the respective OpenCL buffer, which currently holds the
        "squeezed" directional quaternions, to hold the main directional
        vectors afterwards.
        """
        # After the next step, the <self._buf_qeigs> buffer also contains the
        # main directions
        self.extract_main_directions_kernel(self._loader.queue,
                                            (self.response.size, ), None,
                                            self._buf_qeigs, self._buf_qeigs)
        main_dirs = np.empty_like(self.qeigs)
        cl.enqueue_copy(self._loader.queue, main_dirs, self._buf_qeigs)
        
        self.__main_dirs_unadjusted = main_dirs.copy()
        return main_dirs
        
    
    def _adjust_ridge_directions(self, labeled_ridge, main_dirs):
        """
        Approach: (1) Calculate a minimum spanning tree (MST) over all ridge
        points, (2) starting from the first ridge point (whose choice is kind
        of arbitrary), propagate its direction along all edges of the MST, (3)
        return (coords, dirs), where <coords> is a uint16 Nx3 Numpy array of
        all ridge coordinates (voxel indices) and <dirs> is float32 Nx3 Numpy
        array of the respective adjusted directional vectors.
        """
        # Get all ridge points, then calculate their distances
        i_ridge = zip(*np.where(labeled_ridge))  # Nx3; tuples in rows
        dists = pdist(np.multiply(i_ridge, self.voxel_size), "euclidean")
        
        # Get the minimum spanning tree (MST); the result is a sparse matrix
        mst = minimum_spanning_tree(squareform(dists), overwrite=True)
        
        # Create array that holds the adjusted ridge point directions
        adjusted = np.ones_like(i_ridge, dtype=np.float32) * np.float32(np.nan)

        # Use direction of zeroth ridge point as reference
        adjusted[0] = main_dirs[i_ridge[0]]
        
        # Traverse the MST starting from 0, adjust all vectors on the way
        # (nonrecursive traversal, avoiding maximum recursion depth problems)
        
        get_children = lambda i : np.r_[mst[i].nonzero()[1],
                                        mst[:, i].nonzero()[0]]
        # ^ children = nonzero entries in the MST matrix (need to make the
        # returned 2D indices 1D by disposing of the input index dimension)
        visited = lambda i : not np.isnan(adjusted[i, 0])
        adjust = lambda cur, ref : cur if np.dot(cur, ref) >= 0 else -cur
        # ^ Adjust current direction <cur> w.r.t. reference direction <ref>

        stack = []
        stack.append(0)
        
        while stack:
            
            i = stack.pop()
            dir_i = adjusted[i]
            children = get_children(i)
            
            for child in children:
                if not visited(child):
                    
                    # Actual child adjustment: index into 4D <main_dirs>
                    # array via 3D indices in <i_ridge> to get the child's
                    # direction, store the adjusted direction in <adjusted>
                    dir_child = main_dirs[i_ridge[child]]
                    adjusted[child] = adjust(dir_child, dir_i)
                    stack.append(child)
                        
        return(np.asarray(i_ridge, dtype=np.uint16, order="C"), adjusted)
    
    def _adjust_all_directions(self, ridge_coords, ridge_dirs):
        """
        Based on the given <ridge_coords> (Nx3 uint16 array) and the respective
        adjusted <ridge_dirs> (Nx3 float32 array), adjust the signs of all
        directional vectors that are currently stored in the respective OpenCL
        buffer so that for each directional vector the dot product with the
        direction of the nearest ridge point is maximized.
        
        Return a 4D Numpy array containing the result.
        """
        num_ridge_coords = len(ridge_coords)
        assert len(ridge_coords) == len(ridge_dirs)
        
        queue = self._loader.queue
        voxel_size = np.array(np.r_[self.voxel_size, np.nan], dtype=np.float32)
        # ^ Recall that float3 in OpenCL is aligned to 4-elements => need a 4th
        # value as placeholder here
        
        coords_image = to_cl_image_8192(self._loader.ctx, ridge_coords, fill=np.nan)
        dirs_image = to_cl_image_8192(self._loader.ctx, ridge_dirs, fill=np.nan)
        
        self.adjust_non_ridge_signs_kernel(queue, tuple(self.response.shape[::-1]), self._lsize_3d,
                                           self._buf_qeigs, coords_image, dirs_image,
                                           voxel_size, num_ridge_coords)
        
        dirs = np.empty_like(self.qeigs)
        cl.enqueue_copy(queue, dirs, self._buf_qeigs)
        return dirs
        

class TransformationCompleter(TreePrintable):
    """
    Complete a vector field to a field of transformation matrices by finding
    for each vector two orthogonal ones that together form a matrix with a
    determinant of value 1.
    """
    def __init__(self, dirs, loader=None):
        """
        <dirs>
            The vector field to be completed (4-dimensional Numpy array
            expected, with vectors along its last axis). May be None to be set
            later.
        <loader>
            A <ClProgramLoader> instance. Will be used if given, will be
            created if None.
        """
        
        self.__dirs = None
        self._buf = None  # Used for both input and result
        
        self._dtype = np.float32
        self._loader = loader if loader is not None else ClProgramLoader()
        self._init_cl_programs()
        
        if dirs is not None:
            self.dirs = dirs
        
    def _init_cl_programs(self):
        
        SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        PROGRAM_PATH = os.path.join(SRC_DIR, "clcode", "adjust_directions.cl")
        INCLUDE_PATH = os.path.join(SRC_DIR, "clcode")
        
        self.complete_transformations_kernel = self._loader.program_from_file(PROGRAM_PATH,
                                                                              defs={"THRESHOLD" : -1},  # dummy value
                                                                              includes=[INCLUDE_PATH]).completeTransformations
                                                                             
    def complete(self):
        """
        Assuming that the given vectors are the first columns of transformation
        matrices, complete each of them to a full transformation matrix by (1)
        normalizing it, (2) finding two other arbitrary unit vectors that are
        orthogonal to the given one and that together result in a matrix with a
        determinant of value 1, (3) converting the matrix to an equivalent unit
        quaternion and "squeezing" it to three values (cf.
        <quaternion_math.qsqueeze()>).
        
        Return the result as a Numpy array of same shape.
        """
        queue = self._loader.queue
        cl.enqueue_copy(queue, self._buf, self.dirs)
        self.complete_transformations_kernel(queue, (self.dirs.size // 3, ), None,
                                             self._buf, self._buf)
        quats = np.empty_like(self.dirs)
        cl.enqueue_copy(queue, quats, self._buf)
        return quats
    
    @property
    def dirs(self):
        return self.__dirs
    
    @dirs.setter
    def dirs(self, value):
        recreate_buffers = (self.__dirs is None) or np.prod(self.__dirs.shape) != np.prod(value.shape)
        self.__dirs = np.require(value, dtype=self._dtype, requirements=["C", "A"])
        if recreate_buffers:
            self._buf = cl.Buffer(self._loader.ctx,
                                  cl.mem_flags.READ_WRITE,
                                  self.__dirs.nbytes)
