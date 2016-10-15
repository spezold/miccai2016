#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import numpy as np
import os

from helpers.cl_helpers import cl_workgroup_size_3d_for


class RawRidgeDetector(object):
    """
    Find the ridge in the given Frangi vesselness response.
    """
    
    def __init__(self, loader, threshold):
        """
        <loader>
            A <ClProgramLoader> instance.
        <threshold>
            Absolute value below which values in the vesselness response will
            not be considered as ridge candidates -- for noise robustness.
        """
        self.__threshold = threshold
        
        self._dtype = np.float32
        self._loader = loader
        self._shape = None
        
        self._init_cl_programs()
        
    def _init_cl_programs(self):
        
        SRC_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        PROGRAM_PATH = os.path.join(SRC_DIR, "clcode", "detect_ridges.cl")
        INCLUDE_PATH = os.path.join(SRC_DIR, "clcode")
        
        defs = {"THRESHOLD" : self.threshold}
        prg = self._loader.program_from_file(PROGRAM_PATH, defs=defs,
                                             includes=[INCLUDE_PATH])
        self._detect_ridges_kernel = prg.detectRidges
        
    @property
    def threshold(self):
        
        return self.__threshold
    
    def execute(self, result_buffer, response_buffer, qeigs_buffer, shape):
        """
        <result_buffer>
            Buffer that will hold the result as float32 values.
        <response_buffer>
            Buffer that holds the vesselness response as float32 values.
        <qeigs_buffer>
            The full directional matrices of the vesselness response, each
            represented as a "squeezed" unit quaternion of three consecutive
            float32 values.
        <shape>
            The 3D shape of the image data to be processed; given as a three-
            element sequence.
            
        Return
            None
        """
        assert len(shape) == 3
        data_bytes = np.prod(shape) * self._dtype().nbytes
        assert result_buffer.size >= data_bytes
        assert response_buffer.size >= data_bytes
        assert qeigs_buffer.size >= 3 * data_bytes
        
        if np.any(shape != self._shape):
            self._shape = shape
            self._lsize_3d = cl_workgroup_size_3d_for(shape, reverse=True)

        self._detect_ridges_kernel(self._loader.queue, tuple(shape[::-1]), self._lsize_3d,
                                   result_buffer, response_buffer, qeigs_buffer)


def hysteresis_threshold(cand, sure):
    """
    Calculate hysteresis threshold for the given labeled voxels, similar to the
    Canny edge detector.
    
    CAUTION: The values of <cand> and <sure> will be adjusted, <sure> will hold
    the end result.
    
    <cand>
        Voxels with value >= lower hysteresis threshold should be labeled 1,
        background voxels should be labeled 0 (it does not matter whether this
        overlaps with <sure> or not).
    <sure>
        Voxels with value >= upper hysteresis threshold should be labeled 1,
        background voxels should be labeled 0.
        
    Return
        The altered <sure> array for convenience, with the foreground voxels
        labeled 1 and the background voxels labeled 0.
    """
    assert cand.shape == sure.shape
    
    def find_neighbors(pos, cand):
        
        lower_bound = np.maximum(0, np.subtract(pos, 1))
        # ^ Avoid negative indices
        neighborhood = cand[lower_bound[0]:pos[0]+2,
                            lower_bound[1]:pos[1]+2,
                            lower_bound[2]:pos[2]+2]
        neighbors = zip(*np.where(neighborhood))
        # Special treatment of neighbors being empty is necessary, as an
        # empty list cannot be added to <lower_bound>
        if neighbors:
            neighbors = lower_bound + neighbors
            # Indexing needs tuples
            neighbors = [tuple(n) for n in neighbors]
        return neighbors
    
    stack = zip(*np.where(sure))
    
    while stack:
        
        c = stack.pop()
    
        # Label the current coordinate as sure and remove it from
        # candidates, then push the neighboring candidates to the stack
        sure[c] = True
        cand[c] = False
        neighbors = find_neighbors(c, cand)
        stack.extend(neighbors)
        
    return sure

