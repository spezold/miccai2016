#!/usr/bin/env python
# coding: utf-8

"""
Functions for slicing image stacks.
"""

from __future__ import division

import numpy as np


def bounding_box(mask, margin=0):
    """
    Calculate the bounding box around all <mask> values (i.e. values != 0) and
    return the respective slicing parameters. Set a <margin> value in order to
    additionally include (if > 0) or exclude (if < 0) values towards all sides
    if desired (may be a scalar or a d-tuple, where <d> is the number of
    dimensions of <mask>; will be rounded to integers with <np.round()>).
    """
    margin = (np.ones(mask.ndim) * np.round(margin)).astype(np.int)
    # Get all mask position indices (d-tuple of N-length arrays, where <N> is
    # the number of mask positions and <d> is the mask's dimensionality). Using
    # <nonzero()> instead of <argwhere()> (the latter is proposed in [1]) does
    # not really make a difference, apart from the structure of the result
    #
    # References
    # [1] http://stackoverflow.com/questions/4808221/ (20121121)
    i = np.nonzero(mask)  # dxN
    try:
        # Get the minima and maxima for all dimensions as slicing boundaries,
        # adjust for margin value, avoid negative values
        lowers = np.maximum(0, np.min(i, axis=1) - margin)
        uppers = np.maximum(0, np.max(i, axis=1) + margin + 1)
        # ^ "+1" because the upper boundary is excluded when slicing
    except ValueError:
        # If any of the dimensions is zero (i.e., the mask is empty), set the
        # lower value to the upper value, resulting in an empty slice
        lowers = uppers = np.zeros(mask.ndim, dtype=np.int)
    return tuple(slice(l, u, 1) for l, u in zip(lowers, uppers))
    # ^ lower, upper, step


def add_margin_to_slc(margin, slc):
    """
    Add the given <margin> (d-tuple or scalar; will be rounded to integers with
    <np.round()>) to the given slicing information <slc> (d-tuple of <slice>
    instances that all have a <start>, <stop>, and <step> value set), avoiding
    negative indices.
    
    Return the adjusted slicing information as a new d-tuple of <slice>
    instances.
    """
    d = len(slc)
    margin = (np.ones(d) * np.round(margin)).astype(np.int)
    slc_array = np.asarray([[s.start, s.stop, s.step] for s in slc])
    # ^ dx3 (start, stop, step)
    slc_array[:, 0] = slc_array[:, 0] - margin  # Margin to lower boundaries
    slc_array[:, 1] = slc_array[:, 1] + margin  # Margin to upper boundaries
    slc_array[:, :2] = np.clip(slc_array[:, :2], 0, np.inf)  # Avoid negative indices
    return tuple(slice(*row) for row in slc_array)
