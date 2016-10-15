#!/usr/bin/env python
# coding: utf-8

"""
Miscellaneous helper functions
"""

from __future__ import division

import numpy as np


def human_readable_bytes(byte_count, si=True, decimals=1):
    """
    Return a string that represents a <byte_count> in human readable form,
    either as a multiple of 1000 (if <si> is True; default) or as a multiple of
    1024 (if <si> is False).
    
    Print the given number of <decimals> for each value that is greater than
    1000 or 1024, respectively.
    
    Reimplementation of [1]. Should work until yottabytes/yobibytes. For even
    greater values, return "<byte_count> B" in scientific notation for
    convenience.
    
    References
    [1] http://stackoverflow.com/questions/3758606/ (20130903)
    """
    unit = 1000 if si else 1024
    if byte_count < unit:
        return "%i B" % byte_count
    exponent = int(np.log(byte_count) / np.log(unit))
    try:
        prefix = (("k" if si else "K") + "MGTPEZY")[exponent-1]
        prefix = prefix if si else prefix + "i"
        byte_formatted = byte_count / unit ** exponent
        return ("%." + str(decimals) + "f %sB") % (byte_formatted, prefix)
    except IndexError:
        # Beyond "Y"
        return ("%." + str(decimals) + "e B") % byte_count


def dump(obj, path, use_cpickle=True):
    """
    Dump the object <obj> to the given <path>, using the <pickle> or <cPickle>
    module.
    """
    if use_cpickle: import cPickle as pickle  # @UnusedImport
    else: import pickle  # @Reimport
    with open(path, "wb") as ofile: pickle.dump(obj, ofile, protocol=2)  # protocol=2 for huge speedup!

    
def load_dumped(path, use_cpickle=True):
    """
    Load and return an object that was dumped to the given <path>, using the
    <pickle> or <cPickle> module.
    """
    if use_cpickle: import cPickle as pickle  # @UnusedImport
    else: import pickle  # @Reimport
    with open(path, "rb") as ifile: return pickle.load(ifile)


def normalize(data, dst_min=0.0, dst_max=1.0, perc_low=None, perc_up=None, clip=True):
    """
    Normalize the given <data> (Numpy array expected) to the range
    [dst_min, dst_max].
    
    If <perc_low> is given (float in [0, 100] expected), it will be interpreted
    as a lower percentile value: Normalize the data, so that the smallest
    <perc_low> percent of values are mapped to <dst_min>. If <perc_low> is
    None, map the minimum value in <data> to <dst_min> instead.

    If <perc_up> is given (float in [0, 100] expected), it will be interpreted
    as an upper percentile value: Normalize the data, so that the largest
    <perc_up> percent of values are mapped to <dst_max>. If <perc_up> is None,
    map the maximum value in <data> to <dst_max> instead.
    
    If <clip> is True, ensure that after percentile normalization values that
    turn out smaller than <dst_min> are actually set to <dst_min> and values
    greater than <dst_max> are set to <dst_max>. If <clip> is False, leave
    these values unchanged (making them lie outside [dst_min, dst_max]).
    
    Return the normalized values in a new Numpy array. If the difference
    between the minimum and the maximum value (or the respective percentile
    values) evaluates to zero, return an array with of <data>'s shape with its
    values set to <dst_min>.
    """
    use_percentiles = perc_low is not None or perc_up is not None
    
    if use_percentiles:
        # Make exactly one call to <np.percentile()>, so that <data> is sorted
        # only once
        perc_low =   0. if perc_low is None else perc_low
        perc_up  = 100. if perc_up  is None else perc_up
        src_min, src_max = np.percentile(data, [perc_low, perc_up])
    else:
        src_min = np.min(data)
        src_max = np.max(data)
    
    if src_max - src_min == 0:
        return np.ones_like(data, dtype=np.float) * dst_min
    
    # First map to [0, 1], then to [dst_min, dst_max] interval
    result = (1. / (src_max - src_min)) * (data.astype(np.float) - src_min)
    if dst_max != 1 and dst_min == 0:
        result = result * dst_max
    elif dst_min != 0:
        result = result * (dst_max - dst_min) + dst_min

    # Chop off the values beyond the percentiles if desired and necessary
    if clip and use_percentiles:
        np.clip(result, dst_min, dst_max, out=result)

    return result
