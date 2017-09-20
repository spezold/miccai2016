#!/usr/bin/env python
# coding: utf-8

"""
Miscellaneous helper functions
"""

from __future__ import division

import numpy as np
import os
import pyopencl as cl

from helpers.misc import human_readable_bytes


def cl_workgroup_size_3d_for(shape_3d, reverse=True, verbose=True):
    """
    For a 3D image volume with the given <shape_3d> calculate an OpenCL
    workgroup size such that the volume's dimension 2 is partitioned into the
    largest fitting power of two <= 64.
    
    Return the respective workgroup size as a three-tuple. If <reverse> is True
    (default), reverse the determined size's order, i.e. return the
    partitioning for volume dimension 2 in workgroup size position 0 etc.
    """
    lsize_3d = [1, 1, 1]
    optimal = True
        
    shape_2s = [64, 32, 16, 8, 4, 2]
    for shape_2 in shape_2s:
        if shape_2 == 16:
            optimal = False
        if not (shape_3d[2] % shape_2):
            lsize_3d[0 if reverse else 2] = shape_2
            break
    if not (shape_3d[1] % 2):
        lsize_3d[1] = 2
    lsize_3d = tuple(lsize_3d)
    if verbose:
        print "Working with" + (" suboptimal " if not optimal else " ") + \
              "OpenCL workgroup size: %s" % (lsize_3d, )
    return lsize_3d


def to_cl_image_8192(ctx, values, fill=0):
    """
    Convert the given Nx3 array <values> to an RGBA image with a data type
    that is compatible to the datatype of <values>, with min(len(values),
    8192) values along its axis 1, and the necessary number of values along
    axis 0, such that the image is filled putting values[0] into result[0,
    0], values[1] into result[0, 1], etc. Excess image values and all
    values of the A channel are filled with the given <fill> value
    (defaults to 0).
    
    Note that, for whatever reason, OpenCL image axis 0 maps to array axis
    1 and vice versa (cf. help(cl._cl.Image)) -- which should not make a
    difference if the result of this function is used transparently.
    
    Return the resulting <pyopencl._cl.Image> instance.
    """
    assert values.shape[1] == 3    
    num_values = len(values)
    assert num_values > 0

    # Determine image dimensions
    img_dim_1 = np.minimum(8192, num_values)
    img_dim_0 = (int(num_values - 1) >> 13) + 1  # 2 ** 13 = 8192
    
    # Create appropriately sized Numpy array, then fill it
    values_2d = np.asarray(np.ones((img_dim_1 * img_dim_0, 4),
                                   dtype=values.dtype) * fill,
                           dtype=values.dtype)
    values_2d[:num_values, :3] = values
    values_2d = values_2d.reshape(img_dim_0, img_dim_1, 4)
    values_2d = np.swapaxes(values_2d, 0, 1)
    values_2d = np.require(values_2d, requirements=["A", "C"])
    
    # Create <cl._cl.Image> instance
    values_cl = cl.image_from_array(ctx, values_2d, num_channels=4, mode="r")
    return values_cl


class ClProgramLoader(object):
    """
    Create an OpenCl context, load, and hold available, OpenCl programs from
    text files and strings. Inspired by [1].
    
    References
    [1] https://github.com/enjalot/adventures_in_opencl/blob/master/python/part1/main.py (20130726)
    """
    
    def __init__(self):
        
        self.ctx = self.__create_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.programs = {}  # The built programs
        
    def __create_context(self):
        """
        Create a context for the GPU device with most RAM.
        """
        devices = []
        for p in cl.get_platforms():
            devices.extend(p.get_devices(device_type=cl.device_type.GPU))
        key = lambda device : device.get_info(cl.device_info.GLOBAL_MEM_SIZE)
        try:
            device = max(devices, key=key)
            print "Using %s" % device.get_info(cl.device_info.NAME),
            print "with %s global device memory." % human_readable_bytes(key(device))
            ctx = cl.Context(devices=[device])
            return ctx
        except ValueError:
            raise RuntimeError("No OpenCL GPU device found!")
        
    def program_from_file(self, program_path, program_id=None, defs=None, includes=None):
        """
        Create and build a <pyopencl.Program> from the file content of the
        given <program_path>.
        
        Hold the result available as <self.programs[program_id]>. If
        <program_id> is None, create an id from the <program_path>'s file name
        (use the part before the first "." in the file name). If <program_id>
        is False, don't hold available the created <pyopencl.Program> instance,
        just return it.
        
        Via <defs> a dictionary of macros may be given. They will be added to
        the program build options and are equivalent to
        
            #define <key> <value>
            
        for each (key, value) tuple in <defs> if <value> is not None, or else
        
            #define <key>.
            
        If <defs> is an ordered dictionary, its order will be maintained.
        
        Via <includes>, a list of folders may be given that shall be searched
        for header files (order will be maintained).
        
        Return the created <pyopencl.Program> instance for convenience.
        """
        
        with open(program_path, "r") as f:
            program_string = "".join(f.readlines())
            
        if program_id is None:
            program_id = os.path.basename(program_path).split(".", 1)[0]
            
        return self.program_from_string(program_string, program_id, defs, includes)
    
    def program_from_string(self, program_string, program_id, defs, includes):
        """
        See <program_from_file()>, but expects the actual source code as
        <program_string> (rather than the path to the source code file).
        """
        
        options = []
        
        if defs is not None:
            macros = []
            for (k, v) in defs.items():
                macro_string = "-D '%s'='(%s)'" % (k, v) if v is not None else "-D '%s'" % (k, )
                macros.append(macro_string)
            options += macros
            
        if includes is not None:
            include_strings = ["-I '%s'" % i for i in includes]
            options += include_strings
        
        # Workaround for some weird bug: We need a valid current working
        # directory before calling cl.Program().build(), but for whatever
        # reason, this is not always the case
        try:        
            os.getcwd()
        except OSError:
            print ("Warning: Current working directory is invalid, " +
                   "trying to change it to user's home")
            os.chdir(os.path.expanduser("~"))
        program = cl.Program(self.ctx, program_string).build(options=options)
        if program_id is not False:
            self.programs[program_id] = program
        return program



class Reducer(object):
    """
    Reduces an array in as many passes as necessary.
    """
    
    def __init__(self, prg_gpu, prg_cpu, wgsize, queue):
        """
        <prg_gpu>
            The program used for reduction on the GPU (<pyopencl.Program>
            instance). The following signature is assumed
            - input data (global buffer)
            - output data (global buffer)
            - input size (number of elements in the input data buffer)
            Identical behavior to the sum_abs kernel is assumed; see there.
        <prg_cpu>
            function that reduces the last remaining elements on the CPU.
            Should take an array as input and return a scalar
        <wgsize>
            the work group size that the program has been initialized with
            (scalar)
        <queue>
            OpenCl command queue instance to be used
        """
        self._prg_gpu = prg_gpu
        self._prg_cpu = prg_cpu
        self._WGSIZE = wgsize
        self._local_size = (wgsize, )
        self._queue = queue
        
    def swap_buffer_bytes_for(self, n_in, dtype):
        """
        Return the minimum size (in bytes) required for the <swap_buffer> in
        the <reduce()> method, given the number of input elements <n_in> of the
        given (Numpy) data type <dtype>.
        """
        result = self._calc_outputsize(n_in) * dtype().nbytes
        return result
    
    def _calc_outputsize(self, n_in):
        """
        Return resulting number of elements in the output buffer, given the
        number of elements in the input buffer <n_in>.
        """
        return int(np.ceil(np.ceil(n_in / 2.) / self._WGSIZE))
    
    def reduce(self, in_buffer, in_number, swap_buffer, dtype=np.float32,
               preserve_in_buffer_contents=True):
        """
        Actually reduce an array/buffer.
        
        <in_buffer>
            The <pyopencl.Buffer> instance whose content is to be reduced.
            Caution: Its contents may be altered (see the
            <preserve_in_buffer_contents> option).
        <in_number>
            The number of elements in <in_Buffer>.
        <swap_buffer>
            A <pyopencl.Buffer> instance that is used for reading and writing
            intermediate results. Its necessary size can be queried via the
            <swap_buffer_bytes_for()> method.
        <dtype>
            The data type of <in_buffer>'s content.
        <preserve_in_buffer_contents>
            If True (default), make sure that on return, the original input of
            <in_buffer> is restored. As this step involves a host-device and a
            device-host copy of some data (albeit usually a small amount), it
            should be ommitted if <in_buffer>'s original content is not needed
            after reduction.
            
        Return
            The reduction result as a scalar.
        """
        input_size = int(in_number)
        input_buffer = in_buffer
        output_buffer = swap_buffer
        
        assert input_size > self._WGSIZE
        
        if preserve_in_buffer_contents:
            
            # We need to secure the first N elements of <in_buffer>. N is the
            # number of elements that remain after the second reduction pass,
            # i.e. when <swap_buffer> serves as input and <in_buffer> serves as
            # output
            num_preserved = self._calc_outputsize(self._calc_outputsize(input_size))
            data_preserved = np.empty(num_preserved, dtype=dtype)
            cl.enqueue_copy(self._queue, data_preserved, in_buffer)
        
        while input_size > self._WGSIZE:
            
            output_size = self._calc_outputsize(input_size)
            global_size = output_size * self._WGSIZE

            # Reduce current input, swap buffers, update input/output_size
            self._prg_gpu(self._queue, (global_size, ), self._local_size,
                          input_buffer, output_buffer, input_size)
            
            input_buffer, output_buffer = output_buffer, input_buffer
            input_size = output_size
        
        # Read back the remaining values to the CPU (we have to read from
        # <input_buffer>, as we swapped the buffers after the last reduction),
        # then perform the final reduction there
        gpu_result = np.empty(input_size, dtype=dtype)
        cl.enqueue_copy(self._queue, gpu_result, input_buffer)
        
        if preserve_in_buffer_contents:
            
            # Restore <in_buffer>'s first N elements
            cl.enqueue_copy(self._queue, in_buffer, data_preserved)
        
        result = self._prg_cpu(gpu_result)
        return result
