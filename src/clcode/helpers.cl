/*
Helper kernels.

Note that the kernel functionality for initialization (i.e. the <to_*()>
kernel functionality) can more easily be achieved starting from OpenCL 1.2,
using <clEnqueueFillBuffer()>.
*/


/*
Sets all array elements to zero.
*/
__kernel void to_zeros(__global float *a)
{
    a[get_global_id(0)] = 0.0f;
}


/*
Sets all array elements to one.
*/
__kernel void to_ones(__global float *a)
{
    a[get_global_id(0)] = 1.0f;
}


/*
Sets all array elements to the given value.
*/
__kernel void to_value(__global float *a, const float val)
{
    a[get_global_id(0)] = val;
}


/*
Set all array elements in <a> to their absolute value, write result to <out>.

The <a> and <out> buffers may be the same or different buffers.
*/
__kernel void el_abs(__global const float *a,
                     __global float *out)
{
    const size_t i = get_global_id(0);
    out[i] = fabs(a[i]);
}


/*
Write element-wise sum of <a> and <b> to <out>.

Any combination of <a>, <b>, and <out> may refer to the same or different buffers.
*/
__kernel void el_add(__global const float *a,
                     __global const float *b,
                     __global float *out)
{
    const size_t i = get_global_id(0);
    out[i] = a[i] + b[i];
}


/*
Write element-wise minimum of <a> and <b> to <out>.

Any combination of <a>, <b>, and <out> may refer to the same or different buffers.
*/
__kernel void el_min(__global const float *a,
                     __global const float *b,
                     __global float *out)
{
    const size_t i = get_global_id(0);
    out[i] = fmin(a[i], b[i]);
}


/*
Write element-wise maximum of <a> and <b> to <out>.

Any combination of <a>, <b>, and <out> may refer to the same or different buffers.
*/
__kernel void el_max(__global const float *a,
                     __global const float *b,
                     __global float *out)
{
    const size_t i = get_global_id(0);
    out[i] = fmax(a[i], b[i]);
}
