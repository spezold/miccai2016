#include "basedefs.h"
#include "quaternion_math.h"

/*
Updates the flow field p. Implements Equation (5.3) of [1].

Variable names:
    alpha: \alpha(x) (penalty for total-variation term)
    squeezed: the rotational part or the anisotropy matrix ("squeezed"
        quaternion, "eigenvectors in columns")
    sqeigs: the square roots of the non-negative eigenvalues of the anisotropy
        matrix
    eigs:
        (1) If CONST_EVALS is defined: Array with 3 elements along its last
        dimension, holding <squeezed> (see above); (2) If CONST_EVALS is not
        defined: Array with 6 elements along its last dimension: first 3
        values: <sqeigs>, last 3 values: <squeezed> (see above)
    hp0: H p(x) (transformed flow field, component of dimension/axis 0)
    hp1: H p(x) (transformed flow field, component of dimension/axis 1)
    hp2: H p(x) (transformed flow field, component of dimension/axis 2)
    ps: p_s(x)
    pt: p_t(x)
    tmp: intermediate results
    steps: c (Equation (5.3); step size for the projection step)
    u: \lambda(x)
    labels: foreground/background seeds for the supervised version of the cut;
        foreground seeds are labeled with 1, background seeds are labeled with
        2, unlabeled voxels are 0

References
[1] J. Yuan, E. Bae, X.-C. Tai, and Y. Boykov, “A study on continuous max-flow
    and min-cut approaches,” UCLA, CAM, UCLA, technical report CAM 10-61, 2010.
*/

#define sqroot(x) native_sqrt(x)

// #define RS0 (1.)  // reciprocal of the voxel scaling along dimension 0 [voxel/mm]
// #define RS1 (1.)  // reciprocal of the voxel scaling along dimension 1 [voxel/mm]
// #define RS2 (1.)  // reciprocal of the voxel scaling along dimension 2 [voxel/mm]
// #define SS0 (1.)  // steps / s0 (s0 = voxel scaling along dimension 0 [mm/voxel])
// #define SS1 (1.)  // steps / s1 (s1 = voxel scaling along dimension 1 [mm/voxel])
// #define SS2 (1.)  // steps / s2 (s2 = voxel scaling along dimension 2 [mm/voxel])
// #define RCC (.2)  // reciprocal of the step size for augmented Lagrangian method)

// #define CONST_EVALS  // Use the same eigenvalues for all points? If so ...
// #define SQEVAL0 (1.)   // ... then this is the square root of eigenvalue 0, ...
// #define SQEVAL1 (1.)   // ... the square root of eigenvalue 1, ...
// #define SQEVAL2 (1.)   // ... and the square root of eigenvalue 2

// #define SUPERVISED  // Use the supervised version of the cut?


/**
Suppose the anisotropy matrix A is decomposed as H H.T, calculate the matrix
product between H and inOther.

In practice, use the rotational part inRot and the square roots of the
eigenvalues inSqeigs to achieve the multiplication.

@param[in]  inRot the quaternion that represents the rotational part of A (i.e.
            "A's eigenvectors in columns")
@param[in]  inSqeigs the square roots of A's eigenvalues, may be the same as
            outResult
@param[in]  inOther the vector to be multiplied, may be the same as outResult
@param[out] outResult may be the same as inOther and inSqeigs
*/
#ifdef CONST_EVALS
void h_dot(const qat inRot, const vec inOther, vec outResult)
{
    outResult[0] = SQEVAL0 * inOther[0];
    outResult[1] = SQEVAL1 * inOther[1];
    outResult[2] = SQEVAL2 * inOther[2];
    qrot(inRot, outResult, outResult);
}
#else
void h_dot(const qat inRot, const vec inSqeigs, const vec inOther, vec outResult)
{
    outResult[0] = inSqeigs[0] * inOther[0];
    outResult[1] = inSqeigs[1] * inOther[1];
    outResult[2] = inSqeigs[2] * inOther[2];
    qrot(inRot, outResult, outResult);
}
#endif  // CONST_EVALS

/**
Same as h_dot(), but calculate the matrix product between H.T and inOther,
using inRot and inSqeigs. The inRot quaternion is altered during function
execution, but is guaranteed to again hold its original values on return.
*/
#ifdef CONST_EVALS
void h_t_dot(qat inRot, const vec inOther, vec outResult)
{
    inRot[0] = -inRot[0];  // We need the transpose here ...
    qrot(inRot, inOther, outResult);
    inRot[0] = -inRot[0];  // ... but also want to keep inRot unaltered later
    outResult[0] *= SQEVAL0;
    outResult[1] *= SQEVAL1;
    outResult[2] *= SQEVAL2;
}
#else
void h_t_dot(qat inRot, const vec inSqeigs, const vec inOther, vec outResult)
{
    inRot[0] = -inRot[0];  // We need the transpose here ...
    qrot(inRot, inOther, outResult);
    inRot[0] = -inRot[0];  // ... but also want to keep inRot unaltered later
    outResult[0] *= inSqeigs[0];
    outResult[1] *= inSqeigs[1];
    outResult[2] *= inSqeigs[2];
}
#endif  // CONST_EVALS

/**
Same as h_dot(), but calculate the matrix product between H^-1 and inOther,
using inRot and inSqeigs. The inRot quatenion is altered during function
execution, but is guaranteed to again hold its original values on return.
*/
#ifdef CONST_EVALS
void h_inv_dot(qat inRot, const vec inOther, vec outResult)
{
    inRot[0] = -inRot[0];  // We need the transpose here ...
    qrot(inRot, inOther, outResult);
    inRot[0] = -inRot[0];  // ... but also want to keep inRot unaltered later
    outResult[0] /= SQEVAL0;
    outResult[1] /= SQEVAL1;
    outResult[2] /= SQEVAL2;
}
#else
void h_inv_dot(qat inRot, const vec inSqeigs, const vec inOther, vec outResult)
{
    inRot[0] = -inRot[0];  // We need the transpose here ...
    qrot(inRot, inOther, outResult);
    inRot[0] = -inRot[0];  // ... but also want to keep inRot unaltered later
    outResult[0] /= inSqeigs[0];
    outResult[1] /= inSqeigs[1];
    outResult[2] /= inSqeigs[2];
}
#endif  // CONST_EVALS

/*
Calculates

    p <- H^-1 hp(x) + c * H.T grad(tmp)  with  tmp <- div(H p) - p_s + p_t - u / cc,
    
(c == steps), then project the vectors of p to the sphere with radius \alpha,
and recalculate hp <- H p.

Assumes 3D arrays of identical size. Assumes to be called with the reversed
image shape.
*/
__kernel void update_p(__global ftype *hp0,
                       __global ftype *hp1,
                       __global ftype *hp2,
                       __global const ftype *tmp,
                       __global const ftype *eigs,
                       __global const ftype *alpha)
{
    const size_t i0 = get_global_id(2);
    const size_t i1 = get_global_id(1);
    const size_t i2 = get_global_id(0);
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);

    
    // Calculate the current index

    const itype i = i2 + dim2 * (i1 + dim1 * i0);
    
    #ifdef CONST_EVALS
    const itype i_eigs = i * 3;  // index for the <eigs> buffer
    #else
    const itype i_eigs = i * 6;
    #endif  // CONST_EVALS


    // c * grad(tmp) (c is part of SS0/SS1/SS2)

    vec g_i = {i0 < dim0 - 1 ? (tmp[i + dim2 * dim1] - tmp[i]) * SS0 : 0.f,
               i1 < dim1 - 1 ? (tmp[i + dim2       ] - tmp[i]) * SS1 : 0.f,
               i2 < dim2 - 1 ? (tmp[i + 1          ] - tmp[i]) * SS2 : 0.f};
 
     
    // c * H.T grad(tmp) == H.T c * grad(tmp)

    #ifdef CONST_EVALS
    vec eigs_i = {eigs[i_eigs], eigs[i_eigs + 1], eigs[i_eigs + 2]};
    // ^ squeezed quaternion
    qat quat_i;
    qexpand(eigs_i, quat_i);
    h_t_dot(quat_i, g_i, g_i);
    #else
    vec eigs_i = {eigs[i_eigs + 3], eigs[i_eigs + 4], eigs[i_eigs + 5]};
    // ^ squeezed quaternion
    qat quat_i;
    qexpand(eigs_i, quat_i);
    eigs_i[0] = eigs[i_eigs]; eigs_i[1] = eigs[i_eigs + 1]; eigs_i[2] = eigs[i_eigs + 2];
    // ^ square roots of the eigenvalues
    h_t_dot(quat_i, eigs_i, g_i, g_i);
    #endif  // CONST_EVALS
    
    // p + c * H.T grad(tmp)
    vec p_i = {hp0[i], hp1[i], hp2[i]};
    #ifdef CONST_EVALS
    h_inv_dot(quat_i, p_i, p_i);
    #else
    h_inv_dot(quat_i, eigs_i, p_i, p_i);
    #endif  // CONST_EVALS
    
    p_i[0] += g_i[0]; p_i[1] += g_i[1]; p_i[2] += g_i[2];

    
    // Project p, recalulate H p, and write back results

    const ftype norm = sqroot(square(p_i[0]) + square(p_i[1]) + square(p_i[2]));
    const ftype radius = alpha[i];
    const ftype scale = norm > radius ? radius / norm : 1.f;
    
    p_i[0] *= scale;
    p_i[1] *= scale;
    p_i[2] *= scale;
    
    #ifdef CONST_EVALS
    h_dot(quat_i, p_i, p_i);
    #else
    h_dot(quat_i, eigs_i, p_i, p_i);
    #endif  // CONST_EVALS
    
    hp0[i] = p_i[0];
    hp1[i] = p_i[1];
    hp2[i] = p_i[2];
}

/**
Updates ps, pt, and finally u, then calculate tmp as div(H p) - ps + pt, which
is needed for evaluating the convergence criterion.

Assumes 3D arrays of identical size. Assumes to be called with the reversed
image shape.
*/
__kernel void update_u(__global const ftype *hp0,
                       __global const ftype *hp1,
                       __global const ftype *hp2,
                       __global ftype *tmp,
                       __global const ftype *cs,
                       __global const ftype *ct,
                       __global ftype *pt,
                       __global ftype *u
                       #ifdef SUPERVISED
                       , __global const uchar *labels
                       #endif  // SUPERVISED
                       )
{
    const size_t i0 = get_global_id(2);
    const size_t i1 = get_global_id(1);
    const size_t i2 = get_global_id(0);
    
    const size_t dim0 = get_global_size(2);
    const size_t dim1 = get_global_size(1);
    const size_t dim2 = get_global_size(0);

    
    // Calculate the current index

    const itype i = i2 + dim2 * (i1 + dim1 * i0);
    
    // Calculate div(H p)
    
    const ftype divhp_i = \
    (((i0 < dim0 - 1 ? hp0[i] : 0.f) - (i0 > 0 ? hp0[i - dim2 * dim1] : 0.f)) * RS0 +
     ((i1 < dim1 - 1 ? hp1[i] : 0.f) - (i1 > 0 ? hp1[i - dim2       ] : 0.f)) * RS1 +
     ((i2 < dim2 - 1 ? hp2[i] : 0.f) - (i2 > 0 ? hp2[i - 1          ] : 0.f)) * RS2);
     
    // Update ps and pt (already making use of the new ps for pt)
    
    ftype u_i = u[i];
    
    #ifdef SUPERVISED
    const uchar uf = labels[i] & 1;
    const uchar ub = !(labels[i] & 2);
    // ^ Logical "not" because ub labels the values that do NOT belong to the
    // background (see Yuan's paper, eq. (26))
    const ftype ps_i = min((ftype)( divhp_i + pt[i] - RCC * (u_i - ub)), (ftype)cs[i]);
    const ftype pt_i = min((ftype)(-divhp_i + ps_i  + RCC * (u_i - uf)), (ftype)ct[i]);
    #else
    const ftype ps_i = min((ftype)( divhp_i + pt[i] - RCC * (u_i - 1)),  (ftype)cs[i]);
    const ftype pt_i = min((ftype)(-divhp_i + ps_i  + RCC *  u_i),       (ftype)ct[i]);
    #endif  // SUPERVISED
    pt[i] = pt_i;
    
    
    // Update the error term, update u, finally update tmp
    
    const ftype error_i = divhp_i - ps_i + pt_i;

    u_i -= error_i / RCC;
    
    u[i] = u_i;
    tmp[i] = error_i;
}

/**
Updates tmp as needed for the update_p kernel, i.e. calculate

    tmp <- tmp - u / c,
    
assuming that tmp had beed calculated as div(H p) - ps + pt before.

Assumes 1D arrays of identical size.
*/
__kernel void update_tmp(__global ftype *tmp,
                         __global const ftype *u)
{
    const size_t i = get_global_id(0);
    tmp[i] = tmp[i] - u[i] * RCC;
}
