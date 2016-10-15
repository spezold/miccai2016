#ifndef _QUATERNION_MATH_H
#define _QUATERNION_MATH_H

#include "basedefs.h"

#define safeatan2(x, y) ((x) == (ftype)0 && (y) == (ftype)0 ? (ftype)0 : atan2((ftype)(x), (ftype)(y)))
#define rsqroot(x) native_rsqrt(x)
#define EPS FLT_EPSILON

/**
 * Squeeze the given unit quaternion to an equivalent vector by calculating
 * the respective Euler angles (z-x-z extrinsic) [1].
 *
 * References
 * [1] https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions (20150617)
 *
 * @param[in]  inQ the unit quaternion to be squeezed
 * @param[out] outV the squeezed result
 */
void qsqueeze(const qat inQ, vec outV)
{
    const ftype q1spq2s = square(inQ[1]) + square(inQ[2]);

    bool cond1 = q1spq2s < EPS;
    bool cond2 = cond1 || (1 - q1spq2s) < EPS;

    const ftype q1q3 = inQ[1] * inQ[3];
    const ftype q2q0 = inQ[2] * inQ[0];
    const ftype q1q0 = inQ[1] * inQ[0];
    const ftype q2q3 = inQ[2] * inQ[3];

    outV[0] /* phi */   = safeatan2((q1q3 + q2q0), (q1q0 - q2q3));
    outV[1] /* theta */ = (cond1 ? 0 : M_PI) * cond2 + acos(clamp((1 - 2 * q1spq2s), (ftype)-1, (ftype)1)) * (!cond2);
    
    ftype at2 = (inQ[1] * inQ[2] - inQ[0] * inQ[3]) * cond2 + (q1q3 - q2q0) * (!cond2);
    at2 = safeatan2(at2, (.5 - (square(inQ[2]) + square(inQ[3]))) * cond2 + (q1q0 + q2q3) * (!cond2));
    
    outV[2] /* psi */   = (at2 - outV[0]) * (cond1 ? 1 : -1) * cond2 + at2 * (!cond2);
}

/**
 * Same as qsqueeze(), but assumes that the real/scalar part is the last
 * element of the given quaternion.
 */
void qsqueezeLast(const qat inQ, vec outV)
{
    const ftype q1 = inQ[0];  // We create these for better readability and
    const ftype q2 = inQ[1];  // hope for an intelligent compiler optimizing
    const ftype q3 = inQ[2];  // them away ...
    const ftype q0 = inQ[3];

    const ftype q1spq2s = square(q1) + square(q2);

    bool cond1 = q1spq2s < EPS;
    bool cond2 = cond1 || (1 - q1spq2s) < EPS;

    const ftype q1q3 = q1 * q3;
    const ftype q2q0 = q2 * q0;
    const ftype q1q0 = q1 * q0;
    const ftype q2q3 = q2 * q3;

    outV[0] /* phi */   = safeatan2((q1q3 + q2q0), (q1q0 - q2q3));
    outV[1] /* theta */ = (cond1 ? 0 : M_PI) * cond2 + acos(clamp((1 - 2 * q1spq2s), (ftype)-1, (ftype)1)) * (!cond2);
    
    ftype at2 = (q1 * q2 - q0 * q3) * cond2 + (q1q3 - q2q0) * (!cond2);
    at2 = safeatan2(at2, (.5 - (square(q2) + square(q3))) * cond2 + (q1q0 + q2q3) * (!cond2));
    
    outV[2] /* psi */   = (at2 - outV[0]) * (cond1 ? 1 : -1) * cond2 + at2 * (!cond2);
}

/**
 * The inverse operation of qsqueeze(), except for a possible sign change
 * (which can be ignored, as each unit quaternion q represents the same
 * rotation as -q).
 *
 * @param[in]  inV unit quaternion that was squeezed via qsqueeze()
 * @param[out] outQ the reconstructed quaternion
 */
void qexpand(const vec inV, qat outQ)
{
    const ftype phi_p_psi_h = (inV[0] + inV[2]) * 0.5;
    const ftype phi_m_psi_h = (inV[0] - inV[2]) * 0.5;
    const ftype cos_theta_h = cos(inV[1] * 0.5);
    const ftype sin_theta_h = sin(inV[1] * 0.5);

    outQ[0] = cos(phi_p_psi_h) * cos_theta_h;
    outQ[1] = cos(phi_m_psi_h) * sin_theta_h;
    outQ[2] = sin(phi_m_psi_h) * sin_theta_h;
    outQ[3] = sin(phi_p_psi_h) * cos_theta_h;
}

/**
 * Multiply the two given unit quaternions i.e. calculate the Hamilton product
 * between inL and inR.
 * @param[in]  inL left factor of the product, may *not* be the same as out
 * @param[in]  inR right factor of the product, may *not* be the same as out
 * @param[out] out may *not* be the same as inL nor inR
 */
void qmult(const qat inL, const qat inR, qat out)
{
    out[0] = inL[0]*inR[0] - (inL[1]*inR[1] + inL[2]*inR[2] + inL[3]*inR[3]);
    out[1] = inL[0]*inR[1] +  inL[1]*inR[0] + inL[2]*inR[3] - inL[3]*inR[2];
    out[2] = inL[0]*inR[2] -  inL[1]*inR[3] + inL[2]*inR[0] + inL[3]*inR[1];
    out[3] = inL[0]*inR[3] +  inL[1]*inR[2] - inL[2]*inR[1] + inL[3]*inR[0];
}

/**
 * Calculate the Hamilton product between the given unit quaternion and vector,
 * interpreting the vector as a unit quaternion with scalar/real part == 0.
 * @param[in]  inQ left factor of the product, may *not* be the same as out
 * @param[in]  inv right factor of the product
 * @param[out] out may *not* be the same as inQ
 */
void qmult_vector(const qat inQ, const vec inV, qat outQ)
{
    outQ[0] = -(inQ[1]*inV[0] + inQ[2]*inV[1] + inQ[3]*inV[2]);
    outQ[1] =   inQ[0]*inV[0] + inQ[2]*inV[2] - inQ[3]*inV[1];
    outQ[2] =   inQ[0]*inV[1] - inQ[1]*inV[2] + inQ[3]*inV[0];
    outQ[3] =   inQ[0]*inV[2] + inQ[1]*inV[1] - inQ[2]*inV[0];
}

/**
 * Rotate the given vector by the given unit quaternion that represents the
 * desired rotation.
 * @param[in]  inQ  desired rotation
 * @param[in]  inV  may be the same as outV
 * @param[out] outV may be the same as inV
 */
void qrot(const qat inQ, const vec inV, vec outV)
{
    // Calculate result = q v inv(q) in two steps

    // Step 1: tmp = q v

    qat tmpQ;
    qmult_vector(inQ, inV, tmpQ);

    // Step 2: result = tmp inv(q)

    // Same as qmult(), but (1) don't calculate scalar/real part of the
    // resulting quaternion (must be 0 anyway), (2) multiply with the inverse
    // of the right quaternion. Inversion means: Swap the sign of all elements
    // of inQ's vector/imaginary part (i.e. inQ[1], inQ[2], inQ[3])
    outV[0] = -tmpQ[0]*inQ[1] + tmpQ[1]*inQ[0] - tmpQ[2]*inQ[3] + tmpQ[3]*inQ[2];
    outV[1] = -tmpQ[0]*inQ[2] + tmpQ[1]*inQ[3] + tmpQ[2]*inQ[0] - tmpQ[3]*inQ[1];
    outV[2] = -tmpQ[0]*inQ[3] - tmpQ[1]*inQ[2] + tmpQ[2]*inQ[1] + tmpQ[3]*inQ[0];
}

/**
 * Rotate the given matrix column-wise by the given unit quaternion that
 * represents the desired rotation.
 * @param[in]  inQ  desired rotation
 * @param[in]  inM  may be the same as outM
 * @param[out] outM may be the same as inM
 */
void qmatrot(const qat inQ, const mat inM, mat outM)
{
    vec tmpV;
    for(int i=0; i<3; ++i)
    {
        tmpV[0] = inM[0][i];
        tmpV[1] = inM[1][i];
        tmpV[2] = inM[2][i];
        qrot(inQ, tmpV, tmpV);
        outM[0][i] = tmpV[0];
        outM[1][i] = tmpV[1];
        outM[2][i] = tmpV[2];
    }
}

/**
 * Invert quaternion, not to be confused with transpose (see qtrans()).
 * @param[in]  in  may be the same as out
 * @param[out] out may be the same as in
 */
void qinv(const qat in, qat out)
{
    out[0] =  in[0];
    out[1] = -in[1];
    out[2] = -in[2];
    out[3] = -in[3];
}

/**
 * Transpose quaternion, not to be confused with invert (see qinv()).
 * @param[in]  in  may be the same as out
 * @param[out] out may be the same as in
 */
void qtrans(const qat in, qat out)
{
    out[0] = -in[0];
    out[1] =  in[1];
    out[2] =  in[2];
    out[3] =  in[3];
}

/**
 * Normalize the given quaternion to unit length.
 * @param[in]  in  may be the same as out
 * @param[out] out may be the same as in
 */
void qnormalize(const qat in, qat out)
{
    const ftype rnrm = rsqroot(square(in[0]) + square(in[1]) + square(in[2]) + square(in[3]));

    out[0] = in[0] * rnrm;
    out[1] = in[1] * rnrm;
    out[2] = in[2] * rnrm;
    out[3] = in[3] * rnrm;
}

/**
 * Convert the given unit quaternion to a matrix that represents the same
 * rotation.
 *
 * The implementation generatens the matrix given by [*] under the headline
 * "Equations".
 *
 * References
 * [*] http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm (20150421)
 */
void quat2mat(const qat inQ, mat outM)
{
    // inQuat == {qw, qx, qy, qz}
    const ftype qx2 = square(inQ[1]);
    const ftype qy2 = square(inQ[2]);
    const ftype qz2 = square(inQ[3]);

    const ftype qwqx = inQ[0] * inQ[1];
    const ftype qwqy = inQ[0] * inQ[2];
    const ftype qwqz = inQ[0] * inQ[3];
    const ftype qxqy = inQ[1] * inQ[2];
    const ftype qxqz = inQ[1] * inQ[3];
    const ftype qyqz = inQ[2] * inQ[3];

    // Column 0
    outM[0][0] = 1 - 2 * (qy2 + qz2);
    outM[1][0] = 2 * (qxqy + qwqz);
    outM[2][0] = 2 * (qxqz - qwqy);

    // Column 1
    outM[0][1] = 2 * (qxqy - qwqz);
    outM[1][1] = 1 - 2 * (qx2 + qz2);
    outM[2][1] = 2 * (qyqz + qwqx);

    // Column 2
    outM[0][2] = 2 * (qxqz + qwqy);
    outM[1][2] = 2 * (qyqz - qwqx);
    outM[2][2] = 1 - 2 * (qx2 + qy2);
}

/**
Same as <quat2mat()> function, but calculating the columns of the
resulting matrix in float3 instances. Any of the resulting columns might be
passed a null pointer in case its value is not needed.
*/
void quat2matFloat3(const qat inQ, float3 *oCol0, float3 *oCol1, float3 *oCol2)
{
    // inQuat == {qw, qx, qy, qz}
    const ftype qx2 = square(inQ[1]);
    const ftype qy2 = square(inQ[2]);
    const ftype qz2 = square(inQ[3]);

    const ftype qwqx = inQ[0] * inQ[1];
    const ftype qwqy = inQ[0] * inQ[2];
    const ftype qwqz = inQ[0] * inQ[3];
    const ftype qxqy = inQ[1] * inQ[2];
    const ftype qxqz = inQ[1] * inQ[3];
    const ftype qyqz = inQ[2] * inQ[3];

    if (oCol0)
    {
        oCol0->s0 = 1 - 2 * (qy2 + qz2);
        oCol0->s1 = 2 * (qxqy + qwqz);
        oCol0->s2 = 2 * (qxqz - qwqy);
    }
    if (oCol1)
    {
        oCol1->s0 = 2 * (qxqy - qwqz);
        oCol1->s1 = 1 - 2 * (qx2 + qz2);
        oCol1->s2 = 2 * (qyqz + qwqx);
    }
    if (oCol2)
    {
        oCol2->s0 = 2 * (qxqz + qwqy);
        oCol2->s1 = 2 * (qyqz - qwqx);
        oCol2->s2 = 1 - 2 * (qx2 + qy2);
    }
}

/**
 * Convert the given rotation matrix to a unit quaternion that represents the
 * same rotation.
 *
 * The implementation follows [*], using the stable version given under the
 * headline "C++ Code (kindly sent to me by Angel)".
 *
 * References
 * [*] http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm (20150421)
 */
void mat2quat(const mat inM, qat outQ)
{
    // We use s to hold the trace until it is overwritten
    ftype s = inM[0][0] + inM[1][1] + inM[2][2];

    if (s > 0)  // if trace > 0 ...
    {
        s = rsqroot(s + 1.0) * 0.5;
        // ^ s = 1 / (2 * sqrt(trace + 1)) <=> s = 1 / (4 * qw)
        // (last place where we needed the trace)
        outQ[0] = 0.25 / s;
        outQ[1] = (inM[2][1] - inM[1][2]) * s;
        outQ[2] = (inM[0][2] - inM[2][0]) * s;
        outQ[3] = (inM[1][0] - inM[0][1]) * s;
    }
    else if (inM[0][0] > inM[1][1] && inM[0][0] > inM[2][2])
    {
        s = rsqroot(1.0 + inM[0][0] - inM[1][1] - inM[2][2]) * 0.5;  // s = 1 / (4 * qx)
        outQ[0] = (inM[2][1] - inM[1][2]) * s;
        outQ[1] = 0.25 / s;
        outQ[2] = (inM[0][1] + inM[1][0]) * s;
        outQ[3] = (inM[0][2] + inM[2][0]) * s;
    }
    else if (inM[1][1] > inM[2][2])
    {
        s = rsqroot(1.0 + inM[1][1] - inM[0][0] - inM[2][2]) * 0.5;  // s = 1 / (4 * qy)
        outQ[0] = (inM[0][2] - inM[2][0]) * s;
        outQ[1] = (inM[0][1] + inM[1][0]) * s;
        outQ[2] = 0.25 / s;
        outQ[3] = (inM[1][2] + inM[2][1]) * s;
    }
    else
    {
        s = rsqroot(1.0 + inM[2][2] - inM[0][0] - inM[1][1]) * 0.5;  // s = 1 / (4 * qz)
        outQ[0] = (inM[1][0] - inM[0][1]) * s;
        outQ[1] = (inM[0][2] + inM[2][0]) * s;
        outQ[2] = (inM[1][2] + inM[2][1]) * s;
        outQ[3] = 0.25 / s;
    }
}

#endif  // _QUATERNION_MATH_H

