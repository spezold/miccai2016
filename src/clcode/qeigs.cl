#include "basedefs.h"
#include "quaternion_math.h"

#define EPS    FLT_EPSILON
#define SQ2    (0.7071067811865475244)  // 1 / sqrt(2) 
#define GAMMA  (5.8284271247461900976)  // 3 + 2 * sqrt(2)
#define C_STAR (0.9238795325112867561)  // cos(pi / 8)
#define S_STAR (0.3826834323650897717)  // sin(pi / 8)

#ifndef MAXITER
    #define MAXITER 50
#endif
// ^ Maximum number of full Jacobi sweeps (50 should be way more than enough)

#ifndef eigcmp
    #define eigcmp(a, b) (a > b)
#endif
// ^ May be adjusted to determine the order of the eigenvalues and eigenvectors
// after calculating the eigendecomposition. Needs the condition that brings a
// to the front and b to the back. Default (see above): descending order. For
// no reordering, eigcmp() should always be true.

/**
 * Multiply the two given unit quaternions i.e. calculate the Hamilton product
 * between inL and inR.
 *
 * CAUTION: Real/scalar part is supposed to be the last element of the given
 * quaternions here and will also be returned as the last element in the
 * result.
 *
 * @param[in]  inL left factor of the product, may *not* be the same as out
 * @param[in]  inR right factor of the product, may *not* be the same as out
 * @param[out] out may *not* be the same as inL nor inR
 */
void cqmult(__constant qat inL, const qat inR, qat out)
{
    out[3] = inL[3]*inR[3] - (inL[0]*inR[0] + inL[1]*inR[1] + inL[2]*inR[2]);
    out[0] = inL[3]*inR[0] +  inL[0]*inR[3] + inL[1]*inR[2] - inL[2]*inR[1];
    out[1] = inL[3]*inR[1] -  inL[0]*inR[2] + inL[1]*inR[3] + inL[2]*inR[0];
    out[2] = inL[3]*inR[2] +  inL[0]*inR[1] - inL[1]*inR[0] + inL[2]*inR[3];
}

__constant qat SORTER_QUAT[6] = {{  0.,   0.,   0.,  1.  },   // order: 0, 1, 2
                                 {  0.,  -SQ2,  SQ2, 0.  },   // order: 0, 2, 1
                                 { -SQ2,  SQ2,  0.,  0.  },   // order: 1, 0, 2
                                 { -0.5, -0.5, -0.5, 0.5 },   // order: 1, 2, 0
                                 {  0.5,  0.5,  0.5, 0.5 },   // order: 2, 0, 1
                                 { -SQ2,  0.,   SQ2, 0.  }};  // order: 2, 1, 0
                                 
/**
 * Sort the eigenvalues and adjust the quaternion respectively. The sorting
 * order is determined by the eigcmp macro.
 *
 * CAUTION: Real/scalar part is supposed to be the last element of the given
 * quaternion here and will also be returned as the last element in the result.
 *
 * @param[in,out] ioQ the quaternion that corresponds to the eigenvectors
 * @param[in,out] ioV the eigenvalues to be sorted
 */
void sortEigs(qat ioQ, vec ioV)
{
    itype order[3];  // the newly determined sorting order, in terms of the old indices
    const qat inQ = {ioQ[0], ioQ[1], ioQ[2], ioQ[3]};

    itype oTmp;  // temporary element for order swapping
    ftype vTmp;  // the currently most extreme value

    bool cond;

    // Sort the eigenvalues, store the sorting order

    // Compare elements 0 and 1
    cond = eigcmp(ioV[0], ioV[1]);
    vTmp   = ioV[0];
    ioV[0] = cond ? ioV[0] : ioV[1];    order[0] = cond ? 0 : 1;
    ioV[1] = cond ? ioV[1] : vTmp;      order[1] = cond ? 1 : 0;
    // Compare elements 1 and 2
    cond = eigcmp(ioV[1], ioV[2]);
    vTmp   = ioV[1];                    oTmp     = order[1];
    ioV[1] = cond ? ioV[1] : ioV[2];    order[1] = cond ? order[1] : 2;
    ioV[2] = cond ? ioV[2] : vTmp;      order[2] = cond ? 2 : oTmp;
    // Compare elements 0 and 1 again
    cond = eigcmp(ioV[0], ioV[1]);
    vTmp   = ioV[0];                    oTmp     = order[0];
    ioV[0] = cond ? ioV[0] : ioV[1];    order[0] = cond ? order[0] : order[1];
    ioV[1] = cond ? ioV[1] : vTmp;      order[1] = cond ? order[1] : oTmp;

    // Adjust the quaternion
    cqmult(SORTER_QUAT[ (order[0] << 1) | (order[1] > order[2]) ], inQ, ioQ);
    // ^ The bit operation maps the order to the 1D index of SORTER_QUAT
    // ([0, 1, 2] -> 0, [0, 2, 1] -> 1, [1, 0, 2] -> 2, [1, 2, 0] -> 3, ...)
}


/**
 * Calculate the eigendecomposition of the given real symmetric 3x3 matrix.
 *
 * Uses the Jacobi eigenvalue algorithm [1] in its cyclic version, i.e. each
 * iteration accesses each off-diagonal element once and tries to zero it out.
 * The implementation follows [2,p.430] with two adjustments: First, for the
 * Givens/Jacobi rotation, it uses numerical approximations as proposed in [3].
 * Second, it uses quaternion math wherever possible.
 *
 * Returns the eigendecomposition in terms of the sorted eigenvalues and a
 * quaternion that represents the matrix of the sorted eigenvectors. The
 * sorting order may be determined by redefining the eigcmp() macro.
 *
 * If the calculation does not converge within the given maximum number of
 * iterations, return the eigenvalues and the quaternion as NANs.
 *
 * CAUTION: outQuat represents the respective rotation matrix rot such that
 *
 *     rot.T inMat rot
 *
 * becomes the diagonal matrix of eigenvalues.
 * In other words, outQuat represents the eigenvector matrix of inMat with
 * eigenvectors in columns.
 *
 * Assumes 1D arrays of identical size, assumes to be called with the number of
 * given matrices.
 *
 * References
 * [1] https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm (20150609)
 * [2] Gene H. Golub and C. F. Van Loan, “Matrix computations”, 3rd ed.
 *     Baltimore: Johns Hopkins University Press, 1996.
 * [3] A. McAdams, A. Selle, R. Tamstorf, J. Teran, and E. Sifakis, “Computing
 *     the Singular Value Decomposition of 3×3 matrices with minimal branching
 *     and elementary floating point operations,” University of
 *     Wisconsin–Madison, Madison, WI, USA, technical report 1690, May 2011.
 *
 * @param[in]  inMat the matrices to be decomposed, giving the 6 unique
 *             elements of the i-th matrix in consecutive order according to 
 *             the following scheme:
 *             [i + 0] -> [0][0], [i + 1] -> [1][1], [i + 2] -> [2][2],
 *             [i + 3] -> [0][1], [i + 4] -> [0][2], [i + 5] -> [1][2].
 *             May be the same as outEigs
 * @param[out] outEigs the resulting decomposition, giving each 6 resulting
 *             values in consecutive order: The first 3 elements hold the
 *             sorted eigenvalues, sorting order is determined by the eigcmp()
 *             macro. The last 3 elements hold a "squeezed" version of the unit
 *             quaternion that represents the matrix of sorted eigenvectors, as
 *             produced by the qsqueeze() function; it can be expanded to four
 *             elements with qexpand() (also be sure to read CAUTION above).
 *             May be the same as inMat
 */
__kernel void eigs(__global const ftype *inMat,
                   __global ftype *outEigs)
{
    const size_t i = get_global_id(0) * 6;  // global start index of the current matrix

    itype niter;  // iteration count
    itype p, q; // Matrix indices for the sweeps (p < q)
    itype r;    // The remaining matrix index != p, q (i.e. 2 if p == 0 and q == 1 etc.)

    ftype c;  // The non-zero quaternion elements for the current Givens/Jacobi
    ftype s;  // rotation and the respective rotation matrix elements
    ftype tmp1, tmp2;  // Intermediate/temporary results
    vec onRotd;  // On-diagonal elements of the rotated matrix
    vec offRotd;  // Off-diagonal elements of the rotated matrix (stores element of 2D position [p][q] at index [r])
    
    // Local representation of the output quaternion; initialized with the
    // identity rotation. For the ease of indexing, keep the real/scalar part
    // in the last position (only fix this before writing to global memory)
    qat oQ = {0., 0., 0., 1.};

    bool cond;

    onRotd[0] = inMat[i]; onRotd[1] = inMat[i + 1]; onRotd[2] = inMat[i + 2];
    offRotd[2] = inMat[i + 3]; offRotd[1] = inMat[i + 4]; offRotd[0] = inMat[i + 5];

    // Convergence criterion via maximum absolute value of the input elements
    // (maybe a better one can be chosen)
    tmp1 = fabs(onRotd[0]);
    tmp2 = fabs(onRotd[1]);  tmp1 = (tmp1 < tmp2 ? tmp2 : tmp1);
    tmp2 = fabs(onRotd[2]);  tmp1 = (tmp1 < tmp2 ? tmp2 : tmp1);
    tmp2 = fabs(offRotd[0]); tmp1 = (tmp1 < tmp2 ? tmp2 : tmp1);
    tmp2 = fabs(offRotd[1]); tmp1 = (tmp1 < tmp2 ? tmp2 : tmp1);
    tmp2 = fabs(offRotd[2]); tmp1 = (tmp1 < tmp2 ? tmp2 : tmp1);
    const ftype TOL = tmp1 * EPS;  // == max(abs(inMat)) * EPS

    for (niter = 0; niter < MAXITER; ++niter)
    {
        // Check for convergence
        if ((fabs(offRotd[0]) + fabs(offRotd[1]) + fabs(offRotd[2])) <= TOL)
        {
            // Sort eigenvalues according to given sorting order (onRotd should
            // by now contain the eigenvalues)
            sortEigs(oQ, onRotd);
            // Transpose returned quaternion as we want "eigenvectors in
            // columns" while for now we worked with "eigenvectors in rows"
            // (recall: real/scalar part is still the last element of oQ)
            oQ[3] = -oQ[3];
            // Squeeze the quaternion, then write to global memory (misuse
            // offRotd here)
            qsqueezeLast(oQ, offRotd);
            outEigs[i]     = onRotd[0];  outEigs[i + 1] = onRotd[1];  outEigs[i + 2] = onRotd[2];
            outEigs[i + 3] = offRotd[0]; outEigs[i + 4] = offRotd[1]; outEigs[i + 5] = offRotd[2];    
            return;  
        }

        // Enter the sweep loops: access all off-diagonal elements once
        for (p = 0; p < 2; ++p)
        {
            for (q = p + 1; q < 3; ++q)
            {
                r = 3 ^ (p | q);  // == 3 - (p + q) -- Remaining index, given p and q

                // Find the Givens/Jacobi rotation for the current index p, q
                // (corresponds to "sym.schur2()" in [2], but implements algorithm 2 in [3].)
                c = 2. * (onRotd[q] - onRotd[p]);
                s = offRotd[r];

                tmp1 = square(s);
                tmp2 = square(c);

                cond = (GAMMA * tmp1 < tmp2);
                tmp1 = rsqrt(tmp1 + tmp2);  // omega in [3]

                c = cond ? tmp1 * c : C_STAR;
                s = cond ? tmp1 * s : S_STAR;
                // End of algorithm 2 in [3] (sign for s is not adjusted, yet)

                // Combine the current quaternion with the accumulated
                // rotations via the Hamilton product (adjust sign for s)

                // Recall: Real/scalar part of the quaternions is currently at index 3
                tmp1 = c * oQ[p] - s * oQ[q];
                tmp2 = c * oQ[q] + s * oQ[p];
                oQ[p] = tmp1; oQ[q] = tmp2;
                cond = (r == 1);
                tmp1 = c * oQ[3] + (cond ? s : -s) * oQ[r];
                tmp2 = c * oQ[r] + (cond ? -s : s) * oQ[3];
                oQ[3] = tmp1; oQ[r] = tmp2;

                // Create the respective rotation matrix J (not explicitly)
                tmp1 = 1 - 2 * square(s);
                s = 2 * c * s;  // off-diagonal elements
                c = tmp1;       // on-diagonal elements

                // Rotate M <- J M J.T (greatly simplified through many zeros; correct order of operations matters)
                tmp1       = offRotd[r] * c + onRotd[p]  * s;
                tmp2       = onRotd[q]  * c + offRotd[r] * s;
                onRotd[p]  = c * (onRotd[p]  * c - offRotd[r] * s) -
                             s * (offRotd[r] * c - onRotd[q]  * s);
                onRotd[q]  = c * tmp2       + s * tmp1;
                offRotd[r] = c * tmp1       - s * tmp2;        // [p][q]
                tmp2       = offRotd[q];
                offRotd[q] = c * tmp2       - s * offRotd[p];  // [p][r]
                offRotd[p] = c * offRotd[p] + s * tmp2;        // [q][r]
            }
        }
    }
    // Did not converge
    outEigs[i]     = NAN; outEigs[i + 1] = NAN; outEigs[i + 2] = NAN;
    outEigs[i + 3] = NAN; outEigs[i + 4] = NAN; outEigs[i + 5] = NAN;      
}

