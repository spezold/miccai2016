#ifndef _MATRIX_MATH_H
#define _MATRIX_MATH_H

#include "basedefs.h"

/**
 * Return the determinant for the given 3x3 matrix [1].
 *
 * References
 * [1] https://en.wikipedia.org/wiki/Determinant (20151019)
 *
 * @param[in] inM the matrix whose determinant is to be determined.
 * @return the determinant of the given matrix.
 */
ftype det(const mat inM)
{
    return inM[0][0] * (inM[1][1] * inM[2][2] - inM[1][2] * inM[2][1]) -
           inM[0][1] * (inM[1][0] * inM[2][2] - inM[1][2] * inM[2][0]) +
           inM[0][2] * (inM[1][0] * inM[2][1] - inM[1][1] * inM[2][0]);
}

#endif  // _MATRIX_MATH_H
