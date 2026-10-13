
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Sep  8 19:20:20 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Effective Degrees of Freedom (EDoF)
 *           AI Assisted Code
 *
 *  @see     `RidgeRegression` for a test case
 */

package scalation
package modeling

import scalation.mathstat.{Fac_QR, MatrixD}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EffectiveDoF` trait provides a method for calculating Effective Degrees of Freedom
 *  (EDoF).
 */
trait EffectiveDoF:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate EDoF strictly using the Upper Triangular R matrix generated from
     *  the augmented QR factorization path.  Computes R^-1 efficiently as R
     *  (from the augmented QR) is n x n upper-triangular.
     *  @param x    the original design matrix X (m x n)
     *  @param fac  the matrix factorization (currently requires `Fac_QR`)
     */
    def calcEDoF (x: MatrixD, fac: Fac_QR): Double =
        val transformedX = x * fac.r_inverse         // transform original X matrix: Z = X * R^-1
        transformedX.normFSq                         // edof = tr(H) = ||X * R^-1||_F^2
    end calcEDoF

end EffectiveDoF

