
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Mar  5 16:10:39 EST 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Regularization Method for MV
 */

package scalation
package modeling
package neuralnet

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegularizedMV` trait describes the `center` method that is to be supported
 *  by all companion objects supporting regularized MV regression.
 */
trait RegularizedMV:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Regularized MV Regression from a data matrix and response matrix.
     *  This function centers the data.  Implementations should return specific return types.
     *  @param x       the un-centered data/input m-by-n matrix, NOT augmented with a first column of ones
     *  @param y       the un-centered response/output matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term 'lambda * norm of b'
     */
    def center (x: MatrixD, y: MatrixD, fname: Array [String] = null,
                hparam: HyperParameter = RidgeRegression.hp): PredictorMV

end RegularizedMV

