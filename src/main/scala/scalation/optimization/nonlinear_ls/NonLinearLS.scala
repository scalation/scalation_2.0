
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Sep 16 13:28:38 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Base Trait for Nonlinear Least Squares (NLS) Optimizers
 *           AI Assisted Code
 */

package scalation
package optimization
package nonlinear_ls

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NonLinearLS` trait for multi-dimensional Nonlinear Least Squares (NLS) Optimizers.
 *  @param maxIter     the maximum iteration depth allowance (defaults to 200)
 *  @param lambdaInit  the initial Levenberg-Marquardt damping factor (defaults to 1e-3)
 *  @param eps         the convergence optimization tolerance, parameter update norm (defaults to 1e-7)
 */
trait NonLinearLS (val maxIter: Int = 200, val lambdaInit: Double = 1e-3, val eps: Double = 1e-7):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The vectorized residual function evaluating r = y - f(x, b)
     */
    def residualFn: VectorD => VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The optional vectorized analytical Jacobian matrix function computing J = -∂f(x, b) / ∂b
     */
    def jacobianFn: Option [VectorD => MatrixD]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Execute the optimization algorithm from an initial parameter guess.
     *  @param b0  the intial guess/starting point
     */
    def optimize (b0: VectorD): VectorD

end NonLinearLS

