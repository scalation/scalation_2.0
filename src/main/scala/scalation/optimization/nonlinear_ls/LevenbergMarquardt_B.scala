
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Sep 22 12:10:51 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Levenberg-Marquardt Algorithm for Nonlinear Least Squares (NLS) with Bounds
 *           AI Assisted Code
 */

package scalation
package optimization
package nonlinear_ls

import scala.math.{log, max, min, pow}

//import scalation.calculus.Differential.jacobian
import scalation.mathstat._
//import scalation.random.Normal

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LevenbergMarquardt_B` class implements a Non-Linear Least Squares solver
 *  with explicit box constraints (lower and upper parameter bounds).
 *  It extends or wraps the core Levenberg-Marquardt logic using a projected step method.
 *  @param n       the number of parameters (columns)
 *  @param lb      the lower bounds vector for parameters
 *  @param ub      the upper bounds vector for parameters
 *  @param rFn     the vectorized residual function
 *  @param jFn     the optional vectorized analytical Jacobian function
 */
class LevenbergMarquardt_B (n: Int, lb: VectorD, ub: VectorD,
                            rFn: VectorD => VectorD,
                            jFn: Option[VectorD => MatrixD] = None)
      extends LevenbergMarquardt (n, rFn, jFn):

    private val debug = debugf ("LevenbergMarquardt_B", true)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project a parameter vector 'b' back into the feasible region defined by lb and ub.
     *  @param b  the current parameter vector estimate
     */
    private def project (b: VectorD): VectorD =
        val p = new VectorD (n)
        for j <- 0 until n do p(j) = max (lb(j), min (ub(j), b(j)))
        p
    end project

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Override the update step to enforce hard box constraints at every iteration.
     *  @param b      the current parameter vector estimate
     *  @param delta  the calculated parameter displacement (step) vector from the normal equations
     */
    override protected def computeNextB (b: VectorD, delta: VectorD): VectorD = project (b + delta)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Overrides or extends the optimization method to apply bound constraints during
     *  the parameter update phase.  Clamps the initial guess to ensure it starts in a
     *  feasible state, then executes the parent loop.
     *  @param b0  the initial guess parameter vector
     */
    override def optimize (b0: VectorD): VectorD =
        // FIX - need a better way to deal with boundary conditions
        // Initialize with simple projection to ensure a feasible starting state
        val p = project (b0) 
        debug ("optimize", s"project ($b0) = $p")
        project (super.optimize (p))
    end optimize

end LevenbergMarquardt_B


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `levenbergMarquardt_BTest` main method tests the `LevenbergMarquardt_B` class.
 *  It uses an analytically computed Jacobian and input x is multi-dimensional.
 *  > runMain scalation.optimization.nonlinear_ls.levenbergMarquardt_BTest
 */
@main def levenbergMarquardt_BTest (): Unit =

    val m = 30
    val n = 3
    
    // Re-use the multi-variable data setup matrix 'x'
    val x = new MatrixD (m, 2)
    for i <- 0 until m do
        x(i, 0) = 1.0 + (i * 0.1)
        x(i, 1) = 5.0 - (i * 0.1)

    val x1 = x(?, 0)
    val x2 = x(?, 1)

    val beta  = VectorD (1.5, 0.8, -0.4)
    val beta0 = VectorD (1.0, 0.2, 0.0)

    val y = new VectorD (m)
    for i <- 0 until m do y(i) = beta(0) * pow (x1(i), beta(1)) * pow (x2(i), beta(2))

    // Functions
    val residualFunc = (b: VectorD) =>
        y - (x1.map (pow (_, b(1))) * x2.map (pow (_, b(2))) * b(0))

    val analyticalJacobian = (b: VectorD) =>
        val p1 = x1.map(pow (_, b(1)))
        val p2 = x2.map(pow (_, b(2)))
        val baseGroup = p1 * p2
        val col0 = -baseGroup
        val col1 = x1.map (log) * (-b(0)) * baseGroup
        val col2 = x2.map (log) * (-b(0)) * baseGroup
        MatrixD (col0, col1, col2).transpose

    // Define strict limits. 
    // Say we happen to know from domain expertise that b0 cannot exceed 1.45!
    val lower = VectorD (0.0,  0.0, -1.0)
    val upper = VectorD (1.45, 2.0,  1.0) // Clamping b0 short of its real 1.5 target

    // Execute Bounded Optimizer
    val lmb      = new LevenbergMarquardt_B (n, lower, upper, residualFunc, Some (analyticalJacobian))
    val solution = lmb.optimize (beta0)

    banner ("BOUNDED LEVENBERG-MARQUARDT NLS OPTIMIZATION RESULT")
    println (s"Enforced Parameter Upper Bounds:  $upper")
    println (s"Target Real Unconstrained Beta:  $beta")
    println (s"Bounded Final Recovered Solution: $solution")

end levenbergMarquardt_BTest

