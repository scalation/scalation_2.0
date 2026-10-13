
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Sep 16 13:28:38 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Levenberg-Marquardt Algorithm for Nonlinear Least Squares (NLS)
 *           AI Assisted Code
 */

package scalation
package optimization
package nonlinear_ls

//import scala.annotation.unused
import scala.math.{abs, exp, log, max, pow}

import scalation.calculus.Differential.jacobian
import scalation.mathstat._
import scalation.random.Normal

/*
 * HOW TO WRITE VECTORIZED NON-LINEAR LEAST SQUARES FUNCTIONS
 *
 * EXAMPLE FUNCTION: f(x, b) = b0 * exp (b1 * x)
 *
 * 1. VECTORIZED RESIDUALS TEMPLATE (residualFn):
 *    Mathematical Equation: r = y - f(x, b)
 * 
 *    val residualFunction = (b: VectorD) => 
 *        y - ((x * b(1)).exp * b(0))
 * 
 * 2. VECTORIZED ANALYTICAL JACOBIAN TEMPLATE (jacobianFn):
 *    Mathematical Equation: J = -∂f(x, b) / ∂b
 * 
 *    val analyticalJacobian = (b: VectorD) =>
 *        val expTerm = (x * b(1)).exp
 *        val col0    = -expTerm
 *        val col1    = x * (-b(0)) * expTerm
 *        MatrixD (col0, col1).transpose
 *
 * See test cases below:
 *     levenbergMarquardtTest:   Uses numerically computed Jacobian
 *     levenbergMarquardtTest2:  Uses analytically computed Jacobian
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LevenbergMarquardt` class implements the Levenberg-Marquardt (LM) algorithm
 *  for solving Non-Linear Least Squares (NLS) problems.
 *  This class minimizes the objective function: F(b) = 0.5 * ||r(b)||^2
 *  where r(b) is a vector-valued residual function r: R^n -> R^m.
 *  @nate m is the number of residuals/equations (e.g., number of data points)
 *  @param n           the number of parameters to optimize
 *  @param residualFn  the vector residual function returning a VectorD of size m given parameters VectorD of size n
 *  @param jacobianFn  optional analytical Jacobian function returning an m x n MatrixD given parameters VectorD of size n.
 */
class LevenbergMarquardt (n: Int,
                          val residualFn: VectorD => VectorD,
                          val jacobianFn: Option [VectorD => MatrixD] = None)
      extends NonLinearLS ():

    private val debug = debugf ("LevenbergMarquardt", true)
    private val flaw  = flawf ("LevenbergMarquardt")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Numerically approximates the m x n Jacobian Matrix using a central finite difference schema.
     *  J_ij = ∂r_i / ∂b_j
     *  Uses the analytical option if supplied; otherwise, falls back to a central finite difference schema.
     *  @param b  the current parameter vector
     */
    def computeJacobian (b: VectorD): MatrixD =
        jacobianFn match
            case Some (jFn) => jFn (b)                          // Jacobian via user supplied analytic matrix function
            case None       => jacobian (residualFn, b)         // Jacobian via numerical finite differences
    end computeJacobian

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For the Linear Model y-hat = Xb, solve the normal equations for the parameter
     *  vector b in (X^TX)b = X^ty.  Use the QR factorization solver.
     *  @param x_  the matrix of input/predictor variables
     *  @param y_  the vector holding the output/response vector
     */
    protected def solveNormalEq (x_ : MatrixD, y_ : VectorD): VectorD =
        val qr = Fac_QR (x_)                
        qr.factor ()
        qr.solve (y_)
    end solveNormalEq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the next parameter candidate.  The method is set up for overridding by
     *  subclasses to inject box constraint projections.
     *  @param b      the current parameter vector estimate
     *  @param delta  the calculated parameter displacement (step) vector from the normal equations
     */
    protected def computeNextB (b: VectorD, delta: VectorD): VectorD = b + delta

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Hook to filter or modify the Jacobian matrix before forming normal equations.
     *  Overridden by subclasses to implement active-set column freezing.
     *  @param J  the Jacobian matrix
     *  @param b  the current parameter vector estimate (used in subclasses)
     */
//  protected def filterJacobian (J: MatrixD, @unused b: VectorD): MatrixD = J

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Execute the Levenberg-Marquardt optimization loop to find the optimal parameters
     *  and return the optimized parameter vector.
     *  @param b0  the initial parameter guess vector/starting point of size n
     */
    def optimize (b0: VectorD): VectorD =
        var b      = b0.copy
        var lambda = lambdaInit
        val factor = 10.0
        
        var r   = residualFn (b)
        var sse = r dot r                                       // sum of Squared Errors loss

        var iter      = 1
        var converged = false

        // Refactored to map state variables using standard loops without non-local returns
        while iter <= maxIter && ! converged do
            val J   = computeJacobian (b)                       // compute the Jacobian matrix
            val Jt  = J.transpose
            val JtJ = Jt * J                                    // approximated Hessian matrix base
            val g   = (-Jt * r)                                 // gradient descent driving force vector

            var stepAccepted = false
            
            // Adjust damping parameter adaptively until an acceptable local descent step is found
            while ! stepAccepted && lambda < 1e9 do
                val A = JtJ.copy
                for j <- 0 until n do
                    // Apply Marquardt scaling to regularize along diagonal variance
                    A(j, j) += lambda * max (JtJ(j, j), 1e-4)

                // Solve the normal equations for the parameter update delta: A * delta = g
                val delta = solveNormalEq (A, g)

                if delta == null || delta.notFinite then
                    lambda *= factor                            // ill-conditioned matrix state, enhance damping stiffness
                    flaw ("optimize", s"delta = $delta, new lambda = $lambda")
                else
                    val bNew     = computeNextB (b, delta)
                    val rNew     = residualFn (bNew)
                    val sseNew   = rNew dot rNew
                    val relError = abs (sse - sseNew) / (sse + 1e-12)
                    debug ("optimize", s"sseNew = $sseNew, delta = $delta, bNew = $bNew")

                    if ! sseNew.isNaN && ! sseNew.isInfinite && sseNew < sse then
                        // Functional improvement verified
                        b            = bNew
                        r            = rNew
                        sse          = sseNew
                        lambda      /= factor
                        stepAccepted = true
                        
                        // Convergence check based on parameter update vector magnitude
                        if delta.norm < eps || relError < 1e-13 then converged = true
                    else
                        // Performance degraded, reject step and try stiffer damping
                        lambda *= factor
            end while
            debug ("optimize", s"iter = $iter, sse = $sse, b = $b")
            iter += 1
        end while
        b
    end optimize

end LevenbergMarquardt


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `levenbergMarquardtTest` main method tests the `LevenbergMarquardt` class.
 *  It uses an numerically computed Jacobian.
 *  > runMain scalation.optimization.nonlinear_ls.levenbergMarquardtTest
 */
@main def levenbergMarquardtTest (): Unit =

    // Generate independent variable data points x_i
    val x = VectorD.range (0, 30).map (_ * 0.2)
    val m = x.dim
    val n = 2                                        // 2 parameters to find

    val beta  = VectorD (2.0, -0.5)                  // target parameter values for y = b0 * exp(b1 * x)
    val beta0 = VectorD (1.0, -0.1)                  // initial guess
    val noise = new Normal (0.0, 0.001)              // Normally distributed noise

    // Generate synthetic response values vector y with artificial noise
    val y = new VectorD (m)
    for i <- 0 until m do y(i) = beta(0) * exp (beta(1) * x(i)) + noise.gen

    // Formulate a vectorized residual function: r = y - f(x, b)
    val residualFunc = (b: VectorD) => y - (x * b(1)).exp * b(0)

    // Instantiate and execute our standalone mathematical optimizer
    val optimizer = new LevenbergMarquardt (n, residualFunc)
    val solution  = optimizer.optimize (beta0)

    banner ("NUMERICAL JACOBIAN LEVENBERG-MARQUARDT NLS OPTIMIZATION RESULT")
    println (s"Target Real Values:   $beta")
    println (s"Optimized Solution:   $solution")

end levenbergMarquardtTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `levenbergMarquardtTest2` main method tests the `LevenbergMarquardt` class.
 *  It uses an analytically computed Jacobian.
 *  > runMain scalation.optimization.nonlinear_ls.levenbergMarquardtTest2
 */
@main def levenbergMarquardtTest2 (): Unit =

    val x = VectorD.range (0, 30).map(_ * 0.2)
    val m = x.dim
    val n = 2                                        // 2 parameters to find

    val beta  = VectorD (2.0, -0.5)                  // target parameter values for y = b0 * exp(b1 * x)
    val beta0 = VectorD (1.0, -0.1)                  // initial guess
    val noise = new Normal (0.0, 0.001)              // Normally distributed noise

    // Generate synthetic response values vector y with artificial noise
    val y = new VectorD (m)
    for i <- 0 until m do y(i) = beta(0) * exp (beta(1) * x(i)) + noise.gen

    // Residual function mapping: r_i = y_i - (b0 * exp(b1 * x_i))
    val residualFunc = (b: VectorD) => y - (x * b(1)).exp * b(0)

    // Analytical Jacobian function mapping: J_ij = ∂r_i / ∂b_j
    // ∂r / ∂b0 = -exp (x * b1)
    // ∂r / ∂b1 = x * (-b0) * exp (b1 * x_i)
    val analyticalJacobian = (b: VectorD) =>
        val expTerm = (x * b(1)).exp
        val col0    = -expTerm
        val col1    = x * (-b(0)) * expTerm
        MatrixD (col0, col1).transpose

    // Instantiate optimizer with the structural analytical partial derivatives passed explicitly!
    val optimizer = new LevenbergMarquardt (n, residualFunc, Some (analyticalJacobian))
    val solution  = optimizer.optimize (beta0)

    banner ("ANALYTICAL JACOBIAN LEVENBERG-MARQUARDT NLS OPTIMIZATION RESULT")
    println (s"Target Real Values:   $beta")
    println (s"Optimized Solution:   $solution")

end levenbergMarquardtTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `levenbergMarquardtTest3` main method tests the `LevenbergMarquardt` class.
 *  It uses an analytically computed Jacobian and input x is multi-dimensional.
 *  > runMain scalation.optimization.nonlinear_ls.levenbergMarquardtTest3
 */
@main def levenbergMarquardtTest3 (): Unit =

    // Setup Data Matrix 'x' with 2 independent variables (columns) across 30 instances
    // Column 0 = Temperature (x1), Column 1 = Pressure (x2)
    val m = 30
    val n = 3                                      // have 3 parameters to find: b0, b1, b2
    
    val x = new MatrixD (m, 2)
    for i <- 0 until m do
        x(i, 0) = 1.0 + (i * 0.1)                  // x1 goes up steadily
        x(i, 1) = 5.0 - (i * 0.1)                  // x2 goes DOWN steadily (breaks linear correlation!)

    // Real true coefficients we want the optimizer to recover
    val beta  = VectorD (1.5, 0.8, -0.4) 
    val beta0 = VectorD (1.0, 0.2, 0.0)
    val noise = new Normal (0.0, 0.001, 0)

    // Slice out whole columns using ScalaTion's native column extraction syntax 'x(?, col)'
    val x1 = x(?, 0)
    val x2 = x(?, 1)

    // Generate dependent response vector 'y'
    val y = new VectorD (m)
    for i <- 0 until m do
        y(i) = beta(0) * pow (x1(i), beta(1)) * pow (x2(i), beta(2)) + noise.gen

    // Vectorized Residual Function: r = y - (b0 * x1^b1 * x2^b2)
    // Uses ScalaTion 2's element-wise arithmetic operators and custom vector maps
    val residualFunc = (b: VectorD) =>
        val f_x = x1.map (pow (_, b(1))) * x2.map(pow (_, b(2))) * b(0)
        y - f_x

    // Vectorized Analytical Jacobian Matrix Template: J = -∂f(x, b) / ∂b
    // ∂f/∂b0 = x1^b1 * x2^b2
    // ∂f/∂b1 = b0 * ln(x1) * x1^b1 * x2^b2
    // ∂f/∂b2 = b0 * ln(x2) * x1^b1 * x2^b2
    val analyticalJacobian = (b: VectorD) =>
        val p1 = x1.map (pow (_, b(1)))
        val p2 = x2.map (pow (_, b(2)))
        val baseGroup = p1 * p2
        val col0 = -baseGroup
        val col1 = x1.map (log) * (-b(0)) * baseGroup
        val col2 = x2.map (log) * (-b(0)) * baseGroup
        MatrixD (col0, col1, col2).transpose

    // Execute Optimizer 
    val optimizer = new LevenbergMarquardt (n, residualFunc, Some (analyticalJacobian))
    val solution  = optimizer.optimize (beta0)

    banner ("MULTI-VARIABLE LEVENBERG-MARQUARDT NLS OPTIMIZATION RESULT")
    println (s"Independent Matrix Size:   ${x.dim} rows x ${x.dim2} columns")
    println (s"Target Real Parameters:    $beta")
    println (s"Optimized Recovered Parameters: $solution")

end levenbergMarquardtTest3

