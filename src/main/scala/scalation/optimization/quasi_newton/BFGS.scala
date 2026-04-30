
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng, André Laranjeira
 *  @version 2.0
 *  @note    Fri Sep 30 13:37:32 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Broyden–Fletcher–Goldfarb–Shanno (BFGS) Quasi-Newton Optimizer
 *
 *  @see The Superlinear Convergence of a Modified BFGS-Type Method for Unconstrained Optimization
 *  @see On the Robustness of Conjugate-Gradient Methods and Quasi-Newton Methods
 *  @see Limited Memory BFGS for Nonsmooth Optimization
 *  @see http://en.wikipedia.org/wiki/BFGS_method
 *  @see http://www.personal.psu.edu/cxg286/Math555.pdf
 *  @see http://people.orie.cornell.edu/aslewis/publications/bfgs_inexactLS.pdf
 *  @see http://people.orie.cornell.edu/aslewis/publications/bfgs_exactLS.pdf
 */

package scalation
package optimization
package quasi_newton

import scala.math.{abs, max}
import scala.util.control.Breaks.{break, breakable}

import scalation.calculus.Differential.∇
import scalation.mathstat.*
import scalation.optimization.functions.*
import scalation.scala2d.writeImage

import MatrixD.eye
import QNewton.aHi_inc

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BFGS` the class implements the Broyden–Fletcher–Goldfarb–Shanno (BFGS)
 *  Quasi-Newton Algorithm for solving Non-Linear Programming (NLP) problems.
 *  BFGS determines a search direction by deflecting the steepest descent direction
 *  vector (opposite the gradient) by  multiplying it by a matrix that approximates
 *  the inverse Hessian.  Note, this implementation may be set up to work with the
 *  matrix b (approximate Hessian) or directly with the aHi matrix (the inverse of b).
 *
 *  minimize    f(x)
 *  subject to  g(x) <= 0   [ optionally g(x) == 0 ]
 *
 *  @param f        the objective function to be minimized
 *  @param g        the constraint function to be satisfied, if any
 *  @param ineq     whether the constraint is treated as inequality (default) or equality
 *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
 *                            or inexact (e.g., `WolfeLS`) Line Search
 */
class BFGS (f: FunctionV2S, g: FunctionV2S = null,
            ineq: Boolean = true, exactLS: Boolean = false)
      extends Minimizer, PathMonitor:

    private val debug  = debugf ("BFGS", false)                 // debug function
//  private val flaw   = flawf ("BFGS")                         // flaw function
    private val WEIGHT = 1000.0                                 // weight on penalty for constraint violation
    private var bfgs   = true                                   // use BFGS (true) or Gradient Descent (false)

//  private var df: Array [FunctionV2S] = null                  // gradient as explicit functions for partials - FIX - should use

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use the Gradient Descent algorithm rather than the default BFGS algorithm.
     */
    def setSteepest (): Unit = bfgs = false

//  private var b: MatrixD    = null                            // approx. Hessian matrix (use b or aHi)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Update the b matrix, whose inverse is used to deflect -gradient to a
     *  better direction than steepest descent (-gradient).
     *  @param s  the step vector (next point - current point)
     *  @param y  the difference in the gradients (next - current)
     */
//  def updateB (s: VectorD, y: VectorD): Unit =
//      var sy = s dot y                                        // dot product of s and y
//      if abs (sy) < TOL then sy = TOL
//      val sb = s * b
//      b += MatrixD.outer (y, y) / sy - MatrixD.outer (sb, sb) / (sb dot s)
//  end updateB

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the partial derivative functions.  If these functions are available,
     *  they are more efficient and more accurate than estimating the values
     *  using difference quotients (the default approach).
     *  @param grad  the gradient as explicit functions for partials
     *  FIX - should use
    def setDerivatives (grad: Array [FunctionV2S]): Unit =
        if g != null then flaw ("setDerivatives", "only works for unconstrained problems")
        df = grad                                               // use given functions for partial derivatives
    end setDerivatives
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The objective function f plus a weighted penalty based on the constraint
     *  function g.
     *  @param x  the coordinate values of the current point
     */
    override def fg (x: VectorD): Double =
        val f_x = f(x)
        if g == null then                                       // unconstrained
            f_x
        else                                                    // constrained, g(x) <= 0
            val penalty = if ineq then max (g(x), 0.0) else abs (g(x))
            f_x + abs (f_x) * WEIGHT * penalty * penalty
    end fg

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an exact GoldenSectionLS or inexact WolfeLS Line Search.
     *  Search in direction dir, returning the distance z to move in that direction.
     *  Default to 
     *  @param x     the current point
     *  @param dir   the direction to move in
     *  @param step  the initial step size
     */
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double =
        def f_1D (z: Double): Double = fg(x + dir * z)          // create a 1D function
        val ls = if exactLS then new GoldenSectionLS (f_1D)     // Golden Section Line Search
                 else new WolfeLS (f_1D)                        // Wolfe line search ((c1 = .0001, c2 = .9)
        ls.search (step)                                        // perform a Line Search
    end lineSearch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the following Non-Linear Programming (NLP) problem using BFGS:
     *      min { f(x) | g(x) <= 0 }.
     *  It computes numerical gradients.  To use explicit functions for gradient,
     *  add 'grad' parameter to replace '∇ (fg)(xn)'.
     *  @param x0     the starting point 
     *  @param step_  the initial step size
     *  @param toler  the tolerance
     */
    def solve2 (x0: VectorD, step_ : Double = STEP, toler: Double = TOL): FuncVec =
        debug ("solve2", s"x0 = $x0, step_ = $step_, toler = $toler")
        clearPath ()

        var step = step_                                        // set the current step size
        var x    = (x0, ∇ (fg)(x0))                             // current (point, gradient)
        var xx:  (VectorD, VectorD) = (null, null)              // next (point, gradient)
        var dir: VectorD = null                                 // initial direction is -gradient
        var s:   VectorD = null                                 // step vector

        val aHi = eye (x0.dim, x0.dim)                          // approximate Hessian inverse (aHi) matrix
                                                                // start with identity matrix

        debug ("solve2", s"||gradient||^2 = ${x._2.normSq}")

        var mgn         = 0.0                                   // mean gradient normSq
        var diff        = 0.0                                   // diff between current and next point
        val diffTol     = toler * toler                         // tolerance for changes in diff
        var count       = 0                                     // number of times mgn stayed roughly same (< diffTol)
        val maxCount    = 10                                    // max number of times mgn stayed roughly same => terminate
        val n           = x0.dim                                // size of the parameter vector
        var goodGrad    = true                                  // good gradient value flag (not NaN nor infinity)
        var xn: VectorD = null                                  // next value for x (point)

        breakable {
            for it <- 1 to MAX_IT do
                if goodGrad then
                    dir = if bfgs then -(aHi * x._2) else -x._2
                s  = dir * lineSearch (x._1, dir, step)         // update step vector
                xn = x._1 + s                                   // next x point
                if goodGrad then
                    for xx_i <- xn if xx_i.isNaN || xx_i.isInfinite do break ()
                    diff = (xn - x._1).normSq / n               // measure of distance moved
                xx = (xn, ∇ (fg)(xn))                           // compute the next point, gradient
                mgn = xx._2.normSq / n                          // compute mean gradient normSq

                debug ("solve2", s"start of iteration $it: step = $step, f(x) = ${fg(x._1)}, mgn = $mgn")

                if mgn.isNaN || mgn.isInfinite then
                    if ! goodGrad then break ()                 // the gradient is still bad => break
                    goodGrad = false                            // gradient blew up
                    step /= 2.0                                 // halve the step size 
                else if mgn < toler || count > maxCount then
                    { x = xx; break () }                        // return on vanished gradient or haven't moved
                else if goodGrad then
                    if diff < diffTol then count += 1           // increment no movement counter
                    if step < step_   then step  *= 1.5         // increase step size by 50%
                else
                    goodGrad = true                             // gradient is currently fine

                if goodGrad then
                    if bfgs then aHi += aHi_inc (aHi, s, xx._2 - x._2)   // update the deflection matrix aHi
//                  debug ("solve2", s"(it = $it) move from ${x._1} to ${xx._1} where fg(xx._1) = ${fg(xx._1)}")
                    x = xx                                      // make the next point the current point

                add2Path (x._1)                                 // collect points visited
            end for
        } // breakable
        (fg(x._1), x._1)                                        // return functional value and current point
    end solve2

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for an optima by finding a local optima close to the starting point/guess 'x0'.
     *  This version uses explicit functions for the gradient (partial derivatives).
     *  @param x0     the starting point/guess
     *  @param grad   the gradient as explicit functions for partials
     *  @param step_  the initial step size
     *  @param toler  the tolerance
     */
    def solve3 (x0: VectorD, grad: FunctionV2V, step_ : Double = STEP, toler: Double = TOL): FuncVec =
        debug ("solve3", s"x0 = $x0, step_ = $step_, toler = $toler")
        clearPath ()

        var step = step_                                        // set the current step size
        var x    = (x0, grad (x0))                              // current (point, gradient)
        var xx:  (VectorD, VectorD) = (null, null)              // next (point, gradient)
        var dir: VectorD = null                                 // initial direction is -gradient
        var s:   VectorD = null                                 // step vector

        val aHi = eye (x0.dim, x0.dim)                          // approximate Hessian inverse (aHi) matrix
                                                                // start with identity matrix
        add2Path(x._1)

        debug ("solve3", s"||gradient||^2 = ${x._2.normSq}")

        var mgn         = 0.0                                   // mean gradient normSq
        var diff        = 0.0                                   // diff between current and next point
        val diffTol     = toler * toler                         // tolerance for changes in diff
        var count       = 0                                     // number of times mgn stayed roughly same (< diffTol)
        val maxCount    = 10                                    // max number of times mgn stayed roughly same => terminate
        val n           = x0.dim                                // size of the parameter vector
        var goodGrad    = true                                  // good gradient value flag (not NaN nor infinity)
        var xn: VectorD = null                                  // next value for x (point)

        breakable {
            for it <- 1 to MAX_IT do
                if goodGrad then
                    dir = if bfgs then -(aHi * x._2) else -x._2
                s  = dir * lineSearch (x._1, dir, step)         // update step vector
                xn = x._1 + s                                   // next x point
                if goodGrad then
                    for xx_i <- xn if xx_i.isNaN || xx_i.isInfinite do break ()
                    diff = (xn - x._1).normSq / n               // measure of distance moved
                xx = (xn, grad (xn))                            // compute the next point, gradient
                mgn = xx._2.normSq / n                          // compute mean gradient normSq

                debug ("solve3", s"start of iteration $it: step = $step, f(x) = ${fg(x._1)}, mgn = $mgn")

                if mgn.isNaN || mgn.isInfinite then
                    if ! goodGrad then break ()                 // the gradient is still bad => break
                    goodGrad = false                            // gradient blew up
                    step /= 2.0                                 // halve the step size
                else if mgn < toler || count > maxCount then
                    { x = xx; break () }                        // return on vanished gradient or haven't moved
                else if goodGrad then
                    if diff < diffTol then count += 1           // increment no movement counter
                    if step < step_   then step  *= 1.5         // increase step size by 50%
                else
                    goodGrad = true                             // gradient is currently fine

                if goodGrad then
                    if bfgs then aHi += aHi_inc (aHi, s, xx._2 - x._2)   // update the deflection matrix aHi
//                  debug ("solve3", s"(it = $it) move from ${x._1} to ${xx._1} where fg(xx._1) = ${fg(xx._1)}")
                    x = xx                                      // make the next point the current point

                add2Path (x._1)                                 // collect points visited
            end for
        } // breakable
        (fg(x._1), x._1)                                        // return functional value and current point
    end solve3

    def solve (x0: VectorD, step_ : Double = STEP, toler: Double = TOL): FuncVec =
        solve (x0, step_, toler, LBFGSLineSearchAlg.Default, LBFGSLineSearchPrms ())
    end solve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for an optima by finding a local optima close to the starting
     *  point/guess 'x0'.
     *  It computes numerical gradients.  To use explicit functions for gradient,
     *  add 'grad' parameter to replace '∇ (fg)(xn)'.
     *  This version uses a specified line search algorithm implementations developed
     *  for L-BFGS (except for `BacktrackingOrthantWise`, which is currently NOT
     *  supported).  More details can be found in `LBFGSLineSearchAlg`.
     *
     *  @param x0           the starting point/guess
     *  @param step_        the initial step size
     *  @param toler        the tolerance
     *  @param lSearchAlg   `LBFGSLineSearchAlg` representing the line search algorithm chosen for the
     *                      optimization.  Cannot be`BacktrackingOrthantWise`, is not currently supported.
     *  @param lSearchPrms  `LBFGSLineSearchPrms` representing  parameters that control line search execution.
     */
    def solve (x0: VectorD, step_ : Double, toler: Double,
               lSearchAlg: LBFGSLineSearchAlg,
               lSearchPrms: LBFGSLineSearchPrms): FuncVec =
        debug ("solve", s"x0 = $x0, step_ = $step_, toler = $toler")
        clearPath ()

        val step = step_                                        // set the current step size
        var x    = (x0, ∇ (fg)(x0))                             // current (point, gradient)
        var xx:  (VectorD, VectorD) = (null, null)              // next (point, gradient)
        var dir: VectorD = null                                 // initial direction is -gradient
        var s:   VectorD = null                                 // step vector
        add2Path (x._1)

        if lSearchAlg == LBFGSLineSearchAlg.BacktrackingOrthantWise then
            debug("solve", "orthantwise is not currently supported by BFGS")
            return (fg(x._1), x._1)

        val lSearchImple = LBFGSLineSearch.getImple (lSearchAlg)
        val aHi = eye (x0.dim, x0.dim)                          // approximate Hessian inverse (aHi) matrix
                                                                // start with identity matrix

        debug ("solve", s"||gradient||^2 = ${x._2.normSq}")

        var mgn         = 0.0                                   // mean gradient normSq
        var diff        = 0.0                                   // diff between current and next point
        val diffTol     = toler * toler                         // tolerance for changes in diff
        var count       = 0                                     // number of times mgn stayed roughly same (< diffTol)
        val maxCount    = 10                                    // max number of times mgn stayed roughly same => terminate
        val n           = x0.dim                                // size of the parameter vector
        var xn: VectorD = null                                  // next value for x (point)
        val dampeningCoefficient = 0.99                         // how much the increase to aHi should be dampened

        dir = if bfgs then -(aHi * x._2) else -x._2
//      step = 1 / dir.norm

        breakable {
            for it <- 1 to MAX_IT do
                dir = if bfgs then -(aHi * x._2) else -x._2

                lSearchImple.lineSearch (n, x._1, f(x._1), x._2, dir, step,
                                         LBFGSCallbackData (n, None, FunctionEvaluation (f)),
                                         lSearchPrms)
                match
                    case lSearchStep: LBFGSLineSearchStep =>
                        s = dir * lSearchStep.step
                    case lSearchFailure: LBFGSLineSearchFailure =>
                        debug ("solve", s"line search failed: $lSearchFailure")
                        break ()

                xn = x._1 + s                                   // next x point

                for xx_i <- xn if xx_i.isNaN || xx_i.isInfinite do break ()
                diff = s.normSq / n                             // measure of distance moved

                xx  = (xn, ∇ (fg)(xn))                          // compute the next point, gradient
                mgn = xx._2.normSq / n                          // compute mean gradient normSq

                debug ("solve", s"start of iteration $it: step = $step, f(x) = ${fg(x._1)}, mgn = $mgn")

                if mgn < toler || count > maxCount then
                    { x = xx; break () }                        // return on vanished gradient or haven't moved
                else if diff < diffTol then
                    count += 1                                  // increment no movement counter

                if bfgs then aHi += aHi_inc (aHi, s, xx._2 - x._2) * dampeningCoefficient    // update the deflection matrix aHi
//              debug ("solve", s"(it = $it) move from ${x._1} to ${xx._1} where fg(xx._1) = ${fg(xx._1)}")
                x = xx                                          // make the next point the current point

//              step = lSearchPrms.defaultStep
                add2Path (x._1)                                 // collect points visited
            end for
        } // breakable

        (fg(x._1), x._1)                                        // return functional value and current point
    end solve
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for an optima by finding a local optima close to the starting
     *  point/guess 'x0'.
     *  This version uses explicit functions for the gradient (partials
     *  derivatives) and a specified line search algorithm implementations developed
     *  for L-BFGS (except for `BacktrackingOrthantWise`, which is currently NOT
     *  supported).  More details can be found in `LBFGSLineSearchAlg`.
     *
     *  @param x0           the starting point/guess
     *  @param grad         the gradient as explicit functions for partials
     *  @param step_        the initial step size
     *  @param toler        the tolerance
     *  @param lSearchAlg   `LBFGSLineSearchAlg` representing the line search algorithm chosen for the
     *                      optimization.  Cannot be`BacktrackingOrthantWise`, is not currently supported.
     *  @param lSearchPrms  `LBFGSLineSearchPrms` representing  parameters that control line search execution.
     */
    def solve_ (x0: VectorD, grad: FunctionV2V, step_ : Double = STEP, toler: Double = TOL,
                lSearchAlg: LBFGSLineSearchAlg = LBFGSLineSearchAlg.Default,
                lSearchPrms: LBFGSLineSearchPrms = LBFGSLineSearchPrms ()): FuncVec =
        debug ("solve_", s"x0 = $x0, step_ = $step_, toler = $toler")
        clearPath ()

        var step = step_                                        // set the current step size
        var x    = (x0, grad (x0))                              // current (point, gradient)
        var xx:  (VectorD, VectorD) = (null, null)              // next (point, gradient)
        var dir: VectorD = null                                 // initial direction is -gradient
        var s:   VectorD = null                                 // step vector
        add2Path (x._1)

        if lSearchAlg == LBFGSLineSearchAlg.BacktrackingOrthantWise then
            debug("solve_", "orthantwise is not currently supported by BFGS")
            return (fg(x._1), x._1)

        val lSearchImple = LBFGSLineSearch.getImple (lSearchAlg)
        val aHi = eye (x0.dim, x0.dim)                          // approximate Hessian inverse (aHi) matrix
                                                                // start with identity matrix

        debug ("solve_", s"||gradient||^2 = ${x._2.normSq}")

        var mgn         = 0.0                                   // mean gradient normSq
        var diff        = 0.0                                   // diff between current and next point
        val diffTol     = toler * toler                         // tolerance for changes in diff
        var count       = 0                                     // number of times mgn stayed roughly same (< diffTol)
        val maxCount    = 10                                    // max number of times mgn stayed roughly same => terminate
        val n           = x0.dim                                // size of the parameter vector
        var xn: VectorD = null                                  // next value for x (point)
        val dampeningCoefficient = 0.99                         // how much the increase to aHi should be dampened

        dir = if bfgs then -(aHi * x._2) else -x._2
        step = 1 / dir.norm

         breakable {
            for it <- 1 to MAX_IT do
                dir = if bfgs then -(aHi * x._2) else -x._2

                lSearchImple.lineSearch (n, x._1, f(x._1), grad(x._1), dir, step,
                                         LBFGSCallbackData (n, None, FunctionEvaluation (f, grad)),
                                         lSearchPrms)
                match
                    case lSearchStep: LBFGSLineSearchStep =>
                        s = dir * lSearchStep.step
                    case lSearchFailure: LBFGSLineSearchFailure =>
                        debug ("solve_", s"line search failed: $lSearchFailure")
                        break ()

                xn = x._1 + s                                   // next x point

                for xx_i <- xn if xx_i.isNaN || xx_i.isInfinite do break ()
                diff = s.normSq / n                             // measure of distance moved

                xx  = (xn, grad (xn))                           // compute the next point, gradient
                mgn = xx._2.normSq / n                          // compute mean gradient normSq

                debug ("solve_", s"start of iteration $it: step = $step, f(x) = ${fg(x._1)}, mgn = $mgn")

                if mgn < toler || count > maxCount then
                    { x = xx; break () }                        // return on vanished gradient or haven't moved
                else if diff < diffTol then
                    count += 1                                  // increment no movement counter

                if bfgs then aHi += aHi_inc (aHi, s, xx._2 - x._2) * dampeningCoefficient    // update the deflection matrix aHi
//              debug ("solve_", s"(it = $it) move from ${x._1} to ${xx._1} where fg(xx._1) = ${fg(xx._1)}")
                x = xx                                          // make the next point the current point

                step = lSearchPrms.defaultStep
                add2Path (x._1)                                 // collect points visited
            end for
        } // breakable

        (fg(x._1), x._1)                                        // return functional value and current point
    end solve_

end BFGS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BFGS` companion object provides factory methods.
 */
object BFGS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Steepest Descent (default) or BFGS optimizer.
     *  @param f         the objective function to be minimized
     *  @param g         the constraint function to be satisfied, if any
     *  @param ineq      whether the constraint is treated as inequality (default) or equality
     *  @param exactLS   whether to use exact (e.g., `GoldenLS`)
     *                             or inexact (e.g., `WolfeLS`) Line Search
     *  @param steepest  whether to use Steepest Descent rather than BFGS
     */
    def apply (f: FunctionV2S, g: FunctionV2S = null,
               ineq: Boolean = true, exactLS: Boolean = false,
               steepest: Boolean = true): BFGS =
        if steepest then
           val steep = new BFGS (f, g, ineq, exactLS)
           steep.setSteepest ()
           steep
        else
           new BFGS (f, g, ineq, exactLS)
    end apply

end BFGS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSTest` main function is used to test the `BFGS` class on f(x):
 *      f(x) = (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.bFGSTest
 */
@main def bFGSTest (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting point

    banner ("Minimize: (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    def f    = ParaboloidFunction.objFunction
    def grad = ParaboloidFunction.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

end bFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSTest2` main function is used to test the `BFGS` class on f(x):
 *      f(x) = x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.bFGSTest2
 */
@main def bFGSTest2 (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting point

    banner ("Minimize: x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    def f    = QuarticFunction.objFunction
    def grad = QuarticFunction.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

end bFGSTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSTest3` main function is used to test the `BFGS_NoLS` class.
 *  This test uses the Rosenbrock function.
 *      f(x) = (1 - x_0)^2 + 100 (x_1 - x_0^2^)^2
 *  > runMain scalation.optimization.quasi_newton.bFGSTest3
 */
@main def bFGSTest3 (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting point

    banner ("Minimize: (1 - x_0)^2 + 100 (x_1 - x_0^2)^2")
    def f    = RosenbrockFunction.objFunction
    def grad = RosenbrockFunction.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

end bFGSTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSTest4` main function is used to test the `BFGS` class on f(x):
 *      f(x) = 1/x(0) + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.bFGSTest4
 */
@main def bFGSTest4 (): Unit =  

    val step = 1.0                                             // step size (may need adjustment)
    val x0   = VectorD (0.1, 0.0)                              // starting location

    banner ("Minimize: 1/x(0) + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    def f    = ReciprocalFunction.objFunction
    def grad = ReciprocalFunction.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSBoothFunction` main function is used to test the `BFGS` class on f(x):
 *      f(x) = (x(0) + 2 * x(1) - 7)^2 + (2 * x(0) + x(1) - 5)^2
 *  > runMain scalation.optimization.quasi_newton.bFGSBoothFunction
 */
@main def bFGSBoothFunction (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting location

    banner ("Minimize: (x(0) + 2 * x(1) - 7)^2 + (2 * x(0) + x(1) - 5)^2")
    def f    = BoothFunction.objFunction
    def grad = BoothFunction.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSBoothFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSBealeFunction` main function is used to test the `BFGS` class on f(x):
 *      f(x) = (1.5 - x(0) + x(0)*x(1))^2 + (2.25 - x(0) + x(0)*(x(1)^2^))^2 + (2.625 - x(0) + x(0)*(x(1)^3))^2
 *  > runMain scalation.optimization.quasi_newton.bFGSBealeFunction
 */
@main def bFGSBealeFunction (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val x0   = VectorD (2, -2)                                 // starting location
    val functionDomainLowerBound = VectorD (-10, -10)
    val functionDomainUpperBound = VectorD (10, 10)

    banner ("Minimize: (1.5 - x(0) + x(0)*x(1))^2 + (2.25 - x(0) + x(0)*(x(1)^2))^2 + (2.625 - x(0) + x(0)*(x(1)^3))^2")
    def f    = BealeFunction.objFunction
    def grad = BealeFunction.gradFunction

    val optimizer = new BFGS (f)
    val opt = optimizer.solve_ (x0, grad, step)
    println (s"][ optimal solution (f(x), x) = $opt")

    val plot = new PlotC (BealeFunction.objFunction, functionDomainLowerBound, functionDomainUpperBound,
                          optimizer.getPath, BealeFunction.functionMinimum)
    writeImage ("./plots/BFGS/BFGS_bealeFunction_plot.png", plot)

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSBealeFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSBohachevsky1Function` main function is used to test the `BFGS` class on f(x):
 *      f(x) = x(0)^2 + 2*x(1)^2 - 0.3*cos(3*Pi*x(0)) - 0.4*cos(4*Pi*x(1)) + 0.7
 *  > runMain scalation.optimization.quasi_newton.bFGSBohachevsky1Function
 */
@main def bFGSBohachevsky1Function (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting location

    banner ("Minimize: x(0)^2 + 2*x(1)^2 - 0.3*cos(3*Pi*x(0)) - 0.4*cos(4*Pi*x(1)) + 0.7")
    def f    = Bohachevsky1Function.objFunction
    def grad = Bohachevsky1Function.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSBohachevsky1Function


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSBohachevsky2Function` main function is used to test the `BFGS` class on f(x):
 *      f(x) = x(0)^2 + 2*x(1)^2 - 0.3*cos(3*Pi*x(0))*cos(4*Pi*x(1)) + 0.3
 *  > runMain scalation.optimization.quasi_newton.bFGSBohachevsky2Function
 */
@main def bFGSBohachevsky2Function (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting location

    banner ("Minimize: x(0)^2 + 2*x(1)^2 - 0.3*cos(3*Pi*x(0))*cos(4*Pi*x(1)) + 0.3")
    def f    = Bohachevsky2Function.objFunction
    def grad = Bohachevsky2Function.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSBohachevsky2Function


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSBohachevsky3Function` main function is used to test the `BFGS` class on f(x):
 *      f(x) = x(0)^2 + 2*x(1)^2 - 0.3*cos(3*Pi*x(0)+4*Pi*x(1)) + 0.3
 *  > runMain scalation.optimization.quasi_newton.bFGSBohachevsky3Function
 */
@main def bFGSBohachevsky3Function (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting location

    banner ("Minimize: x(0)^2 + 2*x(1)^2 - 0.3*cos(3*Pi*x(0)+4*Pi*x(1)) + 0.3")
    def f    = Bohachevsky3Function.objFunction
    def grad = Bohachevsky3Function.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSBohachevsky3Function


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSCamel3Function` main function is used to test the `BFGS` class on f(x):
 *      f(x) = 2*x(0)^2 - 1.05*x(0)^4 + (1/6.0)*x(0)^6 + x(0)*x(1) + x(1)^2
 *  > runMain scalation.optimization.quasi_newton.bFGSCamel3Function
 */
@main def bFGSCamel3Function (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting location

    banner ("Minimize:  2*x(0)^2 - 1.05*x(0)^4 + (1/6.0)*x(0)^6 + x(0)*x(1) + x(1)^2")
    def f    = Camel3Function.objFunction
    def grad = Camel3Function.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSCamel3Function

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSCubeFunction` main function is used to test the `BFGS` class on f(x):
 *      f(x) = 100*(x(1) - x(0)^3)^2 + (1-x(0))^2
 *  > runMain scalation.optimization.quasi_newton.bFGSCubeFunction
 */
@main def bFGSCubeFunction (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting location

    banner ("Minimize:  100*(x(1) - x(0)^3)^2 + (1-x(0))^2")
    def f    = CubeFunction.objFunction
    def grad = CubeFunction.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSCubeFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSFreudensteinRothFunction` main function is used to test the `BFGS` class on f(x):
 *      f(x) = (x(0) - 13 + x(1)*((5-x(1))*x(1) -2))^2 + (x(0) -29 + x(1)*((x(1) + 1)*x(1) -14))^2
 *  > runMain scalation.optimization.quasi_newton.bFGSFreudensteinRothFunction
 */
@main def bFGSFreudensteinRothFunction (): Unit =

    val step = 1.0                                             // step size (may need adjustment)
    val n    = 2                                               // dimension of the search space
    val x0   = new VectorD (n)                                 // starting location

    banner ("Minimize:  (x(0) - 13 + x(1)*((5-x(1))*x(1) -2))^2 + (x(0) -29 + x(1)*((x(1) + 1)*x(1) -14))^2")
    def f    = FreudensteinRothFunction.objFunction
    def grad = FreudensteinRothFunction.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSFreudensteinRothFunction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bFGSFreudensteinRothFunction` main function is used to test the `BFGS` class on f(x):
 *      f(x) = (x(0) - 13 + x(1)*((5-x(1))*x(1) -2))^2 + (x(0) -29 + x(1)*((x(1) + 1)*x(1) -14))^2
 *  > runMain scalation.optimization.quasi_newton.bFGSFreudensteinRothFunction
 */
@main def bFGSMcCormickFunction (): Unit =

    // Variable declaration.
    val functionDomainLowerBound = VectorD(-4, -4)
    val functionDomainUpperBound = VectorD(4, 4)
    val functionMinimum = McCormickFunction.functionMinimum

    val step = 1.0                                             // step size (may need adjustment)
    val x0   = VectorD (2.5, 3.5)                              // starting location

    banner ("Minimize:  sin(x(0) + x(1)) + (x(0) - x(1))^2 - 1.5 * x(0) + 2.5 * x(1) + 1")
    def f    = McCormickFunction.objFunction
    def grad = McCormickFunction.gradFunction

    val optimizer = new BFGS (f)
//  val opt = optimizer.solve2 (x0, step)                      // use numerical partials
    val opt = optimizer.solve3 (x0, grad, step)                // use functions for partials
    println (s"][ optimal solution (f(x), x) = $opt")

    new PlotC (f, functionDomainLowerBound, functionDomainUpperBound, optimizer.getPath, functionMinimum)

//  opt = optimizer.resolve (n)                                // try multiple starting points
//  println (s"][ optimal solution (f(x), x) = $opt")

end bFGSMcCormickFunction

