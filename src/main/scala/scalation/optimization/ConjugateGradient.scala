
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Sep 30 13:37:32 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Polak-Ribiere Conjugate Gradient (PR-CG)
 *
 *  @see http://www.neos-guide.org/NEOS/index.php/Nonlinear_Conjugate_Gradient_Method
 */

package scalation
package optimization

import scala.math.{abs, max}

import scalation.calculus.Differential.∇
import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ConjugateGradient` class implements the Polak-Ribiere Conjugate Gradient (PR-CG)
 *  Algorithm for solving Non-Linear  Programming (NLP) problems.  PR-CG determines
 *  a search direction as a weighted combination of the steepest descent direction
 *  (-gradient) and the previous direction.  The weighting is set by the beta function,
 *  which for this implementation used the Polak-Ribiere technique.
 * 
 *  dir_k = - grad (x) + beta * dir_k-1 
 *
 *  minimize    f(x)
 *  subject to  g(x) <= 0    [ optionally g(x) == 0 ]
 *
 *  @param f        the objective function to be minimized
 *  @param g        the constraint function to be satisfied, if any
 *  @param ineq     whether the constraint function must satisfy inequality or equality
 *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
 *                            or inexact (e.g., `WolfeLS`) Line Search
 */
class ConjugateGradient (f: FunctionV2S, g: FunctionV2S = null,
                         ineq: Boolean = true, exactLS: Boolean = true)
      extends Minimizer:

    private val debug  = debugf ("ConjugateGradient", true)   // debug function
//  private val flaw   = flawf ("ConjugateGradient")          // flaw function
    private val WEIGHT = 1000.0                               // weight on penalty for constraint violation

//  private var gr: FunctionV2V = null                        // gradient (vector of partial derivatives) - FIX - should be used

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the beta function using the Polak-Ribiere (PR) technique.  The
     *  function determines how much of the prior direction is mixed in with -gradient.
     *  @param sd1  the search direction at the previous point
     *  @param sd2  the search direction at the current point
     */
    private inline def beta (sd1: VectorD, sd2: VectorD): Double =
        max (0.0, (sd2 dot (sd2 - sd1)) / (sd1.normSq + EPSILON))   // PR-CG (Polak-Ribiere)
    end beta

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the partial derivative functions.  If these functions are available,
     *  they are more efficient and more accurate than estimating the values
     *  using difference quotients (the default approach).
     *  @param partials  the array of partial derivative functions
     *  FIX - should be used
     * 
    def setDerivatives (partials: FunctionV2V): Unit =
        if g != null then flaw ("setDerivatives", "only works for unconstrained problems")
        gr = partials                           // use given functions for partial derivatives
    end setDerivatives
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The objective function f plus a weighted penalty based on the constraint
     *  function g.
     *  @param x  the coordinate values of the current point
     */
    override def fg (x: VectorD): Double =
        val f_x = f(x)
        if g == null then                                     // unconstrained
            f_x
        else                                                  // constrained, g(x) <= 0
            val penalty = if ineq then max (g(x), 0.0) else abs (g(x))
            f_x + abs (f_x) * WEIGHT * penalty * penalty
    end fg

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an exact `GoldenSectionLS` or inexact `WolfeLS` line search.
     *  Search in direction dir, returning the distance z to move in that direction.
     *  @param x     the current point
     *  @param dir   the direction to move in
     *  @param step  the initial step size
     */
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double =
        debug ("lineSearch", s"x = $x, dir = $dir, step = $step")

        def f_1D (z: Double): Double = fg(x + dir * z)        // create a 1D function
        val ls = if exactLS then new GoldenSectionLS (f_1D)   // Golden Section line search
                 else new WolfeLS (f_1D, .0001, .1)           // Wolfe line search (c1 = .0001, c2 = .1)
        ls.search (step)                                      // perform a line search
    end lineSearch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the Non-Linear Programming (NLP) problem using the PR-CG algorithm.
     *  To use explicit functions for gradient, replace ∇(fg, x) with gr(x).
     *  @param x0     the starting point 
     *  @param step   the initial step size
     *  @param toler  the tolerance
     */
    def solve (x0: VectorD, step: Double = STEP, toler: Double = EPSILON): FuncVec =
        debug ("solve", s"x0 = $x0, step = $step, toler = $toler")

        var x    = x0                                         // current point
        var f_x  = fg(x)                                      // objective function at current point
        var y    = VectorD.nullv                              // next point
        var f_y  = 0.0                                        // objective function at next point
        var dir  = - ∇(fg)(x)                                 // initial direction is -gradient
        var dir0 = VectorD.nullv                              // keep the previous direction
        var dist = 1.0                                        // distance between current and next point
        var down = true                                       // moving down flag

        for t <- 1 to MAX_IT if down && dist > toler && dir.normSq > toler do
            debug ("solve", "iteration $t: f(x) = $f_x, x = $x")

            y   = x + dir * lineSearch (x, dir, step)         // determine the next point
            f_y = fg(y)                                       // objective function value for next point
            dir0 = dir                                        // save the current direction
            dir = - ∇(fg)(y)                                  // next search direction using Gradient Descent
            if t > 1 then dir += dir0 * beta (dir0, dir)      // modify search direction using PR-CG

            debug ("solve", s"t = $t, y = $y, f_y = $f_y, dir = $dir")

            dist = (x - y).normSq                             // calc the distance between current and next point
            down = f_y < f_x                                  // still moving down?
            if down then { x = y; f_x = f_y }                 // make the next point, the current point
        end for
        (f_x, x)                                              // return functional value and current point
    end solve

end ConjugateGradient


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `conjugateGradientTest` main function is used to test the `ConjugateGradient
 *  class.
 *      f(x) = (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.conjugateGradientTest
 */
@main def conjugateGradientTest (): Unit =

    val n  = 2
    val x0 = new VectorD (n)

    banner ("Minimize: (x_0 - 3)^2 + (x_1 - 4)^2 + 1")

    def f (x: VectorD): Double = (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1.0

    val optimizer = new ConjugateGradient (f)
    val opt = optimizer.solve (x0)
    println (s"][ optimal solution (f(x), x) = $opt")

end conjugateGradientTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `conjugateGradientTest2` main function is used to test the `ConjugateGradient`
 *  class.
 *      f(x) = x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.conjugateGradientTest2
 */
@main def conjugateGradientTest2 (): Unit =

    val n  = 2
    val x0 = new VectorD (n)

    banner ("Minimize: x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")

    def f (x: VectorD): Double = x(0)~^4 + (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1.0

    val optimizer = new ConjugateGradient (f)
    val opt = optimizer.solve (x0)
    println (s"][ optimal solution (f(x), x) = $opt")

end conjugateGradientTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `conjugateGradientTest3` main function is used to test the `ConjugateGradient`
 *  class.
 *      f(x) = 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.conjugateGradientTest3
 */
@main def conjugateGradientTest3 (): Unit =

    val n  = 2
    val x0 = VectorD (0.1, 0)

    banner ("Minimize: 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")

    def f (x: VectorD): Double = 1/x(0) + x(0)~^4 + (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1.0

    val optimizer = new ConjugateGradient (f)
    var opt = optimizer.solve (x0)
    println (s"][ solve: optimal solution (f(x), x) = $opt")

    opt = optimizer.resolve (n)
    println (s"][ resolve: optimal solution (f(x), x) = $opt")

end conjugateGradientTest3

