
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Aug 24 19:53:22 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    1D Approximate Wolfe Line Search Optimizer
 *
 *  @see pages.cs.wisc.edu/~ferris/cs730/chap3.pdf
 *  @see Limited Memory BFGS for Nonsmooth Optimization
 */

package scalation
package optimization

import scala.math.abs

import scalation.calculus.Differential.Ⅾ
import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `WolfeLS` class performs an inexact line search on f to find a point x
 *  that exhibits
 *      (1) SDC: sufficient decrease (f(x) enough less that f(0)) and
 *      (2) CC:  the slope at x is less steep than the slope at 0.
 *  That is, the line search looks for a value for x satisfying the two Wolfe conditions.
 *  
 *     f(x) <= f(0) + c1 * f'(0) * x      Wolfe condition 1 (Armijo condition)
 *  |f'(x)| <= |c2 * f'(0)|               Wolfe condition 2 (Strong version)
 *    f'(x) >= c2 * f'(0)                 Wolfe condition 2 (Weak version, more robust)
 *  
 *  It works on scalar functions (@see `wolfeLSTest`).  If starting with a vector function
 *  f(x), simply defines a new function fl(a) = x0 + direction * a (@see `wolfeLSTest2`).
 *  
 *  @param f   the scalar objective function to minimize
 *  @param c1  constant for sufficient decrease (Wolfe condition 1)
 *  @param c2  constant for curvature/slope constraint (Wolfe condition 2)
 */
class WolfeLS (f: FunctionS2S, c1: Double = 0.0001, c2: Double = 0.9)
      extends LineSearch:

    private val debug   = debugf ("WolfeLS", false)                 // debug function
    private val POS_INF = Double.PositiveInfinity                   // Positive Infinity
    private val MAX_IT  = 20                                        // maximum number of iterations
    private val df0     = Ⅾ (f)(0.0)                                // derivative at the origin f'(0)
    private val c2_df0  = c2 * df0                                  // pre-multiplication for Wolfe condition 2

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an inexact Line Search (LS) using the Wolfe approach with defaults.
     *  @param step  the initial step size
     */
    def search (step: Double = 1.0): Double = lsearch_ (step)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an inexact Line Search (LS) on the function f to find a point
     *  x that satisfies the Wolfe Conditions 1 and 2.
     *  @param x0   the current point
     *  @param lo0  the lower bound for x
     */
    def lsearch (x0: Double = 1.0, lo0: Double = 0.0): Double = lsearch_ (x0, lo0)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an inexact Line Search (LS) on the function f to find a point
     *  x that satisfies the Wolfe Conditions 1 and 2.
     *  A Bisection Method for the Wolfe Conditions
     *  @see sites.math.washington.edu/~burke/crs/408/notes/nlp/line.pdf
     *  @param x0    the current point (defaults to 1)
     *  @param lo0   the lower bound for x (defaults to 0)
     *  @param weak  whether to use the weak (true) or strong (false) Wolfe conditions
     */
    def lsearch_ (x0: Double = 1.0, lo0: Double = 0.0, weak: Boolean = true): Double =
        var (lo, hi) = (lo0, POS_INF)                               // starting lower and upper bound for x
        var x    = x0                                               // starting point
        val f0   = f(0.0)                                           // function at 0, f(0)
        val dfx0 = Ⅾ (f)(0.0)                                       // derivative at 0, f'(0)

        var (go, it) = (true, 0)
        cfor (go && it < MAX_IT, it += 1) {

            var fx = f(x)                                           // new point
            var dfx = Ⅾ (f)(x)                                      // new derivative

            if fx > f0 + c1 * x * dfx0 then                         // Wolfe condition 1 fails
                hi = x
                x  = (lo + hi) / 2.0
            else if weak && dfx < c2 * dfx0 then                    // Weak Wolfe condition 2 fails
                lo = x
                x  = if hi < POS_INF then (lo + hi) / 2.0 else x + x
            else if ! weak && abs (dfx) > abs (c2_df0) then         // Strong Wolfe condition 2 fails
                lo = x
                x  = if hi < POS_INF then (lo + hi) / 2.0 else x + x
            else
                go = false                                          // both conditions satisfied

            fx = f(x); dfx = Ⅾ (f)(x)                               // recompute f(x) and f'(x)
            debug ("lsearch_", s"(it = $it) x = $x, f(x) = $fx")
        } // cfor
        x
    end lsearch_

end WolfeLS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `wolfeLSTest` main function is used to test the `WolfeLS` class on scalar functions.
 *  > runMain scalation.optimization.wolfeLSTest
 */
@main def wolfeLSTest (): Unit =

    def f(x: Double): Double = (x - 4.0) * (x - 4.0) + 1.0          // no expansion phase
//  def f(x: Double): Double = (x - 40.0) * (x - 40.0) + 1.0        // requires expansion phase

    val wls = new WolfeLS (f)

    println ("\nProblem 1: (x - 4)^2 + 1") 
    val a = wls.search ()
    println ("optimal a solution = " + a)
    println ("optimal f solution = " + f(a))

end wolfeLSTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `wolfeLSTest2` main function is used to test the `WolfeLS` class on vector functions.
 *  > runMain scalation.optimization.wolfeLSTest2
 */
@main def wolfeLSTest2 (): Unit =

    val z0 = VectorD (0.0, 0.0)                                     // zero vector, the origin
    val p  = VectorD (1.0, 1.0)                                     // direction to search in
    var a  = 0.0                                                    // displacement along search direction
    var x  = z0                                                     // starting point

    def f(x: VectorD): Double  = (x(0) - 2.0) * (x(0) - 2.0) + (x(1) - 3.0) * (x(1) - 3.0) + 1.0
    def fl(a: Double): Double  = f(z0 + p * a)

    val wls  = new WolfeLS (fl)

    println ("\nProblem 1: (x_0 - 2)^2 + (x_1 - 3)^2 + 1") 
    a = wls.search ()
    x = z0 + p * a
    println ("optimal a solution = " + a)
    println ("optimal x solution = " + x)
    println ("optimal f solution = " + f(x))

    def f2(x: VectorD): Double = x(0)/4 + 5*x(0)~^2 + x(0)~^4 - 9*x(0)~^2*x(1) + 3*x(1)~^2 + 2*x(1)~^4
    def fl2(a: Double): Double = f2(z0 + p * a)

    val wls2 = new WolfeLS (fl2)

    println ("\nProblem 4: x_0/4 + 5x_0^2 + x_0^4 - 9x_0^2 x_1 + 3x_1^2 + 2x_1^4")
    // @see http://math.fullerton.edu/mathews/n2003/gradientsearch/GradientSearchMod/Links/GradientSearchMod_lnk_5.html
    a = wls2.search ()
    x = z0 + p * a
    println ("optimal a solution = " + a)
    println ("optimal x solution = " + x)
    println ("optimal f solution = " + f(x))

end wolfeLSTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `wolfeLSTest3` main function is used to test the `WolfeLS2` class on vector functions.
 *  This test uses the Rosenbrock function.
 *  @see https://mikl.dk/post/2019-wolfe-conditions/
 *  > runMain scalation.optimization.wolfeLSTest3
 */
@main def wolfeLSTest3 (): Unit =

    def f (x: VectorD): Double = (1.0 - x(0))~^2 + 100.0 * (x(1) - x(0)~^2)~^2

//  def g (x: VectorD): VectorD = VectorD (-2.0 * (1 - x(0)) - 400.0 * x(0) * (x(1) - x(0)~^2),
//                                         200 * (x(1) - x(0)~^2))

    val z0 = VectorD (-2.2, 3)
    val p  = VectorD (1625.6, 368)
    def fl (a: Double): Double = f(z0 + p * a)

    val wls = new WolfeLS (fl)

    println ("\nProblem 1: (1 - x_0)^2 + 100 (x_1 - x_0^2)^2")
    println (s"start z0 = $z0, search direction = $p")
    val a = wls.search ()
    val x = z0 + p * a
    println ("optimal a solution = " + a)
    println ("optimal x solution = " + x)
    println ("optimal f solution = " + f(x))

end wolfeLSTest3

