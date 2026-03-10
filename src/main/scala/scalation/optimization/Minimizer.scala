
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Oct 24 16:25:29 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Base Trait for Minimizers
 */

package scalation
package optimization

import scalation.mathstat._
import scalation.random.RandomVecD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Minimizer` trait sets the pattern for optimization algorithms for solving
 *  Non-Linear Programming (NLP) problems of the form:
 * 
 *  minimize    f(x)
 *  subject to  g(x) <= 0    [ optionally g(x) == 0 ]
 *
 *  where f is the objective function to be minimized
 *        g is the constraint function to be satisfied, if any
 *
 *  Classes mixing in this trait must implement a function fg that rolls the
 *  constraints into the objective functions as penalties for constraint violation,
 *  a one-dimensional Line Search (LS) algorithm lineSearch and an iterative
 *  method (solve) that searches for improved solutions x-vectors with lower
 *  objective function values f(x).
 */
trait Minimizer:

    protected val EPSILON = 1E-10                        // number close to zero
                                                         //   between machine epsilon and its square root
    protected val TOL     = 100.0 * EPSILON              // default tolerance level more relaxed
    protected val STEP    = 0.5                          // default initial step size
    protected val MAX_IT  = 400                          // maximum number of major steps/iterations 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The objective function f plus a weighted penalty based on the constraint
     *  function g.  Override for constrained optimization and ignore for
     *  unconstrained optimization.
     *  @param x  the coordinate values of the current point
     */
    def fg (x: VectorD): Double = 0.0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an exact, e.g., `GoldenSectionLS` or inexact, e.g., `WolfeLS` line search.
     *  Search in direction dir, returning the distance z to move in that direction.
     *  @param x     the current point
     *  @param dir   the direction to move in
     *  @param step  the initial step size
     */
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the Non-Linear Programming (NLP) problem by starting at x0 and
     *  iteratively moving down in the search space to a minimal point.
     *  Return the optimal point/vector x and its objective function value.
     *  @param x0     the starting point 
     *  @param step   the initial step size (may default to STEP)
     *  @param toler  the tolerance (may default to TOL)
     */
    def solve (x0: VectorD, step: Double, toler: Double): FuncVec

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the following Non-Linear Programming (NLP) problem:
     *  min { f(x) | g(x) <= 0 }.  To use explicit functions for gradient,
     *  replace gradient (fg, x._1 + s) with gradientD (df, x._1 + s).
     *  This method uses multiple random restarts.
     *  @param n      the dimensionality of the search space
     *  @param step_  the initial step size
     *  @param toler  the tolerance
     */
    def resolve (n: Int, step_ : Double = STEP, toler: Double = TOL): FuncVec =
        val rvg = RandomVecD (n, 0.5, -0.5)              // note max before min 
        var opt = (MAX_VALUE, VectorD.nullv)
        for i <- 0 until 2*n do
            val x0 = rvg.gen
            println (s"==> resolve: random restart $i at x0 = $x0")
            opt = better (solve (x0, step_, toler), opt)
        end for
        opt
    end resolve

end Minimizer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Minimizer` object provides multiple testing functions.
 */
object Minimizer:

    val EPSILON = 1E-10                                  // number close to zero
    val STEP    = 0.6                                    // step size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the optimizer's solve method with the given objective function and
     *  and starting point.
     *  @param optimizer  the NLP optimizer object to apply
     *  @param x0         the starting point for the optimzer
     */
    def test (optimizer: Minimizer, x0: VectorD): FuncVec =
        val opt = optimizer.solve (x0, STEP, EPSILON)
        banner (s"** solve: optimal solution (f(x), x) = $opt **")
        opt
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the optimizer's resolve method with the given objective function and
     *  and the dimensionality of the seatch space.  This method uses multiple
     *  restarts.
     *  @param optimizer  the NLP optimizer object to apply
     *  @param n          the dimensionality of the search space
     */
    def test2 (optimizer: Minimizer, n: Int): FuncVec =
        val opt = optimizer.resolve (n)
        banner (s"** resolve: optimal solution (f(x), x) = $opt **")
        opt
    end test2

end Minimizer


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Minimizer_NoLS` trait specifies that the extending optimizer/minimizer
 *  does not use a line search algorithm.
 * 
trait Minimizer_NoLS extends Minimizer:

    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double =
        throw new UnsupportedOperationException ("lineSearch: not provided by this optimizer")
    end lineSearch

end Minimizer_NoLS
 */

