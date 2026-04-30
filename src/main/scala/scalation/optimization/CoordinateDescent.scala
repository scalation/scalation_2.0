
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Apr 18 11:58:39 EDT 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Coordinate Descent for Non-Linear Optimization
 *
 *  @see     arxiv.org/pdf/1502.04759.pdf
 */

package scalation
package optimization

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CoordinateDescent` class solves unconstrained Non-Linear Programming (NLP)
 *  problems using the Coordinate Descent algorithm.  Given a function f and a
 *  starting point x0, the algorithm picks coordinate directions (cyclically) and
 *  takes steps in the those directions.  The algorithm iterates until it converges.
 *
 *      dir_k = kth coordinate direction
 *
 *      min f(x)
 *
 *  @param f        the vector-to-scalar objective function
 *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
 *                            or inexact (e.g., `WolfeLS`) Line Search
 */
class CoordinateDescent (f: FunctionV2S, exactLS: Boolean = true)
      extends Minimizer:

    private val debug  = debugf ("CoordinateDescent", true)     // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an exact `GoldenSectionLS` or inexact `WolfeLS` line search.
     *  Search in direction dir, returning the distance z to move in that direction.
     *  @param x     the current point
     *  @param dir   the direction to move in
     *  @param step  the initial step size
     */
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double =
        def f_1D (z: Double): Double = f(x + dir * z)         // create a 1D function
        val ls = if exactLS then new GoldenSectionLS (f_1D)   // Golden Section line search
                 else new WolfeLS (f_1D, .0001, .1)           // Wolfe line search (c1 = .0001, c2 = .1)
        ls.search (step)                                      // perform a line search
    end lineSearch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the Non-Linear Programming (NLP) problem using the Coordinate Descent
     *  algorithm.
     *  @param x0     the starting point
     *  @param step   the initial step size
     *  @param toler  the tolerance
     */
    def solve (x0: VectorD, step: Double = STEP, toler: Double = EPSILON): FuncVec =
        val n    = x0.dim
        var x    = x0                                         // current point
        var fx   = f(x)                                       // objective function at current point
        var y    = VectorD.nullv                              // next point
        var fy   = 0.0                                        // objective function at next point
        val dir  = new VectorD (n)                            // adjust direction by cycling thru coordinates
        var dist = 1.0                                        // distance between current and next point
        var down = true                                       // moving down flag

        var it = 1
        cfor (it <= MAX_IT && down && dist > toler, it += 1) {   // FIX - use better stopping rule
            
            for fb <- 1 to -1 by -2; j <- 0 until n do        // cycle thru coordinates - establish direction
                            
                if j > 0 then dir(j-1) = 0.0
                dir(j) = fb                                   // set direction forward or backward by fb
                y      = x + dir * lineSearch (x, dir, step)  // determine the next point
                fy     = f(y)                                 // objective function value for next point

                debug ("solve", s"it = $it, y = $y, fy = $fy, dir = $dir")

                dist = (x - y).normSq                         // calc the distance between current and next point
                down = fy < fx                                // still moving down?
                if down then { x = y; fx = fy }               // make the next point, the current point
            end for
        } // cfor
        (fx, x)                                               // return the functional value and current point
    end solve

end CoordinateDescent


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `coordinateDescentTest` main function is used to test the `CoordinateDescent` class.
 *  > runMain scalation.optimization.coordinateDescentTest
 */
@main def coordinateDescentTest (): Unit =

    var x0 = VectorD (0.0, 0.0)                               // starting point

    banner ("Minimizer: (x_0 - 3)^2 + (x_1 - 4)^2 + 1") 
    def f(x: VectorD): Double = (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1
    var optimizer = new CoordinateDescent (f)
    var opt = optimizer.solve (x0)
    println (s"optimal solution (f(x), x) = $opt")

    banner ("Minimize: x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    def g(x: VectorD): Double = x(0)~^4 + (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1
    optimizer = new CoordinateDescent (g)
    opt = optimizer.solve (x0)
    println (s"optimal solution (g(x), x) = $opt")

    banner ("Minimizer: x_0/4 + 5x_0^2 + x_0^4 - 9x_0^2 x_1 + 3x_1^2 + 2x_1^4")
    // @see math.fullerton.edu/mathews/n2003/gradientsearch/GradientSearchMod/Links/GradientSearchMod_lnk_5.html
    x0 = VectorD (0.0, 0.0)
    def f3 (x: VectorD): Double = x(0)/4 + 5*x(0)~^2 + x(0)~^4 - 9*x(0)~^2*x(1) + 3*x(1)~^2 + 2*x(1)~^4
    optimizer = new CoordinateDescent (f3)
    opt = optimizer.solve (x0)
    println (s"optimal solution (f3(x), x) = $opt")

end coordinateDescentTest

