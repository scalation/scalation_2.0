
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun May  5 13:13:42 EDT 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Nelder-Mead Simplex Derivative-Free Optimization
 *
 *  @see     http://www.scholarpedia.org/article/Nelder-Mead_algorithm
 *  @see     http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2097904
 *  @see     https://onlinelibrary.wiley.com/doi/epdf/10.1002/anac.200410015
 */

package scalation
package optimization

import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{breakable, break}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NelderMeadSimplex2` solves Non-Linear Programming (NLP) problems using
 *  the Nelder-Mead Simplex algorithm.  Given a function f and its dimension
 *  n,  the algorithm moves a simplex defined by n + 1 points in order to find
 *  an optimal solution.  The algorithm is derivative-free.
 *
 *  minimize    f(x)
 *
 *  @param f  the vector-to-scalar objective function
 *  @param n  the dimension of the search space
 */
class NelderMeadSimplex2 (f: FunctionV2S, n: Int, checkCon: Boolean = false,
                         lower: VectorD = null, upper: VectorD = null)
      extends Minimizer
         with BoundsConstraint (lower, upper)
         with MonitorEpochs:

    private val debug   = debugf ("NelderMeadSimplex2", true) // debug function
    private val flaw    = flawf ("NelderMeadSimplex2")        // flaw function
    private val np1     = n + 1                              // number of vertices/points in simplex
    private val simplex = Array.ofDim [FuncVec] (np1)        // simplex { vetices } used for search

    private val alpha = 1.0                                  // alpha (> 0)  parameter for reflection
    private val beta  = 0.5                                  // beta  (0, 1) parameter for contraction
    private val gamma = 2.0                                  // gamma (> 1)  parameter for expansion
    private val delta = 0.5                                  // delta (0, 1) parameter for shrinkage

    private var (f_h, f_s, f_l) = (0.0, 0.0, 0.0)            // worst, second worst, best functional values

    if n < 2 then flaw ("init", "requires at least a 2-dimensional problem")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize the search simplex by setting n + 1 vertices and computing
     *  their functional values.
     *  @param x0  the given starting point
     *  @param step  the step size
     */
    def initSimplex (x0: VectorD, step: Double): Unit =
        simplex(0) = (f(x0), x0)                             // given starting point and its functional value
        for i <- 1 to n do
            val x = x0 + VectorD.oneAt (i-1, x0.dim) * step
            simplex(i) = (f(x), x)
        end for
        sort ()                                              // order vertices high to low
    end initSimplex

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort the vertices in non-increasing order (high to low).  Then the key
     *  indices are worst/highest (h=0), second worst (s=1), and best/lowest (l=n).
     */
    private def sort (): Unit =
        for i <- 0 until n do
            var im = i
            for j <- i+1 to n if simplex(j)._1 > simplex(im)._1 do im = j
            if im != i then
                val t = simplex(i); simplex(i) = simplex(im); simplex(im) = t
        end for
    end sort

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the centroid of the best-side of the simplex (excluding h=0),
     *  returning it and its functional value.
     */
    private def centroid (): FuncVec =
        val c = new VectorD (n)                              // the centroid of the simplex
        for i <- 1 to n do c += simplex(i)._2                // add vertex points, except h=0
        val x_c = c / n.toDouble                             // divide by # vertices - 1
        (f(x_c), x_c)
    end centroid

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reflect: compute the reflection point of the worst point (h=0) across
     *  the centroid.
     *  @param x_c  the best-side centroid of the simplex
     */
    private def reflect (x_c: VectorD): FuncVec =
        val x_r = x_c + (x_c - simplex(0)._2) * alpha
        (f(x_r), x_r)
    end reflect

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand: compute the expansion point beyond the reflection point.
     *  @param x_c  the best-side centroid of the simplex
     *  @param x_r  the reflection point
     */
    private def expand (x_c: VectorD, x_r: VectorD): FuncVec = 
        val x_e = x_c + (x_r - x_c) * gamma
        (f(x_e), x_e)
    end expand

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Contract: compute the outer contraction point between x_r and x_c.
     *  @param x_c  the best-side centroid of the simplex
     *  @param x_r  the reflection point
     */
    private def contractOut (x_c: VectorD, x_r: VectorD): FuncVec =
        val x_co = x_c + (x_r - x_c) * beta
        (f(x_co), x_co)
    end contractOut

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Contract: compute the inner contraction point between x_h and x_c.
     *  @param x_c  the best-side centroid of the simplex
     */
    private def contractIn (x_c: VectorD): FuncVec =
        val x_ci = x_c + (simplex(0)._2 - x_c) * beta
        (f(x_ci), x_ci)
    end contractIn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shrink: fixing the best/lowest point (l=n), move the rest of the points
     *  toward it.
     */
    private def shrink (): Unit =
        val x_l = simplex(n)._2                              // the best vertex point
        for i <- 0 until n do
            val x = x_l + (simplex(i)._2 - x_l) * delta
            if checkCon then constrain (x)
            simplex(i) = (f(x), x)                           // updated vertex
        end for
    end shrink

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an exact (e.g., `GoldenSectionLS`) or inexact (e.g., `WolfeLS`) line search.
     *  Search in direction dir, returning the distance z to move in that direction.
     *  Currently NOT USED, but may be used to find a better point to add to simplex.
     *  @param x     the current point
     *  @param dir   the direction to move in
     *  @param step  the initial step size
     */
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double = 0.0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace the worst vertex (h=0) in the simplex with the new point.
     *  @param  x_n  the new replacement point
     */
    private def replace (x_n: VectorD): Unit =
        simplex(0) = (f(x_n), x_n)                           // put new vertex at index h=0
        debug ("replace", s"0 with ${simplex(0)}")
        sort ()                                              // re-establish the vertex order
    end replace

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Improve the simplex by replacing the worst/highest vertex (x_h) with a
     *  a better one found on the line containing x_h and the centroid (x_c).
     *  Try the reflection, expansion, outer contraction and inner contraction
     *  points, in that order.  If none succeeds, shrink the simplex and iterate.
     *  Return both distance and difference between x_h (worst) and x_l (best).
     *  @param toler  the tolerance used for termination
     */
    def improveSimplex (toler: Double = EPSILON): (Double, Double) =
        var dist = (simplex(0)._2 - simplex(n)._2).norm             // distance between x_h and x_l
        var diff =  simplex(0)._1 - simplex(n)._1                   // difference between f_h and f_l

        breakable {
            for _ <- 0 until MAX_IT do
                f_h = simplex(0)._1                                        // functional value for x_h (highest/worst)
                f_s = simplex(1)._1                                        // functional value for x_s (second worst)
                f_l = simplex(n)._1                                        // functional value for x_l (lowest/best)
                val (_, x_c) = centroid ()                                 // compute best-side centroid of simplex, f_c not used
                val (f_r, x_r) = reflect (x_c)                             // compute reflection point
                val smaller = f_r <  f_l                                   // f_r smaller than best
                val larger  = f_r >= f_s                                   // f_r at least as large as second worst

                if ! smaller && ! larger then { replace (x_r); break () }  // f_r in middle, replace x_h with x_r

                if smaller then
                    val (f_e, x_e) = expand (x_c, x_r)                     // expand beyond reflection point
                    if f_e < f_r then  { replace (x_e); break () }         // replace worst x_h with x_e
                    else { replace (x_r); break () }                       // replace worst x_h with x_r

                if larger then                                             // contract back from reflection point
                    if f_r < f_h then                                      // f_r between second worst and worst
                        val (f_co, x_co) = contractOut (x_c, x_r)
                        if f_co <= f_r then { replace (x_co); break () }   // replace worst x_h with x_co
                    else                                                   // f_r at least as large as worst
                        val (f_ci, x_ci) = contractIn (x_c)
                        if f_ci <= f_h then { replace (x_ci); break () }   // replace worst x_h with x_ci
                end if

                shrink ()                                                  // shrink the size of the simplex
                sort ()                                                    // re-establish vertex order
                dist = (simplex(0)._2 - simplex(n)._2).norm                // recompute the distance
                diff =  simplex(0)._1 - simplex(n)._1                      // recompute the difference
                if dist < toler || diff < toler then break ()              // check termination condition
            end for
        } // breakable

        (dist, diff)                                                // return distance and difference
    end improveSimplex

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the Non-Linear Programming (NLP) problem using the Nelder-Mead
     *  Simplex algorithm.
     *  @param x0     the given starting point
     *  @param step   the initial step size
     *  @param toler  the tolerance used for termination
     */
    def solve (x0: VectorD, step: Double = STEP, toler: Double = EPSILON): FuncVec =
        initSimplex (x0, step)
        debug ("solve", s"0:\tdist = MAX, diff = MAX, \n\tsimplex = ${stringOf (simplex)}")

        breakable {
            for k <- 1 to MAX_IT do
                val (dist, diff) = improveSimplex ()
                debug ("solve", s"$k:\tdist = $dist, diff = $diff, \n\tsimplex = ${stringOf (simplex)}")
                epochLoss += f_l
                if dist < toler || diff < toler then break ()       // check termination condition
            end for
        } // breakable

        val opt = simplex(n)
        println (s"solve: optimal function, vertex = $opt")
        opt                                                         // return the best functional value and vertex point
    end solve

end NelderMeadSimplex2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nelderMeadSimplex2Test` main function is used to test the `NelderMeadSimplex2` class.
 *  > runMain scalation.optimization.nelderMeadSimplex2Test
 */
@main def nelderMeadSimplex2Test (): Unit =

    val x0 = VectorD (1.0, 1.0)                                     // starting point

    println ("\nProblem 1: (x_0 - 2)^2 + (x_1 - 3)^2 + 1") 
    def f (x: VectorD): Double = (x(0) - 2.0) * (x(0) - 2.0) + (x(1) - 3.0) * (x(1) - 3.0) + 1.0

    val optimizer = new NelderMeadSimplex2 (f, 2)
    val opt = optimizer.solve (x0)                                  // optimal solution
    println (s"optimal solution = (f(x), x) = $opt")

    optimizer.plotLoss ()

end nelderMeadSimplex2Test

