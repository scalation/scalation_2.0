
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yulong Wang
 *  @version 2.0
 *  @date    Thurs Jun  29 13:13:42 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Nelder-Mead Simplex Algorithm for Non-Linear Optimization
 *
 *  @see     DOI 10.1002/anac.200410015  by Singer and Singer 2004
 *  @see     The Dr. Thomas Harvey Rowan PhD dissertation 1990
 */

package scalation
package optimization

import scala.math.min
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NelderMeadSimplex` solves Non-Linear Programming (NLP) problems using
 *  the Nelder-Mead Simplex algorithm.  Given a function f and its dimension
 *  n,  the algorithm moves a simplex defined by n + 1 points in order to find
 *  an optimal solution.  The algorithm is derivative-free.
 *
 *  minimize    f(x)
 *
 *  The algorithm requires between 1 to n+2 function evaluations per iteration
 *  @param f  the vector-to-scalar objective function
 *  @param n  the dimension of the search space
 */
class NelderMeadSimplex (f: FunctionV2S, n: Int)
      extends Minimize:

    private val debug   = debugf ("NelderMeadSimplex", true)         // debug function
    private val flaw    = flawf ("NelderMeadSimplex")                // flaw function
    private val np1     = n + 1                                      // number of vertices/points in simplex
    private val simplex = Array.ofDim [FuncVec] (np1)                // simplex { vetices } used for search

    private val alpha = 1.0                                          // alpha (> 0)  parameter for reflection
    private val beta  = 0.5                                          // beta  (0, 1) parameter for contraction
    private val gamma = 2.0                                          // gamma (> 1)  parameter for expansion
    private val delta = 0.5                                          // delta (0, 1) parameter for shrinkage

    if n < 2 then flaw ("init", "requires at least a 2-dimensional problem")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize the search simplex by setting n + 1 vertices and computing
     *  their functional values.
     *  @param x0    the given starting point
     *  @param step  the step size
     */
    def initSimplex (x0: VectorD, step: Double): Unit =
        simplex(0) = (f(x0), x0)                                     // given starting point and its functional value
        for i <- 1 to n do
            val x = x0 + VectorD.oneAt (i-1, x0.dim) * step
            simplex(i) = (f(x), x)
        end for
        sort ()                                                      // order vertices high to low
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
    private def centroid (): VectorD =
        val c = new VectorD (n)                                      // the centroid of the simplex
        for i <- 1 to n do c += simplex(i)._2                        // add vertex points, except h=0
        val x_c = c / n.toDouble                                     // divide by # vertices - 1
        x_c
    end centroid

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reflect: compute the reflection point of the worst point (h=0) across
     *  the centroid.
     *  @param x_c  the best-side centroid of the simplex
     *  @param x_h  the lowest point
     */
    private inline def reflect (x_c: VectorD, x_h: VectorD): FuncVec =
        val x_r = x_c + (x_c - x_h) * alpha
        (f(x_r), x_r)
    end reflect

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand: compute the expansion point beyond the reflection point.
     *  @param x_c  the best-side centroid of the simplex
     *  @param x_r  the reflection point
     */
    private inline def expand (x_c: VectorD, x_r: VectorD): FuncVec =
        val x_e = x_c + (x_r - x_c) * gamma
        (f(x_e), x_e)
    end expand

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Contract: compute the outer contraction point between x_r and x_c.
     *  @param x_c  the best-side centroid of the simplex
     *  @param x_r  the reflection point
     */
    private inline def contractOut (x_c: VectorD, x_r: VectorD): FuncVec =
        val x_o = x_c + (x_r - x_c) * beta
        (f(x_o), x_o)
    end contractOut

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Contract: compute the inner contraction point between x_h and x_c.
     *  @param x_c  the best-side centroid of the simplex
     *  @param x_h  the lowest point
     */
    private inline def contractIn (x_c: VectorD, x_h: VectorD): FuncVec =
        val x_i = x_c + (x_h - x_c) * beta
        (f(x_i), x_i)
    end contractIn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shrink: fixing the best/lowest point (l=n), move the rest of the points
     *  toward it.
     */
    private def shrink (): Unit =
        val x_l = simplex(n)._2                                      // the best vertex point
        for i <- 0 until n do                                        // i != n as until is exclusive
            val x = x_l + (simplex(i)._2 - x_l) * delta
            simplex(i) = (f(x), x)                                   // updated vertex
        end for
    end shrink

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace the worst vertex (h=0) in the simplex with the new point.
     *  @param x_n  the new replacement point
     *  @param f_n  the new function value; otherwise time-consuming if objective function computationally expensive
     *  @param why  indicated which operation is the basis for replacement
     */
    private def replace (x_n: VectorD, f_n: Double, why: String): Unit =
        debug ("replace", s"the worst point ${simplex(0)} with the new point ($x_n, $f_n based on $why")
        simplex(0) = (f_n, x_n)
    end replace

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transform/improve the simplex by replacing the worst/highest vertex (x_h)
     *  with a better one found on the line containing x_h and the centroid (x_c).
     *  Try the reflection, expansion, outer contraction and inner contraction
     *  points, in that order.  If none succeeds, shrink the simplex and iterate.
     *  Return both distance between x_h (worst) and x_l (best).
     */
    def transform (): Double =
        val (f_h, x_h) = simplex(0)                                  // functional value for x_h (highest/worst)
        val f_s = simplex(1)._1                                      // functional value for x_s (second worst)
        val f_l = simplex(n)._1                                      // functional value for x_l (lowest/best)
        val x_c = centroid ()                                        // compute best-side centroid of simplex

        val (f_r, x_r) = reflect (x_c, x_h)                          // try to REFLECT the simplex
        if f_r < f_s then
            replace (x_r, f_r, "reflect")                            // accept REFLECT
            if f_r < f_l then
                val (f_e, x_e) = expand (x_c, x_r)                   // try to EXPAND
                if f_e < f_l then replace (x_e, f_e, "expand")       // accept EXPAND as greedy expansion else stick to reflect
        else                                                         // we have fr ≥ fs. REFLECT if it helps, and try to CONTRACT
            val (f_con, x_con) =
            if f_r < f_h then contractOut (x_c, x_r)                 // outer contraction
            else contractIn (x_c, x_h )                              // inner contraction
            if f_con < min (f_r, f_h) then
                replace (x_con, f_con, "contract")                   // accept CONTRACT, inside or outside
            else
                shrink ()
        sort ()                                                      // re-establish vertex order
        (simplex(0)._2 - simplex(n)._2).norm                         // distance between worst x_h and best x_l
    end transform

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the Non-Linear Programming (NLP) problem using the Nelder-Mead
     *  Simplex algorithm.
     *  @param x0    the given starting point
     *  @param step  the initial step size
     */
    def solve (x0: VectorD, step: Double = 1): FuncVec =
        initSimplex (x0, step)
        debug ("solve", s"simplex = ${stringOf (simplex)}")

        var (k, go) = (1, true)
        cfor (k <= MAX_IT && go, k += 1) {
            val dist = transform ()
            debug ("solve", s"$k:\tdist = $dist, \n\tsimplex = ${stringOf (simplex)}")
            if dist < TOL then go = false                            // check termination condition
        } // cfor
        val opt = simplex(n)
        println (s"solve: optimal function, vertex = $opt")
        opt                                                          // return the best functional value and vertex point
    end solve

end NelderMeadSimplex


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nelderMeadSimplexTest` main function is used to test the `NelderMeadSimplex` class.
 *  > runMain scalation.optimization.nelderMeadSimplexTest
 */
@main def nelderMeadSimplexTest (): Unit =

    val x0 = VectorD (1.0, 1.0)                                     // starting point

    banner ("Problem 1: (x_0 - 2)^2 + (x_1 - 3)^2 + 1")
    def f (x: VectorD): Double = (x(0) - 2)~^2 + (x(1) - 3)~^2 + 1

    val optimizer = new NelderMeadSimplex (f, 2)
    val opt = optimizer.solve (x0)                                  // optimal solution
    println (s"optimal solution = (f(x), x) = $opt")

end nelderMeadSimplexTest

