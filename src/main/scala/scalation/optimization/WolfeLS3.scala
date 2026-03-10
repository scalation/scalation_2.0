
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Jun 19 01:30:58 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Modified Wolfe Line Search Optimizer
 *
 *  @see arch.library.northwestern.edu/concern/generic_works/jm214p59n?locale=en
 */

package scalation
package optimization

import scala.math.abs

import scalation.calculus.Differential.∇
import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `WolfeLS3` class performs an inexact line search on f to find (1) a point x
 *  that exhibits
 *      (1) SDC: sufficient decrease (f(x) enough less that f(0)) and
 *      (2) CC:  the slope at x is less steep than the slope at 0.
 *  That is, the line search looks for a value for x satisfying the two Wolfe conditions.
 *  
 *     f(x) <= f(0) + c1 * f'(0) * x      Wolfe condition 1 (Armijo condition)
 *  |f'(x)| <= |c2 * f'(0)|               Wolfe condition 2 (Strong version)
 *    f'(x) >= c2 * f'(0)                 Wolfe condition 2 (Weak version, more robust)
 *
 *  The it uses bisection (or interpolative search) to find an approximate local minimal point.
 *  Currently, the strong version is not supported.
 *  Note: c1 and c2 defaults below intended for Quasi Newton methods such as BFGS or L-BFGS
 *  
 *  @param f   the objective/loss function to minimize
 *  @param g   the gradient of the objective/loss function
 *  @param c1  constant for sufficient decrease (Wolfe condition 1: .0001 to .001)
 *  @param c2  constant for curvature/slope constraint (Wolfe condition 2: .9 to .8)
 *  @param c3  constant for noise control condition
 *  @param eg  estimate of gradient noise
 */
class WolfeLS3 (f: FunctionV2S, var g: FunctionV2V, c1: Double = 0.0001, c2: Double = 0.9,
                c3: Double = 0.0, eg: Double = 0.0):

    private val debug   = debugf ("WolfeLS3", true)                 // debug function
    private val MAX_IT  = 30                                        // maximum number of iterations

    if g == null then g = (x: VectorD) => ∇ (f)(x)                  // no function for gradient => use numerical

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether Wolfe condition 1, the Sufficient Decrease Condition (SDC) is satisfied.
     *  @param fx   the functional value of the original point
     *  @param fy   the functional value of the new point y = x + p * a
     *  @param a    the displacement in the search direction
     *  @param gxp  the dot product of the gradient vector g(x) and the search vector p
     */
    inline def wolfe1 (fx: Double, fy: Double, a: Double, gxp: Double): Boolean =
        fy <= fx + c1 * a * gxp

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether Wolfe condition 2, the Curvature Condition (CC) is satisfied.
     *  @param p    the search direction vector
     *  @param gy   the gradient at new point y = x + p * a
     *  @param gxp  the dot product of the gradient vector g(x) and the search vector p
     */
    inline def wolfe2 (p: VectorD, gy: VectorD, gxp: Double): Boolean =
        (gy dot p) >= c2 * gxp

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the noise control condition is satisfied.
     *  @param p   the search direction
     *  @param gx  the gradient at original point x
     *  @param gy  the gradient at new point y = x + p * a
     */
    inline def noiseControl (p: VectorD, gx: VectorD, gy: VectorD): Boolean =
        if c3 > 0.0 then abs (gy - gx dot p) < 2 * (1 + c3) * eg * p.norm 
        else false

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Two-Phase Armijo-Wolfe Line Search and Lengthening: Initial Phase, Algorithm 3.2 in 
     *  @see arch.library.northwestern.edu/concern/generic_works/jm214p59n?locale=en
     *  @param x     the current point
     *  @param p     the current search direction
     *  @param step  the initial step size
     */
    def lsearch (x: VectorD, p: VectorD, step: Double = 1.0): (Double, Double) =
        var split = true                                            // whether to do split phase
        val (fx, gx) = (f(x), g(x))                                 // function and gradient at x: f(x), g(x)
        val gxp = gx dot p                                          // gradient dot search direction
        debug ("lsearch", s"x = $x, fx = $fx, gx = $gx, p = $p, gxp = $gxp")
        var a  = step                                               // start for a = step size

        var l = 0.0                                                 // lower bound
        var u = Double.MaxValue                                     // upper bound

        var (go, it) = (true, 0)
        cfor (go && it < MAX_IT, it += 1) {

            val y = x + p * a                                       // new point
            val (fy, gy) = (f(y), g(y))                             // function and gradient at y: f(y), g(y)

            val wolf1 = wolfe1 (fx, fy, a, gxp)                     // Wolfe condition 1: SDC
            val wolf2 = wolfe2 (p, gy, gxp)                         // Wolfe condition 2: CC
            debug ("lsearch", s"(it = $it) before a = $a, y = $y, fy = $fy, gy = $gy, wolf1 = $wolf1, wolf2 = $wolf2")

            if ! wolf1 then                                         // Wolfe condition 1: SDC fails
                u = a
                a = (l + u) / 2.0                                   // backtrack
            else if noiseControl (gx, gy, p) then                   // noise control condition
                go = false
            else if ! wolf2 then                                    // Wolfe condition 2: CC fails
                l = a
                a = if u == Double.MaxValue then 2.0 * a else (l + u) / 2.0
            else
                split = false
                go = false
            debug ("lsearch", s"(it = $it) after  a = $a in [$l, $u]")
        } // cfor

        val b = a
        if split then splitPhase (x, fx, gx, p, gxp, a, b)
        else (a, b)
    end lsearch

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Two-Phase Armijo-Wolfe Line Search and Lengthening: Split Phase, Algorithm 3.3 in 
     *  @see arch.library.northwestern.edu/concern/generic_works/jm214p59n?locale=en
     *  @param x    the initial point
     *  @param fx   the functional value at x
     *  @param gx   the gradient at x
     *  @param p    the search direction
     *  @param gxp  the dot product if g(x) and p
     *  @param a_   the initial step length
     *  @param b_   the initial lengthening parameter
     */
    def splitPhase (x: VectorD, fx: Double, gx: VectorD, p: VectorD, gxp: Double,
                    a_ : Double, b_ : Double): (Double, Double) =
        var (a, b) = (a_, b_)
        while
            val y  = x + p * a                                      // new point based on a
            val fy = f(y)                                           // functional value
            debug ("splitPhase", s"a = $a, y = $y, fy = $fy")
            wolfe1 (fx, fy, a, gxp) == false                        // Wolfe condition 1: SDC
        do a /= 10.0                                                // backtrack

        while
            val y  = x + p * a                                      // new point based on b
            noiseControl (gx, g(y), p)                              // noise control condition
        do b *= 2                                                   // lengthen step

        (a, b)
    end splitPhase

end WolfeLS3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `wolfeLS3Test` main function is used to test the `WolfeLS3` class on scalar functions.
 *  > runMain scalation.optimization.wolfeLS3Test
 */
@main def wolfeLS3Test (): Unit =

    val z0 = VectorD (0.0)                                          // zero vector, the origin
    val p  = VectorD (1.0)                                          // direction to search in

    def f(x: VectorD): Double = (x(0) - 4.0) * (x(0) - 4.0) + 1.0      // no expansion phase
//  def f(x: VectorD): Double = (x(0) - 40.0) * (x(0) - 40.0) + 1.0    // requires expansion phase

    def g(x: VectorD): VectorD = VectorD (2.0 * (x(0) - 4.0))

    val wls    = new WolfeLS3 (f, g)
    val (a, b) = wls.lsearch (z0, p)
    val x      = z0 + p * a

    println ("\nProblem 1: (x - 4)^2 + 1") 
    println (s"start z0 = $z0, search direction = $p")
    println (s"displacement a     = $a")
    println (s"lengthening b      = $b")
    println (s"optimal x solution = $x")
    println (s"optimal f solution = ${f(x)}")

end wolfeLS3Test


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `wolfeLS3Test2` main function is used to test the `WolfeLS3` class on scalar functions.
 *  > runMain scalation.optimization.wolfeLS3Test2
 */
@main def wolfeLS3Test2 (): Unit =

    val z0 = VectorD (0.0, 0.0)                                     // zero vector, the origin
    val p  = VectorD (1.0, 1.0)                                     // direction to search in

    def f(x: VectorD): Double  = (x(0) - 2.0) * (x(0) - 2.0) + (x(1) - 3.0) * (x(1) - 3.0) + 1.0
    def g(x: VectorD): VectorD = VectorD (2.0 * (x(0) - 2.0), 2.0 * (x(1) - 3.0))

    val yy = VectorD (0.5, 0.5)
    banner (s"yy = $yy, fyy = ${f(yy)}, gyy = ${g(yy)}")

    val wls    = new WolfeLS3 (f, g)
    val (a, b) = wls.lsearch (z0, p)
    val x      = z0 + p * a

    println ("\nProblem 1: (x_0 - 2)^2 + (x_1 - 3)^2 + 1")
    println (s"start z0 = $z0, search direction = $p")
    println (s"displacement a     = $a")
    println (s"lengthening b      = $b")
    println (s"optimal x solution = $x")
    println (s"optimal f solution = ${f(x)}")

/*
    def f2(x: VectorD): Double  = x(0)/4 + 5*x(0)~^2 + x(0)~^4 - 9*x(0)~^2*x(1) + 3*x(1)~^2 + 2*x(1)~^4
    def g2(x: VectorD): VectorD = VectorD (0.25 + 10 * x(0) + 4.0 * x(0)~^3 - 18.0 * x(0) * x(1),
                                           9.0 * x(0)~^2 + 6.0 * x(1) + 8.0 * x(1)~^3)

    val wls2     = new WolfeLS3 (f2, g2)
    val (a2, b2) = wls2.lsearch (z0, p)
    val x2       = z0 + p * a2

    // @see http://math.fullerton.edu/mathews/n2003/gradientsearch/GradientSearchMod/Links/GradientSearchMod_lnk_5.html

    println ("\nProblem 4: x_0/4 + 5x_0^2 + x_0^4 - 9x_0^2 x_1 + 3x_1^2 + 2x_1^4")
    println (s"start z0 = $z0, search direction = $p")
    println (s"displacement a     = $a2")
    println (s"lengthening b      = $b2")
    println (s"optimal x solution = $x2")
    println (s"optimal f solution = ${f2(x2)}")
*/

end wolfeLS3Test2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `wolfeLS3Test3` main function is used to test the `WolfeLS3` class on vector functions.
 *  This test uses the Rosenbrock function.
 *  @see https://mikl.dk/post/2019-wolfe-conditions/
 *  > runMain scalation.optimization.wolfeLS3Test3
 */ 
@main def wolfeLS3Test3 (): Unit =

    def f (x: VectorD): Double = (1.0 - x(0))~^2 + 100.0 * (x(1) - x(0)~^2)~^2

    def g (x: VectorD): VectorD = VectorD (-2.0 * (1 - x(0)) - 400.0 * x(0) * (x(1) - x(0)~^2),
                                           200 * (x(1) - x(0)~^2))

    val z0 = VectorD (-2.2, 3)
    val p  = VectorD (1625.6, 368)       // -gradient at z0

    val wls    = new WolfeLS3 (f, g)
    val (a, b) = wls.lsearch (z0, p)
    val x      = z0 + p * a

    println ("\nProblem 1: (1 - x_0)^2 + 100 (x_1 - x_0^2)^2")
    println (s"start z0 = $z0, search direction = $p")
    println (s"displacement a     = $a")
    println (s"lengthening b      = $b")
    println (s"optimal x solution = $x")
    println (s"optimal f solution = ${f(x)}")

end wolfeLS3Test3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `wolfeLS3Test4` main function is used to test the `WolfeLS3` class on scalar functions.
 *  > runMain scalation.optimization.wolfeLS3Test4
 */
@main def wolfeLS3Test4 (): Unit =

    val aa = new VectorD (60)
    val ff = new VectorD (60)
    val ww = new VectorD (60)

    val x = VectorD (0.0, 0.0)                                      // zero vector, the origin
    val p = VectorD (1.0, 1.0)                                      // direction to search in

    banner ("Problem: (x_0 - 2)^2 + (x_1 - 3)^2 + 1")

    def f(x: VectorD): Double  = (x(0) - 2.0) * (x(0) - 2.0) + (x(1) - 3.0) * (x(1) - 3.0) + 1.0
    def g(x: VectorD): VectorD = VectorD (2.0 * (x(0) - 2.0), 2.0 * (x(1) - 3.0))

    val wls = new WolfeLS3 (f, g)
    val fx  = f(x)
    val gx  = g(x)
    val gxp = gx dot p
    println (s"x = $x, fx = $fx, gx = $gx")

    for i <- 0 until 60 do
        val a  = 0.1 * i
        val y  = x + p * a
        val fy = f(y)
        val gy = g(y)
        val wolf1 = wls.wolfe1 (fx, fy, a, gxp)
        val wolf2 = wls.wolfe2 (p, gy, gxp)
        println (s"a = $a, y = $y, fy = $fy, wolf1 = $wolf1, wolf2 = $wolf2")
        aa(i) = a
        ff(i) = fy
        ww(i) = 2.0 + is (wolf1) + 2.0 * is (wolf2)
    end for
    new Plot (aa, ff, ww, "fy vs. a")

end wolfeLS3Test4

