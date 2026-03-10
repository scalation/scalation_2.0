
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Dec 28 14:29:00 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    (p,q)th-Order Explicit, Adpative Stepsize (Embedded) Runge-Kutta Integrators (RKpq)
 *
 *  @see www.mathworks.com/help/matlab/math/choose-an-ode-solver.html
 *  @see en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
 */

package scalation
package dynamics

import scala.annotation.nowarn
import scala.math.{abs, E, pow}
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RungeKutta3` class provides implementations of several Runge-Kutta numerical
 *  ODE solvers.  Given an unknown, time-dependent function y(t) governed by an
 *  Ordinary Differential Equation (ODE) of the form:
 *      d/dt y(t) = y'(t) = f(t, y)
 *      y(t0) = y0
 *  Note: the `integrateV` method for a system of separable ODEs is mixed in from the
 *  `Integrator` trait.
 *  The ODE method is defined by its Extended Butcher Tablaeu (a, b, b_, c).
 *  @param name  the name the ODE solver
 *  @param a     the lower triangular matrix of constants multiplying the stage derivatives
 *  @param b     the vector of constants for computing the weighted average of stage derivatives
 *  @param b_    same for the embedded solver
 *  @param c     the vector of constants for shifting time
 */
class RungeKutta3 (val name: String, a: MatrixD, b: VectorD, b_ : VectorD, c: VectorD)
       extends Integrator:

    private val debug = debugf ("RungeKutta3", false)                           // debug function
    private val flaw  = flawf ("RungeKutta3")                                   // flaw function
    private val ovf   = Double.MaxValue / 10.0                                  // too big, may overflow
    private val s     = b.dim                                                   // number of stages
    private val b_b   = b - b_                                                  // difference in contants

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute y(t) governed by a differential equation using numerical integration
     *  of the derivative function f(t, y) using a 4th-order Runge-Kutta method to
     *  return the value of y(t) at time t.
     *  @param f     the derivative function f(t, y) where y is a scalar
     *  @param y0    the value of the y-function at time t0, y0 = y(t0)
     *  @param t     the time value at which to compute y(t)
     *  @param t0    the initial time
     *  @param step  the step size
     */
    def integrate (f: Derivative, y0: Double, t: Double,
                   t0: Double = 0.0, step: Double = defaultStepSize): Double =
        integrate2 (f, y0, t, 0.5 * step, 2.0 * step, t0)
    end integrate

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute y(t) governed by a differential equation using numerical integration
     *  of the derivative function f(t, y) using a (4,5)-order Dormand-Prince method to
     *  return the value of y(t) at time t.  The method provides more customization
     *  options.
     *  @param f         the derivative function f(t, y)
     *  @param y0        value of the y-function at time t0, y0 = y(t0)
     *  @param t         the time value at which to compute y(t)
     *  @param hmin      the minimum step size
     *  @param hmax      the maximum step size
     *  @param t0        the initial time
     *  @param tol       the tolerance
     *  @param maxSteps  the maximum number of steps
     */
    def integrate2 (f: Derivative, y0: Double, t: Double, hmin: Double = 0.01, hmax: Double = 1.0,
                   t0: Double = 0.0, tol: Double = 1E-5, maxSteps: Int = 1000): Double =

        var h  = hmax                                                           // initial step size
        var tn = t0                                                             // initialize nth time tn to t0
        var y  = y0                                                             // initialize y at t0 to y0
        val k  = new VectorD (s)                                                // s stage derivatives x h

        breakable {
            for n <- 1 to maxSteps do
                cfor (0, s) { i =>                                              // for each stage
                    val ti = tn + c(i) * h                                      // time for stage i
                    val yi = y + Σ (0, i) { l => a(i, l) * k(l) * h }           // y-scalar for stage i
                    k(i)   = f(ti, yi)                                          // i-ith stage derivative x h
                } // cfor
                debug ("integrate2", s"k = $k")

                error = abs (b_b dot k)                                         // estimate the error
                if error < tol then
                    y += h * (b dot k)                                          // update scalar y
                    tn += h                                                     // move ahead in time
                debug ("integrate2", s"for step n = $n: error = $error, y = $y")
//              if n == 10 then System.exit (0)                                 // for debugging

                val delta = 0.84 * pow (tol / error, 0.2)                       // error control
                h *= (if delta <= 0.1 then 0.1                                  // adjust the step size
                      else if delta >= 4.0 then 4.0
                      else delta)
                if h > hmax then h = hmax

                if tn >= t then break ()                                        // at or past end time
                else if tn + h > t then h = t - tn                              // complete last time increment
                else if h < hmin then break ()                                  // step size too small

                if abs (y) > ovf then flaw ("integrate2", s"probable overflow since y = $y")
                if n % 100 == 0 then println (s"integrate2: step $n: tn = $tn, y = $y")
            end for
        } // breakable
        y                                  // the value of the function at time t, y
    end integrate2

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute y(t), a vector, governed by a system of differential equations using
     *  numerical integration of the derivative function f(t, y) using a 4th-order
     *  Runge-Kutta method to return the value of y(t) at time t.
     *  @param f     the array of derivative functions [f(t, y)] where y is a vector
     *  @param y0    the value of the y-function at time t0, y0 = y(t0)
     *  @param t     the time value at which to compute y(t)
     *  @param t0    the initial time
     *  @param step  the step size
     */
    def integrateVV (f: Array [DerivativeV], y0: VectorD, t: Double,
                     t0: Double = 0.0, step: Double = defaultStepSize): VectorD =

// FIX - has a bug - should give the same r4sults as DormandPrince.scala @see Test2
// likely error in computing k(j. i)
// note: y + (j, x) add x only for jth element

        val maxSteps = 1000
        val hmin     = 0.5 * step
        val hmax     = 2.0 * step
        val tol      = 1E-5
        var h        = hmax
        var tn       = t0
        val y        = y0.copy
        val k        = new MatrixD (y0.dim, s)                                  // for each variable, s stage derivatives

        breakable {
            for n <- 1 to maxSteps do

                cfor (0, s) { i =>                                              // for each stage i
                    val ti = tn + c(i) * h                                      // time for stage i
                    cfor (0, y.dim) { j =>                                      // for each variable j
                        val yi = y + (j, h * Σ (0, i) { l => a(i, l) * k(j, l) })  // y-scalar for stage i
//                      debug ("integrateVV", s"for i = $i, j = $j: yi = $yi")
                        k(j, i)  = f(j)(ti, yi)                                 // i-ith stage derivative x h
                    } // cfor
                    debug ("integrateVV", s"ith stage derivatives @ ti = $ti is k(?, $i) = ${k(?, i)}")
                } // cfor

                error = abs ( Σ (0, s) { i => (b(i) - b_(i)) * k(?, i).norm })  // estimate the error
                if error < tol then
                    cfor (0, y.dim) { j => y(j) += h * (b dot k(j)) }           // update scalar y
                    tn += h                                                     // move ahead in time
                debug ("integrateVV", s"for step n = $n: error = $error, y = $y")
//              if n == 4 then System.exit (0)

                val delta = 0.84 * pow (tol / error, 0.2)                       // error control
                h *= (if delta <= 0.1 then 0.1                                  // adjust the step size
                      else if delta >= 4.0 then 4.0
                      else delta)
                if h > hmax then h = hmax

                if tn >= t then break ()                                        // at or past end time
                else if tn + h > t then h = t - tn                              // complete last time increment
                else if h < hmin then break ()                                  // step size too small

                if y.mag > ovf then flaw ("integrateVV", s"probable overflow since y = $y")
                if n % 100 == 0 then println (s"integrateVV: step $n: tn = $tn, y = $y")
            end for
        } // breakable
        y                                  // the value of the function at time t, y
    end integrateVV

end RungeKutta3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RungeKutta3` object provides several Explicit, Fixed Stepsize Runge-Kutta ODE solvers.
 *  The ODE method is defined by its Extended Butcher Tablaeu (a, b, b_, c).
 *  @see en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
 */
object RungeKutta3:

    // Compute y(t) using a (2,3)-order, 4-stage Runge-Kutta Integrator (RK23).
    // Bogacki–Shampine Runge-Kutta method
    // = new RungeKutta3 ("rk23", rk23_a, rk23_b, rk23_b_, rk23_c)

    val rk23_a  = MatrixD.low (4) (0.5,
                              0.0, 0.75,
                              2/9.0, 1/3.0, 4/9.0)
    val rk23_b  = VectorD (2/9.0, 1/3.0, 4/9.0, 0.0)
    val rk23_b_ = VectorD (7/24.0, 0.25, 1/3.0, 0.125)
    val rk23_c  = VectorD (0.0, 0.5, 0.75, 1.0)


    // Compute y(t) using a (4,5)-order, 7-stage Runge-Kutta Integrator (RK45).
    // Dormand-Prince Runge-Kutta method
    // = new RungeKutta3 ("rk45", rk45_a, rk45_b, rk45_b_, rk45_c)

    val rk45_a  = MatrixD.low (7) (1/5.0,
                                   3/40.0, 9/40.0,
                                   44/45.0, -56/15.0, 32/9.0,
                                   19372/6561.0, -25360/2187.0, 64448/6561.0, -212/729.0,
                                   9017/3168.0, -355/33.0, 46732/5247.0, 49/176.0, -5103/18656.0,
                                   35/384.0, 0.0, 500/1113.0, 125/192.0, -2187/6784.0, 11/84.0)
    val rk45_b  = VectorD (35/384.0, 0.0, 500/1113.0, 125/192.0, -2187/6784.0, 11/84.0, 0.0)
    val rk45_b_ = VectorD (5179/57600.0, 0.0, 7571/16695.0, 393/640.0, -92097/339200.0, 187/2100.0, 1/40.0)
    val rk45_c  = VectorD (0.0, 1/5.0, 3/10.0, 4/5.0, 8/9.0, 1.0, 1.0)


end RungeKutta3

import RungeKutta3._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rungeKutta3Test` main function is used to test the `RungeKutta3` class.
 *  Test the `integrate` and `integrateV` methods for one dimensional and separable
 *  ODE problems.
 *  > runMain scalation.dynamics.rungeKutta3Test
 */
@main def rungeKutta3Test (): Unit =

    val y0 = 1.0
    val t_ = 2.0

//  val solver = new RungeKutta3 ("rk23", rk23_a, rk23_b, rk23_b_, rk23_c)
    val solver = new RungeKutta3 ("rk45", rk45_a, rk45_b, rk45_b_, rk45_c)

    banner (s"Test ODE Solver ${solver.name} compute y(2) where y0 = y(0) = 1")

    banner ("Test `integrate` on y' = f(t, y) = y")
    @nowarn def derv1 (t: Double, y: Double) = y               // f(t, y) for differential equation is e^t
    var y_ = (t: Double) => E~^t                               // symbolic solution
    var y  = solver.integrate (derv1, y0, t_)                  // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct: E~^t = ${y_(t_)} ")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrate` on y' = f(t, y) = 2.0 * t")
    @nowarn def derv2 (t: Double, y: Double) = 2.0 * t         // solution to differential equation is t^2 + 1
    y_ = t => t~^2 + 1                                         // symbolic solution
    y  = solver.integrate (derv2, y0, t_)                      // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct t~^2 + 1 = ${y_(t_)}")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrate` on y' = f(t, y) = t + y")
    def derv3 (t: Double, y: Double) = t + y                    // f(t, y) for ordinary differential equation
    y_ = t => 2*E~^t - t - 1                                    // symbolic solution
    y  = solver.integrate (derv3, y0, t_)                       // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct: 2*E~^t - t - 1 = ${y_(t_)}")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrateV` on y' = f(t, y) = [2.0 * t, y]")
    println (s"\n==> at t = $t_: y = ${solver.integrateV (Array (derv1, derv2), VectorD (1.0, 2.0), t_)}")

end rungeKutta3Test


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rungeKutta3Test2` main function is used to test the `RungeKutta3` class.
 *  Test the `integrateVV` methods for systems of ODEs problems.
 *  @see web.uvic.ca/~tbazett/diffyqs/sec_introtosys.html
 *  > runMain scalation.dynamics.rungeKutta3Test2
 */
@main def rungeKutta3Test2 (): Unit =

//  val solver = new RungeKutta3 ("rk23", rk23_a, rk23_b, rk23_b_, rk23_c)
    val solver = new RungeKutta3 ("rk45", rk45_a, rk45_b, rk45_b_, rk45_c)

    banner (s"Test ODE Solver ${solver.name} compute y(1) where y0 = y(0) = 1,24")

    @nowarn def dy0_dt (t: Double, y: VectorD) = y(0)
    @nowarn def dy1_dt (t: Double, y: VectorD) = y(0) - y(1)
    val odes = Array [DerivativeV] (dy0_dt, dy1_dt)

    def y_(t: Double): VectorD = VectorD (E~^t, 0.5 * E~^t + 1.5 * E~^(-t))

    banner ("Test `integrateVV` on y' = f(t, y)")

    val t  = new VectorD (51)
    val y  = new MatrixD (t.dim, 2)
    val yy = new MatrixD (t.dim, 2)
    t(0)  = 0.0
    y(0)  = VectorD (1.0, 2.0)
    yy(0) = y_(0)
    for i <- 1 to 50 do
        t(i)  = i * 0.2
        yy(i) = y_(t(i))                                         // symbolic solution
        y(i)  = solver.integrateVV (odes, y(0), t(i))            // numeric solution
    end for

    println (s"t  = $t")
    println (s"yy = $yy")
    println (s"y  = $y")

    new Plot (t, yy(?, 0), y(?, 0), "Plot yy(?, 0), y(?, 0) vs. t", lines = true)
    new Plot (t, yy(?, 1), y(?, 1), "Plot yy(?, 1), y(?, 1) vs. t", lines = true)

    val t_ = t.last
    val i_ = t.dim - 1
    println (s"\n==> at t = $t_: y = ${y(i_)}")
    println (s"\n==> correct = ${y_(t_)}")
    println (s"\n==> error = ${y_(t_) - y(i_)}")

end rungeKutta3Test2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rungeKutta3Test3` main function is used to test the `RungeKutta3` class.
 *  Test the `integrateVV` methods for systems of ODEs problems.
 *  @see www.mathworks.com/help/techdoc/ref/ode23.html (Example 1)
 *  > runMain scalation.dynamics.rungeKutta3Test3
 */
@main def rungeKutta3Test3 (): Unit =

//  val solver = new RungeKutta3 ("rk23", rk23_a, rk23_b, rk23_b_, rk23_c)
    val solver = new RungeKutta3 ("rk45", rk45_a, rk45_b, rk45_b_, rk45_c)

    banner (s"Test ODE Solver ${solver.name} compute y(1) where y0 = y(0) = 1,24")

/*
    banner ("Test RungeKutta on y' = t + y with y0 = 1.24 at t_ = 1.0")
    def derv1 (t: Double, y: Double) = t + y
    val y0 = 1.24
    println (s"\n==> at t = $t_: y = ${solver.integrate (derv1, y0, 1.0)}")
*/

    banner ("Test RungeKutta on System of ODEs with y0 = 1.24 at t_ = 1.0")

    @nowarn def dx_dt (t: Double, p: VectorD) =  p(1) * p(2)
    @nowarn def dy_dt (t: Double, p: VectorD) = -p(0) * p(2)
    @nowarn def dz_dt (t: Double, p: VectorD) = -0.51 * p(0) * p(1)
    val odes = Array [DerivativeV] (dx_dt, dy_dt, dz_dt)

    val ti  = 0.2
    var p   = VectorD (0.0, 1.0, 1.0)
    val p_r = new MatrixD (61, 3); for k <- p.indices do p_r(0, k) = p(k)
    val tt  = new VectorD (61)
    for i <- 1 to 60 do
        tt(i) = ti * i
        p = solver.integrateVV (odes, p, ti)
        println (s"\n==> at tt = ${tt(i)}, p = $p")
        for k <- p.indices do p_r(i, k) = p(k)
//      p_r(i) = p
    end for

    new Plot (tt, p_r(?, 0), p_r(?, 1), "Plot p(0), p(1) vs. t", lines = true)

end rungeKutta3Test3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rungeKutta3Test3` main function is used to test the `RungeKutta3` class.
 *  Test the values in the Extended Butcher Tableau (a, b, b_, c) for
 *  `rk45` vs. `DormandPrince`.
 *  > runMain scalation.dynamics.rungeKutta3Test3
@main def rungeKutta3Test3 (): Unit =

    import DormandPrince._

    banner("Check that Extended Butcher Tableau (a, b, b_, c) is the same for rk45 and DormandPrince")

    val dp_a = MatrixD.low (7) (a21,
                                a31, a32,
                                a41, a42, a43,
                                a51, a52, a53, a54,
                                a61, a62, a63, a64, a65,
                                a71, a72, a73, a74, a75, a76)
    assert (dp_a =~ rk45.a)
    println ("Checked a  matrix: dp_a  =~ rk45.a")

    val dp_b = VectorD (b1, b2, b3, b4, b5, b6, b7)
    assert (dp_b =~ rk45.b)
    println ("Checked b  vector: dp_b  =~ rk45.b")

    val dp_b_ = VectorD (b_1, b_2, b_3, b_4, b_5, b_6, b_7)
    assert (dp_b_ =~ rk45.b_)
    println ("Checked b_ vector: dp_b_ =~ rk45.b_")

    val dp_c = VectorD (0.0, c2, c3, c4, c5, c6, c7)
    assert (dp_c =~ rk45.c)
    println ("Checked c  vector: dp_c  =~ rk45.c")

end rungeKutta3Test3
 */

