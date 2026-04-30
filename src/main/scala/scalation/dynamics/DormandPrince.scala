
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Mar 29 14:59:50 EDT 2010
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    (4,5)-Order Dormand-Prince ODE Integrator (DOPRI) or ode45
 */

package scalation
package dynamics

import scala.annotation.nowarn
import scala.math.{abs, E, pow}
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DormandPrince` object provides a state-of-the-art numerical ODE solver.
 *  Given an unknown, time-dependent function y(t) governed by an Ordinary
 *  Differential Equation (ODE) of the form
 *      d/dt y(t) = y'(t) = f(t, y)
 *      y(t0) = y0
 *  compute y(t) using a (4,5)-order Dormand-Prince Integrator (DOPRI) or ode45.
 *  Note: the integrateV method for a system of separable ODEs is mixed in from the
 *  `Integrator` trait.
 *  @see http://adorio-research.org/wordpress/?p=6565
 */
object DormandPrince
       extends Integrator:

    private val debug = debugf ("DormandPrince", false)        // debug function
    private val flaw  = flawf ("DormandPrince")                // flaw function
    private val ovf   = Double.MaxValue / 10.0                 // too big, may overflow

    /** Extended Butcher Tableau @see http://en.wikipedia.org/wiki/Dormand–Prince_method
     */
    val a21                            = 1/5.0
    val (a31, a32)                     = (3/40.0, 9/40.0)
    val (a41, a42, a43)                = (44/45.0, -56/15.0, 32/9.0)
    val (a51, a52, a53, a54)           = (19372/6561.0, -25360/2187.0, 64448/6561.0, -212/729.0)
    val (a61, a62, a63, a64, a65)      = (9017/3168.0, -355/33.0, 46732/5247.0, 49/176.0, -5103/18656.0)
    val (a71, a72, a73, a74, a75, a76) = (35/384.0, 0.0, 500/1113.0, 125/192.0, -2187/6784.0, 11/84.0)
 
    val b1 =  35/384.0                               // weights for 4th order solution
    val b2 =  0.0
    val b3 =  500/1113.0
    val b4 =  125/ 192.0
    val b5 = -2187/6784.0
    val b6 =  11/84.0
    val b7 =  0.0
 
    val b_1 =  5179/57600.0;   val b_b1 = b1 - b_1   // weights for 5th order used for error estimation
    val b_2 =  0.0;            val b_b2 = 0.0
    val b_3 =  7571/16695.0;   val b_b3 = b3 - b_3
    val b_4 =  393/640.0;      val b_b4 = b4 - b_4
    val b_5 = -92097/339200.0; val b_b5 = b5 - b_5
    val b_6 =  187/2100.0;     val b_b6 = b6 - b_6
    val b_7 =  1/40.0;         val b_b7 = b7 - b_7

    val c2 = 1/5.0                                   // time offsets ci * h for stage derivatives where c1 = 0
    val c3 = 3/10.0
    val c4 = 4/5.0
    val c5 = 8/9.0
    val c6 = 1.0
    val c7 = 1.0
 
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute y(t) governed by a differential equation using numerical integration
     *  of the derivative function f(t, y) using a (4,5)-order Dormand-Prince method to
     *  return the value of y(t) at time t.
     *  @param f     the derivative function f(t, y)
     *  @param y0    value of the y-function at time t0, y0 = y(t0)
     *  @param t     the time value at which to compute y(t)
     *  @param t0    the initial time
     *  @param step  the middle step size
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

        var h  = hmax                                                   // initial step size
        var tn = t0                                                     // initialize nth time tn to t0
        var y  = y0                                                     // initialize y at t0 to y0

        var k1, k2, k3, k4, k5, k6, k7 = 0.0
 
        breakable {
            for n <- 1 to maxSteps do
                k1 = f(tn,          y)
                k2 = f(tn + c2 * h, y + (a21*k1) * h)
                k3 = f(tn + c3 * h, y + (a31*k1 + a32*k2) * h)
                k4 = f(tn + c4 * h, y + (a41*k1 + a42*k2 + a43*k3) * h)
                k5 = f(tn + c5 * h, y + (a51*k1 + a52*k2 + a53*k3 + a54*k4) * h)
                k6 = f(tn +      h, y + (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5) * h)
                k7 = f(tn +      h, y + (a71*k1 + a72*k2 + a73*k3 + a74*k4 + a75*k5 + a76*k6) * h)
                debug ("integrate2", s"stage derivatives k = ${VectorD (k1, k2, k3, k4, k5, k6, k7)}")

                error = abs ( b_b1 * k1 + b_b3 * k3 + b_b4 * k4 + b_b5 * k5 + b_b6 * k6 + b_b7 * k7 )
                if error < tol then
                    y  += h * (b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
                    tn += h
                debug ("integrate2", s"for step n = $n: error = $error, y = $y")
//              if n == 10 then System.exit (0)
 
                val delta = 0.84 * pow (tol / error, 0.2)               // error control
                h *= (if delta <= 0.1 then 0.1                          // adjust the step size
                      else if delta >= 4.0 then 4.0
                      else delta)
                if h > hmax then h = hmax
 
                if tn >= t then break ()                                // at or past end time
                else if tn + h > t then h = t - tn                      // complete last time increment
                else if h < hmin then break ()                          // step size too small

                if abs (y) > ovf then flaw ("integrate2", s"probable overflow since y = $y")
                if n % 100 == 0 then println (s"integrate2: step $n: tn = $tn, y = $y")
            end for
        } // breakable
        y                                  // the value of the function at time t, y = f(t)
    end integrate2

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute y(t), a vector, governed by a system of differential equations using
     *  numerical integration of the derivative function f(t, y) using a (4,5)-order
     *  Dormand-Prince method to return the value of y(t) at time t.
     *  @param f     the array of derivative functions [f(t, y)] where y is a vector
     *  @param y0    the value of the y-function at time t0, y0 = y(t0)
     *  @param t     the time value at which to compute y(t)
     *  @param t0    the initial time
     *  @param step  the step size
     */
    def integrateVV (f: Array [DerivativeV], y0: VectorD, t: Double,
                     t0: Double = 0.0, step: Double = defaultStepSize): VectorD =

        val maxSteps = 1000
        val hmin     = 0.5 * step
        val hmax     = 2.0 * step
        val tol      = 1E-5
        var h        = hmax
        var tn       = t0
        val y        = y0.copy

        val k1 = new VectorD (y.dim)
        val k2 = new VectorD (y.dim)
        val k3 = new VectorD (y.dim)
        val k4 = new VectorD (y.dim)
        val k5 = new VectorD (y.dim)
        val k6 = new VectorD (y.dim)
        val k7 = new VectorD (y.dim)

        var ti = 0.0
        var yi: VectorD = null

        breakable {
            for n <- 1 to maxSteps do
                ti = tn
                cfor (0, y.dim) { j => yi = y
                                  k1(j) = f(j)(ti, yi) }
                debug ("integrateVV", s"ith stage derivatives @ ti = $ti is k1 = $k1")

                ti = tn + c2 * h
                cfor (0, y.dim) { j => yi = y + (k1*a21) * h
                                  k2(j) = f(j)(ti, yi) }
                debug ("integrateVV", s"ith stage derivatives @ ti = $ti is k2 = $k2")

                ti = tn + c3 * h
                cfor (0, y.dim) { j => yi =  y + (k1*a31 + k2*a32) * h
                                  k3(j) = f(j)(ti, yi) }
                debug ("integrateVV", s"ith stage derivatives @ ti = $ti is k3 = $k3")

                ti = tn + c4 * h
                cfor (0, y.dim) { j => yi = y + (k1*a41 + k2*a42 + k3*a43) * h
                                  k4(j) = f(j)(ti, yi) }
                debug ("integrateVV", s"ith stage derivatives @ ti = $ti is k4 = $k4")

                ti = tn + c5 * h
                cfor (0, y.dim) { j => yi = y + (k1*a51 + k2*a52 + k3*a53 + k4*a54) * h
                                  k5(j) = f(j)(ti, yi) }
                debug ("integrateVV", s"ith stage derivatives @ ti = $ti is k5 = $k5")

                ti = tn + h
                cfor (0, y.dim) { j => yi = y + (k1*a61 + k2*a62 + k3*a63 + k4*a64 + k5*a65) * h
                                  k6(j) = f(j)(ti, yi) }
                debug ("integrateVV", s"ith stage derivatives @ ti = $ti is k6 = $k6")

                ti = tn + h
                cfor (0, y.dim) { j => yi = y + (k1*a71 + k2*a72 + k3*a73 + k4*a74 + k5*a75 + k6*a76) * h
                                  k7(j) = f(j)(ti, yi) }
                debug ("integrateVV", s"ith stage derivatives @ ti = $ti is k7 = $k7")

//              debug ("integrateVV", s"h = $h, stage derivatives k = ${MatrixD (k1, k2, k3, k4, k5, k6, k7).transpose}")

                error = abs ( (b1-b_1) * k1.norm + (b3-b_3) * k3.norm + (b4-b_4) * k4.norm +
                              (b5-b_5) * k5.norm + (b6-b_6) * k6.norm + (b7-b_7) * k7.norm )
                if error < tol then
                    cfor (0, y.dim) { j => y(j) += h * (b1*k1(j) + b3*k3(j) + b4*k4(j) + b5*k5(j) + b6*k6(j)) }
                    tn += h
                debug ("integrate2", s"for step n = $n: error = $error, y = $y")
//              if n == 4 then System.exit (0)

                val delta = 0.84 * pow (tol / error, 0.2)               // error control
                h *= (if delta <= 0.1 then 0.1                          // adjust the step size
                      else if delta >= 4.0 then 4.0
                      else delta)
                if h > hmax then h = hmax

                if tn >= t then break ()                                // at or past end time
                else if tn + h > t then h = t - tn                      // complete last time increment
                else if h < hmin then break ()                          // step size too small

                if y.mag > ovf then flaw ("integrate2", s"probable overflow since y = $y")
                if n % 100 == 0 then println (s"integrate2: step $n: tn = $tn, y = $y")
            end for
        } // breakable
        y                       // the value of the function at time t, y = f(t)
    end integrateVV
 
end DormandPrince

import DormandPrince._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dormandPrinceTest` main function is used to test the `DormandPrince` object.
 *  This test is for non-stiff equations.  Compare this ode45 with `RungeKutta`.
 *  > runMain scalation.dynamics.dormandPrinceTest
 */
@main def dormandPrinceTest (): Unit =

    val y0 = 1.0
    val t_ = 2.0

    banner (s"Test ODE Solver Dormand-Prince compute y(2) where y0 = y(0) = 1")

    banner ("Test `integrate` on y' = f(t, u) = 2.0 * t")
    @nowarn def derv1 (t: Double, y: Double) = y             // solution to differential equation is e^t
    var y_ = (t: Double) => E~^t                             // symbolic solution
    var y  = integrate (derv1, y0, t_)                       // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct t~^2 + 1 = ${y_(t_)}")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrate` on y' = f(t, u) = y")
    @nowarn def derv2 (t: Double, y: Double) = 2.0 * t       // f(t, y) for differential equation is t^2 + 1
    y_ = t => t~^2 + 1                                       // symbolic solution
    y  = integrate (derv2, y0, t_)                           // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct: E~^t = ${y_(t_)} ")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrate` on y' = f(t, u) = t + y")
    def derv3 (t: Double, y: Double) = t + y                  // f(t, y) for ordinary differential equation
    y_ = t => 2*E~^t - t - 1                                  // symbolic solution
    y  = integrate (derv3, y0, t_)                            // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct: 2*E~^t - t - 1 = ${y_(t_)}")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrateV` on y' = f(t, u) = [2.0 * t, y]")
    println (s"\n==> at t = $t_: y = ${integrateV (Array (derv1, derv2), VectorD (1.0, 2.0), t_)}")

end dormandPrinceTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dormandPrinceTest2` main function is used to test the `DormandPrince` class.
 *  Test the `integrateVV` methods for systems of ODEs problems.
 *  @see web.uvic.ca/~tbazett/diffyqs/sec_introtosys.html
 *  > runMain scalation.dynamics.dormandPrinceTest2
 */
@main def dormandPrinceTest2 (): Unit =

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
        yy(i) = y_(t(i))                                      // symbolic solution
        y(i)  = integrateVV (odes, y(0), t(i))                // numeric solution
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

end dormandPrinceTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dormandPrinceTest3` main function is used to test the `DormandPrince` object.
 *  This test is for a system of non-stiff equations.  Compare this ode45 with Runge-Kutta
 *  and ode23.
 *  @see http://www.mathworks.com/help/techdoc/ref/ode23.html (Example 1)
 *
 *  Runge-Kutta-2  at tt = 12.0: p = VectorD(-0.759943, -0.647107, 0.852941)
 *  Runge-Kutta-3  at tt = 12.0: p = VectorD(-0.760170, -0.646877, 0.852844)
 *  Runge-Kutta-4  at tt = 12.0: p = VectorD(-0.744718, -0.739402, 0.867864)
 *  Runge-Kutta-5  at tt = 12.0: p = VectorD(-0.760170, -0.646877, 0.852844)
 *  Dormand-Prince at tt = 12.0: p = VectorD(-0.705398, -0.708812, 0.863847)
 *
 *  > runMain scalation.dynamics.dormandPrinceTest3
 */
@main def dormandPrinceTest3 (): Unit =

/*
    banner ("Test DormandPrince on y' = t + y with y0 = 1.24 at t_ = 1.0")
    def derv1 (t: Double, y: Double) = t + y
    val y0   = 1.24
    println (s"\n==> at t = $t_: y = ${integrate (derv1, y0, 1.0)}")
*/

    banner ("Test DormandPrince on system of ODEs with y0 = 1.24 at t_ = 1.0")
    @nowarn def dx_dt (t: Double, p: VectorD) =  p(1) * p(2)
    @nowarn def dy_dt (t: Double, p: VectorD) = -p(0) * p(2)
    @nowarn def dz_dt (t: Double, p: VectorD) = -.51 * p(0) * p(1)
    val odes = Array [DerivativeV] (dx_dt, dy_dt, dz_dt)

    val ti  = 0.2
    var p   = VectorD (0.0, 1.0, 1.0)
    val p_r = new MatrixD (61, 3); for k <- p.indices do p_r(0, k) = p(k)
    val tt  = new VectorD (61)
    for i <- 1 to 60 do
        tt(i) = ti * i
        p = integrateVV (odes, p, ti)
        println (s"\n==> at tt = ${tt(i)}: p = $p")
        for k <- p.indices do p_r(i, k) = p(k)
//      p_r(i) = p
    end for

    new Plot (tt, p_r(?, 0), p_r(?, 1), "Plot p(0), p(1) vs. t", lines = true)

end dormandPrinceTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dormandPrinceTest4` main function is used to test the `DormandPrince` object.
 *  This test is for stiff equations.  Compare ode45 with ode23s.
 *  > runMain scalation.dynamics.dormandPrinceTest4
 */
@main def dormandPrinceTest4 (): Unit =

    println ("dormandPriceTest4 not yet implemented")         // FIX - implement

end dormandPrinceTest4

