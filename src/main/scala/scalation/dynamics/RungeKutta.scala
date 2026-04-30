
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Oct 25 18:41:26 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    4th-Order Runge-Kutta Integrator (RK4) or ode44
 */

package scalation
package dynamics

import scala.annotation.nowarn
import scala.math.{abs, E, round}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RungeKutta` object provides an implementation of a classical Runge-Kutta
 *  numerical ODE solver.  Given an unknown, time-dependent function y(t) governed by
 *  an Ordinary Differential Equation (ODE) of the form:
 *      d/dt y(t) = y'(t) = f(t, y)
 *      y(t0) = y0
 *  Compute y(t) using a 4th-order Runge-Kutta Integrator (RK4).  Note: the
 *  'integrateV' method for a system of separable ODEs is mixed in from the
 *  `Integrator` trait.
 */
object RungeKutta
       extends Integrator:

    private val debug = debugf ("RungeKutta", true)                 // debug function
    private val flaw  = flawf ("RungeKutta")                        // flaw function
    private val ovf   = Double.MaxValue / 10.0                      // too big, may overflow

    // Butcher Tableau a, b, c with a hardcoded 
    private val b     = VectorD (1/6.0, 1/3.0, 1/3.0, 1/6.0)
    private val c     = VectorD (0.0, 1/2.0, 1/2.0, 1.0)

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

        val t_t0  = t - t0                                          // time interval
        val steps = (round (t_t0 / step)).toInt                     // number of steps
        val k     = new VectorD (4)                                 // 4 stage derivatives times h
        var h     = t_t0 / steps.toDouble                           // adjusted step size
        var tn    = t0                                              // initialize ith time tn to t0
        var y     = y0                                              // initialize y = f(t) to y0

        for n <- 1 to steps do
            if tn > t then { h -= tn - t; tn = t }                  // don't go past t

            k(0) = h * f(tn + c(0)*h, y)
            k(1) = h * f(tn + c(1)*h, y + 0.5*k(0))
            k(2) = h * f(tn + c(2)*h, y + 0.5*k(1))
            k(3) = h * f(tn + c(3)*h, y + k(2))
            y += b dot k
            debug ("integrate", s"for n = $n: k = $k, y = $y")

            if abs (y) > ovf then flaw ("integrate", s"probable overflow since y = $y")
            if n % 100 == 0 then println (s"integrate: step $n: tn = $tn, y = $y")
            tn += h                                                 // take the next step
        end for
        y                           // the value of the function at time t, y = f(t)
    end integrate

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

        val t_t0  = t - t0                                          // time interval
        val steps = (round (t_t0 / step)).toInt                     // number of steps
        val k     = new MatrixD (y0.dim, 4)                         // for each variable, 4 stage derivatives times h
        var h     = t_t0 / steps.toDouble                           // adjusted step size
        var tn    = t0                                              // initialize ith time tn to t0
        val y     = y0                                              // initialize y = f(t) to y0

        for n <- 1 to steps do
            if tn > t then { h -= tn - t; tn = t }                  // don't go past t

            cfor (0, y0.dim) { j =>                                 // for each variable
                val kj = k(j)
                kj(0) = h * f(j)(tn + c(0)*h, y)
                kj(1) = h * f(j)(tn + c(1)*h, y + 0.5*kj(0))
                kj(2) = h * f(j)(tn + c(2)*h, y + 0.5*kj(1))
                kj(3) = h * f(j)(tn + c(3)*h, y + kj(2))
                y(j) += b dot kj
            } // cfor
            debug ("integrateVV", s"for n = $n: y = $y")

            if abs (y(0)) > ovf then flaw ("integrateVV", s"probable overflow since y = $y")
            if n % 100 == 0 then println (s"integrateVV: step n: tn = $tn, y = $y")
            tn += h                                                 // take the next step
        end for
        y                  // the value of the function at time t, y = f(t)
    end integrateVV

end RungeKutta

import RungeKutta._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rungeKuttaTest` main function is used to test the `RungeKutta` object.
 *  Test the `integrate` and `integrateV` methods for one dimensional and separable
 *  ODE problems.
 *  > runMain scalation.dynamics.rungeKuttaTest
 */
@main def rungeKuttaTest (): Unit =

    val y0 = 1.0
    val t_ = 2.0

    banner (s"Test ODE Solver Classic Runge-Kutta compute y(2) where y0 = y(0) = 1")

    banner ("Test `integrate` on y' = f(t, u) = 2.0 * t")
    @nowarn def derv1 (t: Double, y: Double) = 2.0 * t       // solution to differential equation is t^2
    var y_ = (t: Double) => t~^2 + 1                         // symbolic solution
    var y  = integrate (derv1, y0, t_)                       // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct t~^2 + 1 = ${y_(t_)}")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrate` on y' = f(t, u) = y")
    @nowarn def derv2 (t: Double, y: Double) = y             // f(t, y) for differential equation is e^t
    y_ = t => E~^t                                           // symbolic solution
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

end rungeKuttaTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rungeKutta2Test` main function is used to test the `RungeKutta` object.
 *  Test the `integrateVV` methods for systems of ODEs problems.
 *  @see http://www.mathworks.com/help/techdoc/ref/ode23.html (Example 1)
 *  > runMain scalation.dynamics.rungeKuttaTest2
 */
@main def rungeKuttaTest2 (): Unit =

    banner ("Test RungeKutta on y' = t + y with y0 = 1.24 at t_ = 1.0")
    def derv1 (t: Double, y: Double) = t + y
    val y0 = 1.24
    val t_ = 1.0

    println (s"\n==> at t = $t_: y = ${integrate (derv1, y0, t_)}")

    banner ("Test RungeKutta on System of ODEs with y0 = 1.24 at t_ = 1.0")
    @nowarn def dx_dt (t: Double, p: VectorD) =  p(1) * p(2)
    @nowarn def dy_dt (t: Double, p: VectorD) = -p(0) * p(2)
    @nowarn def dz_dt (t: Double, p: VectorD) = -.51 * p(0) * p(1)
    val odes = Array [DerivativeV] (dx_dt, dy_dt, dz_dt)

    val ti  = 0.2
    var p   = VectorD (0.0, 1.0, 1.0)
    val p_r = new MatrixD (61, 3); for k <- p.indices do p_r(0, k) = p(k)
    val tt  = new VectorD (61)
    for i <- 1 to 60 do                               // 6 or 60
        tt(i) = ti * i
        p = integrateVV (odes, p, ti)
        println ("\n==> at tt = " + tt(i) + " p = " + p)
        for k <- p.indices do p_r(i, k) = p(k)
//      p_r(i) = p
    end for

    new Plot (tt, p_r(?, 0), p_r(?, 1), "Plot p(0), p(1) vs. t", lines = true)

end rungeKuttaTest2

