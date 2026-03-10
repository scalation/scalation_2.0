
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Dec 28 14:29:00 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    pth-Order Explicit, Fixed Stepsize Runge-Kutta Integrators (RKp)
 *
 *  @see www.mathworks.com/help/matlab/math/choose-an-ode-solver.html
 *  @see en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
 *  @see people.cs.vt.edu/~asandu/Public/Qual2011/DiffEqn/Butcher_1996_RK-history.pdf
 */

package scalation
package dynamics

import scala.annotation.nowarn
import scala.math.{abs, E, round}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RungeKutta2` class provides implementations of several Runge-Kutta numerical
 *  ODE solvers.  Given an unknown, time-dependent function y(t) governed by an
 *  Ordinary Differential Equation (ODE) of the form:
 *      d/dt y(t) = y'(t) = f(t, y)
 *      y(t0) = y0
 *  Note: the `integrateV` method for a system of separable ODEs is mixed in from the
 *  `Integrator` trait.
 *  The ODE method is defined by its Butcher Tablaeu (a, b, c).
 *  @param name  the name the ODE solver
 *  @param a     the lower triangular matrix of constants multiplying the stage derivatives
 *  @param b     the vector of constants for computing the weighted average of stage derivatives
 *  @param c     the vector of constants for shifting time
 */
case class RungeKutta2 (name: String, a: MatrixD, b: VectorD, c: VectorD)
       extends Integrator:

    private val debug = debugf ("RungeKutta2", false)          // debug function
    private val flaw  = flawf ("RungeKutta2")                  // flaw function
    private val ovf   = Double.MaxValue / 10.0                 // too big, may overflow
    private val s     = b.dim                                  // number of stages

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
        val k     = new VectorD (s)                                 // s stage derivatives x h
        var h     = t_t0 / steps.toDouble                           // adjusted step size
        var tn    = t0                                              // initialize nth time tn to t0
        var y     = y0                                              // initialize y at t0 to y0

        for n <- 1 to steps do
            if tn > t then { h -= tn - t; tn = t }                  // don't go past t

            cfor (0, s) { i =>                                      // for each stage
                val ti = tn + c(i) * h                              // time for stage i
                val yi = y + Σ (0, i) { l => a(i, l) * k(l) }       // y-scalar for stage i
                k(i) = h * f(ti, yi)                                // i-ith stage derivative x h
            } // cfor
            y += b dot k                                            // update scalar y
            debug ("integrate", s"for n = $n: k = $k, y = $y")

            if abs (y) > ovf then flaw ("integrate", s"probable overflow since y = $y")
            if n % 100 == 0 then println (s"integrate: step $n: tn = $tn, y = $y")
            tn += h                                            // take the next step
        end for
        y                           // the value of the function at time t, y
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
        val k     = new MatrixD (y0.dim, s)                         // for each variable, s stage derivatives times h
        var h     = t_t0 / steps.toDouble                           // adjusted step size
        var tn    = t0                                              // initialize nth time tn to t0
        val y     = y0                                              // initialize y at t0 to y0

        for n <- 1 to steps do
            if tn > t then { h -= tn - t; tn = t }                  // don't go past t

            cfor (0, y0.dim) { j =>                                 // for each variable j
                val kj = k(j)
                cfor (0, s) { i =>                                  // for each stage i
                    val ti = tn + c(i) * h                          // time for stage i
                    val yi = y + Σ (0, i) { l => a(i, l) * kj(l) }  // y-scalar for stage i
                    kj(i) = h * f(j)(ti, yi)                        // i-ith stage derivative x h
                } // cfor
                y(j) += b dot kj
            } // cfor
            debug ("integrateVV", s"for n = $n: y = $y")

            if abs (y(0)) > ovf then flaw ("integrateVV", s"probable overflow since y = $y")
            if n % 100 == 0 then println (s"integrateVV: iteration n: tn = $tn, y = %y")
            tn += h                                                   // take the next step
        end for
        y                  // the value of the function at time t, y
    end integrateVV

end RungeKutta2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RungeKutta2` object provides several Explicit, Fixed Stepsize Runge-Kutta ODE solvers.
 *  The ODE method is defined by its Butcher Tablaeu (a, b, c).
 *  @see en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
 */
object RungeKutta2:

    // Compute y(t) using a 2nd-order Runge-Kutta Integrator (RK2).
    // Modified Euler (Explicit Midpoint) method

    val rk2 = new RungeKutta2 ("rk2",
                  a = MatrixD.low (2) (0.5),
                  b = VectorD (0.0, 1.0),
                  c = VectorD (0.0, 0.5))

    // Compute y(t) using a 3rd-order Runge-Kutta Integrator (RK3).
    // Strong Stability Preserving Runge-Kutta (SSPRK3) method

    val rk3 = new RungeKutta2 ("rk3",
                  a = MatrixD.low (3) (1.0,
                                       0.25, 0.25),
                  b = VectorD (1/6.0, 1/6.0, 2/3.0),
                  c = VectorD (0.0, 1.0, 1/2.0))

    // Compute y(t) using a 4th-order Runge-Kutta Integrator (RK4).
    // Classic fourth-order Runge-Kutta method

    val rk4 = new RungeKutta2 ("rk4",
                  a = MatrixD.low (4) (0.5,
                                       0.0, 0.5,
                                       0.0, 0.0, 1.0),
                  b = VectorD (1/6.0, 1/3.0, 1/3.0, 1/6.0),
                  c = VectorD (0.0, 1/2.0, 1/2.0, 1.0))

    // Compute y(t) using a 5th-order Runge-Kutta Integrator (RK5).
    // Butcher's fifth-order Runge-Kutta method
    // www.researchgate.net/publication/326510985_A_Comparative_Study_on_Fourth_Order_and_Butcher's_Fifth_Order_Runge-Kutta_Methods_with_Third_Order_Initial_Value_Problem_IVP

    val rk5 = new RungeKutta2 ("rk5",
                  a = MatrixD.low (6) (1/4.0,
                                       1/8.0,  1/8.0,
                                       0.0,   -1/2.0, 1.0,
                                       3/16.0, 0.0,   0.0,     9/16.0,
                                      -3/7.0,  2/7.0, 12/7.0, -12/7.0, 8/7.0),
                  b = VectorD (7/90.0, 0.0, 32/90.0, 12/90.0, 32/90.0, 7/90.0),
                  c = VectorD (0.0, 1/4.0, 1/4.0, 1/2.0, 3/4.0, 1.0))

end RungeKutta2

import RungeKutta2._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rungeKutta2Test` main function is used to test the `RungeKutta2` class.
 *  Test the `integrate` and `integrateV` methods for one dimensional and separable
 *  ODE problems.
 *  > runMain scalation.dynamics.rungeKutta2Test
 */
@main def rungeKutta2Test (): Unit =

    val y0 = 1.0
    val t_ = 2.0

    val solver = rk5
    banner (s"Test ODE Solver ${solver.name} compute y(2) where y0 = y(0) = 1")

    banner ("Test `integrate` on y' = f(t, u) = 2.0 * t")
    @nowarn def derv1 (t: Double, y: Double) = 2.0 * t       // solution to differential equation is t^2
    var y_ = (t: Double) => t~^2 + 1                         // symbolic solution
    var y  = solver.integrate (derv1, y0, t_)                // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct t~^2 + 1 = ${y_(t_)}")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrate` on y' = f(t, u) = y")
    @nowarn def derv2 (t: Double, y: Double) = y             // f(t, y) for differential equation is e^t
    y_ = t => E~^t                                           // symbolic solution
    y  = solver.integrate (derv2, y0, t_)                    // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct: E~^t = ${y_(t_)} ")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrate` on y' = f(t, u) = t + y")
    def derv3 (t: Double, y: Double) = t + y                 // f(t, y) for ordinary differential equation
    y_ = t => 2*E~^t - t - 1                                 // symbolic solution
    y  = solver.integrate (derv3, y0, t_)                    // numeric solution
    println (s"\n==> at t = $t_: y = $y")
    println (s"\n==> correct: 2*E~^t - t - 1 = ${y_(t_)}")
    println (s"\n==> error = ${y_(t_) - y}")

    banner ("Test `integrateV` on y' = f(t, u) = [2.0 * t, y]")
    println (s"\n==> at t = $t_: y = ${solver.integrateV (Array (derv1, derv2), VectorD (1.0, 2.0), t_)}")

end rungeKutta2Test


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rungeKutta2Test2` main function is used to test the `RungeKutta2` class.
 *  Test the `integrateVV` methods for systems of ODEs problems.
 *  @see www.mathworks.com/help/techdoc/ref/ode23.html (Example 1)
 *  > runMain scalation.dynamics.rungeKutta2Test2
 */
@main def rungeKutta2Test2 (): Unit =

    val solver = rk5
    banner (s"Test ODE Solver ${solver.name} compute y(1) where y0 = y(0) = 1,24")

    banner ("Test RungeKutta on y' = t + y with y0 = 1.24 at t_ = 1.0")
    def derv1 (t: Double, y: Double) = t + y
    val y0 = 1.24
    val t_ = 1.0

    println (s"\n==> at t = $t_: y = ${solver.integrate (derv1, y0, t_)}")

    banner ("Test RungeKutta on System of ODEs with y0 = 1.24 at t_ = 1.0")

    @nowarn def dx_dt (t: Double, p: VectorD) =  p(1) * p(2)
    @nowarn def dy_dt (t: Double, p: VectorD) = -p(0) * p(2)
    @nowarn def dz_dt (t: Double, p: VectorD) = -.51 * p(0) * p(1)
    val odes = Array [DerivativeV] (dx_dt, dy_dt, dz_dt)

    val ti  = 0.2
    var p   = VectorD (0.0, 1.0, 1.0)
    val p_r = new MatrixD (61, 3); for k <- p.indices do p_r(0, k) = p(k)
    val tt  = new VectorD (61)
    for i <- 1 to 60 do                                // 6 or 60
        tt(i) = ti * i
        p = solver.integrateVV (odes, p, ti)
        println (s"\n==> at tt = ${tt(i)}, p = $p")
        for k <- p.indices do p_r(i, k) = p(k)
//      p_r(i) = p
    end for

    new Plot (tt, p_r(?, 0), p_r(?, 1), "Plot p(0), p(1) vs. t", lines = true)

end rungeKutta2Test2

