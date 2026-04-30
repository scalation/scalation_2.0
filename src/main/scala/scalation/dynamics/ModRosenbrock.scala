
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Nov  6 15:54:34 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    (2,3)-Order Modified Rosenbrock Stiff ODE Integrator or ode23s
 *
 *  @see     rotordynamics.wordpress.com/2014/06/18/the-modified-rosenbrock-triple/
 */

package scalation
package dynamics

import scala.annotation.nowarn
import scala.math.{abs, max, min, pow}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ModRosenbroc` object provides a numerical ODE solver for stiff systems.
 *  Given an unknown, time-dependent function y(t) governed by an Ordinary
 *  Differential Equation (ODE) of the form
 *      d/dt y(t) = f(t, y)
 *  compute y(t) using a (2,3)-order Modified Rosenbrock ODE Integrator (Modified Rosenbrock)
 *  or ode.23s
 *  Note: the integrateV method for a system of separable ODEs is mixed in from the
 *  `Integrator` trait.
 *  @see http://adorio-research.org/wordpress/?p=6565
 */
object ModRosenbrock
       extends Integrator:

    private val flaw = flawf ("ModRosenbrock")

    private val a21 = 1.0
//  private val a31 = 1.0
//  private val a32 = 0.0

    private val c21 = -1.0156171083877702091975600115545
    private val c31 = 4.0759956452537699824805835358067
    private val c32 = 9.2076794298330791242156818474003

    private val b1 = 1.0
    private val b2 = 6.1697947043828245592553615689730
    private val b3 = -0.4277225654321857332623837380651

    private val e1 = 0.5
    private val e2 = -2.9079558716805469821718236208017
    private val e3 = 0.2235406989781156962736090927619

    private val gamma = 0.43586652150845899941601945119356
    private val c2 = 0.43586652150845899941601945119356

    private val d1 = 0.43586652150845899941601945119356
    private val d2 = 0.24291996454816804366592249683314
    private val d3 = 2.1851380027664058511513169485832

    private val safeScale = 0.9
    private val alphaInc  = 0.2
    private val alphaDec  = 0.25
    private val minScale  = 0.2
    private val maxScale  = 10.0

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute y(t) governed by a differential equation using numerical integration
     *  of the derivative function f(t, y) using a (2,3)-order Modified Rosenbrock method
     *  to return the value of y(t) at time t.
     *  @param f     the derivative function f(t, y)
     *  @param y0    value of the y-function at time t0, y0 = y(t0)
     *  @param t     the time value at which to compute y(t)
     *  @param t0    the initial time
     *  @param step  the middle step size
     */
    def integrate (f: Derivative, y0: Double, t: Double,
                   t0: Double = 0.0, step: Double = defaultStepSize): Double =
        0.0                                                                       // FIX - implement
    end integrate
 
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute y(t), a vector, governed by a system of differential equations using
     *  numerical integration of the derivative function f(t, y) using a (2,3)-order
     *  Modified Rosenbrock method to return the value of y(t) at time t.
     *  @param f     the array of derivative functions [f(t, y)] where y is a vector
     *  @param y0    the value of the y-function at time t0, y0 = y(t0)
     *  @param t     the time value at which to compute y(t)
     *  @param t0    the initial time
     *  @param step  the step size
     */
    def integrateVV (f: Array [DerivativeV], y0: VectorD, t: Double,
                     t0: Double = 0.0, step: Double = defaultStepSize): VectorD =

        val n    = y0.dim
        var k1   = new VectorD (n)
        var k2   = new VectorD (n)
        var k3   = new VectorD (n)
        val err  = new VectorD (n)                                        // error vector
        val dfdt = new VectorD (n)                                        // df/dt
        val a    = new MatrixD (n, n)
//      val pivotIndices = Array.ofDim [Int] (n)                          // @see `mathstat.Fac_LU`

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Solve a single step dt and return new point y and the error.
         *  Adpated from
         *  @see github.com/OpenFOAM/OpenFOAM-4.x/blob/master/src/ODE/ODESolvers/Rosenbrock23/Rosenbrock23.H
         */
        def solve (t0: Double, y0: VectorD, dydt0: VectorD, dt: Double): VectorD =

            val y = new VectorD (n)                                       // determine the new point y

            val dfdy = ode_jacobian (t0, y0, dfdt)                        // compute the Jacobian nxn-matrix df/dy

            cfor (0, n) { i =>
                cfor (0, n) { j => a(i, j) = - dfdy(i, j) }
                a(i, i) += 1.0 / (gamma * dt)
            } // cfor

            // lUDecompose (a, pivotIndices)
            val lu = new Fac_LU (a)
            lu.factor ()

            // Calculate k1:
            cfor (0, n) { i => k1(i) = dydt0(i) + dt * d1 * dfdt(i) }

            // lUBacksubstitute (a, pivotIndices, k1)
            k1 = lu.bsolve (k1)

            // Calculate k2:
            cfor (0, n) { i => y(i) = y0(i) + a21 * k1(i) }

            val dydt = ode_derivatives (f, t0 + c2 * dt, y)               // compute the time derivative n-vector dy/dt

            cfor (0, n) { i => k2(i) = dydt(i) + dt * d2 * dfdt(i) + c21 * k1(i) / dt }

            // lUBacksubstitute (a, pivotIndices, k2)
            k2 = lu.bsolve (k2)

            // Calculate k3:
            cfor (0, n) { i => k3(i) = dydt(i) + dt * d3 * dfdt(i) + (c31 * k1(i) + c32 * k2(i)) / dt }

            // lUBacksubstitute (a, pivotIndices, k3)
            k3 = lu.bsolve (k3)

            // Calculate error and update state:
            cfor (0, n) { i =>
                y(i)   = y0(i) + b1 * k1(i) + b2 * k2(i) + b3 * k3(i)
                err(i) = e1 * k1(i) + e2 * k2(i) + e3 * k3(i)
            } // cfor

            normalizeError (y0, y, err)
            y
        end solve

        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Return the maximum normalized error.
         *  Note:  assuming mag is `abs`
         *  Adpated from
         *  @see github.com/OpenFOAM/OpenFOAM-4.x/blob/master/src/ODE/ODESolvers/ODESolver/ODESolver.C
         */
        def normalizeError (y0: VectorD, y: VectorD, err: VectorD): Double =
            val absTol = 1E-5                                             // use instead of absTol_(i)
            val relTol = 1E-4                                             // use instead of relTol_(i)
            var maxErr = 0.0
            cfor (0, n) { i =>
                val tol = absTol + relTol * max (abs (y0(i)), abs (y(i)))
                maxErr  = max (maxErr, abs (err(i)) / tol)
            } // cfor
            maxErr
        end normalizeError

        val derv0 = null                                                  // FIX - need dy/dt at 0
        val yy    = solve (t0, y0, dydt0 = derv0, dt = step)
        yy
    end integrateVV

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute and return the time derivative vector dy/dt at point/vector y and time t
     *  by evaluating the the array derivative functions f.
     *  @param f  the array of derivative functions [f(t, y)] where y is a vector
     *  @param t  the time
     *  @param y  the value of the y-function at time t, y = y(t)
     */
    def ode_derivatives (f: Array [DerivativeV], t: Double, y: VectorD): VectorD =
        VectorD (for i <- f.indices yield f(i)(t, y)) 
    end ode_derivatives

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute and return the Jacobian matrix at point/vector y and time t.
     *  @param t     the time
     *  @param y     the value of the y-function at time t, y = y(t)
     *  @param dfdt  the vector value derivative df/dt
     */
    def ode_jacobian (t: Double, y: VectorD, dfdt: VectorD): MatrixD =
        // either the user should define computation for Jacobian
        // or compute numerically from the array of derivative function f
        println (s"t = $t, y = $y, dfdt = $dfdt")
        null                                                          // FIX - to be implemented
    end ode_jacobian
 
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve a single step dt and return the error.
     *  @param t0     the initial time
     *  @param y0     the value of the y-function at time t0, y0 = y(t0)
     *  @param dtTry  the time-step to try
     */
    def solve_step (t0: Double, y0: VectorD, dydt0: VectorD, dt: Double, y: VectorD): Double = ???

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for ...
     *  @see github.com/OpenFOAM/OpenFOAM -4.x/blob/master/src/ODE/ODESolvers/adaptiveSolver/adaptiveSolver.C
     *  @param f      the array of derivative functions [f(t, y)] where y is a vector
     *  @param t0     the initial time
     *  @param y0     the value of the y-function at time t0, y0 = y(t0)
     *  @param dtTry  the time-step to try
     */
    def adaptiveSolver_solve (f: Array [DerivativeV], t0: Double, y0: VectorD, dtTry: Double): Double =

        val VSMALL = 1E-6
        val dydt0  = ode_derivatives (f, t0, y0)
        var dt     = dtTry
        var err    = 0.0
        var t      = t0
        var y      = y0

        // loop over solver and adjust step-size as necessary to achieve desired error
        var go = true
        val yTemp = y
        while go do
            // solve step and provide error estimate
            err = solve_step (t, y, dydt0, dt, yTemp)

            // if error is large reduce dt
            if err > 1 then
                val scale = max (safeScale * pow (err, - alphaDec), minScale)
                dt *= scale
                if dt < VSMALL then flaw ("adaptiveSolver_solve", "FatalErrorInFunction stepsize underflow")
            if err <= 1 then go = false
        end while

        // update the state (time t and point y)
        t += dt
        y  = yTemp

        // if the error is small increase the step-size, return updated dtTry
        if err > pow (maxScale /safeScale, -1.0 / alphaInc) then
            min (max (safeScale * pow (err, -alphaInc), minScale), maxScale) * dt
        else
            safeScale * maxScale * dt
    end adaptiveSolver_solve

end ModRosenbrock

import ModRosenbrock._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `modRosenbrockTest` main function is used to test the `ModRosenbrock` object.
 *  This test is for non-stiff equations.  Compare ode23s with ode45.
 *  > runMain scalation.dynamics.modRosenbrockTest
 */
@main def modRosenbrockTest (): Unit =

/*
    def derv1 (t: Double, y: Double) = t + y
    val y0   = 1.24
    val t    = 1.0
    val hmin = 0.01
    val hmax = 1.0

    def integrate (f: Derivative, y0: Double, t: Double, hmin: Double, hmax: Double,
                   t0: Double = 0., tol: Double = 1E-5, maxSteps: Int = 1000): Double =
    println ("\n==> at t = " + t + " y = " + integrate2 (derv1, y0, t, hmin, hmax))
*/

    // @see http://www.mathworks.com/help/techdoc/ref/ode23.html (Example 1)

    @nowarn def dx_dt (t: Double, p: VectorD) =  p(1) * p(2)
    @nowarn def dy_dt (t: Double, p: VectorD) = -p(0) * p(2)
    @nowarn def dz_dt (t: Double, p: VectorD) = -0.51 * p(0) * p(1)
    val odes = Array [DerivativeV] (dx_dt, dy_dt, dz_dt)

    val ti  = 0.2
    var p   = VectorD (0.0, 1.0, 1.0)
    val p_r = new MatrixD (61, 3); for k <- p.indices do p_r(0, k) = p(k)
    val tt  = new VectorD (61); tt(0) = 0.0
    for i <- 1 to 60 do
        tt(i) = ti * i
        p = integrateVV (odes, p, ti)
        println ("\n==> at tt = " + tt(i) + " p = " + p)
        for k <- p.indices do p_r(i, k) = p(k)
//      p_r(i) = p
    end for

    new Plot (tt, p_r(?, 0), p_r(?, 1), "Plot p(0), p(1) vs. t")

end modRosenbrockTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `modRosenbrockTest2` main function is used to test the `ModRosenbrock` object.
 *  This test is for stiff equations.  Compare ode23s with ode45.
 *  > runMain scalation.dynamics.modRosenbrockTest2
 */
@main def modRosenbrockTest2 (): Unit =

    println ("modRosenbrockTest2 not yet implemented")             // FIX - implement

end modRosenbrockTest2

