
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Nirupom Bose Roy, John Miller
 *  @version 2.0
 *  @date    Wed May 27 20:37:41 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Kalman Filter
 *
 *  @see web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf
 *  @see en.wikipedia.org/wiki/Kalman_filter
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._
import scalation.random.NormalVec

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KalmanFilter` class provides a simple implementation of a Kalman filter.
 *  It is useful for smoothing noisy data and for providing better estimates of the
 *  state of a system.
 *  @param f  the n × n state transition matrix
 *  @param q  the n × n process noise covariance matrix
 *  @param h  the m × n measurement matrix
 *  @param r  the m × m measurement noise covariance matrix
 *  @param x  the n initial state vector
 *  @param p  the n × n initial covariance matrix
 */
class KalmanFilter (val f: MatrixD, val q: MatrixD,
                    val h: MatrixD, val r: MatrixD,
                    var x: VectorD, var p: MatrixD):

    private val n = x.dim                                            // dimension of the state vector
    private val m = h.dim                                            // dimension of the measurement vector
    require (f.dims == (n, n) && p.dims == (n, n) && q.dims == (n, n) &&
             h.dim2 == n && r.dims == (m, m), "KalmanFilter matrix size mismatch")

    private val MAX_ITER = 20                                        // maximum number of iterations
    private val doPlot   = true                                      // flag for drawing plot
    private val i        = MatrixD.eye (p.dim, p.dim)                // identity matrix

    // prior (predicted) quantities
    private var xPred: VectorD = x.copy
    private var pPred: MatrixD = p.copy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the state of the process at the next time point.
     */
    def predict (): Unit = 
        xPred = f * x                                                // new predicted prior state
        pPred = f * p * f.ᵀ + q                                      // new predicted prior covariance
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the state and covariance estimates with the current and possibly noisy measurements,
     *  using Joseph stabilized covariance update.
     *  @param z  current measurement/observation of the state
     */
    def update (z: VectorD): Unit =
        val y = z - h * xPred                                        // measurement residual
        val s = h * pPred * h.ᵀ + r                                  // residual covariance
        val k = pPred * h.ᵀ * s.inverse                              // optimal Kalman gain
        x = xPred + k * y                                            // updated state estimate

        val ikh = i - k * h
        p = ikh * pPred * ikh.ᵀ + k * r * k.ᵀ                        // updated covariance estimate
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Take one step forward.
     *  @param z  current measurement/observation of the state
     */
    def step (z: VectorD): VectorD =
        predict ()
        update (z)
        x
    end step

// Code for internal simulation

    private var xTrue = x.copy                                       // true state, e.g., from a simulator
    val traj  = if doPlot then new MatrixD (MAX_ITER, n+1) else new MatrixD (0, 0)
    val traj2 = if doPlot then new MatrixD (MAX_ITER, n+1) else new MatrixD (0, 0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def evolveTrueState (): Unit =
        val w = new NormalVec (0.0, q).gen                           // process noise
        xTrue = f * xTrue + w
    end evolveTrueState

   //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def simulateMeasurement (): VectorD =
        val v = new NormalVec (0.0, r).gen                           // observation noise
        h * xTrue + v
    end simulateMeasurement

   //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Simulate the trajectory for x using predict and update phases.
     *  @param dt  the time increment (delta t)
     */
    def simulate (dt: Double): VectorD =
        var t  = 0.0                                                 // initial time

        for k <- 0 until MAX_ITER do
            t += dt                                                  // advance time

            evolveTrueState ()                                       // evolve the simulated system

            val z = simulateMeasurement ()                           // generate noisy observation

            step (z)                                                 // Kalman filter estimation
            println (s"simulate: for step k = $k @ time t = $t, xTrue = $xTrue, x = $x")
            if doPlot then traj(k)  = x :+ t                         // add current time t, state x to trajectory
            if doPlot then traj2(k) = xTrue :+ t                     // add current time t, true state x to trajectory2
        end for
        x
    end simulate

/*
        for k <- 0 until MAX_ITER do
            t += dt                                                  // advance time

            // predict
            predict ()                                               // estimate new state x and covariance pp
            if doPlot then traj(k) = xPred :+ t                      // add current time t, state x to trajectory

            // update
            val v = NormalVec (_0, r).gen                            // observation noise
            val z = h * xPred + v                                    // new observation
            update (z)
        end for
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Copy this Kalman Filter and return it.
     */
    def copyFilter (): KalmanFilter =
        new KalmanFilter (f.copy, q.copy, h.copy, r.copy, x.copy, p.copy)
    end copyFilter

end KalmanFilter


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kalmanFilterTest` main function tests the `KalmanFilter` class based on
 *  tracking a simple simulated trajectory.
 *  @see en.wikipedia.org/wiki/Kalman_filter
 *  > runMain scalation.modeling.forecasting.kalmanFilterTest
 */
@main def kalmanFilterTest (): Unit =

    banner ("KalmanFilterTest")

    val dt    = 0.1                                                  // time increment (delta t)
    val var_a = 0.5                                                  // variance of uncontrolled acceleration a
    val var_z = 0.5                                                  // variance from observation noise

    val ff = MatrixD ((2, 2), 1.0, dt,                               // state transition matrix
                              0.0, 1.0)

    val qq = MatrixD ((2, 2), dt~^4/4, dt~^3/2,                      // process noise covariance matrix
                              dt~^3/2, dt~^2) * var_a

    val hh = MatrixD ((1, 2), 1.0, 0.0)                              // measurement matrix

    val rr = MatrixD ((1, 1), var_z)                                 // measurement noise covariance matrix

    val x0 = VectorD (0.0, 0.0)                                      // initial state vector

    val n  = ff.dim
    val pp = new MatrixD (n, n)                                      // initial covariance estimate matrix

    val kf = new KalmanFilter (ff, qq, hh, rr, x0, pp)

    println ("simulate = " + kf.simulate (dt))
    println ("traj     = " + kf.traj)

    new Plot (kf.traj(?, 2), kf.traj(?, 0), kf.traj(?, 1), "traj: KF", lines = true)
    new Plot (kf.traj2(?, 2), kf.traj2(?, 0), kf.traj2(?, 1), "traj2 Sim", lines = true)

end kalmanFilterTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kalmanFilterTest2` main function tests the `KalmanFilter` class by estimating
 *  coefficients for an AR(2) for Covid-19.
 *  > runMain scalation.modeling.forecasting.kalmanFilterTest2
 */
@main def kalmanFilterTest2 (): Unit =

    println ("TBD")

end kalmanFilterTest2

