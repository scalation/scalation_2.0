
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Nirupom Bose Roy
 *  @version 2.0
 *  @date    Sun Sep 13 20:37:41 EDT 2015
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
 *  @param f  the state transition matrix
 *  @param q  the process noise covariance matrix
 *  @param h  the measurement matrix
 *  @param r  the measurement noise covariance matrix
 *  @param x  the initial state vector
 *  @param p  the initial covariance matrix
 */
class KalmanFilter (val f: MatrixD, val q: MatrixD,
                    val h: MatrixD, val r: MatrixD,
                    var x: VectorD, var p: MatrixD):

    private val MAX_ITER = 20                                        // maximum number of iterations
    private val doPlot   = true                                      // flag for drawing plot
    private val n        = f.dim                                     // dimension of the state vector
    private val _0       = VectorD (n)                               // vector of 0's

    val traj = if doPlot then new MatrixD (MAX_ITER, n+1) else new MatrixD (0, 0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the state of the process at the next time point.
     */
    def predict (): Unit = 
        x = f * x                                                    // new predicted state
        p = f * p * f.ᵀ + q                                          // new predicted covariance
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the state and covariance estimates with the current and possibly noisy measurements
     *  @param z  current measurement/observation of the state
     */
    def update (z: VectorD): Unit =
        val y = z - h * x                                             // measurement residual
        val s = h * p * h.ᵀ + r                                       // residual covariance
        val k = p * h.ᵀ * s.inverse                                   // optimal Kalman gain
        x = x + k * y                                                 // updated state estimate

        val i   = MatrixD.eye (p.dim, p.dim)                          // identity matrix
        val ikh = i - k * h
        p = ikh * p * ikh.ᵀ + k * r * k.ᵀ                             // updated covariance estimate
    end update

   //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Iteratively solve for x using predict and update phases.
     *  @param dt  the time increment (delta t)
     */
    def solve (dt: Double): VectorD =
        var t  = 0.0                                                 // initial time

        for k <- 0 until MAX_ITER do
            t += dt                                                  // advance time
            if doPlot then traj(k) = x :+ t                          // add current time t, state x to trajectory

            // predict
            predict ()                                               // estimate new state x and covariance pp

            // update
            val v = NormalVec (_0, r).gen                            // observation noise
            val z = h * x + v                                        // new observation
            update (z)
        end for
        x
    end solve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Copy this Kalman Filter and return it.
     */
    def copyFilter (): KalmanFilter =
        new KalmanFilter (f.copy, q.copy, h.copy, r.copy, x.copy, p.copy)
    end copyFilter

end KalmanFilter


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kalmanFilterTest` main function is used to test the `KalmanFilter` class.
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

    println ("solve = " + kf.solve (dt))
    println ("traj  = " + kf.traj)

    new Plot (kf.traj(?, 2), kf.traj(?, 0), kf.traj(?, 1))

end kalmanFilterTest

