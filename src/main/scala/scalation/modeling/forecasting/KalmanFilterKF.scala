
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, John Miller, Nirupom Bose Roy, Lokesh Adusumilli
 *  @version 2.0
 *  @date    Wed May 27 20:37:41 EDT 2026
 *  @see     LICENSE (MIT style license file).
 */
package scalation
package modeling
package forecasting

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KalmanFilterKF` class is a minimal Kalman filter used by `ARMA_KF` for
 *  Kalman-Filter maximum-likelihood estimation (KF-MLE).
 *
 *  It differs from the general `KalmanFilter` in this package in one essential
 *  way: `predict()` advances the public state `x` and covariance `p` IN PLACE,
 *  so the predicted (prior) state is readable on `x`/`p` between `predict()` and
 *  `update()`.  `ARMA_KF` reads `x`/`p` at exactly that point to accumulate the
 *  Gaussian log-likelihood, so it requires this in-place semantics.  (The general
 *  `KalmanFilter` instead stashes the prediction in private `xPred`/`pPred` and
 *  leaves `x`/`p` at the previous posterior — correct for its simulation use, but
 *  unusable for KF-MLE.)  Kept as a separate class so `ARMA.scala` and the
 *  `KalmanFilter` simulation tests are unaffected.
 *
 *  @param f  the state transition matrix
 *  @param q  the process noise covariance matrix
 *  @param h  the measurement matrix
 *  @param r  the measurement noise covariance matrix
 *  @param x  the initial state vector (advanced in place by `predict`/`update`)
 *  @param p  the initial covariance matrix (advanced in place by `predict`/`update`)
 */
class KalmanFilterKF (val f: MatrixD, val q: MatrixD,
                      val h: MatrixD, val r: MatrixD,
                      var x: VectorD, var p: MatrixD):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Time-update (predict) step: advance the state and covariance one step,
     *  writing the predicted prior into the public `x` and `p`.
     */
    def predict (): Unit =
        x = f * x
        p = f * p * f.transpose + q
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Measurement-update (correct) step given observation `z`.
     *  @param z  the measurement vector
     */
    def update (z: VectorD): Unit =
        val y = z - h * x                                            // measurement residual
        val s = h * p * h.transpose + r                             // residual covariance
        val k = p * h.transpose * s.inverse                        // optimal Kalman gain
        x = x + k * y
        p = (MatrixD.eye (p.dim, p.dim) - k * h) * p
    end update

end KalmanFilterKF

