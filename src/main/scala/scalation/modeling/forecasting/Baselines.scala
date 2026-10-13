
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Nov 15 19:50:32 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Baseline Time Series Models in Tabular Format
 */

package scalation
package modeling
package forecasting

import scala.math.sqrt

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Baseline` class supports simple baseline time series models showing their
 *  In-Sample Testing in an easy to understand tabular format.
 *  One-step ahead forecasts are produced for all but the first time point (t = 0).
 *  Currently supports "NULL", "RW". "AR1", and "AR2".
 *  @see otexts.com/fpp3/acf.html for Auto-Correlation Function (ACF)
 *  @param y      the time series vector
 *  @param mtype  the type of model as a string
 */
class Baseline (y: VectorD, mtype: String)
      extends FitM:

    val hdr  = VectorS ("t", "yt", "zt", "zˆt", "yˆt", "ε", "ε2")
    val mm   = y.dim                                           // length of time series starting at t = 0
    val t    = VectorD.range (0, mm)                           // time vector
    val x    = new MatrixD (mm, hdr.dim)                       // hold tabular results
    val ybar = y.mean                                          // mean of whole time series
    val r    = y(0 until mm-1) corr y(1 until mm)              // HW - why are r and r1 different
    val r1   = y.acorr (1)                                     // lag-1 auto-correlation
    val r2   = y.acorr (2)                                     // lag-2 auto-correlation
    println (s"ybar = $ybar, rho_1: r = $r, r1 = $r1, r2 = $r2")

    val y_    = y(1 until mm)                                  // time series from t = 1
    val y_bar = y_.mean                                        // mean of time series from t = 1

    banner (s"Baseline: In-Sample, One-Step Ahead Forecasting for $mtype model")

    x(?, 0) = t                                                // time
    x(?, 1) = y                                                // time series y_t

    mtype match
    case "NULL" =>
//      for i <- 1 until mm do x(i, 4) = ybar                  // y_t-hat, does not results in R^2 = 0
        for i <- 1 until mm do x(i, 4) = y_bar                 // y_t-hat
    case "RW" =>
        for i <- 1 until mm do x(i, 4) = x(i-1, 1)             // y_t-hat
    case "AR1" => 
        println (s"parameters: $r1")
        x(?, 2) = y - ybar                                     // centered z_t
        for i <- 1 until mm do x(i, 3) = r1 * x(i-1, 2)        // z_t-hat, b0 = φ0 = r1
        x(?, 4) = x(?, 3) + ybar                               // uncentered y_t-hat
    case _ => // "AR2"
        val b0 = r1 * (1 - r2) / (1 - r1~^2)                   // b0 = φ0
        val b1 = (r2 - r1~^2) / (1 - r1~^2)                    // b1 = φ1
        println (s"parameters: $b0, $b1")
        x(?, 2) = y - ybar                                     // centered z_t
        for i <- 1 until mm do
            x(i, 3) = b0 * x(i-1, 2) + b1 * x(max0 (i-2), 2)   // z_t-hat
        x(?, 4) = x(?, 3) + ybar                               // uncentered y_t-hat

    x(?, 5) = y - x(?, 4)                                      // error ε
    x(?, 6) = x(?, 5) ~^ 2                                     // squared error
    println (hdr)
    println (x)

    // Self-diagnose -- skipping time t = 0 <= can't forecast with back-casting
    val yp     = x(1 until mm, 4)                              // y-hat from t = 1
    val e      = x(1 until mm, 5)                              // errors from t = 1
    val se     = x(1 until mm, 6)                              // errors squared from t = 1

    val _m     = mm - 1                                        // number data points used for metrics (skip t = 0)
    val _sse   = se.sum                                        // conditional sum of squared errors
    val _sst   = (y_ - y_.mean).normSq                         // sum of squares total
    val _rSq   = 1 - _sse / _sst                               // coefficient of determination R^2 using mean
    val _sde   = e.stdev                                       // standard deviation of errors 
    val _mse0  = _sse / _m                                     // raw/MLE mean squared error (MSE0)
    val _rmse  = sqrt (_mse0)                                  // root mean squared error (RMSE)                
    val _mae   = e.abs.mean                                    // mean absolute error (MAE)
    val _smape = 200 * (e.abs / (y_.abs + yp.abs)).mean        // symmetric Mean Absolute Percentage Error (sMAPE)
                                                               // diagnose calls smapeF that handles y_i = yp_i = 0
    
    banner ("CBaseline: ompare self-diagnosis (_) with call to `diagnose` from `FitM`")
    val _qof = VectorD (_rSq, _sst, _sse, _sde, _mse0, _rmse, _mae, _smape, _m)
    println (s"_qof = ${FitM.fitMap (_qof)}")

    val qof = diagnose (y_, yp)
    println (s"qof  = ${FitM.fitMap (qof)}")

end Baseline


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `baselineTest` main function is used to test the `Baseline` class.
 *  It can performs Null, RW, AR(1), or AR(2) time series model calculations.
 *  > runMain scalation.modeling.forecasting.baselineTest
 */
@main def baselineTest (): Unit =

    val model = "NULL"
//  val model = "RW"
//  val model = "AR1"
//  val model = "AR2"

    val y = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3)

    new Baseline (y, model)

end baselineTest

