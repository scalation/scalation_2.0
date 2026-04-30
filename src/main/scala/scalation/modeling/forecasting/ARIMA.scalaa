
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jun 13 01:27:00 EST 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive, Integrated, Moving Average (ARIMA)
 *
 *  @see en.wikipedia.org/wiki/Autoregressive%E2%80%93moving-average_model
 *  @see www.emu.edu.tr/mbalcilar/teaching2007/econ604/lecture_notes.htm
 *  @see www.stat.berkeley.edu/~bartlett/courses/153-fall2010
 *  @see www.princeton.edu/~apapanic/ORFE_405,_Financial_Time_Series_%28Fall_2011%29_files/slides12-13.pdf
 */
 
package scalation
package modeling
package forecasting

import scalation.mathstat._

import ARIMA_diff._
import Forecaster.differ

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARIMA` class provides basic time series analysis capabilities for Auto-
 *  Regressive 'AR' Integrated 'I' Moving-Average 'MA' models.  In an
 *  ARIMA(p, d, q) model, p and q refer to the order of the Auto-Regressive
 *  and Moving-Average components of the model; d refers to the order of
 *  differencing.  Given time series data stored in vector y, its next value y_t = y(t)
 *  may be predicted based on prior values of y and its noise:
 *
 *      y_t = δ + Σ(φ_i y_t-i) + Σ(θ_i e_t-i) + e_t
 *
 *  where δ is a constant, φ is the auto-regressive coefficient vector,
 *  θ is the moving-average coefficient vector, and e is the noise vector.
 *------------------------------------------------------------------------------
 *  If d > 0, then the time series must be differenced first before applying
 *  the above model.
 *------------------------------------------------------------------------------
 *  @param y        the response vector (time series data)
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to AR.hp)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class ARIMA (y: VectorD, hh: Int, tRng: Range = null,
             hparam: HyperParameter = AR.hp,
             bakcast: Boolean = false)
      extends ARMA (diff (y, hparam("d").toInt), hh, tRng, hparam, bakcast):

    private val debug  = debugf ("ARIMA", true)                        // debug function
    private val flaw   = flawf ("ARIMA")                               // flaw function
    private val d      = hparam("d").toInt                             // the number of differences to take
    private val v      = getY                                          // get series passed to ARMA

    if d out (0, 2) then flaw ("init", s"difference d = $d must be in {0, 1, 2}")

    _modelName = s"ARIMA_${p}_${d}_$q"                                 // name of model

    debug ("init", s"$modelName")

    new Plot (null, y, null, s"Plot $modelName: y vs. t", lines = true)
    if d > 0 then new Plot (null, v, null, s"Plot $modelName: v = diff (y, d) vs. t", lines = true)

    ////////////////////////////////////////////////////////////////////////////////
    // Make predictions/forecasts on the original scale time-series (not differenced).

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict all values corresponding to the given vector v_.  If differenced,
     *  tranform back to the original scale.
     *  @param v_  the actual values to use in making predictions (as passed to ARMA).
     */
    def predictAll2 (v_ : VectorD = v): VectorD =
        val vp = predictAll (v_)
        if d > 0 then backform (vp, y, d) else vp
    end predictAll2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign to FORECAST MATRIX and return h-step ahead forecast.
     *  @param yf  the forecast matrix (time x horizons)
     *  @param y_  the actual values to use in making forecasts
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *
    def forecastAt2 (yf: MatrixD, y_ : VectorD, h: Int): VectorD =
        val yfh = arma.forecastAt (yf, pick (y_), h)
        if d > 0 then yfh + y_ else yfh
    end forecastAt2
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points and all horizons (1 through h-steps ahead).
     *  Record these in the yf matrix, where
     *      yf(t, k) = k-steps ahead forecast for y_t
     *  Note, column 0, yf(?, 0), is set to y (the actual time-series values).
     *  Forecast recursively down diagonals in the yf forecast matrix.
     *  The top right and bottom left triangles in yf matrix are not forecastable.
     *  @param y_  the actual values to use in making forecasts
     *  @param h   the maximum forecasting horizon, number of steps ahead to produce forecasts
     *
    def forecastAll2 (y_ : VectorD, h: Int): MatrixD =
        debug ("forecastAll2", s"y_.dim = ${y_.dim}, e.dim = ${e.dim}")
        yf = new MatrixD (y_.dim+h, h+2)                               // forecasts for all time points t & horizons to h
        for t <- y_.indices do yf(t, 0) = y_(t)                        // first column is the timestep (e.g., logical day)
        for k <- 1 to h do forecastAt2 (yf, y_, k)                     // forecast k-steps into the future
        for t <- yf.indices do yf(t, h+1) = t                          // last column is time (logical day)
        yf                                                             // return matrix of forecasted values
    end forecastAll2
     */

end ARIMA


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRIMATest` main function tests the `ARIMA` class on simulated data.
 *  Test predictions (one step ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRIMATest
 */
@main def aRIMATest (): Unit =

    import AR.hp
    println (s"hp = $hp")

    val y  = makeTSeries ()                                            // create simulated time-series (see `Stationary`)
    val hh = 2

    banner (s"Test Predictions: ARIMA(1, 0, 1) on simulated time-series")
    var mod = new ARIMA (y, hh)                                        // create model for time-series data ARIMA(1, 1)
    mod.trainNtest ()()                                                // train and test on full dataset

    banner (s"Test Predictions: ARIMA(1, 0, 0) on simulated time-series")
    hp("q") = 0
    mod = new ARIMA (y, hh)                                            // create model for time-series data ARIMA(1, 0)
    mod.trainNtest ()()      

    banner ("Select model based on ACF and PACF")
    mod.plotFunc (mod.acF, "ACF")                                      // Auto-Correlation Function (ACF)
    mod.plotFunc (mod.pacF, "PACF")                                    // Partial Auto-Correlation Function (PACF)

end aRIMATest

import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRIMATest2` main function tests the `ARIMA` class on real data:
 *  Forecasting lake levels.
 *  Test predictions (one step ahead forecasts) with no differencing
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRIMATest2
 */
@main def aRIMATest2 (): Unit =

    import AR.hp
    hp("d") = 0            // (no differencing) => should give same results as ARMA (@see `aRMATest2`)
    println (s"hp = $hp")
    val hh = 2

    for p <- 1 to 5; q <- 0 to 2 do
        hp("p") = p; hp("q") = q                                       // set p (AR) and q (MA) hyper-parameters
        val mod = new ARIMA (y, hh)                                    // create model for time-series data ARIMA(p, q)
        banner (s"Test Predictions: ${mod.modelName} on LakeLevels Dataset")
        mod.trainNtest ()()                                            // trainb and test the model on full dataset

        val yp = mod.predictAll2 (y)                                   // results on original scale
        new Plot (null, y, yp, s"Plot: ${mod.modelName} predictAll2: y, yp vs t", lines = true)
    end for

end aRIMATest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRIMATest3` main function tests the `ARIMA` class on real data:
 *  Forecasting lake levels.
 *  Test predictions (one step ahead forecasts) taking one difference.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRIMATest3
 */
@main def aRIMATest3 (): Unit =

    import AR.hp
    val d = 1
    hp("d") = d                                                        // first differencing 
    val hh = 2

    val v = diff (y, d)                                                // take the first difference of time-series y
    differ (y, backform (v, y))                                        // verify recovery of original times-series

    for p <- 1 to 4; q <- 0 to 1 do
        hp("p") = p; hp("q") = q                                       // set p (AR) and q (MA) hyper-parameters
        val mod = new ARIMA (y, hh)                                    // create model for time-series data ARIMA(p, q)
        banner (s"Test Predictions: ${mod.modelName} on LakeLevels Dataset")
        val vp = mod.trainNtest ()()._1                                // test and test the model on full dataset
        val yp = mod.predictAll2 (y)                                   // results on original scale

        new Plot (null, v, vp, s"Plot: ${mod.modelName} predictAll2: v, vp vs t", lines = true)
        new Plot (null, y, yp, s"Plot: ${mod.modelName} predictAll2: y, yp vs t", lines = true)
    end for

end aRIMATest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRIMATest4` main function tests the `ARIMA` class on real data:
 *  Forecasting lake levels.
 *  Test forecasts (1 to h steps ahead forecasts) for several values of p and q.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  @see `aRMATest5`
 *  > runMain scalation.modeling.forecasting.aRIMATest4
 *
@main def aRIMATest4 (): Unit =

    import AR.hp

    val hh = 2                                                        // maximum forecasting horizon

    val v = Δ (y)                                                     // velocity series (first differences)

    for p <- 2 to 2; q <- 0 to 0 do
        hp("p") = p; hp("q") = q                                      // set p (AR) and q (MA) hyper-parameters
        val mod = new ARIMA (y)                                       // create model for time series data
        banner (s"Test: ${mod.modelName} on LakeLevels Dataset")
        mod.trainNtest ()()                                           // train and test the model on full dataset

        val yf_ = mod.forecastAll (y, hh)                             // forecast using differenced values h-steps ahead for all y
        val yf  = transformBack (yf_, y, hh)                          // transform back to original scale
        println (s"yf_ = $yf_")                                       // forecast matrix on differenced values
        println (s"yf  = $yf")                                        // forecast matrix on original scale
        println (s"y   = $y")                                         // observed values on original scale

        val tf = new TestFit (y.dim)
        val vh1 = yf_(?, 1)                                           // test on differenced scale
        val vh2 = yf_(?, 2)
        println (tf.testDiagnose (v, vh1))
        println (tf.testDiagnose (v, vh2))
        new Plot (null, v, vh1, "v, vh1 vs. t", lines = true)
        new Plot (null, v, vh2, "v, vh2 vs. t", lines = true)

        val yh1 = yf(?, 1)                                            // test on original scale
        val yh2 = yf(?, 2)
        println (tf.testDiagnose (y, yh1))
        println (tf.testDiagnose (y, yh2))
        new Plot (null, y, yh1, "y, yh1 vs. t", lines = true)
        new Plot (null, y, yh2, "y, yh2 vs. t", lines = true)

//      Forecaster.checkForecastMatrix (yf, y, yp)                    // FIX - differences & un-differenced
    end for

end aRIMATest4
 */
