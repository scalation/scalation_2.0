
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Linear Trend Model
 *
 *  @see ruqinren.wordpress.com/2020/02/21/all-the-confusion-about-arima-arimax-transfer-function-dynamic-regression-models/
 *       robjhyndman.com/hyndsight/arimax/
 *       medium.com/@xwang222/forecasting-101-ep07-multivariate-models-9f3a11fbb374
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TrendModel` class provides basic time series analysis capabilities for
 *  TrendModel models.  TrendModel models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = f(t) = b_0 + b_1 t.
 *
 *      y_t = f(t) + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param y        the response vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (none => use null)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class TrendModel (y: VectorD, hh: Int, tRng: Range = null,
                  hparam: HyperParameter = null,
                  bakcast: Boolean = false)
      extends Forecaster (y, hh, tRng, hparam, bakcast)
         with NoSubModels:

    private val flaw = flawf ("TrendModel")                             // flaw function

    _modelName = "TrendModel"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a time series y_, train the forecasting function y_ = f(lags (y_)) + e,
     *  where f(lags (y_)) is a function of the lagged values of y_,
     *  by fitting its parameters.
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the testing/full response/output vector (e.g., full y)
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =
        val t_ = VectorD.range (0, y_.dim)                             // regress on time axis
        b      = SimpleRegression.coeff (t_, y_)                       // compute b = [b_0, b_1]
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = f (y_t-1, ...) = b_0 + b_1 t    (trend model, slope, intercept)
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions (not used)
     */
    override def predict (t: Int, y_ : VectorD): Double = b(0) + b(1) * t

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with rolling validation (analog of predict method).
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     */
    override def forecast (t: Int, y_ : VectorD = yb): VectorD =
        val yh = new VectorD (hh)                                       // hold forecasts for each horizon
        for h <- 1 to hh do
            val pred = predict (t+h-1, y_)                              // slide in prior forecasted values
            yf(t, h) = pred                                             // record in forecast matrix
            yh(h-1)  = pred                                             // record forecasts for each horizon
        yh                                                              // return forecasts for all horizons
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the h-steps ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1.
     *  @see `forecastAll` method in `Forecaster` trait.
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the actual values to use in making forecasts
     */
    override def forecastAt (h: Int, y_ : VectorD = yb): VectorD =
        if h < 2 then flaw ("forecastAt", s"horizon h = $h must be at least 2")

        for t <- y_.indices do                                          // make forecasts over all time points for horizon h
            yf(t, h) = predict (t+h-1, y_)                              // record in forecast matrix
        yf(?, h)                                                        // return the h-step ahead forecast vector
    end forecastAt

end TrendModel


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TrendModel` companion object provides factory methods for the `TrendModel` class.
 */
object TrendModel:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TrendModel` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = null): TrendModel =
        new TrendModel (y, hh, tRng, hparam)
    end apply

end TrendModel


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `trendModelTest` main function tests the `TrendModel` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.trendModelTest
 */
@main def trendModelTest (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new TrendModel (y, hh)                                      // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end trendModelTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `trendModelTest2` main function tests the `TrendModel` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.trendModelTest2
 */
@main def trendModelTest2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new TrendModel (y, hh)                                      // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end trendModelTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `trendModelTest3` main function tests the `TrendModel` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.trendModelTest3
 */
@main def trendModelTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new TrendModel (y, hh)                                      // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end trendModelTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `trendModelTest4` main function tests the `TrendModel` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.trendModelTest4
 */
@main def trendModelTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new TrendModel (y, hh)                                      // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end trendModelTest4

