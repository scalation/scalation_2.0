
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Simple Moving Average (not the same as MA in ARMA)
 */

package scalation
package modeling
package forecasting

import scala.math.max

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleMovingAverage` class provides basic time series analysis capabilities for
 *  SimpleMovingAverage models.  SimpleMovingAverage models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = mean of last q values.
 *
 *      y_t = mean (y_t-1, ..., y_t-q) + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param y        the response vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to SimpleMovingAverage.hp)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class SimpleMovingAverage (y: VectorD, hh: Int, tRng: Range = null,
                           hparam: HyperParameter = SimpleMovingAverage.hp,
                           bakcast: Boolean = false)
      extends Forecaster (y, hh, tRng, hparam, bakcast)
         with NoSubModels:

    private val debug = debugf ("SimpleMovingAverage", true)            // debug function
    private val flaw  = flawf ("SimpleMovingAverage")                   // flaw function
    private val q     = hparam("q").toInt                               // take mean of last q values

    _modelName = s"SimpleMovingAverage_$q"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = f (y_t-1, ...) = mean of last q values    (simple moving average model)
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions (mean (inclusive, exclusice))
     */
    override def predict (t: Int, y_ : VectorD): Double =
        if t < 1 then -0.0                                              // not enough prior data
        else y_.mean (max0 (t-q), t)                                    // mean of prior q actual values
    end predict

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
            val pred = if t < 1 then -0.0                               // not enough prior data
                       else forge (t, h).mean                           // record in forecast matrix
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
            yf(t, h) = if t < 1 then -0.0                               // not enough prior data
                       else forge (t, h).mean                           // record in forecast matrix
        yf(?, h)                                                        // return the h-step ahead forecast vector
    end forecastAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a vector from actual (move up column 0 in yf) and prior forecasted values
     *  (move right from column h in yf) to be used in the moving average calculation. 
     *  @param t   the time point from which to make forecasts
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */ 
    def forge (t: Int, h: Int): VectorD = 
        var yft = yf(t, max (1, h-q) until h)                           // @t: get prior forecasts h-q .. h-1
        if yft.dim < q then
            val gap = q - yft.dim                                       // still need gap values
            yft = yft ++ yf(max0 (t-gap) until t, 0)                    // get remaining values from actuals (column 0)
        debug ("forge", s"($t, $h) = $yft")
        yft
    end forge

end SimpleMovingAverage


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleMovingAverage` companion object provides factory methods for the
 *  `SimpleMovingAverage` class.
 */
object SimpleMovingAverage:

    /** Base hyper-parameter specification for `SimpleMovingAverage` and `WeightedMovingAverage` classes
     */
    val hp = new HyperParameter
    hp += ("q", 2, 2)                           // number of prior values for mean
    hp += ("u", 1.0, 1.0)                       // slider from flat (0.0) to linear (1.0) weights

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SimpleMovingAverage` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = hp): SimpleMovingAverage =
        new SimpleMovingAverage (y, hh, tRng, hparam)
    end apply

end SimpleMovingAverage

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverageTest` main function tests the `SimpleMovingAverage` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.simpleMovingAverageTest
 */
@main def simpleMovingAverageTest (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new SimpleMovingAverage (y, hh)                             // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end simpleMovingAverageTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverageTest2` main function tests the `SimpleMovingAverage` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.simpleMovingAverageTest2
 */
@main def simpleMovingAverageTest2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new SimpleMovingAverage (y, hh)                             // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end simpleMovingAverageTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverageTest3` main function tests the `SimpleMovingAverage` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.simpleMovingAverageTest3
 */
@main def simpleMovingAverageTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new SimpleMovingAverage (y, hh)                             // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end simpleMovingAverageTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverageTest4` main function tests the `SimpleMovingAverage` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.simpleMovingAverageTest4
 */
@main def simpleMovingAverageTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new SimpleMovingAverage (y, hh)                             // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end simpleMovingAverageTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverageTest5` main function tests the `SimpleMovingAverage` class
 *  a simple data using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.simpleMovingAverageTest5
 */
@main def simpleMovingAverageTest5 (): Unit =

    import SimpleMovingAverage.hp

    val y_ = VectorD (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)

    hp("q") = 3                                                           // size of moving average window: test 2 and 3
    val hh  = 5                                                           // maximum forecasting horizon

    val mod = new SimpleMovingAverage (y_, hh)                            // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on Simple Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    val yf_ = mod.getYf
    mod.diagnoseAll (y_, yf_)
    println (s"Final In-ST Forecast Matrix yf = $yf_")

end simpleMovingAverageTest5

