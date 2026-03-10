//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Ruthvik Mankari
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Simple Moving Average (not the same as MA in ARMA)
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._
import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleMovingAverage2` class provides basic time series analysis capabilities for
 *  SimpleMovingAverage2 models.  SimpleMovingAverage2 models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = mean of last q values.
 *
 *      y_t = mean (y_t-1, ..., y_t-q) + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param y        the response vector (time series data)
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to SimpleMovingAverage2.hp)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class SimpleMovingAverage2 (y: VectorD, hh: Int, tRng: Range = null,
                           hparam: HyperParameter = SimpleMovingAverage2.hp,
                           bakcast: Boolean = false)
  extends Forecaster (y, hh, tRng, hparam, bakcast)
    with NoSubModels:

    private val flaw = flawf ("SimpleMovingAverage2")                    // flaw function
    private val q    = hparam("q").toInt                                // take mean of last q values

    b          = VectorD.one (q) / q                                    // equal weight
    _modelName = s"SimpleMovingAverage2_$q"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = f (y_t-1, ...) = mean of last q values    (simple moving average model)
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions (mean (inclusive, exclusice))
     */
    override def predict (t: Int, y_ : VectorD): Double = y_.mean (max0 (t-q), t)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with rolling validation (analog of predict method).
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     *  @author Ruthvik Mankari
     */
    override def forecast(t: Int, y_ : VectorD = yb): VectorD =
        val yh = new VectorD(hh)

        for h <- 1 to hh do
            // Check if enough history exists
            if t < q + (h - 1) then
                yf(t, h) = -0.0
                yh(h - 1) = -0.0
            else
                // Copy last q points from actuals (up to a = t - h)
                var window = VectorD((t - h - q + 1 to t - h).map(y_(_)).toArray)
                var pred = 0.0

                // Perform recursive forecasting
                for _ <- 0 until h do
                    pred = window.mean
                    window = window(1 until q) :+ pred // slide window forward

                yf(t, h) = pred
                yh(h - 1) = pred
        end for

        yh
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the h-steps ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1.
     *  @see `forecastAll` method in `Forecaster` trait.
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the actual values to use in making forecasts
     * @author Ruthvik Mankari
     */
    override def forecastAt(h: Int, y_ : VectorD = yb): VectorD =
        if h < 1 then flaw("forecastAt", s"horizon h = $h must be ≥ 1")

        for t <- y_.indices do
            if t < q + (h - 1) then
                yf(t, h) = -0.0
            else
                var window = VectorD((t - h - q + 1 to t - h).map(y_(_)).toArray)
                var pred = 0.0
                for _ <- 0 until h do
                    pred = window.mean
                    window = window(1 until q) :+ pred
                yf(t, h) = pred
        end for

        yf(?, h)
    end forecastAt
end SimpleMovingAverage2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleMovingAverage2` companion object provides factory methods for the
 *  `SimpleMovingAverage2` class.
 */
object SimpleMovingAverage2:

    /** Base hyper-parameter specification for `SimpleMovingAverage2` and `WeightedMovingAverage2` classes
     */
    val hp = new HyperParameter
    hp += ("q", 2, 2)                           // number of prior values for mean
    hp += ("u", 1.0, 1.0)                       // slider from flat (0.0) to linear (1.0) weights

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SimpleMovingAverage2` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = hp): SimpleMovingAverage2 =
        new SimpleMovingAverage2 (y, hh, tRng, hparam)
    end apply

end SimpleMovingAverage2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverage2Test` main function tests the `SimpleMovingAverage2` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.simpleMovingAverage2Test
 */
@main def simpleMovingAverage2Test (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new SimpleMovingAverage2 (y, hh)                             // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end simpleMovingAverage2Test


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverage2Test2` main function tests the `SimpleMovingAverage2` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.simpleMovingAverage2Test2
 */
@main def simpleMovingAverage2Test2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new SimpleMovingAverage2 (y, hh)                             // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end simpleMovingAverage2Test2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverage2Test3` main function tests the `SimpleMovingAverage2` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.simpleMovingAverage2Test3
 */
@main def simpleMovingAverage2Test3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon
//  SimpleMovingAverage2.hp("q") = 3.0

    val mod = new SimpleMovingAverage2 (y, hh)                             // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end simpleMovingAverage2Test3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverage2Test4` main function tests the `SimpleMovingAverage2` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.simpleMovingAverage2Test4
 */
@main def simpleMovingAverage2Test4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new SimpleMovingAverage2 (y, hh)                             // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end simpleMovingAverage2Test4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleMovingAverage2Test5` main function tests the `SimpleMovingAverage2` class
 *  on a simple dataset using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.simpleMovingAverage2Test5
 */
@main def simpleMovingAverage2Test5 (): Unit =

    import SimpleMovingAverage2.hp

    val y_ = VectorD (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)

    hp("q") = 3                                                           // size of moving average window: test 2 and 3
    val hh  = 5                                                           // maximum forecasting horizon

    val mod = new SimpleMovingAverage2 (y_, hh)                           // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on Simple Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    val yf_ = mod.getYf
    mod.diagnoseAll (y_, yf_)
    println (s"Final In-ST Forecast Matrix yf = $yf_")

end simpleMovingAverage2Test5

