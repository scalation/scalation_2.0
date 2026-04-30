
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Random Walk Adjusted by Slope
 *
 *  @see     `RandomWalk` in RandomWalk.scala -- the regular random walk (s = 0)
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomWalkS` class provides basic time series analysis capabilities for
 *  RandomWalkS models.  RandomWalkS models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t is the previous
 *  value adjusted by the slope weighted by s. 
 *
 *      y_t = y_t-1 + s (y_t-1 - y_t-2) + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param y        the response vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `RandomWalkS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class RandomWalkS (y: VectorD, hh: Int, tRng: Range = null,
                   hparam: HyperParameter = RandomWalkS.hp,
                   bakcast: Boolean = false)
      extends Forecaster (y, hh, tRng, hparam, bakcast)
         with NoSubModels:

    private val sw = hparam("sw").toDouble                              // slope weight (same as RW when sw = 0

    _modelName = "RandomWalkS"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast based on the 
     *  a slope (S) adjusted random walk.
     *
     *      y_t+1 = y_t + s (y_t - y_t-1)
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    override def predict (t: Int, y_ : VectorD): Double =
        super.predict (t, y_) + sw * (y_(max0(t-1)) - y_(max0(t-2)))
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
            val pred = predict (t, y_)                                  // use only actual values
            yf(t, h) = pred                                             // record in forecast matrix
            yh(h-1)  = pred                                             // record forecasts for each horizon
        yh                                                              // return forecasts for all horizons
    end forecast

end RandomWalkS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomWalkS` companion object provides factory methods for the
 *  `RandomWalkS` class.
 */
object RandomWalkS:

    /** Base hyper-parameter specification for the `RandomWalkS` classes
     */
    val hp = new HyperParameter
    hp += ("sw", 0.1, 0.1)                           // slope weight (sw) used for slope adjustment

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RandomWalkS` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null,
               hparam: HyperParameter = hp): RandomWalkS =
        new RandomWalkS (y, hh, tRng, hparam)
    end apply

end RandomWalkS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkSTest` main function tests the `RandomWalkS` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.randomWalkSTest
 */
@main def randomWalkSTest (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new RandomWalkS (y, hh)                                     // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end randomWalkSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkSTest2` main function tests the `RandomWalkS` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.randomWalkSTest2
 */
@main def randomWalkSTest2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new RandomWalkS (y, hh)                                     // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end randomWalkSTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkSTest3` main function tests the `RandomWalkS` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.randomWalkSTest3
 */
@main def randomWalkSTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    for i <- 1 to 10 do
        RandomWalkS.hp("s") = i / 10.0                                    // tune the slope weight
        val mod = new RandomWalkS (y, hh)                                 // create model for time series data
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()                                               // train and test on full dataset

        mod.forecastAll ()                                                // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (y, mod.getYf)
        println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    end for

end randomWalkSTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkSTest4` main function tests the `RandomWalkS` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.randomWalkSTest4
 */
@main def randomWalkSTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    for i <- 1 to 10 do
        RandomWalkS.hp("s") = i / 10.0                                    // tune the slope weight
        val mod = new RandomWalkS (y, hh)                                 // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()

        mod.rollValidate ()                                               // TnT with Rolling Validation
        println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end randomWalkSTest4

