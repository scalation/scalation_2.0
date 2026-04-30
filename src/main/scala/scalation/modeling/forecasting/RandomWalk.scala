
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Random Walk (RW)
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._
import scalation.random.Randi

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomWalk` class provides basic time series analysis capabilities for
 *  RandomWalk models.  RandomWalk models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t+1 = y(t+1)
 *  may be predicted based on its past value of y:
 *
 *      y_t+1 = y_t + e_t+1
 *
 *  where y_t is the previous value of y and e_t+1 is the new residual/error term.
 *  @param y        the response vector (time series data)
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (none => use null)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class RandomWalk (y: VectorD, hh: Int, tRng: Range = null,
                  hparam: HyperParameter = null,
                  bakcast: Boolean = false)
      extends Forecaster (y, hh, tRng, hparam, bakcast)
         with NoSubModels:

    _modelName = "RandomWalk"

end RandomWalk


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomWalk` companion object provides factory methods for the `RandomWalk` class.
 */
object RandomWalk:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RandomWalk` forecasting object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time vector, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters (defaults to null)
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = null): RandomWalk =
        new RandomWalk (y, hh, tRng, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a `RandomWalk` time series vector by recording integer locations for a random walk.
     *  @param y0   the previous value (e.g., from a training set)
     *  @param d    the maximum step size {-d, ..., d} in the random walk
     *  @param n    the number of steps to make in the random walk
     *  @param stm  the random number stream to use
     */
    def make (y0: Int, d: Int, n: Int, stm: Int = 0): VectorD =
        val rng = Randi (-d, d, stm)
        val y   = new VectorD (n)
        var x   = y0
        for i <- y.indices do { x += rng.igen; y(i) = x }
        y
    end make

end RandomWalk

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkTest` main function tests the `RandomWalk` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.randomWalkTest
 */
@main def randomWalkTest (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y, hh)                                      // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll (y)
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end randomWalkTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkTest2` main function tests the `RandomWalk` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.randomWalkTest2
 */
@main def randomWalkTest2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y, hh)                                      // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.setSkip (0)
    mod.rollValidate ()                                                   // TnT with Rolling Validation
    mod.diagnoseAll (y, mod.getYf, Forecaster.teRng (y.dim))              // only diagnose on the testing set
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end randomWalkTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkTest3` main function tests the `RandomWalk` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.randomWalkTest3
 */
@main def randomWalkTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y, hh)                                      // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll (y)                                                   // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end randomWalkTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkTest4` main function tests the `RandomWalk` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.randomWalkTest4
 */
@main def randomWalkTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y, hh)
    banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()

    mod.setSkip (0)
    mod.rollValidate ()                                                   // TnT with Rolling Validation
    mod.diagnoseAll (y, mod.getYf, Forecaster.teRng (y.dim))              // only diagnose on the testing set
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")              // flat forecast matrix yf
    println (s"Final TnT Forecast Matrix yf = ${mod.slant (mod.getYf)}")  // slanted yf -- easier to visualize

end randomWalkTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkTest5` main function tests the `RandomWalk` class on a simple dataset
 *  using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.randomWalkTest5
 */
@main def randomWalkTest5 (): Unit =

    val y_ = VectorD (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y_, hh)                                     // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on Simple Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y_, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end randomWalkTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkTest6` main function tests the `RandomWalk` class on small dataset.
 *  Test forecasts (h = 1 step ahead forecasts).
 *  > runMain scalation.modeling.forecasting.randomWalkTest6
 */
@main def randomWalkTest6 (): Unit =

    val y  = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3)

    val mod = new RandomWalk (y, 1)                                       // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "RW")

end randomWalkTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalkTest7` main function tests the `RandomWalk` class on small dataset.
 *  Test forecasts (h = 1 step ahead forecasts).
 *  > runMain scalation.modeling.forecasting.randomWalkTest7 <stm>
 *  @param stm  the random number stream to use (command-line argument, e.g., 2)
 */
@main def randomWalkTest7 (stm: Int): Unit =

    val y  = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3)
    val n  = y.dim
    val yy = RandomWalk.make (y(n-1).toInt, 2, n, stm)

    println (s"training set y  = $y")
    println (s"testing  set yy = $yy")

    new Plot (null, y ++ yy, null, "training and testing sets", lines = true)

end randomWalkTest7

