
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Simple Exponential Smoothing
 */

package scalation
package modeling
package forecasting

import scala.math.min

import scalation.mathstat._
//import scalation.optimization.quasi_newton.{BFGS => Optimizer}       // change import to change optimizer
//import scalation.optimization.quasi_newton.{LBFGS => Optimizer}
import scalation.optimization.quasi_newton.{LBFGS_B => Optimizer}
import scalation.optimization.quasi_newton.LBFGS_B.makeBounds

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleExpSmoothing` class provides basic time series analysis capabilities for
 *  Simple Exponential Smoothing models.  SimpleExpSmoothing models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = mean of last q values.
 *
 *      s_t  = α y_t-1 + (1 - α) s_t-1       smoothing equation
 *      yf_t = s_t                           forecast equation
 *
 *  where vector s is the smoothed version of vector y.
 *  @param y        the response vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to SimpleExpSmoothing.hp)`
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class SimpleExpSmoothing (y: VectorD, hh: Int, tRng: Range = null,
                          hparam: HyperParameter = SimpleExpSmoothing.hp,
                          bakcast: Boolean = false)  
      extends Forecaster (y, hh, tRng, hparam, bakcast)
         with Filter (y)
         with NoSubModels:

    private val TOL   = 1E-4                                             // tolerance
    private val lo_up = makeBounds (1, 0.0, 1.05)                        // lower & upper bounds on α for optimizer (1.0 + slack)

    private var α     = hparam("α").toDouble                             // default value for the smoothing parameter
    private var s     = VectorD.nullv                                    // vector of smoothed/leveled values (state)
//  private val sf    = new VectorD (y.dim)                              // to hold smooth values for a forecast horizon
    private var opt   = true                                             // whether to optimize the smoothing parameter

    _modelName = "SimpleExpSmoothing"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the smoothing parameter α.
     *  @param a  the smoothing parameter
     */
    def reset (a: Double): Unit = α = a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Toggle the opt flag that indicates whether optimization should be used to set α.
     */
    def toggleOpt (): Unit = opt = ! opt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Smooth the time-series data y, returning the leveled/smoothed data s.
     *  May be viewed as unoptimized training.
     *  @see Smoothing Equation in section 7.1.
     *      s_t+1 = α y_t + (1 - α) s_t                                  // smoothing equation
     *  @param y_  the response/output vector (training/full)
     *  @param a   the smoothing parameter (decay rate for older values)
     */
    def smooth (y_ : VectorD = y, a: Double = α): VectorD =
        s = new VectorD (y_.dim)
        s(0) = y(0)
        for t <- 0 until y_.dim-1 do s(t+1) = a * y_(t) + (1 - a) * s(t)
        s
    end smooth

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `SimpleExpSmoothing` model to the times-series data in vector y_.
     *  If `opt` is true, optimize the smoothing parameter α.
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the training/full response vector (e.g., full y)
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =

        def f_obj (x: VectorD): Double = (y_ - smooth (y_, x(0))).normSq   // only one parameter

        if opt then
            val optimizer = new Optimizer (f_obj, l_u = lo_up)             // Bounded Quasi-Newton optimizer
//          val optimizer = new Optimizer (f_obj)                          // Quasi-Newton optimizer
            val opt = optimizer.solve (VectorD (α), toler = TOL)           // optimize value for α
            α = (opt._2)(0)                                                // pull α from vector result
        s = smooth (y_, α)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter vector for the SimpleExpSmoothing model.
     */
    override def parameter: VectorD = VectorD (α)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = s_t
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    override def predict (t: Int, y_ : VectorD): Double = s(min (t, s.dim-1))

end SimpleExpSmoothing


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleExpSmoothing` companion object provides factory methods for the
 *  `SimpleExpSmoothing` class.
 */
object SimpleExpSmoothing:

    /** Base hyper-parameter specification for the `SimpleExpSmoothing` class
     */
    val hp = new HyperParameter
    hp += ("α", 0.9, 0.9)                           // default value for the smoothing parameter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SimpleExpSmoothing` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = hp): SimpleExpSmoothing =
        new SimpleExpSmoothing (y, hh, tRng, hparam)
    end apply

end SimpleExpSmoothing


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleExpSmoothingTest` main function tests the `SimpleExpSmoothing` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.simpleExpSmoothingTest
 */
@main def simpleExpSmoothingTest (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new SimpleExpSmoothing (y, hh)                              // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end simpleExpSmoothingTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleExpSmoothingTest2` main function tests the `SimpleExpSmoothing` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.simpleExpSmoothingTest2
 */
@main def simpleExpSmoothingTest2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new SimpleExpSmoothing (y, hh)                              // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end simpleExpSmoothingTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleExpSmoothingTest3` main function tests the `SimpleExpSmoothing` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.simpleExpSmoothingTest3
 */
@main def simpleExpSmoothingTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new SimpleExpSmoothing (y, hh)                              // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end simpleExpSmoothingTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleExpSmoothingTest4` main function tests the `SimpleExpSmoothing` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.simpleExpSmoothingTest4
 */
@main def simpleExpSmoothingTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new SimpleExpSmoothing (y, hh)                              // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end simpleExpSmoothingTest4

