
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive (AR)
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AR` class provides basic time series analysis capabilities for Auto-Regressive
 *  (AR) models.  AR models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = b dot [1, y_t-1, ..., y_t-p) + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param y         the response vector (time series data) 
 *  @param hh        the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng      the time range, if relevant (time index may suffice)
 *  @param hparam    the hyper-parameters (defaults to AR.hp)
 *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
 *  @param adjusted  whether in `Correlogram` when calculating auto-covarainces/auto-correlations
 *                   to adjust to account for the number of elements in the sum Σ (or use dim-1)
 *                   @see `VectorD.acov`
 *  @oaram tForm     the transformation applied
 */
class AR (y: VectorD, hh: Int, tRng: Range = null,
          hparam: HyperParameter = AR.hp,
          bakcast: Boolean = false, adjusted: Boolean = true,
          tForm: Transform = null)
      extends Forecaster (y, hh, tRng, hparam, bakcast)
         with Correlogram (y, adjusted)
         with NoSubModels:

    private   val flaw = flawf ("AR")                                   // flaw function
    protected val p    = hparam("p").toInt                              // use the last p values
    protected var δ    = NO_DOUBLE                                      // drift/intercept/constant term

    _modelName = s"AR_$p"
    yForm      = tForm                                                  // defined in `Fit` via hierarchy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `AR` model to the times-series data in vector y_.
     *  Estimate the coefficient vector b (φ) for a p-th order Auto-Regressive AR(p) model.
     *  Uses Durbin-Levinson Algorithm (in `Correlogram`) to determine the coefficients.
     *  The b (φ) vector is p-th row of psi matrix (ignoring the first (0th) column).
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the training/full response vector (e.g., full y)
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =
        makeCorrelogram (y_)                                            // correlogram computes psi matrix
        b = psiM(p)(1 until p+1).reverse                                // coefficients = p-th row, columns 1, 2, ... p
        δ = statsF.mu * (1 - b.sum)                                     // compute drift/intercept
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter vector for the AR(p) model.
     */
    override def parameter: VectorD = δ +: b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = δ +  b_0 y_t-1 + b_1 y_t-2 + ... + b_p-1 y_t-p
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    override def predict (t: Int, y_ : VectorD): Double =
//      δ + rdot (b, y_, t-1)                                          // old way
        val x = y_.prior (p, t)                                        // x = [ y_t-p, ... y_t-1 ]
        δ + (b dot x)
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a vector from va;ues in the FORECAST MATRIX yf. 
     *  @patam yf  the forecast matrix
     *  @param t   the time point from which to make forecasts
     *  @param h   the forecasting horizon
     */
    def forge (yf: MatrixD, t: Int, h: Int): VectorD =
        val x_act   = yf(?, 0).prior (max0 (p-h+1), t)                 // get actual lagged y-values (endogenous)
        val x_fcast = yf(t)(1 until h)                                 // get forecasted y-values
        x_act ++ x_fcast
    end forge

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
//          val pred = δ + rdot (b, yf, t, h-1)                         // slide in prior forecasted values
            val x = forge (yf, t, h)
            val pred = δ + (b dot x)
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
//          yf(t, h) = δ + rdot (b, yf, t, h-1)                         // record in forecast matrix (old way)
            val x = forge (yf, t, h)
            yf(t, h) = δ + (b dot x)                                    // record in forecast matrix
        yf(?, h)                                                        // return the h-step ahead forecast vector
    end forecastAt

end AR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AR` companion object provides factory methods for the
 *  `AR` class.
 */
object AR:

    /** Base hyper-parameter specification for the `AR`, ARMA, ARIMA, `SARIMA` and  `SARIMAX` classes
     */
    val hp = new HyperParameter
    hp += ("p", 1, 1)                               // number of Auto-Regressive (AR) parameters
    hp += ("d", 1, 1)                               // number of Differences to take
    hp += ("q", 1, 1)                               // number of Moving-Average (MA) parameters
    hp += ("P", 1, 1)                               // number of Seasonal Auto-Regressive (AR) parameters
    hp += ("D", 1, 1)                               // number of Seasonal Differences to take
    hp += ("Q", 1, 1)                               // number of Seasonal Moving-Average (MA) parameters
    hp += ("s", 7, 7)                               // length of the Seasonal Period
    hp += ("a", 1, 1)                               // the first lag for the eXogenous variables
    hp += ("b", 2, 2)                               // the last lag for the eXogenous variables

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `AR` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     *  @param adjusted  whether in `Correlogram` when calculating auto-covarainces/auto-correlations
     *                   to adjust to account for the number of elements in the sum Σ (or use dim-1)
     *                   @see `VectorD.acov`
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = hp,
               bakcast: Boolean = false, adjusted: Boolean = true): AR =
        new AR (y, hh, tRng, hparam, bakcast, adjusted)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `AR` object by building an input matrix xy and then calling the
     *  `AR` constructor.  Also rescale the input data.
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tForm    the z-transform (rescale to standard normal)
     */
    def rescale (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = AR.hp,
                 bakcast: Boolean = false, adjusted: Boolean = true): AR =

        val tr_size = Model.trSize (y.dim)
        val tForm_y = NormForm (y(0 until tr_size))                     // use (mean, std) of training set for both In-sample and TnT
//      val tForm_y = tForm(y)                                          // use full dataset

        val y_scl = tForm_y.f(y)
        new AR (y_scl, hh, tRng, hparam, bakcast, adjusted, tForm_y)
    end rescale

end AR

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest` main function tests the `AR` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRTest
 */
@main def aRTest (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new AR (y, hh)                                              // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end aRTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest2` main function tests the `AR` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRTest2
 */
@main def aRTest2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new AR (y, hh)                                              // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.setSkip (0)                                                       // can use values from training set to not skip any in test
    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest3` main function tests the `AR` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRTest3
 */
@main def aRTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    for p <- 1 to 5 do
        AR.hp("p") = p                                                    // number of AR terms
        val mod = new AR (y, hh)                                          // create model for time series data
//      val mod = new AR (y, hh, adjusted = false)                        // use conventional rho estimation
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()                                               // train and test on full dataset

//      mod.setSkip (p)                                                   // full AR-formula available when t >= p
        mod.forecastAll ()                                                // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (y, mod.getYf)
//      println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
//      println (s"Final In-ST Forecast Matrix yf = ${mod.getYf.shiftDiag}")
    end for

end aRTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest4` main function tests the `AR` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRTest4
 */
@main def aRTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    for p <- 1 to 5 do
        AR.hp("p") = p                                                    // number of AR terms
        val mod = new AR (y, hh)                                          // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
//      mod.setSkip (p)                                                   // may wish to skip until all p past values are available
        mod.trainNtest ()()

        mod.setSkip (0)                                                   // can use values from training set to not skip any in test
        mod.rollValidate ()                                               // TnT with Rolling Validation
        println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest5` main function tests the `AR` class on small dataset.
 *  Test forecasts (h = 1 step ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRTest5
 */
@main def aRTest5 (): Unit =

    val y  = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3)

    var mod = new AR (y, 1)                                               // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "AR1")

    AR.hp ("p") = 2
    mod = new AR (y, 1)                                                   // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "AR2")

end aRTest5

