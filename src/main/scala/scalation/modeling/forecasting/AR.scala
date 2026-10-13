
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

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AR` class provides basic time series analysis capabilities for Auto-Regressive
 *  (AR) models.  AR models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = δ +  b ∙ [ y_t-p, ... y_t-2, y_t-1 ] + e_t
 *      y_t = δ + b_0 y_t-p + ... + b_p-2 y_t-2 + b_p-1 y_t-1 + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param y         the response vector (time series data) 
 *  @param hh        the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng      the time range, if relevant (time index may suffice)
 *  @param hparam    the hyper-parameters (defaults to AR.hp)
 *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
 *  @param adjusted  whether in `Correlogram` when calculating auto-covariances/auto-correlations
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

    private   val flaw = flawf ("AR")                                  // flaw function
    protected val p    = hparam("p").toInt                             // use the last p values
    protected var δ    = NO_DOUBLE                                     // drift/intercept/constant term

    _modelName = s"AR_$p"
    yForm      = tForm                                                 // defined in `Fit` via hierarchy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `AR` model to the times-series data in vector y_.
     *  Estimate the coefficient vector b (φ) for a p-th order Auto-Regressive AR(p) model.
     *  Uses Durbin-Levinson Algorithm (in `Correlogram`) to determine the coefficients.
     *  The b (φ) vector is p-th row of psi matrix (ignoring the first (0t)ŷ.
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the training/full response vector (e.g., full y)
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =
        makeCorrelogram (y_)                                           // correlogram computes psi matrix
        b = psiM(p)(1 until p+1).reverse                               // coefficients = p-th row, columns 1, 2, ... p
        δ = statsF.mu * (1 - b.sum)                                    // compute drift/intercept
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter vector for the AR(p) model (drift δ, b = φ).
     */
    override def parameter: VectorD = δ +: b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *  @note: order of variables is set to be compatible with `ARY`.
     *
     *      ŷ_t = δ + b ∙ [ y_t-p, ... y_t-2, y_t-1 ]
     *      ŷ_t = δ + b_0 y_t-p + ... + b_p-2 y_t-2 + b_p-1 y_t-1
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    override def predict (t: Int, y_ : VectorD): Double =
        val x = y_.lag (p, t)                                          // x = [ y_t-p, ... y_t-2, y_t-1 ]
        δ + b ∙ x                                                      // drift + AR terms, b = φ
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a x-vector from old enough actual lagged y-values and use previously forecasted
     *  values from the FORECAST MATRIX yf for the rest. 
     *
     *  h = 2: ŷ_t+1 form x = [ y_t-p, ... y_t-3, y-2, ŷ_t-1 ] 
     *  h = 3: ŷ_t+2 form x = [ y_t-p, ... y_t-3, ŷ-2, ŷ_t-1 ] 
     *
     *  @patam yf  the forecast matrix
     *  @param t   the time point from which to make forecasts
     *  @param h   the forecasting horizon
     */
    def forge (yf: MatrixD, t: Int, h: Int): VectorD =
        val x_act   = yf(?, 0).lag (max0 (p-h+1), t)                   // get actual lagged y-values (endogenous)
        val x_fcast = yf(t)(1 until h)                                 // get forecasted y-values
        x_act ++ x_fcast                                               // x-vector from old actual and forecasted values
    end forge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with ROLLING VALIDATION (analog of predict method).
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     */
    override def forecast (t: Int, y_ : VectorD = yb): VectorD =
        val yh = new VectorD (hh)                                      // hold forecasts for each horizon
        for h <- 1 to hh do
            val x = forge (yf, t, h)                                   // get actual and forecasted values
            val pred = δ + b ∙ x                                       // make forecast prediction
            yf(t, h) = pred                                            // record in forecast matrix
            yh(h-1)  = pred                                            // record forecasts for each horizon
        yh                                                             // return forecasts for all horizons
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

        for t <- y_.indices do                                         // make forecasts over all time points for horizon h
            val x = forge (yf, t, h)
            yf(t, h) = δ + b ∙ x                                       // record in forecast matrix
        yf(?, h)                                                       // return the h-step ahead forecast vector
    end forecastAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF SUMMARY for a model with diagnostics for each predictor x_j
     *  and the overall Quality of Fit (QoF).
     *  FIX - numbers look wrong
     *  @param qof    the Quality of Fit vector (x_ in regular summary)
     *  @param fname  the array of feature/variable names
     *  @param b      the parameters/coefficients for the model
     *  @param vifs   the Variance Inflation Factors (VIFs)
     */
    def summary_ (qof: VectorD, fname: Array [String] = null,
                  b: VectorD = parameter,
                  vifs: VectorD = null): String =

        val (_sse, _rse, _rSq, _rSqBar) = (qof(QoF.sse.ordinal), qof(QoF.rse.ordinal), qof(QoF.rSq.ordinal), qof(QoF.rSqBar.ordinal))

        val stdErr = stdErrAR (p + 1, _sse, df)                        // standard errors of coefficients

        val stats  = (sumCoeff (b, stdErr, vifs), fmt(_rse), fmt(_rSq), fmt(_rSqBar))

        (if fname != null then "fname = " + stringOf (fname) else "") +
        s"""
SUMMARY
    Parameters/Coefficients:
    Var      Estimate    Std. Error \t t value \t Pr(>|t|) \t VIF
----------------------------------------------------------------------------------
${stats._1}
    Residual standard error: ${stats._2} on $df degrees of freedom
    Multiple R-squared:  ${stats._3},   Adjusted R-squared:  ${stats._4}
    F-statistic: $fStat_ on $dfr and $df DF,  p-value: $p_fS_
----------------------------------------------------------------------------------
        """
    end summary_

end AR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AR` companion object provides factory methods for the
 *  `AR` class.
 */
object AR:

    /** Base hyper-parameter specification for the `AR`, ARMA, ARIMA, `SARIMA` and `ARX*` classes
     */
    val hp = new HyperParameter
    hp += ("p", 1, 1)                               // number of Auto-Regressive (AR) parameters
    hp += ("d", 1, 1)                               // number of Differences to take
    hp += ("q", 1, 1)                               // number of Moving-Average (MA) parameters
    hp += ("P", 1, 1)                               // number of Seasonal Auto-Regressive (AR) parameters
    hp += ("D", 1, 1)                               // number of Seasonal Differences to take
    hp += ("Q", 1, 1)                               // number of Seasonal Moving-Average (MA) parameters
    hp += ("s", 7, 7)                               // length of the Seasonal Period

    hp += ("px", 1, 1)                              // number of lags for the eXogenous variables
    hp += ("spec", 1, 1)                            // trend terms: 0 - implicit constant via centering, 1 - linear, 2 - quadratic, 3 - cubic
    hp += ("crx", 0, 0)                             // 0 - no ENDO-EXO cross terms, 1 - include ENDO-EXO cross terms
    hp += ("nneg", 1, 1)                            // 0 - unrestricted, 1 - predictions must be non-negative

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `AR` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     *  @param adjusted  whether in `Correlogram` when calculating auto-covariances/auto-correlations
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

    mod.setSkip ()                                                        // can use values from training set to not skip any in test
    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRTest2

import Example_Covid.{clip, loadData_y}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest3` main function tests the `AR` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRTest3
 */
@main def aRTest3 (): Unit =

    val (_, y) = clip ((null, loadData_y ()))
    val hh     = 6                                                        // maximum forecasting horizon

    for p <- 1 to 5 do
        AR.hp("p") = p                                                    // number of AR terms
        val mod = new AR (y, hh)                                          // create model for time series data
//      val mod = new AR (y, hh, adjusted = false)                        // use conventional rho estimation
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        val qof = mod.inSample_Test ()._2                                 // In-sample Testing
        println (mod.summary_ (qof))                                      // statistical summary of fit
    end for

end aRTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRTest4` main function tests the `AR` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRTest4
 */
@main def aRTest4 (): Unit =

    val (_, y) = clip ((null, loadData_y ()))
    val hh     = 6                                                        // maximum forecasting horizon

    for p <- 1 to 5 do
        AR.hp("p") = p                                                    // number of AR terms
        val mod = new AR (y, hh)                                          // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                   // Train and Test with Rolling Validation
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

