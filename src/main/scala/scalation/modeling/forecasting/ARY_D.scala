
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y (ARY_D) using OLS - Direct Forecasting
 *
 *  @see `scalation.modeling.forecasting.ARY` for recursive forecasting version
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARY_D` class provides basic time series analysis capabilities for
 *  ARY_D models.  ARY_D models are often used for forecasting.
 *  `ARY_D` uses DIRECT (as opposed to RECURSIVE) multi-horizon forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y) @see `ARY_D.apply`
 *  @param y        the response/output matrix (column per horizon) (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARY_D (x: MatrixD, y: MatrixD, hh: Int, fname: Array [String],
             tRng: Range = null, hparam: HyperParameter = AR.hp,
             bakcast: Boolean = false,
             tForms: TransformMap = Map ("tForm_y" -> null))
      extends Forecaster_D (x, y, hh, fname, tRng, hparam, bakcast):      // provides `reg` field

    private val debug = debugf ("ARY_D", false)                           // debug function
    private val p     = hparam("p").toInt                                 // use the last p values (p lags)
    private val spec  = hparam("spec").toInt                              // trend terms: 0 implicit constant, 1 - linear, 2 - quadratic, 3 - cubic
    private val nneg  = hparam("nneg").toInt == 1                         // 0 => unrestricted, 1 => predictions must be non-negative

    _modelName = s"ARY_D_$p"
    yForm      = tForms("tForm_y").asInstanceOf [Transform]

    debug ("init", s"$modelName with trend spec = $spec and reg = ${reg.modelName}")
//  debug ("init", s"[ x | y ] = ${x ++^ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = b_0 + b_1 y_t-1 + b_2 y_t-2 + ... + b_p y_t-p = b dot x_t
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions (ignored)
     */
    def predict (t: Int, y_ : MatrixD): VectorD =
        val yp = rectify (reg.predict (x(t)), nneg)
        if t < y_.dim then
            debug ("predict", s"@t = $t, x(t) = ${x(t)}, yp = $yp vs. y_ = ${y_(t)}")
        yp
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with rolling validation (analog of predict method).
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     */
    override def forecast (t: Int, y_ : VectorD): VectorD =
        val pred = predict (t, MatrixD (y_).ᵀ)
        yf(t, 1 until hh+1) = pred
        pred                                                              // yh is pred
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points all horizons h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the forecast matrix.
     *  @param y_  the matrix of actual response values
     * 
    override def forecastAll (y_ : MatrixD): MatrixD =
        for t <- y_.indices do
            val pred = predict (t, y_)
            for h <- 1 to hh do yf(t, h) = pred(h-1)
        yf
    end forecastAll
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build an `ARY_D` model using the cols with the selected features.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel (cols: LSET [Int] = mcols): ARY_D =
        new ARY_D (x(?, cols), y, hh, cols.toArray.map (fname (_)), tRng, hparam, bakcast, tForms)
    end getModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a single-horizon `ARY` model using the cols with the selected features.
     *  Note: uses `ARY` as it is the base model for ARY*_D.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel_h (cols: LSET [Int] = mcols, h: Int = 1): ARY =
        new ARY (x(?, cols), y(?, h-1), 1, cols.toArray.map (fname (_)), tRng, hparam, bakcast, tForms)
    end getModel_h

end ARY_D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARY_D` companion object provides factory methods for the `ARY_D` class.
 */
object ARY_D extends MakeMatrix4TSY:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY_D` object by building an input matrix x and then calling the constructor.
     *  @param y        the response vector (time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = AR.hp,
               bakcast: Boolean = false): ARY_D =

        val p     = hparam("p").toInt                                   // use the last p values
        val spec  = hparam("spec").toInt                                // 0 - implicit constant, 1 - linear, 2 - quadratic, 3 - cubic
        val xy    = ARY.buildMatrix (y, hparam, bakcast)
        val yy    = makeMatrix4Y (y, hh, bakcast)
        val fname = if fname_ == null then formNames (spec, p) else fname_
        new ARY_D (xy, yy, hh, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY_D` object by building an input matrix xy and then calling the
     *  `ARY_D` constructor.  Also rescale the input data.
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormT   the transform for rescaling endogenous and exogenous
     */ 
    def rescale (y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = AR.hp,
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARY_D =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        val p       = hparam("p").toInt                                 // use the last p values
        val spec    = hparam("spec").toInt                              // 0 - implicit constant, 1 - linear, 2 - quadratic, 3 - cubic
        val tForm_y = tFormT.form(y)
        val y_scl   = tForm_y.f(y)

        val tForms = Map ("tForm_y" -> tForm_y)
        val xy     = ARY.buildMatrix (y_scl, hparam, bakcast)
        val fname  = if fname_ == null then formNames (spec, p) else fname_
        val yy     = makeMatrix4Y (y_scl, hh, bakcast)
        new ARY_D (xy, yy, hh, fname, tRng, hparam, bakcast, tForms)
    end rescale

end ARY_D

import Example_LakeLevels.y
import AR.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_DTest` main function tests the `ARY_D` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRY_DTest
 */
@main def aRY_DTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    for p <- 1 to 5 do
        hp("p") = p                                                     // endo lags
        val mod = ARY_D (y, hh)                                         // create model for time series data
        banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
        mod.inSample_Test ()                                            // In-Sample Testing
        println (mod.summary ())                                        // statistical summary  FIX -- crashes
    end for

end aRY_DTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_DTest2` main function tests the `ARY_D` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRY_DTest2
 */
@main def aRY_DTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    for p <- 1 to 5 do
        hp("p") = p                                                     // endo lags
        val mod = ARY_D (y, hh)                                         // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRY_DTest2

import Example_Covid.{clip, loadData_y}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_DTest3` main function tests the `ARY_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRY_DTest3
 */
@main def aRY_DTest3 (): Unit =

    val (_, y) = clip ((null, loadData_y ()))
    val hh     = 6                                                      // maximum forecasting horizon
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    for p <- 1 to 5; s <- 0 to 0 do
        hp.set (("p", p), ("spec", s))                                  // # endo lags, trend specification: 0, 1, 2, 3
        val mod = ARY_D (y, hh)                                         // create model for time series data
        mod.inSample_Test ()                                            // In-Sample Testing
        println (mod.summary ())                                        // statistical summary -- FIX crashes
    end for

end aRY_DTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_DTest4` main function tests the `ARY_D` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRY_DTest4
 */
@main def aRY_DTest4 (): Unit =

    val (_, y) = clip ((null, loadData_y ()))
    val hh     = 6                                                      // maximum forecasting horizon
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    for p <- 1 to 5; s <- 0 to 1 do
        hp.set (("p", p), ("spec", s))                                  // # endo lags, trend specification: 0, 1, 2, 3
        val mod = ARY_D (y, hh)                                         // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
/*
        mod.trainNtest ()()                                             // use customized trainNtest
        mod.setSkip ()                                                  // don't skip any rows
        mod.rollValidate ()
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))    // only diagnose on the testing set
        println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
*/
    end for

end aRY_DTest4

