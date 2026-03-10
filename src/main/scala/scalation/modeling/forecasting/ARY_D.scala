
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y (ARY_D) using OLS - Direct Forecasting
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._
import scalation.modeling.neuralnet.{RegressionMV => REGRESSION}

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
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARY_D (x: MatrixD, y: MatrixD, hh: Int, fname: Array [String],
             tRng: Range = null, hparam: HyperParameter = hp,
             bakcast: Boolean = false,
             tForms: TransformMap = Map ("tForm_y" -> null))
      extends Forecaster_D (x, y, hh, fname, tRng, hparam, bakcast):

    private val debug = debugf ("ARY_D", true)                            // debug function
    private val p     = hparam("p").toInt                                 // use the last p values (p lags)
    private val spec  = hparam("spec").toInt                              // trend terms: 0 - none, 1 - constant, 2 - linear, 3 - quadratic
                                                                          //              4 - sine, 5 cosine
    private val nneg  = hparam("nneg").toInt == 1                         // 0 => unrestricted, 1 => predictions must be non-negative
    private val reg   = new REGRESSION (x, y, fname, hparam)              // delegate training to multi-variate regression

    _modelName = s"ARY_D_$p"
    yForm      = tForms("tForm_y").asInstanceOf [Transform]

    debug ("init", s"$modelName with additional term spec = $spec")
//  debug ("init", s"[ x | y ] = ${x ++^ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `ARY_D` model to the times-series data in vector y_.
     *  Estimate the coefficient vector b for a p-th order Auto-Regressive ARY_D(p) model.
     *  Uses OLS Matrix Fatorization to determine the coefficients, i.e., the b (φ) vector.
     *  @param x_  the data/input matrix (e.g., full x)
     *  @param y_  the training/full response vector (e.g., full y)
     */
    def train_x (x_ : MatrixD, y_ : MatrixD): Unit =
        debug ("train_x", s"$modelName, x_.dim = ${x_.dim}, y_.dim = ${y_.dim}")
        val idx = y_(?, y.dim2-1).indexOf (NO_DOUBLE)                     // index of first non-value in the last column
        val (x_t, y_t) = if idx < 0 then (x_, y_) else (x_(0 until idx), y_(0 until idx))
        reg.train (x_t, y_t)                                              // train the multi-variate regression model
        bb = reg.parameter                                                // coefficients from regression
    end train_x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = x, fname_ : Array [String] = reg.getFname,
                          b_ : VectorD = b, vifs: VectorD = reg.vif ()): String =
        super.summary (x_, fname_, b_, vifs)                              // summary from `Fit`
    end summary

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
//      if t < y_.dim then
//          debug ("predict", s"@t = $t, x(t) = ${x(t)}, yp = $yp vs. y_ = ${y_(t)}")
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
     */
    override def forecastAll (y_ : MatrixD): MatrixD =
        for t <- y_.indices do
            val pred = predict (t, y_)
            for h <- 1 to hh do yf(t, h) = pred(h-1)
        yf
    end forecastAll

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
               tRng: Range = null, hparam: HyperParameter = hp,
               bakcast: Boolean = false): ARY_D =

        val p     = hparam("p").toInt                                   // use the last p values
        val spec  = hparam("spec").toInt                                // 0 - none, 1 - constant, 2 - linear, 3 -quadratic, 4 - sin, 5 = cos
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
                 tRng: Range = null, hparam: HyperParameter = hp,
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARY_D =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        val p       = hparam("p").toInt                                 // use the last p values
        val spec    = hparam("spec").toInt                              // 0 - none, 1 - constant, 2 - linear, 3 -quadratic, 4 - sin, 5 = cos
        val tForm_y = tFormT.form(y)
        val y_scl   = tForm_y.f(y)

        val tForms = Map ("tForm_y" -> tForm_y)
        val xy     = ARY.buildMatrix (y_scl, hparam, bakcast)
        val fname  = if fname_ == null then formNames (spec, p) else fname_
        val yy     = makeMatrix4Y (y_scl, hh, bakcast)
        new ARY_D (xy, yy, hh, fname, tRng, hparam, bakcast, tForms)
    end rescale

end ARY_D

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_DTest` main function tests the `ARY_D` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRY_DTest
 */
@main def aRY_DTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon
    hp("p")    = 3                                                      // endo lags
    hp("spec") = 1                                                      // trend specification: 0, 1, 2, 3, 5

    val mod = ARY_D (y, hh)                                             // create model for time series data
    mod.inSample_Test ()                                                // In-Sample Testing
//  println (mod.summary ())                                            // statistical summary  FIX -- crashes

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
    hp("p")    = 3                                                      // endo lags
    hp("spec") = 2                                                      // trend specification: 0, 1, 2, 3, 5

    val mod = ARY_D (y, hh)                                             // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.rollValidate ()                                                 // TnT with Rolling Validation
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))     // only diagnose on the testing set
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRY_DTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_DTest3` main function tests the `ARY_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRY_DTest3
 */
@main def aRY_DTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon

    for p <- 1 to 6 do                                                  // number of lags
        hp("p") = p  
        val mod = ARY_D (y, hh)                                         // create model for time series data
        mod.inSample_Test ()                                            // In-Sample Testing
//      println (mod.summary ())                                        // statistical summary -- FIX crashes
    end for

end aRY_DTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_DTest4` main function tests the `ARY_D` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRY_DTest4
 */
@main def aRY_DTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon

    for p <- 5 to 5; s <- 1 to 1 do                                     // number of lags; trend
        hp("p")    = p
        hp("spec") = s
        val mod = ARY_D (y, hh)                                         // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest_x ()()                                           // use customized trainNtest_x

        mod.setSkip (0)
        mod.rollValidate (rc = 2)
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))    // only diagnose on the testing set
        println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRY_DTest4

