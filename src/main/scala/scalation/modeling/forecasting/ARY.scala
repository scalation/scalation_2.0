
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y (ARY) using OLS
 *
 *  @see `scalation.modeling.Regression`
 *  @see `scalation.modeling.forecasting.ARX` when exogenous variable are needed
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARY` class provides basic time series analysis capabilities for ARY models.
 *  ARY models utilize multiple linear regression based on lagged values of y.
 *  Given time series data stored in vector y, its next value y_t = combination of last
 *  p values of y.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y) @see `ARY.apply`
 *  @param y        the response/output vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARY (x: MatrixD, y: VectorD, hh: Int, fname: Array [String],
           tRng: Range = null, hparam: HyperParameter = hp,
           bakcast: Boolean = false,
           tForms: TransformMap = Map ("tForm_y" -> null))
      extends Forecaster_Reg (x, y, hh, fname, tRng, hparam, bakcast):   // no automatic backcasting, @see `ARY.apply`

    private val debug  = debugf ("ARY", true)                           // debug function
    protected val p    = hparam("p").toInt                              // use the last p values (p lags)
    protected val spec = hparam("spec").toInt                           // trend terms: 0 - none, 1 - constant, 2 - linear, 3 - quadratic
                                                                        //              4 - sine, 5 cosine
    _modelName = s"ARY_$p"
    yForm      = tForms("tForm_y").asInstanceOf [Transform]             // yForm defined in `Fit` via hierarchy

    debug ("init", s"$modelName with additional term spec = $spec")
//  debug ("init", s"[ x | y ] = ${x :^+ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a new vector from the first spec values of x, the last p-h+1 values
     *  of x (past values) and recent values 1 to h-1 from the forecasts.
     *  @param xx  the t-th row of the input matrix (lagged actual values)
     *  @param yy  the t-th row of the forecast matrix (forecasted future values)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forge (xx: VectorD, yy: VectorD, h: Int): VectorD =
        val n_endo  = spec + p                                          // number of trend + endogenous values
        val x_trend = xx(0 until spec)                                  // get trend values
        val x_act   = xx(n_endo-(p+1-h) until n_endo)                   // get actual lagged y-values (endogenous)
        val nyy     = p - x_act.dim                                     // number of forecasted values needed
        val x_fcast = yy(h-nyy until h)                                 // get forecasted y-values

        x_trend ++ x_act ++ x_fcast
    end forge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build an `ARY` model using the cols with the selected features.
     *  @param cols  the cols of the input matrix with selected features
     */
    def convertReg2Forc (cols: LSET [Int] = mcols): ARY =
        new ARY (getX(?, cols), getY, hh, cols.toArray.map (fname(_)), tRng, hparam, bakcast, tForms)
    end convertReg2Forc

end ARY


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARY` companion object provides factory methods for the `ARY` class.
 */
object ARY extends MakeMatrix4TSY:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY` object by making/building an input matrix x and then calling the
     *  `ARY` constructor.
     *  @param y        the response vector (time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = hp,
               bakcast: Boolean = false): ARY =

        val p     = hparam("p").toInt                                   // use the last p values
        val spec  = hparam("spec").toInt                                // 0 - none, 1 - constant, 2 - linear, 3 -quadratic, 4 - sin, 5 = cos
        val xy    = buildMatrix (y, hparam, bakcast)
        val fname = if fname_ == null then formNames (spec, p) else fname_
        new ARY (xy, y, hh, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY` object by building an input matrix xy and then calling the
     *  `ARY` constructor.  Also rescale the input data.
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormT   the transform for rescaling endogenous and exogenous
     */
    def rescale (y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = hp,
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARY =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        val p     = hparam("p").toInt                                   // use the last p values
        val spec  = hparam("spec").toInt                                // 0 - none, 1 - constant, 2 - linear, 3 -quadratic, 4 - sin, 5 = cos

        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormT.form(y(0 until tr_size))                   // use (mean, std) of training set for both In-sample and TnT
//      val tForm_y = tForm(y)                                          // use full dataset

        val y_scl   = tForm_y.f(y)
        val tForms  = Map ("tForm_y" -> tForm_y)

        val xy    = buildMatrix (y_scl, hparam, bakcast)
        val fname = if fname_ == null then formNames (spec, p) else fname_
        new ARY (xy, y_scl, hh, fname, tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the p + spec columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  @param y        the response vector (time series data)
     *  @param hp_      the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def buildMatrix (y: VectorD, hp_ : HyperParameter, bakcast: Boolean = false): MatrixD =
        val (p, spec, lwave) = (hp_("p").toInt, hp_("spec").toInt, hp_("lwave").toDouble)
        makeMatrix4T (y, spec, lwave, bakcast) ++^                   // trend terms
        makeMatrix4L (y, p, bakcast)                                 // regular lag terms
    end buildMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY` object for the special case of ARY(1) and use `SimpleRegression`.
     *  @param  the response vector (time series data)
     */
    def ary1 (y: VectorD): SimpleRegression =
        val x = WeightedMovingAverage.backcast (y) +: y(0 until y.dim-1)
        println (MatrixD (x, y).ᵀ)
        SimpleRegression (x, y, null)
    end ary1

end ARY

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRYTest` main function tests the `ARY` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRYTest
 */
@main def aRYTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon
    hp("p")    = 3                                                      // endo lags
    hp("spec") = 2                                                      // trend specification: 0, 1, 2, 3, 5

    val mod = ARY (y, hh)                                               // create model for time series data
    mod.inSample_Test ()                                                // In-Sample Testing
    println (mod.summary ())                                            // statistical summary

end aRYTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRYTest2` main function tests the `ARY` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRYTest2
 */
@main def aRYTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon
    hp("p")    = 3                                                      // endo lags
    hp("spec") = 2                                                      // trend specification: 0, 1, 2, 3, 5

    val mod = ARY (y, hh)                                               // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.rollValidate ()                                                 // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRYTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRYTest3` main function tests the `ARY` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRYTest3
 */
@main def aRYTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks) 

    for p <- 6 to 6; s <- 1 to 1 do                                     // number of lags; trend
        hp("p")    = p                                                  // endo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5

        val mod = ARY (y, hh)                                           // create model for time series data
        mod.inSample_Test ()                                            // In-Sample Testing
        println (mod.summary ())                                        // statistical summary of fit
    end for

end aRYTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRYTest4` main function tests the `ARY` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRYTest4
 */
@main def aRYTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("lwave") =     20                                                // wavelength (distance between peaks) 

    for p <- 6 to 6; s <- 1 to 1 do                                     // number of lags; trend
        hp("p")     = p                                                 // endo lags
        hp("spec")  = s                                                 // trend specification: 0, 1, 2, 3, 5
        val mod = ARY (y, hh)                                           // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest_x ()()                                           // use customized trainNtest_x

        mod.setSkip (0)
        mod.rollValidate (rc = 2)                                       // TnT with Rolling Validation
        println (s"After Roll TnT Forecast Matrix yf = ${mod.getYf}")
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))   // only diagnose on the testing set
//      println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRYTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRYTest5` main function tests the `ARY` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs feature selection.
 *  > runMain scalation.modeling.forecasting.aRYTest5
 */
@main def aRYTest5 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("spec")  = 5                                                     // trend specification: 0, 1, 2, 3, 5
    hp("lwave") = 20                                                    // wavelength (distance between peaks)

    val mod = ARY (y, hh)                                               // create model for time series data
    mod.inSample_Test ()                                                // In-Sample Testing
    println (mod.summary ())                                            // statistical summary of fit

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                              // R^2, R^2 bar, sMAPE, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                            // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    println (s"k = $k")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end aRYTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRYTest6` main function tests the `ARY` object's ability to make/build input
 *  matrices.  Build an input/predictor data matrix for the COVID-19 dataset.
 *  > runMain scalation.modeling.forecasting.aRYTest6
 */
@main def aRYTest6 (): Unit =

    val yy    = loadData_y ()
//  val y     = yy                                                      // full
    val y     = yy(0 until 116)                                         // clip the flat end
    val p     = 2                                                       // the number of lags
    val spec  = 1                                                       // trend specification
    val lwave = 6                                                       // wavelength (distance between peaks)

    val x = makeMatrix4T (y, spec, lwave, false) ++ makeMatrix4L (y, p, false)

    println (s"y = $y \n x = $x")

    val mod = ARY.ary1 (y)                                              // one lag (p = 1)
    banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()

end aRYTest6

