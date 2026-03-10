
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y with quadratic terms (ARY_Quad) using OLS
 *
 *  @see `scalation.modeling.Regression`
 *  @see `scalation.modeling.forecasting.ARX` when exogenous variable are needed
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARY_Quad` class provides basic time series analysis capabilities for ARY quadratic models.
 *  ARY quadratic models utilize quadratic multiple linear regression based on lagged values of y.
 *  Given time series data stored in vector y, its next value y_t = combination of last
 *  p values of y and y^2.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y and y^2) @see `ARY_Quad.apply`
 *  @param y        the response/output vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARY_Quad (x: MatrixD, y: VectorD, hh: Int, fname: Array [String],
                tRng: Range = null, hparam: HyperParameter = hp,
                bakcast: Boolean = false,
                tForms: TransformMap = Map ("tForm_y" -> null))
      extends ARY (x, y, hh, fname, tRng, hparam, bakcast, tForms):

    private val debug = debugf ("ARY_Quad", true)                       // debug function
    private val pow   = Transform.hp("p").toDouble                      // power to raise the endogenous lags to (defaults to quadratic)

    _modelName = s"ARY_Quad_$p"

    debug ("init", s"$modelName with additional term spec = $spec")
//  debug ("init", s"[ x | y ] = ${x :^+ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a new vector from the first spec values of x, the last p-h+1 values
     *  of x (past values) and recent values 1 to h-1 from the forecasts.
     *  @param xx  the t-th row of the input matrix (lagged actual values)
     *  @param yy  the t-th row of the forecast matrix (forecasted future values)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    override def forge (xx: VectorD, yy: VectorD, h: Int): VectorD =
        val n_endo   = spec + p                                         // number of trend + endogenous values
        val x_trend  = xx(0 until spec)                                 // get trend values
        val x_act    = xx(n_endo-(p+1-h) until n_endo)                  // get actual lagged y-values (endogenous)
        val nyy      = p - x_act.dim                                    // number of forecasted values needed
        val x_fcast  = yy(h-nyy until h)                                // get forecasted y-values

        val x2_act   = x_act ~^ pow                                     // get actual y^2-values
        val x2_fcast = x_fcast ~^ pow                                   // get forecasted y^2-values
        x_trend ++ x_act ++ x_fcast ++ x2_act ++ x2_fcast
    end forge

end ARY_Quad


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARY_Quad` companion object provides factory methods for the `ARY_Quad` class.
 */
object ARY_Quad extends MakeMatrix4TSY:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY_Quad` object by building an input matrix xy and then calling the
     *  `ARY_Quad` constructor.
     *  @param y        the response vector (time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = hp,
               bakcast: Boolean = false): ARY_Quad =

        val p     = hparam("p").toInt                                   // use the last p values
        val spec  = hparam("spec").toInt                                // 0 - none, 1 - constant, 2 - linear, 3 -quadratic, 4 - sin, 5 = cos
        val xy    = buildMatrix (y, hparam, bakcast)
        val fname = if fname_ == null then formNames (spec, p) else fname_
        new ARY_Quad (xy, y, hh, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY_Quad` object by building an input matrix xy and then calling the
     *  `ARY_Quad` constructor.  Also rescale the input data.
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
                 tFormT: TransformT = MinMax): ARY_Quad =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                   // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)

        val p     = hparam("p").toInt                                   // use the last p values
        val spec  = hparam("spec").toInt                                // 0 - none, 1 - constant, 2 - linear, 3 -quadratic, 4 - sin, 5 = cos
        val powForm = PowForm (VectorD (0, Transform.hp("p").toDouble))
        val tForms  = Map ("tForm_y" -> tForm_y, "powForm" -> powForm)
        val xy      = buildMatrix (y_scl, hparam, bakcast, powForm)
        val fname   = if fname_ == null then formNames (spec, p) else fname_
        new ARY_Quad (xy, y_scl, hh, fname, tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the p + spec columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  @param y        the response vector (time series data)
     *  @param hp_      the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param powForm  the power transform
     */
    def buildMatrix (y: VectorD, hp_ : HyperParameter, bakcast: Boolean,
                     powForm: Transform = PowForm (VectorD (0, Transform.hp("p").toDouble))): MatrixD =

        val (p, spec, lwave) = (hp_("p").toInt, hp_("spec").toInt, hp_("lwave").toDouble)

        val y_pow   = powForm.f(y)
        val x_endo = MatrixD (y, y_pow).ᵀ

        // add trend terms and terms for the endogenous variable
        makeMatrix4T (y, spec, lwave, bakcast) ++^                      // trend terms
        makeMatrix4L (x_endo, p, bakcast)                               // lagged linear terms
    end buildMatrix

end ARY_Quad

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_QuadTest` main function tests the `ARY_Quad` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRY_QuadTest
 */
@main def aRY_QuadTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon
    hp("p")    = 3                                                      // endo lags
    hp("spec") = 2                                                      // trend specification: 0, 1, 2, 3, 5

    val mod = ARY_Quad (y, hh)                                          // create model for time series data
    mod.inSample_Test ()                                                // In-Sample Testing
    println (mod.summary ())                                            // statistical summary

end aRY_QuadTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_QuadTest2` main function tests the `ARY_Quad` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRY_QuadTest2
 */
@main def aRY_QuadTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon
    hp("p")    = 3                                                      // endo lags
    hp("spec") = 2                                                      // trend specification: 0, 1, 2, 3, 5

    val mod = ARY_Quad (y, hh)                                          // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.rollValidate ()                                                 // TnT with Rolling Validation
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))     // only diagnose on the testing set
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRY_QuadTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_QuadTest3` main function tests the `ARY_Quad` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRY_QuadTest3
 */
@main def aRY_QuadTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks) 
    Transform.hp("p") = 1.5                                             // use 1.5 for the power/exponent (default is 2)

    for p <- 1 to 5; s <- 1 to 2 do                                     // number of lags; trend
        hp("p")    = p                                                  // endo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5

        val mod = ARY_Quad (y, hh)                                      // create model for time series data
        mod.inSample_Test ()                                            // In-Sample Testing
        println (mod.summary ())                                        // statistical summary of fit
    end for

end aRY_QuadTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_QuadTest4` main function tests the `ARY_Quad` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRY_QuadTest4
 */
@main def aRY_QuadTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks) 

    for p <- 1 to 10; s <- 1 to 5 do                                    // number of lags; trend
        hp("p")     = p                                                 // endo lags
        hp("spec")  = s                                                 // trend specification: 0, 1, 2, 3, 5
        val mod = ARY_Quad (y, hh)                                      // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest_x ()()                                           // use customized trainNtest_x

        mod.setSkip (0)
        mod.rollValidate ()                                             // TnT with Rolling Validation
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))   // only diagnose on the testing set
//      println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRY_QuadTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRY_QuadTest5` main function tests the `ARY_Quad` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs feature selection.
 *  > runMain scalation.modeling.forecasting.aRY_QuadTest5
 */
@main def aRY_QuadTest5 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("spec")  = 5                                                     // trend specification: 0, 1, 2, 3, 5
    hp("lwave") = 20                                                    // wavelength (distance between peaks)

    val mod = ARY_Quad (y, hh)                                          // create model for time series data
    mod.inSample_Test ()                                                // In-Sample Testing
    println (mod.summary ())                                            // statistical summary of fit

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                              // R^2, R^2 bar, sMAPE, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                            // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    println (s"k = $k")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end aRY_QuadTest5

