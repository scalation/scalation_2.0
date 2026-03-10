
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe with quadratic terms (ARX_Quad) using OLS
 *
 *  @see `scalation.modeling.Regression`
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer, LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_Quad` class provides basic time series analysis capabilities for ARX quadratic models.
 *  ARX quadratic models utilize quadratic multiple linear regression based on lagged values of y.
 *  ARX models build on `ARY` by including one or more exogenous (xe) variables.
 *  Given time series data stored in vector y, its next value y_t = combination of
 *  last p values of y, y^2 and the last q values of each exogenous variable xe_j.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y, y^2 and xe) @see `ARX_Quad.apply`
 *  @param y        the response/output vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARX_Quad (x: MatrixD, y: VectorD, hh: Int, n_exo: Int, fname: Array [String],
                tRng: Range = null, hparam: HyperParameter = hp,
                bakcast: Boolean = false,
                tForms: TransformMap = Map ("tForm_y" -> null))
      extends ARX (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms):

    private val debug = debugf ("ARX_Quad", true)                       // debug function

    _modelName = s"ARX_Quad_${p}_${q}_$n_exo"

    debug ("init", s"$modelName with with $n_exo exogenous variables and additional term spec = $spec")
//  debug ("init", s"[ x | y ] = ${x :^+ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a new vector from the first spec values of x, the last p-h+1 values
     *  of x (past values) and recent values 1 to h-1 from the forecasts.
     *  @param xx  the t-th row of the input matrix (lagged actual values)
     *  @param yy  the t-th row of the forecast matrix (forecasted future values)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    override def forge (xx: VectorD, yy: VectorD, h: Int): VectorD =
        // add terms for the endogenous variable
        val n_endo  = spec + p                                           // number of trend + endogenous values
        val x_act   = xx(n_endo - (p+1-h) until n_endo)                  // get actual lagged y-values (endogenous)
        val nyy     = p - x_act.dim                                      // number of forecasted values needed
        val x_fcast = yy(h-nyy until h)                                  // get forecasted y-values

        val x_act_pow   = xx(n_endo+p - (p+1-h) until n_endo+p)          // get transformed lagged endogenous variable
//      val x_fcast_pow = scaleCorrection (x_fcast)
        val x_fcast_pow = tForms("powForm").asInstanceOf [Transform].f(x_fcast)

        var xy = x_act ++ x_fcast ++ x_act_pow ++ x_fcast_pow            // add transformed lagged forecasted y-values
        for j <- 0 until n_exo do                                        // for the j-th exogenous variable
            xy = xy ++ hide (xx(n_endo+p + j*q until n_endo+p + (j+1)*q), h)
        xx(0 until spec) ++ xy
    end forge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply scale correction to x_fcast.
     *  @param x_fcast  the vector to apply the scale correction to
     *
    def scaleCorrection (x_fcast: VectorD): VectorD =
        if tForms("tForm_y") != null then
            val f_pp = (tForms("tForm_endo").asInstanceOf [Transform].f(_: VectorD)) ⚬
                       (tForms("powForm").asInstanceOf [Transform].f(_: VectorD)) ⚬
                       (tForms("tForm_y").asInstanceOf [Transform].fi(_: VectorD))
            f_pp (x_fcast)
        else
            tForms("powForm").asInstanceOf [Transform].f(x_fcast)
    end scaleCorrection
     */

end ARX_Quad


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_Quad` companion object provides factory methods for the `ARX_Quad` class.
 */
object ARX_Quad extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_Quad` object by building an input matrix xy and then calling the
     *  `ARX_Quad` constructor.
     *  @param xe          the matrix of exogenous variable values
     *  @param y           the endogenous/response vector (main time series data)
     *  @param hh          the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_      the feature/variable names
     *  @param tRng        the time range, if relevant (time index may suffice)
     *  @param hparam      the hyper-parameters
     *  @param fEndo_enab  the set of transforms to be used for the endogenous
     *  @param fExo_enab   the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast     whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = hp,
               fEndo_enab: LSET [TransformT] = null,
               fExo_enab: Array [LSET [TransformT]] = null,
               bakcast: Boolean = false): ARX_Quad =

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))   // backfill each exogenous variable

        val xy = buildMatrix (xe_bfil, y, hparam, bakcast)
        val fname = formNames (xe.dim2, hparam)
        new ARX_Quad (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_Quad` object by building an input matrix xy and then calling the
     *  `ARX_Quad` constructor.  Also rescale the input data.
     *  @param xe          the matrix of exogenous variable values
     *  @param y           the endogenous/response vector (main time series data)
     *  @param hh          the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_      the feature/variable names
     *  @param tRng        the time range, if relevant (time index may suffice)
     *  @param hparam      the hyper-parameters
     *  @param fEndo_enab  the set of transforms to be used for the endogenous
     *  @param fExo_enab   the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast     whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormT      the transform for rescaling endogenous and exogenous
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = hp,
                 fEndo_enab: LSET [TransformT] = null,
                 fExo_enab: Array [LSET [TransformT]] = null,
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARX_Quad =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                      // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))    // backfill each exogenous variable
            if tFormScale != null then
                val tForm_exo = tFormScale (xe_bfil(0 until tr_size))
                xe_bfil       = tForm_exo.f (xe_bfil)

        val powForm = PowForm (VectorD (0, Transform.hp("p").toDouble))
        val tForms  = Map ("tForm_y" -> tForm_y, "powForm" -> powForm)
        val xy      = buildMatrix (xe_bfil, y_scl, hparam, bakcast, powForm)
        val fname   = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        new ARX_Quad (xy, y_scl, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the p + spec columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  @param xe_bfil  the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hp_      the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param powForm  the power transform
     */
    def buildMatrix (xe_bfil: MatrixD, y: VectorD, hp_ : HyperParameter, bakcast: Boolean,
                     powForm: Transform = PowForm (VectorD (0, Transform.hp("p").toDouble))): MatrixD =

        val (p, q, spec, lwave) = (hp_("p").toInt, hp_("q").toInt, hp_("spec").toInt, hp_("lwave").toDouble)
        val y_pp   = powForm.f(y)                                        // apply power transformation
        val x_endo = MatrixD (y, y_pp).ᵀ

        // add trend terms and terms for the endogenous variable
        var xy = makeMatrix4T (y, spec, lwave, bakcast) ++^              // trend terms
                 makeMatrix4L (x_endo, p, bakcast)                       // lagged linear terms

        if xe_bfil != null and q > 0 then                                // rescale the exogenous variables
            xy = xy ++^ makeMatrix4L (xe_bfil, q, bakcast)
        xy
    end buildMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables (none for `ARX_Quad`)
     *  @param n_fExArr  the number of functions used to map exogenous variables (none for `ARX_Quad`)
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int = 0, n_fExArr: Array [Int] = null): Array [String] =

        val (p, q, spec) = (hp_("p").toInt, hp_("q").toInt, hp_("spec").toInt)
        val names = ArrayBuffer [String] ()
        for j <- 0 until n_exo; k <- q to 1 by -1 do names += s"xe${j}l$k"
        MakeMatrix4TS.formNames (spec, p, Transform.hp("p").toDouble) ++ names.toArray
    end formNames

end ARX_Quad

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest` main function tests the `ARX_Quad` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest
 *
@main def aRX_QuadTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_Quad (y, hh)                                          // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end aRX_QuadTest
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest2` main function tests the `ARX_Quad` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest2
 *
@main def aRX_QuadTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_Quad (y, hh)                                          // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.rollValidate ()                                                 // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRX_QuadTest2
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest3` main function tests the `ARX_Quad` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest3
 */
@main def aRX_QuadTest3 (): Unit =

//  val exo_vars  = NO_EXO
    val exo_vars  = Array ("icu_patients")
//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    Transform.hp("p") = 1.5                                             // use 1.5, 1.9 for the power/exponent (default is 2)
    hp("lwave") = 20                                                    // wavelength (distance between peaks)

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of endo lags; exo lags; trend
        hp("p")    = p                                                  // endo lags
        hp("q")    = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5
        val mod = ARX_Quad (xe, y, hh)                                  // create model for time series data
//      val mod = ARX_Quad.rescale (xe, y, hh)                          // create model for time series data
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit
    end for

end aRX_QuadTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest4` main function tests the `ARX_Quad` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest4
 */
@main def aRX_QuadTest4 (): Unit =

    val exo_vars  = Array ("icu_patients")
//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    Transform.hp("p") = 1.5                                             // use 1.5 for the power/exponent (default is 2)
    hp("lwave") = 20                                                    // wavelength (distance between peaks)

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // endo lags
        hp("q")    = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5
        val mod = ARX_Quad (xe, y, hh)                                  // create model for time series data
//      val mod = ARX_Quad.rescale (xe, y, hh)                          // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest_x ()()                                           // use customized trainNtest_x

        mod.setSkip (0)
        mod.rollValidate ()                                             // TnT with Rolling Validation
        println (s"After Roll TnT Forecast Matrix yf = ${mod.getYf}")
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))   // only diagnose on the testing set
//      println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRX_QuadTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest5` main function tests the `ARX_Quad` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs feature selection.
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest5
 */
@main def aRX_QuadTest5 (): Unit =

    val exo_vars  = Array ("icu_patients")
//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    val p  = 10
    val q  = 10
    hp("p")     = p                                                     // endo lags
    hp("q")     = q                                                     // exo lags
    hp("spec")  = 5                                                     // trend specification: 0, 1, 2, 3, 5
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
    Transform.hp("p") = 1.5                                             // use 1.5 for the power/exponent (default is 2)

    val mod = ARX_Quad (xe, y, hh)                                      // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset
    println (mod.summary ())                                            // statistical summary of fit

//  mod.setSkip (p)                                                     // full AR-formula available when t >= p
    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (mod.getY, mod.getYf)                               // QoF for each horizon
//  println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                              // R^2, R^2 bar, sMAPE, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                            // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    println (s"k = $k")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end aRX_QuadTest5

