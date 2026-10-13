
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

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}

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
 *  @param hparam   the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARX_Quad (x: MatrixD, y: VectorD, hh: Int, n_exo: Int, fname: Array [String],
                tRng: Range = null, hparam: HyperParameter = AR.hp,
                bakcast: Boolean = false,
                tForms: TransformMap = Map ("tForm_y" -> null))
      extends ARX (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms):

    private val debug = debugf ("ARX_Quad", false)                       // debug function

    _modelName = s"ARX_Quad_${p}_${q}_$n_exo"

    debug ("init", s"$modelName with $n_exo exogenous variables and additional trend spec = $spec")
//  debug ("init", s"[ x | y ] = ${x :^+ y}")
    debug ("init", s"tForms= $tForms")

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
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = AR.hp,
               fEndo: LSET [Transform] = null,
               fExo: Array [LSET [Transform]] = Array (),
               bakcast: Boolean = false): ARX_Quad =

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))   // backfill each exogenous variable

        val powForm = PowForm (VectorD (0, Transform.hp("p").toDouble))
        val tForms  = Map ("tForm_y" -> null, "powForm" -> powForm)
        val xy      = buildMatrix (xe_bfil, y, hparam, bakcast, powForm)  // Yousef_added
        val fname   = formNames (xe.dim2, hparam)
        new ARX_Quad (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_Quad` object by building an input matrix xy and then calling the
     *  `ARX_Quad` constructor.  Also rescale the input data.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormT   the transform for rescaling endogenous and exogenous
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = AR.hp,
//               fEndo: LSET [Transform] = null,
//               fExo: Array [LSET [Transform]] = Array (),
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARX_Quad =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                      // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
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

        val (p, q, spec) = (hp_("p").toInt, hp_("px").toInt, hp_("spec").toInt)
        val y_pp   = powForm.f(y)                                        // apply power transformation
        val x_endo = MatrixD (y, y_pp).ᵀ

        // add trend terms and terms for the endogenous variable
        var xy = makeMatrix4T (y, spec, bakcast) ++^                     // trend terms
                 makeMatrix4L (x_endo, p, bakcast)                       // lagged linear terms

        if xe_bfil != null and q > 0 then                                // rescale the exogenous variables
            xy = xy ++^ makeMatrix4L (xe_bfil, q, bakcast)

        println (s"ARX_Quad.buildMatrix: xy.dims = ${xy.dims}")
//      println (s"xy = $xy")
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

        val (p, q, spec) = (hp_("p").toInt, hp_("px").toInt, hp_("spec").toInt)
        val names = VEC [String] ()
        for j <- 0 until n_exo; k <- q to 1 by -1 do names += s"xe${j}l$k"
        MakeMatrix4TS.formNames (spec, p, Transform.hp("p").toDouble) ++ names.toArray
    end formNames

end ARX_Quad

import Example_LakeLevels.y
import AR.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest` main function tests the `ARX_Quad` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest
 */
@main def aRX_QuadTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_Quad (EMPTY_EXO, y, hh)                               // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

end aRX_QuadTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest2` main function tests the `ARX_Quad` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest2
 */
@main def aRX_QuadTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_Quad (EMPTY_EXO, y, hh)                               // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validati

end aRX_QuadTest2

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest3` main function tests the `ARX_Quad` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest3
 */
@main def aRX_QuadTest3 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO)) 
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    Transform.hp("p") = 1.8   // GOLDEN_R                               // use 1.8 for the power/exponent (default is 2)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX_Quad (xe, y, hh)                                  // create model for time series data
//      val mod = ARX_Quad.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
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

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO)) 
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    Transform.hp("p") = 1.8                                             // use 1.8 for the power/exponent (default is 2)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX_Quad (xe, y, hh)                                  // create model for time series data
//      val mod = ARX_Quad.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRX_QuadTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest5` main function tests the `ARX_Quad` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection.
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest5
 */
@main def aRX_QuadTest5 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("px")    = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3
    Transform.hp("p") = 1.5                                             // use 1.5 for the power/exponent (default is 2)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS = true                                  // use full dataset for FS

    val mod = ARX_Quad (xe, y, hh)                                      // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (mod.getY, mod.getYf)                               // QoF for each horizon
//  println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

//  for tech <- values do                                               // try all feature selection techniques
//  for tech <- values if tech == Forward do                            // try a particular one
//  for tech <- values if tech == Backward do                           // try a particular one
//  for tech <- values if tech == Stepwise do                           // try a particular one
    for tech <- values if tech == Beam do                               // try a particular one
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (cols, rSq, tnt = false)

//      val imp = mod.importance (cols.toArray, rSq)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRX_QuadTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest6` main function tests the `ARX_Quad` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection  (training-set, training-set).
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest6
 */
@main def aRX_QuadTest6 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("px")    = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3
    Transform.hp("p") = 1.5                                             // use 1.5 for the power/exponent (default is 2)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    val mod = ARX_Quad (xe, y, hh)                                      // create model for time series data
    banner (s"Training-only Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                 // initially train and test on full dataset
    println (mod.summary ())                                            // statistical summary of fit

    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (mod.getY, mod.getYf)                               // QoF for each horizon
//  println (s"Training-only Forecast Matrix yf = ${mod.getYf}")

/*
    mod.setSkip ()                                                      // don't skip any rows
    mod.rollValidate ()                                                 // TnT with Rolling Validation
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng(y.dim))
*/

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

//  for tech <- values do                                               // try all feature selection techniques
//  for tech <- values if tech == Forward do                            // try a particular one
//  for tech <- values if tech == Backward do                           // try a particular one
//  for tech <- values if tech == Stepwise do                           // try a particular one
    for tech <- values if tech == Beam do                               // try a particular one
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (cols, rSq, inSample = false)

//      val imp = mod.importance (cols.toArray, rSq)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRX_QuadTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_QuadTest7` main function tests the `ARX_Quad` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection  (training-set, testing-set).
 *  > runMain scalation.modeling.forecasting.aRX_QuadTest7
 */
@main def aRX_QuadTest7 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 1  // 6                                               // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("px")    = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3
    Transform.hp("p") = 1.5                                             // use 1.5 for the power/exponent (default is 2)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    val mod = ARX_Quad (xe, y, hh)                                      // create model for time series data
    banner (s"Training-Test Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                 // initially train and test on full dataset
    println (mod.summary ())                                            // statistical summary of fit

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

//  for tech <- values do                                               // try all feature selection techniques
//  for tech <- values if tech == Forward do                            // try a particular one
//  for tech <- values if tech == Backward do                           // try a particular one
    for tech <- values if tech == Stepwise do                           // try a particular one
//  for tech <- values if tech == Beam do                               // try a particular one
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (cols, rSq, inSample = false)               // TnT of the selected model

//      val imp = mod.importance (cols.toArray, rSq.ᵀ)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRX_QuadTest7

