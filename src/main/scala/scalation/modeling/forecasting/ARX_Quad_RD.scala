
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Mar 31 23:28:32 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Quadratic, Auto-Regressive on lagged y and xe (ARX_Quad_RD) using OLS - Direct Forecasting
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_Quad_RD` class provides basic time series analysis capabilities for
 *  ARX_Quad_RD models.  ARX_Quad_RD models are often used for forecasting.
 *  `ARX_Quad_RD` uses DIRECT (as opposed to RECURSIVE) multi-horizon forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y) @see `ARX_Quad_RD.apply`
 *  @param y        the response/output matrix (column per horizon) (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARX_Quad_RD (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String] = null,
                  tRng: Range = null, hparam: HyperParameter = AR.hp,
                  bakcast: Boolean = false,  
                  tForms: TransformMap = Map ("tForm_y" -> null))
      extends ARX_D (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms):

    private val debug = debugf ("ARX_Quad_RD", false)                       // debug function

    _modelName = s"ARX_Quad_RD_${p}_${q}_$n_exo"

    debug ("init", s"$modelName with $n_exo exogenous variables and additional trend spec = $spec")
//  debug ("init", s"[ x | y ] = ${x ++^ y}")

end ARX_Quad_RD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_Quad_RD` companion object provides factory methods for the `ARX_Quad_RD` class.
 */
object ARX_Quad_RD extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_Quad_RD` object by building an input matrix x and then calling the constructor.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters (defaults for `AR.hp`)
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = AR.hp,
               fEndo: LSET [Transform] = null,
               fExo: Array [LSET [Transform]] = null,
               bakcast: Boolean = false): ARX_Quad_RD =

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val powForm = PowForm (VectorD (0, Transform.hp("p").toDouble))    // Yousef_added
        val tForms  = Map ("tForm_y" -> null, "powForm" -> powForm)
        val xy    = ARX_Quad.buildMatrix (xe_bfil, y, hparam, bakcast, powForm)
        val fname = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        val yy    = makeMatrix4Y (y, hh, bakcast)

        banner ("ARX_Quad_RD.apply: Step 1: Run Recursive ARX to get its Forecast Matrix")
//      val arx   = new ARX_Quad (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)  // recusive forecasting

        val xy_a  = ARX.buildMatrix (xe_bfil, y, hparam, bakcast)          // the plain ARX matrix
        val arx   = new ARX (xy_a, y, hh, xe.dim2, ARX.formNames (xe.dim2, hparam), tRng, hparam, bakcast)
        arx.trainNtest ()()                                                 // initially train and test on full dataset
        val yf_r  = arx.backcast (arx.forecastAll ())                       // forecast h-steps ahead (h = 1 to hh) for all y
        println ("ARX_Quad_RD.apply: Forecast Matrix yf_r.dims = ${yf_r.dims}, yf_r = $yf_r")
        val xy_r  = (xy ++^ yf_r(?, 1 until yf_r.dim2-1))                   // append recursive forecast matrix

        new ARX_Quad_RD (xy_r, yy, hh, xe.dim2, fname ++ ARX_RD.yfNames (hh), tRng, hparam, bakcast, tForms)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_Quad_RD` object by building an input matrix xy and then calling the
     * `ARX_Quad_RD` constructor.  Also rescale the input data.
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
                 tFormT: TransformT = MinMax): ARX_Quad_RD =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                       // use (mean, std) of training set for both In-sample and TnT
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
        val xy      = ARX_Quad.buildMatrix (xe_bfil, y_scl, hparam, bakcast, powForm)
        val yy      = makeMatrix4Y (y_scl, hh, bakcast)
        val fname   = if fname_ == null then formNames (xe.dim2, hparam) else fname_

        banner ("ARX_Quad_RD.rescale: Step 1: Run Recursive ARX to get its Forecast Matrix")
//      val arx   = new ARX_Quad (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)  // recusive forecasting

        val xy_a  = ARX.buildMatrix (xe_bfil, y_scl, hparam, bakcast)      // the plain ARX matrix
        val arx   = new ARX (xy_a, y_scl, hh, xe.dim2, ARX.formNames (xe.dim2, hparam), tRng, hparam, bakcast, tForms)
        arx.trainNtest ()()                                                 // initially train and test on full dataset
        val yf_r  = arx.backcast (arx.forecastAll ())                       // forecast h-steps ahead (h = 1 to hh) for all y
        println ("ARX_Quad_RD.rescale: Forecast Matrix yf_r.dims = ${yf_r.dims}, yf_r = $yf_r")
        val xy_r  = (xy ++^ yf_r(?, 1 until yf_r.dim2-1))                   // append recursive forecast matrix

        new ARX_Quad_RD (xy_r, yy, hh, xe.dim2, fname ++ ARX_RD.yfNames (hh), tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables (none for `ARX_Quad_RD`)
     *  @param n_fExArr  the number of functions used to map exogenous variables (none for `ARX_Quad_RD`)
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int = 0, n_fExArr: Array [Int] = null): Array [String] =
        ARX_Quad.formNames (n_exo, hp_, n_fEn, n_fExArr)
    end formNames

end ARX_Quad_RD

import Example_LakeLevels.y
import AR.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_RDTest` main function tests the `ARX_Quad_RD` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_Quad_RDTest
 */
@main def aRX_Quad_RDTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_Quad_RD (EMPTY_EXO, y, hh)                            // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

end aRX_Quad_RDTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_RDTest2` main function tests the `ARX_Quad_RD` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_Quad_RDTest2
 */
@main def aRX_Quad_RDTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_Quad_RD (EMPTY_EXO, y, hh)                            // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validation

end aRX_Quad_RDTest2

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_RDTest3` main function tests the `ARX_Quad_RD` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_Quad_RDTest3
 */
@main def aRX_Quad_RDTest3 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    Transform.hp("p") = 1.8                                             // power on Pow transform
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
//      val mod = ARX_Quad_RD (xe, y, hh)                               // create model for time series data
        val mod = ARX_Quad_RD.rescale (xe, y, hh, tFormT = Log1p) 
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit  FIX - crashes
    end for

end aRX_Quad_RDTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_RDTest4` main function tests the `ARX_Quad_RD` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_Quad_RDTest4
 */
@main def aRX_Quad_RDTest4 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    Transform.hp("p") = 1.8                                             // power on Pow transform
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do 
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX_Quad_RD (xe, y, hh)                               // create model for time series data
//      val mod = ARX_Quad_RD.rescale (xe, y, hh, tFormT = Log1p) 
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRX_Quad_RDTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_RDTest5` main function tests the `ARX_Quad_RD` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection.
 *  > runMain scalation.modeling.forecasting.aRX_Quad_RDTest5
 */
@main def aRX_Quad_RDTest5 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("px")    = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3
    Transform.hp("p") = 1.5                                             // power on Pow transform
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS = true                                  // use full dataset for FS

    val mod = ARX_Quad_RD (xe, y, hh)                                   // create model for time series data
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
        val (colsArr, rSqArr) = mod.selectFeatures (tech, "none", h = 1)   // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (colsArr, rSqArr, tnt = false)              // In-ST of the selected multi-horizon model

//      val imp = mod.importance (cols.toArray, rSq)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRX_Quad_RDTest5

