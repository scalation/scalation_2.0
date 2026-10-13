
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe (ARX_RD) using OLS - Direct Forecasting
 *
 *  @see `scalation.modeling.forecasting.ARX` for recursive forecasting version
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_RD` class provides basic time series analysis capabilities for
 *  ARX_RD models.  ARX_RD models are often used for forecasting.
 *  `ARX_RD` uses DIRECT (as opposed to RECURSIVE) multi-horizon forecasting.
 *  @note: `ARX_RD` is dependent on [[ARX]] class for feature selection.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y) @see `ARX_RD.apply`
 *  @param y        the response/output matrix (column per horizon) (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARX_RD (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String],
             tRng: Range = null, hparam: HyperParameter = AR.hp,
             bakcast: Boolean = false,
             tForms: TransformMap = Map ("tForm_y" -> null))
      extends Forecaster_D (x, y, hh, fname, tRng, hparam, bakcast):      // provides `reg` field

    private   val debug = debugf ("ARX_RD", false)                        // debug function
    protected val p     = hparam("p").toInt                               // use the last p endogenous values (p lags)
    protected val q     = hparam("px").toInt                              // use the last q exogenous values (q lags)
    protected val spec  = hparam("spec").toInt                            // trend terms: 0 - implicit constant, 1 - linear, 2 - quadratic, 3 - cubic
    protected val nneg  = hparam("nneg").toInt == 1                       // 0 => unrestricted, 1 => predictions must be non-negative

    _modelName = s"ARX_RD_${p}_${q}_$n_exo"
    yForm      = tForms("tForm_y").asInstanceOf [Transform]

    debug ("init", s"$modelName with $n_exo exogenous variables and additional trend spec = $spec")
    debug ("init", s"[ x | y ] = ${x ++^ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = b_0 + b_1 y_t-1 + b_2 y_t-2 + ... + b_p y_t-p = b dot x_t
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions (ignored)
     */
    def predict (t: Int, y_ : MatrixD): VectorD =
//      println (s"ARX_RD.predict: for t = $t reg.predict (${x(t)}) = ${reg.predict (x(t))}")   // original x
//      println (s"ARX_RD.predict: x(t) = ${x(t)} vs. reg.getX(t) = ${reg.getX(t)}")            // centered x
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
        println (s"ARX_RD.forecast: yf($t) = ${yf(t)}")
        pred                                                              // yh is pred
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build an `ARX_RD` model using the cols with the selected features.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel (cols: LSET [Int] = mcols): ARX_RD =
        new ARX_RD (x(?, cols), y, hh, n_exo, cols.toArray.map (fname(_)), tRng, hparam, bakcast, tForms)
    end getModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a single-horizon `ARX` model using the cols with the selected features.
     *  Note: uses `ARX` as it is the base model for ARX*_D.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel_h (cols: LSET [Int] = mcols, h: Int = 1): ARX =
        new ARX (x(?, cols), y(?, h-1), 1, n_exo, cols.toArray.map (fname(_)), tRng, hparam, bakcast, tForms)
    end getModel_h

end ARX_RD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_RD` companion object provides factory methods for the `ARX_RD` class.
 */
object ARX_RD extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the names of the hh appended recursive-forecast columns: yf_h1, ..., yf_hhh
     *  (column yf_hh holds the recursive model's h-steps ahead forecast of y_t).
     *  @param hh  the maximum forecasting horizon
     */
    def yfNames (hh: Int): Array [String] = (1 to hh).toArray.map (h => s"yf_h$h")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_RD` object by building an input matrix xy and then calling the constructor.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters (defaults for `MakeMatrix4TS.hp`)
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous 
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = AR.hp,
               fEndo: LSET [Transform] = null,
               fExo: Array [LSET [Transform]] = Array (),
               bakcast: Boolean = false): ARX_RD =

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
            xe_bfill = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val xy    = ARX.buildMatrix (xe_bfill, y, hparam, bakcast)
        val fname = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        val yy    = makeMatrix4Y (y, hh, bakcast)

        banner ("ARX_RD.apply: Step 1: Run Recursive ARX to get its Forecast Matrix")
        val arx   = new ARX (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast)  // recusive forecasting
        arx.trainNtest ()()                                                 // initially train and test on full dataset
        val yf_r  = arx.backcast (arx.forecastAll ())                       // forecast h-steps ahead (h = 1 to hh) for all y
        println ("ARX_RD.apply: Forecast Matrix yf_r.dims = ${yf_r.dims}, yf_r = $yf_r")
        val xy_r  = (xy ++^ yf_r(?, 1 until yf_r.dim2-1))                   // append recursive forecast matrix

        new ARX_RD (xy_r, yy, hh, xe.dim2, fname ++ yfNames (hh), tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_RD` object by building an input matrix xy and then calling the
     *  `ARX_RD` constructor, with rescaling of endogneous and exogenous variable values.
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
                 tFormT: TransformT = MinMax): ARX_RD =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                       // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
            xe_bfill = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))    // backfill each exogenous variable
            if tFormScale != null then
                val tForm_exo = tFormScale (xe_bfill(0 until tr_size))
                xe_bfill      = tForm_exo.f (xe_bfill)                      // rescale the backfilled exogenous variable

        val tForms = Map ("tForm_y" -> tForm_y)
        val xy     = ARX.buildMatrix (xe_bfill, y_scl, hparam, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        val yy     = makeMatrix4Y (y_scl, hh, bakcast)

        banner ("ARX_RD.rescale: Step 1: Run Recursive ARX to get its Forecast Matrix")
        val arx   = new ARX (xy, y_scl, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)  // recusive forecasting
        arx.trainNtest ()()                                                 // initially train and test on full dataset
        val yf_r  = arx.backcast (arx.forecastAll ())                       // forecast h-steps ahead (h = 1 to hh) for all y
        println ("ARX_RD.rescale: Forecast Matrix yf_r.dims = ${yf_r.dims}, yf_r = $yf_r")
        val xy_r  = (xy ++^ yf_r(?, 1 until yf_r.dim2-1))                   // append recursive forecast matrix

        new ARX_RD (xy_r, yy, hh, xe.dim2, fname ++ yfNames (hh), tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables ((none for `ARX_RD`)
     *  @param n_fExArr  the number of functions used to map exogenous variables ((none for `ARX_RD`)
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int = 0, n_fExArr: Array [Int] = null): Array [String] =
        ARX.formNames (n_exo, hp_, n_fEn, n_fExArr)
    end formNames

end ARX_RD

import Example_LakeLevels.y
import AR.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_RDTest` main function tests the `ARX_RD` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_RDTest
 */
@main def aRX_RDTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_RD (EMPTY_EXO, y, hh)                                 // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit  FIX - only takes last horizon

end aRX_RDTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_RDTest2` main function tests the `ARX_RD` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_RDTest2
 */
@main def aRX_RDTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_RD (EMPTY_EXO, y, hh)                                 // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validation

end aRX_RDTest2

import Example_Covid.{BEST_EXO, BEST2_EXO, clip, loadData}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_RDTest3` main function tests the `ARX_RD` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_RDTest3
 */
@main def aRX_RDTest3 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))                // for testing against `ARY, should be the same for h = 1
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    if DO_PLOT then
        new Plot (null, y, null, s"y (new_deaths) vs. t", lines = true)
        if xe.dim2 > 0 then new Plot (null, xe(?, 0), null, s"x_0 (${BEST_EXO(0)}) vs. t", lines = true)

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
//      val mod = ARX_RD (xe, y, hh)                                    // create model for time series data
        val mod = ARX_RD.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit  FIX - crashes
    end for

end aRX_RDTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_RDTest4` main function tests the `ARX_RD` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_RDTest4
 */
@main def aRX_RDTest4 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))                // for testing against `ARY, should be the same for h = 1
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    RidgeRegression.hp("lambda") = 1.0

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX_RD (xe, y, hh)                                    // create model for time series data
//      val mod = ARX_RD.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRX_RDTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_RDTest5` main function tests the `ARX_RD` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection.
 *  > runMain scalation.modeling.forecasting.aRX_RDTest5
 */
@main def aRX_RDTest5 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("px")    = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS = true                                  // use full dataset for FS

    val mod = ARX_RD (xe, y, hh)                                        // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

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

end aRX_RDTest5

