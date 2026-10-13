
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe (ARX_D) using OLS - Direct Forecasting
 *
 *  @see `scalation.modeling.forecasting.ARX` for recursive forecasting version
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_D` class provides basic time series analysis capabilities for
 *  ARX_D models.  ARX_D models are often used for forecasting.
 *  `ARX_D` uses DIRECT (as opposed to RECURSIVE) multi-horizon forecasting.
 *  @note: `ARX_D` is dependent on [[ARX]] class for feature selection.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x             the data/input matrix (lagged columns of y) @see `ARX_D.apply`
 *  @param y             the response/output matrix (column per horizon) (time series data) 
 *  @param hh            the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo         the number of exogenous variables
 *  @param fname         the feature/variable names
 *  @param tRng          the time range, if relevant (time index may suffice)
 *  @param hparam        the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast       whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms        the map of transformations applied
 *  @param scaled        whether the regression standardizes the columns of x (training mean and
 *                           standard deviation) instead of only centering them (defaults to false)
 */
class ARX_D (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String],
             tRng: Range = null, hparam: HyperParameter = AR.hp,
             bakcast: Boolean = false,
             tForms: TransformMap = Map ("tForm_y" -> null),
             scaled: Boolean = false)
      extends Forecaster_D (x, y, hh, fname, tRng, hparam, bakcast, scaled = scaled):   // provides `reg` field

    private   val debug = debugf ("ARX_D", false)                         // debug function
    protected val p     = hparam("p").toInt                               // use the last p endogenous values (p lags)
    protected val q     = hparam("px").toInt                              // use the last q exogenous values (q lags)
    protected val spec  = hparam("spec").toInt                            // trend terms: 0 - implicit constant, 1 - linear, 2 - quadratic, 3 - cubic
    protected val nneg  = hparam("nneg").toInt == 1 &&                    // 0 => unrestricted, 1 => predictions must be non-negative
                          ! ARX_D.centered (tForms        )               // (never in a centered/standardized space, @see `centered`)

    _modelName = s"ARX_D_${p}_${q}_$n_exo"
    yForm      = tForms("tForm_y").asInstanceOf [Transform]

    debug ("init", s"$modelName with $n_exo exogenous variables and additional trend spec = $spec")
    debug ("init", s"x.dims = ${x.dims}, y.dims = ${y.dims}")
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
//      println (s"ARX_D.predict: for t = $t reg.predict (${x(t)}) = ${reg.predict (x(t))}")   // original x
//      println (s"ARX_D.predict: x(t) = ${x(t)} vs. reg.getX(t) = ${reg.getX(t)}")            // centered x
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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build an `ARX_D` model using the cols with the selected features (same
     *  transforms and scaling as this model).
     *  @param cols  the cols of the input matrix with selected features
     */
    def getModel (cols: LSET [Int] = mcols): ARX_D =
        new ARX_D (x(?, cols), y, hh, n_exo, cols.toArray.map (fname(_)), tRng, hparam, bakcast, tForms, scaled)
    end getModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a single-horizon `ARX` model using the cols with the selected features.
     *  @note: uses `ARX` as it is the base model for ARX*_D.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel_h (cols: LSET [Int] = mcols, h: Int = 1): ARX =
        new ARX (x(?, cols), y(?, h-1), 1, n_exo, cols.toArray.map (fname(_)), tRng, hparam, bakcast, tForms, scaled)
    end getModel_h

end ARX_D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_D` companion object provides factory methods for the `ARX_D` class.
 */
object ARX_D extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_D` object by building an input matrix xy and then calling the constructor.
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
               fExo: Array [LSET [Transform]] = Array (),
               bakcast: Boolean = false): ARX_D =

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
            xe_bfill = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val xy    = ARX.buildMatrix (xe_bfill, y, hparam, bakcast)
        val fname = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        val yy    = makeMatrix4Y (y, hh, bakcast)
        new ARX_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_D` object by building an input matrix xy and then calling the
     *  `ARX_D` constructor, with rescaling of endogneous and exogenous variable values.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous 
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tForm    the transform for rescaling endogenous and exogenous
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = AR.hp,
//               fEndo: LSET [Transform] = null,
//               fExo: Array [LSET [Transform]] = Array (),
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARX_D =

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
        new ARX_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end rescale

   //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_D` object from the terms of a formula (@see `ARX_Formula`),
     *  with one column per term (lagged variable, transform, interaction or wave),
     *  so skipped lags and interactions that contiguous lags 1 to p/px cannot
     *  express are supported.  Rows are not trimmed: each term pads its first lag
     *  values as `ARX.buildMatrix` does (@see `VectorD.lagv`).  When tFormT is given,
     *  every base column is transformed (fit on its training slice, as in `rescale`)
     *  before the terms are evaluated, and the response's transform is kept for
     *  inverting forecasts.  A term whose column is not finite (e.g., log of 0, exp
     *  of a large value) is rejected here, before the regression sees it.
     *  The matrix has NO intercept column (the regression centers, @see `Forecaster_D`),
     *  so feature selection on it must use first = 0 (@see `Forecaster_D.selectFeatures`).
     *  Train on a window x(0 until n), yy(0 until n) to keep rows from n on out of
     *  sample; the regression fits its means (and scales) on that window only.
     *  @param resp     the response/endogenous variable
     *  @param preds    the predictor terms from the right hand side of the formula
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names (defaults to the terms' labels)
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters (defaults to `AR.hp`; never modified here)
     *  @param bakcast  whether a backcasted value is prepended (not supported, must be false)
     *  @param tFormT   the transform type applied to every base column (null => none)
     *  @param trSize   the number of leading rows the base-column transforms are fit on
     *                      (defaults to -1 => `Model.trSize (y.dim)`, as in `rescale`);
     *                      pass the size of the training window so its statistics never
     *                      include validation or testing rows
     *  @param scaled   whether the regression standardizes the term columns (defaults to false)
     *  @param dataset  the map of named data columns (given)
     */
    def fromTerms (resp: FormulaVar, preds: VEC [Term], hh: Int,
                   fname_ : Array [String] = null, tRng: Range = null,
                   hparam: HyperParameter = AR.hp, bakcast: Boolean = false,
                   tFormT: TransformT = null, trSize: Int = -1, scaled: Boolean = false)
                   (using dataset: Map [String, VectorD]): ARX_D =

        val td    = ARX_Formula.prepare (resp, preds, tFormT, trSize, bakcast)   // checks and base-column transforms
        val xy    = ARX_Formula.matrix (preds, td.data)                          // one column per term
        val yy    = makeMatrix4Y (td.y, hh, bakcast)
        val fname = if fname_ == null then ARX_Formula.names (preds) else fname_
        val (hp_, n_exo) = ARX_Formula.lagHP (resp, preds, hparam)
        new ARX_D (xy, yy, hh, n_exo, fname, tRng, hp_, bakcast, Map ("tForm_y" -> td.tForm_y), scaled)
    end fromTerms
/*
    def fromTerms (resp: FormulaVar, preds: VEC [Term], hh: Int,
                   fname_ : Array [String] = null, tRng: Range = null,
                   hparam: HyperParameter = AR.hp, bakcast: Boolean = false,
                   tFormT: TransformT = null, trSize: Int = -1, scaled: Boolean = false)
                   (using dataset: Map [String, VectorD]): ARX_D =

        require (! bakcast, "fromTerms: bakcast not supported")

        def fail (msg: String): Nothing = throw new IllegalArgumentException (s"fromTerms: $msg")

        if preds.isEmpty then fail ("no predictor terms (the mean model has no design matrix)")
        val y = dataset.getOrElse (resp.name, fail (s"response '${resp.name}' not in dataset"))
        for pred <- preds; nm <- pred.vars do
            if ! dataset.contains (nm) then fail (s"variable '$nm' not in dataset")
            if dataset(nm).dim != y.dim then fail (s"'$nm' length ${dataset(nm).dim} != ${y.dim}")
        for pred <- preds if pred.lag >= y.dim do fail (s"lag ${pred.lag} >= series length ${y.dim}")

        // Transform every base column on its own TRAINING slice (as in `rescale`), before
        // the terms are evaluated; keep the response's transform for forecast inversion
        // (the regression itself never rectifies in a centered/standardized space, @see `centered`)

        val tr_size = if trSize > 0 then trSize else Model.trSize (y.dim)
        if tr_size > y.dim then fail (s"trSize $tr_size > series length ${y.dim}")
        val tData   = if tFormT == null then dataset
                      else dataset.map ((nm, col) => nm -> tFormT.form (col(0 until tr_size)).f (col))
        val tForm_y = if tFormT == null then null else tFormT.form (y(0 until tr_size))

        // One column per term, full length (first lag values padded, @see `VectorD.lagv`)
        val xy = MatrixD (preds.map (_.column (tData))).ᵀ
        for j <- preds.indices if xy(?, j).exists (v => v.isNaN || v.isInfinite) do
            fail (s"term '${preds(j).label}' is not finite (e.g., log of 0 or exp of a large value)")

        val y_scl  = tData (resp.name)
        val yy     = makeMatrix4Y (y_scl, hh, bakcast)
        val tForms = Map ("tForm_y" -> tForm_y)
        val fname  = if fname_ == null then preds.map (_.label).toArray else fname_

        val pY    = preds.filter (_.vars == Set (resp.name)).map (_.lag).maxOption.getOrElse (0)
        val pX    = preds.filter (t => t.vars.nonEmpty && t.vars != Set (resp.name)).map (_.lag).maxOption.getOrElse (0)
        val n_exo = (preds.flatMap (_.vars).toSet - resp.name).size        // distinct exogenous variables in the terms
        val hp_   = hparam.updateReturn (("p", pY), ("px", pX), ("spec", 0))
        new ARX_D (xy, yy, hh, n_exo, fname, tRng, hp_, bakcast, tForms, scaled)
    end fromTerms
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables ((none for `ARX_D`)
     *  @param n_fExArr  the number of functions used to map exogenous variables ((none for `ARX_D`)
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int = 0, n_fExArr: Array [Int] = null): Array [String] =
        ARX.formNames (n_exo, hp_, n_fEn, n_fExArr)
    end formNames

end ARX_D

import Example_LakeLevels.y
import AR.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest` main function tests the `ARX_D` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_DTest
 */
@main def aRX_DTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_D (EMPTY_EXO, y, hh)                                  // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit  FIX - only takes last horizon

end aRX_DTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest2` main function tests the `ARX_D` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_DTest2
 */
@main def aRX_DTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_D (EMPTY_EXO, y, hh)                                  // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validation

end aRX_DTest2

import Example_Covid.{BEST_EXO, BEST2_EXO, clip, loadData}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest3` main function tests the `ARX_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Output (5): smape -> VectorD(13.3316, 20.4325, 27.2434, 32.7079, 38.5554, 43.2620)
 *  > runMain scalation.modeling.forecasting.aRX_DTest3
 */
@main def aRX_DTest3 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))                // for testing against `ARY, should be the same for h = 1
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    if DO_PLOT then
        new Plot (null, y, null, s"y (new_deaths) vs. t", lines = true)
        if xe.dim2 > 0 then new Plot (null, xe(?, 0), null, s"x_0 (${BEST_EXO(0)}) vs. t", lines = true)

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX_D (xe, y, hh)                                     // create model for time series data
//      val mod = ARX_D.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit  FIX - crashes
    end for

end aRX_DTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest4` main function tests the `ARX_D` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Output (5): smape -> VectorD(15.8157, 21.0612, 28.8321, 37.4038, 41.3978, 49.3985)
 *  > runMain scalation.modeling.forecasting.aRX_DTest4
 */
@main def aRX_DTest4 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))                // for testing against `ARY, should be the same for h = 1
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    RidgeRegression.hp("lambda") = 1.0

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX_D (xe, y, hh)                                     // create model for time series data
//      val mod = ARX_D.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRX_DTest4

// Note: `aRX_DTest5` to `aRX_DTest13` test the feature selection of a direct model, ONE MODE PER
//       TEST, all in the same shape: build the model, test it (the reference), select, report
//       and test the selected model.  No method is defined inside a test: the selection methods
//       return the column set(s) and trace(s), and `reportSelection` (@see `Forecaster_D`) prints
//       them and tests the selected model(s) with the steps of `inSample_Test` / `tnT_Test`, so
//       every TnT line below follows the same protocol (train, `setSkip`, `rollValidate`, diagnose
//       on the testing rows) and is comparable.
//
//           test  selection                          model            reportSelection
//           ------------------------------------------------------------------------------------
//           5     selectFeatures     (h = 1 decides) ARX_D.rescale    inSample, the multi-horizon model
//           6     selectFeaturesAtH  (h = 1)         ARX_D.rescale    tnt, the h = 1 model (the only AtH test)
//           7     selectFeatures     (mean J)        ARX_D            inSample, the multi-horizon model
//           8     selectFeatures     (h = 1 decides) ARX_D.rescale    tnt, the multi-horizon model
//           9     selectFeaturesPerH (one set per h) ARX_D            tnt, one model per horizon
//           10    h = 1, h = hh and mean J           ARX_D            tnT_Test of the full model + all three
//           11    selectFeatures     (mean J)        ARX_D.fromTerms  tnt, the multi-horizon model
//           12    none: the training-window contract of `fromTerms`
//           13    selectFeatures     (CEM, prior)    ARX_D.fromTerms  tnt, the multi-horizon model
//
//       Selection sees the rows the test allows it to see: `fullset_FS = true` for the in-sample
//       tests, false for the TnT tests (select on the training rows, test on the rest).  Tests 5
//       and 6 keep the setting (p = 10, px = 5, spec = 2, a rescaled model); tests 7 to 10
//       use p = 6, px = 6, spec = 1 (19 columns), so their tables compare with one another.
// YousefChange --------- Start
// Note: wording only -- a base file does not name the sub-package built on it
//       Nothing here imports any sub-package built on it (formula searches): they depend on this
//       package, never the reverse.
// YousefChange --------- end

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest5` main function tests the `ARX_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection AT HORIZON 1 (`selectFeaturesAtH`) on a
 *  RESCALED model (tFormT = Log1p) and tests the selected single-horizon model
 *  in-sample.  The single-horizon model at h = 1 is printed first as the reference
 *  for reading the selection traces in ORIGINAL space: a Forward trace's last row
 *  (the full model) equals that model's in-sample QoF, and its first row (intercept
 *  only) is a genuine original-space sMAPE.
 *  > runMain scalation.modeling.forecasting.aRX_DTest5
 */
@main def aRX_DTest5 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp.set (("p", 10), ("px", 5), ("spec", 2))                          // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS  = true                                 // in-sample test: select on all the rows

    val mod = ARX_D.rescale (xe, y, hh, tFormT = Log1p)                 // create model for time series data
    mod.inSample_Test ()                                                // the full model: the reference
    println (mod.summary ())                                            // statistical summary of fit

    // Note: the reference for reading a selection trace below is the single-horizon model at h = 1
    //       that drives the selection (`getModel_h`: an `ARX` on the same columns, transforms and
    //       rows), trained and diagnosed in-sample -- original space through `yForm`; the trace's
    //       last row (Forward: the full model) equals its QoF.  The ARX_D's own h = 1 fitMap row is
    //       NOT that reference: `Forecaster_D.train` cuts the rows of every horizon at the last
    //       horizon's first undefined response, so its h = 1 equation is fit on hh - 1 fewer rows
    //       than the single-horizon model (9.34 vs 9.29 sMAPE here).
    banner (s"reference for the selection trace: ${mod.modelName} at h = 1 as a single-horizon ARX")
    mod.getModel_h (h = 1).inSample_Test ()                             // in-sample QoF of the h = 1 model

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam, CEM

    // Note: one line picks the techniques: all of them, or one at a time.
//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
//  val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
    val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech (at h = 1)")
        val (colsArr, rSqArr) = mod.selectFeatures (tech, "none", h = 1)   // horizon 1 decides, its columns shared by all horizons
        mod.reportSelection (colsArr, rSqArr, tnt = false)              // print, plot; In-sample Test of the selected model
    end for

end aRX_DTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest6` main function tests the `ARX_D` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection AT HORIZON 1 (`selectFeaturesAtH`) on a
 *  RESCALED model (tFormT = Log1p) and tests the selected model with TnT.  What is
 *  tested is the single-horizon model the selection returns (an `ARX` at h = 1, so
 *  its QoF has one horizon); `aRX_DTest10` shares the columns of h = 1 with all the
 *  horizons of the multi-horizon model instead.
 *  > runMain scalation.modeling.forecasting.aRX_DTest6
 */
@main def aRX_DTest6 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp.set (("p", 10), ("px", 5), ("spec", 2))                          // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS  = false                                // TnT test: select on the training rows

    val mod = ARX_D.rescale (xe, y, hh, tFormT = Log1p)                 // create model for time series data
    mod.getModel ().tnT_Test (showYf = false)                           // the full model: the reference -- a COPY,
                                                                        // as `tnT_Test` sets skip = 0 and the
                                                                        // selection below must keep the burn-in

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam, CEM

    // Note: one line picks the techniques: all of them, or one at a time.
//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
//  val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
    val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech (at h = 1)")
        val (cols, rSq) = mod.selectFeaturesAtH (tech, "none")          // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (Array (cols), Array (rSq), h = 1, inSample = false)   // print, plot; TnT of the selected h = 1 model
    end for

end aRX_DTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
// YousefChange --------- Note: comment only -- the scaladoc below no longer names the sub-package built on it
/** The `aRX_DTest7` main function tests the `ARX_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection over ALL HORIZONS JOINTLY
 *  (`selectFeatures`, h = 0): every candidate column set is fit as ONE multi-horizon
 *  model and scored by J, the mean FPE-form sMAPE over the horizons
 *  (`Forecaster_D.aggFPE`, also the selection score of structure searches).  The selected
 *  multi-horizon model is then tested in-sample against the full model above.
 *  > runMain scalation.modeling.forecasting.aRX_DTest7
 */
@main def aRX_DTest7 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp.set (("p", 6), ("px", 6), ("spec", 1))                           // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS  = true                                 // in-sample test: select on all the rows

    val mod = ARX_D (xe, y, hh)                                         // create model for time series data
//  val mod = ARX_D.rescale (xe, y, hh, tFormT = Log1p)                 // or the rescaled version
    mod.inSample_Test ()                                                // the full model: the reference

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam, CEM

    // Note: one line picks the techniques: all of them, or one at a time.
//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
    val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
//  val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech (all horizons, J = mean FPE-form sMAPE)")
        val (colsArr, rSqArr) = mod.selectFeatures (tech)               // one multi-horizon model per candidate set
        mod.reportSelection (colsArr, rSqArr, tnt = false)              // print, plot; In-sample Test of the selected model
    end for

end aRX_DTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest8` main function tests the `ARX_D` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection over ALL HORIZONS JOINTLY
 *  (`selectFeatures`, h = 0, scored by J = the mean FPE-form sMAPE over the
 *  horizons) on the training rows, and tests the selected multi-horizon model with
 *  TnT against the full model above.
 *  > runMain scalation.modeling.forecasting.aRX_DTest8
 */
@main def aRX_DTest8 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp.set (("p", 6), ("px", 6), ("spec", 1))                           // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS  = false                                // TnT test: select on the training rows

    // val mod = ARX_D (xe, y, hh)                                         // create model for time series data
    val mod = ARX_D.rescale (xe, y, hh, tFormT = Log1p)                 // or the rescaled version
    mod.getModel ().tnT_Test (showYf = false)                           // the full model: the reference -- a COPY,
                                                                        // as `tnT_Test` sets skip = 0 and the
                                                                        // selection below must keep the burn-in

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam, CEM

    // Note: one line picks the techniques: all of them, or one at a time.
//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
//  val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
    val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech (all horizons, J = mean FPE-form sMAPE)")
        val (colsArr, rSqArr) = mod.selectFeatures (tech, h = 1)        // the single-horizon model at h = 1 decides
        mod.reportSelection (colsArr, rSqArr, inSample = false)         // print, plot; TnT of the selected multi-horizon model
    end for

end aRX_DTest8


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest9` main function tests the `ARX_D` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection PER HORIZON (`selectFeaturesPerH`): one
 *  column set and one single-horizon model per horizon, each rolling-validated on
 *  the testing rows by the selection itself.  The table at the end lists, per
 *  horizon, how many columns were kept, that model's TnT sMAPE and its features;
 *  the full model's TnT above is the reference, and `aRX_DTest8` is the same test
 *  with ONE column set for all the horizons.
 *  > runMain scalation.modeling.forecasting.aRX_DTest9
 */
@main def aRX_DTest9 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp.set (("p", 6), ("px", 6), ("spec", 1))                           // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS  = false                                // TnT test: select on the training rows

    val mod = ARX_D (xe, y, hh)                                         // create model for time series data
//  val mod = ARX_D.rescale (xe, y, hh, tFormT = Log1p)                 // or the rescaled version
    mod.getModel ().tnT_Test (showYf = false)                           // the full model: the reference -- a COPY,
                                                                        // as `tnT_Test` sets skip = 0 and the
                                                                        // selection below must keep the burn-in

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam, CEM

    // Note: one line picks the techniques: all of them, or one at a time.
//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
    val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
//  val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech (per horizon)")
        val (colsArr, rSqArr) = mod.selectFeaturesPerH (tech)           // one column set per horizon
        val (_, _, ftMat)     = mod.reportSelection (colsArr, rSqArr, inSample = false)   // print, TnT of each horizon's model

        banner (s"${mod.modelName}: per-horizon selection ($tech), TnT sMAPE on the testing rows")
        println ("  h   k   TnT smape   features")
        for h <- 1 to hh do
            println (f"$h%3d ${colsArr(h-1).size}%3d ${ftMat(h-1, QoF.smape.ordinal)}%11.3f   " +
                     colsArr(h-1).toSeq.sorted.map (mod.getFname (_)).mkString (", "))
        end for
        println ("kept at every horizon: " +
                 colsArr.reduce (_ intersect _).toSeq.sorted.map (mod.getFname (_)).mkString (", "))
    end for

end aRX_DTest9


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest10` main function tests the `ARX_D` class on real data:
 *  Forecasting COVID-19 and ILI using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version compares the three ways one column set can be chosen for a
 *  multi-horizon model -- `selectFeatures (h = 1)` (horizon 1 decides),
 *  `selectFeatures (h = hh)` (the last horizon decides) and `selectFeatures ()`
 *  (all horizons jointly, J = the mean FPE-form sMAPE) -- keeps the three selected
 *  models and runs TnT on all three (and on the full model) at the end, so their
 *  QoF blocks can be read side by side.
 *  > runMain scalation.modeling.forecasting.aRX_DTest10
 */
@main def aRX_DTest10 (): Unit =

    val hh = 6                                                          // maximum forecasting horizon
    hp.set (("p", 6), ("px", 6), ("spec", 1))                           // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS  = false                                // TnT test: select on the training rows

    val covid = clip (loadData (BEST2_EXO))                             // icu_patients, positive_rate
    val ili   = Example_ILI.loadData (Array ("%WEIGHTED ILI", "OT"), Example_ILI.response)
    val datasets = Seq (("Covid-19 (xe0 = icu_patients, xe1 = positive_rate)", covid),
                        ("ILI (xe0 = %WEIGHTED ILI, xe1 = OT)", ili))

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam, CEM

    // Note: one line picks the techniques: all of them, or one at a time.
//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
    val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
//  val techs = Array (CEM)

    for (tag, (xe, y)) <- datasets; tech <- techs do
        val mod = ARX_D (xe, y, hh)                                     // create model for time series data
        banner (s"$tag: ${mod.modelName}, ${mod.getX.dim2} columns, technique $tech")

        banner (s"$tag (1) at h = 1: horizon 1 decides, its columns shared by all horizons")
        val cols1 = mod.selectFeatures (tech, h = 1)._1(0)
        val mod1  = mod.getModel (cols1)                                // the first model to keep

        banner (s"$tag (2) at h = hh = $hh: the last horizon decides, its columns shared by all horizons")
        val colsH = mod.selectFeatures (tech, h = hh)._1(0)
        val modH  = mod.getModel (colsH)                                // the second model to keep

        banner (s"$tag (3) all horizons jointly: J = mean FPE-form sMAPE over the horizons")
        val cols0 = mod.selectFeatures (tech)._1(0)
        val mod0  = mod.getModel (cols0)                                // the third model to keep

        banner (s"$tag: the three selected column sets")
        println (s"at h = 1      k = ${cols1.size}: ${stringOf (mod1.getFname)}")
        println (s"at h = $hh    k = ${colsH.size}: ${stringOf (modH.getFname)}")
        println (s"all (mean J)  k = ${cols0.size}: ${stringOf (mod0.getFname)}")

        banner (s"$tag: TnT of the full model (the reference)")
        mod.tnT_Test  (showYf = false)
        banner (s"$tag: TnT of the model selected at h = 1")
        mod1.tnT_Test (showYf = false)
        banner (s"$tag: TnT of the model selected at h = $hh")
        modH.tnT_Test (showYf = false)
        banner (s"$tag: TnT of the model selected over all horizons (mean J)")
        mod0.tnT_Test (showYf = false)
    end for

end aRX_DTest10


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest11` main function tests `ARX_D.fromTerms` on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  The model is built from a FORMULA's terms -- lag windows of y, icu and pRate plus
 *  log1p and squared terms -- so skipped lags and transforms that contiguous lags
 *  1 to p cannot express are included; the matrix has no intercept column, hence
 *  first = 0.  Feature Selection runs over all horizons jointly (J = the mean
 *  FPE-form sMAPE) and the selected model is tested with TnT.
 *  > runMain scalation.modeling.forecasting.aRX_DTest11
 */
@main def aRX_DTest11 (): Unit =

    val (xe, y_) = clip (loadData (BEST2_EXO))                          // icu_patients, positive_rate
    given dataContext: Map [String, VectorD] = Map (
        "y"     -> y_,
        "icu"   -> xe.col(0),
        "pRate" -> xe.col(1))

    val y     = Vars.y
//  val icu   = Vars.icu
//  val pRate = Vars.pRate
    val hh    = 6                                                       // maximum forecasting horizon
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS  = false                                // TnT test: select on the training rows

    banner ("library: lag windows of y, icu and pRate plus log1p and squared terms")
//  val lib = y ~= y~(6,1) + icu~(6,1) + pRate~(6,1) + log1p (icu~(3,1)) + sq (pRate~(3,1))
    val lib = y ~= y~(5,1)                                              // Same as ARX_D_5_0_0
    val mod = ARX_D.fromTerms (lib.resp, lib.preds, hh)                 // create model for time series data
    println (s"formula = $lib")
    println (s"library columns = ${mod.getX.dim2}: ${stringOf (mod.getFname)}")
    mod.getModel ().tnT_Test ()                                         // the full model: the reference -- a COPY,
                                                                        // as `tnT_Test` sets skip = 0 and the
                                                                        // selection below must keep the burn-in

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam, CEM

    // Note: one line picks the techniques: all of them, or one at a time.
//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
    val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
//  val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech (all horizons, first = 0: no intercept column)")
        val (colsArr, rSqArr) = mod.selectFeatures (tech, first = 0)    // one multi-horizon model per candidate set
        mod.reportSelection (colsArr, rSqArr, inSample = false)         // k, features, trace, plot; TnT of the selected model
    end for

end aRX_DTest11


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_DTest12` main function tests `ARX_D.fromTerms` with the training-window
 *  contract on the COVID-19 dataset: a model built from a formula on the FULL series
 *  is trained on its leading trSize rows only, then its forecast matrix is filled
 *  for every row, so the rows from trSize on are out-of-sample.  Checks:
 *  (1) the fit uses the training rows only: its h = 1 parameters equal those of a
 *      `CenteredRidgeMV` fit directly on the training rows, and that model's
 *      original-space intercept and slopes reproduce `predict` on a testing row,
 *  (2) scaled = true with lambda = 0 gives the same forecasts as scaled = false,
 *  (3) a standardized response is never rectified and the global hp("nneg") is
 *      left alone by `fromTerms`.
 *  Feature selection on a `fromTerms` matrix is tested by `aRX_DTest11`.
 *  > runMain scalation.modeling.forecasting.aRX_DTest12
 */
@main def aRX_DTest12 (): Unit =

    import scalation.modeling.neuralnet.CenteredRidgeMV

    val (xe, y_) = clip (loadData (BEST2_EXO))
    given dataContext: Map [String, VectorD] = Map (
        "y"     -> y_,
        "icu"   -> xe.col(0),
        "pRate" -> xe.col(1))

    val y     = Vars.y
    val icu   = Vars.icu
    val pRate = Vars.pRate
    val hh    = 6                                                       // maximum forecasting horizon
    val f     = y ~= y~(3,1) + icu~1 + log1p (pRate~2)                  // formula: 3 lags of y, icu~1, log1p (pRate~2)
    val trSz  = Model.trSize (y_.dim)                                   // training window: first 80% of the rows
    val lam   = RidgeRegression.hp("lambda")                            // remember the global lambda
    RidgeRegression.hp("lambda") = 0.0                                  // use 0.0 for OLS, so (2) is exact

    banner ("(1) fromTerms on the full series, trained on the first trSz rows only")
    val mod = ARX_D.fromTerms (f.resp, f.preds, hh, trSize = trSz)
    val (x, yy) = (mod.getX, mod.getYy)
    println (s"modelName = ${mod.modelName}, fname = ${stringOf (mod.getFname)}, x.dims = ${x.dims}, trSz = $trSz")
    mod.train (x(0 until trSz), yy(0 until trSz))                       // means of the training rows only
    mod.test (x, yy)                                                    // fill the forecast matrix for every row
    val qof = mod.diagnoseAll (mod.getY, mod.getYf, trSz until y_.dim)  // out-of-sample QoF per horizon
    println (s"out-of-sample smape per horizon = ${qof(?, QoF.smape.ordinal)}")

    val crm = new CenteredRidgeMV (x(0 until trSz), yy(0 until trSz))   // the same fit, made directly
    crm.train ()
    assert ((crm.parameter(?, 0) - mod.parameter).normSq < 1E-10 * (1.0 + mod.parameter.normSq))
    val t      = trSz + 5                                               // a testing row
    val manual = (crm.intercept + (crm.slopes dot x(t))).map (v => math.max (0.0, v))   // b0 + x_t dot slopes, rectified
    println (s"row $t: predict = ${mod.predict (t, yy)}\n         manual  = $manual")
    assert ((mod.predict (t, yy) - manual).normSq < 1E-8 * (1.0 + manual.normSq))

    banner ("(2) scaled = true with lambda = 0 gives the same forecasts")
    val modS = ARX_D.fromTerms (f.resp, f.preds, hh, trSize = trSz, scaled = true)
    modS.train (x(0 until trSz), yy(0 until trSz))
    modS.test (x, yy)
    assert ((modS.getYf - mod.getYf).normFSq < 1E-8 * (1.0 + mod.getYf.normFSq))
    println ("scaled and unscaled forecasts agree")

    banner ("(3) a standardized response is never rectified; hp(\"nneg\") is not modified")
    val nneg0 = AR.hp("nneg")
    val fN    = y ~= y~(3,1) + icu~1 + pRate~2                          // no log1p: a standardized column can be negative
    val modN  = ARX_D.fromTerms (fN.resp, fN.preds, hh, tFormT = TransformT.Norm, trSize = trSz)
    val (xN, yyN) = (modN.getX, modN.getYy)                             // standardized columns
    modN.train (xN(0 until trSz), yyN(0 until trSz))
    modN.test (xN, yyN)
    assert (AR.hp("nneg") == nneg0, "fromTerms changed the global hp(\"nneg\")")
    assert (ARX_D.centered (Map ("tForm_y" -> NormForm (y_))) && ! ARX_D.centered (Map ("tForm_y" -> Log1pForm (y_))))
    println (s"standardized-space forecast at t = 0: ${modN.getYf(0, 1)} (may be negative), yForm = ${modN.getYForm}")

    RidgeRegression.hp("lambda") = lam                                  // restore the global lambda
    println ("aRX_DTest12: passed")

end aRX_DTest12

