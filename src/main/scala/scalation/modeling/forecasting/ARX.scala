
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe (ARX) using OLS
 *
 *  @see `scalation.modeling.RidgeRegression`
 *  @see `scalation.modeling.forecasting.ARY` when no exogenous variable are needed
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX` class provides basic time series analysis capabilities for ARX models.
 *  ARX models build on `ARY` by including one or more exogenous (xe) variables.
 *  Given time series data stored in vector y, its next value y_t = combination of
 *  last p values of y and the last q values of each exogenous variable xe_j.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y and xe) @see `ARX.apply`
 *  @param y        the response/output vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 *  @param scaled   whether the regression standardizes the columns of x (training mean and
 *                      standard deviation) instead of only centering them (defaults to false)
 */
class ARX (x: MatrixD, y: VectorD, hh: Int, n_exo: Int, fname: Array [String],
           tRng: Range = null, hparam: HyperParameter = AR.hp,
           bakcast: Boolean = false,
           tForms: TransformMap = Map ("tForm_y" -> null),
           scaled: Boolean = false)
      extends Forecaster_Reg (x, y, hh, fname, tRng, hparam, bakcast, scaled = scaled):

    private   val debug = debugf ("ARX", false)                         // debug function
    protected val p     = hparam("p").toInt                             // use the last p endogenous values (p lags)
    protected val q     = hparam("px").toInt                            // use the last q exogenous values (q lags)
    protected val spec  = hparam("spec").toInt                          // trend terms: 0 - implicit constant, 1 - linear, 2 - quadratic, 3 - cubic
    override protected val nneg = hparam("nneg").toInt == 1 &&          // 0 => unrestricted, 1 => predictions must be non-negative
                                  ! ARX.centered (tForms)               // (never in a centered/standardized space, @see `ARX_D.centered`)

    _modelName = s"ARX_${p}_${q}_$n_exo"
    yForm      = tForms("tForm_y").asInstanceOf [Transform]

    debug ("init", s"$modelName with $n_exo exogenous variables and additional trend spec = $spec, x.dims = ${x.dims}")
//  debug ("init", s"[ x | y ] = ${x :^+ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a new vector from the first spec values of x, the last p-h+1 values
     *  of x (past values), values 1 to h-1 from the forecasts, and available values
     *  from exogenous variables.
     *  @param xx  the t-th row of the input matrix (lagged actual values)
     *  @param yy  the t-th row of the forecast matrix (forecasted future values)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forge (xx: VectorD, yy: VectorD, h: Int): VectorD =
        // add terms for the endogenous variable
        val n_endo  = spec + p                                       // number of trend + endogenous values
        val x_act   = xx(n_endo-(p+1-h) until n_endo)                // get actual lagged y-values (endogenous)
        val nyy     = p - x_act.dim                                  // number of forecasted values needed
        val x_fcast = yy(h-nyy until h)                              // get forecasted y-values

        var xy = x_act ++ x_fcast
        if n_exo > 0 and q > 0 then
            for j <- 0 until n_exo do                                // for the j-th exogenous variable
                xy = xy ++ hide (xx(n_endo + j*q until n_endo + (j+1)*q), h)
        xx(0 until spec) ++ xy
    end forge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Hide values at the end of vector z (last h-1 values) as the increasing horizon  
     *  turns them in future values (hence unavailable).  Set these values to either
     *  zero (the default) or the last available value.
     *  @param z     the vector to shift
     *  @param h     the current horizon (number of steps ahead to forecast)
     *  @param fill  whether to backfill with the rightmost value (true) or with 0 (false)
     */
    def hide (z: VectorD, h: Int, fill: Boolean = true): VectorD =
        val zl = z(z.dim - 1)                                        // last available z value per horizon
        val z_ = new VectorD (z.dim)
        for k <- z.indices do
            z_(k) = if k <= z.dim - h then z(k+h-1) else if fill then zl else 0.0
        z_
    end hide

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a `ARX` model using the cols with the selected features.
     *  @param cols  the cols of the input matrix with selected features
     */
    def getModel (cols: LSET [Int] = mcols): ARX =
        new ARX (getX(?, cols), getY, hh, n_exo, cols.toArray.map (fname (_)), tRng, hparam, bakcast, tForms, scaled)
    end getModel

end ARX


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX` companion object provides factory methods for the `ARX` class.
 */
object ARX extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX` object by building an input matrix xy and then calling the
     *  `ARX` constructor.
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
               bakcast: Boolean = false): ARX =

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
            xe_bfill = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val xy = buildMatrix (xe_bfill, y, hparam, bakcast)
        val fname = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        new ARX (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX` object by building an input matrix xy and then calling the
     *  `ARX` constructor, with rescaling of endogneous and exogenous variable values.
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
                 tFormT: TransformT = MinMax): ARX =

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
        val xy     = buildMatrix (xe_bfill, y_scl, hparam, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        new ARX (xy, y_scl, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX` object by building an input matrix xy and then calling the
     *  `ARX` constructor.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def trend (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = AR.hp,
               bakcast: Boolean = false): ARX =

        val hp2 = hparam.updateReturn (("p", 0), ("px", 0))
        val xe_bfill: MatrixD = null

        val xy = buildMatrix (xe_bfill, y, hparam, bakcast)
        val fname = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        new ARX (xy, y, hh, xe.dim2, fname, tRng, hp2, bakcast)
    end trend

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the spec + p columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  When cross = true, additional cross terms will be added.  Columns produced
     *  by transformations will be added as well.
     *  @param xe_bfill  the matrix of exogenous variable values
     *  @param y         the endogenous/response vector (main time series data)
     *  @param hp_       the hyper-parameters
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     */
    def buildMatrix (xe_bfill: MatrixD, y: VectorD, hp_ : HyperParameter, bakcast: Boolean): MatrixD =

        val (p, q, spec) = (hp_("p").toInt, hp_("px").toInt, hp_("spec").toInt)

        // make matrix xy for trend terms and lagged terms of the endogenous variable
        var xy = makeMatrix4T (y, spec, bakcast) ++^                        // trend terms
                 makeMatrix4L (y, p, bakcast)                               // lagged linear terms

        // apply transformations fExo to the exogenous variables and add there columns to x_exo
        if xe_bfill != null and q > 0 then
            xy = xy ++^ makeMatrix4L (xe_bfill, q, bakcast)                 // add lagged exogenous term to xy

        println (s"xy.dims = ${xy.dims}")
        xy
    end buildMatrix

   //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX` object from the terms of a formula (@see `ARX_Formula`),
     *  with one column per term (lagged variable, transform, interaction or wave),
     *  so skipped lags and interactions that contiguous lags 1 to p/px cannot
     *  express are supported.  Rows are not trimmed: each term pads its first lag
     *  values as `ARX.buildMatrix` does (@see `VectorD.lagv`).  When tFormT is given,
     *  every base column is transformed (fit on its training slice, as in `rescale`)
     *  before the terms are evaluated, and the response's transform is kept for
     *  inverting forecasts.  A term whose column is not finite (e.g., log of 0, exp
     *  of a large value) is rejected here, before the regression sees it.
     *  The matrix has NO intercept column (the regression centers, @see `Forecaster_Reg,
     *  so feature selection on it must use first = 0 (@see `Forecaster_Reg.selectFeatures`).
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
                   (using dataset: Map [String, VectorD]): ARX =

        val td    = ARX_Formula.prepare (resp, preds, tFormT, trSize, bakcast)   // checks and base-column transforms
        val xy    = ARX_Formula.matrix (preds, td.data)                          // one column per term
        val fname = if fname_ == null then ARX_Formula.names (preds) else fname_
        val (hp_, n_exo) = ARX_Formula.lagHP (resp, preds, hparam)
        new ARX (xy, td.y, hh, n_exo, fname, tRng, hp_, bakcast, Map ("tForm_y" -> td.tForm_y), scaled)
    end fromTerms
/*
    def fromTerms (resp: FormulaVar, preds: VEC [Term], hh: Int,
                   fname_ : Array [String] = null, tRng: Range = null,
                   hparam: HyperParameter = AR.hp, bakcast: Boolean = false,
                   tFormT: TransformT = null, trSize: Int = -1, scaled: Boolean = false)
                   (using dataset: Map [String, VectorD]): ARX =

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
//      val yy     = makeMatrix4Y (y_scl, hh, bakcast)
        val tForms = Map ("tForm_y" -> tForm_y)
        val fname  = if fname_ == null then preds.map (_.label).toArray else fname_

        val pY    = preds.filter (_.vars == Set (resp.name)).map (_.lag).maxOption.getOrElse (0)
        val pX    = preds.filter (t => t.vars.nonEmpty && t.vars != Set (resp.name)).map (_.lag).maxOption.getOrElse (0)
        val n_exo = (preds.flatMap (_.vars).toSet - resp.name).size        // distinct exogenous variables in the terms
        val hp_   = hparam.updateReturn (("p", pY), ("px", pX), ("spec", 0))
        new ARX (xy, y_scl, hh, n_exo, fname, tRng, hp_, bakcast, tForms, scaled)
    end fromTerms
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables (none for `ARX`)
     *  @param n_fExArr  the number of functions used to map exogenous variables (none for `ARX`)
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int = 0, n_fExArr: Array [Int] = null): Array [String] =

        val (p, q, spec) = (hp_("p").toInt, hp_("px").toInt, hp_("spec").toInt)
        val names = VEC [String] ()
        for j <- 0 until n_exo; k <- q to 1 by -1 do names += s"xe${j}l$k"
        MakeMatrix4TS.formNames (spec, p) ++ names.toArray
    end formNames

end ARX

import Example_LakeLevels.y
import AR.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest` main function tests the `ARX` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRXTest
 */
@main def aRXTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX (EMPTY_EXO, y, hh)                                    // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

end aRXTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest2` main function tests the `ARX` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRXTest2
 */
@main def aRXTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX (EMPTY_EXO, y, hh)                                    // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validation

end aRXTest2

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest3` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for AR(p), ARX(p), ARX_D(p), ARX_RD_p),
 *  ARX_Quad(p), ARX_Quad_D, ARX_Quad_RD, ARX_SR
 *
 *  20.3933,    30.9930,    40.4824,    48.8231,    56.4826,    63.4345  RW
 *
 *  20.1949,    29.8361,    38.2024,    46.2682,    54.1661,    60.6188  AR(1)
 *  20.1798,    29.8594,    38.1450,    45.5636,    52.3473,    57.4466  ARX(1)
 *  20.4293,    30.9204,    39.6444,    45.6790,    49.7719,    52.6263  ARX_D(1)
 *  20.5401,    32.9343,    42.2589,    47.9236,    50.9710,    51.2585  ARX_RD(1)
 *  20.2187,    29.8534,    38.0669,    45.5270,    52.2238,    57.2811  ARX_Quad(1) @1.8 Pow
 *  19.5432,    31.5957,    40.6543,    45.9527,    49.1867,    50.3784  ARX_Quad_D(1)
 *  19.3691,    31.7913,    40.8996,    46.1389,    49.1578,    50.3728  ARX_Quad_RD(1)
 *  19.8426,    29.4308,    37.7936,    45.2651,    52.1618,    57.6120  ARX_SR(1) @0.2 Pow, 0.2mx DQuad
 *  20.1293,    30.3451,    37.7556,    43.4998,    47.3151,    49.6745  ARX_SR_D(1)
 *  20.4948,    31.1367,    38.5559,    44.2841,    47.4833,    49.4377  ARX_SR_RD(1)
 *
 *  16.3579,    24.7155,    33.0480,    40.1643,    46.8762,    53.2178  AR(2)
 *  16.2602,    23.7221,    32.0026,    39.0003,    44.9917,    50.2442  ARX(2)
 *  16.3135,    23.7149,    30.9058,    36.2627,    41.2780,    46.3900  ARX_D(2)
 *  15.3328,    23.5382,    30.7663,    36.9936,    42.1214,    45.9565  ARX_RD(2)
 *  16.2799,    22.9376,    31.4021,    37.9903,    44.3656,    50.0553  ARX_Quad(2)
 *  14.4011,    20.4339,    27.2183,    33.5937,    39.3402,    44.4232  ARX_Quad_D(2)
 *  14.2257,    20.5106,    27.7323,    34.0540,    39.3691,    44.4389  ARX_Quad_RD(2)
 *  15.5101,    26.0884,    33.7850,    39.6578,    45.6508,    50.7941  ARX_SR(2)
 *  15.6475,    23.4081,    28.2301,    32.5121,    37.9013,    44.0801  ARX_SR_D(2)
 *  15.4762,    23.6560,    28.3018,    32.5947,    37.7637,    43.8000  ARX_SR_RD(2)
 *
 *  14.5467,    21.5279,    29.0858,    35.3761,    41.0748,    46.9813  AR(3)
 *  14.3865,    21.1050,    28.7310,    34.8707,    40.4586,    45.8412  ARX(3)
 *  14.5005,    21.3985,    28.5130,    33.8650,    39.3643,    43.8694  ARX_D(3)
 *  13.3470,    20.7260,    29.0960,    35.9171,    41.7920,    45.5489  ARX_RD(3)
 *  14.2592,    20.2073,    28.0450,    34.3864,    40.3448,    45.7260  ARX_Quad(3)
 *  12.7571,    18.6820,    27.0169,    33.8436,    39.5992,    44.3995  ARX_Quad_D(3)
 *  12.4017,    18.6742,    27.0352,    33.9936,    39.6830,    44.3689  ARX_Quad_RD(3)
 *  13.7214,    22.5882,    28.8758,    37.6656,    44.2559,    49.8216  ARX_SR(3)
 *  13.8104,    20.3868,    25.2633,    30.4033,    36.2286,    41.7853  ARX_SR_D(3)
 *  13.7913,    20.5442,    25.4682,    30.4177,    36.6624,    41.7404  ARX_SR_RD(3)
 *
 *  13.3945,    21.0643,    27.8030,    33.0090,    38.4717,    43.6289  AR(4)
 *  13.3183,    20.6711,    27.3836,    32.7528,    38.3256,    43.4674  ARX(4)
 *  13.4085,    21.0572,    27.9240,    33.1802,    38.7518,    43.0492  ARX_D(4)
 *  13.3690,    20.8569,    28.8816,    35.8043,    41.7175,    45.8119  ARX_RD(4)
 *  13.4614,    20.3714,    28.1860,    34.3754,    39.9186,    44.7558  ARX_Quad(4)
 *  12.5816,    18.9073,    26.9637,    33.2016,    38.6594,    43.5066  ARX_Quad_D(4)
 *  12.5075,    19.0267,    27.1824,    33.4721,    38.7702,    43.4985  ARX_Quad_RD(4)
 *  12.8991,    20.9464,    27.8061,    36.3788,    42.0512,    46.2560  ARX_SR(4)
 *  13.1272,    18.9123,    23.4027,    29.5380,    35.1066,    40.7952  ARX_SR_D(4)
 *  13.4334,    19.3549,    23.8732,    29.6392,    35.9812,    41.2532  ARX_SR_RD(4)
 *
 *  13.4819,    20.5284,    27.0367,    32.3694,    38.0965,    42.5019  AR(5)
 *  13.2489,    20.0563,    26.5879,    32.0772,    37.9229,    43.3617  ARX(5)
 *  13.3972,    20.4249,    27.1716,    32.5678,    38.4480,    43.1454  ARX_D(5)
 *  13.0204,    20.6342,    27.8612,    35.2164,    41.3175,    45.1612  ARX_RD(5)
 *  12.9587,    20.0597,    28.0812,    34.7720,    40.6403,    45.5188  ARX_Quad(5)
 *  12.3520,    18.2275,    24.3962,    29.7891,    35.1957,    40.3741  ARX_Quad_D(5)
 *  12.1063,    18.2434,    24.3356,    29.7247,    35.1368,    40.3257  ARX_Quad_RD(5)
 *  11.7793,    19.5368,    26.7329,    33.0278,    38.6541,    40.5255  ARX_SR(5)
 *  11.9531,    18.2499,    22.9514,    28.3416,    33.4735,    39.1343  ARX_SR_D(5)
 *  12.2984,    18.3807,    23.0066,    29.0608,    34.8016,    40.4868  ARX_SR_RD(5)
 *
 *  > runMain scalation.modeling.forecasting.aRXTest3
 */
@main def aRXTest3 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))                // for testing against `ARY, should be the same
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

//  new Plot (null, y, null, s"y (new_deaths) vs. t", lines = true)
//  if xe.dim2 > 0 then new Plot (null, xe(?, 0), null, s"x_0 (${BEST_EXO(0)}) vs. t", lines = true)

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX (xe, y, hh)                                       // create model for time series data
//      val mod = ARX.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit
    end for

/*
    banner ("Test NormForm Transform")
    val yform  = NormForm (y)
    val y_     = yform.f (y)
    val xeform = NormForm (xe(?, 0))
    val xe_    = xeform.f (xe(?, 0))
    new Plot (null, y_, xe_, s"y (new_deaths) vs. t", lines = true)

    for s <- 0 to 5 do
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3
        val mod = ARX.trend (xe, y, hh)                                 // create model for time series data
        banner (s"In-ST Trend (s = $s) Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
//      println (mod.summary ())                                        // statistical summary of fit
    end for
*/

end aRXTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest4` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for AR(p), ARX(p), ARX_D(p), ARX_RD_p),
 *  ARX_Quad(p), ARX_Quad_D, ARX_Quad_RD, ARX_SR
 *
 *  18.1532,    27.2211,    40.3519,    52.3739,    62.5276,    73.6424  RW
 *
 *  18.4029,    29.5363,    41.9465,    53.3411,    62.7636,    73.1441  AR(1)
 *  18.5538,    30.1712,    43.2886,    55.0143,    63.8081,    72.8042  ARX(1)
 *  18.5489,    31.4088,    45.3246,    57.0302,    64.1919,    70.4402  ARX_D(1)
 *  18.5349,    31.3709,    45.2523,    56.9240,    64.0752,    70.4146  ARX_RD(1)
 *  18.7321,    29.6130,    43.3588,    55.6745,    64.8136,    74.4036  ARX_Quad(1) @1.8 Pow
 *  18.7194,    30.6062,    45.5023,    58.0389,    65.7463,    72.5828  ARX_Quad_D(1)
 *  18.7227,    30.6158,    45.5883,    58.1697,    65.8960,    72.6589  ARX_Quad_RD(1)
 *  18.8207,    29.4770,    43.5409,    56.0759,    65.5002,    75.2714  ARX_SR(1) @0.2 Pow, 0.6mx DQuad
 *  18.8062,    30.4702,    45.8893,    58.6089,    66.0756,    72.4736  ARX_SR_D(1)
 *  18.8058,    30.4615,    45.8339,    58.4783,    65.8705,    72.2785  ARX_SR_RD(1)
 *
 *  18.0433,    26.2445,    38.7122,    49.5094,    58.1561,    69.7676  AR(2)
 *  17.8528,    28.6166,    41.5335,    53.6381,    61.8345,    70.6173  ARX(2)
 *  17.8309,    25.9739,    36.0776,    45.6313,    50.8811,    60.4757  ARX_D(2)
 *  17.8142,    25.9406,    36.0205,    45.5000,    50.7264,    60.4388  ARX_RD(2)
 *  18.4966,    29.5766,    44.3132,    57.9942,    67.5151,    76.8433  ARX_Quad(2)
 *  18.4745,    26.8347,    36.7151,    47.8507,    53.3711,    64.2207  ARX_Quad_D(2)
 *  18.4802,    26.8263,    36.7704,    47.9813,    53.5240,    64.2968  ARX_Quad_RD(2)
 *  18.0815,    29.0127,    44.1220,    58.3131,    68.4855,    78.5462  ARX_SR(2)
 *  18.0902,    27.0981,    35.6004,    46.8282,    51.4226,    62.8227  ARX_SR_D(2) 
 *  18.0918,    27.0977,    35.5748,    46.7552,    51.2960,    62.7028  ARX_SR_RD(2)
 *
 *  16.4218,    23.5256,    32.6385,    44.4111,    51.8210,    63.7353  AR(3)
 *  16.2212,    23.5287,    33.8367,    44.2112,    52.7507,    61.8977  ARX(3)
 *  16.2013,    21.9274,    28.8492,    37.8377,    41.2254,    52.1445  ARX_D(3)
 *  16.1788,    21.8858,    28.7777,    37.6909,    41.0501,    52.0969  ARX_RD(3)
 *  16.3746,    22.7779,    35.5390,    47.5890,    57.4027,    68.1882  ARX_Quad(3)
 *  16.3590,    20.9860,    28.9631,    36.7369,    43.3968,    54.4968  ARX_Quad_D(3)
 *  16.3603,    20.9617,    28.9858,    36.8270,    43.4656,    54.5511  ARX_Quad_RD(3)
 *  15.9230,    22.0030,    33.1480,    44.8791,    54.7551,    66.3781  ARX_SR(3)
 *  15.9602,    20.6944,    25.9128,    37.2356,    39.1055,    52.5149  ARX_SR_D(3)
 *  15.9625,    20.6943,    25.9127,    37.2002,    39.0479,    52.4522  ARX_SR_RD(3)
 *
 *  14.8153,    21.8989,    29.9364,    39.3910,    47.8124,    53.5577  AR(4)
 *  15.5229,    21.6827,    28.9578,    39.2641,    41.9809,    51.6632  ARX(4)
 *  15.4977,    21.5394,    29.1667,    37.7948,    41.7922,    50.3497  ARX_D(4)
 *  15.4740,    21.4949,    29.0590,    37.6429,    41.6133,    50.2969  ARX_RD(4)
 *  14.4700,    20.1519,    28.2077,    37.2514,    42.3221,    50.5717  ARX_Quad(4)
 *  14.4294,    19.8261,    26.7720,    36.2867,    41.5804,    50.6255  ARX_Quad_D(4)
 *  14.4352,    19.8153,    26.7311,    36.4097,    41.7308,    50.7064  ARX_Quad_RD(4)
 *  12.2379,    18.3076,    23.5460,    33.6728,    36.7798,    46.2768  ARX_SR(4)
 *  12.2569,    18.1510,    22.7928,    34.7904,    36.5732,    47.9044  ARX_SR_D(4)
 *  12.2592,    18.1502,    22.7983,    34.7559,    36.5120,    47.8381  ARX_SR_RD(4)
 *
 *  14.3593,    21.9036,    29.1791,    40.6232,    46.1989,    55.2531  AR(5)
 *  15.9034,    21.1997,    29.2566,    39.1411,    42.1765,    50.4310  ARX(5)
 *  15.8876,    21.1755,    29.3186,    38.2420,    42.0767,    50.5327  ARX_D(5)
 *  15.8657,    21.1132,    29.2284,    38.1140,    41.9263,    50.4838  ARX_RD(5)
 *  15.0209,    20.1297,    28.1222,    36.3393,    40.4033,    48.4057  ARX_Quad(5)
 *  14.9983,    20.7931,    26.3594,    37.5566,    42.3055,    51.3716  ARX_Quad_D(5)
 *  15.0160,    20.7740,    26.5098,    37.8117,    42.6775,    51.5523  ARX_Quad_RD(5)
 *  15.0146,    20.6906,    26.3743,    37.4673,    42.1440,    51.1924  ARX_Quad_RD(5)
 *  11.7856,    17.9338,    21.8066,    31.9418,    33.1844,    43.3829  ARX_SR(5)
 *  11.8188,    19.1837,    22.9515,    35.0939,    36.4313,    46.3892  ARX_SR_D(5)
 *  11.8225,    19.1836,    22.9488,    35.0419,    36.3341,    46.2893  ARX_SR_RD(5)
 *
 *  > runMain scalation.modeling.forecasting.aRXTest4
 */
@main def aRXTest4 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))                // for testing against `ARY, should be the same
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    Fit.hp("omega") = 0.0                                               // blending parameter WAPE (0) <-> NMAE (1)
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX (xe, y, hh)                                       // create model for time series data
//      val mod = ARX.rescale (xe, y, hh, tFormT = Log1p)
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRXTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest5` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection.
 *  > runMain scalation.modeling.forecasting.aRXTest5
 */
@main def aRXTest5 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp.set (("p", 10), ("px", 5), ("spec", 2))                          // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS = true                                  // use full dataset for FS

    val mod = ARX (xe, y, hh)                                           // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

    // Note: one line picks the techniques: all of them, or one at a time.
    val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
//  val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
//  val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (cols, rSq, tnt = false)

//      val imp = mod.importance (cols.toArray, rSq.ᵀ)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRXTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest6` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection (training-set, training-set).
 *  > runMain scalation.modeling.forecasting.aRXTest6
 */
@main def aRXTest6 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp.set (("p", 10), ("px", 5), ("spec", 2))                          // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    val mod = ARX (xe, y, hh)                                           // create model for time series data
    banner (s"Training-only Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                 // initially train and test on full dataset

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

//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
//  val techs = Array (Backward)
//  val techs = Array (Stepwise)
//  val techs = Array (Beam)
    val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (cols, rSq, tnt = false)

//      val imp = mod.importance (cols.toArray, rSq.ᵀ)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRXTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest7` main function tests the `ARX` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection (training-set, testing-set).
 *  > runMain scalation.modeling.forecasting.aRXTest7
 */
@main def aRXTest7 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 1  // 6                                               // maximum forecasting horizon
    hp.set (("p", 10), ("px", 5), ("spec", 2))                          // endo lags, exo lags, trend spec: 0, 1, 2, 3
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    val mod = ARX (xe, y, hh)                                           // create model for time series data
    banner (s"Training-Test Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                 // initially train and test on full dataset

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

    // Note: one line picks the techniques: all of them, or one at a time.
//  val techs = values                                                  // try all feature selection techniques
//  val techs = Array (Forward)                                         // try a particular one
//  val techs = Array (Backward)
//  val techs = Array (Stepwise)
    val techs = Array (Beam)
//  val techs = Array (CEM)

    for tech <- techs do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech, "none")             // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (cols, rSq, inSample = false)               // TnT of the selected model

//      val imp = mod.importance (cols.toArray, rSq.ᵀ)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRXTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRXTest8` main function tests the `ARX` class in terms of model specification
 *  formulas/equations.  Nore:  y~2 means y(t-2)
 *  > runMain scalation.modeling.forecasting.aRXTest8
 */
@main def aRXTest8 (): Unit =

    // Setup mock historic data vectors
    val y = VectorD (12.0, 15.0, 14.0, 16.0, 18.0, 21.0, 19.0, 22.0)
    val x = VectorD (4.0,   5.2,  4.8,  5.5,  6.0,  6.2,  5.9,  6.5)

    // Model Coefficients (Estimated via Ordinary Least Squares or similar)
    val c  = 1.5
    val φ = VectorD (-0.2, 0.6)
    val β = VectorD (1.1, 0.8)
    val r = (2, 1)                               // back 2, back 1

    println ("--- Step-by-Step ARX Prediction Loop ---")
    for timestamp <- 2 until y.dim do

        given t: Int = timestamp                 // activates contextual time lookup

        banner (s"timestamp t = $t")

        val y_pred = c + φ(0) * y~2 + φ(1) * y~1 + β(0) * x~2 + β(1) * x~1
        val y_pre2 = c + φ ∙ (y~r) + β ∙ (x~r)
        
        println (s"Scalar Formula: Time $t | Actual: ${y(t)} | Predicted: $y_pred")
        println (s"Vector Formula: Time $t | Actual: ${y(t)} | Predicte2: $y_pre2")

end aRXTest8

