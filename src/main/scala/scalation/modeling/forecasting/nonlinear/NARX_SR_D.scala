
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Wed Nov 19 22:58:47 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe with SR terms (NARX_SR_D) whose
 *           nonlinear parameters are fitted by a nonlinear optimizer - Direct Forecasting
 *           AI Assisted Code
 *
 *  @see `scalation.optimization.nonlinear_ls.VarPro`
 *  @see `NARX_Formula` for models entered as formulas with free parameters
 */

package scalation
package modeling
package forecasting
package nonlinear

import scala.collection.mutable.{ArrayBuffer, LinkedHashSet => LSET}
import scala.math.{abs, max, sqrt}

import scalation.mathstat._
import scalation.optimization.nonlinear_ls.{LevenbergMarquardt_TR, VarPro}
import scalation.optimization.quasi_newton.LBFGS_B

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `OptimizerNL` enum lists the nonlinear optimizers `NARX_SR_D.optimize` can use
 *  to fit the nonlinear parameters a of the transforms.  All of them minimize the
 *  same objective, the ridge least squares criterion of the model's regression
 *  (centered/scaled as the regression is),
 *
 *      || y - X (a) b ||^2 + lambda || b ||^2,
 *
 *  summed over the horizons, but they handle the linear coefficients b differently:
 *  VarPro   eliminates b (b (a) by QR) and runs `LevenbergMarquardt_TR` on the reduced
 *           residual (Variable Projection, Kaufman/finite-difference Jacobian)
 *  LBFGS_B  eliminates b in the same way and runs the bounded quasi-Newton `LBFGS_B`
 *           on the reduced objective (finite-difference gradient, exact line search)
 *  Joint    runs `LevenbergMarquardt_TR` over a and b together (b unbounded), starting
 *           from b (a0)
 */
enum OptimizerNL:
    case VarPro, LBFGS_B, Joint
end OptimizerNL

/** The nonlinear optimizer used by `NARX_SR_D.optimize` (a test may reassign it).
 */
var OPTIMIZER_NL: OptimizerNL = OptimizerNL.VarPro

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NARX_SR_D` class provides time series analysis capabilities for NARX Symbolic
 *  Regression (SR) models with Direct (D) forecasting.  These models include trend, linear,
 *  power, root, log, ... and cross terms for the single endogenous (y) variable and zero
 *  or more exogenous (xe) variables.
 *  Given time series data stored in vector y and matrix xe, its next value y_t = combination
 *  of last p values of y and of its transforms f_i (y), and the last q values of each exogenous
 *  variable xe_j and of its transforms g_ji (xe_j) (as well as ENDO-EXO cross terms).
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t, x_t is a vector of inputs, and e_t is the
 *  residual/error term.  The model is LINEAR in the coefficients b and NONLINEAR in the
 *  parameters of the transforms (e.g., the shift s and the power p of (x + s)^p): the
 *  nonlinear parameters are fitted once on the training rows, by the optimizer
 *  `OPTIMIZER_NL` (Variable Projection by default, @see `NARX_SR_D.rescale`), and are
 *  then held fixed, while the linear coefficients are (re)fitted by the regression of
 *  `ARX_D` on every training window.
 *  @see `MakeMatrix4TS` for hyper-parameter specifications.
 *  @param x        the data/input matrix (lagged columns of y and xe) @see `NARX_SR_D.apply`
 *  @param y        the response/output matrix (column per horizon) (time series data)
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 *  @param w_nl     the nonlinear parameters: two per transform (shift, scale/power), in the
 *                      order endogenous transforms, then the transforms of each exogenous variable
 *  @param scaled   whether the regression standardizes the columns of x (training mean and
 *                      standard deviation) instead of only centering them (defaults to false)
 *  @param n_free   the number of nonlinear parameters that were FITTED (they take degrees of
 *                      freedom like the coefficients, @see `mod_resetDF`); -1 => all of w_nl
 */
class NARX_SR_D (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String],
                 tRng: Range = null, hparam: HyperParameter = AR.hp,
                 bakcast: Boolean = false,
                 tForms: TransformMap = Map ("tForm_y" -> null), w_nl: VectorD = new VectorD (0),
                 scaled: Boolean = false, n_free: Int = -1)
      extends ARX_D (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms, scaled):

    private val debug = debugf ("NARX_SR_D", true)                          // debug function

    _modelName = s"NARX_SR_D($p, $q, $n_exo)"

    debug ("init", s"$modelName with $n_exo exogenous variables and additional trend spec = $spec, x.dims = ${x.dims}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the nonlinear parameters of the transforms (two per transform).
     */
    def parameter_nl: VectorD = w_nl

// Note: the fitted nonlinear parameters count as degrees of freedom taken by the model, so the
//       penalized measures (R^2 bar, AIC, sMAPE-IC, the FPE-form sMAPE of the selection score J)
//       charge a free parameter as they charge a coefficient

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of fitted nonlinear parameters.
     */
    def nFree: Int = if n_free < 0 then w_nl.dim else n_free

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the Degrees of Freedom (DF): the k slopes (@see `Forecaster_D.mod_resetDF`)
     *  plus the fitted nonlinear parameters.
     *  @param size  the size of dataset (full, train, or test) after skipped rows
     */
    override def mod_resetDF (size: Int): Unit = resetDF_k (size, parameter.size + nFree)

end NARX_SR_D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NARX_SR_D` companion object provides factory methods for the `NARX_SR_D` class.
 *  In `apply` and `rescale` the transforms are given by TYPE (`TransformT`): each type
 *  carries the default values of its two parameters (shift, scale/power) and their
 *  bounds (@see `TransformT.wlu`).  In `fromTerms` the model comes from the terms of a
 *  formula whose free parameters carry their own start values and bounds
 *  (@see `NARX_Formula`, `Par`).
 */
object NARX_SR_D:

    private val debug = debugf ("NARX_SR_D", true)                          // debug function
    private val flaw  = flawf ("NARX_SR_D")                                 // flaw function
    private val BIG   = 1e9                                                 // a bound this large means no bound

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `NARX_SR_D` object by building an input matrix xy and then calling the
     *  `NARX_SR_D` constructor.  The data are NOT rescaled (@see `rescale`).  With
     *  tune = true the nonlinear parameters of the transforms are fitted by the optimizer
     *  `OPTIMIZER_NL` (@see `optimize`) on the training rows; otherwise they keep their
     *  default values (`TransformT.wlu`), which gives the `ARX_SR_D` model with those
     *  transforms.  A transform that is not finite on some row (e.g., a root of a
     *  negative value) is reported by `flaw`.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transform types to be used for the endogenous variable
     *  @param fExo     the array containing the sets of transform types to be used for the
     *                      exogenous variables (null => `Pow` for each of them)
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tune     whether to fit the nonlinear parameters (@see `optimize`)
     *                      (false => keep their default values)
     *  @param trSize   the number of leading rows used for fitting the nonlinear parameters
     *                      (defaults to -1 => `Model.trSize (y.dim)`)
     *  @param scaled   whether the regression (and the fit of the nonlinear parameters)
     *                      standardizes the columns of x, so that the ridge penalty does not
     *                      depend on the units of a column (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = AR.hp,
               fEndo: LSET [TransformT] = LSET (Root),
               fExo: Array [LSET [TransformT]] = null,
               bakcast: Boolean = false,
               tune: Boolean = true, trSize: Int = -1, scaled: Boolean = false): NARX_SR_D =

        val xe_bfill = if xe.dim2 > 0 && hparam("px").toInt > 0 then backfill (xe) else null
        make (xe_bfill, xe.dim2, y, hh, fname_, tRng, hparam, fEndo, exoSets (fExo, xe.dim2), bakcast,
              null, tune, trSize, scaled)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `NARX_SR_D` object by building an input matrix xy and then calling the
     *  `NARX_SR_D` constructor, with rescaling of endogenous and exogenous variable values
     *  and fitting of the nonlinear parameters of the transforms.
     *  Rescales (with the statistics of the training rows), then transforms, then lags.
     *  With tune = true the nonlinear parameters are fitted by the optimizer `OPTIMIZER_NL`
     *  (@see `optimize`) on the training rows; otherwise they keep their default values.
     *  The default min-max range lu = [1, 5] keeps the rescaled training values positive,
     *  as the power, root and log transforms require; a transform that is not finite on
     *  some row of the series (e.g., a testing value far below the training minimum) is
     *  reported by `flaw`.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transform types to be used for the endogenous variable
     *  @param fExo     the array containing the sets of transform types to be used for the
     *                      exogenous variables (null => `Pow` for each of them)
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormT   the transform type for rescaling y and xe (defaults to `MinMax`)
     *  @param lu       the range [l, u] of the min-max rescaling (defaults to [1, 5])
     *  @param tune     whether to fit the nonlinear parameters (@see `optimize`)
     *                      (false => keep their default values)
     *  @param trSize   the number of leading rows used for rescaling and for fitting the
     *                      nonlinear parameters (defaults to -1 => `Model.trSize (y.dim)`)
     *  @param scaled   whether the regression (and the fit of the nonlinear parameters)
     *                      standardizes the columns of x, so that the ridge penalty does not
     *                      depend on the units of a column (defaults to false)
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = AR.hp,
                 fEndo: LSET [TransformT] = LSET (Root),
                 fExo: Array [LSET [TransformT]] = null,
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax, lu: VectorD = VectorD (1, 5),
                 tune: Boolean = true, trSize: Int = -1, scaled: Boolean = false): NARX_SR_D =

        val tr_size = if trSize > 0 then trSize else Model.trSize (y.dim)

        // rescale y: use the statistics of the training rows for both In-sample and TnT
        val tForm_y = tFormT.form (y(0 until tr_size))
        tForm_y.setLU (lu)
        val y_scl   = tForm_y.f (y)

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 && hparam("px").toInt > 0 then
            xe_bfill = backfill (xe)                                        // backfill each exogenous variable
            val tForm_exo = tFormT.form (xe_bfill(0 until tr_size))
            tForm_exo.setLU (lu)
            xe_bfill = tForm_exo.f (xe_bfill)                               // rescale the backfilled exogenous variables

        make (xe_bfill, xe.dim2, y_scl, hh, fname_, tRng, hparam, fEndo, exoSets (fExo, xe.dim2), bakcast,
              tForm_y, tune, tr_size, scaled)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `NARX_SR_D` object from (rescaled or not) data: set the nonlinear
     *  parameters (fitted on the training rows, or the defaults), build the input matrix
     *  with the transforms at those parameters and call the constructor
     *  (the common part of `apply` and `rescale`).
     *  @param xe_bfill  the matrix of backfilled exogenous variable values (null => none)
     *  @param n_exo     the number of exogenous variables
     *  @param y         the endogenous/response vector (rescaled when tForm_y is given)
     *  @param hh        the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_    the feature/variable names
     *  @param tRng      the time range, if relevant (time index may suffice)
     *  @param hparam    the hyper-parameters
     *  @param fEndo     the set of transform types to be used for the endogenous variable
     *  @param fExo      the array containing the sets of transform types for the exogenous variables
     *  @param bakcast   whether a backcasted value is prepended to the time series
     *  @param tForm_y   the transform that rescaled y (null => none), kept for inverting forecasts
     *  @param tune      whether to fit the nonlinear parameters (false => keep their default values)
     *  @param trSize    the number of training rows (-1 => `Model.trSize (y.dim)`)
     *  @param scaled    whether the regression standardizes the columns of x
     */
    private def make (xe_bfill: MatrixD, n_exo: Int, y: VectorD, hh: Int, fname_ : Array [String], tRng: Range,
                      hparam: HyperParameter, fEndo: LSET [TransformT], fExo: Array [LSET [TransformT]],
                      bakcast: Boolean, tForm_y: Transform, tune: Boolean, trSize: Int, scaled: Boolean): NARX_SR_D =

        // set the nonlinear parameters: fitted on the training rows, or the defaults
        val w_nl = if tune then optimize (xe_bfill, y, hh, hparam, fEndo, fExo, bakcast, trSize, scaled)
                   else initializeW (fEndo, fExo)._1
        debug ("make", s"nonlinear parameters (shift, scale/power per transform) w_nl = $w_nl")

        val (fEn, fEx) = getTransforms (w_nl, fEndo, fExo)
        val tForms: TransformMap = Map ("tForm_y" -> tForm_y)
        val xy     = buildMatrix (xe_bfill, y, hparam, fEn, fEx, bakcast)
        val fname  = if fname_ == null then formNames (n_exo, hparam, fEndo.size, fExo.map (_.size)) else fname_
        for j <- xy.indices2 if xy(?, j).exists (v => v.isNaN || v.isInfinite) do
            flaw ("make", s"column ${fname(j)} is not finite: a transform is outside its domain on some row")
        val yy     = makeMatrix4Y (y, hh, bakcast)
        val n_free = if tune then tunable (fEndo, fExo).size else 0         // the fitted parameters
        new NARX_SR_D (xy, yy, hh, n_exo, fname, tRng, hparam, bakcast, tForms, w_nl, scaled, n_free)
    end make

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the spec + p columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  When cross = true, additional cross terms will be added.  Columns produced
     *  by transformations will be added as well.
     *  @param xe_bfill  the matrix of exogenous variable values (null => none)
     *  @param y         the endogenous/response vector (main time series data)
     *  @param hp_       the hyper-parameters
     *  @param fEndo     the transformation functions to apply on the endogenous variable
     *  @param fExo      the transformation functions to apply on the exogenous variables
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     */
    def buildMatrix (xe_bfill: MatrixD, y: VectorD, hp_ : HyperParameter,
                     fEndo: Array [Transform], fExo: Array [Array [Transform]], bakcast: Boolean): MatrixD =

        val (p, q, spec, cross) = (hp_("p").toInt, hp_("px").toInt, hp_("spec").toInt, hp_("crx").toInt == 1)

        // apply transformations fEndo to the endogenous variable and add these columns to x_endo
        var x_endo = MatrixD (y).ᵀ                                         // make a matrix out of vector y
        for tr <- fEndo do x_endo = x_endo :^+ tr.f(y)                     // add each transformation of the endogenous variable

        // make matrix xy for trend terms and lagged terms of the endogenous variable
        var xy = makeMatrix4T (y, spec, bakcast) ++^                       // trend terms
                 makeMatrix4L (x_endo, p, bakcast)                         // lagged linear terms

        // apply transformations fExo to the exogenous variables and add these columns to x_exo
        if xe_bfill != null && q > 0 then
            var x_exo = new MatrixD (xe_bfill.dim, 0)
            for j <- xe_bfill.indices2 do
                val xe_j = xe_bfill(?, j)                                  // extract the (j+1)th exogenous variable
                x_exo = x_exo :^+ xe_j                                     // add the exogenous variable
                if j < fExo.length then
                    for tr <- fExo(j) do x_exo = x_exo :^+ tr.f(xe_j)      // add each transformation of the exogenous variable

            // add cross terms of the endogenous and exogenous variables
            if cross then x_exo = x_exo ++^ y *~: xe_bfill                 // element-wise multiplication of vector y and matrix xe
            xy = xy ++^ makeMatrix4L (x_exo, q, bakcast)                   // add lagged exogenous terms to xy

        xy                                                                 // return the built matrix xy
    end buildMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit the nonlinear parameters of the transforms with the optimizer `OPTIMIZER_NL`
     *  (@see `OptimizerNL`), returning ALL the nonlinear parameters (two per transform;
     *  the ones that are not tuned keep their default values).  The model is separable:
     *  for given nonlinear parameters the coefficients of all horizons are the solution
     *  of a linear (ridge) least squares problem, so `VarPro` and `LBFGS_B` search only
     *  the nonlinear parameters, within their bounds, while `Joint` searches them
     *  together with the coefficients; the nonlinear parameters are shared by the hh
     *  horizons.
     *  The fit uses the rows whose hh targets all lie in the training rows (0 until
     *  trSize - hh + 1) and the centering, scaling and ridge penalty of the model's
     *  regression.  Without scaling the penalty depends on the units of a column, which
     *  the nonlinear parameters change (a larger power gives a larger column, hence a
     *  smaller coefficient and a smaller penalty); scaled = true removes that effect.
     *  @param xe_bfill  the matrix of exogenous variable values (null => none)
     *  @param y         the endogenous/response vector (main time series data)
     *  @param hh        the maximum forecasting horizon (h = 1 to hh)
     *  @param hparam    the hyper-parameters
     *  @param fEndo     the set of transform types to be used for the endogenous variable
     *  @param fExo      the array containing the sets of transform types for the exogenous variables
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     *  @param trSize    the number of training rows (defaults to -1 => `Model.trSize (y.dim)`)
     *  @param scaled    whether the regression standardizes the columns of x (defaults to false)
     */
    def optimize (xe_bfill: MatrixD, y: VectorD, hh: Int, hparam: HyperParameter = AR.hp,
                  fEndo: LSET [TransformT] = LSET (Root),
                  fExo: Array [LSET [TransformT]] = Array (),
                  bakcast: Boolean = false, trSize: Int = -1, scaled: Boolean = false): VectorD =

        val (w0, l, u) = initializeW (fEndo, fExo)
        val free       = tunable (fEndo, fExo)                             // indices of the parameters to tune

        if free.isEmpty then w0                                            // nothing to tune
        else
            val tr_size = if trSize > 0 then trSize else Model.trSize (y.dim)
            val fitRng  = 0 until tr_size - hh + 1                         // rows whose targets lie in the training rows
            val yy      = makeMatrix4Y (y, hh, bakcast)(fitRng)
            val lb      = VectorD (free.map (i => if l(i) <= -BIG then Double.NegativeInfinity else l(i)))
            val ub      = VectorD (free.map (i => if u(i) >=  BIG then Double.PositiveInfinity else u(i)))

            val full = (a: VectorD) =>                                     // all parameters, given the tuned ones
                val w = w0.copy
                for i <- free.indices do w(free(i)) = a(i)
                w

            val basis = (a: VectorD) =>                                    // the input matrix for the fitting rows
                val (fEn, fEx) = getTransforms (full (a), fEndo, fExo)
                buildMatrix (xe_bfill, y, hparam, fEn, fEx, bakcast)(fitRng)

            debug ("optimize", s"fit on rows $fitRng the parameters ${free.mkString ("(", ", ", ")")}")
            full (fitNL (basis, yy, VectorD (free.map (w0(_))), lb, ub, ridge (hparam), scaled))
        end if
    end optimize

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `NARX_SR_D` object from the terms of a formula (@see `NARX_Formula`,
     *  `ARX_D.fromTerms`), with one column per term, whose free nonlinear parameters
     *  (@see `Par`) are fitted by `OPTIMIZER_NL` on the rows whose hh targets all lie
     *  in the training rows (0 until trSize - hh + 1), with the centering, scaling and
     *  ridge penalty of the model's regression; a parameter that the linear fit
     *  already covers (@see `NARX_Formula.redundant`) is reported and kept at its
     *  start value, as is one with equal bounds.  The parameters are then held at the
     *  fitted values (the term labels, hence the feature names, show them in the
     *  plain grammar) and the linear coefficients are (re)fitted by `ARX_D`'s
     *  regression on every training window.  Every base column is transformed on its
     *  training slice when tFormT is given, before the terms are evaluated (the checks,
     *  transforms, columns, names and lag hyper-parameters are those of `ARX_D.fromTerms`,
     *  shared through `ARX_Formula.prepare`, `matrix`, `names` and `lagHP`).
     *  The matrix has NO intercept column (the regression centers), so feature
     *  selection on it must use first = 0 (@see `Forecaster_D.selectFeatures`).
     *  @param resp     the response/endogenous variable
     *  @param preds    the predictor terms from the right hand side of the formula
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names (defaults to the terms' labels)
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters (defaults to `AR.hp`; never modified here)
     *  @param bakcast  whether a backcasted value is prepended (not supported, must be false)
     *  @param tFormT   the transform type applied to every base column (null => none)
     *  @param trSize   the number of leading rows the base-column transforms and the
     *                      nonlinear parameters are fit on (-1 => `Model.trSize (y.dim)`)
     *  @param scaled   whether the regression (and the fit of the nonlinear parameters)
     *                      standardizes the term columns (defaults to false)
     *  @param dataset  the map of named data columns (given)
     */
    def fromTerms (resp: FormulaVar, preds: ArrayBuffer [Term], hh: Int,
                   fname_ : Array [String] = null, tRng: Range = null,
                   hparam: HyperParameter = AR.hp, bakcast: Boolean = false,
                   tFormT: TransformT = null, trSize: Int = -1, scaled: Boolean = false)
                   (using dataset: Map [String, VectorD]): NARX_SR_D =

        val td = ARX_Formula.prepare (resp, preds, tFormT, trSize, bakcast)   // checks and base-column transforms
        require (td.trSize >= hh, s"fromTerms: trSize ${td.trSize} < hh = $hh: no row has all its targets in the training rows")

        // the free nonlinear parameters: all, minus the redundant and the fixed ones
        val pars = NARX_Formula.params (preds)
        pars.foreach (_.reset ())
        NARX_Formula.resolve (preds, td.data, td.trSize)                   // defaults from the training rows, if unset
        val redundant = preds.flatMap (NARX_Formula.redundant)
        for p <- redundant do flaw ("fromTerms", s"parameter $p is covered by the linear fit: kept at its start value")
        val free = pars.filter (p => p.free && ! redundant.exists (_ eq p))

        if free.nonEmpty then
            val fitRng = 0 until td.trSize - hh + 1                        // rows whose targets lie in the training rows
            val yy     = makeMatrix4Y (td.y, hh, bakcast)(fitRng)
            val basis  = (a: VectorD) =>
                for i <- free.indices do free(i).update (a(i))
                ARX_Formula.matrix (preds, td.data, check = false)(fitRng)
            val a0 = VectorD (free.map (_.w0).toArray)
            val lb = VectorD (free.map (_.lb).toArray)
            val ub = VectorD (free.map (_.ub).toArray)
            debug ("fromTerms", s"fit on rows $fitRng the parameters ${free.mkString ("(", ", ", ")")}")
            val a  = fitNL (basis, yy, a0, lb, ub, ridge (hparam), scaled)
            for i <- free.indices do free(i).update (a(i))
        end if
        pars.foreach (p => p.set (p.value))                                // from now on the labels show the values

        val xy    = ARX_Formula.matrix (preds, td.data)                    // one column per term, at the fitted values
        val yy    = makeMatrix4Y (td.y, hh, bakcast)
        val fname = if fname_ == null then ARX_Formula.names (preds) else fname_
        val w_nl  = VectorD (pars.map (_.value).toArray)
        val (hp_, n_exo) = ARX_Formula.lagHP (resp, preds, hparam)
        new NARX_SR_D (xy, yy, hh, n_exo, fname, tRng, hp_, bakcast, Map ("tForm_y" -> td.tForm_y), w_nl, scaled,
                       free.size)                                          // the fitted parameters take DF
    end fromTerms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit nonlinear parameters a, within [lb, ub], so that the ridge least squares
     *  criterion of the linear fit of yy on the basis (a) is minimal, with the
     *  optimizer `OPTIMIZER_NL` (@see `OptimizerNL`); return the fitted a.
     *  @param basis   the input matrix (for the fitting rows) as a function of a
     *  @param yy      the response matrix (one column per horizon) for the fitting rows
     *  @param a0      the starting point
     *  @param lb      the lower bounds (-infinity for none)
     *  @param ub      the upper bounds (+infinity for none)
     *  @param lambda  the ridge penalty of the model's regression
     *  @param scaled  whether the regression standardizes the columns
     */
    private def fitNL (basis: VectorD => MatrixD, yy: MatrixD, a0: VectorD, lb: VectorD, ub: VectorD,
                       lambda: Double, scaled: Boolean): VectorD =
        val varPro = new VarPro (a0.dim, basis, yy, lb, ub, lambda = lambda, scaled = scaled)
        val (a, steps) = OPTIMIZER_NL match
            case OptimizerNL.VarPro  => (varPro.optimize (a0), varPro.stepCounts)
            case OptimizerNL.LBFGS_B =>
                val lbfgs = new LBFGS_B (varPro.objective, exactLS = true, l_u = (lb, ub))
                (lbfgs.solve (a0)._2, Map ())                              // (objective value, argument) => argument
            case OptimizerNL.Joint   => joint (varPro, a0, lb, ub, lambda)
        debug ("fitNL", s"$OPTIMIZER_NL: a = $a, objective = ${varPro.objective (a)}, steps = $steps")
        a
    end fitNL

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit the nonlinear parameters a and the linear coefficients b of all the
     *  responses TOGETHER with `LevenbergMarquardt_TR` (the alternative to Variable
     *  Projection), on the residual [ y_j - X (a) b_j; -sqrt (lambda) b_j ] stacked
     *  over the responses j, with the centering and scaling of varPro's inner problem.
     *  The Jacobian is analytic for b (-X (a) and -sqrt (lambda) I) and by finite
     *  differences for a.  Return the nonlinear parameters and the step counts.
     *  @param varPro  the Variable Projection solver holding the problem (basis, responses)
     *  @param a0      the starting point for the nonlinear parameters
     *  @param lb      the lower bounds on the nonlinear parameters
     *  @param ub      the upper bounds on the nonlinear parameters
     *  @param lambda  the ridge penalty on the coefficients
     */
    private def joint (varPro: VarPro, a0: VectorD, lb: VectorD, ub: VectorD, lambda: Double):
                      (VectorD, Map [String, Int]) =
        val k   = a0.dim                                                   // number of nonlinear parameters
        val yc  = varPro.response                                          // the (centered) responses
        val b0  = varPro.coefficients (a0)                                 // coefficients at a0, n-by-ny
        val (m, ny, n) = (yc.dim, yc.dim2, b0.dim)
        val pen = lambda > 0.0
        val ma  = if pen then m + n else m                                 // residuals per response
        val slam = sqrt (lambda)

        val res = (w: VectorD) =>                                          // w = (a, b_1, ..., b_ny)
            val xs = varPro.system (w(0 until k))
            val r  = new VectorD (ny * ma)
            for j <- 0 until ny do
                val bj = w(k + j * n until k + (j + 1) * n)
                val rj = yc(?, j) - xs * bj
                for t <- 0 until m do r(j * ma + t) = rj(t)
                if pen then for i <- 0 until n do r(j * ma + m + i) = -slam * bj(i)
            r

        val jac = (w: VectorD) =>
            val a  = w(0 until k)
            val xs = varPro.system (a)
            val r0 = res (w)
            val jm = new MatrixD (ny * ma, k + n * ny)
            for i <- 0 until k do                                          // nonlinear parameters: finite differences
                val h  = 1e-7 * max (1.0, abs (a(i)))
                val s  = if a(i) + h <= ub(i) then h else -h
                val w2 = w.copy
                w2(i)  = a(i) + s
                jm(?, i) = (res (w2) - r0) / s
            for j <- 0 until ny; i <- 0 until n do                         // linear coefficients: analytic
                val c = k + j * n + i
                for t <- 0 until m do jm(j * ma + t, c) = -xs(t, i)
                if pen then jm(j * ma + m + i, c) = -slam
            jm

        val inf = Double.PositiveInfinity
        val lm  = new LevenbergMarquardt_TR (k + n * ny, lb ++ VectorD.fill (n * ny)(-inf),
                                             ub ++ VectorD.fill (n * ny)(inf), res, Some (jac))
        val w   = lm.optimize (a0 ++ b0.transpose.flatten)                 // b response by response
        (w(0 until k), lm.stepCounts)
    end joint

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the ridge penalty the model's regression will use: the one in hparam
     *  if present, the one of `RidgeRegression.hp` otherwise (@see `Forecaster_D`).
     *  @param hparam  the hyper-parameters
     */
    private def ridge (hparam: HyperParameter): Double =
        try hparam("lambda").toDouble
        catch case _: NoSuchElementException => RidgeRegression.hp("lambda").toDouble
    end ridge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sets of transform types for the n_exo exogenous variables: `Pow`
     *  for each when fExo is null, an empty set for a variable beyond fExo's length.
     *  @param fExo   the array containing the sets of transform types for the exogenous variables
     *  @param n_exo  the number of exogenous variables
     */
    private def exoSets (fExo: Array [LSET [TransformT]], n_exo: Int): Array [LSET [TransformT]] =
        if fExo == null then Array.fill (n_exo)(LSET (Pow))
        else Array.tabulate (n_exo)(j => if j < fExo.length then fExo(j) else LSET [TransformT] ())
    end exoSets

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return all the transform types in the order of the nonlinear parameters:
     *  endo's transforms, exo1's transforms, exo2's transforms, ...
     *  @param fEndo  the set of transform types to be used for the endogenous variable
     *  @param fExo   the array containing the sets of transform types for the exogenous variables
     */
    private def allForms (fEndo: LSET [TransformT], fExo: Array [LSET [TransformT]]): Array [TransformT] =
        (fEndo.toSeq ++ fExo.toSeq.flatMap (_.toSeq)).toArray
    end allForms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form vectors for the default nonlinear parameters and their bounds for the
     *  transforms: two per transform (shift, scale/power; a transform type without
     *  parameters, e.g., `Log1p`, contributes none).
     *  @param fEndo  the set of transform types to be used for the endogenous variable
     *  @param fExo   the array containing the sets of transform types for the exogenous variables
     */
    def initializeW (fEndo: LSET [TransformT], fExo: Array [LSET [TransformT]]):
                    (VectorD, VectorD, VectorD) =
        var w = new VectorD (0)
        var l = new VectorD (0)
        var u = new VectorD (0)
        for t <- allForms (fEndo, fExo) if t.wlu != null do
            w = w ++ t.wlu.w
            l = l ++ t.wlu.l
            u = u ++ t.wlu.u
        (w, l, u)
    end initializeW

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether parameter i (0 = shift, 1 = scale/power) of transform type t
     *  must be searched as a nonlinear parameter.  A parameter that only shifts or
     *  rescales the column is already covered by the intercept and the linear
     *  coefficient, so tuning it would make the search singular:
     *      log (a x + s) = log a + log (x + s/a)  => the scale a is a constant plus a shift,
     *      exp (a x + s) = e^s exp (a x)          => the shift s is a linear coefficient.
     *  The power of `PowR` is rounded to a rational, so it cannot be tuned smoothly.
     *  @param t  the transform type
     *  @param i  the parameter (0 = shift, 1 = scale/power)
     */
    private def nonlinear (t: TransformT, i: Int): Boolean =
        t match
        case Log        => i == 0
        case Exp | NExp => i == 1
        case PowR       => i == 0
        case _          => true
    end nonlinear

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the indices (into the vector of nonlinear parameters) of the parameters
     *  to tune: those whose bounds differ (lower < upper; equal bounds fix a parameter)
     *  and that are truly nonlinear (@see `nonlinear`).
     *  @param fEndo  the set of transform types to be used for the endogenous variable
     *  @param fExo   the array containing the sets of transform types for the exogenous variables
     */
    def tunable (fEndo: LSET [TransformT], fExo: Array [LSET [TransformT]]): Array [Int] =
        val idx = ArrayBuffer [Int] ()
        var k   = 0
        for t <- allForms (fEndo, fExo) if t.wlu != null do
            for i <- 0 until 2 do
                if t.wlu.l(i) < t.wlu.u(i) && nonlinear (t, i) then idx += k + i
            k += 2
        idx.toArray
    end tunable

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form arrays of transform objects using the vector of nonlinear parameters.
     *  @param w_nl   the vector of nonlinear parameters (two per transform)
     *  @param fEndo  the set of transform types to be used for the endogenous variable
     *  @param fExo   the array containing the sets of transform types for the exogenous variables
     */
    def getTransforms (w_nl: VectorD, fEndo: LSET [TransformT], fExo: Array [LSET [TransformT]]):
                      (Array [Transform], Array [Array [Transform]]) =
        var k = 0
        def make (t: TransformT): Transform =
            if t.wlu == null then t.form (null)                            // a transform without parameters
            else
                k += 2
                t.form (VectorD (w_nl(k-2), w_nl(k-1)))
        end make

        val listEndo = fEndo.toArray.map (make)
        val listExo  = fExo.map (_.toArray.map (make))
        (listEndo, listExo)
    end getTransforms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables
     *  @param n_fExArr  the number of functions used to map each exogenous variables
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int, n_fExArr: Array [Int]): Array [String] =

        val (spec, p, q, cross) = (hp_("spec").toInt, hp_("p").toInt, hp_("px").toInt, hp_("crx").toInt)
        val names = ArrayBuffer [String] ()
        for i <- 0 until n_fEn; j <- p to 1 by -1 do names += s"f$i(yl$j)"           // function lags endo terms

        // exogenous (match build order):
        for j <- 0 until n_exo do
            for k <- q to 1 by -1 do names += s"xe${j}l$k"                           // raw exogenous lags

            val n_fEx_j = if j < n_fExArr.length then n_fExArr(j) else 0             // transformations for this exo j
            for i <- 0 until n_fEx_j; k <- q to 1 by -1 do names += s"g$j,$i(xe${j}l$k)"
        end for

        if cross == 1 then
            for j <- 0 until n_exo; k <- q to 1 by -1 do names += s"xe${j}l$k*yl$k"  // lagged cross terms

        MakeMatrix4TS.formNames (spec, p) ++ names.toArray
    end formNames

end NARX_SR_D

import Example_LakeLevels.y
import AR.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest` main function tests the `NARX_SR_D` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  The data are not rescaled; the root transform of the y lags, (y + s)^r, has its
 *  shift s and root r fitted by `OPTIMIZER_NL` (`apply` with tune = true, the default).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_SR_DTest
 */
@main def nARX_SR_DTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = NARX_SR_D (EMPTY_EXO, y, hh)                              // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit
    println (s"nonlinear parameters (shift, root) of f0 (y) = ${mod.parameter_nl}")

end nARX_SR_DTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest2` main function tests the `NARX_SR_D` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_SR_DTest2
 */
@main def nARX_SR_DTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = NARX_SR_D (EMPTY_EXO, y, hh)                              // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validation
    println (s"nonlinear parameters (shift, root) of f0 (y) = ${mod.parameter_nl}")

end nARX_SR_DTest2

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest3` main function tests the `NARX_SR_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  The data are not rescaled.  The model is built twice: with the DEFAULT nonlinear
 *  parameters of its transforms (tune = false, the `ARX_SR_D` model with these
 *  transforms) and with the parameters fitted by `OPTIMIZER_NL` (`apply`'s default).
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_SR_DTest3
 */
@main def nARX_SR_DTest3 (): Unit =

    val exo_vars  = Array ("icu_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

    val xe = xxe(0 until 116)                                           // clip the flat end
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("crx") = 1                                                       // 1 => add cross terms

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // endo lags
        hp("px")   = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3

        banner ("NARX_SR_D with the DEFAULT nonlinear parameters")
        val mod0 = NARX_SR_D (xe, y, hh, tune = false)                  // create model for time series data
        mod0.inSample_Test ()                                           // In-sample Testing
        println (mod0.summary ())                                       // statistical summary of fit

        banner (s"NARX_SR_D with the nonlinear parameters fitted by $OPTIMIZER_NL")
        val mod = NARX_SR_D (xe, y, hh)                                 // create model for time series data
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit

        banner ("nonlinear parameters: (shift, root) of f0 (y); (shift, power) of g0,0 (icu_patients)")
        println (s"default: ${mod0.parameter_nl}")
        println (s"fitted:  ${mod.parameter_nl}")
    end for

end nARX_SR_DTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest4` main function tests the `NARX_SR_D` class on real data:
 *  Forecasting COVID-19 using Train and Test (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  The data are not rescaled.  As in `nARX_SR_DTest3`, the model is built with the
 *  DEFAULT nonlinear parameters (tune = false) and with the fitted ones.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_SR_DTest4
 */
@main def nARX_SR_DTest4 (): Unit =

    val exo_vars  = Array ("icu_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

    val xe = xxe(0 until 116)                                           // clip the flat end
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("crx") = 0                                                       // 1 => add cross terms

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // endo lags
        hp("px")   = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3

        banner ("NARX_SR_D with the DEFAULT nonlinear parameters")
        val mod0 = NARX_SR_D (xe, y, hh, tune = false)                  // create model for time series data
        mod0.tnT_Test (showYf = false)                                  // TnT with Rolling Validation

        banner (s"NARX_SR_D with the nonlinear parameters fitted by $OPTIMIZER_NL")
        val mod = NARX_SR_D (xe, y, hh)                                 // create model for time series data
        mod.tnT_Test (showYf = false)                                   // TnT with Rolling Validation

        banner ("nonlinear parameters: (shift, root) of f0 (y); (shift, power) of g0,0 (icu_patients)")
        println (s"default: ${mod0.parameter_nl}")
        println (s"fitted:  ${mod.parameter_nl}")
    end for

end nARX_SR_DTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest5` main function fits the linear and nonlinear parameters
 *  and tests the `NARX_SR_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing and Train and Test (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  The same rescaled model is tested twice -- with the DEFAULT nonlinear parameters
 *  of its root transforms (tune = false) and with the ones fitted by `OPTIMIZER_NL`
 *  on the training rows (tune = true) -- so the two TnT QoF blocks show
 *  whether tuning the nonlinear parameters helps out of sample.  Feature selection
 *  at horizon 1 then runs on the fitted model.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_SR_DTest5
 */
@main def nARX_SR_DTest5 (): Unit =

    val exo_vars  = Array ("icu_patients", "positive_rate")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

    val xe = xxe(0 until 116)                                       // clip the flat end
    val y  = yy(0 until 116)                                        // clip the flat end
    val hh = 6                                                      // maximum forecasting horizon
    hp("crx") = 0                                                   // 1 => add cross terms
    RidgeRegression.hp("lambda") = 6.0
    RidgeRegression.hp("factorization") = "Fac_Cholesky"

    hp("p")    = 6                                                  // endo lags
    hp("px")   = 4                                                  // exo lags
    hp("spec") = 1                                                  // trend specification: 0, 1, 2, 3

    val fEndo: LSET [TransformT] = LSET (Root)                      // transform types for the endogenous variable
    val fExo: Array [LSET [TransformT]] = Array (LSET (Root), LSET (Root))   // and for each exogenous variable
                                                                    // (must be exo_vars.length == fExo.length)

    banner ("NARX_SR_D with the DEFAULT nonlinear parameters")
    val mod0 = NARX_SR_D.rescale (xe, y, hh, fEndo = fEndo, fExo = fExo, tune = false)
    mod0.inSample_Test ()                                           // In-sample Testing
    mod0.tnT_Test (rc = 2, showYf = false)                          // TnT with Rolling Validation

    banner (s"NARX_SR_D with the nonlinear parameters fitted by $OPTIMIZER_NL")
    val mod = NARX_SR_D.rescale (xe, y, hh, fEndo = fEndo, fExo = fExo)
    mod.inSample_Test ()                                            // In-sample Testing
    mod.tnT_Test (rc = 2, showYf = false)                           // TnT with Rolling Validation

    banner ("nonlinear parameters (shift, root) of f0 (y), g0,0 (icu_patients), g1,0 (positive_rate)")
    println (s"default: ${mod0.parameter_nl}")
    println (s"fitted:  ${mod.parameter_nl}")

    banner ("Feature Selection (Backward, horizon 1 decides) on the model with the fitted nonlinear parameters")
    FeatureSelection.fullset_FS = false                             // TnT test: select on the training rows
    val (colsArr, rSqArr) = mod.selectFeatures (SelectionTech.Backward, "none", h = 1)
    mod.reportSelection (colsArr, rSqArr, inSample = false, rc = 2) // TnT of the selected multi-horizon model

end nARX_SR_DTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest6` main function fits the linear and nonlinear parameters
 *  and tests the `NARX_SR_D` class on real data, with a POWER term for the
 *  endogenous variable, (y + s)^p, and a LOG term for each exogenous variable,
 *  log (xe + s): the coefficients of the terms are linear parameters, while the
 *  shifts s and the power p are nonlinear parameters fitted by the optimizer `OPTIMIZER_NL`.
 *  Forecasting COVID-19 using In-Sample Testing and Train and Test (TnT) with Rolling
 *  Validation, with the default (tune = false) and the fitted (tune = true) nonlinear parameters.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_SR_DTest6
 */
@main def nARX_SR_DTest6 (): Unit =

    val exo_vars  = Array ("icu_patients", "positive_rate")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

    val xe = xxe(0 until 116)                                       // clip the flat end
    val y  = yy(0 until 116)                                        // clip the flat end
    val hh = 6                                                      // maximum forecasting horizon
    hp("crx") = 0                                                   // 1 => add cross terms
    RidgeRegression.hp("lambda") = 6.0

    hp("p")    = 6                                                  // endo lags
    hp("px")   = 4                                                  // exo lags
    hp("spec") = 1                                                  // trend specification: 0, 1, 2, 3

    val fEndo: LSET [TransformT] = LSET (Pow)                       // (y + s)^p: shift s and power p are tuned
    val fExo: Array [LSET [TransformT]] = Array (LSET (Log), LSET (Log))     // log (xe + s): shift s is tuned

    banner ("NARX_SR_D with the DEFAULT nonlinear parameters")
    val mod0 = NARX_SR_D.rescale (xe, y, hh, fEndo = fEndo, fExo = fExo, tune = false)
    mod0.inSample_Test ()                                           // In-sample Testing
    mod0.tnT_Test (rc = 2, showYf = false)                          // TnT with Rolling Validation

    banner (s"NARX_SR_D with the nonlinear parameters fitted by $OPTIMIZER_NL")
    val mod = NARX_SR_D.rescale (xe, y, hh, fEndo = fEndo, fExo = fExo)
    mod.inSample_Test ()                                            // In-sample Testing
    mod.tnT_Test (rc = 2, showYf = false)                           // TnT with Rolling Validation

    banner ("nonlinear parameters: (shift, power) of f0 (y); (shift, scale) of g0,0 (icu_patients), g1,0 (positive_rate)")
    println (s"default: ${mod0.parameter_nl}")
    println (s"fitted:  ${mod.parameter_nl}")

end nARX_SR_DTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest7` main function repeats `nARX_SR_DTest6` (a POWER term for the
 *  endogenous variable and a LOG term for each exogenous variable) with a UNIT-FREE
 *  ridge penalty: scaled = true standardizes the columns in the regression and in the
 *  fit of the nonlinear parameters.  Without it a nonlinear parameter can lower the
 *  penalty just by making its column larger; with it only the shape of the column counts.
 *  Forecasting COVID-19 using In-Sample Testing and Train and Test (TnT) with Rolling
 *  Validation, with the default (tune = false) and the fitted (tune = true) nonlinear parameters.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_SR_DTest7
 */
@main def nARX_SR_DTest7 (): Unit =

    val exo_vars  = Array ("icu_patients", "positive_rate")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

    val xe = xxe(0 until 116)                                       // clip the flat end
    val y  = yy(0 until 116)                                        // clip the flat end
    val hh = 6                                                      // maximum forecasting horizon
    hp("crx") = 0                                                   // 1 => add cross terms
    RidgeRegression.hp("lambda") = 1.0                              // penalty on standardized columns

    hp("p")    = 6                                                  // endo lags
    hp("px")   = 4                                                  // exo lags
    hp("spec") = 1                                                  // trend specification: 0, 1, 2, 3

    val fEndo: LSET [TransformT] = LSET (Pow)                       // (y + s)^p: shift s and power p are tuned
    val fExo: Array [LSET [TransformT]] = Array (LSET (Log), LSET (Log))     // log (xe + s): shift s is tuned

    banner ("NARX_SR_D (scaled) with the DEFAULT nonlinear parameters")
    val mod0 = NARX_SR_D.rescale (xe, y, hh, fEndo = fEndo, fExo = fExo, tune = false, scaled = true)
    mod0.inSample_Test ()                                           // In-sample Testing
    mod0.tnT_Test (rc = 2, showYf = false)                          // TnT with Rolling Validation

    banner (s"NARX_SR_D (scaled) with the nonlinear parameters fitted by $OPTIMIZER_NL")
    val mod = NARX_SR_D.rescale (xe, y, hh, fEndo = fEndo, fExo = fExo, scaled = true)
    mod.inSample_Test ()                                            // In-sample Testing
    mod.tnT_Test (rc = 2, showYf = false)                           // TnT with Rolling Validation

    banner ("nonlinear parameters: (shift, power) of f0 (y); (shift, scale) of g0,0 (icu_patients), g1,0 (positive_rate)")
    println (s"default: ${mod0.parameter_nl}")
    println (s"fitted:  ${mod.parameter_nl}")

end nARX_SR_DTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest8` main function compares the NONLINEAR OPTIMIZERS of `OptimizerNL`
 *  on the problem of `nARX_SR_DTest6` (a POWER term for the endogenous variable and a
 *  LOG term for each exogenous variable, lambda = 6): for each optimizer it fits the
 *  nonlinear parameters, reports them with the fitting time (the objective and the
 *  steps are in the debug line of `optimize`) and runs Train and Test (TnT) with
 *  Rolling Validation.  The three optimizers minimize the same objective, so they
 *  should agree on the parameters when they converge; they differ in cost.
 *  > runMain scalation.modeling.forecasting.nonlinear.nARX_SR_DTest8
 */
@main def nARX_SR_DTest8 (): Unit =

    val exo_vars  = Array ("icu_patients", "positive_rate")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

    val xe = xxe(0 until 116)                                       // clip the flat end
    val y  = yy(0 until 116)                                        // clip the flat end
    val hh = 6                                                      // maximum forecasting horizon
    hp("crx") = 0                                                   // 1 => add cross terms
    RidgeRegression.hp("lambda") = 6.0

    hp("p")    = 6                                                  // endo lags
    hp("px")   = 4                                                  // exo lags
    hp("spec") = 1                                                  // trend specification: 0, 1, 2, 3

    val fEndo: LSET [TransformT] = LSET (Pow)                       // (y + s)^p: shift s and power p are tuned
    val fExo: Array [LSET [TransformT]] = Array (LSET (Log), LSET (Log))     // log (xe + s): shift s is tuned

    val w_nl = new Array [VectorD] (OptimizerNL.values.length)
    val secs = new Array [Double] (OptimizerNL.values.length)
    for opt <- OptimizerNL.values do
        OPTIMIZER_NL = opt
        banner (s"NARX_SR_D with the nonlinear parameters fitted by $opt")
        val (mod, ms) = timed { NARX_SR_D.rescale (xe, y, hh, fEndo = fEndo, fExo = fExo) }
        w_nl(opt.ordinal) = mod.parameter_nl
        secs(opt.ordinal) = ms / 1000.0
        mod.tnT_Test (rc = 2, showYf = false)                       // TnT with Rolling Validation
    end for
    OPTIMIZER_NL = OptimizerNL.VarPro                               // restore the default

    banner ("nonlinear parameters: (shift, power) of f0 (y); (shift, scale) of g0,0 (icu_patients), g1,0 (positive_rate)")
    for opt <- OptimizerNL.values do
        println (s"$opt: fitted ${w_nl(opt.ordinal)}, fitting time ${secs(opt.ordinal)} s")

end nARX_SR_DTest8
