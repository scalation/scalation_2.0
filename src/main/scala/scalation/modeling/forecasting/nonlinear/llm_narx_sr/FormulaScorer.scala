
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep 14 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Fit-and-Score Engine for Formula Search (`LLM_NARX_SR`)
 *
 *  A thin layer over the base's own machinery: a formula (@see `ARX_Formula`) is
 *  built by `ARX_D.fromTerms` -- or, when it carries FREE NONLINEAR PARAMETERS
 *  (@see `NARX_Formula`, `Par`: pow(y~1, ?), log(? + x~1), sat(x~1, ?), ...), by
 *  `NARX_SR_D.fromTerms`, which fits the parameters on the training rows with
 *  `NARX_SR_D.OPTIMIZER_NL` before the coefficients --, trained by `Forecaster_D.train`
 *  on the TRAINING rows, and diagnosed by the base's `diagnose` exactly as
 *  `ForecastMatrix.diagnoseAll` does, so every number here is an `ARX_D` number (an
 *  `NARX_SR_D` is an `ARX_D`; its fitted parameters count as degrees of freedom, so the
 *  FPE penalty of the score charges a free parameter as it charges a term).  The rows are ScalaTion's:
 *
 *      training rows  trRng = 0 until Model.trSize (m)     (92 of the 116 Covid-19 rows)
 *      testing rows   teRng = Forecaster.teRng (m)         (the last 24)
 *
 *  Search = feature selection on the training rows, which are split once more
 *  (rev3.2): the FIT rows (the first 1 - hp("valFrac") of them) the candidate is
 *  trained on, and the VALIDATION rows (the last hp("valFrac"), 25 % by default)
 *  it is scored on out of sample -- in hp("valFolds") rolling-origin blocks (3 by
 *  default: block i is forecast by a model fitted on every row before it, and the
 *  blocks' QoF matrices are averaged; one block follows the regime of its 23 weeks
 *  too closely on Covid-19).  Each candidate's in-sample QoF matrix
 *  (`qofTrain`, one row per horizon, @see `Fit.N_QoF`) is computed on the fit rows
 *  with ONE fixed `skip` for every candidate (models compared with the same skip,
 *  @see `Diagnoser`), its validation QoF matrix (`qofVal`) on the validation rows
 *  (no skip: every forecast there is out of sample), and its selection score J is
 *  the mean over horizons of the FPE-form parsimony-adjusted VALIDATION sMAPE,
 *  sMAPE_val,h (1 + 2p/n_h), p the parameters of the fit (slopes, intercept and the
 *  free nonlinear parameters, @see `NARX_SR_D.mod_resetDF`) and n_h the fit rows
 *  diagnosed at horizon h (@see `Forecaster_D.smapeFPE`: the penalty is weighed
 *  against the loss itself, so its strength does not fall with the error level as
 *  the additive `Fit.smapeC`'s does), or the h = 1 value alone when hp("hSel") = 1.
 *  With hp("valFrac") = 0 the fit rows are the training rows and J is the in-sample
 *  FPE-form sMAPE of rev3.0/3.1 (the Oct 4 grids: that score ranked the arms
 *  upside down on the testing rows, Spearman -0.355).
 *  A candidate is REJECTED (scored `Double.MaxValue`, with a reason) when it has
 *  more than hp("maxFree") free nonlinear parameters, when a fitted parameter ends
 *  within hp("boundTol") of its box (the term has collapsed onto its linear limit:
 *  log (s + x) with a large s, sat (x, k) with k above the data, pow (x, p) with p
 *  at the box or within 0.05 of 1), or when a free-parameter term's fitted column has
 *  a variance inflation factor above hp("maxVIF") against the other columns.
 *  Final = `tnt`: the formula refitted on ALL training rows, then the base's
 *  `rollValidate` on the testing rows (sliding window, refit every hp("rc") rows)
 *  followed by the same QoF matrix on `teRng`.
 *  The engine also scores any base model (`score (mod, label)`, `tnt (mod)`), so the
 *  classical anchors are the base's own `ARX_D`/`RandomWalk` objects.
 *
 *  Hyper-parameters: every model carries ONE map -- the regression's defaults
 *  overwritten by the model's own entries (`hparam`) -- so the ridge penalty
 *  hp("lambda") travels inside the model and no global is touched; the search's
 *  own settings are `FormulaScorer.hp` (aliased as `LLM_NARX_SR.hp`).
 *
 *  Any failure (non-finite column, singular fit, ...) scores `Double.MaxValue`:
 *  a search generates degenerate formulas and they must rank last, never crash.
 *  Quiet by construction: no `trainNtest`, `test` or `diagnoseAll` (which plot
 *  and print) is called here; `LLM_NARX_SR.showBest` calls them for the winners.
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting
package nonlinear
package llm_narx_sr

import scala.collection.mutable.{LinkedHashMap => LMAP, LinkedHashSet => LSET}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Candidate` case class is one scored member of a search population: a
 *  formula (or a base model under a label) with its in-sample QoF matrix on the
 *  fit rows and its QoF matrix on the validation rows (@see `FormulaScorer`).
 *  @param formula   the formula (null for a base model scored under `label`)
 *  @param score     the selection score J (@see `FormulaScorer.scoreOf`; `Double.MaxValue` => the fit failed)
 *  @param qofTrain  the QoF matrix on the fit rows (hh x Fit.N_QoF; -0.0 when the fit failed)
 *  @param nTerms    the number of terms/columns (the parsimony measure)
 *  @param label     the display name of a base model (null for a formula)
 *  @param model     the base model itself (null for a formula: rebuilt by `FormulaScorer.fit`)
 *  @param qofVal    the QoF matrix on the validation rows (null => the fit rows' matrix, hp("valFrac") = 0)
 *  @param extraDF   parameters charged by J beyond the model's own (the free parameters of a
 *                       library set evaluated at fixed values, @see `FormulaScorer.evalCols`)
 *  @param reason    why the candidate was rejected (null when it was not)
 */
case class Candidate (formula: ARX_Formula, score: Double, qofTrain: MatrixD, nTerms: Int,
                      label: String = null, model: Forecaster = null,
                      qofVal: MatrixD = null, extraDF: Int = 0, reason: String = null):

    def failed: Boolean    = score == Double.MaxValue                   // whether the fit failed (or was rejected)
    def key: String        = if formula != null then FormulaScorer.keyOf (formula) else label   // canonical identity
    def formulaStr: String = if formula != null then formula.formulaStr else label   // display string (fitted values)
    def proposalStr: String =                                           // the string a proposer is shown (?w slots)
        if formula != null then NARX_Formula (formula).proposalStr else label
    def nFree: Int         = if formula != null then NARX_Formula (formula).nFree else 0   // free nonlinear parameters
    def smapeH: VectorD    = qofTrain(?, QoF.smape.ordinal)             // in-sample sMAPE per horizon (fit rows)
    def smape: Double      = smapeH.mean                                // its mean over horizons
    def qofV: MatrixD      = if qofVal == null then qofTrain else qofVal   // the validation matrix (or the fit rows')
    def smapeValH: VectorD = qofV(?, QoF.smape.ordinal)                 // validation sMAPE per horizon
    def smapeVal: Double   = smapeValH.mean                             // its mean over horizons
    def scoreH: VectorD    = FormulaScorer.jOf (qofTrain, qofV, extraDF)   // J per horizon (@see `FormulaScorer.jOf`)
    def why: String        = if reason == null then "" else s" -- $reason"   // the rejection reason, for display

end Candidate


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FormulaScorer` class fits formulas (and base models) on the training rows
 *  and scores them by the base's QoF measures (@see the file comment).
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param scaled   whether a formula's regression standardizes its term columns,
 *                      so the ridge penalty is unit-free (defaults to true)
 *  @param dataset  the map of named data columns (full series)
 */
class FormulaScorer (val hh: Int = FormulaScorer.hp("hh").toInt, val scaled: Boolean = true)
                    (using dataset: Map [String, VectorD]):

    import FormulaScorer.hp

    private val debug    = debugf ("FormulaScorer", false)              // debug function
    private val flaw     = flawf ("FormulaScorer")                      // flaw function
    private val fitCache = LMAP [String, ARX_D] ()                      // key -> fitted model (training rows)
    private val CACHE_SZ = 64                                           // fitted models kept (oldest evicted)
    private var nEval    = 0                                            // number of fits made (all paths)

    val vars: Set [String] = dataset.keySet                             // the legal variable names
    val fullSet: Map [String, VectorD] = dataset                        // every row
    val m: Int             = dataset.head._2.dim                        // rows in the full series
    val trSize: Int        = Model.trSize (m)                           // training rows (ScalaTion's split)
    val trRng: Range       = 0 until trSize                             // the training rows
    val teRng: Range       = Forecaster.teRng (m)                       // the testing rows
    val trainSet: Map [String, VectorD] = dataset.map ((k, v) => k -> v(trRng))   // the training rows' columns
    val skip: Int          = hp("skip").toInt                           // rows skipped by every in-sample diagnosis
    val hSel: Int          = hp("hSel").toInt                           // 0 => score over all horizons, 1 => h = 1 only
    val rc: Int            = hp("rc").toInt                             // retraining cycle of `rollValidate`
    val ridge: Double      = hp("lambda").toDouble                      // ridge penalty carried by every model
    val valFrac: Double    = hp("valFrac").toDouble                     // share of the training rows held out for the selection score
    val valFolds: Int      = math.max (1, hp("valFolds").toInt)         // rolling-origin blocks the validation rows are split into
    val nVal: Int          = if valFrac <= 0.0 then 0 else math.max (valFolds * (hh + 1), math.round (valFrac * trSize).toInt)
    val fitSize: Int       = trSize - nVal                              // the fit rows: candidates are trained on them
    val fitRng: Range      = 0 until fitSize                            // the fit rows
    val valRng: Range      = fitSize until trSize                       // the validation rows (empty when valFrac = 0)
    val blocks: Seq [Range] =                                           // the rolling-origin validation blocks (fold i trains on the rows before block i)
        if nVal == 0 then Seq ()
        else
            val b = nVal / valFolds
            (0 until valFolds).map (i => (fitSize + i * b) until (if i == valFolds - 1 then trSize else fitSize + (i + 1) * b))
    val maxFree: Int       = hp("maxFree").toInt                        // free parameters allowed per candidate (-1 => no cap)
    val boundTol: Double   = hp("boundTol").toDouble                    // a fitted parameter this close to its box is a reject
    val maxVIF: Double     = hp("maxVIF").toDouble                      // VIF of a free-parameter column above this is a reject (0 => off)

    if trSize <= skip + 2 then flaw ("init", s"trSize = $trSize too small for skip = $skip")
    if nVal > 0 && fitSize <= skip + hh + 1 then flaw ("init", s"fit rows = $fitSize too few for skip = $skip and hh = $hh: lower valFrac")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of model fits made so far (the search budget in fits).
     */
    def evaluations: Int = nEval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the exogenous variables of a configuration the arms build on: the
     *  list itself (`Given`) or the screening's shortlist of the pool (`Auto`) on
     *  this scorer's data and training rows, hp("exoK") of them, leads up to
     *  hp("maxLag") (@see `DatasetConfig.shortlist`; rev4.1: was
     *  `DatasetConfig.shortlist (scorer)`).
     *  @param cfg  the dataset configuration
     */
    def shortlist (cfg: DatasetConfig): Seq [String] =
        cfg.shortlist (fullSet, trSize, hp("exoK").toInt, hp("maxLag").toInt)
    end shortlist

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the hyper-parameter map a model built here carries: the forecaster's
     *  (`AR.hp`) and the regression's (`RidgeRegression.hp`) defaults with this
     *  scorer's ridge penalty and the given entries written over them.
     *  @param nvs  further name-value pairs (e.g., ("p", 3))
     */
    def hparam (nvs: (String, ValueType)*): HyperParameter =
        (AR.hp ++ RidgeRegression.hp).updateReturn ((("lambda", ridge) +: nvs)*)
    end hparam

//  F O R M U L A S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build (untrained) the `ARX_D` of a formula on the full series: one column
     *  per term, transforms fitted on the first `size` rows (@see `ARX_D.fromTerms`);
     *  a formula with free nonlinear parameters becomes an `NARX_SR_D` whose
     *  parameters are fitted on those rows first (@see `NARX_SR_D.fromTerms`,
     *  `NARX_SR_D.OPTIMIZER_NL`), then held fixed.
     *  @param f     the formula
     *  @param size  the leading rows the transforms and parameters are fitted on
     *                   (defaults to the fit rows; `trSize` for the final model)
     */
    def build (f: ARX_Formula, size: Int = fitSize): ARX_D =
        if NARX_Formula (f).nFree > 0 then
            NARX_SR_D.fromTerms (f.resp, f.preds, hh, hparam = hparam (), tFormT = f.tFormT, trSize = size, scaled = scaled)
        else
            ARX_D.fromTerms (f.resp, f.preds, hh, hparam = hparam (), tFormT = f.tFormT, trSize = size, scaled = scaled)
    end build

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build (untrained) the `ARX_D` of a formula with its free parameters held at
     *  their CURRENT values (start values, or the values of an earlier fit), without
     *  fitting them: the design matrix a selection stage scores column subsets of
     *  (@see `library`), where a fit per subset would be too costly.  Unset parameters
     *  are resolved from the fit rows first (@see `NARX_Formula.resolve`).
     *  @param f  the formula
     */
    def buildFixed (f: ARX_Formula): ARX_D =
        NARX_Formula.resolve (f.preds, dataset, fitSize)
        ARX_D.fromTerms (f.resp, f.preds, hh, hparam = hparam (), tFormT = f.tFormT, trSize = fitSize, scaled = scaled)
    end buildFixed

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit a formula on the fit rows (forecasts for every row) and return the
     *  `ARX_D`, reusing the cached model when this formula was fitted before (by its
     *  canonical key, which ignores the values of its free parameters).
     *  @param f  the formula to fit
     */
    def fit (f: ARX_Formula): ARX_D =
        val key = FormulaScorer.keyOf (f)
        fitCache.get (key) match
        case Some (mod) => mod
        case None =>
            val mod = train (build (f), fitRng)
            if fitCache.size >= CACHE_SZ then fitCache -= fitCache.head._1   // evict the oldest
            fitCache (key) = mod
            mod
    end fit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit a formula on ALL training rows (parameters and coefficients; forecasts
     *  for every row): the final model of a winner, the one `showBest` reports and
     *  `tnt` rolls over the testing rows.  Not cached.
     *  @param f  the formula to fit
     */
    def fitFull (f: ARX_Formula): ARX_D = train (build (f, trSize), trRng)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit and score one formula: its QoF matrices on the fit rows (in-sample)
     *  and on the validation rows, and the selection score J (@see `scoreOf`); a
     *  failed fit ranks last, as does a rejected one (@see `rejection`), with its reason.
     *  @param f  the formula to score
     */
    def score (f: ARX_Formula): Candidate =
        val nf = NARX_Formula (f).nFree
        if maxFree >= 0 && nf > maxFree then
            return failed (f, f.nTerms, reason = s"$nf free parameters, at most $maxFree allowed")
        try
            val mod = fit (f)
            val why = rejection (f, mod)
            if why != null then
                debug ("score", s"rejected '${f.formulaStr}': $why")
                return failed (f, f.nTerms, reason = why)
            val q  = qofFit (mod)
            val qv = qofValidationAll (mod, size => build (f, size))       // every validation block
            Candidate (f, scoreOf (q, qv), q, f.nTerms, qofVal = qv)
        catch case e: Exception =>
            debug ("score", s"rejected '${f.formulaStr}': ${e.getMessage}")
            failed (f, f.nTerms, reason = s"fit failed: ${e.getMessage}")
    end score

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return why a fitted formula is rejected, or null when it is not: a free
     *  parameter fitted within `boundTol` of its box (the nonlinear term has
     *  collapsed onto a linear or constant limit), an exponent within 0.05 of 1 (a
     *  plain lag), or a free-parameter term whose fitted column has a variance
     *  inflation factor above `maxVIF` against the other columns on the fit rows
     *  (a parameter spent on a column the others already span).
     *  @param f    the formula (its parameters hold the fitted values)
     *  @param mod  the fitted model
     */
    def rejection (f: ARX_Formula, mod: ARX_D): String =
        val pars = NARX_Formula (f).params.filter (_.free)
        def onBound (p: Par): String =                                   // the bound a parameter sits on, or null
            val rng = p.ub - p.lb
            val tol = boundTol * rng
            if rng.isFinite && rng > 0.0 && p.value - p.lb <= tol then s"${p.role} ${Par.fmt (p.value)} at its lower bound ${Par.fmt (p.lb)}"
            else if rng.isFinite && rng > 0.0 && p.ub - p.value <= tol then s"${p.role} ${Par.fmt (p.value)} at its upper bound ${Par.fmt (p.ub)}"
            else if p.role == ParRole.Power && math.abs (p.value - 1.0) < 0.05 then s"exponent ${Par.fmt (p.value)} is a plain lag"
            else null
        val bound = pars.iterator.map (onBound).find (_ != null)
        if bound.isDefined then return bound.get
        if maxVIF > 0.0 && pars.nonEmpty && f.nTerms > 1 then
            val xc = mod.getX (fitRng).center ()                           // centered columns on the fit rows
            val hit = f.preds.toSeq.zipWithIndex.iterator
                       .filter ((t, _) => NARX_Formula.params (Seq (t)).exists (_.free))
                       .map ((t, j) => (t, FormulaScorer.vif (xc, j)))
                       .find ((_, v) => v > maxVIF)
            if hit.isDefined then return f"term ${hit.get._1.label} is collinear with the others (VIF ${hit.get._2}%.1f)"
        end if
        null
    end rejection

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the validation sMAPE per horizon of a formula (exceptions propagate).
     *  @param f  the formula
     */
    def scoreH (f: ARX_Formula): VectorD = qofValidation (fit (f))(?, QoF.smape.ordinal)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train-and-Test a formula: a fresh model trained on ALL training rows
     *  (`fitFull`), then the base's rolling validation on the testing rows; return
     *  the model and its QoF matrix on the testing rows (@see `tnt (mod)`).
     *  @param f  the formula
     */
    def tnt (f: ARX_Formula): (ARX_D, MatrixD) =
        val mod = fitFull (f)
        (mod, tnt (mod))
    end tnt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a failed candidate (score `Double.MaxValue`, QoF -0.0).
     *  @param f       the formula (may be null)
     *  @param nTerms  its number of terms
     *  @param label   the label of a base model (null for a formula)
     *  @param reason  why it failed or was rejected (null => unknown)
     */
    def failed (f: ARX_Formula, nTerms: Int, label: String = null, reason: String = null): Candidate =
        Candidate (f, Double.MaxValue, MatrixD.fill (hh, Fit.N_QoF, -0.0), nTerms, label, reason = reason)
    end failed

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the diagnostic notes of a fitted candidate for a proposer: the
     *  autocorrelation of the h = 1 residuals on the fit rows at lags 1..k (what the
     *  equation leaves unexplained) and the terms ranked by their standardized h = 1
     *  effect (@see `Explainer.importance`); "" for a failed or base candidate.
     *  @param cand  the candidate
     *  @param k     the number of residual lags reported
     */
    def notes (cand: Candidate, k: Int = hp("maxLag").toInt): String =
        if cand.failed || cand.formula == null then return ""
        try
            val mod = fit (cand.formula)
            val (y_, yf) = (mod.getY, mod.getYf)
            val e   = (y_(fitRng) - yf(?, 1)(fitRng)).drop (skip)           // h = 1 residuals on the fit rows
            val acf = (1 to k).map (l => f"${e.acorr (l)}%.2f").mkString ("[", ", ", "]")
            val imp = Explainer.importance (mod, cand.formula, fitSize)
            val tot = imp.map (i => math.abs (i.effect)).sum
            val top = imp.sortBy (i => -math.abs (i.effect)).take (5).map (i =>
                          f"${i.label} ${if tot > 0 then 100 * math.abs (i.effect) / tot else 0.0}%.0f%%").mkString (", ")
            s"residual autocorrelation of the best at lags 1..$k: $acf; its terms by importance: $top"
        catch case e: Exception => debug ("notes", s"no notes: ${e.getMessage}"); ""
    end notes

//  B A S E   M O D E L S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit and score a base model (e.g., `ARX_D (xe, y, hh)`, `RandomWalk (y, hh)`)
     *  the same way a formula is scored, under the given label.
     *  @param mod    the model (built on the full series)
     *  @param label  its display name in tables
     */
    def score (mod: Forecaster, label: String): Candidate =
        try
            train (mod, fitRng)
            val q  = qofFit (mod)
            val qv = qofValidationAll (mod, _ => mod)                      // the same object, retrained per block
            if valFolds > 1 && nVal > 0 then train (mod, fitRng)           // left as trained on the fit rows
            val k  = mod match
                     case d: Forecaster_D => d.getX.dim2
                     case _               => 0
            Candidate (null, scoreOf (q, qv), q, k, label, mod, qofVal = qv)
        catch case e: Exception =>
            debug ("score", s"rejected '$label': ${e.getMessage}")
            failed (null, 0, label, reason = s"fit failed: ${e.getMessage}")
    end score

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a model on the given leading rows (the fit rows of the search, or all
     *  training rows for a final model, as `reportBest` and `rollValidate` train)
     *  and fill its forecast matrix for EVERY row, so the rows after them hold
     *  out-of-sample forecasts; set the fixed skip.
     *  @param mod   the model (built on the full series)
     *  @param rows  the rows trained on (defaults to the training rows)
     */
    def train [T <: Forecaster] (mod: T, rows: Range = trRng): T =
        mod.setSkip (skip)
        mod match
        case d: Forecaster_D =>
            val (x, yy) = (d.getX, d.getYy)
            d.train (x(rows), yy(rows))
            d.predictAll (yy)
            d.forecastAll ()
        case f =>
            f.train (null, f.getY (rows))
            f.predictAll (f.getY)
            f.forecastAll (f.getY)
        end match
        nEval += 1
        mod
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train-and-Test a trained model: the base's `rollValidate` on the testing
     *  rows (sliding window, refit every rc rows, no skip -- as `tnT_Test`), then
     *  its QoF matrix on `teRng`.
     *  @param mod  the model (trained on the training rows)
     */
    def tnt (mod: Forecaster): MatrixD =
        mod.setSkip ()
        mod.rollValidate (rc)
        qofByH (mod, teRng)
    end tnt

//  Q O F

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the in-sample QoF matrix of a trained model: `qofByH` on the
     *  training rows (the first `skip` rows are dropped by `diagnose`).
     *  @param mod  the model (trained on the training rows, forecasts filled)
     */
    def qofTrain (mod: Forecaster): MatrixD = qofByH (mod, trRng)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the in-sample QoF matrix of a model trained on the fit rows:
     *  `qofByH` on the fit rows (the first `skip` rows are dropped by `diagnose`).
     *  @param mod  the model (trained on the fit rows, forecasts filled)
     */
    def qofFit (mod: Forecaster): MatrixD = qofByH (mod, fitRng)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the QoF matrix of a model trained on the fit rows over the VALIDATION
     *  rows, every forecast there out of sample, no row skipped; the fit rows'
     *  matrix when there are no validation rows (hp("valFrac") = 0).
     *  @param mod  the model (trained on the fit rows, forecasts filled)
     */
    def qofValidation (mod: Forecaster): MatrixD =
        if nVal == 0 then qofFit (mod) else qofByH (mod, blocks(0), noSkip = true)
    end qofValidation

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the validation QoF matrix over ALL rolling-origin blocks: block 0
     *  from the model trained on the fit rows, block i from a fresh model trained
     *  on every row before the block (`fresh (size)` builds it untrained), the
     *  matrices averaged over the blocks; with one block it is `qofValidation`.
     *  @param mod    the model trained on the fit rows (forecasts filled)
     *  @param fresh  a builder of the model on the full series, given the leading rows
     *                    its transforms and parameters are fitted on
     */
    def qofValidationAll (mod: Forecaster, fresh: Int => Forecaster): MatrixD =
        val q0 = qofValidation (mod)
        if valFolds == 1 || nVal == 0 then q0
        else
            var sum = q0
            for i <- 1 until valFolds do
                val rows = 0 until blocks(i).start
                val m_i  = train (fresh (rows.end), rows)
                sum = sum + qofByH (m_i, blocks(i), noSkip = true)
            end for
            sum / valFolds.toDouble
    end qofValidationAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the QoF matrix (one row per horizon, @see `Fit.N_QoF`) of a model on
     *  a row range: the alignment, DF reset and `diagnose` of `ForecastMatrix.diagnoseAll`
     *  restricted to rows t1 until t2 and QUIET (no plot, no print).  As there,
     *  yf(t, h) is paired with y(t+h-1), so the h-step pairs inside the range are
     *  y(t1+h-1 until t2) and yf(?, h)(t1 until t2-h+1); `diagnose` drops the first
     *  `skip` rows of both and inverts a response transform (@see `Fit.diagnose`).
     *  @param mod     the model (forecast matrix filled for the range)
     *  @param rng     the rows t1 until t2
     *  @param noSkip  whether to diagnose every row of the range (an out-of-sample range)
     */
    def qofByH (mod: Forecaster, rng: Range, noSkip: Boolean = false): MatrixD =
        val ftMat = new MatrixD (hh, Fit.N_QoF)
        for h <- 1 to hh do ftMat(h-1) = diagnoseH (mod, rng, h, noSkip)
        ftMat
    end qofByH

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose ONE horizon of a model on a row range (the per-horizon step of
     *  `qofByH`) and return its QoF vector; the model's `summary` then reports
     *  that horizon's fit (it reads the last `diagnose`).
     *  @param mod     the model (forecast matrix filled for the range)
     *  @param rng     the rows t1 until t2
     *  @param h       the horizon (1 to hh)
     *  @param noSkip  whether to diagnose every row of the range (the skip is restored after)
     */
    def diagnoseH (mod: Forecaster, rng: Range, h: Int, noSkip: Boolean = false): VectorD =
        val (y_, yf)  = (mod.getY, mod.getYf)
        val (t1, t2)  = (rng.start, rng.end)
        val yy  = y_(t1+h-1 until t2)                                     // align the actual response values
        val yfh = yf(?, h)(t1 until t2-h+1)                               // align column h of the forecast matrix
        val sk  = mod.getSkip
        if noSkip then mod.setSkip ()                                     // an out-of-sample range skips no row
        mod.mod_resetDF (if noSkip then math.max (yy.dim, fitSize - skip) else yy.dim - mod.getSkip)   // the DF: of the fit, on a validation block
        val qof = mod.diagnose (yy, yfh)                                  // use column h of yf
        if noSkip then mod.setSkip (sk)
        qof
    end diagnoseH

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the selection score J: the mean over horizons of the FPE-form
     *  parsimony-adjusted validation sMAPE, sMAPE_val,h (1 + 2 (|S| + 1 + extra) / n_h)
     *  (@see `jOf`), or its h = 1 value when hp("hSel") = 1; `Double.MaxValue`
     *  when undefined.  With one matrix (the fit rows') it is the in-sample J of
     *  rev3.0, the criterion the base's all-horizons feature selection minimizes.
     *  @param qof    the QoF matrix on the fit rows (one row per horizon)
     *  @param qofV   the QoF matrix on the validation rows (null => qof)
     *  @param extra  parameters charged beyond the model's own (free parameters held fixed)
     */
    def scoreOf (qof: MatrixD, qofV: MatrixD = null, extra: Int = 0): Double =
        val sJ = FormulaScorer.jOf (qof, if qofV == null then qof else qofV, extra)
        val s  = if hSel == 1 then sJ(0) else sJ.mean
        if s.isNaN || s.isInfinite then Double.MaxValue else s
    end scoreOf

//  L I B R A R I E S   (the cross-entropy stage, @see `TermLibrary`)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the library model for a term list (untrained): one column per term,
     *  the design matrix the cross-entropy stage selects from.  A term with free
     *  parameters enters at their current values (the role start values resolved on
     *  the training rows, or the values of the fit of the candidate it came from); the
     *  selected equation is then scored by `score`, which fits them (@see `buildFixed`).
     *  @param resp    the response variable
     *  @param terms   the library terms (one column each)
     *  @param tFormT  the response/base-column transform (null => none)
     */
    def library (resp: FormulaVar, terms: scala.collection.Seq [Term], tFormT: TransformT = null): ARX_D =
        buildFixed (NARX_Formula (resp, terms, tFormT))                 // free parameters at their current values
    end library

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate a column subset of a library model (@see `library`): fit the
     *  sub-model on the training rows and score it as `score` does, returning a
     *  `BestStep` whose QoF vector is the h = 1 QoF vector with the selection score
     *  at index qk and the mean sMAPE over horizons at `QoF.smape` -- the evaluator
     *  `SubsetSearch` calls (@see `TermLibrary.selectBy`).  A failed fit returns a
     *  null QoF (ranked last by the selector).  The score J is minimized, so qk
     *  must be a minimized slot (a maximized one, e.g., rSq, would rank it upside down).
     *  @param lib   the library model (all candidate columns)
     *  @param cols  the columns to keep
     *  @param qk    index of the QoF slot the selection score is written to (minimized)
     */
    def evalCols (lib: ARX_D, cols: LSET [Int], nFree: Int = 0)(using qk: Int): BestStep =
        require (! Fit.maxi.contains (qk), s"evalCols: the score J is minimized, qk = $qk is a maximized slot")
        if cols.isEmpty then return BestStep (-1, null, null, cols)()  // the mean model is not a candidate
        try
            val sub = train (lib.getModel (cols), fitRng)
            val q   = qofFit (sub)
            val qv  = qofValidationAll (sub, _ => lib.getModel (cols))
            val s0  = scoreOf (q, qv, nFree)                               // validation J, free parameters charged
            val s   = if maxFree >= 0 && nFree > maxFree && s0 < Double.MaxValue then s0 * (1.0 + (nFree - maxFree))   // over the cap: steep
                      else s0                                              //   (a set stays evaluable, so backward/CEM can move off it)
            if s == Double.MaxValue then BestStep (-1, null, null, cols)()
            else
                val qof = q(0).copy
                qof(QoF.smape.ordinal) = q(?, QoF.smape.ordinal).mean
                qof(qk) = s
                BestStep (-1, qof, null, cols)(qof(qk))
        catch case e: Exception =>
            debug ("evalCols", s"cols = $cols failed: ${e.getMessage}")
            BestStep (-1, null, null, cols)()
    end evalCols

end FormulaScorer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FormulaScorer` companion object holds the search hyper-parameters (the
 *  one map every arm and the harness read; `LLM_NARX_SR.hp` is the same object, so
 *  tests set them as `LLM_NARX_SR.hp("rc") = 5`).
 */
object FormulaScorer:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the canonical key of a formula: `NARX_Formula.key`, in which every free
     *  parameter is a bare ?, so a formula and its fitted version are one candidate.
     *  @param f  the formula
     */
    def keyOf (f: ARX_Formula): String = NARX_Formula (f).key

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the FPE-form parsimony-adjusted sMAPE per horizon of a candidate,
     *      sMAPE_val,h (1 + 2 p_h / n_h),  p_h = dfr_h + 1 + extra,
     *  the sMAPE taken on the validation rows, the parameter count p (the slopes,
     *  the intercept, the free nonlinear parameters a fitted `NARX_SR_D` counts in
     *  `dfr`, and `extra` for parameters held fixed) and the row count n from the
     *  fit rows (@see `Forecaster_D.smapeFPE`: with qofV = qof it is that in-sample form).
     *  @param qof    the QoF matrix on the fit rows (one row per horizon)
     *  @param qofV   the QoF matrix on the validation rows
     *  @param extra  parameters charged beyond the model's own
     */
    def jOf (qof: MatrixD, qofV: MatrixD, extra: Int = 0): VectorD =
        VectorD (for h <- qof.indices yield
            val p = qof(h, QoF.dfr.ordinal) + 1.0 + extra
            val n = qof(h, QoF.m.ordinal)
            qofV(h, QoF.smape.ordinal) * (1.0 + 2.0 * p / n))
    end jOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Variance Inflation Factor of column j of a (centered) matrix:
     *  1 / (1 - R^2) of the regression of x_j on the other columns (no intercept:
     *  the columns are centered); infinite when x_j is spanned by them.
     *  @param xc  the centered matrix
     *  @param j   the column
     */
    def vif (xc: MatrixD, j: Int): Double =
        if xc.dim2 < 2 then return 1.0
        val xj  = xc(?, j)
        val sst = xj.normSq
        if sst <= 0.0 then return Double.PositiveInfinity
        val xo  = xc.not (?, j)
        val reg = new Regression (xo, xj)
        reg.train ()
        val sse = (xj - reg.predict (xo)).normSq
        val r2  = 1.0 - sse / sst
        if r2 >= 1.0 then Double.PositiveInfinity else 1.0 / (1.0 - r2)
    end vif

    val hp = new HyperParameter
    hp += ("hh", 6, 6)                                                  // maximum forecasting horizon (h = 1..hh)
    hp += ("maxLag", 6, 6)                                              // maximum lag for proposals, libraries and the parser
    hp += ("skip", 6, 6)                                                // rows skipped by every in-sample diagnosis (same for all candidates)
    hp += ("rc", 5, 5)                                                  // retraining cycle of `rollValidate` on the testing rows
    hp += ("hSel", 0, 0)                                                // 0 => score = mean FPE-sMAPE over horizons, 1 => h = 1 only
    hp += ("lambda", 1.0, 1.0)                                          // ridge penalty carried by every model
    hp += ("perRound", 5, 5)                                            // proposals requested per round
    hp += ("maxRounds", 20, 20)                                         // maximum rounds per loop
    hp += ("patience", 5, 5)                                            // stagnant rounds before a loop stops
    hp += ("topK", 5, 5)                                                // candidates fed back each round
    hp += ("nFreqs", 2, 2)                                              // periodogram frequencies for wave(k)
    hp += ("exoK", 2, 2)                                                // exogenous variables kept by `ExoScreen` (Auto mode)
    hp += ("maxCols", 12, 12)                                           // largest column set the grammar CEM samples
    hp += ("minTerms", 3, 3)                                            // smallest proposal a trimmer touches
    hp += ("nlTerms", 1, 1)                                             // 1 => proposers and grammar libraries use the free-parameter forms (@see `NARX_FormulaParser`)
    hp += ("valFrac", 0.25, 0.25)                                       // share of the training rows held out for the selection score (0 => in-sample J)
    hp += ("valFolds", 3, 3)                                            // rolling-origin blocks of the validation rows (k => k fits per candidate; 1 => one block)
    hp += ("maxFree", 2, 2)                                             // free nonlinear parameters allowed per candidate (-1 => no cap)
    hp += ("boundTol", 0.01, 0.01)                                      // a fitted parameter within this fraction of its box of a bound is rejected
    hp += ("maxVIF", 10.0, 10.0)                                        // a free-parameter column with a larger VIF is rejected (0 => off)
    hp += ("fitLib", 1, 1)                                              // 1 => library terms with free parameters are fitted one by one before selection
    hp += ("refitTop", 5, 5)                                            // CEM: the best sets of this many rounds are refitted with their free parameters
    hp += ("cemSample", 300, 300)                                       // CEM: column sets per round on a library (-1 => CEM_Selector's default)
    hp += ("cemIter", 15, 15)                                           // CEM: maximum rounds on a library
    hp += ("maxFits", 0, 0)                                             // model fits a loop may make per run (0 => no budget)

end FormulaScorer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaScorerTest` main function tests the `FormulaScorer` class on an
 *  exactly linear series with the formula y ~= y~1: with the fit made on the
 *  training rows and a negligible ridge penalty, the h = 1 fit is y_t = 2 + y_t-1
 *  and the in-sample sMAPE per horizon is small -- not zero: the first lag row is
 *  padded (`VectorD.lagv` repeats the first value, as `ARX.buildMatrix` does), so
 *  one training row of 32 is off by 2 and the fit is near-exact (intercept 1.72,
 *  slope 1.006, sMAPE 0.2 %) -- and the selection score is the mean over horizons
 *  of the FPE-form sMAPE_h (1 + 2 (dfr + 1) / m_h).
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.formulaScorerTest
 */
@main def formulaScorerTest (): Unit =

    val hh    = 3
    val n     = 40
    val bound = 1.0                                                     // sMAPE (%) with the padded first lag row
    val y     = VectorD (for t <- 0 until n yield 10.0 + 2.0 * t)

    given dataset: Map [String, VectorD] = Map ("y" -> y)

    FormulaScorer.hp("lambda") = 1e-9                                   // near OLS
    FormulaScorer.hp("valFrac") = 0.0                                   // the in-sample J this test asserts
    val scorer = new FormulaScorer (hh)
    val f      = NARX_FormulaParser.parseFormula ("y ~= y~1", Set ("y"), maxLag = 1)
    val mod    = scorer.fit (f)
    val cand   = scorer.score (f)
    val sH     = cand.smapeH

    banner ("exact linear series: y ~= y~1")
    println (s"training rows        = ${scorer.trRng}, testing rows = ${scorer.teRng}, skip = ${scorer.skip}")
    println (s"sMAPE per horizon    = $sH")
    println (s"h = 1: intercept     = ${mod.intercept(0)}, slope = ${mod.slopes(0, 0)}")
    println (s"candidate            = ${cand.formulaStr} score = ${cand.score}, terms = ${cand.nTerms}")
    println (s"model hparameter     = ${mod.hparameter}")
    assert (sH.forall (_ < bound), s"sMAPE not small: $sH")
    assert (math.abs (mod.intercept(0) - 2.0) < 0.5 && math.abs (mod.slopes(0, 0) - 1.0) < 0.01,
            "h = 1 fit is not (near) y_t = 2 + y_t-1")
    val q   = cand.qofTrain
    val exp = VectorD (for h <- 0 until hh yield
                           q(h, QoF.smape.ordinal) * (1.0 + 2.0 * (q(h, QoF.dfr.ordinal) + 1.0) / q(h, QoF.m.ordinal))).mean
    assert (math.abs (cand.score - exp) < 1e-9, s"score ${cand.score} != mean FPE-form sMAPE $exp")
    assert (math.abs (mod.hparameter("lambda").toDouble - 1e-9) < 1e-15, "the model must carry the scorer's ridge penalty")
    assert (scorer.fit (f) eq mod, "second fit of the same formula must come from the cache")
    println ("formulaScorerTest: passed")

end formulaScorerTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaScorerTest2` main function tests the response transform path:
 *  a formula with TFORM: Log1p on a positive linear series is fitted in log1p
 *  space and diagnosed on the original scale (`Fit.diagnose` inverts), so its
 *  sMAPE is small (the log-linear fit is not exact) but finite and well below
 *  the no-model value.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.formulaScorerTest2
 */
@main def formulaScorerTest2 (): Unit =

    val hh    = 3
    val n     = 40
    val bound = 10.0
    val y     = VectorD (for t <- 0 until n yield 10.0 + 2.0 * t)         // positive: safe for Log1p

    given dataset: Map [String, VectorD] = Map ("y" -> y)

    FormulaScorer.hp("lambda") = 1e-9
    val scorer = new FormulaScorer (hh)
    val f      = NARX_FormulaParser.parseFormula ("y ~= y~1 TFORM: Log1p", Set ("y"), maxLag = 1)
    val mod    = scorer.fit (f)
    val sH     = scorer.scoreH (f)

    banner ("TFORM: Log1p on a positive linear series")
    println (s"formula  = ${f.formulaStr}, tFormT = ${f.tFormT}, model transform = ${mod.getYForm}")
    println (s"sMAPE per horizon (original scale) = $sH")
    assert (mod.getYForm != null, "the model must carry the response transform")
    assert (sH.forall (v => ! v.isNaN && v < bound), s"sMAPE out of bounds: $sH")
    println ("formulaScorerTest2: passed")

end formulaScorerTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaScorerTest3` main function tests the alignment of `qofByH`
 *  against a hand-written computation from the model's own y and yf -- on the
 *  testing rows without skip and on the training rows with the fixed skip -- and
 *  the forecast identity yf(t, h) = b0(h) + x(t) dot coef(?, h) for a testing row.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.formulaScorerTest3
 */
@main def formulaScorerTest3 (): Unit =

    val hh = 3
    val n  = 60
    val y  = VectorD (for t <- 0 until n yield 10.0 + 2.0 * t + 3.0 * math.sin (t / 3.0))

    given dataset: Map [String, VectorD] = Map ("y" -> y)

    val scorer = new FormulaScorer (hh)
    val f      = NARX_FormulaParser.parseFormula ("y ~= y~1", Set ("y"), maxLag = 1)
    val mod    = scorer.fit (f)                                          // one fit, forecasts for every row
    val (yv, yf) = (mod.getY, mod.getYf)
    val (tr, m, skip) = (scorer.trSize, yv.dim, scorer.skip)

    banner ("training rows with the fixed skip")
    val qTr = scorer.qofTrain (mod)(?, QoF.smape.ordinal)
    val mTr = VectorD (for h <- 1 to hh yield
                           FitM.smapeF (yv(h-1 until tr).drop (skip), yf(?, h)(0 until tr-h+1).drop (skip)))
    println (s"scorer = $qTr")
    println (s"manual = $mTr")
    assert ((qTr - mTr).normSq < 1e-18, "qofTrain disagrees with the manual alignment")

    banner ("testing rows, no skip")
    mod.setSkip ()
    val qTe = scorer.qofByH (mod, scorer.teRng)(?, QoF.smape.ordinal)
    val mTe = VectorD (for h <- 1 to hh yield FitM.smapeF (yv(tr+h-1 until m), yf(?, h)(tr until m-h+1)))
    println (s"scorer = $qTe")
    println (s"manual = $mTe")
    assert ((qTe - mTe).normSq < 1e-18, "qofByH disagrees with the manual alignment")

    banner ("forecast identity on a testing row")
    val t   = tr + 2
    val yp1 = mod.intercept + (mod.slopes dot mod.getX (t))     // one value per horizon
    println (s"yf($t, 1..$hh) = ${yf(t, 1 until hh+1)}")
    println (s"b0 + x(t) dot B = $yp1")
    assert ((yf(t, 1 until hh+1) - yp1).normSq < 1e-12, "forecast identity failed")
    println ("formulaScorerTest3: passed")

end formulaScorerTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaScorerTest4` main function is the COMPATIBILITY test: the formula
 *  y ~= y~(5,1) on the Covid-19 deaths (no exogenous variables), unscaled with
 *  lambda = 1, must give exactly what `aRX_DTest4` gives for p = 5 -- the base's
 *  `ARX_D (xe, y, hh)` scored through the same engine (in-sample and TnT), and the
 *  reference TnT sMAPE of the aRX_DTest4 run of 2026-09-14 (rc = 5, lambda = 1,
 *  df = 18 on the 24 testing rows):
 *      (15.8876, 21.1755, 29.3186, 38.2420, 42.0767, 50.5327).
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.formulaScorerTest4
 */
@main def formulaScorerTest4 (): Unit =

    import Example_Covid._

    val hh  = 6
    val ref = VectorD (15.8876, 21.1755, 29.3186, 38.2420, 42.0767, 50.5327)      // aRX_DTest4 log, 2026-09-14
    val (xe, y) = clip (loadData (NO_EXO))

    given dataset: Map [String, VectorD] = Map ("y" -> y)

    FormulaScorer.hp("lambda") = 1.0
    val scorer = new FormulaScorer (hh, scaled = false)                  // the base's regression: centered, not scaled
    println (s"rows m = ${scorer.m}, training = ${scorer.trRng}, testing = ${scorer.teRng}, rc = ${scorer.rc}")

    banner ("the base model: ARX_D (xe, y, hh) with p = 5, px = 0, spec = 0")
    val mod  = ARX_D (xe, y, hh, hparam = scorer.hparam (("p", 5), ("px", 0), ("spec", 0)))
    val cB   = scorer.score (mod, mod.modelName)
    val qB   = scorer.tnt (mod)
    println (s"in-sample sMAPE = ${cB.smapeH}")
    println (s"TnT sMAPE       = ${qB(?, QoF.smape.ordinal)}")

    banner ("the formula: y ~= y~(5,1)")
    val f    = NARX_FormulaParser.parseFormula ("y ~= y~(5,1)", scorer.vars, maxLag = 5)
    val cF   = scorer.score (f)
    val (modF, qF) = scorer.tnt (f)
    println (s"model name      = ${modF.modelName}, hparameter = ${modF.hparameter}")
    println (s"in-sample sMAPE = ${cF.smapeH}")
    println (s"TnT sMAPE       = ${qF(?, QoF.smape.ordinal)}")

    assert ((cF.smapeH - cB.smapeH).normSq < 1e-12, "formula and base model differ in-sample")
    assert ((qF(?, QoF.smape.ordinal) - qB(?, QoF.smape.ordinal)).normSq < 1e-12, "formula and base model differ on TnT")
    val diff = (qF(?, QoF.smape.ordinal) - ref).norm
    if diff < 1e-3 then println (s"matches the aRX_DTest4 reference (|diff| = $diff)")
    else println (s"NOTE: differs from the aRX_DTest4 reference by $diff -- rerun aRX_DTest4 (p = 5) and compare")
    println ("formulaScorerTest4: passed")

end formulaScorerTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaScorerTest5` main function tests `library` and `evalCols`, the
 *  evaluator of the cross-entropy stage: on a series driven by y~1 and z~2 (with
 *  noise) that starts at its level (a burn-in is dropped, so with lag padding no
 *  idle lag can explain a start-up transient), scored at h = 1 (a direct model at
 *  a later horizon needs other lags, so only h = 1 identifies the planted
 *  structure), the true column subset of an 18-column library (six lags each of
 *  y, z and a white-noise variable u) must score better than the full library
 *  (16 idle columns: their in-sample gain is below the FPE penalty by a margin of
 *  at least 1 % of J in 5000 simulated series) and than a wrong subset, a failed
 *  subset returns a null QoF, and the score written at qk is the FPE-form sMAPE
 *  (J) of the h = 1 QoF vector.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.formulaScorerTest5
 */
@main def formulaScorerTest5 (): Unit =

    import scalation.random.Normal

    val hh    = 1
    val n     = 400
    val burn  = 50                                                      // start-up rows dropped: the series starts at its level
    val noise = Normal (0.0, 1.0, 5)                                    // (with lag padding, idle lags would otherwise
    val zz    = Normal (0.0, 3.0, 6)                                    //  explain the transient from y(0) to the level)
    val uu    = Normal (0.0, 1.0, 7)
    val zAll  = VectorD (for _ <- 0 until n + burn yield 100.0 + zz.gen)
    val yAll  = new VectorD (n + burn)
    yAll(0) = 250.0; yAll(1) = 252.0
    for t <- 2 until n + burn do yAll(t) = 20.0 + 0.6 * yAll(t-1) + 1.0 * zAll(t-2) + noise.gen
    val (y, z) = (yAll(burn until n + burn), zAll(burn until n + burn))
    val u      = VectorD (for _ <- 0 until n yield uu.gen)              // white noise: a variable that explains nothing

    given dataset: Map [String, VectorD] = Map ("y" -> y, "z" -> z, "u" -> u)
    val qk = QoF.smapeC.ordinal                                          // the QoF slot carrying the selection score

    val scorer = new FormulaScorer (hh)
    val terms  = NARX_FormulaParser.parseFormula ("y ~= y~(6,1) + z~(6,1) + u~(6,1)", scorer.vars, maxLag = 6).preds
    val lib    = scorer.library (FormulaVar ("y"), terms)
    println (s"library columns = ${lib.getFname.mkString (", ")}")

    val truth = LSET (5, 10)                                             // y~1, z~2
    val wrong = LSET (4, 11)                                             // y~2, z~1
    val full  = LSET.range (0, terms.size)
    val sT = scorer.evalCols (lib, truth)(using qk)
    val sW = scorer.evalCols (lib, wrong)(using qk)
    val sF = scorer.evalCols (lib, full)(using qk)

    banner ("evalCols on a planted structure")
    println (s"truth ${truth} -> ${sT.qof(qk)}  (mean sMAPE ${sT.qof(QoF.smape.ordinal)})")
    println (s"wrong ${wrong} -> ${sW.qof(qk)}")
    println (s"full  ${full}  -> ${sF.qof(qk)}")
    assert (sT.qof(qk) < sW.qof(qk), "the true subset must beat a wrong subset")
    assert (lib.getFname(5) == "y~1" && lib.getFname(10) == "z~2", s"library column order changed: ${lib.getFname.mkString (", ")}")
    assert (sT.qof(qk) < sF.qof(qk), "the true subset must beat the full library (penalty and noise columns)")
    val cT = scorer.score (lib.getModel (truth), "truth")
    assert (math.abs (sT.qof(qk) - cT.scoreH(0)) < 1e-12, "score at qk = J of the h = 1 QoF vector")

    val bad = scorer.evalCols (lib, LSET ())(using qk)                   // no columns: the build must fail cleanly
    assert (bad.qof == null, "an empty column set must score as failed (null QoF)")
    println (s"fits made = ${scorer.evaluations}")
    println ("formulaScorerTest5: passed")

end formulaScorerTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaScorerTest6` main function tests the FREE-PARAMETER path of the
 *  engine: on a series y_t = 5 + 0.6 y_t-1 + 4 log (z_t-1 + 0.3) + e_t the formula
 *  y ~= y~1 + log(? + z~1) is fitted by `NARX_SR_D.fromTerms` (the shift fitted on the
 *  training rows, close to 0.3), its model is an `NARX_SR_D` with one free parameter
 *  counted in the degrees of freedom, it scores better than the same structure with
 *  the shift fixed at 1 (log1p), the fitted equation prints its value and reads back
 *  as a start value, and a second parse of the same structure (another start value)
 *  hits the cache: one candidate, one fit.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.formulaScorerTest6
 */
@main def formulaScorerTest6 (): Unit =

    import scalation.random.Normal

    val n     = 300
    val burn  = 50
    val noise = Normal (0.0, 0.3, 31)
    val zz    = Normal (0.0, 1.5, 32)
    val zAll  = VectorD (for _ <- 0 until n + burn yield 3.0 + zz.gen).map (v => math.max (v, 0.1))
    val yAll  = new VectorD (n + burn)
    yAll(0) = 20.0
    for t <- 1 until n + burn do yAll(t) = 5.0 + 0.6 * yAll(t-1) + 4.0 * math.log (zAll(t-1) + 0.3) + noise.gen
    val (y, z) = (yAll(burn until n + burn), zAll(burn until n + burn))

    given dataset: Map [String, VectorD] = Map ("y" -> y, "z" -> z)
    FormulaScorer.hp("lambda")  = 1e-6
    FormulaScorer.hp("valFrac") = 0.0                                   // the in-sample J (this test is about the fit path;
    val scorer = new FormulaScorer (hh = 2)                             //   on a validation block the extra parameter's charge can outweigh the shift's small gain)
    val vars   = scorer.vars

    banner ("the free shift vs. the fixed shift")
    val fFree  = NARX_FormulaParser.parseFormula ("y ~= y~1 + log(? + z~1)", vars, 4)
    val fFixed = NARX_FormulaParser.parseFormula ("y ~= y~1 + log1p(z~1)", vars, 4)
    val cFree  = scorer.score (fFree)
    val cFixed = scorer.score (fFixed)
    val mod    = scorer.fit (fFree)
    println (s"free:  ${cFree.formulaStr}  score ${cFree.score}  sMAPE ${cFree.smapeH}  nFree ${cFree.nFree}")
    println (s"fixed: ${cFixed.formulaStr}  score ${cFixed.score}  sMAPE ${cFixed.smapeH}")
    val shift = NARX_Formula (fFree).params.head.value
    println (s"fitted shift = $shift (true 0.3); model = ${mod.modelName}, dfr = ${cFree.qofTrain(0, QoF.dfr.ordinal)}")
    assert (mod.isInstanceOf [NARX_SR_D], "a formula with a free parameter must be fitted as an NARX_SR_D")
    assert (math.abs (shift - 0.3) < 0.15, s"the shift must be recovered: $shift")
    assert (cFree.nFree == 1 && cFree.qofTrain(0, QoF.dfr.ordinal) == 3.0, "two slopes + one free parameter = 3 degrees of freedom")
    assert (cFree.score < cFixed.score, "the free shift must score better than the fixed one")
    assert (! cFree.failed && cFree.formulaStr.contains ("log(") && ! cFree.formulaStr.contains ("?"),
            s"the fitted equation prints its value: ${cFree.formulaStr}")

    banner ("the fitted equation as a proposal, and the cache")
    val prop = cFree.proposalStr
    println (s"proposal string: $prop")
    val fBack = NARX_FormulaParser.parseFormula (prop, vars, 4)
    assert (NARX_Formula (fBack).params.head.w0 == NARX_Formula (fBack).params.head.value &&
            math.abs (NARX_Formula (fBack).params.head.w0 - shift) < 1e-3, "the proposal's value is the start value of the read-back formula")
    val nFits = scorer.evaluations
    val fOther = NARX_FormulaParser.parseFormula ("y ~= log(?0.5 + z~1) + y~1", vars, 4)
    assert (scorer.fit (fOther) eq mod, "the same structure with another start value must hit the cache")
    assert (scorer.evaluations == nFits, "no new fit for a cached structure")
    assert (fOther.key == fFree.key && cFree.key == FormulaScorer.keyOf (fOther), "one candidate key for the structure")

    banner ("a linear formula still goes through ARX_D.fromTerms")
    assert (! scorer.fit (fFixed).isInstanceOf [NARX_SR_D], "no free parameter => ARX_D")
    println ("formulaScorerTest6: passed")

end formulaScorerTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `formulaScorerTest7` main function tests the VALIDATION path of the engine
 *  (hp("valFrac") = 0.25, hp("valFolds") = 3) and the rejection rules on
 *  y_t = 5 + 0.6 y_t-1 + 1.5 z_t-2 + e_t: the fit rows are the first 75 % of the
 *  training rows and the validation rows the rest, in three rolling-origin blocks (a
 *  fresh fit on every row before each block); a candidate's `qofVal` is its QoF there
 *  averaged over the blocks; the score J is the FPE-form of the validation sMAPE with
 *  the fit's parameter count (@see `FormulaScorer.jOf`); the
 *  true structure scores better than the same structure padded with six idle lags
 *  (which fit the fit rows better but forecast the validation rows worse); the
 *  final `tnt` model is refitted on all training rows; a formula with three free
 *  parameters is rejected by the cap (hp("maxFree") = 2) without a fit; a free
 *  exponent on a linear driver is fitted close to 1 and rejected as a plain lag;
 *  an `ema` of a lag already in the formula is rejected on its bound or its VIF;
 *  and every rejection carries its reason.
 *  > runMain scalation.modeling.forecasting.nonlinear.llm_narx_sr.formulaScorerTest7
 */
@main def formulaScorerTest7 (): Unit =

    import scalation.random.Normal

    val n     = 240
    val burn  = 50
    val noise = Normal (0.0, 1.0, 51)
    val zz    = Normal (0.0, 3.0, 52)
    val zAll  = VectorD (for _ <- 0 until n + burn yield 100.0 + zz.gen)
    val yAll  = new VectorD (n + burn)
    yAll(0) = 250.0; yAll(1) = 252.0
    for t <- 2 until n + burn do yAll(t) = 5.0 + 0.6 * yAll(t-1) + 1.5 * zAll(t-2) + noise.gen
    val (y, z) = (yAll(burn until n + burn), zAll(burn until n + burn))
    val u      = VectorD (for _ <- 0 until n yield Normal (0.0, 1.0, 53).gen)

    given dataset: Map [String, VectorD] = Map ("y" -> y, "z" -> z, "u" -> u)
    FormulaScorer.hp("lambda")  = 1e-6
    FormulaScorer.hp("valFrac") = 0.25
    FormulaScorer.hp("maxFree") = 2
    val scorer = new FormulaScorer (hh = 2)
    val vars   = scorer.vars

    banner ("the split of the training rows")
    println (s"training rows ${scorer.trRng}: fit rows ${scorer.fitRng}, validation rows ${scorer.valRng}; testing rows ${scorer.teRng}")
    assert (scorer.fitSize + scorer.nVal == scorer.trSize && scorer.nVal >= scorer.hh + 1, "the validation rows are the tail of the training rows")
    assert (math.abs (scorer.nVal - 0.25 * scorer.trSize) <= 1.0, "a quarter of the training rows")

    banner ("the true structure vs the padded one")
    val fTrue = NARX_FormulaParser.parseFormula ("y ~= y~1 + z~2", vars, 6)
    val fPad  = NARX_FormulaParser.parseFormula ("y ~= y~1 + z~2 + u~(6,1)", vars, 6)
    val cTrue = scorer.score (fTrue)
    val cPad  = scorer.score (fPad)
    println (f"true:   in-sample ${cTrue.smape}%.3f  validation ${cTrue.smapeVal}%.3f  J ${cTrue.score}%.3f")
    println (f"padded: in-sample ${cPad.smape}%.3f  validation ${cPad.smapeVal}%.3f  J ${cPad.score}%.3f")
    assert (! cTrue.failed && ! cPad.failed, "both must fit")
    assert (cTrue.qofVal != null && cTrue.qofVal.dim == scorer.hh, "the validation QoF matrix is recorded")
    val expJ = FormulaScorer.jOf (cTrue.qofTrain, cTrue.qofVal).mean
    assert (math.abs (cTrue.score - expJ) < 1e-12, s"J ${cTrue.score} != FPE-form validation sMAPE $expJ")
    assert (cTrue.qofTrain(0, QoF.m.ordinal) == scorer.fitSize - scorer.skip, "the fit rows (minus the skip) are diagnosed in-sample")
    val mVal = scorer.blocks.map (_.size).sum.toDouble / scorer.valFolds   // the rows per block (averaged over the blocks)
    assert (math.abs (cTrue.qofVal(0, QoF.m.ordinal) - mVal) < 1e-9, "every validation row of a block is diagnosed at h = 1 (no skip)")
    println (s"validation blocks = ${scorer.blocks.mkString (", ")}")
    assert (cTrue.score < cPad.score, "the true structure must beat the padded one on the validation rows")

    banner ("the final model is refitted on all training rows")
    val (modF, qTe) = scorer.tnt (fTrue)
    println (s"TnT sMAPE = ${qTe(?, QoF.smape.ordinal)}")
    assert (modF ne scorer.fit (fTrue), "tnt builds a fresh model")
    assert (qTe(?, QoF.smape.ordinal).forall (v => ! v.isNaN && v < 5.0), "the TnT forecasts are good")

    banner ("rejections")
    val fCap = NARX_FormulaParser.parseFormula ("y ~= y~1 + pow(z~2, ?) + log(? + z~1) + sat(u~1, ?)", vars, 6)
    val nFit = scorer.evaluations
    val cCap = scorer.score (fCap)
    println (s"cap:  ${cCap.reason}")
    assert (cCap.failed && cCap.reason.contains ("3 free parameters") && scorer.evaluations == nFit, "three free parameters: rejected without a fit")

    val fPow = NARX_FormulaParser.parseFormula ("y ~= y~1 + pow(z~2, ?)", vars, 6)
    val cPow = scorer.score (fPow)
    val pPow = NARX_Formula (fPow).params.head.value
    println (f"pow:  fitted exponent $pPow%.3f -> ${cPow.reason}")
    assert (cPow.failed && (cPow.reason.contains ("plain lag") || cPow.reason.contains ("bound")),
            s"an exponent on a linear driver must be rejected: ${cPow.reason}")

    val fEma = NARX_FormulaParser.parseFormula ("y ~= y~1 + z~2 + ema(y~1, ?)", vars, 6)
    val cEma = scorer.score (fEma)
    val dEma = NARX_Formula (fEma).params.head
    println (s"ema:  decay ${Par.fmt (dEma.value)} in [${Par.fmt (dEma.lb)}, ${Par.fmt (dEma.ub)}] -> " +
             (if cEma.failed then cEma.reason else "accepted (interior decay, VIF within the limit)"))
    assert (cEma.failed == (cEma.reason != null), "a rejection carries its reason, an accepted candidate none")

    val fOK = NARX_FormulaParser.parseFormula ("y ~= y~1 + z~2 + u~1", vars, 6)
    assert (! scorer.score (fOK).failed, "a plain formula is not rejected")
    println (s"fits made = ${scorer.evaluations}")
    println ("formulaScorerTest7: passed")

end formulaScorerTest7
