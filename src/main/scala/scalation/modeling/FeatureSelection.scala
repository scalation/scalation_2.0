
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Sep 27 20:58:20 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Support for Feature Selection and Best-Step
 *
 *  @see     bookdown.org/max/FES/selection.html
 *
 *  There are two important given instances the user may change (see below):
 *      qk          the QoF metric index used for comparing models
 *      fullset_FS  whether to use the full dataset or the training set for Feature Selection
 */

package scalation
package modeling

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SelectionTech` enumeration indicates the available feature selection
 *  techniques.
 */
enum SelectionTech:

     case Forward                                                       // Forward Selection (add best x_j)
     case Backward                                                      // Backward Elimination (remove worst x_j
     case Stepwise                                                      // Stepwise Selection (best step add/remove)
     case Beam                                                          // Beam Search (improve multiple candidate sub-models)
     case CEM                                                           // Cross-Entropy Method (@see `CEM_Selector`)

end SelectionTech


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Make a new restricted array of strings for the feature names based on the
 *  selected columns.
 *  @param fname  the original/full set of feature names
 *  @param cols   the selected columns
 */
def newFname (fname: Array [String], cols: LSET [Int]): Array [String] = cols.map (fname(_)).toArray 
def newFname (fname: Array [String], cols: VectorI): Array [String] = cols.map (fname(_)).toArray 


// G I V E N S

// Change as needed the default (given instance) QoF metric used for Feature Selection (FS)

//given qk: Int = QoF.rSqBar.ordinal                                    // which QoF metric index to use by default - Regression
//given qk: Int = QoF.smape.ordinal                                     // which QoF metric index to use by default - Time Series (no penalty)
given qk: Int = QoF.smapeC.ordinal                                      // which QoF metric index to use by default - Time Series
//given qk: Int = QoF.aic.ordinal                                       // which QoF metric index to use by default - either

// Change as needed the default (given instance) slack base used for Stepwise Feature Selection (FS)
// FIX -- need a more dataset sensitive value rather than a fixed magnitude of 25.0.

given slack_base: Double = if Fit.maxi.contains (qk) then 25.0          // new vs. old - slack when a high is good metric (qk -> rSqBar)
                           else -25.0                                   // new vs. old - slack when a low is good metric (qk -> smapeC)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FeatureSelection` object provides getter/setter methods for determining
 *  whether FS works with the FULL DATASET or just the TRAINING SET.
 */
object FeatureSelection:

    private var _fullset_FS = false                                     // FS uses: true => full dataset, false => training set
                                                                        // defaults to false, must call setter to change (fullset_FS = true)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Getter/Setter for `fullset_FS`.  If true, Feature Selection (FS) is performed
     *  on the entire (full) dataset (e.g., for In-Sample Testing).  If false, FS is
     *  restricted to the training set to prevent "data leakage" and ensure unbiased
     *  evaluation on the test set.  Note, any `validate` method should use only the
     *  training set.
     */
    inline def fullset_FS: Boolean = _fullset_FS
    def fullset_FS_= (fullset_FS_ : Boolean): Unit = _fullset_FS = fullset_FS_

end FeatureSelection


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SelectionReport` trait reports and tests the result of a feature selection
 *  the same way for every model family: `reportSelection` prints, for each selected
 *  column set, its size, columns and feature names and its QoF trace (and plots the
 *  trace when `DO_PLOT`), then builds the selected model(s) and tests them In-Sample
 *  and/or Out-of-Sample, the QoF of every model in ONE table.  What a "selected
 *  model" is and how it is tested differ per family and are the two hooks:
 *  `selectedModel` (a sub-model of a regression, the forecaster on the columns,
 *  the single-horizon forecaster at h) and `testSelected` (In-Sample: train on all
 *  the rows and report; Out-of-Sample: k-fold cross-validation for a predictor,
 *  rolling validation for a forecaster -- the steps of the family's own
 *  `inSample_Test` / `crossValidate` / `tnT_Test`).  The arrays may hold one column
 *  set (`selectFeatures`: one model) or one per horizon (`selectFeaturesPerH`: one
 *  single-horizon model per horizon, their QoF rows assembled into the table).
 */
trait SelectionReport:
    this: Model =>

    /** The type of model a selection yields in this family (a sub-model of a
     *  regression, a forecaster), the type `reportSelection` returns.
     */
    type SelectedModel <: Model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the model of a selected column set (hook).
     *  @param cols  the selected columns
     *  @param h     the horizon of a single-horizon selection (0 => none, or all horizons)
     */
    protected def selectedModel (cols: LSET [Int], h: Int): SelectedModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a selected model (hook) in-sample (train on all the rows, report) and/or
     *  out-of-sample (a predictor: k-fold cross-validation; a forecaster: rolling
     *  validation on the testing rows) and return the two QoF tables -- one row per
     *  horizon for a forecaster, one row for a predictor; null for a test not run.
     *  @param mod       the selected model
     *  @param inSample  whether to test in-sample
     *  @param tnt       whether to test out-of-sample (cross-validation / rolling validation)
     *  @param rc        the retraining cycle of a rolling validation (a forecaster)
     *  @param showYf    whether to print the forecast matrix (a forecaster)
     */
    protected def testSelected (mod: SelectedModel, inSample: Boolean, tnt: Boolean, rc: Int,
                                showYf: Boolean): (MatrixD, MatrixD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the column set to build the model from (hook): the set itself, or
     *  for a family whose selection returns the path (@see `FeatureSelection`) the
     *  best step along it.
     *  @param cols  the selected columns, as returned by the selection
     */
    protected def bestCols (cols: LSET [Int]): LSET [Int] = cols

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the name of column j for the report: the feature name, or x_j when
     *  the model names fewer columns than its matrix has.
     *  @param j  the column
     */
    protected def colName (j: Int): String =
        val fn = getFname
        if fn != null && j < fn.length then fn(j) else s"x$j"
    end colName

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Report and test a feature selection: print, for each selected column set,
     *  its size k (of n), columns and feature names and its QoF trace per step
     *  (`Fit.qofVector`: 100 R^2, 100 R^2 bar, sMAPE, 100 R^2 cv or sMAPE-IC), plot
     *  the trace (when `DO_PLOT`), then build the model(s) of the best set(s) and
     *  test them (@see the hooks); return the model(s) (an `IndexedSeq`: the model type
     *  is abstract) and the in-sample and out-of-sample QoF tables (null when not run), row h-1 = horizon h for a
     *  forecaster.  One set => one model; one set per horizon (hh sets) => one
     *  single-horizon model per horizon, whose QoF rows are assembled into the
     *  table and printed (`Fit.showFitMap`).
     *  @param colsArr    the selected column set(s): one, or one per horizon
     *  @param rSqArr     the QoF trace(s) of the selection(s), one per column set
     *  @param h          the horizon of a single-horizon selection (`selectFeaturesAtH`),
     *                        0 => none / all horizons (defaults to 0)
     *  @param inSample   whether to test the selected model(s) in-sample (defaults to true)
     *  @param tnt        whether to test them out-of-sample: a predictor's k-fold cross-validation,
     *                        a forecaster's rolling validation (defaults to true)
     *  @param rc         the retraining cycle of a rolling validation (defaults to 5)
     *  @param showTrace  whether to print the QoF trace(s) (defaults to true)
     *  @param plot       whether to plot the QoF trace(s), when `DO_PLOT` is set (defaults to true)
     *  @param showYf     whether to print the forecast matrices (defaults to false)
     */
    def reportSelection (colsArr: Array [LSET [Int]], rSqArr: Array [VEC [VectorD]], h: Int = 0,
                         inSample: Boolean = true, tnt: Boolean = true, rc: Int = 5,
                         showTrace: Boolean = true, plot: Boolean = true, showYf: Boolean = false):
                        (IndexedSeq [SelectedModel], MatrixD, MatrixD) =
        require (colsArr.length == rSqArr.length, "reportSelection: one trace per column set")
        require (colsArr.length >= 1, "reportSelection: no column set")
        val perH = colsArr.length > 1                                    // one single-horizon model per horizon
        require (h == 0 || ! perH, "reportSelection: a single-horizon selection has one column set")
        val n = getX.dim2

        val models = IndexedSeq.tabulate (colsArr.length)(k => selectedModel (bestCols (colsArr(k)), if perH then k + 1 else h))
        for k <- colsArr.indices do                                      // report the selection(s)
            val cols = colsArr(k)
            val best = bestCols (cols)
            val what = if perH then s" for horizon ${k+1}" else if h > 0 then s" for horizon $h" else ""
            banner (s"$modelName: selected$what: k = ${cols.size} of n = $n, cols = $cols")
            if best ne cols then
                println (s"best step: k = ${best.size}, cols = $best")
            println (s"features: ${best.toSeq.sorted.map (colName).mkString (", ")}")
            if showTrace then
                println ("trace per step: 100 R^2, 100 R^2 bar, sMAPE, 100 R^2 cv (or sMAPE-IC)")
                for r_ <- rSqArr(k) do println (s"    $r_")
            if plot && DO_PLOT && rSqArr(k).nonEmpty then
                new PlotM (null, rSqArr(k), Regression.metrics, s"QoF vs step for $modelName$what", lines = true)
        end for

        var qofIn: MatrixD  = null
        var qofTnT: MatrixD = null
        if ! perH then
            val (qi, qt) = testSelected (models(0), inSample, tnt, rc, showYf)
            qofIn = qi; qofTnT = qt
        else
            val hh = colsArr.length
            if inSample then qofIn  = new MatrixD (hh, Fit.N_QoF)
            if tnt      then qofTnT = new MatrixD (hh, Fit.N_QoF)
            for k <- 0 until hh do
                val (qi, qt) = testSelected (models(k), inSample, tnt, rc, showYf)
                if inSample then qofIn(k)  = qi(0)
                if tnt      then qofTnT(k) = qt(0)
            end for
            if inSample then
                banner (s"$modelName: In-Sample QoF per horizon, one model per horizon")
                println (Fit.showFitMap (qofIn.ᵀ))
            if tnt then
                banner (s"$modelName: Out-of-Sample QoF per horizon, one model per horizon")
                println (Fit.showFitMap (qofTnT.ᵀ))
        end if
        (models, qofIn, qofTnT)
    end reportSelection

end SelectionReport


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FeatureSelection` trait establishes a framework for feature selection,
 *  i.e., selecting the features (e.g., variable x_j, cross term x_j x_k, or
 *  functional form x_j^2) to include in the model.
 */
trait FeatureSelection
      extends SelectionReport:

    this: Model =>

    private val debug = debugf ("FeatureSelection", true)               // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the columns for the best model, if available.
     *  @param cols  the selected columns, as returned by the selection
     */
    protected override def bestCols (cols: LSET [Int]): LSET [Int] =
        val best = getBest
        if best != null && best.mod_cols != null && best.qof != null then best.mod_cols else cols
    end bestCols

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform feature selection to find the most predictive features/variables
     *  to  have in the model, returning the features/variables added and the new
     *  Quality of Fit (QoF) measures/metrics for all steps.
     *  @note:  every selection returns ARRAYS -- (Array [LSET [Int]], Array [VEC [VectorD]]) with ONE
     *  element: the column set (the path's order for forward/backward, @see `importance`) and
     *  its QoF trace -- the shape `forecasting.Forecaster_D.selectFeaturesPerH` fills with one
     *  element per horizon, so `reportSelection` prints and tests every selection alike; the
     *  best step along the path is `getBest`.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param tech   the feature selection technique to apply
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param first  first variable to consider for elimination
     *                    (default (1) assume intercept x_0 will be in any model)
     *  @param swap   whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def selectFeatures (tech: SelectionTech, cross: String = "many",
                        first: Int = 1, swap: Boolean = true)(using qk: Int):
                       (Array [LSET [Int]], Array [VEC [VectorD]]) =
        debug ("selectFeatures", s"use $tech to select features based on QoF metric with index qk = $qk")
        tech match
        case SelectionTech.Forward  => forwardSelAll (cross)
        case SelectionTech.Backward => backwardElimAll (first, cross)
        case SelectionTech.Stepwise => stepwiseSelAll (cross, swap)
        case SelectionTech.Beam     => beamSelAll (cross)
        case SelectionTech.CEM      => cemSelAll (cross, first)
    end selectFeatures

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform FORWARD SELECTION to find the MOST predictive features/variables
     *  to ADD into the model, returning the features/variables added and the new
     *  Quality of Fit (QoF) measures/metrics for all steps.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSelAll (cross: String = "many")(using qk: Int): (Array [LSET [Int]], Array [VEC [VectorD]])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform BACKWARD ELIMINATION to find the LEAST predictive features/variables
     *  to REMOVE from the full model, returning the features/variables left and the
     *  new Quality of Fit (QoF)  measures/metrics for all steps.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param first  first variable to consider for elimination
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def backwardElimAll (first: Int = 1, cross: String = "many")(using qk: Int):
                        (Array [LSET [Int]], Array [VEC [VectorD]])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform STEPWISE SELECTION to find a GOOD COMBINATION of predictive features/variables
     *  to have in the model, returning the features/variables selected and the new Quality of Fit
     *  (QoF) measures/metrics for all steps.  At each step, it calls forward and backward
     *  and takes the best of the two actions.  Stops when neither action yields improvement.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param swap   whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def stepwiseSelAll (cross: String = "many", swap: Boolean = true)(using qk: Int):
                       (Array [LSET [Int]], Array [VEC [VectorD]])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform BEAM SEARCH SELECTION to find a GOOD COMBINATION of predictive features/variables to
     *  have in the model, returning the top k sets of features/variables selected and the new Quality of
     *  Fit (QoF) measures/metrics for all steps.  At each step, iterate over the models in the beam
     *  (top k) and create candidates by adding features (phase 1) and then removing features (phase 2).
     *  From all the candidates, keep the best k and start a new iteration.  Stops when there is
     *  no improvement in any of top k or the maximum number of features is reached.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param bk     the beam width holding the top k models (defaults to 3)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def beamSelAll (cross: String = "many", bk: Int = 3)(using qk: Int): (Array [LSET [Int]], Array [VEC [VectorD]])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform CROSS-ENTROPY SELECTION (@see `CEM_Selector`) to find a GOOD COMBINATION of
     *  predictive features/variables: sample feature sets from per-feature inclusion
     *  probabilities, keep the elite sets and move the probabilities towards them,
     *  returning the best set found and the QoF measures/metrics of the best-so-far
     *  model after each round.  Not greedy, keeps diversity by sampling and accepts a
     *  prior over the features.  Models that support it override this method
     *  (@see `forecasting.Forecaster_Reg`); the default reports it as unsupported.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param cross    indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param first    the number of leading features always kept (default 1: the intercept x_0)
     *  @param prior    the prior inclusion probability per feature (null => 0.5 each)
     *  @param nSample  the number of feature sets sampled per round
     *  @param maxIter  the maximum number of rounds
     *  @param maxCols  the maximum number of features in a sampled set (-1 => no cap)
     *  @param stream   the random number stream
     *  @param qk       index of Quality of Fit (QoF) to use for comparing quality
     */
    def cemSelAll (cross: String = "many", first: Int = 1, prior: VectorD = null,
                   nSample: Int = CEM_Selector.N_SAMPLE, maxIter: Int = CEM_Selector.MAX_ITER,
                   maxCols: Int = -1, stream: Int = 0)(using qk: Int): (Array [LSET [Int]], Array [VEC [VectorD]]) =
        throw new UnsupportedOperationException (
            s"cemSelAll (cross = $cross, first = $first, prior = $prior, nSample = $nSample, maxIter = $maxIter, " +
            s"maxCols = $maxCols, stream = $stream, qk = $qk) is not supported by this model")
    end cemSelAll

end FeatureSelection


type Model_FS = (Predictor | neuralnet.PredictorMV) & Fit


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BestStep` is used to record the best improvement step found so far during
 *  feature selection.  Note, best depends on whether maximizing or minimizing
 *  @param col       the column/variable to ADD/REMOVE for this step
 *  @param qof       the Quality of Fit (QoF) for this step
 *  @param mod       the model including selected features/variables for this step
 *  @param mod_cols  the columns selected for mod
 *  @param qk        the index for the Quality of Fit (QoF) measure/metric used for comparison
 *  @param bestq     the best QoF for metric qk so far
 */
case class BestStep (col: Int = -1, qof: VectorD = null, mod: Model_FS = null, mod_cols: LSET [Int] = null)
                    (using qk: Int)(bestq: Double = Fit.extreme (qk))
     extends Ordered [BestStep]:

    private val debug = debugf ("BestStep", false)

    debug ("BestStep", s"bestq = $bestq")                          // needed for unused explicit parameter warning

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this step is better than that step.
     *  @param that_qof  the Qof for that step
     */
    infix def gt (that_qof: Double): Boolean =
        if Fit.maxi.contains (qk) then qof(qk) > that_qof          // maximize, e.g., R^2
        else qof(qk) < that_qof                                    // minimize, e.g., mse, smape
    end gt
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this step is better than or equal to that step.
     *  @param that_qof  the QoF for that step
     */
    infix def ge (that_qof: Double): Boolean =
        if Fit.maxi.contains (qk) then qof(qk) >= that_qof         // maximize, e.g., R^2
        else qof(qk) <= that_qof                                   // minimize, e.g., mse, smape
    end ge
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the better between this and candidate step.
     *  @param cand  the new candidate
     */
    def better (cand: BestStep): BestStep =
        debug ("better", s"cand = $cand vs. this = $this")
        if qof == null then cand
        else if cand gt qof(qk) then cand else this
    end better

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the better between this and the to be formed candidate step.
     *  @param j         the index of the feature/variable
     *  @param qof_j     the QoF for mod_j
     *  @param mod_j     the model with j
     *  @param mod_cols  the columns selected for mod_j
     */
    def better (j: Int, qof_j: VectorD, mod_j: Model_FS, mod_cols: LSET [Int]): BestStep =
        better (BestStep (j, qof_j, mod_j, mod_cols)(qof_j(qk)))
    end better

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the comparison result between this and that step.
     *  @param that  the other candidate
     */
    def compare (that: BestStep): Int =
        if qof == null && that.qof == null then 0
//      else if qof == null then -1
//      else if that.qof == null then 1
        else qof(qk).compare (that.qof(qk))
    end compare

end BestStep


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Update the key metrics, e.g., rSq-based and smape QoF results for the next
 *  iteration of feature selection.
 *  @see `Predictor`
 *  @param rSq    the VEC containing information about r-Sq-based QoF measures
 *  @param cross  indicator to include "many" cross-validation, "one" validation, or "none" nothing
 *  @param best   the best step so far
 */
def updateQoF (rSq: VEC [VectorD], cross: String, best: BestStep): Unit =
    cross match
    case "many" =>
        rSq += Fit.qofVector (best.qof, best.mod.crossValidate ())          // results for model mod_l, with cross-validation
    case "one" =>
        val qof = best.mod.validate ()()._2
        qof match 
        case qofv: VectorD => 
            rSq += Fit.qofVector (best.qof, best.mod.qof2Stat (qofv))       // results for model mod_l, with validation
        case qofm: MatrixD =>
            rSq += Fit.qofVector (best.qof, best.mod.qof2Stat (qofm(0)))    // results for model mod_l, with validation
    case _ =>
        rSq += Fit.qofVector (best.qof, null)                               // results for model mod_l, with nothing
end updateQoF

