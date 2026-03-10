
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

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SelectionTech` enumeration indicates the available feature selection
 *  techniques.
 */
enum SelectionTech:

     case Forward, Backward, Stepwise, Beam

end SelectionTech

// G I V E N S

// Change as needed the default (given instance) QoF metric used for Feature Selection (FS)

given qk: Int = QoF.rSqBar.ordinal                                      // which QoF metric index to use by default - Regression
//given qk: Int = QoF.smapeC.ordinal                                    // which QoF metric index to use by default - Time Series

// Change as needed the default (given instance) data to be used for Feature Selection (FS)

//given fullset_FS: Boolean = true                                      // use full dataset for Feature Selection
given fullset_FS: Boolean = false                                       // use the training set for Feature Selection
                                                                        // for example the training set used in the `validate` method

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Make a new restricted array of strings for the feature names based on the
 *  selected columns.
 *  @param fname  the original/full set of feature names
 *  @param cols   the selected columns
 */
def newFname (fname: Array [String], cols: LSET [Int]): Array [String] = cols.map (fname(_)).toArray 
def newFname (fname: Array [String], cols: VectorI): Array [String] = cols.map (fname(_)).toArray 


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FeatureSelection` trait establishes a framework for feature selection,
 *  i.e., selecting the features (e.g., variable x_j, cross term x_j x_k, or
 *  functional form x_j^2) to include in the model.
 */
trait FeatureSelection:

    private val debug = debugf ("FeatureSelection", true)               // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform feature selection to find the most predictive features/variables
     *  to  have in the model, returning the features/variables added and the new
     *  Quality of Fit (QoF) measures/metrics for all steps.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param tech   the feature selection technique to apply
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def selectFeatures (tech: SelectionTech, cross: String = "many")(using qk: Int):
                       (LSET [Int], MatrixD) =
        debug ("selectFeatures", s"select features based on QoF metric with index qk = $qk")
        tech match
        case SelectionTech.Forward  => forwardSelAll (cross)
        case SelectionTech.Backward => backwardElimAll (1, cross)
        case SelectionTech.Stepwise => stepwiseSelAll (cross)
        case SelectionTech.Beam     => beamSelAll (cross)
    end selectFeatures

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform FORWARD SELECTION to find the MOST predictive features/variables
     *  to ADD into the model, returning the features/variables added and the new
     *  Quality of Fit (QoF) measures/metrics for all steps.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSelAll (cross: String = "many")(using qk: Int): (LSET [Int], MatrixD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform BACKWARD ELIMINATION to find the LEAST predictive features/variables
     *  to REMOVE from the full model, returning the features/variables left and the
     *  new Quality of Fit (QoF)  measures/metrics for all steps.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param first  first variable to consider for elimination
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def backwardElimAll (first: Int = 1, cross: String = "many")(using qk: Int): (LSET [Int], MatrixD)

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
                       (LSET [Int], MatrixD)

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
    def beamSelAll (cross: String = "many", bk: Int = 3)(using qk: Int): (LSET [Int], MatrixD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the x-columns of matrix xy based on the two thresholds, returning
     *  the filtered matrix and the column indices/predictor variables selected.
     *  @param xy    the [ x, y ] combined data-response matrix
     *  @param thr1  the threshold used to compare the predictor x-columns to the y-column
     *               only want variables above some minimal dependency level
     *  @param thr2  the threshold used to compare the predictor x-columns with each other
     *               only want variables below some cut-off dependency/collinearity level
     *  @param dep   the variable/column dependency measure (defaults to correlation)
     *
    def filter (xy: MatrixD, thr1: Double = 0.2, thr2: Double = 0.8)
               (dep: MatrixD = xy.corr): (MatrixD, VectorI) =

        val lst  = dep.dim2 - 1                                                    // the index of last column (holds y)
        val depY = dep(?, lst)                                                     // the dependency sub-matrix for xy vs. y (last column)
        val depX = dep(0 until lst, 0 until lst)                                   // the dependency sub-matrix for x vs. x
        val indices  = for i <- 0 until lst if abs (depY(i)) > thr1 yield i        // row indices that match (> thr1)
        val sIndices = indices.sortBy (i => -abs (depY(i)))                        // sort indices from highest dep to lowest

        // only add index i if its dependency with all selected columns < thr2
        val selected = ArrayBuffer [Int] ()
        for i <- sIndices do
            if selected.forall (k => abs (depX(i, k)) < thr2) then selected += i   // row indices that also match (< thr2)
        val selected_ = selected.sorted

        (xy(?, selected_), new VectorI (selected_.size, selected_.toArray))
    end filter
     */

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
        else if qof == null then -1
        else if that.qof == null then 1
        else qof(qk).compare (that.qof(qk))
    end compare

end BestStep


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Update the rSq-based and smape QoF results for the l-th iteration of feature
 *  selection.
 *  @see `Predictor`
 *  @param rSq    the matrix contain information about r-Sq-based QoF measures
 *  @param l      the l-th iteration
 *  @param cross  indicator to include "many" cross-validation, "one" validation, or "none" nothing
 *  @param best   the best step so far
 */
def updateQoF (rSq: MatrixD, l: Int, cross: String, best: BestStep): Unit =
    rSq(l) =
        cross match
        case "many" =>
            Fit.qofVector (best.qof, best.mod.crossValidate ())       // results for model mod_l, with cross-validation
        case "one" =>
            val qof = best.mod.validate ()()._2
            qof match 
            case qofv: VectorD => 
                Fit.qofVector (best.qof, best.mod.qof2Stat (qofv))    // results for model mod_l, with validation
            case qofm: MatrixD =>
                Fit.qofVector (best.qof, best.mod.qof2Stat (qofm(0)))   // results for model mod_l, with validation
        case _     =>
            Fit.qofVector (best.qof, null)                            // results for model mod_l, with nothing
end updateQoF

