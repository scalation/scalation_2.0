
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Predictor for Matrix Input, Vector Output
 */

package scalation
package modeling

import scala.collection.mutable.{ArrayBuffer => VEC, IndexedSeq, LinkedHashSet => LSET}
import scala.math.{max, min}
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._
//import scalation.random.RandomVecSample

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Shifted Rectified Linear Unit (Shifted ReLU) for a scalar.
 *  @param x  the scalar to be rectified
 *  @param a  the slope parameter: a = 1 => __/ , a = -1 => \__
 *  @param b  the shift (intercept analog) parameter: b = 0 => __/ , b = 2 => __/
 *                                                               0            0
 */
inline def srelu (x: Double, a: Double = 1.0, b: Double = 0.0): Double = max (0, a * x - b)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Rectify the prediction/forecast when they are required to be non-negative, by
 *  setting negative values to zero.
 *  @param yp    the predicted/forecasted value
 *  @param nneg  whether the values are required to be non-negative (e.g., counts)
 */
inline def rectify (yp: Double, nneg: Boolean = true): Double =
    if nneg && yp < 0.0 then 0.0 else yp
end rectify


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Rectify the prediction/forecast when they are required to be non-negative, by
 *  setting negative values in the vector to zero.
 *  @param yp    the predicted/forecasted vector
 *  @param nneg  whether the values are required to be non-negative (e.g., counts)
 */
inline def rectify (yp: VectorD, nneg: Boolean): VectorD =
    if nneg then yp.map (rectify (_)) else yp
end rectify


// G I V E N S

// Change as needed the default (given instance) whether to display plots

//given DO_PLOT: Boolean = false
given DO_PLOT: Boolean = true

//given DO_REPORT: Boolean = false
given DO_REPORT: Boolean = true

//given SHOW_QoF: Boolean = false   // Yousef_added
given SHOW_QoF: Boolean = true


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Predictor` trait provides a framwork for multiple predictive analytics
 *  techniques, e.g., `Regression`.  x is multi-dimensional [1, x_1, ... x_k].
 *  Fit the parameter vector b in for example the regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  @param x       the input/data m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model)
 *  @param y       the response/output m-vector
 *  @param fname   the feature/variable names (if null, use x_j's)
 *  @param hparam  the hyper-parameters for the model
 */
trait Predictor (x: MatrixD, y: VectorD, protected var fname: Array [String], hparam: HyperParameter)
      extends Model
         with FeatureSelection:

    private   val debug   = debugf ("Predictor", true)                       // debug function
    private   val flaw    = flawf ("Predictor")                              // flaw function

    if x != null then
        if x.dim != y.dim then flaw ("init", "row dimensions of x and y are incompatible")
        if x.dim2 < 1 then flaw ("init", s"dim2 = ${x.dim2} of the x matrix must be at least 1")
        if x.dim <= x.dim2 then
            flaw ("init", s"Predictor requires more rows ${x.dim} than columns ${x.dim2}")

    private val MIN_FOLDS = 3                                                // minimum number of folds for cross validation
    private val stream    = 0                                                // random number stream to use
    private val permGen   = TnT_Split.makePermGen (y.dim, stream)            // permutation generator

    protected var b: VectorD = null                                          // parameter/coefficient vector [b_0, b_1, ... b_k]

    if x != null && fname == null then fname = x.indices2.map ("x" + _).toArray  // default feature/variable names

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the set of columns (numbers) for the features in this model.
     */
    def mcols: LSET [Int] = LSET.range (0, getX.dim2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used data matrix x.  Mainly for derived classes where x is expanded
     *  from the given columns in x_, e.g., `SymbolicRegression.quadratic` adds squared columns.
     */
    def getX: MatrixD = x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response vector y.  Mainly for derived classes where y is
     *  transformed, e.g., `TranRegression`, `ARX`.
     */
    def getY: VectorD = y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the feature/variable names.
     */
    def getFname: Array [String] = fname

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of terms/parameters in the model, e.g., b_0 + b_1 x_1 + b_2 x_2 
     *  has three terms.
     */
    def numTerms: Int = getX.dim2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a predictive model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training
     *  dataset.  Training involves estimating the model parameters b.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The train2 method should work like the train method, but should also
     *  optimize hyper-parameters (e.g., shrinkage or learning rate).
     *  Only implementing classes needing this capability should override this method.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def train2 (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        throw new UnsupportedOperationException ("train2: not supported - no hyper-parameters to optimize")
    end train2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF vector.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.  Return the predictions and QoF.
     *  FIX - currently must override if y is transformed, @see `TranRegression`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output vector (defaults to full y)
     */
    def trainNtest (x_ : MatrixD = x, y_ : VectorD = y)
                   (xx: MatrixD = x, yy: VectorD = y): (VectorD, VectorD) =
        train (x_, y_)                                                       // train the model on training set
        debug ("trainNTest", s"b = $b")
        val (yp, qof) = test (xx, yy)                                        // test the model on testing set
        println (report (qof))                                               // report on Quality of Fit (QoF)
        Predictor.plotPrediction (yy, yp, modelName, doPlot = false)         // plot actual and predicted for test-set
        Predictor.plotPrediction (yy, yp, modelName, doPlot = DO_PLOT)       // plot actual and predicted for test-set (reordered)
        (yp, qof)
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z,
     *  e.g., (b_0, b_1, b_2) dot (1, z_1, z_2).
     *  Must override when using transformations, e.g., `ExpRegression`.
     *  @param z  the new vector to predict
     */
    def predict (z: VectorD): Double = b dot z

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of vector y = f(x_, b), e.g., x_ * b for `Regression`.
     *  May override for efficiency.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    def predict (x_ : MatrixD): VectorD =
        VectorD (for i <- x_.indices yield predict (x_(i)))
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a conformal PREDICTION INTERVAL half width for each prediction yp (y-hat).
     *  Implements Algorithm 2 Split Conformal Prediction from
     *  @see www.stat.cmu.edu/~ryantibs/papers/conformal.pdf
     *  @param x_  the testing/full data/input matrix
     *  @param y_  the testing/full response/output vector
     *  @param α   the significance level (1 - p_)
     */
    def predictCInt (x_ : MatrixD, y_ : VectorD, α: Double = .1): VectorD =
        val n     = y_.dim                                                   // number of instances
        var n_by2 = n / 2                                                    // number of instances for half
        val idx   = testIndices (n_by2, true)                                // randomly split into equal-size subsets idx, rest 
        val (x_e, x_t, y_e, y_t) = TnT_Split (x_, y_, idx)                   // Test-n-Train Split: test _e, train _t
        train (x_t, y_t)                                                     // train model on the TRAINING-set (rest)
        val yp = predict (x_e)                                               // predict values for TESTING-set (idx)

        val r  = y_e - yp                                                    // compute the residuals/errors for TESTING-set
        n_by2  = r.dim / 2                                                   // number of residuals for half
        val k  = math.ceil ((n_by2 + 1) * (1.0 - α)).toInt                   // determine k based on significance level   
        val d  = r.median (k)                                                // the k-th smallest residual from TESTING-set
        VectorD.fill (n)(d)                                                  // return as a vector (general form)
    end predictCInt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the hyper-parameters.
     */
    def hparameter: HyperParameter = hparam

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of parameter/coefficient values.
     */
    def parameter: VectorD = b

//  F E A T U R E   S E L E C T I O N

    // @see givens in `modeling.FeatureSelection`

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  Must be implemented for models that support feature selection.
     *  Otherwise, use @see `NoBuildModel
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): Predictor & Fit

    private var theBest = BestStep ()()                                      // record the best model from feature selection
    private val t_rng   = if FeatureSelection.fullset_FS then 0 until y.dim  // use full dataset for Feature Selection (FS)
                          else 0 until Model.trSize (y.dim)                  // use training set for Feature Selection (FS)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the best-step to default
     */
    def resetBest (): Unit = theBest = BestStep ()()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the best model found from feature selection.
     */
    def getBest: BestStep = theBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** When the new best-step is better than theBest, replace theBest.
     *  @param best  new best-step found during feature selection
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    private def updateBest (best: BestStep)(using qk: Int): Unit =
        if best.qof != null then
            if theBest.qof == null || (best gt theBest.qof(qk)) then theBest = best
    end updateBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variable to add the
     *  existing model, returning the variable to add and the new model.
     *  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols  the columns of matrix x currently included in the existing model
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSel (cols: LSET [Int])(using qk: Int): BestStep =
        var best = BestStep ()()                                             // best step so far

        for j <- x.indices2 if ! (cols contains j) do
            val cols_j = cols union LSET (j)                                 // try adding variable/column x_j
            val x_cols = x(?, cols_j)                                        // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols, newFname (fname, cols_j))       // regress with x_j added

            val (x_tr, y_tr) = (x_cols(t_rng), y(t_rng))                     // get full/training data
            mod_j.train (x_tr, y_tr)                                         // train model
            best = best.better (j, mod_j.test (x_tr, y_tr)._2, mod_j, cols_j)   // which is better
        end for

        if best.col == -1 then
            flaw ("forwardSel", "could not find a variable x_j to add: best.col = -1")
        best
    end forwardSel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evalaute the model with only one column, e.g., intercept only model.
     *  @param qk  index of Quality of Fit (QoF) to use for comparing quality
     */
    def select0 (qk: Int): BestStep =
        val x_cols = x(?, LSET (0))                                          // x projected onto columns {0}
        val mod_0  = buildModel (x_cols, newFname (fname, LSET (0)))         // regress with x_0 added
        mod_0.train ()                                                       // train model
        val qof_0 = mod_0.test ()._2
        BestStep (0, qof_0, mod_0)(qof_0(qk))                                // result for intercept only
    end select0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform FORWARD SELECTION to find the MOST predictive variables to have
     *  in the model, returning the variables added and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSelAll (cross: String = "many")(using qk: Int): (LSET [Int], VEC [VectorD]) =
        resetBest ()
        val rSq  = VEC [VectorD] ()                                          // QoF: R^2, R^2 Bar, sMAPE, R^2 cv
        val cols = LSET (0)                                                  // start with x_0 in model (e.g., intercept)
        updateQoF (rSq, cross, select0 (qk))                                 // update Qof results for 0-th variable

        banner (s"forwardSelAll: (l = 0) INITIAL variable (0, ${fname(0)}) => cols = $cols")

        breakable {
            for l <- 1 until x.dim2 do
                val best = forwardSel (cols)                                 // add most predictive variable
                if best.col == -1 then break ()                              // could not find variable to add
                updateBest (best)
                cols += best.col                                             // add variable x_j
                updateQoF (rSq, cross, best)                                 // update QoF results for l-th variable
                val (jj, jj_qof) = (best.col, best.qof(qk))
                banner (s"forwardSelAll: (l = $l) ADD variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")
            end for
        } // breakable

        (cols, rSq)
    end forwardSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the relative importance of selected variables, ordered highest to
     *  lowest, rescaled so the highest is one.
     *  @param cols  the selected columns/features/variables
     *  @param rSq   the matrix R^2 values (stand in for sse)
     */
    def importance (cols: Array [Int], rSq: MatrixD): Array [(Int, Double)] =
        val r2  = rSq(?, 0)                                                  // use column 0 for R^2
        val imp = Array.ofDim [(Int, Double)] (r2.dim)                       // for variables, except intercept
        val sf  = 1.0 / (r2(1) - r2(0))                                      // scale factor, so most important = 1
        imp(0)  = (cols(0), -0.0)
        for j <- 1 until imp.size do imp(j) = (cols(j), sf * (r2(j) - r2(j-1)))   // scaled improvement in R^2 (2 => cv)
        imp                                                                  // return the importance
    end importance

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform backward elimination to find the least predictive variable to remove
     *  from the existing model, returning the variable to eliminate, the new parameter
     *  vector and the new Quality of Fit (QoF).  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols   the columns of matrix x currently included in the existing model
     *  @param first  first variable to consider for elimination
     *                      (default (1) assume intercept x_0 will be in any model)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def backwardElim (cols: LSET [Int], first: Int = 1)(using qk: Int): BestStep =
        var best = BestStep ()()                                             // best step so far

        for j <- first until x.dim2 if cols contains j do
            val cols_j = cols diff LSET (j)                                  // try removing variable/column x_j
            val x_cols = x(?, cols_j)                                        // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols, newFname (fname, cols_j))       // regress with x_j added

            val (x_tr, y_tr) = (x_cols(t_rng), y(t_rng))                     // get full/training data
            mod_j.train (x_tr, y_tr)                                         // train model
            best = best.better (j, mod_j.test (x_tr, y_tr)._2, mod_j, cols_j)   // which is better
        end for

        if best.col == -1 then
            flaw ("backwardElim", "could not find a variable x_j to eliminate: best.col = -1")
        best
    end backwardElim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the full model before variable elimination as a starting point for
     *  backward elimination.
     *  @param qk  index of Quality of Fit (QoF) to use for comparing quality
     */
    def fullModel (qk: Int): BestStep =
        val mod_a = buildModel (x, fname)                                    // regress with all variables x_j

        val (x_tr, y_tr) = (x(t_rng), y(t_rng))                              // get full/training data
        mod_a.train (x_tr, y_tr)                                             // train model
        val qof_a = mod_a.test (x_tr, y_tr)._2                               // get test qof for mod_a
        BestStep (-1, qof_a, mod_a, mcols)(qof_a(qk))                        // result for full only
    end fullModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform BACKWARD ELIMINATION to find the LEAST predictive variables to remove
     *  from the full model, returning the variables left and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param first  first variable to consider for elimination
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def backwardElimAll (first: Int = 1, cross: String = "many")(using qk: Int):
                        (LSET [Int], VEC [VectorD]) =
        resetBest ()
        val rSq  = VEC [VectorD] ()                                          // R^2, R^2 Bar, sMAPE, R^2 cv
        val cols = mcols                                                     // start with all x_j in model
        val rem  = VEC [Int] ()                                              // start with no columns removed

        val best0 = fullModel (qk)                                           // start with all columns
        updateBest (best0)
        updateQoF (rSq, cross, best0)                                        // update QoF results for full model
        val jj_qof = best0.qof(qk)
        debug ("backwardElimAll", s"(l = 0) INITIAL variables (all) => cols = $cols @ $jj_qof")

        breakable {
            for l <- 1 until x.dim2 - 1 do                                   // l indicates number of variables eliminated
                val best = backwardElim (cols, first)                        // remove least predictive variable
                if best.col == -1 then break ()                              // could not find variable to remove
                updateBest (best)
                cols -= best.col                                             // remove variable x_j
                rem  += best.col                                             // keep track of removed columns
                updateQoF (rSq, cross, best)                                 // update QoF results
                val (jj, jj_qof) = (best.col, best.qof(qk))
                debug ("backwardElimAll", s"(l = $l) REMOVE variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")
            end for
        } // breakable

        updateQoF (rSq, cross, select0 (qk))                                 // update Qof results for 0-th variable
        rem += cols.max                                                      // remove last non-zero column
        rem += 0                                                             // remove column 0

        (LSET.from (rem.reverse), rSq.reverse)                               // reverse the order results
    end backwardElimAll 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform STEPWISE SELECTION to find a GOOD COMBINATION of predictive variables to have
     *  in the model, returning the variables selected and the new Quality of Fit (QoF)
     *  measures for all steps.  At each step it calls forwardSel and backwardElim
     *  and takes the best of the two actions.  Stops when neither action yields improvement.
     *  @see `Fit` for index of QoF measures.
     *  @see `modeling.FeatureSelection` for GIVENS for qk and slack_base
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param swap   whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def stepwiseSelAll (cross: String = "many", swap: Boolean = true)(using qk: Int):
                       (LSET [Int], VEC [VectorD]) =
        resetBest ()
        val rSq    = VEC [VectorD] ()                                        // QoF: R^2, R^2 Bar, sMAPE, R^2 cv
        val cols   = LSET (0)                                                // start with x_0 in model
        var last_q = Fit.extreme (qk)                                        // current best QoF
        val vars   = VEC [Int] ()

        banner (s"stepwiseSelAll: (qk = $qk, l = 0) INITIAL variable (0, ${fname(0)}) => cols = $cols")

        breakable {
            for l <- 1 until x.dim2 - 1 do
                val bestf = forwardSel (cols)                                // add most predictive variable OR
                val bestb = backwardElim (cols, 1)                           // remove least predictive variable
                debug ("stepwiseSelAll", s"bestf = $bestf, bestb = $bestb")

                val slack = slack_base / l~^2                                // increase slack-base amgnitude to include more features
                                                                             // slack => likely to ADD features at the beginning

                if (bestb.col == -1 || (bestf ge bestb.qof(qk) - slack)) &&     // forward as good as backward
                   (bestf.col != -1 && (bestf gt last_q - slack)) then          // a better model has been found
                    updateBest (bestf)
                    vars  += bestf.col
                    cols  += bestf.col                                       // ADD variable bestf.col
                    last_q = bestf.qof(qk)
                    updateQoF (rSq, cross, bestf)                            // update QoF results
                    println (s"\nstepwiseSelAll: (l = $l) ADD variable $bestf")
                    val (jj, jj_qof) = (bestf.col, last_q)
                    banner (s"stepwiseSelAll: (l = $l) ADD variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")

                else if bestb.col != -1 && (bestb gt last_q) then            // a better model has been found
                    updateBest (bestb)
                    vars  += bestb.col
                    cols  -= bestb.col                                       // REMOVE variable bestb.col 
                    last_q = bestb.qof(qk)
                    updateQoF (rSq, cross, bestb)                            // update QoF results
                    println (s"\nstepwiseSelAll: (l = $l) REMOVE variable $bestb")
                    val (jj, jj_qof) = (bestb.col, last_q)
                    banner (s"stepwiseSelAll: (l = $l) REMOVE variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")

                else
                    if ! swap then break ()
                    val (out, in) = (bestb.col, bestf.col)
                    val bestfb = swapVars (cols, out, in, qk)
                    updateBest (bestfb)
                    if out != -1 && in != -1 && (bestfb gt last_q) then      // a better model has been found
                        vars  += bestb.col
                        vars  += bestf.col
                        cols  -= bestb.col                                   // REMOVE variable bestb.col (swap out)
                        cols  += bestf.col                                   // ADD variable bestf.col (swap in)
                        last_q = bestfb.qof(qk)
                        updateQoF (rSq, cross, bestfb)                       // update QoF results
                        println (s"\nstepwiseSelAll: (l = $l) SWAP variable $bestb with $bestf")
                    else
                        println (s"\nstepwiseSelAll: (l = $l) last_q = $last_q better ($bestb, $bestf)")
                        break ()                                             // can't find a better model -> quit
                end if

                val x_cols = x(?, cols)                                      // x projected onto cols columns
                val mod_   = buildModel (x_cols, newFname (fname, cols))     // regress on this x
                mod_.train ()                                                // train model
                println (mod_.report (mod_.test ()._2))                      // test and report
            end for
        } // breakable

        println (s"stepwiseSelAll: selected features = $cols")
        println (s"stepwiseSelAll: selected features = ${cols.map (fname (_))}")
        println (s"stepwiseSelAll: features in/out   = $vars")

        (cols, rSq)
//      (cols, rSq(1 until cols.size))
    end stepwiseSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap out variable with in variable.
     *  @param cols  the columns of matrix x currently included in the existing model
     *  @param out   the variable to swap out
     *  @param in    the variable to swap in
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    def swapVars (cols: LSET [Int], out: Int, in: Int, qk: Int): BestStep =
        val cols_  = cols diff LSET (out) union LSET (in)  // swap out var with in var
        val x_cols = x(?, cols_)                                             // x projected onto cols_j columns
        val mod_j  = buildModel (x_cols)                                     // regress with x_out removed and x_in added
        mod_j.train ()                                                       // train model
        val qof_in = mod_j.test ()._2
        BestStep (in, qof_in, mod_j)(qof_in(qk))                             // candidate step
    end swapVars

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
    def beamSelAll (cross: String = "many", bk: Int = 3)(using qk: Int): (LSET [Int], VEC [VectorD]) =

        // FIX - to be implemented

        null       
    end beamSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Variance Inflation Factor (VIF) for each variable to test
     *  for multi-collinearity by regressing x_j against the rest of the variables.
     *  A VIF over 50 indicates that over 98% of the variance of x_j can be predicted
     *  from the other variables, so x_j may be a candidate for removal from the model.
     *  Note:  override this method to use a superior regression technique.
     *  @param skip  the number of columns of x at the beginning to skip in computing VIF
     */
    def vif (skip: Int = 1): VectorD =
        val vifV = new VectorD (x.dim2 - skip)                               // VIF vector for x columns except skip columns
        if vifV.dim == 1 then { vifV(0) = 1.0; return vifV }                 // no other variables
        for j <- skip until x.dim2 do
            val x_j   = x(?, j)                                              // column j vector
            val x_noj = x.not (?, j)                                         // all columns except j matrix                   
            val mod_j = new Regression (x_noj, x_j)                          // regress with x_j removed
            mod_j.train ()                                                   // train model
            val rSq_j = (mod_j.test ()._2)(QoF.rSq.ordinal)                  // R^2 for predicting x_j
            if rSq_j.isNaN then Fac_LU.diagnoseMat (x_noj)                   // check for problems with matrix
//          debug ("vif", s"for variable x_$j, rSq_$j = $rSq_j")
            vifV(j-1) =  1.0 / (1.0 - rSq_j)                                 // store vif for x_1 in vifV(0)
        end for
        vifV
    end vif

//  T E S T I N G   S C E N A R I O S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform In-Sample Testing, i.e., train and test on the same FULL data set.
     *  Return the prediction and the Quality of Fit.
     *  @param skip    the number of initial data points to skip (due to insufficient information)
     *  @param showYp  whether to show the prediction vector
     */
    def inSample_Test (skip: Int = 0, showYp: Boolean = false): (VectorD, VectorD) =
        banner (s"In-Sample Test: $modelName")
        val (x_, y_)  = (x.drop (skip), y.drop (skip))
        val (yp, qof) = trainNtest (x_, y_)(x_, y_)
        if showYp then
            println (s"Final In-Sample Prediction Vector yp = $yp")
        (yp, qof)
    end inSample_Test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the indices for the test-set for (1) RANDONLY or (2) FIRST.
     *  @see `scalation.mathstat.TnT_Split`
     *  @param n_test  the size of test-set
     *  @param rando   whether to select indices randomly or in blocks
     */
    inline def testIndices (n_test: Int, rando: Boolean): IndexedSeq [Int] =
        TnT_Split.testIndices (permGen, n_test, rando)
    end testIndices

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the indices for the test-set for (1) RANDONLY or (3) LAST
     *  @see `scalation.mathstat.TnT_Split`
     *  @param n_total  the size of full dataset
     *  @param n_test   the size of test-set
     *  @param rando    whether to select indices randomly or in blocks
     */
    inline def testIndices (n_total: Int, n_test: Int, rando: Boolean): IndexedSeq [Int] =
        TnT_Split.testIndices (permGen, n_total, n_test, rando)
    end testIndices

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use validation to compute test Quality of Fit (QoF) measures by dividing
     *  the full dataset into a TESTING-set and a TRAINING-set, returning qof and yp.
     *  The testing-set is defined by idx and the rest of the data is the training-set.
     *  Select the TESTING-set to be (@see `mathstat.TnT_Split`)
     *      1. RANDOM pass rando = true
     *      2. FIRST  pass rando = false and
     *                     idx   = testIndices ((ratio * y.dim).toInt, rando)
     *      3. LAST   pass rando = false and
     *                     idx   = testIndices (y.dim, (ratio * y.dim).toInt, rando)
     *      4. CUSTOM pass rando = false and
     *                     idx   = indices specified by user
     *  @caveat:  loosely uses In-Sample "Degrees of Freedom", models may override to correct this.
     *            @see `Regression.validate`
     *  @param rando  flag indicating whether to use randomized or simple validation
     *  @param ratio  the ratio of the TESTING-set to the full dataset (most common 70-30 (.3), 80-20 (.2))
     *  @param idx    the prescribed TESTING-set indices (default => generate)
     */
    def validate (rando: Boolean = true, ratio: Double = Model.TE_RATIO)
//               (idx: IndexedSeq [Int] = testIndices ((ratio * y.dim).toInt, rando)):
                 (idx: IndexedSeq [Int] = testIndices (y.dim, (ratio * y.dim).toInt, rando)):
                 (VectorD, VectorD) =
        debug ("validate", s"n_test = ${(ratio * y.dim).toInt}, rando = $rando")
        val (x_e, x_, y_e, y_) = TnT_Split (x, y, idx)                       // Test-n-Train Split

        train (x_, y_)                                                       // train model on the TRAINING-set
        val (yp, qof) = test (x_e, y_e)                                      // test on TESTING-set and get its yp and QoF measures
        if qof(QoF.sst.ordinal) <= 0.0 then                                  // requires variation in TESTING-set
            flaw ("validate", "chosen testing set has no variability")
        println (FitM.fitMap (qof, QoF.values.map (_.toString)))
        (yp, qof)
    end validate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use k-fold cross-validation to compute test Quality of Fit (QoF) measures
     *  by iteratively dividing the full dataset into a TESTING set and a TRAINING set.
     *  Each test set is defined by idx and the rest of the data is the training set.
     *  @see showQofStatTable in `Fit` object for printing the returned stats.
     *  @param k      the number of cross-validation iterations/folds (defaults to 5x).
     *  @param rando  flag indicating whether to use randomized or simple cross-validation
     */
    def crossValidate (k: Int = 5, rando: Boolean = true): Array [Statistic] =
        if k < MIN_FOLDS then flaw ("crossValidate", s"k = $k must be at least $MIN_FOLDS")
        val stats   = Fit.qofStatTable                                       // create table for QoF measures
        val fullIdx = if rando then permGen.igen                             // permuted indices
                      else VectorI.range (0, y.dim)                          // ordered indices
        val sz      = y.dim / k                                              // size of each fold
        val ratio   = 1.0 / k                                                // fraction of dataset used for testing

        for fold <- 0 until k do
            banner (s"crossValidate: fold $fold: train-test splits sizes = (${y.dim - sz}, $sz)")
            val idx = fullIdx (fold * sz until (fold+1) * sz).toMuIndexedSeq   // instance indices for this fold
            val qof = validate (rando, ratio)(idx)._2
            debug ("crossValidate", s"fold $fold: qof = $qof")
            if qof(QoF.sst.ordinal) > 0.0 then                               // requires variation in test-set
                for q <- qof.indices do stats(q).tally (qof(q))              // tally these QoF measures
        end for

        stats
    end crossValidate

end Predictor


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Predictor` companion object provides a method for testing predictive models.
 */
object Predictor:

    private val LIMIT   = 5000                                               // do not plot more than 5000 points

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make the PREDICTION INTERVALS for the given model.
     *  @param mod  the model making the predictions
     *  @param xx   the aligned data/input matrix to use (test/full)
     *  @param yy   the aligned actual response/output vector to use (test/full)
     *  @param yp   the corresponding vector of predicted values
     */
    def makePredictionInt (mod: Predictor & Fit, xx: MatrixD,
                           yy: VectorD, yp: VectorD): (MatrixD, MatrixD) = 
        banner (s"Prediction Intervals for ${mod.modelName}")
        val l_u           = mod.PIbounds (yp, mod.predictInt_ (xx))          // make PI lower and upper bound matrices from yp and ihw
        val (qof_all, iα) = mod.diagnose_pi (yy, yp, l_u)                    // compute metrics for both point and interval predictions
        mod.showQoF (qof_all)                                                // show all the QoF metrics
        plotPredictionInt (yy, yp, MatrixD.at (l_u, iα), mod.modelName)      // plot ordered actual, predicted, lower, upper
        l_u
    end makePredictionInt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Plot the actual and predicted values/vectors both ordered by increasing yy values.
     *  @param yy      the aligned actual response/output vector to use (test/full)
     *  @param yp      the corresponding vector of predicted values
     *  @param mName   the model name
     *  @param order   whether to order all vectors by y-actual
     *  @param doPlot  whether to plot y-actual vs. predictions
     */
    def plotPrediction (yy: VectorD, yp: VectorD, mName: String,
                        order: Boolean = true, doPlot: Boolean = true): Unit =
        if doPlot then
            val r  = 0 until min (yy.dim, LIMIT)                             // limited index range
            var ys = (yy(r), yp(r))                                          // slice to LIMIT
            if order then ys = orderByY (ys._1, ys._2)                       // order all by yy
            new Plot (null, ys._1, ys._2,                                    // plot ordered actual, predicted
                      s"Plot $mName predictions: yy black/actual vs. yp red/predicted", lines = true)
    end plotPrediction

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Plot the PREDICTION INTERVALS with all vectors ordered by increasing yp values.
     *  @param yy      the aligned actual response/output vector to use (test/full)
     *  @param yp      the corresponding vector of predicted values
     *  @param low_up  the predicted (lower, upper) bound vectors
     *  @param mName   the model name
     *  @param order   whether to order all vectors by y-actual
     *  @param doPlot  whether to plot y-actual vs. predictions as well as prediction intervals
     */
    def plotPredictionInt (yy: VectorD, yp: VectorD,
                           low_up: (VectorD, VectorD), mName: String,
                           order: Boolean = true, doPlot: Boolean = true): Unit =
        if doPlot then
            val r  = 0 until min (yy.dim, LIMIT)                             // limited index range
            var ys = MatrixD (yp(r), yy(r), low_up._1(r), low_up._2(r))      // slice to LIMIT and order all by yp
            if order then ys = orderByY (ys)                                 // order all by yy
            new PlotM (null, ys, Array ("yp", "yy", "low", "up"),            // plot ordered actual, predicted, lower, upper
                       s"Plot $mName prediction intervals [low, up]", lines = true)
    end plotPredictionInt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test (in-sample) by training and testing on the FULL dataset.
     *  Test (out-of-sample) by training on the TRAINING set and testing on the TESTING set.
     *  @param mod    the model to be used
     *  @param ext    the model subtype extension (e.g., indicating the transformation function used)
     *  @param check  whether to check the assertion that the in-sample and out-of-sample results
     *                are in rough agreement (e.g., at 20%)
     */
    def test (mod: Predictor, ext: String = "", check: Boolean = true): Unit =
        val iq = QoF.rSq.ordinal
        banner (s"Test ${mod.modelName} $ext")
        val qof = mod.trainNtest ()()._2                                     // train and test the model on full dataset (in-sample)

        println ("Validate: Out-of-Sample Testing")
        val qof2 = mod.validate ()()._2                                      // train on training-set, test on testing-set
        if check then assert (rel_diff (qof(iq), qof2(iq)) < 0.2)            // check agreement of in-sample and out-of-sample results
        println (FitM.fitMap (qof2, QoF.values.map (_.toString)))
    end test

end Predictor


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `predictorTest` main function is used to test the `Predictor` trait
 *  and its derived classes using the `Example_AutoMPG` dataset containing
 *  data matrices x, ox and response vector y.
 *  Shift imports for the Example_BasketBall or Example_BPressure datasets.
 *  @see `Example_AutoMPG_Correlation
 *  > runMain scalation.modeling.predictorTest
 */
@main def predictorTest (): Unit =

    import Example_AutoMPG._
//  import Example_BasketBall._
//  import Example_BPressure._
    import Predictor.test

    val fname_6  = Array ("modelyear")                                       // modelyear has highest positive correlation
    val fname_4  = Array ("weight")                                          // weight has highest correlation magnitude
    val fname_04 = Array ("intercept", "weight")
//  val hp2      = Regression.hp                                             // the hyper-parameters of Regression

    test (new NullModel (y), check = false)                                  // 1
    test (new SimplerRegression (ox(?, LSET (6)), y, fname_6))               // 2
    test (new SimpleRegression (ox(?, LSET (0, 4)), y, fname_04))            // 3
    test (new Regression (x, y, x_fname))                                    // 4 - no intercept
    test (new Regression (ox, y, ox_fname))                                  // 5
    test (RidgeRegression.center (x, y, x_fname))                            // 6 - no intercept
    test (LassoRegression.center (x, y, x_fname))                            // 7 - no intercept
    test (new RegressionWLS (ox, y, ox_fname))                               // 8
/*
    test (new TranRegression (ox, y, ox_fname, hp2, id, id), "id")           // 10 - id
    test (new TranRegression (ox, y, ox_fname, hp2, sqrt, sq), "sqrt")       // 11 - sqrt
    test (new TranRegression (ox, y, ox_fname, hp2, cbrt, cb), "cbrt")       // 12 - cbrt
    test (new TranRegression (ox, y, ox_fname), "log")                       // 13 - log
    test (TranRegression (ox, y, ox_fname), "box-cox")                       // 14 - box-cox
*/
    test (SymbolicRegression.quadratic (x, y, x_fname))                      // 15
    test (SymbolicRegression.quadratic (x, y, x_fname, true))                // 16
    test (SymbolicRegression.cubic (x, y, x_fname))                          // 17
    test (SymbolicRegression.cubic (x, y, x_fname, true))                    // 18
    test (SymbolicRegression (x, y, x_fname, LSET (-2.0, -1, 2, 3, 4)))      // 19
    test (new PolyRegression (ox(?, LSET (4)), y, 4, fname_4))               // 20
    test (new PolyORegression (ox(?, LSET (4)), y, 4, fname_4))              // 21 
    test (new TrigRegression (ox(?, LSET (4)), y, 8, fname_4))               // 22
    test (new ExpRegression (ox, y, ox_fname))                               // 23
    test (new KNN_Regression (x, y, x_fname), "k=3")                         // 25
    KNN_Regression.hp("kappa") = 5
    test (new KNN_Regression (x, y, x_fname), "k=5")                         // 24
    KNN_Regression.hp("kappa") = 7
    test (new KNN_Regression (x, y, x_fname), "k=7")                         // 26
    test (RegressionCat (oxr, y, 6, xr_fname))                               // 27 - include the origin cat. col.

end predictorTest

