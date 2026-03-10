
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jan 21 19:16:18 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Predictor for Matrix Input, Matrix Output
 */

package scalation
package modeling
package neuralnet

import scala.collection.mutable.{ArrayBuffer, IndexedSeq, LinkedHashSet => LSET}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMV` trait provides a framwork for multiple predictive analytics
 *  techniques, e.g., Multi-variate Regression and Neural Netoworks.
 *  x is multi-dimensional [1, x_1, ... x_k] and so is y.
 *  Fit the `NetParam` parameters bb in for example the regression equation
 *      y  =  f(bb dot x) + e
 *  bb is an array of `NetParam` where each component is a weight matrix and
 *  a bias vector.
 *  @see `NetParam`
 *  @param x       the input/data m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model or use bias)
 *  @param y       the response/output m-by-ny matrix
 *  @param fname   the feature/variable names (if null, use x_j's)
 *  @param hparam  the hyper-parameters for the model/network
 */
trait PredictorMV (x: MatrixD, y: MatrixD, protected var fname: Array [String],
                   hparam: HyperParameter)
      extends Model
         with FeatureSelection:

    protected val DO_PLOT = true                                             // whether to plot y vs yp
    private   val debug   = debugf ("PredictorMV", true)                     // debug function
    private   val flaw    = flawf ("PredictorMV")                            // flaw function

    if x != null then
        if x.dim != y.dim then flaw ("init", "row dimensions of x and y are incompatible")
        if x.dim2 < 1 then flaw ("init", s"dim2 = ${x.dim2} of the x matrix must be at least 1")
        if x.dim <= x.dim2 then
            flaw ("init", s"PredictorMV requires more rows ${x.dim} than columns ${x.dim2}")

    private val MIN_FOLDS = 3                                                // minimum number of folds for cross validation
    private val stream    = 0                                                // random number stream to use
    private val permGen   = TnT_Split.makePermGen (y.dim, stream)            // permutation generator

    protected var bb: Array [NetParam]  = null                               // array of network parameters - init/alloc in class
    protected var e: MatrixD            = null                               // residual/error matrix

    if x != null && fname == null then fname = x.indices2.map ("x" + _).toArray  // default feature/variable names

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the set of columns (numbers) for the features in this model.
     */
    def mcols: LSET [Int] = LSET.range (0, getX.dim2)  

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used data matrix x.  Mainly for derived classes where x is expanded
     *  from the given columns in x_.
     */
    def getX: MatrixD = x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response vector y (first colum in matrix).
     */
    def getY: VectorD = y(?, 0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response matrix y.  Mainly for derived classes where y is
     *  transformed.
     */
    override def getYY: MatrixD = y

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
     *  matrix and y_ is the response/output matrix.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training
     *  dataset.  Training involves estimating the model parameters bb.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     */
    def train (x_ : MatrixD = x, y_ : MatrixD = y): Unit

    def train (x_ : MatrixD, y_ : VectorD): Unit =                           // first column only
        train (x_, MatrixD (y_).transpose)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The train2 method should work like the train method, but should also
     *  optimize hyper-parameters (e.g., shrinkage or learning rate).
     *  Only implementing classes needing this capability should override this method.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     */
    def train2 (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        throw new UnsupportedOperationException ("train2: not supported - no hyper-parameters to optimize")
    end train2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF matrix.
     *  Each variable predictions and QoF values are returned in columns of respective matrices.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output matrix (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : MatrixD = y): (MatrixD, MatrixD)

    def test (x_ : MatrixD, y_ : VectorD): (VectorD, VectorD) =               // first column only
        val (yp, qof) = test (x_, MatrixD (y_).transpose)
        (yp(?, 0), qof(?, 0))
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.
     *  FIX - currently must override if y is transformed, @see `TranRegression`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output matrix (defaults to full y)
     */
    def trainNtest (x_ : MatrixD = x, y_ : MatrixD = y)
                   (xx: MatrixD = x, yy: MatrixD = y): (MatrixD, MatrixD) =
        train (x_, y_)
        val (yp, qof) = test (xx, yy)
        println (report (qof))
        if DO_PLOT then makePlots (yy, yp)
        (yp, qof)
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.  This version does auto-tuning.
     *  FIX - currently must override if y is transformed, @see `TranRegression`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output matrix (defaults to full y)
     */
    def trainNtest2 (x_ : MatrixD = x, y_ : MatrixD = y)
                    (xx: MatrixD = x, yy: MatrixD = y): (MatrixD, MatrixD) =
        train2 (x_, y_)
        val (yp, qof) = test (xx, yy)
        println (report (qof))
        if DO_PLOT then makePlots (yy, yp)
        (yp, qof)
    end trainNtest2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make plots for each output/response variable (column of matrix y).
     *  Must override if the response matrix is transformed or rescaled.
     *  @param yy  the testing/full actual response/output matrix (defaults to full y)
     *  @param yp  the testing/full predicted response/output matrix (defaults to full y)
     */
    def makePlots (yy: MatrixD, yp: MatrixD): Unit =
        val (ryy, ryp) = orderByYY (yy, yp)                               // order by yy
        for k <- ryy.indices2 do
            new Plot (null, ryy(?, k), ryp(?, k), s"$modelName: y$k black/actual vs. red/predicted", lines = true)
    end makePlots

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a basic report on a trained and tested multi-variate model.
     *  @param ftMat  the matrix of qof values produced by the `Fit` trait
     */
    override def report (ftMat: MatrixD): String =
        s"""
REPORT
    ----------------------------------------------------------------------------
    modelName  mn  = $modelName
    ----------------------------------------------------------------------------
    hparameter hp  = $hparameter
    ----------------------------------------------------------------------------
    features   fn  = ${stringOf (getFname)}
    ----------------------------------------------------------------------------
    parameter  bb  = ${stringOf (parameters)}
    ----------------------------------------------------------------------------
    fitMap     qof = ${Fit.showFitMap (ftMat)}
    ----------------------------------------------------------------------------
        """
    end report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Order vectors y_ and yp_ accroding to the ascending order of y_.
     *  @param y_   the vector to order by (e.g., true response values)
     *  @param yp_  the vector to be order by y_ (e.g., predicted response values)
     */
    def orderByY (y_ : VectorD, yp_ : VectorD): (VectorD, VectorD) =
        val rank = y_.iqsort                                                  // rank order for vector y_
        (y_.reorder (rank), yp_.reorder (rank))                               // (y_ in ascending order, yp ordered by y_)
    end orderByY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z,
     *  e.g., (b_0, b_1, b_2) dot (1, z_1, z_2).
     *  Must override when using transformations, e.g., `ExpRegression`.
     *  @param z  the new vector to predict
     */
    def predict (z: VectorD): VectorD                                         // = b dot z

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of vector y = f(x_, b), e.g., x_ * b for `Regression`.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    def predict (x_ : MatrixD): MatrixD =
        MatrixD (for i <- x_.indices yield predict (x_(i)))
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the hyper-parameters.
     */
    def hparameter: HyperParameter = hparam

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return only the first matrix of parameter/coefficient values.
     */
    def parameter: MatrixD = bb(0).toMatrixD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the array of network parameters (weight matrix, bias vector) bb.
     */
    def parameters: NetParams = bb

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the matrix of residuals/errors.
     */
    def residual: MatrixD = e

//  F E A T U R E   S E L E C T I O N

    // @see givens in `modeling.FeatureSelection`

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  Override for models that support feature section.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): PredictorMV & Fit

    protected val USE_MEAN = true                                             // use mean vs. first of qof for feature selection
    private   var theBest  = BestStep ()()                                    // record the best model from feature selection
    private   val t_rng   = if fullset_FS then 0 until y.dim                  // use full dataset for Feature Selection (FS)
                            else 0 until Model.trSize (y.dim)                 // use training set for Feature Selection (FS)

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
     *  Note: for QoF where smaller if better, must switch to '<'.
     *  @param best  new best-step found during feature selection
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    private def updateBest (best: BestStep)(using qk: Int): Unit =
        if best.qof != null then
            if theBest.qof == null || (best gt theBest.qof(qk)) then theBest = best
    end updateBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the rSq-based QoF results for the l-th iteration.
     *  @param rSq    the matrix contain information about r-Sq-based QoF measures
     *  @param l      the l-th iteration
     *  @param cross  indicator of whether cross-validation are to be included
     *  @param fit_l  the fit vector for the l-th iteration
     *  @param mod_l  the predictive model for the l-th iteration
     *  // FIX - wrong param & can remove?
    private def updateQoF (rSq: MatrixD, l: Int, cross: Boolean, best: BestStep): Unit =
        rSq(l) =
            if cross then
                Fit.qofVector (best.qof, best.mod.crossValidate ())           // results for model mod_l, with cross-validation
            else
                Fit.qofVector (best.qof, null)                                // results for model mod_l, no cross-validation
    end updateQoF
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform forward selection to find the most predictive variable to add the
     *  existing model, returning the variable to add and the new model.
     *  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols  the columns of matrix x currently included in the existing model
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSel (cols: LSET [Int])(using qk: Int): BestStep =
        var best = BestStep ()()                                              // best step so far

        for j <- x.indices2 if ! (cols contains j) do
            val cols_j = cols union LSET (j)                                  // try adding variable/column x_j
            val x_cols = x(?, cols_j)                                         // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols, newFname (fname, cols_j))        // regress with x_j added

            val (x_tr, y_tr) = (x_cols(t_rng), y(t_rng))                      // get full/training data
            mod_j.train (x_tr, y_tr)                                          // train model
            val qof = mod_j.test (x_tr, y_tr)._2                              // get test qof for mod_j
            if USE_MEAN then
                best = best.better (j, qof.meanRow, mod_j, cols_j)            // which is better based on mean of all targets
            else
                best = best.better (j, qof(?, 0), mod_j, cols_j)              // which is better based of first target
        end for

        if best.col == -1 then
            flaw ("forwardSel", "could not find a variable x_j to add: best.col = -1")
        best
    end forwardSel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform FORWARD SELECTION to find the MOST predictive variables to have
     *  in the model, returning the variables added and the new Quality of Fit (QoF)
     *  measures for all steps.
     *  @see `Fit` for index of QoF measures.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def forwardSelAll (cross: String = "many")(using qk: Int): (LSET [Int], MatrixD) =
        resetBest ()
        val rSq  = new MatrixD (x.dim2 - 1, Fit.qofVectorSize)                // QoF: R^2, R^2 Bar, smape, R^2 cv
        val cols = LSET (0)                                                   // start with x_0 in model
//      updateQoF (rSq, 0, cross, select0 (qk))                               // update Qof results for 0-th variable FIX?

        banner (s"forwardSelAll: (l = 0) INITIAL variable (0, ${fname(0)}) => cols = $cols")

        breakable {
            for l <- 1 until x.dim2 do
                val best = forwardSel (cols)                                  // add most predictive variable
                if best.col == -1 then break ()                               // could not find variable to add
                updateBest (best)
                cols += best.col                                              // add variable x_j
                updateQoF (rSq, l-1, cross, best)                             // update QoF results
                val (jj, jj_qof) = (best.col, best.qof(qk))
                banner (s"forwardSelAll: (l = $l) ADD variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")
            end for
        } // breakable

        (cols, rSq(0 until cols.size-1))
    end forwardSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform backward elimination to find the least predictive variable to remove
     *  from the existing model, returning the variable to eliminate, the new parameter
     *  matrix and the new Quality of Fit (QoF).  May be called repeatedly.
     *  @see `Fit` for index of QoF measures.
     *  @param cols   the columns of matrix x currently included in the existing model
     *  @param first  first variable to consider for elimination
     *                      (default (1) assume intercept x_0 will be in any model)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def backwardElim (cols: LSET [Int], first: Int = 1)(using qk: Int): BestStep =
        var best = BestStep ()()                                              // best step so far

        for j <- first until x.dim2 if cols contains j do
            val cols_j = cols diff LSET (j)                                   // try removing variable/column x_j
            val x_cols = x(?, cols_j)                                         // x projected onto cols_j columns
            val mod_j  = buildModel (x_cols, newFname (fname, cols_j))        // regress with x_j added

            val (x_tr, y_tr) = (x_cols(t_rng), y(t_rng))                      // get full/training data
            mod_j.train (x_tr, y_tr)                                          // train model
            val qof = mod_j.test (x_tr, y_tr)._2                              // get test qof for mod_j
            if USE_MEAN then
                best = best.better (j, qof.meanRow, mod_j, cols_j)            // which is better based on mean of all targets
            else
                best = best.better (j, qof(?, 0), mod_j, cols_j)              // which is better based of first target
        end for

        if best.col == -1 then
            flaw ("backwardElim", "could not find a variable x_j to eliminate: best.col = -1")
        best
    end backwardElim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the full model before variable elimination as a starting point for
     *  backward elimination.
     */
    private def fullModel (qk: Int): BestStep =
        val mod_a = buildModel (x, fname)                                     // regress with all variables x_j

        val (x_tr, y_tr) = (x(t_rng), y(t_rng))                               // get full/training data
        mod_a.train (x_tr, y_tr)                                              // train model
        val qof_a = mod_a.test (x_tr, y_tr)._2                                // get test qof for mod_a
        val qof_ = if USE_MEAN then
            qof_a.meanRow                                                     // results for full model based on mean of all targets
        else
            qof_a(?, 0)                                                       // results for full model based of first target
        BestStep (-1, qof_, mod_a)(qof_(qk))                                  // return best step
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
    def backwardElimAll (first: Int = 1, cross: String = "many")(using qk: Int): (LSET [Int], MatrixD) =
        resetBest ()
        val rSq  = new MatrixD (x.dim2 - 1, Fit.qofVectorSize)                // R^2, R^2 Bar, smape, R^2 cv
        val cols = LSET.range (0, x.dim2)                                     // start with all x_j in model

        val best0 = fullModel(qk)
        updateQoF (rSq, 0, cross, best0)                                      // update QoF results for full model
        val jj_qof = best0.qof(qk)
        banner (s"backwardElimAll: (l = 0) INITIAL variables (all) => cols = $cols @ $jj_qof")

        breakable {
            for l <- 1 until x.dim2 - 1 do                                    // l indicates number of variables eliminated
                val best = backwardElim (cols, first)                         // remove least predictive variable
                if best.col == -1 then break ()                               // could not find variable to remove
                updateBest (best)
                cols -= best.col                                              // remove variable x_j
                updateQoF (rSq, l, cross, best)                               // update QoF results
                val (jj, jj_qof) = (best.col, best.qof(qk))
                banner (s"backwardElimAll: (l = $l) REMOVE variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")
            end for
        } // breakable

        (cols, rSq.reverse)                                                   // reverse the order results
    end backwardElimAll 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform STEPWISE SELECTION to find a GOOD COMBINATION of predictive variables to have
     *  in the model, returning the variables selected and the new Quality of Fit (QoF)
     *  measures for all steps.  At each step it calls 'forwardSel' and 'backwardElim'
     *  and takes the best of the two actions.  Stops when neither action yields improvement.
     *  @see `Fit` for index of QoF measures.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def stepwiseSelAll (cross: String = "many", swap: Boolean = true)(using qk: Int):
                       (LSET [Int], MatrixD) =
        resetBest ()
        val rSq    = new MatrixD (x.dim2 - 1, Fit.qofVectorSize)              // QoF: R^2, R^2 Bar, smape, R^2 cv
        val cols   = LSET (0)                                                 // start with x_0 in model
        var last_q = Fit.extreme (qk)                                         // current best QoF
        val vars   = ArrayBuffer [Int]()

        banner (s"stepwiseSelAll: (l = 0) INITIAL variable (0, ${fname(0)}) => cols = $cols")

        breakable {
            for l <- 1 until x.dim2 - 1 do
                val bestf = forwardSel (cols)                                 // add most predictive variable OR
                val bestb = backwardElim (cols, 1)                            // remove least predictive variable
                debug ("stepwiseSelAll", s"bestf = $bestf, bestb = $bestb")

                if (bestb.col == -1 || (bestf ge bestb.qof(qk))) &&           // forward as good as backward
                   (bestf.col != -1 && (bestf gt last_q)) then                // a better model has been found
                    updateBest (bestf)
                    vars  += bestf.col
                    cols  += bestf.col                                        // ADD variable bestf.col
                    last_q = bestf.qof(qk)
                    updateQoF (rSq, l, cross, bestf)                          // update QoF results
                    println (s"\nstepwiseSelAll: (l = $l) ADD variable $bestf")
                    val (jj, jj_qof) = (bestf.col, last_q)
                    banner (s"stepwiseSelAll: (l = $l) ADD variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")

                else if bestb.col != -1 && (bestb gt last_q) then             // a better model has been found
                    updateBest (bestb)
                    vars  += bestb.col
                    cols  -= bestb.col                                        // REMOVE variable bestb.col 
                    last_q = bestb.qof(qk)
                    updateQoF (rSq, l, cross, bestb)                          // update QoF results
                    println (s"\nstepwiseSelAll: (l = $l) REMOVE variable $bestb")
                    val (jj, jj_qof) = (bestb.col, last_q)
                    banner (s"stepwiseSelAll: (l = $l) REMOVE variable ($jj, ${fname(jj)}) => cols = $cols @ $jj_qof")

                else
                    if ! swap then break ()
                    val (out, in) = (bestb.col, bestf.col)
                    val bestfb = swapVars (cols, out, in, qk)
                    updateBest (bestfb)
                    if out != -1 && in != -1 && (bestfb gt last_q) then       // a better model has been found
                        vars  += bestb.col
                        vars  += bestf.col
                        cols  -= bestb.col                                    // REMOVE variable bestb.col (swap out)
                        cols  += bestf.col                                    // ADD variable bestf.col (swap in)
                        last_q = bestfb.qof(qk)
                        updateQoF (rSq, l, cross, bestfb)                     // update QoF results
                        println (s"\nstepwiseSelAll: (l = $l) SWAP variable $bestb with $bestf")
                    else
                        break ()                                              // can't find a better model -> quit
                end if
            end for
        } // breakable

        println (s"stepwiseSelAll: selected features = $cols")
        println (s"stepwiseSelAll: selected features = ${cols.map (fname (_))}")
        println (s"stepwiseSelAll: features in/out   = $vars")

        (cols, rSq(1 until cols.size))
    end stepwiseSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap out variable with in variable.
     *  @param cols  the columns of matrix x currently included in the existing model
     *  @param out   the variable to swap out
     *  @param in    the variable to swap in
     */
    private def swapVars (cols: LSET [Int], out: Int, in: Int, qk: Int): BestStep =
        val cols_  = cols diff LSET (out) union LSET (in)                     // swap out var with in var
        val x_cols = x(?, cols_)                                              // x projected onto cols_j columns
        val mod_j  = buildModel (x_cols, newFname (fname, cols_))             // regress with x_out removed and x_in added
        mod_j.train ()                                                        // train model
        val qof_in = mod_j.test ()._2(?, 0)
        BestStep (in, qof_in, mod_j)(qof_in(qk))                              // candidate step
    end swapVars

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform BEAM SEARCH SELECTION to find a GOOD COMBINATION of predictive features/variables to
     *  have in the model, returning the top k sets of features/variables selected and the new Quality of
     *  Fit (QoF) measures/metrics for all steps.  At each step, iterate over the models in the beam
     *  (top k) and create candidates by adding features (phase 1) and then removing (phase 2).
     *  From all the candidates, keep the best k and start a new iteration.  Stops when there is
     *  no improvement in any of top k (or the maximum number of features is reached.
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param bk     the beam width holding the top k models (defaults to 3)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def beamSelAll (cross: String = "many", bk: Int = 3)(using qk: Int): (LSET [Int], MatrixD) =

        // FIX -- to be implemented

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
        val vifV = new VectorD (x.dim2 - skip)                                // VIF vector for x columns except skip columns
        for j <- skip until x.dim2 do
            val x_j   = x(?, j)                                               // column j vector
            val x_noj = x.not (?, j)                                          // all columns except j matrix                   
            val mod_j = new Regression (x_noj, x_j)                           // regress with x_j removed
            mod_j.train ()                                                    // train model
            val rSq_j = (mod_j.test ()._2)(QoF.rSq.ordinal)                   // R^2 for predicting x_j
            if rSq_j.isNaN then Fac_LU.diagnoseMat (x_noj)                    // check for problems with matrix
//          debug ("vif", s"for variable x_$j, rSq_$j = $rSq_j")
            vifV(j-1) =  1.0 / (1.0 - rSq_j)                                  // store vif for x_1 in vifV(0)
        end for
        vifV
    end vif

//  T E S T I N G   S C E N A R I O S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform In-Sample Testing, i.e., train and test on the full data set.
     *  @param skip    the number of initial data points to skip (due to insufficient information)
     *  @param showYp  whether to show the prediction vector
     */
    def inSample_Test (skip: Int = 0, showYp: Boolean = false): Unit =
        val (x_, y_) = (x.drop (skip), y.drop (skip))
        val yp = trainNtest (x_, y_)(x_, y_)._1
        if showYp then
            println (s"Final In-Sample Prediction Vector yp = $yp")
    end inSample_Test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the indices for the test-set.
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
     *  the full dataset into a TESTING-set and a TRAINING-set.
     *  The test set is defined by idx and the rest of the data is the training set.
     *  @see `modeling.Predictor.validate` about the RANDOM, FIRST, and LAST options
     *  for selecting the testing-set.
     *  @param rando  flag indicating whether to use randomized or simple validation
     *  @param ratio  the ratio of the TESTING-set to the full dataset (most common 70-30, 80-20)
     *  @param idx    the prescribed TESTING-set indices (default => generate)
     */
    def validate (rando: Boolean = true, ratio: Double = Model.TE_RATIO)
//               (idx: IndexedSeq [Int] = testIndices ((ratio * y.dim).toInt, rando)):
                 (idx: IndexedSeq [Int] = testIndices (y.dim, (ratio * y.dim).toInt, rando)):
                 (MatrixD, MatrixD) =
        debug ("validate", s"n_test = ${(ratio * y.dim).toInt}, rando = $rando")
        val (x_e, x_, y_e, y_) = TnT_Split (x, y, idx)                        // Test-n-Train Split

        train (x_, y_)                                                        // train model on the training set
        val (yp, qof) = test (x_e, y_e)                                       // test on test-set and get QoF measures
        if qof(QoF.sst.ordinal)(0) <= 0.0 then                                // requires variation in test-set
            flaw ("validate", "chosen testing set has no variability")
//      println (FitM.fitMap (qof, QoF.values.map (_.toString)))
        (yp, qof)
    end validate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert QoF results into an array (of size 1) of `Statistic` for compatibility
     *  with the `crossValidate` method.
     *  @param qof  the Quality of Fit (QoF) results
    def qof2Stat (qof: MatrixD): Array [Statistic] =
        val stats = Fit.qofStatTable                                          // create table for QoF measures
        if qof(QoF.sst.ordinal)(0) > 0.0 then                                 // requires variation in test-set
            for q <- qof.indices do stats(q).tally (qof(q)(0))                // tally these QoF measures
        stats
    end qof2Stat
     */

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
        val stats   = Fit.qofStatTable                                        // create table for QoF measures
        val fullIdx = if rando then permGen.igen                              // premuted indices
                      else VectorI.range (0, y.dim)                           // ordered indices
        val sz      = y.dim / k                                               // size of each fold
        val ratio   = 1.0 / k                                                 // fraction of dataset used for testing

        for fold <- 0 until k do
            val idx = fullIdx (fold * sz until (fold+1) * sz).toMuIndexedSeq  // instance indices for this fold
            debug ("crossValidate", s"fold $fold: test set size = $sz")
            val qof = validate (rando, ratio)(idx)._2
            debug ("crossValidate", s"fold $fold: qof = $qof")
            if qof(QoF.sst.ordinal)(0) > 0.0 then                             // requires variation in test-set
                for q <- qof.indices do stats(q).tally (qof(q)(0))            // tally these QoF measures
        end for

        stats
    end crossValidate

end PredictorMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PredictorMV` companion object provides a method for testing predictive
 *  models.
 */
object PredictorMV:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test (in-sample) by training and testing on the FULL dataset.
     *  Test (out-of-sample) by training on the TRAINING set and testing on the TESTING set.
     *  @param mod    the model to be used
     *  @param ext    the model subtype extension (e.g., indicating the transformation function used)
     *  @param check  whether to check the assertion that the in-sample and out-of-sample results
     *                are in rough agreement (e.g., at 20%)
     */
    def test (mod: PredictorMV, ext: String = "", check: Boolean = true): Unit =
        val iq = QoF.rSq.ordinal
        banner (s"Test ${mod.modelName} $ext")
        val qof = mod.trainNtest ()()._2                                      // train and test the model on full dataset (in-sample)

        println ("Validate: Out-of-Sample Testing")
        val qof2 = mod.validate ()()._2                                       // train on training set, test on testing set
        if check then assert (rel_diff (qof(iq)(0), qof2(iq)(0)) < 0.2)       // check agreement of in-sample and out-of-sample results
        println (Fit.showFitMap (qof2))
    end test

end PredictorMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `predictorMVTest` main function is used to test the `PredictorMV` trait
 *  and its derived classes using the `Example_Concrete` dataset containing
 *  data matrices x, ox and response matrix y.
 *  > runMain scalation.modeling.predictorMVTest
 */
@main def predictorMVTest (): Unit =

//  import Example_Concrete._

    println ("TBD")

end predictorMVTest

