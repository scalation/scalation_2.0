
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jan 17 15:04:21 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Abstract Class for Forecasters that utilize Regression
 *           Extending classes include ARY, ARX, ARX_Quad, ARX_SR, ...
 *
 *  @see `scalation.modeling.RidgeRegression` regularization needed due to multi-collinearity
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}
import scala.math.min
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

// Select via import the type of regularized regression:
// Ridge (L_2), Lasso (L_1), Bridge (L_q), RidgeBridge (L_2 and L_q)

import scalation.modeling.{CenteredRidge => REGRESSION}     // default
//import scalation.modeling.{RidgeRegression => REGRESSION}
//import scalation.modeling.{LassoRegression => REGRESSION}
//import scalation.modeling.{BridgeRegression => REGRESSION}
//import scalation.modeling.{RidgeBridgeRegression => REGRESSION}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Forecaster_Reg` abstract class provides base methods for use by extending classes
 *  that utilize regularized regression for time series forecasting.
 *  @param x             the data/input matrix (lagged columns of y and xe) @see `ARX.apply`
 *                           for regularization, the data will be centered (so NO INTERCEPT COLUMN)
 *  @param y             the response/output vector (time series data)
 *  @param hh            the maximum forecasting horizon (h = 1 to hh)
 *  @param fname         the feature/variable names
 *  @param tRng          the time range, if relevant (time index may suffice)
 *  @param hparam        the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast       whether a backcasted value is prepended to the time series (defaults to false)
 *  @param hasIntercept  whether the original model has an intercept (defaults to true)
 *  @param scaled        whether the regression standardizes the columns of x with the training
 *                           mean and standard deviation instead of only centering them, so the
 *                           ridge penalty is unit-free (defaults to false, @see `CenteredRidgeMV`)
 */
abstract class Forecaster_Reg (x: MatrixD, y: VectorD, hh: Int, fname: Array [String],
                               tRng: Range = null, var hparam: HyperParameter = AR.hp,
                               bakcast: Boolean = false, hasIntercept: Boolean = true,
                               scaled: Boolean = false)
      extends Forecaster (y, hh, tRng, hparam, bakcast, hasIntercept)
         with FeatureSelection:

    private val debug  = debugf ("Forecaster_Reg", false)                 // debug function
    private val flaw   = flawf ("Forecaster_Reg")                         // flaw function

    hparam = REGRESSION.hp ++ hparam
    protected val reg  = new REGRESSION (x, y, fname, hparam, scaled)     // delegate training to regularized regression
    protected val nneg = hparam("nneg").toInt == 1                        // 0 => unrestricted, 1 => predictions must be non-negative

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the set of columns (numbers) for the features in this model.
     */
    def mcols: LSET [Int] = LSET.range (0, getX.dim2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the data/input matrix built from lagged y (and optionally xe) values.
     */
    override def getX: MatrixD = x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the feature/variable names.  Overrides definition in `Forecast` trait.
     */
    override def getFname: Array [String] = fname

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit a `Forecaster_Reg` model to the times-series data y_ = f(x_).
     *  Estimate the coefficient vector b for a `Forecaster_Reg` model.
     *  Uses OLS Matrix Fatorization to determine the coefficients, e.g., the b (φ) vector.
     *  @param x_  the data/input matrix (e.g., full x)
     *  @param y_  the training/full response vector (e.g., full y)
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit =
        debug ("train", s"$modelName, x_.dims = ${x_.dims}, y_.dim = ${y_.dim}")
        val idx = y_.indexOf (NO_DOUBLE)                                  // index of first non-value
        val (x_t, y_t) = if idx < 0 then (x_, y_)
                         else (x_(0 until idx), y_(0 until idx))
        reg.train (x_t, y_t)                                              // train the regression model
        b = reg.parameter                                                 // coefficients from regression
        debug ("train", s"parameter vector b = $b")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *  @see `modeling.rectify` define in `Predictor.scala`
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    override def predict (t: Int, y_ : VectorD): Double =
        val yp = rectify (reg.predict (x(t)), nneg)
        if t < y_.dim then
            debug ("predict", s"@t = $t, b = $b dot x(t) = ${x(t)} = yp = $yp vs. y_ = ${y_(t)}")
        yp
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a new vector from the first spec values of x, the last p-h+1 values
     *  of x (past values) and recent values 1 to h-1 from the forecasts.
     *  @param xx  the t-th row of the input matrix (lagged actual values)
     *  @param yy  the t-th row of the forecast matrix (forecasted future values)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forge (xx: VectorD, yy: VectorD, h: Int): VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with rolling validation (analog of predict method).
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     */
    override def forecast (t: Int, y_ : VectorD = y): VectorD =
        val yh = new VectorD (hh)                                         // hold forecasts for each horizon
        for h <- 1 to hh do
            val xy   = if h == 1 then x(t) else forge (x(t), yf(t), h)    // pull past and prior forecasted values
            val pred = rectify (reg.predict (xy), nneg)                   // slide in prior forecasted values
//          debug ("forecast", s"h = $h, @t = $t, xy = $xy, yp = $pred, y_ = ${y_(t)}")
            yf(t, h) = pred                                               // record in forecast matrix
            yh(h-1)  = pred                                               // record forecasts for each horizon
        yh                                                                // return forecasts for all horizons
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the h-steps ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1.
     *  @see `forecastAll` method in `Forecaster` trait.
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the actual values to use in making forecasts
     */
    override def forecastAt (h: Int, y_ : VectorD = y): VectorD =
        if h < 2 then flaw ("forecastAt", s"horizon h = $h must be at least 2")

        for t <- y_.indices do                                            // make forecasts over all time points for horizon h
            val xy   = forge (x(t), yf(t), h)
            val pred = rectify (reg.predict (xy), nneg)
//          debug ("forecastAt", s"h = $h, @t = $t, xy = $xy, yp = $pred, y_ = ${y_(t)}")
            yf(t, h) = pred                                               // record in forecast matrix
        yf(?, h)                                                          // return the h-step ahead forecast vector
    end forecastAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Models need to provide a means for updating the Degrees of Freedom (DF):
     *  dfr (degrees of freedom taken by regression/model), df (degrees of freedom left for error).
     *  @note: Need the correct size taking into account whether initial rows were skipped.
     *  @note: Assume regularized regression where the intercept parameter is removed.
     *  @param size  the size of dataset (full, train, or test)
     * 
    override def mod_resetDF (size: Int): Unit =
        // In CenteredRidge, parameter.size is exactly the number of slopes (k).
        // So dfr (model df) is exactly parameter.size, regardless of intercept presence.
        val dfr = parameter.size.toDouble

        // Compute error df: size minus total parameters.
        // If hasIntercept is true, the total parameters estimated are slopes + 1 recovered intercept.
        val total_params = parameter.size + (if hasIntercept then 1 else 0)
        val df           = (size - total_params).toDouble

        debug ("mod_resetDF", s"dfr = $dfr, df = $df")
        resetDF (dfr, df)
    end mod_resetDF
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the Degrees of Freedom (DF).  The design matrix x has NO intercept
     *  column: the centered regression returns the k slopes as `parameter` and
     *  recovers the intercept from the training means, so k = parameter.size and
     *  the recovered intercept is counted by `resetDF_k` via `hasIntercept`
     *  (which must be true for a centered regression: the intercept is always there).
     *  @param size  the size of dataset (full, train, or test) after skipped rows
     */
    override def mod_resetDF (size: Int): Unit = resetDF_k (size, parameter.size)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix (defualt x, getX)
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     * 
    override def summary (x_ : MatrixD = x, fname_ : Array [String] = fname,
                          b_ : VectorD = b, vifs: VectorD = reg.vif ()): String =
        super.summary (x_, fname_, b_, vifs)                              // summary from `Fit`
    end summary
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary: the implicit intercept recovered by the centered
     *  regression followed by the slopes in original units (`reg.parameter_org`),
     *  with a column of ones prepended to x so `Fit.summary`'s standard errors and
     *  VIF indexing (column 0 = intercept) apply; the VIFs are those of the slope
     *  columns given an intercept (as in `ARX_D.summary`).  Reflects the last
     *  `diagnose` call.
     *  @param x_      the testing/full data/input matrix (no intercept column)
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters (null => intercept +: slopes)
     *  @param vifs    the Variance Inflation Factors (null => computed here)
     */
    override def summary (x_ : MatrixD = x, fname_ : Array [String] = fname,
                          b_ : VectorD = null, vifs: VectorD = null): String =
        val x1 = VectorD.one (x_.dim) +^: x_                              // column of ones for the implicit intercept
        val b1 = if b_ != null then b_ else reg.parameter_org
        val v1 = if vifs != null then vifs else new Regression (x1, y).vif ()   // one VIF per slope column
        super.summary (x1, "intercept" +: fname_, b1, v1)
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform In-Sample Testing, i.e. train and test on the full data set.
     *  Return the prediction and the Quality of Fit.
     *  @param showYf  whether to show the forecast matrix
     * 
    override def inSample_Test (showYf: Boolean = false): (VectorD, VectorD) =
        banner (s"In-Sample Test: $modelName")
        val (yp, qof) = trainNtest ()()                                   // train on full and test on full
        forecastAll ()                                                    // forecast over all horizons
        diagnoseAll (getY, getYf)                                         // compute metrics for all horizons
        if showYf then
            println (s"Final In-Sample Forecast Matrix yf = ${getYf}")
//          println (s"Final In-Sample Forecast Matrix yf = ${getYf.shiftDiag}")
        (yp, qof)
    end inSample_Test
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform Train-n-Test (TnT) Testing, i.e. train and test with rolling validation.
     *  @param rc      the retraining cycles (how often to retrain the model)
     *  @param showYf  whether to show the forecast matrix
     *
    override def tnT_Test (rc: Int = 5, showYf: Boolean = false): Unit =
        banner (s"TnT Test: $modelName")
        trainNtest ()()                                                   // initial training updated by `rollValidate`
        setSkip ()                                                        // diagnose: don't skip any rows
        rollValidate (rc)                                                 // TnT with Rolling Validation
        diagnoseAll (getY, getYf, Forecaster.teRng (y.dim))               // only diagnose on the testing set
        if showYf then
            println (s"Final TnT Forecast Matrix yf = ${getYf}")
//          println (s"Final TnT Forecast Matrix yf = ${getYf.shiftDiag}")
    end tnT_Test
     */

//  F E A T U R E   S E L E C T I O N

    // @see givens in `modeling.FeatureSelection`

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  Must be implemented for models that support feature selection.
     *  Otherwise, use @see `NoBuildModel
     *  @note: Forecasting models should use this method to build there own sub-models.  FIX.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): REGRESSION =
        reg.buildModel (x_cols, fname2)
    end buildModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a `Forecaster_Reg` model using the cols with the selected features,
     *  e.g., the best model after feature selection: `getModel (getBest.mod_cols)`.
     *  @param cols  the cols of the input matrix with selected features
     */
    def getModel (col: LSET [Int]): Forecaster_Reg

    private var theBest = BestStep ()()                                   // record the best model from Feature Selection (FS)
    private val t_rng   = if FeatureSelection.fullset_FS then 0 until y.dim  // use full dataset for Feature Selection (FS)
                          else 0 until Model.trSize (y.dim)                  // use training set for Feature Selection (FS)

    /** The rows feature selection fits on: t_rng cut off at the first `NO_DOUBLE`
     *  response value (a horizon h > 1 response ends h-1 rows early, @see `ARX_D.getModel_h`),
     *  so no undefined row can enter a sub-model's regression (as `train` guarantees
     *  for the full model).
     */
    private val fs_rng  =
        val end = y.indexOf (NO_DOUBLE)
        if end < 0 then t_rng else t_rng.start until min (t_rng.end, end)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the best-step to default
     */
    def resetBest (): Unit = theBest = BestStep ()()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the best step/model found from feature selection.
     *  @note:  Use `getModel` to turn it into a forecasting model.
     */
    def getBest: BestStep = theBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** When the new best-step is better than theBest, replace theBest.
     *  Note: for QoF where smaller if better, must switch to '<'.
     *  @param best  new best-step found during feature selection
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     *               defaults to smapeC, could try rSqBar could work better
     */
    private def updateBest (best: BestStep) (using qk: Int): Unit =
        if best.qof != null then
            if theBest.qof == null || (best gt theBest.qof(qk)) then theBest = best
    end updateBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a model from the given columns `col_j`, run it, and returning the new model
     *  and its qof.
     *  @param cols_j  the lags/columns to be included in the new model
     */
    private def buildNrun (cols_j: LSET [Int]): (Model_FS, VectorD) =
        val x_cols = x(?, cols_j)                                         // x projected onto cols_j columns
        val mod_j  = reg.buildModel (x_cols, cols_j.toArray.map (cname (_)))   // regress with x_j added (change: "reg.")
//      val (x_tr, y_tr) = (x_cols(t_rng), y(t_rng))                      // get full/training data
        val (x_tr, y_tr) = (x_cols(fs_rng), y(fs_rng))                    // get full/training data
        mod_j.train (x_tr, y_tr)                                          // train model
        val qof_j = mod_j.test (x_tr, y_tr)._2                            // test for qof
        (mod_j, qof_j)                                                    // return model and its qof
    end buildNrun

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Correct the QoF in `best`: re-diagnose the sub-model on the rows feature
     *  selection trains on (fs_rng: the training set, or the full dataset when
     *  `FeatureSelection.fullset_FS`), leaving out the first skp rows (their lags
     *  are clamped), in ORIGINAL space: when the data were transformed (`rescale`,
     *  `fromTerms` with a tFormT), `yForm.fi` maps both the responses and the
     *  sub-model's predictions back before `diagnose`, so the criterion compared,
     *  printed and stored by every selection step is the one the final model
     *  reports.  Diagnosing on fs_rng keeps the selection criterion free of the
     *  testing rows and comparable with the full model (`fullModel` is diagnosed
     *  on fs_rng as well).
     *  @param skp   the number of time-steps to skip at the beginning
     *  @param best  the best model selected so far
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    def correctQoF (skp: Int, best: BestStep)(using qk: Int): BestStep =
        if skp > 0 || yForm != null && best.mod != null then
            val bmod = best.mod.asInstanceOf [REGRESSION]
            val yp   = bmod.predict (bmod.getX (fs_rng))                  // predictions on the FS rows only
            val (yo, ypo) = if yForm == null then (y(fs_rng), yp)
                            else (yForm.fi (y(fs_rng)), yForm.fi (yp))       // back to original space
            val qof  = bmod.diagnose (yo.drop (skp), ypo.drop (skp), null)
            BestStep (best.col, qof, bmod, best.mod_cols)(qof(qk))
        else
            best
    end correctQoF

//  T H E  S E L E C T I O N   T E C H N I Q U E S   (@see `SubsetSearch`)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate one column set the way every selection step does: build and run
     *  the sub-model on the FS rows (`buildNrun`) and correct its QoF (`correctQoF`:
     *  skip rows dropped, original space).  The evaluator of every technique.
     *  @param cols  the columns of the set
     *  @param qk    index of Quality of Fit (QoF) to use for comparing quality
     */
    private def evalCols (cols: LSET [Int])(using qk: Int): BestStep =
        val (mod_j, qof_j) = buildNrun (cols)
        correctQoF (getSkip, BestStep (-1, qof_j, mod_j, cols)(qof_j(qk)))
    end evalCols

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the name of column j for the banners and traces: fname(j), or x_j
     *  when a model names fewer columns than its matrix has (e.g., `ARY_Quad`
     *  before its power terms were named), so a print never fails a selection.
     *  @param j  the column
     */
    private def cname (j: Int): String = if j < fname.length then fname(j) else s"x$j"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Record a search path: the best-so-far after each step into the rSq trace
     *  (@see `updateQoF`) and the best step found (@see `updateBest`).
     *  @param path   the search path
     *  @param rSq    the trace of key QoF measures, one row per step
     *  @param cross  indicator to include the cross-validation/validation QoF measure
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    private def record (path: SubsetSearch.Path, rSq: VEC [VectorD], cross: String)(using qk: Int): Unit =
        for step <- path.steps do updateQoF (rSq, cross, step)
        updateBest (path.best)
    end record

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform FORWARD SELECTION to find the MOST predictive variables to have
     *  in the model, returning the variables added and the new Quality of Fit (QoF)
     *  measures for all steps (@see `SubsetSearch.forward`: the full path from the
     *  intercept-only model, the best set along it is `getBest`).
     *  @see `modeling.Fit` for index of QoF measures.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    override def forwardSelAll (cross: String = "many")(using qk: Int): (Array [LSET [Int]], Array [VEC [VectorD]]) =
        resetBest ()
        val rSq  = VEC [VectorD] ()                                       // QoF: R^2, R^2 Bar, sMAPE, R^2 cv
        banner (s"forwardSelAll: (qk = $qk, l = 0) INITIAL variable (0, ${cname (0)}) => cols = ${LSET (0)}")
        val path = SubsetSearch.forward (x.dim2, evalCols, LSET (0))
        record (path, rSq, cross)
        for (step, l) <- path.steps.zipWithIndex if l > 0 do
            banner (s"forwardSelAll: (l = $l) ADD variable (${step.col}, ${cname (step.col)}) => cols = ${step.mod_cols} @ ${step.qof(qk)}")
        val cols = if path.steps.isEmpty then LSET (0) else path.steps.last.mod_cols
        (Array (cols), Array (rSq))
    end forwardSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform BACKWARD ELIMINATION to find the LEAST predictive variables to remove
     *  from the full model, returning the variables in reverse order of elimination
     *  and the new Quality of Fit (QoF) measures for all steps, from the smallest
     *  model up (@see `SubsetSearch.backward`: the full path down to two columns,
     *  the best set along it is `getBest`).
     *  @see `modeling.Fit` for index of QoF measures.
     *  @param first  first variable to consider for elimination (the leading `first` columns are kept)
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    override def backwardElimAll (first: Int = 1, cross: String = "none")(using qk: Int):
                                 (Array [LSET [Int]], Array [VEC [VectorD]]) =
        resetBest ()
        val rSq  = VEC [VectorD] ()                                       // R^2, R^2 Bar, sMAPE, sMAPEC
        val path = SubsetSearch.backward (x.dim2, evalCols, LSET.from (0 until first), minCols = 2)
        record (path, rSq, cross)
        for (step, l) <- path.steps.zipWithIndex do
            if l == 0 then debug ("backwardElimAll", s"(qk = $qk, l = 0) INITIAL variables (all) => cols = ${step.mod_cols} @ ${step.qof(qk)}")
            else debug ("backwardElimAll", s"(l = $l) REMOVE variable (${step.col}, ${cname (step.col)}) => cols = ${step.mod_cols} @ ${step.qof(qk)}")
        updateQoF (rSq, cross, correctQoF (getSkip, reg.select0 (qk)))   // the 0-th variable row (fs_rng, skip, original space)
        val rem  = VEC [Int] ()                                           // the columns removed, in order
        for step <- path.steps.drop (1) do rem += step.col
        val cols = if path.steps.isEmpty then reg.mcols else path.steps.last.mod_cols
        rem += cols.max                                                   // remove last non-zero column
        rem += 0                                                          // remove column 0
        (Array (LSET.from (rem.reverse)), Array (rSq.reverse))            // reverse the order results
    end backwardElimAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform STEPWISE SELECTION to find a GOOD COMBINATION of predictive variables to have
     *  in the model, returning the variables selected and the new Quality of Fit (QoF)
     *  measures for all steps (@see `SubsetSearch.stepwise`: at each step the best
     *  forward and the best backward move, a swap when neither improves, stop when
     *  no move improves; the slack rule of `Predictor.stepwiseSelAll`).
     *  @see `modeling.Fit` for index of QoF measures.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param swap   whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    override def stepwiseSelAll (cross: String = "none", swap: Boolean = true)(using qk: Int):
                                (Array [LSET [Int]], Array [VEC [VectorD]]) =
        resetBest ()
        val rSq  = VEC [VectorD] ()                                       // QoF: R^2, R^2 Bar, sMAPE, sMAPEC
        banner (s"stepwiseSelAll: (qk = $qk, l = 0) INITIAL variable (0, ${cname (0)}) => cols = ${LSET (0)}")
        val path = SubsetSearch.stepwise (x.dim2, evalCols, LSET (0), swap = swap)
        record (path, rSq, cross)
        var prev = 1                                                      // size of the previous set (intercept only)
        for (step, l) <- path.steps.zipWithIndex do
            val cols = step.mod_cols
            val move = if cols.size > prev then "ADD" else if cols.size < prev then "REMOVE" else "SWAP"
            banner (s"stepwiseSelAll: (l = ${l+1}) $move variable (${step.col}, ${cname (step.col)}) => cols = $cols @ ${step.qof(qk)}")
            prev = cols.size
        end for
        val cols = if path.steps.isEmpty then LSET (0) else path.steps.last.mod_cols
        println (s"stepwiseSelAll: selected features = $cols")
        println (s"stepwiseSelAll: selected features = ${cols.map (cname (_))}")
        (Array (cols), Array (rSq))
    end stepwiseSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform BEAM SEARCH SELECTION to find a GOOD COMBINATION of predictive features/variables to
     *  have in the model, returning the best set found and the new Quality of Fit (QoF)
     *  measures after each level (@see `SubsetSearch.beam`: phase I keeps the bk best single
     *  features, phase II expands every set of the beam by the move set (1,0), (2,0), (0,1),
     *  (0,2), (1,1), (2,1) and keeps the bk best, until patience runs out).
     *  @see `Fit` for index of QoF measures/metrics.
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "many")
     *  @param bk     the beam width holding the top k models (defaults to 8)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def beamSelAll (cross: String = "many", bk: Int = 8)(using qk: Int): (Array [LSET [Int]], Array [VEC [VectorD]]) =
        resetBest ()
        val rSq  = VEC [VectorD] ()                                       // QoF: R^2, R^2 Bar, sMAPE, R^2 cv
        updateQoF (rSq, cross, correctQoF (getSkip, reg.select0 (qk)))    // the 0-th variable row (fs_rng, skip, original space)
        banner (s"beamSelAll: (qk = $qk, l = 0) INITIAL variable (0, ${cname (0)}) => cols = ${LSET (0)}")
        val path = SubsetSearch.beam (x.dim2, evalCols, LSET (0), bk = bk)
        record (path, rSq, cross)
        for (step, l) <- path.steps.zipWithIndex do
            banner (s"beamSelAll: level $l best so far => cols = ${step.mod_cols} @ ${step.qof(qk)}")
        (Array (path.cols), Array (rSq))
    end beamSelAll

    private var lastProbs: VectorD = null                                 // the last CEM run's inclusion probabilities

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform CROSS-ENTROPY SELECTION (@see `SubsetSearch.cem`, `CEM_Selector`) to find
     *  a GOOD COMBINATION of predictive features/variables: each round samples column
     *  sets from per-column inclusion probabilities, evaluates each set as every other
     *  technique does (`evalCols`), keeps the elite sets and moves the probabilities
     *  towards them.  Returns the best set found and the QoF of the best-so-far model
     *  after each round; `cemProbs` returns the final inclusion probabilities.
     *  @see `Fit` for index of QoF measures.
     *  @param cross    indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param first    the number of leading columns always kept (default 1: the intercept x_0;
     *                  use 0 when x has no intercept column, e.g., a matrix built by `ARX_D.fromTerms`)
     *  @param prior    the prior inclusion probability per column of x (null => 0.5 each)
     *  @param nSample  the number of column sets sampled per round
     *  @param maxIter  the maximum number of rounds
     *  @param maxCols  the maximum number of columns in a sampled set (-1 => no cap)
     *  @param stream   the random number stream
     *  @param qk       index of Quality of Fit (QoF) to use for comparing quality
     */
    override def cemSelAll (cross: String = "none", first: Int = 1, prior: VectorD = null,
                            nSample: Int = CEM_Selector.N_SAMPLE, maxIter: Int = CEM_Selector.MAX_ITER,
                            maxCols: Int = -1, stream: Int = 0)(using qk: Int): (Array [LSET [Int]], Array [VEC [VectorD]]) =
        resetBest ()
        val rSq  = VEC [VectorD] ()                                       // QoF of the best-so-far model per round
        val path = SubsetSearch.cem (x.dim2, evalCols, LSET.from (0 until first), prior, nSample, maxIter, maxCols, stream)
        record (path, rSq, cross)
        lastProbs = path.probs
        if ! path.found then
            flaw ("cemSelAll", "no column set could be evaluated")
            return (Array (LSET [Int] ()), Array (rSq))
        val best = path.best
        banner (s"cemSelAll: ${path.nEval} column sets evaluated => cols = ${best.mod_cols} @ ${best.qof(qk)}")
        println (s"cemSelAll: selected features = ${best.mod_cols.map (cname (_))}")
        println (s"cemSelAll: inclusion probabilities = ${path.probs}")
        (Array (best.mod_cols), Array (rSq))
    end cemSelAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the inclusion probability per column left by the last `cemSelAll`
     *  run (null if none ran): columns near one are the stable core of the model.
     */
    def cemProbs: VectorD = lastProbs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the relative importance of selected variables, ordered highest to
     *  lowest, rescaled so the highest is one.
     *  @param cols  the selected columns/features/variables
     *  @param rSq   the matrix R^2 values (stand in for sse)
     */
    def importance (cols: Array [Int], rSq: MatrixD): Array [(Int, Double)] =
        reg.importance (cols, rSq)
    end importance

//  R E P O R T I N G   T H E   S E L E C T I O N

    type SelectedModel = Forecaster_Reg

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the forecaster on a selected column set (the hook of `reportSelection`): `getModel`.
     *  @param cols  the selected columns
     *  @param h     ignored (a single-horizon model forecasts every horizon recursively)
     */
    protected def selectedModel (cols: LSET [Int], h: Int): Forecaster_Reg = getModel (cols)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a selected forecaster (the hook of `reportSelection`) with the steps of
     *  `inSample_Test` (train on all the rows, forecast, diagnose) and/or of `tnT_Test`
     *  (train, `setSkip`, rolling validation with retraining every rc rows, diagnose
     *  on the testing rows); each QoF table has one row per horizon.
     *  Note: a sub-model (`getModel (cols)`) holds the selected columns only, while
     *  `forge` (the recursive forecasts for h > 1) assumes the full lag structure
     *  (@see the FIX of `buildModel`), so a selected forecaster with hh > 1 is
     *  tested at h = 1 only (one row): its predictions, rolled over the testing rows.
     *  @param mod       the selected forecaster
     *  @param inSample  whether to test in-sample
     *  @param tnt       whether to test with rolling validation
     *  @param rc        the retraining cycle of the rolling validation
     *  @param showYf    whether to print the forecast matrices
     */
    protected def testSelected (mod: Forecaster_Reg, inSample: Boolean, tnt: Boolean, rc: Int,
                                showYf: Boolean): (MatrixD, MatrixD) =
        var qofIn: MatrixD  = null
        var qofTnT: MatrixD = null
        val h1   = mod.getYf.dim2 > 2                                     // hh > 1 (yf has hh + 2 columns): h = 1 only (see the note)
        val note = if h1 then s" (h = 1 only: a sub-model's forge needs the full lag structure)" else ""

        if inSample then
            banner (s"$modelName: In-Sample Test of the selected model on ${stringOf (mod.getFname)}$note")
            val (_, qof) = mod.trainNtest ()()                            // train on all the rows, test on all the rows
            if h1 then
                qofIn = MatrixD (qof)                                     // QoF at h = 1
                println ("fitMap QoF = "); println (Fit.showFitMap (qofIn.ᵀ))
            else
                mod.forecastAll ()                                        // forecast over all horizons
                qofIn = mod.diagnoseAll (mod.getY, mod.getYf)             // QoF for all horizons
                if showYf then println (s"Forecast Matrix yf = ${mod.getYf}")

        if tnt then
            banner (s"$modelName: TnT Test (rolling validation, rc = $rc) of the selected model on ${stringOf (mod.getFname)}$note")
            mod.trainNtest ()()                                           // initial training updated by the rolling validation
            mod.setSkip ()                                                // diagnose: don't skip any rows
            if h1 then
                val (x, y) = (mod.getX, mod.getY)
                val teRng  = Forecaster.teRng (y.dim)                     // the testing rows
                val yp     = new VectorD (teRng.size)                     // their 1-step ahead predictions
                for i <- yp.indices do
                    val t = teRng.start + i                               // the next time point to predict
                    if i % rc == 0 then mod.train (x(i until t), y(i until t))     // retrain on the sliding training rows
                    yp(i) = mod.predict (t, y)
                end for
                mod.mod_resetDF (yp.dim)                                  // reset the Degrees of Freedom (DF)
                qofTnT = MatrixD (mod.diagnose (y(teRng), yp))            // QoF at h = 1 on the testing rows
                println ("fitMap QoF = "); println (Fit.showFitMap (qofTnT.ᵀ))
            else
                mod.rollValidate (rc)                                     // TnT with Rolling Validation
                qofTnT = mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (mod.getY.dim))   // the testing rows only
                if showYf then println (s"Forecast Matrix yf = ${mod.getYf}")
        (qofIn, qofTnT)
    end testSelected

end Forecaster_Reg

