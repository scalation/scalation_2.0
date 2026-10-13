
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Abstract Class for Forecasters with Matrix Input
 *           most models will need to override `train`, `predict`, `forecast` and `forecastAt`
 *
 *  @see `scalation.modeling.neuralnet.RidgeRegressionMV`
 */

package scalation
package modeling
package forecasting

import scala.annotation.unused
import scala.collection.mutable.{ArrayBuffer => VEC, LinkedHashSet => LSET}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import scalation.modeling.neuralnet.{CenteredRidgeMV => REGRESSION}     // default
//import scalation.modeling.neuralnet.{RidgeRegressionMV => REGRESSION}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Forecaster_D` abstract class provides a common framework for several forecasters.
 *  @note `Forecaster_D` is dependent on [[Forecaster_Reg]] class to do feature selection.
 *  Note, the `train` method must be called first followed by `test`.
 *  @param x             the input lagged time series data
 *  @param y             the response matrix (time series data per horizon)
 *  @param hh            the maximum forecasting horizon (h = 1 to hh)
 *  @param fname         the feature/variable names
 *  @param tRng          the time range, if relevant (index as time may suffice)
 *  @param hparam        the hyper-parameters for models extending this abstract class
 *  @param bakcast       whether a backcasted value is prepended to the time series (defaults to false)
 *  @param hasIntercept  whether the original model has an intercept (defaults to true)
 *                           -- only affects the degrees of freedom: the intercept itself is implicit,
 *                           recovered from the training means by the centered regression
 *  @param scaled        whether the regression standardizes the columns of x with the training
 *                           mean and standard deviation instead of only centering them, so the
 *                           ridge penalty is unit-free (defaults to false, @see `CenteredRidgeMV`)
 */
abstract class Forecaster_D (x: MatrixD, y: MatrixD, hh: Int, fname: Array [String],
                             tRng: Range = null, var hparam: HyperParameter = AR.hp,
                             bakcast: Boolean = false, hasIntercept: Boolean = true,
                             scaled: Boolean = false)
      extends Forecaster (y(?, 0), hh, tRng, hparam, bakcast, hasIntercept)   // no automatic backcasting, @see `ARY_D.apply`
         with SelectionReport: 

    private val debug = debugf ("Forecaster_D", false)                  // debug function

    println (s"Forecaster_D.init: hasIntercept = $hasIntercept")

//  protected val reg = REGRESSION.center (x, y, fname, hparam ++ REGRESSION.hp)  // delegate training to multi-variate regularized regression

    // delegate training to multi-variate regularized regression
    // it re-fits its centering/scaling on every window passed to `train`, so rolling windows never see later rows
    hparam = REGRESSION.hp ++ hparam
    protected val reg = new REGRESSION (x, y, fname, hparam, scaled)

    protected var bb: MatrixD = null                                    // use parameter matrix bb instead of vector b
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the set of columns (numbers) for the features in this model.
     */
    def mcols: LSET [Int] = LSET.range (0, getX.dim2)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the data/input matrix built from lagged y (and optionally xe) values.
     */
    override def getX: MatrixD = x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response/output matrix y.
     */
    def getYy: MatrixD = y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the feature/variable names.  Overrides definition in `Forecaster`
     */
    override def getFname: Array [String] = fname

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for horizon 1: the implicit intercept recovered by
     *  the centered regression (`reg.intercept`) followed by the slopes in
     *  original units (`reg.slopes`), with a column of ones prepended to x so
     *  `Fit.summary`'s standard errors and VIF indexing (column 0 = intercept)
     *  apply; the VIFs are those of the slope columns given an intercept.
     *  Reflects the last `diagnose` call (the last horizon after `diagnoseAll`).
     *  @param x_      the testing/full data/input matrix (no intercept column)
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters (null => intercept +: slopes for h = 1)
     *  @param vifs    the Variance Inflation Factors (null => computed here)
     */
    override def summary (x_ : MatrixD = x, fname_ : Array [String] = reg.getFname,
                          b_ : VectorD = null, vifs: VectorD = null): String =
        val x1 = VectorD.one (x_.dim) +^: x_                              // column of ones for the implicit intercept
        val b1 = if b_ != null then b_ else reg.intercept(0) +: reg.slopes(?, 0)
        val v1 = if vifs != null then vifs else new Regression (x1, getY).vif ()   // one VIF per slope column
        super.summary (x1, "intercept" +: fname_, b1, v1)
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a horizon-specific QoF summary (coefficient table) for the ARX_D-style
     *  report of a direct model.
     *  @param h  the given horizon
     */
    def summary (h: Int): String =
        require (1 <= h && h <= hh, s"summary: horizon h = $h out of range [1, $hh]")
        s"\nhorizon h = $h\n" + summary (b_ = reg.intercept(h-1) +: reg.slopes(?, h-1))
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit the model on a response VECTOR (as a one-column matrix).
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit = train (x_, MatrixD (y_).transpose)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit the regression on an ORIGINAL-space window (x_, y_): drop the
     *  trailing rows whose last horizon column is undefined (`NO_DOUBLE`, the
     *  last hh-1 rows), let `reg` refit its centering (and scaling) on the rows
     *  that remain and solve for the parameter matrix, then expose it as bb.
     *  Override for models that are not fit by `reg` (e.g., `NeuralNet_3L4TS`).
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output matrix (one column per horizon)
     */
    def train (x_ : MatrixD, y_ : MatrixD): Unit =
        debug ("train", s"$modelName, x_.dim = ${x_.dim}, y_.dim = ${y_.dim}")
        val idx = y_(?, y_.dim2-1).indexOf (NO_DOUBLE)                    // index of first non-value in the last column
        val (x_t, y_t) = if idx < 0 then (x_, y_) else (x_(0 until idx), y_(0 until idx))
        reg.train (x_t, y_t)                                              // train the multi-variate regression model
        bb = reg.parameter                                                // coefficients from regression
        debug ("train", s"parameter matrix bb = $bb")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the forecasting model y_ = f(y-past) + e and report its QoF
     *  and plot its predictions.  Return the predictions and QoF.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     *  @param xx  the testing/full data/input matrix
     *  @param yy  the testing/full response/output vector
     */
    def trainNtest (x_ : MatrixD, y_ : MatrixD)
                   (xx: MatrixD, yy: MatrixD): (VectorD, VectorD) =
        trainNtest_ (x_, y_)(xx, yy)
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test PREDICTIONS of a forecasting model y_ = f(lags (y_)) + e
     *  and return its predictions and  QoF vector.  Testing may be in-sample
     *  (on the training set) or out-of-sample (on the testing set) as determined
     *  by the parameters passed in.  Note: must call train before test.
     *  Must override to get Quality of Fit (QoF).
     *  @param x_  the data/input matrix
     *  @param y_  the actual testing/full response/output matrix
     */
    def test (x_ : MatrixD, y_ : MatrixD): (VectorD, VectorD) =
        val m = y_.dim
        predictAll (y_)                                                 // make all predictions - saved in yf
        debug ("test", s"x_.dims = ${x_.dims}, y_.dims, ${y_.dims}, yf.dims = ${yf.dims}")

        val y0  = y_(?, 0)                                              // actual values from column 0
        val yf1 = yf(0 until m, 1)                                      // forecasted values for h = 1

        if DO_PLOT then
            new Plot (null, y0, yf1, s"test: Plot of y0, yf1 for $modelName vs. t", true)
        mod_resetDF (m - getSkip)                                       // reset the degrees of freedom
        (yf1, diagnose (y0, yf1))                                       // return predicted and QoF vectors
    end test
/*
    def test (x_ : MatrixD, y_ : MatrixD): (VectorD, VectorD) =
        val m = y_.dim
        predictAll (y_)                                                 // make all predictions - saved in yf
        debug ("test", s"x_.dims = ${x_.dims}, y_.dims, ${y_.dims}, yf.dims = ${yf.dims}")

        val y0  = y_(0 until m, 0)                                      // actual values (except last) for h = 1
        val yf1 = yf(0 until m, 1)                                      // forecasted values for h = 1
        if DO_PLOT then
            new Plot (null, y0, yf1, s"test: Plot of y0, yf1 for $modelName vs. t", true)
        mod_resetDF (y0.dim)                                            // reset the degrees of freedom
        (yf1, diagnose (y0, yf1))                                       // return predicted and QoF vectors
    end test
*/

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
    /** Reset the Degrees of Freedom (DF).  As in `Forecaster_Reg`: x has no
     *  intercept column, `parameter` (= bb(?, 0)) holds the k slopes and the
     *  centered regression recovers the intercept, counted via `hasIntercept`.
     *  @param size  the size of dataset (full, train, or test) after skipped rows
     */
    override def mod_resetDF (size: Int): Unit = resetDF_k (size, parameter.size)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of parameter/coefficient values (they are model specific).
     *  Override for models with other parameters besides bb(?, 0).
     */
    override def parameter: VectorD = bb(?, 0)                          // parameter vector (first column)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the implicit intercepts in ORIGINAL units, one per horizon, recovered
     *  by the centered regression from the means of the last training window
     *  (@see `CenteredRidgeMV.intercept`).  Must be called after `train`.
     */
    def intercept: VectorD = reg.intercept

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the slopes in ORIGINAL units (k-by-hh: one column per horizon), so
     *  that forecast (t) = intercept + slopes dot x(t) (@see `CenteredRidgeMV.slopes`).
     *  Must be called after `train`.
     */
    def slopes: MatrixD = reg.slopes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a vector of values for y_t using the 1 to h-steps ahead forecasts.
     *
     *      y_t = b_0 + b_1 y_t-1 + b_2 y_t-2 + ... + b_p y_t-p = b dot x_t
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions (ignored)
     */
    def predict (t: Int, y_ : MatrixD): VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict all values (for all horizons) corresponding to the given time series vector y_.
     *  Create FORECAST MATRIX yf and return it.
     *  Note `forecastAll` simply returns the values produced by `predictAll`.
     *  @param y_  the actual time series values to use in making predictions
     */
    def predictAll (y_ : MatrixD): MatrixD =
        for t <- 0 until y_.dim do
            val pred = predict (t, y_)
//          debug ("predictAll", s"pred = $pred")
            yf(t, 1 until hh+1) = pred                                  // FIX - yf for VAR is different
        yf
    end predictAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points and all horizons (1 through hh-steps ahead).
     *  Record these in the FORECAST MATRIX yf, where
     *
     *      yf(t, h) = h-steps ahead forecast for y_t
     *
     *  @param y_  the actual values to use in making forecasts (ignored, needed for signature match)
     */
    override def forecastAll (@unused y_ : VectorD = yb): MatrixD = yf

//  T E S T I N G   S C E N A R I O S

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use rolling-validation to compute test Quality of Fit (QoF) measures
     *  by dividing the dataset into a TRAINING SET (tr) and a TESTING SET (te).
     *  as follows:  [ <-- tr_size --> | <-- te_size --> ]
     *  Calls forecast for h-steps ahead out-of-sample forecasts.
     *  Return the forecasted values in the FORECAST MATRIX,.
     *  @param rc_      the retraining cycle (number of forecasts until retraining occurs)
     *  @param growing  whether the training grows as it roll or kepps a fixed size
     *  @param doPlot   whether to show the plots
     */
    override def rollValidate (rc: Int = 5, growing: Boolean = false, doPlot: Boolean = false): MatrixD =
        banner (s"rollValidate: Evaluate ${modelName}'s QoF for horizons 1 to $hh:")

        val yf      = getYf                                             // get the full in-sample forecast matrix
        val te_size = Model.teSize (y.dim)                              // size of testing set
        val tr_size = Model.trSize (y.dim)                              // size of initial training set
        debug ("rollValidate", s"y.dims = ${y.dims}, train: tr_size = $tr_size; test: te_size = $te_size, rc = $rc")

        val yp = new VectorD (te_size)
        for i <- yp.indices do                                          // iterate through testing set
            val (is, t) = (if growing then 0 else i, tr_size + i)       // start, next time point to forecast
            if i % rc == 0 then
                val x_ = if x != null then x(is until t) else null
                train (x_, y(is until t))                               // retrain on sliding training set
                debug ("rollValidate", s"retrain on i = $i, rc = $rc, bb = $bb")

            yp(i)  = predict (t, y)(0)                                  // predict the next value (only for h=1)
            val yd = forecast (t, y(?, 0))                              // forecast the next hh-values, yf is updated
            debug ("rollValidate", s"yf(t, 0) = ${yf(t, 0)}, yp(i) = ${yp(i)}, yd = $yd")
        end for

        if doPlot then
            val (t, y_) = align (tr_size, y(?, 0))
            new Plot (t, y_, yp, s"rollValidate: Plot y_, yp vs. t for $modelName", lines = true)
        yf
    end rollValidate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform In-Sample Testing, i.e., train and test on the full data set.
     *  Return the prediction and the Quality of Fit.
     *  @note: Must override from `Forecaster` due to DIRECT forecasting.
     *  @param showYf  whether to show the forecast matrix
     */
    override def inSample_Test (showYf: Boolean = false): (VectorD, VectorD) =
        banner (s"In-Sample Test: $modelName")
        val (yp, qof) = trainNtest (x, y)(x, y)                           // train on full and test on full
        forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
        banner (s"diagnoseAll: $modelName")
        diagnoseAll (getY, getYf)                                         // compute metrics for all horizons
        if showYf then
            println (s"Final In-Sample Forecast Matrix yf = ${getYf}")
//          println (s"Final In-Sample Forecast Matrix yf = ${getYf.shiftDiag}")
        (yp, qof)
    end inSample_Test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform Train-n-Test (TnT) Testing, i.e. train and test with rolling validation.
     *  @note: Must override from `Forecaster` due to DIRECT forecasting.
     *  @param rc      the retraining cycles (how often to retrain the model)
     *  @param showYf  whether to show the forecast matrix
     */
    override def tnT_Test (rc: Int = 5, showYf: Boolean = true): Unit =
        banner (s"TnT Test: $modelName")
        trainNtest (x, y)(x, y)                                           // initial training updated by `rollValidate`
        forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
        setSkip ()                                                        // diagnose: don't skip any rows
        val yf_ = rollValidate (rc)                                       // TnT with Rolling Validation
        debug ("tnT_Test", s"Roll's TnT Forecast Matrix yf_ = $yf_")
        banner (s"diagnoseAll: $modelName")
        diagnoseAll (getY,getYf, Forecaster.teRng (y.dim))                // only diagnose on the testing set
        if showYf then
            println (s"Final TnT Forecast Matrix yf = ${getYf}")
//          println (s"Final TnT Forecast Matrix yf = ${getYf.shiftDiag}")
    end tnT_Test

//  F E A T U R E   S E L E C T I O N

    import SelectionTech._

// Note: `selectFeatures` and `selectFeaturesPerH` return ARRAYS (one column set and trace, or one per
//       horizon); `selectFeaturesAtH` returns the column set and trace (the model is `getModel_h`);
//       `reportSelection` prints and tests any of them, in-sample and/or TnT, one QoF table per test

// Note: the three selections of a direct multi-horizon model and what they return:
//       `selectFeatures` -- ONE column set for all the horizons (h = 0: every candidate set is
//       fit as ONE multi-horizon model and scored over ALL horizons by `agg`; h >= 1: the
//       single-horizon model at h decides, its columns are shared by every horizon), the
//       selected multi-horizon model's regression in `getBest`;
//       `selectFeaturesAtH` -- the column set of the single-horizon `Forecaster_Reg` at h (Dr. Miller's);
//       `selectFeaturesPerH` -- one column set PER HORIZON (renamed).
//       Every candidate set is scored on the FS rows in ORIGINAL space with the skip rows
//       dropped (@see `Forecaster_Reg.correctQoF`); the algorithms are `SubsetSearch`'s; with
//       qk = smapeC and h = 0 the criterion is the selection score J of the `llmsr` structure
//       search (the FPE-form sMAPE of the companion object, mean over horizons), so the base's
//       models and the search select on the same criterion through the same algorithms.

    private var theBest: BestStep = null                                // the best step of the last selection
    private var lastProbs: VectorD = null                               // the last CEM selection's probabilities

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the best step/model of the last feature selection run.  After `selectFeatures`
     *  its `mod` is the regression of the selected multi-horizon model (fitted on the
     *  FS rows, feature names kept) and `mod_cols` its columns, so
     *  `getModel (getBest.mod_cols)` is the selected forecaster -- as `getBest.mod`
     *  and `getModel (getBest.mod_cols)` are for a `Forecaster_Reg`.  After
     *  `selectFeaturesAtH` (or `selectFeaturesPerH`: its last horizon) it is the
     *  best step of that single-horizon model.
     */
    def getBest: BestStep =
        if theBest == null then
            throw new IllegalStateException ("Forecaster_D.getBest: no feature selection has run, so theBest is null")
        theBest
    end getBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the inclusion probability per column left by the last CEM selection
     *  (`selectFeatures` or `selectFeaturesAtH` with `SelectionTech.CEM`), or null
     *  if none ran (@see `Forecaster_Reg.cemProbs`).
     */
    def cemProbs: VectorD = lastProbs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build an `Forecaster_D` model using the cols with the selected features.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel (cols: LSET [Int] = mcols): Forecaster_D

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a single-horizon `Forecaster_Reg` model using the cols with the selected features.
     *  Note: uses `Forecaster_Reg` as it is the base model for Forecaster_D.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel_h (cols: LSET [Int] = mcols, h: Int = 1): Forecaster_Reg

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit the multi-horizon model of a column set on the FS rows (the training
     *  rows, or all rows when `FeatureSelection.fullset_FS`) and return it with its
     *  QoF matrix (one row per horizon) on those rows in ORIGINAL space with the
     *  first `getSkip` rows left out -- the evaluation of every candidate set of
     *  `selectFeatures` (h = 0) and the fit behind `getBest` (h >= 1).
     *  @param cols  the columns of the set
     */
    private def fitCols (cols: LSET [Int]): (Forecaster_D, MatrixD) =
        val rng = if FeatureSelection.fullset_FS then 0 until y.dim else 0 until Model.trSize (y.dim)
        val sub = getModel (cols)                                        // this model's transforms and scaling
        sub.setSkip (getSkip)
        sub.train (sub.getX (rng), sub.getYy (rng))
        sub.predictAll (sub.getYy)
        sub.forecastAll ()
        val qof = new MatrixD (hh, Fit.N_QoF)
        for h <- 1 to hh do
            val yy  = sub.getY (rng.start + h - 1 until rng.end)          // align the actual response values
            val yfh = sub.getYf (?, h)(rng.start until rng.end - h + 1)    // align column h of the forecast matrix
            sub.mod_resetDF (yy.dim - sub.getSkip)
            qof(h-1) = sub.diagnose (yy, yfh)                              // original space via `yForm`
        end for
        (sub, qof)
    end fitCols

    // Note: `selectFeaturesPerH` selects one column-set PER HORIZON
    //       `selectFeatures` selects one column-set for all of them

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform feature selection for this multi-horizon model, returning the columns
     *  selected and the QoF measures for all steps, as `selectFeatures` of a
     *  `Forecaster_Reg` does (@see `FeatureSelection.selectFeatures`) -- as ARRAYS of
     *  ONE column set and ONE trace, the shape `selectFeaturesPerH` returns per horizon,
     *  so `reportSelection` prints and tests both alike; the selected model is `getBest`
     *  (its regression, fitted on the FS rows) and `getModel (getBest.mod_cols)` (the
     *  forecaster, to train or roll-validate).
     *  Two ways to score a candidate column set:
     *  h = 0  -- ALL HORIZONS JOINTLY: every set is fit as ONE multi-horizon model
     *            (`fitCols`: this model's transforms and scaling, the FS rows, per-horizon
     *            QoF in original space with the skip rows dropped) and scored by `agg` over
     *            the horizons (default `Forecaster_D.aggFPE`: the mean of every measure, the
     *            sMAPE-IC slot holding the mean FPE-form sMAPE, so with qk = smapeC the
     *            criterion is the selection score J of the `llmsr` search).
     *            The trace `rSq` holds the best-so-far per step (aggregated measures); no
     *            cross-validation column (the candidates are forecasters, not the regressions
     *            `updateQoF`'s validation expects); a set whose fit fails is skipped.
     *  h >= 1 -- AT ONE HORIZON: the single-horizon model at h decides (@see
     *            `selectFeaturesAtH`: `Forecaster_Reg`'s selection on `getModel_h (h)`, so
     *            `cross` applies and the trace is that model's) and its columns are shared
     *            by every horizon; `getBest` then carries that step's QoF with the
     *            multi-horizon regression of the columns.
     *  @param tech    the selection technique (Forward, Backward, Stepwise, Beam, CEM)
     *  @param cross   indicator to include the cross-validation/validation QoF measure (h >= 1 only)
     *  @param first   the number of leading columns always kept (1 => the intercept column 0;
     *                     0 => none, e.g., a matrix built by `ARX_D.fromTerms`)
     *  @param swap    whether stepwise may swap a column in for one out
     *  @param h       the horizon whose single-horizon model decides (1 to hh), or 0 => all
     *                     horizons jointly (defaults to 0)
     *  @param prior   the prior inclusion probability per column (CEM only; null => 0.5 each)
     *  @param stream  the random number stream (CEM only, h = 0)
     *  @param bk      the beam width (h = 0; the single-horizon model's own default otherwise)
     *  @param agg     the aggregation of the hh-by-N_QoF matrix (one row per horizon) into one QoF
     *                     vector (h = 0; defaults to `Forecaster_D.aggFPE`)
     *  @param qk      index of Quality of Fit (QoF) to use for comparing quality
     */
    def selectFeatures (tech: SelectionTech, cross: String = "none", first: Int = 1, swap: Boolean = true,
                        h: Int = 0, prior: VectorD = null, stream: Int = 0, bk: Int = 3,
                        agg: MatrixD => VectorD = Forecaster_D.aggFPE)(using qk: Int):
                       (Array [LSET [Int]], Array [VEC [VectorD]]) =
        require (0 <= h && h <= hh, s"selectFeatures: horizon h = $h out of range [0, $hh]")

        if h > 0 then                                                     // the single-horizon model at h decides
            val (cols, rSq) = selectFeaturesAtH (tech, cross, first, swap, h, prior)
            val best = theBest                                            // that model's best step
            if best.qof != null then                                      // its QoF, the multi-horizon regression
                theBest = try BestStep (best.col, best.qof, fitCols (cols)._1.reg, cols)(best.qof(qk))
                          catch case e: Exception =>
                              debug ("selectFeatures", s"fit of $cols failed: ${e.getMessage}")
                              best
            (Array (cols), Array (rSq))
        else                                                              // all horizons jointly
            val rSq = VEC [VectorD] ()                                    // the best-so-far per step

            def evalJoint (cols: LSET [Int]): BestStep =                  // fit and score one set over all horizons
                try
                    val (sub, qof) = fitCols (cols)
                    val q = agg (qof)
                    BestStep (-1, q, sub.reg, cols)(q(qk))
                catch case e: Exception =>
                    debug ("selectFeatures", s"cols = $cols failed: ${e.getMessage}")
                    BestStep (-1, null, null, cols)()
            end evalJoint

            val path = SubsetSearch.run (tech, x.dim2, evalJoint, LSET.from (0 until first), swap = swap, bk = bk,
                                         prior = prior, stream = stream)
            for step <- path.steps do updateQoF (rSq, "none", step)
            theBest   = path.best
            lastProbs = path.probs
            banner (s"selectFeatures ($tech, all horizons): ${path.nEval} column sets evaluated => cols = ${path.cols}" +
                    (if path.found then s" @ ${path.best.qof(qk)}" else " (no set could be fit)"))
            (Array (path.cols), Array (rSq))
        end if
    end selectFeatures

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform feature selection for this multi-horizon model, returning the columns
     *  selected and the QoF measures for all steps, as `selectFeatures` of a
     *  `Forecaster_Reg` does (@see `FeatureSelection.selectFeatures`); the selected
     *  model is `getBest` (its regression, fitted on the FS rows) and
     *  `getModel (getBest.mod_cols)` (the forecaster, to train or roll-validate).
     *  Two ways to score a candidate column set:
     *  h = 0  -- ALL HORIZONS JOINTLY: every set is fit as ONE multi-horizon model
     *            (`fitCols`: this model's transforms and scaling, the FS rows, per-horizon
     *            QoF in original space with the skip rows dropped) and scored by `agg` over
     *            the horizons (default `Forecaster_D.aggFPE`: the mean of every measure, the
     *            sMAPE-IC slot holding the mean FPE-form sMAPE, so with qk = smapeC the
     *            criterion is the selection score J of the `llmsr` search).
     *            The trace `rSq` holds the best-so-far per step (aggregated measures); no
     *            cross-validation column (the candidates are forecasters, not the regressions
     *            `updateQoF`'s validation expects); a set whose fit fails is skipped.
     *  h >= 1 -- AT ONE HORIZON: the single-horizon model at h decides (@see
     *            `selectFeaturesAtH`: `Forecaster_Reg`'s selection on `getModel_h (h)`, so
     *            `cross` applies and the trace is that model's) and its columns are shared
     *            by every horizon; `getBest` then carries that step's QoF with the
     *            multi-horizon regression of the columns.
     *  @param tech    the selection technique (Forward, Backward, Stepwise, Beam, CEM)
     *  @param cross   indicator to include the cross-validation/validation QoF measure (h >= 1 only)
     *  @param first   the number of leading columns always kept (1 => the intercept column 0;
     *                     0 => none, e.g., a matrix built by `ARX_D.fromTerms`)
     *  @param swap    whether stepwise may swap a column in for one out
     *  @param h       the horizon whose single-horizon model decides (1 to hh), or 0 => all
     *                     horizons jointly (defaults to 0)
     *  @param prior   the prior inclusion probability per column (CEM only; null => 0.5 each)
     *  @param stream  the random number stream (CEM only, h = 0)
     *  @param bk      the beam width (h = 0; the single-horizon model's own default otherwise)
     *  @param agg     the aggregation of the hh-by-N_QoF matrix (one row per horizon) into one QoF
     *                     vector (h = 0; defaults to `Forecaster_D.aggFPE`)
     *  @param qk      index of Quality of Fit (QoF) to use for comparing quality
     * 
    def selectFeatures (tech: SelectionTech, cross: String = "none", first: Int = 1, swap: Boolean = true,
                        h: Int = 0, prior: VectorD = null, stream: Int = 0, bk: Int = 3,
                        agg: MatrixD => VectorD = Forecaster_D.aggFPE)(using qk: Int):
                       (LSET [Int], VEC [VectorD]) =
        require (0 <= h && h <= hh, s"selectFeatures: horizon h = $h out of range [0, $hh]")

        if h > 0 then                                                     // the single-horizon model at h decides
            val (cols, rSq, _) = selectFeaturesAtH (tech, cross, first, swap, h, prior)
            val best = theBest                                            // that model's best step
            if best.qof != null then                                      // its QoF, the multi-horizon regression
                theBest = try BestStep (best.col, best.qof, fitCols (cols)._1.reg, cols)(best.qof(qk))
                          catch case e: Exception =>
                              debug ("selectFeatures", s"fit of $cols failed: ${e.getMessage}")
                              best
            (cols, rSq)
        else                                                              // all horizons jointly
            val rSq = VEC [VectorD] ()                                    // the best-so-far per step

            def evalJoint (cols: LSET [Int]): BestStep =                  // fit and score one set over all horizons
                try
                    val (sub, qof) = fitCols (cols)
                    val q = agg (qof)
                    BestStep (-1, q, sub.reg, cols)(q(qk))
                catch case e: Exception =>
                    debug ("selectFeatures", s"cols = $cols failed: ${e.getMessage}")
                    BestStep (-1, null, null, cols)()
            end evalJoint

            val path = SubsetSearch.run (tech, x.dim2, evalJoint, LSET.from (0 until first), swap = swap, bk = bk,
                                         prior = prior, stream = stream)
            for step <- path.steps do updateQoF (rSq, "none", step)
            theBest   = path.best
            lastProbs = path.probs
            banner (s"selectFeatures ($tech, all horizons): ${path.nEval} column sets evaluated => cols = ${path.cols}" +
                    (if path.found then s" @ ${path.best.qof(qk)}" else " (no set could be fit)"))
            (path.cols, rSq)
        end if
    end selectFeatures
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform regression-based feature selection to find the most predictive variables
     *  to have in the model, returning the variables left and the new Quality of Fit
     *  (QoF) measures for all steps; the selected single-horizon model is
     *  `getModel_h (cols, h)` (@see `reportSelection` with h to print and test it).
     *  @see `scalation.modeling.forecasting.Forecaster_Reg` for index of QoF measures.
     *  @see `scalation.modeling.FeatureSelection.selectFeatures`
     *  Unlike `selectFeatures` allows feature selection to be horizon (h) specific
     *  @param tech   the type of the feature selection to use
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param first  first variable to consider for elimination
     *                    (default (1) assume intercept x_0 will be in any model)
     *  @param swap   whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param h      the number of the horizon
     *  @param prior  the prior inclusion probability per column (CEM only; null => 0.5 each)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def selectFeaturesAtH (tech: SelectionTech, cross: String = "none",
                           first: Int = 1, swap: Boolean = true, h: Int = 1,
                           prior: VectorD = null)(using qk: Int):
                          (LSET [Int], VEC [VectorD]) =
        require (1 <= h && h <= hh, s"horizon h = $h out of range [1, $hh]")

        val fsFun: (Forecaster_Reg => (Array [LSET [Int]], Array [VEC [VectorD]])) = tech match   // choose the FS routine once
        case Forward  => (m: Forecaster_Reg) => m.forwardSelAll (cross)
        case Backward => (m: Forecaster_Reg) => m.backwardElimAll (first, cross)
        case Stepwise => (m: Forecaster_Reg) => m.stepwiseSelAll (cross, swap)
        case Beam     => (m: Forecaster_Reg) => m.beamSelAll (cross)
        case CEM      => (m: Forecaster_Reg) => m.cemSelAll (cross, first, prior)

        val mod_h = getModel_h (h = h)                            // build a single-horizon model bound to y(:, k-1) with hh=1
        val rSq   = fsFun (mod_h)._2(0)                           // its trace (the single-horizon model's selection)
        val cols  = mod_h.getBest.mod_cols
        theBest   = mod_h.getBest
        lastProbs = mod_h.cemProbs
        (cols, rSq)
    end selectFeaturesAtH

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform regression-based feature selection to find the most predictive variables
     *  to have in the model, returning the variables left and the new Quality of Fit
     *  (QoF) measures for all steps.
     *  @see `scalation.modeling.forecasting.Forecaster_Reg` for index of QoF measures.
     *  @see `scalation.modeling.FeatureSelection.selectFeatures`
     *  Unlike `selectFeatures` allows feature selection to be horizon (h) specific
     *  @param tech   the type of the feature selection to use
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param first  first variable to consider for elimination
     *                    (default (1) assume intercept x_0 will be in any model)
     *  @param swap   whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param h      the number of the horizon
     *  @param prior  the prior inclusion probability per column (CEM only; null => 0.5 each)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     *
    def selectFeaturesAtH (tech: SelectionTech, cross: String = "none",
                           first: Int = 1, swap: Boolean = true, h: Int = 1,
                           prior: VectorD = null)(using qk: Int):
                          (LSET [Int], VEC [VectorD], Forecaster_Reg) =
        require (1 <= h && h <= hh, s"horizon h = $h out of range [1, $hh]")

        val fsFun: (Forecaster_Reg => (LSET [Int], VEC [VectorD])) = tech match     // choose the FS routine once
        case Forward  => (m: Forecaster_Reg) => m.forwardSelAll (cross)
        case Backward => (m: Forecaster_Reg) => m.backwardElimAll (first, cross)
        case Stepwise => (m: Forecaster_Reg) => m.stepwiseSelAll (cross, swap)
        case Beam     => (m: Forecaster_Reg) => m.beamSelAll (cross)
        case CEM      => (m: Forecaster_Reg) => m.cemSelAll (cross, first, prior)

        val mod_h    = getModel_h (h = h)                         // build a single-horizon model bound to y(:, k-1) with hh=1
        val (_, rSq) = fsFun (mod_h)
        val cols     = mod_h.getBest.mod_cols
        val modForc  = getModel_h (cols, h)
        theBest      = mod_h.getBest
        lastProbs    = mod_h.cemProbs
        (cols, rSq, modForc)
    end selectFeaturesAtH
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform regression-based feature selection PER HORIZON: `selectFeaturesAtH`
     *  for h = 1 to hh, one column set per horizon; returns the column sets and the
     *  QoF traces (element h-1 = horizon h), the shape `reportSelection` prints and
     *  tests (one single-horizon `Forecaster_Reg` per horizon, `getModel_h`).
     *  `getBest` is left at the last horizon's best step.
     *  @param tech   the type of the feature selection to use
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param first  first variable to consider for elimination
     *                    (default (1) assume intercept x_0 will be in any model)
     *  @param swap   whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     */
    def selectFeaturesPerH (tech: SelectionTech, cross: String = "none",
                            first: Int = 1, swap: Boolean = true)(using qk: Int):
                           (Array [LSET [Int]], Array [VEC [VectorD]]) =
        val colsArr = new Array [LSET [Int]] (hh)
        val rSqArr  = new Array [VEC [VectorD]] (hh)

        for h <- 1 to hh do
            val (cls, rSq) = selectFeaturesAtH (tech, cross, first, swap, h)
            colsArr(h-1) = cls
            rSqArr(h-1)  = rSq
        end for
        (colsArr, rSqArr)
    end selectFeaturesPerH

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform regression-based feature selection PER HORIZON: `selectFeaturesAtH`
     *  for h = 1 to hh, one column set and one single-horizon `Forecaster_Reg` per
     *  horizon, each rolling-validated on the testing rows; returns the column sets,
     *  the QoF traces, the models, the TnT forecast matrix (column h-1 = horizon h)
     *  and the TnT QoF matrix (row h-1 = horizon h).  `getBest` is left at the last
     *  horizon's best step.
     *  @param tech   the type of the feature selection to use
     *  @param cross  indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param first  first variable to consider for elimination
     *                    (default (1) assume intercept x_0 will be in any model)
     *  @param swap   whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param qk     index of Quality of Fit (QoF) to use for comparing quality
     * 
    def selectFeaturesPerH (tech: SelectionTech, cross: String = "none",
                            first: Int = 1, swap: Boolean = true)(using qk: Int):
                           (Array [LSET [Int]], Array [VEC [VectorD]], Array [Forecaster_Reg], MatrixD, MatrixD) =
        val colsArr = new Array [LSET [Int]] (hh)
        val rSqArr  = new Array [VEC [VectorD]] (hh)
        val modArr  = new Array [Forecaster_Reg] (hh)
        val yf      = new MatrixD (y.dim, hh)
        val ftMat   = new MatrixD (hh, Fit.N_QoF)

        for h <- 1 to hh do
            val (cls, rSq, modForc) = selectFeaturesAtH (tech, cross, first, swap, h)
            colsArr(h-1) = cls
            rSqArr(h-1)  = rSq
            modForc.setSkip ()                                               // don't skip any rows
            modForc.rollValidate ()
            ftMat(h-1)   = modForc.diagnoseAll (modForc.getY, modForc.getYf, Forecaster.teRng (y.dim))(0)
            yf(?, h - 1) = modForc.getYf(?, 1)
            modArr(h-1)  = modForc
        end for
        (colsArr, rSqArr, modArr, yf, ftMat)
    end selectFeaturesPerH
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform attention feature selection to find the best model
     *  @param scoredCols  the columns ordered based on feature scores
     *
    def featureSelectionAtt (scoredCols: LSET [Int]): (Array [MatrixD], Array [MatrixD], Array [MatrixD]) =
        val total = scoredCols.size
        val min_cols = 4
        val qof_all = new Array[MatrixD](hh)
        val qof_inSample = new Array[MatrixD](hh)
        val qof_TnT = new Array[MatrixD](hh)

        for k <- 1 to hh do
            println (s"k = $k")
            val qof_k = new MatrixD (3*(total-min_cols), Fit.N_QoF)
            qof_inSample(k-1) = new MatrixD (total-min_cols, Fit.N_QoF)
            qof_TnT(k-1) = new MatrixD (total-min_cols, Fit.N_QoF)
            for i <- 0 until total-min_cols do
                val cls = scoredCols.take (total-i)
                val mod_i = getModel_h (cls, k)
                val (x_mod_i, y_mod_i) = (mod_i.getX, mod_i.getY)
                val t_rng   = 0 until Model.trSize (y_mod_i.dim) 
                val (x_tr, y_tr) = (x_mod_i(t_rng), y_mod_i(t_rng)) 
                val (_, qof) = mod_i.trainNtest (x_tr, y_tr)(x_tr, y_tr)                
                qof_k(3*i) += i 
                qof_k(3*i + 1) = qof
                qof_inSample(k-1)(i) = qof_k(3*i + 1)

                mod_i.setSkip ()                                             // don't skip any rows
                mod_i.rollValidate ()
                qof_k(3*i + 2) = mod_i.diagnoseAll (mod_i.getY, mod_i.getYf, Forecaster.teRng (y.dim))(0)
                qof_TnT(k-1)(i) = qof_k(3*i + 2)                
            end for
            qof_all(k-1) = qof_k
        end for
        (qof_all, qof_inSample, qof_TnT)
    end featureSelectionAtt
     */

//  R E P O R T I N G   T H E   S E L E C T I O N   (@see `SelectionReport.reportSelection`)

    // Note: `reportSelection (colsArr, rSqArr, h, inSample, tnt, rc, ...)` prints every column set with
    //       its trace and tests the selected model(s): from `selectFeatures` (one set, h = 0) the
    //       multi-horizon model on the columns (`getModel`); from `selectFeaturesAtH` (one set, h given)
    //       the single-horizon model at h (`getModel_h`); from `selectFeaturesPerH` (hh sets) one
    //       single-horizon model per horizon, their QoF rows assembled into one table.  The two hooks:

    type SelectedModel = Forecaster

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the forecaster of a selected column set (the hook of `reportSelection`):
     *  the multi-horizon model on the columns, or the single-horizon model at h.
     *  @param cols  the selected columns
     *  @param h     the horizon of a single-horizon selection (0 => the multi-horizon model)
     */
    protected def selectedModel (cols: LSET [Int], h: Int): Forecaster =
        if h > 0 then getModel_h (cols, h) else getModel (cols)
    end selectedModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a selected forecaster (the hook of `reportSelection`) with the steps of
     *  `inSample_Test` (train on all the rows, forecast, diagnose every horizon) and/or
     *  of `tnT_Test` (train, forecast, `setSkip`, `rollValidate (rc)`, diagnose on the
     *  testing rows); each QoF table has one row per horizon of the model (one row for
     *  a single-horizon model).
     *  @param mod       the selected forecaster (multi- or single-horizon)
     *  @param inSample  whether to test in-sample
     *  @param tnt       whether to test with rolling validation
     *  @param rc        the retraining cycle of the rolling validation
     *  @param showYf    whether to print the forecast matrices
     */
    protected def testSelected (mod: Forecaster, inSample: Boolean, tnt: Boolean, rc: Int,
                                showYf: Boolean): (MatrixD, MatrixD) =
        def train (): Unit = mod match                                     // train on all the rows
            case d: Forecaster_D => d.trainNtest (d.getX, d.getYy)(d.getX, d.getYy)
            case r               => r.trainNtest ()()
        var qofIn: MatrixD  = null
        var qofTnT: MatrixD = null
        if inSample then
            banner (s"$modelName: In-Sample Test of the selected model on ${stringOf (mod.getFname)}")
            train ()
            mod.forecastAll ()                                             // forecast h-steps ahead for all y
            qofIn = mod.diagnoseAll (mod.getY, mod.getYf)                  // QoF for each horizon
            if showYf then println (s"Forecast Matrix yf = ${mod.getYf}")
        if tnt then
            banner (s"$modelName: TnT Test (rolling validation) of the selected model on ${stringOf (mod.getFname)}")
            train ()                                                       // initial training updated by `rollValidate`
            mod.forecastAll ()
            mod.setSkip ()                                                 // diagnose: don't skip any rows
            mod.rollValidate (rc)                                          // TnT with Rolling Validation
            qofTnT = mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (mod.getY.dim))   // the testing rows only
            if showYf then println (s"Forecast Matrix yf = ${mod.getYf}")
        (qofIn, qofTnT)
    end testSelected

end Forecaster_D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Forecaster_D` companion object holds the parsimony-adjusted sMAPE in the
 *  form of Akaike's Final Prediction Error, the selection criterion of the
 *  all-horizons feature selection and of the `llmsr` structure search.
 */
object Forecaster_D:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the FPE-form parsimony-adjusted sMAPE per horizon,
     *      sMAPE_h (1 + 2 p / n_h),  p = |S| + 1 (the slopes and the intercept),
     *  n_h the rows diagnosed at horizon h (the `dfr` and `m` slots of the QoF
     *  matrix).  Akaike's FPE = sigma^2 (n + p) / (n - p) ~ sigma^2 (1 + 2p/n)
     *  charges a term in proportion to the loss, so the weight of the penalty
     *  against the loss is the same at every error level (@see the note above).
     *  @param qof  the QoF matrix, one row per horizon (@see `Fit.N_QoF`)
     */
    def smapeFPE (qof: MatrixD): VectorD =
        VectorD (for h <- qof.indices yield
            val p = qof(h, QoF.dfr.ordinal) + 1.0
            val n = qof(h, QoF.m.ordinal)
            qof(h, QoF.smape.ordinal) * (1.0 + 2.0 * p / n))
    end smapeFPE

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Aggregate a QoF matrix over the horizons for `selectFeatures` (h = 0): the
     *  mean of every measure, with the sMAPE-IC slot holding the mean FPE-form
     *  sMAPE over the horizons (`smapeFPE`), the selection score J of the `llmsr`
     *  search; with qk = smapeC the joint selection therefore minimizes J.
     *  @param qof  the QoF matrix, one row per horizon
     */
    def aggFPE (qof: MatrixD): VectorD =
        val v = qof.mean
        v(QoF.smapeC.ordinal) = smapeFPE (qof).mean
        v
    end aggFPE

end Forecaster_D

