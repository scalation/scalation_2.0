
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Abstract Class for Forecasters with Matrix Input
 *           most models will need to override `train`, `predict`, `forecast` and `forecastAt`
 */

package scalation
package modeling
package forecasting

import scala.annotation.unused
import scala.collection.mutable.{LinkedHashSet => LSET}
import scala.math.max

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Forecaster_D` abstract class provides a common framework for several forecasters.
 *  @note `Forecaster_D` is dependent on [[Forecaster_Reg]] class to do feature selection.
 *  Note, the `train_x` method must be called first followed by `test`.
 *  @param x        the input lagged time series data
 *  @param y        the response matrix (time series data per horizon)
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (index as time may suffice)
 *  @param hparam   the hyper-parameters for models extending this abstract class
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
abstract class Forecaster_D (x: MatrixD, y: MatrixD, hh: Int, fname: Array [String],
                             tRng: Range = null, hparam: HyperParameter = MakeMatrix4TS.hp,
                             bakcast: Boolean = false)
      extends Forecaster (y(?, 0), hh, tRng, hparam, bakcast):          // no automatic backcasting, @see `ARY_D.apply`

    private val debug = debugf ("Forecaster_D", false)                  // debug function

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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit e.g., an `ARY_D` model to the times-series data in vector y_.
     *  Estimate the coefficient vector b for a p-th order Auto-Regressive ARY_D(p) model.
     *  Uses OLS Matrix Fatorization to determine the coefficients, i.e., the b (φ) vector.
     *  @param x_  the data/input matrix (e.g., full x)
     *  @param y_  the training/full response vector (e.g., full y)
     */
    def train_x (x_ : MatrixD, y_ : MatrixD): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the forecasting model y_ = f(y-past) + e and report its QoF
     *  and plot its predictions.  Return the predictions and QoF.
     *  NOTE: must use `trainNtest_x` when an x matrix is used, such as in `ARY_D`.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output vector (defaults to full y)
     */
    def trainNtest_x (x_ : MatrixD = x, y_ : MatrixD = y)(xx: MatrixD = x, yy: MatrixD = y):
                     (VectorD, VectorD) =
        train_x (x_, y_)                                                // train the model on training set
        val (yp, qof) = test (xx, yy)                                   // test the model on testing set
        if DO_REPORT then
            println (report (qof))                                      // report on Quality of Fit (QoF)
        (yp, qof)
    end trainNtest_x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test PREDICTIONS of a forecasting model y_ = f(lags (y_)) + e
     *  and return its predictions and  QoF vector.  Testing may be in-sample
     *  (on the training set) or out-of-sample (on the testing set) as determined
     *  by the parameters passed in.  Note: must call train before test.
     *  Must override to get Quality of Fit (QoF).
     *  @param x_null  the data/input matrix
     *  @param y_      the actual testing/full response/output matrix
     */
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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Models need to provide a means for updating the Degrees of Freedom (DF).
     *  @param size  the size of dataset (full, train, or test)
     */
    override def mod_resetDF (size: Int): Unit =
        val dfr = max (1, parameter.size - 1)                           // degrees of freedom for regression/model
        debug ("mod_resetDF", s"dfr = $dfr, df = ${size-dfr}")
        resetDF (dfr, size - dfr)
    end mod_resetDF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of parameter/coefficient values (they are model specific).
     *  Override for models with other parameters besides bb(?, 0).
     */
    override def parameter: VectorD = bb(?, 0)                          // parameter vector (first column)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = b_0 + b_1 y_t-1 + b_2 y_t-2 + ... + b_p y_t-p = b dot x_t
     *
     *  FIX - parameter order is in conflict with AR models.
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
     *  @param y_  the actual values to use in making forecasts
     */
    def forecastAll (@unused y_ : MatrixD): MatrixD = yf

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
    override def rollValidate (rc: Int = 2, growing: Boolean = false, doPlot: Boolean = false): MatrixD =
        banner (s"rollValidate: Evaluate ${modelName}'s QoF for horizons 1 to $hh:")

        val yf      = getYf                                             // get the full in-sample forecast matrix
        val te_size = Model.teSize (y.dim)                              // size of testing set
        val tr_size = Model.trSize (y.dim)                              // size of initial training set
        debug ("rollValidate", s"y.dims = ${y.dims}, train: tr_size = $tr_size; test: te_size = $te_size, rc = $rc")

        val yp = new VectorD (te_size)
        for i <- yp.indices do                                          // iterate through testing set
            val is = if growing then 0 else i
            val t  = tr_size + i                                        // next time point to forecast
            if i % rc == 0 then
                val x_ = if x != null then x(is until t) else null
                train_x (x_, y(is until t))                             // retrain on sliding training set
                debug ("rollValidate", s"retrain on i = $i, bb = $bb")
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
    /** Perform In-Sample Testing, i.e. train and test on the full data set.
     *  @param skip    the number of initial time points to skip (due to insufficient past)
     *  @param showYf  whether to show the forecast matrix
     */
    override def inSample_Test (skip: Int = 2, showYf: Boolean = false): Unit =
        banner (s"In-Sample Test: $modelName")
        trainNtest_x ()()                                                 // train on full and test on full
        setSkip (skip)                                                    // diagnose: skip the first 'skip' rows
        diagnoseAll (getY, getYf)                                         // compute metrics for all horizons
        if showYf then
            println (s"Final In-Sample Forecast Matrix yf = ${getYf}")
//          println (s"Final In-Sample Forecast Matrix yf = ${getYf.shiftDiag}")
    end inSample_Test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform Train-n-Test (TnT) Testing, i.e. train and test with rolling validation.
     *  @param skip    the number of initial time points to skip (due to insufficient past)
     *  @param rc      the retraining cycles (how often to retrain the model)
     *  @param showYf  whether to show the forecast matrix
     */
    override def tnT_Test (skip: Int = 0, rc: Int = 2, showYf: Boolean = false): Unit =
        banner (s"TnT Test: $modelName")
        trainNtest_x ()()                                                 // initial training updated by `rollValidate`
        setSkip (skip)                                                    // diagnose: skip the first 'skip' rows
        rollValidate (rc)                                                 // TnT with Rolling Validation
        diagnoseAll (getY,getYf, Forecaster.teRng (y.dim))                // only diagnose on the testing set
        if showYf then
            println (s"Final TnT Forecast Matrix yf = ${getYf}")
//          println (s"Final TnT Forecast Matrix yf = ${getYf.shiftDiag}")
    end tnT_Test

//  F E A T U R E   S E L E C T I O N

    import SelectionTech._

    def getBest: BestStep = ???   // FIX -- implement or throw exception

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
    /** Perform regression-based feature selection to find the most predictive variables
     *  to have in the model, returning the variables left and the new Quality of Fit
     *  (QoF) measures for all steps.
     *  @see `scalation.modeling.Forecaster_Reg` for index of QoF measures.
     *  @param h       the number of the horizon
     *  @param fsType  the type of the feature selection to use
     *  @param cross   indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param first   first variable to consider for elimination
     *                     (default (1) assume intercept x_0 will be in any model)
     *  @param swap    whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param qk      index of Quality of Fit (QoF) to use for comparing quality
     */
    def featureSelectAtHorizon (h: Int, fsType: SelectionTech, cross: String = "none",
                                first: Int = 1, swap: Boolean = true)
                               (using qk: Int): (LSET [Int], MatrixD, Forecaster_Reg) =
        require (1 <= h && h <= hh, s"horizon h=$h out of range [1, $hh]")

        val fsFun: (Forecaster_Reg => (LSET [Int], MatrixD)) = fsType match            // choose the FS routine once
        case Forward  => (m: Forecaster_Reg) => m.forwardSelAll (cross)
        case Backward => (m: Forecaster_Reg) => m.backwardElimAll (first, cross)
        case Stepwise => (m: Forecaster_Reg) => m.stepwiseSelAll (cross, swap)
        case Beam     => (m: Forecaster_Reg) => m.beamSelAll (cross)

        val mod_h = getModel_h (h = h)                                       // build a single-horizon model bound to y(:, k-1) with hh=1
        val (_, rSq) = fsFun (mod_h)
        val cols = mod_h.getBest.mod_cols
        val modForc = getModel_h (cols, h)
        (cols, rSq, modForc)
    end featureSelectAtHorizon

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform regression-based feature selection to find the most predictive variables
     *  to have in the model, returning the variables left and the new Quality of Fit
     *  (QoF) measures for all steps.
     *  @param fsType  the type of the feature selection to use
     *  @param cross   indicator to include the cross-validation/validation QoF measure (defaults to "none")
     *  @param first   first variable to consider for elimination
     *                     (default (1) assume intercept x_0 will be in any model)
     *  @param swap    whether to allow a swap step (swap out a feature for a new feature in one step)
     *  @param qk      index of Quality of Fit (QoF) to use for comparing quality
     */
    def featureSelection (fsType: SelectionTech = Stepwise, cross: String = "none",
                          first: Int = 1, swap: Boolean = true)
                         (using qk: Int): (Array [LSET [Int]], Array [MatrixD], Array [Forecaster_Reg], MatrixD, MatrixD) =
        val colsArr = new Array [LSET [Int]](hh)
        val rSqArr  = new Array [MatrixD](hh)
        val modArr  = new Array [Forecaster_Reg](hh)
        val yf      = new MatrixD(y.dim, hh)
        val ftMat   = new MatrixD (hh, Fit.N_QoF)

        for h <- 1 to hh do
            val (cls, rSq, modForc) = featureSelectAtHorizon (h, fsType, cross, first, swap)
            colsArr(h-1) = cls
            rSqArr(h-1)  = rSq
            modForc.setSkip (0)
            modForc.rollValidate (rc = 2)
            ftMat(h-1)   = modForc.diagnoseAll (modForc.getY, modForc.getYf, Forecaster.teRng (y.dim))(0)
            yf(?, h - 1) = modForc.getYf(?, 1)
            modArr(h-1)  = modForc
        end for
        (colsArr, rSqArr, modArr, yf, ftMat)
    end featureSelection

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
                val (_, qof) = mod_i.trainNtest_x (x_tr, y_tr)(x_tr, y_tr)                
                qof_k(3*i) += i 
                qof_k(3*i + 1) = qof
                qof_inSample(k-1)(i) = qof_k(3*i + 1)
                mod_i.setSkip (0)
                mod_i.rollValidate (rc = 2)
                qof_k(3*i + 2) = mod_i.diagnoseAll (mod_i.getY, mod_i.getYf, Forecaster.teRng (y.dim))(0)
                qof_TnT(k-1)(i) = qof_k(3*i + 2)                
            end for
            qof_all(k-1) = qof_k
        end for
        (qof_all, qof_inSample, qof_TnT)
    end featureSelectionAtt
     */

end Forecaster_D

