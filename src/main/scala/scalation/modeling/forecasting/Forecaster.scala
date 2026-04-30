
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Abstract Class for Forecasters with Vector Input + Random Walk (RW)
 *           most models will need to override `train`, `predict`, `forecast` and `forecastAt`
 */

package scalation
package modeling
package forecasting

import scala.annotation.unused
import scala.collection.mutable.IndexedSeq
import scala.math.{abs, max}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Forecast` trait serves a minimal adpater of `Model` to the `Forecast` task.
 *  Most of implementation is in the `Forecaster` abstract class below.
 */
trait Forecast
      extends Model:

    _taskType = TaskType.Forecast                                          // the type of task performed

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum lag used by the model (its capacity to look into the past).
     *  Models that use more than one past value to make predictions/forecasts must
     *  override this method, e.g., ARMA (2, 3) should set the cap to max(p, q) = 3.
     */
    def cap: Int = 1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used data/input matrix.  Model that use x should override.
     */
    def getX: MatrixD = null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the feature/variable names.  Model that use x should override.
     */
    def getFname: Array [String] = Array ("no-x features")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a time series y_, train the forecasting function y_ = f(lags (y_)) + e,
     *  where f(lags (y_)) is a function of the lagged values of y_,
     *  by fitting its parameters.
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the testing/full response/output vector (e.g., full y)
     */
    def train (x_null: MatrixD, y_ : VectorD): Unit = {}

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The standard signature for prediction does not apply to time series.
     */
    def predict (z: VectorD): Double =
        throw new UnsupportedOperationException ("predict (VectorD) use the alternative predict")
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The standard signature for prediction does not apply to time series.
     */
    def crossValidate (k: Int, rando: Boolean): Array [Statistic] =
        throw new UnsupportedOperationException ("Use `rollValidate` instead of `crossValidate`")
    end crossValidate

end Forecast


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Forecaster` abstract class provides a common framework for several forecasters.
 *  Note, the train method must be called first followed by test.
 *  @param y        the response vector (time series data)
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (index as time may suffice)
 *  @param hparam   the hyper-parameters for models extending this abstract class
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
abstract class Forecaster (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = null,
                           bakcast: Boolean = false)
      extends Diagnoser (dfr = 1, df = y.dim - 1)                          // dfr and df are updated later per model
         with ForecastMatrix (y, hh, tRng)
         with Forecast:

    private val debug = debugf ("Forecaster", false)                       // debug function
    private val flaw  = flawf ("Forecaster")                               // flaw function

    protected val yb = if bakcast then WeightedMovingAverage.backcast (y) +: y   // prepend by adding one backcasted value
                       else y
    protected val mm = yb.dim                                              // size of augmented time series with one backcasted value 
    protected val e  = new VectorD (mm)                                    // residual/error vector [e_0, e_1, ... e_m]
    protected var b  = VectorD (0)                                         // parameter vector for forecasting models
    protected val yf = makeForecastMatrix (yb, hh)                         // forecasts for all time points t & horizons to h

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response/output vector y.
     */
    def getY: VectorD = y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used original/untransformed response/output vector y.
     */
    def getY_org: VectorD = if yForm == null then y else yForm.fi(y)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response/output vector yb (y prepended by one backcast value).
     */
    def getYb: VectorD = yb

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used FORECAST MATRIX yf.
     */
    def getYf: MatrixD = yf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test PREDICTIONS of a forecasting model y_ = f(lags (y_)) + e
     *  and return its predictions and  QoF vector.  Testing may be in-sample
     *  (on the training set) or out-of-sample (on the testing set) as determined
     *  by the parameters passed in.  Note: must call train before test.
     *  Must override to get Quality of Fit (QoF).
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the actual testing/full response/output vector
     */
    def test (x_null: MatrixD, y_ : VectorD): (VectorD, VectorD) =
        val yy = if bakcast then y_(1 until y_.dim)                       // align the actual values
                 else y_
        val yp = predictAll (yy)                                          // make all predictions

        println (s"yy.dim = ${yy.dim}, yp.dim = ${yp.dim}")
//      Forecaster.differ (yy, yfh)                                       // uncomment for debugging
        assert (yy.dim == yp.dim)                                         // make sure the vector sizes agree

        if DO_PLOT then
            new Plot (null, yy, yp, s"test: Plot of yy, yp for $modelName vs. t", true)
        mod_resetDF (yy.dim - skip)                                       // reset the degrees of freedom
//      println (s"test: yy = $yy,\n yp = $yp")
        (yp, diagnose (yy, yp))                                           // return predicted and QoF vectors
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Models need to provide a means for updating the Degrees of Freedom (DF).
     *  @param size  the size of dataset (full, train, or test)
     */
    override def mod_resetDF (size: Int): Unit =
        val dfr = max (1, parameter.size - 1)                              // degrees of freedom for regression/model
        debug ("mod_resetDF", s"dfr = $dfr, df = ${size-dfr}")
        resetDF (dfr, size - dfr) 
    end mod_resetDF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the forecasting model y_ = f(y-past) + e and report its QoF
     *  and plot its predictions.  Return the predictions and QoF.
     *  @param y_  the training/full response/output vector (defaults to full y)
     *  @param yy  the testing/full response/output vector (defaults to full y)
     */
    def trainNtest (y_ : VectorD = yb)(yy: VectorD = yb): (VectorD, VectorD) =
        train (null, y_)                                                  // train the model on training set
        val (yp, qof) = test (null, yy)                                   // test the model on testing set
        println (report (qof))                                            // report on Quality of Fit (QoF)
        (yp, qof)
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test FORECASTS of a forecasting model y_ = f(lags (y_)) + e
     *  and RETURN (1) aligned actual values, (2) its forecasts and (3) QoF vector.
     *  Testing may be in-sample (on the training set) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.  Note: must call train and forecastAll
     *  before testF.
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the testing/full response/output vector
     *
    def testF (h: Int, y_ : VectorD): (VectorD, VectorD, VectorD) =
        val yfh = yf(?, h)(0 until y_.dim-h)                              // column h of the forecast matrix
        val yy  = y_(h until y_.dim)                                      // align the actual values
        println (s"yy.dim = ${yy.dim}, yfh.dim = ${yfh.dim}")
//      Forecaster.differ (yy, yfh)                                       // uncomment for debugging
        assert (yy.dim == yfh.dim)                                        // make sure the vector sizes agree

        new Plot (null, yy, yfh, s"testF: yy, yfh vs. t for $modelName @h = $h", lines = true)
        mod_resetDF (yy.dim)                                              // reset the degrees of freedom
        (yy, yfh, diagnose (yy, yfh))                                     // return actual, forecasted and QoF vectors
    end testF
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the hyper-parameters.
     */
    def hparameter: HyperParameter = hparam

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of parameter/coefficient values (they are model specific).
     *  Override for models with other parameters besides b.
     */
    def parameter: VectorD = b                                            // parameter vector
    def nparams: Int       = parameter.dim                                // number of parameters

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of residuals/errors.
     */
    def residual: VectorD = { if e == null then flaw ("residual", "must call test method first"); e }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = f (y_t-1, ...) = y_t-1    (random walk -- use previous value)
     *
     *  Override for other models.
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    def predict (t: Int, y_ : VectorD): Double = y_(max0 (t-1))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict all values corresponding to the given time series vector y_.
     *  Update FORECAST MATRIX yf and return PREDICTION VECTOR yp as second (1) column
     *  of yf with last value removed.
     *  @see `forecastAll` to forecast beyond horizon h = 1.
     *  @param y_  the actual time series values to use in making predictions
     */
    def predictAll (y_ : VectorD = yb): VectorD =
        if bakcast then
            for t <- 1 until y_.dim do yf(t-1, 1) = predict (t, y_)           // use model to make predictions
            yf(?, 1)(0 until y_.dim-1)                                        // return yp: first horizon only
        else
//          debug ("predictAll", s"y_.dim = ${y_.dim}, yf.dims = ${yf.dims}")
            yf(0, 1) = NO_DOUBLE                                              // no prior data to make predictions
            for t <- 1 until yf.dim do yf(t, 1) = predict (t, y_)             // skip t = 0
            yf(?, 1)
    end predictAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with rolling validation (analog of predict method).
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     */
    def forecast (t: Int, y_ : VectorD = yb): VectorD =
        val yh = new VectorD (hh)                                         // hold forecasts for each horizon
        val pred = predict (t, y_)                                        // use 1-step ahead forecast (h-step same for RW)
        for h <- 1 to hh do
            yf(t, h) = pred                                               // record in forecast matrix
            yh(h-1)  = pred                                               // record forecasts for each horizon
        yh                                                                // return forecasts for all horizons
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points and all horizons (1 through hh-steps ahead).
     *  Record these in the FORECAST MATRIX yf, where
     *
     *      yf(t, h) = h-steps ahead forecast for y_t
     *
     *  @param y_  the actual values to use in making forecasts
     */
    def forecastAll (y_ : VectorD = yb): MatrixD =
        for h <- 2 to hh do forecastAt (h, y_)                            // forecast k-steps into the future
        yf                                                                // return matrix of forecasted values
    end forecastAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the h-steps ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1 and for random walk the
     *  forecast across all horizons is the same.
     *  Method should be overridden for each model besides `RandomWalk`
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the actual values to use in making forecasts
     */
    def forecastAt (h: Int, @unused y_ : VectorD = yb): VectorD =
        yf(?, h) = yf(?, 1)
        yf(?, h)
    end forecastAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast intervals for all y_.dim time points at horizon h (h-steps ahead).
     *  Create prediction intervals (two vectors) for the given time points at level p.
     *  @caveat:  assumes errors follow a Normal distribution.  Override this method to handle other cases.
     *  @param y_   the aligned actual values to use in making forecasts
     *  @param yfh  the forecast vector at horizon h
     *  @param h    the forecasting horizon, number of steps ahead to produce forecasts
     *  @param p    the level (1 - alpha) for the prediction interval
     */
    def forecastAtI (y_ : VectorD, yfh: VectorD, h: Int, p: Double = 0.9): (VectorD, VectorD) =
        debug ("forecastAtI", s"for h = $h: y_.dim = ${y_.dim}, yfh.dim = ${yfh.dim}")
        val sig_h = (y_ - yfh).stdev                                      // standard error of estimate at horizon h
        val width = z_sigma (sig_h, p)                                    // interval half width
        (yfh - width, yfh + width)                                        // return lower and upper bounds
    end forecastAtI

//  T E S T I N G   S C E N A R I O S

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Align the actual response vector for comparison with the predicted/forecasted
     *  response vector, returning a time vector and sliced response vector.
     *  @param tr_size  the size of the initial training set
     *  @param y        the actual response for the full dataset (to be sliced)
     */
    def align (tr_size: Int, y: VectorD): (VectorD, VectorD) =
        (VectorD.range (tr_size, y.dim), y(tr_size until y.dim))
    end align

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use validation to compute test Quality of Fit (QoF) measures by dividing
     *  the full dataset into a TESTING-set and a TRAINING-set.
     *  Delegates to `rollValidate` with no retraining and only diagnoses the first horizon.
     *  Must call the `set_TE_RATIO` method to change the default TE ratio.
     *  @param rando  flag indicating whether to use randomized or simple validation (false)
     *  @param ratio  the ratio of the TESTING-set to the full dataset (most common 70-30 (.3), 80-20 (.2))
     *  @param idx    the prescribed TESTING-set indices (default => null)
     */
    def validate (rando: Boolean = false, ratio: Double = Model.TE_RATIO)
                 (@unused idx: IndexedSeq [Int] = null):
                 (VectorD, VectorD) =
        debug ("validate", s"rando = $rando (requires false), ratio = $ratio (requires ${Model.TE_RATIO})")
        val te_size = Model.teSize (y.dim)                                // size of testing set
        val yf1 = rollValidate (y.dim, false)(?, 1)                       // get column 1 returned from `rollValidate`
        val y_  = y(y.dim - te_size until y.dim)                          // trim the actual values to testing-set
        val yf_ = yf1(y.dim - te_size until y.dim)                        // trim the forecast at h = 1 to testing-set
        val qof = diagnose (y_, yf_)
        (yf1, qof)
    end validate

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use rolling-validation to compute test Quality of Fit (QoF) measures
     *  by dividing the dataset into a TRAINING SET (tr) and a TESTING SET (te).
     *  as follows:  [ <-- tr_size --> | <-- te_size --> ]
     *  Calls forecast for h-steps ahead out-of-sample forecasts.
     *  Return the FORECAST MATRIX.
     *  @param rc       the retraining cycle (number of forecasts until retraining occurs)
     *  @param growing  whether the training grows as it roll or keeps a fixed size
     *  @param doPlot   whether to show the plots
     */
    def rollValidate (rc: Int = 2, growing: Boolean = false, doPlot: Boolean = true): MatrixD =
        val ftMat   = new MatrixD (hh, Fit.N_QoF)
        banner (s"rollValidate: Evaluate ${modelName}'s QoF for horizons 1 to $hh:")

        val x       = getX                                                // some model use and input matrix, else null
        val y       = getYb                                               // get (expanded) response/output vector
        val yf      = getYf                                               // get the full in-sample forecast matrix
        val te_size = Model.teSize (y.dim)                                // size of testing set
        val tr_size = Model.trSize (y.dim)                                // size of initial training set
        debug ("rollValidate", s"y.dim = ${y.dim}, train: tr_size = $tr_size; test: te_size = $te_size, rc = $rc")

        val yp = new VectorD (te_size)                                    // y-predicted over testing set (only for h=1)
        for i <- 0 until te_size do                                       // iterate through testing set
            val is = if growing then 0 else i
            val t  = tr_size + i                                          // next time point to forecast
            if i % rc == 0 then
                val x_ = if x != null then x(is until t) else null
                train (x_, y(is until t))                                 // retrain on sliding training set
//          yp(i)  = predict (min (t+1, y.dim-1), y)                      // predict the next value (only for h=1)
            yp(i)  = predict (t, y)                                       // predict the next value (only for h=1)
            val yd = forecast (t, y)                                      // forecast the next hh-values, yf is updated
            debug ("rollValidate", s"yf(t, 0) = ${yf(t, 0)}, yp(i) = ${yp(i)}, yd = $yd")
//          assert (yp(i) =~ yd(0))                                       // make sure h=1 forecasts agree with predictions
        end for

        val (t, yy) = align (tr_size, y)                                  // align vectors
        if doPlot then
            new Plot (t, yy, yp, s"rollValidate: Plot yy, yp vs. t for $modelName", lines = true)

        val yf_ = yf(tr_size until y.dim)                                 // forecast matrix for test-set
        for h <- 1 to hh do
            val yy_ = yy(h-1 until yy.dim)                                // trim the actual values
            val yfh = yf_(?, h)(0 until yy.dim-h+1)                       // column h of the forecast matrix

            if doPlot then
                new Plot (t, yy_, yfh, s"rollValidate: Plot yy_, yfh vs. t for $modelName @h = $h", lines = true)
            mod_resetDF (te_size - h)                                     // reset degrees of freedom
            val qof    = diagnose (yy_, yfh)
            ftMat(h-1) = qof
//          println (FitM.fitMap (qof, qoF_names))
        end for
        if DO_REPORT then
            println ("fitMap     qof = ")
            println (Fit.showFitMap (ftMat.ᵀ))
        yf
    end rollValidate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform In-Sample Testing, i.e., train and test on the same FULL data set.
     *  Return the prediction and the Quality of Fit.
     *  @param skip    the number of initial time points to skip (due to insufficient past)
     *  @param showYf  whether to show the forecast matrix
     */
    override def inSample_Test (skip: Int = 2, showYf: Boolean = false): (VectorD, VectorD) =
        banner (s"In-Sample Test: $modelName")
        val (yp, qof) = trainNtest ()()                                   // train on full and test on full 
        forecastAll ()                                                    // forecast over all horizons
        setSkip (skip)                                                    // diagnose: skip the first 'skip' rows
        diagnoseAll (getY, getYf)                                         // compute metrics for all horizons
        if showYf then
            println (s"Final In-Sample Forecast Matrix yf = ${getYf}")
//          println (s"Final In-Sample Forecast Matrix yf = ${getYf.shiftDiag}")
        (yp, qof)
    end inSample_Test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform Train-n-Test (TnT) Testing, i.e., train and test with rolling validation.
     *  @param skip    the number of initial time points to skip (due to insufficient past)
     *  @param rc      the retraining cycles (how often to retrain the model)
     *  @param showYf  whether to show the forecast matrix
     */
    def tnT_Test (skip: Int = 0, rc: Int = 2, showYf: Boolean = false): Unit =
        banner (s"TnT Test: $modelName")
        trainNtest ()()                                                   // initial training updated by `rollValidate`
        setSkip (skip)                                                    // diagnose: skip the first 'skip' rows
        rollValidate (rc)                                                 // TnT with Rolling Validation
        diagnoseAll (getY,getYf, Forecaster.teRng (y.dim))                // only diagnose on the testing set
        if showYf then
            println (s"Final TnT Forecast Matrix yf = ${getYf}")
//          println (s"Final TnT Forecast Matrix yf = ${getYf.shiftDiag}")
    end tnT_Test

end Forecaster


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Forecaster` companion object provides methods useful for classes extending
 *  the `Forecaster` abstract class, i.e., forecasting models with a single input variable.
 */
object Forecaster:

    private val flaw = flawf ("Forecaster")                            // flaw function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the index range for the testing set.
     *  @param m  the size of the full dataset
     */
    def teRng (m: Int): Range = Model.trSize (m) until m

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a reverse dot product of the parameter vector b and the most recent
     *  actual values in the time series y_, going backwards from y_t.
     *  Use max (0, ..) to avoid using negative indices into the y_ vector.
     *  @param b   the parameter/coefficient vector (e.g., φ for AR)
     *  @param y_  the actual time series values to use in making predictions
     *  @param t   the time point FROM WHICH to make prediction
     */
    def rdot (b: VectorD, y_ : VectorD, t: Int): Double =
        var sum = 0.0
        for j <- b.indices do sum += b(j) * y_(max0 (t-j))             // add b_j y_t-j
        sum
    end rdot

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a reverse dot product of the parameter vector b and a row of
     *  the yf matrix starting at element (r, c) and moving back and then moving up.
     *  Use max (0, ..) to avoid using negative indices into the yf matrix.
     *  @param b   the parameter/coefficient vector (e.g., φ for AR)
     *  @param yf  the forecast matrix (time x horizons)
     *  @param r   the starting row in the forecast matrix (time)
     *  @param c   the starting column in the forecast matrix (horizon)
     */
    def rdot (b: VectorD, yf: MatrixD, r: Int, c: Int): Double =
        var sum = 0.0
        for j <- b.indices do
            val k = c - j
            sum += (if k > 0 then b(j) * yf(r, k)                      // move back in row r (prior forecasts)
                    else          b(j) * yf(max0 (r+k), 0))            // move up in column 0 (actual values)
        sum
    end rdot

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Point out the differences between two vectors/time series.
     *  @param u      the first vector/time series
     *  @param v      the second vector/time series
     *  @param scale  the scale factor to set the tolerance 'tol'
     *  @param allow  flag indicating whether allow (via assert) any differences
     */
    def differ (u: VectorD, v: VectorD, scale: Double = 1E-9, allow: Boolean = true): Int =
        if u.dim != v.dim then flaw ("differ", s"requires u.dim = ${u.dim} = v.dim = ${v.dim}")
        val tol = u.mean * scale
        var cnt = 0
//      for t <- u.indices if u(t) !=~ v(t) do                            // machine epsilon
        for t <- u.indices if abs (u(t) - v(t)) > tol do                  // application tolerance
            cnt += 1
            println (s"differ at t = $t: ${u(t)} \t ${v(t)}")
        end for
        banner (s"differ (u, v): found $cnt points that differ")
        if ! allow then assert (cnt == 0)
        cnt
    end differ

end Forecaster

