
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Aug 23 22:54:01 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Forecasters with one Endogenous and Multiple Exogenous Variables
 *
 *  @see ruqinren.wordpress.com/2020/02/21/all-the-confusion-about-arima-arimax-transfer-function-dynamic-regression-models/
 *       robjhyndman.com/hyndsight/arimax/
 *       medium.com/@xwang222/forecasting-101-ep07-multivariate-models-9f3a11fbb374
 */

// FIX - convert the regression trees so the `ForecasterX` trait is not needed and can be eliminated

package scalation
package modeling
package forecasting
package reg_trees

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Given a response vector y, build and return
 *  (1) an input/predictor MATRIX xx and
 *  (2) an output/multi-horizon output/response MATRIX yy.
 *  Used by forecast models that use DIRECT multi-horizon forecasting.
 *  The first response can't be predicted as its inputs are only the backcast value.
 *  Therefore, the number of rows in xx and yy is reduced to "y.dim-1".
 *  FIX - utilize `MakeMatrix4TS`
 *  @param y     the given output/response vector, i.e., the time series
 *  @param lags  the maximum lag included (inclusive)
 *  @param hh    the maximum forecasting horizon (h = 1, 2, ... hh)
 */
def buildMatrix4TS (y: VectorD, lags: Int, hh: Int): (MatrixD, MatrixD) =
    val mm = y.dim - 1
    val yb = WeightedMovingAverage.backcast (y) +: y                       // y prependined with one backcast value
    val xx = new MatrixD (y.dim-1, lags)                                   // input matrix: column for each lag
    val yy = new MatrixD (y.dim-1, hh)                                     // output matrix: column for each horizon
    for t <- 0 until mm do                                                 // skip first row (all the same values)
        for j <- xx.indices2 do xx(t, lags - 1 - j) = yb(max0 (t + 1 - j))
        for j <- yy.indices2 do yy(t, j) = if t + 1 + j >= y.dim then -0.0 else y(t + 1 + j)
    end for
    println (s"buildMatrix4TS: xx.dims = ${xx.dims}, yy.dims = ${yy.dims}")
    (xx, yy)
end buildMatrix4TS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ForecasterX` trait provides a common framework for several forecasting 
 *  models that use 1 ENDOGENOUS variable y and 0 or more EXOGENOUS variables xj.
 *  It provides methods for multi-horizon (1 to h) forecasting using the RECURSIVE technique.
 *  Forecasted values are produced only for the endogenous variable y.
 *  Lower case indicates actual values, while upper case is for forecasted values.
 *
 *  Y_t+1 = f(y_t, y_t-1, ... y_t-p+1, x0_t, x0_t-1, ... x0_t-p+1, x1_t, ...)
 *  Y_t+2 = f(Y_t+1, y_t, ... y_t-p+2, x0_t, x0_t-1, ... x0_t-p+1, x1_t, ...)
 *  ...
 *  Y_t+h = f(Y_t+1, y_t, ... y_t-p+2, x0_t, x0_t-1, ... x0_t-p+1, x1_t, ...)
 *
 *  @see Forecaster - when there are no exogenous variables
 *  @param lags  the lags (p) used for endogenous variable (e.g., 10 => use lags 1 to 10)
 */
trait ForecasterX (lags: Int)
      extends Model:

    private   val debug = debugf ("ForecasterX", true)                   // debug function
    protected var yf: MatrixD = null                                     // forecasts for all time points t & horizons to h

    debug ("init", s"lags = $lags")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the internally row trimed and column expanded input matrix and response vector.
     */
    def getXY: (MatrixD, VectorD)                                        // (getX, getY)

    def trainNtest (x_ : MatrixD, y_ : VectorD)
                   (xx: MatrixD, yy: VectorD): (VectorD, VectorD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t+1 using the 1-step ahead forecast.
     *      y_t+1 = f (y_t, ...) + e_t+1
     *  @param t   the time point from which to make prediction
     *  @param yx  the matrix of endogenous y and exogenous x values
     */
    def predict (t: Int, yx: MatrixD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size h, of 1 through h-steps ahead forecasts for the model.
     *      forecast the following time points:  t+1, ..., t+h.
     *  Note, must create the yf matrix before calling the forecast method.
     *  Intended to work with rolling validation (analog of predict method)
     *  @param t   the time point from which to make forecasts
     *  @param yf  the forecast matrix for the endogenous variable y (time x horizons)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forecast (t: Int, yf: MatrixD, h: Int): VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the h-steps ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1.
     *  @param yf  the forecast matrix for the endogenous variable y (time x horizons)
     *  @param yx  the matrix of endogenous y and exogenous x values
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forecastAt (yf: MatrixD, yx: MatrixD, h: Int): VectorD

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
        val sig_h = (y_ - yfh).stdev                                   // standard error of estimate at horizon h
        val width = z_sigma (sig_h, p)                                 // interval half width
        (yfh - width, yfh + width)                                     // return lower and upper bounds
    end forecastAtI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points and all horizons (1 through h-steps ahead).
     *  Record these in the FORECAST MATRIX yf, where
     *      yf(t, k) = k-steps ahead forecast for y_t
     *  Note, column 0, yf(?, 0), is set to y (the actual time-series values).
     *     last column, yf(?, h+1), is set to t (the time values, for reference).
     *  Forecast recursively down diagonals in the yf forecast matrix.
     *  The top right and bottom left triangles in yf matrix are not forecastable.
     *  @param y_  the actual values to use in making forecasts
     *  @param yx  the matrix of endogenous y and exogenous x values
     *  @param h   the maximum forecasting horizon, number of steps ahead to produce forecasts
     */
    def forecastAll (y_ : VectorD, yx: MatrixD, h: Int): MatrixD =
        val yx_ = yx.copy
        debug ("forecastAll", s"y_.dim = ${y_.dim}, yx_.dims = ${yx_.dims}")
        yf = new MatrixD (y_.dim+h, h+2)                               // forecasts for all time points t & horizons to h
        for t <- y_.indices do yf(t, 0) = y_(t)                        // first column is the actual endogenous y values
        for t <- yf.indices do yf(t, h+1) = t                          // last column is time (logical day)

        for k <- 1 to h do
            if k > 1 then yx_.insert (1, lags, yf(?, k-1))             // insert previous forecasts for endogenous variable
//          if k > 1 then yx_.insert (0, lags-1, yf(?, k-1))           // insert previous forecasts for endogenous variable (no intercept)
// FIX - need forecasted values, but also keep yx intact
            forecastAt (yf, yx_, k)                                    // forecast k-steps into the future
        end for
        yf                                                             // return matrix of forecasted values
    end forecastAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test FORECASTS at horizon h of a forecasting model y_ = f(lags (y_), x) + e
     *  and RETURN (1) aligned actual values, (2) the forecasts and (3) QoF vector.
     *  Testing may be in-sample (on the training set) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.  Note: must call train and forecastAll
     *  before testF.
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the testing/full response/output vector
     *  @param yx  the matrix of endogenous y and exogenous x values
     */
    def testF (h: Int, y_ : VectorD, yx: MatrixD): (VectorD, VectorD, VectorD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test FORECASTS over horizons 1 to h of a forecasting model y_ = f(lags (y_), x) + e
     *  and return its forecasts and QoF vector.  Testing may be in-sample
     *  (on the training set) or out-of-sample (on the testing set) as determined
     *  by the parameters passed in.  Note: must call train and forecastAll before testHorizons.
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the testing/full response/output vector
     *  @param yx  the matrix of endogenous y and exogenous x values
    def testHorizons (h: Int, y_ : VectorD, yx: MatrixD): Unit =
        for k <- 1 to h do
            val (yy, yfh, qof) = testF (k, y_, yx)                     // k-steps ahead forecast and its QoF
            println (s"Evaluate QoF for horizon $k:")
            println (FitM.fitMap (qof, qoF_names))                     // evaluate k-steps ahead forecasts
        end for
    end testHorizons
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set up testing by making h-steps ahead FORECASTS, and then aligning actual
     *  and forecasted values.  Helper method for implementations of testF method.
     *  DROP the first h elements.
     *  @param y_      the testing/full response/output vector
     *  @param yx      the matrix of endogenous y and exogenous x values
     *  @param h       the forecasting horizon, number of steps ahead to produce forecasts
     *  @param doPlot  whether to plot predicted and actual values vs. time t
     */
    protected def testSetupF (y_ : VectorD, yx: MatrixD, h: Int, doPlot: Boolean = true): (VectorD, VectorD) =
        if yf == null || yf.dim2 < h+2 then yf = forecastAll (y_, yx, h)
        val yh  = yf(?, h)                                             // get column h of yf (y-forecasted)
        val yy  = y_(h-1 until y_.dim)                                 // note: day 0 already cut out by `ForecastUtil`
        val yfh = yh(h-1 until y_.dim)                                 // align actual and forecasted vectors
        assert (yy.dim == yfh.dim)                                     // make sure the vector sizes agree
        if doPlot then new Plot (null, yy, yfh, s"Plot of yy, yfh for ForecasterX model (h = $h) vs. t", true)
        (yy, yfh)
    end testSetupF

end ForecasterX


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ForecasterX` companion object provides functions useful for classes extending
 *  the `ForecasterX` trait, i.e., forecasting models with multiple input variables.
 */
object ForecasterX:

    private   val debug = debugf ("ForecasterX", true)                   // debug function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the x matrix and y vector into training and testing sets.
     *  @param x      the x data/input matrix
     *  @param y      the y response/output vector
     *  @param ratio  the ratio of the TESTING set to the full dataset (most common 80-20, 70-30)
     */
    def split_TnT (x: MatrixD, y: VectorD, ratio: Double = 0.20): (MatrixD, VectorD, MatrixD, VectorD) =
        val n       = x.dim
        val tr_size = (n * (1.0 - ratio)).toInt
        debug ("split_TnT", s"train: tr_size = $tr_size, test: te_size = ${n - tr_size}")
        (x(0 until tr_size), y(0 until tr_size), x(tr_size until n), y(tr_size until n))
    end split_TnT

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use rolling-validation to compute test Quality of Fit (QoF) measures
     *  by dividing the dataset into a TRAINING SET (tr) and a TESTING SET (te)
     *  as follows:  [ <-- tr_size --> | <-- te_size --> ]
     *  This version calls predict (RECURSIVE) for h-steps ahead out-of-sample forecasts.
     *  @see `RollingValidation`
     *  @param mod      the forecasting model being used (e.g., `RegressionTreeRF4TS`)
     *  @param rc       the retraining cycle (number of forecasts until retraining occurs)
     *  @param hh       the max forecasting horizon (h = 1, 2, ... hh)
     *  @param te_size  the size of the testing set (negative => use ratio to calculate
     */
    def rollValidate (mod: ForecasterX & Fit, rc: Int, hh: Int, te_size_ : Int = -1): Unit =
        val (x, y)  = mod.getXY
        val yf      = mod.forecastAll (y, x, hh)                          // get in-sample forecast matrix - FIX - simplify
        val ftMat   = new MatrixD (hh, Fit.N_QoF)

        val te_size = if te_size_ > 0 then te_size_                       // size of initial testing set
                      else RollingValidation.teSize (y.dim)               // calculate using testing ratio
        val tr_size = y.dim - te_size                                     // size of initial training set
        debug ("rollValidate", s"y.dim = ${y.dim}, train: tr_size = $tr_size; test: te_size = $te_size, rc = $rc")

//      val yp = new VectorD (te_size)                                    // y-predicted over testing set
        for i <- 0 until te_size do                                       // iterate through testing set
            val t = tr_size + i                                           // next time point to forecast
//          if i % rc == 0 then mod.train (x(0 until t), y(0 until t))    // retrain on sliding training set (growing set)
            if i % rc == 0 then mod.train (x(i until t), y(i until t))    // retrain on sliding training set (fixed size set)
//          yp(i) = mod.predict (x(t-1))                                  // predict the next value (only for h=1)
            mod.forecast (t-1, yf, hh)                                    // forecast the next h-values
        end for                                                           // yf is updated down its diagonals

        val df = x.dim2 - 1                                               // degrees of freedom for model (rough approx)
        mod.resetDF (df, te_size - df)                                    // reset degrees of freedom
        val (t, yy) = RollingValidation.align (tr_size, y)                // align vectors

        for h <- 1 to hh do                                               // move thru each horizon 1 to h
            val yfh = yf(tr_size until y.dim, h)
            debug ("rollValidate", s"horizon $h: yy.dim = ${yy.dim}, yfh.dim = ${yfh.dim}")
            new Plot (t, yy, yfh, s"Plot yy, yfh vs. t for horizon h = $h", lines = true)
            ftMat(h-1) = mod.diagnose (yy, yfh)
        end for

        banner (s"rollValidate: Evaluate ${mod.modelName}'s QoF for the horizons: 1 to $hh")
        println ("fitMap     qof = ")
        println (Fit.showFitMap (ftMat.ᵀ))
    end rollValidate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate the quality of point and optionally interval forecast for horizon (h = 1 to hh).
     *  @param mod   the forecasting model to be evaluated
     *  @param y     the testing/full response/output vector
     *  @param yx    the matrix of endogenous y and exogenous x values
     *  @param hh    the maximum forecasting horizon (h = 1 to hh)
     *  @param ints  whether to evaluate prediction interval forecasts as well as point forecasts
     */
    def evalForecasts (mod: ForecasterX & Fit, y: VectorD, yx: MatrixD,
                       hh: Int, ints: Boolean = false): Unit =
        val ftMat = new MatrixD (hh, Fit.N_QoF)
        val mName = mod.modelName
        banner (s"Evaluate ${mName}'s QoF for horizons 1 to $hh:")

        for h <- 1 to hh do
            val (yy, yfh, qof) = mod.testF (h, y, yx)                     // h-steps ahead forecast and its QoF
            ftMat(h-1) = qof
//          println (FitM.fitMap (qof, qoF_names))                        // evaluate h-steps ahead forecasts

            if ints then
                val low_up  = mod.forecastAtI (yy, yfh, h)               // prediction interval forecasts
                val qof_all = mod.diagnose_ (yy, yfh, low_up)            // fully evaluate h-steps ahead forecasts
                mod.showQoF (qof_all)                                    // show all the QoF metrics
                Predictor.plotPredictionInt (yy, yfh, low_up, mName)     // plot ordered actual, predicted, lower, upper
        end for

        println ("fitMap     qof = ")
        println (Fit.showFitMap (ftMat.ᵀ))
    end evalForecasts

end ForecasterX

