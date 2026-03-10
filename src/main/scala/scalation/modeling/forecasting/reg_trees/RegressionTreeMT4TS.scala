
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jun 21 23:13:48 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Model Trees for Time Series
 */

package scalation
package modeling
package forecasting
package reg_trees

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeMT4TS` class supports Model Trees for Time Series data.
 *  Multi-horizon forecasting supported via the Recursive method.
 *  Given a response vector y, a predictor matrix x is built that consists of
 *  lagged y vectors.   Additional future response vectors are built for training.
 *
 *      y_t = f(x)
 *
 *  where x = [y_{t-1}, y_{t-2}, ... y_{t-lags}].
 *
 *  @param x       the input/predictor matrix built out of lags of y
 *                     (and optionally from exogenous variables ex)
 *  @param yy      the output/response vector trimmed to match x.dim
 *  @param lags    the maximum lag included (inclusive)
 *  @param fname   the feature/variable names
 *  @param hparam  the hyper-parameters (use RegressionTree.hp for default)
 */
class RegressionTreeMT4TS (x: MatrixD, yy: VectorD, lags: Int, fname: Array [String] = null,
                           hparam: HyperParameter = RegressionTree.hp)
      extends RegressionTreeMT (x, yy, fname, hparam)
         with ForecasterX (lags):

    private val debug = debugf ("RegressionTreeMT4TS", true)             // debug function
    private val flaw  = flawf ("RegressionTreeMT4TS")                    // flaw function

    _modelName = s"RegressionTreeMT4TS_$lags"

    debug ("init", s"$modelName: x.dims = ${x.dims}, yy.dim = ${yy.dim}")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the internally row trimed and column expanded input matrix and response vector.
     */
    def getXY: (MatrixD, VectorD) = (x, yy)                              // (getX, getY)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t+1 using the 1-step ahead forecast.
     *      y_t+1 = f (y_t, ...) + e_t+1
     *  @param t   the time point from which to make prediction
     *  @param yx  the matrix of endogenous y and exogenous x values
     */
    def predict (t: Int, yx: MatrixD): Double = ???

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size h, of 1 through h-steps ahead forecasts for the model.
     *      forecast the following time points:  t+1, ..., t-1+h.
     *  Note, must create the yf matrix before calling the forecast method.
     *  Intended to work with rolling validation (analog of predict method)
     *  Must call `forecastAll` first.
     *  @param t   the time point from which to make forecasts
     *  @param yf  the forecast matrix (time x horizons)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forecast (t: Int, yf: MatrixD, h: Int): VectorD =
        if h < 1 then flaw ("forecast", s"horizon h = $h must be at least 1")
        VectorD (for k <- 1 to h yield yf(t+k, k))                       // get yf diagonal from time t
    end forecast

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign to FORECAST MATRIX and return h-step ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1.
     *  @see `forecastAll` method in `Forecaster` trait.
     *  @param yf  the forecast matrix for the endogenous variable y (time x horizons)
     *  @param yx  the matrix of endogenous y and exogenous x values
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forecastAt (yf: MatrixD, yx: MatrixD, h: Int): VectorD =
        if h < 2 then flaw ("forecastAt", s"horizon h = $h must be at least 2")

        for t <- yx.indices do                                           // make forecasts over all time points for horizon h
//          yf(t+h-1, h) = b dot yx(min (t, yx.dim-1))                   // forecast down the diagonal ??
            yf(t+h-1, h) = predict (yx(t))                               // forecast using MT predict: t
//          yf(t+h-1, h) = predict (yx(max0 (t-1)))                      // forecast using MT predict: t-1
        yf(?, h)                                                         // return the h-step ahead forecast vector
    end forecastAt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test FORECASTS of a `RegressionTreeMT4TS` forecasting model y_ = f(x) + e
     *  and return its forecasts and QoF vector.  Testing may be in-sample
     *  (on the training set) or out-of-sample (on the testing set) as determined
     *  by the parameters passed in.  Note: must call train and forecastAll before testF.
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the testing/full response/output vector
     *  @param yx  the matrix of endogenous y and exogenous x values
     */
    def testF (h: Int, y_ : VectorD, yx: MatrixD): (VectorD, VectorD, VectorD) =
        val (yy, yfh) = testSetupF (y_, yx, h)                           // get and align actual and forecasted values
        val params = x.dim2 - 1
        resetDF (params, yy.dim - params)                                // reset the degrees of freedom (rough approx)
        println (s"testF: yy.dim = ${yy.dim}, yfh.dim = ${yfh.dim}")
//      differ (yy, yfh)                                                 // uncomment for debugging
        (yy, yfh, diagnose (yy, yfh))                                    // return aligned actual, forecasted and QoF vectors
    end testF

end RegressionTreeMT4TS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeMT4TS` companion object provides factory methods.
 */
object RegressionTreeMT4TS:

    private val debug = debugf ("RegressionTreeMT4TS", true)             // debug function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeMT4TS` object from a response vector.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  @param y          the original un-expanded output/response vector
     *  @param lags       the maximum lag included (inclusive)
     *  @param h          the forecasting horizon (1, 2, ... h)
     *  @param intercept  whether to add a column of all ones to the matrix (intercept)
     *  @param hparam     the hyper-parameters (use RegressionTree.hp for default)
     */
    def apply (y: VectorD, lags: Int, h: Int, intercept: Boolean = true,
               hparam: HyperParameter = RegressionTree.hp): RegressionTreeMT4TS =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        val x  = if intercept then VectorD.one (yy.dim) +^: x_ else x_    // add first column of all ones
        val y_ = yy(?, 0)                                                 // use first column (h = 1)
        debug ("apply", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        new RegressionTreeMT4TS (x, y_, lags, hparam = hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeMT4TS` object from a response matrix.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  This method provides data rescaling.
     *  @param y          the original un-expanded output/response vector
     *  @param lags       the maximum lag included (inclusive)
     *  @param h          the forecasting horizon (1, 2, ... h)
     *  @param intercept  whether to add a column of all ones to the matrix (intercept)
     *  @param hparam     the hyper-parameters (use RegressionTree.hp for default)
     */
    def rescale (y: VectorD, lags: Int, h: Int, intercept: Boolean = true,
                 hparam: HyperParameter = RegressionTree.hp): RegressionTreeMT4TS =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        var x = scale (extreme (x_), (1.0, 5.0))(x_)                      // rescale vector x matrix to [1, 5]
        if intercept then x = VectorD.one (yy.dim) +^: x                  // add first column of all ones
        val y_ = yy(?, 0)                                                 // use first column
        debug ("rescale", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        new RegressionTreeMT4TS (x, y_, lags, hparam = hparam)
    end rescale

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeMT4TS` object from a response vector.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  In addition, lagged exogenous variables are added.
     *  @param y          the original un-expanded output/response vector
     *  @param lags       the maximum lag included (inclusive)
     *  @parax ex         the input matrix for exogenous variables (one per column)
     *  @param h          the forecasting horizon (1, 2, ... h)
     *  @param intercept  whether to add a column of all ones to the matrix (intercept)
     *  @param hparam     the hyper-parameters (use RegressionTree.hp for default)
     */
    def exo (y: VectorD, lags: Int, ex: MatrixD, h: Int,
             intercept: Boolean = true, hparam: HyperParameter = RegressionTree.hp): RegressionTreeMT4TS =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        var x = if intercept then VectorD.one (yy.dim) +^: x_ else x_     // add first column of all ones
        val endoCols = x.dim2
        println (s"endogenous: columns = $endoCols")

//      x = x ++^ ARX.makeExoCols (lags, ex, elag1, elag2)                // add columns for each lagged exo var
        x = x ++^ MakeMatrix4TS.makeMatrix4EXO (ex, lags, 1)              // add columns for each lagged exo var
        println (s"exogenous: columns = ${x.dim2 - endoCols}")

        val y_ = yy(?, 0)                                                 // use first column
        debug ("exo", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        new RegressionTreeMT4TS (x, y_, lags, hparam = hparam)
    end exo

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use rolling-validation to compute test Quality of Fit (QoF) measures
     *  by dividing the dataset into a TRAINING SET (tr) and a TESTING SET (te)
     *  as follows:  [ <-- tr_size --> | <-- te_size --> ]
     *  This version calls predict (RECURSIVE) for h-steps ahead out-of-sample forecasts.
     *  @see `RollingValidation`
     *  @param mod      the forecasting model being used (e.g., `RegressionTreeMT4TS`)
     *  @param rc       the retraining cycle (number of forecasts until retraining occurs)
     *  @param hh       the max forecasting horizon (h = 1, 2, ... hh)
     *  @param te_size  the size of the testing set (negative => use ratio to calculate
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
     */

end RegressionTreeMT4TS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMT4TSTest` main function tests the `RegressionTreeMT4TS` class.
 *  This test is used to CHECK that the `buildMatrix4TS` function is working correctly.
 *  May get NaN for some maximum lags (p) due to multi-collinearity.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeMT4TSTest
 */
@main def regressionTreeMT4TSTest (): Unit =

    val m  = 30
    val y  = VectorD.range (1, m)                                      // used to CHECK the buildMatrix4TS function
    val hh = 3                                                         // max forecasting horizon

    for p <- 5 to 5 do                                                 // autoregressive hyper-parameter p
        banner (s"Test: RegressionTreeMT4TS with $p lags")
        val mod = RegressionTreeMT4TS (y, p, hh)                       // create model for time series data
        mod.trainNtest ()()                                            // train the model on full dataset
        println (mod.summary)

        val yy = mod.getY
        val yp = mod.predict (mod.getX)
        new Plot (null, yy, yp, s"yy vs. yp for ${mod.modelName} (h=1) with $p lags", lines = true)
    end for

end regressionTreeMT4TSTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMT4TSTest2` main function tests the `RegressionTreeMT4TS` class
 *  on real data:  Forecasting lake levels.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeMT4TSTest2
 */
@main def regressionTreeMT4TSTest2 (): Unit =

    import forecasting.Example_LakeLevels.y
    val hh = 1                                                         // max forecasting horizon

    for p <- 1 to 7 do                                                 // autoregressive hyper-parameter p
        banner (s"Test: RegressionTreeMT4TS with $p lags")
        val mod = RegressionTreeMT4TS (y, p, hh)                       // create model for time series data
        mod.trainNtest ()()                                            // train the model on full dataset
        println (mod.summary)

        banner ("Predictions/Forecasts")                               // direct forecasting technique
        val yy = mod.getY
        val yf = mod.predict (mod.getX)
//      for k <- yf.indices2 do
        new Plot (null, yy, yf, s"yy vs. yf for ${mod.modelName} (h=1) with $p lags", lines = true)
        println (s"yf = $yf")
        println (s"yf.dim = ${yf.dim}")
    end for

end regressionTreeMT4TSTest2

import forecasting.Example_Covid.{loadData, NO_EXO, response}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMT4TSTest3` main function tests the `RegressionTreeMT4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous variable only.
 *  Does In-Sample Testing (In_ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeMT4TSTest3
 */
@main def regressionTreeMT4TSTest3 (): Unit =

    val LAGS = 10                                                      // number of lags
    val hh   =  6                                                      // max forecasting horizon
    val hp   = RegressionTree.hp
    hp("maxDepth") = 8

    val (ex, y) = loadData (NO_EXO, response)
    val yy      = y(0 until 116)                                       // clip the flat part of the data
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeMT4TS on COVID-19 Weekly Data")
    val mod = RegressionTreeMT4TS (yy, LAGS, hh)                       // create model for time series data
//  val mod = RegressionTreeMT4TS.rescale (yy, LAGS, hh)               // create model for time series data - scaling
    val yp  = mod.trainNtest ()()._1                                   // train on full and test on full
    val yy_ = yy.drop (1)                                              // can't forecast first point

    new Plot (null, yy_, yp, s"${mod.modelName}, yy_ vs. yp @ h = 1", lines = true)

    val y_yp = MatrixD (yy_, yp).ᵀ
    println (s"y_yp = $y_yp")

    val xx = mod.getX
    mod.forecastAll (yy, xx, hh)
    ForecasterX.evalForecasts (mod, yy, xx, hh)

/*
    banner (s"Feature Selection Technique: Stepwise")
    val (cols, rSq) = mod.stepwiseSelAll (cross = false)               // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionTreeMT4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"Stepwise: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")
*/

end regressionTreeMT4TSTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMT4TSTest4` main function tests the `RegressionTreeMT4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous variables.
 *  Does Train-n-Test (TnT) Split testing on the model.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeMT4TSTest4
 */
@main def regressionTreeMT4TSTest4 (): Unit =

    val yy = forecasting.Example_Covid.loadData_y (response)
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val rc = 1                                                          // retraining cycle
    val hh = 6                                                          // max forecasting horizon
    val hp = RegressionTree.hp
    hp("maxDepth") = 8

    val te_size = RollingValidation.teSize (y.dim)
    println (s"te_size = $te_size")

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 5 to 7 do                                                  // p (LAGS) can't be too small for MT
        val mod = RegressionTreeMT4TS (y, p, hh)                        // create an MT (no exo) model
        banner (s"TnT Test: ${mod.modelName} Model")
        mod.trainNtest ()()
        ForecasterX.rollValidate (mod, rc, hh, te_size)         // FIX - need to handle multiple horizons
    end for

end regressionTreeMT4TSTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMT4TSTest5` main function tests the `RegressionTreeMT4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeMT4TSTest5
 */
@main def regressionTreeMT4TSTest5 (): Unit =

    val LAGS = 10                                                           // number of lags
    val hh   = 6                                                            // max forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeMT4TS.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeMT4TS.exo (y, LAGS, ex, hh)                     // create model for time series data - with exo
    val yp  = mod.trainNtest ()()._1                                        // train on full and test on full
    val yy_ = y(LAGS until y.dim)
    new Plot (null, yy_, yp, s"${mod.modelName}, yy vs. yp", lines = true)

//  val tech = SelectionTech.Forward                                        // pick one feature selection technique
//  val tech = SelectionTech.Backward
    val tech = SelectionTech.Stepwise

    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = "none")             // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionTreeMT4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

end regressionTreeMT4TSTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMT4TSTest6` main function tests the `RegressionTreeMT4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  Run Train-n-Test (TnT) Split testing on best model.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeMT4TSTest6
 */
@main def regressionTreeMT4TSTest6 (): Unit =

    val LAGS = 10                                                           // number of lags
    val hh   = 6                                                            // max forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeMT4TS.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeMT4TS.exo (y, LAGS, ex, hh)                     // create model for time series data with exo
    val yp  = mod.trainNtest ()()._1                                        // train on full and test on full
    val yy_ = y(LAGS until y.dim)
    new Plot (null, yy_, yp, s"${mod.modelName}, yy vs. yp", lines = true)

//  val tech = SelectionTech.Forward                                        // pick one feature selection technique
//  val tech = SelectionTech.Backward
    val tech = SelectionTech.Stepwise

    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = "none")             // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionTreeMT4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

    banner ("Run TnT on Best model")
//  val bmod = mod.getBest._4                                               // get the best model from feature selection
    val bmod = mod.getBest.mod.asInstanceOf [RegressionTreeMT4TS]           // get the best model from feature selection
    val (x_, y_, xtest, ytest) = ForecasterX.split_TnT (bmod.getX, bmod.getY)
    val yptest = bmod.trainNtest (x_, y_)(xtest, ytest)._1                  // train on (x_, y_) and test on (xtest, ytest)
    new Plot (null, ytest, yptest, s"${mod.modelName}, ytest vs. yptest", lines = true)

end regressionTreeMT4TSTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMT4TSTest7` main function tests the `RegressionTreeMT4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  Run Train-n-Test (TnT) Split testing on best model using Rolling Validation.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeMT4TSTest7
 */
@main def regressionTreeMT4TSTest7 (): Unit =

    val LAGS = 10                                                           // number of lags (values from past)
    val rc   = 1                                                            // retraining cycle
    val hh   = 6                                                            // max forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    val te_size = RollingValidation.teSize (y.dim)
    println (s"te_size = $te_size")

    banner ("Test In-Sample RegressionTreeMT4TS.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeMT4TS.exo (y, LAGS, ex, hh)                     // create model for time series data with exo
    val yp = mod.trainNtest ()()._1                                         // train on full and test on full
    val yy_ = y(LAGS until y.dim)
    new Plot (null, yy_, yp, s"${mod.modelName}, yy vs. yp", lines = true)

//  val tech = SelectionTech.Forward                                        // pick one feature selection technique
//  val tech = SelectionTech.Backward
    val tech = SelectionTech.Stepwise

    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = "none")             // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionTreeMT4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

    banner ("Run Rolling Validation on RegressionTreeMT4TS Best model")
//  val bmod = mod.getBest._4
    val bmod = mod.getBest.mod.asInstanceOf [RegressionTreeMT4TS]            // get the best model from feature selection - FIX: cast
    ForecasterX.rollValidate (bmod, rc, hh, te_size)

end regressionTreeMT4TSTest7

