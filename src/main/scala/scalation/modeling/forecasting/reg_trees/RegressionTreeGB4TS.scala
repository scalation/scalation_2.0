
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jun 21 23:13:48 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Gradient Boosting for Time Series
 */

package scalation
package modeling
package forecasting
package reg_trees

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeGB4TS` class supports Gradient Boosting for Time Series data.
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
 *  @param hparam  the hyper-parameters (use REGRESSION.hp for default)
 */
class RegressionTreeGB4TS (x: MatrixD, yy: VectorD, lags: Int, fname: Array [String] = null,
                           hparam: HyperParameter = RegressionTree.hp)
      extends RegressionTreeGB (x, yy, fname, hparam)
         with ForecasterX (lags):

    private val debug = debugf ("RegressionTreeGB4TS", true)             // debug function
    private val flaw  = flawf ("RegressionTreeGB4TS")                    // flaw function

    _modelName = s"RegressionTreeGB4TS_$lags"

    debug ("init", s"$modelName: x.dims = ${x.dims}, yy.dim = ${yy.dim}")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the internally row trimed and column expanded input matrix and reposnse vector.
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
            yf(t+h-1, h) = predict (yx(t))                               // forecast using GB predict: t
//          yf(t+h-1, h) = predict (yx(max0 (t-1)))                      // forecast using GB predict: t-1
        yf(?, h)                                                         // return the h-step ahead forecast vector
    end forecastAt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test FORECASTS of a `RegressionTreeGB4TS` forecasting model y_ = f(x) + e
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

end RegressionTreeGB4TS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeGB4TS` companion object provides factory methods.
 */
object RegressionTreeGB4TS:

    private val debug = debugf ("RegressionTreeGB4TS", true)             // debug function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeGB4TS` object from a response vector.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  @param y          the original un-expanded output/response vector
     *  @param lags       the maximum lag included (inclusive)
     *  @param h          the forecasting horizon (1, 2, ... h)
     *  @param intercept  whether to add a column of all ones to the matrix (intercept)
     *  @param hparam     the hyper-parameters (use RegressionTree.hp for default)
     */
    def apply (y: VectorD, lags: Int, h: Int, intercept: Boolean = true,
               hparam: HyperParameter = RegressionTree.hp): RegressionTreeGB4TS =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        val x  = if intercept then VectorD.one (yy.dim) +^: x_ else x_    // add first column of all ones
        val y_ = yy(?, 0)                                                 // use first column (h = 1)
        debug ("apply", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        new RegressionTreeGB4TS (x, y_, lags, null, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeGB4TS` object from a response matrix.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  This method provides data rescaling.
     *  @param y          the original un-expanded output/response vector
     *  @param lags       the maximum lag included (inclusive)
     *  @param h          the forecasting horizon (1, 2, ... h)
     *  @param intercept  whether to add a column of all ones to the matrix (intercept)
     *  @param hparam     the hyper-parameters (use RegressionTree.hp for default)
     */
    def rescale (y: VectorD, lags: Int, h: Int, intercept: Boolean = true,
                 hparam: HyperParameter = RegressionTree.hp): RegressionTreeGB4TS =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        var x = scale (extreme (x_), (1.0, 5.0))(x_)                      // rescale vector x matrix to [1, 5]
        if intercept then x = VectorD.one (yy.dim) +^: x                  // add first column of all ones
        val y_ = yy(?, 0)                                                 // use first column
        debug ("rescale", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        new RegressionTreeGB4TS (x, y_, lags, null, hparam)
    end rescale

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeGB4TS` object from a response vector.  The input/data matrix
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
             intercept: Boolean = true, hparam: HyperParameter = RegressionTree.hp): RegressionTreeGB4TS =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        var x = if intercept then VectorD.one (yy.dim) +^: x_ else x_     // add first column of all ones
        val endoCols = x.dim2
        println (s"endogenous: columns = $endoCols")

//      x = x ++^ ARX.makeExoCols (lags, ex, elag1, elag2)                // add columns for each lagged exo var
        x = x ++^ MakeMatrix4TS.makeMatrix4EXO (ex, lags, 1)              // add columns for each lagged exo var
        println (s"exogenous: columns = ${x.dim2 - endoCols}")

        val y_ = yy(?, 0)                                                  // use first column
        debug ("exo", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        new RegressionTreeGB4TS (x, y_, lags, null, hparam)
    end exo

end RegressionTreeGB4TS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TSTest` main function tests the `RegressionTreeGB4TS` class.
 *  This test is used to CHECK that the `buildMatrix4TS` function is working correctly.
 *  May get NaN for some maximum lags (p) due to multi-collinearity.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TSTest
 */
@main def regressionTreeGB4TSTest (): Unit =

    val m  = 30
    val y  = VectorD.range (1, m)                                      // used to CHECK the buildMatrix4TS function
    val hh = 3                                                         // max forecasting horizon

    for p <- 5 to 5 do                                                 // autoregressive hyper-parameter p
        banner (s"Test: RegressionTreeGB4TS with $p lags")
        val mod = RegressionTreeGB4TS (y, p, hh)                       // create model for time series data
        mod.trainNtest ()()                                            // train the model on full dataset
        println (mod.summary)

        val yy = mod.getY
        val yp = mod.predict (mod.getX)
        new Plot (null, yy, yp, s"yy vs. yp for ${mod.modelName} (h=1) with $p lags", lines = true)
    end for

end regressionTreeGB4TSTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TSTest2` main function tests the `RegressionTreeGB4TS` class
 *  on real data:  Forecasting lake levels.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TSTest2
 */
@main def regressionTreeGB4TSTest2 (): Unit =

    import forecasting.Example_LakeLevels.y
    val hh = 1                                                         // max forecasting horizon

    for p <- 1 to 7 do                                                 // autoregressive hyper-parameter p
        banner (s"Test: RegressionTreeGB4TS with $p lags")
        val mod = RegressionTreeGB4TS (y, p, hh)                       // create model for time series data
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

end regressionTreeGB4TSTest2

import forecasting.Example_Covid.{loadData, NO_EXO, response}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TSTest3` main function tests the `RegressionTreeGB4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous variable only.
 *  Does In-Sample Testing (In_ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TSTest3
 */
@main def regressionTreeGB4TSTest3 (): Unit =

    val LAGS = 10                                                      // number of lags
    val hh   =  6                                                      // max forecasting horizon

    val (ex, y) = loadData (NO_EXO, response)
    val yy      = y(0 until 116)                                       // clip the flat part of the data
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeGB4TS on COVID-19 Weekly Data")
    val mod = RegressionTreeGB4TS (yy, LAGS, hh)                       // create model for time series data
//  val mod = RegressionTreeGB4TS.rescale (yy, LAGS, hh)               // create model for time series data - scaling
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
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionTreeGB4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"Stepwise: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")
*/

end regressionTreeGB4TSTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TSTest4` main function tests the `RegressionTreeGB4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous variables.
 *  Does Train-n-Test (TnT) Split testing on the model.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TSTest4
 */
@main def regressionTreeGB4TSTest4 (): Unit =

    val yy = forecasting.Example_Covid.loadData_y (response)
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val rc = 1                                                          // retraining cycle
    val hh = 6                                                          // max forecasting horizon

    val te_size = RollingValidation.teSize (y.dim)
    println (s"te_size = $te_size")

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 5 to 7 do                                                  // p (LAGS) can't be too small for GB
        val mod = RegressionTreeGB4TS (y, p, hh)                        // create an GB (no exo) model
        banner (s"TnT Test: ${mod.modelName} Model")
        mod.trainNtest ()()
//      RegressionTreeGB4TS.rollValidate (mod, rc, hh, te_size)
        ForecasterX.rollValidate (mod, rc, hh, te_size)
    end for

end regressionTreeGB4TSTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TSTest5` main function tests the `RegressionTreeGB4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TSTest5
 */
@main def regressionTreeGB4TSTest5 (): Unit =

    val LAGS = 10                                                           // number of lags
    val hh   = 6                                                            // max forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeGB4TS.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeGB4TS.exo (y, LAGS, ex, hh)                     // create model for time series data - with exo
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
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionTreeGB4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

end regressionTreeGB4TSTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TSTest6` main function tests the `RegressionTreeGB4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  Run Train-n-Test (TnT) Split testing on best model.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TSTest6
 */
@main def regressionTreeGB4TSTest6 (): Unit =

    val LAGS = 10                                                           // number of lags
    val hh   = 6                                                            // max forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeGB4TS.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeGB4TS.exo (y, LAGS, ex, hh)                     // create model for time series data with exo
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
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionTreeGB4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

    banner ("Run TnT on Best model")
//  val bmod = mod.getBest._4                                               // get the best model from feature selection
    val bmod = mod.getBest.mod.asInstanceOf [RegressionTreeGB4TS]           // get the best model from feature selection
    val (x_, y_, xtest, ytest) = ForecasterX.split_TnT (bmod.getX, bmod.getY)
    val yptest = bmod.trainNtest (x_, y_)(xtest, ytest)._1                  // train on (x_, y_) and test on (xtest, ytest)
    new Plot (null, ytest, yptest, s"${mod.modelName}, ytest vs. yptest", lines = true)

end regressionTreeGB4TSTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TSTest7` main function tests the `RegressionTreeGB4TS` class
 *  on real data:  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  Run Train-n-Test (TnT) Split testing on best model using Rolling Validation.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TSTest7
 */
@main def regressionTreeGB4TSTest7 (): Unit =

    val LAGS = 10                                                           // number of lags (values from past)
    val rc   = 1                                                            // retraining cycle
    val hh   = 6                                                            // max forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    val te_size = RollingValidation.teSize (y.dim)
    println (s"te_size = $te_size")

    banner ("Test In-Sample RegressionTreeGB4TS.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeGB4TS.exo (y, LAGS, ex, hh)                     // create model for time series data with exo
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
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionTreeGB4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

    banner ("Run Rolling Validation on RegressionTreeGB4TS Best model")
//  val bmod = mod.getBest._4
    val bmod = mod.getBest.mod.asInstanceOf [RegressionTreeGB4TS]           // get the best model from feature selection
//  RegressionTreeGB4TS.rollValidate (bmod, rc, hh, te_size)
    ForecasterX.rollValidate (bmod, rc, hh, te_size)

end regressionTreeGB4TSTest7

