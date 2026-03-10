
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
/** The `RegressionTreeGB4TS2` object supports Gradient Boosting for Time Series data.
 *  Multi-horizon forecasting supported via the Recursive method.
 *  Given a response vector y, a predictor matrix x is built that consists of
 *  lagged y vectors.  Additional future response vectors are built for training.
 *
 *      y_t = f(x)
 *
 *  where x = [y_{t-1}, y_{t-2}, ... y_{t-lags}].
 */
object RegressionTreeGB4TS2:

    private val debug = debugf ("RegressionTreeGB4TS2", true)                           // debug function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeGB` object from a response vector.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  @param y          the original un-expanded output/response vector
     *  @param lags       the maximum lag included (inclusive)
     *  @param h          the forecasting horizon (1, 2, ... h)
     *  @param intercept  whether to add a column of all ones to the matrix (intercept)
     *  @param hparam     the hyper-parameters (use RegressionTree.hp for default)
     */
    def apply (y: VectorD, lags: Int, h: Int, intercept: Boolean = true,
               hparam: HyperParameter = RegressionTree.hp): RegressionTreeGB =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        val x  = if intercept then VectorD.one (yy.dim) +^: x_ else x_    // add first column of all ones
        val y_ = yy(?, 0)                                                 // use first column (h = 1)
        debug ("apply", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        val mod = new RegressionTreeGB (x, y_, null, hparam)
        mod.modelName = s"RegressionTreeGB4TS2$lags"
        mod
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeGB` object from a response matrix.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  This method provides data rescaling.
     *  @param y          the original un-expanded output/response vector
     *  @param lags       the maximum lag included (inclusive)
     *  @param h          the forecasting horizon (1, 2, ... h)
     *  @param intercept  whether to add a column of all ones to the matrix (intercept)
     *  @param hparam     the hyper-parameters (use RegressionTree.hp for default)
     */
    def rescale (y: VectorD, lags: Int, h: Int, intercept: Boolean = true,
                 hparam: HyperParameter = RegressionTree.hp): RegressionTreeGB =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        var x = scale (extreme (x_), (1.0, 5.0))(x_)                      // rescale vector x matrix to [1, 5]
        if intercept then x = VectorD.one (yy.dim) +^: x                  // add first column of all ones
        val y_ = yy(?, 0)                                                 // use first column
        debug ("rescale", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        val mod = new RegressionTreeGB (x, y_, null, hparam)
        mod.modelName = s"RegressionTreeGB4TS2$lags"
        mod
    end rescale

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeGB` object from a response vector.  The input/data matrix
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
             intercept: Boolean = true, hparam: HyperParameter = RegressionTree.hp): RegressionTreeGB =
        val (x_, yy) = buildMatrix4TS (y, lags, h)                        // column for each lag
        var x = if intercept then VectorD.one (yy.dim) +^: x_ else x_     // add first column of all ones
        val endoCols = x.dim2
        println (s"endogenous: columns = $endoCols")

//      x = x ++^ ARX.makeExoCols (lags, ex, elag1, elag2)                // add columns for each lagged exo var
        x = x ++^ MakeMatrix4TS.makeMatrix4EXO (ex, lags, 1)              // add columns for each lagged exo var
        println (s"exogenous: columns = ${x.dim2 - endoCols}")

        val y_ = yy(?, 0)                                                 // use first column
        debug ("exo", s"x.dims = ${x.dims}, y_.dim = ${y_.dim}")

        val mod = new RegressionTreeGB (x, y_, null, hparam)
        mod.modelName = s"RegressionTreeGB4TS2.exo_$lags"
        mod
    end exo

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the x matrix and y matrix into training and testing sets.
     *  @param x      the x data/input matrix
     *  @param y      the y response/output matrix
     *  @param ratio  the ratio of the TESTING set to the full dataset (most common 70-30, 80-20)
     */
//  def split_TnT (x: MatrixD, y: MatrixD, ratio: Double = 0.20): (MatrixD, MatrixD, MatrixD, MatrixD) =
    def split_TnT (x: MatrixD, y: VectorD, ratio: Double = 0.20): (MatrixD, VectorD, MatrixD, VectorD) =
        val n       = x.dim
        val tr_size = (n * (1.0 - ratio)).toInt
        println (s"RegressionTreeGB4TS2.split_TnT: tr_size = $tr_size, te_size = ${n - tr_size}")
        (x(0 until tr_size), y(0 until tr_size), x(tr_size until n), y(tr_size until n))
    end split_TnT 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Align (for the testing set) the actual response vector for comparison with
     *  the predicted/forecasted response vector, returning a time vector and sliced
     *  response vectors.
     *  @param tr_size  the size of the intial training set
     *  @param y        the actual response for the full dataset (to be sliced)
     *  @param yp       the predicted response for the full dataset (to be sliced)
     *  @param h_       the current forecasting horizon - 1
     */
    def align (tr_size: Int, y: VectorD, yp: VectorD, h_ : Int): (VectorD, VectorD, VectorD) =
        debug ("align:", s"y.dim = ${y.dim}, yp.dim = ${yp.dim}, h_ = $h_")
        (VectorD.range (tr_size, y.dim - h_), y(tr_size until y.dim - h_), yp(0 until yp.dim - h_))
    end align

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use rolling-validation to compute test Quality of Fit (QoF) measures
     *  by dividing the dataset into a TRAINING SET (tr) and a TESTING SET (te)
     *  as follows:  [ <-- tr_size --> | <-- te_size --> ]
     *  This version calls predict (DIRECT) for h-steps ahead out-of-sample forecasts.
     *  @see `RollingValidation`
     *  @param mod      the forecasting model being used (e.g., `RegressionTreeGB4TS2`)
     *  @param rc       the retraining cycle (number of forecasts until retraining occurs)
     *  @param te_size  the size of the testing set
     */
    def rollValidate (mod: Predictor & Fit, rc: Int, te_size_ : Int): VectorD =
        val x       = mod.getX                                            // get data/input matrix
        val y       = mod.getY                                            // get response/output matrix
        val hh      = 1
        val ftMat   = new MatrixD (hh, Fit.N_QoF)
        banner (s"rollValidate: Evaluate ${mod.modelName}'s QoF for the horizons: 1 to $hh")

        val te_size = if te_size_ < 0 then RollingValidation.teSize (y.dim) else te_size_  // size of test set
        val tr_size = y.dim - te_size                                                      // size of initial training set
        debug ("rollValidate", s"y.dim = ${y.dim}, train: tr_size = $tr_size; test: te_size = $te_size, rc = $rc")

//      val yp = new MatrixD (te_size, y.dim2)                            // y-predicted over testing set
        val yp = new VectorD (te_size)                                    // y-predicted over testing set
        for i <- 0 until te_size do                                       // iterate through testing set
            val t = tr_size + i                                           // next time point to forecast
//          if i % rc == 0 then mod.train (x(0 until t), y(0 until t))    // retrain on sliding training set (growing set)
            if i % rc == 0 then mod.train (x(i until t), y(i until t))    // retrain on sliding training set (fixed size set)
            yp(i) = mod.predict (x(t-1))                                  // predict the next value
        end for

        val df = max0 (mod.parameter.dim - 1)                             // degrees of freedom for model
        mod.resetDF (df, te_size - df)                                    // reset degrees of freedom

        for k <- 0 until hh do                                            // move thru each horizon 1 to h
            val (t, yk, ypk) = align (tr_size, y, yp, k)                  // clip ending zeros (0.0 or -0.0)
            debug ("rollValidate", s"horizon $k: yk.dim = ${yk.dim}, ypk.dim = ${ypk.dim}")
            new Plot (t, yk, ypk, s"Plot yy, yp vs. t for horizon ${k+1}", lines = true)
            val qof  = mod.diagnose (yk, ypk)
            ftMat(k) = qof
//          println (FitM.fitMap (qof, qoF_names))
        end for
        println ("fitMap     qof = ")
        println (FitM.showFitMap (ftMat.transpose, QoF.values.map (_.toString)))
        yp
    end rollValidate

end RegressionTreeGB4TS2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TS2Test` main function tests the `RegressionTreeGB4TS2` class.
 *  This test is used to CHECK that the `buildMatrix4TS` function is working correctly.
 *  May get NaN for some maximum lags (p) due to multi-collinearity.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TS2Test
 */
@main def regressionTreeGB4TS2Test (): Unit =

    val m = 30
    val y = VectorD.range (1, m)                                       // used to CHECK the buildMatrix4TS function
    val h = 3                                                          // the forecasting horizon

    for p <- 5 to 5 do                                                 // autoregressive hyper-parameter p
        banner (s"Test: RegressionTreeGB4TS2 with $p lags")
        val mod = RegressionTreeGB4TS2 (y, p, h)                                     // create model for time series data
        mod.trainNtest ()()                                            // train the model on full dataset
        println (mod.summary)

        val yy = mod.getY
        val yp = mod.predict (mod.getX)
        new Plot (null, yy, yp, s"yy vs. yp for ${mod.modelName} (h=1) with $p lags", lines = true)
    end for

end regressionTreeGB4TS2Test


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TS2Test2` main function tests the `RegressionTreeGB4TS2` class on real data:
 *  Forecasting lake levels.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TS2Test2
 */
@main def regressionTreeGB4TS2Test2 (): Unit =

    import forecasting.Example_LakeLevels.y
    val h = 1                                                          // the forecasting horizon

    for p <- 1 to 7 do                                                 // autoregressive hyper-parameter p
        banner (s"Test: RegressionTreeGB4TS2 with $p lags")
        val mod = RegressionTreeGB4TS2 (y, p, h)                                     // create model for time series data
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

end regressionTreeGB4TS2Test2

import forecasting.Example_Covid.{loadData, NO_EXO, response}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TS2Test3` main function tests the `RegressionTreeGB4TS2` class on real data:
 *  Forecasts COVID-19 Weekly Data using endogenous variable only.
 *  Does In-Sample Testing (In_ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TS2Test3
 */
@main def regressionTreeGB4TS2Test3 (): Unit =

    val LAGS = 10                                                      // number of lags
    val h    =  6                                                      // forecasting horizon

    val (ex, y) = loadData (NO_EXO, response)
    val yy      = y(0 until 116)                                       // clip the flat part of the data
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeGB4TS2 on COVID-19 Weekly Data")
    val mod = RegressionTreeGB4TS2 (yy, LAGS, h)                       // create model for time series data
//  val mod = RegressionTreeGB4TS2.rescale (yy, LAGS, h)               // create model for time series data - scaling
    val yp  = mod.trainNtest ()()._1                                   // train on full and test on full
    val yy_ = yy.drop (1)                                              // can't forecast first point

    new Plot (null, yy_, yp, s"${mod.modelName}, yy_ vs. yp @ h = 1", lines = true)

    val y_yp = MatrixD (yy_, yp).transpose
    println (s"y_yp = $y_yp")

//  mod.forecastAll (yy, h)                                            // FIX - to be implemented - see ARX.scala
//  Forecaster.evalForecasts (mod, yy, h)                              // FIX - to be implemented - see ARX.scala

/*
    banner (s"Feature Selection Technique: Stepwise")
    val (cols, rSq) = mod.stepwiseSelAll (cross = false)               // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for RegressionTreeGB4TS2 with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"Stepwise: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")
*/

end regressionTreeGB4TS2Test3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TS2Test4` main function tests the `RegressionTreeGB4TS2` class on real data:
 *  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TS2Test4
 */
@main def regressionTreeGB4TS2Test4 (): Unit =

    val LAGS = 10                                                           // number of lags
    val h    = 6                                                            // forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeGB4TS2.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeGB4TS2.exo (y, LAGS, ex, h)                     // create model for time series data - with exo
    val yp  = mod.trainNtest ()()._1                                        // train on full and test on full
    val yy_ = y(LAGS until y.dim)
    new Plot (null, yy_, yp, s"${mod.modelName}, yy vs. yp", lines = true)

//  val tech = SelectionTech.Forward                                        // pick one feature selection technique
//  val tech = SelectionTech.Backward
    val tech = SelectionTech.Stepwise

    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = false)              // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for RegressionTreeGB4TS2 with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

end regressionTreeGB4TS2Test4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TS2Test5` main function tests the `RegressionTreeGB4TS2` class on real data:
 *  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  Run Train-n-Test (TnT) Split testing on best model.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TS2Test5
 */
@main def regressionTreeGB4TS2Test5 (): Unit =

    val LAGS = 10                                                           // number of lags
    val h    = 6                                                            // forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    banner ("Test In-Sample RegressionTreeGB4TS2.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeGB4TS2.exo (y, LAGS, ex, h)                     // create model for time series data with exo
    val yp  = mod.trainNtest ()()._1                                        // train on full and test on full
    val yy_ = y(LAGS until y.dim)
    new Plot (null, yy_, yp, s"${mod.modelName}, yy vs. yp", lines = true)

//  val tech = SelectionTech.Forward                                        // pick one feature selection technique
//  val tech = SelectionTech.Backward
    val tech = SelectionTech.Stepwise

    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = false)              // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for RegressionTreeGB4TS2 with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

    banner ("Run TnT on Best model")
//  val bmod = mod.getBest._4                                               // get the best model from feature selection
    val bmod = mod.getBest.mod.asInstanceOf [RegressionTreeGB]              // get the best model from feature selection
    val (x_, y_, xtest, ytest) = RegressionTreeGB4TS2.split_TnT (bmod.getX, bmod.getY)
    val yptest = bmod.trainNtest (x_, y_)(xtest, ytest)._1                  // train on (x_, y_) and test on (xtest, ytest)
    new Plot (null, ytest, yptest, s"${mod.modelName}, ytest vs. yptest", lines = true)

end regressionTreeGB4TS2Test5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGB4TS2Test6` main function tests the `RegressionTreeGB4TS2` class on real data:
 *  Forecasts COVID-19 Weekly Data using endogenous and exogenous variables.
 *  Does In-Sample Testing (In-ST).
 *  Determines the terms to include in the model using Feature Selection.
 *  Run Train-n-Test (TnT) Split testing on best model using Rolling Validation.
 *  > runMain scalation.modeling.forecasting.reg_trees.regressionTreeGB4TS2Test6
 */
@main def regressionTreeGB4TS2Test6 (): Unit =

    val LAGS = 10                                                           // number of lags (values from past)
    val rc   = 1                                                            // retraining cycle
    val h    = 6                                                            // forecasting horizon

    val exo_vars = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (ex, y)  = loadData (exo_vars, response)
    println (s"ex.dims = ${ex.dims}, y.dim = ${y.dim}")

    val te_size = RollingValidation.teSize (y.dim)
    println (s"te_size = $te_size")

    banner ("Test In-Sample RegressionTreeGB4TS2.exo on COVID-19 Weekly Data")
    val mod = RegressionTreeGB4TS2.exo (y, LAGS, ex, h)                     // create model for time series data with exo
    val yp  = mod.trainNtest ()()._1                                        // train on full and test on full
    val yy_ = y(LAGS until y.dim)
    new Plot (null, yy_, yp, s"${mod.modelName}, yy vs. yp", lines = true)

//  val tech = SelectionTech.Forward                                        // pick one feature selection technique
//  val tech = SelectionTech.Backward
    val tech = SelectionTech.Stepwise

    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = false)              // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for RegressionTreeGB4TS2 with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

    banner ("Run Rolling Validation on RegressionTreeGB4TS2 Best model")
//  val bmod = mod.getBest._4                                               // get the best model from feature selection
    val bmod = mod.getBest.mod.asInstanceOf [RegressionTreeGB]              // get the best model from feature selection
    RegressionTreeGB4TS2.rollValidate (bmod, rc, te_size)

end regressionTreeGB4TS2Test6

