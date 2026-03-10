
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep  2 14:37:55 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Vector AutoRegressive (VARX)
 *
 *  @see     phdinds-aim.github.io/time_series_handbook/03_VectorAutoregressiveModels/03_VectorAutoregressiveMethods.html
 *           www.lem.sssup.it/phd/documents/Lesson17.pdf
 *           Parameter/coefficient estimation: Multi-variate Ordinary Least Squares (OLS) or
 *                                             Generalized Least Squares (GLS)
 */

package scalation
package modeling
package forecasting
package multivar

import scala.collection.mutable.ArrayBuffer

import scalation.mathstat._

import MakeMatrix4TS._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VARX` class provides multi-variate time series analysis capabilities for VARX models.
 *  VARX models are related to `ARX` and `VAR` models, with some of the exogenous variables
 *  treated as endogenous variables and are themselves forecasted.  Potentially having more
 *  up-to-date forecasted values feeding into multi-horizon forecasting can improve
 *  accuracy, but may also lead to compounding of forecast errors.
 *  Given multi-variate time series data where matrix x holds the input and matrix y holds 
 *  the output, the next vector value y_t = combination of last p vector values in x.
 *
 *      y_t = bb dot x_t + e_t
 *
 *  where y_t is the value of y at time t, bb is the parameter matrix and e_t is the 
 *  residual/error term.
 *  @param y        the response/output matrix (multi-variate time series data)
 *  @param x        the input lagged time series data
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class VARX (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String] = null,
            tRng: Range = null, hparam: HyperParameter = hp,
            bakcast: Boolean = false)                                     // backcasted values only used in `buildMatrix4TS`
//    extends Forecaster_D (x, y, hh, tRng, hparam, bakcast):             // no automatic backcasting, @see `VARX.apply`
      extends Forecaster_RegV (x, y, hh, fname, tRng, hparam, bakcast):

    private val debug = debugf ("VARX", true)                             // debug function
    private val p     = hparam("p").toInt                                 // use the last p endogenous values for each endo variable (p lags)
    private val q     = hparam("q").toInt                                 // use the last q exogenous values for each exo variable (q lags)
    private val spec  = hparam("spec").toInt                              // trend terms: 0 - none, 1 - constant, 2 - linear, 3 - quadratic
                                                                          //              4 - sine, 5 cosine
    private val n      = y.dim2                                           // the total number of variables
    private val n_endo = n - n_exo                                        // the number of endogenous variables

    _modelName = s"VARX_${p}_${q}_$n"

    debug ("init", s"$modelName with $n_endo, $n_exo endo, exo variables and additional term spec = $spec")
    debug ("init", s"[ x | y ] = ${x ++^ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a new vector from the first spec values of x, the last p-h+1 values
     *  of x (past values), values 1 to h-1 from the forecasts, and available values
     *  from exogenous variables.
     *  @param xx  the t-th row of the input matrix (lagged actual values)
     *  @param yy  the t-th row of the forecast tensor (forecasted future values)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    def forge (xx: VectorD, yy: MatrixD, h: Int): VectorD =
        val xy = new VectorD (spec + n * p)
        xy(0 until spec) = xx(0 until spec)                               // get trend values

        var jend = spec + p                                               // ending j index
        for j <- 0 until n do                                             // for each variable
            val x_act   = xx(jend-(p+1-h) until jend)                     // get actual lagged y-values
            val nyy     = p - x_act.dim                                   // number of forecasted values needed
            val x_fcast = yy(h-nyy until h, j)                            // get forecasted y-values
            xy(jend-p until jend) = x_act ++ x_fcast
            jend += p
        end for
        xy
    end forge

end VARX


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VARX` object supports regression for Multivariate Time Series data.
 *  Given a response matrix y, a predictor matrix x is built that consists of
 *  lagged y vectors.   Additional future response vectors are built for training.
 */
object VARX:

//  private val debug = debugf ("VARX", true)                                   // debug function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VARX` object from a response matrix.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the response/output matrix (multi-variate time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: MatrixD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = hp,
               bakcast: Boolean = false): VARX =                               // backcasted values only used in `buildMatrix`
        val xy    = buildMatrix (xe, y, hparam, bakcast)
        val fname = if fname_ == null then formNames (xe.dim2, hparam) else fname_
        new VARX (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the p + spec columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the matrix vector (time series data)
     *  @param hp_      the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def buildMatrix (xe: MatrixD, y: MatrixD, hp_ : HyperParameter, bakcast: Boolean): MatrixD =
        val (p, q, spec, lwave) = (hp_("p").toInt, hp_("q").toInt, hp_("spec").toInt, hp_("lwave").toDouble)
        makeMatrix4T (y(?, 0), spec, lwave, bakcast) ++^             // trend terms
        makeMatrix4L (y, p, bakcast)  ++^                            // regular lag terms
        makeMatrix4EXO (xe, q, 1, bakcast)                           // add exogenous terms
    end buildMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo  the number of exogenous variable
     *  @param hp_    the hyper-parameters
     */
    def formNames (n_exo: Int, hp_ : HyperParameter): Array [String] =
        val (p, q, spec) = (hp_("p").toInt, hp_("q").toInt, hp_("spec").toInt)
        val names = ArrayBuffer [String] ()
        for j <- 0 until n_exo; k <- q to 1 by -1 do names += s"xe${j}l$k"
        MakeMatrix4TS.formNames (spec, p) ++ names.toArray
    end formNames

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the x matrix and y matrix into training and testing sets.
     *  @param x      the x data/input matrix
     *  @param y      the y response/output matrix
     *  @param ratio  the ratio of the TESTING set to the full dataset (most common 70-30, 80-20)
     *
    def split_TnT (x: MatrixD, y: MatrixD, ratio: Double = 0.30): (MatrixD, MatrixD, MatrixD, MatrixD) =
        val n       = x.dim
        val tr_size = (n * (1.0 - ratio)).toInt
        println (s"VARX.split_TnT: tr_size = $tr_size, te_size = ${n - tr_size}")
        (x(0 until tr_size), y(0 until tr_size), x(tr_size until n), y(tr_size until n))
    end split_TnT
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use rolling-validation to compute test Quality of Fit (QoF) measures
     *  by dividing the dataset into a TRAINING SET (tr) and a TESTING SET (te)
     *  as follows:  [ <-- tr_size --> | <-- te_size --> ]
     *  This version calls predict for one-step ahead out-of-sample forecasts.
     *  @see `RollingValidation`
     *  @param mod  the forecasting model being used (e.g., `VARX`)
     *  @param rc   the retraining cycle (number of forecasts until retraining occurs)
     *
    def rollValidate (mod: PredictorMV & Fit, rc: Int): Unit =
        val x       = mod.getX                                                 // get data/input matrix
        val y       = mod.getY                                                 // get response/output vector
        val te_size = RollingValidation.teSize (y.dim)                         // size of testing set
        val tr_size = y.dim - te_size                                          // size of initial training set
        debug ("rollValidate", s"train: tr_size = $tr_size; test: te_size = $te_size, rc = $rc")

        val yp = new MatrixD (te_size, y.dim2)                                 // y-predicted over testing set
        for i <- 0 until te_size do                                            // iterate through testing set
            val t = tr_size + i                                                // next time point to forecast
//          if i % rc == 0 then mod.train (x(0 until t), y(0 until t))         // retrain on sliding training set (growing set)
            if i % rc == 0 then mod.train (x(i until t), y(i until t))         // retrain on sliding training set (fixed size set)
            yp(i) = mod.predict (x(t-1))                                       // predict the next value
        end for

        val df = max (0, mod.parameter(0).dim - 1)                             // degrees of freedom for model
        mod.resetDF (df, te_size - df)                                         // reset degrees of freedom
        for k <- y.indices2 do
            val (t, yk) = RollingValidation.align (tr_size, y(?, k))           // align vectors
            val ypk = yp(?, k)
            banner (s"QoF for horizon ${k+1} with yk.dim = ${yk.dim}, ypk.dim = ${ypk.dim}")
            new Plot (t, yk, ypk, s"Plot yy, yp vs. t for horizon ${k+1}", lines = true)
            println (FitM.fitMap (mod.diagnose (yk, ypk), qoF_names))
        end for
    end rollValidate
     */

end VARX

// import Forecaster_RegV.plotAll

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vARXTest` main function tests the `VARX` class.
 *  This test is used to CHECK that the `buildMatrix` method (@see `apply`) is working correctly.
 *  May get NaN for some maximum lags (p) due to multi-collinearity.
 *  > runMain scalation.modeling.forecasting.multivar.vARXTest
 *
@main def vARXTest (): Unit =

    val m  = 30
    val z  = VectorD.range (1, m)
    val y  = MatrixD (z, -z + m).ᵀ
    val hh = 3                                                                 // the forecasting horizon

    hp("q") = 2
    for p <- 5 to 5 do                                                         // autoregressive hyper-parameter p
        hp("p") = p
        banner (s"Test: VARX with $p lags")
        val mod = VARX (y, hh)                                                  // create model for time series data
        mod.trainNtest_x ()()                                                  // train the model on full dataset
        println (mod.summary)

//      val yy = mod.getY
//      val yp = mod.predict (mod.getX)
//      plotAll (yy, yp, mod.modelName)
//      for k <- yp.indices2 do
//          new Plot (null, yy(?, k), yp(?, k), s"yy_$k vs. yp_$k for ${mod.modelName} (h=${k+1}) with $p lags", lines = true)
    end for

end vARXTest
 */


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vARXTest2` main function tests the `VARX` class on real data:
 *  Forecasting Gas Furnace Data.  Performs In-Sample Testing.
 *  > runMain scalation.modeling.forecasting.multivar.vARXTest2
 *
@main def vARXTest2 (): Unit =

    import Example_GasFurnace._

    val hh   = 4                                                               // forecasting horizon
    val LAGS = 5                                                               // number of lags
    hp("p") = LAGS
    hp("q") = 2

    val y = Example_GasFurnace.loadData_yy (header)
    println (s"y.dims = ${y.dims}")

    banner ("Test In-Sample VARX on GasFurnace Data")
    val mod = VARX (y, hh, header)                                              // create model for time series data
    val yp  = mod.trainNtest_x ()()._1                                         // train on full and test on full
    println (mod.summary)
    val yy_ = y(1 until y.dim)                                                 // can't forecast first values at t = 0
    plotAll (yy_, yp, mod.modelName)

end vARXTest2
 */


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vARXTest3` main function tests the `VARX` class on real data:
 *  Forecasting COVID-19 Weekly Data.  Performs In-Sample Testing.
 *  Goal:  Find the variable that works best with "new_deaths"
 *  > runMain scalation.modeling.forecasting.multivar.vARXTest3
 */
@main def vARXTest3 (): Unit =

    val hh   = 6                                                               // maximum forecasting horizon
    val LAGS = 2                                                               // number of lags
    hp("p") = LAGS
    hp("q") = 2

    val endo_vars = Array ("new_deaths", "icu_patients")
    val exo_vars  = Array ("hosp_patients")
    val fname     = endo_vars ++ exo_vars

    val yy   = Example_Covid.loadData_yy (fname)
//  val y    = yy                                                              // full
    val y    = yy(0 until 116, 0 until 2)                                      // clip the flat end, col. 0, 1 (endo)
    val xe   = yy(0 until 116, 2 until 3)                                      // clip the flat end, col. 2 (exo)
    println (s"y.dims = ${y.dims}")

    for j <- fname.indices do
        new Plot (null, y(?, j), null, s"y_$j (${fname(j)}) vs. t", lines = true)

    banner ("Test In-Sample VARX on COVID-19 Weekly Data")
    val mod = VARX (xe, y, hh)                                                 // create model for time series data
    mod.trainNtest_x ()()                                                      // train on full and test on full
//  println (mod.summary ())
//  val yy_ = y(1 until y.dim)                                                 // can't forecast first values at t = 0
//  plotAll (yy_, yp, mod.modelName)

/*
    banner (s"Feature Selection Technique: Stepwise")
    val (cols, rSq) = mod.stepRegressionAll (cross = false)                    // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for VARX with tech", lines = true)

    banner ("Feature Importance")
    println (s"Stepwise: rSq = $rSq")
*/
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

end vARXTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vARXTest4` main function tests the `VARX` class on real data:
 *  Forecasting COVID-19 Weekly Data.  Performs In-Sample Testing.
 *  Goal:  Find the four variables that works best with "new_deaths"
 *  > runMain scalation.modeling.forecasting.multivar.vARXTest4
 *
@main def vARXTest4 (): Unit =

    val LAGS = 5                                                               // number of lags
    val hh   = 6                                                               // forecasting horizon

    val vars = Array ("new_deaths", "icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val yy = Example_Covid.loadData_yy (vars)
    val iskip = yy(?, 0).indexWhere (_ >= 6.0)                                 // find day with at least 6 deaths
    println (s"iskip = $iskip is first day with at least 6 deaths")
    val y = yy(iskip until yy.dim)                                             // trim away the first iskip rows
    println (s"y.dims = ${y.dims}")

    banner ("Test In-Sample VARX on COVID-19 Weekly Data")
    hp("p") = LAGS
    hp("q") = 2
    val mod = VARX (y, hh)                                                      // create model for time series data - with exo
    mod.trainNtest_x ()()                                                      // train on full and test on full
//  println (mod.summary ())
//  val yy_ = y(1 until y.dim)                                                 // can't forecast first values at t = 0
//  plotAll (yy_, yp, mod.modelName)

//  val tech = SelectionTech.Forward                                           // pick one feature selection technique
//  val tech = SelectionTech.Backward
//  val tech = SelectionTech.Stepwise

/*
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = false)                 // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for VARX with tech", lines = true)

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
*/
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

end vARXTest4
 */


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vARXTest5` main function tests the `VARX` class on real data:
 *  Forecasting COVID-19 Weekly Data. Does TnT Testing on endogenous and exogenous variables.
 *  Determine the terms to include in the model using Stepwise on In-Sample.
 *  > runMain scalation.modeling.forecasting.multivar.vARXTest5
 *
@main def vARXTest5 (): Unit =

    val LAGS = 5                                                               // number of lags
    val hh   = 6                                                               // forecasting horizon

    val vars = Array ("new_deaths", "icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val yy = Example_Covid.loadData_yy (vars)
    val iskip = yy(?, 0).indexWhere (_ >= 6.0)                                 // find day with at least 6 deaths
    println (s"iskip = $iskip is first day with at least 6 deaths")
    val y = yy(iskip until yy.dim)                                             // trim away the first iskip rows
    println (s"y.dims = ${y.dims}")

    hp("p") = LAGS
    hp("q") = 2
    banner ("Test In-Sample VARX on COVID-19 Weekly Data")
    val mod = VARX (y, hh)                                                      // create model for time series data - with exo
    mod.trainNtest_x ()()                                                      // train on full and test on full
//  println (mod.summary ())
//  val yy_ = y(1 until y.dim)                                                 // can't forecast first values at t = 0
//  plotAll (yy_, yp, mod.modelName)

//  val tech = SelectionTech.Forward                                           // pick one feature selection technique
//  val tech = SelectionTech.Backward
//  val tech = SelectionTech.Stepwise

/*
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = false)                 // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for VARX with tech", lines = true)

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
*/
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

/*
    banner ("Run TnT on Best model")
    val bmod = mod.getBest._3                                                  // get the best model from feature selection
    val (x_, y_, xtest, ytest) = VARX.split_TnT (bmod.getX, bmod.getY)
    val yptest = bmod.trainNtest_x (x_, y_)(xtest, ytest)._1                   // train on (x_, y_) and test on (xtest, ytest)
    new Plot (null, ytest(?, 0), yptest(?, 0), s"${mod.modelName}, ytest vs. yptest", lines = true)
*/

end vARXTest5
 */


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vARXTest6` main function tests the `VARX` class on real data:
 *  Forecasting COVID-19 Weekly Data.  Does Rolling Validation on variables.
 *  Determine the terms to include in the model using Stepwise on In-Sample.
 *  > runMain scalation.modeling.forecasting.multivar.vARXTest6
 *
@main def vARXTest6 (): Unit =

    val LAGS = 5                                                               // number of lags
    val h    = 6                                                               // forecasting horizon

    val vars = Array ("new_deaths", "icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val yy = Example_Covid.loadData_yy (vars)
    val iskip = yy(?, 0).indexWhere (_ >= 6.0)                                 // find day with at least 6 deaths
    println (s"iskip = $iskip is first day with at least 6 deaths")
    val y = yy(iskip until yy.dim)                                             // trim away the first iskip rows
    println (s"y.dims = ${y.dims}")

    hp("p") = LAGS
    hp("q") = 2
    banner ("Test In-Sample VARX on COVID-19 Weekly Data")
    val mod = VARX (y, h)                                                       // create model for time series data - with exo
    mod.trainNtest_x ()()                                                      // train on full and test on full
//  println (mod.summary ())
//  val yy_ = y(1 until y.dim)                                                 // can't forecast first values at t = 0
//  plotAll (yy_, yp, mod.modelName)

//  val tech = SelectionTech.Forward                                           // pick one feature selection technique
//  val tech = SelectionTech.Backward
//  val tech = SelectionTech.Stepwise

/*
    banner (s"Feature Selection Technique: $tech")
    val (cols, rSq) = mod.selectFeatures (tech, cross = false)                 // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for VARX with tech", lines = true)

    banner ("Feature Importance")
    println (s"$tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")

    banner ("Run Rolling Validation on VARX Best model")
    val bmod = mod.getBest._3                                                  // get the best model from feature selection
    VARX.rollValidate (bmod, 1)
*/

end vARXTest6
 */

