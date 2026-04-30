
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive, Moving Average (ARMA)
 *
 *  Parameter Estimation: Least Squares, Maximum Likelihood
 *  Conditional Sum-of-Squares (CSS), Negative Log-Likelihood (NLL)
 *  @see     arxiv.org/pdf/1611.00965
 *  @see     arxiv.org/html/2310.01198v2
 *  @see     arxiv.org/pdf/2310.01198
 *  @see     people.stat.sc.edu/hitchcock/stat520ch7slides.pdf
 *  @see     www.emu.edu.tr/mbalcilar/teaching2007/econ604/lecture_notes.htm
 *  @see     www.stat.berkeley.edu/~bartlett/courses/153-fall2010
 *  @see     www.princeton.edu/~apapanic/ORFE_405,_Financial_Time_Series_%28Fall_2011%29_files/slides12-13.pdf
 */

package scalation
package modeling
package forecasting

import scala.annotation.unused

import scalation.mathstat._
//import scalation.optimization.quasi_newton.{BFGS => Optimizer}       // change import to change optimizer
//import scalation.optimization.quasi_newton.{LBFGS => Optimizer}
import scalation.optimization.quasi_newton.{LBFGS_B => Optimizer}
import scalation.random.NormalVec_c

import Forecaster.rdot
import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA` class provides basic time series analysis capabilities for Auto-Regressive,
 *  Moving Average (ARMA) models.  ARMA models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last
 *  p values and q shocks.
 *
 *      y_t = δ + Σ[φ_j y_t-j] + Σ[θ_j e_t-j] + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param y        the response vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to AR.hp)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class ARMA (y: VectorD, hh: Int, tRng: Range = null,
            hparam: HyperParameter = AR.hp,
            bakcast: Boolean = false)
      extends AR (y, hh, tRng, hparam, bakcast):

    private   val debug = debugf ("ARMA", true)                         // debug function
    private   val flaw  = flawf ("ARMA")                                // flaw function
    private   val STEP  = 0.02                                          // step size for optimizer
    protected val q     = hparam("q").toInt                             // use the last q shock/errors
//  private   var z     = VectorD.nullv                                 // var for centered time series (used by first train)
    private   val pnq   = p + q                                         // sum of the orders
    private   val notHR = true                                          // don't use the HR algorithm
    private   val useAR = false                                         // use AR for initial guess for φ parameters

    _modelName = s"ARMA_${p}_$q"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize the model parameters b = φ ++ θ by use the inherited AR for φ and
     *  small random numbers for θ.
     *  @param y_  the training/full response vector (e.g., full y)
     */
    def init_params (y_ : VectorD): VectorD =
        var bb =
        if useAR then
            super.train (null, y_)                                        // option: fit AR to initialize ARMA
            super.parameter(1 until p+1)                                  // use AR parameters to initialize φ for ARMA
        else
            NormalVec_c (p, 0.1, 0.01).gen                                // randomly initialize φ with small values
        if q > 0 then bb = bb ++ NormalVec_c (q, 0.0, 0.01).gen           // randomly initialize θ with small values
        bb
    end init_params

// Use one of the following two train methods: swap names train0 & train and add override

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `ARMA` model to the times-series data in vector y_.
     *  Estimate the coefficient vector b for a (p, q)-th order Auto-Regressive ARMA(p, q) model.
     *  Uses a nonlinear optimizer (e.g., LBFGS_B) to determine the coefficients.
     *  Residuals are re-estimated during optimization (may lead to instability)
     *  NOTE: Requires the error update in `predict` to be uncommented.
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the training/full response vector (e.g., full y)
     */
    def train0 (@unused x_null: MatrixD, y_ : VectorD): Unit =
        banner (s"T R A I N 0  --  for p = $p, q = $q")
        val mu = y_.mean                                                // sample mean of y_
        b = init_params (y_)                                            // initialize parameter vector b = φ ++ θ
//      e.clear ()                                                      // set errors to zero (and uncomment) or try
//      e.set (super.residual)                                          // set errors to AR residuals
        δ = mu * (1 - b(0 until p).sum)                                 // determine intercept before optimization
//      z = y_ - mu                                                     // optimization works better using zero-centered data

        def css (b_ : VectorD): Double =
            b = b_.copy                                                 // copy parameters from b vector
            δ = mu * (1 - b(0 until p).sum)                             // determine updated intercept
            val yp = predictAll (y_)                                    // predicted value for z
            val yy = y_(1 until y_.dim)                                 // skip first (backcasted) value
            val loss = ssef (yy, yp)                                    // compute loss function
//          println (s"css loss = $loss, δ = $δ, b = $b")
            loss
        end css

        debug ("train0", s"before optimization: p = $p, q = $q, δ = $δ, b = $b")
        val optimizer = Optimizer (css, b.dim)                          // apply Quasi-Newton optimizer
        val bb = optimizer.solve (b, STEP)._2                           // optimal solution for loss function and parameters
        b = bb                                                          // assign optimized parameters to vector b
        δ = mu * (1 - b(0 until p).sum)                                 // determine intercept after optimization
        debug ("train0", s"after optimization: p = $p, q = $q, δ = $δ, b = $b")
//      println (s"train0: error e = $e")

    end train0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `ARMA` model to the times-series data in vector y_.
     *  Estimate the coefficient vector b for a (p, q)-th order Auto-Regressive ARMA(p, q) model.
     *  Uses a nonlinear optimizer (e.g., LBFGS_B) to determine the coefficients.
     *  Residuals are estimated before optimization using the Hannan-Rissanen Algorithm.
     *  NOTE: Requires the error update in `predict` to be commented out.
     *  @see faculty.washington.edu/dbp/s519/PDFs/13-overheads-2020.pdf
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the training/full response vector (e.g., full y)
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =
        if notHR then
            train0 (x_null, y_)
        else
            e.clear ()
            δ = 0.0                                                     // intercept for y_
            resid (y_)                                                  // set the residuals using high order AR
            val optimizer = Optimizer (ss, b.dim)                       // apply Quasi-Newton optimizer
            val bb = optimizer.solve (b, STEP)._2                       // optimal solution for loss function and parameters
            b = bb                                                      // recover parameters for z
            δ = y.mean * (1 - b(0 until p).sum)                         // determine intercept after optimization
            debug ("train", s"optimized: p = $p, q - $q, δ = $δ, b = $b")
            println (s"train: error e = $e")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use a higher order AR model to estimate the residuals (unobserved data).
     *  Set the residual/error vector e defined in `Forecaster`.
     *  @param y_      the training/full response vector (e.g., full y)
     */
    def resid (y_ : VectorD): Unit =
        val hp2 = new HyperParameter
        hp2 += ("p", pnq + 3, pnq + 3)                                  // Set the AR order to p + 1 + 3
        val ar = new AR (y, hh, tRng, hp2)                              // create an AR model
        ar.train (null, y_)                                             // train the AR model
        e += ar.residual                                                // use residuals from the AR model
    end resid

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sum of squared errors (loss function).
     *  @param b_ the combined parameters (δ, b) where b = (φ, θ).
     */
    def ss (b_ : VectorD): Double =
        b = b_.copy                                                     // copy parameters from b vector
        val yy  = yb(1 until yb.dim)                                    // skip first (backcasted) value
        δ = yy.mean * (1 - b(0 until p).sum)                            // determine updated intercept
        val yyp = predictAll (yb)                                       // predicted value for yb
//      debug ("ss", s"yy.dim = ${yy.dim}, yyp.dim = ${yyp.dim}")
        ssef (yy, yyp)                                                  // compute loss function
    end ss

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = δ +  φ_0 y_t-1 + φ_1 y_t-2 + ... + φ_p-1 y_t-p
     *                 θ_0 e_t-1 + θ_1 e_t-2 + ... + θ_q-1 e_t-q
     *
     *  where φ = b(0 until p) and θ = b(p until p_q).
     *  When k < 0 let y_k = y_0 (i.e., assume first value repeats back in time),
     *  but do not assume errors repeat.  Note, column 1 of yf (yf(?, 1) holds yp.
     *  Must be executed in time order, so errors are properly recorded in vector e
     *  @see `predictAll` method in `Forecaster` trait.
     *  @see `rdot` in Forecaster.scala for reverse dot product implementation.
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    override def predict (t: Int, y_ : VectorD): Double =
        if t == 0 then e(0) = 0                                         // from backcast: assume no error
        if t == 1 then e(1) = y_(1) - yf(0, 1)                          // first real point

        var sum = δ + rdot (b(0 until p), y_, t-1)                        // intercept + AR terms (use y); b(0 until p) = φ
        for j <- 0 until q do                                           // add MA terms (shocks)
            if t-1-j >= 0 then sum += b(p+j) * e(t-1-j)                     // e(t-j = -1) does not exists; b(p+j) = θ(j)

        if t < y_.dim-1 then e(t) = y_(t) - sum                     // update the error vector (uncomment for first train)
        sum                                                             // prediction yp
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with rolling validation (analog of predict method).
     *  Note, must include [ y_i, e_i ] before horizon and [ yp_i ] after horizon
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     */
    override def forecast (t: Int, y_ : VectorD = yb): VectorD =
        val yh = new VectorD (hh)                                       // hold forecasts for each horizon
        for h <- 1 to hh do
            var sum = δ + rdot (b(0 until p), yf, t, h-1)               // intercept + AR terms (use y and yp); b(0 until p) = φ
            for j <- h-1 until q do                                     // add MA terms (shocks) from before horizon
                if t-j >= 0 then sum += b(p+j) * e(t-j)                 // e(t-j = -1) does not exists; b(p+j) = θ(j)
            yf(t, h) = sum                                              // record in forecast matrix
            yh(h-1)  = sum                                              // record forecasts for each horizon
        yh                                                              // return forecasts for all horizons
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the h-steps ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1.
     *  @see `forecastAll` method in `Forecaster` trait.
     *  Note, must include [ y_i, e_i ] before horizon and [ yp_i ] after horizon
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the actual values to use in making forecasts
     */
    override def forecastAt (h: Int, y_ : VectorD = yb): VectorD =
        if h < 2 then flaw ("forecastAt", s"horizon h = $h must be at least 2")

        for t <- y_.indices do                                          // make forecasts over all time points for horizon h
            var sum = δ + rdot (b(0 until p), yf, t, h-1)               // intercept + AR terms (use y and yp); b(0 until p) = φ
            for j <- h-1 until q do                                     // add MA terms (shocks) from before horizon
                if t-j >= 0 then sum += b(p+j) * e(t-j)                 // e(t-j = -1) does not exists; b(p+j) = θ(j)
            yf(t, h) = sum                                              // record in forecast matrix
        yf(?, h)                                                        // return the h-step ahead forecast vector
    end forecastAt

end ARMA


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA` companion object provides factory methods for the `ARMA` class.
 */
object ARMA:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `ARMA` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = AR.hp): ARMA =
        new ARMA (y, hh, tRng, hparam)
    end apply

end ARMA


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMATest` main function tests the `ARMA` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRMATest
 */
@main def aRMATest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = new ARMA (y, hh)                                          // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                 // train and test on full dataset

    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end aRMATest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMATest2` main function tests the `ARMA` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRMATest2
 */
@main def aRMATest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = new ARMA (y, hh)                                          // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                 // train and test on full dataset

    mod.setSkip (0)
    mod.rollValidate ()                                                 // TnT with Rolling Validation
    mod.diagnoseAll (y, mod.getYf, Forecaster.teRng (y.dim))            // only diagnose on the testing set
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRMATest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMATest3` main function tests the `ARMA` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for AR(p), ARY(p), ARY_D(p), ARMA(p, 0), and ARMA(p, 1).
 *  Note ARX (p, 1, 0), where 0 => no exo vars, duplicates results of ARY(p)
 *
 *  19.0371,    29.5797,    39.0740,    47.4638,    55.1785,    62.1818  RW
 *
 *  18.7298,    28.4908,    37.0997,    45.6487,    51.7248,    56.3708  AR(1)
 *  18.5808,    28.3362,    37.2485,    45.7846,    52.0362,    56.9114  ARY(1)
 *  18.5808,    28.8144,    37.7469,    44.8006,    49.8166,    52.3205  ARY_D(1)
 *  18.5788,    28.3364,    37.2530,    45.7883,    52.0403,    56.9181  ARY_Quad(1)
 *  18.7095,    28.4690,    37.1203,    45.6688,    51.7687,    56.4467  ARMA(1, 0)
 *  17.0508,    26.4669,    35.4906,    43.5707,    49.4949,    54.2347  ARMA(1, 1)
 *
 *  16.3579,    24.7155,    33.0480,    40.0707,    46.0049,    50.8265  AR(2)
 *  16.2270,    23.3708,    31.6615,    38.7385,    44.7630,    50.0814  ARY(2)
 *  16.2270,    22.9698,    30.0933,    35.4960,    40.7977,    46.2700  ARY_D(2)
 *  16.2663,    22.6643,    31.0768,    37.7388,    44.2476,    50.0283  ARY_Quad(2)
 *  19.0826,    29.2723,    37.2914,    44.2636,    49.8307,    53.6992  ARMA(2, 0)
 *  17.0445,    26.6538,    35.5239,    42.9937,    48.7679,    53.3489  ARMA(2, 1)
 *
 *  16.0114,    22.7408,    29.5631,    35.2773,    40.9870,    45.8408  AR(3)
 *  15.7509,    21.9972,    28.8976,    34.6815,    40.7375,    46.1590  ARY(3)
 *  15.7509,    21.8745,    28.2745,    32.9840,    39.1694,    43.9673  ARY_D(3)
 *  15.7262,    21.2578,    28.4101,    34.1532,    40.6659    	46.1492  ARY_Quad(3)
 *  16.7027,    23.4111,    30.5995,    36.7396,    42.6680,    47.1189  ARMA(3, 0)
 *  16.1750,    23.1243,    30.8535,    37.1636,    43.0417,    48.2946  ARMA(3, 1)
 *
 *  15.8988,    22.5738,    28.5298,    33.3360,    39.1586,    43.1606  AR(4)
 *  15.6423,    21.7982,    27.9006,    33.1000,    39.0543,    43.9748  ARY(5)
 *  15.6423,    21.8663,    28.0034,    32.9898,    38.9927,    43.6218  ARY_D(4)
 *  15.5814,    21.2352,    28.5489,    34.4369,    40.3618,    45.2605  ARY_Quad(4)
 *  16.6457,    22.9684,    29.0629,    34.6601,    40.1521,    44.0896  ARMA(4, 0)
 *  15.3290,    21.9965,    27.8397,    34.3507,    40.0857,    45.8402  ARMA(4, 1)
 *
 *  15.9279,    22.5769,    28.5035,    33.3019,    39.1381,    43.0520  AR(5)
 *  15.6349,    21.8003,    27.9084,    33.1127,    39.0628,    44.0175  ARY(5)
 *  15.6349,    21.7885,    28.0114,    33.0117,    39.1418,    43.7715  ARY_D(5)
 *  15.3209,    21.3541,    28.9325,    35.1359,    41.0300,    45.8558  ARY_Quad(5)
 *  16.3720,    22.8047,    28.7702,    33.9232,    39.5677,    43.2628  ARMA(5, 0)
 *  15.3361,    21.9121,    27.6568,    34.0218,    39.6254,    45.2994  ARMA(5, 1)
 *
 *  > runMain scalation.modeling.forecasting.aRMATest3
 */
@main def aRMATest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon

    for p <- 1 to 5; q <- 0 to 1 do
        AR.hp("p") = p                                                  // number of AR terms
        AR.hp("q") = q                                                  // number of MA terms
        val mod = new ARMA (y, hh)                                      // create model for time series data
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()                                             // train and test on full dataset

        mod.forecastAll ()                                              // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (y, mod.getYf)                                  // use showYf = false to not print forecast matrix Yf
    end for

end aRMATest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMATest4` main function tests the `ARMA` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for AR(p), ARY(p), ARY_D(p), ARMA(p, 0), and ARMA(p, 1).
 *
 *  19.1334,    31.1906,    44.3787,    55.1576,    65.1810,    74.0524  AR(1)
 *  19.0397,    30.4570,    43.9113,    54.9642,    65.3163,    74.2124  ARY(1)
 *  19.1718,    30.7038,    44.5265,    55.7794,    66.3876,    75.6566  ARMA(1, 0)
 *  18.3012,    29.3224,    43.0369,    54.5719,    64.9230,    74.2520  ARMA(1, 1)
 *
 *  16.6447,    26.9109,    39.8106,    50.8595,    60.2176,    68.6317  AR(2)
 *  16.8833,    26.4824,    39.2329,    50.8677,    61.0624,    70.3218  ARY(2)
 *  19.4256,    32.8815,    46.4279,    57.2199,    66.8651,    75.3077  ARMA(2, 0)
 *  18.3009,    30.0443,    43.6634,    54.9669,    64.8541,    73.7911  ARMA(2, 1)
 *
 *  15.9232,    23.5929,    34.3577,    44.1784,    53.6513,    62.0129  AR(3)
 *  15.7190,    21.7959,    32.1395,    42.0074,    52.6874,    62.7276  ARY(3)
 *  16.4547,    24.4668,    36.8597,    46.7958,    58.3539,    67.6623  ARMA(3, 0)
 *  17.0353,    24.0309,    36.6585,    46.1961,    57.6348,    67.2332  ARMA(3, 1)
 *
 *  15.3256,    22.6893,    30.7558,    39.6274,    48.6646,    56.7375  AR(4)
 *  14.6791,    19.9940,    26.5644,    35.4590,    41.4955,    50.8660  ARY(4)
 *  14.9687,    22.2599,    29.6359,    39.6018,    48.2853,    56.9797  ARMA(4, 0)
 *  15.2243,    21.4976,    27.7929,    37.9923,    45.0999,    54.3417  ARMA(4, 1)
 *
 *  15.9166,    21.5246,    28.0675,    36.8669,    43.3785,    51.1786  AR(5)
 *  15.0232,    19.4222,    27.1981,    35.4744,    40.3466,    48.4066  ARY(5)
 *  15.5426,    21.0405,    29.1731,    37.8006,    43.3590,    52.6387  ARMA(5, 0)
 *  15.7641,    21.0723,    28.7463,    37.7968,    42.8480,    52.8277  ARMA(5, 1)
 *
 *  > runMain scalation.modeling.forecasting.aRMATest4
 */
@main def aRMATest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon

    for p <- 1 to 5; q <- 0 to 1 do
        AR.hp("p") = p                                                  // number of AR terms
        AR.hp("q") = q                                                  // number of MA terms
        val mod = new ARMA (y, hh)                                      // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()

        mod.setSkip (0)                                                 // using data from training can forecast first in test
        mod.rollValidate ()                                             // TnT with Rolling Validation
        mod.diagnoseAll (y, mod.getYf, Forecaster.teRng (y.dim))        // only diagnose on the testing set
//      println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRMATest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMATest5` main function tests the `ARMA` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for ARMA(p, 1) (i.e., q = 1) for different p orders.
 *  > runMain scalation.modeling.forecasting.aRMATest5
 */
@main def aRMATest5 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon

    AR.hp("q") = 1                                                      // number of MA terms
    for p <- 1 to 5 do
        AR.hp("p") = p                                                  // number of AR terms
        val mod = new ARMA (y, hh)                                      // create model for time series data
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()                                             // train and test on full dataset

        mod.forecastAll ()                                              // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (y, mod.getYf)
        println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    end for

end aRMATest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMATest6` main function tests the `ARMA` class on small dataset.
 *  Test forecasts (h = 1 step ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRMATest6
 */
@main def aRMATest6 (): Unit =

    val y  = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3)

    AR.hp ("q") = 0
    var mod = new ARMA (y, 1)                                             // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "AR1")

    AR.hp ("p") = 2
    mod = new ARMA (y, 1)                                                 // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "AR2")

end aRMATest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMATest7` main function tests the `ARMA` class on small dataset.
 *  Test the generation of ARMA sequences for various p and q values.
 *  > runMain scalation.modeling.forecasting.aRMATest7
 */
@main def aRMATest7 (): Unit =

    val nrg = random.Normal (0.0, 1.0)

    val m = 100
    val y = new VectorD (m)
    val e = new VectorD (m)
    val φ = VectorD (0.8, 0.7)
    val θ = VectorD (0.8, 0.7)

    for p <- 0 to 2; q <- 0 to 2 if p + q > 0 do
        val (rp, rq) = ((0 until p), (0 until q))
        for t <- y.indices do
            e(t) = nrg.gen
            y(t) = rdot (φ(rp), y, t) + rdot (θ(rq), e, t) + e(t)
        end for
        new Plot (null, y, null, s"Plot of y vs. t for p = $p, q = $q", lines = true)
        object CG extends Correlogram (y)
        CG.makeCorrelogram ()
        CG.plotCorrelogram ()
    end for

end aRMATest7

