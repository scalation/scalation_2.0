
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jul 25 15:37:46 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive, Moving Average (ARMA_2SL)
 *
 *  Parameter Estimation: 
 *  Unlike `ARMA_2SL` which can be unsable or `ARMA_KF` which is computationally heavy,
 *  `ARMA_2SL` uses a Two-Stage Linear (2SL) approximation (Durbin / Hannan-Rissanen)
 *  to guarantee global convexity and fast matrix solutions.
 *
 *  Stage One:  Use high order AR(p+q+2) --> yp --> residuals (e_t-j)
 *  Stage Two:  Add a column for each e_t-j to design matrix for ARX or ARX_D 
 */

// FIX - under development

package scalation
package modeling
package forecasting

import scala.annotation.unused

import scalation.mathstat._

import Example_LakeLevels.y

import ARMA_2SL._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA_2SL` class provides basic time series analysis capabilities for Auto-Regressive,
 *  Moving Average (ARMA_2SL) models.  ARMA_2SL models are often used for forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last
 *  p values and q shocks.
 *
 *      y_t = δ + Σ[φ_j y_t-j] + Σ[θ_j e_t-j] + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  The coefficients/parameters are δ, φ, and θ, drift, AR-coeffs, and MA-coeffs.
 *  @see `ARMA_CSS`, `ARMA_KF` for methods that use nonlinear optimization
 *  @param y        the response vector (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to AR.hp)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class ARMA_2SL (y: VectorD, hh: Int, tRng: Range = null,
                hparam: HyperParameter = AR.hp,
                bakcast: Boolean = false)
      extends Forecaster (y, hh, tRng, hparam, bakcast):              // provides `reg` field

//  private val debug = debugf ("ARMA_2SL", false)                    // debug function
//  private val flaw  = flawf ("ARMA_2SL")                            // flaw function
    private val p     = hparam("p").toInt                             // use the last p values (AR-terms)
    private val q     = hparam("q").toInt                             // use the last q shock/errors (MA-terms)
    private var model2: Forecaster_D = null

    _modelName = s"ARMA_2SL_${p}_$q"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `ARMA_2SL` model to the times-series data in vector y_.
     *  Estimate the coefficient vector b for a (p, q)-th order Auto-Regressive ARMA_2SL(p, q) model.
     *  Uses two-stage linear optimizer to determine the coefficients.
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the training/full response vector (e.g., full y)
     */
    override def train (@unused x_null: MatrixD, y_ : VectorD): Unit =
        banner (s"T R A I N  --  for p = $p, q = $q")

        // Stage one
        val e_sh_ = arHR_errors (y_, p, q)                          // approximate the errors

        // Stage two
        given dataContext: Map [String, VectorD] = Map (
            "y"    -> y_,
            "e_sh" -> e_sh_
        )

        val y    = Vars.y
        val e_sh = Vars.e_sh

        // =========================================================================
        // STAGE 2: ARX_D with shock included (MA approximation)
        // =========================================================================
        val formula2 = if q == 0 then y ~= y~(p,1)
                       else y ~= y~(p,1) + e_sh                      // FIX - extend for q > 1
        model2 = formula2.fit_D (hh)

        model2.forecastAll ()
        model2.diagnoseAll (model2.getY, model2.getYf)

//      println (s"train: error e = $e")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *  @note: order of variables is set to be compatible with `ARY` and `AR`.
     *
     *      ŷ = δ + φ ∙ [ y_t-p, ... y_t-2, y_t-1 ]
     *            + θ ∙ [ e_t-q, ... e_t-2, e_t-1 ]
     *
     *  where φ = b(0 until p) and θ = b(p until p+q).
     *  When k < 0 let y_k = y_0 (i.e., assume first value repeats back in time),
     *  but do not assume errors repeat.  Note, column 1 of yf (yf(?, 1) holds yp.
     *  Must be executed in time order, so errors are properly recorded in vector e
     *  @see `predictAll` method in `Forecaster` trait.
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions
     */
    override def predict (t: Int, y_ : VectorD): Double = model2.predict (t, y_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of size hh, h = 1 to hh-steps ahead forecasts for the model,
     *  i.e., forecast the following time points:  t+1, ..., t+h.
     *  Intended to work with ROLLING VALIDATION (analog of predict method).
     *  Note, must include [ y_i, e_i ] before horizon and [ yp_i ] after horizon
     *  @param t   the time point from which to make forecasts
     *  @param y_  the actual values to use in making predictions
     */
    override def forecast (t: Int, y_ : VectorD = yb): VectorD = model2.forecast (t, y_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast values for all y_.dim time points at horizon h (h-steps ahead).
     *  Assign into FORECAST MATRIX and return the h-steps ahead forecast.
     *  Note, `predictAll` provides predictions for h = 1.
     *  @see `forecastAll` method in `Forecaster` trait.
     *  Note, must include [ y_i, e_i ] before horizon and [ yp_i ] after horizon
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     *  @param y_  the actual values to use in making forecasts
     */
    override def forecastAt (h: Int, y_ : VectorD = yb): VectorD = model2.forecastAt (h, y_)

    def getBest: BestStep = ???   // FIX -- implement or throw exception

end ARMA_2SL


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA_2SL` companion object provides factory methods for the `ARMA_2SL` class.
 */
object ARMA_2SL:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `ARMA_2SL` object.
     *  @param y       the response vector (time series data)
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng    the time range, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = AR.hp): ARMA_2SL =
        new ARMA_2SL (y, hh, tRng, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Approximate the errors/shocks using Hannan-Rissanen (HR) method:
     *  HR uses a long AR(p+q+2) model to produce proxies for the errors/shocks.
     *  @param y_  the training/full response vector (e.g., full y)
     *  @param p   the number of past values to include in the model (AR terms)
     *  @param q   the number of past errors/shocks to include in the model (MA terms)
     */
    def arHR_errors (y_ : VectorD, p: Int, q: Int): VectorD =
        AR.hp("p") = p + q + 2
        val arModel = new AR (y_, 1)                                    // use AR model for phi0 and errors er
//      arModel.trainNtest ()()
        arModel.train (null, y_)
        val er = y_ - arModel.predictAll ()                             // residual vector of same length
        er(0)  = er(1)                                                  // can't predict first value, nor the error
        er.shift (1)                                                    // shift to use e_{t-1}
    end arHR_errors

end ARMA_2SL


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_2SLTest` main function tests the `ARMA_2SL` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRMA_2SLTest
 */
@main def aRMA_2SLTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    for p <- 1 to 5; q <- 0 to 2 do
        AR.hp.set (("p", p), ("q", q))                                  // # AR terms, # MA terms
        val mod = new ARMA_2SL (y, hh)                                  // create model for time series data
        banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
//      println (mod.summary ())
    end for

end aRMA_2SLTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_2SLTest2` main function tests the `ARMA_2SL` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRMA_2SLTest2
 */
@main def aRMA_2SLTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

//  for p <- 1 to 5; q <- 0 to 2 do
    for p <- 1 to 5; q <- 0 to 1 do
        AR.hp.set (("p", p), ("q", q))                                  // # AR terms, # MA terms
        val mod = new ARMA_2SL (y, hh)                                  // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRMA_2SLTest2

import Example_Covid.{clip, loadData_y}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_2SLTest3` main function tests the `ARMA_2SL` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for AR(p), ARY(p), ARY_D(p), ARY_Quad(p), and ARMA_2SL(p, q).
 *  Note ARX (p, 1, 0), where 0 => no exo vars, should duplicates results of ARY(p)
 *
 *  19.0371,    29.5797,    39.0740,    47.4638,    55.1785,    62.1818  RW
 *
 *  18.7298,    28.4908,    37.4800,    46.3173,    53.3245,    59.5733  AR(1)
 *  17.9570,    27.8389,    38.2202,    46.8801,    53.9360,    60.0898  ARY(1)
 *  17.9618,    28.2405,    37.9195,    46.3640,    52.7653,    57.5828  ARY_D(1)
 *  18.1510,    28.9302,    39.1532,    47.4220,    53.9358,    59.8057  ARY_Quad(1) @1.8
 *  18.7095,    28.4690,    37.4985,    46.3367,    53.3577,    59.6225  ARMA_2SL(1, 0)
 *  17.0508,    26.4669,   >48.3576,   >55.5031,   >60.6453,   >65.8731  ARMA_2SL(1, 1)
 *  16.3906,    24.5959,   >44.0388,   >69.0787,   >72.1528,   >75.1025  ARMA_2SL(1, 2)
 *
 *  18.7095,    28.4690,    37.4985,    46.3367,    53.3577,    59.6225  NEW ARMA_2SL results
 *  17.0508,    26.4669,   >48.3576,   >55.5031,   >60.6453,   >65.8731  <-- 4 remaining problems
 *  16.3906,    24.5959,   >44.0388,   >69.0787,   >72.1528,   >75.1025  new warning: don't let q > p
 *
 *  16.3579,    24.7155,    33.0480,    40.1643,    46.8762,    53.2178  AR(2)
 *  15.6434,    23.1377,    31.4817,    39.7497,    47.1402,    54.4322  ARY(2)
 *  15.6462,    22.9254,    30.2138,    37.9797,    45.5425,    53.3514  ARY_D(2)
 *  15.7432,    23.3563,    32.3720,    40.6980,    47.5430,    54.6544  ARY_Quad(2)
 *  16.2780,    23.9049,    32.0846,    39.1081,    45.9899,    52.6934  ARMA_2SL(2, 0)
 *  15.9955,    22.1827,    28.4463,   >59.7675,   >59.9407,   >61.8693  ARMA_2SL(2, 1)
 *  15.9825,    22.1786,    28.4184,    34.3234,   >59.4347,   >61.4645  ARMA_2SL(2, 2)
 *
 *  16.2780,    23.9049,    32.0846,    39.1081,    45.9899,    52.6934
 *  15.9802,    22.3028,    28.5484,   >55.6803,    56.0996,    59.1596  <-- 1 remaining problem
 *  15.8822,    22.3551,    28.6950,    35.3678,    50.7729,    54.9230
 *
 *  16.0114,    22.7408,    29.5631,    35.2773,    41.5856,    47.5716  AR(3)
 *  15.2466,    21.9209,    28.9117,    36.8809,    44.3765,    51.6056  ARY(3)
 *  15.2577,    21.7428,    29.0617,    36.9863,    45.2127,    53.1253  ARY_D(3)
 *  15.2283,    21.7942,    29.3575,    37.5463,    43.4328,    50.1804  ARY_Quad(3)
 *  15.8996,    22.5037,    29.2999,    34.8894,    41.4264,    47.6811  ARMA_2SL(3, 0)
 *  15.8583,    22.3514,    28.7946,    34.1015,    40.1397,    45.7540  ARMA_2SL(3, 1)
 *  16.0223,    22.5370,    28.6346,    34.1455,   >56.1685,    49.6751  ARMA_2SL(3, 2)
 *
 *  15.9001,    22.5047,    29.3006,    34.8864,    41.4225,    47.6805
 *  15.8984,    22.4595,    28.9675,    34.3056,    39.9657,    45.7296
 *  16.0228,    22.4953,    28.5992,    34.1132,    52.3294,    49.1479
 *
 *  15.8988,    22.5738,    28.5298,    33.3360,    39.1586,    44.3459  AR(4)
 *  15.2111,    21.6626,    28.2550,    36.4412,    43.5074,    50.8344  ARY(4)
 *  15.2240,    21.7621,    29.0603,    37.3265,    45.6671,    53.4636  ARY_D(4)
 *  15.1834,    21.7630,    29.9297,    37.2337,    43.2314,    48.1949  ARY_Quad(4)
 *  15.7947,    22.3001,    28.3314,    33.2822,    39.1217,    44.7431  ARMA_2SL(4, 0)
 *  15.8113,    22.2825,    28.2934,    33.2403,    39.0975,    44.0504  ARMA_2SL(4, 1)
 *  15.9027,    22.3363,    28.3234,    33.6346,    39.2045,    48.6453  ARMA_2SL(4, 2)
 *
 *  15.7955,    22.2919,    28.3209,    33.2685,    39.1204,    44.7477
 *  15.7547,    22.3055,    28.3846,    33.3654,    39.2047,    47.3164
 *  15.8670,    22.3925,    28.3206,    33.4633,    39.1771,    45.1759
 *
 *  15.9279,    22.5769,    28.5035,    33.3019,    39.1381,    43.0520  AR(5)
 *  14.9614,    21.5226,    28.7885,    36.8389,    44.4641,    51.8397  ARY(5)
 *  15.0044,    21.6113,    29.3029,    37.7081,    45.7620,    53.3911  ARY_D(5)
 *  14.6721,    22.0108,    31.0667,    38.7504,    44.8782,    49.7450  ARY_Quad(5)
 *  15.8239,    22.3370,    28.3568,    33.3036,    39.1090,    43.4518  ARMA_2SL(5, 0)
 *  15.8282,    22.2835,    28.2811,    33.2257,    39.0842,    43.4441  ARMA_2SL(5, 1)
 *  15.7199,    22.2704,    28.2829,    33.4244,    39.2114,    43.5115  ARMA_2SL(5, 2)
 *
 *  15.8017,    22.3003,    28.3201,    33.2638,    39.1084,    43.5058
 *  15.7745,    22.2892,    28.3449,    33.3113,    39.1550,    43.6822
 *  15.6674,    22.2591,    28.2856,    33.5385,    39.2694,    43.5630
 *
 *  > runMain scalation.modeling.forecasting.aRMA_2SLTest3
 */
@main def aRMA_2SLTest3 (): Unit =

    val (_, y) = clip ((null, loadData_y ()))
    val hh     = 6                                                      // maximum forecasting horizon

//  for p <- 1 to 5; q <- 0 to 2 do
    for p <- 1 to 5; q <- 0 to 1 do
        AR.hp.set (("p", p), ("q", q))                                  // # AR terms, # MA terms
        val mod = new ARMA_2SL (y, hh)                                  // create model for time series data
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.train (null, y)                                             // In-sample Testing
//      mod.inSample_Test ()                                            // In-sample Testing
    end for

end aRMA_2SLTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_2SLTest4` main function tests the `ARMA_2SL` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for AR(p), ARY(p), ARY_Quad(p), and ARMA_2SL(p, q).
 *
 *  18.4029,    29.5363,    41.9465,    53.3411,    62.7636,    73.1441  AR(1)
 *  18.3285,    28.3898,    41.5423,    53.4436,    63.2516,    73.7536  ARY(1)
 *  18.3419,    28.5375,    41.3806,    53.0299,    62.1348,    71.9170  ARY_Quad(1) @1.8
 *  18.6509,    30.3039,    42.6829,    54.0981,    63.4351,    73.5673  ARMA_2SL(1, 0)
 *  17.1051,    28.6342,    49.6391,    56.8713,    62.9405,    72.0412  ARMA_2SL(1, 1)
 *  17.0470,    24.9490,    49.5975,   >73.1670,   >76.8413,    79.2300  ARMA_2SL(1, 2)
 *
 *  18.6509,    30.3039,    42.6829,    54.0981,    63.4351,    73.5673  New results
 *  17.1051,    28.6342,    49.6391,    56.8713,    62.9405,    72.0412
 *  17.0470,    24.9490,    49.5975,   >73.1670,   >76.8413,    79.2300  new warning: don't let q > p
 *
 *  18.0433,    26.2445,    38.7122,    49.5094,    58.1561,    69.7676  AR(2)
 *  17.8345,    25.9349,    36.8827,    49.4226,    59.7645,    72.3838  ARY(2)
 *  17.8525,    27.9216,    40.8630,    53.0493,    63.4681,    74.4617  ARY_Quad(2)
 *  17.8276,    28.5037,    41.2556,    51.8990,    60.5259,    71.3822  ARMA_2SL(2, 0)
 *  20.6702,    32.7991,    45.7136,   >77.2935,   >72.1848,    68.6594  ARMA_2SL(2, 1)
 *  21.8203,    31.5059,    45.4001,   >61.9614,   >101.606,   >95.1982  ARMA_2SL(2, 2)
 *
 *  17.8276,    28.5038,    41.2557,    51.8990,    60.5259,    71.3822
 *  19.9514,    30.8196,    41.0785,   >68.8172,    65.0759,    65.1732  1 remaining instability
 *  23.6062,    41.1342,    56.5107,   >62.4819,   >97.7739,   >92.8282  3 remaining instabilities
 *
 *  16.4218,    23.5256,    32.6385,    44.4111,    51.8210,    63.7353  AR(3)
 *  16.7548,    23.0250,    28.4321,    42.2488,    47.9650,    61.1932  ARY(3)
 *  15.8536,    21.9312,    31.5295,    41.1759,    50.6815,    63.0421  ARY_Quad(3)
 *  16.1378,    23.3522,    33.5296,    43.7266,    52.1735,    63.0729  ARMA_2SL(3, 0)
 *  17.1926,    23.5788,    31.4791,    41.0385,    38.8180,    47.4447  ARMA_2SL(3, 1)
 *  17.9176,    25.0781,    34.1451,    42.8025,    56.1022,    60.9597  ARMA_2SL(3, 2)
 *
 *  16.1421,    23.3580,    33.5353,    43.7314,    52.1797,    63.0807
 *  16.7478,    23.1268,    30.6921,    40.3983,    35.1094,    44.4112
 *  17.6777,    24.9393,    34.3359,    43.4910,    52.3126,    59.7013
 *
 *  14.8153,    21.8989,    29.9364,    39.3910,    47.8124,    53.5577  AR(4)
 *  16.1682,    22.6525,    27.6705,    41.4661,    45.9318,    59.0154  ARY(4)
 *  14.2657,    19.9137,    25.6706,    35.6222,    37.5583,    47.4295  ARY_Quad(4)
 *  15.4186,    21.5099,    28.7271,    38.8750,    42.0044,    52.0754  ARMA_2SL(4, 0)
 *  16.3031,    22.0963,    30.1306,    39.7072,    43.6993,    46.6432  ARMA_2SL(4, 1)
 *  16.4455,    23.2830,    32.1969,    42.2797,    47.1758,    61.7078  ARMA_2SL(4, 2)
 *
 *  15.4171,    21.5000,    28.7220,    38.8706,    41.9990,    52.0696
 *  15.8305,    21.5951,    29.4081,    39.1845,    42.6060,    46.9077
 *  16.2333,    23.0588,    32.1199,    41.7076,    47.3536,    57.2018
 *
 *  14.3593,    21.9036,    29.1791,    40.6232,    46.1989,    55.2531  AR(5)
 *  15.8623,    21.9684,    27.4750,    40.5709,    45.4836,    57.7703  ARY(5)
 *  14.3843,    20.0371,    26.2968,    35.7404,    36.7487,    47.2926  ARY_Quad(5)
 *  15.8455,    21.3385,    29.3982,    39.4999,    43.0577,    52.7309  ARMA_2SL(5, 0)
 *  20.3754,    27.8635,    38.6308,    49.8010,   >56.2493,   >65.0986  ARMA_2SL(5, 1)
 *  14.7449,    19.9257,    27.5053,    36.8658,    41.3740,    49.5925  ARMA_2SL(5, 2)
 *
 *  15.7956,    21.2599,    29.3562,    39.4638,    43.0093,    52.6796
 *  15.7504,    20.9657,    29.1495,    38.9534,    42.7263,    51.6061
 *  14.3850,    19.5889,    27.3282,    36.3608,    41.0969,    49.0924
 *
 *  > runMain scalation.modeling.forecasting.aRMA_2SLTest4
 */
@main def aRMA_2SLTest4 (): Unit =

    val (_, y) = clip ((null, loadData_y ()))
    val hh     = 6                                                      // maximum forecasting horizon

    for p <- 1 to 5; q <- 0 to 2 do
        AR.hp.set (("p", p), ("q", q))                                  // # AR terms, # MA terms
        val mod = new ARMA_2SL (y, hh)                                  // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation
    end for

end aRMA_2SLTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_2SLTest5` main function tests the `ARMA_2SL` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for ARMA_2SL(p, 1) (i.e., q = 1) for different p orders.
 *  > runMain scalation.modeling.forecasting.aRMA_2SLTest5
 */
@main def aRMA_2SLTest5 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon

    AR.hp("q") = 1                                                      // number of MA terms
    for p <- 1 to 5 do
        AR.hp("p") = p                                                  // number of AR terms
        val mod = new ARMA_2SL (y, hh)                                  // create model for time series data
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()                                             // train and test on full dataset

        mod.forecastAll ()                                              // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (y, mod.getYf)
        println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    end for

end aRMA_2SLTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_2SLTest6` main function tests the `ARMA_2SL` class on small dataset.
 *  Test forecasts (h = 1 step ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRMA_2SLTest6
 */
@main def aRMA_2SLTest6 (): Unit =

    val y  = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3)

    AR.hp ("q") = 0
    var mod = new ARMA_2SL (y, 1)                                         // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "AR1")

    AR.hp ("p") = 2
    mod = new ARMA_2SL (y, 1)                                             // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on a Small Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    new Baseline (y, "AR2")

end aRMA_2SLTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_2SLTest7` main function tests the `ARMA_2SL` class on small dataset.
 *  Test the generation of ARMA_2SL sequences for various p and q values.
 *  > runMain scalation.modeling.forecasting.aRMA_2SLTest7
 */
@main def aRMA_2SLTest7 (): Unit =

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
            y(t) = φ(rp) ∙ y.lag (p, t) + θ(rq) ∙ e.lag (q, t) + e(t)
        end for
        new Plot (null, y, null, s"Plot of y vs. t for p = $p, q = $q", lines = true)
        object CG extends Correlogram (y)
        CG.makeCorrelogram ()
        CG.plotCorrelogram ()
    end for

end aRMA_2SLTest7

