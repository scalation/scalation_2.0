
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jul 29 11:30:42 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Time Series Data: Covid-19 Weekly Data
 */

package scalation
package modeling
package forecasting

import scala.math.min
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Example_Covid` object provides a convenient way to load Covid-19 weekly data.
 *  See test cases (odd In-Sample, even TnT Split) below for
 *                                         Loss/Equations   Optimizer
 *  (a:  1,  2) Plot and EDA               -                -
 *  Univariate:
 *  (b:  3,  4) Baseline Models            none or CSSE     none or various
 *  (c:  5,  6) AR(p) Models               Yule-Walker      Durbin-Levinson
 *  (d:  7,  8) ARMA(p, q=0) Models        CSSE             BFGS?
 *  (e:  9, 10) ARY(p) Models              CSSE             Cholesky Factorization
 *  (f: 11, 12) ARY_D(p) Models            CSSE + Direct    QR, Cholesky Factorization
 *  (g: 13, 14) ARMA(p, q=1) Models        CSSE             BFGS?
 *  Multivariate:
 *  (h: 15, 16) ARX(p, 2, 2) Models        CSSE             Cholesky Factorization
 *  (i: 17, 18) ARX_D Models               CSSE + Direct    QR, Cholesky Factorization
 *  (j: 19, 20) ARX_Quad(p, 2, 2) Models   CSSE             Cholesky Factorization
 *  (k: 21, 22) ARX_Quad_D Models          CSSE + Direct    QR, Cholesky Factorization
 *
 *  Known Bugs: SMA, WMA, SES, ARMA, ARY_D, ARX_D, ARX_Quad_D
 */
object Example_Covid:

    import scala.collection.mutable.HashMap

    val fileName = "covid_19_weekly.csv"

    val header = Array ("new_cases",
                        "new_deaths",
                        "reproduction_rate",
                        "icu_patients",
                        "hosp_patients",
                        "new_tests",
                        "positive_rate",
                        "tests_per_case",
                        "people_vaccinated",
                        "people_fully_vaccinated",
                        "total_boosters",
                        "new_vaccinations",
                        "excess_mortality_cumulative_absolute",
                        "excess_mortality_cumulative",
                        "excess_mortality",
                        "excess_mortality_cumulative_per_million")

    val response = "new_deaths"                                   // main response/output variable
    val NO_EXO   = Array.ofDim [String] (0)                       // empty array => no exogenous variables

    val yy = Example_Covid.loadData_y ()
//  val y  = yy                                                   // full
    val y  = yy(0 until 116)                                      // clip the flat end

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the Covid-19 weekly data into a matrix for the exogenous variables x
     *  and a vector for the response/endogenous variable y.
     *  @param x_strs  the column names for the exogenous variables x
     *  @param y_str   the column name for the endogenous variable y
     *  @param trim    the number of initial rows to trim away (e.g., they are all 0)
     */
    def loadData (x_strs: Array [String], y_str: String = response, trim: Int = 0): (MatrixD, VectorD) =
        val col = HashMap [String, Int] ()
        for i <- header.indices do col += header(i) -> i

        val data = MatrixD.load (fileName, 1+trim, 1)             // skip first row (header) + trim first column
        val x_cols = for s <- x_strs yield col(s)
        (data(?, x_cols), data(?, col(y_str)))
    end loadData

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the Covid-19 weekly data into a vector for the response/endogenous variable y.
     *  @param y_str  the column name for the endogenous variable y
     *  @param trim   the number of initial rows to trim away (e.g., they are all 0)
     */
    def loadData_y (y_str: String = response, trim: Int = 0): VectorD =
        val col = HashMap [String, Int] ()
        for i <- header.indices do col += header(i) -> i

        val data = MatrixD.load (fileName, 1+trim, 1)             // skip first row (header) + trim first column
        data(?, col(y_str))
    end loadData_y

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Load the Covid-19 weekly data into a matrix for the variables y.
     *  @param y_str  the column names for the variables y (e.g., used in a VAR model)
     *  @param trim   the number of initial rows to trim away (e.g., they are all 0)
     */
    def loadData_yy (y_strs: Array [String], trim: Int = 0): MatrixD =
        val col = HashMap [String, Int] ()
        for i <- header.indices do col += header(i) -> i

        val data = MatrixD.load (fileName, 1+trim, 1)             // skip first row (header) + trim first column
        val y_cols = for s <- y_strs yield col(s)
        data(?, y_cols)
    end loadData_yy

end Example_Covid

import Example_Covid._
import MakeMatrix4TS.hp

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest` main function tests the `Example_Covid` object.
 *  Prints and plots the response column ("new_deaths").
 *  > runMain scalation.modeling.forecasting.example_CovidTest
 */
@main def example_CovidTest (): Unit =

    banner (s"Print the response = $response column for the Covid-19 dataset (${y.dim} points")
    for i <- y.indices do println (s"$i \t ${y(i)}")

    banner (s"Plot the response = $response column for the Covid-19 dataset (${y.dim} points")
    new Plot (null, y, null, s"y ($response)", lines = true)

end example_CovidTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest2` main function tests the `Example_Covid` object.
 *  Performs Exploratory Data Analysis (EDA) to find relationships between
 *  contemporaneous variables.
 *  > runMain scalation.modeling.forecasting.example_CovidTest2
 */
@main def example_CovidTest2 (): Unit =

    import scala.collection.mutable.{LinkedHashSet => LSET}

    val (xx, yy) = loadData (header, response)
//  val (x, y)   = (xx, yy)                                             // full
    val (x, y)   = (xx(0 until 116), yy(0 until 116))                   // clip the flat end

    new Plot (null, y, null, s"y ($response)", lines = true)

    for j <- x.indices2 do
        banner (s"EDA for response = $response vs. ${header(j)}")
        var xj  = x(?, j)                                               // get column j
        xj = scaleV (extreme (xj), (0.0, 2.0))(xj)                      // rescale vector xj to [0, 2]
        val xxj = MatrixD.fromVector (xj)
//      val mod = SymbolicRegression.quadratic (xxj, y)
//      val mod = SymbolicRegression.rescale (xxj, y, null, LSET (1.0, 2.0, 3.0), cross = false)
        val mod = SymbolicRegression (xxj, y, null, LSET (0.5, 1.0, 2.0, 3.0), cross = false)
        mod.trainNtest ()()
        val yp = mod.predict (mod.getX)
        println (mod.summary ())
        new Plot (xj, y, yp, s"y, yp ($response) vs. x_$j (${header(j)})")
    end for

end example_CovidTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest3` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs several baseline models for horizons 1 to 6, see sMAPE metrics below:
 *
55.1927,	53.9282,	52.7133,	51.8648,	51.9621,	52.0771  Null
54.6045,	53.3361,	52.1227,	51.3005,	51.4569,	51.5041  Trend
24.2641,	25.5725,	39.2995,	45.9060,	54.4583,	60.3090  SMA -- FIX Bug h=2 too low
26.4055,	23.3947,	40.9707,	44.6394,	55.1448,	59.5280  WMA -- FIX Bug h=2 too low
18.6934,	29.1811,	38.6542,	47.1281,	54.8713,	61.9944  SES
19.0371,	29.5797,	39.0740,	47.4638,	55.1785,	62.1818  RW
18.3265,	28.7734,	38.2039,	46.7814,	54.5563,	61.7930  RWS
18.7298,	28.4908,	37.4800,	46.3173,	53.3245,	59.5733  AR(1)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest3
 */
@main def example_CovidTest3 (): Unit =

    val hh = 6                                                            // max forecasting horizon

    new Plot (null, y, null, s"y ($response)", lines = true)

    new NullModel (y, hh).inSample_Test ()                                // create a Null Model and do In-Sample Testing
    new TrendModel (y, hh).inSample_Test ()
    new SimpleMovingAverage (y, hh).inSample_Test ()
    new WeightedMovingAverage (y, hh).inSample_Test ()
    new SimpleExpSmoothing (y, hh).inSample_Test ()
    new RandomWalk (y, hh).inSample_Test ()
    new RandomWalkS (y, hh).inSample_Test ()
    new AR (y, hh).inSample_Test ()

end example_CovidTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest4` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs several baseline models for horizons 1 to 6, see sMAPE metrics below:
 *
 *  57.1057,    60.0825,    62.9136,    64.7453,    67.9247,    70.6674  Null
 *  61.9077,    65.1881,    68.7187,    71.4655,    73.9327,    75.9584  Trend
 *  22.3044,    30.4325,    45.3661,    55.7217,    67.6973,    77.4038  SMA
 *  23.8526,    30.0945,    46.9748,    55.8104,    68.7352,    77.7010  WMA
 *  18.3769,    27.1712,    40.3425,    51.8124,    63.7356,    75.0046  SES
 *  18.6713,    27.5720,    40.9387,    52.3496,    64.2481,    75.3015  RW
 *  18.0855,    26.7084,    39.6941,    51.2218,    63.1873,    74.6834  RWS
 *  19.1590,    31.1975,    44.4850,    55.3120,    65.5536,    74.4969  AR(1)

55.0263,	57.1038,	59.9686,	62.7341,	64.4922,	67.5687  Null
58.5433,	61.9389,	65.3934,	69.2238,	72.2127,	75.0520  Trend
9.30514,	20.1768,	31.9284,	44.6519,	56.0476,	67.5464  SMA -- FIX Bug
12.2955,	20.0054,	33.8672,	44.7494,	57.1694,	67.9005  WMA -- FIX Bug
33.3083,	44.2916,	54.1432,	64.0841,	73.5420,	80.7100  SES -- FIX Bug
18.1532,	27.2211,	40.3519,	52.3739,	62.5276,	73.6424  RW
17.8157,	26.6262,	39.4029,	51.5366,	61.7820,	73.3250  RWS
18.4659,	29.8363,	42.1980,	53.5928,	62.9734,	73.3153  AR(1)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest4
 */
@main def example_CovidTest4 (): Unit =

    val hh = 6                                                          // max forecasting horizon

    new Plot (null, y, null, s"y ($response)", lines = true)

    new NullModel (y, hh).tnT_Test ()                                   // create a Null Model and do TnT Testing
    new TrendModel (y, hh).tnT_Test ()
    new SimpleMovingAverage (y, hh).tnT_Test ()
    new WeightedMovingAverage (y, hh).tnT_Test ()
    new SimpleExpSmoothing (y, hh).tnT_Test ()
    new RandomWalk (y, hh).tnT_Test ()
    new RandomWalkS (y, hh).tnT_Test ()
    new AR (y, hh).tnT_Test ()

end example_CovidTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest5` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive AR(p) models for several p values and horizons 1 to 6,
 *  see sMAPE metrics below:
 *
18.7298,	28.4908,	37.4800,	46.3173,	53.3245,	59.5733  AR(1)
16.3579,	24.7155,	33.0480,	40.1643,	46.8762,	53.2178  AR(2)
16.0114,	22.7408,	29.5631,	35.2773,	41.5856,	47.5716  AR(3)
15.8988,	22.5738,	28.5298,	33.3360,	39.1586,	44.3459  AR(4)
15.9279,	22.5769,	28.5035,	33.3019,	39.1381,	43.0520  AR(5)
15.9647,	22.6143,	28.5229,	33.3735,	39.1651,	42.9640  AR(6)
16.0207,	23.2172,	29.4751,	35.2827,	41.0976,	46.1932  AR(7)
16.0501,	22.7281,	28.6740,	34.1866,	39.5963,	44.9223  AR(8)
16.0196,	22.5269,	28.4223,	34.1619,	39.7297,	44.4649  AR(9)
16.1069,	22.6213,	28.6435,	34.2722,	39.9638,	44.8023  AR(10)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest5
 */
@main def example_CovidTest5 (): Unit =

    val hh = 6                                                          // max forecasting horizon
    val hp = AR.hp                                                      // hyper-parameters for AR family of models

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // AR hyper-parameter settings
        hp("p") = p
        new AR (y, hh).inSample_Test ()                                 // create an AR model and do In-Sample Testing
    end for

end example_CovidTest5
 

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest6` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive AR(p) models for several p values and horizons 1 to 6,
 *  see sMAPE metrics below:
 *
18.4659,	29.8363,	42.1980,	53.5928,	62.9734,	73.3153  AR(1)
16.7534,	25.7382,	38.5096,	49.3593,	57.9183,	69.7091  AR(2)
16.3630,	21.1490,	29.8750,	39.7999,	47.3691,	59.0869  AR(3)
15.0428,	20.1558,	29.3151,	38.2679,	43.0488,	51.3448  AR(4)
14.9448,	20.2989,	27.2780,	37.3160,	41.9003,	54.0098  AR(5)
13.9802,	19.7390,	27.2648,	35.6434,	42.2692,	50.8636  AR(6)
14.3902,	22.1659,	32.1102,	44.2965,	50.4653,	59.6916  AR(7)
15.0354,	24.8868,	35.9570,	49.9725,	55.2307,	62.9366  AR(8)
14.3458,	23.0047,	32.7333,	44.5037,	50.6380,	61.1755  AR(9)
14.0441,	23.9778,	35.8541,	48.1709,	53.5309,	63.7929  AR(10)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest6
 */
@main def example_CovidTest6 (): Unit =

    val hh = 6                                                          // max forecasting horizon
    val hp = AR.hp                                                      // hyper-parameters for AR family of models

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // AR hyper-parameter settings
        hp("p") = p
        new AR (y, hh).tnT_Test ()                                      // create an AR model and do TnT Testing
    end for

end example_CovidTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest7` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive, Moving Average ARMA(p, 0) models for several p and
 *  horizons 1 to 6, see sMAPE metrics below:
 *
 *  20.2191,    29.9108,    38.1525,    45.5858,    52.2918,    57.3670  ARMA(1, 0)
 *  17.7900,    25.3293,    33.3283,    39.5055,    44.9095,    50.6043  ARMA(2, 0)
 *  17.4057,    23.9135,    30.5357,    35.5950,    40.6434,    46.4122  ARMA(3, 0)
 *  17.2928,    23.6678,    29.5574,    34.0383,    38.9062,    44.1568  ARMA(4, 0)
 *  17.2850,    23.6708,    29.5699,    34.0520,    38.9330,    44.2125  ARMA(5, 0)
 *  17.3271,    23.9829,    29.9874,    34.6032,    39.0682,    43.6979  ARMA(6, 0)
 *  17.2335,    24.0097,    29.9465,    34.3426,    38.9182,    44.4357  ARMA(7, 0)
 *  17.2811,    23.7288,    29.5992,    34.0946,    38.6983,    44.1365  ARMA(8, 0)
 *  17.2044,    23.6396,    29.5609,    34.2834,    38.9406,    44.1984  ARMA(9, 0)
 *  17.2588,    23.6012,    29.4737,    34.3447,    39.0981,    44.1297  ARMA(10, 0)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest7
 */
@main def example_CovidTest7 (): Unit =

    val hh = 6                                                          // max forecasting horizon
    val hp = AR.hp                                                      // hyper-parameters for AR family of models
    hp("q") = 0                                                         // no MA terms => AR with different optimizer

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARMA hyper-parameter settings
        hp("p") = p
        new ARMA (y, hh).inSample_Test ()                               // create an ARMA model and do In-Sample Testing
    end for

end example_CovidTest7


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest8` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive, Moving Average ARMA(p, 0) models for several p values
 *  and horizons 1 to 6, see sMAPE metrics below:
 *
 *  19.0003,    30.3936,    43.8008,    54.8254,    65.3736,    74.5465  ARMA(1, 0)
 *  17.0385,    26.7633,    39.4985,    51.0132,    61.2488,    70.4454  ARMA(2, 0)
 *  16.0454,    22.1844,    31.7033,    41.1297,    51.6017,    61.3707  ARMA(3, 0)
 *  15.2966,    20.7829,    27.7076,    36.3322,    41.5452,    49.0153  ARMA(4, 0)
 *  15.6244,    20.6003,    29.0435,    36.8354,    43.1722,    48.1613  ARMA(5, 0)
 *  15.6619,    23.1335,    32.0946,    41.3166,    50.0557,    60.0608  ARMA(6, 0)
 *  16.0957,    22.2142,    32.4196,    39.8389,    47.6075,    51.5675  ARMA(7, 0)
 *  15.8659,    25.6319,    36.0707,    45.6189,    54.9417,    58.8670  ARMA(8, 0)
 *  15.5716,    24.2525,    34.1386,    44.2350,    55.1113,    60.8057  ARMA(9, 0)
 *  14.9008,    22.6571,    30.4335,    41.6601,    50.1669,    61.2246  ARMA(10, 0)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest8
 */
@main def example_CovidTest8 (): Unit =

    val hh = 6                                                          // max forecasting horizon
    val hp = AR.hp                                                      // hyper-parameters for AR family of models
    hp("q") = 0                                                         // no MA terms => AR with different optimizer

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARMA hyper-parameter settings
        hp("p") = p
        new ARMA (y, hh).tnT_Test ()                                    // create an ARMA model and do TnT Testing
    end for

end example_CovidTest8


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest9` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive, Lagged Regression ARY(p) models for several p values and
 *  horizons 1 to 6, see sMAPE metrics below:
 *
18.7156,	28.5159,	37.2459,	45.8036,	51.9371,	56.6985  ARY(1)
16.2587,	23.7072,	31.9906,	38.9940,	44.9856,	50.2418  ARY(2)
15.8240,	22.2659,	29.1170,	34.8505,	40.8580,	46.2113  ARY(3)
15.7020,	22.0134,	28.1169,	33.2691,	39.1811,	44.0750  ARY(4)
15.6875,	22.0198,	28.1429,	33.2990,	39.2397,	44.1700  ARY(5)
15.6982,	22.3197,	28.5239,	33.7716,	39.3099,	43.7038  ARY(6)
15.6186,	22.3438,	28.5437,	33.6401,	39.2926,	44.4411  ARY(7)
15.6595,	22.0566,	28.1782,	33.3346,	38.9921,	44.1955  ARY(8)
15.5823,	21.9463,	28.1055,	33.4000,	39.1296,	44.2667  ARY(9)
15.6267,	21.9089,	28.0047,	33.4205,	39.1586,	44.2015  ARY(10)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest9
 */
@main def example_CovidTest9 (): Unit =

    val hh = 6                                                          // max forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARY hyper-parameter settings
        hp("p") = p
        ARY (y, hh).inSample_Test ()                                    // create an ARY model and do In-Sample Testing
    end for

end example_CovidTest9


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest10` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive, Lagged Regression ARY(p) models for several p values,
 *  and horizons 1 to 6, see sMAPE metrics below:
 *
18.4998,	29.4024,	42.6675,	54.2169,	63.0084,	72.2290  ARY(1)
17.8595,	27.8544,	40.7768,	52.5351,	61.0972,	70.1973  ARY(2)
16.4745,	22.9627,	32.9324,	42.8385,	51.6931,	61.5292  ARY(3)
15.7400,	21.2937,	28.9582,	38.8393,	42.0740,	51.3300  ARY(4)
16.2315,	21.2230,	29.6593,	38.9380,	43.1389,	49.8287  ARY(5)
16.1056,	22.6427,	30.9964,	40.9072,	46.1625,	54.0774  ARY(6)
16.6737,	23.1236,	33.1328,	43.0148,	49.7683,	55.2326  ARY(7)
16.7242,	25.7990,	36.3484,	47.6660,	55.1432,	60.7600  ARY(8)
16.5522,	24.9271,	35.1184,	46.2757,	53.5985,	59.8611  ARY(9)
16.0764,	23.6507,	33.7168,	44.2123,	51.0500,	61.1110  ARY(10)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest10
 */
@main def example_CovidTest10 (): Unit =

    val hh = 6                                                          // max forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARY hyper-parameter settings
        hp("p") = p
        ARY (y, hh).tnT_Test ()                                         // create an ARY model and do TnT Testing
    end for

end example_CovidTest10


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest11` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive, Lagged Regression, Direct ARY_D(p) models for several p values,
 *  and horizons 1 to 6, see sMAPE metrics below:
 *
 *  19.9912,    30.1349,    38.7483,    45.1096,    49.5424,    52.5320  ARY_D(1)
 *  17.7245,    24.2871,    31.1716,    35.9357,    40.5132,    46.4806  ARY_D(2)
 *  17.2367,    23.2007,    29.4120,    33.5757,    38.8647,    44.1707  ARY_D(3)
 *  17.1336,    23.1984,    29.1758,    33.5773,    38.6493,    43.8045  ARY_D(4)
 *  17.1196,    23.1224,    29.1769,    33.6120,    38.7839,    43.9346  ARY_D(5)
 *  17.1324,    23.1273,    29.2292,    33.8956,    39.1209,    44.0869  ARY_D(6)
 *  16.9815,    23.2879,    29.2536,    33.9433,    39.1474,    44.2361  ARY_D(7)
 *  17.0492,    23.1888,    29.2826,    34.0878,    39.2379,    44.7474  ARY_D(8)
 *  16.9841,    23.1090,    29.2154,    34.1249,    39.2711,    44.7709  ARY_D(9)
 *  17.0676,    23.1089,    28.9425,    33.9046,    38.9082,    44.0469  ARY_D(10)
 
18.7192,	28.0356,	38.0739,	46.8690,	54.2154,	60.8921  ARY_D(1)  // FIX Bug -- too high
16.2602,	23.9446,	33.8763,	42.7548,	50.5601,	58.3793  ARY_D(2)
15.8284,	23.1419,	32.7795,	41.9619,	50.0289,	57.6217  ARY_D(3)
15.7065,	22.8376,	32.3402,	41.6187,	49.6103,	57.1951  ARY_D(4)
15.6925,	22.8577,	32.3599,	41.6423,	49.6253,	57.2027  ARY_D(5)
15.7054,	22.8457,	32.0807,	41.5924,	49.5162,	57.1057  ARY_D(6)
15.6252,	22.9759,	32.3061,	41.7185,	49.5916,	57.1693  ARY_D(7)
15.6665,	22.8945,	32.1515,	41.6596,	49.5256,	57.1121  ARY_D(8)
15.5888,	22.8066,	32.0457,	41.6402,	49.5181,	57.0147  ARY_D(9)
15.6341,	22.7900,	31.9169,	41.5769,	49.4211,	56.8442  ARY_D(10)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest11
 */
@main def example_CovidTest11 (): Unit =

    val hh = 6                                                          // max forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARY_D hyper-parameter settings
        hp("p") = p
        ARY_D (y, hh).inSample_Test ()                                  // create an ARY_D model and do In-Sample Testing
    end for

end example_CovidTest11


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest12` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive, Lagged Regression, Direct ARY_D(p) models for several p values,
 *  and horizons 1 to 6, see sMAPE metrics below:
 *
18.5012,	30.6359,	44.6907,	56.3819,	63.5357,	70.2028  ARY_D(1)
17.8594,	25.6887,	35.4256,	45.1790,	50.4527,	60.0111  ARY_D(2)
16.4715,	21.8016,	29.0381,	38.5104,	41.7722,	52.5948  ARY_D(3)
15.7366,	21.5619,	29.6182,	38.5514,	42.6748,	50.8053  ARY_D(4)
16.2315,	21.6376,	30.1303,	39.2472,	43.2712,	51.3975  ARY_D(5)
16.1058,	21.4914,	30.3555,	39.4806,	43.6360,	52.2788  ARY_D(6)
16.6739,	24.9328,	34.3104,	43.0995,	50.1624,	55.6230  ARY_D(7)
16.7248,	25.6616,	34.1789,	44.1880,	50.5008,	57.2136  ARY_D(8)
16.5494,	24.1130,	33.8701,	42.6320,	49.7175,	59.3071  ARY_D(9)
16.0705,	24.2940,	34.3169,	43.9710,	52.8199,	63.8433  ARY_D(10)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest12
 */
@main def example_CovidTest12 (): Unit =

    val hh = 6                                                          // max forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARY_D hyper-parameter settings
        hp("p") = p
        ARY_D (y, hh).tnT_Test ()                                       // create an ARY_D model and do TnT Testing
    end for

end example_CovidTest12


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest13` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive, Moving Average ARMA(p, q) models for several p values,
 *  and horizons 1 to 6, see sMAPE metrics below:
 *
 *  FIX - good for h = 1, but then sMAPE scores explode
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest13
 */
@main def example_CovidTest13 (): Unit =

    val hh = 6                                                          // max forecasting horizon
    val hp = AR.hp                                                      // hyper-parameters for AR family of models
    hp("q") = 1                                                         // one MA term

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARMA hyper-parameter settings
        hp("p") = p
        new ARMA (y, hh).inSample_Test ()                               // create an ARMA model and do In-Sample Testing
    end for

end example_CovidTest13


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest14` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive, Moving Average ARMA(p, q) models for several p values.
 *  and horizons 1 to 6, see sMAPE metrics below:
 * 
 *  FIX - for all h sMAPE scores have exploded
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest14
 */
@main def example_CovidTest14 (): Unit =

    val hh = 6                                                          // max forecasting horizon
    val hp = AR.hp                                                      // hyper-parameters for AR family of models
    hp("q") = 1                                                         // one MA term

    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARMA hyper-parameter settings
        hp("p") = p
        new ARMA (y, hh).tnT_Test ()                                    // create an ARMA model and do TnT Testing
    end for

end example_CovidTest14


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest15` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive, Exogenous ARX(p, q, n) models for several p values,
 *  and horizons 1 to 6, see sMAPE metrics below:
 *
16.8457,	24.4372,	32.1950,	40.6005,	47.7611,	53.9067  ARX(1, 1, 2)
14.0454,	18.9222,	26.7142,	35.7187,	43.8227,	50.6001  ARX(2, 2, 2)
13.8452,	18.1834,	24.5848,	32.9174,	40.9337,	47.7760  ARX(3, 2, 2)
13.7872,	17.8297,	23.8105,	31.4904,	39.1773,	46.0194  ARX(4, 2, 2)
13.7843,	17.8311,	23.8111,	31.4940,	39.1855,	46.0289  ARX(5, 2, 2)
13.8789,	17.9482,	24.0073,	31.5983,	38.8170,	45.1675  ARX(6, 2, 2)
13.8181,	17.9809,	24.0099,	31.6827,	38.9771,	45.5835  ARX(7, 2, 2)
13.9439,	17.6081,	23.7523,	31.3991,	38.6531,	44.9676  ARX(8, 2, 2)
13.7840,	17.6349,	23.8658,	31.1578,	38.6519,	44.6899  ARX(9, 2, 2)
13.8821,	17.6217,	23.6753,	31.0605,	38.6224,	44.6828  ARX(10, 2, 2)
 *
18.7156,	28.5159,	37.2459,	45.8036,	51.9371,	56.6985  ARX(1, 1, 0)  Agrees with ARY(p)
16.2587,	23.7072,	31.9906,	38.9940,	44.9856,	50.2418  ARX(2, 2, 0)
15.8240,	22.2659,	29.1170,	34.8505,	40.8580,	46.2113  ARX(3, 2, 0)
15.7020,	22.0134,	28.1169,	33.2691,	39.1811,	44.0750  ARX(4, 2, 0)
15.6875,	22.0198,	28.1429,	33.2990,	39.2397,	44.1700  ARX(5, 2, 0)
15.6982,	22.3197,	28.5239,	33.7716,	39.3099,	43.7038  ARX(6, 2, 0)
15.6186,	22.3438,	28.5437,	33.6401,	39.2926,	44.4411  ARX(7, 2, 0)
15.6595,	22.0566,	28.1782,	33.3346,	38.9921,	44.1955  ARX(8, 2, 0)
15.5823,	21.9463,	28.1055,	33.4000,	39.1296,	44.2667  ARX(9, 2, 0)
15.6267,	21.9089,	28.0047,	33.4205,	39.1586,	44.2015  ARX(10, 2, 0)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest15
 */
@main def example_CovidTest15 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
//  val exo_vars  = Array ("icu_patients", "hosp_patients")
    val exo_vars  = NO_EXO
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    banner (s"exo_vars = ${stringOf (exo_vars)}, endo_var = $response")
    println (s"xe.dims = ${xe.dims}, y.dim = ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARX hyper-parameter settings
        hp("p") = p
        hp("q") = min (2, p)
        ARX (xe, y, hh).inSample_Test ()                                // create an ARX model and do In-Sample Testing
    end for

end example_CovidTest15


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest16` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive, Exogenous ARX(p, q, n) models for several p values,
 *  and horizons 1 to 6, see sMAPE metrics below:
 *
11.0073,	19.3191,	26.2452,	35.9969,	47.8490,	58.9978  ARX(1, 1, 2)
10.4339,	19.6885,	25.2416,	35.4041,	47.6633,	58.1026  ARX(2, 2, 2)
10.0477,	19.6210,	25.8577,	35.9296,	47.6708,	58.2116  ARX(3, 2, 2)
9.34031,	17.6473,	23.3025,	33.0873,	46.0021,	57.6114  ARX(4, 2, 2)
10.5476,	17.6866,	23.1955,	33.2009,	45.4878,	57.6834  ARX(5, 2, 2)
11.5526,	18.6435,	24.7745,	35.5272,	44.6416,	57.7697  ARX(6, 2, 2)
11.3831,	17.8615,	23.0079,	33.1844,	44.4669,	57.2788  ARX(7, 2, 2)
11.1061,	17.4816,	22.1642,	33.2042,	45.1645,	57.6688  ARX(8, 2, 2)
11.3308,	18.0780,	23.5240,	34.5589,	46.3419,	58.6557  ARX(9, 2, 2)
11.4131,	19.5224,	25.7109,	36.8454,	49.5382,	60.9186  ARX(10, 2, 2)
 *
18.4998,	29.4024,	42.6675,	54.2169,	63.0084,	72.2290  ARX(1, 1, 0)  Agrees with ARY(p)
17.8595,	27.8544,	40.7768,	52.5351,	61.0972,	70.1973  ARX(2, 2, 0)
16.4745,	22.9627,	32.9324,	42.8385,	51.6931,	61.5292  ARX(3, 2, 0)
15.7400,	21.2937,	28.9582,	38.8393,	42.0740,	51.3300  ARX(4, 2, 0)
16.2315,	21.2230,	29.6593,	38.9380,	43.1389,	49.8287  ARX(5, 2, 0)
16.1056,	22.6427,	30.9964,	40.9072,	46.1625,	54.0774  ARX(6, 2, 0)
16.6737,	23.1236,	33.1328,	43.0148,	49.7683,	55.2326  ARX(7, 2, 0)
16.7242,	25.7990,	36.3484,	47.6660,	55.1432,	60.7600  ARX(8, 2, 0)
16.5522,	24.9271,	35.1184,	46.2757,	53.5985,	59.8611  ARX(9, 2, 0)
16.0764,	23.6507,	33.7168,	44.2123,	51.0500,	61.1110  ARX(10, 2, 0)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest16
 */
@main def example_CovidTest16 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
//  val exo_vars  = Array ("icu_patients", "hosp_patients")
    val exo_vars  = NO_EXO
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    banner (s"exo_vars = ${stringOf (exo_vars)}, endo_var = $response")
    println (s"xe.dims = ${xe.dims}, y.dim = ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARX hyper-parameter settings
        hp("p") = p
        hp("q") = min (2, p)
        ARX (xe, y, hh).tnT_Test ()                                     // create an ARX model and do TnT Testing
    end for

end example_CovidTest16


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest17` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive, Exogenous ARX_D(p, q, n) models for several p values.
 *
16.8503,	25.2909,	35.4803,	44.6073,	52.2135,	59.4200  ARX_D(1, 1, 2)  FIX Bug -- too high
14.0491,	20.5723,	30.7172,	40.1614,	48.7536,	56.8138  ARX_D(2, 2, 2)
13.8498,	20.2659,	30.2173,	39.6863,	48.4629,	56.3433  ARX_D(3, 2, 2)
13.7940,	19.9815,	29.8293,	39.3993,	48.0656,	55.9535  ARX_D(4, 2, 2)
13.7917,	19.9831,	29.8315,	39.4017,	48.0668,	55.9540  ARX_D(5, 2, 2)
13.8881,	20.0432,	29.5594,	39.4102,	47.9529,	55.8790  ARX_D(6, 2, 2)
13.8256,	20.2026,	29.7336,	39.5616,	48.0363,	55.9488  ARX_D(7, 2, 2)
13.9522,	20.0203,	29.5011,	39.4600,	47.9126,	55.8402  ARX_D(8, 2, 2)
13.7933,	19.8568,	29.2884,	39.3859,	47.7936,	55.7004  ARX_D(9, 2, 2)
13.8925,	19.8232,	29.0014,	39.2233,	47.6079,	55.3331  ARX_D(10, 2, 2)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest17
 */
@main def example_CovidTest17 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val exo_vars  = Array ("icu_patients", "hosp_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    banner (s"exo_vars = ${stringOf (exo_vars)}, endo_var = $response")
    println (s"xe.dims = ${xe.dims}, y.dim = ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARX_D hyper-parameter settings
        hp("p") = p
        hp("q") = min (2, p)
        ARX_D (xe, y, hh).inSample_Test ()                              // create an ARX_D model and do In-Sample Testing
    end for

end example_CovidTest17


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest18` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive, Exogenous ARX_D(p, q, n) models for several p values.
 *
11.0164,	18.1939,	23.2358,	31.6475,	42.8253,	54.3245  ARX_D(1, 1, 2)
10.4521,	13.8955,	13.5074,	24.7088,	35.3807,	59.9904  ARX_D(2, 2, 2)
10.0793,	11.2167,	14.3570,	25.6690,	40.5868,	62.7553  ARX_D(3, 2, 2)
9.34877,	11.5869,	17.2668,	29.0107,	44.4698,	63.0348  ARX_D(4, 2, 2)
10.5457,	13.4293,	17.9561,	31.2434,	47.5865,	68.7770  ARX_D(5, 2, 2)
11.5562,	12.9348,	18.9845,	34.2571,	55.0142,	76.4654  ARX_D(6, 2, 2)
11.3872,	12.5172,	19.2266,	33.6051,	60.1889,	80.5061  ARX_D(7, 2, 2)
11.1091,	12.5290,	17.6071,	34.1205,	61.5160,	76.5128  ARX_D(8, 2, 2)
11.3466,	12.4147,	18.3370,	34.4428,	61.0988,	78.5466  ARX_D(9, 2, 2)
11.4265,	12.6378,	17.6644,	34.6527,	61.0367,	81.0210  ARX_D(10, 2, 2)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest18
 */
@main def example_CovidTest18 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val exo_vars  = Array ("icu_patients", "hosp_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    banner (s"exo_vars = ${stringOf (exo_vars)}, endo_var = $response")
    println (s"xe.dims = ${xe.dims}, y.dim = ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARX_D hyper-parameter settings
        hp("p") = p
        hp("q") = min (2, p)
        ARX_D (xe, y, hh).tnT_Test ()                                   // create an ARX_D model and do TnT Testing
    end for

end example_CovidTest18


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest19` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive, Exogenous ARX_Quad(p, q, n) models for several p values.
 *  Uses RidgeRegression with lambda = 0.1; y^pow with pow = 1.5.
 *
16.6072,	24.4626,	32.6457,	40.8155,	47.9617,	54.1519  ARX_Quad(1, 1, 2)
14.8114,	18.7980,	24.6991,	33.2430,	42.0688,	49.9135  ARX_Quad(2, 2, 2)
14.3609,	18.4683,	24.3359,	32.3833,	40.2261,	48.3873  ARX_Quad(3, 2, 2)
14.1085,	18.5579,	24.8609,	32.6986,	40.6706,	48.4073  ARX_Quad(4, 2, 2)
13.8807,	18.3509,	24.6423,	32.0085,	40.2473,	47.6040  ARX_Quad(5, 2, 2)
13.6989,	18.4814,	24.6467,	32.5355,	40.7019,	48.2871  ARX_Quad(6, 2, 2)
13.7876,	18.4203,	24.6844,	32.2166,	40.6423,	48.0845  ARX_Quad(7, 2, 2)
13.9035,	17.5713,	23.6566,	31.4147,	40.4540,	48.7819  ARX_Quad(8, 2, 2)
13.7435,	17.6190,	23.4393,	31.4997,	40.7884,	49.3476  ARX_Quad(9, 2, 2)
13.8470,	17.8286,	22.5861,	30.3541,	38.8571,	47.1813  ARX_Quad(10, 2, 2)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest19
 */
@main def example_CovidTest19 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val exo_vars  = Array ("icu_patients", "hosp_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    banner (s"exo_vars = ${stringOf (exo_vars)}, endo_var = $response")
    println (s"xe.dims = ${xe.dims}, y.dim = ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARX_Quad hyper-parameter settings
        hp("p") = p
        hp("q") = min (2, p)
        ARX_Quad (xe, y, hh).inSample_Test ()                           // create an ARX_Quad model and do In-Sample Testing
    end for

end example_CovidTest19


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest20` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive, Exogenous ARX_Quad(p, q, n) models for several p values.
 *  Uses RidgeRegression with lambda = 0.1; y^pow with pow = 1.5.
 *
11.8379,	18.8631,	25.3769,	34.6704,	46.0594,	58.3846  ARX_Quad(1, 1, 2)
11.4651,	13.9379,	17.4264,	29.5020,	41.4593,	69.7777  ARX_Quad(2, 2, 2)
10.9527,	12.5247,	16.9508,	28.3049,	44.0043,	66.2080  ARX_Quad(3, 2, 2)
10.0516,	12.8597,	17.9446,	30.4416,	45.4654,	64.7888  ARX_Quad(4, 2, 2)
10.9845,	13.3932,	18.5001,	32.2812,	48.1254,	69.1962  ARX_Quad(5, 2, 2)
11.1224,	13.8001,	20.4705,	36.9029,	56.0602,	82.4567  ARX_Quad(6, 2, 2)
11.8387,	13.8277,	21.2469,	37.6986,	63.8038,	88.8678  ARX_Quad(7, 2, 2)
11.5418,	14.6203,	21.4781,	40.8681,	70.1170,	88.3799  ARX_Quad(8, 2, 2)
12.3724,	14.8509,	21.5095,	40.9054,	70.3691,	87.8717  ARX_Quad(9, 2, 2)
12.6086,	15.7098,	21.6644,	41.5958,	69.4722,	82.9493  ARX_Quad(10, 2, 2)
 *
 *  > runMain scalation.modeling.forecasting.example_CovidTest20
 */
@main def example_CovidTest20 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val exo_vars  = Array ("icu_patients", "hosp_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    banner (s"exo_vars = ${stringOf (exo_vars)}, endo_var = $response")
    println (s"xe.dims = ${xe.dims}, y.dim = ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARX_Quad_D hyper-parameter settings
        hp("p") = p
        hp("q") = min (2, p)
        ARX_Quad_D (xe, y, hh).tnT_Test ()                              // create an ARX_Quad_D model and do TnT Testing
    end for

end example_CovidTest20


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest21` main function tests the `Example_Covid` object.
 *  Uses In-Sample Testing, i.e., train and test on the same data.
 *  Runs Auto-Regressive, Exogenous ARX_Quad_D(p, q, n) models for several p values.
 *  > runMain scalation.modeling.forecasting.example_CovidTest21
 */
@main def example_CovidTest21 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val exo_vars  = Array ("icu_patients", "hosp_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    banner (s"exo_vars = ${stringOf (exo_vars)}, endo_var = $response")
    println (s"xe.dims = ${xe.dims}, y.dim = ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARX_Quad_D hyper-parameter settings
        hp("p") = p
        hp("q") = min (2, p)
        ARX_Quad_D (xe, y, hh).inSample_Test ()                         // create an ARX_Quad_D model and do In-Sample Testing
    end for

end example_CovidTest21


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `example_CovidTest22` main function tests the `Example_Covid` object.
 *  Uses Train-n-Test Split (TnT) with Rolling Validation.
 *  Runs Auto-Regressive, Exogenous ARX_Quad_D(p, q, n) models for several p values.
 *  > runMain scalation.modeling.forecasting.example_CovidTest22
 */
@main def example_CovidTest22 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val exo_vars  = Array ("icu_patients", "hosp_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
//  hp("lambda") = 1.0                                                  // regularization parameter

    banner (s"exo_vars = ${stringOf (exo_vars)}, endo_var = $response")
    println (s"xe.dims = ${xe.dims}, y.dim = ${y.dim}")
    new Plot (null, y, null, s"y ($response)", lines = true)

    for p <- 1 to 10 do                                                 // ARX_Quad_D hyper-parameter settings
        hp("p") = p
        hp("q") = min (2, p)
        ARX_Quad_D (xe, y, hh).tnT_Test ()                              // create an ARX_Quad_D model and do TnT Testing
    end for

end example_CovidTest22

