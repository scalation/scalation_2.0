
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, John Miller
 *  @version 2.0
 *  @date    Thu May 26 13:00:49 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Statistical Test for Time Series Stationarity
 *
 *  Unit Root Tests for Time Series Stationarity
 *  (1 is a root of the process characteristic equation)
 *  @see github.com/olmallet81/URT
 */

package scalation
package modeling
package forecasting

import scala.collection.immutable.HashMap
import scala.Double.NaN
import scala.math.max
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._
import scalation.random.{Normal, Variate}

type CriticalValues = HashMap [Int, VectorD]

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `UnitRoot` trait provides a common framework for various unit root testers
 *  for Time Series Stationarity.
 *  This code is translated from the C++ code found in
 *  @see github.com/olmallet81/URT.
 *  @param testName     the name of test, e.g., KPSS 
 *  @param nobs         the number of observations (length of time-series)
 *  @param validTrends  vector of test valid trends types, e.g., constant, linear trend
 *  @param lagsType     default lags value long or short time-series
 *  @param lags         the number of lags to use
 *  @param trend        type of trend to test for
 */
trait UnitRoot (protected val testName: String, protected val nobs: Int,
                protected val validTrends: VectorS, protected var lagsType: String,
                protected var lags: Int, protected var trend: String):

    protected var coeff: HashMap [Double, CriticalValues] = null        // HashMap containing critical values coefficients
    protected var pval               = 1.0                              // test p-value
    protected var newLags            = false                            // control if a new number of lags has been chosen
    protected var newTrend           = false                            // control if a new trend has been chosen

    private var maxLags              = 0                                // maximum number of lags for lag length optimization
    private var npar                 = 0                                // number of parameters excluding lag difference terms
    private var prevLagsType: String = null                             // previous type of lags
    private var prevTrend: String    = null                             // previous regression trend
    private var trendType: String    = null                             // regression trend for outputting test results
    private var prevLags             = 0                                // previous number of lags
    private val optim                = false                            // control if lag length is optimized
    private var newTest              = false                            // control if new test is run (true) or all parameters remain same (false)

    private val probas       = Array (0.001, 0.005, 0.01, 0.025, 0.05,
                                      0.10 , 0.20 , 0.50, 0.80 , 0.90,
                                      0.95 , 0.975, 0.99, 0.995, 0.999)  // array of probabilities for p-value computation
    private val criticalVals = Array.ofDim [Double] (probas.length)      // array containing tne test critical values

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the values for lags for trend.
     *  @param lags   the number of lags to use
     *  @param trend  type of trend to test for
     */
    def reset (lags_ : Int, trend_ : String): Unit = { lags = lags_; trend = trend_ }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get test pvalue.
     */
    def getPval (): Double = pval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get test valid trends.
     */
    def getTrends (): VectorS = validTrends

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute test statistic.
     */
    def statistic (): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute test statistic p-value.
     */
    def pvalue (stat: Double): Double =
        if stat.isNaN then pval = NaN
        else if newTest then                                            // if a new test has been run
            computeCV ()                                                // computing critical values
            computePval (stat)                                          // computing p-value
            newTest = false
        pval
    end pvalue

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Output test results (can be overridden by derived classes).
     */
    def show (stat: Double): Unit =
        banner (s"$testName Test Results")                              // outputting test name
        println (s"  Statistic = $stat")                                // outputting test statistic
        print   ("  P-value = ")                                        // outputting p-value

        if      pval <= probas(0) then  print ("< 0.001")
        else if pval >= probas(14) then print ("> 0.999")
        else print (pval)
        println ()

        println (s"  Lags  = $lags")
        println (s"  Trend = $trendType")
        println ("  ------------------------------------\n")

        println ("  Test Hypothesis")                                   // outputting test hypothesis
        println ("  ------------------------------------")

        println ("  H0: The process is weakly stationary")              // KPSS hypotheses
        println ("  H1: The process contains a unit root")
        println ()

        println ("  Critical Values")                                   // outputting critical values
        println ("  ---------------")

        val idx = Array (12, 10, 9)                                     // declaring array of critical value indexes

        if stat.isNaN then
            println (s"   1% $NaN")
            println (s"   5% $NaN")
            println (s"  10% $NaN")
        else
            println (s"   1% ${criticalVals(idx(0))}")
            println (s"   5% ${criticalVals(idx(1))}")
            println (s"  10% ${criticalVals(idx(2))}")

        println ("  ---------------")
        println ("  Test Conclusion")                                   // outputting test conclusion

        if      pval <= 0.01 then println ("We can reject H0 at the 1% significance level")
        else if pval <= 0.05 then println ("We can reject H0 at the 5% significance level")
        else if pval <= 0.10 then println ("We can reject H0 at the 10% significance level")
        else if ! pval.isNaN then println ("We cannot reject H0")
        else println ("We cannot conclude, NaN produced")
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set number of lags, checking input validity. This method needs to know
     *  optim so it needs to be run after set_method ().
     */
    protected def setLags (): Unit =
        if ! optim then
            if lags < 0 then                                            // number of lags cannot be strictly negative
                lags = 0
                println ("\n  WARNING: number of lags cannot be negative, it has been set to 0 by default.\n")
            if ! lagsType.isEmpty && lags != prevLags then lagsType = ""   // if user has switched from a default lags value
                                                                           // to a value of his choice (for all tests)
        else if maxLags < 0 then  // maxLags cannot be strictly negative
            maxLags = 0
            println ("\n  WARNING: maximum number of lags cannot be negative, it has been set to a default value (L12-rule).\n")

        // updating lags only for PP and KPSS tests, for ADF and DFGLS tests lags will be updated at the next optimization or
        // set back to prevLags if maxLags, trend, method and level are the same as before

        if optim && maxLags == 0 || ! lagsType.isEmpty then             // computing default lags value for KPSS test
            maxLags =
            if lagsType == "short" then (4 *  (0.01 * nobs)~^0.25).toInt   // short => L4-rule (Schwert 1989)
            else (12 * (0.01 * nobs)~^0.25).toInt                       // long => L12-rule (Schwert 1989)
            lags = maxLags
        end if

        if ! optim && lags != prevLags then                             // if number of lags is different than previous value
            newTest  = true
            newLags  = true
            prevLags = lags
    end setLags

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set lags type long or short for PP and KPSS default lags value or ADF
     *  and DFGLS default maxlags value.
     */
    protected def setLagsType (): Unit =
        if lagsType.isEmpty || lagsType == prevLagsType then return     // skipping method if lagsType is empty or did not change
        if lagsType != "long" && lagsType != "short" then
            println("\n  WARNING: unknown default type of lags, long has been selected by default.\n")
            lagsType = "long"                                           // default lags type is long
        prevLagsType = lagsType
    end setLagsType

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set regression trend.
     */
    protected def setTrend (): Unit =
        if trend == prevTrend then return                              // skip method if trend did not change

        if ! validTrends.contains (trend) then                         // invalid => set default trend to constant
            trend     = "c"
            trendType = "constant"; npar = 2
            println ("\n  WARNING: unknown regression trend selected, 'constant term' has been selected by default.\n")
            println (s"  Possible trends for this test are $validTrends")
        else if trend == "c" then
            trendType = "constant"; npar = 2
        else if trend == "nc" then
            trendType = "no constant"; npar = 1
        else if trend == "ct" then
            trendType = "constant trend"; npar = 3
        else if trend == "ctt" then
            trendType = "quadratic trend"; npar = 4
        println (s"setTrend: trend = $trend, npar = $npar")

        if trend != prevTrend then
            newTest   = true
            newTrend  = true
            prevTrend = trend
    end setTrend

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute critical value from probabilities.
     */
    private def computeCV (): Unit =
        val n = nobs - lags - 1                                        // computing adjusted number of observations

        for i <- probas.indices do
            criticalVals(i) = 0                                      // computing critical value
            val n0 = coeff.getOrElse (probas(i), null).getOrElse (0, null).size
            for j <- 0 until n0 do
                criticalVals(i) += coeff.getOrElse (probas(i), null).getOrElse (0, null)(j) / (n~^j)
            end for

            val n1 = coeff.getOrElse (probas(i), null).getOrElse (1, null).size
            for j <- 0 until n1 do
                criticalVals(i) += coeff.getOrElse (probas(i), null).getOrElse (1, null)(j) * ((lags.toDouble/n)~^(j+1))
            end for
        end for
    end computeCV

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute p-value by linear interpolation from critical values.
     */
    private def computePval (stat: Double): Unit =
        if stat <= criticalVals(0) then pval = probas(0)             // if stat <= critical value for first probability (in abs value)
        else
            breakable {
                for i <- 1 until probas.length if stat <= criticalVals(i) do
                    pval = probas(i-1) + (stat - criticalVals(i-1)) * (probas(i) - probas(i-1)) / (criticalVals(i) - criticalVals(i-1))
                    break ()
                end for
            } // breakable
        end if
        if stat > criticalVals.last then pval = probas.last          // if stat > critical value for last probability (in abs value)
    end computePval

end UnitRoot


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `makeTSeries` top level function generates time-series data.
 *  @param signal  the function of time used to make the deterministic part
 *  @param m       the length of the time series
 *  @param noise   the random variate generator used for the noise part
 */
def makeTSeries (signal: FunctionS2S = (t: Double) => 100 + 40 * (t-1) - (t-2) * (t-2),
                 m: Int = 50, noise: Variate = Normal (0.0, 10000.0)): VectorD =
    VectorD (for t <- 0 until m yield signal (t) + noise.gen)
end makeTSeries


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `makeTSeries` top level function recursively generates time-series data
 *  by simulating and AR process.
 *      y_t+1 = δ + Σ(φ_j y_t-j) + e_t+1
 *  Note: all defaults generates white noise with variance 1
 *  @param c      the initial value for the time series
 *  @param φ      the auto-regressive coefficients
 *  @param m      the length of the time series
 *  @param noise  the random variate generator used for the noise part
 */
def makeTSeriesR (c: Double = 0.0, φ: VectorD = new VectorD (0), m: Int = 50,
                  noise: Variate = Normal ()): VectorD =
    val y = new VectorD (m)
    y(0)  = c
    for t <- 0 until m-1 do
        var sum = 0.0
        for j <- φ.indices do
            sum += φ(j) * y(max (0, t-j))
        end for
        y(t+1) = sum + noise.gen
    end for
    y
end makeTSeriesR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `stationaryTest` main function tests the `Stationary` class on a simulated
 *  time-series.
 *  > runMain scalation.modeling.forecasting.stationaryTest
 */
@main def stationaryTest (): Unit =

    val y = makeTSeries ()

    banner ("Test Stationary on simulated time-series")
    val stats = Stats4TS (y, MAX_LAGS)
    println (stats)
    val zero = new VectorD (stats.acr.dim)
    new Plot (null, stats.acr, zero, "ACF vs. k", true)

    new Plot (null, y, null, "simulated time series", lines = true)

end stationaryTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `stationaryTest2` main function tests the `Stationary` class on a simulated
 *  stationary time-series.  An AR(1) is a stationary process when |φ_1| < 1, 
 *  a unit root process when |φ_1| = 1, and explosive otherwise.
 *  > runMain scalation.modeling.forecasting.stationaryTest2
 */
@main def stationaryTest2 (): Unit =

    val rates = Array (0.99, 1.0, 1.01)

    for i <- rates.indices do
        val φ = VectorD (rates (i))
        val y = makeTSeriesR (0, φ, 2000)

        banner (s"Test Stationary on simulated stationary time-series with rate = ${rates(i)}")
        val stats = Stats4TS (y, MAX_LAGS)
        println (stats)
        val zero = new VectorD (stats.acr.dim)
        new Plot (null, stats.acr, zero, s"ACF vs. k for rate = ${rates(i)}", true)
        new Plot (null, y, null, s"simulated time series with rate = ${rates(i)}", lines = true)
    end for

end stationaryTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `stationaryTest3` main function tests the `Stationary` class on a simulated
 *  stationary time-series. An AR(2) is a stationary process when |φ_2| < 1 and |φ_1| < 1 - φ_2,
 *  a unit root process when |φ_2| = 1 or |φ_1| = 1 - φ_2, and explosive otherwise.
 *  > runMain scalation.modeling.forecasting.stationaryTest3
 */
@main def stationaryTest3 (): Unit =

    val rates = Array (Array ((0.49, 0.49), (0.50, 0.50), (0.51, 0.51)),
                       Array ((0.01, 0.98), (0.01, 0.99), (0.01, 1.00)),
                       Array ((1.98, -0.99), (1.99, -0.99), (2.00, -0.99)))

    for i <- rates.indices; j <- rates(0).indices do
        val φ = VectorD (rates (i)(j)._1, rates(i)(j)._2)
        val y = makeTSeriesR (0, φ, 2000)

        banner (s"Test Stationary on simulated stationary time-series with rate = ${rates(i)(j)}")
        val stats = Stats4TS (y, MAX_LAGS)
        println (stats)
//      val zero = new VectorD (stats.acr.dim)
//      new Plot (null, stats.acr, zero, s"ACF vs. k for rate = ${rates(i)(j)}", true)
        new Plot (null, y, null, s"simulated time series with rate = ${rates(i)(j)}", lines = true)
    end for

end stationaryTest3

