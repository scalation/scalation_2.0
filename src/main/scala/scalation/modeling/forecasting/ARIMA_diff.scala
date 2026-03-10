
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, John Miller
 *  @version 2.0
 *  @date    Sat Jun 13 01:27:00 EST 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Differencing methods for ARIMA models.
 */
 
package scalation
package modeling
package forecasting

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARIMA_diff` object provides methods for taking first and second order
 *  differences, as well as transforming back to the original scale.
 *
 *  diff:              position y  --> velocity v  --> acceleration a    (actual)
 *                              |               |                   |
 *  backform, undiff:  position yp <-- velocity vp <-- acceleration ap   (predicted)
 *
 *  @see www.jstatsoft.org/article/view/v027i03/v27i03.pdf
 *  @see stats.stackexchange.com/questions/32634/difference-time-series-before-arima-or-within-arima
 */
object ARIMA_diff:

    private val debug = debugf ("ARIMA_diff", true)                  // debug function
    private val flaw  = flawf ("ARIMA_diff")                         // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Take the d'th difference of the position time series for d in {0, 1, 2}.
     *  A new vector (of length y.dim-d) is returned even when there is no difference
     *  taken (d = 0), to ensure the original is preserved.
     *  E.g., for d = 1:  Position y --> Velocity v.
     *  @param y  the actual position time series to be differenced
     *  @param d  the order of simple differencing (defaults to 1)
     */
    def diff (y: VectorD, d: Int = 1): VectorD =
        debug ("diff", s"y.dim = ${y.dim}, d = $d")
        d match
        case 0 => y.copy
        case 1 => VectorD (for i <- 0 until y.dim-1 yield y(i+1) - y(i))
        case 2 => VectorD (for i <- 0 until y.dim-2 yield y(i+2) - 2*y(i+1) + y(i))
        case _ => flaw ("diff", s"does not support differencing higher than 2"); null
    end diff

    inline def Δ (y: VectorD): VectorD = diff (y)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Undifference the velocity time series by adding the difference to the previous value.
     *  Velocity v -> Position y.
     *  @param v   the differenced time series (velocity)
     *  @param y0  the first value in the original time series
     */
    def undiff (v: VectorD, y0: Double): VectorD =
        debug ("undiff", s"v.dim = ${v.dim}, y0 = $y0")
        val y = new VectorD (v.dim + 1)
        y(0)  = y0
        for t <- 1 until y.dim do y(t) = v(t-1) + y(t-1)
        y
    end undiff

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backform by transforming back the predicted values of a differenced time series
     *  to the original scale using actual values y.
     *  E.g., for d = 1:  Velocity vp -> Position yp.
     *  @param vp  the predicted differenced (velocity/acceleration) time series
     *  @param y   the actual position time series vector (first d values needed)
     *  @param d   the order of simple differencing (defaults to 1)
     */
    def backform (vp: VectorD, y: VectorD, d: Int = 1): VectorD =
        debug ("backform", s"vp.dim = ${vp.dim}, y.dim = ${y.dim}, d = $d")
        val yp = new VectorD (y.dim)
        d match
        case 0 => vp
        case 1 => yp(0) = y(0)
                  for t <- 0 until y.dim-1 do yp(t+1) = vp(t) + y(t)
                  yp
        case 2 => yp(0) = y(0); yp(1) = y(1)
                  for t <- 0 until y.dim-2 do yp(t+2) = vp(t) + 2*y(t+1) - y(t)
                  yp
        case _ => flaw ("backform", "does not support differencing higher than 2"); null
    end backform

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transform the forecast values in the FORECAST MATRIX of a differenced time series
     *  back to the original scale for all horizons (1 to h).
     *  @param vf  the matrix of all multi-horizon forecasted values (differenced)
     *  @param y   the original actual time series vector (undifferenced)
     *  @param d   the order of simple differencing
     */
    def transformBack (vf: MatrixD, y: VectorD, d: Int): MatrixD =
        val h    = vf.dim2 - 2                                           
        val yy   = y ++ VectorD(0)
        val yf   = new MatrixD (vf.dim, vf.dim2)
        yf(?, 0) = yy
        for k <- 1 to h do yf(?, k) = backform (vf(?, k), yy, d)
        yf(?, h+1) = VectorD.range (0 until vf.dim)
        yf
    end transformBack

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transform the forecast values from time point t of a differenced time series back
     *  to the original scale for all horizons (1 to h).
     *  @param vh  the vector of forecasted differenced values for times t+1, ... t+h
     *  @param y   the original actual time series vector (undifference)
     *  @param d   the order of simple differencing
     *  @param t   the time point being forecasted (@see the `forecast` method)
     */
    def transformBack (vh: VectorD, y: VectorD, d: Int, t: Int): VectorD =
        d match
        case 0 => vh
        case 1 => val yh = y(t - 1 to t) ++ vh
                  for i <- 1 until yh.dim do yh(i) += yh(i-1)
                  yh(1 to yh.dim)
        case 2 => val yh = y(t-2 to t) ++ vh
                  for i <- 2 until yh.dim do yh(i) += (2*yh(i-1) - yh(i-2))
                  yh(2 to yh.dim)
        case _ => flaw ("transformBack", "does not support differencing higher than 2"); null
    end transformBack

end ARIMA_diff

import ARIMA_diff._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRIMA_diffTest` main function tests the `ARIMA_diff` object on real data:
 *  Forecasting lake levels comparing ARMA, AR1MA, Differenced ARMA, Transformed-Back
 *  Differenced ARMA.  Observe that `backform` is better than `undiff` on predictions.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 * > runMain scalation.modeling.forecasting.aRIMA_diffTest
 */
@main def aRIMA_diffTest (): Unit =

    import Example_LakeLevels.y
    import AR.hp
    val tf = new TestFit (y.dim)
    val hh = 2

    banner ("Test ARMA (2, 0) on Lake Level Dataset")
    hp("p") = 2; hp("q") = 0
    val yp = new ARMA (y, hh).trainNtest ()()._1
    new Plot (null, y, yp, "ARMA y and yp vs. time", lines = true)

    banner ("Test Differenced ARMA (2, 0) on Lake Level Dataset")
    val v  = diff (y)                                               // first difference on y (size of v one less than y)
    val yy = undiff (v, y(0))                                       // reverse the diff
    Forecaster.differ (y, yy)                                       // verify recovery of original time series
    val vp_ = new ARMA (v, hh).trainNtest ()()._1                   // predictions skip the first value (no past)
//  val vp = v(0) +: vp_                                            // prepend the first actual value (want same size as v)
    val vp = vp_                                                    // prepend the first actual value (want same size as v)

    banner ("Test Transformed-Back Differenced ARMA (2, 0) on Lake Level Dataset")
    println (s"predictAll: y.dim = ${y.dim}, vp.dim = ${vp.dim}")
    val yp1 = undiff (vp, y(0))                                     // transform vp back to original (y) scale using undiff
    val yp2 = backform (vp, y)                                      // transform vp back to original (y) scale using backform
    
    banner ("Test Transformed-Back using undiff")
    println (tf.testDiagnose (y, yp1))                              // determine the quality of fit for yp1
    banner ("Test Transformed-Back using backform")
    println (tf.testDiagnose (y, yp2))                              // determine the quality of fit for yp2
    new Plot (null, y, yp1, "undiff: y and yp1 vs. time", lines = true)
    new Plot (null, y, yp2, "backform: y and yp2 vs. time", lines = true)
    new Plot (null, yp1, yp2, "yp1 and yp2 vs. time", lines = true)

/*
    banner ("Test AR1MA (2, 0) on Lake Level Dataset")              // AR1MA automatically takes first differences
    val mod = new AR1MA (y)
    val (vp3_, qof3) = mod.trainNtest ()()
    val vp3 = v(0) +: vp3_                                          // prepend the first actual value (want same size as v)
    val yp3 = backform (vp3, y)                                     // transform vp back to original (y) scale using backform
    println (tf.testDiagnose (y, yp3))                              // determine the quality of fit for yp3
    new Plot (null, y, yp3, "y and yp3 vs. time", lines = true)
*/
    
end aRIMA_diffTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRIMA_diffTest2` main function tests the `ARIMA_diff` object on real data:
 *  Forecasting Covid-19 new deaths comparing ARMA, AR1MA, Differenced ARMA, Transformed-Back
 *  Differenced ARMA.  Observe that `backform` is better than `undiff` on predictions.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 * > runMain scalation.modeling.forecasting.aRIMA_diffTest2
 */
@main def aRIMA_diffTest2 (): Unit =

    import Example_Covid.y
    import AR.hp
    val tf = new TestFit (y.dim)
    val hh = 6

    banner ("Test ARMA (2, 0) on Covid-19 Dataset")
    hp("p") = 2; hp("q") = 0
    val yp = new ARMA (y, hh).trainNtest ()()._1
    new Plot (null, y, yp, "ARMA y and yp vs. time", lines = true)

    banner ("Test Differenced ARMA (2, 0) on Covid-19 Dataset")
    val v  = diff (y)                                               // first difference on y (size of v one less than y)
    val yy = undiff (v, y(0))                                       // reverse the diff
    Forecaster.differ (y, yy)                                       // verify recovery of original time series
    val vp_ = new ARMA (v, hh).trainNtest ()()._1                   // predictions skip the first value (no past)
//  val vp = v(0) +: vp_                                            // prepend the first actual value (want same size as v)
    val vp = vp_                                                    // prepend the first actual value (want same size as v)

    banner ("Test Transformed-Back Differenced ARMA (2, 0) on Covid-19 Dataset")
    println (s"predictAll: y.dim = ${y.dim}, vp.dim = ${vp.dim}")
    val yp1 = undiff (vp, y(0))                                     // transform vp back to original (y) scale using undiff
    val yp2 = backform (vp, y)                                      // transform vp back to original (y) scale using backform

    banner ("Test Transformed-Back using undiff")
    println (tf.testDiagnose (y, yp1))                              // determine the quality of fit for yp1
    banner ("Test Transformed-Back using backform")
    println (tf.testDiagnose (y, yp2))                              // determine the quality of fit for yp2
    new Plot (null, y, yp1, "undiff: y and yp1 vs. time", lines = true)
    new Plot (null, y, yp2, "backform: y and yp2 vs. time", lines = true)
    new Plot (null, yp1, yp2, "yp1 and yp2 vs. time", lines = true)

end aRIMA_diffTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRIMA_diffTest3` main function tests the `ARIMA_diff` object on real data:
 *  Forecasting Covid-19 new deaths examining phase spece plots.
 * > runMain scalation.modeling.forecasting.aRIMA_diffTest3
 */
@main def aRIMA_diffTest3 (): Unit =

    import Example_Covid.y                                          // position: level of the time series

    val v  = diff (y)                                               // velocity: first difference on y (size of v one less than y)

    new Plot (y, v, null, "Phase Space: Velocity vs. Position", lines = true)
    new Plot (null, y, v, "Phase Space: Velocity, Position vs. Time", lines = true)

    // Standarize

    val ys = y.standardize
    val vs = v.standardize

    new Plot (ys, vs, null, "Phase Space: Standardized Velocity vs. Position", lines = true)
    new Plot (null, ys, vs, "Phase Space: Standardized Velocity, Position vs. Time", lines = true)

    // Log transformation gives growth rates

    val u = diff (y.log)
    val us = u.standardize

    new Plot (ys, us, null, "Phase Space: Standardized Growth-Rate vs. Position", lines = true)
    new Plot (null, ys, us, "Phase Space: Standardized Growth-Rate, Position vs. Time", lines = true)

end aRIMA_diffTest3

