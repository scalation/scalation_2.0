
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Forecast Matrix time x horizons
 *
 *  @see     `randomWalkTest4` to see flat and slanted versions of a forecast matrix
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

/*----------------------------------------------------------------------------

The FORECAST MATRIX yf: Example Values (made up (e.g., daily) values)

yf(t, h) = element for base time t and horizon h (its time = t+h)
yf(?, h) shows column h that corresponds to a forecast vector for horizon h

y_t   h=1   h=2   t
-------------------
1.0   1.1   1.2   0    backcasted value inserted to time t = 0
2.0   2.1   2.2   1
3.0   3.1   3.2   2
4.0   4.1   4.2   3
5.0   5.1   5.2   4
6.0   6.1   6.2   5

Column h = 0: zeroth horizon forecasts are the actual (e.g., today's known) values in the time series
Column h = 1: horizon one forecasts are the one-step ahead (e.g., tomorrow's) forecasts
Column h = 2: horizon two forecasts are the two-steps ahead (e.g., day after tomorrow's) forecasts

Row time t = 3: yf(3, 0) = 4.0 = the actual value for day 3,
                yf(3, 1) = 4.1 = the one-step ahead (h=1) forecast for day 4
                yf(3, 2) = 4.2 = the two-steps ahead (h=2) forecast for day 5

----------------------------------------------------------------------------*/

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ForecastMatrix` trait provides a common framework for holding forecasts
 *  over time and multiple horizons.
 *  @param y     the response vector (time series data)
 *  @param hh    the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng  the time vector, if relevant (index as time may suffice)
 */
trait ForecastMatrix (y: VectorD, hh: Int, tRng: Range = null):

    private val debug = debugf ("ForecastMatrix", false)                   // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make the full FORECAST MATRIX where the zeroth column holds the actual time series
     *  and the last column is its time/time index.  Columns 1, 2, ... hh are for h steps
     *  ahead forecasts.
     *  @param y_   the actual time series vector to use in making forecasts
     *  @param hh_  the maximum forecasting horizon, number of steps ahead to produce forecasts
     */
    def makeForecastMatrix (y_ : VectorD = y, hh_ : Int = hh): MatrixD =
        val yf_ = new MatrixD (y_.dim, hh_ + 2)                            // forecasts for all time points t & horizons to h
        debug ("makeForecastMatrix", s"forecast matrix: y_.dim = ${y_.dim} --> yf_.dims = ${yf_.dims}")

        for t <- y_.indices do yf_(t, 0) = y_(t)                           // first column (0) holds the actual time series values
        if tRng == null then
            for t <- yf_.indices do yf_(t, hh_ + 1) = t                    // last column (h+1) holds time (logical day)
        else 
            for t <- tRng do yf_(t, hh_ + 1) = t                           // last column (h+1) holds time (logical day)
        yf_
    end makeForecastMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the slanted/diagonalized-down version of the forecast matrix where each
     *  row is at a fixed time point and, for example, the random walk model simply
     *  pushes the values down diagonals.  Note `unshiftDiag` reverses the process.
     *  Also reset the last column that holds the time index to 0, 1, 2, 3, ...
     *  @param yf  the current forecast matrix
     */
    def slant (yf: MatrixD): MatrixD =
        val yf1 = yf(?, 0) ++ new VectorD (yf.dim2-2)
        val yf2 = yf(?, 1 until yf.dim2)                                    // start shift from col 1, flat yf aleady has col 0 to col 1 shift
        val yf_ = yf1 +^: yf2.shiftDiag
        val j   = yf_.dim2 - 1
        for i <- yf_.indices do yf_(i, j) = i
        yf_
    end slant
/*
    def slant (yf: MatrixD): MatrixD =
        val yf_ = yf.shiftDiag
        val j   = yf_.dim2 - 1
        for i <- yf_.indices do yf_(i, j) = i
        yf_
    end slant
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  from the error/residual vector and the predicted & actual responses.
     *  For some models the instances may be weighted.
     *  For time series, the first few predictions use only part of the model, so may be skipped.
     *  @param y_   the actual response/output vector to use (test/full)
     *  @param yfh  the predicted response/output vector (test/full)
     *  @param w    the weights on the instances (defaults to null)
     */
    def diagnose (y_ : VectorD, yfh: VectorD, w: VectorD = null): VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Models need to provide a means for updating the Degrees of Freedom (DF).
     *  @param size  the size of dataset (full, train, or test)
     */
    def mod_resetDF (size: Int): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  for all horizons and print the results in a table.
     *  For time series, the first few predictions use only part of the model, so may be skipped.
     *  The version is for models that perform RECURSIVE multi-horizon forecasting.
     *  @param y_      the actual response/output vector
     *  @param yf      the entire FORECAST MATRIX
     *  @param rRng    the time range, defaults to null (=> full time range)
     *  @param showYf  whether to show the forecast matrix
     */
    def diagnoseAll (y_ : VectorD, yf: MatrixD, tRng: Range = null, showYf: Boolean = false): MatrixD =
        val ftMat = new MatrixD (hh, Fit.N_QoF)
        val t1 = if tRng == null then 0 else tRng.start                    // first time point
        for h <- 1 to hh do
            val yy  = y_(t1+h-1 until y_.dim)                              // align the actual response values
            val yfh = yf(?, h)(t1 until y_.dim-h+1)                        // align column h of the forecast matrix
            debug ("diagnoseAll", s"yy.dim = ${yy.dim}, yfh.dim = ${yfh.dim}")
//          println (s"for h = $h: ${MatrixD (yy, yfh).ᵀ}")
//          Forecaster.differ (yy, yfh)                                    // uncomment for debugging
            assert (yy.dim == yfh.dim)                                     // make sure the vector sizes agree

            mod_resetDF (yy.dim)                                           // reset the Degrees of Freedom (DF)  
            val qof    = diagnose (yy, yfh)                                // use column h of yf
            new Plot (null, yy, yfh, s"Plot yy vs. yfh for horizon $h", lines = true)
            ftMat(h-1) = qof
//          println (FitM.fitMap (qof, qoF_names))
        end for
        if showYf then println (s"Final Forecast Matrix yf = ${slant (yf)}")
        println ("fitMap QoF = ")
        println (Fit.showFitMap (ftMat.ᵀ))
        ftMat
    end diagnoseAll

// had  *  @param sft     the amount of shift for yfh (defaults to 0, FIX - ideally unify the code and remove sft)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  for all horizons and print the results in a table.
     *  For time series, the first few predictions use only part of the model, so may be skipped.
     *  The version is for models that perform DIRECT multi-horizon forecasting.
     *  @param yy  the actual response/output matrix over all horizons
     *  @param yf  the entire FORECAST MATRIX
     */
    def diagnoseAll (yy: MatrixD, yf: MatrixD): MatrixD =
        val ftMat = new MatrixD (hh, Fit.N_QoF)
        for h <- 1 to hh do
            val qof    = diagnose (yy(?, h-1), yf(?, h))                   // use column h of yf
            ftMat(h-1) = qof
//          println (FitM.fitMap (qof, qoF_names))
        end for
        println ("fitMap QoF = ")
        println (Fit.showFitMap (ftMat.ᵀ))
        ftMat
    end diagnoseAll

end ForecastMatrix

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `forecastMatrixTest` main function tests the `RandomWalk` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.forecastMatrixTest
 */
@main def forecastMatrixTest (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y, hh)                                      // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end forecastMatrixTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `forecastMatrixTest2` main function tests the `RandomWalk` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.forecastMatrixTest2
 */
@main def forecastMatrixTest2 (): Unit =

    val hh = 3                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y, hh)                                      // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end forecastMatrixTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `forecastMatrixTest3` main function tests the `RandomWalk` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.forecastMatrixTest3
 */
@main def forecastMatrixTest3 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y, hh)                                      // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    mod.forecastAll ()                                                    // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end forecastMatrixTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `forecastMatrixTest4` main function tests the `RandomWalk` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.forecastMatrixTest4
 */
@main def forecastMatrixTest4 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new RandomWalk (y, hh)
    banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest ()()

    mod.rollValidate ()                                                   // TnT with Rolling Validation
    banner ("diagnoseAll")
    mod.diagnoseAll (y, mod.getYf, Forecaster.teRng (y.dim))              // only diagnose on the testing set
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end forecastMatrixTest4

