
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Forecast Tensor time x horizons x variable
 *                                            row    column     sheet
 */

package scalation
package modeling
package forecasting
package multivar

import scalation.mathstat._

/*----------------------------------------------------------------------------

The FORECASTING TENSOR yf: Example Values (made up (e.g., daily) values)

yf(t, h, j) = element for base time t, horizon h, and variable j (its time = t+h)
yf(?, ?, j) shows sheet j that corresponds to a forecast matrix for variable j

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
/** The `ForecastTensor` trait provides a common framework for holding forecasts
 *  over time and multiple horizons.
 *  @param y     the multi-variate time series data matrix [y_tj]
 *  @param hh    the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng  the time vector, if relevant (index as time may suffice)
 */
trait ForecastTensor (y: MatrixD, hh: Int, tRng: Range = null):

    private val debug = debugf ("ForecastTensor", true)                    // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make the full FORECAST TENSOR where (for each sheet j) the zeroth column holds
     *  the actual time series and the last column is its time/time index.
     *  Columns 1, 2, ... hh are for h steps ahead forecasts.
     *  @param y_  the actual multi-variate times values to use in making forecasts
     *  @param hh  the maximum forecasting horizon, number of steps ahead to produce forecasts
     */
    def makeForecastTensor (y_ : MatrixD = y, hh_ : Int = hh): TensorD =
        val yf_ = new TensorD (y_.dim, hh_ + 2, y_.dim2)                    // forecasts for all time points t & horizons to h
        debug ("makeForecastTensor", s"forecast tensor: y_.dim = ${y_.dim} --> yf_.dims = ${yf_.dims}")

        for j <- y_.indices2 do                                            // for each variable
            for t <- y_.indices do yf_(t, 0, j) = y_(t, j)                 // first column (0) holds the actual time series values
            if tRng == null then
                for t <- yf_.indices do yf_(t, hh_ + 1, j) = t             // last column (h+1) holds time (logical day)
            else 
                for t <- tRng do yf_(t, hh_ + 1, j) = t                    // last column (h+1) holds time (logical day)
        end for
        yf_
    end makeForecastTensor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the diagonalized version of the forecst tensor, i.e., for each of
     *  the j-th forecast matrices each row is at a fixed time point and, for example,
     *  the random walk model simply pushes the values down diagonals.
     *  Note 'unshiftDiag' reverses the process.
     *  @param yf  the current forecast tensor
     */
    def diagonalize (yf: TensorD): TensorD =
        val yf2 = new TensorD (yf.dim, yf.dim2, yf.dim3)
        for j <- yf.indices3 do                                            // for each variable j
            yf2(?, ?, j) = yf(?, ?, j).shiftDiag                                          // shift j-th sheet
        yf2
    end diagonalize

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
     *  @param y_    the actual multi-variate time series matrix to use in making forecasts
     *  @param yf    the entire FORECAST TENSOR
     *  @param rRng  the time range, defaults to null (=> full time range)
     *  @param sft   the amount of shift for yfh (FIX - ideally unify the code and remove sft)
     */
    def diagnoseAll (y_ : MatrixD, yf: TensorD, tRng: Range = null, sft: Int = 0): Unit =
        val ftMat = new TensorD (hh, Fit.N_QoF, yf.dim3)
        val t1 = if tRng == null then 0 else tRng.start                    // first time point
        for j <- yf.indices3; h <- 1 to hh do
            val yy  = y_(t1+h until y_.dim, j)                             // align the actual response values
            val yfh = yf(?, h, j)(t1+sft until y_.dim-h+sft)               // align column h of the forecast matrix
            println (s"yy.dim = ${yy.dim}, yfh.dim = ${yfh.dim}")
//          println (s"for h = $h: ${MatrixD (yy, yfh).transpose}")
//          Forecaster.differ (yy, yfh)                                    // uncomment for debugging
            assert (yy.dim == yfh.dim)                                     // make sure the vector sizes agree

            mod_resetDF (yy.dim)                                           // reset the Degrees of Freedom (DF)  
            val qof = diagnose (yy, yfh)                                   // use column h of yf
            ftMat(h-1, ?, j) = qof
//          println (FitM.fitMap (qof, qoF_names))
        end for
        for j <- yf.indices3 do
            println (s"fitMap QoF for variable $j = ")
            println (Fit.showFitMap (ftMat (?, ?, j).ᵀ))
        end for
    end diagnoseAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  for all horizons and print the results in a table.
     *  For time series, the first few predictions use only part of the model, so may be skipped.
     *  The version is for models that perform DIRECT multi-horizon forecasting.
     *  @param yy  the actual response/output matrix over all horizons
     *  @param yf  the entire FORECAST TENSOR
     */
    def diagnoseAll (yy: TensorD, yf: TensorD): Unit =
        val ftMat = new MatrixD (hh, Fit.N_QoF)
        for h <- 1 to hh do
            val qof    = diagnose (yy(?, h-1), yf(?, h))                   // use column h of yf
            ftMat(h-1) = qof
//          println (FitM.fitMap (qof, qoF_names))
        end for
        println ("fitMap QoF = ")
        println (Fit.showFitMap (ftMat.ᵀ))
    end diagnoseAll

end ForecastTensor

import Example_Covid.loadData_yy

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `forecastTensorTest3` main function tests the `RandomWalk` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting2.forecastTensorTest
 */
@main def forecastTensorTest (): Unit =

    val vars = Array ("new_deaths", "icu_patients")
    val yy = loadData_yy (vars)
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = RandomWalk_Star (y, hh)                                     // create model for time series data
    for j <- mod.indices do
        banner (s"In-ST Forecasts: ${mod(j).modelName} for var $j on COVID-19 Dataset")
        mod(j).trainNtest ()()                                            // train and test on full dataset

        mod(j).forecastAll ()                                             // forecast h-steps ahead (h = 1 to hh) for all y
        mod(j).diagnoseAll (y, mod(j).getYf)                              // should agree with evalForecasts
//      Forecaster.evalForecasts (mod(j), mod(j).getYb, hh)
        println (s"Final In-ST Forecast Tensor yf = ${mod(j).getYf}")
    end for

end forecastTensorTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `forecastTensorTest2` main function tests the `RandomWalk` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting2.forecastTensorTest2
 */
@main def forecastTensorTest2 (): Unit =

    val vars = Array ("new_deaths", "icu_patients")
    val yy = loadData_yy (vars)
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = RandomWalk_Star (y, hh)
    for j <- mod.indices do
        banner (s"TnT Forecasts: ${mod(j).modelName} for var $j on COVID-19 Dataset")
        mod(j).trainNtest ()()

        mod(j).rollValidate ()                                            // TnT with Rolling Validation
        banner ("diagnoseAll")
        mod(j).diagnoseAll (y(?, j), mod(j).getYf, Forecaster.teRng (y.dim))    // only diagnose on the testing set
        println (s"Final TnT Forecast Tensor yf = ${mod(j).getYf}")
    end for

end forecastTensorTest2

