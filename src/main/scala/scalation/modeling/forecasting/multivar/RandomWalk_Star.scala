
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep  1 23:44:49 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Array of Random Walk (RW) Models
 */

package scalation
package modeling
package forecasting
package multivar

import scala.collection.mutable.IndexedSeq

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomWalk_Star` class is used to make an array of random walk models,
 *  e.g., one for each variable in a multi-variate time series.
 *  @param y        the response/output matrix (multi-variate time series data)
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (none => use null)
 */
class RandomWalk_Star (y: MatrixD, hh: Int, fname: Array [String] = null, tRng: Range = null,
                       hparam: HyperParameter = null)
      extends Diagnoser (dfr = 1, df = y.dim - 1)
         with ForecastTensor (y, hh, tRng)
         with Forecast
         with NoSubModels:

    private val debug = debugf ("RandomWalk_Star", true)                  // debug function
    private val yf    = makeForecastTensor (y, hh)                        // make the forecast tensor

    _modelName = s"RandomWalk_Star_${y.dim2}"

    private val mod = (for j <- y.indices2 yield new RandomWalk (y(?, j), hh, tRng, hparam)).toArray

    def getY: VectorD = y(?, 0)
    def hparameter: HyperParameter = hparam
    def inSample_Test (skip: Int, showYp: Boolean): Unit = ???
    def parameter: VectorD | MatrixD = ???
    def test (x_ : MatrixD, y_ : VectorD): (VectorD, VectorD) = ???
    def validate (rando: Boolean, ratio: Double) (idx: IndexedSeq [Int]):
                 (VectorD | MatrixD, VectorD | MatrixD) = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the feature/variable names.  Overrides definition in `Forecast` trait.
     */
    override def getFname: Array [String] = fname

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test each forecasting model y_ = f(y-past) + e and report its QoF
     *  and plot its predictions.
     *  @param y_  the training/full response/output vector (defaults to full y)
     *  @param yy  the testing/full response/output vector (defaults to full y)
     */
    def trainNtest (y_ : MatrixD = y)(yy: MatrixD = y): Unit =
        for k <- mod.indices do
            mod(k).trainNtest (y_(?, k))(yy(?, k))
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For each model, forecast values for all y_.dim time points and all horizons
     *  (1 through hh-steps ahead).  Record these in the FORECAST TENSOR yf.
     *  @param y_  the actual values to use in making forecasts
     */
    def forecastAll (y_ : MatrixD = y): TensorD =
        for k <- mod.indices do
            yf(?, ?, k) = mod(k).forecastAll (y_(?, k))                   // pack forecast matrix into forecast tensor
        debug ("forecastAll", s"forecast tensor yf = $yf")
        yf
    end forecastAll

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For each model, use rolling-validation to compute test Quality of Fit (QoF) measures
     *  by dividing the dataset into a TRAINING SET (tr) and a TESTING SET (te).
     *  as follows:  [ <-- tr_size --> | <-- te_size --> ]
     *  Record the forecasted values in the FORECAST TENSOR yf.
     *  @param rc       the retraining cycle (number of forecasts until retraining occurs)
     *  @param growing  whether the training grows as it roll or kepps a fixed size
     */
    def rollValidate (rc: Int = 2, growing: Boolean = false): TensorD =
        for k <- mod.indices do
            yf(?, ?, k) = mod(k).rollValidate (rc, growing)               // pack forecast matrix into forecast tensor
        debug ("rollValidate", s"forecast tensor yf = $yf")
        yf
    end rollValidate

end RandomWalk_Star


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomWalk_Star` companion object is used to make an array of random walk models,
 *  e.g., one for each variable in a multi-variate time series.
 */
object RandomWalk_Star:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RandomWalk_Star` object consisting of several `RandomWalk` models.
     *  @param y       the multi-variate time series matrix [y_tj]
     *  @param hh      the maximum forecasting horizon (h = 1 to hh)
     *  @param fname   the feature/variable names
     *  @param tRng    the time vector, if relevant (time index may suffice)
     *  @param hparam  the hyper-parameters (defaults to null)
     */
    def apply (y: MatrixD, hh: Int, fname: Array [String] = null, tRng: Range = null,
               hparam: HyperParameter = null): Array [RandomWalk] =
        println (s"apply: create a RandomWalk model for each $fname")
        (for j <- y.indices2 yield new RandomWalk (y(?, j), hh, tRng, hparam)).toArray
    end apply

end RandomWalk_Star

import Example_Covid.loadData_yy

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalk_StarTest` main function tests the `RandomWalk_Star` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting2.multivar.randomWalk_StarTest
 */
@main def randomWalk_StarTest (): Unit =

    val vars = Array ("new_deaths", "icu_patients")
    val yy = loadData_yy (vars)
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new RandomWalk_Star (y, hh, vars)                           // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} for COVID-19 Dataset")
    mod.trainNtest ()()                                                   // train and test on full dataset

    val yf = mod.forecastAll ()                                           // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, yf)                                               // diagnose model QoF

end randomWalk_StarTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalk_StarTest2` main function tests the `RandomWalk_Star` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version explicitly uses an array of `RandomWalk` models.
 *  > runMain scalation.modeling.forecasting2.multivar.randomWalk_StarTest2
 */
@main def randomWalk_StarTest2 (): Unit =

    val vars = Array ("new_deaths", "icu_patients")
    val yy = loadData_yy (vars)
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = RandomWalk_Star (y, hh, vars)                               // use factory apply method for array of models
    for j <- mod.indices do
        banner (s"TnT Forecasts: ${mod(j).modelName} for var $j on COVID-19 Dataset")
        mod(j).trainNtest ()()                                            // train each model

        mod(j).rollValidate ()                                            // TnT with Rolling Validation
        println (s"Final TnT Forecast Matrix yf = ${mod(j).getYf}")       // print each forecast matrix
    end for

end randomWalk_StarTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWalk_StarTest3` main function tests the `RandomWalk_Star` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting2.multivar.randomWalk_StarTest3
 */
@main def randomWalk_StarTest3 (): Unit =

    val vars = Array ("new_deaths", "icu_patients")
    val yy = loadData_yy (vars)
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end
    val hh = 6                                                            // maximum forecasting horizon

    val mod = new RandomWalk_Star (y, hh, vars)                           // use constructor for multi-model
    banner (s"TnT Forecasts: ${mod.modelName} for on COVID-19 Dataset")
    mod.trainNtest ()()                                                   // train the multi-model

    val yf = mod.rollValidate ()                                          // TnT with Rolling Validation
    println (s"Final TnT Forecast Tensor yf = $yf")

end randomWalk_StarTest3

