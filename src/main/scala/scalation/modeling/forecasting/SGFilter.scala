
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu May 22 01:21:46 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Smoother: Savitzky–Golay Filter
 *
 *  @see en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

import Example_Covid.loadData_y
import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SGFilter` class provides basic time series capabilities for Savitzky–Golay
 *  filters that are used to smooth data.
 *  Note, would need to be adapted for use as a forecaster as it uses future data.
 *  @see `WeightedMovingAverage`
 *  @param y  the response vector (time series data) 
 */
class SGFilter (y: VectorD)
      extends Filter (y):

    val c = VectorD (-3, 12, 17, 12, -3) / 35.0                         // Convolution coefficients for
                                                                        // 5 point quadratic

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a smoothed version of the given time series vector.
     *
     *      z(t) = c dot y(t-2 until t+3) = c dot [ y(t-2), y(t-1), y(t), y(t+1), y(t+2) ]
     *
     *  @param y_  the actual time series values to be smoothed
     *  @param a   the smoothing parameter
     */
    def smooth (y_ : VectorD = y, a: Double = 0.0): VectorD =
        val n = y_.dim
        val z = new VectorD (n)
        z(0) = (c(2 until 5) dot y(0 until 3)) * (35.0/26.0)             // use 3 points for z(0) the first point
        z(1) = (c(1 until 5) dot y(0 until 4)) * (35.0/38.0)             // use 4 points for z(1)
        for t <- 2 until n-2 do z(t) = c dot y(t-2 until t+3)            // use 5 points, excepts first and last 2
        z(n-2) = (c(0 until 4) dot y(n-4 until n)) * (35.0/38.0)         // use 4 points for z(n-2)
        z(n-1) = (c(0 until 3) dot y(n-3 until n)) * (35.0/26.0)         // use 3 points for z(n-1) the last point
        z
    end smooth

end SGFilter


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sGFilterTest` main function tests the `SGFilter` class on real data:
 *  Smoothing Lake Levels data.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.sGFilterTest
 */
@main def sGFilterTest (): Unit =

    val filter = new SGFilter (y)                                            // create smoother for time series data
    val ys     = filter.smooth ()
    new Plot (null, y, ys, "Plot y (data) and ys (smoothed data) vs. time", lines = true)

end sGFilterTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sGFilterTest2` main function tests the `SGFilter` class on real data:
 *  Smoothing COVID-19 data.
 *  > runMain scalation.modeling.forecasting.sGFilterTest2
 */
@main def sGFilterTest2 (): Unit =

    val yy = loadData_y ()
//  val y  = yy                                                           // full
    val y  = yy(0 until 116)                                              // clip the flat end

    val filter = new SGFilter (y)                                         // create smoother for time series data
    val ys     = filter.smooth ()
    new Plot (null, y, ys, "Plot y (data) and ys (smoothed data) vs. time", lines = true)

end sGFilterTest2

