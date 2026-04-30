
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu May 22 01:21:46 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Filter: Base Trait for Filters
 *
 *  @see www.motamed.nl/assets/pdf/2018_Motamed_Performance.pdf
 *  A Performance Analysis of Filtering Methods applied to WiFi-based Position Reconstruction
 *  Exponential, Moving-Average, Gaussian, Savitzky-Golay, Kalman Filters 
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Filter` trait provides basic time series capabilities for filters.
 *  A filter is used to pull out the important information from a time series.
 *  Commonly, this involves improving the signal-to-noise ratio, which is often
 *  accomplished by using a low-pass filter that remove high frequencies.  Such
 *  filters are also called smoothers (the smoothed time series has less abrupt changes)
 *  @param y  the response vector (time series data) 
 */
trait Filter (y: VectorD):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a smoothed version of the given time series vector.
     *  @param y_  the actual time series values to be smoothed
     *  @param a   the smoothing parameter
     */
    def smooth (y_ : VectorD = y, a: Double = 0.0): VectorD

end Filter

