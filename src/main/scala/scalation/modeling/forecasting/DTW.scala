
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Aug 26 13:42:17 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Tool: Dynamics Time Warping (DTW)
 */

package scalation
package modeling
package forecasting

import scala.math.abs

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DTW` class is used for aligning two time series.
 *  @param y   the first time series vector
 *  @param q   use the L_q norm (defaults to 2 (Euclidean))
 *  @param ty  the corresponding date-time vector
 */
class DTW (y: VectorD, q: Int = 2)(ty: VectorT =  VectorT.range (0, y.dim)):

    private val debug = debugf ("DTW", true)                              // debug function

    private val inf = POSITIVE_INFINITY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the difference between a and b.
     *  @param a  the first value
     *  @param b  the second value
     */
    def d (a: Double, b: Double): Double = a - b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an alignment score for the two time series y and z.
     *  Translated from @see rtavenar.github.io/blog/dtw.html
     *  Note that the 3 elements in VectorD cannot all be infinity if we have (i > 0 or j > 0)
     *  @param z  the second time series vector
     *  @param w  the size of the locally window (to limit the index span between index i and j)
     */
    def dtw_score (z: VectorD, w: Int = y.dim): Double =
        val r = new MatrixD (y.dim, z.dim)                                // warped distance
        for i <- y.indices; j <- z.indices do
            r(i, j) = d(y(i), z(j)) ~^ q
            if (i > 0 || j > 0) && abs (i - j) <= w then
               r(i, j) += VectorD (
                   if i > 0 then r(i-1, j) else inf,
                   if j > 0 then r(i, j-1) else inf,
                   if i > 0 && j > 0 then r(i-1, j-1) else inf).min
        end for
        r.last ~^ (1.0 / q)
    end dtw_score

// FIX -- need to get the warping path

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Align the two time series.
     *  @param z   the second time series vector
     *  @param tz  the corresponding date-time vector
     */
    def align (z: VectorD)(tz: VectorT = VectorT.range (0 until z.dim)):
              (VectorD, VectorD) = 
        debug ("align", s"y = $y @ ty = $ty \n z = $z @ tz = $tz") 
        (null, null)
    end align

// FIX -- need a method to get an optimal global time shift

end DTW


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dTWTest` main function tests the `DTW` class on real data.
 *  > runMain scalation.modeling.forecasting.dTWTest
 */
@main def dTWTest (): Unit =

    import Example_Covid.{clip, loadData_yy}

    val vars = Array ("new_deaths", "icu_patients")
    val (yy, _) = clip ((loadData_yy (vars), null))
    val y0 = yy(?, 0)                                                     // column 0
    val y1 = yy(?, 1)                                                     // column 1

    val warp = DTW (y0)()                                                 // apply DTW to columns 0 and 1
    val score = warp.dtw_score (y1) 
    println (s"score    = $score")                                        // DTW distance score
    println (s"distance = ${(y0 - y1).norm}")                             // Euclidean distance
    warp.align (y1)()

end dTWTest

