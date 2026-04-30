
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Santosh Uttam Bobade, John Miller
 *  @version 2.0
 *  @date    Sat Jun 9 14:09:25 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Outlier Detection and Handling
 *
 *  @see Ch. 5 in Computational Data Science
 */

package scalation
package modeling

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Outlier` trait specifies an outlier detection operation to be defined by
 *  the objects implementing it, i.e.,
 *      `DistanceOutlier`  - outlier = beyond 'STDEV_CUTOFF' units from mean
 *      `QuantileOutlier`  - outlier = in the 'PERCENTILE' tails of the distribution
 *      `QuartileOutlier`  - outlier = 'X_MULTIPLIER' times beyond the middle two quartiles
 *  Leaving extreme values in datasets that are highly unlikely to represent legitimate
 *  values will reduce the quality of models.  However, removing legitimate extreme values
 *  will only make the model appear to be good, and it may fail in the real world.
 *  @see `Imputation` as an alternative to removal of outliers
 */
trait Outlier:

    protected val flaw = flawf ("Outlier")                                  // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the lower and upper bound for acceptable values for vector y.
     *  @param y       the vector with the possible outlier values
     *  @param factor  the factor used in computing the bound (method dependent)
     */
    def calcBounds (y: VectorD, factor: Double): (Double, Double)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find/detect all outliers in vector y outside the bounds and return their element indices.
     *  @param y       the vector with the possible outlier values
     *  @param bounds  the acceptable lower and upper bounds for element values
     */
    def findOutliers (y: VectorD, bounds: (Double, Double)): Set [Int] =
        (for i <- y.indices if y(i) <= bounds._1 || y(i) >= bounds._2  yield i).toSet
    end findOutliers

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove all outliers from vector y specified by indices in toRemove.
     *  @param y         the vector with the possible outlier values
     *  @param toRemove  the indices of elements to be removed
     */
    def removeOutliers (y: VectorD, toRemove: Set [Int]): VectorD =
        val yr = new VectorD (y.dim - toRemove.size)
        var k  = 0
        for i <- y.indices if ! (toRemove contains i) do { yr(k) = y(i); k += 1 } 
        yr
    end removeOutliers

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove all outliers from matrix x and vector y specified by indices in toRemove.
     *  @param x         the predictor matrix: y = f(x)
     *  @param y         the vector with the possible outlier values
     *  @param toRemove  the indices of elements to be removed
     */
    def removeOutliers (x: MatrixD, y: VectorD, toRemove: Set [Int]): (MatrixD, VectorD) =
        if x.dim != y.dim then flaw ("removeOutliers", s"x.dim = ${x.dim} != y.dim = ${y.dim}")
        val mr = y.dim - toRemove.size
        val xr = new MatrixD (mr, x.dim2)
        val yr = new VectorD (mr)
        var k  = 0
        for i <- y.indices if ! (toRemove contains i) do { xr(k) = x(i); yr(k) = y(i); k += 1 }
        (xr, yr)
    end removeOutliers

end Outlier


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Detect outliers in the vector by treating anything that falls outside of some distance
 *  from the mean as an outlier.  Common number of standard deviation units are 2.5, 2.7, 3 and 3.5.
 *  The larger the dataset, the greater the number of units that should be used.
 */
object DistanceOutlier extends Outlier:

    private val STDEV_CUTOFF = 2.7                                          // default stdev cutoff value
                                                                            // aligned with IQR @ 1.5

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the lower and upper bound for acceptable values for vector y.
     *  Treat anything that falls outside of some factor standard deviation units
     *  from the mean as an outlier.
     *  @param y       the vector with the possible outlier values
     *  @param factor  the factor used in computing the bound (# std deviations)
     */
    def calcBounds (y: VectorD, factor: Double = STDEV_CUTOFF): (Double, Double) =
        val (mu, sig) = (y.mean, y.stdev)
        (mu - factor * sig, mu + factor * sig)
    end calcBounds
        
end DistanceOutlier


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Detect outliers in the vector by treating anything that falls below the 1st Quartile
 *  or above the 3rd Quartile as an Outlier.  Common values for X_MULTIPLIER are 1.5 and 2.0.
 */
object QuartileXOutlier extends Outlier:

    private val QUARTILE      = 0.25                                        // value for quartile as fraction
    private val X_MULTIPLIER  = 1.5                                         // default eXpansion multiplier

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the lower and upper bound for acceptable values for vector y.
     *  Treat anything that falls X times below the 1st Quartile or above the 3rd Quartile
     *  as an Outlier.
     *  @param y       the vector with the possible outlier values
     *  @param factor  the factor used in computing the bound (multiple of IRQ)
     */
    def calcBounds (y: VectorD, factor: Double = X_MULTIPLIER): (Double, Double) =
        val (q1, q3) = (y.quantile (QUARTILE), y.quantile (1 - QUARTILE))
        val iqr      = q3 - q1                                              // inter-quartile range
        (q1 - factor * iqr, q3 + factor * iqr)
    end calcBounds

end QuartileXOutlier


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Detect outliers in the vector by treating anything that falls outside 1-st or
 *  99-th percentile.  Common percentiles that may be passed in factor are .0035, .005, .01, .02, and .05.
 *  Note, extreme 2% as discussed in textbook corresponds to 1% in left tail and 1% in right tail.
 */
object QuantileOutlier extends Outlier:

    private val PERCENTILE  = 0.0035                                        // default percentile as fraction
                                                                            // aligned with IQR @ 1.5

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the lower and upper bound for acceptable values for vector y.
     *  Treat anything that falls below the factor percentile or above the (1 - factor)
     *  percentile as an outlier.
     *  @param y       the vector with the possible outlier values
     *  @param factor  the factor used in computing the bound (percentile)
     */
    def calcBounds (y: VectorD, factor: Double = PERCENTILE): (Double, Double) =
        (y.quantile (factor), y.quantile (1 - factor))
    end calcBounds

end QuantileOutlier


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `outlierTest` main function is used to test the Outliers Detection techniques
 *  presented in the `Outliers` trait for a vector of doubles.
 *  > runMain scalation.modeling.outlierTest
 */
@main def outlierTest (): Unit =

    val y = VectorD (10, 20, 30, 40, 50, 2, 3, 4, 5, 6, 7, 8, -50.0, 5, 5, 5)

    banner ("Standard Deviation Method: DistanceOutlier")
    println (s"Original: $y")
    var bounds = DistanceOutlier.calcBounds (y)
    var idx    = DistanceOutlier.findOutliers (y, bounds)
    var yr     = DistanceOutlier.removeOutliers (y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      $idx")
    println (s"After:    $yr")

    banner ("IQR Method: QuartileXOutlier")
    println (s"Original: $y")
    bounds = QuartileXOutlier.calcBounds (y)
    idx    = QuartileXOutlier.findOutliers (y, bounds)
    yr     = QuartileXOutlier.removeOutliers (y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      $idx")
    println (s"After:    $yr")

    banner ("Percentile Method: QuantileOutlier")
    println (s"Original: $y")
    bounds = QuantileOutlier.calcBounds (y)
    idx    = QuantileOutlier.findOutliers (y, bounds)
    yr     = QuantileOutlier.removeOutliers (y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      $idx")
    println (s"After:    $yr")

end outlierTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `outlierTest2` main function is used to test the Outliers Detection techniques
 *  presented in the `Outliers` trait for a matrix and vector doubles.
 *  > runMain scalation.modeling.outlierTest2
 */
@main def outlierTest2 (): Unit =

    val x = MatrixD ((16, 1), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
    val y = VectorD (10, 20, 30, 40, 50, 2, 3, 4, 5, 6, 7, 8, -50.0, 5, 5, 5)

    banner ("Standard Deviation Method: DistanceOutlier")
    println (s"Original: $x")
    println (s"Original: $y")
    var bounds = DistanceOutlier.calcBounds (y)
    var idx    = DistanceOutlier.findOutliers (y, bounds)
    var xr_yr  = DistanceOutlier.removeOutliers (x, y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      $idx")
    println (s"After:    $xr_yr")

    banner ("IQR Method: QuartileXOutlier")
    println (s"Original: $x")
    println (s"Original: $y")
    bounds = QuartileXOutlier.calcBounds (y)
    idx    = QuartileXOutlier.findOutliers (y, bounds)
    xr_yr  = QuartileXOutlier.removeOutliers (x, y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      $idx")
    println (s"After:    $xr_yr")

    banner ("Percentile Method: QuantileOutlier")
    println (s"Original: $x")
    println (s"Original: $y")
    bounds = QuantileOutlier.calcBounds (y)
    idx    = QuantileOutlier.findOutliers (y, bounds)
    xr_yr  = QuantileOutlier.removeOutliers (x, y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      $idx")
    println (s"After:    $xr_yr")

end outlierTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `outlierTest3` main function is used to test the Outliers Detection techniques
 *  presented in the `Outliers` trait for a matrix and vector doubles.
 *  > runMain scalation.modeling.outlierTest3
 */
@main def outlierTest3 (): Unit = 

    import scalation.random.Normal

    val normal = Normal ()
    val y      = VectorD (for _ <- 0 until 10000 yield normal.gen)

    banner ("Standard Deviation Method: DistanceOutlier")
    var bounds = DistanceOutlier.calcBounds (y)
    var idx    = DistanceOutlier.findOutliers (y, bounds)
    var yr     = DistanceOutlier.removeOutliers (y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      ${idx.size}")
    println (s"After:    ${yr.dim}")

    banner ("IQR Method: QuartileXOutlier")
    bounds = QuartileXOutlier.calcBounds (y)
    idx    = QuartileXOutlier.findOutliers (y, bounds)
    yr     = QuartileXOutlier.removeOutliers (y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      ${idx.size}")
    println (s"After:    ${yr.dim}")

    banner ("Percentile Method: QuantileOutlier")
    bounds = QuantileOutlier.calcBounds (y)
    idx    = QuantileOutlier.findOutliers (y, bounds)
    yr     = QuantileOutlier.removeOutliers (y, idx)
    println (s"bounds:   $bounds")
    println (s"idx:      ${idx.size}")
    println (s"After:    ${yr.dim}")

end outlierTest3

