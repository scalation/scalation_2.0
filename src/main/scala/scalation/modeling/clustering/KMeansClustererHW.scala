
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, John Miller
 *  @version 2.0
 *  @date    Mon Apr 22 14:53:24 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: K-Means Clustering (use Hartigan-Wong algorithm for point reassignment)
 */

package scalation
package modeling
package clustering

import scala.util.control.Breaks.{breakable, break}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KMeansClustererHW` class cluster several vectors/points using
 *  the Hartigan-Wong algorithm.
 *  @param x      the vectors/points to be clustered stored as rows of a matrix
 *  @param k      the number of clusters to make
 *  @param flags  the flags used to adjust the algorithm
 */
class KMeansClustererHW (x: MatrixD, k: Int, flags: Array [Boolean] = Array (false, false))
      extends KMeansClusterer (x, k, flags):

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reassign each vector/point to the cluster with the closest centroid.
     *  This one follows a version of the Hartigan-Wong algorithm.
     *  Indicate done, if no points changed clusters (for stopping rule).
     *  Note: randomized order for index 'i' tends to work better.
     *  @see www.tqmp.org/RegularArticles/vol09-1/p015/p015.pdf
     */
    protected override def reassign (): Boolean =
        var done = true                                                     // done indicates no changes
//      for i <- x.indices do                                               // standard order for index i

        breakable {
            for i <- raniv.igen do                                          // randomize order of index i
                val c1 = to_c(i)                                            // c1 = current cluster for point x_i
                if sz(c1) > 1 then                                          // if size of c1 > 1
                    val d  = distance2 (x(i), cent, c1)                     // adjusted distances to all centroid
                    val c2 = d.argmin ()                                    // c2 = cluster with closest centroid to x_i
                    if d(c2) < d(c1) then                                   // if closest closer than current
                        sz(c1) -= 1                                         // decrement the size of cluster c1
                        sz(c2) += 1                                         // increment size of cluster c2
                        to_c(i) = c2                                        // reassign point x_i to cluster c2
                        done = false                                        // changed clusters => not done
                        if immediate then break ()                          // optionally return after first change
                end if
            end for
        } // breakable
        done                                                                // return whether there were no changes
    end reassign

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the adjusted distance to point 'u' according to the R2 value
     *  described in the Hartigan-Wong algorithm.
     *  @param u     the point in question
     *  @param cent  the matrix holding the centroids
     *  @param cc    the current cluster for point u
     */
    def distance2 (u: VectorD, cent: MatrixD, cc: Int): VectorD =
        val d = new VectorD (cent.dim)
        for c <- 0 until k do
            d(c) = if c == cc then (sz(c) * dist (u, cent(c))) / (sz(c) - 1)
                   else (sz(c) * dist (u, cent(c))) / (sz(c) + 1)
        end for
        d
    end distance2

end KMeansClustererHW

import Clusterer.test

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansClustererTestHW` object is used to test the `KMeansClustererHW` class.
 *  > runMain scalation.modeling.clustering.kMeansClustererHWTest
 */
@main def kMeansClustererHWTest (): Unit =

    import Clusterer.x

    val k   = 3
    val opt = 3.0

    println (s"x = $x")
    println (s"k = $k")
    println ("-" * 60)

    val tf = Array (true, false)
    for fl0 <- tf; fl1 <- tf do
        val fls = Array (fl0, fl1)
        test (x, fls, new KMeansClustererHW (x, k, fls), opt)
    end for

    new Plot (x(?, 0), x(?, 1), null, "x0 vs x1")

end kMeansClustererHWTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansClustererHWTest2` object is used to test the `KMeansClustererHW` class.
 *  > runMain scalation.modeling.clustering.kMeansClustererHWTest2
 */
@main def kMeansClustererHWTest2 (): Unit =

    //                         x0    x1
    val x = MatrixD ((8, 2),  1.0,  1.0,
                              1.0,  3.0,
                              5.0, 18.0,
                              5.0, 20.0,
                              9.0, 10.0,
                              9.0, 12.0,
                             15.0, 30.0,
                             15.0, 32.0)

    val k   = 4
    val opt = 8.0

    println (s"x = $x")
    println (s"k = $k")
    println ("-" * 60)

    val tf = Array (true, false)
    for fl0 <- tf; fl1 <- tf do
        val fls = Array (fl0, fl1)
        test (x, fls, new KMeansClustererHW (x, k, fls), opt)
    end for

    new Plot (x(?, 0), x(?, 1), null, "x0 vs x1")

end kMeansClustererHWTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansClustererHWTest3` object is used to test the `KMeansClustererHW` class.
 *  > runMain scalation.modeling.clustering.kMeansClustererHWTest3
 */
@main def kMeansClustererHWTest3 (): Unit =

    import scalation.random.{Bernoulli, Normal}

    val coin  = Bernoulli ()
    val dist1 = Normal (2.0, 1.0)
    val dist2 = Normal (8.0, 1.0)
    val x     = new MatrixD (50, 2)
    val k     = 4
    val opt   = 76.0

    for i <- x.indices do x(i) = VectorD (if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen)

    println (s"x = $x")
    println (s"k = $k")
    println ("-" * 60)

    val tf = Array (true, false)
    for fl0 <- tf; fl1 <- tf do
        val fls = Array (fl0, fl1)
        test (x, fls, new KMeansClustererHW (x, k, fls), opt)
    end for

    new Plot (x(?, 0), x(?, 1))

end kMeansClustererHWTest3

