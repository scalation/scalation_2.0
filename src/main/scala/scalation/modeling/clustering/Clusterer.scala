
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue May 29 14:45:32 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Base Trait for Clustering Models/Algorithms
 */

package scalation
package modeling
package clustering

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Clusterer` object provides a simple dataset (matrix of data points) for
 *  initial testing of clustering algorithms.
 */
object Clusterer:

//                            x0   x1
    val x = MatrixD ((6, 2), 1.0, 2.0,                                   // data matrix (6 points)
                             2.0, 1.0,
                             4.0, 5.0,
                             5.0, 4.0,
                             8.0, 9.0,
                             9.0, 8.0)

    private val NTESTS = 1000                                            // number of tests/streams

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the quality of the variants of the `KMeansClusterer` class.
     *  @param x    the data matrix holding the points/vectors
     *  @param fls  the array of flags 
     *  @param alg  the clustering algorithm to test
     *  @param opt  the known optimum for see (ignore if not known)
     */
    def test (x: MatrixD, fls: Array [Boolean], alg: Clusterer, opt: Double = -1.0): Unit =
        banner (s"test (flags = ${stringOf (fls)}, opt = $opt)")
        val stat_sst = new Statistic ("sst")
        val stat_sse = new Statistic ("sse")
        val stat_rSq = new Statistic ("rSq")
        var ok = 0
        for s <- 0 until NTESTS do                                       // test with different random streams
            alg.setStream (s)
            alg.train ()                                                 // train, i.e., apply clustering algorithm
            val clus = alg.cluster                                       // current clustering assignment
            val (sst, sse) = (alg.sst (x), alg.sse (x, clus))            // compute sum of squares, total and error
            if s < 5 then println (s"cluster = ${stringOf (clus)}, sse = $sse")   // print first few assignments
            stat_sst.tally (sst)
            stat_sse.tally (sse)
            stat_rSq.tally (1.0 - sse / sst)
            if opt >= 0.0 && alg.checkOpt (x, alg.cluster, opt) then ok += 1
        end for
        if opt != -1 then println (s"ok = $ok of $NTESTS tests")
        println (Statistic.labels)
        println (stat_sst)
        println (stat_sse)
        println (stat_rSq)
    end test

end Clusterer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Clusterer` trait provides a common framework for several clustering
 *  algorithms.
 */
trait Clusterer:

    private var _name: Array [String] = null                         // optional names for clusters

    protected var stream = 0                                         // the stream to use for random numbers

//  _taskType = TaskType.Cluster                                     // the type of task performed -- FIX -- subtrait of Model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the names for the clusters.
     *  @param nm  the array of names
     */
    def name_ (nm: Array [String]): Unit = _name = nm

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the name of the 'c'-th cluster.
     *  @param c  the c-th cluster
     */
    def name (c: Int): String = 
        if _name != null && c < _name.length then _name(c) else "unknown"
    end name

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the random stream to 's'.  Method must be called in implemeting classes
     *  before creating any random generators.
     *  @param s  the new value for the random number stream
     */
    def setStream (s: Int): Unit = stream = s

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a set of points/vectors, put them in clusters, returning the cluster
     *  assignments.  A basic goal is to minimize the sum of squared errors (sse)
     *  in terms of squared distances of points in the cluster to its centroid.
     */
    def train (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cluster assignments.  Should only be called after 'train'.
     */
    def cluster: Array [Int]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sizes (number of points within) of the clusters.  Should only
     *  be called after train. 
     */
    def csize: VectorI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the centroids (a centroid is the mean of points in a cluster).
     *  Should only be called after train. 
     */
    def centroids: MatrixD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the centroids have been initialized.
     */
    def initCentroids (): Boolean = false

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the centroids based on current assignment of points to clusters
     *  and update the 'cent' matrix that stores the centroids in its rows.
     *  @param x     the data matrix holding the points {x_i = x(i)} in its rows
     *  @param to_c  the cluster assignment array
     *  @param sz    the sizes of the clusters (number of points)
     *  @param cent  the matrix holding the centroids in its rows
     */
    def calcCentroids (x: MatrixD, to_c: Array [Int], sz: VectorI, cent: MatrixD): Unit =
        cent.setAll (0.0)                                            // set cent matrix to all zeros
        for i <- x.indices do
            val c   = to_c(i)                                        // x_i currently assigned to cluster c
            cent(c) = cent(c) + x(i)                                 // add the next vector in cluster
        end for
        for c <- cent.indices do cent(c) = cent(c) / sz(c)           // divide to get averages/means
    end calcCentroids

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new point/vector z, determine which cluster it belongs to.
     *  @param z  the vector to classify
     */
    def classify (z: VectorD): Int

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the distances between vector/point 'u' and the points stored as
     *  rows in matrix 'cn'
     *  @param u    the given vector/point (u = x_i)
     *  @param cn   the matrix holding several centroids
     *  @param kc_  the number of centroids so far
     */
    def distance (u: VectorD, cn: MatrixD, kc_ : Int = -1): VectorD =
        val kc = if kc_ < 0 then cn.dim else kc_
        val du = new VectorD (kc)
        for c <- 0 until kc do du(c) = dist (u, cn(c))
        du
    end distance

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sum of squared errors within all clusters, where error is
     *  indicated by e.g., the distance from a point to its centroid.
     *  @param x     the data matrix holding the points
     *  @param to_c  the cluster assignments
     */
    def sse (x: MatrixD, to_c: Array [Int]): Double =
        val cent = centroids
        var sum  = 0.0
        for i <- x.indices do sum += dist (x(i), cent(to_c(i)))
        sum
    end sse

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sum of squared errors from the points in cluster 'c' to the
     *  cluster's centroid.
     *  @param x     the data matrix holding the points
     *  @param c     the current cluster
     *  @param to_c  the cluster assignments
     */
    def sse (x: MatrixD, c: Int, to_c: Array [Int]): Double =
        val cent = centroids
        var sum  = 0.0
        for i <- x.indices if to_c(i) == c do sum += dist (x(i), cent(c))
        sum
    end sse

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sum of squares total for all the points from the mean.
     *  @param x  the data matrix holding the points
     */
    def sst (x: MatrixD): Double =
        val xmean = x.mean
        var sum   = 0.0
        for i <- x.indices do sum += dist (x(i), xmean)
        sum
    end sst

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check to see if the sum of squared errors is optimum.
     *  @param x     the data matrix holding the points
     *  @param to_c  the cluster assignments
     *  @param opt   the known (from human/oracle) optimum
     */
    def checkOpt (x: MatrixD, to_c: Array [Int], opt: Double): Boolean = sse (x, to_c) <= opt

end Clusterer

