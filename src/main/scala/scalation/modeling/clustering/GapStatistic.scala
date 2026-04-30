
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Michael Cotterell, John Miller, Hao Peng
 *  @version 2.0
 *  @date    Thu Mar  9 15:08:30 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Gap Statistic (for determining the optimal number of clusters)
 */

package scalation
package modeling
package clustering

import scala.annotation.unused
import scala.math.log

import scalation.mathstat._
import scalation.random.RandomVecD

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GapStatistic` object is used to help determine the optimal number
 *  of clusters for a clusterer by comparing results to a reference 
 *  distribution.
 *-----------------------------------------------------------------------------
 *  @see web.stanford.edu/~hastie/Papers/gap.pdf 
 */
object GapStatistic:

    import Algorithm._

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a reference distribution based on a set of points.
     *  @param x        the vectors/points to be clustered stored as rows of a matrix
     *  @param useSVD   use SVD to account for the shape of the points (default = true)
     *  @param s        the random number stream (to vary the clusters made)
     */
    def reference (x: MatrixD, useSVD: Boolean = true, stream: Int = 0): MatrixD =
        var ref = new MatrixD (x.dim, x.dim2)
        if useSVD then
            val mean  = x.mean
            val xzero = x - mean
            val svd   = new Fac_SVD (xzero)
            val vt    = svd.factor123 ()._3                   // factors into u, s, vt
            val xp    = xzero * vt.transpose
            val zp    = new MatrixD (x.dim, x.dim2)
            for i <- zp.indices2 do
                val ci = xp(?, i)
                zp(?, i) = RandomVecD (zp.dim, ci.max, ci.min, stream = (stream + i) % 1000).gen
            ref = (zp * vt) + mean
        else
            for i <- ref.indices2 do
                val ci = x(?, i)
                ref(?, i) = RandomVecD (ref.dim, ci.max, ci.min, stream = (stream + i) % 1000).gen
        end if
        ref 
    end reference

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a sum of pairwise distances between points in each cluster (in
     *  one direction). 
     *  @param x       the vectors/points to be clustered stored as rows of a matrix
     *  @param cl      the `Clusterer` use to compute the distance metric 
     *  @param clustr  the cluster assignments
     *  @param k       the number of clusters
     */
    def cumDistance (x: MatrixD, @unused cl: Clusterer, clustr: Array [Int], k: Int): VectorD =
        val sums = new VectorD (k)
        for i <- 0 until x.dim-1; j <- i+1 until x.dim if clustr(i) == clustr(j) do
            sums(clustr(j)) += dist (x(i), x(j))
        end for
        sums
    end cumDistance

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the within sum of squared errors in terms of distances between
     *  between points within a cluster (in one direction).
     *  @param x       the vectors/points to be clustered stored as rows of a matrix
     *  @param cl      the `Clusterer` use to compute the distance metric 
     *  @param clustr  the cluster assignments
     *  @param k       the number of clusters
     */
    def withinSSE (x: MatrixD, cl: Clusterer, clustr: Array [Int], k: Int): Double =
        (cumDistance (x, cl, clustr, k) / cl.csize.toDouble).sum
    end withinSSE

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a `KMeansPPClusterer` clustering on the given points with
     *  an optimal number of clusters `k` chosen using the Gap statistic.
     *  @param x       the vectors/points to be clustered stored as rows of a matrix
     *  @param kMax    the upper bound on the number of clusters
     *  @param algo    the reassignment aslgorithm used by `KMeansPPClusterer`
     *  @param b       the number of reference distributions to create (default = 1)
     *  @param useSVD  use SVD to account for the shape of the points (default = true)
     *  @param plot    whether or not to plot the logs of the within-SSEs (default = false)
     */
    def kMeansPP (x: MatrixD, kMax: Int, algo: Algorithm = HARTIGAN, @unused b: Int = 1, useSVD: Boolean = true,
                  plot: Boolean = false): (KMeansPPClusterer, Array [Int], Int) =
        val awk = new VectorD (kMax)
        val rwk = new VectorD (kMax)
        val gap = new VectorD (kMax)
        val kv  = VectorD.range (1, kMax+1)
        var opk = -1
//      var opcl: KMeansPPClusterer = null
//      var opcls: Array [Int] = null

        for k <- 0 until kMax do
            val ref = GapStatistic.reference (x, useSVD)
            val acl = KMeansPPClusterer (x,   k+1, algo)
            val rcl = KMeansPPClusterer (ref, k+1, algo)
            awk(k)  = log (GapStatistic.withinSSE (x,   acl, acl.cluster, k+1))
            rwk(k)  = log (GapStatistic.withinSSE (ref, rcl, rcl.cluster, k+1))
            gap(k)  = rwk(k) - awk(k)
            if k != 0 && opk == -1 && gap(k-1) >= gap(k) - gap(k)*0.1 then
                // TODO use stddev instead of 0.01*gap
                opk = k
        end for

        if plot then
            new Plot (kv, awk, rwk, "Actual wSSE and Reference wSSE vs. k") // , true)
            new Plot (kv, gap, null, "Gap vs. k") // , true)

        val cl = KMeansPPClusterer (x, opk, algo)   // TODO used saved instead of reclustering
        (cl, cl.cluster, opk)
    end kMeansPP

end GapStatistic


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `gapStatisticTest` main function is used to test the `GapStatistic` object.
 *  > runMain scalation.modeling.clustering.gapStatisticTest
 */
@main def gapStatisticTest (): Unit =

    import Clusterer.x

    val maxK = 6

    val (cl, cls, k) = GapStatistic.kMeansPP (x, maxK, useSVD = false, plot = true)
    println (s"  k = $k")
    println (s"sse = ${cl.sse (x, cls)}")

end gapStatisticTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `gapStatisticTest2` main function is used to test the `GapStatistic` object.
 *  > runMain scalation.modeling.clustering.gapStatisticTest2
 */
@main def gapStatisticTest2 (): Unit =

    import scalation.random.{Normal, Bernoulli}

    val coin  = Bernoulli ()
    val dist1 = Normal (2.0, 0.1)
    val dist2 = Normal (8.0, 0.1)
    val x     = new MatrixD (50, 2)
    val maxK  = 10 

    for i <- x.indices do x(i) = VectorD (if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen)

    val (cl, cls, k) = GapStatistic.kMeansPP (x, maxK, useSVD = false, plot = true)
    println (s"  k = $k")
    println (s"sse = ${cl.sse (x, cls)}")

end gapStatisticTest2

