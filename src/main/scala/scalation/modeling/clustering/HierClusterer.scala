
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Karen Gilmer
 *  @version 2.0
 *  @date    Wed Jan  9 15:38:04 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Hierarchical Clustering
 */
 
package scalation
package modeling
package clustering

import scala.runtime.ScalaRunTime.stringOf
import scala.collection.mutable.{Set, ArrayBuffer}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Cluster several vectors/points using hierarchical clustering.  Start with
 *  each point forming its own cluster and merge clusters until there are only 'k'.
 *  @param x  the vectors/points to be clustered stored as rows of a matrix
 *  @param k  stop when the number of clusters equals k
 */
class HierClusterer (x: MatrixD, k: Int = 2)
      extends Clusterer:

    private val debug = debugf ("HierClusterer", false)   // debug function
    private val flaw  = flawf ("HierClusterer")           // flaw function

    if k >= x.dim then flaw ("init", "k must be less than the number of vectors")

    private val cent  = new MatrixD (k, x.dim2)           // the k centroids of clusters
    private val to_c  = Array.ofDim [Int] (x.dim)         // assignment of vectors to clusters
    private val clust = ArrayBuffer [Set [Int]] ()        // the list of clusters as sets
    private val sz    = new VectorI (k)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Iteratively merge clusters until the number of clusters equals 'k'.
     */
    def train (): Unit =
        sz.set (0)                                        // initialize cluster sizes to zero
        initClusters ()                                   // make a cluster for each point

        for kk <- x.dim until k by -1 do
            val (si, sj) = bestMerge (kk)                 // find the 2 closest clusters
            clust += si | sj                              // add the union of sets i and j
            clust -= si                                   // remove set i
            clust -= sj                                   // remove set j
            debug ("train", s"for cluster (${kk-1}), clust = $clust")
        end for

        finalClusters ()                                  // make final cluster assignments
        calcCentroids (x, to_c, sz, cent)                 // calculate centroids for clusters
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the best two clusters 'si', 'sj' to merge by trying all combinations
     *  and taking the pair with the least distance.
     *  @param kk  the current number of clusters
     */
    private def bestMerge (kk: Int): (Set [Int], Set [Int]) =
        var si, sj: Set [Int] = null
        var minDist = Double.PositiveInfinity
        for i <- 0 until kk-1; j <- i+1 until kk do
            val d_ij = clustDist (clust(i), clust(j))
            if d_ij < minDist then
                minDist = d_ij                            // update minimum distance
                si = clust(i); sj = clust(j)              // remember point sets i and j
        end for
        (si, sj)
    end bestMerge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cluster assignment vector.  Should only be called after `train`.
     */
    def cluster: Array [Int] = to_c

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the centroids.  Should only be called after `train`. 
     */
    def centroids: MatrixD = cent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sizes of the centroids.  Should only be called after `train`. 
     */
    def csize: VectorI = sz

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum distance between cluster 'setA' and 'setB'.
     *  @param setA  the first set
     *  @param setB  the second set
     */
    private def clustDist (setA: Set [Int], setB: Set [Int]): Double =
        var dm = Double.PositiveInfinity
        for i <- setA; j <- setB do
            val d = dist (x(i), x(j))
            if d < dm then dm = d
        end for
        dm
    end clustDist

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create initial clusters where each point forms its own cluster.
     */
    private def initClusters (): Unit = for i <- x.indices do clust += Set (i)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For each data point, determine its final cluster assignment, finalize
     *  the to_c array of cluster assignments.
     */
    private def finalClusters (): Unit =
        for c <- 0 until k; i <- clust(c) do { to_c(i) = c; sz(c) += 1 }
    end finalClusters

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new point/vector y, determine which cluster it belongs to.
     *  @param z  the vector to classify
     */
    def classify (z: VectorD): Int = distance (z, cent).argmin ()

end HierClusterer

import Clusterer.test

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `hierClustererTest` object is used to test the `HierClusterer` class.
 *  > runMain scalation.modeling.clustering.hierClustererTest
 */
@main def hierClustererTest (): Unit =

    import Clusterer.x

    banner ("HierClustererTest")

    val y = VectorD (10.0, 10.0)
    val z = VectorD ( 2.0,  4.0)
    println ("x = " + x)
    println ("y = " + y)
    println ("z = " + z)
    println ("-" * 60)

    val k   = 3
    val opt = 3.0
    val cl = new HierClusterer (x, k)                 
    test (x, Array (false), cl, opt)

//  cl.train ()
    println ("--- final cluster = " + stringOf (cl.cluster) + "\n")
    println ("--- classify " + y + " = " + cl.classify (y) + "\n")
    println ("--- classify " + z + " = " + cl.classify (z) + "\n")
    println (s"sse = ${cl.sse (x, cl.cluster)}")

end hierClustererTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `hierClustererTest2` object is used to test the `HierClusterer` class.
 *  > runMain scalation.modeling.clustering.hierClustererTest2
 */
@main def hierClustererTest2 (): Unit =

    import scalation.random.{Bernoulli, Normal}

    val coin  = Bernoulli ()
    val dist1 = Normal (2.0, 1.0)
    val dist2 = Normal (8.0, 1.0)
    val v     = new MatrixD (50, 2)
    for i <- v.indices do v(i) = VectorD (if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen)

    println (s"v = $v")
    println ("-" * 60)

    val cl = new HierClusterer (v, 4)                 
    cl.train ()
    println ("--- final cluster = " + stringOf (cl.cluster) + "\n")
    println (s"sse = ${cl.sse(v, cl.cluster)}")

end hierClustererTest2

