
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue May 29 14:45:32 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: K-Means Clustering (randomly assign points to clusters)
 */

package scalation
package modeling
package clustering

import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{breakable, break}

import scalation.mathstat._
import scalation.random.{PermutedVecI, Randi}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KMeansClusterer` class cluster several vectors/points using k-means
 *  clustering.  Randomly assign points to k clusters (primary technique).
 *  Iteratively, reassign each point to the cluster containing
 *  the closest centroid.  Stop when there are no changes to the clusters.
 *  @see `KMeansClusterer2` for secondary technique.
 *-----------------------------------------------------------------------------
 *  @param x      the vectors/points to be clustered stored as rows of a matrix
 *  @param k      the number of clusters to make
 *  @param flags  the array of flags used to adjust the algorithm
 *                    default: no post processing, no immediate return upon change
 */
class KMeansClusterer (x: MatrixD, k: Int, val flags: Array [Boolean] = Array (false, false))
      extends Clusterer:

    private val debug = debugf ("KMeansClusterer", false)                // debug function
    private val flaw  = flawf ("KMeansClusterer")                        // flaw function

    if k >= x.dim then        flaw ("init", "k must be less than the number of vectors")
    if flags.length != 2 then flaw ("init", "KMeansClusterer requires 2 flags")

    protected val MAX_IT = 1000                                          // the maximum number of iterations
    protected val cent   = new MatrixD (k, x.dim2)                       // the k centroids of clusters
    protected val sz     = new VectorI (k)                               // the cluster sizes
    private [clustering] val to_c = Array.ofDim [Int] (x.dim)            // assignment of vectors to clusters

    protected val (post, immediate) = (flags(0), flags(1))               // (post processing swapping, immediate return upon change)
    protected var raniv: PermutedVecI = null                             // generator of permutations

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Iteratively recompute clusters until the assignment of points does not
     *  change.  Initialize by randomly assigning points to k clusters.
     */
    def train (): Unit =
        sz.set (0)                                                       // cluster sizes initialized to zero
        raniv = PermutedVecI (VectorI.range (0, x.dim), stream)          // for randomizing index order
        assign ()                                                        // randomly assign points to clusters
        fixEmptyClusters ()                                              // move points into empty clusters
        if ! initCentroids () then calcCentroids (x, to_c, sz, cent)     // pick points for initial centroids
        show (0)

        breakable {
            for l <- 1 to MAX_IT do
                if reassign () then break ()                             // reassign points to clusters (no change => break)
                calcCentroids (x, to_c, sz, cent)                        // re-calculate the centroids
                show (l)
            end for
        } // breakable
        val ce = sz.indexOf (0)                                          // check for empty clusters
        if ce != -1 then throw new Exception (s"Empty cluster c = $ce")

        if post then swap ()                                             // swap points to improve sse
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cluster assignment vector.  Should only be called after `train`.
     */
    def cluster: Array [Int] = to_c

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the centroids. Should only be called after `train`. 
     */
    def centroids: MatrixD = cent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sizes of the centroids.  Should only be called after `train`. 
     */
    def csize: VectorI = sz

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly assign each vector/point x(i) to a random cluster.
     *  Primary technique for initiating the clustering.
     */
    protected def assign (): Unit =
        val ran = new Randi (0, k-1, stream)                             // for random integers: 0, ..., k-1
        for i <- x.indices do
            to_c(i)      = ran.igen                                      // randomly assign x(i) to a cluster
            sz(to_c(i)) += 1                                             // increment size of that cluster
        end for
    end assign

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fix all empty clusters by taking a point from the largest cluster.
     */
    protected def fixEmptyClusters (): Unit =
        for c <- 0 until k if sz(c) < 1 do                               // for each empty cluster
            debug ("fixEmptyClusters", s"cluster c = $c is empty!")
            val largest = sz.argmax ()                                   // largest cluster
            val indices = to_c.indices.filter (to_c(_) == largest)       // indices of elements in largest cluster            

            val ran = new Randi (0, indices.size-1)                      // random integer generator
            val i   = indices(ran.igen)                                  // randomly pick one point from largest cluster
            sz(to_c(i)) -= 1                                             // decrement size of previous cluster
            to_c(i)      = c                                             // reassign vector x(i) to cluster c
            sz(c)       += 1                                             // increment size of cluster c
            debug ("fixEmptyClusters", s"to_c = ${stringOf (to_c)}")
        end for
    end fixEmptyClusters

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reassign each vector/point to the cluster with the closest centroid.
     *  Indicate done, if no points changed clusters (for stopping rule).
     */
    protected def reassign (): Boolean =
        var done = true                                                  // done indicates no changes
//      for i <- raniv.igen do                                           // randomize order of index i
//      for i <- x.indices do                                            // standard order for index i

        var (go, i) = (true, 0)
        cfor (go && i < x.dim, i += 1) {                                 // standard order for index i
            val c1 = to_c(i)                                             // c1 = current cluster for point x_i
            if sz(c1) > 1 then                                           // if size of c1 > 1
                val d  = distance (x(i), cent)                           // distances to all centroid
                val c2 = d.argmin ()                                     // c2 = cluster with closest centroid to x_i
                if d(c2) < d(c1) then                                    // if closest closer than current
                    sz(c1) -= 1                                          // decrement size of current cluster
                    sz(c2) += 1                                          // increment size of new cluster
                    to_c(i) = c2                                         // reassign point x_i to cluster c2
                    done    = false                                      // changed clusters => not done
                    if immediate then go = false                         // optionally return after first change
            end if
        } // cfor
        done                                                             // return whether there were no changes
    end reassign

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap clusters for points x(i) and x(j).
     *  param i  the index for point x(i)
     *  param j  the index for point x(j)
     */
    private def swapPoints (i: Int, j: Int): Unit =
        val temp = to_c(i)
        to_c(i)  = to_c(j)
        to_c(j)  = temp
        calcCentroids (x, to_c, sz, cent)
    end swapPoints

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Try all pairwise swaps and make them if sse improves.
     */
    protected def swap (): Unit =
        for i <- 0 until x.dim-1; j <- i+1 until x.dim if to_c(i) != to_c(j) do
            val sum1 = sse (x, to_c(i), to_c) + sse (x, to_c(j), to_c)
            swapPoints (i, j)
            val sum2 = sse (x, to_c(i), to_c) + sse (x, to_c(j), to_c)
            debug ("swap", s"sum1 = $sum1 vs. sum2 = $sum2")
            if sum2 > sum1 then swapPoints (i, j)                           // if not better, swap back
        end for
    end swap

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new point/vector z, determine which cluster it belongs to,
     *  i.e., the cluster whose centroid it is closest to.
     *  @param z  the vector to classify
     */
    def classify (z: VectorD): Int = distance (z, cent).argmin ()

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the state of the algorithm at iteration l.
     *  @param l  the current iteration
     */
    def show (l: Int): Unit = println (s"($l) to_c = ${stringOf (to_c)} \n($l) cent = $cent")

end KMeansClusterer

import Clusterer.test

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansClustererTest` object is used to test the `KMeansClusterer` class.
 *  > runMain scalation.modeling.clustering.kMeansClustererTest
 */
@main def kMeansClustererTest (): Unit =

    import Clusterer.x

    val k   = 3
    val opt = 3.0

    println (s"x = $x")
    println (s"k = $k")
    println ("-" * 60)

    val tf = Array (true, false)
    for fl0 <- tf; fl1 <- tf do
         val fls = Array (fl0, fl1)
         test (x, fls, new KMeansClusterer (x, k, fls), opt)
    end for

    new Plot (x(?, 0), x(?, 1), null, "x0 vs x1")

end kMeansClustererTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansClustererTest2` object is used to test the `KMeansClusterer` class.
 *  > runMain scalation.modeling.clustering.kMeansClustererTest2
 */
@main def kMeansClustererTest2 (): Unit =

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
         test (x, fls, new KMeansClusterer (x, k, fls), opt)
    end for

    new Plot (x(?, 0), x(?, 1), null, "x0 vs x1")

end kMeansClustererTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansClustererTest3` object is used to test the `KMeansClusterer` class.
 *  > runMain scalation.modeling.clustering.kMeansClustererTest3
 */
@main def kMeansClustererTest3 (): Unit =

    import scalation.random.{Bernoulli, Normal}

    val coin  = Bernoulli ()
    val dist1 = Normal (2.0, 1.0)
    val dist2 = Normal (8.0, 1.0)
    val x    = new MatrixD (50, 2)
    val k    = 4
    val opt  = 76.0

    for i <- x.indices do x(i) = VectorD (if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen)

    println (s"x = $x")
    println (s"k = $k")
    println ("-" * 60)

    val tf = Array (true, false)
    for fl0 <- tf; fl1 <- tf do
         val fls = Array (fl0, fl1)
         test (x, fls, new KMeansClusterer (x, k, fls), opt)
    end for

    new Plot (x(?, 0), x(?, 1))    

end kMeansClustererTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansClustererTest4` object is used to test the `KMeansClusterer` class.
 *  > runMain scalation.modeling.clustering.kMeansClustererTest4
 */
@main def kMeansClustererTest4 (): Unit =

    import scalation.random.{Normal, Bernoulli}

    val coin  = Bernoulli ()
    val dist1 = Normal (2.0, 1.0)
    val dist2 = Normal (8.0, 1.0)
    val x    = new MatrixD (100, 2)
    val k    = 4
    val opt  = 171.0

    for i <- x.indices do x(i) = VectorD (if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen)

//  import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer
//  val cl = new KMeansPlusPlusClusterer (k)
    
    println (s"x = $x")
    println (s"k = $k")
    println ("-" * 60)

    val tf = Array (true, false)
    for fl0 <- tf; fl1 <- tf do
         val fls = Array (fl0, fl1)
         test (x, fls, new KMeansClusterer (x, k, fls), opt)
    end for

    new Plot (x(?, 0), x(?, 1))

end kMeansClustererTest4

