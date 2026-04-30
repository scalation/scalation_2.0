
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Michael Cotterell, John Miller, Hao Peng
 *  @version 2.0
 *  @date    Tue Mar  7 22:10:21 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Implementation of k-means++ Clustering Algorithm
 *
 *  @see webee.technion.ac.il/people/koby/publications_all/pdfs//conf_ijcai_SlonimAC13.pdf
 *  @see cseweb.ucsd.edu/~avattani/papers/hartigan.pdf
 */

package scalation
package modeling
package clustering

//  U N D E R   D E V E L O P M E N T
//  FIX - merge with KMeansClustererPP

import scala.util.boundary, boundary.break

import scalation.mathstat._
import scalation.random._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Algorithm` enum specifies which algorithm to use.
 */
enum Algorithm:

    case HARTIGAN, LLOYD

end Algorithm

import Algorithm._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KMeansPPClusterer` class cluster several vectors/points using 
 *  the k-means++ clustering technique.  
 *-----------------------------------------------------------------------------
 *  @see ilpubs.stanford.edu:8090/778/1/2006-13.pdf 
 *-----------------------------------------------------------------------------
 *  @param x      the vectors/points to be clustered stored as rows of a matrix
 *  @param k      the number of clusters to make
 *  @param algo   the clustering algorithm to use
 *  @param flags  the flags used to adjust the algorithm
 */
class KMeansPPClusterer (x: MatrixD, k: Int, algo: Algorithm = HARTIGAN,
                         flags: Array [Boolean] = Array (false, false))
      extends KMeansClusterer (x, k, flags):

    private   val debug = debugf ("KMeansPPClusterer", false)    // debug flag
    protected var _k    = 0                                      // last centroid chosen 
    protected val pdf   = new VectorD (x.dim)                    // pdf for choosing centroids

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a set of points/vectors, put them in clusters, returning the cluster
     *  assignment vector.  A basic goal is to minimize the sum of the distances
     *  between points within each cluster.
     */
    override def train (): Unit =
        raniv   = PermutedVecI (VectorI.range (0, x.dim), stream)
        initCentroids ()
        algo match
            case LLOYD    => clusterLloyd ()
            case HARTIGAN => clusterHartigan ()

        val ce = sz.indexOf (0)                                          // check for empty clusters
        if ce != -1 then throw new Exception (s"Empty cluster c = $ce")
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the shortest distance from a data point 'u' to the closest 
     *  centroid we have already chosen.
     *  @param u  the vector/point
     */
    private def shortestDistance (u: VectorD): Double = 
        var min = Double.PositiveInfinity
        for i <- 0 to _k do
            val d = dist (u, cent(i))
            if d < min then min = d
        min
    end shortestDistance

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sum of the shortest distances from each data point to its
     *  closest centroid.
     */
    private def pseudoSSE (): Double =
        var sum = 0.0
        for i <- x.indices do sum += shortestDistance (x(i))
        sum
    end pseudoSSE

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the probability distribution for choosing the next centroid.
     */
    private def updatePDF (): Unit =
        val psse = pseudoSSE ()
        for i <- x.indices do pdf(i) = shortestDistance (x(i)) / psse
        debug ("updatePDF", s"updated pdf = $pdf")
    end updatePDF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize the centroids according to the k-means++ technique.
     */
    override def initCentroids (): Boolean =
        val ran1 = new Randi (0, x.dim-1, stream)                // uniform integer distribution 
        cent(0)  = x(ran1.igen)                                  // pick first centroid uniformly at random
        for i <- 1 until k do                                    // pick remaining centroids
            updatePDF ()                                         // update probability distribution
            val ran2 = Discrete (pdf, stream = (stream+i) % 1000)
            cent(i)  = x(ran2.igen)                              // pick next centroid according to pdf
            _k       = i                                         // update last centroid chosen
        debug ("initCentroids", s"initial cent = $cent")
        true
    end initCentroids

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reassign each vector/point to the cluster with the closest centroid.
     *  Indicate done, if no points changed clusters (for stopping rule).
     *  @param first  whether this is the first call to 'reassign'
     */
    private def reassign (first: Boolean): Boolean =
        var done = true
        for i <- x.indices do
            if first then                                        // first call => no c0
                val cc    = classify (x(i))                      // find closest cluster for x(i)
                to_c(i) = cc                                     // assign x(i) to this cluster
                sz(cc) += 1                                      // increment the size of this cluster
                done    = false                                  // change => not done
            else
                val c0 = to_c(i)                                 // record current cluster assignment c0
                if sz(c0) > 1 then                               // make sure cluster c0 has multiple points
                    val cc  = classify (x(i))
                    if cc != c0 then                             // if closest is not the current cluster
                        sz(c0) -= 1                              // decrement the number of points in current cluster
                        to_c(i) = cc                             // reassign x(i) to the closest cluster
                        sz(cc) += 1                              // increment the size of this cluster
                        done    = false                          // change => not done
            end if
        end for
        done                                                     // return whether there was a change during this pass
    end reassign

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cluster the points using Lloyd's algorithm.
     */
    private def clusterLloyd (): Array [Int] = boundary:
        for l <- 1 to MAX_IT do
            if reassign (l == 1) then break (to_c)              // reassign points to clusters (no change => break)
            calcCentroids (x, to_c, sz, cent)                   // re-calculate the centroids
            show (l)
        to_c
    end clusterLloyd

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cluster the points using a version of the Hartigan-Wong algorithm.
     *  @see www.tqmp.org/RegularArticles/vol09-1/p015/p015.pdf
     */
    def clusterHartigan (): Array [Int] = boundary:
        reassign (true)
        calcCentroids (x, to_c, sz, cent)
        for l <- 1 to MAX_IT do
            if reassign2 () then break (to_c)                    // reassign points to clusters (no change => break)
            show (l)
        to_c
    end clusterHartigan
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reassign each vector/point to the cluster with the closest centroid.
     *  This one follows a version of the Hartigan-Wong algorithm.
     *  Indicate done, if no points changed clusters (for stopping rule).
     *  @see www.tqmp.org/RegularArticles/vol09-1/p015/p015.pdf
     */
    private def reassign2 (): Boolean =
        var done = true                                          // done indicates no changes
        val iv   = raniv.igen                                    // randomly order the points
        for i <- iv do
            val c0 = to_c(i)                                     // record point v's current cluster (c0)
            if sz(c0) > 1 then                                   // make sure not to empty a cluster
                calcCentroids (x, to_c, sz, cent)                // re-calculate the centroids
                val c1 = closestByR2 (i)                         // find closest cluster to point v
                sz(c0) -= 1                                      // decrement the size of cluster c0
                to_c(i) = c1                                     // reassign point v to cluster c1
                sz(c1) += 1                                      // increment size of cluster c1
                if c1 != c0 then done = false                    // changed clusters => not done
        end for
        done                                                     // return whether there were no changes
    end reassign2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the closest cluster to point 'u' according to the R2 value 
     *  described in the Hartigan-Wong algorithm.
     */
    private def closestByR2 (i: Int): Int =
        val u    = x(i)
        var (cmin, min) = (0, dist (u, cent(0)))
        var r2   = 0.0
        val cc   = to_c(i)
        for c <- 1 until k do
            r2 = if c == cc then (sz(c) * dist (u, cent(c))) / (sz(c) - 1)
                 else            (sz(c) * dist (u, cent(c))) / (sz(c) + 1)
            if r2 < min then { cmin = c; min = r2 }
        end for
        cmin
    end closestByR2

end KMeansPPClusterer


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KMeansPPClusterer` companion object supplies a factory function.
 */
object KMeansPPClusterer:

    import scalation.random.PermutedVecI

    private var streams: VectorI = VectorI.range (0, 1000)               // vector of random number streams

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a vector that is a permutation of the current vector of random number streams.
     *  @param stream  the stream to use in this process
     */
    def permuteStreams (stream: Int = 0): Unit =
        streams = PermutedVecI (streams, stream).igen

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `KMeansPPClusterer` with random retarts.  It will restart enough
     *  times to see a local minumum 'check' times.
     *  @param x      the vectors/points to be clustered stored as rows of a matrix
     *  @param k      the number of clusters to make
     *  @param algo   the clustering algorithm to use
     *  @param check  the number of times to see local minumum before stopping
     */
    def apply (x: MatrixD, k: Int, algo: Algorithm = HARTIGAN, check: Int = 3):
              KMeansPPClusterer = boundary:
        var seen   = 0
        var ssemin = Double.PositiveInfinity
        var kmpp_min: KMeansPPClusterer = null

        for s <- streams do
            val kmpp = new KMeansPPClusterer (x, k, algo)
            kmpp.setStream (s)
            kmpp.train ()
            val sse = kmpp.sse (x, kmpp.to_c)
//          println (s"sse = $sse")
            if sse == ssemin then
                if seen == check then break (kmpp_min)
                else                  seen += 1
            if sse < ssemin then { ssemin = sse; kmpp_min = kmpp; seen = 0 }
        end for
        kmpp_min
    end apply

end KMeansPPClusterer


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KMeansPPClustererTester` object includes test methods to aid in the
 *  testing of the `KMeansPPClusterer` class. 
 */
object KMeansPPClustererTester:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the `KMeansPPClusterer` class using sequential streams.
     *  @param x         the data matrix (each row is a data point)
     *  @param k         the number of clusters for form
     *  @param opt       optionally pass in known optimal value
     *  @param plot      whether to plot the data points in the matrix. 
     *  @param nstreams  the number of random number streams to use
     */
    def test (x: MatrixD, k: Int, opt: Double = -1, plot: Boolean = false, nstreams: Int = 1000): Unit =
        banner (s"Testing KMeansPPCluster")
        println (s"x.dims = ${x.dims}")
        println (s"     k = $k")
        if plot then new Plot (x.col(0), x.col(1))
        for algo <- Algorithm.values do
            banner (s"test (algo = $algo)")
            val statSSE = new Statistic ("sse")
            var ok = 0
            for s <- 0 until nstreams do
                val cl = new KMeansPPClusterer (x, k, algo)
                cl.setStream (s)
                cl.train ()
                val sse = cl.sse (x, cl.cluster)
                // println (s"stream $s, sse = $sse")
                statSSE.tally (sse)
                if opt != -1 && sse <= opt then ok += 1
            end for
            println (Statistic.labels)
            println (statSSE)
            println (s"min sse = ${statSSE.min}")
            if opt != -1 then println (s"optimal = $ok / $nstreams")            
        end for 
    end test

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the `KMeansPPClusterer` class using randomized streams.
     *  @param x         the data matrix (each row is a data point)
     *  @param k         the number of clusters for form
     *  @param opt       optionally pass in known optimal value
     *  @param plot      whether to plot the data points in the matrix. 
     *  @param nstreams  the number of random number streams to use
     */
    def test2 (x: MatrixD, k: Int, opt: Double = -1, plot: Boolean = false, nstreams: Int = 1000): Unit =
        banner (s"Testing KMeansPPCluster object")
        println (s"x.dims = ${x.dims}")
        println (s"     k = $k")
        if plot then new Plot (x.col(0), x.col(1))
        for algo <- Algorithm.values do
            banner (s"test2 (algo = $algo)")
            val statSSE = new Statistic ("sse")
            var ok = 0
            for s <- 0 until nstreams do
                KMeansPPClusterer.permuteStreams (s)           // randomize the streams
                val cl = KMeansPPClusterer (x, k, algo)
                val sse = cl.sse (x, cl.cluster)
                // println (s"stream $s, sse = $sse")
                statSSE.tally (sse)
                if opt != -1 && sse <= opt then ok += 1
            end for
            println (Statistic.labels)
            println (statSSE)
            println (s"min sse = ${statSSE.min}")
            if opt != -1 then println (s"optimal = $ok / $nstreams")            
        end for 
    end test2

end KMeansPPClustererTester

import KMeansPPClustererTester._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansPPClustererTest` main function is used to test the `KMeansPPClusterer`
 *  class.
 *  > runMain scalation.modeling.clustering.kMeansPPClustererTest
 */
@main def kMeansPPClustererTest (): Unit =

    import Clusterer.x

    val k   = 3
    val opt = 3.0
    test (x, k, opt)
    test2 (x, k, opt)

end kMeansPPClustererTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansPPClustererTest2` main function is used to test the `KMeansPPClusterer`
 *  class.
 *  > runMain scalation.modeling.clustering.kMeansPPClustererTest2
 */
@main def  kMeansPPClustererTest2 (): Unit =

    import scalation.random.{Normal, Bernoulli}

    val coin  = Bernoulli ()
    val dist1 = Normal (2.0, 1.0)
    val dist2 = Normal (8.0, 1.0)
    val x     = new MatrixD (50, 2)
    for i <- x.indices do x(i) = VectorD (if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen)
    val k   = 4
    val opt = 76                       // rounded up

    test (x, k, opt)
    test2 (x, k, opt)

end kMeansPPClustererTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansPPClustererTest3` main function is used to test the `KMeansPPClusterer`
 *  class.
 *  > runMain scalation.modeling.clustering.kMeansPPClustererTest3
 */
@main def kMeansPPClustererTest3 (): Unit =

    import scalation.random.{Normal, Bernoulli}

    val coin  = Bernoulli ()
    val dist1 = Normal (2.0, 1.0)
    val dist2 = Normal (8.0, 1.0)
    val x     = new MatrixD (50, 3)
    for i <- x.indices do x(i) = VectorD (if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen)

    val k   = 8
    val opt = 106                      // rounded up

    test (x, k, opt)
    test2 (x, k, opt)

end kMeansPPClustererTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kMeansPPClustererTest4` main function is used to test the `KMeansPPClusterer`
 *  class.
 *  > runMain scalation.modeling.clustering.kMeansPPClustererTest4
 */
@main def kMeansPPClustererTest4 (): Unit =

    import scalation.random.{Normal, Bernoulli}

    val coin  = Bernoulli ()
    val dist1 = Normal (2.0, 1.0)
    val dist2 = Normal (8.0, 1.0)
    val x     = new MatrixD (100, 4)
    for i <- x.indices do x(i) = VectorD (if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen,
                                          if coin.gen == 0 then dist1.gen else dist2.gen)

    val k   = 16
    val opt = 290                    // rounded up

    test (x, k, opt)
    test2 (x, k, opt)

end kMeansPPClustererTest4

