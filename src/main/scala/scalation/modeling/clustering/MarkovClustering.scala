
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue May 29 14:45:32 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Markov Clustering Algorithm (MCL) for Graphs
 *
 *  @see     www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf
 *  @see     www.ebi.ac.uk/enright/docs/stijnmcl.pdf
 */

package scalation
package modeling
package clustering

import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{breakable, break}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MarkovClusterer` class implements a Markov Clustering Algorithm 'MCL'
 *  and is used to cluster nodes in a graph.  The graph is represented as an
 *  edge-weighted adjacency matrix (a non-zero cell indicates nodes i and j are
 *  connected).
 *  The primary constructor takes either a graph (adjacency matrix) or a
 *  Markov transition matrix as input.  If a graph is passed in, the normalize
 *  method must be called to convert it into a Markov transition matrix.
 *  Before normalizing, it may be helpful to add self loops to the graph.
 *  The matrix (graph or transition) may be either dense or sparse.
 *  See the `MarkovClustererTest` object at the bottom of the file for examples.
 *  @param t  either an adjacency matrix of a graph or a Markov transition matrix
 *  @param k  the strength of expansion
 *  @param r  the strength of inflation
 */
class MarkovClusterer (t: MatrixD, k: Int = 2, r: Double = 2.0)
      extends Clusterer:

    private val debug   = debugf ("MarkovClusterer", false)       // debug flag
    private val MAX_IT  = 200                                     // maximum number of iterations
    private val EPSILON = 1E-7                                    // number close to zero

    private val clustr = Array.ofDim [Int] (t.dim2)               // vector of cluster assignments

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cluster the nodes in the graph by interpreting the processed matrix t.
     *  Nodes not clustered will be in group 0; otherwise, they will be grouped
     *  with their strongest positive attractor.
     */
    def train (): Unit =
        val force = new VectorD (t.dim2)                           // force of attractor, initially 0
        var group = 1                                              // first real group is 1
        for i <- t.indices do
            var found = false
            for j <- t.indices2 do
                if t(i, j) > force(j) then                         // if attractor has greater force
                    clustr(j) = group                              // assign node j to this group
                    force(j) = t(i, j)                             // make t(i, j) the new force
                    found = true                                   // a group was found for this row
            end for
            if found then group += 1                               // increment the group number
        end for
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cluster assignment vector.  Should only be called after `train`.
     */
    def cluster: Array [Int] = clustr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the centroids.  Should only be called after 'train'.
     */
    def centroids: MatrixD = throw new UnsupportedOperationException ("not applicable")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sizes of the centroids.  Should only be called after 'train'.
     */
    def csize: VectorI = throw new UnsupportedOperationException ("not applicable")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add self-loops by setting the main diagonal to the weight parameter.
     *  @param weight  the edge weight on self-loops to be added.
     */
    def addSelfLoops (weight: Double = 1.0): Unit =
        t(?, ?) = weight                                // add self-loops (make diagonal 1)
    end addSelfLoops

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Normalize the matrix t so that each column sums to 1, i.e., convert
     *  the adjacency matrix of a graph into a Markov transition matrix.
     */
    def normalize (): Unit =
        val sum = new VectorD (t.dim2)                  // to hold column sums
        for j <- t.indices2 do sum(j) = t(?, j).sum
        for j <- t.indices2 do t(?, j) = t(?, j) / sum(j)
    end normalize

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expansion tends to grow clusters (flow along path in graph).
     *  Expand by raising the matrix t to the k-th power.
     */
    private def expand (): Unit = t~^^k

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inflation tends to strengthen strong connections and weaken weak ones.
     *  Inflate by raising each cell to the r-th power and normalize column-by-column.
     *  If a cell is close to zero, set it to zero (prune).  Also, detect
     *  convergence by making sure that the variance in each column is small
     *  enough.
     */
    private def inflate (): Boolean =
        var done = true
        for j <- t.indices2 do
            var sum   = 0.0                             // column sum
            var sumSq = 0.0                             // column sum of squares
            var n     = 0.0                             // number of non-zero entries in column

            for i <- t.indices do
                if t(i, j) < EPSILON then
                    t(i, j) = 0.0                       // prune this cell
                else
                    t(i, j) = t(i, j) ~^ r              // raise cell (i, j) to the r-th power
                    sum   += t(i, j)                    // collect sum
                    sumSq += t(i, j) * t(i, j)          // collect sum of squares
                    n += 1.0
            end for

            for i <- t.indices do t(i, j) /= sum                    // normalize
            if sumSq - (sum * sum) / n > EPSILON then done = false  // variance in column too high
        end for
        done                                            // whether convergence has been detected
    end inflate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the processed matrix t.  The matrix is processed by repeated
     *  steps of expansion and inflation until convergence is detected.
     */
    def processMatrix (): MatrixD = 
        breakable {
            for l <- 1 to MAX_IT do
                expand ()                                // expansion step
                if inflate () then break ()              // inflation step with convergence check
                debug ("processMatrix", s"($l): t = $t")
            end for
        } // breakable
        t
    end processMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** This clustering method is not applicable to graph clustering.
     *  @param y  unused parameter
     */
    def classify (y: VectorD): Int = throw new UnsupportedOperationException ()

end MarkovClusterer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `markovClustererTest` object is used to test the `MarkovClusterer` class.
 *  @see www.cs.ucsb.edu/~xyan/classes/CS595D-2009winter/MCL_Presentation2.pdf
 ^  > runMain scalation.modeling.clustering.markovClustererTest
 */
@main def markovClustererTest (): Unit =

    // Test the MCL Algorithm on a graph represented as an adjacency matrix.

    val g = MatrixD ((12, 12),
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,  0.0,
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,
        0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0,  0.0,
        0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0,  0.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  0.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,  0.0,
        0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,  0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,  1.0,
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,  0.0,
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,  1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  0.0) 

    println ("-" * 60)
    println ("g = " + g)
    val mg = new MarkovClusterer (g)
    mg.addSelfLoops ()
    mg.normalize ()
    println ("result  = " + mg.processMatrix ())
    mg.train ()
    println ("cluster = " + stringOf (mg.cluster))

    // Test the MCL Algorithm on a Markov transition matrix.

    val t = MatrixD ((12, 12),
        0.2, 0.25, 0.0,  0.0, 0.0, 0.333, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0,
        0.2, 0.25, 0.25, 0.0, 0.2, 0.0,   0.0,  0.0, 0.0, 0.0,  0.0, 0.0,
        0.0, 0.25, 0.25, 0.2, 0.2, 0.0,   0.0,  0.0, 0.0, 0.0,  0.0, 0.0,
        0.0, 0.0,  0.25, 0.2, 0.0, 0.0,   0.0,  0.2, 0.2, 0.0,  0.2, 0.0,
        0.0, 0.25, 0.25, 0.0, 0.2, 0.0,   0.25, 0.2, 0.0, 0.0,  0.0, 0.0,
        0.2, 0.0,  0.0,  0.0, 0.0, 0.333, 0.0,  0.0, 0.0, 0.25, 0.0, 0.0,
        0.2, 0.0,  0.0,  0.0, 0.2, 0.0,   0.25, 0.0, 0.0, 0.25, 0.0, 0.0,
        0.0, 0.0,  0.0,  0.2, 0.2, 0.0,   0.0,  0.2, 0.2, 0.0,  0.2, 0.0,
        0.0, 0.0,  0.0,  0.2, 0.0, 0.0,   0.0,  0.2, 0.2, 0.0,  0.2, 0.333,
        0.2, 0.0,  0.0,  0.0, 0.0, 0.333, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0,
        0.0, 0.0,  0.0,  0.2, 0.0, 0.0,   0.0,  0.2, 0.2, 0.0,  0.2, 0.333,
        0.0, 0.0,  0.0,  0.0, 0.0, 0.0,   0.0,  0.0, 0.2, 0.0,  0.2, 0.333) 

    println ("-" * 60)
    println ("t = " + t)
    val mt = new MarkovClusterer (t)
    println ("result  = " + mt.processMatrix ())
    mt.train ()
    println ("cluster = " + stringOf (mt.cluster))

    // Test the MCL Algorithm on a graph represented as a sparse adjacency matrix.

//  import scala.collection.mutable.ListMap
//  val x = new SparseMatrixD (12, 12)
//  x(0)  = ListMap ((1, 1.0), (5, 1.0), (6, 1.0), (9, 1.0))
//  x(1)  = ListMap ((0, 1.0), (2, 1.0), (4, 1.0))
//  x(2)  = ListMap ((1, 1.0), (3, 1.0), (4, 1.0))
//  x(3)  = ListMap ((2, 1.0), (7, 1.0), (8, 1.0), (10, 1.0))
//  x(4)  = ListMap ((1, 1.0), (2, 1.0), (6, 1.0), (7, 1.0))
//  x(5)  = ListMap ((0, 1.0), (9, 1.0))
//  x(6)  = ListMap ((0, 1.0), (4, 1.0), (9, 1.0))
//  x(7)  = ListMap ((3, 1.0), (4, 1.0), (8, 1.0), (10, 1.0))
//  x(8)  = ListMap ((3, 1.0), (7, 1.0), (10, 1.0), (11, 1.0))
//  x(9)  = ListMap ((0, 1.0), (5, 1.0), (6, 1.0))
//  x(10) = ListMap ((3, 1.0), (7, 1.0), (8, 1.0), (11, 1.0))
//  x(11) = ListMap ((8, 1.0))

//  println ("-" * 60)
//  println ("x = " + x)
//  val mx = new MarkovClusterer (x)
//  mx.addSelfLoops ()
//  mx.normalize ()
//  println ("result  = " + mx.processMatrix ())
//  println ("cluster = " + stringOf (mx.cluster))

end markovClustererTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `markovClustererTest2` object is used to test the `MarkovClusterer` class.
 ^  > runMain scalation.modeling.clustering.markovClustererTest2
 */
@main def markovClustererTest2 (): Unit =

    // Test the MCL Algorithm on a randomly generated graph represented as an adjacency matrix.

    val rg = new RandomGraph (1000, .05, 10)
    val y  = rg.gen ()
    println ("-" * 60)
    val t0 = System.nanoTime ()
    val my = new MarkovClusterer (y)
    my.addSelfLoops ()
    my.normalize ()
    my.processMatrix ()
    my.train ()
    val cluster = my.cluster
    println ("Elapsed time = " + (System.nanoTime - t0) + " ns")
    println ("-" * 60)
    println ("cluster = " + stringOf (cluster))

end markovClustererTest2

