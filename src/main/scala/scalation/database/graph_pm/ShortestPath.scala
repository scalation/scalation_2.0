
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 10 14:26:34 EDT 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Single-Source Shortest Path (SSSP) Problem
 */

package scalation
package database
package graph_pm

import scala.collection.mutable.{Map, PriorityQueue}
import scala.collection.mutable.{Set => SET}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ShortestPath` class is used to solve shortest path problems for graphs
 *  stored in matrices.  It solves the Single-Source Shortest Path (SSSP) problem
 *  for directed graphs.  The `ShortestPath`  companion object is used to form
 a matrix from a `Graph`.
 *----------------------------------------------------------------------------
 *  The edge cost/distance (must be non-negative) is stored in a dense matrix.
 *  Dijkstra's Algorithm is used:  @see en.wikipedia.org/wiki/Dijkstra%27s_algorithm
 *----------------------------------------------------------------------------
 *  To extend to multi-digraphs, each multi-edge between a pair vertices has it
 *  own edge weight.  Take the minimum when forming the corresponding matrix.
 *  @see thescipub.com/PDF/jcssp.2013.377.382.pdf
 *----------------------------------------------------------------------------
 *  @param c  the cost/distance matrix, where a value of zero implies no connection.
 *            If the actual distance is zero, use a very small number instead.
 *  @param s  the single-source vertex
 */
class ShortestPath (c: MatrixD, s: Int):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Item` inner case class has two fields, vertex id and distance from
     *  vertex s (the source) as well as a compare method based on distance.
     *  @param id  the id of the vertex
     *  @param dd  the vertex's distance from vertex s
     */
    case class Item (id: Int, dd: Double) extends Ordered [Item]:
        def compare (v: Item) = v.dd compare dd
    end Item

    private val debug = debugf ("ShortesPath", true)         // debug function
    private val MAX   = Double.MaxValue                      // infinity (indicates no path so far)
    private val n     = c.dim                                // the number of vertices
    private val q     = PriorityQueue.empty [Item]           // priority queue ordered by distance

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the shortest path from vertex s to each vertex j returning
     *  the vector d giving the distance from s to all other vertices and
     *  the vector p of predecessor vertices.  The path from s to each vertex
     *  can be deduced from the p vector.
     */
    def spath (): (VectorD, VectorI) =
        val d = c(s)                                         // the distance from vertex s to each vertex j
        val p = new VectorI (n)                              // create predecessor vector
        for j <- p.indices if c(s, j) == 0.0 do d(j) = MAX   // set distance to infinity if no direct edge from s
        d(s)  = 0.0                                          // zero distance from s to s
        debug ("spath", s"d = $d, p = $p, c = $c")

        for j <- p.indices do
            p(j) = if d(j) != MAX then s else -1             // initialize predecessor vertices
            q += Item (j, d(j))                              // add each vertex j incl. its distance to q
        end for

        var go = true                                        // set the go flag to true
        while go && q.nonEmpty do                            // iteratively, try to find a shortcut

            val v = q.dequeue ()                             // vertex v in q with least distance dd from s
            if v.dd == MAX then go = false                   // no shortcuts left, so quit
            else
                for j <- p.indices if c(v.id, j) > 0.0 do    // check vertex v's neighbors
                    val alt = v.dd + c(v.id, j)              // compute alternate distance from s to j
                    if alt < d(j) then
                        p(j) = v.id; d(j) = alt; q += Item (j, d(j))
                end for
                debug ("spath", s"updated distance (s, v) : ($s, ${v.id}) = $d")
            end if
            
        end while
        (d, p)                                               // return distance and predecessor vectors
    end spath

end ShortestPath


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ShortestPath` companion object provides factory methods for the 
 *  `ShortestPath` class.
 */
object ShortestPath:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ShortestPath` object from a digraph.  First convert the graph's
     *  adjacency set representation to an adjacency matrix with elabel edge cost.
     *  @param g  the digraph to use
     *  @param s  the source vertex for the digraph
     */
    def apply (g: Graph, s: Int): ShortestPath =
        val n = g.size
        val c = new MatrixD (n, n)                           // cost/distance matrix
        for u <- c.indices; v <- g.ch(u) do
            c(u, v) = g.elabel((u, v)).toDouble              // edge cost u -> v is elabel
        end for
        new ShortestPath (c, s)
    end apply

end ShortestPath


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `shortestPathTest` main function is used to test the `ShortestPath` class.
 *  Input is in the form of matrices (`MatrixD`).
 *  > runMain scalation.database.graph_pm.ShortestPathTest
 */
@main def shortestPathTest (): Unit =

    // matrix representation for the graph, where d_ij = distance from i to j

    val c = MatrixD ((3, 3),   0.0,   2.0, 100.0,
                             100.0,   0.0,   3.0,
                               4.0, 100.0,   0.0)
    println (c)
    val sp = new ShortestPath (c, 0)
    println ("(d, p) = " + sp.spath ())            // shortest distance from s to all vertices)

end shortestPathTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `shortestPathTest2` main function is used to test the `ShortestPath` class.
 *  Input is in the form of graphs (`Graph`).
 *  @see http://thescipub.com/PDF/jcssp.2013.377.382.pdf (Fig. 1)
 *  > runMain scalation.database.graph_pm.shortestPathTest2
 */
@main def shortestPathTest2 (): Unit =

    val g = new Graph (Array (SET (1, 2),                       // ch(0)
                              SET (0, 3),                       // ch(1)
                              SET (0, 1, 3),                    // ch(2)
                              SET (1, 2)),                      // ch(3)
                       Array (10.0, 11.0, 12.0, 13.0),          // for A, B, C, D
                       Map ((0, 1) -> 65.0,
                            (0, 2) -> 46.0,
                            (1, 0) -> 39.0,
                            (1, 3) -> 14.0,
                            (2, 0) -> 46.0,
                            (2, 1) -> 37.0,
                            (2, 3) -> 21.0,
                            (3, 1) -> 19.0,
                            (3, 2) -> 15.0),
                       false, "g")

    g.checkEdges
    g.checkElabels
    g.printG ()

    val s  = 0
    val sp = ShortestPath (g, s)
    println ("(d, p) = " + sp.spath ())        // shortest distance from s to all vertices

end shortestPathTest2

