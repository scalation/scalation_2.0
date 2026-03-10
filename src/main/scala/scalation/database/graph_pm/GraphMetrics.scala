
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Nov 11 19:03:45 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Computes Graph Metrics
 */

package scalation
package database
package graph_pm

import scala.collection.mutable.{ArrayBuffer, Queue}
import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GraphMetrics` class provides methods for determining graph metrics that
 *  can be efficiently computed using Breadth-First Search (BFS).  This works for 
 *  undirected graphs.  If a directed graph is passed in, it will be converted to 
 *  a corresponding undirected graph.
 *  @param g            the graph whose metrics are sought
 *  @param isUndirected indicates whether the graph is undirected
 */
class GraphMetrics (val g: Graph, isUndirected: Boolean = true):

    private val debug = debugf ("GraphMetrics", false)        // debug flag
    private val n     = g.size                                // number of vertices in g
    private val qu    = new Queue [Int] ()                    // a queue supporting BFS
    private val go    = Array.ofDim [Boolean] (n)             // vertex visitation flag
    private val len   = Array.ofDim [Int] (n)                 // path-length from vertex i to j
    
    if ! isUndirected then
        for i <- 0 until n; j <- g.ch (i) do g.ch (j) += i    // converting directed to undirected

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the diameter of graph g (i.e., maximum eccentricity).  This also
     *  equals the "longest shortest path" between any pair of vertices in graph g.
     */
    def diam: Int =
        var max = ecc (0)
        for i <- 1 until n do { val ecc_i = ecc (i); if ecc_i > max then max = ecc_i }
        max
    end diam

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the radius of graph g (i.e., minimum eccentricity).
     */
    def rad: Int =
        var min = ecc (0)
        for i <- 1 until n do { val ecc_i = ecc (i); if ecc_i < min then min = ecc_i }
        min
    end rad

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the central vertices, those with eccentricities equal to the radius.
     */
    def central: Array [Int] =
        val centr = ArrayBuffer [Int] ()
        val ecc_v = Array.ofDim [Int] (n)
        for i <- 0 until n do ecc_v(i) = ecc (i)
        var rd = ecc_v(0)
        for i <- 1 until n do if ecc_v(i)  < rd then rd = ecc_v(i)
        for i <- 0 until n do if ecc_v(i) == rd then centr += i
        centr.toArray
    end central 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the eccentricity of vertex i (the length of the longest path from
     *  vertex i to any other vertex).
     *  @param i  the vertex whose eccentricity is sought
     */
    def ecc (i: Int): Int =
        for j <- 0 until n do { go(j) = true; len(j) = 0 }
        qu.enqueue (i)
        while ! qu.isEmpty do visit ()                       // visit vertices in BFS order
        var ecc_i = len(0)
        for j <- 1 until n do if len(j) > ecc_i then ecc_i = len(j)
        debug ("ecc", s"($i) = $ecc_i")
        ecc_i                                                // return max path-length
    end ecc

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Visit the next vertex (at the head of queue qu), mark it, compute the
     *  path-length len for each of its children and put them in the queue. 
     */
    private def visit (): Unit =
        val j = qu.dequeue ()                                // take next vertex from queue
        go(j) = false                                        // mark as visited
        val len_c = len(j) + 1                               // path-length to child vertices
        debug ("visit", s"len ($j) = ${len(j)}")
        for c <- g.ch(j) do                                  // for each child of vertex j
            if go(c) && len(c) == 0 then
                len(c) = len_c                               // distance from vertex i to c
                qu.enqueue (c)                               // put child c in queue
        end for
    end visit

end GraphMetrics


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GraphMetrics` companion object provides basic statistics about graphs.
 */
object GraphMetrics:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return basic statistics about graph g, currently (1) number of vertices,
     *  (2) number of edges, and (3) average out-degree.
     *  @param g  the given graph
     */
    def stats (g: Graph): Seq [AnyVal] =
        val (n, e) = (g.size, g.nEdges)
        Seq (n, e, e / n.toDouble)
    end stats

end GraphMetrics


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphMetricsTest` main function is used to test the `GraphMetrics` class.
 *  @see http://math.stackexchange.com/questions/240556/radius-diameter-and-center-of-graph
 *  > runMain scalation.database.graph_pm.graphMetricsTest
 */
@main def graphMetricsTest (): Unit =

    val g = ExampleGraphD.g1

    g.printG ()
    val bfs = new GraphMetrics (g)

    // Compute the diameter of graph g
    var dia = 0
    cfor (0, 10) { _ =>
        time { dia = bfs.diam }
        println (s"diameter  = $dia")
    } // cfor
    
    // Compute the radius of graph g
    var rd = 0
    cfor (0, 10) { _ =>
        time { rd = bfs.rad }
        println (s"radius  = $rd")
    } // cfor

    // Return the central vertices of graph g
    var ctr: Array [Int] = null
    cfor (0, 10) { _ =>
        time { ctr = bfs.central }
        println (s"central  = ${stringOf (ctr)}")
    } // cfor

end graphMetricsTest

