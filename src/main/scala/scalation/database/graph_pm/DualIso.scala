
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Matthew Saltz, John Miller
 *  @version 2.0
 *  @date    Thu Nov 10 14:14:46 EST 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    `Graph` Dual Isomorphism Using Mutable Sets
 */

package scalation
package database
package graph_pm

import scala.collection.mutable.Map
import scala.collection.mutable.{Set => SET}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DualIso` class provides an implementation for Subgraph Isomorphism
 *  that uses Dual Graph Simulation for pruning.
 *  @param g  the data graph  G(V, E, l)
 *  @param q  the query graph Q(U, D, k)
 */
class DualIso (g: Graph, q: Graph)
      extends GraphMatcher (g, q):

    private val duals        = new DualSim (g, q)                  // object for Dual Simulation algorithm
    private var matches      = SET [Array [SET [Int]]] ()          // initialize matches to empty
    private var noBijections = true                                // no results yet
    private var limit        = 1000000                             // limit on number of matches

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set an upper bound on the number matches to allow before quitting.
     *  @param _limit  the number of matches before quitting
     */
    def setLimit (_limit: Int): Unit = limit = _limit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply the Dual Subgraph Isomorphism algorithm to find subgraphs of data
     *  graph g that isomorphically match query graph q.  These are represented
     *  by a set of single-valued bijections {psi} where each psi function
     *  maps each query graph vertex u to a data graph vertices v.
     */
    override def bijections (): SET [Array [Int]] =
        matches = SET [Array [SET [Int]]] ()                       // initialize matches to empty
        val φ = duals.feasibleMates                                // initial mappings from label match
        saltzDualIso (duals.prune (φ), 0)                          // recursively find all bijections
        val psi = simplify (matches)                               // pull bijections out matches
        noBijections = false                                       // results now available
        psi                                                        // return the set of bijections
    end bijections

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply the Dual Subgraph Isomorphism pattern matching algorithm to find
     *  the mappings from the query graph q to the data graph g.  These are
     *  represented by a  multi-valued function φ that maps each query graph
     *  vertex u to a set of data graph vertices {v}.
     *  @param ignoreEdgeLabels  whether to ingore edge labels during matching    // FIX - check this
     */
    override def mappings (ignoreEdgeLabels: Boolean = false): Array [SET [Int]] =
        var psi: SET [Array [Int]] = null                          // mappings from Dual Simulation
        if noBijections then psi = bijections ()                   // if no results, create them
        merge (psi)                                                // merge bijections to create mappings
    end mappings

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the count of the number of matches.
     */
    def numMatches: Int = matches.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Refine the mappings φ using the Dual Subgraph Isomorphism algorithm.
     *  Enumerate bijections by using an Ullmann-like recursion that uses Dual
     *  Graph Simulation for pruning.
     *  @param φ    array of mappings from a query vertex u_q to { graph vertices v_g }
     *  @param depth  the depth of recursion
     */
    private def saltzDualIso (φ: Array [SET [Int]], depth: Int): Unit =
        if depth == q.size then
            if ! φ.isEmpty then
                matches += φ
                if matches.size % CHECK == 0 then println ("saltzDualIso: matches so far = " + matches.size)
        else if ! φ.isEmpty then
            breakable {
                for i <- φ (depth) if ! contains (φ, depth, i) do
                    val φCopy = φ.map (x => x)                     // make a copy of φ
                    φCopy (depth) = SET [Int] (i)                  // isolate vertex i
                    if matches.size >= limit then break ()         // quit if at LIMIT
                    saltzDualIso (duals.prune (φCopy), depth + 1)  // solve recursively for the next depth
                end for
            } // breakable
        end if
    end saltzDualIso

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether vertex v is contained in any φ(i) for the previous depths.
     *  @param φ    array of mappings from a query vertex u_q to { graph vertices v_g }
     *  @param depth  the current depth of recursion
     *  @param v      the vertex to check
     */
    private def contains (φ: Array [SET [Int]], depth: Int, v: Int): Boolean =
        var found = false
        breakable {
            for i <- 0 until depth do if φ(i) contains v then
                found = true
                break ()
            end for
        } // breakble
        found
    end contains

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an array to hold matches for each vertex u in the query graph
     *  q and initialize it to contain all empty sets.  Then for each bijection,
     *  add each element of the bijection to its corresponding match set.
     *  @param psi  the set of bijections
     */
    private def merge (psi: SET [Array [Int]]): Array [SET [Int]] =
        val matches = Array.ofDim [SET [Int]] (q.size).map (_ => SET [Int] ())
        val bijects = if psi != null then psi else bijections ()
        for b <- bijects; i <- b.indices do matches(i) += b(i)
        matches
    end merge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pull the bijections out of the complete match set.
     *  @param matches  the complete match set embedding all bijections
     */
    private def simplify (matches: SET [Array [SET [Int]]]): SET [Array [Int]] =
        matches.map (m => m.map (set => set.iterator.next ()))
    end simplify

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The prune0 and prune are not needed, pruning is delegated to incorporated graph
     *  simulation algorithms.
     *  @param φ  array of mappings from a query vertex u_q to { graph vertices v_g }
     */
    def prune0 (φ: Array [SET [Int]]): Array [SET [Int]] = throw new UnsupportedOperationException ()
    def prune (φ: Array [SET [Int]]): Array [SET [Int]] = throw new UnsupportedOperationException ()

end DualIso


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dualIsoTest` main function is used to test the `DualIso` class.
 *  > runMain scalation.database.graph_pm.dualIsoTest
 */
@main def dualIsoTest (): Unit =

    val g = new Graph (Array (SET (1, 3),                        // ch(0)
                              SET (2),                           // ch(1)
                              SET (),                            // ch(2)
                              SET (),                            // ch(3)
                              SET (2),                           // ch(4)
                              SET (4)),                          // ch(5)
                       Array (10.0, 11.0, 11.0, 11.0, 11.0, 10.0),
                       Map ((0, 1) -> -1.0,
                            (0, 3) -> -1.0,
                            (1, 2) -> -1.0,
                            (4, 2) -> -1.0,
                            (5, 4) -> -1.0),                     // change from -1 to -2 filter out vertices
                       false, "g")

    val q = new Graph (Array (SET (1, 3),                        // ch(0)
                              SET (2),                           // ch(1)
                              SET (),                            // ch(2)
                              SET (2)),                          // ch(3)
                       Array (10.0, 11.0, 11.0,11.0),            // vertex labels
                       Map ((0, 1) -> -1.0,                      // edge labels
                            (0, 3) -> -1.0,
                            (1, 2) -> -1.0,
                            (3, 1) -> -1.0),
                       false, "g")

    println (s"g.checkEdges   = ${g.checkEdges}")
    println (s"g.checkElabels = ${g.checkElabels}")
    g.printG ()
    println (s"q.checkEdges   = ${q.checkEdges}")
    println (s"q.checkElabels = ${q.checkElabels}")
    q.printG ()

    val matcher = new DualIso (g, q)                             // Dual Subgraph Isomorphism Pattern Matcher
    val psi = time { matcher.bijections () }                     // time the matcher
    println ("Number of Matches: " + matcher.numMatches)
    for p <- psi do println (s"psi = ${stringOf (p)}")

    matcher.test ("DualIso")

end dualIsoTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dualIsoTest2` main function is used to test the `DualIso` class.
 *  > runMain scalation.database.graph_pm.dualIsoTest2
 */
@main def dualIsoTest2 (): Unit =

    import graph_pm.{ExampleGraphD => EX_GRAPH}
    import MatchAnswers._

    val g = EX_GRAPH.g2
    val q = EX_GRAPH.q2

    println (s"g.checkEdges   = ${g.checkEdges}")
    println (s"g.checkElabels = ${g.checkElabels}")
    g.printG ()
    println (s"q.checkEdges   = ${q.checkEdges}")
    println (s"q.checkElabels = ${q.checkElabels}")
    q.printG ()

    val matcher = new DualIso (g, q)                             // Dual Subgraph Isomorphism Pattern Matcher
    val psi = time { matcher.bijections () }                     // time the matcher
    println ("Number of Matches: " + matcher.numMatches)
    for p <- psi do println (s"psi = ${stringOf (p)}")

    matcher.test ("DualIso", shift (dualIso))

end dualIsoTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dualIsoTest3` main function is used to test the `DualIso` class.
 *  > runMain scalation.database.graph_pm.dualIsoTest3
 */
@main def dualIsoTest3 (): Unit =

    import graph_pm.{ExampleGraphD => EX_GRAPH}

    val g = EX_GRAPH.g1
    val q = EX_GRAPH.q1

    println (s"g.checkEdges   = ${g.checkEdges}")
    println (s"g.checkElabels = ${g.checkElabels}")
    g.printG ()
    println (s"q.checkEdges   = ${q.checkEdges}")
    println (s"q.checkeLabels = ${q.checkElabels}")
    q.printG ()

    val matcher = new DualIso (g, q)                             // Dual Subgraph Isomorphism Pattern Matcher
    val psi = time { matcher.bijections () }                     // time the matcher
    println ("Number of Matches: " + matcher.numMatches)
    for p <- psi do println (s"psi = ${stringOf (p)}")

    matcher.test ("DualIso")

end dualIsoTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dualIsoTest4` main function is used to test the `DualIso` class.
 *  > runMain scalation.database.graph_pm.dualIsoTest4
 *
@main def dualIsoTest4 (): Unit =

    val mgGen = new GraphGen [Double]

    val gSize     = 1000         // size of the data graph
    val qSize     =   10         // size of the query graph
    val nLabels   =  100         // number of distinct vertex labels
    val eLabels   =   10         // number of distinct edge labels
    val gAvDegree =    5         // average vertex out degree for data graph
    val qAvDegree =    2         // average vertex out degree for query graph

    val g = mgGen.genRandomGraph (gSize, nLabels, eLabels, gAvDegree, false, "g")
    val q = mgGen.genBFSQuery (qSize, qAvDegree, g, false, "q")

    println (s"q.checkEdges = ${q.checkEdges}")
    q.printG ()

    val matcher = new DualIso (g, q)                             // Dual Subgraph Isomorphism Pattern Matcher
    val psi = time { matcher.bijections () }                     // time the matcher
    println ("Number of Matches: " + matcher.numMatches)
    for p <- psi do println (s"psi = ${stringOf (p)}")

end dualIsoTest4
 */

