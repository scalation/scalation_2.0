
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Matthew Saltz, John Miller, Aravind Kalimurthy
 *  @version 2.0
 *  @date    Thu Jul 25 11:28:31 EDT 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Random Graph Generator
 */

package scalation
package database
package graph_pm

import scala.collection.mutable.{Map, Queue}
import scala.collection.mutable.{Set => SET}
import scala.math.abs
import scala.runtime.ScalaRunTime.stringOf

import scalation.random._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GraphGen` class is used to build random graph with various characteristics.
 *  Needs to generate vertex labels of various types including `Int`, `Double`,
 *  `String` based on the type.
 *  @param typeSelector  the variable for type matches (work around for generic erasure)
 *  @param stream        the random number stream to use (0 to 999)
 */
class GraphGen (typeSelector: Char, stream: Int = 0):

    private val flaw  = flawf ("GraphGen")                      // flaw function
//  private val ran   = Random (stream)                         // random number generator with interval (0, 1)
    private val rng   = Randi0 (stream = stream)                // random number/integer generator: 0, 1, ...
    private val rsg   = RandomSet (stream = stream)             // random set generator
    private var rlg: Variate = null                             // random label generator for typeSelector parameter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the Random Variate Generator, using the one based on the type.
     *  @param nLabs  the number of labels to generate
     */
    def setVariate (nLabs: Int): Unit =
        typeSelector match
        case 'I' => rlg = Randi (1, nLabs, stream)
        case 'D' => rlg = Randi (1, nLabs, stream)
        case 'S' => rlg = RandomWord (nLabs, stream = stream)
        case _   => flaw ("setVariate", "label type not supported")
        end match
    end setVariate

    //------------------------------------------------------------------------
    // Methods generating random graphs where the number of outgoing edges (the degree)
    // for a vertex is uniformly distributed.
    //------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generates a random graph with the specified size (number of vertices), 
     *  average degree and labels evenly distributed across vertices from 0 to
     *  nLabels - 1.  Not necessarily a connected graph.
     *  @param size      the number of vertices to generate
     *  @param nLabels   the number of vertex labels (distributed uniformly)
     *  @param eLabels   the number of edge labels (distributed uniformly)
     *  @param avDegree  the average degree
     *  @param inverse   whether to create inverse adjacency (parents)
     *  @param name      the name of the graph
     *  @param locality  whether to select children from anywhere in graph or locally
     */
    def genRandomGraph (size: Int, nLabels: Int, eLabels: Int, avDegree: Int, inverse: Boolean = false,
                        name: String = "g", locality: Boolean = true): Graph =
        val hwidth = avDegree + 4                                   // half width for generation window
        val width  = 2 * hwidth                                     // full width
        val ch     = Array.fill [SET [Int]] (size)(SET [Int] ())    // holder for children

        for i <- ch.indices do
            val degree = rng.igen1 (avDegree * 2 + 1)               // desired out-degree for vertex i
            if locality then
                val delta  = rsg.igen (degree, width, i)            // offsets for children
                for j <- delta do
                    var ch_i = abs (i + j - hwidth)
                    if ch_i >= size then ch_i -= ch_i - size + 1
                    if ch_i == i    then ch_i = abs (i - 1)
                    ch(i) += ch_i                                   // generate children randomly with locality
                end for
            else
                ch(i) = rsg.igen (degree, size-1, i)                // generate children uniformly across graph
        end for

        val label  = randDistLabels (size, nLabels)                 // randomly assign vertex labels
        val elabel = randDisteLabels (ch, eLabels)                  // randomly assign edge labels
        new Graph (ch, label, elabel, inverse, name)
    end genRandomGraph

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generates a random connected graph by using genRandomGraph and
     *  checking whether it is connected.
     *  @param size      the number of vertices to generate
     *  @param nLabels   the number of vertex labels (distributed uniformly)
     *  @param eLabels   the number of edge labels (distributed uniformly)
     *  @param avDegree  the average degree
     *  @param inverse   whether to create inverse adjacency (parents)
     *  @param name      the name of the graph
     */
    def genRandomConnectedGraph (size: Int, nLabels: Int, eLabels: Int, avDegree: Int,
                                 inverse: Boolean = true, name: String = "g"): Graph =
        var g: Graph = null
        var it = 0
        while
            g = genRandomGraph (size, nLabels, eLabels, avDegree, inverse, name)
            it += 1
            g.isConnected == false
        do ()
        println (s"genRandomConnectedGraph: $it iterations")
        g
    end genRandomConnectedGraph
  
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generates a random graph with labels distributed based on a power law
     *  distribution (currently with the magic number 2.1 for the power law exponent).
     *  @param size      the number of vertices to generate
     *  @param nLabels   the number of labels (distributed according to power law)
     *  @param eLabels   the number of edge labels (distributed uniformly)
     *  @param avDegree  the average out-degree
     *  @param distPow   the power/exponent (2.1 is used in WWW graph pg 72 of m&m graph data)
     *  @param inverse   whether to create inverse adjacency (parents)
     *  @param name      the name of the graph
     */
    def genRandomGraph_PowLabels (size: Int, nLabels: Int, eLabels: Int, avDegree: Int, distPow: Double = 2.1,
                                  inverse: Boolean = true, name: String = "g"): Graph =
        val ch = Array.ofDim [SET [Int]] (size)
        for i <- ch.indices do
            val degree = rng.igen1 (avDegree * 2 + 1)                    // out-degree for vertex i
            ch(i)      = rsg.igen (degree, size - 1, i)                  // children of vertex i
        end for

        val label = powDistLabels (size, nLabels, distPow)              // power law for vertex lables
        val elabel = randDisteLabels (ch, eLabels)                      // randomly assign edge labels
        new Graph (ch, label, elabel, inverse, name)
    end genRandomGraph_PowLabels

    //------------------------------------------------------------------------
    // Methods generating random graphs where the number of outgoing edges (the degree)
    // for a vertex follows a power law distribution.
    //------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generates a graph with power law degree distribution with exponent distPow
     *  and uniformly distributed labels.
     *  @param size       the number of vertices 
     *  @param nLabels    the number of labels (distributed uniformly)
     *  @param eLabels    the number of edge labels (distributed uniformly)
     *  @param maxDegree  the maximum allowed degree for any vertex
     *  @param distPow    the power/exponent (2.1 is used in WWW graph pg 72 of m&m graph data)
     *  @param inverse    whether to create inverse adjacency (parents)
     *  @param name       the name of the graph
     */
    def genPowerLawGraph (size: Int, nLabels: Int, eLabels: Int, maxDegree: Int, distPow: Double = 2.1,
                          inverse: Boolean = true, name: String = "g"): Graph =
        val powLaw = PowerLaw (1, maxDegree, distPow)
        val ch = Array.ofDim [SET [Int]] (size)
        for i <- ch.indices do
            val degree = powLaw.igen                                    // out-degree
            ch(i)      = rsg.igen (degree, size - 1, i)                 // children of vertex i
        end for

        val label  = randDistLabels (size, nLabels)                     // randomly assign vertex labels
        val elabel = randDisteLabels (ch, eLabels)                      // randomly assign edge labels
        new Graph (ch, label, elabel, inverse, name)
    end genPowerLawGraph

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generates a graph with power law degree distribution with exponent distPow
     *  and power law distributed labels.
     *  @param size       the number of vertices 
     *  @param nLabels    the number of labels (distributed according to power law)
     *  @param eLabels    the number of edge labels (distributed uniformly)
     *  @param maxDegree  the maximum allowed degree for any vertex
     *  @param distPow    the power/exponent (2.1 is used in WWW graph pg 72 of m&m graph data)
     *  @param inverse    whether to create inverse adjacency (parents)
     *  @param name       the name of the graph
     */
    def genPowerLawGraph_PowLabels (size: Int, nLabels: Int, eLabels: Int, maxDegree: Int, distPow: Double = 2.1,
                                    inverse: Boolean = false, name: String = "g"): Graph =
        val powLaw = PowerLaw (1, maxDegree, distPow)
        val ch = Array.ofDim [SET [Int]] (size)
        for i <- ch.indices do
            val degree = powLaw.igen                                    // out-degree
            ch(i)      = rsg.igen (degree, size - 1, i)                 // children of vertex i
        end for

        val label  = powDistLabels (size, nLabels, distPow)             // power law for vertex lables
        val elabel = randDisteLabels (ch, eLabels)                      // randomly assign edge labels
        new Graph (ch, label, elabel, inverse, name)
    end genPowerLawGraph_PowLabels

    //------------------------------------------------------------------------
    // Methods for generating/extracting query graphs from data graphs.
    // Ensures that matches will exist. 
    //------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a graph g, performs a breadth first search starting at a random vertex
     *  until the breadth first tree contains size vertices.  At each junction,
     *  it chooses a random number of children to traverse, with that random
     *  number averaging to avDegree.
     *  @param size      the number of vertices to extract
     *  @param avDegree  the average out-degree
     *  @param g         the data graph to extract from
     *  @param inverse   whether to create inverse adjacency (parents)
     *  @param name      the name of the graph
     */
    def genBFSQuery (size: Int, avDegree: Int, g: Graph, inverse: Boolean = false,
                     name: String = "g"): Graph =
        val maxRestarts = 1000                                            // limit on retries
        val nedges      = avDegree * size                                 // desired number of edges

        var nRestarts   = 0                                               // restarts so far
        var nodes       = SET [Int] ()                                    // current set of extracted vertices
        var edges       = 0                                               // number of edges so far
        var maxEdges    = 0                                               // number of edges in best try
        var maxNodes: SET [Int] = null                                    // nodes in best try
        var chMap:    Map [Int, SET [Int]] = null                         // child map
        var maxChMap: Map [Int, SET [Int]] = null                         // child map in best try

        while (nodes.size < size || edges < nedges) && nRestarts < maxRestarts do
            nodes   = SET [Int] ()
            chMap   = Map [Int, SET [Int]] ()
            val q   = Queue [Int] ()                                      // queue for BFS visitation
            var v_s = -1                                                  // starting vertex

            var degree = 1 + rng.igen1 (avDegree * 2)                     // desired out-degree
            while
                v_s = rng.igen1 (g.size - 1)                              // randomly pick a start vertex
                g.ch(v_s).size < degree + 1
            do ()

            nodes += v_s                                                  // add starting vertex to nodes
            q.enqueue (v_s)                                               // place it in the BFS queue
            edges = 0                                                     // no edges yet

            while ! q.isEmpty && nodes.size < size do                     // need size number of vertices

                val v = q.dequeue ()                                      // candidate vertex v
                val v_chs = g.ch(v)                                       // all its children
                degree = 1 + rng.igen1 (avDegree * 2)                     // desired out-degree
                val chs   = pickChildren (v, v_chs.toArray, degree)       // pick random subset of v's children
                val chs2  = SET [Int] ()                                  // those making the cut

                for v_ch <- chs do                                        // chs2: appropriate vertices from chs
                    degree = 1 + rng.igen1 (avDegree * 2)
                    if g.ch(v_ch).size >= degree + 1 then
                        if nodes.size < size then                         // still need vertices => take edges
                            if ! (nodes contains v_ch) then
                                nodes += v_ch                             // add child vertex to nodes
                                q.enqueue (v_ch)                          // put it on the BFS queue
                            chs2 += v_ch
                            edges += 1
                        else if nodes contains v_ch then                  // can only take edge if child in nodes
                            chs2 += v_ch
                            edges += 1
                        else
                            println (s"genBFSquery: can't find enough child vertices for $v")
                    end if
                end for

                println (s"nodes = $nodes")
                chMap += (v -> (chMap.getOrElse (v, SET [Int] ()) ++ chs2))
            end while

            println (s"chMap = $chMap")

            if edges > maxEdges then                                      // if most edges so far, save as best
                maxEdges = edges
                maxNodes = nodes.clone ()
                maxChMap = chMap.clone ()

            if nodes.size < size || edges < nedges then                   // not enough vertices/edges, try again
                nRestarts += 1
                println ("=" * 80)
                println (s"RESTART: #restarts = $nRestarts, #nodes = ${nodes.size}, #edges = $edges")
                println ("=" * 80)
            end if
        end while

        println (s"maxEdges = $maxEdges")

        if nRestarts == maxRestarts then println ("genBFSQuery: could not find a query graph with enough edges")

        GraphGen.buildQGraph (maxNodes, maxChMap, g.label, g.elabel, inverse, name)
    end genBFSQuery

    //------------------------------------------------------------------------
    // Private helper methods.
    //------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly Pick degree children of vertex v.  Requires v to have at
     *  least degree children.
     *  @param v       the given vertex
     *  @param v_ch    vertex v's children as an array
     *  @param degree  the number of children to pick (out-degree)
     */
    private def pickChildren (v: Int, v_ch: Array [Int], degree: Int): SET [Int] =
        println (s"pickChildren: v = $v, v_ch = ${stringOf (v_ch)}, degree = $degree")
        val chs = SET [Int] ()                                       // set of children to be formed
        var it  = 0                                                  // iteration counter
        while
            val v_c = v_ch(rng.igen1 (v_ch.size - 1))
            if v_c != v then chs += v_c
            it += 1
            if it > 100 * degree then { println ("pickChildren: can't find enough children"); return SET [Int] () }
            chs.size < degree
        do ()
        println (s"pickChildren: chs = $chs")
        chs
    end pickChildren

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an array with labels distributed between 1 and nLabels.
     *  based on a uniform distribution.
     *  @param size     the number of vertices
     *  @param nLabels  the number of labels
     */
    private def randDistLabels (size: Int, nLabels: Int): Array [ValueType] =
        setVariate (nLabels)                                                                // set rlg
        typeSelector match
        case 'I' => Array.ofDim [Int] (size).map (_ => rlg.igen.asInstanceOf [Int])         // for `Int`
        case 'D' => Array.ofDim [Double] (size).map (_ => rlg.gen.asInstanceOf [Double])    // for `Double`
        case 'S' => Array.ofDim [String] (size).map (_ => rlg.sgen.asInstanceOf [String])   // for `String`
        case _   => { flaw ("randDistLabels", "label type not supported"); null.asInstanceOf [Array [ValueType]] }
        end match
    end randDistLabels

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an array with edge labels distributed between 0 and eLabels - 1.
     *  based on a uniform distribution.
     *  @param ch       the adjacency (child array)
     *  @param nLabels  the number of edge labels
     */
    private def randDisteLabels (ch: Array [SET [Int]], eLabels: Int): Map [(Int, Int), ValueType] =
        val elab = Map [(Int, Int), ValueType] ()
        setVariate (eLabels)                                                                // set rlg
        typeSelector match
        case 'I' => for i <- ch.indices; c <- ch(i) do elab += (i, c) -> rlg.igen.asInstanceOf [Int]  
        case 'D' => for i <- ch.indices; c <- ch(i) do elab += (i, c) -> rlg.gen.asInstanceOf [Double]  
        case 'S' => for i <- ch.indices; c <- ch(i) do elab += (i, c) -> rlg.sgen.asInstanceOf [String]  
        case _   => { flaw ("randDistLabels", "label type not supported"); null.asInstanceOf [Array [ValueType]] }
        end match
        elab
    end randDisteLabels

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an array with labels distributed between 0 and nLabels - 1.
     *  based on a power law distribution.
     *  @param size     the number of vertices
     *  @param nLabels  the number of labels
     *  @param pow      the power/exponent
     */
    private def powDistLabels (size: Int, nLabels: Int, pow: Double): Array [ValueType] =
        val powLaw = PowerLaw (1.0, nLabels + 1.0, pow, stream)
        typeSelector match
        case 'I' => Array.ofDim [Int] (size).map (_ => powLaw.igen.asInstanceOf [Int])
        case 'D' => Array.ofDim [Double] (size).map (_ => powLaw.gen.asInstanceOf [Double])
        case 'S' => Array.ofDim [String] (size).map (_ => powLaw.sgen.asInstanceOf [String])
        case _   => { flaw ("powDistLabels", "label type not supported"); null.asInstanceOf [Array [ValueType]] }
        end match
    end powDistLabels

end GraphGen


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GraphGen` companion object provides simple methods for creating data
 *  and query graphs.
 */
object GraphGen:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly generate a data graph g.
     *  @param gSize      the size of the data graph
     *  @param nLabels    the number of distinct labels
     *  @param gAvDegree  the average vertex out-degree for data graph
     *  @param addPa      whether to add direct references to parents
     *  @param typeSel    the type selector
     */
    def genGraph (typeSel: Char, stream: Int = 0, gSize: Int = 100,
                  nLabels: Int = 50, eLabels: Int = 5, gAvDegree: Int = 8, addPa: Boolean = false):
        Graph =
        val gGen = new GraphGen (typeSel, stream)
        gGen.genRandomConnectedGraph (gSize, nLabels, eLabels, gAvDegree, addPa, "g")   // data graph
    end genGraph

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly generate both a data graph g and a query graph q.
     *  @param gSize      the size of the data graph
     *  @param qSize      the size of the query graph
     *  @param nLabels    the number of distinct labels
     *  @param gAvDegree  the average vertex out-degree for data graph
     *  @param qAvDegree  the average vertex out-degree for query graph
     *  @param addPa      whether to add direct references to parents
     *  @param typeSel    the type selector
     */
    def genGraphs(typeSel: Char, stream: Int = 0,
                   gSize: Int = 100, qSize: Int = 10, nLabels: Int = 50, eLabels: Int = 5,
                   gAvDegree: Int = 8, qAvDegree: Int = 2, addPa: Boolean = false):
        (Graph, Graph) =
        val gGen = new GraphGen (typeSel, stream)
        val g = gGen.genRandomConnectedGraph (gSize, nLabels, eLabels, gAvDegree, addPa, "g")   // data graph
        val q = gGen.genBFSQuery (qSize, qAvDegree, g, addPa, "q")                              // query graph
        (g, q)
    end genGraphs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly generate both a data graph g and a query graph q
     *  using Power Law
     *  @param gSize      the size of the data graph
     *  @param qSize      the size of the query graph
     *  @param nLabels    the number of distinct labels
     *  @param gAvDegree  the average vertex out-degree for data graph
     *  @param qAvDegree  the average vertex out-degree for query graph
     *  @param addPa      whether to add direct references to parents
     *  @param typeSel    the type selector
     */
    def genPowerGraphs (typeSel: Char, stream: Int = 0,
                        gSize: Int = 100, qSize: Int = 10, nLabels: Int = 50, eLabels: Int = 5,
                        gAvDegree: Int = 8, qAvDegree: Int = 2, addPa: Boolean = false):
        (Graph, Graph) =
        val gGen = new GraphGen (typeSel, stream)
        val g = gGen.genPowerLawGraph (gSize, nLabels, eLabels, gAvDegree)             // data graph
        val q = gGen.genBFSQuery (qSize, qAvDegree, g, addPa, "q")                     // query graph
        (g, q)
    end genPowerGraphs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Extracts a subgraph of size vertices from graph g by performing a
     *  breadth-first search from a random vertex.
     *  @param size     the number of vertices to extract
     *  @param g        the data graph to extract from
     *  @param inverse  whether to create inverse adjacency (parents)
     *  @param name     the name of the graph
     */
    def extractSubgraph (size: Int, g: Graph, inverse: Boolean = false, 
                         name: String = "g", stream: Int = 0): Graph =
        val maxRestarts = 1000
        var nRestarts   = 0
        var chMap: Map [Int, SET [Int]] = null
        var nodes = SET [Int] ()                                         // current set of extracted vertices
        val rng = Randi0 (stream = stream)

        while nodes.size < size && nRestarts < maxRestarts do
            if nRestarts % 100 == 0 then println ("restarting " + nRestarts)
            chMap     = Map [Int, SET [Int]] ()
            nodes     = SET [Int] ()
            val q     = Queue [Int] ()
            val start =  rng.igen1 (g.size - 1)                           // randomly pick a start node in ch
            println ("extractSubgraph: start node: " + start)
            q.enqueue (start)
            nodes += start

            while ! q.isEmpty && nodes.size < size do
                val newNode = q.dequeue ()
                val newNodeChildren = g.ch (newNode)
                if ! newNodeChildren.isEmpty then
                    for newChild <- newNodeChildren if nodes.size < size do
                        if ! nodes.contains (newChild) then { nodes += newChild; q.enqueue (newChild) }
                end if
            end while

            for n <- nodes do { val chs = g.ch(n) intersect nodes; chMap += (n -> chs ) }
            if nodes.size < size then
                nRestarts += 1
                println ("nodes.size only " + nodes.size)
        end while

        if nRestarts == maxRestarts then { println ("extractSubgraph: could not find a good query"); return null }

        buildQGraph (nodes, chMap, g.label,g.elabel, inverse, name)

    end extractSubgraph

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a query graph from the subgraph extracted from the data graph.
     *  Includes the renumbering of vertex ids.
     *  @param nodes    the nodes extracted from the data graph
     *  @param chMap    the child map: vertex -> children
     *  @param gLabel   the labels from the data graph
     *  @param inverse  whether to create inverse adjacency (parents)
     *  @param name     the name for the new query graph
     */
    def buildQGraph (nodes: SET [Int], chMap: Map [Int, SET [Int]], gLabel: Array [ValueType],
                     gELabel: Map [(Int, Int), ValueType], inverse: Boolean, name: String): Graph =
        // create a vertex map from old to new ids, e.g., 7 -> 0, 11 -> 1, 15 -> 2
        val vertexMap = Map [Int, Int] ()
        var c = 0
        for v <- nodes do { vertexMap += (v -> c); c += 1 }

        // for each new id, record the old id
        val new2OldIds = Array.ofDim [Int] (nodes.size)
        vertexMap.foreach { case (oldId, newId) => new2OldIds(newId) = oldId }

        // for each mapped vertex, assign its mapped children
        val ch = Array.ofDim [SET [Int]] (nodes.size).map (_ => SET [Int] ())
        for (v, v_ch) <- chMap do ch(vertexMap (v)) = v_ch.map (vertexMap (_))

        // map the vertex and edge labels
        val label   = new2OldIds.map (gLabel(_)).toArray
        val qelabel = Map [(Int, Int), ValueType] ()                   // FIX - fill in map

        for u <- ch.indices do
            val p = new2OldIds (u)
            for v <- ch(u) do
                val q = new2OldIds (v)
                qelabel += (u, v) -> gELabel (p, q)
            end for
        end for

        println ("new2OldIds = " + stringOf (new2OldIds))

        new Graph (ch, label, qelabel, inverse, name)

    end buildQGraph

end GraphGen


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphGenTest` main function is used to test the `GraphGen` class for building
 *  random graphs where a vertex's degree is uniformly distributed.
 *  This work build graphs with `Int` vertex/edge labels.
 *  > runMain scalation.database.graph_pm.graphGenTest
 */
@main def graphGenTest (): Unit =

    val gGen = new GraphGen ('I')

    println ("graphGenTest: test genRandomGraph")
    (1 to 5).foreach { _ =>
        val g = gGen.genRandomGraph (4, 100, 5, 1)
        g.printG ()
        println ("CONNECTED?  " + g.isConnected)
    } // foreach

    println ("graphGenTest: test genRandomConnectedGraph")
    (1 to 5).foreach { _ => gGen.genRandomConnectedGraph (4, 100, 5, 1).printG () }

    println ("graphGenTest: test genRandomGraph_PowLabels")
    val g1 = gGen.genRandomGraph_PowLabels (200, 50, 3, 2)
    g1.printG ()
    println (s"g1.labelMap = ${g1.labelMap}")
 
end graphGenTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphGenTest2` main function is used to test the `GraphGen` class for building
 *  random graphs where a vertex's degree is uniformly distributed.
 *  This work build graphs with `Double` vertex/edge labels.
 *  > runMain scalation.database.graph_pm.graphGenTest2
 */
@main def graphGenTest2 (): Unit =

    val gGen = new GraphGen ('D')

    println ("graphGenTest: test genRandomGraph")
    (1 to 5).foreach { _ =>
        val g = gGen.genRandomGraph (4, 100, 5, 1)
        g.printG ()
        println ("CONNECTED?  " + g.isConnected)
    } // foreach

    println ("graphGenTest2: test genRandomConnectedGraph")
    (1 to 5).foreach { _ => gGen.genRandomConnectedGraph (4, 100, 5, 1).printG () }

    println ("graphGenTest2: test genRandomGraph_PowLabels")
    val g1 = gGen.genRandomGraph_PowLabels (50, 10, 10, 5)
//  val g1 = gGen.genRandomGraph_PowLabels (200, 50, 2)
    g1.printG ()
    println (s"g1.labelMap = ${g1.labelMap}")

end graphGenTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphGenTest3` main function is used to test the `GraphGen` class for building
 *  random graphs where a vertex's degree is uniformly distributed.
 *  This work build graphs with `String` vertex/edge labels.
 *  > runMain scalation.database.graph_pm.graphGenTest3
 */
@main def graphGenTest3 (): Unit =

    val gGen = new GraphGen ('S')

    println ("graphGenTest: test genRandomGraph")
    (1 to 5).foreach { _ =>
        val g = gGen.genRandomGraph (4, 100, 5, 1)
        g.printG ()
        println ("CONNECTED?  " + g.isConnected)
    } // foreach

    println ("graphGenTest3: test genRandomConnectedGraph")
    (1 to 5).foreach { _ => gGen.genRandomConnectedGraph (4, 100, 5, 1).printG () }

    println ("graphGenTest3: test genRandomGraph_PowLabels")
    val g1 = gGen.genRandomGraph_PowLabels (50, 10, 10, 5)
    g1.printG ()
    println (s"g1.labelMap = ${g1.labelMap}")

end graphGenTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphGenTest4` main function is used to test the `GraphGen` class for building
 *  power law graphs.
 *  > runMain scalation.database.graph_pm.graphGenTest4
 */
@main def graphGenTest4 (): Unit =

    val gGen = new GraphGen ('I')

    println ("graphGenTest2: test genPowerLawGraph")
    val g2 = gGen.genPowerLawGraph (50, 10, 10, 5)
    g2.printG ()
    g2.ch.sortBy (_.size).foreach { println(_) }

    println ("graphGenTest2: test genPowerLawGraph_PowLabels")
    val g3 = gGen.genPowerLawGraph_PowLabels (50, 10, 10, 5)
    g3.printG ()
    g3.ch.sortBy (_.size).foreach { println (_) }
    println (s"g3.labelMap = ${g3.labelMap}")

end graphGenTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphGenTest5` main function is used to test the `GraphGen` class for extracting
 *  query graphs from data graphs (note: data graph should be connected).
 *  > runMain scalation.database.graph_pm.graphGenTest5
 */
@main def graphGenTest5 (): Unit =

    val gGen = new GraphGen ('D')

    println ("graphGenTest5: test genRandomConnectedGraph")
    val nVertices = 10000
    val nLabels   =    10
    val eLabels   =     5
    val avDegree  =    16
    val g = gGen.genRandomConnectedGraph (nVertices, nLabels, eLabels, avDegree)
    println ("done generating data graph")
    println (GraphMetrics.stats (g))

    println ("graphGenTest5: test genBFSQuery")
    (1 to 5).foreach { _ =>
        val q = gGen.genBFSQuery (25, 3, g)
        q.printG ()
        println (GraphMetrics.stats (g))
    } // foreach
    println ("done")

end graphGenTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphGenTest6` main function is used to test the `GraphGen` companion object
 *  for generating both data and query graphs.
 *  > runMain scalation.database.graph_pm.graphGenTest6
 */
@main def graphGenTest6 (): Unit =

    for stream <- 0 until 3 do
        val (g, q) = GraphGen.genGraphs ('S', stream)
        banner ("data graph")
        g.printG ()
        println (GraphMetrics.stats (g))
        banner ("query graph")
        q.printG ()
        println (GraphMetrics.stats (q))
    end for

end graphGenTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphGenTest7` main function is used to test the `GraphGen` companion object
 *  for generating data graphs.
 *  > runMain scalation.database.graph_pm.graphGenTest7
 */
@main def graphGenTest7 (): Unit =

    for stream <- 0 until 3 do
        val g = GraphGen.genGraph ('D', stream)
        banner ("data graph")
        g.printG ()
        println (GraphMetrics.stats (g))
    end for

end graphGenTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphGenTest8` main function is used to test the `GraphGen` companion object
 *  for generating both data and query graphs.
 *  > runMain scalation.database.graph_pm.graphGenTest8
 */
@main def graphGenTest8 (): Unit =

    val g = GraphGen.genGraph ('S')
    banner ("data graph")
    g.printG ()
    println (GraphMetrics.stats (g))
    banner ("query graph")
    val q = GraphGen.extractSubgraph(5, g)
    q.printG ()
    println (GraphMetrics.stats (q))

end graphGenTest8

