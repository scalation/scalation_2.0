
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Matthew Saltz, Supriya Ramireddy
 *  @version 2.0
 *  @date    Tue Nov  1 19:12:16 EDT 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Graph Data Structure Using Mutable Sets Supporting Pattern Matching
 */

package scalation
package database
package graph_pm

import scala.collection.mutable.{Map, Set => SET}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat.MatrixD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Graph` class stores vertex/edge-labeled directed graphs using an
 *  adjacency set (ch) representation, e.g., ch = { {1, 2}, {0}, {1} } means
 *  that the graph has the following edges { (0, 1), (0, 2), (1, 0), (2, 1) }.
 *  Optionally, inverse adjacency via the pa array can be stored at the cost
 *  of nearly doubling the storage requirements.
 *----------------------------------------------------------------------------
 *  @param ch       the array of child (adjacency) vertex sets (outgoing edges)
 *  @param label    the array of vertex labels: v -> vertex label
 *  @param elabel   the map of edge labels: (u, v) -> edge label
 *  @param inverse  whether to store inverse adjacency sets (parents)
 *  @param name     the name of the digraph
 *  @param schema   optional schema: map from label to label type
 */
case class Graph (ch: Array [SET [Int]],
                  label: Array [ValueType],
                  elabel: Map [(Int, Int), ValueType] = Map (),
                  inverse: Boolean = false,
                  name: String = "g",
                  schema: Array [String] = Array ())
     extends Cloneable:

    private val debug = debugf ("Graph", false)                 // debug function
    private val flaw  = flawf ("Graph")                         // flaw function

    val id = Array.range (1, ch.size + 1)                       // array of vertex id's
    val labelMap = buildLabelMap (label)                        // map from label to set of vertices
    val pa = Array.ofDim [SET [Int]] (ch.size)                  // opt. array of vertex inverse (parent) adjacency sets
    val indexMap = Map [(ValueType, ValueType), SET [Int]] ()   // index from (source node, edge) to destination node
                                                                // must use getOrElse to avoid exceptions
    val count = Map [(ValueType, ValueType, Int), Int] ()       // count of no. of occurences of a pair of vertices
                                                                // with a particular edge
    val edgeMap = Map [ValueType, SET [(Int, Int)]] ()          // index from edges to set of pairs of nodes

    if inverse then addPar ()                                   // by default, don't use pa

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the inverse adjacency sets for rapid accesses to parent vertices.
     */
    def addPar (): Unit =
        for j <- pa.indices do pa(j) = SET [Int] ()
        for i <- ch.indices; j <- ch(i) do pa(j) += i
    end addPar

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size of graph in terms of the number of vertices.
     */
    def size: Int = ch.length

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of edges in this digraph.
     */
    def nEdges: Int = ch.foldLeft (0) { (n, i) => n + i.size }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given an array of labels, return an index from labels to the sets of
     *  vertices containing those labels.
     *  @param label  the array of vertex labels
     */
    def buildLabelMap (label: Array [ValueType]): Map [ValueType, SET [Int]] =
        debug ("buildLabelMap", s"label = ${stringOf (label)}")
        val labelMap = Map [ValueType, SET [Int]] ()
        for i <- label.indices do                               // for each vertex i
            val lab = label(i)                                  // label for vertex i
            val st  = labelMap.getOrElse (lab, null)            // get set of vertices with that label
            if st != null then st.add (i)                       // add to existing set
            else labelMap.put (lab, SET (i))                    // make a new set
        end for
        labelMap
    end buildLabelMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the children of vertex u that are connected via an edge labeled elab.
     *  @param u     the source vertex
     *  @param elab  the edge label
     */
    def children (u: Int, elab: ValueType): SET [Int] =
        for v <- ch(u) if elabel((u, v)) == elab yield v
    end children

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parents of vertex v that are connected via an edge labeled
     *  elab.  Requires the parents pa to be added (@see `Graph`).
     *  @param v    the destination vertex
     *  @param elab the edge label
     */
    def parents (v: Int, elab: ValueType): SET [Int] =
        for u <- pa(v) if elabel((u, v)) == elab yield u
    end parents

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clone (make a deep copy) of this digraph.
     */
    override def clone: Graph =
        val ch2    = Array.ofDim [SET [Int]] (ch.length)
        val label2 = Array.ofDim [ValueType] (ch.length)
        for i <- ch2.indices do
//          ch2(i) = SET (ch(i).toArray: _*)
            ch2(i) = ch(i).clone                                // FIX - make sure it is a deep copy
            label2(i) = label(i)
        end for
        new Graph (ch2, label2, elabel.clone, inverse, name)
    end clone

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the end-point vertex id of each edge is within bounds: 0 .. maxId.
     */
    def checkEdges: Boolean =
        val maxId = ch.size - 1
        var okay  = true
        breakable {
            for u <- ch.indices; u_c <- ch(u) do
                if u_c < 0 || u_c > maxId then
                    flaw ("checkEdges", s"child of $u, with vertex id $u_c not in bounds 0..$maxId")
                    okay = false
                    break ()
            end for
        } // breakable
        okay
    end checkEdges

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the edges in the elabel map correspond to edges in the
     *  the adjacency list.
     */
    def checkElabels: Boolean =
        var okay = true
        breakable {
            for (u, v) <- elabel.keys do
                if ! (ch(u) contains v) then
                    flaw ("checkElabels", s"no such edge from $u to $v")
                    okay = false
                    break ()
            end for
        } // breakable
        okay
    end checkElabels

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Builds index from (start vertex, edge) -> end vertex.
     */
    def buildIndex (): Unit =
        for k <- elabel.keys do
            val e     = elabel(k)
            count    += (label(k._1), e.asInstanceOf [ValueType], k._2) ->
                        (count.getOrElse ((label(k._1), e.asInstanceOf[ValueType], k._2), 0) + 1)
            indexMap += (label(k._1), e.asInstanceOf [ValueType]) ->
                        (indexMap.getOrElse ((label(k._1), e.asInstanceOf [ValueType]), SET()) union SET(k._2))
//                      (indexMap.getOrElse ((label(k._1), e.asInstanceOf [ValueType]), SET()) + k._2)

        println ("count of triples")
        for (k, v) <- count do println((k, v))
        println ("-" * 50)
        println ("indexMap")
        for (k, v) <- indexMap do println((k, v))
        println ("-" * 60)
    end buildIndex

    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build and index from edge -> (Vertex pair).
     */
    def indexEdges (): Unit =
        for k <- elabel.keys do
            val e    = elabel(k)
            edgeMap += e -> (edgeMap.getOrElse (e, SET()) union SET(k))
        end for
        println ("edgeMap" + edgeMap)
        println ("-" * 50)
    end indexEdges

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make this directed graph work like an undirected graph by making sure that
     *  for every edge u -> v, there is a v -> u edge and that they have same edge label.
     */
    def makeUndirected (): Graph =
        for u <- 0 until size; v <- ch(u) do ch(v) += u
        val edges = elabel.clone.keys
        for (u, v) <- edges do elabel += (v, u) -> elabel(u, v)
        this
    end makeUndirected

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this digraph is (weakly) connected.
     */
    def isConnected: Boolean = (new GraphDFS (this)).weakComps == 1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this digraph to a string in a shallow sense.
     *  Large arrays are not converted.  Use print to show all information.
     */
    override def toString: String =
        s"Graph (ch.length = ${ch.length}, label.length = ${label.length}" +
        s"elabel.size = ${elabel.size}, inverse = $inverse, name = $name)"
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the i'th row/line of this digraph to a string.
     *  @param i  the i'th row/line
     */
    def toLine (i: Int): String =
        val ch_i = ch(i).toString.replace ("HashSet(", "").replace(")", "")
        s"vertex id = ${id(i)}, " +
        s"label = ${label(i)}, " +
        s"ch = $ch_i"
    end toLine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print this digraph in a deep sense with all the information.
     */
    def printG (): Unit =
        println (s"Graph ($name, $inverse, $size")
        for i <- ch.indices do println (toLine (i))
        for (k, v) <- elabel do println (s"edge $k -> $v")
        println (")")
    end printG

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an array vertices from selected vertices from graph g, those in vset.
     *  If makeMap, record the mapping between g's vertices and the new vertices.
     *  @param g        graph to be used for adding vertex labels
     *  @param vset     selected vertices from graph g
     *  @param makeMap  whether to make a map from new vertex id's to old vertex id's
     */
    def addVertices (g: Graph, vset: SET [Int], makeMap: Boolean = false):
                    (Array [ValueType], Map [Int, Int]) =
        val lv   = Array.ofDim [ValueType] (vset.size)
        val vmap = Map [Int, Int] ()

        if makeMap then
            var n = 0
            for i <- vset do
                lv(n) = g.label(i)                              // nth vertex gets g's ith label
                vmap += n -> i                                  // records mapping from n to i
                n    += 1
            end for
        else
            for i <- vset do lv(i) = g.label(i)                 // direct correspondence for labels 
        (lv, vmap)
    end addVertices

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** To check if two graphs are equal.
     *  @param g2_ the graph to be compared with this graph
     */
    override def equals (g2_ : Any): Boolean =
        g2_ match
        case _: Graph =>
                val g2 = g2_.asInstanceOf [Graph]
                if ! (ch    sameElements g2.ch) then    { println ("failed on ch"); return false }
                if ! (id    sameElements g2.id) then    { println ("failed on id"); return false }
                if ! (label sameElements g2.label) then { println ("failed on label"); return false }
                if elabel != g2.elabel then             { println ("failed on elabel"); return false }
                true
        case _ => false
        end match
    end equals

end Graph


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Graph` companion object provides builder methods and example query digraphs.
 */
object Graph:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a `Graph` from the given vertex labels and edge mappings.
     *  @param label    the vertex labels
     *  @param elabel   the map of edges, vertex pair -> edge label
     *  @param inverse  whether to store inverse adjacency sets (parents)
     *  @param name     the name of the graph
     *  @param schema   the type of the vertices
     */
    def apply (label: Array [ValueType],
               elabel: Map [(Int, Int), ValueType],
               inverse: Boolean,
               name: String,
               schema: Array [String]): Graph =
        val n  = label.length
        val ch = Array.fill (n)(SET [Int] ())
        for (e, _) <- elabel do ch(e._1) += e._2
        new Graph (ch, label, elabel, inverse, name, schema)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a `Graph` from an adjacency matrix (assumes no self-loops).
     *  @param adj      the adjacency matrix representation for the graph
     *  @param ordered  implies edges i -> j are ordered i < j
     *  @param inverse  whether to store inverse adjacency sets (parents)
     *  @param name     the name of the graph
     *  @param schema   the type of the vertices
     */
    def fromMatrix (adj: MatrixD, ordered: Boolean = true, inverse: Boolean = false,
                    name: String = "g", schema: Array [String] = null): Graph =
        val n    = adj.dim
        val ch   = Array.fill (n)(SET [Int] ())
        val elab = Map [(Int, Int), ValueType] ()
        val lab  = Array.ofDim [ValueType] (n)
        for i <- adj.indices do
            lab(i) = i                                          // vertex label
            if ordered then                                     // i -> j requires i < j
                for j <- i + 1 until n do 
                    ch(i) += j                                  // add j as child of i
                    elab  += (i, j) -> adj(i, j)                // edge label
            else
                for j <- 0 until n if j != i do                 // no vertex order restriction
                    ch(i) += j
                    elab  += (i, j) -> adj(i, j)
        end for
        new Graph (ch, lab, elab, inverse, name, schema)
    end fromMatrix

end Graph


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphTest` main function tests the `Graph` class using examples from
 *  the `ExampleGraphD` object, which contains digraphs whose vertex and edge
 *  labels are of type `Double`.
 *  > runMain scalation.database.graph_pm.graphTest
 */
@main def graphTest (): Unit =

    import ExampleGraphD._

    g1.printG ()
    q1.printG ()
    g2.printG ()
    q2.printG ()

end graphTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphTest2` main function tests the `Graph` class using examples from
 *  the `ExampleGraphD` object, which contains digraphs whose vertex and edge
 *  labels are of type `String`.
 *  > runMain scalation.database.graph_pm.graphTest2
 */
@main def graphTest2 (): Unit =

    import ExampleGraphS._

    g1.printG ()
    q1.printG ()
    g2.printG ()
    q2.printG ()

end graphTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphTest3` main function tests the `Graph` class by building a randomly
 *  generated digraph.
 *  > runMain scalation.database.graph_pm.graphTest3
 *
@main def graphTest3 (): Unit =

    val gGen = new GraphGen (0.0)

    private val nVertices = 20         // number of vertices
    private val nLabels   = 5          // number of distinct vertex labels
    private val eLabels   = 3          // number of distinct edge labels
    private val outDegree = 2          // average out degree
    private val inverse   = false      // whether inverse adjacency is used (parents)
    private val name      = "gr"       // name of the graph

    val graph = gGen.genRandomGraph (nVertices, nLabels, eLabels, outDegree, inverse, name)
    Graph (graph, name).printG ()

end graphTest3
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphTest4` main function tests the `Graph` class by building a graph
 *  from an adjacency matrix called cmi for Conditional Mutual Information.
 *  @see scalation.modeling.classifying.BayesClassifier
 *  > runMain scalation.database.graph_pm.graphTest4
 */
@main def graphTest4 (): Unit =

    val cmi = MatrixD ((4, 4), 0.00000, 0.419593, 0.222815, 0.311752,
                               0.00000, 0.00000,  0.419593, 0.168895,
                               0.00000, 0.00000,  0.00000,  0.0610538,
                               0.00000, 0.00000,  0.00000,  0.00000)
    println (s"cmi = $cmi")

    banner ("Graph from Adjacency Matrix")
    val g = Graph.fromMatrix (cmi)
    g.printG ()

end graphTest4

