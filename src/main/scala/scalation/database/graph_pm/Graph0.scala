
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

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Graph0` class stores vertex-labeled directed graphs using an
 *  adjacency set (ch) representation, e.g., ch = { {1, 2}, {0}, {1} } means
 *  that the graph has the following edges { (0, 1), (0, 2), (1, 0), (2, 1) }.
 *  Optionally, inverse adjacency via the pa array can be stored at the cost
 *  of nearly doubling the storage requirements.
 *----------------------------------------------------------------------------
 *  @param ch       the array of child (adjacency) vertex sets (outgoing edges)
 *  @param label    the array of vertex labels: v -> vertex label
 *  @param inverse  whether to store inverse adjacency sets (parents)
 *  @param name     the name of the digraph
 */
case class Graph0 (ch: Array [SET [Int]], label: Array [ValueType],
                   inverse: Boolean = false, name: String = "g")
     extends Cloneable:

    private val flaw = flawf ("Graph0")                         // flaw function

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
    /** Create an array [1, 2, ..., size] for default values for id's.
     */
    def buildId: Array [Int] = Array.range (1, ch.size + 1)

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
        println (s"buildLabelMap: label = ${stringOf (label)}")
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
    /** Clone (make a deep copy) of this digraph.
     */
    override def clone: Graph0 =
        val ch2    = Array.ofDim [SET [Int]] (ch.length)
        val label2 = Array.ofDim [ValueType] (ch.length)
        for i <- ch2.indices do
//          ch2(i) = SET (ch(i).toArray: _*)
            ch2(i) = ch(i).clone                                // FIX - make sure it is a deep copy
            label2(i) = label(i)
        end for
        new Graph0 (ch2, label2, inverse, name)
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
    /** Make this directed graph work like an undirected graph by making sure that
     *  for every edge u -> v, there is a v -> u edge and that they have same edge label.
     */
    def makeUndirected (): Graph0 =
        for u <- 0 until size; v <- ch(u) do ch(v) += u
        this
    end makeUndirected

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this digraph to a string in a shallow sense.
     *  Large arrays are not converted.  Use print to show all information.
     */
    override def toString: String =
        s"Graph0 (ch.length = ${ch.length}, label.length = ${label.length}" +
        s"inverse = $inverse, name = $name)"
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
        println (s"Graph0 ($name, $inverse, $size")
        for i <- ch.indices do println (toLine (i))
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
        case _: Graph0 =>
                val g2 = g2_.asInstanceOf [Graph]
                if ! (ch    sameElements g2.ch) then    { println ("failed on ch"); return false }
                if ! (id    sameElements g2.id) then    { println ("failed on id"); return false }
                if ! (label sameElements g2.label) then { println ("failed on label"); return false }
                true
        case _ => false
        end match
    end equals

end Graph0

