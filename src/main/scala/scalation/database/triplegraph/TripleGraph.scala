
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Oct  9 15:40:47 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Triple-Graph Data Structure Using Mutable Sets Supporting Pattern Matching
 */

package scalation
package database
package triplegraph

import scala.collection.mutable.{ArrayBuffer => Bag, Map, Set => SET}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Triple` class holds information about a triple (3 part edge). 
 *  @param h  the head vertex
 *  @param r  the relation/edge-label
 *  @param t  the tail vertex
 */
case class Triple (h: Int, r: ValueType, t: Int)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RDFTriple` class holds information about a triple (3 part edge). 
 *  It may be viewed as a statement with a subject, predicate and object.
 *  @param s  the head vertex (subject = International Resourse Indentifier (IRI))
 *  @param p  the relation/edge-label (predicate)
 *  @param o  the tail vertex (object = IRI or literal value)
 */
case class RDFTriple (s: String, p: ValueType, o: ValueType)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TripleGraph` class stores vertex/edge-labeled triple graphs by maintaining
 *  bags of vertices and triples.
 *----------------------------------------------------------------------------
 *  @param label    the array of vertex labels
 *  @param triples  the bag of triples in the triple-graph
 *  @param name     the name of the triple-graph
 *  @param schema   optional schema: map from label to label type
 */
case class TripleGraph (label: Array [ValueType],
                        triples: Bag [Triple],
                        name: String = "g",
                        schema: Array [String] = Array ())
     extends Cloneable:

    val hindex    = Map [Int, SET [Triple]] ()                  // index to triples with the given head vertex
    val tindex    = Map [Int, SET [Triple]] ()                  // index to triples with the given tail vertex
    val labelMap  = buildLabelMap                               // map from vertex labels to vertices containing
    val elabelMap = buildElabelMap                              // map from vertex pair to set of relations/edge-labels

    buildIndices ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size in terms of the vertices.
     */
    def size: Int = label.length

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the tail vertices in the triples (h, *, ?).
     *  @param h  the head vertex
     */
    def ch (h: Int): SET [Int] =
        val edges = hindex.getOrElse (h, SET [Triple] ())
        for e <- edges yield e.t
    end ch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the tail vertices in the triples (h, r, ?).
     *  @param h  the head vertex
     *  @param r  the relation/edge label
     */
    def children (h: Int, r: ValueType): SET [Int] =
        for e <- hindex(h) if e.r == r yield e.t
    end children

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the head vertices in the triples (?, r, t).
     *  @param t  the tail vertex
     *  @param r  the relation/edge label
     */
    def parents (t: Int, r: ValueType): SET [Int] =
        for e <- tindex(t) if e.r == r yield e.h
    end parents

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the indices.  FIX - add edges to existing set
     */
    def buildIndices (): Unit =
        for e <- triples do                                     // for each triple e
            hindex += e.h -> SET (e)
            tindex += e.t -> SET (e)
        end for
    end buildIndices

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a map from vertex labels to the sets of vertices containing
     *  those labels.
     */
    def buildLabelMap: Map [ValueType, SET [Int]] =
        val labelMap = Map [ValueType, SET [Int]] ()
        for v <- label.indices do                               // for each vertex v
            val lab = label(v)                                  // label for vertex v
            val st  = labelMap.getOrElse (lab, null)            // get set of vertices with that label
            if st != null then st.add (v)                       // add to existing set
            else labelMap.put (lab, SET (v))                    // make a new set
        end for
        labelMap
    end buildLabelMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a map from a vertex pair (u, v) to the set relations/edge-labels
     *  on edges connecting vertex u to v.
     */
    def buildElabelMap: Map [(Int, Int), SET [ValueType]] =
        val elabelMap = Map [(Int, Int), SET [ValueType]] ()
        for e <- triples do                                     // for each triple e
            val st = elabelMap.getOrElse ((e.h, e.t), null)     // get set of edge-labels
            if st != null then st.add (e.r)                     // add to existing set
            else elabelMap.put ((e.h, e.t), SET (e.r))          // make a new set
        end for
        elabelMap
    end buildElabelMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this triple-graph to a string in a shallow sense.
     *  Large arrays are not converted.  Use print to show all information.
     */
    override def toString: String =
        s"TripleGraph (sixe = $size, name = $name)"
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print this triple-graph in a deep sense with all the information.
     */
    def printG (): Unit =
        println (s"TripleGraph ($name, $size")
        println ("Triples: ")
        for e <- triples do println (s"\t $e")
    end printG

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** To check if two graphs are equal.
     *  @param g2_ the graph to be compared with this graph
     */
    override def equals (g2_ : Any): Boolean =
        g2_ match
        case _: TripleGraph =>
                val g2 = g2_.asInstanceOf [TripleGraph]
                if ! (label   sameElements g2.label)   then { println ("failed on label");   return false }
                if ! (triples sameElements g2.triples) then { println ("failed on triples"); return false }
                true
        case _ => false
        end match
    end equals

end TripleGraph


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tripleGraphTest` main function is used to test the `TripleGraph` class using
 *  examples from the `ExampleTrGraphD` object, which contains triple-digraphs whose
 *  vertex and edge labels are of type `Double`.
 *  > runMain scalation.database.triplegraph.tripleGraphTest
 *
@main def tripleGraphTest (): Unit =

    import ExampleTrGraphD._

    g1.printG ()
    q1.printG ()
    g2.printG ()
    q2.printG ()

end tripleGraphTest
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tripleGraphTest2` main function is used to test the `TripleGraph` class using
 *  examples from the `ExampleTrGraphD` object, which contains triple-digraphs whose
 *  vertex and edge labels are of type `String`.
 *  > runMain scalation.database.triplegraph.tripleGraphTest
 *
@main def tripleGraphTest2 (): Unit =

    import ExampleMuGraphS._

    g1.printG ()
    q1.printG ()
    g2.printG ()
    q2.printG ()

end tripleGraphTest2
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tripleGraphTest3` main function is used to test the `TripleGraph` class using
 *  a randomly generated triple-graph.
 *  > runMain scalation.database.triplegraph.tripleGraphTest3
 *
@main def tripleGraphTest3 (): Unit =

    val mgGen = new TripleGraphGen (0.0)

    private val nVertices = 20         // number of vertices
    private val nLabels   = 5          // number of distinct vertex labels
    private val eLabels   = 3          // number of distinct edge labels
    private val outDegree = 2          // average out degree
    private val inverse   = false      // whether inverse adjacency is used (parents)
    private val name      = "gr"       // name of the graph

    val mGraph = mgGen.genRandomGraph (nVertices, nLabels, eLabels, outDegree, inverse, name)
    MuGraph (mGraph, "mu" + name).printG ()

end tripleGraphTest3
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tripleGraphTest4` main function is used to test the `TripleGraph` class using
 *  the given data graph g and query graph q.
 *  > runMain scalation.database.triplegraph.tripleGraphTest4
 */
@main def tripleGraphTest4 (): Unit =

    val g = TripleGraph (Array (10.0, 11.0, 11.0, 11.0, 11.0, 10.0),
                         Bag (Triple (0, -1.0, 1),
                              Triple (0, -2.0, 1),
                              Triple (0, -3.0, 1),
                              Triple (0, -1.0, 3),
                              Triple (1, -1.0, 2),
                              Triple (4, -1.0, 2),
                              Triple (5, -1.0, 4),
                              Triple (5, -2.0, 4)),
                         "g")

    val q = TripleGraph (Array (10.0, 11.0, 11.0),
                         Bag (Triple (0, -1.0, 1),
                              Triple (0, -2.0, 1),
                              Triple (1, -1.0, 2)),
                         "q")

    banner ("Data Graph g")
    g.printG ()
    banner ("Query Graph q")
    q.printG ()

end tripleGraphTest4

