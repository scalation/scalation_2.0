
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Matthew Saltz, John Miller
 *  @version 2.0
 *  @date    Fri Jul 10 12:39:33 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Methods for Reading from and Wrting to Files
 */

package scalation
package database
package graph_pm

import java.io.PrintWriter

import scala.collection.mutable.Map
import scala.collection.mutable.{Set => SET}
import scala.io.Source.fromFile

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GraphIO` class is used to write digraphs to a file.
 *  @param g  the digraph to write
 */
class GraphIO (g: Graph):

    import GraphIO.EXT

    private val debug = debugf ("GraphIO", true)                 // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write digraph g to a file in the following format:
     *      Graph (<name>, <inverse>, <nVertices>
     *      <vertexId> <label> <chVertex0> <chVertex1> ...
     *      ...
     *      )
     *  @param name  the file-name containing the graph's vertex, edge and label information
     *  @param base  the base sub-directory for storing graphs
     *  @param ext   the standard file extension for graph
     */
    def write (name: String = g.name, base: String = BASE_DIR, ext: String = EXT): Unit =
        val gFile = base + name + ext                             // relative path-name for file
        val pw    = new PrintWriter (gFile)
        debug ("write", s"gFile = $gFile")
        pw.println (s"Graph (${g.name}, ${g.inverse}, ${g.size}")
        for i <- g.ch.indices do  pw.println (g.toLine (i))
        for (k, v) <- g.elabel do pw.println (s"$k -> $v")
        pw.println (")")
        pw.close ()
    end write

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write the graph to TWO 'igraph' compatible files.
     *  @see igraph.sourceforge.net
     *  @param prefix
     */
    def write2IgraphFiles (prefix: String): (String, String) =
        val lFile = prefix + "igl.txt"
        val eFile = prefix + "ige.txt"
        val lOut  = new PrintWriter (lFile)
        g.label.foreach (lOut.println (_))
        lOut.close
        val eOut  = new PrintWriter (eFile)
        for i <- g.ch.indices do g.ch(i).foreach (x => eOut.println (s"$i $x"))
        eOut.close
        (lFile, eFile)
    end write2IgraphFiles

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Write the graph to TWO 'Neo4J' compatible files: 'lFile' and 'eFile' so that
     *  they may be fed into 'Neo4j' with one of its utilities.
     *  FIX:  need to handle multiple edge types.
     *  @param lFile  the file containing the graph labels (line: vertex-id TAB label)
     *  @param eFile  the file the edges (line: start-id TAB end-id TAB type)
     */
    def write2Neo4JFiles (lFile: String, eFile: String): Unit =
        val vertexLine = new PrintWriter (lFile)             // write the vertex ids and their labels
        vertexLine.println ("id\tlabel")
        g.label.foldLeft (1) { (i, l) => vertexLine.println (s"$i \t $l"); i + 1 }
        vertexLine.close
        val edgeLine = new PrintWriter (eFile)              // write the edges and their types.
        edgeLine.println ("start\tend\ttype")
        g.ch.foldLeft (1) { (i, v) =>
            v.foreach { c => edgeLine.println (s"$i \t ${(c+1)} \tEDGE") }
            i + 1
        } // foldLeft
        edgeLine.close
    end write2Neo4JFiles

end GraphIO


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GraphIO` object is the companion object to the `GraphIO` class and
 *  is used for reading graphs from files or graph databases.
 */
object GraphIO:

    val EXT = ".dg"                                         // standard file extension for digraphs

    private val flaw = flawf ("GraphIO")                    // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a set of child vertices '(v_0, v_1, ...)' from a string array.
     *  @param strArr  the string array 
     */
    def makeSet (strArr: Array [String]): SET [Int] =
        if strArr(0) == "" then SET [Int] ()
        else SET.from (strArr.map (_.toInt))
    end makeSet

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make an edge tuple '(u, v)' from a string array.
     *  @param strArr  the string array 
     */
    def makeTuple (strArr: Array [String]): (Int, Int) =
        (strArr(0).replace ("(", "").toInt, strArr(1).replace (")", "").toInt)
    end makeTuple

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a string into a label according to the type selector.
     *  @param s        the string to convert
     *  @param typeSel  the type selector
     */
    def toLabel (s: String, typeSel: Char = 'D'): ValueType =
        typeSel match
        case 'D' => s.toDouble.asInstanceOf [Double]
        case 'I' => s.toInt.asInstanceOf [Int]
        case 'S' => s.asInstanceOf [String]
        case _   => { flaw ("toLabel", "label type not supported"); null.asInstanceOf [ValueType] }
    end toLabel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read a digraph from a file based on the format used by 'print' and 'write':
     *      Graph (<name>, <inverse>, <nVertices>
     *      <vertexId>, <label>, <chVertex0>, <chVertex1> ...
     *      ...
     *      )
     *  @param name  the file-name containing the graph's vertex, edge and label information
     *  @param base  the base sub-directory for storing graphs
     *  @param ext   the standard file extension for graph
     *  @param sep   the character separating the values (e.g., ',', ' ', '\t')
     */
    def apply (name: String, base: String = BASE_DIR, ext: String = EXT, sep: Char = ','): Graph =
        val gFile  = base + name + ext                             // relative path-name for file
        val l      = fromFile (gFile).getLines ().toArray          // get the lines from gFile
        val l0     = l(0).split ('(')(1).split (sep).map (_.trim)  // array for line 0
        val n      = l0(2).toInt                                   // number of vertices
        val ch     = Array.ofDim [SET [Int]] (n)                   // adjacency: array of children (ch)
        val label  = Array.ofDim [ValueType] (n)                   // array of vertex labels
        val elabel = Map [(Int, Int), ValueType] ()                // map of edge labels
        println (s"apply: read $n vertices from $gFile")

        for i <- ch.indices do
            val li   = l(i+1).split (sep).map (_.trim)             // line i (>0) splits into i, label, ch
            label(i) = toLabel (li(1))                             // make vertex label
            ch(i)    = makeSet (li.slice (2, li.length) )          // make ch set
        end for
        for i <- n+1 until l.length-1 do
            val li  = l(i).split ("->").map (_.trim)               // line i (>n) splits into (u, v) -> elabel
            elabel += makeTuple (li(0).split (sep)) -> toLabel (li(1))
        end for
        new Graph (ch, label, elabel, l0(1) == "true", l0(0))
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read a graph from TWO files:
     *  'lFile' is a file with one label per line, where each line represents
     *      the vertex with id <lineNumber>.
     *  'eFile' is a file with each line representing the vertex with id
     *      <lineNumber>, and each line contains a space-separated list of vertices
     *      to which the current vertex is adjacent.
     *  @param lFile  the file containing the graph labels
     *  @param eFile  the file the edges (to create adjacency sets)
     */
    def read2Files (lFile: String, eFile: String): Graph =
        val lLines = fromFile (lFile).getLines ()                  // get the lines from lFile
        val label  = lLines.map (x => toLabel (x.trim)).toArray    // make the label array
        val eLines = fromFile (eFile).getLines ()                  // get the lines from eFile
        val ch     = eLines.map (line =>                           // make the adj array
            if line.trim != "" then
                line.split (" ").map (x => x.trim.toInt).toSet.asInstanceOf [SET [Int]]
            else
                SET [Int] ()
        ).toArray
        new Graph (ch, label, null)                                // FIX: elabels?
    end read2Files

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Read a graph from TWO specially formatted Pajek files.
     *  @param lFile  the file containing the graph labels
     *  @param eFile  the file the edges (to create adjacency sets)
     */
    def read2PajekFile (lFile: String, eFile: String): Graph =
        val lLines = fromFile (lFile).getLines ()                  // get the lines from lFile
        val label  = lLines.map (x => toLabel (x.trim)).toArray
        val ch     = Array.ofDim [SET [Int]] (label.size)
        for (i <- ch.indices) ch(i) = SET [Int] ()
        val eLines = fromFile (eFile).getLines ()                  // get the lines from eFile
        for line <- eLines do
            val splitL = line.split (" ").map (_.trim)
            val adjs   = splitL.slice (1, splitL.length).map(_.trim.toInt).toSet
            ch(splitL(0).toInt-1) ++= adjs
        end for
        new Graph (ch, label, null)                                 // FIX: elabels?
    end read2PajekFile

end GraphIO


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `graphIOTest` main function is used to test the `GraphIO` class and object.
 *  > runMain scalation.database.graph_pm.graphIOTest
 */
@main def graphIOTest (): Unit =

    val gGen = new GraphGen ('D')

    val name     = "ran_graph"    // the name of the graph
    val size     = 50             // size of the graph
    val nLabels  = 10             // number of distinct vertex labels
    val eLabels  =  5             // number of distinct edge labels
    val avDegree =  3             // average vertex out degree for the graph
    val inverse  = false

    // Create a random graph and print it out

    val ran_graph = gGen.genRandomGraph (size, nLabels, eLabels, avDegree, inverse, "ran_graph")
    println (s"ran_graph = $ran_graph")
    ran_graph.printG ()
    ran_graph.printG ()

    // Write the graph to a file

    val mgIO = new GraphIO (ran_graph)
    println ("start writing graph to " + name)
    mgIO.write ()
    println ("end writing graph to " + name)

    // Read the file to create a new identical graph

    val g = GraphIO (name)
    println (s"g = $g")
    g.printG ()

end graphIOTest

