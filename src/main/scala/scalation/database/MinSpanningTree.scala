
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Peng Hao
 *  @version 2.0
 *  @date    Sat Nov  7 21:01:31 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Minimum Spanning Tree Implementing Prim's Algorithm
 *           Use negative edge labels/weights for Maximum Spanning Tree
 */

package scalation
package database

import scala.collection.mutable.{Map, Set => SET}
import scala.runtime.ScalaRunTime.stringOf

import scalation.database.graph_pm.Graph
import scalation.mathstat.{MatrixD, VectorI}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MinSpanningTree` class is used to build minimum cost spanning trees
 *  from graphs.  Edge cost/weights are given by edge labels. `MinSpanningTree`
 *  implements Prim's algorithm.
 *  `scalation.PriorityQueue` ScalaTion's extension adds increaseKey, printInOrder
 *  @see www.cse.ust.hk/~dekai/271/notes/L07/L07.pdf
 *  @param g           the digraph to build the spanning tree from
 *  @param undirected  whether the graph is already undirected
 */
class MinSpanningTree (g: Graph, undirected: Boolean = true):

    private val debug = debugf ("MinSpanningTree", true)                  // debug flag
    private var stree: Tree = null                                        // spanning tree built by calling span
    private val size  = g.size                                            // the number of nodes for the spanning tree
//  private val root  = new TreeNode (0, 0, 0.0)                          // for vertex 0 in g, create a root node
    private val key   = Array.fill (size)(MAX_VALUE)                      // cost/key array
    private val out   = Array.fill (size)(true)                           // status of outside spanning tree
    private val qu    = PriorityQueue ()(using NodeOrder)                 // priority queue of vertices
    for i <- 0 until size do qu.enqueue (Elem (i, key(i)))                // put all vertices in priority queue

    debug ("init", s"size = $size, already undirected = $undirected")

    if ! undirected then g.makeUndirected ()                              // algorithm works on undirected graphs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the spanning tree.
     */
    def printSTree (): Unit = stree.printTree ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a minimum cost spanning tree for the given graph, returning true
     *  if a complete spanning tree connecting all of g's vertices can be created.
     */
    def span (): Tree =
        val pred = makeITree ()                                           // make an inverted tree
        val el   = Array.ofDim [ValueType] (pred.length)                  // copy elabel value from g into a pred elabel array
        for i <- 1 until el.length do el(i) = g.elabel(pred(i), i).toDouble  // skipping root node (0)
        stree = Tree (pred, el, 4, "st")                                  // build spanning tree from pred vector
        stree
    end span

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Elem` class is used for ordering elements on a priority queue.
     *  @param idx  the index of a node
     *  @param key  the ordering key (based on cost) for a node
     */
    case class Elem (idx: Int, key: Double)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `NodeOrder` object defines the order of node indices based on
     *  their 'key' value.  Using -key to get "smallest first" in priority queue.
     *  This is for minimum spanning trees ('min' = true)
     */
    object NodeOrder extends Ordering [Elem]:
        def compare (e1: Elem, e2: Elem): Int = -e1.key compare -e2.key
    end NodeOrder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make an inverted tree by recording the predecessor/parent array.
     *  Each node except the root will have one parent.  See pseudo-code on p. 28
     *  @see www.cse.ust.hk/~dekai/271/notes/L07/L07.pdf
     */
    def makeITree (): VectorI =
        val pred = VectorI.fill (size)(-1)                                // predecessor node array
        key(0)   = null.asInstanceOf [Double]                             // start at the root (node index 0)

        while qu.nonEmpty do                                              // until all vertices in spanning tree
//          qu.printInOrder                                               // print qu in order, comment out to reducing printing
            val i = qu.dequeue ().idx                                     // return and remove least cost vertex
            debug ("makeITree", s"dequeued i = $i")
            for j <- g.ch(i) if out(j) do                                 // iterate through its outside children
                val cost = g.elabel (i, j).toDouble                       // get cost from edge label

                if cost < key(j) then
                   qu.increaseKey (Elem (j, key(j)), Elem (j, cost))      // reposition j toward front in priority queue
                   key(j)  = cost                                         // lower the cost for node index j
                   pred(j) = i                                            // set pred of j to parent i
                end if
            end for
            out(i) = false                                                // now finished with i
        end while
        debug ("makeITree", s"pred = ${stringOf (pred)}")
        pred
    end makeITree

end MinSpanningTree


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MinSpanningTree` companion object provides factory methods for building
 *  Minimum Spanning Trees.
 */ 
object MinSpanningTree:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Minimum Spanning Tree from a graph's an adjacent matrix representation.
     *  @param adj         the adjacent matrix representation of a digraph
     *  @param undirected  whether the graph is already undirected
     */
    def apply (adj: MatrixD, undirected: Boolean = true): MinSpanningTree =
        val g = Graph.fromMatrix (adj)
        new MinSpanningTree (g, undirected) 
    end apply

end MinSpanningTree


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MaxSpanningTree` object provides factory methods for building
 *  Maximum Spanning Trees.  The adjacent matrix's edge weights are negated to
 *  produce maximum.
 */   
object MaxSpanningTree:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Maximum Spanning Tree from a graph's an adjacent matrix representation.
     *  @param adj         the adjacent matrix representation of a digraph
     *  @param undirected  whether the graph is already undirected
     */
    def apply (adj: MatrixD, undirected: Boolean = true): MinSpanningTree =
        val g = Graph.fromMatrix (-adj)                                   // negate edge weights
        new MinSpanningTree (g, undirected)
    end apply

end MaxSpanningTree


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `minSpanningTreeTest` main function tests the `MinSpanningTree` class.
 *  @see www.cse.ust.hk/~dekai/271/notes/L07/L07.pdf
 *  > runMain scalation.database.minSpanningTreeTest
 */
@main def minSpanningTreeTest: Unit =

    val g = new Graph (Array (SET (1, 3, 4),               // ch(0)
                              SET (2, 3),                  // ch(1)
                              SET (3, 5),                  // ch(2)
                              SET (4, 5),                  // ch(3)
                              SET (),                      // ch(4)
                              SET ()),                     // ch(5)
                              Array.fill (6)(-1.0),        // vertex labels
                       Map ((0, 1) -> 1.0,                 // edge labels
                            (0, 3) -> 10.0,
                            (0, 4) -> 3.0,
                            (1, 2) -> 2.0,
                            (1, 3) -> 3.0,
                            (2, 3) -> 4.0,
                            (2, 5) -> 5.0,
                            (3, 4) -> 4.0,
                            (3, 5) -> 1.0))
    g.printG ()

    val st = new MinSpanningTree (g)
    st.span ()
    println ("spanning tree for graph " + g.name)
    println ("-" * 60)
    st.printSTree ()

end minSpanningTreeTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `minSpanningTreeTest2` main function tests the `MinSpanningTree` class.
 *  This test the Maximum Spanning Tree option.
 *  @see www.cse.ust.hk/~dekai/271/notes/L07/L07.pdf
 *  > runMain scalation.database.minSpanningTreeTest2
 */
@main def minSpanningTreeTest2 (): Unit =

    val g = new Graph (Array (SET (1, 3, 4),               // ch(0)
                              SET (2, 3),                  // ch(1)
                              SET (3, 5),                  // ch(2)
                              SET (4, 5),                  // ch(3)
                              SET (),                      // ch(4)
                              SET ()),                     // ch(5)
                              Array.fill (6)(-1.0),        // vertex labels
                       Map ((0, 1) -> 1.0,                 // edge labels
                            (0, 3) -> 10.0,
                            (0, 4) -> 3.0,
                            (1, 2) -> 2.0,
                            (1, 3) -> 3.0,
                            (2, 3) -> 4.0,
                            (2, 5) -> 5.0,
                            (3, 4) -> 4.0,
                            (3, 5) -> 1.0))
    g.printG ()

    val st = new MinSpanningTree (g, false)
    st.span ()
    println ("spanning tree for graph " + g.name)
    println ("-" * 60)
    st.printSTree ()

end minSpanningTreeTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `minSpanningTreeTest3` main function tests the `MinSpanningTree` class by
 *  building a graph from an adjacency matrix called cmi for Conditional Mutual
 *  Information and creating a minimum spanning tree from it.
 *  @see scalation.modeling.classifying.BayesClassifier
 *  > runMain scalation.database.minSpanningTreeTest3
 */
@main def minSpanningTreeTest3 (): Unit =

    val cmi = MatrixD ((4, 4), 0.00000, 0.419593, 0.222815, 0.311752,
                               0.00000, 0.00000,  0.419593, 0.168895,
                               0.00000, 0.00000,  0.00000,  0.0610538,
                               0.00000, 0.00000,  0.00000,  0.00000)
    println (s"cmi = $cmi")

    banner ("Graph from Adjacency Matrix")
    val g = Graph.fromMatrix (cmi)
    g.printG ()

    banner ("Minimum Spanning Tree from Directed Graph")
    val st = new MinSpanningTree (g, false)
    st.span ()
    println ("-" * 60)
    st.printSTree ()

end minSpanningTreeTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `minSpanningTreeTest4` main function tests the `MinSpanningTree` class by
 *  building a graph from an adjacency matrix called cmi for Conditional Mutual
 *  Information and creating a maximum spanning tree from it.
 *  @see scalation.modeling.classifying.BayesClassifier
 *  > runMain scalation.database.minSpanningTreeTest4
 */
@main def minSpanningTreeTest4 (): Unit =

    import scalation.mathstat.MatrixD

    val cmi = MatrixD ((4, 4), 0.00000, 0.419593, 0.222815, 0.311752,
                               0.00000, 0.00000,  0.419593, 0.168895,
                               0.00000, 0.00000,  0.00000,  0.0610538,
                               0.00000, 0.00000,  0.00000,  0.00000)
    println (s"cmi = $cmi")

    banner ("Graph from Adjacency Matrix")
    val g = Graph.fromMatrix (-cmi)                                // use negative edge weights for max
    g.printG ()

    banner ("Maximum Spanning Tree from Directed Graph")
    val st = new MinSpanningTree (g, false)
    st.span ()
    println ("-" * 60)
    st.printSTree ()

end minSpanningTreeTest4

