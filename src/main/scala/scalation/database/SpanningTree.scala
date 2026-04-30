
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Nov  7 21:01:31 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Spanning Tree
 *
 *  @see    `MinSpanningTree`
 */

package scalation
package database

import scala.collection.mutable.{Map, Queue}

import scalation.mathstat.MatrixD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SpanningTree` class is used to build spanning trees from graphs.
 *  @param g  the digraph (adjacency matrix) to build the spanning tree from
 */
class SpanningTree (g: MatrixD):

    private val debug = debugf ("SpanningTree", true)         // debug function

    private val size  = g.dim                                 // the number of nodes for the spanning tree
    private val root  = new TreeNode (0, 0, 0.0)              // for vertex 0 in g, create a root node
    private val stree = new Tree (root, 4)                    // make a tree based on this root, est. depth 
    private val n_map = Map [Int, TreeNode] ()                // node map from node id to tree node
    private val qu    = Queue (0)                             // queue of vertices in the spanning tree
    private val out   = Array.fill (size)(true)               // status of outside spanning tree
    out(0) = false                                            // root is in (not outside)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the spanning tree.
     */
    def printSTree (): Unit = stree.printTree ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a spanning tree for the given graph, returning true if a complete
     *  spanning tree connecting all of g's nodes can be created.
     */
    def span (): Boolean =
        n_map += 0 -> root                                    // put the root in the map   
        var cnt = 0
        var i   = 0
        while cnt < size - 1 && qu.nonEmpty do
            i = qu.front                                      // get node i from queue
            var j = -1 
            while
                j = findNext (i, j)
                cnt < size-1 && j != -1
            do
                cnt += 1
                debug ("span", s"cnt = $cnt, i = $i, j = $j")
                val p = n_map (i)                             // parent node from parent index
                val n = new TreeNode (j, p.lev+1, 0.0)        // make a tree node for ni
                n_map += j -> n                               // put node j in the map
                stree.add (p, n)                              // connect parent p to node n
            end while
            qu.dequeue ()                                     // nothing left for node i 
        end while
        cnt == size - 1
    end span

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find and return the first node j to connect with current node i to expand
     *  the spenning tree.  Node j must be out (i.e., not yet in the spanning tree).
     *  @param i   the index of current node to make to connection from
     *  @param j0  look for node j past j0 (the last one found)
     */
    private def findNext (i: Int, j0: Int): Int =
        var j = j0 + 1
        while j < size && i == j && g(i, j) == 0.0 && ! out(j) do j += 1
        if j < size then                                      // node j is next
            out(j) = false                                    // j is no longer outsise the tree
            qu.enqueue (j)
            return j
        -1
    end findNext

end SpanningTree


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `spanningTreeTest` object is used to test the `SpanningTree` class.
 *  @see scalation.modeling.classifying.BayesClassifier
 *  > runMain scalation.database.spanningTreeTest
 */
@main def spanningTreeTest (): Unit =

    val cmi = MatrixD ((4, 4), 0.00000, 0.419593, 0.222815, 0.311752,
                               0.00000, 0.00000,  0.419593, 0.168895,
                               0.00000, 0.00000,  0.00000,  0.0610538,
                               0.00000, 0.00000,  0.00000,  0.00000)
    println (s"cmiAns = $cmi")

    val st = new SpanningTree (cmi)
    if ! st.span () then print ("un")
    println ("able to complete a spanning tree for this graph")
    println ("-" * 60)
    st.printSTree ()

end spanningTreeTest

