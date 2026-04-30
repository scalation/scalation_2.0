
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Apr  9 13:31:26 EDT 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Multi-way Tree Data Structure
 */

package scalation
package database

import scala.collection.mutable.{ArrayBuffer, Map}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat.VectorI
import scalation.random.{Randi, Variate}

type Pair = (Int, Int)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TreeNode` class is for a node in a tree.
 *  @param nid    the unique identifier for the node
 *  @param lev    the level of the node in the tree
 *  @param label  the node/incoming edge label
 *  @param ord    the birth order
 */
class TreeNode (val nid: Int, val lev: Int, var label: ValueType = null, var ord: Int = 0):

    val child = new ArrayBuffer [TreeNode] ()
    var parent: TreeNode = null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check if this node is an ancestor of node n.
     *  @param n  target node 
     */           
    def isAncestor (n: TreeNode): Boolean =
        if n == null then false else if this == n then true else isAncestor (n.parent)
    end isAncestor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this node is a leaf.
     */           
    def isLeaf: Boolean = child.size == 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the left sibling of 'this' node.
     */           
    def leftSibling: TreeNode =
        if parent != null && ord > 0 then parent.child(ord - 1)
        else null
    end leftSibling

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the right sibling of 'this' node.
     */           
    def rightSibling: TreeNode =
        if parent != null && ord < parent.child.size - 1 then parent.child(ord + 1)
        else null
    end rightSibling

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a tree node to a string.
     */
    override def toString: String = s"[ $nid, $label ]"

end TreeNode


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Tree` companion object provides methods for building trees.
 */
object Tree:

    private val debug        = debugf ("Tree", true)        // debug function
    private var rng: Variate = null                         // random generator for # children

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly generate a tree.
     *  @param depth   the depth of the tree
     *  @param minOut  the minimum number of children allowed (0 => binary tree)
     *  @param maxOut  the maximum number of children allowed (2 => binary tree)
     *  @param stream  the random number stream
     */
    def apply (depth: Int, minOut: Int = 0, maxOut: Int = 2, stream: Int = 0): Tree =
        rng = new Randi (minOut, maxOut, stream)            // random generator for # children
        val root = new TreeNode (0, 0)                      // make the root node of tree
        val tree = new Tree (root, depth)                   // make a tree from root
        if depth > 0 then
            val imax = rng.igen
            cfor (0, imax) { _ => genPre (depth, root, 1) }  // i, imax)       // add root's children

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Recursive helper method for generating a tree using a pre-order traversal.
         *  @param depth  the depth of the tree
         *  @param p      the parent node
         *  @param lev    the level of the node
         *  -param ord    the birth order of the node
         *  -param sibs   the number of siblings
         */
//      def genPre (depth: Int, p: TreeNode, lev: Int, ord: Int, sibs: Int): Unit =
        def genPre (depth: Int, p: TreeNode, lev: Int): Unit =
            val n = tree.add (p)                                            // add node n to tree
            if lev < depth then
                val imax = rng.igen
                cfor (0, imax) { _ => genPre (depth, n, lev+1) }  // i, imax)  // add n's children
        end genPre

        tree
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a tree from an inverted tree, i.e., a predecessor/parent list.
     *  @see `treeTest3` for an example.
     *  @param pred   the predecessor/parent vector
     *  @param labl   the node labels
     *  @param depth  the estimated average depth of the tree
     *  @param name   the name of tree
     */
    def apply (pred: VectorI, labl: Array [ValueType], depth: Int, name: String): Tree =
        debug ("apply", s"pred = ${stringOf (pred)}, name = $name")

        val lab   = if labl == null then Array.fill (pred.length)(null.asInstanceOf [ValueType])
                    else labl
        val root  = new TreeNode (0, 0, lab(0))             // for vertex 0 in g, create a root node
        val tree  = new Tree (root, depth, name)            // make a tree based on this root, est. depth
        val n_map = Map [Int, TreeNode] ()                  // node map from node id to tree node
        n_map += 0 -> root                                  // put the root node in the tree map

        for ni <- 1 until pred.length do
            val pi = pred(ni)
            if pi >= 0 then
                debug ("apply", s"pi = $pi, ni = $ni")
                val p = n_map.getOrElse (pi, { val pp = new TreeNode (pi, 1, lab(pi))
                                               n_map += pi -> pp; pp })
                val n = n_map.getOrElse (ni, { val nn = new TreeNode (ni, p.lev+1, lab(ni))
                                               n_map += ni -> nn; nn })
                tree.add (p, n)
            end if
        end for
        tree
    end apply

end Tree

import Tree._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Tree` class provides a data structure for multi-way trees.
 *  @param root   the root node of the tree
 *  @param depth  the estimated average depth of the tree
 *  @param name   the name of the tree
 */
class Tree (val root: TreeNode, depth: Int, val name: String = "tree"):

    private val debug = debugf ("Tree", true)               // debug function
    private val TAB   = "    "                              // spaces for TAB
//  private val MID   = 600.0                               // x coordinate for root
//  private val TOP   = 100.0                               // y coordinate for root
//  private val DIA   = 15.0                                // diameter for circles
    private val nodes = ArrayBuffer (root)                  // list of all nodes
    private var nCount = 0                                  // node counter for nid auto-increment

    debug ("init", s"create a tree of depth = $depth and name = $name")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th node.
     *  @param  the index of the node to return
     */
    def apply (i: Int): TreeNode = nodes(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create and add a new node, link it in the tree and return it.
     *  @param p  the parent node (null for root)
     */
    def add (p: TreeNode): TreeNode =
        nCount  += 1
        val n    = new TreeNode (nCount, p.lev+1)           // add node n
        nodes   += n                                        // add node n to nodes list
        n.parent = p                                        // comment out, if parent references not needed
        if p != null then
            p.child += n                                    // add n as child of p
            n.ord = p.child.size - 1                        // record n's birth order
        n                                                   // return node n
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a new node, link it in the tree and return it.
     *  @param p  the parent node (null for root)
     *  @param n  the new node to be added
     */
    def add (p: TreeNode, n: TreeNode): TreeNode =
        debug ("add", s"p = $p, n = $n")
        nCount  += 1
        nodes   += n                                        // add node n to nodes list
        n.parent = p                                        // comment out, if parent references not needed
        if p != null then
            p.child += n                                    // add n as child of p
            n.ord = p.child.size - 1                        // record n's birth order
        n                                                   // return node n
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of nodes in the tree.
     */
    def size: Int = nodes.length

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a map of labels for nodes with incoming edges in the tree.
     */
    def labelMap: Map [Pair, ValueType] =
        val labMap = Map [Pair, ValueType] ()
        for n <- nodes if n != root do labMap += (n.parent.nid, n.nid) -> n.label
        labMap
    end labelMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the tree.
     */
    def printTree (): Unit = printPre (root, 0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursive helper method for printing the tree using a preorder traversal.
     *  @param n    the current node to print
     *  @param lev  the level of the node => amount of indentation
     */
    private def printPre (n: TreeNode, lev: Int): Unit =
        print (TAB * lev)                                   // indent
        println (n)                                         // print node n
        for c <- n.child do printPre (c, lev+1)             // print subtrees
    end printPre

end Tree


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `treeTest` main function test2 the `Tree` class by randomly building a tree.
 *  > runMain scalation.database.treeTest
 */
@main def treeTest (): Unit =

    val ct = Tree (3, 2, 3)
    ct.printTree ()

end treeTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `treeTest2` main function tests the `Tree` class by manually building a tree.
 *  > runMain scalation.database.treeTest2
 */
@main def treeTest2 (): Unit =

    val FANOUT = 3
    val root = new TreeNode (0, 0)                          // nid = 0, lev = 0
    val ct = new Tree (root, 2)                             // root, depth = 2
    cfor (0, FANOUT) { _ =>
        val n = ct.add (ct.root)
        cfor (0, FANOUT) { _ => ct.add (n) }
    } // cfor
    ct.printTree ()

end treeTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `treeTest3` main function tests the `Tree` class by manually building a tree.
 *  > runMain scalation.database.treeTest3
 */
@main def treeTest3 (): Unit =

    val pred = VectorI (-1, 0, 0, 1, 1, 2, 2)
    val labl = Array [ValueType] (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
    val tree = Tree (pred, labl, 3, "t3")
    tree.printTree ()

end treeTest3

