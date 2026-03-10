
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed May 22 14:17:49 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Base Trait for Decision Trees
 */

package scalation
package modeling
package classifying

import scala.collection.mutable.{ArrayBuffer, Set, SortedMap}

import scalation.mathstat.{VectorD, VectorI}
import scalation.mathstat.Probability.entropy

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree` companion object provides the hyper-parameters for the
 *  decision trees, bagging trees, and random forests.
 *  @see `scalation.modeling.HyperParameter`
 */
object DecisionTree:

    /** hyper-parameters for tuning decision trees and random forests
     */
    val hp = new HyperParameter
    hp += ("height",  4, 4)            // the height (edges in longest path) limit for the decision trees
    hp += ("cutoff",  0.01, 0.01)      // the cutoff (stop splitting) entropy threshold
    hp += ("nTrees",  11, 11)          // the (odd) number of trees to create for the forest (e.g., 11 to 51)
    hp += ("bRatio",  0.7, 0.7)        // the bagging ratio (fraction of the data/rows to be used in building trees)
    hp += ("fbRatio", 0.7, 0.7)        // the feature bagging ratio (fraction of the features/columns to be used in building trees)

    // Before creating a new model, update some the hyper-parameter - the rest will take default values, e.g.,
    //
    // Decision.hp("height") = 5
    //
    // Many trees may be needed to get good results for `RandomForest`
    //
    // val hp2 = hp.updateReturn (("nTrees", 51), ("bRatio", 0.6), ("height", 6.0), ("fbRatio", 0.9))
    // val mod = new RandomForest (x, y, fname, k, cname, conts, hp2)

end DecisionTree


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree` trait provides common capabilities for all types of
 *  decision trees.
 */
trait DecisionTree:

    private val debug      = debugf ("DecisionTree", true)           // debug function
    private var root: Node = null                                    // the root node
    private [classifying] val leaves = ArrayBuffer [Node] ()         // array buffer of leaf nodes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the root node to the tree.
     *  @param r  the root node of the tree
     */
    def addRoot (r: Node): Unit = root = r

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add child node c to the tree via branch v from node n.
     *  @param n  the parent node
     *  @param v  the branch value from the parent node
     *  @param c  the child node
     */
    def add (n: Node, v: Int, c: Node): Unit =
        c.pv = v                                                 // branch value from parent to child
        n.branch += v -> c                                       // add to parent's branch map
        if c.leaf then leaves += c                               // if leaf, add to leaves
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add multiple child nodes to the tree via branchs from node 'n'.
     *  @param n   the parent node
     *  @param vc  the branch value and child node, repeatable
     */
    def add (n: Node, vc: (Int, Node)*): Unit =
        for (v, c) <- vc do add (n, v, c)
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** As part of tree pruning, turn an internal node into a leaf.
     *  @param n  the node to turn into a leaf (pruning all nodes below it)
     */
    def makeLeaf (n: Node): Unit =
        if ! n.leaf then
            for c <- n.branch.values do leaves -= c              // remove children from leaves
            n.branch.clear ()                                    // clear branch map
            n.leaf = true                                        // set leaf flag
            leaves += n                                          // add n to leaves
        else
            println (s"makeLeaf: node $n already is a leaf")
    end makeLeaf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether all the children of node n are leaf nodes.
     *  @param n  the node in question
     */
    def leafChildren (n: Node): Boolean = n.branch.values.forall (_.leaf)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find candidate nodes that may be pruned, i.e., those that are parents
     *  of leaf nodes, restricted to those that don't have any children that
     *  are themselves internal nodes.
     */
    def candidates: Set [Node] =
        val can = Set [Node] ()
        for n <- leaves do
            val p = n.parent
            if leafChildren (p) then can += p
        end for
        can
    end candidates

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Of all the pruning candidates, find the one with the least gain.
     *  @param can  the nodes that are canidates for pruning
     */
    def bestCandidate (can: Set [Node]): (Node, Double) =
        var min = Double.MaxValue
        var best: Node = null
        for n <- can if n.gn < min do { min = n.gn; best = n }
        (best, min)
    end bestCandidate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the entropy of the tree as the weighted average over the list
     *  of nodes (defaults to leaves).
     *  @param nodes  the nodes to compute the weighted entropy over
     */
    def calcEntropy (nodes: ArrayBuffer [Node] = leaves): Double =
        var sum, ent = 0.0
        for n <- nodes do
            sum += n.nu_sum                                      // add number of counts for node n
            ent += n.nu_sum * entropy (n.nu)                     // unnormalized weighted entropy
        end for
        debug ("calcEntropy", s"number of nodes = ${nodes.size}, sum = $sum, ent = $ent")
        ent / sum                                                // normalized weighted entropy
    end calcEntropy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Auxilliary predict method facilitating recursion for `VectorI`.
     *  @param z  the data vector to classify
     *  @param n  the current node in the tree
     */
    def predictIrec (z: VectorI, n: Node = root): Int =
        if n.leaf then n.y
        else
            val zj = z(n.j)
            try predictIrec (z, n.branch(zj))
            catch
                case _ : NoSuchElementException => n.nu.argmax ()    // take consensus of node n
        end if
    end predictIrec

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Auxilliary classify method facilitating recursion for `VectorD`.
     *  @param z  the data vector to classify
     *  @param n  the current node in the tree
     */
    def predictIrecD (z: VectorD, n: Node = root): Int =
        if n.leaf then n.y
        else
            val zj = z(n.j)
            try
                val cont = n.thres > Double.NegativeInfinity
                if cont then predictIrecD (z, if zj <= n.thres then n.branch(0) else n.branch(1))
                else predictIrecD (z, n.branch(zj.toInt))
            catch
                case _ : NoSuchElementException => n.nu.argmax ()    // take consensus of node n
        end if
    end predictIrecD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the decision tree using 'prinT' method from `Node` class.
     */
    def printTree (): Unit =
        println ("Decision Tree:")
        Node.printT (root, 0)
        println ()
    end printTree

end DecisionTree


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Node` class is used to hold information about a node in the decision tree.
 *  @param j       the feature/variable number used for splitting (negative => leaf)
 *  @param gn      the information gain recorded at this node
 *  @param nu      the frequency count
 *  @param parent  the parent node (null for root)
 *  @param y       the response/decision value
 *  @param leaf    whether the node is a leaf (terminal node)
 */
case class Node (j: Int, gn: Double, nu: VectorI, parent: Node = null, y: Int,
                 private [classifying] var leaf: Boolean = false)
     extends Cloneable:

    private [classifying] var pv = -1                                  // the branch value from the parent node to this node
    private [classifying] var thres: Double = Double.NegativeInfinity  // threshold for continuous/ordinal features
    private [classifying] val nu_sum = nu.sum                          // sum of frequency counts
    private [classifying] val branch = SortedMap [Int, Node] ()        // maps the branch value, e.g., f2 has values 0, 1, 3,
                                                                       //                              for nodes n0, n1, n2

    override def toString: String = 
       if ! leaf && thres > Double.NegativeInfinity then
           s"$pv -> \t Node (j = $j, nu = $nu, y = $y, leaf = $leaf, thres = $thres)"
       else
           s"$pv -> \t Node (j = $j, nu = $nu, y = $y, leaf = $leaf)"

end Node


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Node` companion object provides helper functions.
 */
object Node:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively print the decision tree nodes, indenting each level.
     *  Requires node n to be not null
     *  @param n       the current node
     *  @param level   the level of node in the tree
     */
    def printT (n: Node, level: Int): Unit =
        if n == null then println (s"X printT: n = $n, level = $level")
        if n.leaf then
            println ("\t" * level + "[ " + n + " ]")
        else
            println ("\t" * level + "[ " + n)
            for c <- n.branch.values do printT (c, level + 1)
            println ("\t" * level + "]")
    end printT

end Node


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTreeTest` main function is used to test the `DecisionTree` class.
 *  > runMain scalation.modeling.classifying.decisionTreeTest
 */
@main def decisionTreeTest (): Unit =

    object Tree extends DecisionTree

    val n0 = Node (0, -0, VectorI (5, 9), null, 1)
    Tree.addRoot (n0)                                            // add root n0
        val n1 = Node (3, -0, VectorI (2, 3), n0, 1)
            val n2 = Node (-1, -0, VectorI (0, 3), n1, 1, true)
            val n3 = Node (-2, -0, VectorI (2, 0), n1, 0, true)
        Tree.add (n1, (0, n2), (1, n3))                          // add children of n1
        val n4 = Node (-3, -0, VectorI (0, 4), n0, 1, true)
        val n5 = Node (2, -0, VectorI (3, 2), n0, 0)
            val n6 = Node (-4, -0, VectorI (0, 2), n5, 1, true)
            val n7 = Node (-5, -0, VectorI (3, 0), n5, 0, true)
        Tree.add (n5, (0, n6), (1, n7))                          // add children of n5
    Tree.add (n0, (0, n1), (1, n4), (2, n5))                     // add children of n0

    Tree.printTree ()
    println (s"inital entropy = ${Tree.calcEntropy (ArrayBuffer (n0))}")
    println (s"final  entropy = ${Tree.calcEntropy ()}")

    println ("Classify New Data")
    val z = VectorI (2, 2, 1, 1)                                 // new data vector to classify
    println (s"predictI ($z) = ${Tree.predictIrec (z)}")

end decisionTreeTest

