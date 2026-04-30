
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Mar 14 19:19:32 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Simple Binary Tree Implementation
 */

package scalation
package database

import scala.collection.mutable.{ArrayBuffer => VEC}

class BinTree [T] (val elem: T):

    private var left: BinTree [T] = null
    private var righ: BinTree [T] = null

    def addLeft (item: T): BinTree [T] = { left = new BinTree (item); left }
    def addRigh (item: T): BinTree [T] = { righ = new BinTree (item); righ }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the binary tree.
     */
    def printTree (): Unit = printPre (this, 0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursive helper method for printing the tree using a pre-order traversal.
     *  @param n    the current node to print
     *  @param lev  the level of the node => amount of indentation
     */
    private def printPre (t: BinTree [T], lev: Int): Unit =
        print ("    " * lev)                                // indent
        println (t.elem)                                    // print node n
        if t.left != null then printPre (t.left, lev+1)     // print left sub-tree
        if t.righ != null then printPre (t.righ, lev+1)     // print right sub-tree
    end printPre

    private val leaves = VEC [T] ()                         // for collecting the leaves

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Collect all the leaves of the binary tree.
     */
    def collectLeaves (): VEC [T] = { collectPre (this); leaves }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursive helper method for collecting the leaves of the tree using a pre-order traversal.
     *  @param t  the current sub-tree
     */
    private def collectPre (t: BinTree [T]): Unit =
        if t.left == null && t.righ == null then leaves += t.elem
        else
            if t.left != null then collectPre (t.left)
            if t.righ != null then collectPre (t.righ)
    end collectPre

end BinTree

