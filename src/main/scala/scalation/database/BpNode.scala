
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Aug  9 01:25:50 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    B+Tree Node with find, add, split, remove, and merge Operations
 */

package scalation
package database

import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BpNode` companion object provides settings for node sizes.
 */
object BpNode:

    private var DEFAULT_DLINK = true                               // default value for DLINK

    private val debug = debugf ("BpNode", false)                   // debug function
    private val flaw  = flawf ("BpNode")                           // flaw function

    private var order = 5                                          // maximum number of references (reset as needed)
    private var half  = order / 2                                  // half of order keys (floor)
    private var halfp = order - half                               // rest of the overflowed keys
    private var mink  = halfp - 1                                  // minimum number of keys (before underflow)

    debug ("init", s"order = $order, half = $half, halfp = $halfp, mink = $mink")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the leaf node DEFAULT_DLINK to dlink (false => forward links only via ref(0),
     *  true => both forward links via ref(0) and backward links via pre).
     *  @see `BpNode` class for specification of ref(0) and pre
     *  @param dlink  whether to support unidirectionsl (false) or bidirections (true) linkage
     */
    def set_DEFAULT_DLINK (dlink: Boolean): Unit = DEFAULT_DLINK = dlink

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the order of BpNodes to order_.
     *  @param order_  the new order for BpNodes (must be at least 4)
     */
    def setOrder (order_ : Int): Unit =
        if order < 3 then flaw ("setOrder", s"order_ = $order_ must be at least 3")
        else
            order = order_
            half  = order / 2
            //          half  = (order - 1) / 2
            halfp = order - half
            mink  = half - 1
    end setOrder

end BpNode

import BpNode._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BpNode` class defines nodes of size order that that may be stored in a B+tree.
 *  Keys have type `ValueType` and may reference values of `Any` type.
 *  @param  keys    number of active keys
 *  @param  isLeaf  whether this node is a leaf
 *  @param  DLINK   whether the leaf nodes support linkage in both directions (ref(0) & pre)
 */
class BpNode (private [database] var keys: Int, val isLeaf: Boolean, DLINK: Boolean = DEFAULT_DLINK)
    extends Serializable:

    private [database] val key = Array.ofDim [ValueType] (order)    // array to hold keys
    private [database] val ref = Array.ofDim [Any] (order + 1)      // array to hold values or reference nodes
    private [database] var pre: BpNode = null                       // reference to previous LEAF node (if needed)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a new root node with one key (and two references) in it.
     *  @param left   the left node (< dkey)
     *  @param dkey   the divider key
     *  @param right  the right node (>= dkey)
     */
    def this (left: BpNode, dkey: ValueType, right: BpNode) =
        this (1, false)
        key(0) = dkey                                               // divider key
        ref(0) = left; ref(1) = right                               // left and right references
    end this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this node has overflowed (too many keys).
     */
    inline def overflow: Boolean = keys >= order

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this node has underflowed (too few keys).
     */
    inline def underflow: Boolean = keys < mink

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this node is rich (i.e., has surplus keys).
     */
    inline def rich: Boolean = keys > mink

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the key at index position i.
     *  @param i  the index position in this node
     */
    inline def apply (i: Int): ValueType =
        if i >= keys then flaw ("apply", s"index i = $i must be less than keys = $keys")
        key(i)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the key at index position i.
     *  @param i  the index position in this node
     *  @param k  the new value for key(i)
     */
    inline def update (i: Int, k: ValueType): Unit =
        if i >= keys then flaw ("update", s"index i = $i must be less than keys = $keys")
        key(i) = k
    end update

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find and return the first position where 'k < key_i' in this node.
     *  If none, return keys (corresponds to last ref).
     *  @param k  the key whose index position is sought
     */
    def find (k: ValueType): Int =
        var (found, i) = (false, 0)
        while ! found && i < keys do
            if k < key(i) then found = true
            else i += 1
        i
    end find

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find and return the first position where 'k == key_i' in this node.
     *  If none, return -1 meaning not found.
     *  @param k  the key whose index position is sought
     */
    def findEq (k: ValueType): Int =
        var (found, i) = (false, 0)
        while ! found && i < keys do
            if k == key(i) then found = true
            else i += 1
        if i < keys then i else -1
    end findEq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shift keys from ip to keys one position to the RIGHT (make room for insertion).
     *  Also shift all references to the right of key ip.
     *  @param ip  the future insertion position
     */
    def shiftR (ip: Int): Unit =
        for i <- keys until ip by -1 do                             // make room by shifting keys right
            key(i)   = key(i-1)                                     // move key right
            ref(i+1) = ref(i)                                       // ref to the right of key
    end shiftR

    def shiftIR(ip: Int): Unit =
        ref(keys + 1) = ref(keys)
        for i <- keys until ip by -1 do // make room by shifting keys right
            key(i) = key(i - 1) // move key right
            ref(i) = ref(i - 1) // ref to the right of key
    end shiftIR


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the new key k and value v into this node at the to be found insertion
     *  position (ip).  If a duplicate key is entered, return the OLD VALUE stored
     *  with the key, otherwise return None (meaning no duplicate was found)..
     *  @param k  the new key
     *  @param v  the new value (or node for internal nodes)
     */
    def add (k: ValueType, v: Any): Option [Any] =
        val ip = find (k)                                           // find insertion position (ip)
        val duplicate = ip != 0 && k == key(ip-1)                   // determine whether the new key is a duplicate
        debug ("add", s"(k = $k, v = $v) pair at ip = $ip")

        if duplicate then
            val old = ref(ip)                                       // save the OLD VALUE for duplicate key
            ref(ip) = v                                             // replace OLD VALUE with new value
            Some (old)                                              // return the OLD VALUE, indicates duplicate
        else
            shiftR (ip)                                             // make room by shifting keys right
            key(ip)   = k                                           // insert new key
            ref(ip+1) = v                                           // insert new value (right of key)
            keys     += 1                                           // increment the number of active keys
            None                                                    // indicates no duplicate found
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split this LEAF node by creating a right sibling node (rt) and moving
     *  half the keys and references to that new node, leaving halfp.
     *  Return the divider key and the right sibling node.
     */
    def split (): (ValueType, BpNode) =
        val rt = new BpNode (half, true)                            // allocate leaf right sibling node (rt)
        for i <- 0 until half do                                    // move largest half of keys (with refs) to rt
            rt.key(i)   = key(halfp + i)
            rt.ref(i+1) = ref(halfp + i + 1)                        // refs are right of keys
        rt.ref(0) = ref(0)                                          // update LINKED LIST of nodes
        if DLINK then rt.pre = this                                 // reverse link
        ref(0)    = rt                                              // this -> rt -> old-right
        keys      = halfp                                           // reset number of active keys to help plus
        (rt.key(0), rt)                                             // (divider key (smallest right) and right sibling
    end split

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split this INTERNAL node by creating a right sibling rt and moving half
     *  the keys and references to that new node, leaving halfp - 1.
     *  Return the divider key and the right sibling node.
     */
    def splitI (): (ValueType, BpNode) =
        val rt = new BpNode (half, false)                           // allocate internal right sibling node (rt)
        for i <- 0 until half do                                    // move largest half of keys (with refs) to rt
            rt.key(i)   = key(halfp + i)
            rt.ref(i) = ref(halfp + i)
        rt.ref(half) = ref(keys)                                    // copy over the last ref
        keys = halfp - 1                                            // reset number of active keys to help plus - 1
        (key(keys), rt)                                             // divider key (middle key) and right sibling
    end splitI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove key at dp and its reference from this node and check for underflow.
     *  @param dp  the deletion index position (may use findEq to find it)
     */
    def remove (dp: Int): Boolean =
        debug ("remove", s"dp = $dp, key = ${key(dp)} ")
        for i <- dp until keys do                                   // remove at dp by shifting keys left
            key(i)   = key(i+1)
            ref(i+1) = ref(i+2)                                     // case: ref on right
        keys -= 1                                                   // decrement the number of keys
        underflow                                                   // if true, node underflows
    end remove

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove key at dp and its reference from this INTERNAL node and check for underflow.
     *  @param dp  the deletion index position (may use findEq to find it)
     */
    def removeI (dp: Int): Boolean =
        debug("remove", s"dp = $dp, key = ${key(dp)}")
        for i <- dp until keys do                                   // remove at dp by shifting keys left
            key(i) = key(i + 1)
            ref(i) = ref(i + 1)                                     // case: ref on left
        keys -= 1                                                   // decrement the number of keys
        underflow                                                   // if true, node underflows
    end removeI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Merge this LEAF node with its right sibling node (rt).
     *  @param rt  the right sibling node
     */
    def merge (rt: BpNode): Unit =
        for i <- 0 until rt.keys do                                 // move keys from rt into this node
            key(keys + i)     = rt.key(i)
            ref(keys + i + 1) = rt.ref(i + 1)
        keys   += rt.keys                                           // add the number of keys from rt
        rt.keys = 0                                                 // make rt empty
        val rt_next = rt.ref(0).asInstanceOf [BpNode]               // the node to the right of rt
        ref(0) = rt_next                                            // unlink node rt in the forward (ref(0)) direction
        if DLINK && rt_next != null then rt_next.pre = this         // unlink node rt in the backward (pre) direction
    end merge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Merge this INTERNAL node with its right sibling node (rt).
     *  @param dk  the divider key from parent
     *  @param rt  the right sibling node
     */
    def mergeI (dk: ValueType, rt: BpNode): Unit =
        key(keys)     = dk                                          // move divider key
        ref(keys + 1) = rt.ref(0)                                   // node corresponding to divider key
        for i <- 0 until rt.keys do                                 // move keys from rt into this node
            key(keys + i + 1) = rt.key(i)
            ref(keys + i + 2) = rt.ref(i + 1)
        keys   += rt.keys + 1                                       // add the number of keys from rt
        rt.keys = 0                                                 // make rt empty
    end mergeI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this node to a string.
     */
    override def toString: String =
        val sb = StringBuilder ("[ . " )
        for i <- 0 until keys do sb ++= (s"${key(i)} . ")
        sb ++= ("]" )
        sb.toString
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the node's data structure.
     */
    def show (): Unit =
        println (s"isLeaf = $isLeaf")
        println (s"keys   = $keys")
        println (s"key    = ${stringOf (key)}")
        println (s"ref    = ${stringOf (ref)}")
        if DLINK then println (s"pre    = $pre")
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the node's references.
     */
    def showRef (): Unit =
        println (s"ref = ${stringOf (ref)}")
    end showRef

end BpNode


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bpNodeTest` main function tests the `BpNode` class by inserting random
 *  key values and testing a leaf node split.
 *  > runMain scalation.database.bpNodeTest
 */
@main def bpNodeTest (): Unit =

    import java.util.Random

    banner ("Example of an Leaf Node Split")

    val totKeys = 5
    val mx      = 10 * totKeys
    val seed    = 1
    val rng     = new Random (seed)
    val node    = new BpNode (0, true)                              // empty leaf node
    var right: BpNode = null

    cfor (0, totKeys) { _ =>
        val key = rng.nextInt (mx)
        banner (s"put key = $key")
        node.add (key, 2 * key)
        println (s"node = $node")
        if node.overflow then
            println (s"BEFORE split: node = $node")
            node.showRef ()
            val (dk, rt) = node.split ()                            // split keys between node (3) and right (2)
            right = rt
            println (s"AFTER split:  node = $node, dk = $dk, rt = $rt")
            node.showRef (); rt.showRef ()
    } // cfor

    banner ("Show Arrays")
    node.show ()

    banner ("Example of an Leaf Node Merge")
    node.remove (2)
    node.merge (right)
    println (s"AFTER merge:  node = $node, right = $right")
    node.show ()

end bpNodeTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bpNodeTest2` main function tests the `BpNode` class by inserting random
 *  key values and testing a internal node split.
 *  > runMain scalation.database.bpNodeTest2
 */
@main def bpNodeTest2 (): Unit =

    import java.util.Random

    banner ("Example of an Internal Node Split")

    val totKeys = 5
    val mx      = 10 * totKeys
    val seed    = 1
    val rng     = new Random (seed)
    val node    = new BpNode (0, false)                             // empty internal node: false => not isLeaf

    cfor (0, totKeys) { _ =>
        val key = rng.nextInt (mx)
        banner (s"put key = $key")
        node.add (key, 2 * key)
        println (s"node = $node")
        if node.overflow then
            println (s"BEFORE split: node = $node")
            node.showRef ()
            val (dk, rt) = node.splitI ()                           // splitI keys between node (2) and right (2)
            println (s"AFTER split:  node = $node, dk = $dk, rt = $rt")
            node.showRef (); rt.showRef ()
    } // cfor

    banner ("Show Arrays")
    node.show ()

end bpNodeTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bpNodeTest3` main function tests the `BpNode` class by inserting random
 *  and removing key values.
 *  > runMain scalation.database.bpNodeTest3
 */
@main def bpNodeTest3 (): Unit =

    banner ("Example of key insertion and removal")

    val keyArr = Array (35, 47, 4, 38)
    val node   = new BpNode (0, true)                               // empty internal node: false => not isLeaf

    for k <- keyArr do
        banner (s"put key = $k")
        node.add (k, 2 * k)
        println (s"node = $node")
    end for

    for k <- keyArr do
        banner (s"remove key = $k")
        val dp = node.findEq (k)                                    // find the deletion position (dp)
        node.remove (dp)                                            // remove the key and its reference
        println (s"AFTER remove: node = $node")
        node.showRef ()
    end for

    banner ("Show Arrays")
    node.show ()

end bpNodeTest3
