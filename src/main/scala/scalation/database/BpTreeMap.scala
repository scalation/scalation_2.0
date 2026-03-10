
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Aug 11 00:26:03 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Sorted Map Implemented Using B+Trees (Indexed and Sequential Access)
 *
 *  Split Nodes on Overflow
 *  Structure for order = 5 (max of 4 keys), upon first split
 *  [ . k4 . -- . -- . -- . ]
 *      [ . k1 . k2 . k3 . -- . ]
 *      [ . k4 . k5 . -- . -- . ]
 *  Rules: divider key (k4 added to parent in this case) is the smallest key
             in the right subtree (SMALLEST RIGHT)
 *         split node n into (n, right_sibling_node) with larger half staying in n
 *         internal node split promotes middle key to parent as the divider key
 *
 *  Borrow/Merge Nodes on Underflow
 *  Rules: try to borrow one key from an adjacent (left or right) rich sibling node
 *         otherwise merge with sibling node
 *
 *  Optionally supports bidirectional linkage of leaf nodes for Sequential Access
 *         forward via ref(0)  [ n1 ] -> [ n2 ] -> [ n3 ]
 *         backward via pre    [ n1 ] <- [ n2 ] <- [ n3 ]   optional DLINK = true
 */

package scalation
package database

import scala.collection.mutable.{AbstractMap, SortedMap}
import scala.reflect.ClassTag

import BpNode._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BpTreeMap` class provides sorted maps that use the B+Tree Data Structure.
 *  Inserts may cause the splitting of nodes, while deletes may cause borrowing
 *  if keys or merging of nodes.
 *  @tparam V     the type of the values assigned to keys in this sorted map
 *  @param  name  the name of the B+Tree (used for indentification)
 */
class BpTreeMap [V: ClassTag] (name: String = "BpTreeMap")
      extends AbstractMap [ValueType, V]
         with SortedMap [ValueType, V]
         with Serializable:

    private val debug  = debugf ("BpTreeMap", false)                     // debug function
    private val flaw   = flawf ("BpTreeMap")                             // flaw function

    private var kCount = 0                                               // counter for total number of keys
    private [database] var count  = 0                                    // count # nodes accessed (performance)
    private var root   = new BpNode (0, true)                            // root node of this B+Tree (initially empty)
    private val first  = root                                            // first leaf node of this B+Tree
    private var last_  = root                                            // lastleaf node of this B+Tree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the first value in the B+Tree (note ref(0) points at next leaf node).
     */
    def getFirst: V = first.ref(1).asInstanceOf [V]
    def getLast: V = last_.ref(last_.keys).asInstanceOf [V]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `SortedMap` trait requires `Ordering` with a compare method to be defined.
     *  @see https://scala-lang.org/api/3.3.0/scala/math/Ordering.html
     *  @see ValueType.scala in `scalation.package`
     */
    def ordering: Ordering [ValueType] = ValueTypeOrd

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size (number of keys) of this B+Tree.
     */
    inline override def size: Int = kCount

    //--------------------------------------------------------------------------
    // Retrieve values or ranges (subtrees)
    //--------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `TreeIterator` inner class supports iterating over all the elements
     *  in a B+Tree by traversing through the LEAF nodes of the tree.
     *  @param ns  the starting leaf node (defaults to first)
     *  @param js  the starting within node index (defaults to -1)
     */
    class TreeIterator (ns: BpNode = first, js: Int = -1) extends Iterator [(ValueType, V)]:
        private var (n, j) = (ns, js)

        def hasNext: Boolean = j < n.keys-1 || n.ref(0) != null

        def next (): (ValueType, V) =
//          debug ("next", s"node n = $n, j = $j, n.keys = ${n.keys}")
            if j < n.keys-1 then j += 1
            else { n = n.ref(0).asInstanceOf [BpNode]; j = 0 }
            (n(j), n.ref(j+1).asInstanceOf [V])
        end next
    end TreeIterator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an iterator for retrieving all the elements in this B+Tree.
     *  @see scala.collection.IterableOnce
     */
    def iterator: Iterator [(ValueType, V)] = new TreeIterator ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an iterator for retrieving all the elements in this B+Tree starting
     *  from key start.  Returns null if all keys in tree are smaller than start.
     *  @see scala.collection.SortedMapOps
     *  @param start  the key to start with (inclusive)
     */
    def iteratorFrom (start: ValueType): Iterator [(ValueType, V)] =
        val (ns, js) = findp (start, root)                               // find position: node ns and key index js
        debug ("iteratorFrom", s"(ns, js) = ($ns, $js)")
        val jjs = math.max (js-1, -1)
        if ns != null then new TreeIterator (ns, jjs)
        else null
    end iteratorFrom

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an iterator for retrieving all the keys in this B+Tree starting
     *  from key start.
     *  @see scala.collection.SortedMapOps
     *  @param start  the key to start with
     */
    def keysIteratorFrom (start: ValueType): Iterator [ValueType] =
        throw new UnsupportedOperationException ("keysIteratorFrom not available, use iteratorFrom instead")
    end keysIteratorFrom

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the submap starting at from and ending before until.
     *  @see scala.collection.SortedOps
     *  @param from   the starting key (inclusive)
     *  @param until  the ending key (exclusive)
     */
    def rangeImpl (from: Option [ValueType], until: Option [ValueType]): BpTreeMap [V] =
        val subtree = new BpTreeMap [V] (name + from)
        val it = if from.isDefined then iteratorFrom (from.get)
        else iterator
        var cont = true
        while cont && it.hasNext do
            val (k, v) = it.next ()
            if ! until.isDefined || k < until.get then
                if ! from.isDefined || k >= from.get then subtree.addOne ((k, v))
            else
                cont = false
        subtree
    end rangeImpl

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the value associated with the key by looking it up in this B+Tree.
     *  @param key  the key used for look up
     */
    def get (key: ValueType): Option [V] = Option (find (key))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the given key in this B+tree and return its corresponding value.
     *  Calls the recursive findp method.
     *  @param key  the key to find
     */
    inline def find (key: ValueType): V =
        val (ln, ip) = findp (key, root)                                 // leaf node, index position
        if ip >= 0 then ln.ref(ip+1).asInstanceOf [V]
        else null.asInstanceOf [V]
    end find

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursive helper method for finding the position of the given key in this B+tree.
     *  @param key  the key to find
     *  @param n    the current node
     */
    private def findp (key: ValueType, n: BpNode): (BpNode, Int) =
        count += 1
        if n.isLeaf then (n, n.findEq (key))
        else findp (key, n.ref(n.find (key)).asInstanceOf [BpNode])
    end findp

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the first (key, value) pair whose key is larger than (beyond) the search key (skey).
     *  @author Amily Chowdhury
     *  @param skey  the search key
     */
    def findFirstBeyond (skey: ValueType): Option [(ValueType, V)] =
        val (ln, ip) = findp (skey, root)                                // locate leaf node and index position for skey
        if ln != null then
            if ip >= 0 && ip < ln.keys - 1 then
                // Case 1: If the key is found and not the last one, return the next key in the leaf node
                Some ((ln(ip + 1), ln.ref(ip + 2).asInstanceOf[V]))      // use ip + 1 to get the next key
            else
                // Case 2: Move to the next leaf node and return the first key there
                val nxLeaf = ln.ref(0).asInstanceOf [BpNode]             // move to the next leaf node
                if nxLeaf != null && nxLeaf.keys > 0 then
                    Some ((nxLeaf(0), nxLeaf.ref(1).asInstanceOf [V]))   // get first key and reference
                else
                    None
        else
            None
    end findFirstBeyond

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the last (key, value) pair whose key is smaller than (below) the search key (skey).
     *  @author Amily Chowdhury
     *  @param skey  the search key
     */
    def findLastBelow (skey: ValueType): Option [(ValueType, V)] =
        val (ln, ip) = findp (skey, root)                                // locate leaf node and index position for skey
        if ln != null then
            if ip > 0 then
                // Case 1: If the key is found and not the first one, return the previous key in the leaf node
                Some ((ln.key(ip - 1), ln.ref(ip).asInstanceOf [V]))     // use ip - 1 to get the previous key
            else
                // Case 2: Move to the previous leaf node and return the last key there
                val pvLeaf = ln.pre                                      // move to the previous leaf node
                if pvLeaf != null && pvLeaf.keys > 0 then
                    val iLast = pvLeaf.keys - 1
                    Some ((pvLeaf(iLast), pvLeaf.ref(pvLeaf.keys).asInstanceOf [V]))   // get last key and reference
                else
                    None
        else
            None
    end findLastBelow

    //--------------------------------------------------------------------------
    // Add key-value pairs into the B+Tree
    //--------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one key-value pair into this B+Tree and return this.
     *  Called by the put method in `AbstractMap`.
     *  Note:  it splits the node that overflows
     *  @param elem   the key-value pair to add/insert
     */
    def addOne (elem: (ValueType, V)): this.type =
        val (key, value) = elem
        insert (key, value, root)                                        // call the recursive insert
        this
    end addOne

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursive helper method for inserting a key and ref into this B+tree.
     *  Returns the divider key and right sibling upon split, else null.
     *  @param key  the key to insert
     *  @param ref  the value/node to insert
     *  @param n    the current node
     */
    private def insert (key: ValueType, ref: V, n: BpNode): (ValueType, BpNode) =
        var k_r: (ValueType, BpNode) = null

        if n.isLeaf then                                                 // handle LEAF node
            k_r = add (n, key, ref)
            if k_r != null then
                if n != root then return k_r
                root = new BpNode (root, k_r._1, k_r._2)                 // make a new root

        else                                                             // handle INTERNAL node
            k_r = insert (key, ref, n.ref(n.find (key)).asInstanceOf [BpNode])
            if k_r != null then
                k_r = addI (n, k_r._1, k_r._2)
                if k_r != null && n == root then root = new BpNode (root, k_r._1, k_r._2)
        end if
        k_r
    end insert

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add new key k and value v into LEAF node n.  Upon overflow, split node n,
     *  in which case the divider key and new right sibling node are returned.
     *  If you split in leaf node and leaf node is last node, then newly created leaf node is last node
     *  @param n  the current node
     *  @param k  the new key
     *  @param v  the new value
     */
    private def add (n: BpNode, k: ValueType, v: V): (ValueType, BpNode) =
        var k_r: (ValueType, BpNode) = null                              // divider key, right sibling
        val duplicate = n.add (k, v)                                     // add into node n
        if duplicate == None then kCount += 1                            // increment the key count unless its a duplicate
        else flaw ("add", s"key $k is a duplicate, old value = $duplicate")
        if n.overflow then
            k_r = n.split ()                              // its full, must split
            if n == last_ then last_ = k_r._2
        k_r
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add new key k and value v into INTERNAL node n.  Upon overflow, split node n,
     *  in which case the divider key and new right sibling node are returned.
     *  @param n  the current node
     *  @param k  the new key
     *  @param v  the new left value (ref a node)
     */
    private def addI (n: BpNode, k: ValueType, v: BpNode): (ValueType, BpNode) =
        var k_r: (ValueType, BpNode) = null                              // divider key, right sibling
        n.add (k, v)                                                     // add into node n
        //      n.showRef ()
        if n.overflow then k_r = n.splitI ()                             // its full, must split
        k_r
    end addI

    //--------------------------------------------------------------------------
    // Remove key-value pair from the B+Tree
    //--------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the one element (key-value pair) with the given key and return
     *  whether it matches the value expected.
     *  @param key    the key whose element is to be removed
     *  @param value  the value expected of the removed element
     */
    def checkedRemove (key: ValueType, value: V): Boolean = remove (key) == value

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract/remove the one element (key-value pair) with the given key.
     *  Called by the remove method in `AbstractMap`.
     *  @param key  the key whose element is to be removed
     */
    def subtractOne (key: ValueType): this.type =
        kCount -= 1                                                      // decrement the key count
        delete (key, root)                                               // call the recursive delete
        if ! root.isLeaf && root.keys == 0 then
            root = root.ref(0).asInstanceOf [BpNode]                     // remove empty root by resetting the root reference
        this
    end subtractOne

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursive helper method for deleting a key with its ref from this B+tree.
     *  @param key  the key to delete
     *  @param n    the current node
     *  @param par  the parent node (null for root)
     */
    private def delete (key: ValueType, n: BpNode, par: BpNode = null): Unit =

        if n.isLeaf then                                                 // handle LEAF node
            val dp = n.findEq (key)                                      // deletion position in LEAF node n
            n.remove (dp)                                                // remove key at index position dp

            // upon underflow do a leaf node borrow or merge

            if n != root && n.underflow then                             // unless root, check for underflow
                val j = par.find (key)                                   // j-th index position in parent
                val (sib, left) = richestSib (par, j)                    // richest sib and whether it's left
                debug ("delete", s"needs to handle underflow of LEAF node n = $n, sib = $sib, left = $left")

                if sib.rich then borrow (n, sib, left, par, j)           // leaf borrow a key from rich sib
                else merge (n, sib, left, par, j)                        // leaf merge nodes n and sib, may cause parent to underflow
            end if

        else                                                             // handle INTERNAL node
            val dp = n.find (key)                                        // deletion position in INTERNAL node n
            delete (key, n.ref(dp).asInstanceOf [BpNode], n)             // recursive call to delete

            // upon underflow do an internal node borrow (borrowI) or merge (mergeI)

            if n != root && n.underflow then                             // unless root, check for underflow
                val j = par.find (key)                                   // j-th index position in parent
                val (sib, left) = richestSib (par, j)                    // richest sib and whether it's left
                debug ("delete", s"needs to handle underflow of INTERNAL node n = $n, sib = $sib, left = $left")
                if sib.rich then borrowI (n, sib, left, par, j)          // internal borrow a key from rich sib
                else mergeI (n, sib, left, par, j)                       // internal merge nodes n and sib, may cause parent to underflow
            end if

        end if
    end delete

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return node n's richest sibling (and whether it is left/true or right/false)
     *  found from parent node.
     *  @param par  the parent node of node n that has underflowed
     *  @param i    the position of node n as a child of node par
     */
    private def richestSib (par: BpNode, i: Int): (BpNode, Boolean) =
        debug ("richSib", s"return node n's (@ $i) richest sibling (left or right)")
        val leftn  = if i-1 >= 0 then par.ref(i-1).asInstanceOf [BpNode] else null
        val rightn = if i+1 <= par.keys then par.ref(i+1).asInstanceOf [BpNode] else null

        if leftn == null then (rightn, false)
        else if rightn == null then (leftn, true)
        else if leftn.keys >= rightn.keys then (leftn, true)
        else (rightn, false)
    end richestSib

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Borrow a key-value pair from a rich sibling, so LEAF node n won't underflow.
     *  For borrow LEFT, last key in left sib k2 moves to n, then k2 replaces k3 in par.
     *         [ ... k3 ... ]              [ ... k2 ... ]
     *  [ ... k1 k2 ]  [ k3 ... ]  TO  [ ... k1 ]  [ k2 k3 ... ]
     *  @param n     the current node that has underflowed
     *  @param sib   the rich sibling node
     *  @param left  the whether the sib is left or right
     *  @param par   the parent node
     *  @param j     the index position in the parent node
     */
    private def borrow (n: BpNode, sib: BpNode, left: Boolean, par: BpNode, j: Int): Unit =
        val i = if left then sib.keys-1 else 0
        val bkey = sib(i)
        debug ("borrow", s"key $bkey from rich sib = $sib node for node n = $n having par = $par at j = $j, left=$left")
        val bref = sib.ref(i+1).asInstanceOf [V]
        sib.remove (i)
        add (n, bkey, bref); kCount -= 1                                 // key is moved, not really added (=> -= 1)
        if left then par(j-1) = bkey else par(j) = sib(0)                // the divider key for parent node
    end borrow

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Borrow a key-value pair from a rich sibling, so INTERNAL node n won't underflow.
     *  For borrow LEFT, last key in left sib k2 rotates into par, whose key k3 rotates to n.
     *         [ ... k3 ... ]              [ ... k2 ... ]
     *  [ ... k1 k2 ]  [ k4 ... ]  TO  [ ... k1 ]  [ k3 k4 ... ]
     *  @param n     the current node that has underflowed
     *  @param sib   the rich sibling node
     *  @param left  the whether the sib is left or right
     *  @param par   the parent node
     *  @param j     the index position in the parent node
     */
    private def borrowI (n: BpNode, sib: BpNode, left: Boolean, par: BpNode, j: Int): Unit =

        if left then                                                     // borrow from LEFt sib
            // j is incorrect when you are doing borrow on the left -- should be zero
            val ip = 0                                                   // node n's insertion position is 0
            debug ("borrowI", s"LEFT sib = $sib, n = $n [ip = $ip], par = $par [j = $j]")
            n.shiftIR (ip)                                               // shift right to make room in node n
            n.key(ip) = par.key(j-1)                                     // move par key @ j to node n
            n.keys += 1
            n.ref(ip) = sib.ref(sib.keys)                                // move sib's last ref to node n

            par.key(j-1) = sib.key(sib.keys-1)                             // promote sib's last key to par @ j
            sib.keys -= 1                                                // effectively removes last key in sib

        else                                                             // borrow from RIGHT sib
            val ip = n.keys                                              // node n's insertion position is n.keys
            debug ("borrowI", s"n = $n [ip = $ip], RIGHT sib = $sib, par = $par [j = $j]")
            n.key(ip)   = par.key(j)                                     // move par key @ j to node n
            n.ref(ip+1) = sib.ref(0)                                     // move sib's first ref to node n
            n.keys += 1

            par.key(j) = sib.key(0)                                      // promote sib's first key to par @ j
            sib.removeI (0)                                              // remove sib's first key
    end borrowI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Merge LEAF node n with its sibling returning whether the parent node
     *  has underflowed.
     *  If you delete, if it in last node, the merged node will be the new last node.
     *  @param n     the current node that has underflowed
     *  @param sib   the sibling node
     *  @param left  the whether the sib is left or right
     *  @param par   the parent node
     *  @param j     the index position in the parent node
     */
    private def merge (n: BpNode, sib: BpNode, left: Boolean, par: BpNode, j: Int): Boolean =
        debug ("merge", s"LEAF node n = $n that underflows with sib = $sib having par = $par at j = $j ,left=$left")
        if left then
            sib.merge (n)
            if n == last_ then last_ = sib
            par.remove(j-1)
        else
            n.merge (sib)
            if sib == last_ then last_ = n
            par.remove (j)                           // true means parent underflowed
    end merge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Merge INTERNAL node n with its sibling returning whether the parent node
     *  has underflowed.
     *  @param n     the current node that has underflowed
     *  @param sib   the sibling node
     *  @param left  the whether the sib is left or right
     *  @param par   the parent node
     *  @param j     the index position in the parent node
     */
    private def mergeI (n: BpNode, sib: BpNode, left: Boolean, par: BpNode, j: Int): Boolean =
        if left then
            debug ("mergeI", s"LEFT sib = $sib, n = $n, par = $par [j = $j]")
            sib.mergeI (par.key(j-1), n)
            par.remove(j-1)                                              // true means parent underflowed
        else
            debug ("mergeI", s"n = $n, RIGHT sib = $sib, par = $par [j = $j]")
            n.mergeI (par.key(j), sib)
            par.remove (j)                                               // true means parent underflowed
    end mergeI

    //--------------------------------------------------------------------------
    // Print/show the B+Tree
    //--------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this B+Tree has the same entries, (key, value) pairs,
     *  (in the same order) as another sorted map.
     *  @param that  the other sorted map
     */
    infix def equals (that: SortedMap [ValueType, V]): Boolean =
        println (s"this.size = ${this.size} ==? that.size = ${that.size}")
        if this.size != that.size then return false
        var same = true
        val it1  = this.iterator
        val it2  = that.iterator
        while same && it1.hasNext do
            val (k1, v1) = it1.next ()
            val (k2, v2) = it2.next ()
            debug ("equals", s"($k1, $v1) ==? ($k2, $v2)")
            same = k1 == k2 && v1 == v2
        same
    end equals

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print this B+Tree.
     */
    def show (): Unit =
        println (s"BpTreeMap_$name")
        printT (root, 0)
        println ("-" * 60)
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print this B+Tree's leaf node links.
     */
    def showLink (): Unit =
        println (s"BpTreeMap_$name Leaf Node Linkage")
        println("=" * 60)
        var n = first
        while n != null do { n.show (); n = n.ref(0).asInstanceOf [BpNode]}
        println("=" * 60)
    end showLink

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print this B+Tree map using a pre-order traversal and indenting each level.
     *  @param n      the current node to print
     *  @param level  the current level in the B+Tree
     */
    private def printT (n: BpNode, level: Int): Unit =
        if n != null then
            println ("\t" * level + n)
            if ! n.isLeaf then
                for j <- 0 to n.keys do printT (n.ref(j).asInstanceOf [BpNode], level + 1)
    end printT

end BpTreeMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bpTreeMapTest` main function used for testing the `BpTreeMap` class by
 *  inserting increasing key values.
 *  > runMain scalation.database.bpTreeMapTest
 */
@main def bpTreeMapTest (): Unit =

    banner ("Insert Increasing Integer Keys")
    val totKeys = 60
    val tree    = new BpTreeMap [Int] ("Test")

    for i <- 1 until totKeys by 2 do
        banner (s"put ($i, ${i * i})")
        tree.put (i, i * i)
        tree.show ()
        tree.showLink()
    end for

    banner ("Find Keys")
    for i <- 0 until totKeys do println (s"key = $i, value = ${tree.get(i)}")
    println ("-" * 60)

    banner ("Iterate Through the B+Tree")
    for it <- tree.iterator do println (it)
    println ("-" * 60)
    tree.foreach (println (_))

    banner ("Iterate Through the B+Tree from Key = 11")
    for it <- tree.iteratorFrom (11) do println (it)
    println ("-" * 60)
    tree.foreach (println (_))
    println ("-" * 60)

    banner ("Find Keys in Range")
    println (s"Range Query 11 until 20: key-value pairs = ${tree.range (11, 20)}")
    println ("-" * 60)

    banner ("Print Statistics")
    println (s"size = ${tree.size}")
    println (s"Average number of nodes accessed = ${tree.count / totKeys.toDouble}")

    banner ("Delete Keys")
    tree.show ()
    val toRemove = Array (29, 31,27, 33, 35, 25, 23, 13, 7, 1, 59, 55, 47, 53, 3, 5, 9, 21, 17, 19, 21, 11, 15, 37, 39, 41, 43, 49, 47, 51, 45)
//  val toRemove = Array (7, 1, 3, 5, 9, 21, 17, 19, 21, 11, 15, 13, 29, 31, 27, 33, 35, 23, 39, 37, 41, 25)

    for key <- toRemove do
        banner (s"remove ($key)")
        tree.remove (key)
        tree.show ()
        tree.showLink()
    end for

end bpTreeMapTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bpTreeMapTest2` main function used for testing the `BpTreeMap` class by
 *  inserting random key values.
 *  > runMain scalation.database.bpTreeMapTest2
 */
@main def bpTreeMapTest2 (): Unit =

    import java.util.Random

    banner ("Insert Random Integer Keys")
    val totKeys = 60
    val mx      = 10 * totKeys
    val seed    = 1
    val rng     = new Random (seed)
    val tree    = new BpTreeMap [Int] ("Test2")

    cfor (0, totKeys) { _ =>
        val key = rng.nextInt (mx)
        banner (s"put ($key, ${2 * key})")
        tree.put (key, 2 * key)
        tree.show ()
        tree.showLink()
    } // cfor

    banner ("Print Statistics")
    println (s"size = ${tree.size}")
    println (s"Average number of nodes accessed = ${tree.count / totKeys.toDouble}")

end bpTreeMapTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bpTreeMapTest3` main function used for testing the `BpTreeMap` class by
 *  inserting and removing keys and values into/from a B+Tree and a TreeMap.
 *  Performs AUTOMATED TESTING.
 *  > runMain scalation.database.bpTreeMapTest3
 */
@main def bpTreeMapTest3 (): Unit =

    import java.util.Random
    import scala.collection.mutable.TreeMap

    banner ("AutoTest: Insert Random Integer Keys into BpTreeMap and TreeMap")
    val totKeys = 60
    val mx      = 10 * totKeys
    val seed    = 1
    val rng     = new Random (seed)
    val tree    = new BpTreeMap [Int] ("Test3")
    val tree2   = new TreeMap [ValueType, Int] ()(using ValueTypeOrd)

    cfor (0, totKeys) { _ =>
        val key = rng.nextInt (mx)
        tree.put (key, 2 * key)
        tree2.put (key, 2 * key)
    } // cfor

    var same = tree equals tree2
    println (s"tree equals tree2 = $same")
    assert (same)

    tree.show ()

    banner ("AutoTest: Remove All Keys from BpTreeMap and TreeMap")
    val toRemove = tree2.keys
    for key <- toRemove do
        banner (s"remove ($key)")
        tree.remove (key)
        tree2.remove (key)
        tree.show ()
        tree.showLink()
        same = tree equals tree2
        println (s"tree equals tree2 = $same")
        assert (same)
    end for

end bpTreeMapTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bpTreeMapTest4` main function used for testing the `BpTreeMap` class by
 *  inserting keys and values into B+Trees, and performing range queries.
 *  @author Amily Chowdhury
 *  > runMain scalation.database.bpTreeMapTest4
 */
@main def bpTreeMapTest4 (): Unit =

    banner("Insert Increasing Integer Keys")
    val tree = new BpTreeMap [String] ("Range_test")

    tree.put (147.4, "C1")
    tree.put (153.4, "C2")
    tree.put (163.4, "C3")
    tree.put (173.4, "C4")
    tree.put (180.4, "C5")

    tree.show ()
    tree.showLink ()

    banner ("Find Keys in Range")
    //  val lb = 137.4
    //  val ub = 180.4
    var lb = 147.4
    var ub = 163.4
    println (s"Range Query lb = $lb: until up = $ub: key-value pairs")
    println (s"Range Query lb = $lb: until up = $ub: key-value pairs = ${tree.range (lb, ub)}")
    println ("-" * 60)

    banner("Find First Key Beyond Upper Bound of Range")
    println(s"Find First Key Beyond $ub")
    var result = tree.findFirstBeyond (ub)
    println(s"The first key-value pair beyond $ub is: $result")

    ub = 180.4
    println (s"Find First Key Beyond $ub")
    result = tree.findFirstBeyond (ub)
    println (s"The first key-value pair beyond $ub is: $result")

    banner("Find Last Key Below Lower Bound")
    lb = 163.4
    println (s"Find Last Key Below $lb")
    result = tree.findLastBelow (lb)
    println(s"The last key-value pair below $lb is: $result")

    lb = 147.4
    println (s"Find Last Key Below $lb")
    result = tree.findLastBelow (lb)
    println(s"The last key-value pair below $lb is: $result")

end bpTreeMapTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bpTreeMapTest5` main function used for testing the `BpTreeMap` class by
 *  inserting keys and values into B+Trees, one representing each of two lanes.
 *  Can be used for finding cars in a traffic simulation.
 *  > runMain scalation.database.bpTreeMapTest5
 */
@main def bpTreeMapTest5 (): Unit =

    import java.util.Random

    case class Car (vin: Int, dist: Double)

    banner ("Insert Random Integer Keys")
    val totKeys = 60
    val seed    = 1
    val rng     = new Random (seed)
    val lane1   = new BpTreeMap [Car] ("lane1")        // index for lane1
    val lane2   = new BpTreeMap [Car] ("lane2")        // index for lane2

    var dist = 0.0                                     // distance from end of lane
    var ord  = 0                                       // rank order from end of lane
    for i <- 1 to totKeys do
        dist += rng.nextInt (5)
        val c_i = Car (i, dist)                        // the car being put into lane1's B+Tree
        ord += 10                                      // rank order of car toward end of lane
        banner (s"put ($ord, $c_i)")
        lane1.put (ord, c_i)
        lane1.show ()
        lane1.showLink()
    end for

    dist = 0.0                                         // distance from end of lane
    ord  = 0
    for i <- 1 to totKeys do
        dist += rng.nextInt (5)
        val c_i = Car (totKeys + i, dist)              // the car being put into lane2's B+Tree
        ord += 10                                      // rank order of car toward end of lane
        banner (s"put ($ord, $c_i)")
        lane2.put (ord, c_i)
        lane2.show ()
        lane2.showLink()
    end for

// find the j-th car in lane1 call it car1
// find the corresponding j-th car in lane2 call it car2
// check whether car2 is behind car1 in the other lane
// may need a doubly linked list of nodes at the leaf-level to search forward and backward
// find the closest car in the other lane that is behind you
// if its distance is large enough, make the lane change
// may need gaps in ord so lane changing car can get an ord without making all care reassign theirs

end bpTreeMapTest5

