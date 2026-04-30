
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Jul  4 12:55:00 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Map Implemented Using Linear Hashing
 *
 *  Split Home Buckets when load factor is too high
 *  Merge Home Buckets when load factor is too low (not yet implemented)
 */

package scalation
package database

import scala.collection.mutable.{AbstractMap, ArrayBuffer}
import scala.reflect.ClassTag

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LinHashMap` class provides hash maps that use the Linear Hashing algorithm.
 *  A hash table is created that is an expandable array-list (`ArrayBuffer`) of buckets.
 *  Inserts may cause the allocation of overflow buckets as well as the splitting of
 *  home buckets.  Uses the SRI rule: Split, Rehash, and then Insert.
 *  @tparam K        the type of the keys contained in this hash map
 *  @tparam V        the type of the values assigned to keys in this hash map
 *  @param  name     the name of Linear Hash Map
 *  @param  order    the number of slots per bucket
 *  @param  loadFac  the (lower, upper) bound on the load factor (# keys over # home slots)
 */
class LinHashMap [K: ClassTag, V: ClassTag] (name: String, order: Int = 4,
                  loadFac: (Double, Double) = (0.3, 1.2))
      extends AbstractMap [K, V]
         with Serializable:

    var count = 0                                             // count # buckets accessed (performance)

    private val debug = debugf ("LinHashMap", true)           // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Bucket` inner class defines buckets that are stored in this hash table.
     */
    class Bucket ()
          extends Serializable:

        val key   = Array.ofDim [K] (order)                   // array to hold keys
        val value = Array.ofDim [V] (order)                   // array to hold values
        var nKeys = 0                                         // number of active keys 
        var next: Bucket = null                               // reference to next bucket in chain

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Find key k in this bucket, returning the corresponding value or
         *  null if not found.
         *  @param k  the key to find in this bucket
         */
        def find (k: K): V = 
            val j = findPos (k)
            if j == -1 then null.asInstanceOf [V] else value(j)
        end find

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Find the position of key k in this bucket, returning its position in
         *  the bucket, or -1 if not found.
         *  @param k  the key to find in this bucket
         */
        inline def findPos (k: K): Int = key.indexOf (k)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Add the given key-value pair to this bucket, returning true if successful
         *  and false otherwise (when this bucket is full).
         *  @param k  the key
         *  @param v  the value
         */
        def add (k: K, v: V): Boolean =
            if nKeys < order then
                key(nKeys)   = k
                value(nKeys) = v
                nKeys += 1
                true
            else false
        end add
 
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Move the j-th element (key-value pair) to bucket b2, returning true
         *  if successful and false otherwise (when bucket b2 is full).
         *  Simply remove the j-th element when b2 is null.
         *  @param j   the j-th key-value pair
         *  @param b2  the other bucket to move the pair to
         */
        def move (j : Int, b2: Bucket): Boolean =
            val (ky, vl) = (key(j), value(j))
            if b2 == null || b2.add (ky, vl) then
                nKeys -= 1                                    // remove element j
                if j != nKeys then
                    key(j)   = key(nKeys)                     // overwrite with last pair
                    value(j) = value(nKeys)
                true
            else false
        end move

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Show/print the contents of this bucket.
         */
        def show (): Unit =
            print ("[ " )
            for j <- 0 until nKeys do print (s"${key(j)} . ")
            println ("]" )
        end show

    end Bucket


    private var mod1     = 4                                  // modulus for low resolution hashing
    private var mod2     = 2 * mod1                           // modulus for high resolution hashing
    private var isplit   = 0                                  // index of the next bucket to split
    private var keyCount = 0                                  // counter for total number of keys

    /** The list of buckets making up this hash table.
     */
    private val hTable = new ArrayBuffer [Bucket] ()
    cfor (0, mod1) { _ => hTable += new Bucket () }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `HTIterator` inner class supports iterating over all the elements
     *  in a Hash-Table by traversing through the buckets in this hash table.
     *  @param is  the starting bucket chain (defaults to 0)
     *  @param bs  the starting home bucket (defaults to hTable(0))
     *  @param js  the starting within node index (defaults to 0)
     */
    class HTIterator (is: Int = 0, bs: Bucket = hTable(0), js: Int = 0) extends Iterator [(K, V)]:
        private var (i, b, j) = (is, bs, js)
        def hasNext: Boolean = j < b.nKeys - 1 || b.next != null || is < size0 - 1
        def next (): (K, V) =
            if j < b.nKeys - 1 then j += 1                     // next position in bucket
            else if b.next != null then { b = b.next; j = 0 }  // next overflow bucket
            else { i += 1; b = hTable(i); j = 0 }              // next bucket chain
            (b.key(j), b.value(j))
        end next
    end HTIterator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an iterator for retrieving all the elements in this hash table.
     *  @see scala.collection.IterableOnce
     */
    def iterator: Iterator [(K, V)] = new HTIterator ()
        
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the value associated with the key by looking it up in this hash table.
     *  @param key  the key used for look up
     */
    def get (key: K): Option [V] = find (key, hTable(h (key)), true)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one key-value pair into this hash table and return this (called by put).
     *  Note:  it splits the 'isplit' bucket chain when the load factor is exceeded.
     *  @param elem   the key-value pair to add/insert
     */
    def addOne (elem: (K, V)): this.type =
        val (key, value) = elem
        keyCount += 1                                         // increment the key count
        val lf    = loadFactor                                // compute the load factor
        debug ("addOne", s"$key -> $value, load factor = $lf")
        if lf > loadFac._2 then split ()                      // split when load factor is too high

        var b = hTable(h (key))                               // find space in bucket chain
        while { if b.nKeys < order then { b.add (key, value); return this }
                b.next != null } do
            b = b.next
        end while

        val bnew = new Bucket ()                              // needs a new overflow bucket
        bnew.add (key, value)                                 // put pair in new bucket
        b.next = bnew                                         // add new bucket at end of chain
        this
    end addOne

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract/remove the one element (key-value pair) with the given key.
     *  Note:  it merges home buckets when load factor is too low.
     *  @param key  the key whose element is to be removed
     */
    def subtractOne (key: K): this.type =
        keyCount -= 1                                         // decrement the key count
        val lf    = loadFactor                                // compute the load factor
        debug ("subtractOne", s"key = $key, load factor = $lf")
        if lf < loadFac._1 then merge ()                      // merge when load factor is too low

        val bh = hTable(h (key))                              // home bucket
        var b  = bh                                           // current bucket
        var bp = bh                                           // previous bucket
        var removed = false
        cfor (! removed && b != null, {bp = b; b = b.next}) {   // iterate through bucket chain
            val j = b.findPos (key)
            if j >= 0 then                                    // position of key is found
                b.move (j, null)                              // move element j to null bucket
                if b != bh && b.nKeys == 0 then
                    bp.next = b.next                          // remove empty overflow bucket
                removed = true                                // element successfully removed, end loop
            end if
        } // cfor
        debug ("subtractOne", s"could not find key $key to remove it")
        this
    end subtractOne

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print this hash table.
     */
    def show (): Unit =
        println (s"LinHashMap_$name")
        println ("-" * 60)

        for i <- hTable.indices do
            print (s"Bucket [ $i ] = ")
            var j = 0
            var b = hTable(i)
            cfor (b != null, {b = b.next}) {
                if j > 0 then print (" \t\t --> ")
                b.show ()
                j += 1
            } // cfor
        end for
        println ("-" * 60)
    end show
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size (order * number of home buckets) of this hash table. 
     */
    override def size: Int = order * (mod1 + isplit)
    inline def size0: Int = order * (mod1 + isplit)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split bucket chain 'isplit' by creating a new bucket chain at the end of the
     *  hash table and redistributing the keys according to the high resolution hash
     *  function 'h2'.  Increment 'isplit'.  If current split phase is complete,
     *  reset 'isplit' to zero, and update the hash functions.
     */
    private def split (): Unit =
        debug ("split", s"bucket chain $isplit")

        val bnew = new Bucket ()                              // create a new home bucket          
        hTable  += bnew                                       // add to end of hash table
        val b    = hTable(isplit)                             // b is the bucket chain to be split
        val isp  = isplit                                     // original value of isplit
        isplit  += 1                                          // increment to split pointer
        if isplit == mod1 then                                // has it reached end in current phase?
            isplit = 0                                        // if so, reset to zero
            mod1   = mod2                                     // double first hash function's modulus
            mod2  *= 2                                        // double second hash function's modulus
        rehash (isp, b, bnew)                                 // rehash some keys in chain b to bnew
        show ()
    end split

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Merge bucket chain 'isplit-1' by merging this bucket chain with the last
     *  bucket chain.  Deccrement 'isplit'.  If 'isplit' becomes negative,
     *  set back to previous phase.
     */
    private def merge (): Unit =
        debug ("merge", s"bucket chain $isplit")

        val bend = hTable.last                                // reference the last bucket chain
        hTable  -= bend                                       // remove the last chain from the table
        isplit  -= 1                                          // decrement to split pointer
        if isplit < 0 then                                    // reset back to previous phase
           mod2   = mod1
           mod1  /= 2
           isplit = mod1 - 1

        val bh = hTable(isplit)                               // home bucket for merge chain
        var b  = bh
        cfor (b.next != null, {b = b.next}) {}                // move to the end of this chain
        b.next = bend                                         // link last chain to merge chain 
        show ()
    end merge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rehash/redistribute the key-value pairs in the bucket chain for b1.
     *  Based on h2 some pairs may need to move to bucket chain for b2.
     *  @param isp  the original value of isplit (before the split)
     *  @param b1   the first bucket chain's home bucket (split bucket)
     *  @param b2   the second bucket chain's home bucket (new bucket)
     */
    private def rehash (isp: Int, b1: Bucket, b2: Bucket): Unit =
        var b  = b1                                           // bucket chain for isp
        var c  = b2                                           // bucket chain at end of hash table
        var bp = b1                                           // previous bucket
        cfor (b != null, {bp = b; b = b.next}) {              // iterate through isp bucket chain
            var j = 0
            while b.nKeys > 0 && j < b.nKeys do
                val keyj = b.key(j)                           // j-th key in bucket b
                val i2 = h2(keyj)                             // recompute its hash function
                debug ("rehash", s"keyj = $keyj, is = $isp, i2 = $i2") 
                if i2 != isp then                             // see if the hash has changed?
                    if ! b.move (j, c) then                   // move j-th pair to bucket chain c
                        val b3 = new Bucket ()                // requires a new overflow bucket
                        c.next = b3                           // link it in th chain
                        c = b3                                // set c to new overflow bucket
                        b.move (j, c)                         // move j-th pair to overflow bucket
                    else
                        if b != b1 && b.nKeys == 0 then
                            bp.next = b.next                  // remove empty overflow bucket
                    end if
                else
                    j += 1
            end while
        } // cfor
    end rehash

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the load factor (# keys over # home slots) for this hash table.
     */
    private def loadFactor: Double = keyCount / size.toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the key in the bucket chain that starts with home bucket bh and
     *  return the current value stored stored for the key.
     *  @param key     the key to find
     *  @param bh      the given home bucket
     *  @param by_get  whether 'find' is called from 'get' (performance monitoring)
     */
    private def find (key: K, bh: Bucket, by_get: Boolean): Option [V] =
        var b = bh
        var res: Option [V] = None
        cfor (b != null && res == None, {b = b.next}) {
            if by_get then count += 1
            val result = b.find (key)
            if result != null then res = Some (result)
        } // cfor
        res 
    end find

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Hash the key using the low resolution hash function and if needed, the
     *  high resolution hash function. Return the location of the bucket chain
     *  containing the key-value pair.
     *  @param key  the key to hash
     */
    private def h (key: K): Int = 
        val i = key.hashCode () % mod1
        if i < isplit then h2 (key) else i
    end h

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Hash the key using the high resolution hash function.  Return the location
     *  of the bucket chain containing the key-value pair.
     *  @param key  the key to hash
     */
    private def h2 (key: K): Int = key.hashCode () % mod2

end LinHashMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `linHashMapTest` main function used for testing the `LinHashMap` class.
 *  > runMain scalation.database.linHashMapTest
 */
@main def linHashMapTest (): Unit =

    import scalation.random.Randi0

    val RANDOMLY  = false
    val totalKeys = 50

    val ht = new LinHashMap [Int, Int] ("Test")

    banner ("Insert Keys")
    if RANDOMLY then
        val rng = Randi0 (2 * totalKeys)
        for i <- 1 to totalKeys by 2 do ht.put (rng.igen, i * i)
    else 
        for i <- 1 to totalKeys by 2 do ht.put (i, i * i)
    ht.show ()

    banner ("Find Keys")
    for i <- 0 to totalKeys do println (s"key = $i, value = ${ht.get (i)}")

    banner ("Delete Keys")
    ht -= 41
    ht.show ()
    ht -= 33 
    ht.show ()

    banner ("Analysis Performance")
    println (s"Average number of buckets accessed = ${ht.count / totalKeys.toDouble}")

end linHashMapTest

