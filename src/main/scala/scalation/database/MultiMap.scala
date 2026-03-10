
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Aug  9 19:29:27 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Multi-Map (K, Set [V]) Index Structures for Non-Unique Indices
 *
 *  val index = new LinHashMultiMap [Int, Int] ()           // using ScalaTion's Linear Hash Map
 *  val index = new HashMultiMap [Int, Int] ()              // using Scala's Hash Map
 *  val index = new JHashMultiMap [Int, Int] ()             // using Java's Hash Map
 *  val index = new BpTreeMultiMap [Int, Int] ()            // using ScalaTion's B+Tree Map
 *  val index = new TreeMultiMap [Int, Int] ()              // using Scala's Tree Map
 *  val index = new JTreeMultiMap [Int, Int] ()             // using Java's Tree Map
 */

package scalation
package database

import scala.collection.mutable.{HashMap, TreeMap, Set}
import scala.reflect.ClassTag

//  H a s h   B a s e d   I n d e x

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LinHashMultiMap` class provides hash maps that use the Linear Hashing algorithm.
 *  It build on `LinHashMap` allowing values to multi-valued `Set [V]` and can be
 *  used for building Non-Unique Indices.
 *  @see scalation.database.LinHashMap
 *  @tparam K           the type of the keys contained in this hash map
 *  @tparam V           the base-type of the values assigned to keys in this hash map
 *  @param  name        the name of the Linear Hash MultiMap
 *  @param  order       the number of slots per bucket
 *  @param  loadFactor  the (lower, upper) bound on the load factor (# keys over # home slots)
 */
//class LinHashMultiMap [K: ClassTag, V: ClassTag] (name: String, order: Int = 4,
class LinHashMultiMap [K: ClassTag, V] (name: String, order: Int = 4,
                                        loadFactor: (Double, Double) = (0.3, 1.2))
      extends LinHashMap [K, Set [V]] (name, order, loadFactor):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one key-value pair into this hash table and return this (called by put).
     *  The addOne method adds a set of values, whereas addOne1 adds a single value.
     *  @param elem   the key-value pair to add/insert for an individual value
     */
    def addOne1 (elem: (K, V)): this.type =
        val (key, value) = elem
        val current = getOrElse (key, null)
        val multiElem = if current == null then (key, Set (value))
                        else (key, current union Set (value))
        addOne (multiElem)
    end addOne1

end LinHashMultiMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `HashMultiMap` class provides hash maps that use the Separate Chaining algorithm.
 *  It build on `HashMap` allowing values to multi-valued `Set [V]` and can be
 *  used for building Non-Unique Indices.
 *  @see scala.collection.mutable.HashMap
 *  @tparam K           the type of the keys contained in this hash map
 *  @tparam V           the base-type of the values assigned to keys in this hash map
 *  @param  initialCap  the initial hash table size (number of slots)
 *  @param  loadFactor  the load factor (number of keys over number of slots)
 */
class HashMultiMap [K, V] (initialCap: Int    = HashMap.defaultInitialCapacity,
                           loadFactor: Double = HashMap.defaultLoadFactor)
      extends Serializable:

    protected val hmap = new HashMap [K, Set [V]] (initialCap,
                                                   loadFactor)    // delegate to HashMap
    val count = 0                                                 // count # nodes accessed (performance)
                                                                  // not implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one key-value pair into this hash map and return this (called by put).
     *  The addOne method adds a set of values, whereas addOne1 adds a single value.
     *  @param elem   the key-value pair to add/insert for an individual value
     */
    def addOne1 (elem: (K, V)): hmap.type =
        val (key, value) = elem
        val current = hmap.getOrElse (key, null)
        val multiElem = if current == null then (key, Set (value))
                        else (key, current union Set (value))
        hmap.addOne (multiElem)
    end addOne1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print this HashMap.
     */
    def show (): Unit =
        println ("HashMap")
        println ("-" * 60)
        for elem <- hmap do println (elem)
        println ("-" * 60)
    end show

    export hmap.*

end HashMultiMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `JHashMultiMap` class provides hash maps that use the Separate Chaining algorithm
 *  with several optimizations.  It build on Java's `HashMap` allowing values to
 *  multi-valued `Set [V]` and can be used for building Non-Unique Indices.
 *  @see java.util.HashMap
 *  @tparam K           the type of the keys contained in this hash map
 *  @tparam V           the base-type of the values assigned to keys in this hash map
 *  @param  initialCap  the initial hash table size (number of slots)
 *  @param  loadFactor  the load factor (number of keys over number of slots)
 */
class JHashMultiMap [K, V] (initialCap: Int   = 16,
                            loadFactor: Float = 0.75)
      extends Serializable:

    protected val hmap = new java.util.HashMap [K, Set [V]] (initialCap,
                                                             loadFactor)    // delegate to Java's HashMap
    val count = 0                                                 // count # nodes accessed (performance)
                                                                  // not implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one key-value pair into this hash map and return this (called by put).
     *  The addOne method adds a set of values, whereas addOne1 adds a single value.
     *  @param elem   the key-value pair to add/insert for an individual value
     */
    def addOne1 (elem: (K, V)): hmap.type =
        val (key, value) = elem
        val current = hmap.getOrDefault (key, null)
        val multiElem = if current == null then (key, Set (value))
                        else (key, current union Set (value))
        hmap.put (multiElem._1, multiElem._2)
        hmap
    end addOne1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print this HashMap.
     */
    def show (): Unit =
        println ("HashMap")
        println ("-" * 60)
        val it = hmap.entrySet.iterator
        while it.hasNext do println (it.next ())
        println ("-" * 60)
    end show

    export hmap.{toString as _, hashCode as _, equals as _, clone as _, *}

end JHashMultiMap

//  T r e e   B a s e d   I n d e x

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BpTreeMultiMap` class provides tree maps that use the B+Tree algorithm.
 *  It build on `BpTreeMap` allowing values to multi-valued `Set [V]` and can be
 *  used for building Non-Unique Indices.
 *  -tparam K      the type of the keys contained in this tree map (FIX add this type)
 *  @tparam V      the base-type of the values assigned to keys in this tree map
 *  @param  order  the number of order (maximum number of children) of the tree
 *  @param  ord    the implicit ordering used to compare objects of type K
 */
class BpTreeMultiMap [V] (order: Int = 4)
      extends BpTreeMap [Set [V]] ("BpTreeMultiMap"):     // FIX: add ", order"
 
    println (s"BpTreeMultiMap: order = $order")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one key-value pair into this tree map and return this (called by put).
     *  The addOne method adds a set of values, whereas addOne1 adds a single value.
     *  @param elem   the key-value pair to add/insert for an individual value
     */
    def addOne1 (elem: (ValueType, V)): this.type =
        val (key, value) = elem
        val current = getOrElse (key, null)
        val multiElem = if current == null then (key, Set (value))
                        else (key, current union Set (value))
        addOne (multiElem)
    end addOne1

end BpTreeMultiMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TreeMultiMap` class provides tree maps that use the Red-Black Tree algorithm.
 *  It build on `TreeMap` allowing values to multi-valued `Set [V]` and can be
 *  used for building Non-Unique Indices.
 *  @see scala.collection.mutable.TreeMap
 *  @tparam K    the type of the keys contained in this tree map
 *  @tparam V    the base-type of the values assigned to keys in this tree map
 *  @param  ord  the implicit ordering used to compare objects of type K
 */
class TreeMultiMap [K, V] (implicit val ord: Ordering [K])
      extends Serializable:

    protected val tree = new TreeMap [K, Set [V]] ()              // delegate to TreeMap
    val count = 0                                                 // count # nodes accessed (performance)
                                                                  // not implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one key-value pair into this tree map and return this (called by put).
     *  The addOne method adds a set of values, whereas addOne1 adds a single value.
     *  @param elem   the key-value pair to add/insert for an individual value
     */
    def addOne1 (elem: (K, V)): tree.type =
        val (key, value) = elem
        val current = tree.getOrElse (key, null)
        val multiElem = if current == null then (key, Set (value))
                        else (key, current union Set (value))
        tree.addOne (multiElem)
    end addOne1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print this TreeMap.
     */
    def show (): Unit =
        println ("TreeMap")
        println ("-" * 60)
        for elem <- tree do println (elem)
        println ("-" * 60)
    end show

    export tree.*

end TreeMultiMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TreeMultiMap` class provides tree maps that use the Red-Black Tree algorithm.
 *  It build on Java's `TreeMap` allowing values to multi-valued `Set [V]` and can be
 *  used for building Non-Unique Indices.
 *  @see java.util.TreeMap
 *  @tparam K    the type of the keys contained in this tree map
 *  @tparam V    the base-type of the values assigned to keys in this tree map
 *  @param  ord  the implicit ordering used to compare objects of type K
 */
class JTreeMultiMap [K, V] (implicit val ord: Ordering [K])
      extends Serializable:

    protected val tree = new java.util.TreeMap [K, Set [V]] ()    // delegate to Java's TreeMap
    val count = 0                                                 // count # nodes accessed (performance)
                                                                  // not implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one key-value pair into this tree map and return this (called by put).
     *  The addOne method adds a set of values, whereas addOne1 adds a single value.
     *  @param elem   the key-value pair to add/insert for an individual value
     */
    def addOne1 (elem: (K, V)): tree.type =
        val (key, value) = elem
        val current = tree.getOrDefault (key, null)
        val multiElem = if current == null then (key, Set (value))
                        else (key, current union Set (value))
        tree.put (multiElem._1, multiElem._2)
        tree
    end addOne1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print this TreeMap.
     */
    def show (): Unit =
        println ("TreeMap")
        println ("-" * 60)
        val it = tree.entrySet.iterator
        while it.hasNext do println (it.next ())
        println ("-" * 60)
    end show

    export tree.{toString as _, hashCode as _, equals as _, clone as _, *}

end JTreeMultiMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `multiMapTest` main function used for testing the Multi-Map classes.
 *  > runMain scalation.database.multiMapTest
 */
@main def multiMapTest (): Unit =

    import scalation.random.Randi0

    val RANDOMLY  = false
    val totalKeys = 50

//  val index = new LinHashMultiMap [Int, Int] ()
//  val index = new HashMultiMap [Int, Int] ()
//  val index = new JHashMultiMap [Int, Int] ()

//  val index = new BpTreeMultiMap [Int, Int] ()
    val index = new TreeMultiMap [Int, Int] ()
//  val index = new JTreeMultiMap [Int, Int] ()

    banner ("Insert Keys into Index")
    if RANDOMLY then
        val rng = Randi0 (2 * totalKeys)
        for i <- 1 to totalKeys by 2 do index.put (rng.igen, Set (i~^2, i~^3))
    else
        for i <- 1 to totalKeys by 2 do index.put (i, Set (i~^2, i~^3))
    index.show ()

    banner ("Find Keys")
    for i <- 0 to totalKeys do println (s"key = $i, value = ${index.get (i)}")

    banner ("Delete Keys")
    index -= 41
    index.show ()
    index -= 33
    index.show ()

    banner ("Analysis Performance")
    println (s"Average number of buckets accessed = ${index.count / totalKeys.toDouble}")

end multiMapTest

