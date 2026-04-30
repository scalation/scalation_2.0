
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Aug  9 19:29:27 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Map (K, V) Index Structures for Java Maps
 *
 *  val index = new JHashMap [Int, Int] ()                  // using Java's Hash Map
 *  val index = new JTreeMap [Int, Int] ()                  // using Java's Tree Map
 */

package scalation
package database

//  H a s h   B a s e d   I n d e x

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `JHashMap` class provides hash maps that use the Separate Chaining algorithm
 *  with several optimizations.  It uses on Java's `HashMap` and can be used for
 *  building Unique Indices.
 *  @see java.util.HashMap
 *  @tparam K           the type of the keys contained in this hash map
 *  @tparam V           the base-type of the values assigned to keys in this hash map
 *  @param  initialCap  the initial hash table size (number of slots)
 *  @param  loadFactor  the load factor (number of keys over number of slots)
 */
class JHashMap [K, V] (initialCap: Int   = 16,
                       loadFactor: Float = 0.75)
      extends Serializable:

    private   val flaw = flawf ("JHashMap")
    protected val hmap = new java.util.HashMap [K, V] (initialCap,
                                                       loadFactor)    // delegate to Java's HashMap
    val count = 0                                                 // count # nodes accessed (performance)
                                                                  // not implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the entry from this map having the given key.
     *  @param  the key of the entry to be removed
     */
    def -= (key: K): hmap.type =
        val old = hmap.remove (key)
        if old == null then flaw ("-=", s"the key = $key to be removed is not in this map")
        hmap
    end -=

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

end JHashMap

//  T r e e   B a s e d   I n d e x

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TreeMMap` class provides tree maps that use the Red-Black Tree algorithm.
 *  It uses on Java's `TreeMap` and can be  used for building Unique Indices.
 *  @see java.util.TreeMap
 *  @tparam K    the type of the keys contained in this tree map
 *  @tparam V    the base-type of the values assigned to keys in this tree map
 *  @param  ord  the implicit ordering used to compare objects of type K
 */
class JTreeMap [K, V] (implicit val ord: Ordering [K])
      extends Serializable:

    private   val flaw = flawf ("JTreeMap")
    protected val tree = new java.util.TreeMap [K, V] ()          // delegate to Java's TreeMap
    val count = 0                                                 // count # nodes accessed (performance)
                                                                  // not implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the entry from this map having the given key.
     *  @param  the key of the entry to be removed
     */
    def -= (key: K): tree.type =
        val old = tree.remove (key)
        if old == null then flaw ("-=", s"the key = $key to be removed is not in this map")
        tree
    end -=

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

end JTreeMap


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `javaMapTest` main function used for testing the Java-Map classes.
 *  > runMain scalation.database.javaMapTest
 */
@main def javaMapTest (): Unit =

    import scalation.random.Randi0

    val RANDOMLY  = false
    val totalKeys = 50

    val index = new JHashMap [Int, Int] ()
//  val index = new JTreeMap [Int, Int] ()

    banner ("Insert Keys into Index")
    if RANDOMLY then
        val rng = Randi0 (2 * totalKeys)
        for i <- 1 to totalKeys by 2 do index.put (rng.igen, i~^2)
    else
        for i <- 1 to totalKeys by 2 do index.put (i, i~^2)
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

end javaMapTest

