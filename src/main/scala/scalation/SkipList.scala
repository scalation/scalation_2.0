
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Korede Bishi
 *  @version 2.0
 *  @date    Thu Aug 17 19:08:17 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Skip List
 */

//  U N D E R   D E V E L O P M E N T

package scalation

import scala.util.Random

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SkipList` class ...
 *  @tparam K         the type of the keys contained in this sorted map
 *  @tparam V         the type of the values assigned to keys in this sorted map
 *  @param  ordering  the implicit ordering used to compare objects of type K
 */
class SkipList [K, V] (using ordering: Ordering [K]):

    private val maxlevel = 10                                      // maximum number of levels for the skip  list
    private val random   = new Random ()
    private val head     = new SkipNodeType [K, V] (null.asInstanceOf [K], null.asInstanceOf [V], Array.fill (maxlevel)(null))
//  private var topLevel = 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Search for the given key in this skip list.
     *  @param key  the key to search for, returning the assocaited value
     */
    def search (key: K): Option [V] =
        var current= head
        var result: Option [V] = None
        for level <- maxlevel -1 to 0 by -1 do
            while current.next(level) != null && ordering.compare (current.next(level).key, key) < 0 do
                current= current.next(level)
            if current.next(0) != null && ordering.compare (current.next(0).key, key) == 0 then
                //val foundNode = current.next(0).value
                //banner(s"The key is found $key with a node value $foundNode")
                result = Some (current.next(0).value)
        result.foreach (foundNode => banner (s"The key is found $key with a node value $foundNode"))
        result
    end search

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert a key (of type K) and value (of type V) into this skip list.
     *  An array named update is created of type `SkipNodeType` with a maximum level as
     *  the maximum level of the SkipList and do an insertion update.
     *  A randomLevel function is used to generate a
     *  @param key    the key to insert
     *  @param value  the key's associated value
     */
    def insert (key: K, value: V): Unit =
        val update  = new Array [SkipNodeType [K, V]] (maxlevel)
        var current = head
        for level <- maxlevel - 1 to 0 by -1 do
            while current.next(level) != null && ordering.compare (current.next(level).key, key) < 0 do
                current = current.next(level)
            update(level) = current
        end for
        current = current.next(0)
        if current != null && ordering.compare (current.key, key) == 0 then
            current.value = value
        else
            val newLevel = randomLevel ()
            val newNode  = new SkipNodeType [K, V] (key, value, Array.fill (newLevel)(null))
            for level <- 0 until newLevel do
                newNode.next(level) = update(level).next(level)
                update(level).next(level) = newNode
            end for
    end insert

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Delete the the entry with the given key.
     *  @param key  the key to delete
     */
    def delete (key: K): Unit =
        val update = new Array [SkipNodeType [K, V]] (maxlevel)
        val toUpdate = new Array [Boolean] (maxlevel)
        var current = head
        for level <- maxlevel - 1 until 0 by -1 do
            while current.next(level) != null && ordering.compare (current.next(level).key, key) < 0 do
                current = current.next(level)
            update(level) = current
            toUpdate(level) = current.next(level) != null && ordering.compare (current.next(level).key, key) == 0
        end for
        current = current.next(0)
        if current != null && ordering.compare (current.key, key) == 0 then
            for level <- 0 until maxlevel do
                if toUpdate(level) then update(level).next(level) = current.next(level)
    end delete

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly generate a level.
     */
    def randomLevel (): Int =
        var level = 1
        while random.nextBoolean () && level < maxlevel do level += 1
        level
    end randomLevel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  The `Node` inner class ...
     *
    private class Node [K, V] (var key: K, var value: V, var levels: Int):

        private val next = new Array [SkipNodeType [K, V]] (levels)
        def getnext (level: Int): SkipNodeType [K, V] = next(level)
        def setNext (level: Int, node: SkipNodeType [K, V]): Unit = next(level) = node

    end Node
     */

end SkipList


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SkipNodeType` ...
 *  @tparam K      the type of the keys contained in this sorted map
 *  @tparam V      the type of the values assigned to keys in this sorted map
 *  @param  key    the key for this node
 *  @param  value  the value for this node
 *  @param  next   array of references to next nodes
 */
class SkipNodeType [K, V] (var key: K, var value: V, val next: Array [SkipNodeType [K, V]])


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `skipListTest` main function test the `SkipList` class.
 *  > runMain scalation.database.skipListTest
 */ 
@main def skipListTest (): Unit =

    val skipList = new SkipList [Int, String] ()

    // Insert elements into the Skip List

    skipList.insert (20, "Value 2")
    skipList.insert (40, "Value 3")
    skipList.insert (30, "Value 4")
    skipList.insert (25, "Value 2")

    // Search for elements

    println (skipList.search (20))      // Output: Some("Value 2")
    println (skipList.search (25))      // Output: Some("Value 2")
    println (skipList.search (30))      // Output: Some("Value 4")

    // Delete elements from the Skip List

    skipList.delete (25)

    println (skipList.search (25))      // Output: None

end skipListTest

