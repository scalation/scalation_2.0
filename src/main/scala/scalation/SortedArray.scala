
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 11 13:25:46 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Fixed Size Circular Array (SortedArray)
 */

package scalation

import scala.collection.mutable._
import scala.reflect.ClassTag
import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SortedArray` class provides a circular array that can be used to store the
 *  latest cap elements.  Example to add elements 1, 2, 3, 4, 5, 6, 7 keeping
 *  the most recebt 5:
 *  [ 1, 2, 3, 4, 5 ]   f = 0, l = 4
 *  [ 6, 7, 3, 4, 5 ]   f = 2, l = 1
 *  @param cap   the capacity or maximum number of elements that can be stored
 *  @param zero  an element that indicates zero for type A
 */
class SortedArray [A] (cap: Int) (using ord: Ordering [A], tag: ClassTag [A])
      extends AbstractIterable [A]
         with Builder [A, SortedArray [A]]
         with Serializable:

    import ord.mkOrderingOps 

    private var size_ = 0                                       // number of elements
    private val max   = cap - 1                                 // max available index position
    private val store = Array.ofDim [A] (cap)                   // storage space for elements

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return this sorted array's known size.
     */
    override def knownSize: Int = size_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return this sorted array.
     */
    def result (): SortedArray [A] = this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an iterator to traverse the sorted array.
     */
    override def iterator: Iterator [A] = store.iterator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one element in the correct order.
     *  @param elem  the next element
     */
    def addOne (elem: A): this.type =
        if size_ == cap then
            if elem < store(max) then shiftNinsert (elem, max)
        else
            shiftNinsert (elem, size_)
            size_ += 1
        this
    end addOne

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Enqueue the next element into the sorted array.
     *  @param elem  the next element
     */
    def enqueue (elem: A): Unit = addOne (elem)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move backward by shifting elements right until the insertion spot is found.
     *  @param elem      the next element
     *  @param startIdx  where to start looking for the insertion spot
     */  
    private def shiftNinsert (elem: A, startIdx: Int): Unit =
        var i = startIdx
        while i > 0 && store(i - 1) > elem do
            store(i) = store(i - 1)
            i -= 1
        store(i) = elem
    end shiftNinsert

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clear the sorted array.
     */
    def clear (): Unit = size_ = 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Copy (shallow) this sorted array into another sorted array.
     */
    def copy (): SortedArray [A] =
        val sa = new SortedArray (cap)
        for i <- 0 until size do sa.addOne (store(i))
        sa
    end copy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the minimum:  The root is always the minimum.
     */
    def findMin: Option [A] = if size_ > 0 then Some (store(0)) else None

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the maximum:  The maximum is the larger of the root's children.
     */
    def findMax: Option [A] = if size_ > 0 then Some (store(size_ - 1)) else None

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the elements in the sorted array in order (min to max).
     */
    def printInOrder (): Unit =
        print ("SortedArray: ")
        for i <- 0 until size do println (store(i))
    end printInOrder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show/print the elements in the sorted array in logical order.
     */
    def show (): Unit =
        println ("-" * 60)
        print ("SortedArray(") 
        for j <- 0 until size_ do
            print (store(j))
            if j < size_ - 1 then print (", ")
        println (")")
    end show

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this sorted array into a string showing the storage/physical view.
     */
    override def toString: String = s"SortedArray(${stringOf (store)})"

end SortedArray


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sortedArrayTest` main function is used to test the `SortedArray` class.
 *  > runMain scalation.sortedArrayTest
 */
@main def sortedArrayTest (): Unit =

    val sa = new SortedArray [Double] (5)
    banner ("sa.addOne (1)")
    sa.addOne (1); sa.show (); println (sa)
    banner ("sa.addOne (7)")
    sa.addOne (7); sa.show (); println (sa)
    banner ("sa.addOne (3)")
    sa.addOne (3); sa.show (); println (sa)
    banner ("sa.addOne (4)")
    sa.addOne (4); sa.show (); println (sa)
    banner ("sa.addOne (5)")
    sa.addOne (5); sa.show (); println (sa)
    banner ("sa.addOne (2)")
    sa.addOne (2); sa.show (); println (sa)
    banner ("sa.addOne (6)")
    sa.addOne (6); sa.show (); println (sa)

    banner (s"sa.findMin = ${sa.findMin}")
    banner (s"sa.findMax = ${sa.findMax}")

end sortedArrayTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sortedArrayTest2` main function is used to test the `SortedArray` class.
 *  This test used the reverse order.
 *  @note, may prefer to use `Ordering.Double.TotalOrdering.reverse`
 *  > runMain scalation.sortedArrayTest2
 */
@main def sortedArrayTest2 (): Unit =

    given ord: Ordering [Double] = summon [Ordering [Double]].reverse

    val sa = new SortedArray [Double] (5)
    banner ("sa.addOne (1)")
    sa.addOne (1); sa.show (); println (sa)
    banner ("sa.addOne (7)")
    sa.addOne (7); sa.show (); println (sa)
    banner ("sa.addOne (3)")
    sa.addOne (3); sa.show (); println (sa)
    banner ("sa.addOne (4)")
    sa.addOne (4); sa.show (); println (sa)
    banner ("sa.addOne (5)")
    sa.addOne (5); sa.show (); println (sa)
    banner ("sa.addOne (2)")
    sa.addOne (2); sa.show (); println (sa)
    banner ("sa.addOne (6)")
    sa.addOne (6); sa.show (); println (sa)

    banner (s"sa.findMin = ${sa.findMin}")
    banner (s"sa.findMax = ${sa.findMax}")

end sortedArrayTest2

