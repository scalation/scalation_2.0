
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Mar 20 12:05:49 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Data Structure:  Min-Max Heap using Array as the Backing Store
 *           adapted from AI generated code
 */

package scalation

import scala.collection.mutable._
import scala.reflect.ClassTag

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MinMaxHeap` class stores elements in a heap that is organized to make
 *  finding both the minimum and maximum elements efficient.
 *  It includes methods to serve as a Double-Ended Priority Queue (DEPQ).
 *  @param cap    the capacity/limit on the number of elements allowed in the heap
 *  @param heap_  optionally provide an initial heap ordered array
 */
class MinMaxHeap [T] (val cap: Int, heap_ : Array [T] = null)
                     (implicit ord: Ordering [T], tag: ClassTag [T])
      extends AbstractIterable [T]
         with Builder [T, MinMaxHeap [T]]
         with Serializable:

    private val debug = debugf ("MinMaxHeap", false)                 // debug function
    private val flaw  = flawf ("MinMaxHeap")                         // flaw function
    private val heap  = if heap_ != null then heap_                  // use existing heap ordered array
                        else new Array [T] (cap)                     // new array to store the heap
    private var n     = if heap_ != null then heap_.length           // current number of elements in heap_
                        else 0                                       // new array => empty (0)

    debug ("init", s"cap = $cap, n = $n")
    if heap.length != cap then flaw ("init", s"capacity cap = $cap must = heap.length = ${heap.length}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clear/empty the heap.
     */
    def clear (): Unit = n = 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return this heap.
     */
    def result (): MinMaxHeap [T] = this
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an iterator to traverse the heap.
     */
    override def iterator: Iterator [T] = heap.take (n).iterator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size of heap (number of elements).
     */
    override def size: Int = n
    override def knownSize: Int = n

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the heap is empty.
     */
    override inline def isEmpty: Boolean = n == 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add one element to the heap.
     *  @param elem  the new element to insert
     */
    def addOne (elem: T): this.type = { insert (elem); this }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert a new elem into the heap.
     *  @param elem  the new element to insert
     */
    def insert (elem: T): Unit =
        if n >= cap then throw new RuntimeException ("insert: heap overflow")
        heap(n) = elem
        pushUp (n)
        n += 1
    end insert

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Enqueue a new elem into the heap by inserting unless there is no room,
     *  then replace the maximum element currently in the heap. 
     *  @param elem  the new element to insert
     */
    def enqueue (elem: T): Unit =
        if n < cap then insert (elem)
        else replaceMax (elem)
    end enqueue

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace the maximum elem in the heap with the new elem if it is smaller.
     *  Assumes the heap is not empty (i.e., n > 0).
     *  @param elem  the new element to replace the max
     */
    def replaceMax (elem: T): Unit =
        val maxI = if n == 1 then 0 else if n == 2 || ord.gt (heap(1), heap(2)) then 1 else 2
        val maxE = heap(maxI)
        if ord.gt (maxE, elem) then
            heap(maxI) = elem                                        // elem smaller => replace
            if maxI > 0 && ord.lt (heap(maxI), heap(0)) then         // elem < root => toot not min
                swap (maxI, 0)                                       // max elem the new root
            pushDownMax (maxI)                                       // fix the heap
    end replaceMax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a Min-Max Heap from an unordered array in O(n) time.
     *  @param elems  the initial elements to populate the heap
     */
    def buildHeap (elems: Array [T]): Unit =
        if elems.length > cap then 
            throw new IllegalArgumentException ("buildHeap: array exceeds heap capacity")
        
        // Copy elements into the internal array
        System.arraycopy (elems, 0, heap, 0, elems.length)
        n = elems.length
        
        // Start from the last non-leaf node (at n/2 - 1) and push down to the root
        for i <- n/2 - 1 to 0 by -1 do pushDown (i)
    end buildHeap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Copy (shallow) this heap into another heap.
     */
    def copy (): MinMaxHeap [T] = new MinMaxHeap (cap, heap.clone ())

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the minimum:  The root is always the minimum.
     */
    def findMin: Option [T] = if n > 0 then Some (heap(0)) else None

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the maximum:  The maximum is the larger of the root's children.
     */
    def findMax: Option [T] =
        n match
            case 0 => None
            case 1 => Some (heap(0))
            case 2 => Some (heap(1))
            case _ => Some (if ord.gt (heap(1), heap(2)) then heap(1) else heap(2))
    findMax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Dequeue the minimum element minE that is at the root and reestablish the heap.
     *  As the default, dequeue is dequeueMin.
     */
    def dequeue (): T =
        if isEmpty then throw new NoSuchElementException ("Heap is empty")
        val minE = heap(0)                                           // minimum element at the root
        n -= 1                                                       // decrement the number of elements
        if n > 0 then
            heap(0) = heap(n)                                        // put the last element at the root
            pushDown (0)                                             // push it down the heap if needed
        minE
    end dequeue

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Dequeue the maximum element maxE, the larger of the root's children and
     *  reestablish the heap
     */
    def dequeueMax (): T =
        if isEmpty then throw new NoSuchElementException ("Heap is empty")
        if n == 1 then
            n -= 1
            heap(0)                                                  // case of only one element
        else
            val maxI = if n == 2 || ord.gt (heap(1), heap(2)) then 1 else 2
            val maxE = heap(maxI)                                    // maximum element is at maxI
            n -= 1                                                   // decrement the number of elements
            if maxI < n then
                heap(maxI) = heap(n)                                 // put the last element where max was
                pushDown (maxI)                                      // push it down the heap if needed
            maxE
    end dequeueMax

    //--------------------------------------------------------------------------
    // --- private Push Up methods (used for insertion)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Push the element at i up the tree based on its level in the heap.
     *  @param i  the current node/index position
     */
    private def pushUp (i: Int): Unit =
        val level = (log2 (i + 1)).toInt                             // @see `scalation.CommonFunctions`
        if level % 2 == 0 then                                       // min level
            if i > 0 && ord.gt (heap(i), heap(parent(i))) then
                swap (i, parent(i))
                pushUpMax (parent(i))
            else pushUpMin (i)
        else                                                         // max level
            if i > 0 && ord.lt (heap(i), heap(parent(i))) then
                swap (i, parent(i))
                pushUpMin (parent(i))
            else pushUpMax (i)
    end pushUp

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Push the element at i up the min part of the tree.
     *  @param i  the current node/index position
     */
    private def pushUpMin (i: Int): Unit =
        val gp = grandparent(i)
        if gp >= 0 && ord.lt (heap(i), heap(gp)) then
            swap (i, gp)
            pushUpMin (gp)
    end pushUpMin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Push the element at i up the max part of the tree.
     *  @param i  the current node/index position
     */
    private def pushUpMax (i: Int): Unit =
        val gp = grandparent(i)
        if gp >= 0 && ord.gt (heap(i), heap(gp)) then
            swap (i, gp)
            pushUpMax (gp)
    end pushUpMax

    //--------------------------------------------------------------------------
    // --- private Push Down methods (used for deletion)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Push the element at i down the tree based on its level in the heap.
     *  @param i  the current node/index position
     */
    private def pushDown (i: Int): Unit =
        val level = (log2 (i + 1)).toInt
        if level % 2 == 0 then pushDownMin (i)
        else pushDownMax (i)
    end pushDown

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Push the element at i down the min part of the tree.
     *  @param i  the current node/index position
     */
    private def pushDownMin (i: Int): Unit =
        val m = findSmallestChildOrGrandchild (i)
        if m != -1 then
            if isGrandchild (i, m) then
                if ord.lt (heap(m), heap(i)) then
                    swap (m, i)
                    if ord.gt (heap(m), heap(parent (m))) then swap (m, parent (m))
                    pushDownMin (m)
            else if ord.lt (heap(m), heap(i)) then
                swap (m, i)
    end pushDownMin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Push the element at i down the max part of the tree.
     *  @param i  the current node/index position
     */
    private def pushDownMax (i: Int): Unit =
        val m = findLargestChildOrGrandchild (i)
        if m != -1 then
            if isGrandchild (i, m) then
                if ord.gt (heap(m), heap(i)) then
                    swap (m, i)
                    if ord.lt (heap(m), heap(parent (m))) then swap (m, parent (m))
                    pushDownMax (m)
            else if ord.gt (heap(m), heap(i)) then
                swap (m, i)
    end pushDownMax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the elements in the min-max heap in order (min to max).
     */
    def printInOrder (): Unit =
        val cp = copy ()
        print ("MinMaxHeap: ")
        while ! cp.isEmpty do println (cp.dequeue ())
    end printInOrder

    //--------------------------------------------------------------------------
    // --- private search helper methods

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the index of i's parent.
     *  @param i  the current node
     */
    private inline def parent (i: Int): Int = (i - 1) / 2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the index of i's grand-parent.
     *  @param i  the current node
     */
    private inline def grandparent (i: Int): Int = if i <= 2 then -1 else (i - 3) / 4

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the index of i's smallest child or grand-child.
     *  @param i  the current node
     */
    private def findSmallestChildOrGrandchild (i: Int): Int =
        var m = -1
        inline def check (idx: Int): Unit =
            if idx < n && (m == -1 || ord.lt (heap(idx), heap(m))) then m = idx

        check (2 * i + 1)                                            // child 1
        check (2 * i + 2)                                            // child 2
        for gc <- 4 * i + 3 until math.min (4 * i + 7, n) do check (gc)   // grand-children
        m
    end findSmallestChildOrGrandchild

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the index of i's largest child or grand-child.
     *  @param i  the current node
     */
    private def findLargestChildOrGrandchild (i: Int): Int =
        var m = -1
        inline def check (idx: Int): Unit =
            if idx < n && (m == -1 || ord.gt (heap(idx), heap(m))) then m = idx

        check (2 * i + 1)                                            // child 1
        check (2 * i + 2)                                            // child 1
        for gc <- 4 * i + 3 until math.min (4 * i + 7, n) do check (gc)   // grand-children
        m
    end findLargestChildOrGrandchild

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether node i is a grand-child of node m.
     *  @param i  the current node
     *  @param m  the other node
     */
    private inline def isGrandchild (i: Int, m: Int): Boolean = m > 2 * i + 2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap the elements at nodes i and j.
     *  @param i  the first node
     *  @param j  the second node
     */
    private def swap (i: Int, j: Int): Unit = 
        val temp = heap(i)
        heap(i)  = heap(j)
        heap(j)  = temp
    end swap

end MinMaxHeap


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `minMaxHeapTest` main function tests the `MinMaxHeapTest` class.
 *  > runMain scalation.minMaxHeapTest
 */
@main def minMaxHeapTest (): Unit =

//  given ord: Ordering [Double] = summon [Ordering [Double]].reverse

    banner ("Test MinMaxHeap")
    val cap = 9                                            // position 0 used by data structure => cap = 8 elements + 1
    val pq  = new MinMaxHeap [Double] (cap)
    for i <- 1 to cap do
        pq.enqueue (i * i)
        println (pq)
//  println ("Insert 20"); pq.insert (20)                  // causes overflow
    println ("Insert 20"); pq.enqueue (20)                 // new element replaces the max
    println (pq)
    while ! pq.isEmpty do println (pq.dequeue ())          // dequeue in min order
//  while ! pq.isEmpty do println (pq.dequeueMax ())       // dequeue in max order

end minMaxHeapTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `minMaxHeapTest2` main function tests the `MinMaxHeapTest` class.
 *  @see `scalation.Timer.time`
 *  > runMain scalation.minMaxHeapTest2
 */
@main def minMaxHeapTest2 (): Unit =

//  given ord: Ordering [Double] = summon [Ordering [Double]].reverse

    banner ("Timing Test MinMaxHeap")

    for cap <- 100000 until 1000000 by 100000 do
        print (s"MinMaxHeap      at $cap ")
        time {
            val pq  = new MinMaxHeap [Double] (cap)
            for i <- 1 to cap do pq.enqueue (i * i)
            while ! pq.isEmpty do pq.dequeue ()
        } // end time

        print (s"PriorityQueueFW at $cap ")
        time {
            val pq2  = new PriorityQueueFW [Double] (cap)
            for i <- 1 to cap do pq2.enqueue (i * i)
            while ! pq2.isEmpty do pq2.dequeue ()
        } // end time
    end for

end minMaxHeapTest2

