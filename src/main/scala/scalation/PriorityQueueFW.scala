
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  EPFL and Lightbend, Inc., John Miller
 *  @version 2.0
 *  @date    Sun Jun  6 15:07:08 EDT 2021
 *
 *  @note    Fixed Width Extension to Scala's Priority Queue class
 *           with decreaseKey, increaseKey, printInOrder 
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/* Scala (https://www.scala-lang.org)
 *
 * Copyright EPFL and Lightbend, Inc.
 *
 * Licensed under Apache License 2.0
 * (http://www.apache.org/licenses/LICENSE-2.0).
 *
 * See the NOTICE file distributed with this work for
 * additional information regarding copyright ownership.
 *
 * @author John Miller added decreaseKey, increaseKey, printInOrder
 */

package scalation

import scala.collection.{AbstractIterator, IterableOnce, IterableOps,
                         SortedIterableFactory, StrictOptimizedIterableOps}
import scala.collection.generic.DefaultSerializationProxy
import scala.collection.immutable
import scala.collection.mutable.{AbstractIterable, ArrayBuilder, ArrayBuffer,
                                 Builder, Cloneable, Growable, Iterable, Queue}
import scala.math.Ordering
import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PriorityQueueFW` class implements priority queues using a heap.
 *  To prioritize elements of type A there must be an implicit
 *  Ordering [A] available at creation.
 *
 *  If multiple elements have the same priority in the ordering of this
 *  PriorityQueueFW, no guarantees are made regarding the order in which elements
 *  are returned by `dequeue` or `dequeueAll`. In particular, that means this
 *  class does not guarantee first in first out behaviour that may be
 *  incorrectly inferred from the Queue part of the name of this class.
 *
 *  Only the `dequeue` and `dequeueAll` methods will return elements in priority
 *  order (while removing elements from the heap).  Standard collection methods
 *  including `drop`, `iterator`, and `toString` will remove or traverse the heap
 *  in whichever order seems most convenient.
 *
 *  Therefore, printing a `PriorityQueueFW` will not reveal the priority order of
 *  the elements, though the highest-priority element will be printed first.  To
 *  print the elements in order, one must duplicate the `PriorityQueueFW` (by using
 *  `clone`, for instance) and then dequeue them:
 *
 *  @example {{{
 *  val pq = collection.mutable.PriorityQueueFW(1, 2, 5, 3, 7)
 *  println(pq)                  // elements probably not in order
 *  println(pq.clone.dequeueAll) // prints ArraySeq(7, 5, 3, 2, 1)
 *  }}}
 *
 *  @tparam A   type of the elements in this priority queue.
 *  @param ord  implicit ordering used to compare the elements of type `A`.
 *  @param cap  the capacity (maximum number of elements + 1) of this fixed width priority queue
 *
 *  @define Coll PriorityQueueFW
 *  @define coll priority queue
 *  @define orderDependent
 *  @define orderDependentFold
 *  @define mayNotTerminateInf
 *  @define willNotTerminateInf
 */
//class PriorityQueueFW [A] (implicit val ord: Ordering [A])
class PriorityQueueFW [A] (cap: Int)(using val ord: Ordering [A])
      extends AbstractIterable [A]
         with Iterable [A]
         with IterableOps [A, Iterable, PriorityQueueFW [A]]
         with StrictOptimizedIterableOps [A, Iterable, PriorityQueueFW [A]]
         with Builder [A, PriorityQueueFW [A]]
         with Cloneable [PriorityQueueFW [A]]
         with Growable [A]
         with Serializable:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `ResizableArrayAccess` class extends `ArrayBuffer` with methods below.
     *  The size of the ArrayBuffer should be fixed at 
     *  Used for internal storage of heap.
     */
    private class ResizableArrayAccess [A0] extends ArrayBuffer [A0] (cap):

        override def mapInPlace (f: A0 => A0): this.type =
            var i = 1           // see "we do not use array(0)" comment below (???)
            val siz = this.size
            while i < siz do { this(i) = f(this(i)); i += 1 }
            this
        end mapInPlace

        def p_size0 = size0
        def p_size0_=(s: Int) = size0 = s
        def p_array = array
//      def p_ensureSize (n: Int) = super.ensureSize (n)
        def p_swap (a: Int, b: Int): Unit =
            val h = array(a); array(a) = array(b); array(b) = h
        end p_swap
    end ResizableArrayAccess

    private val debug  = debugf ("PriorityQueueFW", true)    // debug function

    private val resarr = new ResizableArrayAccess [A]    // internal storage for priority queue

    resarr.p_size0 += 1                                  // we do not use array(0) TODO: explain -- what is the first element even for?

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the length of this priority queue.
     */
    def length: Int = resarr.length - 1                  // adjust length accordingly

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size of this priority queue.
     */
    override def size: Int = length

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the known size of this priority queue.
     */
    override def knownSize: Int = length

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this priority is empty.
     */
    override def isEmpty: Boolean = resarr.p_size0 < 2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a priority from a collection type that is not eligible for
     *  EvidenceIterableFactoryDefaults since C != CC[A] (PriorityQueueFW[A] != Iterable[A])
     *  @param coll  the collection of elements to be turned into a priority queue
     */
    override protected def fromSpecific (coll: scala.collection.IterableOnce [A]): PriorityQueueFW [A] =
        PriorityQueueFW.from (coll)
    end fromSpecific

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a priority queue builder.
     */
    override protected def newSpecificBuilder: Builder [A, PriorityQueueFW [A]] = PriorityQueueFW.newBuilder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make this priority empty.
     */
    override def empty: PriorityQueueFW [A] = PriorityQueueFW.empty

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply/map the function f in-place
     *  @param f  the function to be applied
     */
    def mapInPlace (f: A => A): this.type =
        resarr.mapInPlace (f)
        heapify (1)
        this
    end mapInPlace

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return this priority queue.
     */
    def result () = this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Treat an `AnyRef` reference as having type A.
     *  @param x  the refereence
     */
    private def toA (x: AnyRef): A = x.asInstanceOf [A]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make swaps going up the tree.
     *  Use `ord` directly to avoid allocating `OrderingOps`
     *  @param as  array reference to heap 
     *  @param m   start index
     */
    protected def fixUp (as: Array [AnyRef], m: Int): Unit =
        var k: Int = m
        while k > 1 && ord.lt (toA (as(k / 2)), toA (as(k))) do
            resarr.p_swap (k, k / 2)
            k = k / 2
        end while
    end fixUp

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make swaps going down the tree.
     *  Returns true if any swaps were done (used in heapify).
     *  Use `ord` directly to avoid allocating `OrderingOps`.
     *  @param as  array reference to heap 
     *  @param m   start index
     *  @param n   end index
     */
    protected def fixDown (as: Array [AnyRef], m: Int, n: Int): Boolean =
        var k: Int = m
        while n >= 2 * k do
            var j = 2 * k
            if j < n && ord.lt (toA (as(j)), toA (as(j + 1))) then j += 1
            if ord.gteq (toA (as(k)), toA (as(j))) then return k != m
            else 
                val h = as(k)
                as(k) = as(j)
                as(j) = h
                k = j
            end if
        end while
        k != m
    end fixDown

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert a single element into the priority queue.
     *  @param elem  the element to insert.
     *  @return      this $coll.
     */
    def addOne (elem: A): this.type =
//      resarr.p_ensureSize (resarr.p_size0 + 1)
        if resarr.p_size0 < cap then 
            resarr.p_array(resarr.p_size0) = elem.asInstanceOf [AnyRef]
            fixUp (resarr.p_array, resarr.p_size0)
            resarr.p_size0 += 1
        else
            replaceLow (elem)                   // no space, replace lower of new and old lowest
        this
    end addOne

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** When the priority queue is full, only insert the new element if it is better than
     *  the lowest priority element.  In which case, replace the lowest priority element.
     *  FIX - should work off of `ord` rather than using max (which assumes higher is worse)
     *  @param elem  the new element to possibly insert
     */
    private def replaceLow (elem: A): Unit =
        debug ("replaceLow", s"${stringOf (resarr.p_array)}")
        val (maxI, maxE) = findLow
        debug ("replaceLow", s"ord.gt ($elem, $maxE) = ${ord.gt (elem, maxE)}") 
        if ord.gt (elem, maxE) then resarr.p_array(maxI) = elem.asInstanceOf [AnyRef]    // overwrite if new is better
    end replaceLow

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the lowest priority element in the queue; it will be in one of the leaf nodes.
     *  FIX - should work off of `ord` rather than using max (which assumes higher is worse)
     */
    private def findLow: (Int, A) =
        val n    = resarr.p_size0-1
        var maxI = n
        var maxE = toA (resarr.p_array(maxI))
        for i <- n/2 + 1 until n do
            val el = toA (resarr.p_array(i))
            if ord.gt (maxE, el) then { maxI = i; maxE = el }
        (maxI, maxE)
    end findLow

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add all elements.
     *  @param xs  the elements to be added
     */
    override def addAll (xs: IterableOnce [A]): this.type =
        val from = resarr.p_size0
//      for x <- xs.iterator do unsafeAdd (x)
        for x <- xs.iterator do addOne (x)
        heapify (from)
        this
    end addAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an element without calling heapify.
     *  Like += but skips fixUp, which breaks the ordering invariant.
     *  A series of unsafeAdds MUST be followed by heapify.
     *  @param elem  the element to add
     *
    private def unsafeAdd (elem: A): Unit =
//      resarr.p_ensureSize (resarr.p_size0 + 1)
        resarr.p_array(resarr.p_size0) = elem.asInstanceOf [AnyRef]
        resarr.p_size0 += 1
    end unsafeAdd
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reestablish the heap order.
     *  Elements at indices 1..from-1 were already in heap order before any adds.
     *  Elements at indices from..n are newly added, their order must be fixed.
     *----------------------------------------------------------------------------
     *  Do fixDown on the parents of all the new elements, except the parent of the
     *  first new element, which is in the queue (that parent is treated specially
     *  because it might be the root)
     *  @param from  fix the heap from position/index from to the end of the heap
     */
    private def heapify (from: Int): Unit =
        val n = length

        if from <= 2 then              // no pre-existing order to maintain, do the textbook heapify algorithm
            for i <- n/2 to 1 by -1 do fixDown (resarr.p_array, i, n)
        else if n - from < 4 then      // for very small adds, doing the simplest fix is faster
            for i <- from to n do fixUp (resarr.p_array, i)
        else
            var min = from/2           // tracks the minimum element in the queue
            val queue = scala.collection.mutable.Queue [Int](min)

            for i <- n/2 until min by -1 do
                if fixDown (resarr.p_array, i, n) then
                    // there was a swap, so also need to fixDown i's parent
                    val parent = i/2
                    if parent < min then              // make sure same parent isn't added twice
                        min = parent
                        queue += parent
                end if
            end for

            while queue.nonEmpty do
                val i = queue.dequeue ()
                if fixDown (resarr.p_array, i, n) then
                  val parent = i/2
                  if parent < min && parent > 0 then
                      // the "parent > 0" is to avoid adding the parent of the root
                      min = parent
                      queue += parent
                end if
            end while
        end if
    end heapify

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Adds all elements provided by a `IterableOnce` object into the priority queue.
     *  @param xs  a iterable object.
     *  @return    a new priority queue containing elements of both `xs` and `this`.
     */
    def ++ (xs: IterableOnce [A]): PriorityQueueFW [A] = { this.clone () ++= xs }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Adds all elements to the queue.
     *  @param elems  the elements to add.
     */
    def enqueue (elems: A*): Unit = this ++= elems

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the element with the highest priority in the queue, and removes
     *  this element from the queue.
     *  throws NoSuchElementException
     *  @return  the element with the highest priority.
     */
    def dequeue (): A =
        if resarr.p_size0 > 1 then
            resarr.p_size0 = resarr.p_size0 - 1
            val result = resarr.p_array(1)
            resarr.p_array(1) = resarr.p_array(resarr.p_size0)
            resarr.p_array(resarr.p_size0) = null             // erase reference from array
            fixDown (resarr.p_array, 1, resarr.p_size0 - 1)
            toA (result)
        else
            throw new NoSuchElementException ("no element to remove from heap")
    end dequeue

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns and removes all the elements in the priority queue.
     */
    def dequeueAll [A1 >: A]: immutable.Seq [A1] =
        val b = ArrayBuilder.make [Any]
        b.sizeHint (size)
        while nonEmpty do b += dequeue ()
        immutable.ArraySeq.unsafeWrapArray (b.result ()).asInstanceOf [immutable.ArraySeq [A1]]
    end dequeueAll

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Since the element's priority is being decreased, move it to a lower priority
     *  position (toward the back).
     *  @param elem    the element to reposition
     *  @param upElem  the updated version of the element to reposition
     */
    def decreaseKey (elem: A, upElem: A): Unit =
        if ord.lt (upElem, elem) then                                    // make sure priority is decreased
            val m = resarr.p_array.indexOf (elem.asInstanceOf [AnyRef])  // find the element in the heap
            resarr.p_array(m) = upElem.asInstanceOf [AnyRef]             // replace it with its updated version
            fixDown (resarr.p_array, m, resarr.p_size0)                  // re-position in heap if needed
    end decreaseKey

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Since the element's priority is being increased, move it to a higher priority
     *  position (toward the front).
     *  @param elem    the element to reposition
     *  @param upElem  the updated version of the element to reposition
     */
    def increaseKey (elem: A, upElem: A): Unit =
        if ord.gt (upElem, elem) then                                    // make sure priority is increased
            val m = resarr.p_array.indexOf (elem.asInstanceOf [AnyRef])  // find the element in the heap
            resarr.p_array(m) = upElem.asInstanceOf [AnyRef]             // replace it with its updated version
            fixUp (resarr.p_array, m)                                    // re-position in heap if needed
    end increaseKey

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the elements in the priority queue in order.
     */
    def printInOrder: Unit =
        print ("PriorityQueueFW: ")
        println (clone.dequeueAll)
    end printInOrder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the element with the highest priority in the queue,
     *  or throws an error if there is no element contained in the queue.
     *  @return   the element with the highest priority.
     */
    override def head: A = if resarr.p_size0 > 1 then toA (resarr.p_array(1))
                           else throw new NoSuchElementException ("queue is empty")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Removes all elements from the queue.  After this operation is completed,
     *  the queue will be empty.
     */
    def clear (): Unit =
        resarr.clear ()
        resarr.p_size0 = 1
    end clear

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns an iterator which yields all the elements.  Note: The order of elements
     *  returned is undefined.  If you want to traverse the elements in priority queue
     *  order, use `clone().dequeueAll.iterator`.
     *  @return  an iterator over all the elements.
     */
    override def iterator: Iterator [A] = resarr.iterator.drop (1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the reverse of this priority queue. The new priority queue has
     *  the same elements as the original, but the opposite ordering.
     *  For example, the element with the highest priority in `pq` has the lowest
     *  priority in `pq.reverse`, and vice versa.
     *  Ties are handled arbitrarily.  Elements with equal priority may or
     *  may not be reversed with respect to each other.
     *----------------------------------------------------------------------------
     *  Copy the existing data into the new array backwards; this won't put it exactly
     *  into the correct order, but will require less fixing than copying it in
     *  the original order
     *  @return  the reversed priority queue.
     */
    def reverse: PriorityQueueFW [A] =
//      val revq = new PriorityQueueFW [A] ()(using ord.reverse)
        val revq = new PriorityQueueFW [A] (cap)(using ord.reverse)
        val n = resarr.p_size0
//      revq.resarr.p_ensureSize (n)
        revq.resarr.p_size0 = n
        val from = resarr.p_array
        val to = revq.resarr.p_array
        for i <- 1 until n do to(i) = from(n-i)
        revq.heapify (1)
        revq
    end reverse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns an iterator which yields all the elements in the reverse order
     *  than that returned by the method `iterator`.
     *  Note: The order of elements returned is undefined.
     *  @return  an iterator over all elements sorted in descending order.
     */
    def reverseIterator: Iterator [A] = new AbstractIterator [A] {
        private var i = resarr.p_size0 - 1
        def hasNext: Boolean = i >= 1
        def next (): A =
            val n = resarr.p_array(i)
            i -= 1
            toA (n)
        end next
    } // reverseIterator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns a regular queue containing the same elements.
     *  Note: the order of elements is undefined.
     */
    def toQueue: Queue [A] = new Queue [A] ++= this.iterator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns a textual representation of a queue as a string.
     *  @return the string representation of this queue.
     */
    override def toString () = toList.mkString ("PriorityQueueFW(", ", ", ")")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Converts this $coll to a list.
     *  Note: the order of elements is undefined.
     *  @return a list containing all elements of this $coll.
     */
    override def toList: immutable.List [A] = immutable.List.from (this.iterator)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clone this priority queue.
     *  @return  a priority queue with the same elements.
     */
    override def clone (): PriorityQueueFW [A] =
        val pq = new PriorityQueueFW [A] (cap)
        val n  = resarr.p_size0
//      pq.resarr.p_ensureSize (n)
        java.lang.System.arraycopy (resarr.p_array, 1, pq.resarr.p_array, 1, n-1)
        pq.resarr.p_size0 = n
        pq
    end clone

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Copy the elements in this priority's heap to an array.
     *  @param xs     the array to copy the elements in this priority's heap to
     *  @param start  the index to start copy
     *  @param len    the number of elements to copy
     */
    override def copyToArray [B >: A] (xs: Array [B], start: Int, len: Int): Int =
//      val copied = IterableOnce.elemsToCopyToArray (length, xs.length, start, len)
        val copied = length                // FIXED - protection would allow access
        if copied > 0 then Array.copy (resarr.p_array, 1, xs, start, copied)
        copied
    end copyToArray

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    @deprecated("Use `PriorityQueueFW` instead", "2.13.0")
    def orderedCompanion: PriorityQueueFW.type = PriorityQueueFW

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use Defaultn Serialization Proxy for writing in place.
     */
    protected [PriorityQueueFW] def writeReplace (): AnyRef =
        new DefaultSerializationProxy (PriorityQueueFW.evidenceIterableFactory [A], this)
    end writeReplace

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the name of this class.
     */
    override protected [PriorityQueueFW] def className = "PriorityQueueFW"

end PriorityQueueFW


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PriorityQueueFW` object provides methods for creating priority queues.
 */
@SerialVersionUID(3L)
object PriorityQueueFW extends SortedIterableFactory [PriorityQueueFW]:

    private var _cap = 5                                      // the current queue capacity

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the capacity cap.
     */
    def cap: Int = _cap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the capacity cap.
     *  @param c  the new capacity
     */
    def cap_= (c: Int): Unit = _cap = c

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a priority queue builder.
     */
    def newBuilder [A : Ordering]: Builder [A, PriorityQueueFW [A]] =
        new Builder [A, PriorityQueueFW [A]] {
            val pq = new PriorityQueueFW [A] (_cap)
//          def addOne (elem: A): this.type = { pq.unsafeAdd (elem); this }
            def addOne (elem: A): this.type = { pq.addOne (elem); this }
            def result (): PriorityQueueFW [A] = { pq.heapify (1); pq }
            def clear (): Unit = pq.clear () }
    end newBuilder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an empty priority queue.
     */
    def empty [A : Ordering]: PriorityQueueFW [A] = new PriorityQueueFW [A] (_cap)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a priority queue from an iterable (once) collection.
     *  @param it   the iterable (once) collection
     */
    def from [E : Ordering] (it: IterableOnce [E]): PriorityQueueFW [E] =
        val b = newBuilder [E]
        b ++= it
        b.result ()
    end from

end PriorityQueueFW


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `priorityQueueFWTest` main function tests the `PriorityQueueFW` class.
 *  > runMain scalation.priorityQueueFWTest
 */
@main def priorityQueueFWTest (): Unit =

    given ord: Ordering [Double] = summon [Ordering [Double]].reverse

    banner ("Test PriorityQueueFW")
    val cap = 9                                            // position 0 used by data structure => cap = 8 elements + 1
    val pq  = new PriorityQueueFW [Double] (cap)
    for i <- 1 to 10 do
        pq.enqueue (i * i)
        println (pq) 
    println ("Insert 20")
    pq.enqueue (20)
    println (pq) 
    while ! pq.isEmpty do println (pq.dequeue ())

end priorityQueueFWTest

