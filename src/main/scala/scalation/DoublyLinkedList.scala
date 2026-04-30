
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Korede Bishi
 *  @version 2.0
 *  @date    Sun Feb 25 20:55:28 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Data Structure: Doubly Linked List with head and tail References
 *           suitable for implementing queues supporting removal of any element
 */

package scalation

import scala.collection.mutable.{AbstractIterable, ListBuffer}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DoublyLinkedList` class provides a data structure implementing mutable doubly-linked lists.
 *  Imagine a line of elements/cars moving left to right in a list/lane:
 *    remove head/lead car when it reaches the end of the lane
 *    add tail/last car when it reaches the beginning of the lane
 *
 *      ahead                   -->    -->
 *      tail (last car) --> [c3]   [c2]   [c1] <-- head (lead car)
 *      behind                  <--    <--
 *
 *  @tparam A  the type of the elements/values in the list
 */
class DoublyLinkedList [A]
      extends AbstractIterable [A]
         with Serializable:

    private val debug = debugf ("DoublyLinkedList", true)           // debug  function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Node` inner case class wraps elements in nodes for double linkage.
     *  @param elem    the element in this node (you)
     *  @param ahead   the node ahead of you (e.g., the car ahead)
     *  @param behind  the node behind you   (e.g., the car behind)
     */
    case class Node (elem: A, var ahead: Node, var behind: Node):

        override def toString: String = s"Node ($elem)"

    end Node

    private var head_ : Node = null                                 // head node (lead car)
    private var tail_ : Node = null                                 // tail node (last car)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `NodeIterator` inner class supports iterating over all the nodes in this list,
     *  moving foreward in list/lane (tail to head).
     *  @param ns  the starting node (defaults to tail)
     */
    class NodeIterator (ns: Node = tail_) extends Iterator [Node]:
        var n = ns                                                  // current node (positioned in list)
        def hasNext: Boolean = n != null
        def next (): Node = { val cur = n; n = n.ahead; cur }       // move forward towards the front of list/lane
    end NodeIterator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an iterator for retrieving all the nodes in this list.
     *  @see scala.collection.IterableOnce
     */
    def nodeIterator: Iterator [Node] = new NodeIterator ()

    inline def getAhead (n: Node): Node = n.ahead

    inline def getBehind (n: Node ): Node = n.behind

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `ListIterator` inner class supports iterating over all the elements in this list,
     *  moving foreward in list/lane (tail to head).
     *  @param ns  the starting node (defaults to tail)
     */
    class ListIterator (ns: Node = tail_) extends Iterator [A]:
        var n = ns                                                  // current node (positioned in list)
        def hasNext: Boolean = n != null
        def next (): A = { val cur = n; n = n.ahead; cur.elem }     // move forward towards the front of list/lane
    end ListIterator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an iterator for retrieving all the elements in this list.
     *  @see scala.collection.IterableOnce
     */
    def iterator: Iterator [A] = new ListIterator ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the element in node n (e.g., the current car).
     *  @param n  the node containing the sought element
     */
    inline def elemAt (n: Node): A = n.elem

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the lead/first node in the list (e.g, node holding the lead car).
     */
    inline override def head: A = head_.elem

    inline def headNode: Node = head_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the trail/last node in the list (e.g, node holding the trail car).
     */
    inline override def last: A = tail_.elem

    inline def lastNode: Node = tail_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the list is empty (head and tail are null).
     */
    inline override def isEmpty: Boolean = head_ == null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::fixed
    /** Add the first element (a lead car) to a list and return the new node n.
     *  @param elm  the element to be added
     *  @return the new node be added
     */
    def addFirst (elm: A): Node =
        val n = Node (elm, null, head_)                             // new node has nothing ahead, and its behind is the current head
        if head_ != null then                                       // if list is not empty
            head_.ahead = n                                         // update the head's ahead to point to the new node
        head_ = n                                                   // update head to point to the new node
        if tail_ == null then                                       // if the list was empty (tail is null)
            tail_ = n                                               // set tail to the new node
        debug ("addFirst", s"added node $n as the first element in list")
        n
    end addFirst

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a new element into the list BEFORE/behind the given node `nn` and return
     *  the new node `n`.
     *  Relink:  bn <-> nn  TO  bn <-> n <-> nn
     *  @param elm  the new element to be added
     *  @param nn   the given node (defaults to tail if not given)
     *  @return the new node `n`
     */
    def add (elm: A, nn: Node = tail_): Node =
        if isEmpty || nn == null then
            addFirst (elm)                                          // case 1: List is empty or no reference node
        else
            val bn = nn.behind                                      // bn references the node behind nn
            val n  = Node (elm, nn, bn)                             // new node is inserted with nn ahead and bn behind

            if bn != null then bn.ahead = n                         // fix ahead linkage of bn (behind) node
            nn.behind = n                                           // fix behind linkage of the nn (given) node

            if nn == tail_ then tail_ = n                           // update tail if inserting at the end

            debug ("add", s"[bn = $bn] <-> [n = $n] <-> [nn = $nn]")
            n
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a new element BEFORE the given successor node `nn` and return the new node `n`.
     *  Relink:  pn <-> nn  TO  pn <-> n <-> nn
     *  The predecessor (`pn`) of the successor node `nn` is relinked to point to the new node `n`.
     *  Similarly, the new node `n` links back to `pn` and forward to `nn`. If `nn` is `null`,
     *  this method adds the element as the first element in the list.
     *  @param elm the element to be added
     *  @param nn  the successor node (defaults to `null` if not provided)
     *  @return the newly created node `n` inserted before node `nn`
     *
    def addBefore (elm: A, pn: Node): Node =
        val nn = pn.behind // Get the behind node (car behind `pn`)

        if nn == null then
            // Case 1: `pn` is the head, so insert BEHIND it and assume tail.
            val n = Node(elm, pn, null) // New node's ahead = pn, behind = null
            pn.behind = n // Fix behind linkage
            tail_ = n // Update the tail pointer
            debug("addBefore", s"Inserted node $n behind head $pn (new tail)")
            n
        else
            // Case 2: `pn` has a behind node (normal case, inserting between two nodes)
            val n = Node(elm, pn, nn) // Insert between `pn` (ahead) and `nn` (behind)
            pn.behind = n // Fix pn's behind pointer
            nn.ahead = n // Fix nn's ahead pointer
            debug("addBefore", s" pn= $pn n=$n and $nn")
            n
    end addBefore
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the node `n` from the linked list.
     *  Relink:  bn <-> n <-> an  TO  bn <-> an
     *  @param n  the given node to remove (unlink)
     */
    def remove (n: Node = head_): Unit =
        val an = n.ahead                                            // an = the node/car AHEAD of node n
        val bn = n.behind                                           // bn = the node/car BEHIND node n

        if an != null then an.behind = bn                           // set an's ref: bn <- an
        if bn != null then bn.ahead  = an                           // set nn's ref: bn -> an

        if n == head_ then head_ = bn                               // if n was head, reset to bn
        if n == tail_ then tail_ = an                               // if n was tail, reset to an

        n.ahead  = null                                             // n no longer links
        n.behind = null
        debug ("remove", s"[bn = $bn] <-> [an = $an]")
    end remove

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clear the list of all nodes (and their elements) by setting head_ and tail_
     *  to null, so CG can reclaim the unreferenced nodes.
     */
    def clear (): Unit = { tail_ = null; head_ = null }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this doubly linked list to a string (tail to head).
     */
    override def toString: String =
        val sb = StringBuilder ("DoublyLinkedList (tail -")
        for n <- nodeIterator do sb.append (s"> [ $n ] <-")
        sb.append (" head)").mkString
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the elements of this doubly linked list to a Scala List.
     *  This method is efficient in terms of maintaining the correct order without
     *  needing a separate reverse at the end.
     */
    override def toList: List [A] =
        val buf = ListBuffer [A] ()                                 // use ListBuffer for efficient appends
        for n <- nodeIterator do buf += n.elem                      // traverse using the predefined nodeIterator
        buf.toList                                                  // convert ListBuffer to List
    end toList

end DoublyLinkedList


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `doublyLinkedListTest` main function tests the `DoublyLinkedList` class.
 *  > runMain scalation.doublyLinkedListTest
 */
@main def doublyLinkedListTest (): Unit =

    banner ("Test the addFirst and add methods")
    val dll = DoublyLinkedList [Int] ()
    for i <- 0 until 10 do
        if dll.isEmpty then dll.addFirst (i)
        else dll.add (i)

    banner ("Test the toString method")
    println (s"dll = $dll")

    banner ("Test the remove method")
    while ! dll.isEmpty do
        dll.remove ()
        println (s"dll = $dll")

end doublyLinkedListTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `doublyLinkedListTest` main function tests the `DoublyLinkedList` class.
 *  > runMain scalation.doublyLinkedListTest2
 */
@main def doublyLinkedListTest2 (): Unit =

    banner ("Test the add method")
    val dll = DoublyLinkedList [Int] ()
    for i <- 0 until 10 do dll.add(i)
    val n = dll.headNode

    println (s"n the head node is:  $n")
    println (s"the node behind n is:   ${dll.getBehind (n)}")
    println (s"the node ahead of n is: ${dll.getAhead (n)}")
    println (s"dll = $dll")

    banner ("Test the remove method")
    while ! dll.isEmpty do
        dll.remove ()
        println (dll)

end doublyLinkedListTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `doublyLinkedListTest` main function tests the `DoublyLinkedList` class.
 *  > runMain scalation.doublyLinkedListTest3
 *
@main def doublyLinkedListTest3 (): Unit =

    banner("Test the add and addBefore methods")

    // Create DoublyLinkedList
    val dll = DoublyLinkedList[Int]()

    dll.add(98) // DLL now contains [ 98 ] (single node)
    val head = dll.headNode
    dll.addBefore(99, head) // Insert 99 before Node(98)
    dll.addBefore(95, head) // Insert 99 before Node(98)
    println(dll)

@main def doublyLinkedListTest4(): Unit =
    banner("Test the add and addBefore methods")

    // Case 1: Insert normally at the tail (Default behavior)
    val dll = DoublyLinkedList[Int]()
    for i <- 0 until 5 do dll.add(i)
    println("After normal insertion:")
    println(dll)

    // Case 2: Insert before a given node (Middle of list)
    val refNode = dll.headNode.behind // Second node in the list
    println(s"Inserting before $refNode")
    dll.addBefore(99, refNode)
    println("After inserting 99 before the second node:")
    println(dll)

    // Case 3: Insert before head (Becomes new head)
    dll.addBefore(77, dll.headNode)
    println("After inserting 77 before head:")
    println(dll)

    // Case 4: Insert when list is empty (Should work with addFirst)
    val emptyDll = DoublyLinkedList[Int]()
    emptyDll.addBefore(55, null)
    println("After inserting 55 into empty list:")
    println(emptyDll)

 */

