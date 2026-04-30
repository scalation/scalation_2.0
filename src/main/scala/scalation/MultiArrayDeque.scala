
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Korede Binji
 *  @version 2.0
 *  @date    Wed Jun 21 22:39:42 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Data Structure: Multiple ArrayDeques
 */

package scalation

import scala.collection.mutable.ArrayDeque
import scala.math.round

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MultiArrayDeques` class provides a data structure for storing multiple
 *  `ArrayDeque`s for the purpose of maintaining multiple parallel lanes.
 *  For example cars on a multi-lane road can be positioned in this data structure.
 *  It allows cars to added and removed and supports finding cars at a similar
 *  distance (from the start) to be found in another lane.
 *  @param nLanes  the number of lanes
 */
class MultiArrayDeques [A] (nLanes: Int):

    private val lane = Array.fill (nLanes)(ArrayDeque [A] ())              // allocated n lanes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the l-th lane.
     *  @param l  the l-th lane
     */
    def apply (l: Int): ArrayDeque [A] = lane(l)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th element in lane l.
     *  @param l  the l-th lane
     *  @param i  the index position in lane l
     */
    def apply (l: Int, i: Int): A = lane(l)(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an new element to the beginning of lane l.
     *  @param l     the l-th lane
     *  @param elem  the element to be added
     */
    def add (l: Int, elem: A): Unit = lane(l) += elem

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert a new element AT index position i into lane l.
     *  @param l     the l-th lane
     *  @param elem  the element to be inserted
     *  @param i     the index position in lane l
     */
    def insertAt (l: Int, elem: A, i: Int): Unit = lane(l).insert (i, elem)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the element from the end of lane l.
     *  @param l  the l-th lane
     */
    def remove (l: Int): A = lane(l).removeHead ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the element AT index position i from lane l.
     *  @param l     the l-th lane
     *  @param i  the index position in lane l
     */
    def removeAt (l: Int, i: Int): A = lane(l).remove (i)

end MultiArrayDeques


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `multiArrayDequesTest` main function tests the `MultiArrayDeques` class.
 *  > runMain scalation.multiArrayDequesTest	
 */
@main def multiArrayDequesTest (): Unit =

    case class Car (id: Int, ln: Int, dist: Double)

    val road = MultiArrayDeques [Car] (2)

    val ncars = Array (20, 10)
    val ratio = ncars(1) / ncars(0).toDouble

    var id = 0
    for l <- 0 until 2 do
        for i <- 0 until ncars(l) do
            id += 1
            road.add (l, Car (id, l, (l+1).toDouble * (ncars(l) - i)))

    for l <- 0 until 2 do
        println (s"lane $l = ${road(l)}")

    // Car in lane 0, find car in lane 1 at similar distance
    val car1  = road (0, 15)
    println (s"car1 = $car1")
    val guess = round (15 * ratio).toInt
    println (s"guess for car = $guess")
    val car2  = road (1, guess)
    println (s"car2 = $car2")

    // Car in lane 0, find car in front in the same lane

    // Car in lane 0, find car in back in the same lane

end multiArrayDequesTest

