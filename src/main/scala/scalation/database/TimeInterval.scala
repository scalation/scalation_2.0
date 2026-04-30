
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 27 18:19:54 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    `TimeInterval` implementing Allen's Interval Algebra
 *
 *  @see https://cse.unl.edu/~choueiry/Documents/Allen-CACM1983.pdf
 *  @see https://www.sciencedirect.com/topics/computer-science/interval-algebra
 *  @see https://www.uni-bamberg.de/fileadmin/temp-ba7mm3/SparQ/SparQ-Manual.pdf
 */

package scalation
package database

import scala.collection.mutable.{ArrayBuffer => Bag}

import TimeNum.{min, max}
import TimeOfWeek.fromTimeNum

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeInterval` companion object provides factory functions for creating
 *  time-intervals.
 */
object TimeInterval:

    private val debug = debugf ("TimeInterval", true)                     // debug function
    private val flaw  = flawf ("TimeInterval")                            // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a time-interval from the given string specification.
     *  e.g., "T 03:55 pm-05:10 pm"
     *  @param ti  the time-interval as a string
     */
    def apply (ti: String): TimeInterval =
        val token = ti.split ("-")
        if token.size != 2 then
            flaw ("apply", s"must have two times to make a time-interval")
            null
        else
            val tw1 = TimeOfWeek (token(0))
            val tw2 = TimeOfWeek (s"${token(0)(0)} ${token(1)}")
            debug ("apply", s"create TimeInterval ($tw1, $tw2)")
            TimeInterval (tw1, tw2)
        end if
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a time-interval from two time-of-week objects.
     *  @param tw1  the first time-of-week
     *  @param tw2  the second time-of-week
     */
    def apply (tw1: TimeOfWeek, tw2: TimeOfWeek): TimeInterval =
        TimeInterval (tw1.toTimeNum (), tw2.toTimeNum ())
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a multiple time-interval array from the given string specification.
     *  e.g., "T R 03:55 pm-05:10 pm"
     *  @param mti  multiple time-intervals as a string
     */
    def multiDay (mti: String): Array [TimeInterval] =
        val token = mti.split ("-")
        if token.size != 2 then
            flaw ("multiDay", s"must have two times to make a time-interval")
            null
        else
            val part  = token(0).split (" ")
            val np    = part.size
            val last2 = s"${part(np-2)} ${part(np-1)}" 
            val tia   = Array.ofDim [TimeInterval] (np-2)
           
            for i <- tia.indices do
                val day = part(i)
                if ! TimeOfWeek.isDay (day) then flaw ("multiDay", "expected a day, but found $day")
                val tw1 = TimeOfWeek (s"$day $last2")
                val tw2 = TimeOfWeek (s"$day ${token(1)}")
                debug ("multiDay", s"create TimeInterval ($tw1, $tw2)")
                tia(i) = TimeInterval (tw1, tw2)
            end for
            tia
        end if
    end multiDay

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return an array of time-intervals.
     *  @param tis  the var-arg of time-interval specification strings
     */
    def makeIntervals (tis: String*): Array [TimeInterval] =
        (for ti <- tis yield apply (ti)).toArray
    end makeIntervals

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the time-intervals in the array of time-intervals tia
     *  @param name  the name associated with the time-intervals.
     *  @param tia   the array of time-intervals
     */
    def showIntervals (name: String, tia: Array [TimeInterval]): Unit =
        for ti <- tia do println (s"$name: ${ti.format}")
    end showIntervals

end TimeInterval


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeInterval `class implements Allen's Interval Algebra (first 13).
 *      Basic:    equals
 *      Main:     before, meets, overlaps,  during,   starts,  finishes
 *      Converse: after,  metBy, overlapBy, contains, startBy, finishBy
 *      Merge:    merge,  mergeGap, conflict
 *  @param t1  the time of the start of the interval (inclusive)
 *  @param t2  the time of the end of the interval (inclusive)
 */
case class TimeInterval (t1: TimeNum, t2: TimeNum):

    private val flaw = flawf ("TimeInterval")                             // flaw function

    if t1 > t2 then flaw ("init", s"start time t1 = $t1 cannot be larger than end time t2= $t2")

    // Basic

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this time-interval is the same as time-interval y.
     *  @param y  the other time-interval
     */
    infix def equals (y: TimeInterval): Boolean = t1 == y.t1 && t2 == y.t2                // eq
    inline def == (y: TimeInterval): Boolean = this equals y

    // Main

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this time-interval is entirely before time-interval y.
     *  @param y  the other time-interval
     */
    infix def before   (y: TimeInterval): Boolean = t2 < y.t1                             // b
    inline infix def < (y: TimeInterval): Boolean = this before y
    infix def meets    (y: TimeInterval): Boolean = t2 == y.t1                            // m
    infix def overlaps (y: TimeInterval): Boolean = t1 < y.t1 && y.t1 < t2 && t2 < y.t2   // o
    infix def during   (y: TimeInterval): Boolean = y.t1 < t1 && t2 < y.t2                // d
    infix def starts   (y: TimeInterval): Boolean = t1 == y.t1 && t2 < y.t2               // s
    infix def finishes (y: TimeInterval): Boolean = t2 == y.t2 && y.t1 < t1               // f

    // Converse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this time-interval is entirely after time-interval y.
     *  @param y  the other time-interval
     */
    inline infix def after     (y: TimeInterval): Boolean = y before this                 // bi
    inline infix def >         (y: TimeInterval): Boolean = this after y
    inline infix def metBy     (y: TimeInterval): Boolean = y meets this                  // mi
    inline infix def overlapBy (y: TimeInterval): Boolean = y overlaps this               // oi
    inline infix def contains  (y: TimeInterval): Boolean = y during this                 // di
    inline infix def startBy   (y: TimeInterval): Boolean = y starts this                 // si
    inline infix def finishBy  (y: TimeInterval): Boolean = y finishes this               // fi

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** If time-intervals this and y are disjoint, return null; otherwise return
     *  a merged time-interval.
     *  @param y  the other time-interval
     */
    def merge (y: TimeInterval): TimeInterval =
        if before (y) || after (y) then null
        else TimeInterval (min (t1, y.t1), max (t2, y.t2))
    end merge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** If time-intervals this and y are disjoint beyond the gap, return null;
     *  otherwise return a merged time-interval.
     *  @param y    the other time-interval
     *  @param gap  the allowable time gap (e.g., 15 minutes gap between student classes)
     */
    def mergeGap (y: TimeInterval, gap: Long = 15 * TimeNum.MINUTE): TimeInterval =
        if t2 + gap < y.t1 || y.t2 + gap < t1 then null
        else TimeInterval (min (t1, y.t1), max (t2, y.t2))
    end mergeGap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether time-interval this and time-interval y have a time conflict.
     *  @param y  the other time-interval
     */
    infix def conflict (y: TimeInterval): Boolean = ! (before (y) || after (y))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this time-interval to a range of time-number (date-time) strings.
     */
    override def toString: String = s"[ $t1, $t2 ]"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this time-interval to a range of time-of-week strings.
     */
    def format: String = s"[ ${fromTimeNum (t1).format ()}, ${fromTimeNum (t2).format ()} ]" 

end TimeInterval


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The timeIntervalTest main function is used to test the `TimeInterval` class.
 *  Tests the Basic and Main operators:
 *      Basic: equal
 *      Main:  before, meets, overlaps, starts, during, finishes
 *  > runMain scalation.database.timeIntervalTest
 */
@main def timeIntervalTest (): Unit =

    banner ("timeIntervalTest")
    val t = for i <- 0 until 10 yield TimeNum.fromInt (i)
    println (s"t = $t")

    val a = TimeInterval (t(0), t(3))                                     // [0, 3]
    val b = TimeInterval (t(2), t(5))                                     // [2, 5]
    val c = TimeInterval (t(5), t(7))                                     // [5, 7]
    val d = TimeInterval (t(6), t(9))                                     // [6, 9]
    val e = TimeInterval (t(7), t(8))                                     // [7, 8]
    val f = TimeInterval (t(7), t(9))                                     // [7, 9]
    val g = TimeInterval (t(0), t(3))                                     // [0, 3]

    banner ("Test Basic Operators:")

    println (s"a equals b \t = ${a equals b}")
    println (s"a equals g \t = ${a equals g}")

    println (s"a == b \t = ${a == b}")
    println (s"a == g \t = ${a == g}")

    banner ("Test Main Operators:")

    println (s"a before b \t = ${a before b}")
    println (s"a before c \t = ${a before c}")

    println (s"a < b \t = ${a < b}")
    println (s"a < c \t = ${a < c}")

    println (s"a meets b \t = ${a meets b}")
    println (s"b meets c \t = ${b meets c}")

    println (s"a overlaps c \t = ${a overlaps c}")
    println (s"a overlaps b \t = ${a overlaps b}")

    println (s"e during c \t = ${e during c}")
    println (s"e during d \t = ${e during d}")

    println (s"e starts d \t = ${e starts d}")
    println (s"e starts f \t = ${e starts f}")

    println (s"f finishes e \t = ${f finishes e}")
    println (s"f finishes d \t = ${f finishes d}")

end timeIntervalTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The timeIntervalTest2 main function is used to test the `TimeInterval` class.
 *  Tests the Converse Merge operators.
 *      Convere: after, metBy, overlapBy, contains, startBy, finishBy
 *      Merge:   merge, mergeGap
 *  > runMain scalation.database.timeIntervalTest2
 */
@main def timeIntervalTest2 (): Unit =

    banner ("timeIntervalTest2")
    val t = for i <- 0 until 10 yield TimeNum.fromInt (i)
    println (s"t = $t")

    val a = TimeInterval (t(0), t(3))                                     // [0, 3]
    val b = TimeInterval (t(2), t(5))                                     // [2, 5]
    val c = TimeInterval (t(5), t(7))                                     // [5, 7]
    val d = TimeInterval (t(6), t(9))                                     // [6, 9]
    val e = TimeInterval (t(7), t(8))                                     // [7, 8]
    val f = TimeInterval (t(7), t(9))                                     // [7, 9]
//  val g = TimeInterval (t(0), t(3))                                     // [0, 3]

    banner ("Test Converse Operators:")

    println (s"b after a \t = ${b after a}")
    println (s"c after a \t = ${c after a}")

    println (s"b > a \t = ${b > a}")
    println (s"c > a \t = ${c > a}")

    println (s"b metBy a \t = ${b metBy a}")
    println (s"c metBy b \t = ${c metBy b}")

    println (s"c overlapBy a \t = ${c overlapBy a}")
    println (s"b overlapBy a \t = ${b overlapBy a}")

    println (s"c contains e \t = ${c contains e}")
    println (s"d contains e \t = ${d contains e}")

    println (s"d startBy e \t = ${d startBy e}")
    println (s"f startBy e \t = ${f startBy e}")

    println (s"e finishBy f \t = ${e finishBy f}")
    println (s"d finishBy f \t = ${d finishBy f}")

    banner ("Test Merge Operators:")

    println (s"a.merge (c) \t = ${a.merge (c)}")
    println (s"a.merge (b) \t = ${a.merge (b)}")

    println (s"a.mergeGap (c) \t = ${a.mergeGap (c)}")
    println (s"a.mergeGap (b) \t = ${a.mergeGap (b)}")

end timeIntervalTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The timeIntervalTest3 main function is used to test the `TimeInterval` class.
 *  Tests for a time conflict between a student's class schedule and assigned lab.
 *  @see `scalation.database.table.TA_AssignmentDB`
 *  > runMain scalation.database.timeIntervalTest3
 */
@main def timeIntervalTest3 (): Unit =

    import TimeInterval.{makeIntervals, showIntervals, multiDay}

    banner ("timeIntervalTest3")

    // Determine a merged time schedule for a student taking 

    val schedule = Array (
        Array (52909, 8370, "T R 03:55 pm-05:10 pm", "W 04:10 pm-05:00 pm", "Miller",    30, 0, 30),
        Array (52918, 8790, "T R 12:45 pm-02:00 pm", "W 12:40 pm-01:30 pm", "Ramaswamy", 30, 0, 30),
        Array (52919, 8850, "W 11:30 am-12:20 pm", "T R 11:10 am-12:25 pm", "T. Liu",    45, 0, 45))

    val labs = Array (
        Array (19917, 13011, "M W 03:00 pm-03:50 pm", "", "LaMarca", 30, 2, 28),
        Array (19921, 13011, "T R 12:45 pm-02:00 pm", "", "LaMarca", 30, 0, 30),
        Array (19928, 13011, "T R 03:55 pm-05:10 pm", "", "LaMarca", 30, 1, 29),
        Array (19935, 13011, "T R 05:30 pm-06:45 pm", "", "LaMarca", 30, 9, 21))

    banner ("8370 Periods")
    val _8370 = makeIntervals ("T 03:55 pm-05:10 pm", "R 03:55 pm-05:10 pm", "W 04:10 pm-05:00 pm")
    showIntervals ("8370", _8370)

    banner ("8790 Periods")
    val _8790 = makeIntervals ("T 12:45 pm-02:00 pm", "R 12:45 pm-02:00 pm", "W 12:40 pm-01:30 pm")
    showIntervals ("8790", _8790)

    banner ("8850 Periods")
    val _8850 = makeIntervals ("T 11:10 am-12:25 pm", "R 11:10 am-12:25 pm", "W 11:30 am-12:20 pm")
    showIntervals ("8850", _8850)

    banner ("Test multiDay")
    val period = multiDay ("T R 03:55 pm-05:10 pm")
    showIntervals ("period", period)

    banner ("All Periods in Class Schedule")
    val periods = Bag [TimeInterval] ()
    for cls <- schedule do
        val (per1, per2) = (cls(2).asInstanceOf [String], cls(3).asInstanceOf [String]) 
        periods ++= multiDay (per1)
        if per2 != "" then periods ++= multiDay (per2)
    end for
    showIntervals (s"schedule", periods.toArray)

    banner ("All Periods for Assigned Labs")
    val lperiods = Bag [TimeInterval] ()
    for lab <- labs do
        val (per1, per2) = (lab(2).asInstanceOf [String], lab(3).asInstanceOf [String]) 
        lperiods ++= multiDay (per1)
        if per2 != "" then lperiods ++= multiDay (per2)
    end for
    showIntervals (s"labs", lperiods.toArray)
    
    banner ("Check for Time Conflicts")
    for lab <- lperiods do
        for cls <- periods do
            if lab conflict cls then
                println (s"Found time conflict between lab ${lab.format} and cls ${cls.format}")
    end for

end timeIntervalTest3

