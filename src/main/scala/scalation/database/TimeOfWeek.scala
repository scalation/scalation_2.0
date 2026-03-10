
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 27 18:19:54 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note   `TimeOfWeek` Day and Time within a Week (Relative, not Absolute Time)
 *
 *  @see  https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/time/LocalDateTime.html
 *  @see  https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/time/format/DateTimeFormatter.html
 *
 *  Abbreviations for Days of the Week:
 *
 *  1.  M       - Monday
 *  2.  T or TU - Tuesday
 *  3.  W       - Wednesday
 *  4.  R or TH - Thursday
 *  5.  F       - Friday
 *  6.  S or SA - Saturday
 *  7.  U or SU - Sunday
 */

package scalation
package database

//import java.time.LocalDateTime
//import java.time.format.{DateTimeFormatter, FormatStyle}
import java.time.temporal.ChronoField

//import scala.runtime.ScalaRunTime.stringOf

import TimeNum.{DAY, HOUR, MINUTE}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeOfWeek` companion object provide factory functions for creating
 *  instances of TimeOfWeek.
 */
object TimeOfWeek:

    private val debug = debugf ("TimeOfWeek", true)                   // debug function
    private val flaw  = flawf ("TimeOfWeek")                          // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an instance of `TimeOfWeek` from a formatted string specification.
     *  @param dt  the day-time as a string
     */
    def apply (dt: String): TimeOfWeek =

        val isp = dt.indexOf (" ")
        val (dayStr, timeStr) = dt.splitAt (isp)

        val day = dayStr(0) match
        case 'M' => 1
        case 'T' => if dayStr.size > 1 && dayStr(1).toUpper == 'H' then 4 else 2
        case 'W' => 3
        case 'R' => 4
        case 'F' => 5
        case 'S' => if dayStr.size > 1 && dayStr(1).toUpper == 'U' then 7 else 6
        case 'U' => 7
        case _   => flaw ("apply", "unknown day = $day"); 0

//      val formatter = DateTimeFormatter.ofLocalizedTime (FormatStyle.SHORT)
//      val time = LocalDateTime.parse (timeStr.trim, formatter)

        val time = parseTime (timeStr.trim)

        debug ("apply", s"TimeOfWeek (day = $day, time = $time)")
        TimeOfWeek (day, time)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse the time-string and return the total number of seconds into the day.
     *  @param ts  the time-string in hh:mm [ :ss ] [ | AM | PM ]
     */
    def parseTime (ts: String): Long =
        val isp = ts.indexOf (" ")
        val (time, pm) = ts.splitAt (isp)
        val token = time.split (":")
//      println (s"token = ${stringOf (token)}, pm = $pm")

        var hrs = token(0).toInt
        if ! isPM (pm) && hrs == 12 then hrs = 0
        var sec = hrs * HOUR + token(1).toInt * MINUTE
        if token.size == 3 then sec += token(2).toInt
        if isPM (pm) && hrs != 12 then sec += 12L * HOUR 
        sec
    end parseTime

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this time is AM (false) or PM (true).
     *  @param ts  the time-string in hh:mm:ss [ | AM | PM ]
     */
    def isPM (ts: String): Boolean = ts.toUpperCase `contains` "PM"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the string specification of a day is a correct abbreviation.
     *  @param dayStr  the day specification as a string
     */
    def isDay (dayStr: String): Boolean = 
        dayStr(0) match
        case 'M' | 'W' | 'R' | 'F' | 'U' => true
        case 'T' => dayStr.size == 1 || dayStr(1).toUpper == 'H'      // T, Th, TH
        case 'S' => dayStr.size == 1 || dayStr(1).toUpper == 'U'      // S, Su, SU
        case _   => false
    end isDay

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** From a time-number get the time-of-week, i.e, day and seconds into that day.
     *  @see https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/time/temporal/ChronoField.html
     *  @param dt  the date-time as a time-number
     */
    def fromTimeNum (dt: TimeNum): TimeOfWeek =
        val day = dt.getChrono (ChronoField.DAY_OF_WEEK).toInt
        val sec = dt.getChrono (ChronoField.SECOND_OF_DAY)
        TimeOfWeek (day, sec)
    end fromTimeNum

end TimeOfWeek


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeOfWeek` class keeps track relative date-time within a week,
 *  e.g., for a weekly schedule.
 *  @param day   the day of the week (Mon -> 1, ... Sun -> 7)
 *  @param time  the time of day in seconds (0 to 86399)
 */
case class TimeOfWeek (day: Int, second: Long):

    val dayArr = Array ('?', 'M', 'T', 'W', 'R', 'F', 'S', 'U')        // abbreviations for days

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this time-of-week specification to a `TimeNum`.
     *  It uses the number of seconds since start of the relative week.
     *  @param baseDate  the absolute date for start of the relative week
     *                   if absolute date is unimportant, may pass firstMonday
     */
    def toTimeNum (baseDate: TimeNum = TimeNum.firstMonday): TimeNum =
        baseDate + TimeNum ((day - 1) * DAY + second)
    end toTimeNum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Format this time-of-week in "E HH:mm:ss", E is day-first letter, HH is hour
     *  (2-digit, 24 hour), mm is minute (2-digit), and ss is second (2-digit).
     *  @see https://docs.oracle.com/en/java/javase/17/docs/api/java.base/java/time/format/DateTimeFormatter.html
     *  @param withSec  flag indicating whether to include seconds or stop at minutes
     */
    def format (withSec: Boolean = false): String =
        val E  = dayArr(day)
        val HH = second / HOUR
        val hs = second % HOUR
        val mm = hs / MINUTE
        if withSec then
            val ss = hs % MINUTE
//          s"$E $HH:$mm:$ss"
            f"$E%s $HH%02d:$mm%02d:$ss%02d"
        else
            f"$E%s $HH%02d:$mm%02d"
    end format

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this time-of-week object to a string
     */
    override def toString: String = s"day = $day, second = $second"

end TimeOfWeek


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `timeOfWeekTest` main function is use to test the `TimeOfWeek` class.
 *  > runMain scalation.database.timeOfWeekTest
 */
@main def timeOfWeekTest (): Unit =

    banner ("timeOfWeekTest")

    val dt  = "T 03:55 pm"
    val dt2 = "T 03:55:00 PM"

    banner  (s"dt = $dt")
    val tw = TimeOfWeek (dt)
    println (s"tw           = $tw")
    println (s"tw.format () = ${tw.format ()}")

    banner  (s"dt2 = $dt2")
    val tw2 = TimeOfWeek (dt2)
    println (s"tw2           = $tw2")
    println (s"tw2.format () = ${tw2.format ()}")

end timeOfWeekTest

