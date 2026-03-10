
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Santosh Uttam Bobade, John Miller
 *  @version 2.0
 *  @date    Wed May 27 14:36:12 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    `TimeNum` for Handling Dates and Times
 */

package scalation

import java.time.{DateTimeException, Instant, ZonedDateTime, ZoneId, ZoneOffset}
import java.time.format.DateTimeFormatter
import java.time.temporal.{ChronoField, UnsupportedTemporalTypeException}

import scala.language.implicitConversions
import scala.math.{abs => ABS, sqrt => SQRT}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Add the <<= operator as an extension method to `java.time.Instant`.
 */
extension (s: Instant)
    def <<= (t: Instant): Boolean = s.compareTo (t) <= 0
        

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Given instance to be used with methods such as `sorted` to define the ordering
 *  of elements.
 *  FIX - should be implicit in "sorted (using timeNumOrd)" 
 */
given timeNumOrd: Ordering [TimeNum] = new Ordering [TimeNum]:
    def compare (s: TimeNum, t: TimeNum) = s compare t


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeNum` companion object is used to represent and operate on date-time values.
 *  It contains an implicit class definition for `TimeNum`, which extends
 *  `Instant` with `Numeric` operations that can be used in `VectorT`.
 *  The `java.time.Instant` class stores nano-seconds since the UNIX Epoch
 *  using a `long` for seconds and `int` for nano-seconds (12 bytes or 96 bits).
 *  The `java.time.ZonedDateTime` class is used with it to handle various
 *  date-time formats.
 */
object TimeNum:

    private val debug = debugf ("TimeNum", false)            // debug function

    /** Define the default time zone to be Coordinated Universal Time (UTC).
     *  UTC is the successor to Greenwich Mean Time (GMT) that was developed
     *  at Royal Observatory in Greenwich, UK (0° longitude).
     *  UTC does not observe Daylight Savings Time and serves as universal
     *  time across regions/time zones.
     *  @see en.wikipedia.org/wiki/Coordinated_Universal_Time
     */
    val DEFAULT_TZ = "UTC"

    /** Default date-time format
     *  month, day, year, hour (24), minute, second, nano-seconds zone
     */
    val DEFAULT_DATETIME_FORMAT = "MM/d/yyyy HH:mm:ss:SSSSSSSSS z"

    val SECOND = 1L                                          //       1 second
    val MINUTE = 60L * SECOND                                //      60 seconds
    val HOUR   = 60L * MINUTE                                //    3600 seconds
    val DAY    = 24L * HOUR                                  //  86,400 seconds
    val WEEK   = 7L  * DAY                                   // 604,800 seconds

    // MONTH and YEAR are irregular and require complex calculations

    /** TimeNum (0) as a date-time corresponding to 01-01-1970 00:00:00 UTC,
     *  the UNIX Epoch (00:00:00 UTC on Thursday, 1 January 1970)
     */
    val _0 = TimeNum (0L)                                    // 0 day, Thu, Jan 1

    /** TimeNum (1) as a date-time corresponding to 01-02-1970 00:00:00 UTC
     *  NoteL:  _1 was arbitrarily chosen as one day rather than another time unit
     */
    val _1 = TimeNum (DAY)                                   // 1 day = 1 * 86400 seconds, Fri, Jan 2
    val _2 = TimeNum (2L * DAY)                              // 2 day = 2 * 86400 seconds, Sat, Jan 3
    val _3 = TimeNum (3L * DAY)                              // 3 day = 3 * 86400 seconds, Sun, Jan 4
    val _4 = TimeNum (4L * DAY)                              // 4 day = 4 * 86400 seconds, Mon, Jan 5
    val _5 = TimeNum (5L * DAY)                              // 5 day = 5 * 86400 seconds, Tue, Jan 6
    val _6 = TimeNum (6L * DAY)                              // 6 day = 6 * 86400 seconds, Wed, Jan 7

    val firstMonday = TimeNum._4                             // first Monday in UNIX Epoch: Jan 5, 1970
                                                             // useful for converting to `TimeOfWeek`

    /** TimeNum missing date-time
     */
    val noTimeNum  = null.asInstanceOf [TimeNum]

    /** Ordering for date-time values
     */
//  val ord = new Ordering [TimeNum] { def compare (s: TimeNum, t: TimeNum) = s compare t }

    /** Nano-seconds must be strictly than this limit (billion nanoseconds = 1 second)
     */
    val nanoLimit = 1000000000

    /** Default hour specification
     */
    private val DEFAULT_HOUR = (" 00",  " HH")

    /** Default time-zone
     */
    private val DEFAULT_ZONE = (" UTC", " z")

    /** Format for 12 hour cycle
     */
    private val formats_12h = Array ("M/d/y h:m a z",
                                     "MMM/d/y h:m a z",
                                     "MM/dd/yyyy h:m a z",
                                     "MM/dd/yyyy h:m:s:SSSSSSSSS a z",
                                     "MM/dd/yyyy hh:mm:ss a Z",
                                     "yyyy/M/d h:m:s Z)",
                                     "yyyy/M/d H:m:s z")

    /** Format for 24 hour cycle
     */
    private val formats_24h = Array (DEFAULT_DATETIME_FORMAT,
                                     "MM/dd/yyyy H:m:s",
                                     "M/d/y H:m z",
                                     "MMM/d/y H:m z",
                                     "MM/dd/yyyy H:m z",
                                     "MM/dd/yyyy H:m:s:SSSSSSSSS z",
                                     "yyyy/M/d H:m:s Z",
                                     "yyyy/M/d H:m:s z")

    /** The threshold is use for determining whether two `TimeNum` object are
     *  approximately equal (application dependent).  The initial value is one second,
     *  for one minute set it to 60.0, or for one microsecond set it to 1E-6.
     */
    private var threshold = 1.0

    /** The seconds portion of the threshold
     */
    private var threshold_s = threshold.toLong

    /** The nano-seconds portion of the threshold
     */
    private var threshold_n = ((threshold - threshold_s) * nanoLimit).toInt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the threshold to value more suitable to the current application.
     *  @param thres  the new value for the threshold
     */
    def setThreshold (thres: Double): Unit =
        threshold   = thres
        threshold_s = threshold.toLong
        threshold_n = ((threshold - threshold_s) * nanoLimit).toInt
    end setThreshold

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the absolute value of that date-time number.
     *  @param t  that date-time number
     */
    def abs (t: TimeNum): TimeNum =  TimeNum (ABS (t.inst.getEpochSecond), ABS (t.inst.getNano))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum of two date-time number, s and t.
     *  @param s  the first date-time number to compare
     *  @param t  the second date-time number to compare
     */
    def max (s: TimeNum, t: TimeNum): TimeNum = if t.toLong > s.toLong then t else s

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum of two date-time numbers, s and t.
     *  @param s  the first date-time number to compare
     *  @param t  the second date-time number to compare
     */
    def min (s: TimeNum, t: TimeNum): TimeNum = if t.toLong < s.toLong then t else s

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the square root of a `TimeNum`.
     *  @param t  the date-time to obtain the square root of
     */
    def sqrt (t: TimeNum): TimeNum =
        TimeNum (SQRT (t.inst.getEpochSecond.toDouble).toLong, SQRT (t.inst.getNano).toInt)
    end sqrt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the signum (sgn) of a `TimeNum`.
     *  @param t  the date-time to obtain the signum of
     */
    def signum (t: TimeNum): Int =
        val res = t compare _0
        if res < 0 then -1 else if res > 0 then 1 else 0
    end signum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert from `Double` to `TimeNum`.
     *  @param d  the Double parameter to convert
     */
    def double2TimeNum (d: Double): TimeNum = TimeNum (d.toLong)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TimeNum` object from Number of seconds that have elapsed
     *  since January 1, 1970 UTC.
     *  @param sec  the number of seconds
     *  @param ns   the number of nano-seconds
     */
    def apply (sec: Long, ns: Int = 0): TimeNum =
        val t = new TimeNum (Instant.ofEpochSecond (sec, ns))
//      debug ("apply", s"newly created TimeNum t = $t")
        t
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TimeNum` object by adding default time values.
     *  @param dt         the date-time in string format
     *  @param dtPattern  the pattern/format for the date-time
     */
    def apply (dt: String, dtPattern: String): TimeNum =
        var (dts, dtp) = (dt, dtPattern)
        if ! dtPattern.contains ("h") && ! dtPattern.contains ("H") then
            dts += DEFAULT_HOUR._1; dtp +=  DEFAULT_HOUR._2
        if ! dtPattern.contains ("z") && ! dtPattern.contains ("Z") then
            dts += DEFAULT_ZONE._1; dtp +=  DEFAULT_ZONE._2
        debug ("apply", s"dts = $dts, dtp = $dtp")
        new TimeNum (dts, dtp)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TimeNum` object by adding some default time values, this version
     *  does not accept date pattern from user, it makes a best guess.
     *  @param dt_  the date-time in string format
     */
    def apply (dt_ : String): TimeNum =
        val dt   = dt_.replace ('-', '/')
        var time = noTimeNum
        val timeFormat = if dt.contains ("AM") || dt.contains ("PM") then
                         formats_12h else formats_24h

        var caught = false
        var msg    = ""
        breakable {
            for format <- timeFormat do
                try
                    time = TimeNum (dt, format)
                    caught = false
                    break ()
                catch
                    case ex: Exception => caught = true; msg = ex.getMessage
                end try
            end for
        } // breakable
        if caught then throw new DateTimeException (msg)
        else time
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TimeNum` number from a `ValueType` holding a `TimeNum`.
     *  @param dt  the date-time as a `ValueType`
     */
    def fromValueType (dt: ValueType): TimeNum =
       if dt.isInstanceOf [TimeNum] then dt.asInstanceOf [TimeNum]
       else _0
    end fromValueType

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TimeNum` number from a `Double`, where the seconds are
     *  extracted from the whole number part and the nanoseconds from the
     *  fractional part.
     *  @param d  the source double
     */
    def fromDouble (d: Double): TimeNum =
        val ts = d.toLong
        val tn = ((d - ts) * nanoLimit).toInt
        TimeNum (ts, tn)
    end fromDouble

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TimeNum` number from an `Int`, where the seconds are
     *  given by n and the default (0) is used for the nanoseconds.
     *  @param n  the source integer
     */
    def fromInt (n: Int): TimeNum =
        val ts = n.toLong
        TimeNum (ts)
    end fromInt

end TimeNum

import TimeNum._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeNum` class is used to represent and operate on date-time numbers.
 *  Internally, a date-time number is represented as an `Instant`.
 *  @param inst  the underlying `Instant` of time
 */
class TimeNum (val inst: Instant)
      extends Numeric [TimeNum] with Ordered [TimeNum] with Serializable:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a `TimeNum` object from the given date-time by taking a `ZonedDateTime`
     *  and converting it to the corresponding `Instant`, the underlying type of `TimeNum` class.
     *  @param dt  the given date-time
     */
    def this (dt: ZonedDateTime) = { this (dt.toInstant ()) }
  
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a `TimeNum` object from the given date-time by creating a `ZonedDateTime`
     *  from the string  and converting it to the corresponding `Instant`, the underlying
     *  type of `TimeNum` class.  Format it using dtPattern.
     *  @param dt         the given date-time as a string
     *  @param dtPattern  the given date-time pattern/format
     */
    def this (dt: String, dtPattern: String = DEFAULT_DATETIME_FORMAT) =
        this (ZonedDateTime.parse (dt, DateTimeFormatter.ofPattern (dtPattern)).toInstant ())
    end this

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the unary minus (-).
     */
    def negate (s: TimeNum): TimeNum = TimeNum (-inst.getEpochSecond, -inst.getNano)

    inline def unary_- : TimeNum = negate (this)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this `TimeNum` and `TimeNum` t.  The seconds and nanoseconds are
     *  added separately, with carry from nanoseconds handled.
     *  @param t  the TimeNum to add to this
     */
    def plus (s: TimeNum, t: TimeNum): TimeNum =
        val (ss, sn) = (s.inst.getEpochSecond, s.inst.getNano)
        val (ts, tn) = (t.inst.getEpochSecond, t.inst.getNano)
        var (rs, rn) = (ss + ts, sn + tn)
        if rn >= nanoLimit then { rs += 1; rn -= nanoLimit }
        TimeNum (rs, rn)
    end plus

    inline def + (t: TimeNum): TimeNum = plus (this, t)
    inline def + (tl: Long): TimeNum = plus (this, TimeNum (tl))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract TimeNum` t from this `TimeNum`.
     *  @param t  the TimeNum to subtract from this
     */
    def minus (s: TimeNum, t: TimeNum): TimeNum =
        val (ss, sn) = (s.inst.getEpochSecond, s.inst.getNano)
        val (ts, tn) = (t.inst.getEpochSecond, t.inst.getNano)
        var (rs, rn) = (ss - ts, sn - tn)
        if rn < 0L then { rs -= 1; rn += nanoLimit }
        TimeNum (rs, rn)
    end minus

    inline def - (t: TimeNum): TimeNum = minus (this, t)
    inline def - (tl: Long): TimeNum = minus (this, TimeNum (tl))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this `TimeNum` and `TimeNum` t.
     *  @param t  the TimeNum that multiplies this
     */
    def times (s: TimeNum, t: TimeNum): TimeNum =
        val (ss, sn) = (s.inst.getEpochSecond, s.inst.getNano)
        val (ts, tn) = (t.inst.getEpochSecond, t.inst.getNano)
        val x: Double = ss + sn.toDouble / nanoLimit
        val y: Double = ts + tn.toDouble / nanoLimit
        val xy = x * y
        val rs = xy.toLong
        val rn = ((xy - rs) * nanoLimit).toInt
        TimeNum (rs, rn)
    end times

    inline def * (t: TimeNum): TimeNum = times (this, t)
    inline def * (tl: Long): TimeNum = times (this, TimeNum (tl))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide this `TimeNum` by `TimeNum` t.
     *  @param t  the TimeNum that divides this
     */
    def divide (s: TimeNum, t: TimeNum): TimeNum =
        throw new UnsupportedOperationException ("TimeNum does not support divide operation")
    end divide

    inline def / (t: TimeNum): TimeNum = divide (this, t)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raise this `TimeNum` to power `TimeNum` t.
     *  @param t  the TimeNum that raises this
     */
    def ~^ (t: TimeNum): TimeNum = 
        throw new UnsupportedOperationException ("TimeNum does not support ~^ operation")
    end ~^

    inline def ↑ (t: TimeNum): TimeNum =
        throw new UnsupportedOperationException ("TimeNum does not support ↑ operation")
    end ↑

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether two `TimeNum` numbers are nearly equal.
     *  @param t  the TimeNum to compare with this
     */
    def =~ (t: TimeNum): Boolean =
        val t1 = TimeNum (inst.getEpochSecond - threshold_s, inst.getNano - threshold_n)
        val t2 = TimeNum (inst.getEpochSecond + threshold_s, inst.getNano + threshold_n)
        t1 <= t && t <= t2
    end =~

    inline def ≈ (t: TimeNum): Boolean = this =~ t

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether two `TimeNum` numbers not are nearly equal.
     *  @param t  compare this with t
     */
    def !=~ (t: TimeNum): Boolean = ! (this =~ t)

    inline def ≉ (t: TimeNum): Boolean = this !=~ t

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum of this and t `TimeNum` numbers.
     *  @param t  the TimeNum number to compare with this
     */
    def max (t: TimeNum): TimeNum = if t.inst <<= inst then this else t

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum of this and t `TimeNum` numbers.
     *  @param t  the TimeNum number to compare with this
     */
    def min (t: TimeNum): TimeNum = if inst <<= t.inst then this else t

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare two `TimeNum` numbers (negative for <, zero for ==, positive for >).
     *  @param t  the TimeNum number to compare with this
     */
    infix def compare (t: TimeNum): Int = inst `compareTo` t.inst

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare two `TimeNum` numbers (negative for <, zero for ==, positive for >).
     *  @param s  the first TimeNum number
     *  @param t  the second TimeNum number
     */
    def compare (s: TimeNum, t: TimeNum): Int = s.inst `compareTo` t.inst

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare this `TimeNum` number with that date-time number t for inequality.
     *  @param t  that TimeNum number
     */
    inline def ≠ (t: TimeNum): Boolean = this != t

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare this `TimeNum` number with that date-time number t for less than
     *  or equal to.
     *  @param t  that TimeNum number
     */
    inline def ≤ (t: TimeNum): Boolean = this <= t

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare this `TimeNum` number with that date-time number t for greater
     *  than or equal to.
     *  @param t  that TimeNum number
     */
    inline def ≥ (t: TimeNum): Boolean = t <= this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this is within the given bounds.
     *  @param lim  the given (lower, upper) bounds
     */
    infix def in (lim: (TimeNum, TimeNum)): Boolean = lim._1 <= this && this <= lim._2

    inline def ∈ (lim: (TimeNum, TimeNum)): Boolean  = this in lim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this is in the given set.
     *  @param lim  the given set of values
     */
    infix def in (set: Set [TimeNum]): Boolean = set contains this

    inline def ∈ (set: Set [TimeNum]): Boolean  = this in set

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this is outside the given bounds.
     *  @param lim  the given (lower, upper) bounds
     */
    infix def out (lim: (TimeNum, TimeNum)): Boolean = this < lim._1 || lim._2 < this

    inline def ∉ (lim: (TimeNum, TimeNum)): Boolean = this out lim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this is not in the given set.
     *  @param lim  the given set of values
     */
    infix def not_in (set: Set [TimeNum]): Boolean = ! (set contains this)

    inline def ∉ (set: Set [TimeNum]): Boolean = this not_in set

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get a temporal aspect (e.g., month, week, day, hour, ...).
     *  @see https://docs.oracle.com/en/java/javase/17/docs/api//java.base/java/time/temporal/ChronoField.html
     *  @see https://docs.oracle.com/en/java/javase/17/docs/api//java.base/java/time/ZonedDateTime.html
     *  Currently only four (INSTANT_SECONDS, MICRO_OF_SECOND, MILLI_OF_SECOND, NANO_OF_SECOND)
     *  of the enumeration values are implemented, but ScalaTion adds seven more (see below).
     *  @param chrono  the enumeration type specifying the aspect to return
     */
    def getChrono (chrono: ChronoField, zone: ZoneId = ZoneOffset.UTC): Long =
        try inst.getLong (chrono)                               
        catch
            case ex: UnsupportedTemporalTypeException =>
                chrono match
                case ChronoField.DAY_OF_MONTH
                    => ZonedDateTime.ofInstant (inst, zone).getDayOfMonth ()
                case ChronoField.DAY_OF_WEEK
                    => ZonedDateTime.ofInstant (inst, zone).getDayOfWeek.getValue ()
                case ChronoField.HOUR_OF_DAY
                    => ZonedDateTime.ofInstant (inst, zone).getHour ()
                case ChronoField.MONTH_OF_YEAR
                    => ZonedDateTime.ofInstant (inst, zone).getMonthValue ()
                case ChronoField.YEAR
                    => ZonedDateTime.ofInstant (inst, zone).getYear ()
                case ChronoField.AMPM_OF_DAY
                    => ZonedDateTime.ofInstant (inst, zone).getHour () / 12
                case ChronoField.SECOND_OF_DAY
                    => val zdt = ZonedDateTime.ofInstant (inst, zone)
                       zdt.getHour () * HOUR +
                       zdt.getMinute () * MINUTE +
                       zdt.getSecond ()
                case _
                    => throw ex
        end try
    end getChrono

    def fromDouble (d: Double): TimeNum = TimeNum.fromDouble (d)
    def fromInt (i: Int): TimeNum = TimeNum.fromInt (i)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this date-time number to a `TimeNum`.
     */
    def toTimeNum: TimeNum = this

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the given `TimeNum` number to a `Double`.
     *  @param t  that date-time number to convert
     */
    def toDouble (t: TimeNum): Double = t.inst.getEpochSecond + t.inst.getNano.toDouble / nanoLimit
    def toDouble: Double = toDouble (this)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `TimeNum` number to a `Float`.
     *  @param t  that date-time number to convert
     */
    def toFloat (t: TimeNum): Float = t.toDouble.toFloat
    def toFloat: Float = toFloat (this)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `TimeNum` number to an `Int`.
     *  @param t  that date-time number to convert
     */
    def toInt (t: TimeNum): Int = t.toLong.toInt
    def toInt: Int = toInt (this)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `TimeNum` number to a `Long`.
     *  @param t  that date-time number to convert
     */
    def toLong (t: TimeNum): Long = t.inst.getEpochSecond
    def toLong: Long = toLong (this)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Override equals to determine whether this date-time number equals
     *  date-time t.
     *  @param t  the date-time number to compare with this
     */
    override infix def equals (t: Any): Boolean =
        t match
        case _ : Instant => inst `equals` t
        case _ : TimeNum => inst `equals` t.asInstanceOf [TimeNum].inst
        case _           => false
    end equals

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Must also override hashCode to be compatible with equals.
     */
    override def hashCode: Int = ABS (inst.hashCode)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse the string to create a time number.
     */
    def parseString (str: String): Option [TimeNum] = Some (TimeNum (str))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this date-time number to a String.
     */
    override def toString: String =
//      DateTimeFormatter.ofPattern (DEFAULT_DATETIME_FORMAT).withZone (ZoneId.systemDefault ()).format (inst)
        DateTimeFormatter.ofPattern (DEFAULT_DATETIME_FORMAT).withZone (ZoneId.of (DEFAULT_TZ)).format (inst)
    end toString

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this date time number to a String in the format specified by
     *  the user, use DEFAULT_DATETIME_FORMAT if user does not specify any format.
     *  @param format  the format to be used 
     */
    def toString2 (format: String = DEFAULT_DATETIME_FORMAT): String =
        DateTimeFormatter.ofPattern (format).withZone (ZoneId.of (DEFAULT_TZ)).format (inst)
        DateTimeFormatter.ofPattern (format).withZone (ZoneId.systemDefault ()).format (inst)
    end toString2

end TimeNum


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `timeNumTest` main function is used to test the `TimeNum` class.
 *  > runMain scalation.timeNumTest
 */
@main def timeNumTest (): Unit =

//  val tztest    = ZonedDateTime.now (ZoneId.of("GMT-05:00"))
    val date1     = new TimeNum (ZonedDateTime.now (ZoneId.of ("GMT-05:00")))
    Thread.sleep (2000)                                                   // to create a different TimeNum object
    val date2     = new TimeNum (ZonedDateTime.now (ZoneId.of ("GMT-05:00")))
    val date3     = date1
    val datezero  = ZonedDateTime.ofInstant (Instant.ofEpochSecond (0), ZoneId.of ("UTC"))

    println ("date1                         = " + date1)
    println ("date2                         = " + date2)

    println ("date1 ≥ date2                 = " + (date1 ≥ date2))
    println ("date1 ≤ date2                 = " + (date1 ≤ date2))
    println ("date1 ≠ date2                 = " + (date1 ≠ date2))
    println ("date1.inst.toEpoSecond        = " + date1.inst.getEpochSecond)
    println ("datezero (should print EPOCH) = " + datezero)
    println ("min (date1, date2)            = " + min (date1, date2))
    println ("max (date1, date2)            = " + max (date1, date2))
    println ("date1.hashcode                = " + date1.hashCode)
    println ("date1 equals date2            = " + (date1 equals date2))
    println ("date1 equals date3            = " + (date1 equals date3))
    println ("date1.toLong                  = " + date1.toLong)
    println ("date1.toFloat                 = " + date1.toFloat)
    println ("fromInt                       = " + date1.fromInt (date1.inst.getEpochSecond.toInt))
    println ("TimeNum.fromDouble (600.8)    = " + TimeNum.fromDouble (600.8))
    println ("TimeNum.fromInt (18000)       = " + TimeNum.fromInt (18000))
    println ("date1.toDouble (date2)        = " + date1.toDouble (date2))
    println ("date1.toFloat (date2)         = " + date1.toFloat (date2))
    println ("date1.toInt (date2)           = " + date1.toInt (date2))
    println ("date1.toLong (date2)          = " + date1.toLong (date2))
    println ("date2.hashCode                = " + date1.hashCode)

    val arr = Array (date1, date2, date3)
    println ("compare = " + (date1 compare date2))
    println ("original arr = " + stringOf (arr))
    val sarr = arr.sorted (using timeNumOrd)
    println ("sorted arr   = " + stringOf (sarr))

end timeNumTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `timeNumTest2` main function is used to test the `TimeNum` class regarding
 *  approximately equal date-times.
 *  > runMain scalation.timeNumTest2
 */
@main def timeNumTest2 (): Unit =

    TimeNum.setThreshold (10.1)                                 // 10.1 second - 10 seconds, 100,000,000 nanoseconds

    val dt =  "07/26/2018 1:12:30:800000000 UTC"                // reference

    val dtArray = Array (
        "07/26/2018 1:12:19:800000000 UTC",    // under by seconds F
        "07/26/2018 1:12:20:699999999 UTC",    // under by nanoseconds F
        "07/26/2018 1:12:20:700000000 UTC",    // on threshold T (returns F as nanoThreshold is 99,999,999 instead of 100M)
        "07/26/2018 1:12:20:700000001 UTC",    // on threshold T (returns F as nanoThreshold is 99,999,999 instead of 100M)
        "07/26/2018 1:12:20:899999999 UTC",    // within by nanoseconds T
        "07/26/2018 1:12:20:900000000 UTC",    // within by nanoseconds T
        "07/26/2018 1:12:21:800000000 UTC",    // within by seconds T

        "07/26/2018 1:12:39:800000000 UTC",    // within by seconds T
        "07/26/2018 1:12:40:800000000 UTC",    // within by seconds T
        "07/26/2018 1:12:40:899999999 UTC",    // within by nanoseconds T
        "07/26/2018 1:12:40:900000000 UTC",    // on threshold T (returns F as nanoThreshold is 99,999,999 instead of 100M)
        "07/26/2018 1:12:40:900000001 UTC",    // over by nanoseconds F
        "07/26/2018 1:12:41:800000000 UTC")    // over by seconds F

    val refTimeNum = TimeNum (dt)
    println (s"refTimeNum = $refTimeNum")
    for s <- dtArray do
        val t2 = TimeNum (s)
        println (s"$refTimeNum =~ $t2 is ${refTimeNum =~ t2}")
    end for

end timeNumTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `timeNumTest3` main function is used to test the `TimeNum` class regarding
 *  its getChrono method.
 *  > runMain scalation.timeNumTest3
 */
@main def timeNumTest3 (): Unit =

    val str = "07/26/2018 16:30:01"
    val t   = TimeNum (str)

    println (s"t = $t")
    // prints actual hours and minute
    println (s"early ${ZonedDateTime.ofInstant (t.inst, ZoneOffset.UTC)}")
    // chops hours, minutes and seconds to zero
    println (s"later ${ZonedDateTime.ofInstant (t.inst, ZoneOffset.UTC).withHour(0).withMinute(0).withSecond(0)}")

    println (s"DAY_OF_MONTH   = ${t.getChrono (ChronoField.DAY_OF_MONTH)}")
    println (s"DAY_OF_WEEK    = ${t.getChrono (ChronoField.DAY_OF_WEEK)}")
    println (s"HOUR_OF_DAY    = ${t.getChrono (ChronoField.HOUR_OF_DAY)}")
    println (s"MONTH_OF_YEAR  = ${t.getChrono (ChronoField.MONTH_OF_YEAR)}")
    println (s"YEAR           = ${t.getChrono (ChronoField.YEAR)}")
    println (s"NANO_OF_SECOND = ${t.getChrono (ChronoField.NANO_OF_SECOND)}")
    println (s"AM(0)/PM(1)    = ${t.getChrono (ChronoField.AMPM_OF_DAY)}")

end timeNumTest3

