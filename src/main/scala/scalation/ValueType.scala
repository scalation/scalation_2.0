
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0 
 *  @date    Sat Aug 29 14:14:32 EDT 2020
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    ValueType - Union Datatype for Atomic Database Values
 *           `Double`, `Int`, `Long`, `String`, `TimeNum`
 *           (includes useful related values and methods)
 *
 *  @see     "type ValueType" below
 *
 *  @note    Other Numeric/Numeric-Related Types:
 *           in Scala 3: `math.BigDecimal`, `math.BigInt`, `Byte`, `Char` `Float`, `Short` 
 *           in ScalaTion: `Complex`, `Rat`
 */

package scalation

import java.lang.Double.isNaN

import scala.collection.StringOps
import scala.collection.mutable.Map

//import scala.collection.immutable.{Vector => VEC}                       // for immutable
import scala.collection.mutable.{ArrayBuffer => VEC}                      // for mutable

import scala.math.{abs, max, min, pow, Pi, sqrt}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*  Top-level definitions of values/constants and functions related to basic datatypes.
 *  @see CommonFunctions.scala for Top-level definitions of common scalar functions.
 *  @see Util.scala for top-level definitions utility functions.
 */

/** The most negative 32-bit integer value (-2147483648 = -2^31)
 */
val MIN_INTEGER = java.lang.Integer.MIN_VALUE

/** Largest positive 32-bit integer value (2147483647 = 2^31-1)
 */
val MAX_INTEGER = java.lang.Integer.MAX_VALUE

/** Smallest positive normal value of type double, 2^-1022 (retains full precision).
 *  Also, the smallest double such that 1.0 / 'SAFE_MIN' does not overflow.
 */
val MIN_NORMAL = java.lang.Double.MIN_NORMAL

/** Largest positive finite value of type double, 2^1023
 */
val MAX_VALUE = java.lang.Double.MAX_VALUE

/** Special value representing negative infinity: 1111111111110...0
 *  Ex: -1.0 / 0.0
 *  @see http://stackoverflow.com/questions/13317566/what-are-the-infinity-constants-in-java-really
 */
val NEGATIVE_INFINITY = java.lang.Double.NEGATIVE_INFINITY

/** Special value representing positive infinity: 0111111111110...0
 *  Ex: 1.0 / 0.0
 *  @see http://stackoverflow.com/questions/13317566/what-are-the-infinity-constants-in-java-really
 */
val POSITIVE_INFINITY = java.lang.Double.POSITIVE_INFINITY

/** Smallest double such that 1.0 + EPSILON != 1.0, slightly above 2^-53.
 *  Also, known as the "machine epsilon".
 *  @see https://issues.scala-lang.org/browse/SI-3791
 */
val EPSILON = 1.1102230246251568E-16               // 1 + EPSILON okay

/** Default tolerance should be much larger than the "machine epsilon".
 *  Application dependent => redefine as needed per application.
 */
val TOL = 1000.0 * EPSILON

/** The number 2π (needed in common calculations)
 */
val _2Pi = 2.0 * Pi

/** The number 2/π (needed in common calculations)
 */
val _2byPi = 2.0 / Pi

/** The number π/2 (90 degrees) (needed in common calculations)
 */
val Piby2 = Pi / 2.0

/** The number srqt 2π (needed in common calculations)
 */
val sqrt_2Pi = sqrt (2.0 * Pi)

/** The number srqt 2/π (needed in common calculations)
 */
val sqrt_2byPi = sqrt (2.0 / Pi)

/** Indicators of missing/illegal values per datatype
 */
//val NO_DOUBLE  = -0.0
val NO_DOUBLE  = NEGATIVE_INFINITY
val NO_INT     = java.lang.Integer.MIN_VALUE
val NO_LONG    = java.lang.Long.MIN_VALUE
val NO_STRING  = null.asInstanceOf [String]
val NO_TIMENUM = null.asInstanceOf [TimeNum]

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the maximum of three values: x, y, z.
 *  @param x  the first value
 *  @param y  the second value
 *  @param z  the third value
 */
inline def max3 (x: Double, y: Double, z: Double): Double = max (max (x, y), z)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return x, unless it is lees than the lower limit in absolute value, return limit with x's sign.
 *  @param x      the given value
 *  @param limit  the lower limit (a small positive value)
 */
inline def maxmag (x: Double, limit: Double): Double =
    if abs (x) < limit then if x < 0.0 then -limit else limit
    else x
end maxmag

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the absolute value of x with the sign of y.
 *  @param x  the value contributor
 *  @param y  the sign contributor
 */
inline def sign (x: Double, y: Double): Double = if y < 0.0 then -abs (x) else abs (x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Round the given double x to the desired number of decimal places.
 *  @param x       the double number which needs to be rounded off
 *  @param places  the desired number of decimal places
 */
inline def roundTo (x: Double, places: Int = 4): Double =
    val s = math.pow (10, places)
    math.round (x * s) / s
end roundTo

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Round the given double x to the closest integer (`Int`).
 *  @param x  the double number which needs to be rounded
 */
inline def round2Int (x: Double): Int = (x + 0.5).toInt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Round the given double x to the closest long integer (`Long`).
 *  @param x  the double number which needs to be rounded
 */
inline def round2Long (x: Double): Long = (x + 0.5).toLong

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Power function for scala Longs x ~^ y.  Compute: math.pow (x, y).toLong without using
 *  floating point, so as to not lose precision.
 *  @param x  the Long base parameter
 *  @param y  the Long exponent parameter
 */
def powl (x: Long, y: Long): Long =
    var base   = x
    var exp    = y
    var result = 1L
    while exp != 0L do
        if (exp & 1L) == 1L then result *= base
        exp >>= 1L
        base *= base
    result
end powl

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the relative difference between x and y.
 *  @param x  the first double precision floating point number
 *  @param y  the second double precision floating point number
 */
def rel_diff (x: Double, y: Double): Double = abs (x - y) / max (abs (x), abs (y))

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Determine whether two double precision floating point numbers 'x' and 'y'
 *  are nearly equal.  Two numbers are considered to be nearly equal, if within
 *  '2 EPSILON'.  A number is considered to be nearly zero, if within '2 MIN_NORMAL'.
 *  To accommodate round-off errors, may use 'TOL' instead of 'EPSILON'.
 *----------------------------------------------------------------------------------------
 *  @see stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison
 *----------------------------------------------------------------------------------------
 *  If both 'x' and 'y' are NaN (Not-a-Number), the IEEE standard indicates that should
 *  be considered always not equal.  For 'near_eq', they are considered nearly equal.
 *  Comment out the first line below to conform more closely to the IEEE standard.
 *  @see stackoverflow.com/questions/10034149/why-is-nan-not-equal-to-nan
 *----------------------------------------------------------------------------------------
 *  @param x  the first double precision floating point number
 *  @param y  the second double precision floating point number
 */
def near_eq (x: Double, y: Double): Boolean =
    if isNaN (x) && isNaN (y) then return true              // comment out to follow IEEE standard
    if x == y then return true                              // they are equal

    val diff  = abs (x - y)
    val norm1 = min (abs (x) + abs (y), MAX_VALUE)
    diff < max (MIN_NORMAL, TOL * norm1)
end near_eq

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Format a double value for printing.
 *  @param x  the double value to format
 */
def fmt (x: Double): String = "%.6f".format (x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Extend `Int` to include an exponentiation operator (~^), nearly equal (=~), ≤, ≥, ≠, and in/out.
 */
extension (x: Int)
    inline def ~^ (y: Int): Int = pow (x.toDouble, y.toDouble).toInt
    inline def =~ (y: Int): Boolean = x == y
    inline def ≤ (y: Int): Boolean = x <= y
    inline def ≥ (y: Int): Boolean = x >= y
    inline def ≠ (y: Int): Boolean = x != y
    infix  def in (r: (Int, Int)): Boolean = r._1 <= x && x <= r._2
    inline def ∈ (r: (Int, Int)): Boolean = x in r
    infix  def out (r: (Int, Int)): Boolean = x < r._1 || r._2 < x
    inline def ∉ (r: (Int, Int)): Boolean = x out r

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Extend `Long` to include an exponentiation operator (~^), nearly equal (=~), ≤, ≥, ≠, and in/out.
 */
extension (x: Long)
    inline def ~^ (y: Long): Long = powl (x, y)
    inline def =~ (y: Long): Boolean = x == y
    inline def ≤ (y: Long): Boolean = x <= y
    inline def ≥ (y: Long): Boolean = x >= y
    inline def ≠ (y: Long): Boolean = x != y
    infix  def in (r: (Long, Long)): Boolean = r._1 <= x && x <= r._2
    inline def ∈ (r: (Long, Long)): Boolean = x in r
    infix  def out (r: (Long, Long)): Boolean = x < r._1 || r._2 < x
    inline def ∉ (r: (Long, Long)): Boolean = x out r

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Extend `Double` to include an exponentiation operators (~^, ↑), nearly equal (=~), ≤, ≥, ≠, and in/out.
 */
extension (x: Double)
    inline def ~^ (y: Double): Double = pow (x, y)
    inline def ↑ (y: Rat): Double = pow_ (x, y)             // extended to a negative base
    inline def =~ (y: Double): Boolean = near_eq (x, y)
    inline def ≤ (y: Double): Boolean = x <= y
    inline def ≥ (y: Double): Boolean = x >= y
    inline def ≠ (y: Double): Boolean = x != y
    infix  def in (r: (Double, Double)): Boolean = r._1 <= x && x <= r._2
    inline def ∈ (r: (Double, Double)): Boolean = x in r
    infix  def out (r: (Double, Double)): Boolean = x < r._1 || r._2 < x
    inline def ∉ (r: (Double, Double)): Boolean = x out r

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Extend `String` to include an exponentiation operator (~^), nearly equal (=~), ≤, ≥, ≠, and in/out,
 *  as well operators for numeric types.  Use '.mkDouble' instead of '.toDouble'.
 */
extension (x: String)
    inline def ~^ (y: String): String = "NaN"
    inline def =~ (y: String): Boolean = x.toLowerCase () == y.toLowerCase ()
    inline def ≤ (y: String): Boolean = x <= y
    inline def ≥ (y: String): Boolean = x >= y
    inline def ≠ (y: String): Boolean = x != y
    infix  def in (r: (String, String)): Boolean = r._1 <= x && x <= r._2
    inline def ∈ (r: (String, String)): Boolean = x in r
    infix  def out (r: (String, String)): Boolean = x < r._1 || r._2 < x
    inline def ∉ (r: (String, String)): Boolean = x out r
    def unary_- : String = "-" + x
    def - (y: String): String = x diff y
    def * (y: String): String = x.repeat (y.toInt)
    def * (y: Int): String = x.repeat (y)
    def / (y: String): String = "NaN"
    def / (y: Int): String = "NaN"
    def mkDouble: Double = safe_toDouble (x)
    def mkInt: Int = safe_toInt (x)
    def mkLong: Long = safe_toLong (x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Safe method for converting a `String` into a `Double` that catches the exceptions and avoids
 *  the "infinite loop problem".
 *  @param s  the string to convert to a Double
 */
def safe_toDouble (s: String): Double =
    var d: Double = NO_DOUBLE
    try 
        d = java.lang.Double.parseDouble (s)
    catch
        case _ : java.lang.NullPointerException =>
            println ("safe_toDouble: can't parse null string")
        case _ : java.lang.NumberFormatException =>
            println (s"safe_toDouble: can't parse '$s' to create a Double")
    end try
    d
end safe_toDouble

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Safe method for converting a `String` into a `Int` that catches the exceptions and avoids
 *  the "infinite loop problem".
 *  @param s  the string to convert to an Int
 */
def safe_toInt (s: String): Int =
    var d: Int = NO_INT
    try
        d = java.lang.Integer.parseInt (s)
    catch
        case _ : java.lang.NullPointerException =>
            println ("safe_toInt: can't parse null string")
        case _ : java.lang.NumberFormatException =>
            println (s"safe_toInt: can't parse '$s' to create a Int")
    end try
    d
end safe_toInt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Safe method for converting a `String` into a `Long` that catches the exceptions and avoids
 *  the "infinite loop problem".
 *  @param s  the string to convert to a Long
 */
def safe_toLong (s: String): Long =
    var d: Long = NO_LONG
    try
        d = java.lang.Long.parseLong (s)
    catch
        case _ : java.lang.NullPointerException =>
            println ("safe_toLong: can't parse null string")
        case _ : java.lang.NumberFormatException =>
            println (s"safe_toLong: can't parse '$s' to create a Long")
    end try
    d
end safe_toLong


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ValueType` type is a union type for atomic database values.
 */
type ValueType = ( Double | Int | Long | String | TimeNum )

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ValueTypeOrd` object provides `Ordering` for `ValueType` via the compare method.
 *  Used by methods that need `Ordering` for `ValueType`, e.g., max (ValueTypeOrd).
 */
object ValueTypeOrd extends Ordering [ValueType]:
    def compare (x: ValueType, y: ValueType): Int =
        if x == y then 0 else if x > y then 1 else -1
    end compare
end ValueTypeOrd

val regexL = raw"[\-\+]?\d+".r                            // regular expression for intergers (`Int` or `Long`)
val regexD = raw"[\-\+]?\d*\.\d+".r                       // regular expression for double precision floating point (`Double`)
//val regexT = raw"regex for TimeNum is possible".r       // FIX

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Determine by the form of a string its likely domain using regular expression matching,
 *  returning a character: `D`ouble | `I`nt | `L`ong | `S`tring | `T`imeNum
 *  @see https://dotty.epfl.ch/api/scala/util/matching/Regex.html
 *  @param str  the type un-differentiated string whose domain is sought
 */
def typeOfStr (str: String): Char =
    if regexL.matches (str) then
        val long = str.mkLong
        if long < MIN_INTEGER || long > MAX_INTEGER then 'L'
        else 'I'
    else if regexD.matches (str) then 'D'
//  else if regexT.matches (str) then 'T'                 // FIX - need regex or other means of recognizing date-time (@sse `TimeNum`)
    else 'S'
end typeOfStr

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Extension methods for `ValueType`.
 */
extension (x: ValueType)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The toDouble extension method converts a `ValueType` to a `Double`.
     */
    def toDouble: Double = 
        x match
        case _: Double  => x.asInstanceOf [Double]
        case _: Int     => x.asInstanceOf [Int].toDouble
        case _: Long    => x.asInstanceOf [Long].toDouble
        case _: String  => x.asInstanceOf [String].mkDouble
        case _: TimeNum => x.asInstanceOf [TimeNum].toDouble
    end toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The toInt extension method converts a `ValueType` to an `Int`.
     */
    def toInt: Int = 
        x match
        case _: Double  => x.asInstanceOf [Double].toInt
        case _: Int     => x.asInstanceOf [Int]
        case _: Long    => x.asInstanceOf [Long].toInt
        case _: String  => x.asInstanceOf [String].mkInt
        case _: TimeNum => x.asInstanceOf [TimeNum].toInt
    end toInt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The toLong extension method converts a `ValueType` to an `Long`.
     */
    def toLong: Long = 
        x match
        case _: Double  => x.asInstanceOf [Double].toLong
        case _: Int     => x.asInstanceOf [Int].toLong
        case _: Long    => x.asInstanceOf [Long]
        case _: String  => x.asInstanceOf [String].mkLong
        case _: TimeNum => x.asInstanceOf [TimeNum].toLong
    end toLong

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The < extension method determines whether x < y.
     */
    def < (y: ValueType): Boolean =
        x match
        case _: Double  => x.asInstanceOf [Double]  < y.asInstanceOf [Double]
        case _: Int     => x.asInstanceOf [Int]     < y.asInstanceOf [Int]
        case _: Long    => x.asInstanceOf [Long]    < y.asInstanceOf [Long]
        case _: String  => StringOps (x.asInstanceOf [String]) < y.asInstanceOf [String]
        case _: TimeNum => x.asInstanceOf [TimeNum] < y.asInstanceOf [TimeNum]
    end <

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The <= extension method determines whether x <= y.
     */
    def <= (y: ValueType): Boolean =
        x match
        case _: Double  => x.asInstanceOf [Double]  <= y.asInstanceOf [Double]
        case _: Int     => x.asInstanceOf [Int]     <= y.asInstanceOf [Int]
        case _: Long    => x.asInstanceOf [Long]    <= y.asInstanceOf [Long]
        case _: String  => StringOps (x.asInstanceOf [String]) <= y.asInstanceOf [String]
        case _: TimeNum => x.asInstanceOf [TimeNum] <= y.asInstanceOf [TimeNum]
    end <=

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The > extension method determines whether x > y.
     */
    def > (y: ValueType): Boolean =
        x match
        case _: Double  => x.asInstanceOf [Double]  > y.asInstanceOf [Double]
        case _: Int     => x.asInstanceOf [Int]     > y.asInstanceOf [Int]
        case _: Long    => x.asInstanceOf [Long]    > y.asInstanceOf [Long]
        case _: String  => StringOps (x.asInstanceOf [String]) > y.asInstanceOf [String]
        case _: TimeNum => x.asInstanceOf [TimeNum] > y.asInstanceOf [TimeNum]
    end >

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The >= extension method determines whether x >= y.
     */
    def >= (y: ValueType): Boolean =
        x match
        case _: Double  => x.asInstanceOf [Double]  >= y.asInstanceOf [Double]
        case _: Int     => x.asInstanceOf [Int]     >= y.asInstanceOf [Int]
        case _: Long    => x.asInstanceOf [Long]    >= y.asInstanceOf [Long]
        case _: String  => StringOps (x.asInstanceOf [String]) >= y.asInstanceOf [String]
        case _: TimeNum => x.asInstanceOf [TimeNum] >= y.asInstanceOf [TimeNum]
    end >=


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Property` type is a map from property names to property values.
 */
type Property = Map [String, ValueType]

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The +++ extension method concatenates properties and renames to avoid ambiguity.
 */
extension (p: Property)
    def +++ (q: Property): Property = 
        val pq = p.clone
        for qe <- q do pq += (if p contains qe._1 then (qe._1 + "2", qe._2) else qe)
        pq
    end +++


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `valueTypeTest` main function is used to test the `ValueType` type.
 *  > runMain scalation.valueTypeTest
 */
@main def valueTypeTest (): Unit =

    val store = VEC [ValueType] (0, 1L, 2.0, "three", TimeNum._2)
    println (s"store = $store")
    println (s"store(0) == 1: ${store(0) == 1}")
    println (s"store(0) < 1: ${store(0) < 1}")
    //println (s"store(0) > 1: ${store(0) > 1}")

end valueTypeTest

