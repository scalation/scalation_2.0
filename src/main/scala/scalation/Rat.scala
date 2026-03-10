
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Casey Bowman
 *  @version 2.0
 *  @date    Sat Jul 20 22:24:50 EDT 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Rational (Rat) Numbers
 */

package scalation

//import scala.language.implicitConversions
import scala.math.floor
import scala.util.control.Breaks.{breakable, break}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Rat` class is used to represent and operate on rational numbers.
 *  Internally, a rational number is represented as two long integers.
 *  Externally, two forms are supported:
 *
 *      a/b    = 2/3         via: Rat ("2/3"), 'toString'
 *      (a, b) = (2, 3)      via: create ("(2, 3)") 'toString2'
 *
 *  A `Rat` number can be created without loss of precision using the constructor,
 *  `apply`, `create` or `fromBigDecimal` methods.  Other methods may lose precision.
 *  @param num  the numerator (e.g., 2L)
 *  @param den  the denominator (e.g., 3L)
 */
case class Rat (val num: Long, val den: Long = 1L)
     extends Fractional [Rat]
        with Ordered [Rat]:

    require (den != 0L)         // the denominator must not be zero

    /** General alias for the parts of a complex number
     */
    val (val1, val2) = (num, den)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reduce the magnitude of the numerator and denonimator by dividing
     *  both by their Greatest Common Divisor (GCD).
     */
    def reduce (): Rat =
        val gc = gcd (num, den)
        Rat (num / gc, den / gc)
    end reduce

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the unary minus (-).
     */
    def unary_- : Rat = Rat (-num, -den)
    inline def negate (q: Rat): Rat = -q

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add two rational numbers, this + q.
     *  @param q  add rational q to this
     */
    def + (q: Rat): Rat    = Rat (num * q.den + q.num * den, den * q.den)
    def + (d: Double): Rat = this + fromDouble (d)
    inline def plus (q: Rat, p: Rat): Rat = q + p

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a rational number plus a long, this + l.
     *  @param l  add long l to this
     */
    def + (l: Long): Rat = Rat (num + l * den, den)
    inline def plus (q: Rat, l: Long): Rat = q + l

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract two rational numbers, this - q.
     *  @param q  subtract rational q from this
     */
    def - (q: Rat): Rat    = Rat (num * q.den - q.num * den, den * q.den)
    def - (d: Double): Rat = this - fromDouble (d)
    inline def minus (q: Rat, p: Rat): Rat = q - p

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract: this rational number minus a long, this - l.
     *  @param l  subtract long l from this
     */
    def - (l: Long): Rat = Rat (num - l * den, den)
    inline def minus (q: Rat, l: Long): Rat = q - l

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply two rational numbers, this * q.
     *  @param q  multiply this times rational q
     */
    def * (q: Rat): Rat    = Rat (num * q.num, den * q.den)
    def * (d: Double): Rat = this * fromDouble (d)
    inline def times (q: Rat, p: Rat): Rat = q * p

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply a rational number times a long, this * l.
     *  @param l  multiply this times long l
     */
    def * (l: Long): Rat = Rat (num * l, den)
    inline def times (q: Rat, l: Long): Rat = q * l

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide two rational numbers, this / q.
     *  @param q  divide this by rational q
     */
    def / (q: Rat): Rat    = Rat (num * q.den, den * q.num) 
    def / (d: Double): Rat = this / fromDouble (d)
    inline def div (q: Rat, p: Rat): Rat = q / p

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divide a rational number div a long, this / l
     *  @param l  divide this by long l
     */
    def / (l: Long): Rat = Rat (num, den * l)
    inline def div (q: Rat, l: Long): Rat = q / l

    def ÷ (num: Long, den: Long) = Rat (num, den)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Take the reciprocal of this rational number by swapping `num` and `den`.
     */
    def recip = Rat (den, num)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raise a rational number to the q-th power.
     *  @param q  the rational power/exponent
     */
    def ~^ (q: Rat): Rat = root (this ~^ q.num, q.den)
    inline def pow (q: Rat, p: Rat): Rat = q ~^ p

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raise a rational number to the l-th power.
     *  @param l  the long power/exponent
     */
    def ~^ (l: Long): Rat = Rat (num ~^ l, den ~^ l)
    inline def pow (q: Rat, l: Long): Rat = q ~^ l

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raise a rational number to the q-th power.  Extended to handle a negative base.
     *  @see `pow_` in CommonFunctions.
     *  @param q  the rational power/exponent
     */
    def ↑ (q: Rat): Rat = fromDouble (this.toDouble ↑ q)
    inline def pow_ (q: Rat, p: Rat): Rat = q ↑ p

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Take the l-th root of the rational number q.
     *  @param l  the long root
     */
    def root (q: Rat, l: Long): Rat = Rat (lroot (q.num, l), lroot (q.den, l))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether two rational numbers are nearly equal.
     *  @param q the compare 'this' with q
     */
    def =~ (q: Rat): Boolean = this.toDouble =~ q.toDouble
    inline def ≈ (q: Rat): Boolean = this =~ q
    inline def near_eq (q: Rat, p: Rat): Boolean = q =~ p

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the absolute value of this rational number.
     */
    def abs: Rat = Rat (num.abs, den.abs)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum of this and that rational numbers.
     *  @param q  that rational number to compare with this
     */
    def max (q: Rat): Rat = if q > this then q else this

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum of this and that rational numbers.
     *  @param q  that rational number to compare with this
     */
    def min (q: Rat): Rat = if q < this then q else this

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the square root of that rational number.
     *  @param x  that rational number
     */
    def sqrt: Rat = this ~^ Rat (1L, 2L)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this rational number is integral.
     */
    def isIntegral: Boolean = den == 1L

  //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare two rational numbers (negative for <, zero for ==, positive for >).
     *  @param q  the first rational number to compare
     *  @param p  the second rational number to compare
     */
    def compare (q: Rat, p: Rat): Int = q.num * p.den compare q.num * q.den

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare this rational number with that rational number 'q'.
     *  @param q  that rational number
     */
    def compare (q: Rat): Int = num * q.den compare q.num * den

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare 'this' rational number with that rational number 'q' for inequality.
     *  @param q  that rational number
     */
    def ≠ (q: Rat) = this != q

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare 'this' rational number with that rational number 'q' for less than
     *  or equal to.
     *  @param q  that rational number
     */
    def ≤ (q: Rat) = this <= q

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare 'this' rational number with that rational number 'q' for greater 
     *  than or equal to.
     *  @param q  that rational number
     */
    def ≥ (q: Rat) = this >= q

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether 'this' is within the given bounds
     *  @param lim  the given (lower, upper) bounds
     */
    def in (lim: (Rat, Rat)): Boolean = lim._1 <= this && this <= lim._2
    def ∈ (lim: (Rat, Rat)): Boolean  = lim._1 <= this && this <= lim._2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether 'this' is in the given set.
     *  @param lim  the given set of values
     */
    def in (set: Set [Rat]): Boolean = set contains this
    def ∈ (set: Set [Rat]): Boolean  = set contains this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether 'this' is not within the given bounds
     *  @param lim  the given (lower, upper) bounds
     */
    def not_in (lim: (Rat, Rat)): Boolean = ! (lim._1 <= this && this <= lim._2)
    def ∉ (lim: (Rat, Rat)): Boolean      = ! (lim._1 <= this && this <= lim._2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether 'this' is not in the given set.
     *  @param lim  the given set of values
     */
    def not_in (set: Set [Rat]): Boolean = ! (set contains this)
    def ∉ (set: Set [Rat]): Boolean      = ! (set contains this)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert that/this rational number to a Rat.
     *  @param q  that rational number to convert
     */
    def toRat (q: Rat) = q

    def toRat = this

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert that/this rational number to a `BigDecimal` number.
     *  @param q  that rational number to convert
     */
    def toBigDecimal (q: Rat): BigDecimal = BigDecimal (q.num) / BigDecimal (q.den)

    def toBigDecimal: BigDecimal = BigDecimal (num) / BigDecimal (den)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert that/this rational number to a `Double`.
     *  @param q  that rational number to convert
     */
    def toDouble (q: Rat): Double = q.num.toDouble / q.den.toDouble

    def toDouble: Double = num.toDouble / den.toDouble

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert that/this rational number to a `Float`.
     *  @param q  that rational number to convert
     */
    def toFloat (q: Rat): Float = (q.num.toDouble / q.den.toDouble).toFloat

    def toFloat: Float = (num.toDouble / den.toDouble).toFloat

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert that/this rational number to an `Int`.
     *  @param q  that rational number to convert
     */
    def toInt (q: Rat): Int = (q.num / q.den).toInt

    def toInt: Int = (num / den).toInt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this rational number to a `Long`.
     *  @param q  that rational number to convert
     */
    def toLong (q: Rat): Long = q.num / q.den

    def toLong: Long = num / den

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `BigDecimal` number.
     *  @param y  the `BigDecimal` used to create the rational number
     */
    def fromBigDecimal (y: BigDecimal): Rat = Rat.fromBigDecimal (y)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `Double`.
     *  @see Rat.double2Rat
     *  @param y  the `Double` used to create the rational number
     */
    def fromDouble (y: Double): Rat = Rat.fromDouble (y)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `Float`.  `Float` is currently not fully
     *  supported.
     *  @param y  the `Float` used to create the rational number
     */
    def fromFloat (y: Float): Rat = Rat.fromDouble (y)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from an `Int`.
     *  @param n  the `Int` used to create the rational number
     */
    def fromInt (n: Int): Rat = Rat (n.toLong, 1L)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `Long`.
     *  @param n  the `Long` used to create the rational number
     */
    def fromLong (n: Long): Rat = Rat (n, 1L)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Override equals to determine whether this rational number equals rational 'c'.
     *  @param c  the rational number to compare with this
     */
    override def equals (c: Any): Boolean =
        val q = c.asInstanceOf [Rat]
        (num * q.den).equals (q.num * den)
    end equals

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Must also override hashCode to be be compatible with equals.
     */
    override def hashCode: Int = num.hashCode + 41 * den.hashCode

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Parse the string to create a rational number.
     */
    def parseString (str: String): Option [Rat] = Some (Rat (str))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this rational number to a String of the form 'a/b'.
     */
    override def toString: String = s"$num/$den"

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this rational number to a String of the form '(a, b)'.
     */
    def toString2: String = "(" + num + ", " + den + ")"

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the Great Common Denominator (GCD) of two long integers.
     *  @param l1  the first long number
     *  @param l2  the second long number
     */
    private def gcd (l1: Long, l2: Long): Long =
        BigInt (l1).gcd (l2).toLong
//      if l2 == 0 then l1 else gcd (l2, l1 % l2)
    end gcd

end Rat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Rat` companion object defines the origin (zero), one and minus one
 *  as well as some utility functions.
 */
object Rat:

    /** Zero (0) as a Rat number
     */
    val _0  = Rat ( 0L)

    /** One (1) as a Rat number
     */
    val _1  = Rat ( 1L)

    /** Negative one (-1) as a Rat number
     */
    val _1n = Rat (-1L)

    /** Denominator (2 ~^ 54) big enough to capture largest Double significand (53 bits)
     */
    val maxDen = 0x40000000000000L
//  val maxDen = 18014398509481984L

    /** One in `BigDecimal`
     */
    val _1_big = BigDecimal (1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Implicit conversion from `Double` to `Rat`.
     *  @param d  the Double parameter to convert
     */
//  implicit def double2Rat (d: Double): Rat = fromDouble (d)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a pair of Longs.
     *  @param qt  the tuple form of a rational number
     */
    def apply (qt: (Long, Long)): Rat = Rat (qt._1, qt._2)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from its primary string representation "a/b".
     *  Examples: "2/3", "2".
     *  @param qs  the string form of a rational number
     */
    def apply (qs: String): Rat =
        val pair = qs.split ('/')
        val p0 = pair(0)
        Rat (if pair.length == 1 then (p0.toLong, 1L)
                  else (p0.toLong, pair(1).toLong))
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from its secondary string representation "(a, b)".
     *  Examples: "(2, 3)", "(2, 1)".
     *  @param qs  the string form of a rational number
     */
    def create (qs: String): Rat =
        val pair = qs.split ('/')
        Rat (pair(0).drop(1).toLong, pair(1).dropRight(1).toLong)
    end create

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a rational number from a String of the form "12.3E+7".
     *  @see http://docs.oracle.com/javase/1.5.0/docs/api/java/math/BigDecimal.html
     *       #BigDecimal%28java.lang.String%29
     *  @param s  the given String representation of a number
     */
    def make (s: String): Rat = fromBigDecimal (BigDecimal (s))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the absolute value of that rational number.
     *  @param x  that rational number
     */
    def abs (x: Rat) = x.abs

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum of two rational numbers, q and p.
     *  @param q  the first rational number to compare
     *  @param p  the second rational number to compare
     */
    def max (q: Rat, p: Rat): Rat = if p > q then p else q

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum of two rational numbers, q and p.
     *  @param q  the first rational number to compare
     *  @param p  the second rational number to compare
     */
    def min (q: Rat, p: Rat): Rat = if p < q then p else q

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the signum (sgn) of a rational number.  The values may be -1, 0, or 1.
     *  @param r  the rational number to obtain the sigum of
     */
    def signum (r: Rat): Rat =
        if r.num == 0     then _0
        else if r.num > 0 then fromDouble (math.signum (r.den.toDouble))
        else fromDouble (-math.signum (r.den.toDouble))
    end signum

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the square root of that rational number.
     *  @param x  that rational number
     */
    def sqrt (x: Rat): Rat = x ~^ Rat (1L, 2L)

    /** Ordering for rational numbers
     */
    val ord = new Ordering [Rat]
              { def compare (x: Rat, y: Rat) = x.compare (y) }

    /** Implicit numeric value for establishing type
     */
//  implicit val num = _0

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `BigDecimal` number.
     *  @param y   the `BigDecimal` used to create the rational number
     *  @param md  the maximum denominator
     */
    def fromBigDecimal (y: BigDecimal): Rat = Rat (from_BigDecimal (y))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the numerator and denonimator of the closest rational number
     *  to the given `BigDecimal` number.
     *  @param y   the `BigDecimal` used to create the rational number
     *  @param md  the maximum denominator
     */
    def from_BigDecimal (y: BigDecimal, md: Long = Long.MaxValue): (Long, Long) =
        val epsilon = _1_big / md
        var d = y
        val n = d.setScale (0, BigDecimal.RoundingMode.FLOOR)       // floor (d)
        d -= n
        if d < epsilon then return (n.toLong, 1L)
        else if _1_big - epsilon < d then return (n.toLong + 1L, 1L)

        val dp    = d + epsilon
        val dm    = d - epsilon
        var low_n = 0L                                              // lower numerator
        var low_d = 1L                                              // lower denominator
        var upp_n = 1L                                              // upper numerator
        var upp_d = 1L                                              // upper denominator
        var mid_n = 1L                                              // middle numerator
        var mid_d = 1L                                              // middle denominator

        breakable {
            while true do
                mid_n = low_n + upp_n
                mid_d = low_d + upp_d
                if mid_d * dp < mid_n then
                    upp_n = mid_n
                    upp_d = mid_d
                else if mid_d * dm > mid_n then
                    low_n = mid_n
                    low_d = mid_d
                else
                    break ()
            end while
        } // breakable
        (n.toLong * mid_d + mid_n, mid_d)
    end from_BigDecimal

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `Double`.
     *  @param y   the double used to create the rational number
     *  @param md  the maximum denominator
     */
    def fromDouble (y: Double): Rat = Rat (from_Double (y))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the numerator and denonimator of the closest rational number
     *  to the given `BigDecimal` number.
     *  @see http://stackoverflow.com/questions/5124743/algorithm-for-simplifying-
     *       decimal-to-fractions/5128558#5128558
     *  @param y   the double used to create the rational number
     *  @param md  the maximum denominator
     */
    def from_Double (y: Double, md: Long = maxDen): (Long, Long) =
        val epsilon = 1.0 / md
        var d = y
        val n = floor (d)
        d -= n
        if d < epsilon then return (n.toLong, 1L)
        else if 1.0 - epsilon < d then return (n.toLong + 1L, 1L)
        var low_n = 0L
        var low_d = 1L
        var upp_n = 1L
        var upp_d = 1L
        var mid_n = 1L
        var mid_d = 1L

        breakable {
            while true do
                mid_n = low_n + upp_n
                mid_d = low_d + upp_d
                if mid_d * (d + epsilon) < mid_n then
                    upp_n = mid_n
                    upp_d = mid_d
                else if mid_n < (d - epsilon) * mid_d then
                    low_n = mid_n
                    low_d = mid_d
                else
                    break ()
            end while
        } // breakable
        (n.toLong * mid_d + mid_n, mid_d)
    end from_Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `Double`.
     *  FIX: if den not a power of 2, it goes to 'md'.
     *  @see http://rosettacode.org/wiki/Convert_decimal_number_to_rational
     *  @param y   the double used to create the rational number.
     *  @param md  the maximum denominator
     */
    def fromDouble2 (y: Double, md: Long = maxDen): Rat = 
        if y =~ 0.0 then return _0
        val neg = y < 0.0
        val h   = Array (0L, 1L, 0L)
        val k   = Array (1L, 0L, 0L)

        var f   = if neg then -y else y
        var a   = 0L
        var n   = 1L
        var x   = 0L
        var end = false

        while f != floor (f) do { n <<= 1; f *= 2 }     // double f until no frac
        var d = f.toLong

        breakable {
            for i <- 0 to 63 do
                a = if n != 0L then d / n else 0L
                if i > 0 && a == 0L then break ()
                x = d; d = n; n = x % n
                x = a
                if k(1) * a + k(0) >= md then
                    x = (md - k(0)) / k(1)
                    if x * 2L >= a || k(1) >= md then end = true else break ()

                h(2) = x * h(1) + h(0); h(0) = h(1); h(1) = h(2)
                k(2) = x * k(1) + k(0); k(0) = k(1); k(1) = k(2)

                if end then break ()
            end for
        } // breakable

        Rat (if neg then -h(1) else h(1), k(1))
    end fromDouble2

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number with a small odd denominator from a `Double` y
     *  by running through possible odd denominators d from 1 to `max_d` and creating
     *  two candidate numerators n for each d, one with n/d below and other above.
     *  @param y   the double used to create the rational number.
     */
    def fromDouble3 (y: Double): Rat =
        val max_d = 51                                  // limit the maximum denominator to around 50
        var nb, n = 0                                   // numerator: best and temp
        var db    = 0                                   // denominator: best
        var eb, e = Double.MaxValue                     // error: best and temp

        for d <- 1 to max_d by 2 do                     // try all odd denominators
            val dd = d.toDouble
            n = math.floor (y * d).toInt                // candidate Rat (n, d) below y
            e = math.abs (y - n/dd)
            if e < eb then { eb = e; nb = n; db = d }
            n = math.ceil (y * d).toInt                 // candidate Rat (n, d) above y
            e = math.abs (y - n/dd)
            if e < eb then { eb = e; nb = n; db = d }
        end for
        Rat (nb, db)
    end fromDouble3

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `Float`.
     *  @param y  the float used to create the rational number.
     */
    def fromFloat (y: Float): Rat = fromDouble (y)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from an `Int`.
     *  @param n  the integer used to create the rational number.
     */
    def fromInt (n: Int) = Rat (n)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a rational number from a `Long`.
     *  @param n  the long used to create the rational number.
     */
    def fromLong (n: Long) = Rat (n)

end Rat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ratTest` main function is used to test the `Rat` class.
 *  > runMain scalation.ratTest
 */
@main def ratTest (): Unit =

    import Rat._

    val a = Rat (1L,  4L)
    val b = Rat (1L,  2L)
    val c = Rat (2L,  3L)
    val d = Rat (8L, 10L)
    val e = Rat (5L)
    val f = Rat (4L,  5L)

    println ("maxDen    = " + maxDen)

    println ("a         = " + a)
    println ("b         = " + b)
    println ("c         = " + c)
    println ("d         = " + d)
    println ("e         = " + e)
    println ("-c        = " + -c)
    println ("c + d     = " + (c + d))
    println ("c - d     = " + (c - d))
    println ("c * d     = " + (c * d))
    println ("c / d     = " + (c / d))
    println ("c ~^ 2L   = " + (c ~^ 2L))
    println ("a ~^ b    = " + (a ~^ b))
    println ("c.abs     = " + c.abs)
    println ("a.sqrt    = " + a.sqrt)
    println ("c < d     = " + (c < d))
    println ("d.reduce  = " + d.reduce ())

    println ("fromDouble (.5))    = " + fromDouble (.5))
    println ("fromDouble (.25))   = " + fromDouble (.25))
    println ("fromDouble (.125))  = " + fromDouble (.125))
    println ("fromDouble (.0625)) = " + fromDouble (.0625))
    println ("fromDouble (-.125)) = " + fromDouble (-.125))
    println ("fromDouble (1./3.)) = " + fromDouble (1.0/3.0))
    println ("fromDouble (.334))  = " + fromDouble (.334))
    println ("fromDouble (.2))    = " + fromDouble (.2))
    println ("fromDouble (0.0))   = " + fromDouble (0.0))

    println ("fromDouble3 (0.33))  = " + fromDouble3 (0.33))
    println ("fromDouble3 (0.333)) = " + fromDouble3 (0.333))
    println ("fromDouble3 (0.5))   = " + fromDouble3 (0.5))
    println ("fromDouble3 (0.67))  = " + fromDouble3 (0.67))
    println ("fromDouble3 (0.667)) = " + fromDouble3 (0.667))

    println ("Compare two rational numbers")
    println (s"$c.compare ($f) = ${c.compare (f)}")
    println (s"$c.equals ($f)  = ${c.equals (f)}")
    println (s"$c == $f        = ${c == f}")
    println (s"$c =~ $f        = ${c =~ f}")
    println (s"$d.compare ($f) = ${d.compare (f)}")
    println (s"$d.equals ($f)  = ${d.equals (f)}")
    println (s"$d == $f        = ${d == f}")
    println (s"$d =~ $f        = ${d =~ f}")

end ratTest

