
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jan  1 13:54:44 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Function Type FunctionS2S and Common Functions of that Type
 *
 *  @see https://scala-lang.org/api/3.x/scala/math.html
 *  @see https://en.wikipedia.org/wiki/Trigonometric_functions
 *
 *  def id (x: Double): Double
 *  def max0 (x: Double): Double
 *  def max0 (x: Int): Int
 *  def recip (x: Double): Double
 *  def sq (x: Double): Double
 *  def cb (x: Double): Double
 *  def sec (x: Double): Double
 *  def csc (x: Double): Double
 *  def cot (x: Double): Double
 *  def log2 (x: Double): Double
 *  //def log10 (x: Double): Double
 *  def pow2 (x: Double): Double
 *  def pow10 (x: Double): Double
 *  def pow (x: Double, a: Int, b: Int): Double
 *  //def log1p (x: Double): Double
 *  //def expm1 (x: Double): Double 
 *  def logb (b: Double, x: Double): Double
 *  def ihs (x: Double): Double
 *
 *  Many common functions are also supplied by the `scala.math` package.
 */

package scalation

import scala.math.{cos, exp, expm1, log, log10, log1p, max, pow, sin, sinh, sqrt, tan}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*  Top-level definitions of common scalar functions.
 *  @see Util.scala for top-level definitions related to basic utilities.
 *  @see ValueType.scala for top-level definitions related to basic datatypes.
 */

/** Type definition for functions mapping scalar `Double` to scalar `Double`
 */
type FunctionS2S = Double => Double

/** The value of log 2
 */
val log_2 = log (2.0)

/** The value of log 10
 *
val log_10 = log (10.0)
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The identity function of type `FunctionS2S`.
 *  Its inverse function is itself.
 *  @param x  the value to return
 */
inline def id (x: Double): Double = x

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The rectified linear function (max with 0) of type `FunctionS2S`.
 *  Its inverse function does not exit.
 *  @param x  the value to return
 */
inline def max0 (x: Double): Double = max (0.0, x)
inline def max0 (x: Int): Int = max (0, x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The reciprocal function of type `FunctionS2S`.
 *  Its inverse function is itself.
 *  @param x  the value to take the reciprocal of
 */
inline def recip (x: Double): Double = 1.0 / x

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The square function of type `FunctionS2S`.
 *  Its inverse function is sqrt.
 *  @param x  the value to square
 */
inline def sq (x: Double): Double = x * x

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The cube function of type `FunctionS2S`.
 *  Its inverse function is cbrt.
 *  @param x  the value to cube
 */
inline def cb (x: Double): Double = x * x * x

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The secant function of type `FunctionS2S` (function missing from `scala.math`).
 *  Its inverse function is asec.
 *  @param x  the angle in radians
 */
inline def sec (x: Double): Double = 1.0 / cos (x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The cosecant function of type `FunctionS2S` (function missing from `scala.math`).
 *  Its inverse function is acsc.
 *  @param x  the angle in radians
 */
inline def csc (x: Double): Double = 1.0 / sin (x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The cotangent function of type `FunctionS2S` (function missing from `scala.math`).
 *  Its inverse function is atan.
 *  @param x  the angle in radians
 */
inline def cot (x: Double): Double = 1.0 / tan (x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The log base 2 function of type `FunctionS2S`.
 *  Its inverse function is pow2.
 *  @param x  the value whose log is sought
 */
inline def log2 (x: Double): Double = log (x) / log_2

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The log base 10 function of type `FunctionS2S`.  Built-into Scala.
 *  Its inverse function is pow10.
 *  @param x  the value whose log is sought
 *
inline def log10 (x: Double): Double = log (x) / log_10
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Find the y-th root of x, i.e.,  x ~^ 1/y for Scala Longs.
 *  r = x ~^ 1/y is largest long integer r such that r ~^ y <= x.
 *  @see http://en.wikipedia.org/wiki/Shifting_nth_root_algorithm
 *  @see http://stackoverflow.com/questions/8826822/calculate-nth-root-with-integer-arithmetic
 *  @param x  the Long base parameter
 *  @param y  the Long root level (reciprocal exponent) parameter
 */
def lroot (x: Long, y: Long): Long =
    var r = 1L                               // initial guess for root

    def step: Long = ((y-1) * r + x / r~^(y-1)) / y

    var q = step                             // find better root
    while
        r = q
        q = step
        q < r
    do ()     // repeat looking for better root
    r
end lroot

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The power base 2 function 2^x of type `FunctionS2S`.
 *  Its inverse function is log2.
 *  @param x  the value of the exponent 2^x
 */
inline def pow2 (x: Double): Double = pow (2.0, x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The power base 10 function 10^x of type `FunctionS2S`.
 *  Its inverse function is log10.
 *  @param x  the value of the exponent 10^x
 */
inline def pow10 (x: Double): Double = pow (10.0, x)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Raise x to a rational (fractional) power r = b/c where b and c are longs using logic to handle
 *  the sign separately if the exponent is a simple fraction with an odd denominator. 
 *  In particular, use `pow` when x is non-negative, else the identity (−a)^(b/c) = (−1)^b a^(b/c)
 *  when c is odd, otherwise return Not-a-Number (NaN).
 *  @see math.stackexchange.com/questions/317528/how-do-you-compute-negative-numbers-to-fractional-powers
 *  @param x  the value of the base x^r
 *  @param r  the rational `Rat` exponent: power/root, num/den, b/c
 */
def pow_ (x: Double, r: Rat): Double =
    if x >= 0.0 then pow (x, r.toDouble)
    else if r.den % 2 == 1 then pow (-1, r.num.toDouble) * pow (-x, r.toDouble)
    else Double.NaN
end pow_

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The power to p function x^p of type `FunctionS2S`.
 *  Its inverse function is powTo (1/p)(x)
 *  Usage: define a new function pow3 = powTo (3) so pow3 (x) = x^3
 *  @param p  the value of the exponent/power x^p
 *  @param x  the value of the base x^p
 */
inline def powTo (p: Double)(x: Double): Double = pow (x, p)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The "log one plus", log base e function of x + 1 of type `FunctionS2S`.  Built-into Scala.
 *  Avoids computing log(0).
 *  @see www.johndcook.com/blog/python_log_one_plus_x
 *  Note:  log(1 + x) = x - x^2/2 with error roughly x^3/3, so when |x| < 10^-4,
 *  |x|^3 < 10^-12, and the relative error is less than 10^-8.
 *  Its inverse function is expm1.
 *  @see numpy:  np.log1p
 *  @param x  the value whose log1p is sought
 *
inline def log1p (x: Double): Double =
    if x <= -1.0 then
        Double.NaN                                // return Not-a-Number
    else if abs (x) > 1e-4 then
        log (1.0 + x)                             // x is large enough to use the obvious evaluation
    else
        (-0.5*x + 1.0) * x                        // second order Taylor approx.
end log1p
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The "exp minus one", exp function minus one of type `FunctionS2S`.  Built-into Scala.
 *  This simple approximation uses a second-order Taylor approx. for x close to zero.
 *  @see www.johndcook.com/blog/cpp_expm1
 *  Its inverse function is log1p.
 *  @see numpy:  np.expm1
 *  @see git.musl-libc.org/cgit/musl/tree/src/math/expm1.c  for a more accurate implementation
 *  @param x  the value whose expm1 is sought
 *
inline def expm1 (x: Double): Double = 
    if abs (x) < 1e-5 then
        x + 0.5 * x*x                            // second order Taylor approx.
    else
        exp (x) - 1.0                            // x is large enough to use the obvious evaluation 
end expm1
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The Inverse Hyperbolic Sine (IHS) arsinh function of type `FunctionS2S`.
 *  It approximates the log function and avoids the problem of computing log(0).
 *  @see marcfbellemare.com/wordpress/12856
 *  Its inverse function is sinh.
 *  @param x  the value whose ihs is sought
 */
inline def ihs (x: Double): Double = log (x + sqrt (x*x + 1.0))

// Related Functions:

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The log base b function (note log is the natural log).
 *  Its inverse function is pow.
 *  @param b  the base of the logarithm
 *  @param x  the value whose log is sought
 */
inline def logb (b: Double, x: Double): Double = log (x) / log (b)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `commonFunTest` main function is used to test the functions in CommonFunctions.scala.
 *      np.expm1(1e-10)   = 1.00000000005e-10
 *      np.exp(1e-10) - 1 = 1.000000082740371e-10
 *  @see numpy.org/doc/stable/reference/generated/numpy.expm1.html
 *  Note:  (log, exp) are common transformation pairs and (log1p, expm1) and (lhs, sinh)
 *  are roughly similar (see plots) and handle the log(0) = -infinity problem.
 *  > runMain scalation.commonFunTest
 */
@main def commonFunTest (): Unit =

   val x = 1e-10
   banner (s"Test expm1 (x = $x)")
   val y = expm1 (x)
   val z = exp (x) - 1.0
   println (s"expm1 ($x)   = $y")
   println (s"exp ($x) - 1 = $z")

   banner (s"Test log1p (x = $x)")
   val a = log1p (x)
   val b = log (1.0 + x)
   println (s"log1p ($x)   = $a")
   println (s"log (1 + $x) = $b")

   banner (s"Test expm1 (log1p (x = $x))")
   println (s"expm1 (log1p ($x))   = ${expm1 (a)}")
   println (s"exp (log (1+$x)) - 1 = ${exp (b) - 1}")

   banner (s"Test pow_ (xx, r)")
   val xx = -8.0
   println (s"pow_ ($xx, Rat (2, 3))) = ${pow_ (xx, Rat (2, 3))}")
   println (s"pow_ ($xx, Rat (1, 3))) = ${pow_ (xx, Rat (1, 3))}")
   println (s"pow_ ($xx, Rat (1, 2))) = ${pow_ (xx, Rat (1, 2))}")
   println (s"pow ($xx, 2/3))   = ${pow (xx, 2.toDouble/3)}")
   println (s"pow ($xx, 1/3))   = ${pow (xx, 1.toDouble/3)}")
   println (s"pow ($xx, 1/2))   = ${pow (xx, 1.toDouble/ 2)}")

   banner (s"Test ihs (x = $x)")
   val u = ihs (x)
   println (s"lhs ($x)) = $u")
   println (s"sinh (lhs ($x))) = ${sinh (u)}")

   banner ("Test ihs (x = 1)")
   val v = ihs (1.0)
   println (s"lhs (1)) = $v")
   println (s"sinh (lhs (1))) = ${sinh (v)}")

   val t = mathstat.VectorD.range (1 until 100) / 20.0
   new mathstat.Plot (t, t.map (log (_)), t.map (log1p (_)), "log vs. log1p (red)")
   new mathstat.Plot (t, t.map (log (_)), t.map (ihs (_)), "log vs. ihs (red)")
   new mathstat.Plot (t, t.map (log (_)), t.map (log10 (_)), "log vs. log10 (red)")

end commonFunTest

