
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jan  1 13:54:44 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    A Simple Function Calculator (Double - double precision floating point numbers)
 *  @see     `CommonFunctions`
 */

package scalation

import scala.math._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Calc` object provides a template for evaluating/calculating scalar functions.
 *  Replace the part after the equal sign with your formula.
 */
object Calc:

    def f(x: Double): Double = 6.67430E-11 * 5.97219E24 * x / 6.371E6~^2

    def f2(x: Double): Double = 6.67408E-11 * 5.97219E24 * x / 6.371E6~^2 + log (1.0)

//  def f(x: Double): Double = ceil (log10 (x / 9.0))

end Calc


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runCalc` main function evaluates the function f on the given input.
 *  @param x  the command-line argument passed to function f
 *  > runMain scalation.runCalc <value-for-x>
 */
@main def runCalc (x: Double): Unit =

    import Calc.f

    println (s"f($x) = ${f(x)}")

end runCalc


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runCalcHelp` main function list the operators used in floating point calculations.
 *  > runMain scalation.runCalcHelp
 */
@main def runCalcHelp (): Unit =

    val help = """
        Operators and Functions:
        +	add
        -       subtract
        *       multiply
        /       divide
        %       mod
        ~^      exponentiate
        log     logarithm (base e)
        log2    logarithm (base 2)
        log10   logarithm (base 10)
        sin     sine (unit circle y-ccordinate)
        cos     cosine (unit circle x-ccordinate)
        tan     tangent (sin/cos)
        round   round to nearest integer
        ceil    ceiling (round up)
        floor   floor (round down)
        see     https://scala-lang.org/api/3.x/scala/math.html
    """

    println (help)

end runCalcHelp

