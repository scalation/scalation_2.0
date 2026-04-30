
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Aug 25 15:38:28 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Monte Carlo Integration
 */

package scalation
package simulation
package monte_carlo

import scala.math.sqrt

import scalation.random.Uniform

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MonteCarloIntegration` uses Monte Carlo sampling to approximate the
 *  integral of f(x) over the domain [a, b].
 */
object MonteCarloIntegration:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inegrate the function f on the interval [a, b].
     *  @param f  the function to integrate
     *  @param a  the start of the interval
     *  @param b  the end of the interval
     *  @param m  the number of samples of f to collect
     *  @param s  the random number stream
     */
    def integrate (f: FunctionS2S, a: Double, b: Double, m: Int, s: Int = 0): Double =
        val length = b - a
        val x   = Uniform (a, b, s)
        var sum = 0.0
        cfor (0, m) { _ => sum += f(x.gen) }
        sum * length / m
    end integrate

end MonteCarloIntegration


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MonteCarloIntegrationTest` object provides a simple example of Monte Carlo
 *  Integration.  This example provides a slow converging way to compute pi.
 *  The area of circle with radius one is pi.
 *  > runMain scalation.simulation.monte_carlo.monteCarloIntegrationTest
 */
@main def monteCarloIntegrationTest (): Unit =

    import MonteCarloIntegration.integrate

    def h(x: Double): Double = sqrt (1 - x~^2)

    for k <- 1 to 9; s <- 0 to 1 do
        val pi = 4 * integrate (h, 0, 1, 10~^k, s)
        println (s"for k = $k, s = $s: pi = $pi")
    end for

end monteCarloIntegrationTest

