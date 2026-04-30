
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Aug 25 13:58:23 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Grain Dropping Experiment - Monte Carlo Simulation
 */

package scalation
package simulation
package monte_carlo

import scalation.random.RandomVecD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GrainDropping` class may be used for rough estimates of pi by using
 *  Monte Carlo Simulation.
 *  @see also Buffon Needle Experiment
 *  @param stream  the random number stream to use
 */
class GrainDropping (stream: Int):

    private val grain = RandomVecD (2, max = 1, min = -1, stream = stream)  // random vector generator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the fraction of grains found inside the unit circle. 
     *  @param n  the number of grains to generate
     */
    def fraction (n: Int): Double =
        var count = 0
        cfor (0, n) { _ => if grain.gen.normSq <= 1.0 then count += 1 }
        count / n.toDouble
    end fraction

end GrainDropping


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `grainDroppingTest` main function is used to test the `GrainDropping` class.
 *  > runMain scalation.simulation.monte_carlo.grainDroppingTest
 */
@main def grainDroppingTest (): Unit =

    for stream <- 0 to 5 do
        val bn = new GrainDropping (stream)
        banner (s"Grain Dropping Results for stream = $stream")
        for k <- 1 to 8 do
            val n = 10~^k
            println (s"for n = $n: pi = ${4 * bn.fraction (n)}")
        end for
    end for

end grainDroppingTest

