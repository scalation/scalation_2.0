
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Aug 25 13:58:23 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Roll Dice - Monte Carlo Simulation
 *
 *  @see ocw.mit.edu/high-school/mathematics/combinatorics-the-fine-art-of-counting/assignments/MITHFH_solutions_6.pdf
 *  @see mathworld.wolfram.com/Dice.html
 */

package scalation
package simulation
package monte_carlo

import scala.math.{abs, min}

import scalation.mathstat.{VectorD, VectorI}
import scalation.random.Randi

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RollDice` class is used to represent a deck of playing cards.
 *  @param nDice  the number of dice to roll for one sample
 */
class RollDice (nDice: Int):

    private val NUM_SIDES = 6                                    // number of sides on dice
    private val stream    = 0                                    // random number stream
    private val dice      = Randi (1, NUM_SIDES, stream)         // random variate generator
    private var total     = 0                                    // the total number of rolls
    private val counter   = new VectorI (nDice * (NUM_SIDES-1) + 1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Roll nDice dice, returning a vector of three integers.
     */
    def roll: Int = (for _ <- 0 until nDice yield dice.igen).sum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Collect the result of of rolling nDice dice into the counter vector.
     *  @param sum  the sum of dice from a roll
     */
    def collect (sum: Int): Unit = { counter(sum - nDice) += 1; total += 1 }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the final value of the counter vector divided by the number of rolls.
     */
    def counts: VectorD = counter.toDouble / total

end RollDice


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RollDice` object provides formulas for the coefficients (number of ways)
 *  to roll a certain number.
 */
object RollDice:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the coefficient (number of ways) to roll k using 1 dice.
     *  @param k  the value (sum) from the roll
     */ 
    def coeff_1 (k: Int): Int =
        if k out (1, 6) then 0
        else                 1
    end coeff_1

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the coefficient (number of ways) to roll k using 2 dice.
     *  @param k  the value (sum) from the roll
     */ 
    def coeff_2 (k: Int): Int =
        if k out (2, 12) then 0
        else                  6 - abs (k - 7) 
    end coeff_2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the coefficient (number of ways) to roll k using 3 dice.
     *  @param k  the value (sum) from the roll
     */ 
    def coeff_3 (k: Int): Int =
        if k out (3, 18) then 0
        else if k <= 8   then (k - 1) * (k - 2) / 2
        else if k <= 14  then 21 * k - k~^2 - 83
        else                  (19 - k) * (20 - k) / 2
    end coeff_3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the number of ways to get a sum of k using n_d dice using a recursive
     *  function.
     *  @param n_d  the number of dice to roll
     *  @param k    the value of the roll (sum of the dice)
     */
    def n_ways (n_d: Int, k: Int): Int =
        if n_d == 1 then is (k in (1, 6))
        else
            var cnt = 0
            for j <- 1 to min (6, k) do cnt += n_ways (n_d - 1, k - j)
            cnt
    end n_ways

end RollDice

import RollDice._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rollDiceTest` main function is used to test the `RollDice` class.
 *  > runMain scalation.simulation.monte_carlo.rollDiceTest
 */
@main def rollDiceTest (): Unit =

    val samples = 10000000

    val count_1 = for k <- 1 to 6  yield coeff_1 (k)
    val count_2 = for k <- 2 to 12 yield coeff_2 (k)
    val count_3 = for k <- 3 to 18 yield coeff_3 (k)

    banner ("Monte Carlo: number of ways for 1 Dice")
    val monte_1 = new RollDice (1)
    cfor (0, samples) { _ => monte_1.collect (monte_1.roll) }
    val result_1 = monte_1.counts * 6
    println (s"count_2  = $count_1")
    println (s"result_1 = $result_1")

    banner ("Monte Carlo: number of ways for 2 Dice")
    val monte_2 = new RollDice (2)
    cfor (0, samples) { _ => monte_2.collect (monte_2.roll) }
    val result_2 = monte_2.counts * 6~^2
    println (s"count_2  = $count_2")
    println (s"result_2 = $result_2")

    banner ("Monte Carlo: number of ways for 3 Dice")
    val monte_3 = new RollDice (3)
    cfor (0, samples) { _ => monte_3.collect (monte_3.roll) }
    val result_3 = monte_3.counts * 6~^3
    println (s"count_3  = $count_3")
    println (s"result_3 = $result_3")
    println (s"probt_3  = ${result_3/216.0}")

end rollDiceTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `rollDiceTest2` main function computes the number of ways to get a sum of s
 *  using n_d = 1, 2, 3, 4 dice. 
 *  > runMain scalation.simulation.monte_carlo.rollDiceTest2
 */
@main def rollDiceTest2 (): Unit =

    for n_d <- 1 to 4 do
        val count = VectorI (for s <- n_d to 6 * n_d yield n_ways (n_d, s))
        println (s"count for n_d = $n_d = $count")
    end for

end rollDiceTest2

import scalation.mathstat.Plot

@main def rollDiceTest3 (): Unit =

    val dice = Randi (1, 6)
    val x    = VectorD.range (3, 19)
    val freq = new VectorD (16)
    cfor (0, 1000000) { _ =>
        val sum = dice.igen + dice.igen + dice.igen
        freq(sum-3) += 1
    } // cfor
    new Plot (x, freq)

end rollDiceTest3

