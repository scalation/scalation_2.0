
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Aug 20 20:02:06 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Monte Carlo Simulation of the Monty Hall Problem
 */

package scalation
package simulation
package monte_carlo

import scalation.random.{Bernoulli, Randi}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MontyHall` main function is used to test whether staying or switching picks when
 *  selecting a prize behind one of three doors in "Let's Make a Deal" is the better
 *  strategy.
 *  @see en.wikipedia.org/wiki/Monty_Hall_problem
 *  > runMain scalation.simulation.monte_carlo.montyHall
 */
@main def montyHall (): Unit =

    val limit     =    100                               // for debugging, swap comments
//  val limit     = 100000                               // for better probability estimates

    val debug     = debugf ("montyHall", true)           // debug flag
    val stream    = 0                                    // random number stream (0 to 999)
    val rg        = Randi (0, 2, stream)                 // door selection (0, 1 or 2) random generator 
    val coin      = Bernoulli (stream + 1)               // coin flip generator
    var winStay   = 0                                    // count wins with stay stategy
    var winSwitch = 0                                    // count wins with switch strategy

    cfor (0, limit) { _ =>
        val car  = rg.igen                               // car randomly placed behind this door
        val pick = rg.igen                               // contestant randomly picks a door
        val show = (car, pick) match                     // Monty Hall show other non-car door
            case (0, 1) | (1, 0) => 2                    
            case (0, 2) | (2, 0) => 1
            case (1, 2) | (2, 1) => 0
            case _               => (pick + 1 + coin.igen) % 3

        debug ("main", s"car = $car, pick = $pick, show = $show")
       
        if pick == car then winStay += 1                 // stay with initial pick
        else                winSwitch += 1               // switch to the other door 
    } // cfor

    println (s"winStay   = $winStay")
    println (s"winSwitch = $winSwitch")

end montyHall

