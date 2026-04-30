
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep  7 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: Bank for Process Simulation using 'jump'
 */

package scalation
package simulation
package process
package example_MIR                                    // Method of Independent Replications

import scalation.random.{Exponential, Sharp}
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runBank` function is used to launch the `BankModel` class.
 *  > runMain scalation.simulation.process.example_MIR.runBank
 */
@main def runBank (): Unit = new BankModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BankModel` class defines a simple process-interaction model of a bank
 *  where service is provided by one or more tellers.
 *  This version reduces the impact of transports on simulation by
 *  (1) using the transport's 'jump' method rather than its 'move' method and
 *  (2) reducing the time through the transport from minutes to seconds
 *  It also has animation turned off, by default.
 *  @caveat:  must add 'from' and 'to' components before transport!!
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class BankModel (name: String = "Bank", reps: Int = 40, animating: Boolean = false,
                 aniRatio: Double = 8.0, nStop: Int = 1000, stream: Int = 0)
      extends Model (name, reps, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val lambda   = 12.0  // 6.0                         // customer arrival rate (per hour)
    val mu       = 7.5                                  // customer service rate (per hour)
    val nTellers = 2     // 1                           // the number of bank tellers (servers)

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)
    val serviceRV  = Exponential (HOUR / mu, (stream + 1) % N_STREAMS)
    val moveRV     = Sharp (SECOND, (stream + 2) % N_STREAMS)

    //--------------------------------------------------
    // Create Model Components

    val entry     = Source ("entry", this, () => Customer (), 0, nStop, iArrivalRV, (100, 290))
    val tellerQ   = WaitQueue ("tellerQ", (330, 290))
    val teller    = Resource ("teller", tellerQ, nTellers, serviceRV, (350, 285))
    val door      = Sink ("door", (600, 290))
    val toTellerQ = Transport ("toTellerQ", entry, tellerQ, moveRV)
    val toDoor    = Transport ("toDoor", teller, door, moveRV)

    addComponent (entry, tellerQ, teller, door, toTellerQ, toDoor)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Actor

    case class Customer () extends SimActor ("c", this):

        override def act (): Unit =
            toTellerQ.jump ()                           // junp (one step) to the teller queus
            if teller.busy then tellerQ.waitIn ()       // wait when all tellers are busy
            else tellerQ.noWait ()                      // record no waiting time
            teller.utilize ()                           // process transaction with teller
            teller.release ()                           // release the teller
            toDoor.jump ()                              // jump to door
            door.leave ()                               // exit the bank
        end act

    end Customer

    simulate ()
    waitFinished ()
    Model.shutdown ()

end BankModel

