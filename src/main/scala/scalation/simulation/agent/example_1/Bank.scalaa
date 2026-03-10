
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 26 15:00:24 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: Bank for Agent-Based Simulation
 */

package scalation
package simulation.agent
package example_1                                     // One-Shot

import scalation.mathstat.VectorD
import scalation.random.{Exponential, Sharp, Uniform}
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runBank` function is used to launch the `BankModel` class.
 *  > runMain scalation.simulation.agent.example_1.runBank
 */
@main def runBank (): Unit = new BankModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BankModel` class defines a simple agent-based simulation model of a bank
 *  where service is provided by one or more tellers.
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param startSim   the start time of the simulation
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class BankModel (name: String = "Bank", reps: Int = 1, startSim: Double = 0.0,
                 animating: Boolean = true, aniRatio: Double = 8.0,
                 nStop: Int = 100, stream: Int = 0)
      extends Model (name, reps, startSim, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val lambda   = 6.0                                // customer arrival rate (per hour)
    val mu       = 7.5                                // customer service rate (per hour)
    val nTellers = 1                                  // the number of bank tellers (servers)

    //--------------------------------------------------
    // Create Random Variates (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)
    val serviceRV  = Exponential (HOUR / mu, (stream + 1) % N_STREAMS)
    val moveRV     = Uniform (4 * MINUTE, 6 * MINUTE, (stream + 2) % N_STREAMS)

    //--------------------------------------------------
    // Create the Graph Model: Vertices and Edges

    val entry_pos = Source.at (100, 290)
    val cust_pos  = VectorD (110, 290, 10, 10)         // FIX - need general solution for multiple sources

    val entry     = Source ("entry", this, 0.0, iArrivalRV, () => Customer (), nStop, pos = entry_pos)
    val tellerQ   = WaitQueue ("tellerQ", this, pos = WaitQueue.at (330, 290))
    val teller    = Resource ("teller", this, serviceRV, nTellers, pos = Resource.at (380, 285))
    val door      = Sink ("door", this, pos = Sink.at (600, 290))
    val toTellerQ = Transport ("toTellerQ", this, entry.vert, tellerQ, moveRV)
    val toTeller  = Link ("to", this, tellerQ, teller, Sharp (0.0))
    val toDoor    = Transport ("toDoor", this, teller, door, moveRV)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Agent

    case class Customer () extends SimAgent ("c", director.clock, this, cust_pos.copy):

        def act (): Unit =
            toTellerQ.move (this)
            if teller.busy then tellerQ.waitIn (this) else tellerQ.noWait (this)
            toTeller.jump (this)
            teller.work (this)
            teller.release (this)
            tellerQ.ping ()
            toDoor.move (this)
            door.leave (this)
        end act

    end Customer

    simulate ()
    waitFinished ()
    Model.shutdown ()

end BankModel

