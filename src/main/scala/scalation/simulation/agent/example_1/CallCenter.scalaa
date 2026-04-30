
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Oct 31 01:16:38 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: Call Center for Agent-Based Simulation
 */

package scalation
package simulation.agent
package example_1                                     // One-Shot

import scalation.random.{Exponential, Uniform}
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runCallCenter` function is used to launch the `CallCenterModel` class.
 *  > runMain scalation.simulation.agent.example_1.runCallCenter
 */
@main def runCallCenter (): Unit = new CallCenterModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CallCenterModel` class defines a simple agent-based simulation  model of a
 *  call center where service is provided by one or more tele-service representatives.
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param startSim   the start time of the simulation
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class CallCenterModel (name: String = "CallCenter", reps: Int = 1, startSim: Double = 0.0,
                       animating: Boolean = true, aniRatio: Double = 8.0,
                       nStop: Int = 100, stream: Int = 0)
      extends Model (name, reps, startSim, animating, aniRatio):

    //--------------------------------------------------
    // Intialize Model Constants

    val lambda  = 6.0                                 // customer arrival rate (per hour)
    val mu      = 7.5                                 // customer service rate (per hour)
    val nPhones = 1                                   // the number of Phones (servers)

    //--------------------------------------------------
    // Create Random Variates (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)
    val serviceRV  = Exponential (HOUR / mu, (stream + 1) % N_STREAMS)
    val jTimeRV     = Uniform (0.4 * MINUTE, 0.6 * MINUTE, (stream + 2) % N_STREAMS)

    //--------------------------------------------------
    // Create the Graph Model: Vertices and Edges

    val entry    = Source ("entry", this, 0.0, iArrivalRV, () => Call (), nStop, pos = Source.at (200, 290))
    val phone    = Resource ("phone", this, serviceRV, nPhones, pos = Resource.at (350, 285))
    val hangUp   = Sink ("hangUp", this, pos = Sink.at (500, 290))
    val drop     = Sink ("drop", this, pos = Sink.at (360, 440))
    val toPhone  = Link ("toPhone", this, entry.vert, phone, jTimeRV)
    val toHangUp = Link ("toHangUp", this, phone, hangUp, jTimeRV)
    val toDrop   = Link ("toDrop", this, phone, drop, jTimeRV)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Agent

    case class Call () extends SimAgent ("c", director.clock, this):

        def act (): Unit =
            toPhone.jump (this)
            if phone.busy then
                toDrop.jump (this)
                drop.leave (this)
            else
                phone.work (this)
                phone.release (this)
                toHangUp.jump (this)
                hangUp.leave (this)
            end if
        end act

    end Call

    simulate ()
    waitFinished ()
    Model.shutdown ()

end CallCenterModel

