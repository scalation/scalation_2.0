
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 26 14:15:24 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: SOME for Event-Scheduling Simulation (A Template)
 */

package scalation
package simulation
package event

import scalation.mathstat.Statistic
import scalation.random.Exponential
//import scalation.random.RandomSeeds.N_STREAMS

/*******************************************************************************
See Example Models in sub-directories:
    example_1:    One-Shot Simulation (OSS)
                  defaults: 1 rep, Model (no batching)
    example_MIR:  Method of Independent Replications (MIR)
                  defaults: 10 rep, Model (no batching)
    example_MBM:  Method of Batch Means (MBM)
                  defaults: 1 rep, Model_MBM (batching)
*******************************************************************************/


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runSOME` function is used to launch the `SOMEModel` class.
 *  The code severs as a template for writing useful simulation models.
 *  > runMain scalation.simulation.event.runSOME
 */
@main def runSOME (): Unit = new SOMEModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SOMEModel` class defines a template for eveny-scheduling models.
 *  @param name    the name of the simulation model
 *  @param reps    the number of independent replications to run
 *  @param nStop   the number arrivals before stopping
 *  @param stream  the base random number stream (0 to 999)
 */
class SOMEModel (name: String = "SOME", reps: Int = 1, nStop: Int = 100, stream: Int = 0)
      extends Model (name, reps):

    //--------------------------------------------------
    // Initialize Model Constants

    val lambda = 6.0                                  // customer arrival rate (per hr)

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)

    //--------------------------------------------------
    // Create State Variables

    var nArr      = 0.0                               // number of customers that have arrived

    val t_ia_stat = new Statistic ("t_ia")            // time between Arrivals statistics
    addStats (t_ia_stat)

    //--------------------------------------------------
    // Specify Logic for each Type of Simulation Event

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** `Arrival` is a subclass of `Event` for handling arrival events.
     *  The 'occur' method triggers future events and updates the current state.
     *  @param customer  the entity that arrives
     *  @param delay     the time delay for this event's occurrence
     */
    case class Arrival (customer: Entity, delay: Double)
         extends Event (customer, this, delay, t_ia_stat):

        def occur (): Unit =
            if nArr < nStop - 1 then
                val toArrive = Entity (iArrivalRV.gen, 0.0, SOMEModel.this)
                schedule (Arrival (toArrive, toArrive.iArrivalT))
            nArr += 1                                      // update the current state
        end occur

    end Arrival

    //--------------------------------------------------
    // Start the simulation after scheduling the first priming event

    val firstArrival = Entity (iArrivalRV.gen, 0.0, this)
    schedule (Arrival (firstArrival, firstArrival.iArrivalT))     // first priming event
    simulate ()                                                   // start simulating

    report (("nArr", nArr))
    reportStats ()

end SOMEModel

