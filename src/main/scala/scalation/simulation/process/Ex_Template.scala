

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Sep 11 15:17:56 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: SOME for Process-Interaction Simulation (A Template)
 */

package scalation
package simulation
package process

import scalation.random.Exponential

/*******************************************************************************
See Example Models in sub-directories:
    example_1:    One-Shot Simulation (OSS)
                  defaults: 1 rep, animation on, move method, Model (no batching)
    example_MIR:  Method of Independent Replications (MIR)
                  defaults: 10 rep, animation off, jump method, Model (no batching)
    example_MBM:  Method of Batch Means (MBM)
                  defaults: 1 rep, animation off, jump method, Model_MBM (batching)
*******************************************************************************/


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runSOME` function is used to launch the `SOMEModel` class.
 *  The code severs as a template for writing useful simulation models.
 *  > runMain scalation.simulation.process.runSOME
 */
@main def runSOME (): Unit = new SOMEModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SOMEModel` class defines a template for process-interaction models.
 *  Caveat: must add 'from' and 'to' components before transport!!
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class SOMEModel (name: String = "SOME", reps: Int = 1, animating: Boolean = true,
                 aniRatio: Double = 8.0, nStop: Int = 100, stream: Int = 0)
      extends Model (name, reps, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val lambda = 6.0                                    // customer arrival rate (per hour)

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)

    //--------------------------------------------------
    // Create Model Components

    val entry = Source ("entry", this, () => SOMEActor (), 0, nStop, iArrivalRV, (100, 290))
    val exit  = Sink ("exit", (600, 290))

    addComponent (entry, exit)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Actor

    case class SOMEActor () extends SimActor ("s", this):

        override def act (): Unit =
            println ("SOMEActor: please write the script for this actor")
            exit.leave ()
        end act

    end SOMEActor

    simulate ()
    waitFinished ()
    Model.shutdown ()

end SOMEModel

