
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Oct 20 22:46:14 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: UGA_Bus for Process-Interaction Simulation
 */

// U N D E R   D E V E L O P M E N T

package scalation
package simulation
package process
package example_1                                       // One-Shot

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.random.{Exponential, Randi, Uniform}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runUGA_Bus` function is used to launch the `UGA_BusModel` class.
 *  > runMain scalation.simulation.process.example_1.runUGA_Bus
 */
@main def runUGA_Bus (): Unit = new UGA_BusModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `UGA_BusModel` class defines a simple process-interaction model of a bus
 *  with four stop around campus (NORTH, EAST, SOUTH, WEST).
 *  @caveat:  must add 'from' and 'to' components before transport!!
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class UGA_BusModel (name: String = "UGA_Bus", reps: Int = 1, animating: Boolean = true,
                    aniRatio: Double = 50.0, nStop: Int = 100, stream: Int = 0)
      extends Model (name, reps, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val lambda   = 10.0                                 // rider arrival rate (per hour)
    val cap      = 20                                   // bus capacity
    val NORTH    = 0
    val EAST     = 2
    val SOUTH    = 3
    val WEST     = 4

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)    // use different random number streams for independence
    val jTimeRV    = Uniform (4 * MINUTE, 6 * MINUTE, stream + 1)
    val lTimeRV    = Uniform (4 * MINUTE, 6 * MINUTE, stream + 2)
    val moveRV     = Uniform (4 * MINUTE, 6 * MINUTE, stream + 3)
    val moveRV2    = Uniform (4 * MINUTE, 6 * MINUTE, stream + 4)
    val destRV     = Randi (0, 3, stream + 5)

    //--------------------------------------------------
    // Create Model Components

    val source = Source.group (this, () => Rider (), nStop, (800, 250),
                                     ("sNorth", 0, iArrivalRV, (0, 0)),
                                     ("sEast", 1, iArrivalRV, (230, 200)),
                                     ("sSouth", 2, iArrivalRV, (30, 400)),
                                     ("sWest", 3, iArrivalRV, (-200, 230)))

    val busStop = WaitQueue.group ((800, 430), ("bNorth", (0, 0)),
                                               ("bEast", (50, 20)),
                                               ("bSouth", (30, 70)),
                                               ("bWest", (-20, 50)))

    val junction = Junction.group (jTimeRV, (800, 430), ("jNorth", (0, 0)),
                                                        ("jEast", (50, 20)),
                                                        ("jSouth", (30, 70)),
                                                        ("jWest", (-20, 50)))

    val sink = Sink.group ((830, 250), ("kNorth", (0, 0)),
                                       ("kEast", (200, 230)),
                                       ("kSouth", (-30, 400)),
                                       ("kWest", (-230, 200)))

    val road   = new VEC [Transport] ()
    val walkTo = new VEC [Transport] ()
    val walkFr = new VEC [Transport] ()
    for i <- source.indices do
        road   += Transport ("rd" + i, junction(i), junction((i+1)%4), moveRV)
        walkTo += Transport ("wt" + i, source(i), busStop(i), moveRV2)
        walkFr += Transport ("wf" + i, junction(i), sink(i), moveRV2)

    addComponents (source, busStop, junction, sink, road.toList, walkTo.toList, walkFr.toList)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Actor

    case class Rider () extends SimActor ("r", this):

        override def act (): Unit =
            val i = subtype                             // starting point, based on subtype
            val j = destRV.igen                         // destination
            walkTo(i).move ()                           // move from source to bus stop
            nextTransport = walkFr (j)                  // record destination transport
            busStop(i).waitIn ()                        // wait at bus stop for bus
            walkFr(j).move ()                           // depart the the bus and walk to building
            sink(j).leave ()                            // leave the simulation
        end act

    end Rider

    case class UGA_Bus () extends Bus ("u", this, lTimeRV, cap):

        override def act (): Unit =
            while simulating do                         // bus circulates until simulation is over - FIX use operating time
                road(NORTH).move ()                     // move to the NORTH bus stop
                unload (walkFr(NORTH))                  // unload riders
                load (busStop(NORTH))                   // load new riders

                road(EAST).move ()                      // move to the EAST bus stop
                unload (walkFr(EAST))
                load (busStop(EAST))

                road(SOUTH).move ()                     // move to the SOUTH bus stop
                unload (walkFr(SOUTH))
                load (busStop(SOUTH))

                road(WEST).move ()                      // move to the WEST bus stop
                unload (walkFr(WEST))
                load (busStop(WEST))
            end while
        end act

    end UGA_Bus

    simulate ()
    waitFinished ()
    Model.shutdown ()

end UGA_BusModel

