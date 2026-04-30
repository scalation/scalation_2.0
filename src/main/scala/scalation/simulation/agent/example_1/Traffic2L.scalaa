
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep 20 15:47:16 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: Traffic2L (Two-Lane) for Agent-Based Simulation
 */

package scalation
package simulation.agent
package example_1

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.random.{Sharp, Uniform}
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runTraffic2L` function is used to launch the `Traffic2LModel` class in
 *  structure testing mode, where each `Source` will create just one agent.
 *  > runMain scalation.simulation.agent.example_1.runTraffic2L1
 */
@main def runTraffic2L1 (): Unit = new Traffic2LModel (nStop = 1)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runTraffic2L` function is used to launch the `Traffic2LModel` class.
 *  > runMain scalation.simulation.agent.example_1.runTraffic2L
 */
@main def runTraffic2L (): Unit = new Traffic2LModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Traffic2LModel` class simulates an intersection with four traffic lights
 *  `Gates` and four roads.  Each road consists of two `Transport`s with one in each
 *  direction.
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param startSim   the start time of the simulation
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class Traffic2LModel (name: String = "Traffic2L", reps: Int = 1, startSim: Double = 0.0,
                      animating: Boolean = true, aniRatio: Double = 8.0,
                      nStop: Int = 20, stream: Int = 0)
      extends Model (name, reps, startSim, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val iaTime  = (4000.0, 6000.0)                    // (lower, upper) on inter-arrival time
    val onTime  = 8000.0                              // on (green-light) time for North-South traffic
    val offTime = 6000.0                              // off (red-light) time for North-South traffic
    val mvTime  = (2900.0, 3100.0)                    // (lower, upper) on move time
    val jpTime  = 100.0                               // jump time

    //--------------------------------------------------
    // Create Random Variates (RVs)

    val iArrivalRV = Uniform (iaTime, stream)
    val onTimeRV   = Sharp (onTime, (stream + 1) % N_STREAMS)
    val offTimeRV  = Sharp (offTime, (stream + 2) % N_STREAMS)
    val moveRV     = Uniform (mvTime, (stream + 3) % N_STREAMS)
    val jumpRV     = Sharp (jpTime, (stream + 4) % N_STREAMS)

    //--------------------------------------------------
    // Create the Graph Model: Vertices and Edges

    import Gate.{delX, delY}
    val base = (800.0, 400.0)

    val source = Source.group (this, 0.0, () => Car (), nStop, null, base,
                              ("sN", iArrivalRV, 0, (-delX, -250)),        // from North
                              ("sE", iArrivalRV, 1, (250, -delY)),
                              ("sS", iArrivalRV, 2, (delX, 250)),
                              ("sW", iArrivalRV, 3, (-250, delY)))

    val queue = WaitQueue.group (this, Int.MaxValue, null, base,
                                ("qN", (-delX, -40)),                      // before North light 
                                ("qE", (40, -delY)),
                                ("qS", (delX, 40)),
                                ("qW", (-40, delY)))

    val light = Gate.group4 (this, 0.0, onTimeRV, offTimeRV, 15, null, base,
                            ("lN", queue(0)),                              // traffic from North
                            ("lE", queue(1)),
                            ("lS", queue(2)),
                            ("lW", queue(3)))

    val sink = Sink.group (this, null, base,
                          ("kS", (-delX, 250)),                            // end for North traffic
                          ("kW", (-250, -delY)),
                          ("kN", (delX, -250)),
                          ("kE", (250, delY)))

    val road1 = VEC [Transport] ()
    val link  = VEC [Link] ()
    val road2 = VEC [Transport] ()
    for i <- source.indices do
        road1 += Transport (s"ra$i", this, source(i).vert, queue(i), moveRV)
        link  += Link ("", this, queue(i), light(i).vert, jumpRV)
        road2 += Transport (s"rb$i", this, light(i).vert, sink(i), moveRV)
    end for

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Agent

    case class Car () extends SimAgent ("c", director.clock, this):

        def act (): Unit =
            banner (s"Car $me started")
            val i = subtype                                       // from North (0), East (1), South (2), West (3)
            road1(i).move (this)                                  // move along road segment 1
            if light(i).open (this) then queue(i).noWait (this)   // continue thru for green light
            else queue(i).waitIn (this)                           // stop and wait for red light
            link(i).jump (this)                                   // jump thru gate/intersection
            road2(i).move (this)                                  // move along road segment 2
            sink(i).leave (this)                                  // end at this sink
        end act

    end Car

    simulate ()
    waitFinished ()
    Model.shutdown ()

end Traffic2LModel

