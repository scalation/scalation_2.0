
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Dec  5 18:21:53 EST 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: Agent-Based Simulation for UGA Bus Routes
 */

package scalation
package simulation.agent
package example_1

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.random.{Sharp, Uniform}
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runUGABusRoutes` function is used to launch the `UGABusRoutesModel` class in
 *  structure testing mode, where each `Source` will create just one agent.
 *  > runMain scalation.simulation.agent.example_1.runUGABusRoutes1
 */
@main def runUGABusRoutes1 (): Unit = new UGABusRoutesModel (nStop = 1)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runUGABusRoutes` function is used to launch the `UGABusRoutesModel` class.
 *  > runMain scalation.simulation.agent.example_1.runUGABusRoutes
 */
@main def runUGABusRoutes (): Unit = new UGABusRoutesModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `UGABusRoutesModel` class simulates buses travelling through an intersection
 *  with four traffic lights `Gates` and four roads.  Each road consists of two
 *  `Transport`s with one in each direction.
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param startSim   the start time of the simulation
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class UGABusRoutesModel (name: String = "UGABusRoutes", reps: Int = 1, startSim: Double = 0.0,
                      animating: Boolean = true, aniRatio: Double = 8.0,
                      nStop: Int = 20, stream: Int = 0)
      extends Model (name, reps, startSim, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val iaTime  = (4000.0, 6000.0)                    // (lower, upper) on inter-arrival time
    val onTime  = 8000.0                              // on (green-light) time for North-South traffic
    val offTime = 6000.0                              // off (red-light) time for North-South traffic
    val mvTime  = (2900.0, 3100.0)                    // (lower, upper) on move time
    val bsTime  = (900.0, 1100.0)                     // bus stop time
    val jpTime  = 100.0                               // jump time through intersection

    //--------------------------------------------------
    // Create Random Variates (RVs)

    val iArrivalRV = Uniform (iaTime, stream)
    val onTimeRV   = Sharp (onTime, (stream + 1) % N_STREAMS)
    val offTimeRV  = Sharp (offTime, (stream + 2) % N_STREAMS)
    val moveRV     = Uniform (mvTime, (stream + 3) % N_STREAMS)
    val bStopRV    = Uniform (bsTime, (stream + 4) % N_STREAMS)
    val jumpRV     = Sharp (jpTime, (stream + 5) % N_STREAMS)

    //--------------------------------------------------
    // Create the Graph Model: Vertices and Edges

    import Gate.{delX, delY}
    val base = (800.0, 600.0)

    val source = Source.group (this, 0.0, () => Bus (), nStop, null, base,
                              ("sN", iArrivalRV, 0, (-delX, -450)),        // from North
                              ("sE", iArrivalRV, 1, (250, -delY)),
                              ("sS", iArrivalRV, 2, (delX, 250)),
                              ("sW", iArrivalRV, 3, (-450, delY)))

    val busStop = Junction.group (this, bStopRV, null, base,
                                 ("jN2S", (-delX, -200)),
                                 ("jE2W", (-200, -delY)),
                                 ("jS2N", (delX, -200)),
                                 ("jW2E", (-200, delY)))

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
                          ("kW", (-450, -delY)),
                          ("kN", (delX, -450)),
                          ("kE", (250, delY)))

    val road1 = VEC [Transport] ()                                         // source to bus stop
    val road2 = VEC [Transport] ()                                         // bus stop to light/queue
    val link  = VEC [Link] ()                                              // link through interesection
    val road3 = VEC [Transport] ()                                         // light to sink
    for i <- source.indices do
        if i == 0 || i == 3 then                                           // from North or West
            road1 += Transport (s"ra$i", this, source(i).vert, busStop(i), moveRV)
            road2 += Transport (s"rb$i", this, busStop(i), queue(i), moveRV)
            link  += Link ("", this, queue(i), light(i).vert, jumpRV)
            road3 += Transport (s"rc$i", this, light(i).vert, sink(i), moveRV)
        else
            road1 += Transport (s"ra$i", this, source(i).vert, queue(i), moveRV)
            link  += Link ("", this, queue(i), light(i).vert, jumpRV)
            road2 += Transport (s"rb$i", this, light(i).vert, busStop(i), moveRV)
            road3 += Transport (s"rc$i", this, busStop(i), sink(i), moveRV)
        end if
    end for

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Agent

    case class Bus () extends SimAgent ("b", director.clock, this):

        def act (): Unit =
            banner (s"Bus $me started")
            val i = subtype                                           // from North (0), East (1), South (2), West (3)
            road1(i).move (this)                                      // move along road segment 1
            if i == 0 || i == 3 then                                  // from North or West
                busStop(i).jump (this)                                // jump through junction/bus stop
                road2(i).move (this)                                  // move along road segment 3
                if light(i).open (this) then queue(i).noWait (this)   // continue thru for green light
                else queue(i).waitIn (this)                           // stop and wait for red light
                link(i).jump (this)                                   // jump through gate/intersection
            else
                if light(i).open (this) then queue(i).noWait (this)   // continue thru for green light
                else queue(i).waitIn (this)                           // stop and wait for red light
                road2(i).move (this)                                  // move along road segment 3
                link(i).jump (this)                                   // jump through gate/intersection
                busStop(i).jump (this)                                // jump through junction/bus stop
            end if
            road3(i).move (this)                                      // move along road segment 3
            sink(i).leave (this)                                      // end at this sink
        end act

    end Bus

    simulate ()
    waitFinished ()
    Model.shutdown ()

end UGABusRoutesModel

