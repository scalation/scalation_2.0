
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 26 15:00:24 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: Traffic for Process-Interaction Simulation
 */

package scalation
package simulation
package process
package example_1                                       // One-Shot

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.random.{Bernoulli, Sharp, Uniform}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runTrafficLaneChange` function is used to launch the `TrafficLaneChangeModel` class.
 *  > runMain scalation.simulation.process.example_1.runTrafficLaneChange
 */
@main def runTrafficLaneChange (): Unit = new TrafficLaneChangeModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TrafficLaneChangeModel` class simulates an intersection with four traffic lights
 *  `Gates` and four roads.  Each road consists of two routes with one in each
 *  direction.  Each `Path` has two lanes (`Transport`s).
 *  This mnodel includes making LANE CHANGES.
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class TrafficLaneChangeModel (name: String = "TrafficLaneChange", reps: Int = 1, animating: Boolean = true,
                              aniRatio: Double = 1.0, nStop: Int = 10, stream: Int = 0)
      extends Model (name, reps, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val iaTime  = (4000.0, 6000.0)                      // (lower, upper) on inter-arrival time
    val onTime  = 8000.0                                // on (green-light) time for North-South traffic
    val offTime = 6000.0                                // off (red-light) time for North-South traffic
    val mvTime  = (2900.0, 3100.0)                      // (lower, upper) on move time

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Uniform (iaTime, stream)           // use different random number streams for independence
    val onTimeRV   = Sharp (onTime, stream + 1)
    val offTimeRV  = Sharp (offTime, stream + 2)
    val moveRV     = Uniform (mvTime, stream + 3)
    val laneRV     = Bernoulli (stream = stream + 4)

    //--------------------------------------------------
    // Create Model Components

    val source = Source.group (this, () => Car (), nStop, (800, 250),
                              ("s1N", 0, iArrivalRV, (0, 0)),            // from North
                              ("s1E", 1, iArrivalRV, (230, 200)),
                              ("s1S", 2, iArrivalRV, (30, 400)),
                              ("s1W", 3, iArrivalRV, (-200, 230)))

    val queue = WaitQueue.group ((800, 430), ("q1N", (0, 0)),            // before North light
                                             ("q1E", (50, 20)),
                                             ("q1S", (30, 70)),
                                             ("q1W", (-20, 50)))

    val light = Gate.group (this, nStop, onTimeRV, offTimeRV, (800, 480),
                           ("l1N", queue(0), (0, 0)),                    // traffic from North
                           ("l1E", queue(1), (0, -30)),
                           ("l1S", queue(2), (30, -30)),
                           ("l1W", queue(3), (30, 0)))

    val sink = Sink.group ((830, 250), ("k1N", (0, 0)),
                                       ("k1E", (200, 230)),
                                       ("k1S", (-30, 400)),              // end for North traffic
                                       ("k1W", (-230, 200)))

    val road = VEC [Path] ()
    for i <- source.indices do
        road += Path ("ra" + i, 2, source(i), queue(i), moveRV)
    for i <- source.indices do
        road += Path ("rb" + i, 2, light(i),  sink((i + 2) % 4), moveRV)

    addComponents (source, queue, light, sink, road.toList)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Actor

    case class Car () extends SimActor ("c", this):

        override def act (): Unit =
            banner (s"Car $me started")
            val i = subtype                             // from North (0), East (1), South (2), West (3)
            val j = laneRV.igen                         // randomly select lane j

            road(i).lane(j).move ()                     // FIX - diff segment: move half way down lane j
            val k = (j + 1) % 2                         // index of other lane
            road(i).changeLane (j, k)                   // change from lane j to lane k
            road(i).lane(k).move ()                     // FIX - diff segment: move the rest of the way down lane k

            if light(i).shut then queue(i).waitIn ()    // stop and wait if light is red
            else queue(i).noWait ()                     // record skipping the queue for green light
            road(i+4).lane(j).move ()                   // move along road i+4 in lane j
            sink((i+2)%4).leave ()                      // end at the corresponding sink
        end act

    end Car

    simulate ()
    waitFinished ()
    Model.shutdown ()

end TrafficLaneChangeModel

