
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 26 15:00:24 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: Traffic with Turn for Process-Interaction Simulation
 */

package scalation
package simulation
package process
package example_1                                       // One-Shot

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.random.{Bernoulli, Sharp, Uniform}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runTrafficTurn` function is used to launch the `TrafficModelTurn` class.
 *  > runMain scalation.simulation.process.example_1.runTrafficTurn
 */
@main def runTrafficTurn (): Unit = new TrafficModelTurn ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TrafficModelTurn` class simulates an intersection with four traffic lights
 *  `Gates` and four roads.  Each road consists of two routes with one in each
 *  direction.  Each `Path` has two lanes (`Transport`s).
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class TrafficModelTurn (name: String = "TrafficTurn", reps: Int = 1, animating: Boolean = true,
                        aniRatio: Double = 1.0, nStop: Int = 100, stream: Int = 0)
      extends Model (name, reps, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val iaTime  = (4000.0, 6000.0)                      // (lower, upper) on inter-arrival time
    val onTime  = 8000.0                                // on (green-light) time for North-South traffic
    val offTime = 6000.0                                // off (red-light) time for North-South traffic
    val mvTime  = (2900.0, 3100.0)                      // (lower, upper) on move time
    val pTurn   = 0.25                                  // probability of making a right turn vs. going straight

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Uniform (iaTime, stream)           // use different random number streams for independence
    val onTimeRV   = Sharp (onTime, stream + 1)
    val offTimeRV  = Sharp (offTime, stream + 2)
    val moveRV     = Uniform (mvTime, stream + 3)
    val laneRV     = Bernoulli (stream = stream + 4)
    val turnRV     = Bernoulli (pTurn, stream + 5)

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
            val i  = subtype                            // from North (0), East (1), South (2), West (3)
            val l  = laneRV.igen                        // randomly select lane l
            val tn = turnRV.igen                        // 1 => turn right, 0 => go straight
            road(i).lane(l).move ()
            if light(i).shut then queue(i).waitIn ()    // not implementing turn right on red
            else queue(i).noWait ()                     // record there was no waiting
           
            if tn == 1 then
                road((i + 5) % 8).lane(l).move ()       // add 5 for next segment (turn right)
                sink((i + 3) % 4).leave ()              // add 3
            else 
                road(i + 4).lane(l).move ()             // add 4 for next segment (straight)
                sink((i + 2) % 4).leave ()              // add 2
        end act

    end Car

    simulate ()
    waitFinished ()
    Model.shutdown ()

end TrafficModelTurn

