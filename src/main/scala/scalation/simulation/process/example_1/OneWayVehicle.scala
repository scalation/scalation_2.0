
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 26 15:00:24 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: One-Way-Street (with Vehicle) for Process-Interaction Simulation
 */

package scalation
package simulation
package process
package example_1                                       // One-Shot

import scalation.random.Exponential

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runOneWayVehicle` function is used to launch the `OneWayVehicleModel` class.
 *  > runMain scalation.simulation.process.example_1.runOneWayVehicle
 */
@main def runOneWayVehicle (): Unit = new OneWayVehicleModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `OneWayVehicletModel` class simulates a one-lane roead.
 *  @caveat:  must add 'from' and 'to' components before transport!!
 *  @param name       the name of the simulation model
 *  @param reps       the number of independent replications to run
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param nStop      the number arrivals before stopping
 *  @param stream     the base random number stream (0 to 999)
 */
class OneWayVehicleModel (name: String = "OneWayVehicle", reps: Int = 1, animating: Boolean = true,
                          aniRatio: Double = 8.0, nStop: Int = 10, stream: Int = 0)
      extends Model (name, reps, animating, aniRatio):

    //--------------------------------------------------
    // Initialize Model Constants

    val lambda = 20.0                                     // car arrival rate (per hour)
    val mvTime = (2900.0, 3100.0)                         // (lower, upper) on move time

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)
    val motion     = GippsDynamics

    //--------------------------------------------------
    // Create Model Components

    val entry = VSource ("entry", this, () => Car (), 0, nStop, iArrivalRV, (100, 290))
    val exit  = Sink ("exit", (600, 290))
    val lane  = VTransport ("lane", entry, exit, motion, false, 0.25)

    addComponent (entry, exit, lane)                      // must add from and to before transport!!

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Actor

    case class Car () extends Vehicle ("c", this):

        override def act (): Unit =
            println (s"act: Car $this BEGINS")
//          val carAhead = lane.getFirst                  // find the car-ahead in lane i (the one to follow)
            println (s"act: call getLsst on lane = $lane")
            val carAhead = lane.getLast                   // find the car-ahead in lane i (the one to follow)
            println (s"act: carAhead = $carAhead")
            SimActor.addToAlist (this, carAhead)          // add this car after the car-ahead in alist
            lane.move ()                                  // move down the street/lane to the exit
            println (s"act: $this ENDS")
            SimActor.removeFromAlist (this)               // remove this car from alist
            exit.leave ()                                 // exit the street/lane
        end act

    end Car

    simulate ()
    waitFinished ()
    Model.shutdown ()

end OneWayVehicleModel

