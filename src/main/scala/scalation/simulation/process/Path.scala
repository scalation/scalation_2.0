
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jan 25 19:44:16 EST 2014
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Path for Modeling Multi-Lane Pathway
 */

package scalation
package simulation
package process

import scala.math.{abs, hypot}

import scalation.animation.CommandType._
import scalation.mathstat.VectorD
import scalation.random.Variate
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Path` class provides a multi-lane pathway between two other components.
 *  The components in a `Model` conceptually form a graph in which the edges
 *  are `Transport`s or `VTransport`s and the nodes are other `Component`s.
 *  A `Path` is a composite component that bundles several `Transport`s or `VTransport`s.
 *  @param name      the name of the path
 *  @param k         the number of lanes/transports in the path
 *  @param from      the starting component
 *  @param to        the ending component
 *  @param motion    the variate or dynamics model for the speed/trip-time for motion down the `Path`
 *  @param isSpeed   whether speed or trip-time is used for motion
 *  @param bend      the bend or curvature of the `Path` (0 => line)
 */
class Path (name: String, k: Int, val from: Component, val to: Component,
            motion: Variate | Dynamics, isSpeed: Boolean = false, bend: Double = 0.0)
      extends Component:

    private val GAP   = 10.0                              // gap between lanes 
    private val delta = calcShift                         // amount of shift in x and y directions

    val lane = Array.ofDim [Transport] (k)                // initilize lane array before calling initComponent
    for i <- lane.indices do
        val shift = VectorD ((i - (k - 1) / 2.0) * delta(0), (i - (k - 1) / 2.0) * delta(1))
        lane(i) = if motion.isInstanceOf [Variate] then
            new Transport (s"${name}_$i", from, to, motion.asInstanceOf [Variate],
                           isSpeed, bend, shift, shift)
        else
            new VTransport (s"${name}_$i", from, to, motion.asInstanceOf [Dynamics],
                            isSpeed, bend, shift, shift)
        subpart  += lane(i)
    end for

    initComponent (name, Array ())

    private val debug = debugf ("Path", true)             // debug function
    private val flaw  = flawf ("Path")                    // flaw function

    debug ("init", s"name = $name, from = ${from.name}, to = ${to.name}, delta = $delta")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of lanes.
     */
    def lanes: Int = k

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Change lanes from the current lane l1 to the new lane l2.
     *  @param l1  the actor's current lane
     *  @param l2  the actor's new lane
     */
    def changeLane (l1: Int, l2: Int): Unit =
        if abs (l1 - l2) > 2 then
            flaw ("changeLane", s"UNSAFE to cross multiple lanes at once $l1 to #l2")
        val actor = director.theActor.asInstanceOf [Vehicle]
        val (open, _, _) = laneOpenAt (l2, actor.disp)        // (open, p, s)
        if open then
            debug ("changeLane", s"from lane $l1 to lane $l2")
            director.log.trace (this, s"change lane from $l1 to $l2", actor, director.clock)
            lane(l1).asInstanceOf [VTransport].vtree -= actor
            lane(l2).asInstanceOf [VTransport].vtree += actor
//          lane(l1).asInstanceOf [VTransport].vtree.checkedRemove (actor.disp, actor)
//          lane(l2).asInstanceOf [VTransport].vtree.put (actor.disp, actor)
        else
            director.log.trace (this, s"unable to change lane from $l1 to $l2", actor, director.clock)
    end changeLane

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether it is safe to change to the new lane.
     *  @param newLane       the desired new lane
     *  @param displacement  the displacement (distance from start of the new lane)
     */
    def laneOpenAt (newLane: Int, displacement: Double): (Boolean, Vehicle, Vehicle) =
        println (s"laneOpenAt: newLane = $newLane, displacement = $displacement")
        (false, null, null)                     // FIX -- use B+Tree to see if there is a car in the way
    end laneOpenAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the direction/turn random variate to determine next the direction.
     *  This allows an application model to select the next component.
     *  The decision is delegated to this path's lane(0) transport.
     */
    def selector: Variate = lane(0).selector

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the direction/turn random variate in this path's lane(0) transport.
     *  @param selectorRV  the random variate used to select the direction
     */
    def selector_= (selector: Variate): Unit = lane(0).selector = selector

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the amount of shift in the x and y directions.
     */
    private def calcShift: VectorD =
        val xdist = from.at(0) - to.at(0)
        val ydist = from.at(1) - to.at(1)
        val hyp   = hypot (xdist, ydist)
        VectorD ((ydist / hyp) * GAP, -(xdist / hyp) * GAP)
    end calcShift

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Give the location of the curve to be its starting point.
     */
    override def at: Array [Double] = lane(0).at
        
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tell the animation engine to display this `Transport`.
     */
    def display (): Unit =
//      director.animate (this, CreateEdge, blue, QCurve (), from, to, Array (bend))
        for l <- lane do
            director.animate (l, CreateEdge, blue, l.curve, l.from, l.to,
                              Array (l.p1(0), l.p1(1), l.pc(0), l.pc(1), l.p2(0), l.p2(1)))
    end display

end Path


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `pathTest` main function is used to test the `Path` class, which is a composite
 *  class.  It simulates a two-lane road in one direction.
 *  > runMain scalation.simulation.process.pathTest
 */
@main def pathTest (): Unit =

    import scalation.random.{Bernoulli, Uniform}

    class PathModel (name: String, nArrivals: Int, iArrivalRV: Variate, moveRV: Variate)
          extends Model (name):

        val rng     = Bernoulli ()
        val onRamp  = new Source ("onRamp", this, () => Car (), 0, nArrivals, iArrivalRV, (100.0, 200.0))
        val offRamp = new Sink ("offRamp", (400.0, 200.0))
        val road    = new Path ("lane", 2, onRamp, offRamp, moveRV, false, 0.25)

        addComponent (onRamp, offRamp, road)

        case class Car () extends SimActor ("c", this):

            override def act (): Unit =
                val i = rng.igen             // choose a lane
                road.lane(i).move ()         // move along the lane
                offRamp.leave ()             // exit
            end act

        end Car

    end PathModel

    val rm = new PathModel ("road", 10, Uniform (4000, 6000), Uniform (2900, 3100))
    rm.simulate ()
    Model.shutdown ()

end pathTest

