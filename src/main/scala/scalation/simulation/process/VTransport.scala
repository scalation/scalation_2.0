
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Casey Bowman
 *  @version 2.0
 *  @date    Tue Feb  4 14:56:34 EST 2020 
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Variable Speed Transport is a Pathway between Components
 */

package scalation
package simulation
package process

import scala.collection.mutable.ArrayDeque
import scala.runtime.ScalaRunTime.stringOf

import scalation.animation.CommandType._
//import scalation.database.BpTreeMap
import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VTransport` class provides a variable-speed pathway between two other components.
 *  The components in a `Model` conceptually form a 'graph' in which the edges
 *  are `VTransport`s and the nodes are other `Component`s.
 *  @see `animation.Dgraph.move2Boundary` that aligns edge with node boundaries.
 *  @param name      the name of the variable-speed transport
 *  @param from      the starting component
 *  @param to        the ending component
 *  @param motion    the dynamics model for the speed/trip-time for motion down the `VTransport`
 *  @param isSpeed   whether speed or trip-time is used for motion
 *  @param bend      the bend or curvature of the `VTransport` (0 => line)
 *  @param shift1    the x-y shift for the transport's first end-point (from-side)
 *  @param shift2    the x-y shift for the transport's second end-point (to-side)
 */
class VTransport (name: String, from_ : Component, to_ : Component,
                  motion: Dynamics, isSpeed: Boolean = false, bend: Double = 0.0,
                  shift1: VectorD = VectorD (0, 0), shift2: VectorD = VectorD (0, 0))
      extends Transport (name, from_, to_, null, isSpeed, bend, shift1, shift2):

    private val debug = debugf ("VTransport", true)                     // debug function
//  private val flaw  = flawf ("VTransport")                            // flaw function

    private [process] val vtree = ArrayDeque [Vehicle] ()               // Array Deque for finding vehicles based on entry order
//  private [process] val vtree = new BpTreeMap [Vehicle] (name)        // B+Tree map for finding vehicles by their displacement
                                                                        // key is displacement `Double`, value actor `Vehicle` 

    debug ("init", s"name = $name, p1 = $p1, pc = $pc, p2 = $p2, located at ${stringOf (at)}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the first vehicle in `Vtransport` (the first element in vtree).
     */
    def getFirst: Vehicle =
        val first: Vehicle = if vtree.isEmpty then null else vtree.head
        debug ("getFirst", s"the first vehivle = $first")
        first
    end getFirst

//  def getFirst: Vehicle = vtree.getFirst

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the last vehicle in `Vtransport` (the last element in vtree).
     */
    def getLast: Vehicle = 
        val last: Vehicle = if vtree.isEmpty then null else vtree.last
        debug ("getLast", s"the last vehivle = $last")
        last
    end getLast

//  def getLast: Vehicle = vtree.getLast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move the entity (SimActor) smoothly down this VTransport (e.g., road).
     *  Repeatedely move it along the VTransport/Edge/QCurve.
     *  @caveat:  tokens coordinates are computed using a shadow QCurve
     *            (same coordinates as the one that will be created by the animation engine).
     */
    override def move (): Unit =
        debug ("move", s"get actor ${director.theActor} from director $director")
        println (s"actor ${director.theActor.getClass}")

        val actor = director.theActor.asInstanceOf [Vehicle]

        debug ("move", s"actor = $actor along the VTransport")
        vtree += actor                                                 // add vehicle into vtree deque by entry order
//      vtree.put (actor.disp, actor)                                  // put vehicle into vtree map by initial diplacement
        tally (Vehicle.rt)

        var done = false
        while actor.disp < curve.length && ! done do
            director.log.trace (this, "moves for " + Vehicle.rt, actor, director.clock)
            debug ("move", s"actor = $actor to be moved by motion = $motion")
            motion.updateV (actor, curve.length)                       // update actor/vehicle's motion/position
            debug ("move", s"actor = $actor has moved")
//          vtree.update (actor.disp, actor)                           // reposition vehicle in vtree by new position

            debug ("move", s"${actor.name}, x = ${actor.disp}, VTransport = $name")
            director.animate (actor, MoveToken, null, null, calcPoint (actor.disp))

            debug ("move", s"${actor.name}, check if actor.disp = ${actor.disp} >= curve.length = ${curve.length}")
            if actor.disp >= curve.length then
                done   = true                                          // done as actor/vehicle at end of `VTransporty`
                vtree -= actor
//              if ! vtree.checkedRemove (actor.disp, actor) then      // remove from vtree and check that it is the correct vehicle
//                  flaw ("move", "removed the wrong vehicle from vtree")
            end if

            actor.schedule (Vehicle.rt)
            actor.yieldToDirector ()
        end while
    end move

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the (x, y) point in the simulation space for the vehicle.
     *  `calcPoint0` is for straight lines.
     *  @param s  the current displacement along the road of the vehicle.
     */
    def calcPoint0 (s: Double): Array [Double] =
        val prop = s / curve.length
        val x = p1(0) + (p2(0) - p1(0)) * prop
        val y = p1(1) + (p2(1) - p1(1)) * prop
        Array (x - RAD, y - RAD)
    end calcPoint0

    def calcPoint (s: Double): Array [Double] =
       curve.traj = s / curve.length                                   // percentage of the curve the car has traveled thus far.
       val xy = curve.eval ()
       Array (xy.x, xy.y)
    end calcPoint
    
end VTransport

