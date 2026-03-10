
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep  7 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Transport is a Pathway between Components
 */

package scalation
package simulation
package process

import scala.math.{abs, floor}
import scala.runtime.ScalaRunTime.stringOf

import scalation.animation.CommandType._
import scalation.mathstat.VectorD
import scalation.random.{Discrete, Variate}
import scalation.scala2d.Colors._
import scalation.scala2d.QCurve
import scalation.scala2d.QCurve.calcControlPoint2

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Transport` class provides a pathway between two other components.
 *  The components in a `Model` conceptually form a 'graph' in which the edges
 *  are `Transport`s and the nodes are other `Component`s.
 *  @see `animation.Dgraph.move2Boundary` that aligns edge with node boundaries.
 *  @param name      the name of the transport
 *  @param from      the starting component
 *  @param to        the ending component
 *  @param motion    the random variate for the speed/trip-time for motion down the `Transport`
 *  @param isSpeed   whether speed or trip-time is used for motion
 *  @param bend      the bend or curvature of the `Transport` (0 => line)
 *  @param shift1    the x-y shift for the transport's first end-point (from-side)
 *  @param shift2    the x-y shift for the transport's second end-point (to-side)
 */
class Transport (name: String, val from: Component, val to: Component,
                 motion: Variate, isSpeed: Boolean = false, bend: Double = 0.0,
                 shift1: VectorD = VectorD (0, 0), shift2: VectorD = VectorD (0, 0))
      extends Component:

    initComponent (name, Array ())

    private val debug       = debugf ("Transport", true)                // debug function

    private val EPSILON     = 1E-7                                      // number close to zero
    private val STEP_SIZE   = 5                                         // number of units/pixels to move per step
    private var onTransport = 0                                         // the number of entities/sim-actors on this Transport

    private [process] val curve    = QCurve ()                          // shadow QCurve for computing locations as tokens move along curve
    private [process] val (p1, p2) = calcEndPoints                      // first and second endpoints
    private [process] val pc       = calcControlPoint2 (p1, p2, bend)   // control point (determines curvature)

    private [process] var selector: Variate = Discrete (VectorD (0.25, 0.5, 0.25))
            // Random variate for selecting next direction, defaults to left (.25), straight (.50), right (.25)

    debug ("init", s"name = $name, p1 = $p1, pc = $pc, p2 = $p2, located at ${stringOf (at)}")

    if abs (bend) < EPSILON then
        curve.setLine (p1, p2)
    else
        curve.setLine (p1, p2, bend)
//      curve.setLine (p1, pc, p2)
//      println ("loc = " + curve.getFirst)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Give the location of the curve to be its starting point.
     */
    override def at: Array [Double] =
       println (s"at: curve = $curve")
       val p1 = curve.getFirst
       Array (p1.x, p1.y)
    end at

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the coordinates of the two end-points.
     */
    private def calcEndPoints: (VectorD, VectorD) =
        val w1 = from.at(2)
        val h1 = from.at(3)
        val x1 = from.at(0) + 0.5 * w1 + shift1(0)
        val y1 = from.at(1) + 0.5 * h1 + shift1(1)

        val w2 = to.at(2)
        val h2 = to.at(3)
        val x2 = to.at(0) + 0.5 * w2 + shift2(0)
        val y2 = to.at(1) + 0.5 * h2 + shift2(1)

//      if x1 < x2 then x1 += w1 else x2 += w2
        (VectorD (x1, y1), VectorD (x2, y2))
    end calcEndPoints

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tell the animation engine to display this transport.
     */
    def display (): Unit =
//      director.animate (this, CreateEdge, blue, QCurve (), from, to, Array (bend))
        director.animate (this, CreateEdge, blue, QCurve (), from, to,
                          Array (p1(0), p1(1), pc(0), pc(1), p2(0), p2(1)))
//                        Array (p1.getX (), p1.getY (), pc.getX (), pc.getY (), p2.getX (), p2.getY ()))
    end display

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Jump the entity `SimActor` down this transport.  Place it in the middle
     *  of the `Transport`s `QCurve` for the entire trip time.
     */
    def jump (): Unit =
        val actor    = director.theActor
        val duration = if isSpeed then curve.length / motion.gen else motion.gen
        tally (duration)
        accum (onTransport)
        onTransport += 1
        director.log.trace (this, s"jumps for $duration", actor, director.clock)

        curve.traj = 0.5                                                // jump to middle of curve
        val loc = curve.next (DIAM, DIAM)
        director.animate (actor, MoveToken, null, null, Array (loc.x, loc.y))

        actor.schedule (duration)
        actor.yieldToDirector ()
        accum (onTransport)
        onTransport -= 1
    end jump

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move the entity `SimActor` smoothly down this transport.  Repeatedly
     *  move it along the `Transport`s `QCurve`.  Caveat: tokens coordinates
     *  are computed using a shadow `QCurve` (same coordinates as the one that
     *  will be created by the animation engine).
     */
    def move (): Unit =
        val actor    = director.theActor
        debug ("move", s"actor = $actor along the Transport")
        val duration = if isSpeed then curve.length / motion.gen else motion.gen
        tally (duration)
        accum (onTransport)
        onTransport += 1
        director.log.trace (this, s"moves for $duration", actor, director.clock)

        val dist  = curve.length                                        // distance to move
        val steps = (floor (dist / STEP_SIZE)).toInt                    // number of small steps on QCurve
//      val steps = (floor (curve.length / STEP_SIZE)).toInt            // number of small steps on QCurve

        curve.setSteps (steps)
        curve.traj = actor.trajectory
        var loc    = curve.next (DIAM, DIAM)                            // get the starting position for the entity/token
        actor.trajectory = curve.traj

        cfor (0, steps) { _ =>
            if loc != null then
                director.animate (actor, MoveToken, null, null, Array (loc.x, loc.y))
                actor.schedule (duration / steps.toDouble)
                actor.yieldToDirector ()
//              println ("Transport.move: -- before loc = " + loc)
                curve.traj = actor.trajectory
                loc = curve.next (DIAM, DIAM)
                actor.trajectory = curve.traj
//              println ("Transport.move: -- after  loc = " + loc)
            end if
        } // cfor

        accum (onTransport)
        onTransport -= 1
        actor.trajectory = 0.0                                          // reset for next transport
    end move
    
end Transport

