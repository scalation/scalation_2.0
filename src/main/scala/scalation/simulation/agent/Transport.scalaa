 
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Sep 30 16:46:03 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Transport is a Pathway between Vertices
 */

package scalation
package simulation.agent

import scala.math.{atan2, floor}                            // @see scala-lang.org/api/3.x/scala/math.html#atan2-44b

import scalation.animation.CommandType.MoveToken
import scalation.database.graph.{Edge, Vertex, EdgeType}
import scalation.mathstat.VectorD
import scalation.random.Variate
import scalation.scala2d._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Transport` class provides a pathway between two vertices (motion on an edge).
 *  @param name      the name of this transport
 *  @param director  the `Model` directing the simulation
 *  @param from      the first/starting vertex
 *  @param to        the second/ending vertex
 *  @param moveRV    the movement random variate
 *  @param prop      the properties (Map) of this transport
 *  @param shift1    the x-y shift for the transport's first end-point (from-side)
 *  @param shift2    the x-y shift for the transport's second end-point (to-side)
 *  @param shift     the orthogonal shift of the edge - FIX - make shifting more uniform
 */
class Transport (name: String, director: Model, from: Vertex, to: Vertex,
                 moveRV: Variate = null, prop: Property = null,
                 shift1: VectorD = VectorD (0.0, 0.0), shift2: VectorD = VectorD (0.0, 0.0),
                 shift: Int = 0)
      extends Edge (name, from, prop, to, shift)
         with EdgeAgents
         with Statistical (name):

    Transport.add (this)

    private val debug       = debugf ("Transport", true)           // debug function
    private val STEP_SIZE   = 10                                   // number of units/pixels to move per step
    private var onTransport = 0                                    // the number of entities/sim-agents on this Transport
//  private val p1          = from.pos(0 to 2) + shift1            // position of first endpoint vertex
//  private val p2          = to.pos(0 to 2) + shift2              // position of second endpoint vertex
    private val (p1, p2)    = Transport.calcEndPoints (from, to, shift1, shift2)   // first and second endpoints
    private val diff        = p2 - p1                              // the difference between the end points
    private val distance    = diff.norm                            // the distance from p1 to p2 (length of transport)
    private val angle       = atan2 (diff(1), diff(0))             // the angle from p1 to p2

    debug ("init", s"name = $me, director = ${director.me}, from = ${from.me}, to = ${to.me}, " +
                   s"moveRV = moveRV, prop = $prop, " +
                   s"shift1 = $shift1, shift2 = $shift2, " +
                   s"points: p1 = $p1 to p2 = $p2")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move the entity `SimAgent` smoothly down this transport.  Repeatedly
     *  move it along the transport taking small steps each time.
     *  @param agent     the agent who is moving along this transport
     *  @param duration  the total time it will take agent to move over the transport
     *  @param fraction  the fraction of the remaining transform to move along
     */
    def move (agent: SimAgent, duration: Double = moveRV.gen, fraction: Double = 1.0): Unit =
        collectStats (duration, onTransport, director.clock)
        add (agent)
        onTransport += 1
        director.log.trace (this, s"moves for $duration", agent, director.clock)

        agent.setPos (p1(0), p1(1))

        val dist = fraction * distance                             // fractional distance to move

        val steps = (floor (dist / STEP_SIZE)).toInt               // number of small steps
        debug ("move", s"agent $agent for $duration using $steps steps")
        for i <- 1 to steps do
            agent.updatePos (STEP_SIZE, angle)                     // update position
            agent.updateLoc (STEP_SIZE)                            // update topological location
            director.animate (agent, MoveToken)
            director.schedule (agent, duration / steps.toDouble)
            agent.yieldToDirector ()
        end for

        accumStats (onTransport, director.clock)
        remove (agent)
        onTransport -= 1
    end move

end Transport


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Transport` companion object establishes itself as a `EdgeType` and provides
 *  a method of collecting its edge instances.
 */
object Transport
       extends EdgeType ("Transport", null, Array ("type"), null, color = blue,
                         shape = Arrow ()):

    Model.add (Transport)

    private val debug = debugf ("Transport_", true)                  // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the transport edge to the Transport edge type.
     *  @param transport  the transport edge to add
     */
    def add (transport: Transport): Unit =
        debug ("add", s"add edge $transport to edge type $Transport")
        edges += transport
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the coordinates of the two end-points of the edge/transport.
     *  @param from    the from vertex
     *  @param to      the to vertex
     *  @param shift1  the x-y shift for the transport's first end-point (from-side)
     *  @param shift2  the x-y shift for the transport's second end-point (to-side)
     */
    def calcEndPoints (from: Vertex, to: Vertex,
                       shift1: VectorD = VectorD (0.0, 0.0),
                       shift2: VectorD = VectorD (0.0, 0,0)): (VectorD, VectorD) =
        val w1 = from.pos(2)
        val h1 = from.pos(3)
        val x1 = from.pos(0) + 0.5 * w1 + shift1(0)
        val y1 = from.pos(1) + 0.5 * h1 + shift1(1)

        val w2 = to.pos(2)
        val h2 = to.pos(3)
        val x2 = to.pos(0) + 0.5 * w2 + shift2(0)
        val y2 = to.pos(1) + 0.5 * h2 + shift2(1)

//      if x1 < x2 then x1 += w1 else x2 += w2
        (VectorD (x1, y1), VectorD (x2, y2))
    end calcEndPoints

end Transport

