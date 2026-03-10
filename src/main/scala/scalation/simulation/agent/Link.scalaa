
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Sep 30 16:46:03 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Link is a Pathway between Vertices
 */

package scalation
package simulation.agent

import scala.math.atan2

import scalation.animation.CommandType.MoveToken
import scalation.database.graph.{Edge, Vertex, EdgeType}
import scalation.mathstat.VectorD
import scalation.random.Variate
import scalation.scala2d._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Link` class provides a pathway between two vertices (jump through an edge).
 *  @param name      the name of this link
 *  @param director  the `Model` directing the simulation
 *  @param from      the first/starting vertex
 *  @param to        the second/ending vertex
 *  @param jTimeRV   the jump time random variate
 *  @param prop      the properties (Map) of this link
 *  @param shift1    the x-y shift for the transport's first end-point (from-side)
 *  @param shift2    the x-y shift for the transport's second end-point (to-side)
 *  @param shift     the orthogonal shift of the edge - FIX - make shifting more uniform
 */
class Link (name: String, director: Model, from: Vertex, to: Vertex,
            jTimeRV: Variate, prop: Property = null,
            shift1: VectorD = VectorD (0.0, 0.0), shift2: VectorD = VectorD (0.0, 0.0),
            shift: Int = 0)
      extends Edge (name, from, prop, to, shift)
         with EdgeAgents
         with Statistical (name):

    Link.add (this)
    director.statList += this                                        // add to director's variable to keep track of

    private val debug     = debugf ("Link", false)                   // debug function
    private var onLink    = 0                                        // the number of entities/sim-agents on this Link
    private val p1        = from.pos(0 to 2) + shift1                // position of first endpoint vertex
    private val p2        = to.pos(0 to 2) + shift2                  // position of second endpoint vertex
    private val diff      = p2 - p1                                  // the difference between the end points
    private val distance  = (p2 - p1).norm                           // the distance from p1 to p2 (length of link)
    private val angle     = atan2 (diff(1), diff(0))                 // the angle from p1 to p2
    private val STEP_SIZE = distance / 2.0                           // number of units/pixels to move per step

    debug ("init", s"name = $me, director = ${director.me}, from = ${from.me}, to = ${to.me}, " +
                   s"jTimeRV = $jTimeRV, prop = $prop, " +
                   s"points: p1 = $p1 to p2 = $p2")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Jump the entity `SimAgent` down this link.  Place it in the middle of this
     *  link for the entire trip time.  Set duration to 0.0 for instantanious jump.
     *  @param agent     the agent who is on this link
     *  @param duration  the total time the agent spends on this link
     */
    def jump (agent: SimAgent, duration: Double = jTimeRV.gen): Unit =
        collectStats (duration, onLink, director.clock)
        onLink += 1
        director.log.trace (this, s"jumps for $duration", agent, director.clock)
        agent.updatePos (STEP_SIZE, angle)                           // update position
        agent.updateLoc (STEP_SIZE)                                  // update topological location

        if duration > 0.0 then
            director.animate (agent, MoveToken)                      // duration 0.0 => no animation step
            director.schedule (agent, duration)
            agent.yieldToDirector ()
        end if

        accumStats (onLink, director.clock)
        onLink -= 1
    end jump

end Link


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Link` companion object establishes itself as a `EdgeType` and provides
 *  a method of collecting its edge instances.
 */
object Link
       extends EdgeType ("Link", null, Array ("type"), null, color = black,
                         shape = Arrow ()):

    Model.add (Link)

    private val debug = debugf ("Link_", true)                       // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the link edge to the Link edge type.
     *  @param link  the link edge to add
     */
    def add (link: Link): Unit =
        debug ("add", s"add edge $link to edge type $Link")
        edges += link
    end add

end Link

