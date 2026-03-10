
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Oct  1 00:38:28 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    A Resource Provides Services to SimAgents
 */

package scalation
package simulation.agent

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.animation.CommandType.MoveToken
import scalation.database.graph.{Vertex, VertexType}
import scalation.mathstat.VectorD
import scalation.random.Variate
import scalation.scala2d._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Resource` class provides services to entities (`SimAgents`).
 *  @param name       the name of this server
 *  @param director   the `Model` directing the simulation
 *  @param serviceRV  the service time random variate
 *  @param units      the number of service units (e.g., bank tellers)
 *  @param prop       the properties of this server
 *  @param pos        the Euclidean coordinates of this server
 */
class Resource (name: String, director: Model,
                serviceRV: Variate = null, private var units: Int = 1,
                prop: Property = null, pos: VectorD = null)
      extends Vertex (name, prop, pos)
         with Statistical (name):

    Resource.add (this)
    director.statList += this                                        // add to director's variable to keep track of

    private val debug = debugf ("Resource", false)                   // debug function
    private val flaw  = flawf ("Resource")                           // flaw function
    private var inUse = 0                                            // number of units currently serving agents

    if units < 0 then flaw ("init", s"$name may not have negative units")

    debug ("init", s"name = $me, director = ${director.me}, " +
                   s"serviceRV = $serviceRV, units = $units, " +
                   s"prop = $prop, pos = $pos")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Change the number of units in this resource (e.g., add a teller).
     *  @param dUnits  the number of units to add (+ve) or remove (-ve)
     */
    def changeUnits (agent: SimAgent, dUnits: Int): Unit =
        director.log.trace (this, s"change units by $dUnits", agent, director.clock)
        if units + dUnits >= inUse then units += dUnits
        else flaw ("changeUnits", "attempt to reduce units lower than in use")
    end changeUnits

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this resource is busy (no units available).
     */
    inline def busy: Boolean = inUse == units

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Work with one unit of this server for a period of time (models an activity).
     *  @param agent     the agent who is working with this server
     *  @param duration  the agent's given service time (defaults to serviceRV)
     */
    def work (agent: SimAgent, duration: Double = serviceRV.gen): Unit =
        if busy then flaw ("work", "no units available - call busy first")
        else
            collectStats (duration, inUse, director.clock)
            agent.pos(0) = pos(0) + Resource.wh._1 / 2 - 5           // was wh.1
            agent.pos(1) = pos(1) + Resource.wh._2 / 2 - 5
            director.log.trace (this, s"work for $duration", agent, director.clock)
            director.animate (agent, MoveToken)

            if inUse < units then inUse += 1 else flaw ("utilize", "no units left")
            director.schedule (agent, duration)
            agent.yieldToDirector ()
        end if
    end work

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Release one unit of this server after work is finished.
     *  @param agent  the agent who is finished working with this server
     */
    def release (agent: SimAgent): Unit =
        director.log.trace (this, "releases", agent, director.clock)
        accumStats (inUse, director.clock)
        if inUse > 0 then inUse -= 1 else flaw ("release", "no units currently in use")
    end release

end Resource


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Resource` companion object establishes itself as a `VertexType` and provides
 *  a method of collecting its vertex instances.
 */
object Resource
       extends VertexType ("Resource", Array ("type"), color = lightblue,
                            shape = Rectangle ()):

    Model.add (Resource)

    private val debug = debugf ("Resource_", true)                   // debug function
    private val wh    = (22.0, 30.0)                                 // default display size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the position extended with the wh = (width, height).
     *  @param xy  the coordinates for the server
     */
    def at (xy: (Double, Double)): VectorD =  VectorD (xy._1, xy._2, wh._1, wh._2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the server vertex to the Resource vertex type.
     *  @param server  the server vertex to add
     */
    def add (server: Resource): Unit =
        debug ("add", s"add vertex $server to vertex type $Resource")
        verts += server
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a group of related resources using defaults for width 'w' and height 'h'.
     *  @param director   the `Model` directing the simulation
     *  @param serviceRV  the service time distribution
     *  @param prop       the properties of these servers
     *  @param xy         the (x, y) coordinates for the top-left corner of the reference resource.
     *  @param rsc        repeated resource specific info: <name, units, offset>
     */
    def group (director: Model, serviceRV: Variate,
               prop: Property = null, xy: (Double, Double),
               rsc: (String, Int, (Double, Double))*): VEC [Resource] =
        val resourceG = VEC [Resource] ()
        for r <- rsc do resourceG += Resource (r._1, director, serviceRV, r._2, prop,
                                               at((xy._1 + r._3._1, xy._2 + r._3._2)))
        resourceG
    end group

end Resource

