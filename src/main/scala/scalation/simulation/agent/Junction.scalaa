
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Oct 11 19:46:20 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Junction/Connector between Transports/Routes
 */

package scalation
package simulation.agent

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.animation.CommandType._
import scalation.database.graph.{Vertex, VertexType}
import scalation.mathstat.VectorD
import scalation.random.Variate
import scalation.scala2d.RoundRectangle
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Junction` class provides a connector between two `Transport`s/`Route`s.
 *  Since `Lines` and `QCurves` have limitations (e.g., hard to make a loop back),
 *  a junction may be needed.
 *  @param name      the name of this junction
 *  @param director  the `Model` directing the simulation
 *  @param jTimeRV   the jump-time through the junction
 *  @param prop      the properties of this junction
 *  @param pos       the Euclidean coordinates of this junction
 */
class Junction (name: String, director: Model, jTimeRV: Variate,
                prop: Property = null, pos: VectorD = null)
      extends Vertex (name, prop, pos)
         with Statistical (name):

    Junction.add (this)
    director.statList += this                                        // add to director's variable to keep track of

    private val debug = debugf ("Junction", false)                   // debug function 

    debug ("init", s"name = $me, director = ${director.me}, jTimeRV = $jTimeRV, " +
                   s"prop = $prop, pos = $pos")

    private var onJunction = 0            // the number of entities/sim-agents on this Junction

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Jump the entity 'SimAgent' onto this junction.
     *  @param agent     the agent who is on this junction
     *  @param duration  the total time the agent spends in this junction
     */
    def jump (agent: SimAgent, duration: Double = jTimeRV.gen): Unit =
        collectStats (duration, onJunction, director.clock)
        onJunction += 1
        agent.pos(0) = pos(0) + Junction.wh._1 / 4                   // was / 1
        agent.pos(1) = pos(1) + Junction.wh._2 / 4                   // was / 2
        director.log.trace (this, s"jumps for $duration", agent, director.clock)
        director.animate (agent, MoveToken)

        director.schedule (agent, duration)
        agent.yieldToDirector ()

        accumStats (onJunction, director.clock)
        onJunction -= 1
    end jump

end Junction


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Junction` companion object establishes itself as a `VertexType` and provides
 *  a method of collecting its vertex instances.
 */
object Junction
       extends VertexType ("Junction", Array ("type"), color = lightgrey,
                            shape = RoundRectangle ()):

    Model.add (Junction)

    private val debug = debugf ("Junction", true)                    // debug function
    private val wh    = (20.0, 20.0)                                 // default display size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the position extended with the wh = (width, height).
     *  @param xy  the coordinates for the junction
     */
    def at (xy: (Double, Double)): VectorD =  VectorD (xy._1, xy._2, wh._1, wh._2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the junction vertex to the Junction vertex type.
     *  @param junction  the junction vertex to add
     */
    def add (junction: Junction): Unit =
        debug ("add", s"add vertex $junction to vertex type $Junction")
        verts += junction
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a group of related junctions using defaults for width 'w' and height 'h'.
     *  @param director  the director controlling the model
     *  @param jTimeRV   the jump-time through the junctions
     *  @param prop      the properties of these junctions
     *  @param xy        the (x, y) coordinates for the top-left corner of the reference junction.
     *  @param jnt       repeated junction specific info: <name, offset>
     */
    def group (director: Model, jTimeRV: Variate,
               prop: Property = null, xy: (Double, Double),
               jnt: (String, (Double, Double))*): VEC [Junction] =
        val junctionG = VEC [Junction] ()
        for j <- jnt do junctionG += Junction (j._1, director, jTimeRV, prop,
                                               at((xy._1 + j._2._1, xy._2 + j._2._2)))
        junctionG
    end group

end Junction

