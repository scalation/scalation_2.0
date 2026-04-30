
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Sep 30 16:46:03 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Sink Terminates Entities/SimAgents
 */

package scalation
package simulation.agent

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.animation.CommandType.DestroyToken
import scalation.database.graph.{Vertex, VertexType}
import scalation.mathstat.VectorD
import scalation.scala2d._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Sink` class is used to terminate entities `SimAgent`s when they are finished.
 *  @param name      the name of this sink
 *  @param director  the director controlling the model
 *  @param prop      the properties of this sink
 *  @param pos       the position (Euclidean coordinate) of this sink
 */
class Sink (name: String, director: Model,
            prop: Property = null, pos: VectorD = null)
      extends Vertex (name, prop, pos)
         with Statistical (name):

    Sink.add (this)
    director.statList += this                                        // add to director's variable to keep track of

    private val debug = debugf ("Sink", false)                       // debug function 

    debug ("init", s"name = $me, director = ${director.me}, " +
                   s"prop = $prop, pos = $pos")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Leave the model, effectively terminating the entity `SimAgent`.
     */
    def leave (agent: SimAgent): Unit =
        debug ("leave", s"agent ${agent.id} leaving")
        tallyStats (director.clock - agent.arrivalT)
        director.log.trace (this, "terminates", agent, director.clock)
        director.animate (agent, DestroyToken)
        agent.yieldToDirector (true)                                 // yield and terminate
    end leave

end Sink


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Sink` companion object establishes itself as a `VertexType` and provides
 *  a method of collecting its vertex instances.
 */
object Sink
       extends VertexType ("Sink", Array ("type"), color = darkmagenta,
                           shape = Ellipse ()):

    Model.add (Sink)

    private val debug = debugf ("Sink_", false)                      // debug function
    private val wh = (22.0, 22.0)                                    // default display size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the position extended with the wh = (width, height).
     *  @param xy  the coordinates for the server
     */
    def at (xy: (Double, Double)): VectorD =  VectorD (xy._1, xy._2, wh._1, wh._2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the sink vertex to the Sink vertex type.
     *  @param sink  the sink vertex to add
     */
    def add (sink: Sink): Unit =
        debug ("add", s"add vertex $sink to vertex type $Sink")
        verts += sink
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a group of related sinks using defaults for width 'w' and height 'h'.
     *  @param director  the director controlling the model
     *  @param prop      the properties of these sinks
     *  @param xy        the (x, y) coordinates for the top-left corner of the reference sink.
     *  @param snk       repeated sink specific info: <name, offset>
     */
    def group (director: Model,
               prop: Property = null, xy: (Double, Double),
               snk: (String, (Double, Double))*): VEC [Sink] =
        val sinkG = VEC [Sink] ()
        for s <- snk do sinkG += Sink (s._1, director, prop, at((xy._1 + s._2._1, xy._2 + s._2._2)))
        sinkG
    end group

end Sink

