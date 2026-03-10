
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Sep 30 16:46:03 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Source Creates Entities/SimAgents
 */

package scalation
package simulation.agent

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.animation.CommandType.CreateToken
import scalation.database.graph.{Vertex, VertexType}
import scalation.mathstat.VectorD
import scalation.random.Variate
import scalation.scala2d._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Source` class is used to periodically inject entities (`SimAgent`) into a
 *  running simulation model.  Will act as an arrival generator.  Source is both
 *  a simulation `Vertex` and special `SimAgent` and therefore runs in own thread.
 *  @param name        the name of this source
 *  @param director    the director controlling the model
 *  @param _time       the activation time for this source
 *  @param iArrivalRV  the inter-arrival time distribution
 *  @param makeEntity  the function to make entities of a specified type
 *  @param units       the number of entities to make
 *  @param subtype     the subtype can be used for behavior specialization
 *  @param prop        the properties of this source
 *  @param pos         the position (Euclidean coordinates) of this source
 */
class Source (name: String, director: Model, _time: Double, iArrivalRV: Variate,
              makeEntity: () => SimAgent, units: Int, subtype: Int = 0,
              prop: Property = null, pos: VectorD = null)
      extends SimAgent (name, _time, director, pos, loc = (null, 0.0))
         with Statistical (name):

    private val debug = debugf ("Source", false)                     // debug function
            val vert  = new Vertex (name, prop, pos)                 // internal vertex
    Source.add (this)
    director.statList += this                                        // add to director's variable to keep track of
    
    debug ("init", s"name = $me, director = ${director.me}, time = $time, iArrivalRV = $iArrivalRV, " +
                   s"makeEntity = $makeEntity, units = $units, subtype = $subtype, " +
                   s"prop = $prop, pos = $pos")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compare two spatial objects based on their space coordinates.
     *  @param other  the other item to compare with this item
     */
    override def tryCompareTo [B >: Source: AsPartiallyOrdered] (other: B): Option [Int] =
        val oth = other.asInstanceOf [Source]
        pos tryCompareTo oth.pos
    end tryCompareTo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Source`s as special `SimAgent` will act over time to make entities
     *  (other `SimAgent`s).
     */
    def act (): Unit =
        var prevAgent: SimAgent = null
        for i <- 1 to units do
            debug ("act", s"$me makes SimAgent $i")
            val agent     = makeEntity ()                            // make new agent
            agent.subtype = subtype                                  // set the agent's subtype
            agent.time    = director.clock                           // set the agent's time to now
//          agent.setPos (pos(0), pos(1))                            // FIX - put at boundary, not center
            agent.setPos (pos(0)+pos(2)/2, pos(1)+pos(3)/2)          // pos at boundary

            agent.fore    = prevAgent                                // place this agent in doubly-linked list
            if prevAgent != null then prevAgent.aft = agent
            prevAgent     = agent

            director.log.trace (this, "generates", agent, director.clock)
            director.animate (agent, CreateToken, randomColor (agent.id), Ellipse ())
            director.schedule (agent)
            debug ("act", s"$me schedules $agent")

            if i < units then
                val duration = iArrivalRV.gen
                tallyStats (duration)
                time = director.clock + duration
                director.schedule (this)
                yieldToDirector ()                                   // yield and wait duration time units
            end if
        end for

        director.log.trace (this, "terminates", null, director.clock)
        yieldToDirector (true)                                       // yield and terminate
    end act

end Source


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Source` companion object establishes itself as a `VertexType` and provides
 *  a method of collecting its vertex instances.
 */
object Source
       extends VertexType ("Source", Array ("type"), color = turquoise,
                           shape = Ellipse ()):

    Model.add (Source)

            val sources = VEC [Source] ()                            // collection of sources
    private val debug   = debugf ("Source_", true)                   // debug function
    private val wh      = (22.0, 22.0)                               // default display size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the position extended with the wh = (width, height).
     *  @param xy  the coordinates for the server
     */
    def at (xy: (Double, Double)): VectorD =  VectorD (xy._1, xy._2, wh._1, wh._2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the source vertex to the Source vertex type (also add to sources).
     *  @param source  the source vertex to add
     */
    def add (source: Source): Unit =
        debug ("add", s"add vertex $source to vertex type $Source")
        verts   += source.vert
        sources += source
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a group of related sources using defaults for width 'w' and height 'h'.
     *  @param director    the director controlling the model
     *  @param time        the activation time for these sources
     *  @param makeEntity  the function to make entities of a specified type
     *  @param units       the number of entities to make
     *  @param prop        the properties of these sources
     *  @param xy          the (x, y) coordinates for the top-left corner of the reference source.
     *  @param src         repeated source specific info: <name, distribution, subtype, offset>
     */
    def group (director: Model, time: Double, makeEntity: () => SimAgent, units: Int,
               prop: Property = null, xy: (Double, Double),
               src: (String, Variate, Int, (Double, Double))*): VEC [Source] =
        val sourceG = VEC [Source] ()
        for s <- src do sourceG += Source (s._1, director, time, s._2, makeEntity, units, s._3, prop,
                                           at((xy._1 + s._4._1, xy._2 + s._4._2)))
        sourceG
    end group

end Source

