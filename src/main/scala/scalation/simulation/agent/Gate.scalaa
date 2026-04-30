
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Casey Bowman
 *  @version 2.0
 *  @date    Mon Oct 11 19:46:20 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Gates Can Open (Green) and Close (Red)
 */

// Warning: For long run simulations, increase the number of cycles MAX_FLIPS.

package scalation
package simulation.agent

import scala.collection.mutable.{ArrayBuffer => VEC}
import scala.util.control.Breaks.{breakable, break}

import scalation.animation.CommandType._
import scalation.database.graph.{Vertex, VertexType}
import scalation.mathstat.VectorD
import scalation.random.Variate
import scalation.scala2d.Octagon
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Gate` class models the operation of gate that can open and closed.
 *  When the gate is open, entities can flow through and when closed, they
 *  cannot.  They may wait in a queue or go elsewhere.  A gate can model
 *  a traffic light (green => open, red => closed).
 *  @param name       the name of the gate
 *  @param director   the model/container for this gate
 *  @param time       the activation time for this gate
 *  @param line       the queue holding entities waiting for this gate to open
 *  @param onTimeRV   distribution of time that gate will be open
 *  @param offTimeRV  distribution of time that gate will be closed
 *  @param open0      whether the gate is initially closed (false) or open (true)
 *  @param cap        the maximum number of entities that will be released when the gate is opened
 *  @param prop       the properties of this gate
 *  @param pos        the position (Euclidean coordinates) of this gate
 */
class Gate (name: String, director: Model, time: Double, line: WaitQueue,
            onTimeRV: Variate, offTimeRV: Variate, open0: Boolean = true, cap: Int = 15,
            prop: Property = null, pos: VectorD = null)
      extends SimAgent (name, time, director, pos, loc = (null, 0.0))
         with Statistical (name):

    private val RELEASE_GAP = 50.0                                   // release time gap - FIX should be model dependent
    private val MAX_FLIPS   = 5000                                   // maximum number of flips before forced shut down

    private val debug = debugf ("Gate", false)                       // debug function 

    private var _open = open0                                        // initial value for _open
            val vert  = new Vertex (name, prop, pos)                 // internal vertex
    private var flips = 0                                            // the number of gate flips/cycles

    Gate.add (this)
    director.statList += this
 
    debug ("init", s"name = $me, director = ${director.id}, time = $time, line = ${line.me}, " +
                   s"onTimeRV = $onTimeRV, offTimeRV = $offTimeRV, open0 = $open0, cap = $cap, " +
                   s"prop = $prop, pos = $pos")
 
    director.schedule (this, time)                                   // start this gate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the gate is open (e.g., traffic light is red).
     */
    def open (agent: SimAgent): Boolean =
        debug ("open", s"$agent checks if $me is open = $_open")
        _open
    end open

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Release the gate after service is finished and schedule the agents in the
     *  waiting queue up to cap.
     */
    def release (): Unit =
        breakable {
            for i <- 0 until cap do
                if line.isEmpty then break ()
                val waitingAgent = line.dequeue ()
                val dtime = i * RELEASE_GAP
                director.schedule (waitingAgent, dtime)
                debug ("release", s"$me releases waiting agents $waitingAgent")
                director.log.trace (this, "releases", waitingAgent, waitingAgent.time)
            end for
        } // breakable
    end release

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Specifies how the gate is controlled, cycling through open/green to closed/red.
     */
    def act (): Unit =
        director.animate (this, SetPaintNode, gateColor, Octagon ())      // Octagon
        while flips < MAX_FLIPS && director.simulating do                 // continue until simulation is over
            debug ("act", s"SimAgent.nAgents = ${director.nAgents}")
            val dur = duration                                            // keep gate with current for duration units
            tallyStats (dur)    
            director.schedule (this, dur)                                 // delay until gate color change
            yieldToDirector ()                                            // yield to/through director

            flip ()                                                       // change gate color
            debug ("act", s"$me changes gate color to $gateColor at ${director.clock}")
            if _open then release ()
            director.animate (this, SetPaintNode, gateColor, Octagon ())  // Octagon
        end while
        yieldToDirector (true)                                            // yield to/through director
    end act

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the current color of the gate which indicates (within the animation)
     *  whether the gate is open or closed.
     */
    def gateColor: Color = if _open then green else red

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Toggles the value of open.
     */
    def flip (): Unit =
        flips += 1
        _open   = ! _open
    end flip

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns a Double for the amount of time the gate should stay open or closed
     *  based on whether or not the gate is open or closed
     */
    def duration: Double = if _open then onTimeRV.gen else offTimeRV.gen

end Gate


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Gate` companion object establishes itself as a `VertexType` and provides
 *  a method of collecting its vertex instances.
 */
object Gate
       extends VertexType ("Gate", Array ("type"), color = green,      // color changes during animation
                           shape = Octagon ()):
//                         shape = Ellipse ()):

    Model.add (Gate)

            val gates = VEC [Gate] ()                                // collection of gates
    private val debug = debugf ("Gate", false)                       // debug function
    private val wh    = (20.0, 20.0)                                 // default display size
            val delX  = wh._1 - 3                                    // delta in x direction
            val delY  = wh._2 - 3                                    // delta in y direction

    private val shift = VEC [(Double, Double)] ((-delX, 3),          // shift from center for 4 gates, see group4
                                                (-3, -delY),
                                                (delX, -3),
                                                (3, delY))
                  
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the position extended with the wh = (width, height).
     *  @param xy  the coordinates for the gate
     */
    def at (xy: (Double, Double)): VectorD =  VectorD (xy._1, xy._2, wh._1, wh._2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the gate vertex to the Gate vertex type.
     *  @param gate  the gate vertex to add
     */
    def add (gate: Gate): Unit =
        debug ("add", s"add vertex $gate to vertex type $Gate")
        verts += gate.vert
        gates += gate
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a group of related gates using defaults for width 'w' and height 'h'.
     *  @param director   the director controlling the model
     *  @param time       the activation time for these gates
     *  @param onTimeRV   distribution of time that gates will be open
     *  @param offTimeRV  distribution of time that gates will be closed
     *  @param cap        the maximum number of entities that will be released when the gate is opened
     *  @param prop       the properties of these gates
     *  @param xy         the (x, y) coordinates for the center of the gate group
     *  @param gte        repeated gate specific info: <name, line, offset>
     */
    def group (director: Model, time: Double, onTimeRV: Variate, offTimeRV: Variate, cap: Int = 15,
               prop: Property = null, xy: (Double, Double),
               gte: (String, WaitQueue, (Double, Double))*): VEC [Gate] =
        val gateG = VEC [Gate] ()
        var odd   = false
        for g <- gte do
            val gg = if odd then Gate (g._1, director, time, g._2, onTimeRV, offTimeRV, true, cap, prop,
                                       at((xy._1 + g._3._1, xy._2 + g._3._2)))
            else Gate (g._1, director, time, g._2, offTimeRV, onTimeRV, false, cap, prop,
                       at((xy._1 + g._3._1, xy._2 + g._3._2)))
            gateG += gg
            odd = ! odd
        end for
        gateG
    end group

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a group of four related gates using defaults for width 'w' and height 'h'.
     *  Relative positions predetermined based on shift (e.g., intersection with 4 lights).
     *  @param director   the director controlling the model
     *  @param time       the activation time for these gates
     *  @param onTimeRV   distribution of time that gates will be open
     *  @param offTimeRV  distribution of time that gates will be closed
     *  @param cap        the maximum number of entities that will be released when the gate is opened
     *  @param prop       the properties of these gates
     *  @param xy         the (x, y) coordinates for the center of the gate group
     *  @param gte        repeated gate specific info: <name, line, offset>
     */
    def group4 (director: Model, time: Double, onTimeRV: Variate, offTimeRV: Variate, cap: Int = 15,
               prop: Property = null, xy: (Double, Double),
               gte: (String, WaitQueue)*): VEC [Gate] =
        val gateG = VEC [Gate] ()
        var odd   = false
        var i = 0
        for g <- gte do
            val gg = if odd then Gate (g._1, director, time, g._2, onTimeRV, offTimeRV, true, cap, prop,
                                       at((xy._1 + shift(i)._1, xy._2 + shift(i)._2)))
            else Gate (g._1, director, time, g._2, offTimeRV, onTimeRV, false, cap, prop,
                       at((xy._1 + shift(i)._1, xy._2 + shift(i)._2)))
            gateG += gg
            odd = ! odd
            i  += 1
        end for
        gateG
    end group4

end Gate

