

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Casey Bowman
 *  @version 2.0
 *  @date    Sat 04 Jan 2014 03:18:01 EST 
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Gates Can Open and Shut
 */

package scalation
package simulation
package process

import scala.collection.mutable.{ArrayBuffer => VEC}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{breakable, break}

import scalation.animation.CommandType._
import scalation.random.Variate
import scalation.scala2d.{Ellipse, Rectangle}
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Gate` class models the operation of gate that can open and shut.
 *  When the gate is open, entities can flow through and when shut, they
 *  cannot.  They may wait in a queue or go elsewhere.  A gate can model
 *  a traffic light (green => open, red => shut).
 *  @param name      the name of the gate
 *  @param director  the model/container for this gate
 *  @param line      the queue holding entities waiting for this gate to open
 *  @param units     number of units/phases of operation
 *  @param onTime    distribution of time that gate will be open
 *  @param offTime   distribution of time that gate will be closed
 *  @param loc       the location of the Gate (x, y, w, h)
 *  @param shut0     `Boolean` indicating if the gate is initially opened or closed
 *  @param cap       the maximum number of entities that will be released when the gate is opened
 */
class Gate (name: String, director: Model, line: WaitQueue, units: Int,
            onTime: Variate, offTime: Variate,
            loc: Array [Double], shut0: Boolean = false, cap: Int = 10)
      extends SimActor (name, director)
         with Component:

    initStats (name)
    at = loc

    private val debug = debugf ("Gate", true)                        // debug function 
    private val flaw  = flawf ("Gate")                               // flaw function
 
    private var _shut = shut0                                        // initial value for _shut

    debug ("init", s"name = $name with cor_id $cor_id, located at ${stringOf (at)}")

    if line == null then flaw ("init", "must have line for entities when gate is closed")

    schedule (0.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Auxiliary constructor that uses defaults for width 'w' and height 'h'.
     *  @param name      the name of the gate
     *  @param director  the model/container for this gate
     *  @param line      the queue holding entities waiting for this gate to open
     *  @param units     number of units/phases of operation
     *  @param onTime    distribution of time that gate will be open
     *  @param offTime   distribution of time that gate will be closed
     *  @param xy        the (x, y) coordinates for the top-left corner of the sink.
     *  @param shut0     `Boolean` indicating if the gate is initially opened or closed
     *  @param cap       the maximum number of entities that will be released when the gate is opened
     */
    def this (name: String, director: Model, line: WaitQueue, units: Int,
              onTime: Variate, offTime: Variate,
              xy: (Double, Double), shut0: Boolean, cap: Int) =
        this (name, director, line, units, onTime, offTime, Array (xy._1, xy._2, 20.0, 20.0), shut0, cap)
    end this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the gate is shut (e.g., traffic light is red).
     */
    def shut: Boolean = _shut
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tell the animation engine to display this Gate.
     */
    def display (): Unit =
        director.animate (this, CreateNode, gateColor, Ellipse (), at)
        director.animate (line, CreateNode, cyan, Rectangle (), line.at)
    end display

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Release the Gate after service is finished (also check waiting queue).
     */
    def release (): Unit =
        breakable {
            for i <- 0 until cap do
                if line.isEmpty then break ()
                val actor = director.theActor
                director.log.trace (this, "releases", actor, director.clock)
                val waitingActor = line.dequeue ()
                waitingActor.schedule (i * 500.0)
            end for
        } // breakable
    end release

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Specifies how the gate is controlled.
     */
    override def act (): Unit =
        cfor (0, units) { _ =>
            flip ()
            if ! _shut then release ()
            director.animate (this, SetPaintNode, gateColor, Rectangle (), at)            
            val dur = duration
            tally (dur)    
            schedule (dur)
            yieldToDirector ()
        } // cfor
        yieldToDirector (true)    
    end act

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the current color of the gate which indicates (within the animation)
     *  whether the gate is open or closed.
     */
    def gateColor: Color = if _shut then red else green

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Toggles the value of shut.
     */
    def flip (): Unit = _shut = ! _shut

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns a Double for the amount of time the gate should stay open or closed
     *  based on whether or not the gate is open or closed
     */
    def duration: Double = if _shut then offTime.gen else onTime.gen

end Gate


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Gate` companion object provides a builder method for gates.
 */
object Gate:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a gate using defaults for width 'w' and height 'h'.
     *  @param name      the name of the gate
     *  @param director  the model/container for this gate
     *  @param line      the queue holding entities waiting for this gate to open
     *  @param units     number of units/phases of operation
     *  @param onTime    distribution of time that gate will be open
     *  @param offTime   distribution of time that gate will be closed
     *  @param xy        the (x, y) coordinates for the top-left corner of the sink.
     *  @param shut0     `Boolean` indicating if the gate is initially opened or closed
     *  @param cap       the maximum number of entities that will be released when the gate is opened
     */
    def apply (name: String, director: Model, line: WaitQueue, units: Int, onTime: Variate, offTime: Variate,
              xy: (Int, Int), shut0: Boolean = false, cap: Int = 10): Gate =
        new Gate (name, director, line, units, onTime, offTime,
                  Array (xy._1.toDouble, xy._2.toDouble, 20.0, 20.0), shut0, cap)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a group of related gates using defaults for width 'w' and height 'h'.
     *  @param director  the director controlling the model
     *  @param units     number of units/phases of operation
     *  @param onTime    distribution of time that gate will be open
     *  @param offTime   distribution of time that gate will be closed
     *  @param xy        the (x, y) coordinates for the top-left corner of the reference gate.
     *  @param gte       repeated gate specific info: name, line, offset
     */
    def group (director: Model, units: Int, onTime: Variate, offTime: Variate, xy: (Int, Int),
               gte: (String, WaitQueue, (Int, Int))*): List [Gate] =
        val gateGroup = new VEC [Gate] ()
        var odd = false
        for g <- gte do
            gateGroup += (if odd then Gate (g._1, director, g._2, units, onTime, offTime, 
                                           (xy._1 + g._3._1, xy._2 + g._3._2), true)
                          else        Gate (g._1, director, g._2, units, offTime, onTime,
                                           (xy._1 + g._3._1, xy._2 + g._3._2), false))
            odd = ! odd
        end for
        gateGroup.toList
    end group

end Gate

