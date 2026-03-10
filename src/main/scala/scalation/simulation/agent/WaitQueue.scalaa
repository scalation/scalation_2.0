
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Oct  1 00:38:28 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    WaitQueue Provides a Place for Entities/SimAgents to Wait
 */

package scalation
package simulation.agent

import scala.collection.mutable.{ArrayBuffer => VEC}
import scala.collection.mutable.Queue

import scalation.database.graph.{Vertex, VertexType}
import scalation.mathstat.VectorD
import scalation.scala2d._
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `WaitQueue` class is a wrapper for Scala's `Queue` class, which supports
 *  First-Come, First-Serve 'FCSC' Queues.  It adds monitoring capabilities and
 *  optional capacity restrictions.  If the queue is full, entities (`SimAgent`s)
 *  attempting to enter the queue are 'barred'.  At the model level, such entities
 *  may be (1) held in place, (2) take an alternate route, or (3) be lost (e.g.,
 *  dropped call/packet). 
 *  @param name      the name of this wait-queue
 *  @param director  the `Model` directing the simulation
 *  @param cap       the capacity of the queue (defaults to unbounded)
 *  @param prop      the properties of this wait-queue
 *  @param pos       the Euclidean coordinates for this wait-queue
 */
class WaitQueue (name: String, director: Model, cap: Int = Int.MaxValue,
                 prop: Property = null, pos: VectorD = null)
      extends Vertex (name, prop, pos)
         with QueueOps
         with Statistical (name):

    WaitQueue.add (this)
    director.statList += this                                        // add to director's variable to keep track of

    private val debug   = debugf ("WaitQueue", false)                // debug function
    private val que     = Queue [SimAgent] ()                        // the actual queue
    private var nBarred = 0                                          // number of entities barred from entering when full

    debug ("init", s"name = $me, director = ${director.me}, cap = $cap, " +
                   s"prop = $prop, pos = $pos")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this queue is empty.
     */
    inline def isEmpty: Boolean = que.isEmpty

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this queue is full.
     */
    inline def isFull: Boolean = que.length >= cap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the length of this queue.
     */
    inline def length: Int = que.length

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number entities barred because of this wait-queue being full.
     */
    inline def barred: Int = nBarred

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clear (remove all elements from) the wait queue.
     */
    inline def clear (): Unit = que.clear ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Wait in the queue, recording the waiting time.  Return whether the entity
     *  was able to actually join the queue or was barred.
     *  @param agent  the agent trying to wait in this queue
     */
    def waitIn (agent: SimAgent): Boolean =
        val timeIn = director.clock
        accumStats (que.length, timeIn)                              // collect for persistent statistics
        director.log.trace (this, "wait begins", agent, timeIn)

        val joined = if isFull then
            nBarred += 1                                             // entity/agent barred
            false
        else
            que += agent                                             // entity/agent joins queue
            agent.yieldToDirector ()                                 // indefinite delay
            true
        end joined

        val timeOut = director.clock
        collectStats (timeOut - timeIn, que.length + is (joined), timeOut)
        director.log.trace (this, "wait ends", agent, timeOut)
        joined
    end waitIn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Do not wait in the queue and record zero waiting time.  Call this method
     *  to get average waiting time for all agent.  If you just want the waiting
     *  time for those who wait, do not call this method.
     *  @param agent  the agent skipping past this queue (but recording statistics)
     */
    def noWait (agent: SimAgent): Unit = tallyStats (0.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove and return the agent at the front of the queue.
     */
    def dequeue (): SimAgent = que.dequeue ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Ping the queue when leaving service, so the next agent can end its wait
     *  and likely progress to starting service now.
     */
    def ping (): Unit =
        if que != null && que.nonEmpty then director.reschedule (que.dequeue ())
//      if que.nonEmpty then director.schedule (que.dequeue (), director.clock)
    end ping

end WaitQueue


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `WaitQueue` companion object establishes itself as a `VertexType` and provides
 *  a method of collecting its vertex instances.
 */
object WaitQueue
       extends VertexType ("WaitQueue", Array ("type"), color = sandybrown,
                           shape = Rectangle ()):

    Model.add (WaitQueue)

    private val debug = debugf ("WaitQueue_", false)                 // debug function
    private val wh    = (22.0, 22.0)                                 // default display size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the position extended with the wh = (width, height).
     *  @param xy  the coordinates for the server
     */
    def at (xy: (Double, Double)): VectorD =  VectorD (xy._1, xy._2, wh._1, wh._2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the queue vertex to the WaitQueue vertex type.
     *  @param queue  the queue vertex to add
     */
    def add (queue: WaitQueue): Unit =
        debug ("add", s"add vertex $queue to vertex type $WaitQueue")
        verts += queue
    end add

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a group of related wait-queues using defaults for width 'w' and height 'h'.
     *  @param director  the `Model` directing the simulation
     *  @param cap       the capacity of these queues (defaults to unbounded)
     *  @param prop      the properties of these queues
     *  @param xy        the (x, y) coordinates for the top-left corner of the reference queue.
     *  @param que       repeated queue specific info: <name, offset>
     */
    def group (director: Model, cap: Int = Int.MaxValue,
               prop: Property = null, xy: (Double, Double),
               que: (String, (Double, Double))*): VEC [WaitQueue] =
        val queueG = new VEC [WaitQueue] ()
        for q <- que do queueG += WaitQueue (q._1, director, cap, prop, 
                                             at((xy._1 + q._2._1, xy._2 + q._2._2)))
        queueG
    end group

end WaitQueue

