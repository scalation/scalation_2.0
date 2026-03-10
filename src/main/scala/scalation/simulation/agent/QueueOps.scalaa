
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Oct  1 00:38:28 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    QueueOps Specifies Operations To Be Supported by Wait Queues
 */

package scalation
package simulation.agent

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QueueOps` trait specifies operations to be supported by wait queues.
 */
trait QueueOps:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this queue is empty.
     */
    def isEmpty: Boolean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this queue is full.
     */
    def isFull: Boolean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the length of this queue.
     */
    def length: Int

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number entities barred because of this wait-queue being full.
     */
    def barred: Int

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clear (remove all elements from) the wait queue.
     */
    def clear (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Wait in the queue, recording the waiting time.  Return whether the entity
     *  was able to actually join the queue or was barred.
     *  @param agent  the agent trying to wait in this queue
     */
    def waitIn (agent: SimAgent): Boolean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Do not wait in the queue and record zero waiting time.  Call this method
     *  to get average waiting time for all agent.  If you just want the waiting
     *  time for those who wait, do not call this method.
     *  @param agent  the agent skipping past this queue (but recording statistics)
     */
    def noWait (agent: SimAgent): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove and return the agent at the front of the queue.
     */
    def dequeue (): SimAgent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Ping the queue when leaving service, so the next agent can end its wait
     *  and likely progress to starting service now.
     */
    def ping (): Unit

end QueueOps

