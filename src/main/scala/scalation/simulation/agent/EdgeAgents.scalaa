
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Sep 30 16:46:03 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Collection of Agents on a Edge (e.g., Link, Transport)
 */

package scalation
package simulation.agent

import scala.collection.mutable.ArrayDeque

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EdgeAgents` trait keeps track of which agents are on a particular edge,
 *  e.g., `Link`, `Transport`.  It also allows agents to be found based on their
 *  location on a edge.
 */
trait EdgeAgents:

    private val agentsOn = ArrayDeque [SimAgent] ()               // the list of entities/sim-agents on this edge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the given agent to the end of list (e.g., start of a transport).
     *  @param agent  the agent being added to the end
     */
    def add (agent: SimAgent): ArrayDeque [SimAgent] = agentsOn.addOne (agent)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the agent whose distance down the edge is at least loc.
     *  @param loc  the location at which the agent is sought (closest larger)
     */
    def indexAt (loc: Double): Int = indexWhere ((a: SimAgent) => a.loc._2 >= loc)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the given agent in the list and return its index position.
     *  @param agent  the agent whose index is sought (< 0 => not found)
     */
    def indexOf (agent: SimAgent): Int = agentsOn.indexOf (agent)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the first agent in the list that satifies the given predicate and
     *  return its index position (< 0 => not found).
     *  @param p  the predicate to be satified
     */
    def indexWhere (p: SimAgent => Boolean): Int = agentsOn.indexWhere (p)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert the given agent at the specified index position.
     *  @param agent  the agent being added at the given position
     */
    def insert (index: Int, agent: SimAgent): Unit = agentsOn.insert (index, agent)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Insert the given agent after agent2 in the list.
     *  @param agent   the new agent being added after agent2
     *  @param agent2  the existing agent to insert it after
     */
    def insertAfter (agent: SimAgent, agent2: SimAgent): Unit =
        agentsOn.insert (agentsOn.indexOf (agent2), agent)
    end insertAfter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the given agent from the list.
     *  @param agent  the agent being removed from the list
     */
    def remove (agent: SimAgent): SimAgent = agentsOn.remove (indexOf (agent))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove the first agent from the list.
     */
    def removeFirst (): SimAgent = agentsOn.remove (0)

end EdgeAgents

