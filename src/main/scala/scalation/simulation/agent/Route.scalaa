
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Oct 23 23:27:09 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Route for Modeling Multi-Lane Pathway
 */

package scalation
package simulation.agent

import scala.math.hypot

import scalation.database.graph.Vertex
import scalation.mathstat.VectorD
import scalation.random.Variate
import scalation.simulation.Identifiable

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Route` class provides a multi-lane pathway between two vertices (motion
 *  on multiple edges).
 *  A `Route` is a composite edge that bundles several `Transport`s.
 *  @param _name     the name of the route
 *  @param director  the `Model` directing the simulation
 *  @param k         the number of lanes/transports in the route
 *  @param from      the starting vertex
 *  @param to        the ending vertex
 *  @param moveRV    the movement random variate
 *  @param prop      the properties (Map) of this transport
 */
class Route (_name: String, director: Model, k: Int, from: Vertex, to: Vertex, 
             moveRV: Variate, prop: Property = null)
      extends Identifiable ():

    name = _name

    private val debug = debugf ("Route", true)         // debug function
    private val GAP   = 10.0                           // gap between lanes 
    private val delta = calcShift                      // amount of shift in x and y directions
    debug ("init", s"delta = $delta")

    val lane = Array.ofDim [Transport] (k)
    for i <- lane.indices do
        val shift1 = VectorD ((i - (k - 1) / 2.0) * delta(0), (i - (k - 1) / 2.0) * delta(1))
        val shift  = 1 - k + 2 * i 
        lane(i)    = new Transport (s"${name}_$i", director, from, to, moveRV, prop, shift1, shift1, shift)
    end for

    debug ("init", s"name = $me, director = ${director.me}, k = $k, from = ${from.me} to = ${to.me}, " +
                   s"moveRV = $moveRV, prop = $prop")

//    assert (1 == 2)
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Change lanes for the current lane l1 to the new lane l2 and return the
     *  agent ahead.
     *  @param agent  the agent changing lanes
     *  @param l1     the agent's current lane
     *  @param l2     the agent's new lane
     */
    def changeLane (agent: SimAgent, l1: Int, l2: Int): SimAgent =
        null
    end changeLane

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the amount of shift in the x and y directions.
     */
    private def calcShift: VectorD =
        val xdist = from.pos(0) - to.pos(0)
        val ydist = from.pos(1) - to.pos(1)
        val hyp   = hypot (xdist, ydist)
        VectorD ((ydist / hyp) * GAP, -(xdist / hyp) * GAP)
    end calcShift

end Route

