
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed May  6 14:43:59 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Factor Graph: A Bipartite Graph with Two Types of Nodes:
 *           Variable ( ) and Factor [ ]
 *           Designed to Support Dynamic, Causal Factor Graphs
 *           extends `Place`, `Transition`, and `PetriNet`
 *
 *  Example: COVID-19 Causal Factor Graph
 *
 *            (Mobility) ↘ 
 *                         [F1] → (New_Cases) ↘
 *           (Awareness) ↗                      [F2] → (Mortality)
 *                             (Demographics) ↗ 
 *
 *  Possible Intervention: set Mobility to a lower value
 *
 *  @see     https://dl.acm.org/doi/10.1145/3705297
 *           http://yann.lecun.com/exdb/publis/pdf/mirowski-ecml-09.pdf
 */

package scalation
package simulation
package activity

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.mathstat.VectorD
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariableNode` class extends `PlaceD`, but instead of tokens/fluids, it stores
 *  the 'belief' or marginal distribution.
 *  @param name     the name of variable/feature
 *  @parm  len      the dimension of the belief vector
 *  @param x_coord  the place's x-coordinate
 *  @param y_coord  the place's y-coordinate
 */
class VariableNode (name: String, val len: Int, x_coord: Double, y_coord: Double)
      extends PlaceD (x_coord, y_coord):

    private val debug = debugf ("VariableNode", true)                    // debug function
    private [activity] var belief = new VectorD (len)                    // causal belief info
    private val incomingMessages = VEC [VectorD] ()                      // incoming messages from connected Factors

    debug ("init", s"initialize the $name variable node")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Simple belief update takes the element-wise product of all incoming messages
     *  and then normalizes to sum to 1.0.
     */
    def updateBelief (): Unit =
        belief = incomingMessages.reduce (_ * _).normalize
    end updateBelief

end VariableNode


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FactorNode` class extends `Transition`, but instead of firing tokens,
 *  it computes the causal function (AND/OR/Probabilistic).
 *  @param name     the name of factor
 *  @param logic    the mapping from input vectors (from `VariableNode`s) to output vector
 *  @param x_coord  the transition's x-coordinate
 *  @param y_coord  the transition's y-coordinate
 */
class FactorNode (name: String, val logic: Seq [VectorD] => VectorD,
                  x_coord: Double, y_coord: Double)
      extends Transition (x_coord, y_coord):
    
    private val debug = debugf ("FactorNode", true)                      // debug function

    debug ("init", s"initialize the $name factor node")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Overriding 'fire' to implement the Sum-Product algorithm logic.
     *  In a Factor Graph, 'firing' means sending a message to a Variable.
     *  @param toVariable  the variable to send the result to
     */
    def computeMessage (toVariable: VariableNode): VectorD =
        // 1. Get messages from all other connected variables
        // 2. Multiply them by the factorTable (causal mechanism)
        // 3. Marginalize out the other variables
        // For AND/OR, the factorTable acts as a deterministic constraint
        new VectorD (toVariable.len)                                     // returns the calculated message
    end computeMessage

end FactorNode


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FactorGraph` class extends `PetriNet`, and manages the collection of
 *  Nodes and the propagation cycles.
 *  @param name        the name of factor graph consisting of Variable and Factor nodes
 *  @param colors      array of colors for fluids
 *  @param placeD      array of continuous places
 *  @param transition  array of timed transitions
 */
class FactorGraph (name: String, colors: Array [Color], placeD: Array [PlaceD],
                   transition: Array [Transition])
      extends PetriNet (colors, placeD, transition):

    private val debug = debugf ("FactorGraph", true)                     // debug function

    debug ("init", s"initialize the $name factor graph")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Executes Belief Propagation:  This replaces the standard Petri Net simulation loop.
     *  @param iteration  the number of iterations
     */
    def propagate (iterations: Int = 10): Unit =
        for i <- 0 until iterations do
            // 1. All FactorNodes compute messages for their VariableNodes
            // 2. All VariableNodes update their beliefs
            println (s"Iteration $i: Propagating beliefs ...")
        end for
    end propagate

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Implements the 'Do-operator' (Intervention):  Cuts the incoming causal influence
     *  to a variable and sets it to a constant.
     *  @param variab    the variable whose belief is to adjusted
     *  @param valueIdx  the index to be set
     */
    def doIntervention (variab: VariableNode, valueIdx: Int): Unit =
        val fixedBelief = new VectorD (variab.len)
        fixedBelief(valueIdx) = 1.0
        variab.belief = fixedBelief
        // Logic to disable the 'Transition' (Factor) that normally feeds this Place
    end doIntervention

end FactorGraph

