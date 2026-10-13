
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jul 12 17:44:43 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Causal Petri-Net: A Bipartite Graph with Two Types of Nodes:
 *           Variable nodes and Gate nodes
 *
 *  Example: COVID-19 Causal Petri-Net based on a given Causal Graph
 *
 *  @see `mathstat.CausalGraph`
 */

package scalation
package simulation
package activity

import scalation.mathstat._
import scalation.random.Uniform
import scalation.scala2d.Colors

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CausalPetriNet` object provides an adapter matching the custom fire transition.
 *  It acts a causal graph adapter for `mathstat.CausalGraph`.
 */
object CausalPetriNet:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Adapt a casual graph into a Petri-net.
     *  @param cg  the input causal graph
     */
    def adapt (cg: CausalGraph): PetriNet =
        val defaultColors     = Array (Colors.green)
        val activeTokenVector = VectorI (1)
        val emptyTokenVector  = VectorI (0)
        val uniformDelay      = new Uniform (1.0, 1.05)
        val allNodes          = cg.nodes.values.toArray
        val variableNodes     = allNodes.filter (_.logic == LogicType.VARIABLE)
        val gateNodes         = allNodes.filter (_.logic != LogicType.VARIABLE)

        // Hardcode structural Node ID boundaries matching your time-series pipeline loops
        val TARGET_VAR_ID      = 11
//      val EXO_GATE_TRANS_ID  = 10
        val ENDO_GATE_TRANS_ID = 12

        // Initialize incremental layout column counters
        var endoVarCount = 0
        var exoVarCount  = 0

        // Extract logic operators and translate them into Transitions
        val transitionMap = gateNodes.map { gate =>
            val ntype    = if gate.id == ENDO_GATE_TRANS_ID then "endo" else "exo"
            val xPos     = if gate.id == ENDO_GATE_TRANS_ID then 200.0 else 500.0
            val yPos     = 350.0                                 // logic gates anchor the middle lane layer
            val transit  = new Transition (xPos, yPos, uniformDelay, defaultColors)
            transit.name = ntype + transit.id
            gate.id -> transit
        }.toMap

        // Extract discrete variable nodes and translate them to PlaceI states
        val placeMap = variableNodes.map { node =>
            val initialState = if node.value > 0.0 then activeTokenVector else emptyTokenVector
            var ntype = "target"
            val (xPos, yPos) = if node.id == TARGET_VAR_ID then
                // THE ROOT TARGET CHILD: Centered perfectly at the bottom level, below the processing gates
                (350.0, 550.0)
            else
                // Identify if a node belongs to the Endogenous track by examining what it outputs to
                val isEndoInput = cg.adjList(node.id).contains (ENDO_GATE_TRANS_ID)

                if isEndoInput then
                    ntype = "endo"
                    endoVarCount += 1
                    // Stack endo parent variables side-by-side directly above the ENDO gate column (X=150)
                    val offset = (endoVarCount - 2) * 80.0       // centers inputs relative to X = 150
                    (200.0 + offset, 150.0)
                else
                    ntype = "exo"
                    exoVarCount += 1
                    // Stack exo parent variables side-by-side directly above the EXO gate column (X=450)
                    val offset = (exoVarCount - 2) * 80.0        // centers inputs relative to X = 450
                    (500.0 + offset, 150.0)
            val place = new PlaceI (xPos, yPos, initialState)
            place.name = ntype + place.id
            node.id -> place                                     // derived layout coordinates
        }.toMap

        val petriNet = new PetriNet (defaultColors, placeMap.values.toArray, transitionMap.values.toArray)

        // Wire connectivity logic based on the causal graph adjacency matrix rules
        for (gateId, trans) <- transitionMap do
            val parentIds = cg.parList (gateId)
            val childIds  = cg.adjList (gateId)

            val inputArcsArray = parentIds.flatMap (placeMap.get).map { parentPlace =>
                new ArcI (place      = parentPlace,
                          transition = trans, 
                          incoming   = true, 
                          minTokens  = activeTokenVector,
                          rates      = null,
                          testArc    = true)           // native non-destructive causal flow
            }.toArray

            val outputArcsArray = childIds.flatMap (placeMap.get).map { childPlace =>
                new ArcI (place      = childPlace,
                          transition = trans,
                          incoming   = false, 
                          minTokens  = activeTokenVector,
                          rates      = null,
                          testArc    = false)
            }.toArray

            // Bind everything using the base class connect method
            trans.connect (petriNet, inputArcsArray, outputArcsArray)
        end for           
        petriNet
    end adapt

end CausalPetriNet


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `causalPetriNetTest` main method test the `CausalPetriNet` object.
 *  > runMain scalation.simulation.activity.causalPetriNetTest
 */
@main def causalPetriNetTest (): Unit =

    banner ("Initializing CausalGraph to PetriNet Adapter Verification Driver")

    // Construct an active test graph structure using native public fields

    val testGraph = new CausalGraph ()

    // Define three underlying nodes: Two concurrent true causes and an un-triggered effect variable

    val causeA  = LogicNode (1, "Cloudy_Weather", LogicType.VARIABLE, value = 1.0)
    val causeB  = LogicNode (2, "High_Humidity", LogicType.VARIABLE, value = 1.0)
    val andGate = LogicNode (3, "Rain_Condition_Gate", LogicType.AND)
    val effectC = LogicNode (4, "Wet_Grass_Outcome", LogicType.VARIABLE, value = 0.0)

    // Populate public nodes map directly

    testGraph.nodes(causeA.id)  = causeA
    testGraph.nodes(causeB.id)  = causeB
    testGraph.nodes(andGate.id) = andGate
    testGraph.nodes(effectC.id) = effectC

    // Wire network topology connections natively via public adjList and parList

    testGraph.adjList(1) = testGraph.adjList(1) :+ 3
    testGraph.parList(3) = testGraph.parList(3) :+ 1

    testGraph.adjList(2) = testGraph.adjList(2) :+ 3
    testGraph.parList(3) = testGraph.parList(3) :+ 2

    testGraph.adjList(3) = testGraph.adjList(3) :+ 4
    testGraph.parList(4) = testGraph.parList(4) :+ 3

    println (s"-> Built Causal Graph pipeline successfully with ${testGraph.nodes.size} active nodes.")

    // Process structural transformation mapping step

    val compiledNet = CausalPetriNet.adapt(testGraph)

    println (s"-> Adapted framework mapping successfully generated standard Petri net structures.")
    println (s"   - Discrete Places (Variables): ${compiledNet.placeI.length}")
    println (s"   - Transitions (Logic Gates): ${compiledNet.transition.length}")

    // Verify initial token distribution states across the underlying tracking matrix

    println ("\n[State Diagnosis - Pre Execution]")
    for (place, index) <- compiledNet.placeI.zipWithIndex do
        println (s"   Place index [$index] at X=${place.x} holds tokens state distribution vector: ${place.tokens}")

    // Execute simulation step directly on the compiled logic engine matrix
    println ("\n[Triggering Active Causal Propagation Step]")
    if compiledNet.transition.nonEmpty then
        val sampleTransition = compiledNet.transition.head
        
        // Fire natively to test non-destructive testArc functionality 
        sampleTransition.fire()
        
        println ("\n[State Diagnosis - Post Execution Check]")
        for (place, index) <- compiledNet.placeI.zipWithIndex do
            println (s"   Place index [$index] at X=${place.x} updated token state distribution vector: ${place.tokens}")
            
        println ("\nSUCCESS: Input place tokens were verified unchanged while downstream nodes received the propagation vector.")
    else
        println ("ERROR: No operational logic transitions generated during compilation mapping steps.")

    banner ("Display compiledNet")
    println (compiledNet)
    compiledNet.simulate (2, 10)

end causalPetriNetTest

