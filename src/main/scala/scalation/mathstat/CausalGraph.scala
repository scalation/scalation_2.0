
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jun 27 13:02:25 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Data Structure for Causal Graphs
 *           works either statically or dynamically (with lags)
 *  @note    AI Assisted Code
 */

package scalation
package mathstat

import scala.collection.mutable.{Map => MMap, Stack, Set => MSet, ArrayBuffer => VEC}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Define distinct types for explicit AND-OR logic
 */
enum LogicType:

    case VARIABLE
    case AND
    case OR
    case MAX
    case PRODUCT

end LogicType


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Structurally define a single node in the network.
 */
case class LogicNode (id: Int, name: String, logic: LogicType,
                      var value: Double = 0.0, weights: VEC [Double] = VEC ())

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CausalGraph` provides a Bipartite DAG Data Structure for AND-OR Causal Graphs.
 */
class CausalGraph:

    val nodes   = MMap [Int, LogicNode]()
    val adjList = MMap [Int, VEC [Int]]().withDefaultValue (VEC ())
    val parList = MMap [Int, VEC [Int]]().withDefaultValue (VEC ())

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a node to the graph.
     *  @param id     the node identifier
     *  @param name   a meaningful description of node
     *  @param logic  the logical type of the node
     */
    def addNode (id: Int, name: String, logic: LogicType): Unit =
        nodes(id) = LogicNode (id, name, logic)
    end addNode

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Conveniently register multiple nodes at once using basic types.
     *  @param nodesToAdd  the nodes to add to the graph
     */
    def addNodes (nodesToAdd: (Int, String, LogicType)*): Unit =
        for (id, name, logic) <- nodesToAdd do addNode (id, name, logic)
    end addNodes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an edge to the graph: from -> to
     *  Enforce bipartite rule: Operators shouldn't link directly to Operators
     *  @param fromId  the id of the from node
     *  @param toId    the id of the to node
     *  @param weight  the edge weight indicating level of causality
     */
    def addEdge (fromId: Int, toId: Int, weight: Double = 1.0): Unit =
        val fromNode = nodes(fromId)
        val toNode   = nodes(toId)
        
        if fromNode.logic != LogicType.VARIABLE && toNode.logic != LogicType.VARIABLE then
            throw new IllegalArgumentException ("Bipartite violation: Cannot link an Operator directly to an Operator.")
            
        if ! adjList.contains (fromId) then adjList(fromId) = VEC ()
        if ! parList.contains (toId) then   parList(toId) = VEC ()
        
        adjList(fromId) += toId
        parList(toId)   += fromId
        toNode.weights  += weight
    end addEdge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Conveniently add multiple connections at once (weight defaults to 1.0 if omitted).
     *  @param edgesToAdd  the edges to add to the graph
     */
    def addEdges (edgesToAdd: (Int, Int, Double)*): Unit =
        for (fromId, toId, weight) <- edgesToAdd do addEdge (fromId, toId, weight)
    end addEdges

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the node with the given id.
     *  @param id  the node identifier
     */
    def getNode (id: Int): LogicNode = nodes(id)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the parent nodes with the given id.
     *  @param id  the node identifier
     */
    def getParents (id: Int): Seq [Int]  = parList(id).toSeq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the child nodes with the given id.
     *  @param id  the node identifier
     */
    def getChildren (id: Int): Seq [Int] = adjList(id).toSeq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get all the node ids.
     */
    def getAllNodeIds: Seq [Int] = nodes.keys.toSeq.sorted

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Automate node generation directly from a dataset header.
     *  Maps column names sequentially to numeric IDs starting from a specified offset.
     *  and return a Map linking Column Names to their generated Node IDs.
     *  @param headers  the column headers/names
     *  @param startId  the starting id number
     */
    def addDatasetHeaders (headers: Seq [String], startId: Int = 1): Map [String, Int] =
        val columnIdMap = headers.zipWithIndex.map { (name, idx) =>
            val nodeId = startId + idx
            addNode (nodeId, name, LogicType.VARIABLE)
            name -> nodeId
        }.toMap
        columnIdMap
    end addDatasetHeaders

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the Causal AND-OR Graph Topology.
     */
    def printGraph (): Unit =
        banner ("CAUSAL AND-OR GRAPH TOPOLOGY")

        // Sort nodes numerically or topologically for cleaner rendering
        for nodeId <- getAllNodeIds do
            val node     = getNode(nodeId)
            val children = getChildren(nodeId)
            val parents  = getParents(nodeId)

            // Format Node Metadata
            print (f"Node $nodeId%02d [${node.name}]")
            print (f" (${node.logic})")
            println (f" -> Current Value: ${node.value}%.2f")

            // Print Incoming Causes (Parents)
            if parents.nonEmpty then
                print ("  └── Parents: ")
                val parentStrings = parents.zip(node.weights).map { (pId, w) =>
                    s"${getNode(pId).name} (w: $w)"
                }
                println (parentStrings.mkString (", "))

            // Print Outgoing Effects (Children)
            if children.nonEmpty then
                print ("  └── Children: ")
                val childStrings = children.map (cId => getNode(cId).name)
                println (childStrings.mkString (", "))

            println ("-" * 55)
        end for
    end printGraph

end CausalGraph


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CausalEngine` class ...
 *  @param graph  the causal graph
 */
class CausalEngine (val graph: CausalGraph):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /* Perform a topological sort using Depth-First Search (DFS).
     */
    def topologicalSort (): Seq [Int] =
        val stack   = Stack [Int]()
        val visited = MSet [Int]()

        def dfs (nodeId: Int): Unit =
            visited += nodeId
            for childId <- graph.getChildren (nodeId) do
                if ! visited.contains (childId) then dfs (childId)
            stack.push (nodeId)

        for nodeId <- graph.getAllNodeIds do
            if ! visited.contains (nodeId) then dfs (nodeId)

        stack.toSeq
    end topologicalSort

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Propagate values from root nodes down through the AND-OR logic.
     */
    def evaluateNetwork (): Unit =
        val evaluationOrder = topologicalSort ()

        for nodeId <- evaluationOrder do
            val node = graph.getNode (nodeId)
            val parentIds = graph.getParents (nodeId)

            if parentIds.nonEmpty then
                // val parentValues = parentIds.map (id => graph.getNode (id).value)
                
                // Zip parent values together with their specific edge weights
                // This converts raw parent numbers into "weighted causal signals"
                val weightedInputs = parentIds.zip (node.weights).map { (pId, weight) =>
                    graph.getNode(pId).value * weight
                }

                node.logic match
                    case LogicType.AND =>
                        // Boolean thresholding logic: all must be > 0
                        // node.value = if parentValues.forall (_ > 0.0) then 1.0 else 0.0
                        node.value = weightedInputs.product 
                        
                    case LogicType.OR =>
                        // Boolean thresholding logic: any can be > 0
                        node.value = if weightedInputs.exists (_ > 0.0) then 1.0 else 0.0
                        
                    case LogicType.MAX =>
                        // Pass downstream the dominant continuous causal signal
                        node.value = weightedInputs.max
                        
                    case LogicType.PRODUCT =>
                        // Continuous scaling interaction (X1 * X2 * ...)
                        node.value = weightedInputs.product
                        
                    case LogicType.VARIABLE =>
                        // Base structural variables simply sum their operator inputs
                        node.value = weightedInputs.sum
    end evaluateNetwork

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Audits the exact structural breakdown of a node's immediate upstream causes
     *  and calculated impacts.
     *  @param nodeId  the node identifier
     */
    def traceCausalPath (nodeId: Int): Unit =
        val node     = graph.getNode(nodeId)
        val parents  = graph.getParents(nodeId).map (id => s"${graph.getNode(id).name} (ID: $id)")
        val children = graph.getChildren(nodeId).map (id => s"${graph.getNode(id).name} (ID: $id)")

        println (s"=== Causal Audit Trail for [${node.name}] ===")
        println (s"Current Evaluated Value : ${node.value}")
        println (s"Gating Operational Logic: ${node.logic}")
        println (s"Direct Upstream Parents : ${if parents.isEmpty then "None (Root Factor)" else parents.mkString(", ")}")
        println (s"Direct Downstream Effects: ${if children.isEmpty then "None (Terminal Leaf)" else children.mkString(", ")}")
        println ("=" * 45)
    end traceCausalPath

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Bulk-evaluate an entire single dataset (MatrixD) across the causal graph.
     *  @param dataset     the MatrixD object containing rows of record data.
     *  @param headers     an array of strings matching the columns of the dataset.
     *  @param headerToId  a lookup map generated during node creation linking column names to Node IDs.
     */
    def evaluateDataset (dataset: MatrixD, headers: Array [String], headerToId: Map [String, Int]): Unit =
        
        banner (s"EVALUATING DATASET MATRIX (${dataset.dim}x${dataset.dim2})")

        // Iterate through every row index inside the ScalaTion matrix
        for i <- dataset.indices do
            // Extract the specific row as a VectorD
            val rowVector: VectorD = dataset(i)
            
            // Step A: Ingest column values into corresponding graph input nodes
            for j <- 0 until dataset.dim2 do
                val columnName = headers(j)
                
                // Only inject values if the dataset column exists inside our graph layout
                if headerToId.contains(columnName) then
                    val nodeId = headerToId(columnName)
                    graph.getNode(nodeId).value = rowVector(j)
            
            // Step B: Fire the topological network calculation for this row
            evaluateNetwork ()
            
            // Step C: Simple row trace summary of calculated metrics
            print (f"[Time Index t $i%02d Input] -> ")
            val inputs = headers.map (h => f"$h: ${graph.getNode(headerToId(h)).value}%.1f").mkString(", ")
            print (inputs)
            
            // If you have a target leaf node like 'Transmission_Rate', you can explicitly extract it here
            println ("")
        end for
    end evaluateDataset

end CausalEngine


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `causalGraphTest` main function tests the `CausalGraph` and `CausalEngine` classes.
 *  > runMain scalation.mathstat.causalGraphTest
 */
@main def causalGraphTest (): Unit =

    val graph = new CausalGraph ()

    // Bulk initialize your nodes in a clean list
    graph.addNodes ((1, "High_Temperature",   LogicType.VARIABLE),
                    (2, "Pressure_Drop",      LogicType.VARIABLE),
                    (3, "Primary_Risk_Gate",  LogicType.AND),
                    (4, "Primary_Risk_Status", LogicType.VARIABLE),  // Intermediate Variable
                    (5, "System_Wear_Level",  LogicType.VARIABLE),
                    (6, "Total_Failure_Gate", LogicType.OR),
                    (7, "Engine_Failure_Status", LogicType.VARIABLE))

    // Map out your causal connections simultaneously
    // Format: (From_ID, To_ID, Weight) -> Weight parameter can be omitted if you only pass 2 variables!
    graph.addEdges ((1, 3, 1.0),
                    (2, 3, 1.0),
                    (3, 4, 1.0),
                    (4, 5, 1.0),
                    (5, 6, 1.0),
                    (6, 7, 1.0))

    println (s"Successfully registered ${graph.getAllNodeIds.size} nodes bulk-style.")
    graph.printGraph ()

end causalGraphTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `causalGraphTest2` main function tests the `CausalGraph` and `CausalEngine` classes
 *  using the Covid-19 Dataset.
 *  > runMain scalation.mathstat.causalGraphTest2
 */
@main def causalGraphTest2 (): Unit =

    import scalation.modeling.forecasting.Example_Covid._

    val graph = new CausalGraph ()

    // Target the exogenous tracking metrics we want to correlate
    val exoColumns = Array ("icu_patients", "hosp_patients")

    // Load the companion matrix variables using Example_Covid's pre-trimmed specs
    val (xe, y) = clip (loadData (exoColumns))
    
    // Combine them into a single, unified evaluation dataset matrix
    // This creates a Matrix where Col 0 = icu, Col 1 = hosp, Col 2 = new_deaths
    val combinedDataset = xe :+ y
    val datasetHeaders  = exoColumns :+ response

    // Build our graph architecture programmatically
    val columnIdMap = graph.addDatasetHeaders (datasetHeaders.toIndexedSeq, startId = 100)
    
    graph.addNodes (
        (10, "Severity_Logic_Gate", LogicType.MAX),          // dominant hospital pressure signal
        (11, "Calculated_System_Strain", LogicType.VARIABLE)
    )

    graph.addEdges (
        (columnIdMap("icu_patients"), 10, 1.2),
        (columnIdMap("hosp_patients"), 10, 1.0),
        (10, 11, 1.5),
        (columnIdMap("new_deaths"), 11, 0.5)
    )

    // Run our batch execution engine straight through the matrix layout
    val engine = new CausalEngine (graph)
    engine.evaluateDataset (combinedDataset, datasetHeaders, columnIdMap)
    
    // Check our outcome terminal path metrics
    engine.traceCausalPath (11)
    graph.printGraph ()

end causalGraphTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `causalGraphTest3` main function tests the `CausalGraph` and `CausalEngine` classes
 *  using the Covid-19 Dataset.  Use the coefficients from an `ARX` model for the weights.
 *  > runMain scalation.mathstat.causalGraphTest3
 */
@main def causalGraphTest3 (): Unit =

    import scalation.modeling.forecasting.{AR, ARX}
    import scalation.modeling.forecasting.Example_Covid._
    import scalation.simulation.activity.CausalPetriNet

    val graph = new CausalGraph ()

    // 1. Data Loading Configuration
    val exoColumns = Array ("icu_patients", "hosp_patients")
    val (xe, y) = clip (loadData (exoColumns))

    val (p, q, s) = (3, 2, 0)
    AR.hp.set (("p", p), ("q", q), ("spec", s))

    // Train the model
    val mod = ARX (xe, y, 1)
    mod.trainNtest ()()
    val coeff = mod.parameter
    println (s"model coefficients = $coeff")

    // Register Terminal Target Node and Logic Gate
    val TARGET_ID    = 11
    val EXO_GATE_ID  = 10
    val ENDO_GATE_ID = 12

    graph.addNode (TARGET_ID, s"${response}_Current", LogicType.VARIABLE)
    graph.addNode (EXO_GATE_ID, "Exogenous_Max_Gate", LogicType.MAX)
    graph.addNode (ENDO_GATE_ID, "Endogenous_Sum_Gate", LogicType.PRODUCT)

    graph.addEdge (EXO_GATE_ID, TARGET_ID, 1.0)              // connect EXO-Gate to Target
    graph.addEdge (ENDO_GATE_ID, TARGET_ID, 1.0)             // connect ENDO-Gate to Target

    // Tracker to keep generated node IDs unique
    var currentNodeId = 20 
    var coefIndex     = 0                                    // skip index 0 ?? (the intercept/bias term)

    // LOOP 1: Process Autoregressive Lags (Y)
 
    for lag <- 1 to p do
        val nodeName = s"${response}_Lag$lag"
        val weight   = coeff(coefIndex)
        
        graph.addNode (currentNodeId, nodeName, LogicType.VARIABLE)
        graph.addEdge (currentNodeId, ENDO_GATE_ID, weight)      // endo nodes point directly to the current state node
        
        currentNodeId += 1
        coefIndex     += 1
    end for

    // LOOP 2: Process Exogenous Lags (X)

    for exoVarName <- exoColumns do
        for lag <- 1 to q do
            val nodeName = s"${exoVarName}_Lag$lag"
            val weight   = coeff(coefIndex)
            
            graph.addNode (currentNodeId, nodeName, LogicType.VARIABLE)
            graph.addEdge (currentNodeId, EXO_GATE_ID, weight)       // exog nodes collect into the logical MAX gate first
            
            currentNodeId += 1
            coefIndex     += 1
    end for

    // Print Graph Topology to verify the loops worked perfectly
    banner ("AUTOMATED ARX GRAPH INITIALIZATION COMPLETE")
    graph.printGraph ()

    val visualNet = CausalPetriNet.adapt (graph)
    visualNet.simulate (0.0, 10.0)                           // automatically launch an Animation window

end causalGraphTest3

