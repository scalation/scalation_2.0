
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep 21 15:05:06 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Graph Structure Suitable for Animation
 */

package scalation
package animation

import scala.collection.mutable.{HashSet, ListBuffer}
import scala.math.{abs, atan2, cos, Pi, sin}

import scalation.mathstat.VectorD
import scalation.scala2d._
import scalation.scala2d.Colors._

import Counter.{nextE, nextN}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Dgraph` class is for defining graph structures suitable for animation.
 *  Graphs consist of nodes, edges and tokens.  Tokens can be positioned within
 *  nodes or on edges.  A graph animation class that uses this class would typically
 *  move the tokens by changing there location over time.  This class supports both
 *  directed graphs and bipartite graphs.  Directed graphs contain only primary
 *  nodes, while bipartite graphs have both primary and secondary nodes along with
 *  the rule that edges must go from primaries to secondaries or secondaries to
 *  primaries.  Bipartite graphs can be used to represent Petri Nets by letting
 *  Transitions be primary nodes and Places be secondary nodes.  Everything can be
 *  labeled (nodes, edges and tokens as well as the graph itself).  Nodes and edges
 *  may be added to/removed from graphs, while tokens may be added to/removed from
 *  either nodes or edges.  Tokens may also be free (not bound to nodes or edges).
 *  @param name       the name of the graph
 *  @param bipartite  whether the graph is bipartite (edges only between 2 types of nodes)
 */
class Dgraph (name: String, bipartite: Boolean = false):

    private val debug = debugf ("Dgraph", false)                  // debug function
    private val flaw  = flawf ("Dgraph")                          // flaw function

    debug ("init", s"create graph: name = $name, bipartite = $bipartite")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Node` class is used to represent nodes in the graph.
     *  @param shape    the shape of the node
     *  @param label    the label for the created node
     *  @param primary  whether it is a primary/transition/true or secondary/place node/false
     *  @param color    the color of the node
     *  @param x        the x-coordinate (top left)
     *  @param y        the y-coordinate (top left)
     *  @param w        the width
     *  @param h        the height
     */
//  case class Node (shape: RectangularShape, label: String, primary: Boolean, var color: Color,
    case class Node (shape: RectPolyShape, label: String, primary: Boolean, var color: Color,
                     x: Double, y: Double, w: Double, h: Double):

        shape.setFrame (x, y, w, h)

        private val id       = nextN ()                      // node identifier
                val outEdges = ListBuffer [Edge] ()          // list of outgoing edges
                val tokens   = ListBuffer [Token] ()         // list of tokens current in this node

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Set (or reset) the color.
         *  @param color  the new color
         */
        def setColor (color2: Color): Unit = color = color2

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Add an outgoing edge to this node.
         *  @param edge  the edge to add
         */
        def addEdge (edge: Edge): Boolean =
            if bipartite && edge.from.primary == edge.to.primary then
                flaw ("addEdge", "node types for edge endpoints may not be the same")
                return false
            outEdges += edge
            true
        end addEdge

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Remove an outgoing edge from this node.
         *  @param edge  the edge to remove
         */
        def removeEdge (edge: Edge): Unit = outEdges -= edge

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Add a token from this node.
         *  @param token  the token to add
         */
        def addToken (token: Token): Unit = tokens += token

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Remove a token from this node.
         *  @param token  the token to remove
         */
        def removeToken (token: Token): Unit = tokens -= token

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Convert this node to a string.
         */
        override def toString: String = s"Node $label [ $id ]"

    end Node


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Edge` class is used to represent edges in the graph.  If bend = 0, a
     *  straight line is created, otherwise a quadratic curve is created.
     *  It uses implicit coordinates for the edge endpoints.
     *  @param shape    the shape (line/curve) of the edge
     *  @param label    the label for the created edge
     *  @param primary  whether it is a primary/transition/true or secondary/place node/false
     *  @param color    the color of the edge
     *  @param from     the origination node
     *  @param to       the destination node
     *  @param bend     the amount of bend in the curve (defaults to zero)
     *  @param shift    amount of distance to shift the edge, e.g., to accommodate
     *                  a bundle of edges in a composite edge
     *  @param direct   whether to directly set the line or allow factory function to set it 
     */
    case class Edge (shape: CurvilinearShape, label: String, primary: Boolean, var color: Color,
                     from: Node, to: Node, bend: Double = 0.0, shift: Int = 0, direct: Boolean = true):

        from.addEdge (this)                                           // add edge to outgoing edges of from node

        private val EPSILON = 1E-7                                    // very small real number
        private val id      = nextE ()                                // edge identifier
                val tokens  = ListBuffer [Token] ()                   // list of tokens current on this edge.
        private val p1      = VectorD (from.shape.getCenterX (), from.shape.getCenterY ())
        private val p2      = VectorD (to.shape.getCenterX (),   to.shape.getCenterY ())
        private val gap     = 5                                       // for multiple edges between one pair of nodes

        if abs (bend) > EPSILON then                                  // handle case where "def this" not called first
            shape.setLine (p1, p2, bend)
        else if direct then                                           // directly set the line (use factory methods to move)
            shape.setLine (p1, p2)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Move the edge endpoints, p1 and p2, so edge connects to node boundaries,
         *  rather than centers.  Edge is from p1 to p2:  p1 --> p2.
         *  @param p1  the initial edge starting position (center of the from node)
         *  @param p2  the initial edge ending position (center of the to node)
         */
        def move2Boundary (p1: VectorD, p2: VectorD): Unit =
            val gapShift  = shift * gap
            val frCenterX = from.x + from.w / 2                         // p1(0)
            val frCenterY = from.y + from.h / 2                         // p1(1)
            val toCenterX = to.x + to.w / 2                             // p2(0)
            val toCenterY = to.y + to.h / 2                             // p2(1)
            val angle     = atan2 (toCenterY - frCenterY, toCenterX - frCenterX)

            val radius1 = (from.shape.getWidth () + from.shape.getHeight ()) / 4.0
            p1(0) += radius1 * cos (angle + gapShift)
            p1(1) += radius1 * sin (angle + gapShift)

            val radius2 = (to.shape.getWidth() + to.shape.getHeight()) / 4.0
            p2(0) += radius2 * cos (Pi + angle - gapShift)
            p2(1) += radius2 * sin (Pi + angle - gapShift)

            println (s"move2Boundary: from p1 = $p1, to p2 = $p2")
        end move2Boundary

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Set (or reset) the curve by delegating to shape.
         *  @param p1  the first/starting point of the curve
         *  @param pc  the control point of the curve
         *  @param p1  the second/ending point of the curve
         */
        def setLine (p1: VectorD, pc: VectorD, p2: VectorD): Unit = shape.setLine (p1, pc, p2)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Set (or reset) the line by delegating to shape.
         *  @param p1  the first/starting point of the line
         *  @param p1  the second/ending point of the line
         */
        def setLine (p1: VectorD, p2: VectorD): Unit = shape.setLine (p1, p2)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Set (or reset) the color.
         *  @param color  the new color
         */
        def setColor (color2: Color): Unit = color = color2

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Add a token from this node.
         *  @param token  the token to add
         */
        def addToken (token: Token): Unit = tokens += token

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Convert this edge to a string.
         */
        override def toString: String = s"Edge $label [ $id ]"

    end Edge


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Edge` companion object provides factory mathods for creating various
     *  forms of edges.
     */
    object Edge:

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Construct an edge as a line with explicit coordinates.
         *  @param shape    the shape (line) of the edge
         *  @param label    the label for the created edge
         *  @param primary  whether it is a primary/transition/true or secondary/place node/false
         *  @param color    the color of the edge
         *  @param from     the origination node
         *  @param to       the destination node
         *  @param p1       the (x,y)-coordinates of the edge's start
         *  @param p2       the (x,y)-coordinates of the edge's end
         *  @param shift    amount of distance to shift the edge, e.g., to accommodate composite edges
         */
        def apply (shape: CurvilinearShape, label: String, primary: Boolean, color: Color,
                   from: Node, to: Node, p1: VectorD, p2: VectorD, shift: Int): Edge =
            val e = new Edge (shape, label, primary, color, from, to, 0.0, shift, false)
            e.move2Boundary (p1, p2)
            e.shape.setLine (p1, p2)
            e
        end apply

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Construct an edge as a curve with explicit coordinates.
         *  @param shape    the shape (curve) of the edge
         *  @param label    the label for the created edge
         *  @param primary  whether it is a primary/transition/true or secondary/place node/false
         *  @param color    the color of the edge
         *  @param from     the origination node
         *  @param to       the destination node
         *  @param p1       the (x,y)-coordinates of the edge's start
         *  @param pc       the (x,y)-coordinates of the edge's control point
         *  @param p2       the (x,y)-coordinates of the edge's end
         *  @param shift    amount of distance to shift the edge, e.g., to accommodate composite edges
         */
        def apply (shape: CurvilinearShape, label: String, primary: Boolean, color: Color,
                   from: Node, to: Node, p1: VectorD, pc: VectorD, p2: VectorD, shift: Int): Edge =
            val e = new Edge (shape, label, primary, color, from, to, 0.0, shift, false)
            //e.pc = pc
            e.move2Boundary (p1, p2)
//          e.shape.setLine (p1, p2)                     // no pc => straigt line
            e.shape.setLine (p1, pc, p2)                 // pc => curve
            e
        end apply

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Construct an edge as a curve with explicit coordinates.
         *  @param shape    the shape (curve) of the edge
         *  @param label    the label for the created edge
         *  @param primary  whether it is a primary/transition/true or secondary/place node/false
         *  @param color    the color of the edge
         *  @param from     the origination node
         *  @param to       the destination node
         */
        def apply (shape: CurvilinearShape, label: String, primary: Boolean, color: Color,
                   from: Node, to: Node): Edge =
            val e = new Edge (shape, label, primary, color, from, to, direct = false)
            e.move2Boundary (e.p1, e.p2)
            e.shape.setLine (e.p1, e.p2)
            e
        end apply

    end Edge


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `Token` class is used to represent tokens in the graph.
     *  @param shape    the shape of the token
     *  @param label    the label for the created token
     *  @param primary  whether the token is primary/free/true to secondary/bound/false
     *  @param color    the color of the token
     *  @param onNode   the node the token is on
     *  @param w        the width of the token
     *  @param h        the height of the token
     */
    case class Token (shape: RectangularShape, label: String, primary: Boolean, var color: Color,
                      var onNode: Node, val w: Double, val h: Double):

//      private val id = nextT ()                                // token identifier

        if onNode != null then
            onNode.addToken (this)
            val x = onNode.shape.getCenterX () - w / 2.0
            val y = onNode.shape.getCenterY () - h / 2.0
            shape.setFrame (x, y, w, h)
        end if

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Construct a primary/free token with explicit coordinates.
         *  Such tokens are free to move anywhere in the drawing panel.
         *  @param shape  the shape of the token
         *  @param label  the label for the created token
         *  @param color  the color of the token
         *  @param x      the x-coordinate of the token's location
         *  @param y      the y-coordinate of the token's location
         *  @param w      the width of the token
         *  @param h      the height of the token
         */
        def this (shape: RectangularShape, label: String, color: Color,
                  x: Double, y: Double, w: Double, h: Double) =
            this (shape, label, true, color, null, w, h)
            shape.setFrame (x, y, w, h)
        end this
 
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Set (or reset) the color.
         *  @param color  the new color
         */
        def setColor (color2: Color): Unit = color = color2

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Set the node the token is on.
         *  @param onNode2  the node the token is on
         */
        def setOnNode (onNode2: Node): Unit = onNode = onNode2

    end Token


    /** List of nodes in the graph
     */
    val nodes = ListBuffer [Node] ()

    /** List of edges in the graph
     */
    val edges = ListBuffer [Edge] ()

    /** List of free tokens in the graph (bound tokens must be in a nodes or edges list)
     */
    val freeTokens = ListBuffer [Token] ()

    /** Whether the nodes have been visited (internal use only)
     */
    private val visited = new HashSet [Node] ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a node to the graph.
     *  @param n  the node to add
     */
    def addNode (n: Node): Unit = nodes += n

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove a node from the graph.
     *  @param n  the node to remove
     */
    def removeNode (n: Node): Unit = nodes -= n

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add an edge to the graph.
     *  @param e  the edge to add
     */
    def addEdge (e: Edge): Unit = edges += e

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove an edge from the graph.
     *  @param e  the edge to remove
     */
    def removeEdge (e: Edge): Unit = edges -= e

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a free token to the graph.
     *  @param t  the free token to add
     */
    def addFreeToken (t: Token): Unit = freeTokens += t

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Remove a free token from the graph.
     *  @param t  the free token to remove
     */
    def removeFreeToken (t: Token): Unit = freeTokens -= t

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get all the root nodes (those with no incoming edges).
     */
    def getRoots: ListBuffer [Node] =
        val roots = new ListBuffer [Node] ()
        for n <- nodes do
            var keep = true
            for e <- edges if n == e.to do keep = false
            if keep then roots += n
        end for
        roots
    end getRoots

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Mark all nodes as unvisited by clearing them from the hash set.
     */
    private def clearVisited (): Unit = visited.clear ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively visit all nodes in the graph.
     *  @param n      the current node
     *  @param level  the recursion level
     */
    def traverse (n: Node, level: Int): Unit =
        cfor (0, level) { _ =>  print ("\t") }
        println (n)                              // print visited node
        //visited.add (n)
        val outgoing = n.outEdges
        if outgoing != null then
            for oEdge <- outgoing do
                val next = oEdge.to
                traverse (next, level + 1)
                // if ! visited. contains (next) then traverse (next, level + 1)
            end for
        end if
    end traverse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Traverse the graph printing out its nodes and showing connectivity by indentation.
     */
    def traverseNodes (): Unit =
        clearVisited ()
        // traverse (nodes.get (0), 0)           // only from node 0
        for r <- getRoots do traverse (r, 0)     // from all roots
    end traverseNodes

end Dgraph


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Counter` object maintains counters.
 */
object Counter:

    private var nCounter = 0
    private var eCounter = 0
    private var tCounter = 0

    def nextN (): Int = { nCounter += 1; nCounter }
    def nextE (): Int = { eCounter += 1; eCounter }
    def nextT (): Int = { tCounter += 1; tCounter }

end Counter


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dgraphTest` main function to test the `Dgraph` class.
 *  > runMain scalation.animation.dgraphTest
 */
@main def dgraphTest (): Unit =

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build and test a directed graph.
     */
    def testDirectedGraph (g: Dgraph): Unit =

        // Create nodes
        val n1 = g.Node (Ellipse (), "node1", true, red, 100, 200, 20, 20)
        val n2 = g.Node (Ellipse (), "node2", true, blue, 300, 100, 20, 20)
        val n3 = g.Node (Ellipse (), "node3", true, green, 300, 300, 20, 20)
        val n4 = g.Node (Ellipse (), "node4", true, purple, 500, 200, 20, 20)

        // Create edges
        val e1 = new g.Edge (QCurve (), "edge1", true, black, n1, n2) // 120, 210, 300, 110)
        n1.addEdge (e1)
        val e2 = new g.Edge (QCurve (), "edge1", true, black, n1, n3) // 120, 210, 300, 310)
        n1.addEdge (e2)
        val e3 = new g.Edge (QCurve (), "edge1", true, black, n2, n4) // 320, 110, 500, 210)
        n2.addEdge (e3)
        val e4 = new g.Edge (QCurve (), "edge1", true, black, n3, n4) // 320, 310, 500, 210)
        n3.addEdge (e4)

        // Add the nodes and edges to the directed graph
        g.addNode (n1)
        g.addNode (n2)
        g.addNode (n3)
        g.addNode (n4)
        g.addEdge (e1)
        g.addEdge (e2)
        g.addEdge (e3)
        g.addEdge (e4)

        // Traverse the directed graph printing out its nodes
        g.traverseNodes ()
    end testDirectedGraph

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build and test a bipartite graph.
     */
    def testBipartiteGraph (g: Dgraph): Unit =
        // Create nodes
        val n1 = g.Node (Ellipse (), "node1", false, orange, 100, 100, 30, 30)
        val n2 = g.Node (Ellipse (), "node2", false, orange, 100, 300, 30, 30)
        val n3 = g.Node (Rectangle (), "node2", true, lightgreen, 300, 185, 30, 60)
        val n4 = g.Node (Ellipse (), "node4", false, red, 500, 100, 30, 30)
        val n5 = g.Node (Ellipse (), "node5", false, red, 500, 300, 30, 30)

        // Create edges
        val e1 = new g.Edge (QCurve (), "edge1", true, black, n1, n3) // 130, 115, 300, 215)
        n1.addEdge (e1)
        val e2 = new g.Edge (QCurve (), "edge2", true, black, n2, n3) // 130, 315, 300, 215)
        n2.addEdge (e2)
        val e3 = new g.Edge (QCurve (), "edge3", true, black, n3, n4) // 330, 215, 500, 115)
        n3.addEdge (e3)
        val e4 = new g.Edge (QCurve (), "edge4", true, black, n3, n5) // 330, 215, 500, 315)
        n3.addEdge (e4)

        // Add the nodes and edges to the directed graph
        g.addNode (n1)
        g.addNode (n2)
        g.addNode (n3)
        g.addNode (n4)
        g.addNode (n5)
        g.addEdge (e1)
        g.addEdge (e2)
        g.addEdge (e3)
        g.addEdge (e4)

        // Traverse the directed graph printing out its nodes
        g.traverseNodes ()
    end testBipartiteGraph

    println ("Run DgraphTest - Bipartite Graph Test\n")
    val bg = new Dgraph ("Bipartite_Graph", true)
    testBipartiteGraph (bg)

    println ("Run DgraphTest - Directed Graph Test\n")
    val dg = new Dgraph ("Directed_Graph", false)
    testDirectedGraph (dg)

end dgraphTest

