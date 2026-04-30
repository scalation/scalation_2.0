
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep 14 14:15:51 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Animate the Motion of Objects in a Directed Graph
 */

package scalation
package animation

import scala.collection.mutable.{HashMap, ListBuffer}
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat.VectorD
import scalation.scala2d._
import scalation.scala2d.Colors.Color

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Animator` class implements the commands to create, destroy, move and scale
 *  components (nodes, edges or tokens) in an animated graph.
 *  @param graph  the directed graph to be animated
 */
class Animator (graph: Dgraph)
      extends Transform:

    /** The debug function
     */
    private val debug = debugf ("Animator", true)

    /** The flaw function
     */
    private val flaw = flawf ("Animator")

    /** Map allowing nodes to looked up based on their id's.
     */
    private val nodeMap = HashMap [Int, graph.Node] ()

    /** Map allowing edges to looked up based on their id's.
     */
    private val edgeMap = HashMap [Int, graph.Edge] ()

    /** Map allowing tokens to looked up based on their id's.
     */
    private val tokenMap = HashMap [Int, graph.Token] ()

    /** Time dilation factor for speeding up/slowing down animations
     */
    private var _timeDilationFactor = 1.0

    /*  Number of lost entities/tokens.
     */
//  private var numLost = 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the time dilation factor.
     */
    def timeDilationFactor: Double = _timeDilationFactor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a node at the given location.
     *  @param eid      the external id for the node
     *  @param shape    the shape of the node
     *  @param label    the label for the created node
     *  @param primary  whether node is primary/transition (true) or secondary/place (false)
     *  @param color    the color of the node
     *  @param pts      the coordinates and dimensions of the node
     */
//  def createNode (eid: Int, shape: RectangularShape, label: String, primary: Boolean,
    def createNode (eid: Int, shape: RectPolyShape, label: String, primary: Boolean,
                    color: Color, pts: Array [Double]): Unit =
        var node: graph.Node = null
        val npts: Int        = pts.length

        if npts == 4 then                                        // x, y, w, d
            node = new graph.Node (shape, label, primary, color, pts(0), pts(1), pts(2), pts(3))
        else
            flaw ("createNode", "for node npts = " + npts + " != 4")
            return

        graph.addNode (node)
        nodeMap.put (eid, node)
    end createNode

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an edge at the given location.
     *  @param eid       the external id for the edge
     *  @param shape     the shape (curve) of the edge
     *  @param label     the label for the created edge
     *  @param primary   whether it is a primary (true) or secondary (false)
     *  @param color     the color of the token
     *  @param from_eid  the eid of the origination node
     *  @param to_eid    the eid of the destination node
     *  @param pts       the coordinates and dimensions of the edge
     *  @param shift     amount of distance to shift the edge to accommodate,
     *                   e.g., a bundle of edges in a composite edge
     */
    def createEdge (eid: Int, shape: CurvilinearShape, label: String, primary: Boolean, color: Color,
                    from_eid: Int, to_eid: Int, pts: Array [Double], shift: Int = 0): Unit =
        var edge: graph.Edge = null
        val from: graph.Node = nodeMap.get (from_eid).getOrElse (null)
        val to:   graph.Node = nodeMap.get (to_eid).getOrElse (null)
        val npts = if pts != null then pts.length else 0

        if from == null || to == null then
           flaw ("createEdge", s"found null node - from with id $from_eid = $from  OR to with id $to_eid = $to")
           return

        if npts == 0 then
            // Create a straight line using implicit coordinates derived from node coordinates.
            edge = new graph.Edge (shape, label, primary, color, from, to)

        else if npts == 1 then
            // Create a quadratic curve using implicit coordinates derived from node coordinates.
            edge = new graph.Edge (shape, label, primary, color, from, to, pts(0), shift = shift)

        else if npts == 4 then
            // Create a straight line using explicit coordinates.
//          edge = new graph.Edge (shape, label, primary, color, from, to,
            edge = graph.Edge (shape, label, primary, color, from, to,
                               VectorD (pts(0), pts(1)), VectorD (pts(2), pts(3)), shift = shift)

        else if npts == 6 then
            // Create a quadratic curve using explicit coordinates.
//          edge = new graph.Edge (shape, label, primary, color, from, to,
            edge = graph.Edge (shape, label, primary, color, from, to,
                               VectorD (pts(0), pts(1)), VectorD (pts(2), pts(3)),
                               VectorD (pts(4), pts(5)), shift = shift)
        else
            flaw ("createEdge", s"for edges npts = $npts != 0, 1, 4 or 6")
            return

        debug ("createEdge", s"npts = $npts, edge = $edge, pts = ${stringOf (pts)}")

        from.addEdge (edge)       // add edge to from node's outgoing edges
        graph.addEdge (edge)      // add edge to the graph
        edgeMap.put (eid, edge)   // keep track of the edge in the edge map
    end createEdge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a token at the given location.
     *  @param eid      the external id for the token
     *  @param shape    the shape of the token
     *  @param label    the label for the created token
     *  @param primary  whether it is a primary (true) or secondary (false)
     *  @param color    the color of the token
     *  @param on_eid   the eid of the node it is on
     *  @param pts      the coordinates and dimensions of the token
     */
    def createToken (eid: Int, shape: RectangularShape, label: String, primary: Boolean,
                     color: Color, on_eid: Int, pts: Array [Double]): Unit =
        val default = 8.0                                   // default token size
        val npts    = if pts != null then pts.length else 0
        var token: graph.Token = null

        if primary then                                     // create a free (can go anywhere) token
            if npts == 2 then
                token = new graph.Token (shape, label, color, pts(0), pts(1), default, default)
                graph.addFreeToken (token)                  // add free token to the graph
            else if npts == 4 then
                token = new graph.Token (shape, label, color, pts(0), pts(1), pts(2), pts(3))
                graph.addFreeToken (token)                  // add free token to the graph
            else
                flaw ("createToken", s"for free token npts = $npts != 2 or 4")
                return
        else                                                // create a bound (to a node) token
            val onNode = nodeMap.get (on_eid).getOrElse (null)

            if onNode == null then
               flaw ("createToken", "onNode is null")
               return

            if npts == 0 then
                token = new graph.Token (shape, label, false, color, onNode, default, default)
                val figT = token.shape
                val figN = onNode.shape
                // when a token is added, recompute the location of all the tokens on the node
                adjustTokenLocations (onNode.tokens, figN.getCenterX (), figN.getCenterY (),
                                      figT.getWidth (), figT.getHeight ())
            else if npts == 2 then
                token = new graph.Token (shape, label, false, color, onNode, pts(0), pts(1))
                val figT = token.shape
                val figN = onNode.shape
                // when a token is added, recompute the location of all the tokens on the node
                adjustTokenLocations (onNode.tokens, figN.getCenterX (), figN.getCenterY (),
                                      figT.getWidth (), figT.getHeight ())
            else
                flaw ("createToken", s"for bound token npts = $npts != 0 or 2")
                return
        end if

        tokenMap.put (eid, token)
    end createToken
        
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Destroy the node with the given id.
     *  @param eid  the node's external id
     */        
    def destroyNode (eid: Int): Unit =
        val node = nodeMap.get (eid).getOrElse (null)

        if node == null then
           flaw ("destroyNode", s"node [eid = $eid] is null")
           return
           
        graph.removeNode (node)
        nodeMap.remove (eid)
    end destroyNode
        
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Destroy the edge with the given id.
     *  @param eid  the edges's external id
     */        
    def destroyEdge (eid: Int): Unit =
        val edge = edgeMap.get (eid).getOrElse (null)

        if edge == null then
           flaw ("destroyEdge", "edge [eid = $eid] is null")
           return
           
        graph.removeEdge (edge)
        edgeMap.remove (eid)
    end destroyEdge
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Destroy the token with the given id.
     *  @param eid  the token's external id
     */        
    def destroyToken (eid: Int): Unit =
        val token = tokenMap.get (eid).getOrElse (null)

        if token == null then
           flaw ("destroyToken", "token is null")
           return
           
        if token.primary then                    // destroy free token
            graph.freeTokens -= token
        else                                     // destroy bound token
            val onNode = token.onNode            // the node the token is on
            onNode.removeToken (token)           // remove token from current node
            val figT = token.shape
            val figN = onNode.shape
            // when a token is removed, recompute the location of all remaining tokens on the node
            adjustTokenLocations (onNode.tokens, figN.getCenterX (), figN.getCenterY (),
                                  figT.getWidth (), figT.getHeight ())
        end if

        tokenMap.remove (eid)
    end destroyToken
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move the node to a new (x, y) location.
     *  Edges cannot be moved directly, but must adjust to node movements.
     *  @param eid  the external id of the node to move
     *  @param pts  the new x, y -coordinates
     */        
    def moveNode (eid: Int, pts: Array [Double]): Unit =
        val node = nodeMap.get (eid).getOrElse (null)

        if node == null then
            flaw ("moveNode", s"node [eid = $eid] is null, unable to find token in nodeMap")
        else
            move (node.shape, pts)
            // FIX: also need to move the connected edges
    end moveNode

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move the token to a new (x, y) location.
     *  @param eid  the external id of the token to move
     *  @param pts  the new x, y -coordinates
     */        
    def moveToken (eid: Int, pts: Array [Double]): Unit =
        val token = tokenMap.get (eid).getOrElse (null)

        if token == null then
            flaw ("moveToken", "token is null, unable to find token in tokenMap")
        else if ! token.primary then
            flaw ("moveToken", "arbitrary moves not allowed for bound tokens")
        else
            move (token.shape, pts)
    end moveToken

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move the token onto the new node.
     *  @param eid       the external id of the token to move
     *  @param node_eid  the external id of the node to move onto
     */
    def moveToken2Node (eid: Int, node_eid: Int): Unit =
        val token = tokenMap.get (eid).getOrElse (null)
        moveGivenToken2Node (token, node_eid)
    end moveToken2Node

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move the given token onto the new node.
     *  @param token     the token to move
     *  @param node_eid  the external id of the node to move onto
     */
    private def moveGivenToken2Node (token: graph.Token, node_eid: Int): Unit =
        if token == null then
            flaw ("moveToken", "token is null, unable to find token in tokenMap")
        else if token.primary then
            flaw ("moveToken", "free tokens are not bound to nodes")
        else
            val oldNode = token.onNode
            val newNode = nodeMap.get (node_eid).getOrElse (null)

            if oldNode == null || newNode == null then
                flaw ("moveToken2Node", "oldNode or newNode are null")
            else
                // logically move from old node to new node
                // EXTENSION: what if the token was on an edge?
                // println ("moveGivenToken2Node: from oldNode = " + oldNode + " to newNode = " + newNode)

                oldNode.removeToken (token)
                newNode.addToken (token)
                token.setOnNode (newNode)
                val figT = token.shape

                // when a token is added, recompute the location of all the tokens on newNode
                val figN = newNode.shape
                adjustTokenLocations (newNode.tokens, figN.getCenterX (), figN.getCenterY (),
                                      figT.getWidth (), figT.getHeight ())

                // when a token is removed, recompute the location of all the tokens on oldNode
                val figO = oldNode.shape
                adjustTokenLocations (oldNode.tokens, figO.getCenterX (), figO.getCenterY (),
                                      figT.getWidth (), figT.getHeight ())
            end if
        end if
    end moveGivenToken2Node

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move number tokens of color color from node from_eid to node to_eid.
     *  @param color     the color of the tokens to move
     *  @param from_eid  the external id of the node tokens are to be taken from
     *  @param to_eid    the external id of the node the tokens are to be moved to
     *  @param pts       one dimensional array containing number of tokens to move
     */
    def moveTokens2Node (color: Color, from_eid: Int, to_eid: Int, pts: Array [Double]): Unit =
        if pts.length != 1 then
            flaw ("scaleTokensAt", "pts array should be one dimensional")
        else
            val number  = pts(0).asInstanceOf [Int]
            val oldNode = nodeMap.get (from_eid).getOrElse (null)
            val newNode = nodeMap.get (to_eid).getOrElse (null)

            if oldNode == null || newNode == null then
                flaw ("moveToken2Node", "oldNode or newNode are null")
            else
                var count  = 0
                val tokens = oldNode.tokens

                breakable {
                    for i <- 0 until tokens.length do
                        val token = tokens(i)
                        // println ("moveTokens2Node: token color " + token.color + " vs. " + color)
                        if token.color == color then
                            moveGivenToken2Node (token, to_eid)
                            count += 1
                            if count == number then break ()
                    end for
                } // end breakable

                flaw ("moveTokens2Node", s"could only find $count < $number + tokens of color $color")
            end if
        end if
    end moveTokens2Node

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Adjust the locations of all tokens on a node.
     *  @param tokens  the tokens to adjust
     *  @param xcN     the x-center of containing node
     *  @param ycN     the y-center of containing node
     *  @param w       the width of tokens
     *  @param h       the height of tokens
     */
    private def adjustTokenLocations (tokens: ListBuffer [graph.Token],
                                      xcN: Double, ycN: Double, w: Double, h: Double): Unit =
        val xc     = xcN - w / 2.0               // x-center adjusted by token radius
        val yc     = ycN - h / 2.0               // y-center adjusted by token radius
        val yDelta = h + 2.0                     // y drop between tokens 
        val nt     = tokens.size                 // number of tokens
        val yMid   = (nt - 1) / 2.0              // middle token should be mid height

        for i <- 0 until nt do
            val yy = yc + (i - yMid) * yDelta    // adjust y-coordinate tokens on node
            // println (s"adjustTokenLocations: xc = $xc, yy = $yy")
            move (tokens(i).shape, xc, yy)
        end for
    end adjustTokenLocations

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move the token along the curve for an edge and return false if at end of
     *  curve.
     *  @param eid       the external id for the token to move
     *  @param edge_eid  the external id for edge to move along
     *  @param step      the length of step
     */
    def moveToken2Edge (eid: Int, edge_eid: Int, step: Double): Boolean =
        val edge  = edgeMap.get (edge_eid).getOrElse (null)
        val token = tokenMap.get (eid).getOrElse (null)

        if edge == null || token == null then
            flaw ("moveToken2Edge", "edge or token are null")
        else
            val curve = edge.shape
            val tok   = token.shape
            val tLoc  = curve.next (tok.getWidth (), tok.getHeight ())
            if tLoc != null then                      // null => reached end of curve
                val pts = Array (tLoc.x, tLoc.y)
                moveToken (eid, pts)
                return true                           // continuing on curve
        end if

        return false                                  // exhausted the curve
    end moveToken2Edge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Scale the node, i.e., make it larger or smaller. 
     *  Edges cannot be scaled directly, but must adjust to node scaling.
     *  @param eid  the external id of the node to scale
     *  @param pts  the new width, height dimensions
     */
    def scaleNode (eid: Int, pts: Array [Double]): Unit =
        val node = nodeMap.get (eid).getOrElse (null)

        if node == null then
            flaw ("scaleNode", s"node [eid = $eid] is null")
        else
            scale (node.shape, pts)
            // FIX: also need to move the connected edges
    end scaleNode

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Scale the token, i.e., make it larger or smaller. 
     *  @param eid  the external id of the token to scale
     *  @param pts  the new width, height dimensions
     */
    def scaleToken (eid: Int, pts: Array [Double]): Unit =
        val token = tokenMap.get (eid).getOrElse (null)

        if token == null then flaw ("scaleToken", "token is null")
        else scale (token.shape, pts)
    end scaleToken
        
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Move amount tokens/fluids of color color from node from_eid to node
     *  to_eid by increasing the size of tokens at to_eid while decreasing
     *  the size at from_eid.
     *  @param color     the color of the tokens/fluids to move
     *  @param from_eid  the external id of the node tokens/fluids are to be taken from
     *  @param to_eid    the external id of the node tokens/fluids are to be moved to
     *  @param pts       one dimensional array containing amount of fluids to move
     */
    def scaleTokensAt (color: Color, from_eid: Int, to_eid: Int, pts: Array [Double]): Unit =
        if pts.length != 1 then
            flaw ("scaleTokensAt", "pts array should be one dimensional")
        else
            val amount = pts(0)
            val oldNode = nodeMap.get (from_eid).getOrElse (null)
            val newNode = nodeMap.get (to_eid).getOrElse (null)

            if oldNode == null || newNode == null then
                flaw ("scaleTokens2At", "oldNode or newNode are null")
            else
                scaleGivenTokens (oldNode.tokens, from_eid, color, -amount)  // remove fluid
                scaleGivenTokens (newNode.tokens, to_eid, color, amount)     // add fluid
        end if
    end scaleTokensAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For a given node, look for tokens/fluids by color to increase/decrease
     *  the fluids by the change amount.
     *  @param tokens  the list of tokens/fluids to search
     *  @param color   the color of the token/fluid to change
     *  @param node    the external id of the node whose tokens/fluids are to be changes
     *  @param change  the amount of change of tokens/fluids (+/-)
     */
    private def scaleGivenTokens (tokens: ListBuffer [graph.Token], node_eid: Int,
                                  color: Color, change: Double): Unit =
        var done = false
        breakable {
            for token <- tokens do                                // look through all the tokens in the list
                if token.color == color then                      // find the one with the matching color
                    val size = token.shape.getWidth () + change   // determine new width
                    if size < 0.0 then                            // not enough there to make change
                        flaw ("scaleGivenToken", s"cannot have a negative fluid level for token of color $color")
                    else if size <= 0.0 then                      // no fluid, implies should have no token
                        // destroyToken (token_eid)               // FIX: need to get the token's eid to destroy it
                        scale (token.shape, Array (0.0, 0.0))
                    else                                          // rescale the token
                        scale (token.shape, Array (size, size))
                    done = true
                    break ()                                      // mission accomplished, return now
                end if
            end for
        } // breakable

        if ! done then
            if change < 0.0 then
                flaw ("scaleTokens2At", s"cannot remove fluid, since could not find color $color")
            else if change > 0.0 then
                // No color match, so create a new token/fluid of that color
                val eid = EidCounter.next ()
                createToken (eid, Ellipse (), "t" + eid, false, color, node_eid, Array (change, change))
        end if
    end scaleGivenTokens

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the paint color for the node.
     *  @param eid    the external id for node to paint
     *  @param color  the new color for the node
     */
    def setPaintNode (eid: Int, color: Color): Unit =
        val node = nodeMap.get (eid).getOrElse (null)

        if node == null then flaw ("setPaintNode", s"node [eid = $eid] is null")
        else node.setColor (color)
    end setPaintNode

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the paint color for the edge.
     *  @param eid    the external id for node to paint
     *  @param color  the new color for the node
     */
    def setPaintEdge (eid: Int, color: Color): Unit =
        val edge = edgeMap.get (eid).getOrElse (null)

        if edge == null then flaw ("setPaintEdge", "edge [eid = $eid] is null")
        else edge.setColor (color)
    end setPaintEdge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the paint color for the token.
     *  @param eid    the external id for node to paint
     *  @param color  the new color for the node
     */
    def setPaintToken (eid: Int, color: Color): Unit =
        val token = tokenMap.get (eid).getOrElse (null)

        if token == null then flaw ("setPaintToken", "token [eid = $eid] is null")
        else token.setColor (color)
    end setPaintToken

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Adjust the time dilation: >1 slows down animation, <1 speeds up animation.
     *  @param pts  one dimensional array containing the new time dilation factor
     */
    def timeDilation (pts: Array [Double]): Unit =
        if pts.length != 1 then flaw ("timeDilation", "pts array should be one dimensional")
        else _timeDilationFactor = pts(0)
    end timeDilation 

end Animator


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EidCounter` object is used to provide unique identifiers for internally
 *  created tokens/fluids.
 */
object EidCounter:

    private var count = 1000000    // nodes (places, transitions), edges <= 1000,
                                   // tokens/fluids > 1000
                                   // internally created tokens/fluids > 1000000

    def next (): Int = { count += 1; count }

end EidCounter

