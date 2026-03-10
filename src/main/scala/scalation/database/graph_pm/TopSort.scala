
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, John Miller
 *  @version 2.0
 *  @date    Thu Nov 19 18:43:58 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Topological Sorting Algorithm for Directed Graphs
 */

package scalation
package database
package graph_pm

import scala.collection.mutable.{Set => SET}
import scala.collection.mutable.ArrayBuffer
import scala.runtime.ScalaRunTime.stringOf

import scalation.modeling.autograd.{Variabl, Function}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TrafficLight` object is an enumeration type for traffic light colors.
 *  Vertices are marked Green (unvisited), Yellow (processing), or Red (done with).
 */
enum TrafficLight:

    case Green, Yellow, Red

end TrafficLight

import TrafficLight._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TopSort` object provides the `topSort` method for creating a
 *  topological sort of the vertices in a directed graph.  It also perform
 *  cycle detection.
 */
object TopSort:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Topological sort that returns an edge compatible ordering of the vertices.
     *  Translated from pseudo-code and implements Tarjan's algorithm.
     *  The topological sort will contain NEGATIVE values, if there is a CYCLE.
     *  @see en.wikipedia.org/wiki/Topological_sorting
     *  @param g  the directed graph
     */
    def topSort (g: Graph): Array [Int] =
        val n       = g.size                                   // the number of vertices in g
        val color   = Array.fill (n)(Green)                    // traffic light: Green, Yellow or Red
        val vList   = Array.fill (n)(-1)                       // ordered list of vertices
        var last    = n - 1                                    // last open position in vList
        var acyclic = true                                     // assume acyclic until cycle detected

        for v <- color.indices if acyclic && color(v) == Green do dfs (v)

        /*  Recursively visit vertices, adding vertices onto a list at the end
         *  @param u  the current vertex
         */
        def dfs (u: Int): Unit =
            if acyclic then
                if color(u) == Yellow then
                    vList(last) = -2
                    acyclic = false                            // detected cycle
                else if color(u) == Green then
                    color(u) = Yellow
                    for v <- g.ch(u) do dfs (v)
                    color(u) = Red
                    if vList(last) != -2 then
                        { vList(last) = u; last -= 1 }         // prepend to front of list
            end if
        end dfs

        vList
    end topSort

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Topological sort for Autograd DAGs (Function nodes).
     *  Traverses the computation graph starting from a root `Variabl`
     *  (typically the final loss/output) and returns a topologically
     *  sorted sequence of `Function` nodes (inputs appear before the
     *  Functions that consume them).
     *  @param root  the output variable whose graph should be sorted
     *  @throws IllegalStateException if a cycle is detected
     */
    def topSortFunctions (root: Variabl): Seq [Function] =
        val visited  = SET [Function] ()                       // permanently visited (processed)
        val active   = SET [Function] ()                       // recursion stack for cycle detection
        val ordering = ArrayBuffer [Function] ()

        def dfs (fn: Function): Unit =
            if active.contains (fn) then
                throw new IllegalStateException (s"Cycle detected in autograd graph at Function node: $fn")
            if ! visited.contains (fn) then
                active += fn
                // Recurse to predecessor Functions (those that produce this fn's inputs)
                for in <- fn.inputs; gfn <- in.gradFn do dfs (gfn)
                active   -= fn
                visited  += fn
                ordering += fn                                 // post-order append yields inputs -> outputs
        end dfs

        root.gradFn.foreach (dfs)
        ordering.toSeq
    end topSortFunctions

end TopSort

import TopSort.topSort

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `topSortTest` main method tests the `TopSort` object using a directed graph.
 *  Graphs are created by passing in an array of adjacency sets (one for each vertex).
 *  > runMain scalation.database.graph_pm.topSortTest
 */
@main def topSortTest (): Unit =

    // Test graph 1

    val pg1 = new Graph (Array (SET (1, 2),          // ch(0): edges from 0:  0 -> 1, 0 -> 2
                                SET (2),             // ch(1): edges from 1:  1 -> 2
                                SET ()),             // ch(2): edges from 2:  no such edges
                         Array (10.0, 11.0, 12.0))

    banner ("Precedence Graph pg1")
    pg1.printG ()
    println ("pg1 order = " + stringOf (topSort (pg1)))

    // Test graph 2

    val pg2 = new Graph (Array (SET (1, 2),          // ch(0): edges from 0:  0 -> 1, 0 -> 2
                                SET (2),             // ch(1): edges from 1:  1 -> 2
                                SET (0)),            // ch(2): edges form 2:  2 -> 0
                         Array (10.0, 11.0, 12.0))

    banner ("Precedence Digraph pg2")
    pg2.printG ()
    println ("pg2 order = " + stringOf (topSort (pg2)))

end topSortTest

