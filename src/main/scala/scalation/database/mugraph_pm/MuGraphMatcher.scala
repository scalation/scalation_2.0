
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Matthew Saltz, Aravind Kalimurthy
 *  @version 2.0
 *  @date    Wed May 13 14:58:25 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    MuGraph Pattern Matching Using Mutable Sets
 */

package scalation
package database
package mugraph_pm

import scala.collection.mutable.{Set => SET}
import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MuGraphMatcher` trait serves as a template for implementing specific
 *  algorithms for graph pattern matching.
 *  @param g  the data graph  G(V, E, l) with vertices v in V
 *  @param q  the query graph Q(U, D, k) with vertices u in U
 */
trait MuGraphMatcher (g: MuGraph, q: MuGraph):

    private   val debug      = debugf ("MuGraphMatcher", true)   // debug function
    protected val qRange     = 0 until q.size                    // range for query graph vertices
    protected val gRange     = 0 until g.size                    // range for data graph vertices
    protected val CHECK      = 1024                              // check progress after this many matches
    protected val LIMIT      = 1E7                               // quit after too many matches
    protected val SELF_LOOPS = false                             // whether the directed graph has self-loops

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply a graph pattern matching algorithm to find the mappings from the
     *  query graph q to the data graph g.  These are represented by a
     *  multi-valued function φ that maps each query graph vertex u to a
     *  set of data graph vertices {v}.
     *  @param ignoreEdgeLabels  whether to ingore edge labels during matching
     */
    def mappings (ignoreEdgeLabels: Boolean = false): Array [SET [Int]] =
        val φ = feasibleMates
        if ignoreEdgeLabels then prune0 (φ)
        else prune (φ)
    end mappings

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an initial array of feasible mappings φ from each query
     *  vertex u to the corresponding set of data graph vertices {v} whose
     *  label matches u's.
     */
    def feasibleMates: Array [SET [Int]] =
        val φ = Array.ofDim [SET [Int]] (q.size)
        for u <- qRange do φ(u) = g.labelMap (q.label(u)).clone
        debug ("feasibleMates", s"φ = ${stringOf (φ)}")
        φ
    end feasibleMates

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the mappings φ produced by the feasibleMates method,
     *  prune mappings u -> v where v's children fail to match u's.
     *  This version ignores edge labels.
     *  @param φ  array of mappings from a query vertex u to { graph vertices v }
     */
    def prune0 (φ: Array [SET [Int]]): Array [SET [Int]]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the mappings φ produced by the feasibleMates method,
     *  prune mappings u -> v where v's children fail to match u's.
     *  This version checks edge labels.
     *  @param φ  array of mappings from a query vertex u to { graph vertices v }
     */
    def prune (φ: Array [SET [Int]]): Array [SET [Int]]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the data graph by consider only those vertices and edges which
     *  are part of feasible matches after performing initial dual simulation.
     *  Note, used by strict and tight simulation.
     *  @param φ  mappings from a query vertex u_q to { graph vertices v_g }
     */
    def filterGraph (φ: Array [SET [Int]]): MuGraph =
        val newCh  = Array.fill [SET [Int]] (g.size) (SET [Int] ())
        val newCh2 = Array.ofDim [SET [Int]] (g.size)
        val nodesInSimset = φ.flatten.toSet                      // get all the vertices of feasible matches
        for i <- gRange do newCh2(i) = g.ch(i) & nodesInSimset   // prune via intersection

        for u <- qRange; w <- φ(u) do                            // new ch and pa for data graph based upon feasible vertices
            for v <- q.ch(u) do newCh(w) |= (newCh2(w) & φ(v))
        end for
        new MuGraph (newCh, g.label, g.elabel, g.inverse, g.name + "2")    // create a new data graph
    end filterGraph

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show all mappings between query graph vertices u_i and their sets of
     *  data graph vertices {v}.
     *  @param φ  the set-valued mapping function
     */
    def showMappings (φ: Array [SET [Int]]): Unit =
        println ("query u \t--> graph {v}")
        for i <- φ.indices do println (s"u_$i \t--> ${φ(i)}")
    end showMappings

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the number of mappings between query graph vertices u_i and their sets
     *  of data graph vertices {v}, giving the number of distinct vertices and edges.
     *  @param φ  the set-valued mapping function
     */
    def countMappings (φ: Array [SET [Int]]): (Int, Int) =
        val distVertices = SET [Int] ()
        val distEdges    = SET [(Int, Int)] ()
        for i <- φ.indices do distVertices ++= φ(i)
        for i <- φ.indices do
            for v <- φ(i); v_c <- g.ch(v) if distVertices contains v_c do distEdges += ((v, v_c))
        end for
        (distVertices.size, distEdges.size)
    end countMappings

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the Graph Pattern Matcher.
     *  @param mName   the name of graph pattern matcher
     *  @param ans     the correct answer
     *  @param ignore  whether to ignore the edge labels during matching
     */
    def test (name: String, ans: Array [SET [Int]] = null,
             ignore: Boolean = false): Array [SET [Int]] =
        val φ = time { mappings (ignore) }                                            // time the matcher
        banner (s"query ${q.name} (${q.size}) via $name on data ${g.name} (${g.size})")
        showMappings (φ)                                                               // display results
        println (s"(#distVertices, #distEdges) = ${countMappings (φ)}")

        if ans != null then
            for i <- φ.indices do println (s"$i: ${φ(i)} == ? ${ans(i)}")
            for i <- φ.indices do assert (φ(i) == ans(i))
        φ
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply a graph pattern matching algorithm to find subgraphs of data graph
     *  g that isomorphically match query graph q.  These are represented
     *  by a set of single-valued bijections {psi} where each psi function
     *  maps each query graph vertex u to a data graph vertices v.
     */
    def bijections (): SET [Array [Int]] = throw new UnsupportedOperationException ()

end MuGraphMatcher

