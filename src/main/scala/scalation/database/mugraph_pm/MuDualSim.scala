
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Matthew Saltz, Supriya Ramireddy
 *  @version 2.0
 *  @date    Sun Nov  6 16:04:08 EST 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Multi-Graph `MuGraph` Dual Simulation Using Mutable Sets
 */

package scalation
package database
package mugraph_pm

import scala.collection.mutable.Map
import scala.collection.mutable.{Set => SET}
import scala.util.control.Breaks.{break, breakable}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MuDualSim` class provides an implementation for Dual Graph Simulation
 *  for multi-graphs.
 *  It differs from `DualSim` by not using inverse adjacency sets (pa) in
 *  order to save space.
 *  @param g  the data graph  G(V, E, l)
 *  @param q  the query graph Q(U, D, k)
 */
class MuDualSim (g: MuGraph, q: MuGraph)
      extends MuGraphMatcher (g, q):

    private val debug = debugf ("MuDualSim", true )                    // debug flag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the mappings φ produced by the feasibleMates method,
     *  eliminate mappings u -> v when (1) v's children fail to match u's
     *  or (2) v's parents fail to match u's.
     *  This version ignores edge labels.
     *  @see https://getd.libs.uga.edu/pdfs/saltz_matthew_w_201308_ms.pdf
     *  @param φ  array of mappings from a query vertex u to { graph vertices v }
     */
    def prune0 (φ: Array [SET [Int]]): Array [SET [Int]] =
        var alter = true
        breakable {
            while alter do                                             // check for matching children/parents
                alter = false

                for u <- qRange; u_c <- q.ch(u) do                     // for each u in q and its children u_
                    debug ("prune0", s"for u = $u, u_c = $u_c")
                    val newφ = SET [Int] ()                            // subset of φ(u_c) having a parent in φ(u)

                    for v <- φ(u) do                                   // for each v in g image of u
                        val φInt = g.ch(v) & φ(u_c)                    // children of v contained in φ(u_c)
                        if φInt.isEmpty then
                            φ(u) -= v                                  // remove vertex v from φ(u)
                            if φ(u).isEmpty then break ()              // no match for vertex u => no overall match
                            alter = true
                        // build newφ to contain only those vertices in φ(u_c) which also have a parent in φ(u)
                        newφ ++= φInt
                    end for
    
                    if newφ.isEmpty then break ()                      // empty newφ => no match
                    if newφ.size < φ(u_c).size then alter = true       // since newφ is smaller than φ(u_c)
    
                    if SELF_LOOPS && u_c == u then φ(u_c) &= newφ else φ(u_c) = newφ
                end for

            end while
        } // breakable
        φ
    end prune0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the mappings φ produced by the feasibleMates method,
     *  eliminate mappings u -> v when (1) v's children fail to match u's
     *  or (2) v's parents fail to match u's.
     *  This version checks edge labels.
     *  @param φ  array of mappings from a query vertex u to { graph vertices v }
     */
    def prune (φ: Array [SET [Int]]): Array [SET [Int]] =
        var alter = true
        breakable {
            while alter do                                             // check for matching children/parents
                alter = false

                for u <- qRange; u_c <- q.ch(u) do                     // for each u in q and its children u_
                    debug ("prune", s"for u = $u, u_c = $u_c")
                    val newφ = SET [Int] ()                            // subset of φ(u_c) having a parent in φ(u)
                    val elab_u2u_c = q.elabel ((u, u_c))               // edge label in q for (u, u_c)

                    for v <- φ(u) do                                   // for each v in g image of u
                        val v_c = g.ch(v).filter (elab_u2u_c subsetOf g.elabel (v, _))   // filter on edge labels, using subset
                        debug ("prune", s"v = $v, v_c = $v_c, φ_u_c = " + φ(u_c))

                        val φInt = v_c & φ(u_c)                        // children of v contained in φ(u_c)
                        if φInt.isEmpty then
                            φ(u) -= v                                  // remove vertex v from φ(u)
                            if φ(u).isEmpty then break ()              // no match for vertex u => no overall match
                            alter = true
                        // build newφ to contain only those vertices in φ(u_c) which also have a parent in φ(u)
                        newφ ++= φInt
                    end for
    
                    if newφ.isEmpty then break ()                      // empty newφ => no match
                    if newφ.size < φ(u_c).size then alter = true       // since newφ is smaller than φ(u_c)
    
                    if SELF_LOOPS && u_c == u then φ(u_c) &= newφ else φ(u_c) = newφ
                end for

            end while
        } // breakable
        φ
    end prune

end MuDualSim


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `muDualSimTest` main function is used to test the `MuDualSim` class.
 *  > runMain scalation.database.mugraph_pm.muDualSimTest
 */
@main def muDualSimTest (): Unit =

    val g = new MuGraph (Array (SET (1,3),                       // ch(0)
                                SET (2),                         // ch(1)
                                SET (),                          // ch(2)
                                SET (),                          // ch(3)
                                SET (2),                         // ch(4)
                                SET (4)),                        // ch(5)
                         Array (10.0, 11.0, 11.0, 11.0, 11.0, 10.0),
                         Map ((0, 1) -> SET (-1.0, -2.0, -3.0),
                              (0, 3) -> SET (-1.0),
                              (1, 2) -> SET (-1.0),
                              (4, 2) -> SET (-1.0),
                              (5, 4) -> SET (-1.0)),             // change from -1 to -2 filter out vertices
                         false, "g")

    val q = new MuGraph (Array (SET (1),                         // ch(0)
                                SET (2),                         // ch(1)
                                SET ()),                         // ch(2)
                         Array (10.0, 11.0, 11.0),
                         Map ((0, 1) -> SET (-1.0, -2.0),
                              (1, 2) -> SET (-1.0)),
                         false, "q")

    println (s"g.checkEdges   = ${g.checkEdges}")
    println (s"g.checkElabels = ${g.checkElabels}")
    g.printG ()
    println (s"q.checkEdges   = ${q.checkEdges}")
    println (s"q.checkElabels = ${q.checkElabels}")
    q.printG ()

    val matcher = new MuDualSim (g, q)                           // Dual Graph Simulation Pattern Matcher
    matcher.test ("MuDualSim")

end muDualSimTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `muDualSimTest2` main function is used to test the `MuDualSim` class.
 *  > runMain scalation.database.mugraph_pm.muDualSimTest2
 */
@main def muDualSimTest2 (): Unit =

    import mugraph_pm.{ExampleMuGraphD => EX_GRAPH}
    import MatchAnswers._

    val g = EX_GRAPH.g2p
    val q = EX_GRAPH.q2p

    println (s"g.checkEdges   = ${g.checkEdges}")
    println (s"g.checkElabels = ${g.checkElabels}")
    g.printG ()
    println (s"q.checkEdges   = ${q.checkEdges}")
    println (s"q.checkElabels = ${q.checkElabels}")
    q.printG ()

    val matcher = new MuDualSim (g, q)                           // Dual Graph Simulation Pattern Matcher
    matcher.test ("MuDualSim", shift (dualSim))

end muDualSimTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `muDualSimTest3` main function is used to test the `MuDualSim` class.
 *  > runMain scalation.database.mugraph_pm.muDualSimTest3
 */
@main def muDualSimTest3 (): Unit =

    import mugraph_pm.{ExampleMuGraphD => EX_GRAPH}

    val g = EX_GRAPH.g1p
    val q = EX_GRAPH.q1p

    println (s"g.checkEdges   = ${g.checkEdges}")
    println (s"g.checkElabels = ${g.checkElabels}")
    g.printG ()
    println (s"q.checkEdges   = ${q.checkEdges}")
    println (s"q.checkElabels = ${q.checkElabels}")
    q.printG ()

    val matcher = new MuDualSim (g, q)                           // Dual Graph Simulation Pattern Matcher
    matcher.test ("MuDualSim")

end muDualSimTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `muDualSimTest4` main function is used to test the `MuDualSim` class.
 *  > runMain scalation.database.mugraph_pm.muDualSimTest4
 *
@main def muDualSimTest4 (): Unit =

    val gSize     = 1000         // size of the data graph
    val qSize     =   10         // size of the query graph
    val nLabels   =  100         // number of distinct labels
    val gAvDegree =    5         // average vertex out degree for data graph
    val qAvDegree =    2         // average vertex out degree for query graph

    q.printG ()

    val g = genRandomGraph (gSize, nLabels, gAvDegree, false, "g")
    val q = genBFSQuery (qSize, qAvDegree, g, false, "q")

    val matcher = new MuDualSim (g, q)                           // Dual Graph Simulation Pattern Matcher
    val φ       = time { matcher.mappings () }                   // time the matcher
    matcher.showMappings (φ)                                     // display results

end muDualSimTest4
 */

