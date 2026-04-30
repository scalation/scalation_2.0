
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Dec 12 13:11:30 EST 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Discrete-time Markov Chain
 */

package scalation
package simulation
package state

import scala.math.{cos, Pi, sin}
import scala.runtime.ScalaRunTime.stringOf

import scalation.animation.{AnimateCommand, DgAnimator}
import scalation.animation.CommandType._
import scalation.mathstat._
import scalation.mathstat.MatrixD.eye
import scalation.random.Discrete
import scalation.scala2d.{Ellipse, QArrow}
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MarkoveChain` class supports the creation and use of Discrete-Time Markov Chains
 *  (DTMC)s.  Transient solution: compute the next state p = π * a where π is
 *  the current state probability vector and a is the transition probability matrix.
 *  Equilibrium solution (steady-state): solve for π in π = π * a.
 *  @param a  the transition probability matrix
 */
class MarkovChain (a: MatrixD):

    private val debug = debugf ("MarkovChain", true)                              // debug function
    private val flaw  = flawf ("MarkovChain")                                     // flaw function

    if ! isStochastic then flaw ("init", "transition matrices must be stochastic")

    private val EPSILON = 1E-7                                                    // number close to zero
    private val radius  = 200                                                     // radius of the circle that the nodes are displayed on
    private val xCenter = radius + 100                                            // x-coordinate of the center of the circle
    private val yCenter = radius + 100                                            // y-coordinate of the center of the circle
    private val size    = 30 // The size/diameter of a node
    private val dgAni   = new DgAnimator ("Markov Chain Animator", black, white)  // animation engine
    private val aniQ    = dgAni.getCommandQueue                                   // animation engine's command queue
    private val bend    = .25                                                     // amount of bend in the `QArrow`

    private val animating = true                                                  // animation flag (set to false to turn off animation)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the next probabilistic state π * a.
     *  @param π  the current state probability vector
     */
    def next (π: VectorD): VectorD = π *: a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the k-th next probabilistic state π * a^k.
     *  @param π  the current state probability vector
     *  @param k  compute for the k-th time-step/epoch
     */
    def next (π: VectorD, k: Int): VectorD =
        var p = π.copy
        cfor (0, k) { _ =>  p = p *: a }
        p
    end next

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the limiting probabilistic state π * a^k as k -> infinity, by
     *  solving a left eigenvalue problem: π = π * a => π * (a - I) = 0, where the
     *  eigenvalue is 1.  Solve for π by computing the left nullspace of the a - I
     *  matrix and then normalize π so it adds to 1.
     */
    def limit: VectorD =
        val fac = new Fac_QR ((a - eye (a.dim, a.dim)).transpose, true)
        fac.nullspace (a.dim-1)(?, 0).toProbability
    end limit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Simulate the discrete-time Markov chain, by starting in state i0 and after
     *  the state's holding, making a transition to the next state according to the
     *  jump matrix.
     *  @param i0       the initial/start state
     *  @param endTime  the end time for the simulation
     */
    def simulate (i0: Int, endTime: Int): Unit =
        var clock      = 0                                          // current discrete time
        var i          = i0                                         // current state = start state
        var absorbed   = false                                      // whether it has entered an absorbing state
        val tk_id      = a.dim                                      // the identifier for the token
        val ms_per_sec = 1000.0                                     // 1000 milliseconds per second (animate using seconds)

        animate ()
        aniQ.add (AnimateCommand (CreateToken, tk_id, Ellipse (), "tk" + tk_id, false, black, null, 0.0, i0))

        println (s"simulate: start simulation of Discrete-Time Markov Chain at time $clock")
        println (s"simulate: at time $clock the state is $i")

        while clock < endTime && ! absorbed do
            if a(i, i) =~ 1.0 then
                absorbed = true
                debug ("simulate", s"entered absorbing state $i")
            else
                clock    += 1
                val rowi  = a(i)
                println ("rowi = " + rowi)
                val disRV = Discrete (rowi)
                i         = disRV.igen                              // advance to the next state
            end if
            aniQ.add (AnimateCommand (MoveToken2Node, tk_id, null, null, false, null, null, ms_per_sec * clock, i))
            debug ("simulate", s"at time $clock the state is $i")
        end while

        dgAni.animate (0, ms_per_sec * endTime)
        println (s"simulate: end simulation of Discrete-Time Markov Chain at time $clock")

    end simulate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Animate this Markov Chain.  Place the nodes around a circle and connect them
     *  if there is a such a transition.
     */
    def animate (): Unit =
        if animating then
            val n = a.dim                                           // number of nodes to create
            val ncolor = lightblue                                  // color for nodes
            val ecolor = red                                        // color for edges

            //:: Display the nodes for the Markov Chain

            for i <- a.indices do
                val theta = -Pi + 2.0 * Pi * (i / n.toDouble)
                val shape = Ellipse ()
                val label = "n" + i
                val at    = Array (xCenter + radius * cos (theta),
                                   yCenter + radius * sin (theta), size, size)
                debug ("animate", s"$label.$i CreateNode $ncolor $shape ${stringOf (at)}")
                aniQ.add (AnimateCommand (CreateNode, i, shape, label, true, ncolor, at, 0.0))
            end for

            //:: Display the edges for the Markov Chain

            for i <- a.indices; j <- a.indices2 if a(i, j) > EPSILON do
                val eid   = n * (i + 1) + j
                val shape = QArrow ()
                val label = "" + a(i, j)
                debug ("animate", s"$label.$eid CreateEdge $ecolor $shape $i -> $j")
                if i == j then
                    aniQ.add (AnimateCommand (CreateEdge, eid, shape, label, true, ecolor,
                                              Array (16.0 * bend), 0.0, i, j))
                else
                    aniQ.add (AnimateCommand (CreateEdge, eid, shape, label, true, ecolor,
                                              Array (bend), 0.0, i, j))
            end for
        end if
    end animate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the transition matrix is stochastic (i.e., square, nonnegative,
     *  and rows sum to one).
     */
    def isStochastic: Boolean =
        if a.dim != a.dim2 || ! a.isNonnegative then return false

        var (go, i) = (true, 0)
        cfor (go && i < a.dim, i += 1) {
            if ! (a(i).sum =~ 1.0) then
                println (s"row $i sums to ${a(i).sum}")
                go = false
        } // cfor
        go
    end isStochastic

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this discrete-time Markov Chain to a string.
     */ 
    override def toString: String = s"MarkoveChain($a)"
   
end MarkovChain


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `markovChainTest` main function tests the `MarkovChain` class (Discrete-Time Markov Chains).
 *  > runMain scalation.simulation.state.markovChainTest
 */
@main def markovChainTest (): Unit =

    val endTime = 20          // number of epochs (milliseconds), but may represent any time unit

    val mc = new MarkovChain (MatrixD ((4, 4), .4, .6, .0, .0,    // 4-by-4 matrix
                                               .0, .2, .8, .0,
                                               .3, .0, .5, .2,
                                               .1, .0, .7, .2))
    var π = VectorD (1.0, 0.0, 0.0, 0.0)

    println ("\nDiscrete-Time Markov Chain mc = " + mc + "\n")

    banner ("Discrete-Time Markov Chain: transient solution:")
    println ("\nON epoch 2,\tπ = " + mc.next (π, 2))
    println ("\non epoch 0,\tπ = " + π)

    for k <- 1 to endTime do
        π = mc.next (π)
        println (s"on epoch $k,\tπ = $π")
    end for

    println ("\nDiscrMarkovChainTest2ete-Time Markov Chain: steady-state solution:")
    banner ("steady-state \tπ = " + mc.limit)

    println ("\nDiscrete-Time Markov Chain: simulation:")
    mc.simulate (0, endTime)
   
end markovChainTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `markovChainTest2` main function tests the `MarkovChain` class (Discrete-Time Markov Chains).
 *  Six-state coin game.
 *  > runMain scalation.simulation.state.markovChainTest2
 */
@main def markovChainTest2 (): Unit =

    val endTime = 50                                                // number of epochs/time-steps
    val p       = 0.5                                               // probability of a head
    val q       = 1 - p
    val k       = 3                                                 // enter game with k dollars

    val mc = new MarkovChain (MatrixD ((6, 6), 1, 0, 0, 0, 0, 0,    // 6-by-6 matrix
                                               q, 0, p, 0, 0, 0,
                                               0, q, 0, p, 0, 0,
                                               0, 0, q, 0, p, 0,
                                               0, 0, 0, q, 0, p,
                                               0, 0, 0, 0, 0, 1))
    var π = new VectorD (6); π(k) = 1

    println ("\nDiscrete-Time Markov Chain mc = " + mc + "\n")

    banner ("Discrete-Time Markov Chain: transient solution:")
    println ("\non epoch 0,\tπ = " + π)

    for k <- 1 to endTime do
        π = mc.next (π)
        println (s"on epoch $k,\tπ = $π")
    end for

    banner ("steady-state \tπ = " + mc.limit)

end markovChainTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `markovChainTest3` main function tests the `MarkovChain` class (Discrete-Time Markov Chains).
 *  Six-state coin game: Tail (-1), Head (+1).
 *  > runMain scalation.simulation.state.markovChainTest3
 */
@main def markovChainTest3 (): Unit =

    import scalation.random.Bernoulli

    val p    = 0.5                                                  // probability of a head
    val coin = Bernoulli (p)                                        // RVG for Bernoulli distribution, coin flip
    var lose = 0                                                    // number of games lost
    var win  = 0                                                    // number of games won

    cfor (0, 10000) { _ =>                                          // iterate playing the game
        var j  = 3                                                  // enter game with j dollars
        var go = true                                               // continue with the game
        while go do
            j += (if coin.gen < 0.5 then -1 else 1)
            if j == 0 then { println ("lose"); lose += 1; go = false }
            if j == 5 then { println ("win");  win  += 1; go = false }
        end while
    } // cfor
    println (s"loses = $lose")
    println (s"wins  = $win")

end markovChainTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `markovChainTest4` main function tests the `MarkovChain` class.
 *  @see Introduction to Probabulity Models, 3rd Ed., Ross, p. 146.
 *  > runMain scalation.simulation.state.markovChainTest4
 */
@main def markovChainTest4 (): Unit =

    val a  = MatrixD ((3, 3), .5, .4, .1,                           // 3-by-3 matrix
                              .3, .4, .3,
                              .2, .3, .5)

    val mc = new MarkovChain (a)
    println ("Discrete-Time Markov Chain mc = " + mc + "\n")
    banner ("Discrete-Time Markov Chain: transient solution:")

    var π  = VectorD (.5, .5, 0)
    println ("on epoch 0,\tπ = " + π)
    for k <- 1 to 10 do
        π = mc.next (π)
        println (s"on epoch $k,\tπ = $π")
    end for

    banner ("eigenvector solution for steady-state \tπ = " + mc.limit)

    val aa = MatrixD ((3, 3), -.5,  .3, .2,
                               .4, -.6, .3,
                               1,   1,  1 )
    val b  = VectorD (0, 0, 1)
    val lu = new Fac_LU (aa).factor ()
    banner ("lu factorization for steady-state \tπ = " + lu.solve (b))

end markovChainTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `markovChainTest5` main function tests the `MarkovChain` class.
 *  A simple SEIR model: 40, 8, 10 days between S->E, E->I, I-R.
 *  > runMain scalation.simulation.state.markovChainTest5
 */
@main def markovChainTest5 (): Unit =

    val a  = MatrixD ((4, 4), .95, .05, 0,     0,
                              0,   .875, .125, 0,
                              0,   0,    .9,   .1,
                              0,   0,    0,    1)

    val mc = new MarkovChain (a)
    println ("Discrete-Time Markov Chain mc = " + mc + "\n")
    banner ("Discrete-Time Markov Chain: transient solution:")

    var π  = VectorD (.99, .00, .01, 0)
    println ("on epoch 0,\tπ = " + π)
    for k <- 1 to 100 do
        π = mc.next (π)
        println (s"on epoch $k,\tπ = $π")
    end for

end markovChainTest5

