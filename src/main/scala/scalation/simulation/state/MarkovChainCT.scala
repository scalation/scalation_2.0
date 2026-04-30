
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Dec 12 13:11:30 EST 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Continuous-Time Markov Chains
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
import scalation.random.{Discrete, Exponential}
import scalation.scala2d.{Ellipse, QArrow}
import scalation.scala2d.Colors._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MarkovCT` class supports the creation and use of Continuous-Time Markov Chains
 *  'CTMC's.  Note: the transition matrix 'tr' gives the state transition rates
 *  off-diagonal.  The diagonal elements must equal minus the sum of the rest
 *  of their row.  Transient solution: Solve the Chapman-Kolmogorov differential
 *  equations.  Equilibrium solution (steady-state): solve for p in p * tr = 0.
 *  @see www.math.wustl.edu/~feres/Math450Lect05.pdf
 *  @param tr  the transition rate matrix
 */
class MarkovCT (tr: MatrixD):

    private val debug     = debugf ("MarkovCT", true)          // debug function
    private val flaw      = flawf ("MarkovCT")                 // flaw function

    private val EPSILON   = 1E-7                               // number close to zero
    private val radius    = 200                                // radius of circle for nodes
    private val xCenter   = radius + 100                       // x-coordinate of center of circle
    private val yCenter   = radius + 100                       // y-coordinate of center of circle
    private val size      = 30                                 // size/diameter of a node
    private val bend      = .25                                // amount of bend in `QArrow`

    private val animating = true                               // animation flag (false => turn off)

    /** The jump matrix derived from the transition rate matrix 'tr'
     */
    val jump = new MatrixD (tr.dim, tr.dim2)
   
    if tr.dim != tr.dim2 then flaw ("init", "transition rate matrices must be square")
    for i <- jump.indices do
        val s = tr(i).sum - tr(i, i)                           // sum the ith row of tr skipping i
        for j <- jump.indices2 do
            if i != j then                                     // off-diagonal
                jump(i, j) = if s =~ 0.0 then 0.0 else tr(i, j) / s
            else                                               // on-diagonal
                jump(i, i) = if s =~ 0.0 then 1.0 else 0.0
        end for
    end for

    /** The animation engine
     */
    private val dgAni = new DgAnimator ("Continuous-Time Markov Chain Animator", black, white)

    /** The animation engine's command queue
     */
    private val aniQ = dgAni.getCommandQueue

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the next probabilistic state at t time units in the future.
     *  @param p  the current state probability vector
     *  @param t  compute for time t
     */
    def next (p: VectorD, t: Double = 1.0): VectorD = ???      // FIX, not implemented yet

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the limiting probabilistic state as t -> infinity, by finding the
     *  left nullspace of the tr matrix: solve for p such that p * tr = 0 and
     *  normalize p, i.e., ||p|| = 1.
     */
    def limit: VectorD =
        val fac = new Fac_QR ((tr - eye (tr.dim, tr.dim)).transpose, true)
        fac.nullspace (tr.dim-1)(?, 0).toProbability
    end limit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Simulate the continuous-time Markov chain, by starting in state 'i0' and after
     *  the state's holding, making a transition to the next state according to the
     *  jump matrix.
     *  @param i0       the initial/start state
     *  @param endTime  the end time for the simulation
     */
    def simulate (i0: Int, endTime: Double): Unit =
        var clock      = 0.0                                   // current continuous time
        var i          = i0                                    // current state = start state
        var absorbed   = false                                 // whether it has entered an absorbing state
        val tk_id      = tr.dim                                // the identifier for the token
        val ms_per_sec = 1000.0                                // 1000 milliseconds per second (animate using seconds)

        animate ()
        aniQ.add (AnimateCommand (CreateToken, tk_id, Ellipse (), "tk" + tk_id, false, black, null, 0.0, i0))

        println ("simulate: start simulation of Continuous-Time Markov Chain at time " + clock)
        println ("simulate: at time " + clock + " the state is " + i)

        while clock < endTime && ! absorbed do
            val tr_i = - tr(i, i)                              // holding rate for state i
            if tr_i =~ 0.0 then
                absorbed = true
                debug ("simulate", s"entered absorbing state $i")
            else
                val expRV = Exponential (tr_i)
                clock    += expRV.gen                          // add holding time for state i
                val rowi  = jump(i)
                debug ("simulate", s"rowi = $rowi")
                val disRV = Discrete (rowi)
                i         = disRV.igen                         // advance to the next state
            end if
            aniQ.add (AnimateCommand (MoveToken2Node, tk_id, null, null, false, null, null, ms_per_sec * clock, i))
            debug ("simulate", s"at time $clock the state is $i")
        end while

        dgAni.animate (0, ms_per_sec * endTime)
        println ("simulate: end simulation of Continuous-Time Markov Chain at time " + clock)

    end simulate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Animate 'this' continuous-time Markov Chain.  Place the nodes around a circle
     *  and connect them if there is a such a transition.
     */
    def animate (): Unit =
        if animating then
            val n = tr.dim                                     // number of nodes to create

            //:: Display the nodes for the continuous-time Markov Chain

            for i <- 0 until n do
                val theta = -Pi + 2.0 * Pi * (i / n.toDouble)
                val shape = Ellipse ()
                val label = "n" + i
                val color = lightblue
                val at    = Array (xCenter + radius * cos (theta),
                                   yCenter + radius * sin (theta), size, size)
                println (s"MarkovCT.animate: $label.$i  CreateNode $color $shape ${stringOf (at)}")
                aniQ.add (AnimateCommand (CreateNode, i, shape, label, true, color, at, 0.0))
            end for

            //:: Display the edges for the continuous-time Markov Chain

            for i <- 0 until n do
                for j <- 0 until n if i != j && tr(i, j) > EPSILON do
                    val eid   = n * (i + 1) + j
                    val shape = QArrow ()
                    val label = "" + tr(i, j)
                    val color = red
                    println (s"MarkovCT.animate: $label.$eid CreateEdge $color $shape $i $j")
                    aniQ.add (AnimateCommand (CreateEdge, eid, shape, label, true, color,
                                              Array (bend), 0.0, i, j))
                end for
            end for
        end if
    end animate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this continuous-time Markov Chain to a string.
     */
    override def toString: String = s"MarkovCT($tr)"

end MarkovCT


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `markovCTTest` main function tests the `MarkovCT` class (Continuous-Time Markov Chains).
 *  > runMain scalation.simulation.state.markovCTTest
 */
@main def markovCTTest (): Unit =

    val endTime = 200.0       // number of time units (e.g. in milliseconds)

    val mc = new MarkovCT (MatrixD ((2, 2), -4.0,  4.0,        // 2-by-2 matrix
                                             5.0, -5.0))

    println ("\nContinuous-Time Markov Chain mc = " + mc + "\n")
    println ("\nContinuous-Time Markov Chain: transient solution:")
//  println ("\nAT STEP 1,\tp = " + mc.next (p, 1))

    println ("\nContinuous-Time Markov Chain: steady-state solution:")
    println ("\njump matrix  \tj = " + mc.jump)
    println ("\nsteady-state \tp = " + mc.limit)

    val mc2 = new MarkovCT (MatrixD ((6, 6), -2.0, 1.0,  0.0,  1.0,  0.0, 0.0,   // 6-by-6 matrix
                                              0.0, 0.0,  0.0,  0.0,  0.0, 0.0,
                                              0.0, 1.0, -4.0,  0.0,  0.0, 3.0,
                                              2.0, 0.0,  0.0, -4.0,  2.0, 0.0,
                                              0.0, 3.0,  1.0,  0.0, -5.0, 1.0,
                                              0.0, 0.0,  0.0,  0.0,  0.0, 0.0))

    println ("\nContinuous-Time Markov Chain mc2 = " + mc2 + "\n")
    println ("\nContinuous-Time Markov Chain: simulation:")
    mc2.simulate (0, endTime)

end markovCTTest

