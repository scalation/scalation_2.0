
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Nov 16 14:32:49 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Hidden Markov Models (HMM)
 *
 *  Code adapted from pseudocode in
 *  @see www.cs.sjsu.edu/faculty/stamp/RUA/HMM.pdf
 *
 *  @see people.cs.umass.edu/~mccallum/courses/inlp2004a/lect10-hmm2.pdf
 *  @see web.stanford.edu/~jurafsky/slp3/A.pdf
 *  @see www.parralab.org/teaching/biomed-dsp/class10.pdf
 */

//  U N D E R   D E V E L O P M E N T

package scalation
package modeling
package classifying

import scala.math.log
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._
import scalation.random.ProbabilityVec

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `HiddenMarkov` classes provides Hidden Markov Models (HMM).  An HMM model
 *  consists of a probability vector pi and probability matrices a and b.
 *  The discrete-time system is characterized by a hidden state x(t) and an
 *  observed symbol/value y(t) at time t, which may be viewed as a time series.
 *      pi(i)   = P(x(t) = i)
 *      a(i, j) = P(x(t+1) = j | x(t) = i) 
 *      b(i, k) = P(y(t) = k | x(t) = i)
 *      model (pi, a, b)
 *  @param y       the observation vector/observed discrete-valued time series
 *  @param m       the number of observation symbols/values {0, 1, ... m-1}
 *  @param n       the number of (hidden) states in the model
 *  @param cname_  the class names for the states, e.g., ("Hot", "Cold")
 *  @param pi      the probabilty vector for the initial state
 *  @param a       the state transition probability matrix (n-by-n)
 *  @param b       the observation probability matrix (n-by-m)
 *  @param hparam  the hyper-parameters
 */
class HiddenMarkov (y: VectorI, m: Int, n: Int, cname_ : Array [String] = null,
                    private var pi: VectorD = null,
                    private var a:  MatrixD = null,
                    private var b:  MatrixD = null,
                    hparam: HyperParameter = null)
      extends Classifier (null, y, null, n, cname_, hparam)           // hidden state vector x = null here
         with FitC (n):
                                                                      // feature names fn = null
    private val debug = debugf ("HiddenMarkov", true)                 // debug function
    private val MIT   = 1000                                          // Maximum ITerations
    private val tt    = y.dim                                         // the number of observations
    private val pvm   = ProbabilityVec (m)                            // probability generator (dim = m)
    private val pvn   = ProbabilityVec (n)                            // probability generator (dim = n)

    private val c     = new VectorD (tt)                              // vector of scaling factors
    private val alp   = new MatrixD (tt, n)                           // alpha matrix P(x_t = j  | y = y^t-)
    private val bet   = new MatrixD (tt, n)                           // beta matrix  P(y = y^t+ | x_t = i)
    private val gam   = new MatrixD (tt, n)                           // gamma matrix P(x_t = i  | y = y)
    private val gat   = Array.fill (tt) (new MatrixD (n, n))          // gamma tensor as an array of matrices

    private val rtime  = 0 until tt                                   // range over time
    private val rvalue = 0 until m                                    // range over observation symbols/values
    private val rstate = 0 until n                                    // range over states

    _modelName = s"HiddenMarkov_$n"                                   // name of the model

    if pi == null then
        pi = pvn.gen                                                  // initialize the state probability vector
    if a == null then
        a = new MatrixD (n, n)
        for i <- rstate do a(i) = pvn.gen                             // initialize state transition probability matrix a
    if b == null then
        b = new MatrixD (n, m)
        for i <- rstate do b(i) = pvm.gen                             // initialize observation probability matrix b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size of the (hidden) state space.
     */
    def size: Int = n

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter vector pi.
     */
    override def parameter: VectorD = pi

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter matrices a and b.
     */
    def parameters: (MatrixD, MatrixD) = (a, b)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the joint probability P(x, y).
     *  @param x  the state vector
     */
    def jointProb (x: VectorI): Double =
        var prod = pi(x(0)) * b(x(0), y(0))
        debug ("jointProb", s"pi = ${pi(x(0))}, b = ${b(x(0), y(0))}, ")
        for t <- 1 until x.dim do
            debug ("jointProb", s"a = ${a(x(t-1), x(t))}, b = ${b(x(t), y(t))}, ")
            prod *= a(x(t-1), x(t)) * b(x(t), y(t)) 
        end for
        println ()
        prod
    end jointProb

//----------------------------------------------------------------------------
// Computations without scaling
//----------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The alpha-pass: a forward pass from time t = 0 to tt-1 that computes
     *  alpha alp. 
     */
    def forwardEval0 (): MatrixD =
        for j <- rstate do alp(0, j) = pi(j) * b(j, y(0))            // compute alpha_0 (at time t = 0)
        for t <- 1 until tt; j <- rstate do                          // iterate over time and states
            alp(t, j) = b(j, y(t)) * (alp(t-1) dot a(?, j))
        end for
        alp
    end forwardEval0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability of seeing observation vector y,
     *  given the model pi, a and b.
     *  @param scaled  whether the alpha matrix is scaled
     *  Requires:  alp  the unscaled alpha matrix or
     *             c    the vector of scaling factors
     */
    private def probY (scaled: Boolean = false): Double =
        if scaled then
           var p = 1.0                                                // probability
           for t <- rtime do p *= c(t)                                // product of scaling factors
           1.0 / p                                                    // reciporcal of product
        else
           alp(tt-1).sum                                              // sum of last row
    end probY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the -log of the probability of seeing observation vector y,
     *  given the model pi, a and b.
     *  @param scaled  whether the alpha matrix is scaled
     *  Requires:  alp  the unscaled alpha matrix or
     *             c    the vector of scaling factors
     */
    private def logProbY (scaled: Boolean): Double =
        if scaled then
            var lp = 0.0                                              // log-probability
            for t <- rtime do lp += log (c(t))                        // sum of the log of scaling factors
            -lp                                                       // reciporcal via -log
        else
            -log (alp(tt-1).sum)                                      // - log of sum of last row
    end logProbY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The beta-pass: a backward pass from time t = tt-1 to 0 that computes
     *  beta bet.
     */
    def backwardEval0 (): MatrixD =
        for i <- rstate do bet(tt-1, i) = 1.0                        // initialize beta_{tt-1} to 1
        for t <- tt-2 to 0 by -1; i <- rstate do                     // iterate backover time, over states
            bet(t, i) = 0.0
            for j <- rstate do bet(t, i) += a(i, j) * b(j, y(t+1)) * bet(t+1, j)
        end for
        bet
    end backwardEval0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The gamma-pass: a forward pass from time t = 0 to tt-2 that computes
     *  gamma gam.  Given the observation vector y, find the most probable
     *  sequence of states.
     *  Requires:  alp  the unscaled alpha matrix
     *             bet  the unscaled beta matrix
     */
    def gamma (): MatrixD = (alp *~ bet) / probY ()

//----------------------------------------------------------------------------
// Computations with scaling
//----------------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The alpha-pass: a forward pass from time t = 0 to tt-1 that computes
     *  alpha alp with scaling.
     */
    def forwardEval (): MatrixD =
        for j <- rstate do alp(0, j) = pi(j) * b(j, y(0))             // compute alpha_0 (at time t = 0)
        c(0) = 1.0 / alp(0).sum
        for j <- rstate do alp(0, j) *= c(0)                          // scale alpha_0 (at time 0)

        for t <- 1 until tt do                                        // iterate forward in time
            for j <- rstate do                                        // iterate over states
                alp(t, j) = b(j, y(t)) * (alp(t-1) dot a.col(j))
            end for
            c(t) = 1.0 / alp(t).sum
            for j <- rstate do alp(t, j) *= c(t)                       // scale alpha_t (at time t)
        end for
        alp
    end forwardEval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of scaling factors c.
     */
    def getC: VectorD = c

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The beta-pass: a backward pass from time t = tt-1 to 0 that computes
     *  beta bet with scaling.
     *  Requires:  alp  the scaled alpha matrix  
     *             c    the vector of scaling factors
     */
    def backwardEval (): MatrixD =
        for i <- rstate do bet(tt-1, i) = c(tt-1)                      // initialize beta_{t-1} to c at tt-1
        
        for t <- tt-2 to 0 by -1 do                                    // iterate backward in time
            for i <- rstate do                                         // iterate over states
                bet(t, i) = 0.0
                for j <- rstate do bet(t, i) += a(i, j) * b(j, y(t+1)) * bet(t+1, j)
            end for
            for i <- rstate do bet(t, i) *= c(t)                       // scale beta
        end for
        bet
    end backwardEval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The gamma-pass: a forward pass from time t = 0 to tt-2 that computes
     *  gamma gam and gat.  Given the observation vector y, find the most
     *  probable sequence of states.  Note: gat is computed as a side-effect.
     *  Requires:  alp  the scaled alpha matrix
     *             bet  the scaled beta matrix
     */
    def viterbiDecode (): MatrixD =
        for t <- 0 until tt-1 do                                     // compute gamma (at time t)
            for i <- rstate do
                gam(t, i) = 0.0
                for j <- rstate do
                    gat(t)(i, j) = alp(t, i) * a(i, j) * b(j, y(t+1)) * bet(t+1, j)
                    gam(t, i)   += gat(t)(i, j)
        end for
        for i <- rstate do gam(tt-1, i) = alp(tt-1, i)
        gam
    end viterbiDecode

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Re-estimate the probability vector pi and the probability matrices
     *  a and b.
     *  Requires:  gam  the gamma matrix
     *             gat  the gamma tensor
     */
    def reestimate (): Unit =
        pi = gam(0)                                                   // re-estimate pi

        for i <- rstate do
            val den2 = gam.col(i).sum                                 // denominator for b_ik_
            val den1 = den2 - gam(tt-1, i)                            // for a_ij subtract last element from den2

            for j <- rstate do
                var num = 0.0
                for t <- 0 until tt-1 do num += gat(t)(i, j)
                a(i, j) = num / den1                                  // re-estimate a
            end for

            for k <- rvalue do
                var num = 0.0
                for t <- rtime if y(t) == k do num += gam(t, i)
                b(i, k) = num / den2                                  // re-estimate b
            end for

        end for
    end reestimate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the Hidden Markov Model using the observation vector y to
     *  determine the model pi, a and b.
     *  @param x_  the training/full data/input matrix (ignored)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = null, y_ : VectorI = y): Unit =
        var oldLogPr = 0.0
        breakable {
            for it <- 1 to MIT do                                     // up to Maximum ITerations
                val logPr = logProbY (true)                           // compute the new log probability
                if logPr > oldLogPr then
                    oldLogPr = logPr                                  // improvement => continue
                    forwardEval ()                                    // alpha-pass
                    backwardEval ()                                   // beta-pass
                    viterbiDecode ()                                  // gamma-pass
                    reestimate ()                                     // re-estimate the model (pi, a, b)
                else
                    println (s"train: HMM model converged after $it iterations")
                    break ()
            end for
        } // breakable
        println (s"train: HMM model did not converged after $MIT iterations")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF vector.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (ignored)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = null, y_ : VectorI = y): (VectorI, VectorD) =
        val yp  = predictI (x_)                                       // predicted classes
        val qof = diagnose (y_.toDouble, yp.toDouble)                 // diagnose from actual and predicted
        debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new discrete data vector z, determine which class it belongs to,
     *  returning the best class, its name and its relative probability.
     *  @param z  the observation vector/time series to classify
     */
    override def predictI (z: VectorI): Int =
        val t = z.dim - 1                                             // time of last observation
        alp(t).argmax ()                                              // state with max probability
    end predictI

    inline def predictI (z: VectorD): Int = predictI (z.toInt)

end HiddenMarkov


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `HiddenMarkov` companion object provides a convenience method for testing.
 */
object HiddenMarkov:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For the hmm model, print all state sequences and their joint probability P(x, y).
     *  @param hmm   the Hidden Markov Model
     *  @param xall  the strings for all state sequences
     *  @param xval  the values for all state sequences
     */
    def allState (hmm: HiddenMarkov, xall: Array [String], xval: Array [VectorI]): Unit =
        var sum = 0.0
        for l <- xall.indices do
            val jp = hmm.jointProb (xval(l))
            println (s"P(${xall(l)}, y) = $jp")
            sum += jp
        end for
        println (s"P(y) = $sum")
    end allState

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the hmm model, printing out the unscaled and scaled versions of
     *  the alpha and beta matrices, and the gamma matrix calculated from unscaled
     *  and scaled matrices.
     *  @param hmm  the Hidden Markov Model
     */
    def test (hmm: HiddenMarkov): Unit =
        banner ("Given Parameters")
        println ("parameter  = " + hmm.parameter)
        println ("parameters = " + hmm.parameters)

        banner ("Unscaled Algorithms:")

        banner ("Forward Algorithm: alpha")
        var alp = hmm.forwardEval0 ()                                 // forward algorithm
        println ("alp  = " + alp)                                     // unscaled alpha matrix
        banner ("Backward Algorithm: beta")
        var bet = hmm.backwardEval0 ()                                // backward algorithm
        println ("bet  = " + bet)                                     // unscaled beta matrix
        banner ("Direct Algorithm: gamma")
        var gam = hmm.gamma ()                                        // direct calculation of gamma
        println ("gam  = " + gam)                                     // gamma matrix from unscaled matrices
        println ("P(y) = " + hmm.probY ())                            // probability of observations y given model

        banner ("Scaled Algorithms:")

        banner ("Forward Algorithm: alpha")
        alp = hmm.forwardEval ()                                      // forward algorithm with scaling
        println ("alp = " + alp)                                      // scaled alpha matrix
        banner ("Backward Algorithm: beta")
        bet = hmm.backwardEval ()                                     // backward algorithm with scaling
        println ("bet = " + bet)                                      // scaled beta matrix
        banner ("Viterbi Algorithm: gamma")
        gam = hmm.viterbiDecode ()                                    // Viterbi algorithm
        println ("gam = " + gam)                                      // gamma matrix from scaled matrices
        println ("c    = " + hmm.getC)                                // scaling vector
        println ("P(y) = " + hmm.probY (true))                        // probability of observations y given model

        banner ("Re-Estimate Parameters")
        hmm.reestimate ()                                             // re-estimate the parameters pi, a and b
        println (hmm.report)                                          // report on model
    end test

end HiddenMarkov


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `hiddenMarkovTest` main function is used to test the `HiddenMarkov` class.
 *  Given model (pi, a, b), determine the probability of the observations y.
 *  @see www.cs.sjsu.edu/~stamp/RUA/HMM.pdf (exercise 1).
 *  > runMain scalation.modeling.classifying.hiddenMarkovTest
 */
@main def hiddenMarkovTest (): Unit =

    banner ("Temperature State from Width of Tree Rings") 

    val pi = VectorD (0.0, 1.0)                                       // initial state probability vector: H(0), C(1)

    val a = MatrixD ((2, 2), 0.7, 0.3,                                // state transition matrix
                             0.4, 0.6)

    val b = MatrixD ((2, 3), 0.1, 0.4, 0.5,                           // emission probability matrix
                             0.7, 0.2, 0.1)
    val y = VectorI (1, 0, 2)                                         // observations: M(1), S(0), L(2)
    println (s"y = $y")

    val nSymbols = b.dim2                                             // number of observable symbols |{S, M, L}|
    val nStates  = a.dim                                              // number of states |{H, C}|
    val cn       = Array ("H", "C")                                   // class names for states

    val hmm = new HiddenMarkov (y, nSymbols, nStates, cn, pi, a, b)   // Hidden Markov Model (HMM) 

    banner ("Joint Probabilities: P(x, y)")
    val xall = Array ("HHH", "HHC", "HCH", "HCC",
                      "CHH", "CHC", "CCH", "CCC")
    val xval = Array (VectorI (0, 0, 0), VectorI (0, 0, 1), VectorI (0, 1, 0), VectorI (0, 1, 1),
                      VectorI (1, 0, 0), VectorI (1, 0, 1), VectorI (1, 1, 0), VectorI (1, 1, 1))
    
    HiddenMarkov.allState (hmm, xall, xval)
    HiddenMarkov.test (hmm)

end hiddenMarkovTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `hiddenMarkovTest2` main function is used to test the `HiddenMarkov` class.
 *  Train the model (pi, a, b) based on the observed data.
 *  @see www.cs.sjsu.edu/~stamp/RUA/HMM.pdf.
 *  > runMain scalation.modeling.classifying.hiddenMarkovTest2
 */
@main def hiddenMarkovTest2 (): Unit =

    banner ("Temperature State from Width of Tree Rings") 

    val y = VectorI (0, 1, 0, 2)                                      // observations: S(0), M(1), S(0), L(2)
    println (s"y = $y")

    val nSymbols = 3                                                  // number of observable symbols |{S, M, L}|
    val nStates  = 2                                                  // number of states |{H, C}|
    val cn       = Array ("H", "C")                                   // class names for states

    val hmm = new HiddenMarkov (y, nSymbols, nStates, cn)             // model (pi, a, b) to be determined

    println ("Train the Hidden Markov Model")
    hmm.train (null)
    println (hmm.report)

end hiddenMarkovTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `hiddenMarkovTest3` main function is used to test the `HiddenMarkov` class.
 *  Given model (pi, a, b), determine the probability of the observations y.
 *  @see "Introduction to Data Science using ScalaTion"
 *  > runMain scalation.modeling.classifying.hiddenMarkovTest3
 */
@main def hiddenMarkovTest3 (): Unit =

    banner ("Traffic Accident State from Traffic Flow") 

    val pi = VectorD (0.9, 0.1)                                       // initial state probability vector: N(0), A(1)

    val a = MatrixD ((2, 2), 0.8, 0.2,                                // state transition matrix
                             0.5, 0.5)

    val b = MatrixD ((2, 4), 0.1, 0.2, 0.3, 0.4,                      // emission probability matrix
                             0.5, 0.2, 0.2, 0.1)
    val y = VectorI (3, 3, 0)                                         // observations: 30+, 30+, 0+ vehicles
    println (s"y = $y")

    val nSymbols = b.dim2                                             // number of observable symbols |{0+, 10+, 20+, 30+}|    
    val nStates  = a.dim                                              // number of states |{N, A}|
    val cn       = Array ("N", "A")                                   // class names for states

    val hmm = new HiddenMarkov (y, nSymbols, nStates, cn, pi, a, b)   // Hidden Markov Model (HMM) 

    banner ("Joint Probabilities: P(x, y)")
    val xall = Array ("NNN", "NNA", "NAN", "NAA",
                      "ANN", "ANA", "AAN", "AAA")
    val xval = Array (VectorI (0, 0, 0), VectorI (0, 0, 1), VectorI (0, 1, 0), VectorI (0, 1, 1),
                      VectorI (1, 0, 0), VectorI (1, 0, 1), VectorI (1, 1, 0), VectorI (1, 1, 1))
    
    HiddenMarkov.allState (hmm, xall, xval)
    HiddenMarkov.test (hmm)

end hiddenMarkovTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `hiddenMarkovTest4` main function is used to test the `HiddenMarkov` class.
 *  Train the model (pi, a, b) based on the observed data.
 *  @see "Introduction to Data Science using ScalaTion"
 *  > runMain scalation.modeling.classifying.hiddenMarkovTest4
 */
@main def hiddenMarkovTest4 (): Unit =

    banner ("Traffic Accident State from Traffic Flow") 

    val y = VectorI (3, 3, 0, 0, 3)                                   // observations: 30+, 30+, 0+, 0+, 30+ vehicles
    println (s"y = $y")

    val nSymbols = 4                                                  // number of observable symbols |{0+, 10+, 20+, 30+}|    
    val nStates  = 2                                                  // number of states |{N, A}|
    val cn       = Array ("N", "A")                                   // class names for states

    val hmm = new HiddenMarkov (y, nSymbols, nStates, cn)             // model (pi, a, b) to be determined

    println ("Train the Hidden Markov Model")
    hmm.train (null)
    println (hmm.report)

    val z = VectorI (3, 3, 0)                                         // observations: 30+, 30+, 0+ vehicles
    println (s"classify ($y) = ${hmm.classify (y)}")
    println (s"classify ($z) = ${hmm.classify (z)}")

end hiddenMarkovTest4

