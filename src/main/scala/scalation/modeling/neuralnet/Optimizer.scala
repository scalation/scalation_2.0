
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Sun Jan 27 15:34:08 EST 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Optimization: Stochastic Gradient Descent Optimizer
 */

package scalation
package modeling
package neuralnet

//import java.lang.Double.isNaN

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
import scalation.random.PermutedVecI
import scalation.random.RNGStream.ranStream

import Initializer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Optimizer` object gives defaults for hyper-parameters as well as other
 *  adjustable program constants.
 */
object Optimizer:

    /** hyper-parameters for tuning the optimization algorithms - user tuning
     */
    val hp = new HyperParameter
    hp += ("eta", 0.1, 0.1)                                               // learning/convergence rate (smaller for Adam)
    hp += ("bSize", 20, 20)                                               // mini-batch size, common range 16 to 64
    hp += ("maxEpochs", 400, 400)                                         // maximum number of epochs/iterations
    hp += ("lambda", 0.01, 0.01)                                          // regularization/shrinkage hyper-parameter
    hp += ("upLimit", 4, 4)                                               // up-limit hyper-parameter for stopping rule
    hp += ("beta", 0.9, 0.9)                                              // momentum decay hyper-parameter
    hp += ("beta2", 0.999, 0.999)                                         // momentum decay hyper-parameter (second moment)
    hp += ("nu", 0.9, 0.9)                                                // interpolates between SGD (ν = 0) and
                                                                          // (normalized) SHB (ν = 1)

    /** other constants affecting the optimization algorithms - developer tuning
     */
    val ADJUST_PERIOD  = 100                                              // number of epochs before adjusting learning rate
    val ADJUST_FACTOR  = 1.1                                              // learning rate adjustment factor (1+)
    val NSTEPS         = 16                                               // steps for eta
    val estat          = new Statistic ("epochs")                         // statistics on the number of epochs

end Optimizer

import Optimizer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Optimizer` trait provides methods to optimize and auto_optimize parameters.
 *  Given training data x and y for a Neural Network, fit the parameters b.
 */
trait Optimizer extends MonitorLoss with StoppingRule:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Freeze layer flayer during back-propogation (should only impact the
     *  optimize method in the classes extending this trait).
     *  FIX: make abstract (remove ???) and implement in extending classes
     *  @param flayer  the layer to freeze, e.g., 1 => first hidden layer
     */
    def freeze (flayer: Int): Unit = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a permutation vector generator that will provide a random permutation of
     *  index positions for each call permGen.igen (e.g., used to select random batches).
     *  @param m      the number of data instances
     *  @param rando  whether to use a random or fixed random number stream
     */
    def permGenerator (m: Int, rando: Boolean = true): PermutedVecI =
        val idx    = VectorI.range (0, m)                                 // data instance index range
        val stream = if rando then ranStream else 0                       // use rando, unless testing
        PermutedVecI (idx, stream)                                        // permutation vector generator
    end permGenerator

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x and y for a Neural Network, fit the parameters b,
     *  returning the value of the lose function and the number of epochs.
     *  @param x     the m-by-n input matrix (training data consisting of m input vectors)
     *  @param y     the m-by-ny output matrix (training data consisting of m output vectors)
     *  @param b     the array of parameters (weights & biases) between every two adjacent layers
     *  @param etaI  the lower and upper bounds of learning/convergence rate
     *  @param f     the array of activation function family for every two adjacent layers
     */
    def optimize (x: MatrixD, y: MatrixD, b: NetParams, eta_ : Double, f: Array [AFF]):
                 (Double, Int)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x and y for a Neural Network, fit the parameters b,
     *  returning the value of the lose function and the number of epochs.
     *  Find the best learning rate within the interval etaI.
     *  @param x     the m-by-n input matrix (training data consisting of m input vectors)
     *  @param y     the m-by-ny output matrix (training data consisting of m output vectors)
     *  @param b     the array of parameters (weights & biases) between every two adjacent layers
     *  @param etaI  the lower and upper bounds of learning/convergence rate
     *  @param f     the array of activation function family for every two adjacent layers
     *  @param opti  the array of activation function family for every two adjacent layers
     */
    def auto_optimize (x: MatrixD, y: MatrixD, b: NetParams, etaI: (Double, Double), f: Array [AFF],
                       opti: (MatrixD, MatrixD, NetParams, Double, Array [AFF]) => (Double, Int)):
                      (Double, Int) =
        println (s"auto_optimize: etaI = $etaI")
        var best = (Double.MaxValue, -1)
        var b_best: NetParams = null

        for i <- 0 to NSTEPS do
            val step = (etaI._2 - etaI._1) / NSTEPS                       // compute step size
            val eta  = etaI._1 + i * step                                 // current learning rate
            for b_l <- b do init_weights (b_l)                            // initialize parameters (weights/bias)

            val result = opti (x, y, b, eta, f)                           // run optimizer with given learning rate
            if result._1.isNaN then
                println (s"auto_optimize: FOR eta = $eta, result = $result GIVES Not-a-Number")
            else
                println (s"auto_optimize: eta = $eta, result = $result")
                if result._1 < best._1 then
                    best = result                                             // save it, if better
                    b_best = (for l <- b.indices yield b(l).copy).toArray     // save best parameters
                    println (s"auto_optimize: b = ${stringOf (b)}")
            end if
        end for

        for l <- b.indices do b(l) = b_best(l)                            // use best parameters
        println (s"auto_optimize end: b = ${stringOf (b)}")
        best                                                              // return best loss value & # epochs
    end auto_optimize

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Randomly intialize the weight matrix and optional bias vector for layer l.
     *  @param b_l  the network parameters for layer l
     */
    inline private def init_weights (b_l: NetParam): Unit =
       if b_l.b == null then                                              // bias is null
           b_l.set (weightMat (b_l.w.dim, b_l.w.dim2))                    // randomly assign weights to b_l.w
       else
           b_l.set (weightMat (b_l.w.dim, b_l.w.dim2),                    // randomly assign weights to b_l.w
                    weightVec (b_l.b.dim))                                // randomly assign biases to b_l.b
    end init_weights

end Optimizer

