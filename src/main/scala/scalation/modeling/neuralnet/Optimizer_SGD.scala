
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Sun Feb  6 00:08:23 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Optimization: Stochastic Gradient Descent Optimizer
 */

package scalation
package modeling
package neuralnet

import scala.math.min

import scalation.mathstat._

import Optimizer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Optimizer-SGD` class provides methods to optimize the parameters (weights
 *  and biases) of Neural Networks with various numbers of layers.
 *  This optimizer implements a Stochastic Gradient Descent algorithm.
 */
class Optimizer_SGD extends Optimizer:

    private val debug = debugf ("Optimizer_SGD", true)                    // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x and y for a 2-layer, multi-output Neural Network, fit
     *  the parameter/weight matrix b.  Iterate over several epochs, where each epoch
     *  divides the training set into nB batches.  Each batch is used to update the
     *  parameter's weights.
     *  @param x    the m-by-n input matrix (training data consisting of m input vectors)
     *  @param y    the m-by-ny output matrix (training data consisting of m output vectors)
     *  @param bb   the array of parameters (weights & biases) between every two adjacent layers
     *  @param eta  the initial learning/convergence rate
     *  @param ff   the array of activation function family for every two adjacent layers
     */
    def optimize2 (x: MatrixD, y: MatrixD,
                   bb: NetParams, eta: Double, ff: Array [AFF]): (Double, Int) =
        val permGen   = permGenerator (x.dim)                             // permutation vector generator
        val b         = bb(0)                                             // net-parameters: weight matrix and bias vector
        val f         = ff(0)                                             // activation function
        val bSize     = min (hp("bSize").toInt, x.dim)                    // batch size
        val maxEpochs = hp("maxEpochs").toInt                             // maximum number of epochs
        val upLimit   = hp("upLimit").toInt                               // limit on increasing lose
        var η         = eta                                               // set initial learning rate
        val nB        = x.dim / bSize                                     // the number of batches
        println (s"optimize2: bSize = $bSize, nB = $nB")

        var sse_best_   = -0.0
        var (go, epoch) = (true, 1)
        cfor (go && epoch <= maxEpochs, epoch += 1) {                     // iterate over each epoch
            val batches = permGen.igen.chop (nB)                          // permute indices & chop into nB batches

            for ib <- batches do b -= updateWeight (x(ib), y(ib))         // iteratively update parameters b

            val sse = (y - f.fM (b * x)).normFSq                          // recompute sum of squared errors
            collectLoss (sse)                                             // collect the loss per epoch
//          debug ("optimize2", s"parameters for $epoch th epoch: sse = $sse")
            val (b_best, sse_best) = stopWhen (Array (b), sse)
            if b_best != null then
                b.set (b_best (0))
                sse_best_ = sse_best                                      // save best in sse_best_
                go = false
            else
                if epoch % ADJUST_PERIOD == 0 then η *= ADJUST_FACTOR     // adjust the learning rate
        } // cfor

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Update the parameter/weight matrix b based on the current batch.
         *  Take a step in the direction opposite to the gradient.
         *  @param x  the input matrix for the current batch
         *  @param y  the output matrix for the current batch
         */
        inline def updateWeight (x: MatrixD, y: MatrixD): MatrixD =
            val α  = η / x.dim                                            // eta over the current batch size
            val yp = f.fM (b * x)                                         // prediction: Yp = f(XB)
            val ε  = yp - y                                               // negative of error matrix
            val δ  = f.dM (yp) ⊙ ε                                        // delta matrix for y

            x.ᵀ * δ * α                                                   // return change in parameters (transpose (ᵀ))
        end updateWeight

        debug ("optimize2", s"parameters b = $b")
        if go then ((y - f.fM (b * x)).normFSq, maxEpochs)                // return best and number of epochs
        else       (sse_best_, epoch - upLimit)
    end optimize2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x and y for a 3-layer Neural Network, fit the parameters
     *  (weights and biases) a & b.  Iterate over several epochs, where each epoch divides
     *  the training set into nB batches.  Each batch is used to update the weights.
     *  @param x    the m-by-n input matrix (training data consisting of m input vectors)
     *  @param y    the m-by-ny output matrix (training data consisting of m output vectors)
     *  @param bb   the array of parameters (weights & biases) between every two adjacent layers
     *  @param eta  the initial learning/convergence rate
     *  @param ff   the array of activation function family for every two adjacent layers
     */
    def optimize3 (x: MatrixD, y: MatrixD,
                   bb: NetParams, eta: Double, ff: Array [AFF]): (Double, Int) =
        val permGen   = permGenerator (x.dim)                             // permutation vector generator
        val (a, b)    = (bb(0), bb(1))                                    // two sets of net-parameters
        val (f, f1)   = (ff(0), ff(1))                                    // two activation functions
        val bSize     = min (hp("bSize").toInt, x.dim)                    // batch size
        val maxEpochs = hp("maxEpochs").toInt                             // maximum number of epochs
        val upLimit   = hp("upLimit").toInt                               // limit on increasing lose
        var η         = eta                                               // set initial learning rate
        val nB        = x.dim / bSize                                     // the number of batches
        println (s"optimize3: bSize = $bSize, nB = $nB")

        var sse_best_   = -0.0
        var (go, epoch) = (true, 1)
        cfor (go && epoch <= maxEpochs, epoch += 1) {                     // iterate over each epoch
            val batches = permGen.igen.chop (nB)                          // permute indices & chop into nB batches

            for ib <- batches do
                val ab = updateWeight (x(ib), y(ib))                      // iteratively update parameters a & b
                a -= ab._1; b -= ab._2
            end for

            val sse = (y - b * f1.fM (f.fM (a * x))).normFSq
            collectLoss (sse)                                             // collect the loss per epoch
//          debug ("optimize3", s"parameters for $epoch th epoch: sse = $sse")
            val (b_best, sse_best) = stopWhen (Array (a, b), sse)
            if b_best != null then
                a.set (b_best(0))
                b.set (b_best(1))
                sse_best_ = sse_best                                      // save best in sse_best_
                go = false
            else
                if epoch % ADJUST_PERIOD == 0 then η *= ADJUST_FACTOR     // adjust the learning rate
        } // cfor

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Compute the parameter a & b updates based on the current batch.
         *  A step in the direction opposite to the gradient.
         *  @param x  the input matrix for the current batch
         *  @param y  the output matrix for the current batch
         */
        inline def updateWeight (x: MatrixD, y: MatrixD): (NetParam, NetParam) =
            val α  = η / x.dim                                            // eta over the current batch size
            val z  = f.fM (a * x)                                         // hidden layer: Z  = f(XA)
            val yp = f1.fM (b * z)                                        // prediction:   Yp = f(ZB)
            val ε  = yp - y                                               // negative of the error matrix
            val δ1 = f1.dM (yp) ⊙ ε                                       // delta matrix for y
            val δ0 = f.dM (z) ⊙ (δ1 * b.w.ᵀ)                              // delta matrix for z (transpose (ᵀ))
    
            (NetParam (x.ᵀ * δ0 * α, δ0.mean * η),                        // change to a parameters (weights and biases)
             NetParam (z.ᵀ * δ1 * α, δ1.mean * η))                        // change to b parameters (weights and biases)
        end updateWeight

        debug ("optimize3", s"parameters a = $a \n b = $b")
        if go then ((y - b * f1.fM (f.fM (a * x))).normFSq, maxEpochs)    // return best and number of epochs
        else       (sse_best_, epoch - upLimit)
    end optimize3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x and y for a multi-hidden layer Neural Network, fit the
     *  parameter array b, where each b(l) contains a weight matrix and bias vector.
     *  Iterate over several epochs, where each epoch divides the training set into nB batches.
     *  Each batch is used to update the weights.
     *  @param x    the m-by-n input matrix (training data consisting of m input vectors)
     *  @param y    the m-by-ny output matrix (training data consisting of m output vectors)
     *  @param b    the array of parameters (weights & biases) between every two adjacent layers
     *  @param eta  the initial learning/convergence rate
     *  @param f    the array of activation function family for every two adjacent layers
     */
    def optimize (x: MatrixD, y: MatrixD,
                  b: NetParams, eta: Double, f: Array [AFF]): (Double, Int) =
        val permGen   = permGenerator (x.dim)                             // permutation vector generator
        val bSize     = min (hp("bSize").toInt, x.dim)                    // batch size
        val maxEpochs = hp("maxEpochs").toInt                             // maximum number of epochs
        val upLimit   = hp("upLimit").toInt                               // limit on increasing lose
        var η         = eta                                               // set initial learning rate
        val nB        = x.dim / bSize                                     // the number of batches
        var sse       = 0.0                                               // stores accumulated sse over batches for epoch
        println (s"optimize: bSize = $bSize, nB = $nB")

        val nl     = f.size                                               // number of layers
        val layers = 0 until nl                                           // range for layers
        val z      = Array.ofDim [MatrixD] (nl+1)                         // array to store activations, layer by layer
        val δ      = Array.ofDim [MatrixD] (nl)                           // array to store all deltas

        var sse_best_   = -0.0
        var (go, epoch) = (true, 1)
        cfor (go && epoch <= maxEpochs, epoch += 1) {                     // iterate over each epoch
            sse         = 0.0
            val batches = permGen.igen.chop (nB)                          // permute indices & chop into nB batches

            for ib <- batches do sse += updateWeight (x(ib), y(ib))       // update parameter array b

            collectLoss (sse)                                             // collect the loss per epoch
//          debug ("optimize", s"parameters for $epoch th epoch: b = $b, sse = $sse")
            val (b_best, sse_best) = stopWhen (b, sse)
            if b_best != null then
                for l <- b.indices do b(l) = b_best(l)
                sse_best_ = sse_best                                      // save best in sse_best_
                go = false
            else
                if epoch % ADJUST_PERIOD == 0 then η *= ADJUST_FACTOR     // adjust the learning rate
        } // cfor

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Compute the parameter array b updates based on the current batch.
         *  A step in the direction opposite to the gradient.
         *  @param x  the input matrix for the current batch
         *  @param y  the output matrix for the current batch
         */
        inline def updateWeight (x: MatrixD, y: MatrixD): Double =
            val α = η / x.dim                                             // eta over the current batch size
            z(0)  = x                                                     // initial activation, which is the input matrix
            for l <- layers do z(l+1) = f(l).fM (b(l) * z(l))             // feed-forward and store all activations

            val yp  = z.last                                              // predicted value of y
            val ε   = yp - y                                              // -E where E is the error matrix
            δ(nl-1) = f.last.dM (yp) ⊙ ε                                  // delta for the last layer before output
            for l <- nl-2 to 0 by -1 do
                δ(l) = f(l).dM (z(l+1)) ⊙ (δ(l+1) * b(l+1).w.ᵀ)           // deltas for all previous hidden layers (transpose (ᵀ))
            end for

            for l <- layers do
                b(l) -= (z(l).ᵀ * δ(l) * α,                               // update parameters (weights
                         δ(l).mean * η)                                   // and biases
            end for

            ε.normFSq                                                     // return the sse of this batch
        end updateWeight

        debug ("optimize", s"parameters b = $b")
        if go then (sse, maxEpochs)                                       // return best and number of epochs
        else       (sse_best_, epoch - upLimit)
    end optimize

//              b(l).w *= 1.0 - eta * (lambda / x.dim)                    // regularization factor, weight decay

end Optimizer_SGD

