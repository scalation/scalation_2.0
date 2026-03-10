
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sat Mar  5 22:38:03 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Optimization: ADAptive Moment estimation (Adam) Optimizer
 */

package scalation
package modeling
package neuralnet

import scala.math.min

import scalation.mathstat._

import Optimizer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Optimizer_Adam` class provides functions to optimize the parameters (weights
 *  and biases) of Neural Networks with various numbers of layers.
 *  This optimizer implements an ADAptive Moment estimation (Adam) Optimizer
 *  @see https://arxiv.org/pdf/1412.6980.pdf
 */
class Optimizer_Adam extends Optimizer:

    private val debug = debugf ("Optimizer_Adam", true)                   // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x and y for a 2-layer, multi-output Neural Network, fit
     *  the parameter/weight matrix b.  Iterate over several epochs, where each epoch
     *  divides the training set into nB batches.  Each batch is used to update the
     *  the parameter's weights.
     *  @param x     the m-by-n input matrix (training data consisting of m input vectors)
     *  @param y     the m-by-ny output matrix (training data consisting of m output vectors)
     *  @param bb    the array of parameters (weights & biases) between every two adjacent layers
     *  @param eta_  the initial learning/convergence rate
     *  @param ff    the array of activation function family for every two adjacent layers
     */
    def optimize2 (x: MatrixD, y: MatrixD,
                   bb: NetParams, eta_ : Double, ff: Array [AFF]): (Double, Int) =
        val permGen   = permGenerator (x.dim)                             // permutation vector generator
        val b         = bb(0)                                             // net-parameters: weight matrix and bias vector
        val f         = ff(0)                                             // activation function
        val bSize     = min (hp("bSize").toInt, x.dim)                    // batch size
        val maxEpochs = hp("maxEpochs").toInt                             // maximum number of epochs
        val upLimit   = hp("upLimit").toInt                               // limit on increasing lose
        val β1        = hp("beta").toDouble                               // momentum hyper-parameter
        val β2        = hp("beta2").toDouble                              // second momentum hyper-parameter
        val nB        = x.dim / bSize                                     // the number of batches
        var eta       = eta_                                              // set initial learning rate
        val η         = eta

        var p         = new MatrixD (b.w.dim, b.w.dim2)                   // momentum matrix
        var v         = new MatrixD (b.w.dim, b.w.dim2)                   // second raw moment estimate
        println (s"optimize2: bSize = $bSize, nB = $nB")

        var sse_best_   = -0.0
        var (go, epoch) = (true, 1)
        cfor (go && epoch <= maxEpochs, epoch += 1) {                     // iterate over each epoch
            val batches = permGen.igen.chop (nB)                          // permute indices & split into nB batches

            for ib <- batches do b -= updateWeight (x(ib), y(ib), epoch)  // iteratively update weight matrix b

            val sse = (y - f.fM (b * x)).normFSq                          // recompute sum of squared errors
            collectLoss (sse)                                             // collect loss per epoch
//          debug ("optimize2", s"parameters for $epoch th epoch: sse = $sse")
            val (b_best, sse_best) = stopWhen (Array (b), sse)
            if b_best != null then
                b.set (b_best (0))
                sse_best_ = sse_best                                      // save best in sse_best_
                go = false
            else
                if epoch % ADJUST_PERIOD == 0 then eta *= ADJUST_FACTOR   // adjust the learning rate
        } // cfor

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Update the parameter/weight matrix b based on the current batch.
         *  Take a step in the direction opposite to the gradient.
         *  @see https://arxiv.org/pdf/1412.6980.pdf
         *  @param x  the input matrix for the current batch
         *  @param y  the output matrix for the current batch
         */
        inline def updateWeight (x: MatrixD, y: MatrixD, t: Int): MatrixD =
            val α  = η / x.dim                                            // eta over the current batch size
            val yp = f.fM(b * x)                                          // prediction: Yp = f(XB)
            val ε  = yp - y                                               // negative of error matrix
            val δ  = f.dM(yp) ⊙ ε                                         // delta matrix for y
            val g  = x.ᵀ * δ                                              // + b.w * l -- gradient matrix (transpose (ᵀ))

            p = g * (1 - β1) + p * β1                                     // update biased first moment estimate
            v = v * β2 + g ~^ 2 * (1 - β2)                                // update biased second raw moment estimate

            val pH = p / (1 - β1 ~^ t)                                    // compute bias-corrected first moment estimate
            val vH = v / (1 - β2 ~^ t)                                    // compute bias-corrected second raw moment estimate

            (pH / (vH ~^ 0.5 + EPSILON)) * α                              // gradient-based change in input-output weights (bup)
                                                                          // + b.w * l -- is for AdamW
        end updateWeight

        debug ("optimize2", s"parameters b = $b")
        if go then ((y - f.fM (b * x)).normFSq, maxEpochs)                // return best and number of epochs
        else       (sse_best_, epoch - upLimit)
    end optimize2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x and y for a 3-layer Neural Network, fit the parameters
     *  (weights and biases) a & b.  Iterate over several epochs, where each epoch divides
     *  the training set into nB batches.  Each batch is used to update the weights. 
     *  @param x     the m-by-n input matrix (training data consisting of m input vectors)
     *  @param y     the m-by-ny output matrix (training data consisting of m output vectors)
     *  @param bb    the array of parameters (weights & biases) between every two adjacent layers
     *  @param eta_  the initial learning/convergence rate
     *  @param ff    the array of activation function family for every two adjacent layers
     */
    def optimize3 (x: MatrixD, y: MatrixD,
                   bb: NetParams, eta_ : Double, ff: Array [AFF]): (Double, Int) =
        val permGen = permGenerator (x.dim)                               // permutation vector generator
        val (a, b)  = (bb(0), bb(1))                                      // two sets of net-parameters
        val (f, f1) = (ff(0), ff(1))                                      // two activation functions
        val bSize   = min(hp("bSize").toInt, x.dim)                       // batch size
        val maxEpochs = hp("maxEpochs").toInt                             // maximum number of epochs
        val upLimit = hp("upLimit").toInt                                 // limit on increasing loss
        val β1      = hp("beta").toDouble                                 // first momentum hyper-parameter
        val β2      = hp("beta2").toDouble                                // second momentum hyper-parameter
        val nB      = x.dim / bSize                                       // the number of batches
        var eta     = eta_                                                // set initial learning rate

        var pA = new MatrixD(a.w.dim, a.w.dim2)                           // first moment estimate for a
        var pB = new MatrixD(b.w.dim, b.w.dim2)                           // first moment estimate for b
        var vA = new MatrixD(a.w.dim, a.w.dim2)                           // second moment estimate for a
        var vB = new MatrixD(b.w.dim, b.w.dim2)                           // second moment estimate for b

        var p_biasA = new VectorD(a.w.dim2)
        var p_biasB = new VectorD(b.w.dim2)
        var v_biasA = new VectorD(a.w.dim2)
        var v_biasB = new VectorD(b.w.dim2)

        println (s"optimize3: bSize = $bSize, nB = $nB")

        var sse_best_   = -0.0
        var (go, epoch) = (true, 1)
        cfor (go && epoch <= maxEpochs, epoch += 1) {                     // iterate over each epoch
            val batches = permGen.igen.chop (nB)                          // permute indices & split into nB batches

            for ib <- batches do
                val ab = updateWeight (x(ib), y(ib), epoch)               // update weight matrices a & b
                a -= ab._1; b -= ab._2
            end for

            val sse = (y - b * f1.fM(f.fM(a * x))).normFSq
            collectLoss(sse)                                              // collect the loss per epoch
//          debug ("optimize3", s"parameters for $epoch th epoch: sse = $sse")
            val (b_best, sse_best) = stopWhen (Array (a, b), sse)
            if b_best != null then
                a.set (b_best(0))
                b.set (b_best(1))
                sse_best_ = sse_best                                      // save best in sse_best_
                go = false
            else
                if epoch % ADJUST_PERIOD == 0 then eta *= ADJUST_FACTOR   // adjust the learning rate
        } // cfor

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Compute the parameter/weight matrices a and b updates based on the current batch.
         *  A step in the direction opposite to the gradient.
         *  @param x  the input matrix for the current batch
         *  @param y  the output matrix for the current batch
         */
        inline def updateWeight (x: MatrixD, y: MatrixD, t: Int): (NetParam, NetParam) =
            val α  = eta / x.dim                                          // eta over the current batch size
            val z  = f.fM(a * x)                                          // Z  = f(XA)
            val yp = f1.fM(b * z)                                         // Yp = f(ZB)
            val ε  = yp - y                                               // negative of the error matrix
            val δ1 = f1.dM(yp) ⊙ ε                                        // delta matrix for y
            val δ0 = f.dM(z) ⊙ (δ1 * b.w.transpose)                       // delta matrix for z

            val gA = x.ᵀ * δ0                                             // gradient for a (transpose (ᵀ))
            val gB = z.ᵀ * δ1                                             // gradient for b

            // Update biased first moment estimates for a and b
            pA = gA * (1 - β1) + pA * β1
            pB = gB * (1 - β1) + pB * β1

            // Update biased second moment estimates for a and b
            vA = vA * β2 + gA ~^ 2 * (1 - β2)
            vB = vB * β2 + gB ~^ 2 * (1 - β2)

            // Bias-corrected first moment estimates for a and b
            val pAH = pA / (1 - β1 ~^ t)
            val pBH = pB / (1 - β1 ~^ t)

            // Bias-corrected second moment estimates for a and b
            val vAH = vA / (1 - β2 ~^ t)
            val vBH = vB / (1 - β2 ~^ t)

            // Compute the parameter updates for a and b using the Adam update rule
            val Δa = (pAH / (vAH ~^ 0.5 + EPSILON)) * α
            val Δb = (pBH / (vBH ~^ 0.5 + EPSILON)) * α

            // Update biased first moment estimates for biases
            p_biasA = δ0.mean * (1 - β1) + p_biasA * β1
            p_biasB = δ1.mean * (1 - β1) + p_biasB * β1

            // Update biased second moment estimates for biases
            v_biasA = v_biasA * β2 + (δ0.mean ~^ 2) * (1 - β2)
            v_biasB = v_biasB * β2 + (δ1.mean ~^ 2) * (1 - β2)

            // Bias-corrected first moment estimates for biases
            val p_biasAH = p_biasA / (1 - β1 ~^ t)
            val p_biasBH = p_biasB / (1 - β1 ~^ t)

            // Bias-corrected second moment estimates for biases
            val v_biasAH = v_biasA / (1 - β2 ~^ t)
            val v_biasBH = v_biasB / (1 - β2 ~^ t)

            // Compute the bias updates using the Adam update rule
            val Δ_biasA = (p_biasAH / (v_biasAH ~^ 0.5 + EPSILON)) * α
            val Δ_biasB = (p_biasBH / (v_biasBH ~^ 0.5 + EPSILON)) * α

            (NetParam (Δa, Δ_biasA), NetParam (Δb, Δ_biasB))              // return the updates as NetParam objects
        end updateWeight

        debug ("optimize3", s"parameters a = $a \n b = $b")
        if go then ((y - b * f1.fM(f.fM(a * x))).normFSq, maxEpochs)      // return best and number of epochs
        else (sse_best_, epoch - upLimit)
    end optimize3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x and y, fit the parameter/weight matrices bw and
     *  bias vectors bi. Iterate over several epochs, where each epoch divides the
     *  training set into nB batches. Each batch is used to update the weights.
     *  @param x     the m-by-n input matrix (training data consisting of m input vectors)
     *  @param y     the m-by-ny output matrix (training data consisting of m output vectors)
     *  @param b     the array of parameters (weights & biases) between every two adjacent layers
     *  @param eta_  the initial learning/convergence rate
     *  @param f     the array of activation function family for every two adjacent layers
     */
    def optimize(x: MatrixD, y: MatrixD,
                 b: NetParams, eta_ : Double, f: Array [AFF]): (Double, Int) =
        val permGen = permGenerator(x.dim)                                // permutation vector generator
        val bSize   = min(hp("bSize").toInt, x.dim)                       // batch size
        val maxEpochs = hp("maxEpochs").toInt                             // maximum number of epochs
        val upLimit = hp("upLimit").toInt                                 // limit on increasing loss
        val β1      = hp("beta").toDouble                                 // first momentum hyper-parameter
        val β2      = hp("beta2").toDouble                                // second momentum hyper-parameter
        val nB      = x.dim / bSize                                       // the number of batches
        var eta     = eta_                                                // set initial learning rate
        var sse     = 0.0                                                 // stores accumulated sse over batches for epoch
        println (s"optimize: bSize = $bSize, nB = $nB")

        val nl = f.size                                                   // number of layers
        val layers = 0 until nl                                           // range for layers
        val z = Array.ofDim [MatrixD] (nl + 1)                            // array to store activations, layer by layer
        val δ = Array.ofDim [MatrixD] (nl)                                // array to store all deltas
        val p = Array.ofDim [MatrixD] (nl)                                // first moment estimates (momentum)
        val v = Array.ofDim [MatrixD] (nl)                                // second moment estimates (adaptive learning)
        val p_bias = Array.ofDim [VectorD] (nl)                           // first moment estimates for biases
        val v_bias = Array.ofDim [VectorD] (nl)                           // second moment estimates for biases

        for l <- layers do
            p(l) = new MatrixD (b(l).w.dim, b(l).w.dim2)                  // initialize first moment for each layer
            v(l) = new MatrixD (b(l).w.dim, b(l).w.dim2)                  // initialize second moment for each layer
            p_bias(l) = new VectorD (b(l).b.dim)                          // initialize first moment for biases
            v_bias(l) = new VectorD (b(l).b.dim)                          // initialize second moment for biases

        var sse_best_   = -0.0
        var (go, epoch) = (true, 1)
        cfor (go && epoch <= maxEpochs, epoch += 1) {                     // iterate over each epoch
            sse = 0.0
            val batches = permGen.igen.chop (nB)                          // permute indices &split into nB batches

            for ib <- batches do sse += updateWeight (x(ib), y(ib), epoch)   // update parameter array b

            collectLoss(sse)                                              // collect the loss per epoch
            val (b_best, sse_best) = stopWhen (b, sse)
            if b_best != null then
                for l <- b.indices do b(l).set (b_best(l))                // save the best parameters
                sse_best_ = sse_best                                      // save best in sse_best_
                go = false
            else
                if epoch % ADJUST_PERIOD == 0 then eta *= ADJUST_FACTOR   // adjust the learning rate
        } // cfor

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Compute the parameter array b updates based on the current batch.
         *  A step in the direction opposite to the gradient.
         *  @param x  the input matrix for the current batch
         *  @param y  the output matrix for the current batch
         */
        inline def updateWeight (x: MatrixD, y: MatrixD, t: Int): Double =
            z(0) = x                                                      // initial activation, which is the input matrix
            for l <- layers do z(l + 1) = f(l).fM(b(l) * z(l))            // feedforward and store all activations

            val yp = z.last                                               // predicted value of y
            val ε  = yp - y // negative of the error matrix
            δ(nl - 1) = f.last.dM(yp) ⊙ ε // delta for the last layer before output
            for l <- nl - 2 to 0 by -1 do
                δ(l) = f(l).dM(z(l + 1)) ⊙ (δ(l + 1) * b(l + 1).w.transpose)   // deltas for all previous hidden layers

            val α = eta / x.dim                                           // learning rate scaled by batch size

            for l <- layers do
                val g = z(l).ᵀ * δ(l)                                     // compute the gradient for each layer (transpose (ᵀ))
                val g_bias = δ(l).mean                                    // compute the gradient for the biases

                p(l) = g * (1 - β1) + p(l) * β1                           // update biased first moment estimates for weights
                v(l) = v(l) * β2 + g ~^ 2 * (1 - β2)                      // update biased second moment estimates for weights

                // Update biased first and second moment estimates for biases
                p_bias(l) = g_bias * (1 - β1) + p_bias(l) * β1
                v_bias(l) = v_bias(l) * β2 + (g_bias ~^ 2) * (1 - β2)

                // Compute bias-corrected first and second moment estimates for weights
                val pH = p(l) / (1 - β1 ~^ t)
                val vH = v(l) / (1 - β2 ~^ t)

                // Compute bias-corrected first and second moment estimates for biases
                val p_biasH = p_bias(l) / (1 - β1 ~^ t)
                val v_biasH = v_bias(l) / (1 - β2 ~^ t)

                // Compute parameter and bias updates using the Adam update rule
                val Δ = (pH / (vH ~^ 0.5 + EPSILON)) * α
                val Δ_bias = (p_biasH / (v_biasH ~^ 0.5 + EPSILON)) * α

                b(l) -= (Δ, Δ_bias)                                       // apply the updates to the parameters and biases
            end for

            ε.normFSq                                                     // return the sum of squared errors (sse) for this batch
        end updateWeight

//      debug ("optimize", s"parameters b = $b")
        if go then (sse, maxEpochs)                                       // return best and number of epochs
        else (sse_best_, epoch - upLimit)
    end optimize

end Optimizer_Adam

