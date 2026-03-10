
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Sainagesh Veeravalli
 *  @version 2.0
 *  @date    Thu May 11 15:38:07 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Gated Recurrent Unit (GRU) for Multivariate Time Series
 *
 *  @see https://www.frontiersin.org/articles/10.3389/fncom.2021.678158/full
 * 
 *  Translated from Matlab to Scala
 *  @see https://www.math.ucla.edu/~minchen/doc/BPTTTutorial.pdf
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/* This Matlab program tests the BPTT process we manually developed for GRU.
 * We calculate the gradients of GRU parameters with chain rule, and then
 * compare them to the numerical gradients to check whether our chain rule
 * derivation is correct.
 *
 * Here, we provided 2 versions of BPTT, backward_direct() and backward().
 * The former one is the direct idea to calculate gradient within each step
 * and add them up (O(n_seqˆ2) time) . The latter one is optimized to
 * calculate the contribution of each step to the overall gradient, which is
 * only O(n_seq) time.
 *
 * This is very helpful for people who want to implement GRU in Caffe since
 * Caffe does not support auto−differentiation. This is also very helpful for
 * the people who want to know the details about Back-propagation Through
 * Time algorithm in the Recurrent Neural Networks (such as GRU and LSTM)
 * and also get a sense on how auto−differentiation is possible.
 *
 * NOTE: We does not involve SGD training here. With SGD training, this
 * program would become a complete implementation of GRU which can be
 * trained with sequence data. However, since this is only a CPU serial
 * Matlab version of GRU, applying it on large datasets will be dramatically
 * slow.
 *
 * Matlab code:
 * by Minchen Li, at The University of British Columbia. 2016−04−21
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Standard RNN (GRU/LSTM) Formats for Multivariate Time Series Data:
 *  Matlab:
 *
 *  Keras: 3D format expected by GRU/LSTM is [samples, timesteps, features].
 *         => indexing [timestamp t, lags k, variable j]
 *  PyTorch:
 */

package scalation
package modeling
package forecasting
package neuralforecasting

import scala.math.log

import scalation.mathstat.{MatrixD, Plot, VectorD}
import scalation.random.{NormalMat, NormalVec_c}

import ActivationFun.{sigmoid_, softmax_, tanh_}
import MatrixD.outer

def log_ (x: VectorD): VectorD = x.map (log)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Gate` case class holds information on the gate's value and its partial derivatives.
 *  @param n_seq  the length of the time series
 *  @param n_mem  the size for hidden state (h) (dimensionality of memory)
 *  @param n_var  the number of variables
 */
case class Gate (n_seq: Int, n_mem: Int, n_var: Int):

    val v  = new MatrixD (n_seq, n_mem)                       // gate value: time x state
    var dU = new MatrixD (n_mem, n_var)                       // partial w.r.t. weight matrix U
    var dW = new MatrixD (n_mem, n_mem)                       // partial w.r.t. weight matrix W
    var db = new VectorD (n_mem)                              // partial w.r.t. bias vector b

    def apply (t: Int): VectorD = v(t)

    def update (t: Int, vv: VectorD): Unit = v(t) = vv

    def += (dIn: VectorD, x_t: VectorD, h_tm1: VectorD): Unit =
           { dU += outer (dIn, x_t); dW += outer (dIn, h_tm1); db += dIn }

end Gate


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GRU` class implements Gated Recurrent Unit (GRU) via Back Propagation Through
 *  Time (BPTT).  At each time point x_t, there is a vector representing several variables
 *  or the encoding of a word.  Intended to work for guessing the next work in a sentence
 *  or for multi-horizon forecasting.
 *  Time series: (x_t: t = 0, 1, ..., n_seq-1) where n_seq is the number of time points/words
 *  @param x       the input sequence/time series
 *  @param y       the output sequence/time series
 *  @param fname   the feature/variable names
 *  @param n_mem   the size for hidden state (h) (dimensionality of memory)
 */
class GRU (x: MatrixD, y: MatrixD, fname: Array [String] = null, n_mem: Int = 8):   // 4

    private val CLASSIF    = false                                     // whether to apply classification (e.g., guess next word) or forecast a value 
    private val max_epochs = 31                                        // maximum number of iterations
    private val eta        = 0.001                                     // the learning rate (use 0.25 for gRUTest)

    private val n_seq = x.dim                                          // e.g., 20, number of words in a sentence (including start and end symbol)
    private val n_var = x.dim2                                         // e.g., 64, number of variables or distinct words (vocabulary size)
                                                                       // since we will only use one sentence for training,
                                                                       // this is also the total steps during training.

    private val _1 = VectorD.one (n_mem)                               // vector of all ones for e.g., 1 - z

    // initialize parameters (weights and biases)
    private val rmg1 = NormalMat (n_mem, n_var, 0.0, 0.01)             // random (Normal) matrix generators
    private val rmg2 = NormalMat (n_mem, n_mem, 0.0, 0.01)
    private val rmg3 = NormalMat (n_var, n_mem, 0.0, 0.01)
    private val rvg1 = NormalVec_c (n_mem, 0.0, 0.01)                  // random (Normal) vector generators
    private val rvg3 = NormalVec_c (n_var, 0.0, 0.01)

    private val Uz   = rmg1.gen                                        // parameters for update gate z
    private val Wz   = rmg2.gen
    private val b_z  = rvg1.gen

    private val Ur   = rmg1.gen                                        // parameters for reset gate r
    private val Wr   = rmg2.gen
    private val b_r  = rvg1.gen

    private val Uc   = rmg1.gen                                        // parameters for candidate state mixin c
    private val Wc   = rmg2.gen
    private val b_c  = rvg1.gen

    // decoder for generating output
    private val V    = rmg3.gen                                        // decoder weight matrix
    private val b_V  = rvg3.gen                                        // decoder bias vector

    private val z = Gate (n_seq, n_mem, n_var)                         // update gate z
    private val r = Gate (n_seq, n_mem, n_var)                         // reset gate r
    private val c = Gate (n_seq, n_mem, n_var)                         // candidate state mixin c

    private val h_m1 = rvg1.gen                                        // hidden state @ t = -1 (m1 means minus 1)
    private val h    = new MatrixD (n_seq, n_mem)                      // hidden state h
    private val yp   = new MatrixD (n_seq, n_var)                      // predicted output
    private val L    = new VectorD (n_seq)                             // store loss function values

    // the partial derivative of weights and biases (outside gates)
    private val dh_m1 = new VectorD (h_m1.dim)
    private var db_V: VectorD = null
    private val dV    = new MatrixD (V.dim, V.dim2)

    if fname != null then println (s"GRU: fname = $fname")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the GRU using simple gradient descent.
     */
    def train (): Unit =
        for it <- 1 to max_epochs do
            forward ()            // forward propagate: get intermediate and output results

            println (s"train: for epoch $it: loss function L = $L")
            banner (s"train: for epoch $it: total loss function L.sum = ${L.sum}")

            backward ()           // back propagate: calculate gradients (partial derivatives)

            update_params ()      // update parameters (weights and biases)
        end for
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the RNN predictions.
     */
    def test (): Unit =
        new Plot (null, y(?, 0), yp(?, 0), "Plot of y vs yp for RNN", lines = true)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward propagate calculates yp, loss and intermediate variables for each step.
     */
    def forward (): Unit =
        for t <- 0 until n_seq do
            val h_pre = if t == 0 then h_m1 else h(t-1)                // get previous hidden state
            z(t) = sigmoid_ (Uz * x(t) + Wz * h_pre + b_z)             // update gate
            r(t) = sigmoid_ (Ur * x(t) + Wr * h_pre + b_r)             // reset gate
            c(t) = tanh_ (Uc * x(t) + Wc * (h_pre * r(t)) + b_c)       // candidate state
            h(t) = (_1 - z(t)) * h_pre + z(t) * c(t)                   // hidden state
            if CLASSIF then
                yp(t) = softmax_ (V * h(t) + b_V)                      // activation: softmax for classification
                L(t)  = (-y(t) * log_ (yp(t))).sum                     // cross-entropy loss function
            else
                yp(t) = V * h(t) + b_V                                 // activation: id for forecasting
                L(t)  = (y(t) - yp(t)).normSq	                       // sse loss function
        end for
    end forward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward propagate to calculate gradients using chain rules in O(n_seq) time.
     *  FIX - add option of using sse loss function and fix affected partial derivatives
     */
    def backward (): Unit =

        import ActivationFun.{sigmoidD, tanhD}

        // start back-propagation with the final/feed-forward (ff) layer (uses id for activation)

        val e = yp - y                                                 // negative error matrix
        db_V  = e.sumVr                                                // vector of row sums
        for t <- 0 until n_seq do dV += outer (e(t), h(t))             // outer vector product
        val dh_ff = e * V                                              // partial w.r.t. h: n_seq by n_mem matrix
        var dh = new VectorD (dh_ff.dim2)                              // hold partial for hidden state (dh) @ time t
        var dIn, dhr: VectorD = null

        // calculate the derivative contribution of each step and add them up

        for t <- n_seq-1 to 1 by -1 do                                 // move back in time to t = 1
            dh += dh_ff(t)                                             // update partial for hidden state (dh) @ time t
            val dh_bk = dh                                             // save dh

            dIn = dh * (_1 - z(t)) * tanhD (c(t))                      // input to tanh for candidate mixin c
            c  += (dIn, x(t), h(t-1) * r(t))                           // update partials for c mixin
            dhr = Wc.ᵀ * dIn                                           // ᵀ => matrix transpose
            dh  = dhr * r(t)

            dIn = dhr * h(t-1) * sigmoidD (r(t))                       // input to sigmoid reset gate r
            r  += (dIn, x(t), h(t-1))                                  // update partials for r gate
            dh += Wr.ᵀ * dIn + dh_bk * z(t)

            dIn = dh_bk * (c(t) - h(t-1)) * sigmoidD (z(t))            // input to sigmoid update gate z
            z  += (dIn, x(t), h(t-1))                                  // update partials for z gate
            dh += Wz.ᵀ * dIn
        end for

        // end case @ time t = 0 -> use h_m1 for hidden state

        dh += dh_ff(0)                                                 // update partial for hidden state (dh) @ t = 0

        dIn = dh * (_1 - z(0)) * tanhD (c(0))
        c  += (dIn, x(0), h_m1 * r(0))                                 // update partials for c mixin @ t = 0
        dhr = Wc.ᵀ * dIn
        dh_m1 += dhr * r(0)

        dIn = dhr * h_m1 * sigmoidD (r(0))
        r  += (dIn, x(0), h_m1)                                        // update partials for r gate @ t = 0
        dh_m1 += Wr.ᵀ * dIn + dh * z(0)

        dIn = dh * (c(0) - h_m1) * sigmoidD (z(0))
        z  += (dIn, x(0), h_m1)                                        // update partials for z gate @ t = 0
        dh_m1 += Wz.ᵀ * dIn
    end backward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Based on the calculated partial derivatives, update the parameters (weights
     *  and biases).
     */
    def update_params (): Unit =
        // update gate (z)
        Uz  -= z.dU * eta
        Wz  -= z.dW * eta
        b_z -= z.db * eta

        // reset gate (r)
        Ur  -= r.dU * eta
        Wr  -= r.dW * eta
        b_r -= r.db * eta

        // candidate state (c)
        Uc  -= c.dU * eta
        Wc  -= c.dW * eta
        b_c -= c.db * eta

        // output layer
        V   -= dV * eta
        b_V -= db_V * eta
    end update_params

end GRU


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GRU` companion object provides factory methods.
 */
object GRU:

    import ActivationFun._

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `GRU` with automatic rescaling from a data matrix and response matrix.
     *  @param x       the input/data matrix
     *  @param y       the output/response matrix
     *  @param fname   the feature/variable names
     *  @param n_mem   the size of the hidden state (dimensionality of memory)
     */
    def rescale (x: MatrixD, y: MatrixD, fname: Array [String] = null, n_mem: Int = 4): GRU =
        val x_s = rescaleX (x, f_sigmoid)
        val y_s = rescaleY (y, f_sigmoid)._1

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new GRU (x_s, y_s, fname, n_mem)
    end rescale

end GRU


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Generate a fake sequence dataset:  generate only one sentence for training.
 *  Only for testing. Needs to be changed to read in training data from files.
 *  The words are one-hot encoded into a column vector.
 *  @param n_seq  the sequence size (number of time points/words)
 *  @param n_var  the number of variables/word encoding size
 */
def genSequenceData (n_seq: Int, n_var: Int): (MatrixD, MatrixD) =

    import scalation.random.Randi

    assert (n_var > 2)                                             // for start and end of sentence symbols
    assert (n_seq > 0)

    // define start and end of sentence in the vocabulary
    val SENTENCE_START = new VectorD (n_var)                       // start: [1, 0, 0, ...]
    SENTENCE_START(0) = 1
    val SENTENCE_END = new VectorD (n_var)                         // end:   [0, 1, 0, ...]
    SENTENCE_END(1) = 1

    println (s"SENTENCE_START = $SENTENCE_START")
    println (s"SENTENCE_END   = $SENTENCE_END")

    // generate sentence
    val i_ran = Randi (0, n_var-3)                                 // random integer generator
    val z_t = new MatrixD (n_seq-1, n_var)                         // leave one slot for SENTENCE START
    for t <- 0 until n_seq-1 do
        // generate a random word excludes start and end symbol
        z_t(t, i_ran.igen+2) = 1                                   // set a 1 in position to indicate a word
    end for

    val x_t = SENTENCE_START +: z_t                                // training input matrix (prepend vector)
    val y_t = z_t :+ SENTENCE_END                                  // training output matrix (append vector)

    (x_t, y_t)
end genSequenceData


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `gRUTest` main function tests the `GRU` class on randomly generated
 *  sequence data meant to represent encoded words 
 *  > runMain scalation.modeling.forecasting.neuralforecasting.gRUTest
 */
@main def gRUTest (): Unit =

    val n_seq = 8
    val n_var = 5

    val (x_t, y_t) = genSequenceData (n_seq, n_var)

    println (s"x_t = $x_t")
    println (s"y_t = $y_t")

    banner ("Create a Gated Recurrent Unit (GRU)")
    val mod = new GRU (x_t, y_t)
    mod.train ()
    mod.test ()

end gRUTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `gRUTest2` main function tests the `GRU` class on sequence data read as words
 *  in a file that encoded and pass into `GRU`
 *  > runMain scalation.modeling.forecasting.neuralforecasting.gRUTest2
 */
@main def gRUTest2 (): Unit =

    println ("read words from a text file")

//  FIX - find example text

end gRUTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `gRUTest3` main function tests the `GRU` class on sequence/time series data
 *  corresponding to the lake level dataset using multiple lags.
 *  > runMain scalation.modeling.forecasting.neuralforecasting.gRUTest3
 */
@main def gRUTest3 (): Unit =

    import Example_LakeLevels.y
    import MakeMatrix4TS._
    val hh  = 2                                                       // forecasting horizon - FIX - currently lags == hh
    hp("p") = 2                                                       // number of lags to include

    val y_s = scaleV (extreme (y), (-2.0, 2.0))(y)                    // rescale y to active domain of sigmoid, tanh

    val x  = ARY.buildMatrix (y_s, hp)                                // column for each lag
    val yy = makeMatrix4Y (y_s, hh)

    println (s"x.dims = ${x.dims}, yy.dims = ${yy.dims}")

    banner ("Create a Gated Recurrent Unit (GRU)")
    val mod = new GRU (x, yy)                                         // call constructor
    mod.train ()
    mod.test ()

end gRUTest3

