
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John A. Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Fri Oct 13 22:21:37 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Transformer Encoder Layer
 *
 *  @see sebastianraschka.com/blog/2023/self-attention-from-scratch.html
 *  @see arxiv.org/pdf/1706.03762.pdf (main paper)
 */

package scalation
package modeling
package forecasting
package neuralforecasting

import scalation.mathstat._
import scalation.random.{RandomMatD, RandomTenD}

import ActivationFun._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TrfEncoderLayer` class consists of a Multi-Head Self-Attention and a Feed-Forward
 *  Neural Network (FFNN) sub-layers.
 *  @see pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html#torch.nn.TransformerEncoderLayer
 *  @param x           the input data matrix after embedding (number of rows/instances by embedding dimension)      
 *  @param heads       the number of attention heads (e.g., 1 to 8)
 *  @param f           the activation function family (used by alinear1)
 *  @param p_drop      the probability of setting an element to zero in a dropout layer (e.g., .0 to .5)
 *  @param norm_eps    a small values used in normalization to avoid divide by zero
 *  @param norm_first  whether layer normalization should be done first (see apply method)
 */
class TrfEncoderLayer (x: MatrixD, heads: Int = 4, f: AFF = f_reLU,
                       initW: Array [MatrixD] = null,
                       p_drop: Double = 0.2, norm_eps: Double = 1E-5, norm_first: Boolean = false)
      extends Attention (x.dim2, x.dim2 * heads, heads, x.dim2):

    private val m_x   = x.dim                                             // size/number of instances in input x
    private val d_k   = x.dim2                                            // dimension of query (Q) and key (K)
    private val d_v   = d_k                                               // dimension of value (V) for simplicity same, for flexibility different
    private val d_mod = heads * d_v                                       // the dimensionality of the model (d_model)
    private val d_ff  = 4 * d_mod                                         // the size of the hidden layer (n_z) in the Feed-Forward Neural Network

    private val rmg = RandomMatD (d_k, m_x, 1)  
    private val (w_q, w_k, w_v) = if initW == null then (rmg.gen, rmg.gen, rmg.gen)
                                  else (initW(0), initW(1), initW(2))     // weight matrices for query q, key k, value v

    val rtg   = RandomTenD (heads, d_mod, d_k, 1)                         // random (0, 1) tensor generator for q, k
    val rtg_v = RandomTenD (heads, d_mod, d_v, 1)                         // random (0, 1) tensor generator for v
    val rmg_o = RandomMatD (heads*d_v, d_mod, 1)                          // random (0, 1) matrix generator for for w_o

    private val wt_q = rtg.gen                                            // MH query weight tensor:   heads x d_mod x d_k
    private val wt_k = rtg.gen                                            // MH key weight tensor:     heads x d_mod x d_k
    private val wt_v = rtg_v.gen                                          // MH value weight tensor;   heads x d_mod x d_val
    private val w_o  = rmg_o.gen                                          // MH overall weight matrix: d_mod x d_mod

    private val dropout_sa = DropoutLayer (p_drop)                        // dropout layer (sa_block)

    private val alinear1   = DenseLayer (d_mod, d_ff, f)                  // activated linear layer (ff_block)
    private val dropout1   = DropoutLayer (p_drop)                        // dropout layer (ff_block)
    private val linear2    = DenseLayer (d_ff, d_mod)                     // linear layer (ff_block)
    private val dropout2   = DropoutLayer (p_drop)                        // dropout layer (ff_block)

    private val norm1      = LayerNorm (true, norm_eps)                   // normalization layer (apply)
    private val norm2      = LayerNorm (true, norm_eps)                   // normalization layer (apply)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass:  Compute this encoder layer's result z by using Multi-Head Self-Attention
     *  followed by a Feed-Forward Neural Network.
     */
    def forward (): MatrixD =
        banner ("1. Multi-Head Self-Attention: query q, key k, value v")
        banner ("2. Feed-Forward Neural Network")

        var z: MatrixD = null
        if norm_first then
            z = x + sa_block (norm1 (x))
            z = z + ff_block (norm2 (z))
        else
            z = norm1 (x + sa_block (x))
            z = norm2 (z + ff_block (z))
        z
    end forward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Multi-Head Self-Attention result.
     *  @param x  the input matrix
     */
    def sa_block (x: MatrixD): MatrixD =
        val (q, k, v) = queryKeyValue (x, w_q, w_k, w_v)
        dropout_sa (attentionMH (q, k, v, wt_q, wt_k, wt_v, w_o))
    end sa_block

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Feed-forward Neural Network result.
     *  @param x  the input matrix
     */
    def ff_block (x: MatrixD): MatrixD =
        dropout2 (linear2 (dropout1 (alinear1 (x))))
    end ff_block

end TrfEncoderLayer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `trfEncoderTest` main function illustrates the use of Gradient Descent (GD) to optimize
 *  the weights/parameters of a simple neural network (3-layer (1 hidden) Neural Network).
 *  @see pub.aimind.so/transformer-model-and-variants-of-transformer-chatgpt-3d423676e29c (URL)
 *  > runMain scalation.modeling.forecasting.neuralforecasting.trfEncoderTest
 */
@main def trfEncoderTest (): Unit =

    import math.sqrt
    import ActivationFun.f_softmax

    // three input tokens after embedding (each embedding vector has size 4)
    val x = MatrixD ((3, 4), 1, 0, 1, 0,                // input x0 
                             0, 2, 0, 2,                // input x1
                             1, 1, 1, 1)                // input x2

    println (s"input (after embedding) x = $x")

    val d_k = 3                                         // dimensionality for Q, K, and V (if different need d_v)
//  val heads = 1                                       // number of attention heads (d_model = d_k * heads)

    val wQ = MatrixD ((4, 3), 1, 0, 1,                  // Query weight matrix
                              1, 0, 0,
                              0, 0, 1,
                              0, 1, 1) 
    val wK = MatrixD ((4, 3), 0, 0, 1,                  // Key weight matrix
                              1, 1, 0,
                              0, 1, 0,
                              1, 1, 0) 
    val wV = MatrixD ((4, 3), 0, 2, 0,                  // Value weight matrix
                              0, 3, 0,
                              1, 0, 3,
                              1, 1, 0)

    val q = x * wQ                                      // Query: size of input x d_k
    val k = x * wK                                      // Key:   size of input x d_k
    val v = x * wV                                      // Value: size of input x d_k (or d_v if different)

    val qk  = q * k.ᵀ                                   // repeated dot product
    val sq_ = qk / sqrt (d_k)                           // corrrect scaled dot product formula for sqk
    val sqk = qk / 1.0                                  // approximation used by URL
    val scr = f_softmax.fM (sqk)                        // attention scores
    val att = scr * v                                   // attention (Q, K, V)

    println (s"""
    wQ  = $wQ
    wK  = $wK
    wV  = $wV
    q   = $q
    k   = $k
    v   = $v
    qk  = $qk
    sq_ = $sq_
    sqk = $sqk
    scr = $scr
    att = $att
    """)

end trfEncoderTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `trfEncoderTest2` main function illustrates the use of Gradient Descent (GD) to optimize
 *  the weights/parameters of a simple neural network (3-layer (1 hidden) Neural Network).
 *  @see
 *  > runMain scalation.modeling.forecasting.neuralforecasting.trfEncoderTest2
 *
@main def trfEncoderTest2 (): Unit =

    import Attention.x

    val n_var = x.dim2                                                    // number of variables in input vector x_t
    println (s"n_var = $n_var")
    val n_mod = 72                                                        // size of each query/key vector (q_t, k_t)
    val heads = 3                                                         // number of attention heads
    val n_val = 28                                                        // size of the value vector v_t

    val trf = new TrfEncoderLayer (n_var, n_mod, heads, n_val)
    println (trf.forward ())

end trfEncoderTest2
 */

