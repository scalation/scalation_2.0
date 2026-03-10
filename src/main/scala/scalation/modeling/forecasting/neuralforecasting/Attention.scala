
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Sep  4 13:09:52 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Context and Attention for Transformers
 *
 *  @see https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
 *  @see https://arxiv.org/pdf/1706.03762.pdf (main paper)
 */

package scalation
package modeling
package forecasting
package neuralforecasting

import scala.math.sqrt

import scalation.mathstat._
import scalation.random.{RandomMatD, RandomTenD}

import ActivationFun.f_softmax

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Attention` trait provides methods for computing context vectors, single-head
 *  attention matrices and multi-head attention matrices.
 *  @param n_var  the size of the input vector x_t (number of variables)
 *  @param n_mod  the size of the output (dimensionality of the model, d_model)
 *  @param heads  the number of attention heads
 *  @param n_v    the size of the value vectors
 */
trait Attention (n_var: Int, n_mod: Int = 512, heads: Int = 8, n_v: Int = -1):

    private val debug = debugf ("Attention", true)                       // debug function

    val n_k = n_mod / heads                                              // size per head (dimensionality d_k, d_v)
    debug ("init", s"n_k = $n_k")

    val n_val = if n_v > 0 then n_v else n_k
    val rmg   = RandomMatD (n_mod, n_var, 1)                             // random (0, 1) matrix generator for q, k
    val rmg_v = RandomMatD (heads*n_val, n_var, 1)                       // random (0, 1) matrix generator for v

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Query, Key, Value matrices from the given input and weight matrices.
     *  @param x    the input matrix
     *  @param w_q  the weight matrix for query Q
     *  @param w_v  the weight matrix for key K
     *  @param w_v  the weight matrix for value V
     */
    def queryKeyValue (x: MatrixD, w_q: MatrixD, w_k: MatrixD, w_v: MatrixD): (MatrixD, MatrixD, MatrixD) =
        (x * w_q.ᵀ, x * w_k.ᵀ, x * w_v.ᵀ)
    end queryKeyValue

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a Context Vector from the given query at time t (q_t), key (K) and value (V).
     *  @param q_t  the query vector at time t (based on input vector x_t)
     *  @param k    the key matrix K
     *  @param v    the value matrix V
     */
    def context (q_t: VectorD, k: MatrixD, v: MatrixD): VectorD =
        val root_n = sqrt (q_t.dim)
        v.ᵀ * f_softmax.f_ (k * (q_t / root_n))
    end context

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a Self-Attention Weight Matrix from the given query (Q), key (K) and value (V).
     *  @param q  the query matrix Q (q_t over all time)
     *  @param k  the key matrix K
     *  @param v  the value matrix V
     */
    def attention (q: MatrixD, k: MatrixD, v: MatrixD): MatrixD =
        val root_n = sqrt (q.dim2)
        f_softmax.fM (q * (k.ᵀ / root_n)) * v
    end attention

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a Multi-Head, Self-Attention Weight Matrix by taking attention for each head
     *  and concatenating them; finally multiplying by the overall weight matrix w_o.
     *  The operator ++^ concatenates matrices column-wise.
     *  @param q    the query matrix Q (q_t over all time)
     *  @param k    the key matrix K
     *  @param v    the value matrix V
     *  @param w_q  the weight tensor for query Q (w_q(i) matrix for i-th head)
     *  @param w_v  the weight tensor for key K (w_k(i) matrix for i-th head)
     *  @param w_v  the weight tensor for value V (w_v(i) matrix for i-th head)
     *  @param w_o  the overall weight matrix to be applied to concatenated attention
     */
    def attentionMH (q: MatrixD, k: MatrixD, v: MatrixD,
                     w_q: TensorD, w_k: TensorD, w_v: TensorD,
                     w_o: MatrixD): MatrixD =
        // x: (6, 16) * w_q: (72, 16) --> q=x.w_q.T : (6, 72)
        // q: (6, 72) * wt_q: (72, 72) = q.wt_q: (6, 72)   --> split head: (3, 6, 24)
        // then go to Scaled-dot (3, 6, 24) --> combine head: (6, 72) * w_o: (72, 72)
        // aw: (6, 72)

        debug ("attentionMH", s"q.dims = ${q.dims}, k.dims: ${k.dims}, v.dims: ${v.dims}")
        debug ("attentionMH", s"w_q.dims = ${w_q.dims}, w_k.dims = ${w_k.dims}, w_v.dims = ${w_v.dims}")
        debug ("attensionMH", s"w_o.dims = ${w_o.dims}")

        println (s"(q * w_q(0)).dims: ${(q * w_q(0)).dims}")
        println (s"(k * w_k(0)).dims: ${(k * w_k(0)).dims}")
        println (s"(v * w_v(0)).dims: ${(v * w_v(0)).dims}")

        var att = attention (q * w_q(0), k * w_k(0), v * w_v(0))
        for i <- 1 until heads do
            att = att ++^ attention (q * w_q(i), k * w_k(i), v * w_v(i))
        debug (s"attentionMH", s"att = $att")
        att * w_o
    end attentionMH

end Attention


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Attention` object contains sample a input matrix from
 *  @see https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html
 *  The example is from 6 words with 16 dimensional encoding.
 */
object Attention:

    val x = MatrixD ((6, 16), 0.3374, -0.1778, -0.3035, -0.5880,  0.3486,  0.6603, -0.2196, -0.3792,    // row 0
                              0.7671, -1.1925,  0.6984, -1.4097,  0.1794,  1.8951,  0.4954,  0.2692,

                              0.5146,  0.9938, -0.2587, -1.0826, -0.0444,  1.6236, -2.3229,  1.0878,    // row 1
                              0.6716,  0.6933, -0.9487, -0.0765, -0.1526,  0.1167,  0.4403, -1.4465,

                              0.2553, -0.5496,  1.0042,  0.8272, -0.3948,  0.4892, -0.2168, -1.7472,    // row 2
                             -1.6025, -1.0764,  0.9031, -0.7218, -0.5951, -0.7112,  0.6230, -1.3729,

                             -1.3250,  0.1784, -2.1338,  1.0524, -0.3885, -0.9343, -0.4991, -1.0867,    // row 3
                              0.8805,  1.5542,  0.6266, -0.1755,  0.0983, -0.0935,  0.2662, -0.5850,

                             -0.0770, -1.0205, -0.1690,  0.9178,  1.5810,  1.3010,  1.2753, -0.2010,    // row 4
                              0.4965, -1.5723,  0.9666, -1.1481, -1.1589,  0.3255, -0.6315, -2.8400,

                              0.8768,  1.6221, -1.4779,  1.1331, -1.2203,  1.3139,  1.0533,  0.1388,    // row 5
                              2.2473, -0.8036, -0.2808,  0.7697, -0.6596, -0.7979,  0.1838,  0.2293)

    val m   = x.dim                                                       // number of time points
    val n   = x.dim2                                                      // size of input x_t
    println (s"m = $m, n = $n")
end Attention

import Attention._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `attentionTest` main function tests the `context` and `attention` top-level functions.
 *  Test Self-Attention.
 *  > runMain scalation.modeling.forecasting.neuralforecasting.attentionTest
 */
@main def attentionTest (): Unit =

    val n_var = x.dim2                                                    // number of variables in input vector x_t
    val n_mod = 24                                                        // size of each query/key vector (q_t, k_t)
    val n_val = 28                                                        // size of the value vector v_t
    val heads = 1                                                         // number of attention heads
    object att extends Attention (n_var, n_mod, heads, n_val)

    val w_q = att.rmg.gen
    val w_k = att.rmg.gen
    val w_v = att.rmg_v.gen
    val (q, k, v) = att.queryKeyValue (x, w_q, w_k, w_v)

    banner ("Dimensions for input x, query q, key k, value v")
    println (s"x.dims = ${x.dims}")
    println (s"q.dims = ${q.dims}")
    println (s"k.dims = ${k.dims}")
    println (s"v.dims = ${v.dims}")

    banner ("Attention Matrix")
    val aw = att.attention (q, k, v)
    println (s"aw.dims = ${aw.dims}")
    println (s"aw      = $aw")

    banner ("Context Vectors Collected into Matrix")
    val cxt = new MatrixD (aw.dim, aw.dim2)
    println (s"cxt.dims = ${cxt.dims}")
    for i <- q.indices do cxt(i) = att.context (q(i), k, v)
    println (s"cxt      = $cxt")
    assert (cxt =~ aw)
    println ("succeed")
  
end attentionTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `attentionTest2` main function tests the `attentionMH` top-level function.
 *  Test Multi-Head, Self-Attention.
 *  > runMain scalation.modeling.forecasting..neuralforecastingattentionTest2
 */
@main def attentionTest2 (): Unit =

    val n_var = x.dim2                                                    // number of variables in input vector x_t
    println (s"n_var = $n_var")
    val n_mod = 72                                                        // size of each query/key vector (q_t, k_t)
    val heads = 3                                                         // number of attention heads
    val n_val = 28                                                        // size of the value vector v_t
    object att extends Attention (n_var, n_mod, heads, n_val)

    val w_q = att.rmg.gen
    val w_k = att.rmg.gen
    val w_v = att.rmg.gen
    val (q, k, v) = att.queryKeyValue (x, w_q, w_k, w_v)

    banner ("Dimensions for input x, query q, key k, value v")
    println (s"x.dims = ${x.dims}")
    println (s"q.dims = ${q.dims}")
    println (s"k.dims = ${k.dims}")
    println (s"v.dims = ${v.dims}")

    // Multi-Head (MH)
    println (s"att.n_k: ${att.n_k}")
    val rtg   = RandomTenD (heads, n_mod, att.n_k, 1)                     // random (0, 1) tensor generator for q, k
    val rtg_v = RandomTenD (heads, n_mod, n_val, 1)                       // random (0, 1) tensor generator for v
    val rmg_o = RandomMatD (heads*n_val, n_mod, 1)                        // random (0, 1) matrix generator for for w_o

    val wt_q = rtg.gen                                                    // MH query weight tensor:   heads x n_mod x n_k
    val wt_k = rtg.gen                                                    // MH key weight tensor:     heads x n_mod x n_k
    val wt_v = rtg_v.gen                                                  // MH value weight tensor;   heads x n_mod x n_k
    val w_o  = rmg_o.gen                                                  // MH overall weight matrix: n_mod x n_mod 

    banner ("Dimensions for query wt_q, key wt_k, value wt_v, overall w_o")
    println (s"wt_q.dims = ${wt_q.dims}")
    println (s"wt_k.dims = ${wt_k.dims}")
    println (s"wt_v.dims = ${wt_v.dims}")
    println (s"w_o.dims  = ${w_o.dims}")

    banner ("Multi-Head Attention Matrix")
    val aw = att.attentionMH (q, k, v, wt_q, wt_k, wt_v, w_o)
    println (s"aw.dims = ${aw.dims}")
//  println (s"aw      = $aw")

end attentionTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `attentionTest3` main function tests the `attention` and `context` top-level function.
 *  Test Self-Attention.  Read in weight matrices to compare with PyTorch.
 *  > runMain scalation.modeling.forecasting.neuralforecasting.attentionTest3
 */
@main def attentionTest3 (): Unit =

    val n_var = x.dim2                                                    // number of variables in input vector x_t
    println (s"n_var = $n_var")
    val n_mod = 24                                                        // size of each query/key vector (q_t, k_t, v_t)
    val heads = 1                                                         // number of attention heads
    val n_val = 28
    object att extends Attention (n_var, n_mod, heads, n_val)

    import scalation.database.table.Table

    val data_q = Table.load ("W_query.csv", "W_query", n_var, "0")
    val W_query = data_q.toMatrix ()     // Array(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
//  println (W_query)
    val data_k = Table.load ("W_key.csv", "W_key", n_var, "0")
    val W_key = data_k.toMatrix ()

    val data_v = Table.load ("W_value.csv", "W_value", n_var, "0")
    val W_value = data_v.toMatrix ()

    val (q, k, v) = att.queryKeyValue (x, W_query, W_key, W_value)

    banner ("Dimensions for input x, query q, key k, value v")
    println (s"x.dims = ${x.dims}")
    println (s"q.dims = ${q.dims}")
    println (s"k.dims = ${k.dims}")
    println (s"v.dims = ${v.dims}")
    banner("q Matrix")
    println (q(1))
    banner("k Matrix")
    println (k(1))
    banner("v Matrix")
    println (v(1))

    // Attention
    banner ("Attention Matrix")
    val aw1 = att.attention (q, k, v)
    println (s"aw1.dims = ${aw1.dims}")
    println (s"aw1(1)   = ${aw1(1)}")

    banner ("Context Vectors Collected into Matrix")
    val cxt = new MatrixD (aw1.dim, aw1.dim2)
    println (s"cxt.dims = ${cxt.dims}")
    for i <- q.indices do cxt(i) = att.context (q(i), k, v)
    println (s"cxt(1) = ${cxt(1)}")
    assert (cxt =~ aw1)
    println ("succeed")

end attentionTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `attentionTest4` main function tests the `attentionMH` top-level function.
 *  Test Multi-Head, Self-Attention.  Read in weight matrices to compare with PyTorch.
 *  > runMain scalation.modeling.forecasting.neuralforecasting.attentionTest4
 */
@main def attentionTest4 (): Unit =

    val n_var = x.dim2                                                    // number of variables in input vector x_t
    println (s"n_var = $n_var")
    val n_mod = 72                                                        // size of each query/key vector (q_t, k_t, v_t)
    val heads = 3                                                         // number of attention heads
    val n_val = 24
    object att extends Attention (n_var, n_mod, heads, n_val)

    val q = MatrixD.load ("Q.csv")
    val k = MatrixD.load ("K.csv")
    val v = MatrixD.load ("V.csv")

    // Multi-Head (MH)
    banner ("Dimensions for input x, query q, key k, value v")
    println (s"x.dims = ${x.dims}")
    println (s"q.dims = ${q.dims}")
    println (s"k.dims = ${k.dims}")
    println (s"v.dims = ${v.dims}")
    banner ("q Matrix")
    println (q(1))
    banner ("k Matrix")
    println (k(1))
    banner ("v Matrix")
    println (v(1))

    val w_q = MatrixD.load ("w_q.csv")
    val w_k = MatrixD.load ("w_k.csv")
    val w_v = MatrixD.load ("w_v.csv")
    val w_o = MatrixD.load ("w_o.csv")

    banner ("Dimensions")
    println (s"w_q.dims = ${w_q.dims}")
    println (s"w_k.dims = ${w_k.dims}")
    println (s"w_v.dims = ${w_v.dims}")
    banner ("w_q Matrix")
    println (w_q(1))
    banner ("w_k Matrix")
    println (w_k(1))
    banner ("w_v Matrix")
    println (w_v(1))
    banner ("w_o Tensors")
    println (w_o(1))

    val wt_q = new TensorD (heads, n_mod, att.n_k)                        // MH query weight tensor:   heads x n_mod x n_k
    val wt_k = new TensorD (heads, n_mod, att.n_k)                        // MH key weight tensor:     heads x n_mod x n_k
    val wt_v = new TensorD (heads, n_mod, att.n_k)                        // MH value weight tensor;   heads x n_mod x n_k

    for i <- 0 until heads do                                             // (72, 72)  --->  (3, 72, 24)
        val Mq = w_q(?, (i*att.n_k) to ((i + 1) * att.n_k)-1)
        wt_q(i) = Mq

        val Mk = w_k(?, (i * att.n_k) to ((i + 1) * att.n_k) - 1)
        wt_k(i) = Mk

        val Mv = w_v(?, (i * att.n_k) to ((i + 1) * att.n_k) - 1)
        wt_v(i) = Mv
    end for

    banner ("wt_q, key wt_k, value wt_v after splitting heads (72, 72)  --->  (3, 72, 24)")
    println (s"wt_q.dims = ${wt_q.dims}")
    println (s"wt_k.dims = ${wt_k.dims}")
    println (s"wt_v.dims = ${wt_v.dims}")
    println (s"w_o.dims = ${w_o.dims}")
    banner ("wt_q Tensors")
    println (wt_q(0)(1))
    println (wt_q(1)(1))
    println (wt_q(2)(1))
    banner ("wt_k Tensors")
    println (wt_k(0)(1))
    println (wt_k(1)(1))
    println (wt_k(2)(1))
    banner ("wt_v Tensors")
    println (wt_v(0)(1))
    println (wt_v(1)(1))
    println (wt_v(2)(1))

    banner ("after linear")
    println (s"q * wt_q(1): ${(q * wt_q(1))}")
    println (s"k * wt_k(1): ${(k * wt_k(1))}")
    println (s"v * wt_v(1): ${(v * wt_v(1))}")

    banner ("Multi-Head Attention Matrix")
    val aw = att.attentionMH (q, k, v, wt_q, wt_k, wt_v, w_o)
    banner ("att after w_o")
    println (s"aw.dims = ${aw.dims}")
    for i <- 0 until 3 do println (s"aw($i)      = ${aw(i)}\n\n")

end attentionTest4


def patch (y: VectorD, ps: Int): MatrixD =
    val m = y.dim / ps
    val x = new MatrixD (m, ps)
    for i <- x.indices; j <- x.indices2 do x(i, j) = y(i * ps + j)
    x
end patch


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `attentionTest5` main function tests the `attention` top-level function.
 *  Test Self-Attention with fixed weights
 *  > runMain scalation.modeling.forecasting.neuralforecasting.attentionTest5
 */
@main def attentionTest5 (): Unit =

    import Example_Covid.y

    val ps = 4
    val x  = patch (y, 4)

    println (s"patched time series x = $x")

    val n_var = ps                                                        // number of variables in input vector x_t
    val n_mod = ps                                                        // size of each query/key vector (q_t, k_t)
    val n_val = ps                                                        // size of the value vector v_t
    val heads = 1                                                         // number of attention heads
    object att extends Attention (n_var, n_mod, heads, n_val)

    val w_q = new MatrixD (x.dim, ps); w_q.setAll (1.0/ps)
    val w_k = new MatrixD (x.dim, ps); w_k.setAll (1.0/ps)
    val w_v = new MatrixD (x.dim, ps); w_v.setAll (1.0/ps)
    val (q, k, v) = att.queryKeyValue (x, w_q, w_k, w_v)

    banner ("Dimensions for input x, query q, key k, value v")
    println (s"x.dims = ${x.dims}")
    println (s"q.dims = ${q.dims}")
    println (s"k.dims = ${k.dims}")
    println (s"v.dims = ${v.dims}")

    banner ("Attention Matrix")
    val aw = att.attention (q, k, v)
    println (s"aw.dims = ${aw.dims}")
    println (s"aw      = $aw")

    banner ("Context Vectors Collected into Matrix")
    val cxt = new MatrixD (aw.dim, aw.dim2)
    println (s"cxt.dims = ${cxt.dims}")
    for i <- q.indices do cxt(i) = att.context (q(i), k, v)
    println (s"cxt      = $cxt")
    assert (cxt =~ aw)
    println ("succeed")

end attentionTest5

