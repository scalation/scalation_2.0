
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri November 21 19:48:13 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Multi-Head Attention Module for Transformer Models
 */

package scalation
package modeling
package autograd

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Implements the Multi-Head Attention mechanism, a key component of transformer models.
 *  This class performs linear projections of the input tensors, splits them into multiple
 *  attention heads, applies scaled dot-product attention to each head, and combines the
 *  results into a single output tensor.
 *  @see   https://arxiv.org/abs/1706.03762
 *         "Attention Is All You Need" by Vaswani et al., 2017.
 *  @see   https://dev-discuss.pytorch.org/t/understanding-multi-head-attention-for-ml-framework-developers/1792
 *         "Understanding Multi-Head Attention for ML Framework Developers"
 *  @see   https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
 *         PyTorch MultiheadAttention Documentation
 *  @param numHeads  the number of attention heads
 *  @param dModel    the dimensionality of the model (input and output feature size)
 */
class MultiHeadAttention (numHeads: Int, dModel: Int)
      extends SeqModule (IndexedSeq.empty):
    
    private val headDim: Int = dModel / numHeads
    
    // Linear projections
    private val W_q = Linear (dModel, dModel)
    private val W_k = Linear (dModel, dModel)
    private val W_v = Linear (dModel, dModel)
    private val W_o = Linear (dModel, dModel)
    
    private val sdpa = ScaledDotProductAttention()
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass for the Multi-Head Attention module.
     *  This method takes three input tensors (query, key, value), performs linear projections,
     *  splits them into multiple heads, applies scaled dot-product attention to each head,
     *  and combines the results into a single output tensor.
     *  @param inputs  an `IndexedSeq` containing the query (q), key (k), and value (v) tensors
     *  @return an `IndexedSeq` containing the resulting output tensor
     *  @throws IllegalArgumentException if the number of inputs is not 3
     */
    override def forward (inputs: IndexedSeq [Variabl]): IndexedSeq [Variabl] =
        
        require (inputs.length == 3, s"Expected 3 inputs (q, k, v), got ${inputs.length}")

        val (q, k, v) = (inputs(0), inputs(1), inputs(2))
        
        // Linear projections
        val B = q.shape.head                // batch size
        val T = q.shape(1)                  // sequence length
        
        // Reshape for linear layers
        // TODO: Move the reshape logic inside Linear layer
        val qProj = W_q(q.reshape (Seq (B * T, dModel, 1))).reshape (Seq (B, T, dModel))
        val kProj = W_k(k.reshape (Seq (B * T, dModel, 1))).reshape (Seq (B, T, dModel))
        val vProj = W_v(v.reshape (Seq (B * T, dModel, 1))).reshape (Seq (B, T, dModel))
        
        // Split into multiple heads
        val qHeads = splitHeads (qProj)
        val kHeads = splitHeads (kProj)
        val vHeads = splitHeads (vProj)
        
        // Apply Scaled Dot-Product Attention for each head
        val attHeads = (0 until numHeads).map { h =>
            val att = sdpa.forward (IndexedSeq (qHeads(h), kHeads(h), vHeads(h)))
            att.head
        }
        
        // Combine heads
        val combined = combineHeads (attHeads)
        
        // Final linear projection (reshape for linear layer)
        val output = W_o(combined.reshape (Seq (B * T, dModel, 1))).reshape (Seq (B, T, dModel))
        
        IndexedSeq (output)
    end forward
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Splits the input tensor into multiple attention heads.
     *  This method reshapes the input tensor to produce `numHeads` tensors, each
     *  corresponding to a single attention head.
     *  @param x  the input tensor to split
     *  @return an `IndexedSeq` containing the split tensors for each head
     *  @throws IllegalArgumentException if the input tensor's last dimension is not
     *                                   divisible by the number of heads
     */
    private def splitHeads (x: Variabl): IndexedSeq [Variabl] =
        val shape     = x.shape
//      val batchSize = shape.head      // B
//      val seqLen    = shape(1)        // T
        val dim       = shape(2)        // D
        
        require (dim == dModel,
            s"splitHeads: expected last dim $dModel, got $dim")
        
        require (dModel % numHeads == 0,
            s"splitHeads: dModel = $dModel must be divisible by numHeads = $numHeads")
        
        // Each head dimension
        val hDim = headDim
        
        // We produce H heads, each (B, T, headDim)
        (0 until numHeads).map { h =>
            val start = h * hDim
            val end = start + hDim
            
            // slice last dimension (axis = 2)
            x(?, ?, start until end)
        }
    end splitHeads
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Combines the tensors from multiple attention heads into a single tensor.
     *  This method concatenates the tensors along the last dimension to produce
     *  the final output tensor.
     *  @param heads an `IndexedSeq` containing the tensors for each head
     *  @return a `Variabl` containing the combined tensor
     *  @throws IllegalArgumentException if the number of heads does not match `numHeads`
     */
    private def combineHeads (heads: IndexedSeq [Variabl]): Variabl =
        
        require (heads.length == numHeads,
            s"combineHeads: expected $numHeads heads, got ${heads.length}")
        
        // Concatenate along the last dimension (axis = 2)
        val combined = concat (heads, axis = 2)
        combined
    end combineHeads

end MultiHeadAttention


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Implements the Scaled Dot-Product Attention mechanism.
 *  This class is a sequence module that computes the attention scores and
 *  applies them to the value tensor (v) based on the query (q) and key (k) tensors.
 *  It is a fundamental building block for transformer models.
 *  @see     https://arxiv.org/abs/1706.03762
 *           "Attention Is All You Need" by Vaswani et al., 2017.
 *  @see      https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
 */
class ScaledDotProductAttention
      extends SeqModule (IndexedSeq.empty):
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass for the Scaled Dot-Product Attention module.
     *  This method takes three input tensors (q, k, v), computes the attention
     *  scores, and applies them to the value tensor.
     *  @param inputs  an `IndexedSeq` containing the query (q), key (k), and value (v) tensors
     *  @return        an `IndexedSeq` containing the resulting attention tensor
     *  @throws IllegalArgumentException if the number of inputs is not 3
     */
    override def forward (inputs: IndexedSeq [Variabl]): IndexedSeq [Variabl] =
        require(inputs.length == 3, s"Expected 3 inputs (q, k, v), got ${inputs.length}")
        val q   = inputs(0)
        val k   = inputs(1)
        val v   = inputs(2)
        
        val att = attention (q, k, v)
        IndexedSeq (att)
    end forward
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the attention tensor based on the query, key, and value tensors.
     *  This method calculates the scaled dot-product attention by performing
     *  the following steps:
     *  1. Compute the dot product of the query and the transposed key tensors.
     *  2. Scale the result by the square root of the key dimension.
     *  3. Apply the softmax function to obtain the attention scores.
     *  4. Multiply the attention scores with the value tensor to get the final attention.
     *  @param q  the query tensor
     *  @param k  the key tensor
     *  @param v  the value tensor
     *  @return   the resulting attention tensor
     */
    private def attention (q: Variabl, k: Variabl, v: Variabl): Variabl =
        val d_k = q.shape.last
        val scaleFactor = 1.0 / math.sqrt(d_k)
        
        val kT  = k.transpose(1, 2)
        val qkt = q.bmm(kT)                         // repeated dot product
        
        val sdp = qkt * scaleFactor                 // scaled dot product (sdp)
        val scr = sdp.softmax                       // attention scores
        val att = scr.bmm(v)                        // attention (Q, K, V)
        att
    end attention

end ScaledDotProductAttention

