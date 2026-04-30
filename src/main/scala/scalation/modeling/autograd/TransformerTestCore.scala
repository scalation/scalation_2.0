
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Wed Nov 26 10:44:32 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Unit Tests for Core Transformer Components
 */

package scalation
package modeling
package autograd

import scalation.mathstat.TensorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The TransformerTestCore` tests the `Transformer` class.
 */
object TransformerTestCore:
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test ScaledDotProductAttention forward + backward.
     *  > runMain scalation.modeling.autograd.TransformerTestCore.sdpaTest
     */
    @main def sdpaTest (): Unit =

        banner ("Scaled Dot Product Attention Test")
        
        val R = TestReport()
        val sdpa = ScaledDotProductAttention()
        
        // ---------------------------------------------------------
        // Create a small dummy batch
        // Shapes: (B, T, D) = (2, 3, 4)
        // ---------------------------------------------------------
        val B = 2
        val T = 3
        val D = 4
        
        val q = Variabl (
            TensorD ((B, T, D),
                // batch 0
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                // batch 1
                1.1, 1.0, 0.9, 0.8,
                0.7, 0.6, 0.5, 0.4,
                0.3, 0.2, 0.1, 0.0
            ),
            name = Some("q")
        )
        
        val k = Variabl (
            TensorD ((B, T, D),
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.1, 1.0, 0.9, 0.8,
                0.7, 0.6, 0.5, 0.4,
                0.3, 0.2, 0.1, 0.0
            ),
            name = Some("k")
        )
        
        val v = Variabl (
            TensorD ((B, T, D),
                0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.1, 1.2,
                1.1, 1.0, 0.9, 0.8,
                0.7, 0.6, 0.5, 0.4,
                0.3, 0.2, 0.1, 0.0
            ),
            name = Some("v")
        )
        
        println(s"q shape = ${q.shape}, values = ${q.data}")
        println(s"k shape = ${k.shape}")
        println(s"v shape = ${v.shape}")
        
        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        val out = sdpa (IndexedSeq(q, k, v)).head
        println(s"SDPA output shape = ${out.shape}")
        println(s"SDPA output values = ${out.data}")
        
        // ---------------------------------------------------------
        // Gradient checks
        // ---------------------------------------------------------
        R.record("SDPA - gradCheck(q, k, v)") {
            GradCheck.gradCheckAll(
                Seq(q, k, v),
                () => sdpa(IndexedSeq(q, k, v)).head.sum,
                quiet = false
            )
        }
        
        R.summary("Scaled Dot Product Attention Test")
    
    end sdpaTest
    
    
    @main def mhaTest (): Unit =
        
        banner ("Multi-Head Attention Test")
        
    end mhaTest
    
    
    @main def layerNormTest (): Unit =
        
        banner ("Layer Normalization Test")
        
    end layerNormTest
    
end TransformerTestCore

