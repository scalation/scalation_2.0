
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 19:48:13 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Fully Connected (Linear) Layer for Neural Networks
 */

package scalation
package modeling
package autograd

import scalation.mathstat.TensorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LayerNorm` class implements Layer Normalization as described in:
 *  "Layer Normalization" by Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
 *  @see https://arxiv.org/abs/1607.06450
 *  @param dModel the number of features in the input
 *  @param eps    a small value to avoid division by zero
 *  @param ops    the autograd operations
 */
class LayerNorm (dModel: Int, eps: Double = 1e-6)(using ops: AutogradOps)
      extends Module:

    val gamma = Variabl (ops.onesLike (new TensorD (1, 1, dModel)), name = Some ("gamma"))
    val beta  = Variabl (ops.zerosLike (new TensorD (1, 1, dModel)), name = Some ("beta"))
    
    override def parameters: IndexedSeq [Variabl] = IndexedSeq (gamma, beta)
    
    override def forward (input: Variabl): Variabl =
        val mean = input.meanAxis (axis = 2)
        val variance = input.varAxis (axis = 2)
        val normalized = (input - mean) / (variance + eps).sqrt
        gamma * normalized + beta
    end forward
    
end LayerNorm

