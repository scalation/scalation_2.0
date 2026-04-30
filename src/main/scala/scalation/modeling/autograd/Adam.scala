
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 20:01:00 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Adam Optimizer for Parameter Updates
 */

package scalation
package modeling
package autograd

import scalation.mathstat.TensorD

// FIX -- switch to using hyper-parameters

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Adam` class implements the Adam optimization algorithm for updating model parameters.
 *  The Adam optimizer (Kingma & Ba, 2015) with optional L2 weight decay maintains
 *  first (m) and second (v) moment estimates and applies bias correction.
 *  Classical (non-decoupled) weight decay is applied by adding weightDecay * param to the raw gradient.
 *  @note Call zeroGrad() before backward + step.
 *  @see https://arxiv.org/abs/1412.6980
 *  @param parameters   indexed sequence of Variables representing model parameters.
 *  @param lr           base Learning rate for updating the parameters.
 *  @param beta1        exponential decay rate for the first moment estimates.
 *  @param beta2        exponential decay rate for the second moment estimates.
 *  @param weightDecay  L2 regularization coefficient (0.0 to disable)
 *  @param eps          small constant added for numerical stability.
 */
case class Adam (parameters: IndexedSeq [Variabl], lr: Double = 0.001,
                 beta1: Double = 0.9, beta2: Double = 0.999,
                 weightDecay: Double = 0.0, eps: Double = 1e-8)
    extends Optimizer (parameters, lr):

    /** First moment estimates for each parameter, initialized to zeros with the same shape as the parameter data.
     */
    private val m = parameters.map (p => TensorD.zerosLike (p.data))

    /** Second moment estimates for each parameter, initialized to zeros with the same shape as the parameter data.
     */
    private val v = parameters.map (p => TensorD.zerosLike (p.data))

    /** Time step counter that tracks the number of updates made.
     */
    private var t = 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs a single optimization step using the Adam algorithm.
     *  The step method increments the time step counter, then for each parameter:
     *   - Updates the biased first moment estimate.
     *   - Updates the biased second moment estimate.
     *   - Computes bias-corrected moment estimates.
     *   - Updates the parameter data using the computed moments.
     */
    override def step (): Unit =
        t          += 1                                                // Increment time step
        val beta1_t = (1 - beta1~^t)
        val beta2_t = (1 - beta2~^t)

        for i <- parameters.indices do
            val p   = parameters(i)
            var m_i = m(i)
            var v_i = v(i)

            if p.grad != null then
                val grad = p.grad

                // Apply weight decay if specified (L2 regularization)
                val gradReg = if weightDecay > 0.0 then grad + p.data * weightDecay else grad

                m_i *= beta1                                           // Update biased first moment estimate
                m_i += gradReg * (1 - beta1)

                v_i *= beta2                                           // Update biased second moment estimate
                v_i += gradReg * gradReg * (1 - beta2)

                val biasCorr1 = m_i / beta1_t                          // Compute bias-corrected moment estimates
                val biasCorr2 = v_i / beta2_t

                p.data -= biasCorr1 * learningRate / (biasCorr2.map_ (math.sqrt) + eps)  // Update the parameter with the computed moments
        end for
    end step

end Adam

