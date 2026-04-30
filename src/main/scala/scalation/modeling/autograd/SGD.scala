
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 20:01:00 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: SGD Optimizer for Parameter Updates
 */

package scalation
package modeling
package autograd

import scalation.mathstat.TensorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Implements the Stochastic Gradient Descent (SGD) optimization algorithm.
 *  @param parameters  an indexed sequence of model parameters to be optimized.
 *  @param lr          the learning rate used for updating the parameters.
 *  @param momentum    momentum factor to accelerate convergence (default is 0.0).
 */
case class SGD (parameters: IndexedSeq [Variabl], lr: Double, momentum: Double = 0.0)
     extends Optimizer (parameters, lr):

    /** Velocity for each parameter initialized to zeros matching the shape of parameter data.
     */
    private val velocity = parameters.map (p => TensorD.zerosLike (p.data))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs a single optimization step using the SGD algorithm.
     *  For each parameter:
     *   - Updates the velocity using the momentum factor and the current gradient.
     *   - Updates the parameter data by subtracting the computed velocity.
     */
    override def step (): Unit =
        for i <- parameters.indices do
            val p = parameters(i)
            var v = velocity(i)

            if p.grad != null then
                v *= momentum                         // Update velocity using momentum
                v += p.grad * lr                      // Add the gradient times the learning rate
                p.data -= v                           // Update parameter by subtracting the velocity
        end for
    end step

end SGD

