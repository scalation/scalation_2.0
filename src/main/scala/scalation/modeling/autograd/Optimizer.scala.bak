
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 20:00:17 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Base Class for Gradient-Based Optimization
 */

package scalation
package modeling
package autograd

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Abstract optimizer for updating model parameters.
 *  @param parameters  an indexed sequence of Variables representing the parameters to be optimized.
 */
abstract class Optimizer (parameters: IndexedSeq [Variabl]):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Executes a single optimization step by updating each parameter based on its gradient.
     */
    def step (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Resets the gradient of each parameter to zero.
     *  This is typically called before computing the next set of gradients.
     */
    def zeroGrad (): Unit =
        parameters.foreach { p => if p.grad != null then p.grad.set (0.0) }

end Optimizer

