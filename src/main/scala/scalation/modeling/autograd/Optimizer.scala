
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri Apr 25 20:00:17 EDT 2025
 *  @see     LICENSE (MIT style license file)
 *
 *  @note    Autograd: Base Class for Gradient-Based Optimization
 *
 *  Defines an abstract optimizer for parameter updates in the autograd engine.
 *  Concrete optimizers (e.g., SGD, Adam) should extend this class and implement
 *  the `step()` method to apply parameter updates based on stored gradients.
 *
 *  Each parameter is represented by a `Variabl` node in the computation graph.
 *  The optimizer operates directly on `Variabl.grad`, expecting gradients to be
 *  accumulated via backpropagation before each call to `step()`.
 *
 *  Typical usage:
 *      1.  Call `zeroGrad()` before a forward/backward pass.
 *      2.  Perform forward + backward computation.
 *      3.  Call `step()` to update parameters.
 */

package scalation
package modeling
package autograd

import autograd.Variabl

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Optimizer` abstract class optimizes model parameters.
 *  Notes:
 *      - Subclasses implement the specific update rule in `step()`.
 *      - The optimizer assumes that gradients (`p.grad`) have been computed and
 *        accumulated by the autograd engine before each call to `step()`.
 *      - Parameters with `null` gradients are safely ignored.
 *  @param parameters    the trainable parameters, each wrapped in a `Variabl`
 *  @param learningRate  the step size (η) used for gradient-based updates
 */
abstract class Optimizer (parameters: IndexedSeq [Variabl], var learningRate: Double):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Executes a single optimization step by updating each parameter based on its gradient.
     */
    def step (): Unit
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset gradients for all parameters.
     *  Typically called before the next forward/backward pass.
     *  Only parameters with non-null gradient buffers are updated.
     */
    def zeroGrad (): Unit = 
        parameters.foreach { p => if p.grad != null then p.grad.set (0.0) }
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the global L2 norm of all parameter gradients.
     *  Math:
     *       g = √(∑_p‖grad_p‖² )
     */
    def gradNorm: Double =
        math.sqrt (parameters.map (p => if p.grad != null then p.grad.normFSq else 0.0).sum)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clip the gradients of all parameters by global norm.
     *  Scales gradients so that the total norm ≤ maxNorm.
     *  Math:
     *       Let g = √(∑_p ‖grad_p‖² ).
     *       If g > maxNorm, scale all gradients by (maxNorm / g).
     */
    def clipGradNorm (maxNorm: Double): Unit =
        val totalNorm = math.sqrt (parameters.map (p => if p.grad != null then p.grad.normFSq else 0.0).sum)
        if totalNorm > maxNorm then
            val scale = maxNorm / (totalNorm + 1e-6)
            parameters.foreach { p => if p.grad != null then p.grad *= scale }
    end clipGradNorm

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clip the gradients of all parameters by value (element-wise).
     *  Each gradient entry smaller than `minVal` is set to `minVal`,
     *  and each entry larger than `maxVal` is set to `maxVal`.
     */
    def clipGradValue (minVal: Double, maxVal: Double): Unit =
        parameters.foreach { p => if p.grad != null then p.grad = p.grad.clipByValue (minVal, maxVal) }
    
end Optimizer

