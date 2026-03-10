
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri Nov 7 09:15:25 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Learning Rate Scheduler Interface
 *
 *  Defines the base trait for learning rate schedulers used in gradient-based
 *  optimization. Concrete schedulers should override one or both versions of
 *  `step()` and implement their own logic for updating the learning rate.
 */

package scalation
package modeling
package autograd

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Learning Rate Scheduler (LR Scheduler) trait.
 *  Defines a generic interface for schedulers that adjust the learning rate
 *  during optimization. Concrete implementations may update the learning rate
 *  based on iteration count, loss values, or other criteria.
 *  Notes:
 *      - The parameterless `step()` is intended for schedulers that adjust
 *        learning rate solely based on iteration count.
 *      - The `step(currentLoss)` method is intended for schedulers that adapt
 *        learning rate based on the current loss value.
 *      - By default, both methods throw `UnsupportedOperationException`;
 *        subclasses must override the method(s) they support.
 */
trait LRScheduler:
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Advance the scheduler one step (iteration-based).
     *  Default implementation throws an exception; override if supported.
     */
    def step (): Unit =
        throw new UnsupportedOperationException ("This scheduler does not support step without loss input.")
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Advance the scheduler using the current loss (loss-based scheduling).
     *  Default implementation throws an exception; override if supported.
     *  @param currentLoss  the current loss value used for scheduling
     */
    def step (currentLoss: Double): Unit =
        throw new UnsupportedOperationException ("This scheduler does not support step with loss input.")
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the most recently computed learning rate.
     */
    def getLastLR: Double
    
end LRScheduler

