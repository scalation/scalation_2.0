
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Wed Nov 12 10:32:44 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: StepLR Learning Rate Scheduler
 *
 *  A simple learning rate scheduler that decays the learning rate by a constant
 *  factor `gamma` every `stepSize` epochs.  Call `step()` once per epoch.
 *  `getLastLR` returns the most recently updated learning rate.
 */

package scalation
package modeling
package autograd

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Step-based learning rate scheduler.
 *  Reduces the optimizer's learning rate by multiplying with `gamma`
 *  every `stepSize` epochs.  Matches the behavior of PyTorch's StepLR
 *  for the single-LR (non–param-group) setting.
 *  @param optim     the optimizer whose learning rate will be scheduled
 *  @param stepSize  the interval (in epochs) between LR reductions
 *  @param gamma     the multiplicative decay factor applied every step
 */
final class StepLR (optim: Optimizer, stepSize: Int, gamma: Double)
      extends LRScheduler:
    
    private var epoch = 0
    private var lastLR: Double = optim.learningRate
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Advance the scheduler by one epoch.  When the epoch count is divisible
     * by `stepSize`, reduce the learning rate by multiplying with `gamma`.
     */
    override def step (): Unit =
        epoch += 1
        if epoch % stepSize == 0 then
            optim.learningRate *= gamma
            lastLR = optim.learningRate
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the most recently updated learning rate.
     */
    def getLastLR: Double = lastLR
    
end StepLR

