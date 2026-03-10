
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Sat Nov  8 10:24:12 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: ReduceLROnPlateau Learning Rate Scheduler
 */

package scalation
package modeling
package autograd

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** PyTorch-style ReduceLROnPlateau scheduler.
 *  Monitors a metric each epoch and reduces the learning rate when progress
 *  plateaus.  Supports both "min" (e.g., loss) and "max" (e.g., accuracy) modes,
 *  with relative or absolute thresholds for determining improvement.
 *  The LR is reduced when the number of non-improving epochs exceeds `patience`,
 *  after which a cooldown period prevents further reductions.  Each reduction
 *  follows: newLR = max(oldLR * factor, minLR), skipped when the change is
 *  too small (≤ eps).  Non-finite metric values are ignored.
 *  Call `step(metric)` after each optimizer update.  `getLastLR` returns the
 *  most recent learning rate.
 *  @param optim          the optimizer whose learning rate will be scheduled
 *  @param mode           "min" or "max" (target direction for improvement)
 *  @param factor         multiplicative decay factor in (0,1), i.e., newLR = oldLR * factor
 *  @param patience       number of non-improving epochs tolerated before reduction (strictly `> patience`)
 *  @param threshold      significance threshold (relative or absolute depending on `thresholdMode`)
 *  @param thresholdMode  "rel" for relative margin, "abs" for absolute margin
 *  @param cooldown       epochs to wait after a reduction during which bad-epoch counter stays at 0
 *  @param minLR          lower bound on the learning rate
 *  @param eps            minimal effective LR change required to apply a reduction
 *  @param verbose        if true, prints LR reduction messages
 */
final class ReduceLROnPlateau (optim: Optimizer,
                               mode: String = "min",            // "min" or "max"
                               factor: Double = 0.1,            // newLR = oldLR * factor
                               patience: Int = 10,              // epochs with no significant improvement before reduce
                               threshold: Double = 1e-4,        // significance threshold
                               thresholdMode: String = "rel",   // "rel" or "abs"
                               cooldown: Int = 0,               // epochs to wait after a reduction
                               minLR: Double = 0.0,             // LR floor
                               eps: Double = 1e-8,              // minimal effective LR change
                               verbose: Boolean = false)
      extends LRScheduler:
    
    require (factor > 0.0 && factor < 1.0, "factor must be in (0, 1)")
    require (patience >= 0, "patience must be >= 0")
    require (cooldown >= 0, "cooldown must be >= 0")
    require (threshold >= 0.0, "threshold must be >= 0")
    require (eps >= 0.0, "eps must be >= 0")
    require (mode.equalsIgnoreCase ("min") || mode.equalsIgnoreCase ("max"), "mode must be 'min' or 'max'")
    require (thresholdMode.equalsIgnoreCase ("rel") || thresholdMode.equalsIgnoreCase ("abs"), "thresholdMode must be 'rel' or 'abs'")
    
    private val _modeMin = mode.toLowerCase == "min"
    private val _thrRel  = thresholdMode.toLowerCase == "rel"
    
    private var best: Double =
        if _modeMin then Double.PositiveInfinity else Double.NegativeInfinity
    
    private var numBadEpochs: Int = 0
    private var cooldownCounter: Int = 0
//  private var _lastLR: Double = optim.learningRate
//  private var lastEpoch: Int = -1
    
    private inline def isFinite (x: Double): Boolean = ! x.isNaN && ! x.isInfinity
    private inline def inCooldown: Boolean = cooldownCounter > 0
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the current metric is significantly better than the
     *  stored `best` metric.  Uses both mode ("min"/"max") and threshold mode
     *  ("rel"/"abs") to define improvement.
     *  @param current   the new metric value
     *  @param bestSoFar the best metric recorded so far
     */
    private def isBetter (current: Double, bestSoFar: Double): Boolean =
        if _thrRel then
            if _modeMin then current < bestSoFar * (1.0 - threshold)
            else              current > bestSoFar * (1.0 + threshold)
        else // "abs"
            if _modeMin then current < bestSoFar - threshold
            else              current > bestSoFar + threshold
    end isBetter
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply a learning rate reduction:
     *  newLR = max(oldLR * factor, minLR)
     *  Skips the update if `oldLR - newLR ≤ eps` (no effective change).
     *  Resets bad-epoch counter and enters cooldown.
     */
    private def reduceLR (): Unit =
        val oldLR = optim.learningRate
        val newLR = math.max (oldLR * factor, minLR)
        // eps guard (skip if change too tiny)
        if oldLR - newLR > eps then
            if verbose then
                println (f"[LR Scheduler] ReduceLROnPlateau: reducing LR from $oldLR%.6f to $newLR%.6f")
            optim.learningRate = newLR
//          _lastLR = newLR
            cooldownCounter = cooldown
            numBadEpochs = 0
        end if
    end reduceLR
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Maintain cooldown semantics by decrementing the cooldown counter and
     *  keeping the bad-epoch counter at 0 during cooldown epochs.
     */
    private def resetDuringCooldown (): Unit =
        if cooldownCounter > 0 then cooldownCounter -= 1
        numBadEpochs = 0
    end resetDuringCooldown
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the scheduler using the monitored metric (e.g., validation loss).
     *  Must be called **after** each optimizer update.
     *  Steps:
     *       1. Ignore non-finite metric values.
     *       2. Test for significant improvement versus `best`.
     *       3. Update counters (bad-epochs or reset during cooldown).
     *       4. Trigger LR reduction when plateau persists (`numBadEpochs > patience`).
     *  @param currentMetric the metric value for this epoch
     */
    override def step (currentMetric: Double): Unit =
//      lastEpoch += 1
        if ! isFinite (currentMetric) then return
        
        val better = isBetter (currentMetric, best)
        if better then
            best = currentMetric         // update best only on significant improvement
            numBadEpochs = 0
        else
            if inCooldown then
                resetDuringCooldown ()   // PyTorch keeps bad-epoch counter at 0 during cooldown
            else
                numBadEpochs += 1
        end if
        
        // Reduce if plateau persisted strictly longer than patience
        if numBadEpochs > patience && !inCooldown then reduceLR()
    end step
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the most recently updated learning rate.
     */
    override def getLastLR: Double = optim.learningRate

end ReduceLROnPlateau

