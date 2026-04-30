
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Mar  7 16:46:58 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Optimization: Stopping Rule for Iterative Optimizers
 */

package scalation
package optimization

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `StoppingRule` trait provides stopping rules for early termination
 *  in iterative optimization algorithms.
 *  @param upLimit  the number upward (loss increasing) steps allowed
 */
trait StoppingRule (upLimit: Int = 3):

    private var up        = 0                                             // consecutive steps up
    private var x_best    = VectorD.nullv                                 // best parameter vector
    private var loss_best = Double.MaxValue                               // best loss, so far
    private var loss0     = Double.MaxValue                               // previous loss

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Stop when too many steps have the loss function (e.g., sse) increasing.
     *  Signal a stopping condition by returning the best parameter vector, else null.
     *  @param loss  the current value of the loss function (e.g., sum of squared errors)
     *  @param x     the current value of the parameter vector
     */
    def stopWhen (loss: Double, x: VectorD): FuncVec =
        if loss > loss0 + EPSILON then up += 1                             // getting worse
        else                                                               // getting better
            up = 0
            if loss < loss_best then { loss_best = loss; x_best = x.copy }   // lower loss => save as best
        loss0 = loss                                                       // make current the previous

        if up > upLimit then
            x.set (x_best)                                                 // set new values for x
            (loss_best, x_best)                                            // at limit => return best x
        else
            (loss_best, null)                                              // null => continue search
    end stopWhen

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the best solution found.
     */
    def getBest: FuncVec = (loss_best, x_best)

end StoppingRule

