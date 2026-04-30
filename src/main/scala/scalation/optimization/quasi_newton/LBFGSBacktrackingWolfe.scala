
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Fri Oct 20 14:17:49 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Backtracking Wolfe Line Search Implementation used by the L-BFGS Algorithm.
 *
 *  This Scala implementation was made based on the C implementation of the same
 *  algorithm found in the ink below.
 *
 *  @see github.com/chokkan/liblbfgs
 */

package scalation
package optimization
package quasi_newton

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGSBacktrackingWolfe` object implements the backtracking Wolfe line
 *  search algorithm for use in the native implementation of L-BFGS.
 */
object LBFGSBacktrackingWolfe extends LBFGSLineSearch:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    override def lineSearch (n: Int, x: VectorD, f: Double, g: VectorD, s: VectorD,
                             stp: Double, cd: LBFGSCallbackData, params: LBFGSLineSearchPrms,
                             orthantWise: Option[OrthantWisePrms] = None): LBFGSLineSearchReturn =
        var count  = 0
        var width  = 0.0
        var dg     = 0.0
        var dginit = 0.0
        var dgtest = 0.0
        val dec    = 0.5
        val inc    = 2.1

        var xNew   = x
        var gNew   = g
        var fNew   = f
        var stpNew = stp

        // Check the input parameters for errors
        if stp <= 0.0 then
            return LBFGSLineSearchFailure (LBFGSReturnCode.InvalidPrms, LBFGSLineSearchIncomplete (xNew, fNew))

        // Compute the initial gradient in the search direction
        dginit = g dot s

        // Make sure that s points to a descent direction
        if 0 < dginit then
            return LBFGSLineSearchFailure (LBFGSReturnCode.IncreaseGradient, LBFGSLineSearchIncomplete (xNew, fNew))

        // The initial value of the objective function
        dgtest = params.ftol * dginit

        while true do
            xNew = x + (s * stpNew)

            // Evaluate the function and gradient values
            val evaluationResults = cd.evalLogic.evaluate (cd.instance, xNew, n, stpNew)
            fNew = evaluationResults.objFunctionValue
            gNew = evaluationResults.gradientVector

            count += 1

            if fNew > f + stpNew * dgtest then
                width = dec
            else                                         // check the Wolfe condition
                dg = gNew dot s
                if dg < params.wolfe * dginit then
                    width = inc
                else                                     // exit with the regular Wolfe condition
                    return LBFGSLineSearchStep (xNew, gNew, fNew, stpNew, count)
            end if

            if stpNew < params.minStep then
                // The step is the minimum value
                return LBFGSLineSearchFailure (LBFGSReturnCode.MinimumStep, LBFGSLineSearchIncomplete (xNew, fNew))
            if stpNew > params.maxStep then
                // The step is the maximum value
                return LBFGSLineSearchFailure (LBFGSReturnCode.MaximumStep, LBFGSLineSearchIncomplete (xNew, fNew))
            if params.maxLineSearch <= count then
                // Maximum number of iteration
                return LBFGSLineSearchFailure (LBFGSReturnCode.MaximumLineSearch, LBFGSLineSearchIncomplete (xNew, fNew))

            stpNew *= width
        end while

        LBFGSLineSearchFailure (LBFGSReturnCode.LogicError, LBFGSLineSearchIncomplete (xNew, fNew))

end LBFGSBacktrackingWolfe

