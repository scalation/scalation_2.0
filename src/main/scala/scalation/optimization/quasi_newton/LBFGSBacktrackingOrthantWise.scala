
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Mon Oct 30 17:37:09 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Backtracking Orthant-Wise line Search Implementation used by the L-BFGS Algorithm.
 *
 *  This Scala implementation was made based on the C implementation of the same
 *  algorithm found in the link below.
 *
 *  @see github.com/chokkan/liblbfgs
 */

package scalation
package optimization
package quasi_newton

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGSBacktrackingOrthantWise` object implements the backtracking
 *  Orthant-Wise line search algorithm for use in the native implementation of
 *  L-BFGS.
 */
object LBFGSBacktrackingOrthantWise extends LBFGSLineSearch:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    override def lineSearch (n: Int, x: VectorD, f: Double, g: VectorD, s: VectorD,
                             stp: Double, cd: LBFGSCallbackData, params: LBFGSLineSearchPrms,
                             orthantWise: Option[OrthantWisePrms]): LBFGSLineSearchReturn =
        var count  = 0
        val width  = 0.5
        var norm   = 0.0
        var dgtest = 0.0
        val wp     = new VectorD (n)

        var xNew   = x
        var gNew   = g
        var fNew   = f
        var stpNew = stp

        // Input parameters should have an orthantWise
        if orthantWise.isEmpty then
            return LBFGSLineSearchFailure (LBFGSReturnCode.InvalidPrms, LBFGSLineSearchIncomplete (xNew, fNew))

        val orthantwisePrms = orthantWise.get
        
        // Check the input parameters for errors
        if stp <= 0.0 then
            return LBFGSLineSearchFailure (LBFGSReturnCode.InvalidPrms, LBFGSLineSearchIncomplete (xNew, fNew))

        // Choose the orthant for the new point
        for i <- 0 until n do
            wp(i) = if x(i) == 0.0 then -g(i) else x(i)

        while true do
            xNew = x + (s * stpNew)                             // update the current point

            // The current point is projected onto the orthant
            xNew = orthantwisePrms.project (xNew, wp)

            // Evaluate the function and gradient values
            val evaluationResults = cd.evalLogic.evaluate (cd.instance, xNew, n, stpNew)
            fNew = evaluationResults.objFunctionValue
            gNew = evaluationResults.gradientVector

            // Compute the L1 norm of the variables and add it to the object value
            norm  = orthantwisePrms.x1Norm (xNew)
            fNew += norm * orthantwisePrms.c

            count += 1

            dgtest = 0.0
            for i <- 0 until n do dgtest += (xNew(i) - x(i)) * g(i)

            if fNew <= f + params.ftol * dgtest then
                // The sufficient decrease condition
                return LBFGSLineSearchStep (xNew, gNew, fNew, stpNew, count)

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

end LBFGSBacktrackingOrthantWise

