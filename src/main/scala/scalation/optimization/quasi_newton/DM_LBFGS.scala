
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Mon Jan 8 10:23:10 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Direction Momentum, Limited Memory Broyden–Fletcher–Goldfarb–Shanno (dmL-BFGS) Algorithm
 */

package scalation
package optimization
package quasi_newton

import scala.math.abs

import scalation.mathstat.{PlotC, VectorD}
import scalation.optimization.functions.*

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DM_LBFGS` object implementation of the direction momentum Limited memory
 *  Broyden–Fletcher–Goldfarb–Shanno (BFGS) for unconstrained optimization
 *  (dmL-BFGS) algorithm.
 */
object DM_LBFGS extends PathMonitor:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs the dmL-BFGS optimization that optimizes variables to minimize
     *  a function value using momentum on the variables and the line search direction.
     */
    def dmlbfgsMain (n: Int, x: VectorD,
                     functionLogic: EvaluationLogic | OptimizationLogic,
                     params: LBFGSPrms = LBFGSPrms (), momentum: Double,
                     instance: Any = None): LBFGSResults =
        clearPath ()

        checkdmLBFGSArgs4Errors (n, params, momentum) match
            case Some(errorReturnCode) => return LBFGSResults (errorReturnCode, x, None, None)
            case _ =>

        var xNew: VectorD = x
        add2Path (x)

        var lineSearchResults: LBFGSLineSearchReturn = null
        val useOrthantWiseLogic = params.orthantWise.nonEmpty

        var k = 1
        var end = 0
        var j, ls, bound = 0
        var step = 0.0

        // Constant parameters and their default values
        val m = params.m

        try
            var xp, g, gp, d, dp = new VectorD(n)
            var pg, pf, s, y = VectorD.nullv
            val lm = Array.ofDim [LBFGSIterationData] (m)

            var ys, yy, xnorm, gnorm, beta, fx, rate = 0.0

            val lineSearchImpl: LBFGSLineSearch = determineLineSearchImple (params.lineSearch)

            // Construct a callback data
            val cd = LBFGSCallbackData(n, instance, functionLogic)

            // Allocate working space
            if useOrthantWiseLogic then pg = new VectorD(n)                // allocate working space for OW-LQN
            if 0 < params.past then pf = new VectorD(params.past)          // allocate space for storing previous obj func values

            // Evaluate the function value and its gradient
            val evaluationResults = cd.evalLogic.evaluate (cd.instance, x, cd.n, 0)
            fx = evaluationResults.objFunctionValue
            g  = evaluationResults.gradientVector

            if useOrthantWiseLogic then
                // Compute the L1 norm of the variable and add it to the object value
                val orthantWisePrms = params.orthantWise.get
                xnorm = orthantWisePrms.x1Norm (x)
                fx   += xnorm * orthantWisePrms.c
                pg    = orthantWisePrms.pseudoGradient (x, g)
            end if

            // Store the initial value of the objective function
            if pf != VectorD.nullv then pf(0) = fx

            // Compute the direction: assume initial Hessian matrix H_0 as the identity matrix
            d = if ! useOrthantWiseLogic then -g else -pg

            // Make sure that the initial variables are not a minimizer
            xnorm = x.norm
            gnorm = if ! useOrthantWiseLogic then g.norm else pg.norm
            if xnorm < 1.0 then xnorm = 1.0
            if gnorm / xnorm <= params.epsilon then
                return LBFGSResults (LBFGSReturnCode.AlreadyMinimized, xNew, Some(fx), None)

            // Compute the initial step: step = 1.0 / sqrt(vecdot(d, d, n))
            step = 1.0 / d.norm

            while true do
                xp = xNew                // store the current position, gradient and direction vectors
                gp = g
                dp = d

                // Search for an optimal step
                lineSearchResults = if ! useOrthantWiseLogic then
                    lineSearchImpl.lineSearch (n, xNew, fx, g, d, step, cd, params.lineSearchPrms, params.orthantWise)
                else
                    lineSearchImpl.lineSearch (n, xNew, fx, pg, d, step, cd, params.lineSearchPrms, params.orthantWise)

                lineSearchResults match
                    case lineSearchStep: LBFGSLineSearchStep =>
                        xNew = xp * momentum + lineSearchStep.x * (1 - momentum)
                        g    = lineSearchStep.g
                        fx   = lineSearchStep.fx
                        step = lineSearchStep.step
                        ls   = lineSearchStep.numberOfIterations
                        add2Path (xNew)
                    case lineSearchFailure: LBFGSLineSearchFailure =>
                        val failureReturnCode =
                            if lineSearchFailure.returnCode.isErrorCode then lineSearchFailure.returnCode
                            else LBFGSReturnCode.UnknownError
                        // Return with the value of the previous point
                        return LBFGSResults (failureReturnCode, xp, Some(fx), Some(lineSearchFailure.bestIncompleteResults))

                if useOrthantWiseLogic then
                    val orthantWisePrms = params.orthantWise.get
                    pg = orthantWisePrms.pseudoGradient (xNew, g)

                // Compute x and g norms
                xnorm = xNew.norm
                gnorm = if ! useOrthantWiseLogic then g.norm else pg.norm

                // Report the progress
                functionLogic match
                    case o: OptimizationLogic =>
                        val ret = o.progress (cd.instance, xNew, g, fx, xnorm, gnorm, step, cd.n, k, ls)
                        if ret != LBFGSReturnCode.Success then return LBFGSResults (ret, xNew, Some(fx), None)
                    case _ =>

                // Convergence test: The criterion is given by the following formula: |g(x)| / \max(1, |x|) < \epsilon
                if xnorm < 1.0 then xnorm = 1.0
                if gnorm / xnorm <= params.epsilon then
                    return LBFGSResults (LBFGSReturnCode.Success, xNew, Some(fx), None)

                // Test for stopping criterion: The criterion is given by the following formula:
                //   |(f(past_x) - f(x))| / f(x) < \delta
                if pf != VectorD.nullv then
                    if params.past <= k then                                   // don't test stopping criterion while k < past
                        rate = (pf(k % params.past) - fx) / fx                 // compute relative improvement from the past
                        if abs (rate) < params.delta then                      // the stopping criterion
                            return LBFGSResults(LBFGSReturnCode.Stop, xNew, Some(fx), None)
                    end if
                    pf(k % params.past) = fx                                   // store current value of the objective function
                end if

                if params.maxIterations != 0 && params.maxIterations < k + 1 then
                    return LBFGSResults (LBFGSReturnCode.MaximumIteration, xNew, Some(fx), None)

                // Update vectors s and y:
                //     s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
                //     y_{k+1} = g_{k+1} - g_{k}.
                s = xNew - xp
                y = g - gp

                // Compute scalars ys and yy:
                //     ys = y^t \cdot s = 1 / \rho.
                //     yy = y^t \cdot y.
                // Notice that yy is used for scaling the Hessian matrix H_0 (Cholesky factor)
                ys = y dot s
                yy = y dot y

                lm(end) = LBFGSIterationData (s, y, ys, 0)

                // Recursive formula to compute dir = -(H \cdot g).  This is described in page 779 of: Jorge Nocedal.
                // Updating Quasi-Newton Matrices with Limited Storage.
                // Mathematics of Computation, Vol. 35, No. 151, pp. 773--782, 1980.
                bound = if m <= k then m else k
                k  += 1
                end = (end + 1) % m

                // Compute the steepest direction, i.e., the negative of gradients
                d = if ! useOrthantWiseLogic then -g else -pg

                j = end
                cfor (0, bound) { _ =>
                    j = (j + m - 1) % m                              // if (--j == -1) j = m-1
                    val it = lm(j)
                    it.alpha = it.s dot d                            // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
                    it.alpha = it.alpha / it.ys
                    d += (it.y * (-it.alpha))                        // q_{i} = q_{i+1} - \alpha_{i} y_{i}
                } // cfor

                d *= (ys / yy)

                cfor (0, bound) { _ =>
                    val it = lm(j)
                    beta = it.y dot d                                // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
                    beta /= it.ys
                    d += it.s * (it.alpha - beta)                    // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
                    j = (j + 1) % m                                  // if (++j == m) j = 0
                } // cfor

                // Constrain the search direction for orthant-wise updates
                if useOrthantWiseLogic then
                    val orthantWisePrms = params.orthantWise.get
                    for i <- orthantWisePrms.start until orthantWisePrms.end.getOrElse (d.length) do
                        if d(i) * pg(i) >= 0 then d(i) = 0
                end if

                d = dp * momentum + d * (1 - momentum)

                // Now the search direction d is ready. We try the default step first
                step = params.lineSearchPrms.defaultStep           // previously default step was hardcoded to 1.
            end while

            LBFGSResults (LBFGSReturnCode.UnknownError, xNew, Some(fx), None)
        catch
            case _ : OutOfMemoryError => LBFGSResults (LBFGSReturnCode.OutOfMemory, x, None, None)

    end dmlbfgsMain

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Checks if the dmL-BFGS optimization arguments have an error and returns
     *  an `Option` with the `LBFGSReturnCode` that represents the first error found.
     *
     *  @param n         the number of variables.
     *  @param params    `BFGSPrms` class holding the parameters chosen to control the L-BFGS
     *  @param momentum  the momentum to be applied in the variable and line search direction vectors.
     *  @return          Option [LBFGSReturnCode], the value with a `BFGSReturnCode` return code that
     *                   represents the first error found in the `params` or `momentum` arguments. If
     *                   there are no errors in either, `None` is returned
     */
    private def checkdmLBFGSArgs4Errors (n: Int, prms: LBFGSPrms,
                                         momentum: Double): Option [LBFGSReturnCode] =
        if n <= 0 then             return Some(LBFGSReturnCode.InvalidN)
        if prms.epsilon < 0.0 then return Some(LBFGSReturnCode.InvalidEpsilon)
        if prms.past < 0 then      return Some(LBFGSReturnCode.InvalidTestPeriod)
        if prms.delta < 0.0 then   return Some(LBFGSReturnCode.InvalidDelta)
        if prms.lineSearchPrms.minStep < 0.0 then return Some(LBFGSReturnCode.InvalidMinStep)
        if prms.lineSearchPrms.maxStep < prms.lineSearchPrms.minStep then return Some(LBFGSReturnCode.InvalidMaxStep)
        if prms.lineSearchPrms.ftol < 0.0 then return Some(LBFGSReturnCode.InvalidFTOL)
        if prms.lineSearch == LBFGSLineSearchAlg.BacktrackingWolfe ||
            prms.lineSearch == LBFGSLineSearchAlg.BacktrackingStrongWolfe then
            if prms.lineSearchPrms.wolfe <= prms.lineSearchPrms.ftol || 1.0 <= prms.lineSearchPrms.wolfe then
                return Some(LBFGSReturnCode.InvalidWolfe)
        if prms.lineSearchPrms.gtol < 0.0 then return Some(LBFGSReturnCode.InvalidGTOL)
        if prms.lineSearchPrms.xtol < 0.0 then return Some(LBFGSReturnCode.InvalidXTOL)
        if prms.lineSearchPrms.maxLineSearch <= 0 then return Some(LBFGSReturnCode.InvalidMaxLineSearch)
        if prms.orthantWise.nonEmpty then
            val orthantWisePrms = prms.orthantWise.get
            if orthantWisePrms.c <= 0.0 then return Some(LBFGSReturnCode.InvalidOrthantwise)
            if orthantWisePrms.start < 0 || n <= orthantWisePrms.start then
                return Some(LBFGSReturnCode.InvalidOrthantwiseStart)
            if orthantWisePrms.end.isDefined then
                val orthantWisePrmsEnd = orthantWisePrms.end.get
                if orthantWisePrmsEnd <= orthantWisePrms.start || n < orthantWisePrmsEnd then
                    return Some(LBFGSReturnCode.InvalidOrthantwiseEnd)
            if prms.lineSearch != LBFGSLineSearchAlg.BacktrackingOrthantWise then
                return Some(LBFGSReturnCode.InvalidLineSearch)
        else if prms.lineSearch == LBFGSLineSearchAlg.BacktrackingOrthantWise then
            return Some(LBFGSReturnCode.InvalidLineSearch)
        if momentum < 0 || momentum >= 1 then return Some(LBFGSReturnCode.InvalidMomentum)

        None
    end checkdmLBFGSArgs4Errors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determines what `LBFGSLineSearch` implementation to use in the dmL-BFGS optimization.
     *
     *  @param selection  `LBFGSLineSearchAlg` that describes the user selection for the
                           line search algorithm to be used in the dmL-BFGS optimization.
     *  @return            LBFGSLineSearch implementation of the line search algorithm selected
                           by the user to be used in the dmL-BFGS optimization.
     */
    private def determineLineSearchImple (selection: LBFGSLineSearchAlg): LBFGSLineSearch =
        selection match
            case LBFGSLineSearchAlg.Default                 => LBFGSMoreThuente
            case LBFGSLineSearchAlg.MoreThuente             => LBFGSMoreThuente
            case LBFGSLineSearchAlg.BacktrackingDefault     => LBFGSBacktrackingWolfe
            case LBFGSLineSearchAlg.BacktrackingArmijo      => LBFGSBacktrackingArmijo
            case LBFGSLineSearchAlg.BacktrackingWolfe       => LBFGSBacktrackingWolfe
            case LBFGSLineSearchAlg.BacktrackingStrongWolfe => LBFGSBacktrackingStrongWolfe
            case LBFGSLineSearchAlg.BacktrackingOrthantWise => LBFGSBacktrackingOrthantWise
    end determineLineSearchImple

end DM_LBFGS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `mccormickFunctionDMLBFGSTest` main function uses the McCormick Function
 *  to test the `dmlbfgsMain` method provided by the `dmLBFGS` object.
 *  Multiple tests are performed with different values for the variables.
 *
 *  This test function can be run on the sbt shell with the following command:
 *  > runMain scalation.optimization.quasi_newton.mccormickFunctionDMLBFGSTest
 */
@main def mccormickFunctionDMLBFGSTest(): Unit =

    val functionDomainLowerBound = VectorD(-4, -4)
    val functionDomainUpperBound = VectorD(4, 4)
    val functionOptimizationLogic = FunctionOptimization(McCormickFunction)

    println (DM_LBFGS.dmlbfgsMain (2, VectorD (2.50, 3.50), functionOptimizationLogic,
             params = LBFGSPrms (lineSearchPrms = LBFGSLineSearchPrms (defaultStep = 10.5)), momentum = 0.5))

    new PlotC (McCormickFunction.objFunction, functionDomainLowerBound, functionDomainUpperBound,
               DM_LBFGS.getPath, McCormickFunction.functionMinimum)

end mccormickFunctionDMLBFGSTest

