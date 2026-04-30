
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira, Shivam Anant Rathi
 *  @version 2.0
 *  @note    Tue Aug 22 15:39:53 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Limited Memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) Algorithm
 *  @see     github.com/chokkan/liblbfgs
 */

package scalation
package optimization
package quasi_newton

import scala.math.abs

import scalation.mathstat._
import scalation.optimization.functions.*
import scalation.scala2d.writeImage

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGS` the class implements the Limited-Memory Broyden–Fletcher–Goldfarb–Shanno (BFGS)
 *  Quasi-Newton Algorithm for solving Non-Linear Programming (NLP) problems.
 *      minimize    f(x)
 *  @param f   the multi-variate objective function to be minimized
 *  @param gr  its gradient vector-valued function
 */
case class LBFGS (f: FunctionV2S, g: FunctionV2V = null)
      extends Minimize:

    private val flaw = flawf ("LBFGS")                                // flaw function

    private val funcLogic = if g == null then FunctionEvaluation (f) 
                            else FunctionEvaluation (f, g)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the Non-Linear Programming (NLP) problem by starting at x0 and
     *  iteratively moving down in the search space to a minimal point.
     *  Return the optimal point/vector x and its objective function value.
     *  For more options @see `LBFGS.lbfgsMain`
     *  @param x0  the starting point (initial guess)
     *  @param α   the current learning rate/step size
     */
    def solve (x0: VectorD, α: Double = eta): FuncVec =
        val result = LBFGS.lbfgsMain (x0.dim, x0, funcLogic)
        if result.returnCode != LBFGSReturnCode.Success then
            flaw ("solve", s"failed with return code = ${result.returnCode}")
        (result.finalFunctionValue.get, result.optimizedVariables)
    end solve

end LBFGS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGS` object implementats of the Limited memory Broyden–Fletcher–Goldfarb–Shanno
 *  (BFGS) for unconstrained optimization (L-BFGS) algorithm.  This Scala implementation
 *  was made based on the C implementation of the same algorithm found in the following link.
 *  @see github.com/chokkan/liblbfgs
 */
object LBFGS extends PathMonitor:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs the L-BFGS optimization that optimizes variables to minimize a function value.
     *  @param n              the dimensionality of the optimization problem
     *  @param x              the starting point (initial guess)
     *  @param functionLogic  the logic defining the objective function and its gradient
     *  @param params         the optimization control hyper-parameters (@see `LBFGSPrms`) 
     *  @param instance       an optional user data segment that may be provided when calling this method
     *                        (@see `OptimizationLogic`)
     */
    def lbfgsMain (n: Int, x: VectorD, functionLogic: EvaluationLogic | OptimizationLogic,
                   params: LBFGSPrms = LBFGSPrms (), instance: Any = None): LBFGSResults =
        clearPath ()

        checkLBFGSArgs4Errors (n, params) match
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
            var xp, g, gp, d = new VectorD (n)
            var pg, pf, s, y = VectorD.nullv
            val lm = Array.ofDim [LBFGSIterationData] (m)

            var ys, yy, xnorm, gnorm, beta, fx, rate = 0.0

            val lineSearchImple = LBFGSLineSearch.getImple (params.lineSearch)

            val cd = LBFGSCallbackData (n, instance, functionLogic)      // construct a callback data

            // Allocate working space
            if useOrthantWiseLogic then pg = new VectorD (n)             // allocate working space for OW-LQN

            // Allocate an array for storing previous values of the objective function.
            if 0 < params.past then pf = new VectorD (params.past)

            // Evaluate the function value and its gradient. */
            val evaluationResults = cd.evalLogic.evaluate (cd.instance, x, cd.n, 0)
            fx = evaluationResults.objFunctionValue
            g  = evaluationResults.gradientVector

            if useOrthantWiseLogic then
                // Compute the L1 norm of the variable and add it to the object value.
                val orthantWisePrms = params.orthantWise.get
                xnorm = orthantWisePrms.x1Norm (x)
                fx   += xnorm * orthantWisePrms.c
                pg    = orthantWisePrms.pseudoGradient (x, g)
            end if

            // Store the initial value of the objective function. */
            if pf != VectorD.nullv then pf(0) = fx

            // Compute the direction: assume the initial hessian matrix H_0 as the identity matrix
            d = if ! useOrthantWiseLogic then -g else -pg

            // Make sure that the initial variables are not a minimizer.
            xnorm = x.norm
            gnorm = if ! useOrthantWiseLogic then g.norm else pg.norm
            if xnorm < 1.0 then xnorm = 1.0
            if gnorm / xnorm <= params.epsilon then
                return LBFGSResults (LBFGSReturnCode.AlreadyMinimized, xNew, Some(fx), None)

            // Compute the initial step: step = 1.0 / sqrt(vecdot(d, d, n))
            step = 1.0 / d.norm

            while true do
                xp = xNew                       // Store the current position and gradient vectors
                gp = g

                // Search for an optimal step
                lineSearchResults = if ! useOrthantWiseLogic then
                    lineSearchImple.lineSearch(n, xNew, fx, g, d, step, cd, params.lineSearchPrms, params.orthantWise)
                else
                    lineSearchImple.lineSearch(n, xNew, fx, pg, d, step, cd, params.lineSearchPrms, params.orthantWise)

                lineSearchResults match
                    case lineSearchStep: LBFGSLineSearchStep =>
                        xNew = lineSearchStep.x
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
                    return LBFGSResults(LBFGSReturnCode.Success, xNew, Some(fx), None)

                // Test for stopping criterion:  The criterion is given by the following formula:
                //    |(f(past_x) - f(x))| / f(x) < \delta
                if pf != VectorD.nullv then
                    if params.past <= k then                                   // don't test stopping criterion while k < past
                        rate = (pf(k % params.past) - fx) / fx                 // compute the relative improvement from the past
                        if abs(rate) < params.delta then                       // the stopping criterion
                            return LBFGSResults(LBFGSReturnCode.Stop, xNew, Some(fx), None)
                    end if

                    pf(k % params.past) = fx                                   // Store current value of the objective function
                end if

                if params.maxIterations != 0 && params.maxIterations < k + 1 then
                    return LBFGSResults (LBFGSReturnCode.MaximumIteration, xNew, Some(fx), None)

                // Update vectors s and y:
                //     s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}
                //     y_{k+1} = g_{k+1} - g_{k}
                s = xNew - xp
                y = g - gp

                // Compute scalars ys and yy:
                //    ys = y^t \cdot s = 1 / \rho
                //    yy = y^t \cdot y
                // Notice that yy is used for scaling the hessian matrix H_0 (Cholesky factor).
                ys = y dot s
                yy = y dot y

                lm(end) = LBFGSIterationData (s, y, ys, 0)

                // Recursive formula to compute dir = -(H \cdot g)
                // This is described in page 779 of: Jorge Nocedal.
                // Updating Quasi-Newton Matrices with Limited Storage.
                // Mathematics of Computation, Vol. 35, No. 151, pp. 773--782, 1980.
                bound = if m <= k then m else k
                k  += 1
                end = (end + 1) % m

                // Compute the steepest direction, i.e., the negative of gradients.
                d = if ! useOrthantWiseLogic then -g else -pg

                j = end
                cfor (0, bound) { _ =>
                    j = (j + m - 1) % m                             // if (--j == -1) j = m-1
                    val it = lm(j)
                    it.alpha = it.s dot d                           // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
                    it.alpha = it.alpha / it.ys
                    d += (it.y * (-it.alpha))                       // q_{i} = q_{i+1} - \alpha_{i} y_{i}
                } // cfor

                d *= (ys / yy)

                cfor (0, bound) { _ =>
                    val it = lm(j)
                    beta = it.y dot d                               // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
                    beta /= it.ys
                    d += it.s * (it.alpha - beta)                   // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
                    j = (j + 1) % m                                 // if (++j == m) j = 0
                } // cfor

                // Constrain the search direction for orthant-wise updates.
                if useOrthantWiseLogic then
                    val orthantWisePrms = params.orthantWise.get
                    for i <- orthantWisePrms.start until orthantWisePrms.end.getOrElse(d.length) do
                        if d(i) * pg(i) >= 0 then d(i) = 0
                end if

                // Now the search direction d is ready. We try the default step first.
                step = params.lineSearchPrms.defaultStep          // Previously default step was hardcoded to 1.
            end while

            LBFGSResults(LBFGSReturnCode.UnknownError, xNew, Some(fx), None)
        catch
            case _ : OutOfMemoryError => LBFGSResults(LBFGSReturnCode.OutOfMemory, x, None, None)

    end lbfgsMain


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Checks if the L-BFGS optimization arguments have an error and returns an
     *  `Option` with the `LBFGSReturnCode` that represents the first error  found.
     *
     *  @param n     the number of variables.
     *  @param prms  `BFGSPrms` class representing the parameters chosen to control the
     *               L-BFGS optimization.
     *  @return      Option [LBFGSReturnCode] value with a `BFGSReturnCode` return code that represents
     *               the first error found in the `params` argument. If there are no errors in the `params`
     *               argument, `None` is returned.
     */
    private def checkLBFGSArgs4Errors (n: Int, prms: LBFGSPrms): Option[LBFGSReturnCode] =
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

        None
    end checkLBFGSArgs4Errors

end LBFGS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `boothFunctionLBFGSTest` main function uses the Booth Function to test
 *  the `lbfgsMain` method provided by the `LBFGS` object. Multiple tests are
 *  performed with different values for the variables.
 *
 *  The Booth Function can be described as follows:
 *
 *  - Input dimension: 2;
 *
 *  - Function domain: -10 &le; x,,i,, &le; 10;
 *
 *  - Function definition: f(x) = (x,,0,, + 2 * x,,1,, - 7)^2^ + (2 * x,,0,, +
 *  x,,1,, - 5)^2^;
 *
 *  - Global minimum: x* = (1, 3); f(x*) = 0;
 *
 *  This test function can be run on the sbt shell with the following command:
 *  > runMain scalation.optimization.quasi_newton.boothFunctionLBFGSTest
 */
@main def boothFunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD (-10, -10)
    val functionDomainUpperBound  = VectorD (10, 10)
    val functionOptimizationLogic = FunctionOptimization (BoothFunction)

//  println (LBFGS.lbfgsMain (2, VectorD (1, 3),   functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (2, 3.5), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (0, 0),   functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (-4, 7),  functionOptimizationLogic))

    val plot = new PlotC (BoothFunction.objFunction, functionDomainLowerBound,
               functionDomainUpperBound, LBFGS.getPath, BoothFunction.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_boothFunction_plot.png", plot)

end boothFunctionLBFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bealeFunctionLBFGSTest` main function uses the Beale Function.
 *  > runMain scalation.optimization.quasi_newton.bealeFunctionLBFGSTest
 */
@main def bealeFunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD (-10, -10)
    val functionDomainUpperBound  = VectorD (10, 10)
    val functionOptimizationLogic = FunctionOptimization (BealeFunction)

//  println (LBFGS.lbfgsMain (2, VectorD (-4.5, -4.5), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (-2, 2),      functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (0, 1),       functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (2, -2),      functionOptimizationLogic))

    val plot = new PlotC (BealeFunction.objFunction, functionDomainLowerBound,
               functionDomainUpperBound, LBFGS.getPath, BealeFunction.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_bealeFunction_plot.png", plot)

end bealeFunctionLBFGSTest

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bohachevsky1FunctionLBFGSTest` main function uses the Bohachevsky1 Function.
 *  > runMain scalation.optimization.quasi_newton.bohachevsky1FunctionLBFGSTest
 */
@main def bohachevsky1FunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD (-10, -10)
    val functionDomainUpperBound  = VectorD (10, 10)
    val functionOptimizationLogic = FunctionOptimization (Bohachevsky1Function)

//  println (LBFGS.lbfgsMain (2, VectorD (-10, 10), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (-5, 5),   functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (5, -5),   functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (10, -10), functionOptimizationLogic))

    val plot = new PlotC (Bohachevsky1Function.objFunction, functionDomainLowerBound,
               functionDomainUpperBound, LBFGS.getPath, Bohachevsky1Function.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_bohachevsky1Function_plot.png", plot)

end bohachevsky1FunctionLBFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bohachevsky2FunctionLBFGSTest` main function uses the Bohachevsky2  Function.
 *  > runMain scalation.optimization.quasi_newton.bohachevsky2FunctionLBFGSTest
 */
@main def bohachevsky2FunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD (-10, -10)
    val functionDomainUpperBound  = VectorD (10, 10)
    val functionOptimizationLogic = FunctionOptimization (Bohachevsky2Function)

//  println (LBFGS.lbfgsMain (2, VectorD (-10, 10), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (-5, 5),   functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (5, -5),   functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (10, -10), functionOptimizationLogic))

    val plot = new PlotC (Bohachevsky2Function.objFunction, functionDomainLowerBound,
                   functionDomainUpperBound, LBFGS.getPath, Bohachevsky2Function.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_bohachevsky2Function_plot.png", plot)

end bohachevsky2FunctionLBFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bohachevsky3FunctionLBFGSTest` main function uses the Bohachevsky3 Function.
 *  > runMain scalation.optimization.quasi_newton.bohachevsky3FunctionLBFGSTest
 */
@main def bohachevsky3FunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD (-10, -10)
    val functionDomainUpperBound  = VectorD (10, 10)
    val functionOptimizationLogic = FunctionOptimization (Bohachevsky3Function)

//  println (LBFGS.lbfgsMain (2, VectorD (-10, 10), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (-5, 5),   functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (5, -5),   functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (10, -10), functionOptimizationLogic))

    val plot = new PlotC (Bohachevsky3Function.objFunction, functionDomainLowerBound,
               functionDomainUpperBound, LBFGS.getPath, Bohachevsky3Function.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_bohachevsky3Function_plot.png", plot)

end bohachevsky3FunctionLBFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `camel3FunctionLBFGSTest` main function uses the Camel3 Function.
 *  > runMain scalation.optimization.quasi_newton.camel3FunctionLBFGSTest
 */
@main def camel3FunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD  (-10, -10)
    val functionDomainUpperBound  = VectorD  (10, 10)
    val functionOptimizationLogic = FunctionOptimization (Camel3Function)

//  println (LBFGS.lbfgsMain (2, VectorD (-10, 10), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (-5, 5),   functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (5, -5),   functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (10, -10), functionOptimizationLogic))

    val plot = new PlotC (Camel3Function.objFunction, functionDomainLowerBound,
               functionDomainUpperBound, LBFGS.getPath, Camel3Function.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_camel3Function_plot.png", plot)

end camel3FunctionLBFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cubeFunctionLBFGSTest` main function uses the Cube Function.
 *  > runMain scalation.optimization.quasi_newton.cubeFunctionLBFGSTest
 */
@main def cubeFunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD (-10, -10)
    val functionDomainUpperBound  = VectorD (10, 10)
    val functionOptimizationLogic = FunctionOptimization (CubeFunction)

//  println (LBFGS.lbfgsMain (2, VectorD (-10, 10), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (-5, 5),   functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (5, -5),   functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (10, -10), functionOptimizationLogic))

    val plot = new PlotC (CubeFunction.objFunction, functionDomainLowerBound,
               functionDomainUpperBound, LBFGS.getPath, CubeFunction.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_cubeFunction_plot.png", plot)

end cubeFunctionLBFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `freudensteinRothFunctionLBFGSTest` main function uses the FreudensteinRoth Function.
 *  > runMain scalation.optimization.quasi_newton.freudensteinRothFunctionLBFGSTest
 */
@main def freudensteinRothFunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD (-10, -10)
    val functionDomainUpperBound  = VectorD (10, 10)
    val functionOptimizationLogic = FunctionOptimization (FreudensteinRothFunction)

//  println (LBFGS.lbfgsMain (2, VectorD (-10, 10), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (-5, 5),   functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (5, -5),   functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (10, -10), functionOptimizationLogic))

    val plot = new PlotC (FreudensteinRothFunction.objFunction, functionDomainLowerBound,
               functionDomainUpperBound, LBFGS.getPath, FreudensteinRothFunction.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_freudensteinRothFunction_plot.png", plot)

end freudensteinRothFunctionLBFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `mccormickFunctionLBFGSTest` main function uses the McCormick Function to test
 *  the `lbfgsMain` method provided by the `LBFGS` object. Multiple tests are
 *  performed with different values for the variables.
 *  > runMain scalation.optimization.quasi_newton.mccormickFunctionLBFGSTest
 */
@main def mccormickFunctionLBFGSTest (): Unit =

    val functionDomainLowerBound  = VectorD (-4, -4)
    val functionDomainUpperBound  = VectorD (4, 4)
    val functionOptimizationLogic = FunctionOptimization (McCormickFunction)

//  println (LBFGS.lbfgsMain (2, VectorD (-0.5, -1.5), functionOptimizationLogic))
//  println (LBFGS.lbfgsMain (2, VectorD (0, -0.5),    functionOptimizationLogic))
    println (LBFGS.lbfgsMain (2, VectorD (2.50, 3.50), functionOptimizationLogic,
             params = LBFGSPrms (lineSearchPrms = LBFGSLineSearchPrms (defaultStep = 10))))
//  println (LBFGS.lbfgsMain (2, VectorD (-1.49, -2.99), functionOptimizationLogic, params = LBFGSPrms (defaultStep = 10)))


    val plot = new PlotC (McCormickFunction.objFunction, functionDomainLowerBound,
               functionDomainUpperBound, LBFGS.getPath, McCormickFunction.functionMinimum)
    writeImage ("./plots/LBFGS/LBFGS_mccormickFunction_plot.png", plot)

end mccormickFunctionLBFGSTest

