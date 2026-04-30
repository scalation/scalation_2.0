
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Fri Sep 8 14:20:11 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    MoreThuente Line Search Implementation used by the L-BFGS Algorithm
 *
 *  This Scala implementation was made based on the C implementation of the same
 *  algorithm found in the link below.
 *
 *  @see github.com/chokkan/liblbfgs
 */

package scalation
package optimization
package quasi_newton

import scala.math.{abs, max, min, sqrt}

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGSMoreThuente` object implements the MoreThuente line search
 *  algorithm for use in the native implementation of L-BFGS.
 */
object LBFGSMoreThuente extends LBFGSLineSearch:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `LBFGSUpdateTriIntervalReturn` type is a union type representing
     *  the return value for the `update_trial_interval` method implemented by
     *  this line search algorithm.
     *
     *  A successful execution should return `LineSearchTriInterval`, while
     *  an execution with errors should return a `LBFGSReturnCode` error code.
     *  Returning a `LBFGSReturnCode` success code triggers undefined
     *  behavior.
     */
    private type LBFGSUpdateTriIntervalReturn = LineSearchTriInterval | LBFGSReturnCode

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    override def lineSearch (n: Int, x: VectorD, f: Double, g: VectorD, s: VectorD, stp: Double,
                             cd: LBFGSCallbackData, params: LBFGSLineSearchPrms,
                             orthantWise: Option[OrthantWisePrms] = None): LBFGSLineSearchReturn =
        var count = 0
        var errorCode: Option [LBFGSReturnCode] = None

        var dg = 0.0
        var fxm, dgxm, fym, dgym, fm, dgm = 0.0
        var ftest1 = 0.0
        var stmin, stmax = 0.0

        var xNew = x
        var gNew = g
        var fNew = f
        var stpNew = stp

        // Check the input parameters for errors.
        if stp <= 0 then
            return LBFGSLineSearchFailure (LBFGSReturnCode.InvalidPrms, LBFGSLineSearchIncomplete (xNew, fNew))
        
        // Compute the initial gradient in the search direction.
        val dginit = g dot s

        // Make sure that s points to a descent direction.
        if 0 < dginit then
            return LBFGSLineSearchFailure (LBFGSReturnCode.IncreaseGradient, LBFGSLineSearchIncomplete (xNew, fNew))
        
        // Initialize local variables
        var brackt = false
        var stage1 = true
        val dgtest = params.ftol * dginit
        var width  = params.maxStep - params.minStep
        var prev_width = 2.0 * width

        /* The variables stx, fx, dgx contain the values of the step,
           function, and directional derivative at the best step.
           The variables sty, fy, dgy contain the value of the step,
           function, and derivative at the other endpoint of
           the interval of uncertainty.
           The variables stp, f, dg contain the values of the step,
           function, and derivative at the current step.
        */
        var stx, sty = 0.0
        var fx, fy = f
        var dgx, dgy = dginit

        while true do
            // Set the minimum and maximum steps to correspond to the present interval of uncertainty.
            if brackt then
                stmin = min(stx, sty)
                stmax = max(stx, sty)
            else
                stmin = stx
                stmax = stpNew + 4.0 * (stpNew - stx)

            // Clip the step in the range of [stpmin, stpmax]
            if stpNew < params.minStep then stpNew = params.minStep
            if params.maxStep < stpNew then stpNew = params.maxStep

            // If an unusual termination is to occur then let stp be the lowest point obtained so far.
            if (brackt && ((stpNew <= stmin || stmax <= stpNew) || params.maxLineSearch <= count + 1 || errorCode.nonEmpty)) ||
               (brackt && (stmax - stmin <= params.xtol * stmax)) then
                stpNew = stx

            // Compute the current value of xNew: xNew <- x + (*stp) * s.
            xNew = x + (s * stpNew)

            // Evaluate the function and gradient values
            val evaluationResults = cd.evalLogic.evaluate (cd.instance, xNew, cd.n, stpNew)
            fNew = evaluationResults.objFunctionValue
            gNew = evaluationResults.gradientVector

            dg = gNew dot s

            ftest1 = f + stpNew * dgtest
            count += 1

            // Test for errors and convergence
            if brackt && ((stpNew <= stmin || stmax <= stpNew) || errorCode.nonEmpty) then
                // Rounding errors prevent further progress
                return LBFGSLineSearchFailure (LBFGSReturnCode.RoundingError, LBFGSLineSearchIncomplete (xNew, fNew))
            if stpNew == params.maxStep && fNew <= ftest1 && dg <= dgtest then
                // The step is the maximum value
                return LBFGSLineSearchFailure (LBFGSReturnCode.MaximumStep, LBFGSLineSearchIncomplete (xNew, fNew))
            if stpNew == params.minStep && (ftest1 < fNew || dgtest <= dg) then
                // The step is the minimum value
                return LBFGSLineSearchFailure (LBFGSReturnCode.MinimumStep, LBFGSLineSearchIncomplete (xNew, fNew))
            if brackt && (stmax - stmin) <= params.xtol * stmax then
                // Relative width of the interval of uncertainty is at most xtol
                return LBFGSLineSearchFailure (LBFGSReturnCode.WidthTooSmall, LBFGSLineSearchIncomplete (xNew, fNew))
            if params.maxLineSearch <= count then
                // Maximum number of iteration
                return LBFGSLineSearchFailure (LBFGSReturnCode.MaximumLineSearch, LBFGSLineSearchIncomplete (xNew, fNew))
            if fNew <= ftest1 && abs(dg) <= params.gtol * (-dginit) then
                // The sufficient decrease condition and the directional derivative condition hold
                return LBFGSLineSearchStep (xNew, gNew, fNew, stpNew, count)

            /* In the first stage we seek a step for which the modified
               function has a non-positive value and non-negative derivative.
             */
            if stage1 && fNew <= ftest1 && min(params.ftol, params.gtol) * dginit <= dg then
                stage1 = false

            /* A modified function is used to predict the step only if
               we have not obtained a step for which the modified
               function has a non-positive function value and non-negative
               derivative, and if a lower function value has been
               obtained but the decrease is not sufficient.
             */
            if stage1 && ftest1 < fNew && fNew <= fx then
                // Define the modified function and derivative values
                fm   = fNew - stpNew * dgtest
                fxm  = fx - stx * dgtest
                fym  = fy - sty * dgtest
                dgm  = dg - dgtest
                dgxm = dgx - dgtest
                dgym = dgy - dgtest

                // Call update_trial_interval() to update the interval of uncertainty and to compute the new step.
                update_trial_interval ( stx, fxm, dgxm, sty, fym, dgym, stpNew, fm, dgm, stmin, stmax, brackt) match
                    case trialInterval: LineSearchTriInterval =>
                        stx    = trialInterval.x
                        fxm    = trialInterval.fx
                        dgxm   = trialInterval.dx
                        sty    = trialInterval.y
                        fym    = trialInterval.fy
                        dgym   = trialInterval.dy
                        stpNew = trialInterval.t
                        brackt = trialInterval.brackt
                    case returnCode: LBFGSReturnCode => errorCode = Some(returnCode)

                // Reset the function and gradient values for f
                fx  = fxm + stx * dgtest
                fy  = fym + sty * dgtest
                dgx = dgxm + dgtest
                dgy = dgym + dgtest
            else
                // Call update_trial_interval() to update the interval of uncertainty and to compute the new step.
                update_trial_interval (stx, fx, dgx, sty, fy, dgy,
                                       stpNew, fNew, dg, stmin, stmax, brackt) match
                    case trialInterval: LineSearchTriInterval =>
                        stx    = trialInterval.x
                        fx     = trialInterval.fx
                        dgx    = trialInterval.dx
                        sty    = trialInterval.y
                        fy     = trialInterval.fy
                        dgy    = trialInterval.dy
                        stpNew = trialInterval.t
                        brackt = trialInterval.brackt
                    case returnCode: LBFGSReturnCode => errorCode = Some(returnCode)
            end if

            // Force a sufficient decrease in the interval of uncertainty.
            if brackt then
                if 0.66 * prev_width <= abs(sty - stx) then
                    stpNew = stx + 0.5 * (sty - stx)
                prev_width = width
                width = abs(sty - stx)
            end if
        end while

        LBFGSLineSearchFailure (LBFGSReturnCode.LogicError, LBFGSLineSearchIncomplete (xNew, fNew))

    end lineSearch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    private def update_trial_interval (x: Double, fx: Double, dx: Double, y: Double, fy: Double, dy: Double,
                                       t: Double, ft: Double, dt: Double, tmin: Double, tmax: Double,
                                       brackt: Boolean): LBFGSUpdateTriIntervalReturn =
        var newBrackt: Boolean = brackt
        var newx  = x
        var newfx = fx
        var newdx = dx
        var newy  = y
        var newfy = fy
        var newdy = dy
        var bound: Boolean = false
        val dsign: Boolean = (dt * (dx/abs(dx))) < 0
        var mc: Double = 0.0                                // minimizer of an interpolated cubic
        var mq: Double = 0.0                                // minimizer of an interpolated quadratic
        var newt: Double = 0.0                              // new trial value

        // Check the input parameters for errors
        if brackt then
            if t <= min(x, y) || max(x, y) <= t then
                return LBFGSReturnCode.OutOfInterval        // trival value t is out of the interval
            if 0.0 <= dx * (t - x) then
                return LBFGSReturnCode.IncreaseGradient     // function must decrease from x
            if tmax < tmin then
                return LBFGSReturnCode.IncorrectTMinMax     // incorrect tmin and tmax specified
        end if

        // Trial value selection.
        if fx < ft then
            /* Case 1: a higher function value.
               The minimum is brackt. If the cubic minimizer is closer
               to x than the quadratic one, the cubic one is taken, else
               the average of the minimizers is taken.
             */
            newBrackt = true
            bound = true
            mc = cubicMinimizer(x, fx, dx, t, ft, dt)
            mq = quadMinimizer(x, fx, dx, t, ft)
            newt = if abs(mc - x) < abs(mq - x) then mc else mc + 0.5 * (mq - mc)
        else if dsign then
            /* Case 2: a lower function value and derivatives of
               opposite sign. The minimum is brackt. If the cubic
               minimizer is closer to x than the quadratic (secant) one,
               the cubic one is taken, else the quadratic one is taken.
             */
            newBrackt = true
            bound = false
            mc = cubicMinimizer(x, fx, dx, t, ft, dt)
            mq = quadMinimizer2(x, dx, t, dt)
            newt = if abs(mc - t) > abs(mq - t) then mc else mq
        else if abs(dt) < abs(dx) then
            /* Case 3: a lower function value, derivatives of the
               same sign, and the magnitude of the derivative decreases.
               The cubic minimizer is only used if the cubic tends to
               infinity in the direction of the minimizer or if the minimum
               of the cubic is beyond t. Otherwise the cubic minimizer is
               defined to be either tmin or tmax. The quadratic (secant)
               minimizer is also computed and if the minimum is brackt
               then the the minimizer closest to x is taken, else the one
               farthest away is taken.
             */
            bound = true
            mc = cubicMinimizer2(x, fx, dx, t, ft, dt, tmin, tmax)
            mq = quadMinimizer2(x, dx, t, dt)
            if brackt then
                newt = if abs(t - mc) < abs(t - mq) then mc else mq
            else
                newt = if abs(t - mc) > abs(t - mq) then mc else mq
        else
            /* Case 4: a lower function value, derivatives of the
               same sign, and the magnitude of the derivative does
               not decrease. If the minimum is not brackt, the step
               is either tmin or tmax, else the cubic minimizer is taken.
             */
            bound = false
            newt = if brackt then cubicMinimizer(t, ft, dt, y, fy, dy)
                   else if x < t then tmax
                   else tmin
        end if

        /* Update the interval of uncertainty. This update does not
            depend on the new step or the case analysis above.
            - Case a: if f(x) < f(t),
                x <- x, y <- t.
            - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
                x <- t, y <- y.
            - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
                x <- t, y <- x.
         */
        if fx < ft then             // Case a
            newy = t
            newfy = ft
            newdy = dt
        else
            if dsign then           // Case c
                newy = x
                newfy = fx
                newdy = dx
            newx = t                // Cases b and c
            newfx = ft
            newdx = dt
        end if

        // Clip the new trial value in [tmin, tmax]
        if tmax < newt then newt = tmax
        if newt < tmin then newt = tmin

        // Redefine the new trial value if it is close to the upper bound of the interval
        if newBrackt && bound then
            mq = newx + 0.66 * (newy - newx)
            if newx < newy then
                if mq < newt then newt = mq
            else
                if newt < mq then newt = mq
        end if

        // Return the new trial value
        LineSearchTriInterval (newx, newfx, newdx, newy, newfy, newdy, newt, newBrackt)
    end update_trial_interval

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find a minimizer of an interpolated cubic function.
     *  @param  u       The value of one point, u.
     *  @param  fu      The value of f(u).
     *  @param  du      The value of f'(u).
     *  @param  v       The value of another point, v.
     *  @param  fv      The value of f(v).
     *  @param  dv      The value of f'(v).
     *  @return Double  The minimizer of the interpolated cubic.
     */
    private inline def cubicMinimizer (u: Double, fu: Double, du: Double,
                                       v: Double, fv: Double, dv: Double): Double =
        val d = v - u
        val theta = (fu - fv) * 3/d + du + dv
        var p = abs(theta)
        var q = abs(du)
        var r = abs(dv)
        val s = max(p, max(q, r))
        val a = theta / s
        var gamma = s * sqrt(a~^2 - (du/s) * (dv/ s))
        if v < u then gamma = -gamma
        p = gamma - du + theta
        q = gamma - du + gamma + dv
        r = p / q
        u + r * d
    end cubicMinimizer

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find a minimizer of an interpolated cubic function.
     *  @param  u       The value of one point, u.
     *  @param  fu      The value of f(u).
     *  @param  du      The value of f'(u).
     *  @param  v       The value of another point, v.
     *  @param  fv      The value of f(v).
     *  @param  dv      The value of f'(v).
     *  @param  xmin    The minimum value.
     *  @param  xmax    The maximum value.
     *  @return Double  The minimizer of the interpolated cubic.
     */
    private inline def cubicMinimizer2 (u: Double, fu: Double, du: Double, v: Double,
                                        fv: Double, dv: Double, xmin: Double, xmax: Double): Double =
        val d = v - u
        val theta = (fu - fv) * 3/d + du + dv
        var p = abs(theta)
        var q = abs(du)
        var r = abs(dv)
        val s = max(p, max(q, r))
        val a = theta / s
        var gamma = s * sqrt(max(0, a~^2 - (du/s) * (dv/ s)))
        if u < v then gamma = -gamma
        p = gamma - dv + theta
        q = gamma - dv + gamma + du
        r = p / q
        if r < 0 && gamma != 0 then
            v - r * d
        else if d > 0 then
            xmax
        else
            xmin
    end cubicMinimizer2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find a minimizer of an interpolated quadratic function.
     *  @param  u       The value of one point, u.
     *  @param  fu      The value of f(u).
     *  @param  du      The value of f'(u).
     *  @param  v       The value of another point, v.
     *  @param  fv      The value of f(v).
     *  @return Double  The minimizer of the interpolated quadratic.
     */
    private inline def quadMinimizer (u: Double, fu: Double, du: Double,
                                      v: Double, fv: Double): Double =
        val a = v - u
        u + du / ((fu - fv)/a + du)/2 * a
    end quadMinimizer

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find a minimizer of an interpolated quadratic function.
     *  @param  u       The value of one point, u.
     *  @param  du      The value of f'(u).
     *  @param  v       The value of another point, v.
     *  @param  dv      The value of f'(v).
     *  @return Double  The minimizer of the interpolated quadratic.
     */
    private inline def quadMinimizer2 (u: Double, du: Double, v: Double, dv: Double): Double =
        val a = u - v
        v + dv/(dv - du) * a
    end quadMinimizer2

end LBFGSMoreThuente

