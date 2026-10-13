//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Thu Oct  1 10:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Levenberg-Marquardt for Nonlinear Least Squares (NLS) with Bounds:
 *           Trust-Region Reflective steps (Coleman & Li)
 *           AI Assisted Code
 *
 *  @see Coleman, T. F. and Li, Y. (1996), "An Interior Trust Region Approach for
 *       Nonlinear Minimization Subject to Bounds", SIAM J. Optimization 6 (2), 418-445.
 *  @see Branch, M. A., Coleman, T. F. and Li, Y. (1999), "A Subspace, Interior, and
 *       Conjugate Gradient Method for Large-Scale Bound-Constrained Minimization
 *       Problems", SIAM J. Scientific Computing 21 (1), 1-23.
 *  @see SciPy `least_squares (method = "trf")`, whose step selection this follows.
 */

package scalation
package optimization
package nonlinear_ls

import scala.math.{abs, max, min, sqrt}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LevenbergMarquardt_TR` class solves Non-Linear Least Squares (NLS) problems
 *  with box constraints lb <= b <= ub by the Trust-Region Reflective method of
 *  Coleman and Li, driven by Levenberg-Marquardt damping.  Compared with projecting
 *  the unconstrained LM step (@see `LevenbergMarquardt_B`), which stalls once a
 *  bound is active (the blocked component poisons the step of the free ones and
 *  the step-norm test never fires), this class
 *
 *  1. keeps every iterate STRICTLY inside the box (a fraction theta of the way to a bound),
 *  2. rescales the problem by the Coleman-Li matrix D = diag (sqrt (v)), where v_j is
 *     the distance from b_j to the bound its gradient component points at (1 when
 *     there is none), so the step shrinks in the directions that are about to hit a
 *     bound and the first-order condition becomes || v (*) g ||_inf = 0, which holds
 *     at a bound-constrained optimum (the scaled gradient vanishes where either the
 *     gradient is zero or the variable sits on a bound the gradient points at),
 *  3. solves the damped normal equations in the scaled variables,
 *         (D J^T J D + diag (g (*) dv) + lambda S) s_h = -D g,   S = diag (max (A_jj, 1e-4)),
 *     which with no finite bounds (D = I, dv = 0) is exactly the parent's LM step, and
 *  4. when the step crosses a bound, compares three candidates by the quadratic model
 *     within the implicit trust region of radius || s_h ||: the step truncated to the
 *     interior, the step REFLECTED off the bound it hits, and the scaled steepest
 *     descent step (each with a 1-D minimization along its ray), and takes the best.
 *
 *  The damping lambda is updated as in the parent (divided by 10 on an accepted step,
 *  multiplied by 10 on a rejected one), so the two solvers behave identically until a
 *  bound becomes active.  Stops on the scaled gradient (gtol), on a relative decrease
 *  of the SSE below ftol, on a step shorter than eps, on the parent's maxIter, or when
 *  lambda exceeds 1e9 (no acceptable step).
 *  @param n      the number of parameters
 *  @param lb     the lower bounds (-infinity for none), lb_j < ub_j
 *  @param ub     the upper bounds (+infinity for none)
 *  @param rFn    the vectorized residual function r (b) = y - f (x, b)
 *  @param jFn    the optional vectorized analytical Jacobian J_ij = d r_i / d b_j
 *  @param theta  the fraction of the distance to a bound a step may cover (strict interior)
 *  @param gtol   the tolerance on the scaled gradient || v (*) g ||_inf (first-order optimality)
 *  @param ftol   the tolerance on the relative decrease of the SSE (actual and predicted)
 */
class LevenbergMarquardt_TR (n: Int, lb: VectorD, ub: VectorD,
                             rFn: VectorD => VectorD,
                             jFn: Option [VectorD => MatrixD] = None,
                             theta: Double = 0.995, gtol: Double = 1e-8, ftol: Double = 1e-10)
      extends LevenbergMarquardt (n, rFn, jFn):

    private val debug = debugf ("LevenbergMarquardt_TR", true)
    private val flaw  = flawf ("LevenbergMarquardt_TR")

    private val RSTEP      = 1e-10                                    // relative nudge off a bound
    private val LAMBDA_MAX = 1e9                                      // damping beyond which no step is acceptable
    private val FACTOR     = 10.0                                     // damping factor (as in the parent)

    private var nLM        = 0                                        // accepted steps of the last run, by kind:
    private var nTruncated = 0                                        //   the LM step itself / cut at the box /
    private var nReflected = 0                                        //   reflected off a bound /
    private var nGradient  = 0                                        //   the scaled steepest-descent step

    if lb.dim != n || ub.dim != n then flaw ("init", s"bounds must have n = $n elements: lb.dim = ${lb.dim}, ub.dim = ${ub.dim}")
    for j <- 0 until n if ! (lb(j) < ub(j)) do flaw ("init", s"need lb($j) < ub($j): ${lb(j)} vs ${ub(j)}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Project a parameter vector onto the box [lb, ub].
     *  @param b  the parameter vector
     */
    def project (b: VectorD): VectorD =
        val p = new VectorD (n)
        for j <- 0 until n do p(j) = max (lb(j), min (ub(j), b(j)))
        p
    end project

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of accepted steps of the last `optimize` run by kind:
     *  "lm" (the LM step itself), "truncated" (cut at the box), "reflected" (reflected
     *  off a bound) and "gradient" (the scaled steepest-descent step), @see `selectStep`.
     */
    def stepCounts: Map [String, Int] =
        Map ("lm" -> nLM, "truncated" -> nTruncated, "reflected" -> nReflected, "gradient" -> nGradient)
    end stepCounts

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a copy of parameter vector b moved strictly inside the box: a component
     *  on (or beyond) a bound is placed RSTEP (relative) inside it.
     *  @param b  the parameter vector
     */
    private def interior (b: VectorD): VectorD =
        val p = project (b)
        for j <- 0 until n do
            val (l, u) = (lb(j), ub(j))
            if p(j) <= l then p(j) = l + RSTEP * max (1.0, abs (l))      // l is finite here
            if p(j) >= u then p(j) = u - RSTEP * max (1.0, abs (u))      // u is finite here
            if p(j) <= l then p(j) = 0.5 * (l + u)                       // bounds closer than the nudge
        end for
        p
    end interior

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Coleman-Li scaling vector v and its derivative dv = d v / d b:
     *  v_j = ub_j - b_j (dv_j = -1) when the gradient points at a finite upper bound,
     *  v_j = b_j - lb_j (dv_j = +1) when it points at a finite lower bound, else 1 (0).
     *  @param b  the current parameter vector
     *  @param g  the gradient J^T r
     */
    private def clScaling (b: VectorD, g: VectorD): (VectorD, VectorD) =
        val v  = VectorD.one (n)
        val dv = new VectorD (n)
        for j <- 0 until n do
            if g(j) < 0.0 && ! ub(j).isInfinite then
                v(j) = ub(j) - b(j); dv(j) = -1.0
            else if g(j) > 0.0 && ! lb(j).isInfinite then
                v(j) = b(j) - lb(j); dv(j) = 1.0
        end for
        (v, dv)
    end clScaling

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the largest alpha >= 0 with b + alpha s inside the box (infinity if the
     *  ray never leaves it) and the vector of hits: +1 for a component that reaches
     *  its upper bound there, -1 for its lower bound, 0 otherwise.
     *  @param b  the current parameter vector (inside the box)
     *  @param s  the step direction
     */
    private def stepToBound (b: VectorD, s: VectorD): (Double, VectorD) =
        var alpha = Double.PositiveInfinity
        for j <- 0 until n do
            if s(j) > 0.0 && ! ub(j).isInfinite then alpha = min (alpha, (ub(j) - b(j)) / s(j))
            else if s(j) < 0.0 && ! lb(j).isInfinite then alpha = min (alpha, (lb(j) - b(j)) / s(j))
        end for
        val hits = new VectorD (n)
        if ! alpha.isInfinite then
            for j <- 0 until n do
                val bj = b(j) + alpha * s(j)
                if s(j) > 0.0 && ! ub(j).isInfinite && bj >= ub(j) then hits(j) = 1.0
                else if s(j) < 0.0 && ! lb(j).isInfinite && bj <= lb(j) then hits(j) = -1.0
            end for
        (alpha, hits)
    end stepToBound

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the quadratic model of the change in SSE / 2 for a scaled step s_h:
     *  m (s_h) = 1/2 s_h^T A s_h + g_h . s_h, with A = D J^T J D + diag (g (*) dv).
     *  @param a   the scaled Hessian approximation A
     *  @param gh  the scaled gradient D g
     *  @param sh  the scaled step
     */
    private def model (a: MatrixD, gh: VectorD, sh: VectorD): Double =
        0.5 * (sh dot (a * sh)) + (gh dot sh)
    end model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Minimize the quadratic model along the ray s0 + t dirn for t in [tLo, tHi],
     *  returning the best t and its model value.
     *  @param a     the scaled Hessian approximation A
     *  @param gh    the scaled gradient D g
     *  @param s0    the starting point of the ray (scaled)
     *  @param dirn  the direction of the ray (scaled)
     *  @param tLo   the smallest stride allowed
     *  @param tHi   the largest stride allowed
     */
    private def minQuad1D (a: MatrixD, gh: VectorD, s0: VectorD, dirn: VectorD,
                           tLo: Double, tHi: Double): (Double, Double) =
        val aa = dirn dot (a * dirn)                                      // second-order coefficient (x 2)
        val bb = (dirn dot (a * s0)) + (gh dot dirn)                      // first-order coefficient
        var tBest = tLo
        var vBest = model (a, gh, s0 + dirn * tLo)
        val vHi   = model (a, gh, s0 + dirn * tHi)
        if vHi < vBest then
            tBest = tHi; vBest = vHi
        if aa > 0.0 then
            val t = -bb / aa                                              // the unconstrained minimizer along the ray
            if tLo < t && t < tHi then
                val vt = model (a, gh, s0 + dirn * t)
                if vt < vBest then
                    tBest = t; vBest = vt
        (tBest, vBest)
    end minQuad1D

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the positive stride t at which the ray s0 + t dirn leaves the trust
     *  region || s || <= delta (s0 inside it).
     *  @param s0     the starting point of the ray (scaled)
     *  @param dirn   the direction of the ray (scaled)
     *  @param delta  the trust-region radius
     */
    private def trExit (s0: VectorD, dirn: VectorD, delta: Double): Double =
        val aa = dirn dot dirn
        val bb = 2.0 * (s0 dot dirn)
        val cc = (s0 dot s0) - delta * delta
        if aa <= 0.0 then Double.PositiveInfinity
        else (-bb + sqrt (max (bb * bb - 4.0 * aa * cc, 0.0))) / (2.0 * aa)
    end trExit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select the scaled step to take, given the damped LM step s_h: the step itself
     *  when it stays inside the box; otherwise the best, by the quadratic model, of
     *  (1) the step truncated to the strict interior, (2) the step reflected off the
     *  bound it hits and (3) the scaled steepest descent step, (2) and (3) limited to
     *  the trust region of radius || s_h || and to the strict interior.  Returns the
     *  step, its model value and a label for the trace.
     *  @param b   the current parameter vector (strictly inside the box)
     *  @param a   the scaled Hessian approximation A
     *  @param gh  the scaled gradient D g
     *  @param d   the Coleman-Li scaling sqrt (v)
     *  @param sh  the damped LM step in the scaled variables
     *  @param th  the fraction of the distance to a bound a step may cover
     */
    private def selectStep (b: VectorD, a: MatrixD, gh: VectorD, d: VectorD, sh: VectorD,
                            th: Double): (VectorD, Double, String) =
        val delta = sh.norm                                               // the LM step is the TR step of this radius
        val s     = d * sh                                                // the step in the original variables
        val (alpha, hits) = stepToBound (b, s)

        var bestPh  = sh                                                  // the LM step itself
        var bestVal = model (a, gh, sh)
        var label   = "lm"

        if alpha < 1.0 then                                               // the step leaves the box
            bestPh  = sh * (th * alpha)                                   // (1) truncated to the strict interior
            bestVal = model (a, gh, bestPh)
            label   = "truncated"

            val rh = sh.copy                                              // (2) reflected direction: flip the hit components
            for j <- 0 until n if hits(j) != 0.0 do rh(j) = -rh(j)
            val sbh = sh * alpha                                          // the point on the bound (scaled)
            val (alphaR, _) = stepToBound (b + s * alpha, d * rh)         // where the reflected ray hits the next bound
            val tTr     = trExit (sbh, rh, delta)                         // where it leaves the trust region
            val rStride = min (alphaR, tTr)
            if rStride > 0.0 then
                val tLo = (1.0 - th) * alpha / rStride                    // at least this far: strictly interior again
                val tHi = if alphaR <= tTr then th * alphaR else tTr
                if tLo <= tHi then
                    val (t, value) = minQuad1D (a, gh, sbh, rh, tLo, tHi)
                    if value < bestVal then
                        bestPh = sbh + rh * t; bestVal = value; label = "reflected"
        end if

        val agh   = -gh                                                   // (3) scaled steepest descent
        val agNrm = agh.norm
        if agNrm > 0.0 then
            val (alphaG, _) = stepToBound (b, d * agh)
            val tTrG = delta / agNrm
            val tHiG = if th * alphaG < tTrG then th * alphaG else tTrG
            val (tG, valueG) = minQuad1D (a, gh, new VectorD (n), agh, 0.0, tHiG)
            if valueG < bestVal then
                bestPh = agh * tG; bestVal = valueG; label = "gradient"
        (bestPh, bestVal, label)
    end selectStep

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Execute the bounded Levenberg-Marquardt loop with trust-region reflective
     *  steps and return the optimized parameter vector (projected onto the box).
     *  @param b0  the initial parameter guess (projected inside the box first)
     */
    override def optimize (b0: VectorD): VectorD =
        var b      = interior (b0)
        var lambda = lambdaInit
        var r      = residualFn (b)
        var sse    = r dot r                                              // sum of squared errors

        var iter   = 1
        var status = "maxIter"
        var done   = false
        nLM = 0; nTruncated = 0; nReflected = 0; nGradient = 0
        debug ("optimize", s"interior ($b0) = $b, sse = $sse")

        while iter <= maxIter && ! done do
            val jac = computeJacobian (b)                                 // m x n Jacobian of the residuals
            val g   = jac.transpose * r                                   // gradient of sse / 2
            val (v, dv) = clScaling (b, g)
            val gSc = g * v                                               // the scaled gradient D^2 g
            if gSc.normInf < gtol then
                status = "gtol"; done = true
            else
                val d  = v.sqrt                                           // the Coleman-Li scaling D
                val jh = jac *~ d                                         // J D
                val gh = g * d                                            // D g
                val a  = jh.transpose * jh                                // D J^T J D
                for j <- 0 until n do a(j, j) += g(j) * dv(j)             // + diag (g (*) dv) >= 0
                val th = max (theta, 1.0 - gSc.normInf)                   // closer to the bounds as the gradient vanishes

                var accepted = false
                while ! accepted && ! done && lambda < LAMBDA_MAX do
                    val aL = a.copy
                    for j <- 0 until n do aL(j, j) += lambda * max (a(j, j), 1e-4)    // Marquardt scaling, as in the parent
                    val sh = solveNormalEq (aL, -gh)
                    if sh == null || sh.exists (x => x.isNaN || x.isInfinite) then
                        flaw ("optimize", s"sh = $sh: ill-conditioned, stiffening the damping")
                        lambda *= FACTOR
                    else
                        val (ph, pred, label) = selectStep (b, a, gh, d, sh, th)
                        val bNew   = project (b + d * ph)                 // back to the original variables
                        val rNew   = residualFn (bNew)
                        val sseNew = rNew dot rNew
                        debug ("optimize", s"lambda = $lambda, $label step, sseNew = $sseNew, bNew = $bNew")

                        if ! sseNew.isNaN && ! sseNew.isInfinite && sseNew < sse then
                            val actual    = sse - sseNew                  // actual decrease of the sse
                            val predicted = -2.0 * pred                   // the model's decrease (model is of sse / 2)
                            val dx        = (bNew - b).norm
                            b = bNew; r = rNew; sse = sseNew
                            lambda  /= FACTOR
                            accepted = true
                            label match
                            case "lm"        => nLM += 1
                            case "truncated" => nTruncated += 1
                            case "reflected" => nReflected += 1
                            case _           => nGradient += 1
                            if dx < eps then
                                status = "xtol"; done = true
                            if actual < ftol * sse && predicted < ftol * sse then
                                status = "ftol"; done = true
                        else
                            lambda *= FACTOR                              // no improvement: stiffen the damping
                end while
                if ! accepted && ! done then
                    status = "lambda"; done = true                        // no acceptable step at any damping
            end if
            debug ("optimize", s"iter = $iter, sse = $sse, b = $b")
            iter += 1
        end while
        debug ("optimize", s"stopped on $status after ${iter - 1} iterations: sse = $sse, b = $b, steps = $stepCounts")
        project (b)
    end optimize

end LevenbergMarquardt_TR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `levenbergMarquardt_TRTest` main method tests the `LevenbergMarquardt_TR` class
 *  on the bounded problem of `levenbergMarquardt_BTest` (b0 capped at 1.45 while the
 *  unconstrained optimum has b0 = 1.5) and compares it with `LevenbergMarquardt_B`.
 *  With b0 on its bound the constrained optimum is b1 = 0.814213, b2 = -0.382453 with
 *  SSE = 1.6423e-4 (the two-parameter problem solved to machine precision); the
 *  projected LM stops at SSE = 1.8228e-4 after about 70 iterations.
 *  > runMain scalation.optimization.nonlinear_ls.levenbergMarquardt_TRTest
 */
@main def levenbergMarquardt_TRTest (): Unit =

    val m = 30
    val n = 3

    val x = new MatrixD (m, 2)
    for i <- 0 until m do
        x(i, 0) = 1.0 + (i * 0.1)
        x(i, 1) = 5.0 - (i * 0.1)

    val x1 = x(?, 0)
    val x2 = x(?, 1)

    val beta  = VectorD (1.5, 0.8, -0.4)
    val beta0 = VectorD (1.0, 0.2, 0.0)

    val y = new VectorD (m)
    for i <- 0 until m do y(i) = beta(0) * math.pow (x1(i), beta(1)) * math.pow (x2(i), beta(2))

    val residualFunc = (b: VectorD) =>
        y - (x1.map (math.pow (_, b(1))) * x2.map (math.pow (_, b(2))) * b(0))

    val analyticalJacobian = (b: VectorD) =>
        val p1 = x1.map (math.pow (_, b(1)))
        val p2 = x2.map (math.pow (_, b(2)))
        val baseGroup = p1 * p2
        val col0 = -baseGroup
        val col1 = x1.map (math.log) * (-b(0)) * baseGroup
        val col2 = x2.map (math.log) * (-b(0)) * baseGroup
        MatrixD (col0, col1, col2).transpose

    val lower = VectorD (0.0,  0.0, -1.0)
    val upper = VectorD (1.45, 2.0,  1.0)                                 // b0 capped short of its real 1.5

    banner ("PROJECTED LM (LevenbergMarquardt_B)")
    val lmb  = new LevenbergMarquardt_B (n, lower, upper, residualFunc, Some (analyticalJacobian))
    val solB = lmb.optimize (beta0)
    val rB   = residualFunc (solB)

    banner ("TRUST-REGION REFLECTIVE LM (LevenbergMarquardt_TR)")
    val lmt  = new LevenbergMarquardt_TR (n, lower, upper, residualFunc, Some (analyticalJacobian))
    val solT = lmt.optimize (beta0)
    val rT   = residualFunc (solT)

    banner ("BOUNDED LEVENBERG-MARQUARDT NLS OPTIMIZATION RESULTS")
    println (s"Enforced Parameter Upper Bounds:   $upper")
    println (s"Target Real Unconstrained Beta:    $beta")
    println (s"Constrained optimum (reference):   VectorD(1.45, 0.814213, -0.382453), sse = 1.6423e-4")
    println (s"LevenbergMarquardt_B  solution:    $solB, sse = ${rB dot rB}")
    println (s"LevenbergMarquardt_TR solution:    $solT, sse = ${rT dot rT}")

end levenbergMarquardt_TRTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `levenbergMarquardt_TRTest2` main method tests the `LevenbergMarquardt_TR` class
 *  with infinite bounds: with no finite bound the Coleman-Li scaling is the identity
 *  and every step is the parent's LM step, so it must reproduce the unconstrained
 *  solution of `LevenbergMarquardt`, beta = (1.5, 0.8, -0.4).
 *  > runMain scalation.optimization.nonlinear_ls.levenbergMarquardt_TRTest2
 */
@main def levenbergMarquardt_TRTest2 (): Unit =

    val m = 30
    val n = 3

    val x = new MatrixD (m, 2)
    for i <- 0 until m do
        x(i, 0) = 1.0 + (i * 0.1)
        x(i, 1) = 5.0 - (i * 0.1)

    val x1 = x(?, 0)
    val x2 = x(?, 1)

    val beta  = VectorD (1.5, 0.8, -0.4)
    val beta0 = VectorD (1.0, 0.2, 0.0)

    val y = new VectorD (m)
    for i <- 0 until m do y(i) = beta(0) * math.pow (x1(i), beta(1)) * math.pow (x2(i), beta(2))

    val residualFunc = (b: VectorD) =>
        y - (x1.map (math.pow (_, b(1))) * x2.map (math.pow (_, b(2))) * b(0))

    val analyticalJacobian = (b: VectorD) =>
        val p1 = x1.map (math.pow (_, b(1)))
        val p2 = x2.map (math.pow (_, b(2)))
        val baseGroup = p1 * p2
        val col0 = -baseGroup
        val col1 = x1.map (math.log) * (-b(0)) * baseGroup
        val col2 = x2.map (math.log) * (-b(0)) * baseGroup
        MatrixD (col0, col1, col2).transpose

    banner ("NO FINITE BOUNDS: must match the unconstrained LM")
    val noLower = VectorD.fill (n)(Double.NegativeInfinity)
    val noUpper = VectorD.fill (n)(Double.PositiveInfinity)
    val lm      = new LevenbergMarquardt (n, residualFunc, Some (analyticalJacobian))
    val lmt     = new LevenbergMarquardt_TR (n, noLower, noUpper, residualFunc, Some (analyticalJacobian))
    val solLM   = lm.optimize (beta0)
    val solTR   = lmt.optimize (beta0)
    println (s"Target Real Beta:                  $beta")
    println (s"LevenbergMarquardt    solution:    $solLM, sse = ${residualFunc (solLM).normSq}")
    println (s"LevenbergMarquardt_TR solution:    $solTR, sse = ${residualFunc (solTR).normSq}")
    println (s"LevenbergMarquardt_TR steps:       ${lmt.stepCounts}")
    assert ((solTR - beta).norm < 1e-6, s"unbounded TR must recover beta = $beta, got $solTR")
    assert (lmt.stepCounts("lm") > 0 && lmt.stepCounts("reflected") == 0 && lmt.stepCounts("truncated") == 0,
            s"with no bounds every step must be the LM step: ${lmt.stepCounts}")

end levenbergMarquardt_TRTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `levenbergMarquardt_TRTest3` main method tests the REFLECTION of
 *  `LevenbergMarquardt_TR`: two problems whose LM steps cross a bound that is
 *  active at the constrained optimum, so the step must be reflected off it rather
 *  than cut.  The step counts must show reflected steps and the solution must reach
 *  the constrained optimum; `LevenbergMarquardt_B` is run on the same problems for
 *  comparison (its projected steps stall far from the optimum here).
 *  (1) the power model of `levenbergMarquardt_BTest` with the lower bound b2 >= -0.3
 *      (true b2 = -0.4): constrained optimum (1.2404, 0.878215, -0.3), SSE = 5.2096e-3,
 *      about 6 of 11 steps reflected; the projected LM stalls near SSE = 0.95.
 *  (2) the exponential model y = b0 exp (b1 x) of `levenbergMarquardtTest` with the
 *      lower bound b1 >= -0.3 (true b1 = -0.5): constrained optimum (1.559615, -0.3),
 *      SSE = 1.089173, about 5 of 7 steps reflected; the projected LM stops at SSE = 1.21.
 *  > runMain scalation.optimization.nonlinear_ls.levenbergMarquardt_TRTest3
 */
@main def levenbergMarquardt_TRTest3 (): Unit =

    banner ("REFLECTION 1: power model, lower bound b2 >= -0.3 (active at the optimum)")

    val m = 30
    val n = 3

    val x = new MatrixD (m, 2)
    for i <- 0 until m do
        x(i, 0) = 1.0 + (i * 0.1)
        x(i, 1) = 5.0 - (i * 0.1)

    val x1 = x(?, 0)
    val x2 = x(?, 1)

    val beta  = VectorD (1.5, 0.8, -0.4)
    val beta0 = VectorD (1.0, 0.2, 0.0)

    val y = new VectorD (m)
    for i <- 0 until m do y(i) = beta(0) * math.pow (x1(i), beta(1)) * math.pow (x2(i), beta(2))

    val residualFunc = (b: VectorD) =>
        y - (x1.map (math.pow (_, b(1))) * x2.map (math.pow (_, b(2))) * b(0))

    val analyticalJacobian = (b: VectorD) =>
        val p1 = x1.map (math.pow (_, b(1)))
        val p2 = x2.map (math.pow (_, b(2)))
        val baseGroup = p1 * p2
        val col0 = -baseGroup
        val col1 = x1.map (math.log) * (-b(0)) * baseGroup
        val col2 = x2.map (math.log) * (-b(0)) * baseGroup
        MatrixD (col0, col1, col2).transpose

    val lower = VectorD (0.0, 0.0, -0.3)                                  // b2 may not go below -0.3
    val upper = VectorD (3.0, 2.0,  1.0)
    val sseRef = 5.209632e-3                                              // SSE at the constrained optimum

    val lmb  = new LevenbergMarquardt_B (n, lower, upper, residualFunc, Some (analyticalJacobian))
    val solB = lmb.optimize (beta0)
    val lmt  = new LevenbergMarquardt_TR (n, lower, upper, residualFunc, Some (analyticalJacobian))
    val solT = lmt.optimize (beta0)

    banner ("REFLECTION 1: results")
    println (s"Target Real Unconstrained Beta:    $beta")
    println (s"Constrained optimum (reference):   VectorD(1.2404, 0.878215, -0.3), sse = $sseRef")
    println (s"LevenbergMarquardt_B  solution:    $solB, sse = ${residualFunc (solB).normSq}")
    println (s"LevenbergMarquardt_TR solution:    $solT, sse = ${residualFunc (solT).normSq}")
    println (s"LevenbergMarquardt_TR steps:       ${lmt.stepCounts}")
    assert (lmt.stepCounts("reflected") > 0, s"no reflected step was taken: ${lmt.stepCounts}")
    assert (abs (residualFunc (solT).normSq - sseRef) < 1e-6, s"TR did not reach the constrained optimum: $solT")

    banner ("REFLECTION 2: exponential model y = b0 exp (b1 x), lower bound b1 >= -0.3")

    val xe = VectorD.range (0, 30).map (_ * 0.2)
    val ye = (xe * -0.5).exp * 2.0                                        // b0 = 2, b1 = -0.5 (noise-free)

    val residualExp = (b: VectorD) => ye - (xe * b(1)).exp * b(0)

    val jacobianExp = (b: VectorD) =>
        val expTerm = (xe * b(1)).exp
        MatrixD (-expTerm, xe * (-b(0)) * expTerm).transpose

    val lowerE = VectorD (0.0, -0.3)                                      // b1 may not go below -0.3
    val upperE = VectorD (5.0,  0.0)
    val b0E    = VectorD (1.0, -0.1)
    val sseRefE = 1.089173                                                // SSE at the constrained optimum

    val lmbE  = new LevenbergMarquardt_B (2, lowerE, upperE, residualExp, Some (jacobianExp))
    val solBE = lmbE.optimize (b0E)
    val lmtE  = new LevenbergMarquardt_TR (2, lowerE, upperE, residualExp, Some (jacobianExp))
    val solTE = lmtE.optimize (b0E)

    banner ("REFLECTION 2: results")
    println (s"Target Real Unconstrained Beta:    VectorD(2.0, -0.5)")
    println (s"Constrained optimum (reference):   VectorD(1.559615, -0.3), sse = $sseRefE")
    println (s"LevenbergMarquardt_B  solution:    $solBE, sse = ${residualExp (solBE).normSq}")
    println (s"LevenbergMarquardt_TR solution:    $solTE, sse = ${residualExp (solTE).normSq}")
    println (s"LevenbergMarquardt_TR steps:       ${lmtE.stepCounts}")
    assert (lmtE.stepCounts("reflected") > 0, s"no reflected step was taken: ${lmtE.stepCounts}")
    assert (abs (residualExp (solTE).normSq - sseRefE) < 1e-5, s"TR did not reach the constrained optimum: $solTE")

end levenbergMarquardt_TRTest3

