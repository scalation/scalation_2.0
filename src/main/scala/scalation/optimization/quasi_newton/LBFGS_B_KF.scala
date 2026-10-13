
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, John Miller, Nirupom Bose Roy, Lokesh Adusumilli
 *  @version 2.0
 *  @date    Wed May 27 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Analytic-gradient L-BFGS-B for ARMA KF-MLE (dedicated ARMA_KF optimizer)
 *
 *  Cloned from `scalation.optimization.quasi_newton.LBFGS_B` and patched
 *  to correct six optimizer defects that cause the KF-MLE to converge to
 *  degenerate solutions (near-unit-root AR, θ = 0, μ < 0):
 *
 *  Fix 1  countMax   10  → 200   stops after 10 stagnant-mgn iters, long
 *                                  before MA params are updated from their
 *                                  warm-start of zero.
 *
 *  Fix 2  gtol       1e-2 → 0.9  More-Thuente curvature condition far too
 *                                  tight; rejects almost every step and wastes
 *                                  the line-search budget on μ, starving MA.
 *
 *  Fix 3  maxStep    1.0  → 1e15 caps step at 1.0, blocking large μ moves
 *                                  (COVID deaths ~ 5 000+).
 *
 *  Fix 4  maxLineSearch 20 → 100 too few inner trials; search exits before
 *                                  a Wolfe-acceptable step is found.
 *
 *  Fix 5  hs         5   → 10    L-BFGS curvature memory too small for
 *                                  p = 5 problems (need ≥ p pairs).
 *
 *  Fix 6  h (gradient step)
 *         fixed 1e-6 → adaptive  Differential.∇ uses a hardcoded h = 1e-6.
 *         central-diff               For μ ≈ 5 000 this loses ~6 significant
 *                                  digits: f(5000+1e-6) − f(5000−1e-6) ≈ 0.
 *                                  Replaced by  h_k = max(|x_k|, 1) × ∛ε
 *                                  (≈ 0.03 for μ = 5 000, still 6e-6 for
 *                                  typical φ ∈ (−1,1)).
 *
 *  Dedicated ARMA_KF optimizer; does NOT modify the shared quasi_newton.LBFGS_B
 *  that other models rely on.
 *
 *  > runMain scalation.modeling.forecasting.lBFGS_B_KFTest
 */

package scalation
package optimization
package quasi_newton

import scala.collection.mutable.ArrayBuffer
import scala.math.{abs, cbrt, max, min}
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._
import MatrixD.eye

//import scalation.optimization.{Minimizer, better, blown, FuncVec}
//import scalation.optimization.quasi_newton.{
//    FunctionEvaluation, LBFGSCallbackData,
//    LBFGSLineSearchPrms, LBFGSMoreThuente,
//    LBFGSLineSearchStep, LBFGSLineSearchFailure
//}

/** Local alias for box-constraint tuple so we stay independent of the
 *  quasi_newton package's `Bounds` type alias.
 */
type KFBounds = (VectorD, VectorD)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGS_B_KF` companion object — factory and bounds helpers.
 */
object LBFGS_B_KF:

    val emptyMatrix = new MatrixD (0, 0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `LBFGS_B_KF` with explicit bounds.
     *  @param f     objective to minimise
     *  @param l_u_  optional box bounds; null → (−∞, +∞) in all dims
     *  @param gradF optional analytic gradient; null → adaptive central-diff
     */
    def apply (f: FunctionV2S, l_u_ : KFBounds = null,
               gradF: FunctionV2V = null): LBFGS_B_KF =
        new LBFGS_B_KF (f, l_u_, gradF)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build uniform box bounds (same lo / up in every dimension).
     *  @param n   number of dimensions
     *  @param lo  lower bound (scalar)
     *  @param up  upper bound (scalar)
     */
    inline def makeBounds (n: Int, lo: Double, up: Double): KFBounds =
        (VectorD.fill (n)(lo), VectorD.fill (n)(up))
    end makeBounds

end LBFGS_B_KF


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGS_B_KF` class — L-BFGS-B with all six fixes applied.
 *
 *  The public interface is identical to `LBFGS_B`: construct with `(f, l_u)`
 *  and call `solve(x0)`.
 *
 *  @param f      objective function to minimise
 *  @param l_u    box constraints; null → (−∞, +∞)
 *  @param gradF  analytic gradient (optional); null → adaptive central-diff
 */
class LBFGS_B_KF (f: FunctionV2S,
                  private var l_u: KFBounds = (null, null),
                  gradF: FunctionV2V = null)
      extends Minimizer:

    private val debug           = debugf ("LBFGS_B_KF", false)
    private var ww, mm: MatrixD = null
    private var theta           = 0.0
    private var dim             = 0

    // Fix 5: history size 10 (was 5)
    private var hs = 10

    // Fix 6: adaptive central-difference constants
    //   cbrt(machineEps) ≈ 6.055e-6  → h_k = max(|x_k|, 1) * 6.055e-6
    private val cbrtEps = cbrt (2.220446049250313e-16)   // ≈ 6.055e-6

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Adaptive central-difference gradient (Fix 6).
     *  h_k = max(|x_k|, 1.0) * ∛ε avoids cancellation for large parameters
     *  such as the process mean μ (e.g. μ ≈ 5 000 for COVID weekly deaths).
     *  @param fv  scalar function to differentiate
     *  @param x   evaluation point
     */
    private def adaptiveGrad (fv: FunctionV2S, x: VectorD): VectorD =
        val g = new VectorD (x.dim)
        var i = 0
        while i < x.dim do
            val hi = max (abs (x(i)), 1.0) * cbrtEps
            val xp = x.copy; xp(i) += hi
            val xm = x.copy; xm(i) -= hi
            g(i)   = (fv (xp) - fv (xm)) / (2.0 * hi)
            i += 1
        end while
        g
    end adaptiveGrad

    /** Single dispatch: analytic if provided, else adaptive numerical. */
    private inline def grad (x: VectorD): VectorD =
        if gradF != null then gradF (x) else adaptiveGrad (fg, x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // ── Internal L-BFGS-B geometry helpers (unchanged from LBFGS_B) ─────────
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    private def sortIndices (v: ArrayBuffer [(Int, Double)]): VectorI =
        val sv  = v.sortBy (_._2)
        val idx = new VectorI (sv.length)
        for i <- idx.indices do idx(i) = sv(i)._1
        idx
    end sortIndices

    private def forceBounds (v: VectorD): Unit =
        val (l, u) = l_u
        for i <- v.indices do
            if v(i) > u(i)      then v(i) = u(i)
            else if v(i) < l(i) then v(i) = l(i)
        end for
    end forceBounds

    def setHistorySize (hs_ : Int): Unit = hs = hs_

    private def getGCP (x: VectorD, gr: VectorD): KFBounds =
        debug ("getGCP", s"x = $x, gr = $gr")
        val (l, u) = l_u
        val setOfT = new ArrayBuffer [(Int, Double)] ()
        val d      = -gr
        for j <- 0 until dim do
            if gr(j) == 0 then setOfT.append ((j, MAX_VALUE))
            else
                val tmp = if gr(j) < 0 then (x(j) - u(j)) / gr(j)
                          else (x(j) - l(j)) / gr(j)
                setOfT.append ((j, tmp))
                if tmp == 0 then d(j) = 0
            end if
        end for
        val sortedIndices = sortIndices (setOfT)
        val xCauchy       = x.copy
        val p             = ww.transpose * d
        val c             = new VectorD (ww.dim2)
        var fPrime        = -d dot d
        var fDoublePrime  = max (-theta * fPrime - (p dot mm * p), EPSILON)
        val f_dp_orig     = fDoublePrime
        var dt_min        = -fPrime / fDoublePrime
        var t_old         = 0.0
        var i = 0
        breakable {
            for j <- 0 until dim do
                i = j
                if setOfT (sortedIndices(j))._2 > 0 then break ()
            end for
        } // breakable
        var b  = sortedIndices (i)
        var t  = setOfT (b)._2
        var dt = t
        while dt_min >= dt && i < dim do
            xCauchy(b)   =  if d(b) > 0      then u(b)
                            else if d(b) < 0 then l(b)
                            else xCauchy(b)
            val zb        = xCauchy(b) - x(b)
            c            += p * dt
            val wbt       = ww(b)
            fPrime       += dt * fDoublePrime + gr(b)*gr(b) + theta*gr(b)*zb - gr(b)*(wbt dot mm*c)
            fDoublePrime += -theta*gr(b)*gr(b) - 2.0*(gr(b)*(wbt dot mm*p)) - gr(b)*gr(b)*(wbt dot mm*wbt)
            fDoublePrime  = max (EPSILON * f_dp_orig, fDoublePrime)
            p            += wbt * gr(b)
            d(b)          = 0
            dt_min        = -fPrime / fDoublePrime
            t_old         = t
            i            += 1
            if i < dim then
                b  = sortedIndices (i)
                t  = setOfT (b)._2
                dt = t - t_old
            end if
        end while
        dt_min = max (dt_min, 0.0)
        t_old += dt_min
        for ii <- i until xCauchy.dim do
            val si      = sortedIndices (ii)
            xCauchy(si) = x(si) + t_old * d(si)
        end for
        c += p * dt_min
        (xCauchy, c)
    end getGCP

    private def findAlpha (x_cp: VectorD, du: VectorD, freeVar: ArrayBuffer [Int]): Double =
        val (l, u)    = l_u
        var alphastar = 1.0
        val n         = freeVar.size
        assert (du.dim == n)
        for i <- 0 until n do
            val fi = freeVar(i)
            alphastar = if du(i) > 0 then min (alphastar, (u(fi) - x_cp(fi)) / du(i))
                        else min (alphastar, (l(fi) - x_cp(fi)) / du(i))
        end for
        alphastar
    end findAlpha

    private def subspaceMinimize (x: VectorD, gr: VectorD, xCauchy: VectorD, c: VectorD): VectorD =
        val (l, u)       = l_u
        val thetaInverse = 1.0 / theta
        val freeVarIdx   = new ArrayBuffer [Int] ()
        for i <- xCauchy.indices if xCauchy(i) != u(i) && xCauchy(i) != l(i) do freeVarIdx.append (i)
        val freeVarCount = freeVarIdx.size
        val wwzz         = new MatrixD (ww.dim2, freeVarCount)
        for i <- 0 until freeVarCount do wwzz(?, i) = ww(freeVarIdx(i))
        val rr           = (gr + (xCauchy - x) * theta - ww * (mm * c))
        val r            = new VectorD (freeVarCount)
        for i <- 0 until freeVarCount do r(i) = rr(freeVarIdx(i))
        var v  = mm * (wwzz * r)
        var nn = wwzz * wwzz.transpose * thetaInverse
        nn     = eye (nn.dim, nn.dim) - mm * nn
        val lu = new Fac_LU (nn)
        lu.factor ()
        v = lu.solve (v)
        val du          = r * -thetaInverse - wwzz.transpose * v * thetaInverse * thetaInverse
        val alpha_star  = findAlpha (xCauchy, du, freeVarIdx)
        val dStar       = du * alpha_star
        val subspaceMin = xCauchy.copy
        for i <- 0 until freeVarCount do subspaceMin (freeVarIdx(i)) += dStar(i)
        subspaceMin
    end subspaceMinimize

    private def getMgn (x: VectorD, gr: VectorD): Double =
        val (l, u)     = l_u
        val x_gr       = x - gr
        val checkLower = VectorD (for i <- l.indices yield max (x_gr(i)      , l(i)))
        val checkUpper = VectorD (for i <- u.indices yield min (checkLower(i), u(i)))
        (checkUpper - x).normSq / dim
    end getMgn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // ── Minimizer trait obligations ──────────────────────────────────────────
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    override def fg (x: VectorD): Double = f(x)

    override def lineSearch (x: VectorD, dir: VectorD, step: Double): Double =
        val fv = fg (x)
        val gr = grad (x)                                         // Fix 6
        val (_, _, _, rate) = lineSearchMT (x, fv, gr, dir, step)
        rate
    end lineSearch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** More-Thuente line search with Fixes 2, 3, 4, 6.
     *  @param x          current point
     *  @param fv         f(x)
     *  @param gr         ∇f(x)  (already computed with adaptive step)
     *  @param dir        search direction (subspaceMin − x)
     *  @param alphaInit  initial step size
     */
    private def lineSearchMT (x: VectorD, fv: Double,
                              gr: VectorD, dir: VectorD,
                              alphaInit: Double): (VectorD, VectorD, Double, Double) =

        val dirNorm = dir.norm
        if dirNorm <= EPSILON then return (x, gr, fv, 0.0)

        // Fix 6: provide adaptive gradient to More-Thuente evaluator
        val adaptGrad: FunctionV2V = (z: VectorD) => adaptiveGrad (fg, z)
        val evalLogic = FunctionEvaluation ((z: VectorD) => fg (z), adaptGrad)
        val cd        = LBFGSCallbackData (dim, None, evalLogic)

        val initStep = max (1.0e-12, min (alphaInit, 1.0))

        // Fixes 2, 3, 4 ───────────────────────────────────────────────────────
        val lsPrms = LBFGSLineSearchPrms (
            defaultStep   = initStep,
            minStep       = 1.0e-15,
            maxStep       = 1.0e15,    // Fix 3: was 1.0
            ftol          = 1.0e-4,
            gtol          = 0.9,       // Fix 2: was 1e-2
            xtol          = 1.0e-15,
            maxLineSearch = 100)       // Fix 4: was 20
        // ─────────────────────────────────────────────────────────────────────

        LBFGSMoreThuente.lineSearch (dim, x, fv, gr, dir, initStep, cd, lsPrms) match
            case stepRes: LBFGSLineSearchStep =>
                (stepRes.x, stepRes.g, stepRes.fx, stepRes.step)
            case failRes: LBFGSLineSearchFailure =>
                val inc   = failRes.bestIncompleteResults
                val xInc  = inc.variableValues
                val fInc  = inc.functionValue
                val gInc  = adaptiveGrad (fg, xInc)              // Fix 6: was ∇(fg)(xInc)
                val alpha = ((xInc - x) dot dir) / max (dir dot dir, EPSILON)
                (xInc, gInc, fInc, alpha)
    end lineSearchMT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve  min { f(x) | l ≤ x ≤ u }  using L-BFGS-B with all fixes applied.
     *  @param x0        starting point
     *  @param alphaInit initial line-search step
     *  @param toler     convergence tolerance
     */
    def solve (x0: VectorD, alphaInit: Double = STEP, toler: Double = EPSILON): FuncVec =
        debug ("solve", s"x0 = $x0")

        var best = (MAX_VALUE, VectorD.nullv)

        dim   = x0.size
        theta = 1.0

        if l_u == null || l_u._1 == null || l_u._2 == null then
            l_u = LBFGS_B_KF.makeBounds (dim, NEGATIVE_INFINITY, POSITIVE_INFINITY)

        ww = new MatrixD (dim, 0)
        mm = LBFGS_B_KF.emptyMatrix

        val yHistory = new ArrayBuffer [VectorD] ()
        val sHistory = new ArrayBuffer [VectorD] ()
        var yHistoryMx: MatrixD = null
        var sHistoryMx: MatrixD = null

        var (x, gr) = (x0, grad (x0))                            // Fix 6: adaptive gradient
        var fv      = fg (x)
        var mgn     = 0.0
        var count   = 0
        val countMax = 200                                        // Fix 1: was 10

        breakable {
            for _ <- 1 to MAX_IT do
                val f_old   = fv
                val x_old   = x
                val g_old   = gr
                val mgn_old = mgn

                val (xCauchy, c) = getGCP (x, gr)
                forceBounds (xCauchy)

                val subspaceMin = subspaceMinimize (x, gr, xCauchy, c)
                forceBounds (subspaceMin)

                val dir = subspaceMin - x
                val (xLs, gLs, fLs, _) = lineSearchMT (x, fv, gr, dir, alphaInit)

                x  = xLs
                gr = gLs
                fv = fLs
                forceBounds (x)

                if blown ((fv, x)) then { best = better ((f_old, x_old), best); break () }

                mgn = getMgn (x, gr)
                if mgn < toler || count > countMax then { best = better ((fv, x), best); break () }
                if abs (mgn - mgn_old) < toler then count += 1

                val newY = gr - g_old
                val newS = x - x_old

                val test = abs (newS dot newY)
                if test > EPSILON * newY.normSq then
                    if yHistory.size >= hs then { yHistory.remove (0); sHistory.remove (0) }
                    yHistory append newY
                    sHistory append newS
                    theta = (newY dot newY) / (newY dot newS)
                    yHistoryMx = MatrixD (yHistory).transpose
                    sHistoryMx = MatrixD (sHistory).transpose
                    ww         = yHistoryMx ++^ (sHistoryMx * theta)
                    val aa     = sHistoryMx.transpose * yHistoryMx
                    val ll     = aa.lower
                    ll(?, ?)   = 0.0
                    val dd     = new MatrixD (aa.dim, aa.dim2)
                    dd.setDiag (-aa(?))
                    val mm2    = (dd ++^ ll.transpose) ++ (ll ++^ (sHistoryMx.transpose * sHistoryMx * theta))
                    mm         = Fac_LU.inverse (mm2)()
                end if

                best = better ((fv, x), best)
                if abs (f_old - fv) < toler then break ()
            end for
        } // breakable
        banner (s"solve: optimal solution = $best")
        best
    end solve

end LBFGS_B_KF


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_B_KFTest` main function sanity-checks the fixed optimizer on a
 *  2-D quadratic so we confirm basic convergence is intact after all patches.
 *  > runMain scalation.modeling.forecasting.lBFGS_B_KFTest
 */
@main def lBFGS_B_KFTest (): Unit =

    banner ("LBFGS_B_KF — sanity tests (unconstrained + bounded + large-scale μ)")

    // Test 1: unconstrained quadratic — global min at (3,4), f*=1
    def f1 (x: VectorD): Double = (x(0) - 3.0)~^2 + (x(1) - 4.0)~^2 + 1.0
    val opt1 = LBFGS_B_KF (f1)
    val (fOpt1, xOpt1) = opt1.solve (new VectorD (2))
    println (s"Test 1 [unconstrained]:   f*=$fOpt1  x*=$xOpt1  (expect f*≈1, x*≈[3,4])")
    assert (math.abs (fOpt1 - 1.0) < 1e-5, s"f* should be 1.0, got $fOpt1")

    // Test 2: box-constrained — optimum clipped to x0 ∈ [3.5,5], f*=1.25
    val lu = LBFGS_B_KF.makeBounds (2, 3.5, 5.0)
    val opt2 = LBFGS_B_KF (f1, l_u_ = lu)
    val (fOpt2, xOpt2) = opt2.solve (new VectorD (2))
    println (s"Test 2 [bounded]:         f*=$fOpt2  x*=$xOpt2  (expect x0≈3.5, x1≈4)")
    assert (math.abs (xOpt2(0) - 3.5) < 1e-3, s"x0* should be ≈3.5 (bound), got ${xOpt2(0)}")

    // Test 3: large-scale μ — mimics ARMA_KF where one param is ~ 5000
    //   f(μ,φ) = (μ-5000)²/1e6 + (φ-0.7)²  →  min at (5000, 0.7), f*=0
    def f3 (x: VectorD): Double =
        (x(0) - 5000.0)~^2 / 1.0e6 + (x(1) - 0.7)~^2
    val x0_3 = VectorD (4800.0, 0.3)
    val opt3 = LBFGS_B_KF (f3)
    val (fOpt3, xOpt3) = opt3.solve (x0_3)
    println (s"Test 3 [large-scale μ]:   f*=$fOpt3  x*=$xOpt3  (expect f*≈0, x*≈[5000,0.7])")
    assert (math.abs (xOpt3(0) - 5000.0) < 1.0,  s"μ* should be ≈5000, got ${xOpt3(0)}")
    assert (math.abs (xOpt3(1) - 0.7)    < 1e-4,  s"φ* should be ≈0.7,  got ${xOpt3(1)}")

    println ("\nAll LBFGS_B_KF sanity tests passed.")

end lBFGS_B_KFTest

