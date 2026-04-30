
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, Nirupom Bose Roy
 *  @version 2.0
 *  @date    Fri Oct 7 12:27:00 EDT 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Limited memory BFGS with Bounds (L-BFGS-B)
 *
 *------------------------------------------------------------------------------
 *  Limited memory Broyden–Fletcher–Goldfarb–Shanno (BFGS) for Bound constrained
 *  optimization (L-BFGS-B) algorithm.  Originally proposed by Byrd et al. in 1995.
 *  See the first two links for the original paper and authors' software (written
 *  in Fortran) distribution site, respectively.  This implementation is translated
 *  from a C++ implementation found in the last link.
 *
 *  @see www.ece.northwestern.edu/~nocedal/PSfiles/limited.ps.gz
 *  @see users.iems.northwestern.edu/~nocedal/lbfgsb.html
 *  @see github.com/PatWie/CppNumericalSolvers/blob/master/include/cppoptlib/solver/lbfgsbsolver.h
 */

package scalation
package optimization
package quasi_newton

import scala.collection.mutable.ArrayBuffer
import scala.math.{abs, max, min}
import scala.util.control.Breaks.{break, breakable}

import scalation.calculus.Differential.∇
import scalation.mathstat._
import scalation.optimization.quasi_newton.LBFGS_B_TestUtil._
import scalation.optimization.functions._

import MatrixD.eye

type Bounds = (VectorD, VectorD)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGS_B` companion object provides a factory method for Limited memory
 *  Broyden–Fletcher–Goldfarb–Shanno for Bounds constrained optimization.
 */
object LBFGS_B:

    val emptyMatrix = new MatrixD (0, 0)                          // empty zero dimension matrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `LBFGS_B` object with a given dimensionality and default lower
     *  and upper bounds of -1 and 1, respectively.
     *  @param f        the objective function to be minimized
     *  @param n        the dimensionality of the search space
     *  @param g        the constraint function to be satisfied, if any
     *  @param ineq     whether the constraint is treated as inequality (default) or equality
     *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
     *                            or inexact (e.g., `WolfeLS`) Line Search
     *  @param l_u      (vector, vector) of lower and upper bounds for all input parameters
     */
    def apply (f: FunctionV2S, n: Int, g: FunctionV2S = null,
               ineq: Boolean = true, exactLS: Boolean = false,
               l_u_ : Bounds = null): LBFGS_B =
    
        val l_u = if l_u_ == null then (VectorD.fill (n)(-1), VectorD.fill (n)(1))
                  else l_u_
        new LBFGS_B (f, g, ineq, exactLS, l_u)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make simple bounds where the limits in each dimension is the same.
     *  @param n   the dimensionality of the search space
     *  @param lo  scalar lower bounds for all input parameters
     *  @param up  scalar upper bounds for all input parameters
     */
    inline def makeBounds (n: Int, lo: Double, up: Double): Bounds =
        (VectorD.fill (n)(lo),
         VectorD.fill (n)(up))
    end makeBounds

end LBFGS_B

import LBFGS_B.makeBounds

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGS_B` the class implements the Limited memory Broyden–Fletcher–
 *  Goldfarb–Shanno for Bounds constrained optimization (L-BFGS-B)
 *  Quasi-Newton Algorithm for solving Non-Linear Programming (NLP) problems.
 *  L-BFGS-B determines a search direction by  deflecting the steepest descent direction
 *  vector (opposite the gradient) by *  multiplying it by a matrix that approximates
 *  the inverse Hessian. Furthermore, only a few vectors represent the approximation
 *  of the Hessian Matrix (limited memory). The parameters estimated are also bounded
 *  within user specified lower and upper bounds.
 *
 *  minimize    f(x)
 *  subject to  g(x) <= 0   [ optionally g(x) == 0 ]
 *
 *  @param f        the objective function to be minimized
 *  @param g        the constraint function to be satisfied, if any
 *  @param ineq     whether the constraint is treated as inequality (default) or equality
 *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
 *                            or inexact (e.g., `WolfeLS`) Line Search
 *  @param l_u      (vector, vector) of lower and upper bounds for all input parameters
 */
class LBFGS_B (f: FunctionV2S, g: FunctionV2S = null,
               ineq: Boolean = true, exactLS: Boolean = false,
               private var l_u: Bounds = (null, null))
      extends Minimizer:

    private val debug           = debugf ("L-BFGS", false)        // debug function
    private val WEIGHT          = 1000.0                          // weight on penalty for constraint violation
    private var ww, mm: MatrixD = null                            // workspace matrices
    private var theta           = 0.0                             // a scaling parameter
    private var dim             = 0                               // dimension of the input vector
    private var hs              = 5                               // history size, number of historical vectors to store

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort pairs (k, v) according to v into ascending order.
     *  @param v  ArrayBuffer of Tuple2 to be sorted by the 2nd element
     */
    private def sortIndices (v: ArrayBuffer [(Int, Double)]): VectorI =
        val sv  = v.sortBy (_._2)
        val idx = new VectorI (sv.length)
        for i <- idx.indices do idx(i) = sv(i)._1
        idx
    end sortIndices

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Algorithm CP: Computation of the Generalized Cauchy Point. See page 8 of
     *  @see www.ece.northwestern.edu/~nocedal/PSfiles/limited.ps.gvz
     *  @param x   the parameter vector
     *  @param gr  the gradient vector
     */
    private def getGCP (x: VectorD, gr: VectorD): Bounds =
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
            fPrime       += dt * fDoublePrime + gr(b) * gr(b) + theta * gr(b) * zb - gr(b) * (wbt dot mm * c)
            fDoublePrime += -theta  *  gr(b) *  gr(b) - 2.0   * (gr(b) * (wbt dot mm * p))
                            - gr(b) *  gr(b) * (wbt dot mm * wbt)
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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the alpha* parameter, a positive scalar.  See Equation 5.8 on page 11 of
     *  @see www.ece.northwestern.edu/~nocedal/PSfiles/limited.ps.gvz
     *  @param x_cp     vector of cauchy point
     *  @param du       vector containing intermediate results used to find alpha*
     *  @param freeVar  an ArrayBuffer storing the indices of free variable
     */
    private def findAlpha (x_cp: VectorD, du: VectorD, freeVar: ArrayBuffer [Int]): Double =
        debug ("findAlpha", s"x_cp = $x_cp, du = $du, freeVar = $freeVar")

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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Minimization of the subspace of free variables.  See Section 5 on page 9 of
     *  @see www.ece.northwestern.edu/~nocedal/PSfiles/limited.ps.gvz
     *  @param x        the parameter vector
     *  @param gr       the gradient vector
     *  @param xCauchy  the vector of Cauchy points
     *  @param c        vector obtained from getGCP used to initialize the subspace
     *                  minimization process
     */
    private def subspaceMinimize (x: VectorD, gr: VectorD, xCauchy: VectorD, c: VectorD): VectorD =
        debug ("subspaceMinimize", s"x = $x, gr = $gr, xCauchy = $xCauchy, c = $c")

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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Obtain the mean gradient norm squared
     *  @param x   the parameter vector
     *  @param gr  the gradient vector
     */
    private def getMgn (x: VectorD, gr: VectorD): Double =
        val (l, u)     = l_u
        val x_gr       = x - gr
        val checkLower = VectorD (for i <- l.indices yield max (x_gr(i)      , l(i)))
        val checkUpper = VectorD (for i <- u.indices yield min (checkLower(i), u(i)))
        val mgn        = (checkUpper - x).normSq / dim
        mgn
    end getMgn

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Force the values within 'v' to stay within the pre-defined bounds.
     *  @param v  the Vector containing values to be adjusted
     */
    private def forceBounds (v: VectorD): Unit =
        val (l, u) = l_u
        for i <- v.indices do
            if v(i) > u(i)      then v(i) = u(i)
            else if v(i) < l(i) then v(i) = l(i)
        end for
    end forceBounds

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Modify the number of historical vectors to store.
     *  @param hs_  the new history size
     */
    def setHistorySize (hs_ : Int): Unit = { hs = hs_ }

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The objective function f plus a weighted penalty based on the constraint
     *  function g.
     *  @param x  the coordinate values of the current point
     */
    override def fg (x: VectorD): Double =
        val f_x = f(x)
        if g == null then                             // unconstrained
            f_x
        else                                          // constrained, g(x) <= 0
            val penalty = if ineq then max (g(x), 0.0) else abs (g(x))
            f_x + abs (f_x) * WEIGHT * penalty * penalty
        end if
    end fg

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform a line search from point `x` along direction `dir`, returning the
     *  accepted step length.
     *  This method satisfies the `Minimizer` trait contract.  For exact scalar
     *  search requests it delegates to `lineSearch1D`; otherwise it delegates to
     *  the native More-Thuente bounded line search and returns only the accepted
     *  step length.
     *  @param x     the current point
     *  @param dir   the search direction
     *  @param step  the initial step size
     */
    override def lineSearch (x: VectorD, dir: VectorD, step: Double): Double =
        if exactLS then lineSearch1D (x, dir, step)
        else
            val fv = fg(x)
            val gr = ∇(fg)(x)
            val (_, _, _, rate) = lineSearchMT (x, fv, gr, dir, step)
            rate
    end lineSearch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform a More-Thuente line search along the feasible segment defined by
     *  the current point `x` and search direction `dir`.
     *  For L-BFGS-B, `dir` is typically the vector from the current point to the
     *  subspace minimizer, so the feasible step length is restricted to the
     *  interval `[0, 1]`.
     *  Returns the accepted point, gradient, objective value, and step length.
     *  If the line search fails but provides an incomplete best point, that point
     *  is used as a conservative fallback.
     *  @param x          the current point
     *  @param fv         the objective value at `x`
     *  @param gr         the gradient at `x`
     *  @param dir        the search direction
     *  @param alphaInit  the initial step size
     */
    private def lineSearchMT (x: VectorD, fv: Double,
                              gr: VectorD, dir: VectorD,
                              alphaInit: Double): (VectorD, VectorD, Double, Double) =

        debug ("lineSearchMT", s"x = $x, fv = $fv, gr = $gr, dir = $dir, alphaInit = $alphaInit")

        val dirNorm = dir.norm
        if dirNorm <= EPSILON then return (x, gr, fv, 0.0)

        // Reuse the native L-BFGS callback/evaluation pathway so that the bounded
        // solver uses the same More-Thuente implementation as the unconstrained one.
        val evalLogic = FunctionEvaluation ((z: VectorD) => fg (z))
        val cd        = LBFGSCallbackData (dim, None, evalLogic)

        // Restrict search to the feasible convex segment x + α dir, α in [0, 1].
        val initStep = max (1.0e-12, min (alphaInit, 1.0))
        val lsPrms = LBFGSLineSearchPrms (defaultStep   = initStep,
                                          minStep       = 1.0e-15,
                                          maxStep       = 1.0,
                                          ftol          = 1.0e-4,
                                          gtol          = 1.0e-2,
                                          xtol          = 1.0e-15,
                                          maxLineSearch = 20)

        LBFGSMoreThuente.lineSearch (dim, x, fv, gr, dir, initStep, cd, lsPrms) match
            case stepRes: LBFGSLineSearchStep =>
                (stepRes.x, stepRes.g, stepRes.fx, stepRes.step)

            case failRes: LBFGSLineSearchFailure =>
                val inc   = failRes.bestIncompleteResults
                val xInc  = inc.variableValues
                val fInc  = inc.functionValue
                val gInc  = ∇ (fg)(xInc)
                val alpha = ((xInc - x) dot dir) / max (dir dot dir, EPSILON)
                (xInc, gInc, fInc, alpha)
    end lineSearchMT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform a scalar 1D line search along the direction `dir` from point `x`.
     *  This helper is used only when exact scalar line search is explicitly
     *  requested.  Otherwise, the bounded solver uses the native More-Thuente
     *  line search via `lineSearchMT`.
     *  @param x     the current point
     *  @param dir   the search direction
     *  @param step  the initial step size (may use STEP as default)
     */
    private def lineSearch1D (x: VectorD, dir: VectorD, step: Double): Double =
        debug ("linesearch", s"x = $x, dir = $dir, step = $step")

        def f_1D (z: Double): Double = fg(x + dir * z)          // create a 1D function
        val ls = if exactLS then new GoldenSectionLS (f_1D )    // Golden Section Line Search
                 else new WolfeLS (f_1D)                        // Wolfe line search ((c1 = .0001, c2 = .9)
        ls.search (step)                                        // perform a Line Search
    end lineSearch1D

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the bound-constrained nonlinear optimization problem
     *  min { f(x) | g(x) <= 0 }
     *  using the L-BFGS-B algorithm.
     *  Notes:
     *  - If no bounds are provided, this method installs default unbounded
     *    box constraints `(-∞, +∞)` in every dimension.
     *  - The internal search step is computed along the feasible segment from
     *    the current point `x` to the subspace minimizer.
     *  @param x0        the starting point
     *  @param alphaInit the initial line-search step size
     *  @param toler     the convergence tolerance
     */
    def solve (x0: VectorD, alphaInit: Double = STEP, toler: Double = EPSILON): FuncVec =
        debug ("solve", s"x0 = $x0, alphaInit = $alphaInit, toler = $toler")

        var best = (MAX_VALUE, VectorD.nullv)

        dim   = x0.size
        theta = 1.0

        // Install default unbounded box constraints when bounds were not supplied.
        // Note: `l_u` may be the tuple `(null, null)` rather than `null`, so both
        // tuple members must be checked explicitly.
        val needDefaultBounds =
            l_u == null || l_u._1 == null || l_u._2 == null
        if needDefaultBounds then
            l_u = makeBounds(dim, NEGATIVE_INFINITY, POSITIVE_INFINITY)

//      val (l, u) = l_u

        ww = new MatrixD (dim, 0)                               // FIX - causes empty matrix warning
//      mm = new MatrixD (0, 0)                                 // find alt. to zero dimension matrix
        mm = LBFGS_B.emptyMatrix

        val yHistory = new ArrayBuffer [VectorD] ()
        val sHistory = new ArrayBuffer [VectorD] ()
        var yHistoryMx: MatrixD = null
        var sHistoryMx: MatrixD = null

        var (x, gr)  = (x0, ∇ (fg)(x0))
        var fv       = fg(x)
        var mgn      = 0.0
        var count    = 0
        val countMax = 10

        breakable {
            for k <- 1 to MAX_IT do
//              banner (s"solve: iteration $k: f(x) = $fv, x = $x")
                val f_old   = fv
                val x_old   = x
                val g_old   = gr
                val mgn_old = mgn
    
                // STEP 2: compute the cauchy point
                val (xCauchy, c) = getGCP (x, gr)
                forceBounds (xCauchy)

                // STEP 3: compute a search direction d_k by the primal method for the sub-problem
                val subspaceMin = subspaceMinimize (x, gr, xCauchy, c)
                forceBounds (subspaceMin)

                // STEP 4: perform line search along the feasible segment from
                // the current point to the subspace minimizer.
                val dir = subspaceMin - x
                val (xLs, gLs, fLs, _) =                                // _ for rate
                    if exactLS then
                        val r = lineSearch1D (x, dir, alphaInit)
                        val xNew = x + dir * r
                        forceBounds (xNew)
                        val fNew = fg (xNew)
                        val gNew = ∇ (fg)(xNew)
                        (xNew, gNew, fNew, r)
                    else
                        lineSearchMT (x, fv, gr, dir, alphaInit)

                // STEP 5: accept line-search result
                x  = xLs
                gr = gLs
                fv = fLs
                forceBounds (x)   // safeguard only; should already be feasible if 0 <= rate <= 1

                if blown ((fv, x)) then { best = better ((f_old, x_old), best); break () }

                mgn = getMgn (x, gr)
                if mgn < toler || count > countMax then { best = better ((fv, x), best); break () }
                if abs (mgn - mgn_old) < toler then count += 1

                val newY = gr - g_old                                       // prepare for next iteration
                val newS = x - x_old

                // STEP 6: update limited-memory history if the curvature condition holds.
                val test = abs (newS dot newY)
                if test > EPSILON * newY.normSq then
                    if yHistory.size >= hs then { yHistory.remove (0); sHistory.remove (0) }
                    yHistory append newY
                    sHistory append newS
                    theta = (newY dot newY) / (newY dot newS)

                    // STEP 7: rebuild the compact L-BFGS-B matrices W and M.
                    yHistoryMx = MatrixD (yHistory).transpose
                    sHistoryMx = MatrixD (sHistory).transpose
                    ww         = yHistoryMx ++^ (sHistoryMx * theta)
                    val aa     = sHistoryMx.transpose * yHistoryMx
                    val ll     = aa.lower
                    ll(?, ?)   = 0.0                                       // set ll's diagonal to 0
                    val dd     = new MatrixD (aa.dim, aa.dim2)
                    dd.setDiag (-aa(?))                                    // set dd diagonal to aa's
                    val mm2    = (dd ++^ ll.transpose) ++ (ll ++^ (sHistoryMx.transpose * sHistoryMx * theta))
                    mm         = Fac_LU.inverse (mm2)()
                end if

                debug ("solve", s"(k = $k) move from $x_old to $x where fg(x) = $fv")

                best = better ((fv, x), best)
                if abs (f_old - fv) < toler then break ()                  // successive function values too similar
            end for
        } // breakable
        banner (s"solve: optimal solution = $best")
        best
    end solve

end LBFGS_B


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BTest` main function is used to test the `LBFGS_B` class.
 *      f(x) = (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest
 */
@main def lBFGS_BTest (): Unit =

    val n  = 2
    val x0 = new VectorD (n)
    def f (x: VectorD): Double = (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1

    banner ("Minimize (no bounds): (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    var optimizer = new LBFGS_B (f)
    var opt = optimizer.solve (x0)
    println (s"o][ optimal solution (x, f(x)) = $opt")

    banner ("Minimize (bounds): (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    val lu = makeBounds (x0.dim, 3.5, 5.0)
    optimizer = new LBFGS_B (f, l_u = lu)
    opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

end lBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BTest2` main function is used to test the `LBFGS_B` class.
 *      f(x) = x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest2
 */
@main def lBFGS_BTest2 (): Unit =

    val n  = 2
    val x0 = new VectorD (n)
    def f (x: VectorD): Double = x(0)~^4 + (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1

    banner ("Minimize (no bounds): x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    var optimizer = new LBFGS_B (f)
    var opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

    banner ("Minimize (bounds): x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    val lu = makeBounds (x0.dim, 3.5, 5.0)
    optimizer = new LBFGS_B (f, l_u = lu)
    opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

end lBFGS_BTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BTest3` main function is used to test the `LBFGS_B` class.
 *      f(x) = 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest3
 */
@main def lBFGS_BTest3 (): Unit =

    val n  = 2
    val x0 = VectorD (0.1, 0.0)
    def f (x: VectorD): Double = 1/x(0) + x(0)~^4 + (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1

    banner ("Minimize (no bounds): 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    var optimizer = new LBFGS_B (f)
    var opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

    opt = optimizer.resolve (n)
    println (s"][ optimal solution (x, f(x)) = $opt")

    banner ("Minimize (bounds): 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    val lu = makeBounds (x0.dim, 3.5, 5.0)
    optimizer = new LBFGS_B (f, l_u = lu)
    opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

end lBFGS_BTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BQuadraticInteriorTest` main function test L-BFGS-B on a simple
 *  convex quadratic with an interior optimum.
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BQuadraticInteriorTest
 */
@main def lBFGS_BQuadraticInteriorTest (): Unit =

    val x0 = VectorD (0.0, 0.0)
    def f (x: VectorD): Double = (x(0) - 3.0)~^2 + (x(1) - 4.0)~^2 + 1.0

    runLBFGSB (name    = "Quadratic interior optimum",
               f       = f,
               x0      = x0,
               bounds  = makeBounds (2, -10.0, 10.0),
               exactLS = false)

    compareLS (name   = "Quadratic interior optimum",
               f      = f,
               x0     = x0,
               bounds = makeBounds (2, -10.0, 10.0))

end lBFGS_BQuadraticInteriorTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BQuadraticActiveBoundTest` main function tests L-BFGS-B on a convex
 *  quadratic whose optimum is clipped by active bounds.
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BQuadraticActiveBoundTest
 */
@main def lBFGS_BQuadraticActiveBoundTest (): Unit =

    val x0 = VectorD (0.0, 0.0)
    def f (x: VectorD): Double = (x(0) - 3.0)~^2 + (x(1) - 4.0)~^2 + 1.0

    runLBFGSB (name    = "Quadratic active-bound optimum",
               f       = f,
               x0      = x0,
               bounds  = (VectorD (-10.0, -10.0), VectorD (2.5, 3.5)),
               exactLS = false)

    compareLS (name   = "Quadratic active-bound optimum",
               f      = f,
               x0     = x0,
               bounds = (VectorD (-10.0, -10.0), VectorD (2.5, 3.5)))

end lBFGS_BQuadraticActiveBoundTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BInactiveBoundsVsLBFGSTes` main function compare L-BFGS-B to unconstrained
 *  L-BFGS when bounds are effectively inactive.
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BInactiveBoundsVsLBFGSTes
 */
@main def lBFGS_BInactiveBoundsVsLBFGSTest (): Unit =

    val x0 = VectorD (-4.0, 7.0)
    def f (x: VectorD): Double = (x(0) + 2.0 * x(1) - 7.0)~^2 +
      (2.0 * x(0) + x(1) - 5.0)~^2

    compareToLBFGS (
        name   = "Booth with inactive bounds",
        f      = f,
        x0     = x0,
        bounds = makeBounds (2, -100.0, 100.0)
    )

end lBFGS_BInactiveBoundsVsLBFGSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The boothFunctionLBFGS_BTest tests L-BFGS-B on the Booth function.
 *  > runMain scalation.optimization.quasi_newton.boothFunctionLBFGS_BTest
 */
@main def boothFunctionLBFGS_BTest (): Unit =

    val lo = VectorD (-10, -10)
    val hi = VectorD ( 10,  10)

    runLBFGSB (name    = "Booth",
               f       = BoothFunction.objFunction,
               x0      = VectorD (-4, 7),
               bounds  = (lo, hi),
               exactLS = false)

    compareLS (name   = "Booth",
               f      = BoothFunction.objFunction,
               x0     = VectorD (-4, 7),
               bounds = (lo, hi))

end boothFunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bealeFunctionLBFGS_BTest` main function tests L-BFGS-B on the Beale function.
 *  > runMain scalation.optimization.quasi_newton.bealeFunctionLBFGS_BTest
 */
@main def bealeFunctionLBFGS_BTest (): Unit =

    val lo = VectorD (-10, -10)
    val hi = VectorD ( 10,  10)

    runLBFGSB (name    = "Beale",
               f       = BealeFunction.objFunction,
               x0      = VectorD (2, -2),
               bounds  = (lo, hi),
               exactLS = false)

    compareLS (name   = "Beale",
               f      = BealeFunction.objFunction,
               x0     = VectorD (2, -2),
               bounds = (lo, hi))

end bealeFunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bohachevsky1FunctionLBFGS_BTest` main function tests L-BFGS-B on the Bohachevsky 1 function.
 *  > runMain scalation.optimization.quasi_newton.bohachevsky1FunctionLBFGS_BTest
 */
@main def bohachevsky1FunctionLBFGS_BTest (): Unit =

    runLBFGSB (name    = "Bohachevsky1",
               f       = Bohachevsky1Function.objFunction,
               x0      = VectorD (10, -10),
               bounds  = makeBounds (2, -10.0, 10.0),
               exactLS = false)

    compareLS (name   = "Bohachevsky1",
               f      = Bohachevsky1Function.objFunction,
               x0     = VectorD (10, -10),
               bounds = makeBounds (2, -10.0, 10.0))

end bohachevsky1FunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bohachevsky2FunctionLBFGS_BTest` main function tests L-BFGS-B on the Bohachevsky 2 function.
 *  > runMain scalation.optimization.quasi_newton.bohachevsky2FunctionLBFGS_BTest
 */
@main def bohachevsky2FunctionLBFGS_BTest (): Unit =

    runLBFGSB (name    = "Bohachevsky2",
               f       = Bohachevsky2Function.objFunction,
               x0      = VectorD (10, -10),
               bounds  = makeBounds (2, -10.0, 10.0),
               exactLS = false)

    compareLS (name   = "Bohachevsky2",
               f      = Bohachevsky2Function.objFunction,
               x0     = VectorD (10, -10),
               bounds = makeBounds (2, -10.0, 10.0))

end bohachevsky2FunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bohachevsky3FunctionLBFGS_BTest` main function tests L-BFGS-B on the Bohachevsky 3 function.
 *  > runMain scalation.optimization.quasi_newton.bohachevsky3FunctionLBFGS_BTest
 */
@main def bohachevsky3FunctionLBFGS_BTest (): Unit =

    runLBFGSB (name    = "Bohachevsky3",
               f       = Bohachevsky3Function.objFunction,
               x0      = VectorD (10, -10),
               bounds  = makeBounds (2, -10.0, 10.0),
               exactLS = false)

    compareLS (name   = "Bohachevsky3",
               f      = Bohachevsky3Function.objFunction,
               x0     = VectorD (10, -10),
               bounds = makeBounds (2, -10.0, 10.0))

end bohachevsky3FunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `camel3FunctionLBFGS_BTest` main function tests L-BFGS-B on the Three-Hump Camel function.
 *  > runMain scalation.optimization.quasi_newton.camel3FunctionLBFGS_BTest
 */
@main def camel3FunctionLBFGS_BTest (): Unit =

    runLBFGSB (name    = "Camel3",
               f       = Camel3Function.objFunction,
               x0      = VectorD (10, -10),
               bounds  = makeBounds (2, -10.0, 10.0),
               exactLS = false)

    compareLS (name   = "Camel3",
               f      = Camel3Function.objFunction,
               x0     = VectorD (10, -10),
               bounds = makeBounds (2, -10.0, 10.0))

end camel3FunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cubeFunctionLBFGS_BTest` main function tests L-BFGS-B on the Cube function.
 *  > runMain scalation.optimization.quasi_newton.cubeFunctionLBFGS_BTest
 */
@main def cubeFunctionLBFGS_BTest (): Unit =

    runLBFGSB (name    = "Cube",
               f       = CubeFunction.objFunction,
               x0      = VectorD (5, -5),
               bounds  = makeBounds (2, -10.0, 10.0),
               exactLS = false)

    compareLS (name   = "Cube",
               f      = CubeFunction.objFunction,
               x0     = VectorD (5, -5),
               bounds = makeBounds (2, -10.0, 10.0))

end cubeFunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `freudensteinRothFunctionLBFGS_BTest` main function tests L-BFGS-B on the Freudenstein-Roth function.
 *  > runMain scalation.optimization.quasi_newton.freudensteinRothFunctionLBFGS_BTest
 */
@main def freudensteinRothFunctionLBFGS_BTest (): Unit =

    runLBFGSB (name    = "Freudenstein-Roth",
               f       = FreudensteinRothFunction.objFunction,
               x0      = VectorD (5, -5),
               bounds  = makeBounds (2, -10.0, 10.0),
               exactLS = false)

    compareLS (name   = "Freudenstein-Roth",
               f      = FreudensteinRothFunction.objFunction,
               x0     = VectorD (5, -5),
               bounds = makeBounds (2, -10.0, 10.0))

end freudensteinRothFunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `mccormickFunctionLBFGS_BTest` main function tests L-BFGS-B on the McCormick function.
 *  > runMain scalation.optimization.quasi_newton.mccormickFunctionLBFGS_BTest
 */
@main def mccormickFunctionLBFGS_BTest (): Unit =

    val bounds = (VectorD (-4.0, -4.0), VectorD (4.0, 4.0))

    runLBFGSB (name    = "McCormick",
               f       = McCormickFunction.objFunction,
               x0      = VectorD (2.5, 3.5),
               bounds  = bounds,
               exactLS = false)

    compareLS (name   = "McCormick",
               f      = McCormickFunction.objFunction,
               x0     = VectorD (2.5, 3.5),
               bounds = bounds)

end mccormickFunctionLBFGS_BTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `run_all_LBFGS_B` main function runs a compact but broad L-BFGS-B regression suite.
 *  > runMain scalation.optimization.quasi_newton.run_all_LBFGS_B
 */
@main def run_all_LBFGS_B (): Unit =

    lBFGS_BQuadraticInteriorTest ()
    lBFGS_BQuadraticActiveBoundTest ()
    lBFGS_BInactiveBoundsVsLBFGSTest ()

    boothFunctionLBFGS_BTest ()
    bealeFunctionLBFGS_BTest ()
    bohachevsky1FunctionLBFGS_BTest ()
    bohachevsky2FunctionLBFGS_BTest ()
    bohachevsky3FunctionLBFGS_BTest ()
    camel3FunctionLBFGS_BTest ()
    cubeFunctionLBFGS_BTest ()
    freudensteinRothFunctionLBFGS_BTest ()
    mccormickFunctionLBFGS_BTest ()

end run_all_LBFGS_B

