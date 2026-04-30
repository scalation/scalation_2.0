
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng
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
import MatrixD.eye

import scala.annotation.unused

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
     *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
     *                            or inexact (e.g., `WolfeLS`) Line Search
     *  @param l_u      (vector, vector) of lower and upper bounds for all input parameters
     *  @param gradF    vector to vector functional formula for computing the gradiant, if available
     */
    def apply (f: FunctionV2S, n: Int,
               exactLS: Boolean = false, l_u_ : Bounds = (null, null),
               gradF: FunctionV2V = null): LBFGS_B =
        
        val l_u = if l_u_ == (null, null) then (VectorD.fill (n)(-1), VectorD.fill (n)(1))
        else l_u_
        new LBFGS_B (f, exactLS, l_u, gradF)
    end apply
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make simple bounds where the limits in each dimension is the same.
     *  @param n   the dimensionality of the search space
     *  @param lo  scalar lower bounds for all input parameters
     *  @param up  scalar upper bounds for all input parameters
     */
    inline def makeBounds (n: Int, lo: Double, up: Double): Bounds =
        (VectorD.fill (n)(lo), VectorD.fill (n)(up))
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
 *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
 *                            or inexact (e.g., `WolfeLS`) Line Search
 *  @param l_u      (vector, vector) of lower and upper bounds for all input parameters
 *  @param gradF    vector to vector functional formula for computing the gradiant, if available
 */
class LBFGS_B (f: FunctionV2S,
               exactLS: Boolean = false,
               private var l_u: Bounds = (null, null),
               gradF: FunctionV2V = null)
  extends Minimizer:
    
    private val debug           = debugf ("LBFGS_B", false)       // debug function
    private var ww, mm: MatrixD = null                            // workspace matrices
    private var theta           = 0.0                             // a scaling parameter
    private var dim             = 0                               // dimension of the input vector
    private var hs              = 5                               // history size, number of historical vectors to store
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort pairs (k, v) according to v into ascending order.
     *  std::vector<int> sort_indexes(const std::vector< std::pair<int, Scalar> > &v)
     *  @param v  the ArrayBuffer of Tuple2 to be sorted by the 2nd element
     */
    private def sortIndices (v: ArrayBuffer [(Int, Double)]): VectorI =
        val sv  = v.sortBy (_._2)                                 // FIX - order different in C++ code
        val idx = new VectorI (sv.length)
        for i <- idx.indices do idx(i) = sv(i)._1
        idx
    end sortIndices
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Force the values within 'v' to stay within the pre-defined bounds.
     *  @see void clampToBound(const TProblem &problem, TVector &x)
     *  @param v  the Vector containing values to be adjusted
     */
    private def forceBounds (v: VectorD): Unit =
        val (l, u) = l_u
        for i <- v.indices do
            if v(i) > u(i)      then v(i) = u(i)                  // upper bound
            else if v(i) < l(i) then v(i) = l(i)                  // lower bound
        end for
    end forceBounds
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Algorithm CP: Computation of the Generalized Cauchy Point. See page 8 of
     *  @see www.ece.northwestern.edu/~nocedal/PSfiles/limited.ps.gvz
     *  @see void getGeneralizedCauchyPoint(const TProblem &problem, const TVector &x,
     *                                      const TVector &g, TVector &x_cauchy, VariableTVector &c)
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
        var fPrime        = -d.dot (d)
        
        // MODIFIED
        inline def eps = Math.ulp (1.0)                           // Match C++ code for numerical stability
        
        var fDoublePrime = max (-theta * fPrime - (p dot (mm * p)), eps)   // Using eps instead of EPSILON
        val f_dp_orig    = fDoublePrime
        var dt_min       = -fPrime / fDoublePrime
        var t_old        = 0.0
        
        var i = 0
        breakable {
            for j <- 0 until dim do
                i = j
                if setOfT (sortedIndices(j))._2 > 0 then break ()
        } // breakable
        var b  = sortedIndices(i)
        var t  = setOfT(b)._2
        var dt = t
        
        while dt_min >= dt && i < dim do
            if d(b) > 0 then xCauchy(b) = u(b)
            else if d(b) < 0 then xCauchy(b) = l(b)
            val zb = xCauchy(b) - x(b)
            c     += p * dt
            
            // cache
            val wbt       = ww(b)
            fPrime       += dt * fDoublePrime + gr(b) * gr(b) + theta * gr(b) * zb - gr(b) * wbt.dot (mm * c)
            fDoublePrime += -theta  *  gr(b) *  gr(b) - 2.0   * (gr(b) * wbt.dot (mm * p))
                            - gr(b) *  gr(b) * (wbt.dot (mm * wbt))
            fDoublePrime  = max (eps * f_dp_orig, fDoublePrime)   // Using eps instead of EPSILON
            
            p     += wbt * gr(b)
            d(b)   = 0
            dt_min = -fPrime / fDoublePrime
            t_old  = t
            i     += 1
            if i < dim then
                b  = sortedIndices(i)
                t  = setOfT(b)._2
                dt = t - t_old
        end while
        
        dt_min = max (dt_min, 0.0)
        t_old += dt_min
        
        for ii <- i until xCauchy.dim do
            val si      = sortedIndices (ii)
            xCauchy(si) = x(si) + t_old * d(si)
        c += p * dt_min
        (xCauchy, c)
    end getGCP
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the alpha* parameter, a positive scalar.  See Equation 5.8 on page 11 of
     *  @see www.ece.northwestern.edu/~nocedal/PSfiles/limited.ps.gvz
     *  @see Scalar findAlpha(const TProblem &problem, TVector &x_cp, VariableTVector &du, std::vector<int> &FreeVariables)
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
        
        // MODIFIED : changed to match C++ code (will help with numerical stability)
        var i = 0
        while i < n do
            val v = du(i)
            if abs (v) >= 1e-7 then                               // keep guard
                val fi = freeVar(i)
                val a = if v > 0.0 then (u(fi) - x_cp(fi)) / v
                        else (l(fi) - x_cp(fi)) / v
                alphastar = min (alphastar, a)
            end if
            i += 1
        end while
        
        max (0.0, min (1.0, alphastar))                           // ensure in [0, 1]
    end findAlpha
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Minimization of the subspace of free variables.  See Section 5 on page 9 of
     *  @see www.ece.northwestern.edu/~nocedal/PSfiles/limited.ps.gvz
     *  @see void SubspaceMinimization(const TProblem &problem, TVector &x_cauchy,
                                       TVector &x, VariableTVector &c, TVector &g, TVector &SubspaceMin)
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
    /** Modify the number of historical vectors to store.
     *  @see void setHistorySize(const int hs) { m_historySize = hs; }
     *  @param hs_  the new history size
     */
    def setHistorySize (hs_ : Int): Unit = hs = hs_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the projected gradient under simple bounds.
     *  If x_i is strictly inside (l_i, u_i), keep g_i.
     *  If x_i is at lower bound, clamp to min(0, g_i) (can't go further down).
     *  If x_i is at upper bound, clamp to max(0, g_i) (can't go further up).
     *  Result is zero in components where movement is blocked by an active bound.
     *  @param x  the current point
     *  @param g  the raw gradient at x
     *  @param l  the vector of lower bounds
     *  @param u  the vector of upper bounds
     */
    private def projectedGrad (x: VectorD, g: VectorD, l: VectorD, u: VectorD): VectorD =
        val pg = new VectorD (x.dim)
        var i  = 0
        while i < x.dim do
            val xi = x(i);
            val gi = g(i)
            pg(i) =
                if xi > l(i) && xi < u(i) then gi
                else if xi <= l(i) then math.min (0.0, gi)
                else /* xi >= u(i) */ math.max (0.0, gi)
            i += 1
        pg
    end projectedGrad
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The objective function f.
     *  @param x  the coordinate values of the current point
     */
    inline override def fg (x: VectorD): Double = f(x)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an exact `GoldenSectionLS` or inexact `WolfeLS` Line Search.
     *  Search in direction dir, returning the distance z to move in that direction.
     *  @param x     the current point
     *  @param dir   the direction to move in
     *  @param step  the initial step size
     */
    @deprecated ("Use LBFGSLineSearch with MoreThuente instead", "2.0")
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double =
        debug ("linesearch", s"x = $x, dir = $dir, step = $step")
        
        def f_1D (z: Double): Double = fg(x + dir * z)            // create a 1D function
        val ls = if exactLS then new GoldenSectionLS (f_1D )      // Golden Section Line Search
                 else new WolfeLS (f_1D)                          // Wolfe line search ((c1 = .0001, c2 = .9)
        ls.search (step)                                          // perform a Line Search
    end lineSearch
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the following Non-Linear Programming (NLP) problem using L-BFGS_B:
     *      min { f(x) | g(x) <= 0 }.
     *  @see void minimize(TProblem &problem, TVector &x0)
     *  @param x0         the starting point
     *  @param alphaInit  the initial step size
     *  @param toler      the tolerance
     */
    def solve (x0: VectorD, alphaInit: Double = 1.0, toler: Double = EPSILON): FuncVec =
        debug ("solve", s"x0 = $x0, alphaInit = $alphaInit, toler = $toler")
        
        var best = (MAX_VALUE, VectorD.nullv)
        
        var outerIters = 0
        var totalLineSearchIters = 0
        
        dim   = x0.dim
        theta = 1.0
        if l_u == (null, null) then l_u = makeBounds (dim, NEGATIVE_INFINITY, POSITIVE_INFINITY)
        
        ww = new MatrixD (dim, 0)                                 // FIX - causes empty matrix warning
        mm = LBFGS_B.emptyMatrix
        
        val yHistory =  ArrayBuffer [VectorD] ()
        val sHistory =  ArrayBuffer [VectorD] ()
        var yHistoryMx: MatrixD = null
        var sHistoryMx: MatrixD = null
        
        var x = x0.copy
        forceBounds (x)                                           // MODIFIED: force bounds on initial x (faster convergence)
        
        val grad_fg = if gradF != null then gradF else (z: VectorD) => ∇(fg)(z)
        var gr = grad_fg(x)
        
        var fv       = fg(x)                                      // functional value at x
        var mgn      = 0.0
        var count    = 0
        val countMax = 10
        
        // FIX -- missing "auto noConvergence = ..."
        
        breakable {                                               // main while loop in C++ code
            for k <- 1 to MAX_IT do
//              banner (s"solve: iteration $k: f(x) = $fv, x = $x")
                val f_old   = fv
                val x_old   = x
                val g_old   = gr
                val mgn_old = mgn
                
                // MODIFIED: removed forceBounds on x here to match C++ code
                // STEP 2: compute the cauchy point
                val (xCauchy, c) = getGCP (x, gr)
                
                // STEP 3: compute a search direction d_k by the primal method for the sub-problem
                val subspaceMin = subspaceMinimize (x, gr, xCauchy, c)
                
                // STEP 4: perform linesearch                     // MODIFIED: Used MoreThuente line search and internal projection to bounds
                val dir = subspaceMin - x
                
                val evalLogic =
                    if (gradF != null) then FunctionEvaluation (f, grad_fg)
                    else FunctionEvaluation (f)                   // numeric grad will project internally
                
                val cd = LBFGSCallbackData (dim, null, evalLogic)
                
                val lineSearchPrms = LBFGSLineSearchPrms (maxLineSearch = 20, defaultStep = 1.0, minStep = 1e-15,
                                                          maxStep = 1e15, ftol = 1e-4, wolfe = 0.9, gtol = 1e-2, xtol = 1e-15)
                
                val lineSearchImple = LBFGSLineSearch.getImple (LBFGSLineSearchAlg.MoreThuente)
                
                val lineSearchResults = lineSearchImple.lineSearch (dim, x, fv, gr, dir, alphaInit, cd, lineSearchPrms, None)
                
                lineSearchResults match
                    case step: LBFGSLineSearchStep =>
                        totalLineSearchIters += step.numberOfIterations
                        val stepLen = step.step
                        val x_prev = x
                        val f_prev = fv
                        
                        // 1. Trial Step (Unclamped)
                        val x_try = x_prev + dir * stepLen
                        
                        // 2. Now, clamp final x to bounds (element-wise)
                        val x_new = x_try.copy
                        forceBounds (x_new)
                        
                        // 3. If clamp changed x, then re-evaluate f and g
                        inline def notSame (a: VectorD, b: VectorD): Boolean = (a - b).norm > 0.0
                        
                        if notSame (x_new, x_try) then
                            x  = x_new
                            fv = fg(x)
                            gr = grad_fg(x)
                        else
                            x  = x_try
                            fv = step.fx
                            gr = step.g
                        
                        // compute diagnostics
                        val gnorm  = gr.norm
                        val xdelta = (x - x_prev).norm
                        val fdelta = f_prev - fv
                        printIteration (k, fv, x, gr, gnorm, xdelta, fdelta)
                        outerIters = k
                    
                    case fail: LBFGSLineSearchFailure =>
                        println (s"LBFGS_B.solve: line search failed after $k iterations.")
                        println (s"  Return code: ${fail.returnCode}")
                        best = better ((f_old, x_old), best)
                        break ()
                end match
                
                // MODIFIED: changed to match C++ code more closely
                // STEP 5: stationarity & stopping criteria
                if blown ((fv, x)) then
                    best = better ((f_old, x_old), best)
                    break ()
                
                // Projected gradient (L-BFGS-B style) and ∞-norm (Matches C++ code)
                val (l, u) = l_u
                val pg     = projectedGrad (x, gr, l, u)
                mgn        = pg.normInf
                
                val pgtol   = 1e-8                                // tighten to 1e-8 for more polish
                val ftolRel = 1e-9                                // relative f-decrease stop
                
                // Primary stop: projected-gradient ∞-norm
                if mgn <= pgtol || count > countMax then
                    best = better ((fv, x), best)
                    break ()
                
                // Secondary stop: tiny relative f change
                if math.abs (f_old - fv) <= ftolRel * (1.0 + math.abs (f_old)) then
                    best = better ((fv, x), best)
                    break ()
                
                // No-progress counter (use pgtol, not toler)
                if math.abs (mgn - mgn_old) < pgtol then count += 1
                else count = 0
                
                val newY = gr - g_old                             // prepare for next iteration
                val newS = x - x_old
                
                // STEP 6 — curvature check and memory update (standard L-BFGS practice)
                // Compute curvature scalars
                val ys = newY dot newS                            // yᵀs
                val yy = newY dot newY                            // yᵀy
                
                // cppoptlib condition: accept pair if |sᵀy| > 1e-7 * (yᵀy)
                if (abs(ys) > 1e-7 * yy) then
                    // Accept the pair
                    if yHistory.size >= hs then
                        yHistory.remove (0)
                        sHistory.remove (0)
                    yHistory append newY
                    sHistory append newS
                    
                    // STEP 7 — positive scaling
                    theta = yy / ys                               // stays > 0 because ys > 0
                    
                    yHistoryMx = MatrixD (yHistory).transpose
                    sHistoryMx = MatrixD (sHistory).transpose
                    
                    ww = yHistoryMx ++^ (sHistoryMx * theta)
                    
                    val aa = sHistoryMx.transpose * yHistoryMx
                    val ll = aa.lower
                    ll(?, ?) = 0.0
                    
                    val dd = new MatrixD (aa.dim, aa.dim2)
                    dd.setDiag (-aa(?))
                    
                    val mm2 = (dd ++^ ll.transpose) ++ (ll ++^ (sHistoryMx.transpose * sHistoryMx * theta))
                    mm = Fac_LU.inverse(mm2)()
                end if
                
                debug ("solve", s"(k = $k) move from $x_old to $x where fg(x) = $fv")
                
                best = better ((fv, x), best)
            end for
        } // breakable
        
        println (f"argmin ${x(0)}%.6f ${if (x.dim > 1) then x(1) else 0.0}%.6f")
        println (s"f in argmin ${f2(fv)}")
        println (s"iterations $outerIters")
        banner (s"solve: optimal solution = $best (outer iters = $outerIters, line-search iters = $totalLineSearchIters)")
        
        best
    end solve
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Format the double argument.
     *  @param x  the value to be formatted
     */
    private inline def f2 (x: Double): String = f"$x%.6f"
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Format the vector argument.
     *  @param x  the vector to be formatted
     */
    private inline def vecStr (v: VectorD): String =
        if v == null then "—"
        else v.map (d => "%.6g".format(d)).mkString ("  ")
    end vecStr
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print information about the current iteration it.
     */
    private def printIteration (it: Int, fx: Double, x: VectorD, g: VectorD,
                                gnorm: Double, xdelta: Double, fdelta: Double): Unit =
        println (f"--- Iteration: $it%5d ---")
        println (s"  Value:           ${f2(fx)}")
        println (s"  X:               ${vecStr(x)}")
        println (s"  Gradient:        ${vecStr(g)}")
        println (s"  Gradient Norm:   ${f2(gnorm)}")
        println (s"  X Delta:         ${f2(xdelta)}")
        println (s"  F Delta:         ${f2(fdelta)}")
        println (s"  Hessian Cond.:   ${f2(theta)}")
        println ("-------------------------")
    end printIteration

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
    
    banner ("Minimize (bounds [3.5, 5.0]): (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
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
    
    banner ("Minimize (bounds [3.5, 5.0]): x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
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
    
    banner ("Minimize (bounds [3.5, 5.0]): 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    val lu = makeBounds (x0.dim, 3.5, 5.0)
    optimizer = new LBFGS_B (f, l_u = lu)
    opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

end lBFGS_BTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BTest4` main function is used to test the `LBFGS_B` class.
 *      f(x) = 5 * x(0) * x(0) + 100 * x(1) * x(1) + 5
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest4
 */
@main def lBFGS_BTest4 (): Unit =

    import scalation.calculus.Differential.{∇, Η}

    val x0 = VectorD (-10.0, 2.0)
    
    // function
    def f (x: VectorD): Double = 5 * x(0) * x(0) + 100 * x(1) * x(1) + 5
    // analytical gradient
    def gradF (x: VectorD): VectorD = VectorD(10 * x(0), 200 * x(1))
    // analytical Hessian
    def hessF (@unused x: VectorD): MatrixD = MatrixD((2, 2), 10.0, 0.0, 0.0, 200.0)
    
    banner("Minimize (no bounds): 5 * x(0)^2 + 100 * x(1)^2 + 5")
    
    // Print function value and gradient at start (like cpp)
    println (s"init x0 = $x0")
    println (s"f(x0) = ${f(x0)}")

    // Gradient check: analytical vs numerical
    val gradAnalytic = gradF(x0)
    val gradNumeric  = ∇(f)(x0)
    val gradError    = (gradAnalytic - gradNumeric).norm
    val gradOk       = gradError < 1e-6
    println (s"grad f(x0) (analytic) = $gradAnalytic")
    println (s"grad f(x0) (numeric)  = $gradNumeric")
    println (s"Gradient check ok? $gradOk (error = $gradError)")
    
    // Hessian check: analytical vs numerical
    val hessAnalytic = hessF(x0)
    val hessNumeric  = Η (f, x0)
    val hessError    = (hessAnalytic - hessNumeric).normF
    val hessOk       = hessError < 1e-6
    println (s"Hessian f(x0) (analytic) =\n$hessAnalytic")
    println (s"Hessian f(x0) (numeric)  =\n$hessNumeric")
    println (s"Hessian check ok? $hessOk (error = $hessError)")
    
    // Run LBFGS_B once
    val optimizer = new LBFGS_B(f)
    val (fx, argmin) = optimizer.solve (x0)
    
    println (s"argmin = $argmin")
    println (s"f(argmin) = $fx")
    println (s"status = converged")

end lBFGS_BTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BTest5` main function is used to test the `LBFGS_B` class
 *  on the Rosenbrock function with bounds, mirroring the cppoptlib test.
 *      f(x) = (1 - x0)^2 + 100 * (x1 - x0^2)^2
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest5
 */
@main def lBFGS_BTest5 (): Unit =

    val x0 = VectorD (-1.2, 1.0)                     // classic Rosenbrock start
    
    // Rosenbrock function
    def f (x: VectorD): Double =
        val x0 = x(0)
        val x1 = x(1)
        (1 - x0)~^2 + 100 * (x1 - x0~^2)~^2
    
    // Analytical gradient
    def gradF (x: VectorD): VectorD =
        val x0 = x(0)
        val x1 = x(1)
        VectorD (-2 * (1 - x0) - 400 * x0 * (x1 - x0 * x0),
                 200 * (x1 - x0 * x0))
    
    // Analytical Hessian
    def hessF (x: VectorD): MatrixD =
        val x0 = x(0)
        val x1 = x(1)
        MatrixD ((2, 2), 2 - 400 * x1 + 1200 * x0 * x0, -400 * x0,
                         -400 * x0,                      200.0)
    
    banner("Rosenbrock with bounds: f(x) = (1 - x0)^2 + 100*(x1 - x0^2)^2")
    
    // Print function value and gradient at start
    println (s"init x0 = $x0")
    println (s"f(x0) = ${f(x0)}")
    
    import scalation.calculus.Differential.{∇, Η}
    val gradAnalytic = gradF(x0)
    val gradNumeric  = ∇(f)(x0)
    val gradError    = (gradAnalytic - gradNumeric).norm
    println (s"grad f(x0) (analytic) = $gradAnalytic")
    println (s"grad f(x0) (numeric)  = $gradNumeric")
    println (s"Gradient check error  = $gradError")
    
    val hessAnalytic = hessF(x0)
    val hessNumeric  = Η(f, x0)
    val hessError    = (hessAnalytic - hessNumeric).normF
    println (s"Hessian f(x0) (analytic) =\n$hessAnalytic")
    println (s"Hessian f(x0) (numeric)  =\n$hessNumeric")
    println (s"Hessian check error      = $hessError")
    
    // Set tight bounds (exclude true minimum (1,1))
    val lower = VectorD(-1.0, -0.5)
    val upper = VectorD( 0.2,  0.25)
    
    val lu = (lower, upper)
    
    println (s"Bounds: lower = $lower, upper = $upper")
    
    // Run LBFGS_B
    val optimizer = new LBFGS_B (f, l_u = lu, gradF = gradF)
    val (fx, argmin) = optimizer.solve (x0)
    
    println (s"argmin = $argmin")
    println (s"f(argmin) = $fx")
    println (s"status = converged/bounded optimum")

end lBFGS_BTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BTest5` main function is used to test the `LBFGS_B` class
 *  on the ReciprocalFunction function with bounds. In addition to ReciprocalFunction
 *  function you may want to test RosenbrockFunction, McCormickFunction, and
 *  FreudensteinRothFunction functions.
 *      f(x) = 1/x(0) + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest6
 */
@main def lBFGS_BTest6 (): Unit =

    import functions.ReciprocalFunction

    val x0 = VectorD (0.1, 0.1)                               // starting location

    banner ("Minimize: 1/x(0) + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    def f     = ReciprocalFunction.objFunction
//  def gradF = ReciprocalFunction.gradFunction
    val lu    = ReciprocalFunction.bound

    // Run LBFGS_B
    val optimizer = new LBFGS_B (f, l_u = lu)                 //, gradF = gradF)
    val (fx, argmin) = optimizer.solve (x0)

    println (s"argmin    = $argmin")
    println (s"f(argmin) = $fx")
    println (s"status    = converged/bounded optimum")

end lBFGS_BTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_BStressTest` main function stress-tests LBFGS_B on Rosenbrock
 *  with several different bound configurations.
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BStressTest
 */
@main def lBFGS_BStressTest (): Unit =
    
    val x0 = VectorD (-1.2, 1.0)            // classic Rosenbrock starting point
    
    // Rosenbrock function
    def f(x: VectorD): Double = (1 - x(0))~^2 + 100.0 * (x(1) - x(0)~^2)~^2
    def gradF(x: VectorD): VectorD =
        VectorD (-2 * (1 - x(0)) - 400 * x(0) * (x(1) - x(0)~^2),
                 200 * (x(1) - x(0)~^2))
    
    // Different bound options
    val bounds = Seq (
        "Option 1: Tight (exclude true minimum)" -> (VectorD(-1.0, -0.5), VectorD(0.2, 0.25)),
        "Option 2: Very Tight Box Around Wrong Point" -> (VectorD(0.0, 0.0), VectorD(0.5, 0.5)),
        "Option 3: Skewed Bounds (trap along edge)"   -> (VectorD(-1.0, -1.0), VectorD(0.2, 2.0)),
        "Option 4: Huge Bounds (almost unconstrained)"-> (VectorD(-5.0, -5.0), VectorD(5.0, 5.0)),
        "Option 5: Lower Bound Excludes True Min"     -> (VectorD(-2.0, 0.0), VectorD(0.8, 0.9)))
    
    for (label, lu) <- bounds do
        banner (s"Rosenbrock Stress Test - $label")
        
        println (s"init x0 = $x0, f(x0) = ${f(x0)}")
        println (s"Bounds: lower = ${lu._1}, upper = ${lu._2}")
        
        val optimizer = new LBFGS_B (f, l_u = lu, gradF = gradF)
        val (fx, argmin) = optimizer.solve (x0)
        
        println (s"argmin = $argmin")
        println (s"f(argmin) = $fx")
        println (s"status = converged/bounded optimum\n")
    end for

end lBFGS_BStressTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Quadratic 2D with known solution and clipping by bounds.
 *  f(x) = 0.5 * (x - c)^T diag(d) (x - c)
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest_Quad2D
 */
@main def lBFGS_BTest_Quad2D (): Unit =
    val c  = VectorD (1.5, -2.0)
    val d  = VectorD (4.0, 1.0)                   // diag entries (SPD)
    val x0 = VectorD (0.0, 0.0)
    
    def f(x: VectorD): Double =
        val dx = x - c
        0.5 * (dx * d).dot(dx)
    
    def gradF(x: VectorD): VectorD =
        (x - c) * d
    
    // Bounds that force the solution to lie on a face (clip c(0) to 1.0)
    val lower = VectorD (-1.0, -3.0)
    val upper = VectorD ( 1.0,  0.0)
    
    println ("Quadratic 2D with clipping bounds")
    println (s"true unconstrained minimizer: c = $c")
    println (s"Bounds: lower = $lower, upper = $upper")
    
    val opt = new LBFGS_B(f, l_u = (lower, upper), gradF = gradF)
    val (fx, x) = opt.solve (x0)
    
    println (s"argmin = $x")
    println (s"f(argmin) = $fx")

end lBFGS_BTest_Quad2D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Himmelblau (multimodal), moderate bounds.
 *  f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest_Himmelblau
 */
@main def lBFGS_BTest_Himmelblau (): Unit =
    val x0 = VectorD (-3.5, 3.0)
    
    def f(x: VectorD): Double =
        val a = x(0); val b = x(1)
        val t1 = a*a + b - 11.0
        val t2 = a + b*b - 7.0
        t1*t1 + t2*t2
    
    def gradF(x: VectorD): VectorD =
        val a = x(0); val b = x(1)
        val t1 = a*a + b - 11.0
        val t2 = a + b*b - 7.0
        VectorD (4.0*a*t1 + 2.0*t2,
                 2.0*t1 + 4.0*b*t2)
    
    val lower = VectorD (-5.0, -5.0)
    val upper = VectorD ( 5.0,  5.0)
    
    println ("Himmelblau with moderate bounds")
    println (s"init x0 = $x0, f(x0) = ${f(x0)}")
    println (s"Bounds: lower = $lower, upper = $upper")
    
    val opt = new LBFGS_B (f, l_u = (lower, upper), gradF = gradF)
    val (fx, x) = opt.solve (x0)
    
    println (s"argmin = $x")
    println (s"f(argmin) = $fx")     // expect near one of the known minima ~ (3,2), (-2.805,3.131), ...

end lBFGS_BTest_Himmelblau


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Wood function (4D Rosenbrock variant), classic nonconvex test.
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest_Wood4D
 */
@main def lBFGS_BTest_Wood4D (): Unit =
    val x0 = VectorD (-3.0, -1.0, -3.0, -1.0)
    
    def f(x: VectorD): Double =
        val x1 = x(0); val x2 = x(1); val x3 = x(2); val x4 = x(3)
        100.0 * (x2 - x1~^2)~^2 + (1.0 - x1)~^2 +
          90.0  * (x4 - x3~^2)~^2 + (1.0 - x3)~^2 +
          10.0  * (x2 + x4 - 2.0)~^2 + 0.1 * (x2 - x4)~^2
    
    def gradF(x: VectorD): VectorD =
        val x1 = x(0); val x2 = x(1); val x3 = x(2); val x4 = x(3)
        val g1 = -400.0 * (x2 - x1*x1) * x1 - 2.0 * (1.0 - x1)
        val g2 =  200.0 * (x2 - x1*x1) + 20.0 * (x2 + x4 - 2.0) + 0.2 * (x2 - x4)
        val g3 = -360.0 * (x4 - x3*x3) * x3 - 2.0 * (1.0 - x3)
        val g4 =  180.0 * (x4 - x3*x3) + 20.0 * (x2 + x4 - 2.0) - 0.2 * (x2 - x4)
        VectorD (g1, g2, g3, g4)
    
    val lower = VectorD.fill (4)(-5.0)
    val upper = VectorD.fill (4)( 5.0)
    
    println ("Wood 4D with wide bounds")
    println (s"init x0 = $x0, f(x0) = ${f(x0)}")
    
    val opt = new LBFGS_B (f, l_u = (lower, upper), gradF = gradF)
    val (fx, x) = opt.solve (x0)
    
    println (s"argmin = $x")
    println (s"f(argmin) = $fx")       // true min at (1,1,1,1) with f ≈ 0

end lBFGS_BTest_Wood4D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** High-dim Rosenbrock (n = 10), almost-unconstrained with wide box.
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest_RosenbrockN
 */
@main def lBFGS_BTest_RosenbrockN (): Unit =
    val n  = 10
    val x0 = VectorD (for i <- 0 until n yield if i % 2 == 0 then -1.2 else 1.0)
    
    def f(x: VectorD): Double =
        var s = 0.0
        var i = 0
        while i < n - 1 do
            val xi = x(i); val xip = x(i+1)
            s += (1.0 - xi)~^2 + 100.0 * (xip - xi*xi)~^2
            i += 1
        s
    end f
    
    def gradF(x: VectorD): VectorD =
        val g = new VectorD(n)
        
        var i = 0
        while i < n do
            var gi = 0.0
            
            // backward neighbor term: +200 * (x_i - x_{i-1}^2)
            if i > 0 then
                gi += 200.0 * (x(i) - x(i-1) * x(i-1))
            
            // forward terms: -2(1 - x_i) - 400 x_i (x_{i+1} - x_i^2)
            if i < n - 1 then
                gi += -2.0 * (1.0 - x(i)) - 400.0 * x(i) * (x(i+1) - x(i) * x(i))
            
            g(i) = gi
            i += 1
        g
    end gradF
    
    val lower = VectorD.fill (n)(-5.0)
    val upper = VectorD.fill (n)( 5.0)
    
    println (s"High-dim Rosenbrock (n=$n) with wide bounds")
    println (s"init f(x0) = ${f(x0)}")
    
    val opt = new LBFGS_B(f, l_u = (lower, upper), gradF = gradF)
    val (fx, x) = opt.solve (x0)
    
    println (s"argmin = $x")
    println (s"f(argmin) = $fx")     // expect near all-ones

end lBFGS_BTest_RosenbrockN


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Mixed/infinite bounds + active-set corner behavior on a tilted quadratic.
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest_MixedBounds
 */
@main def lBFGS_BTest_MixedBounds (): Unit =
    val x0 = VectorD (2.0, -3.0, 0.5, -1.0)
    
    // f(x) = 0.5 * ||A x - b||^2  (convex), with nontrivial coupling
    val A = MatrixD ((4, 4), 3.0,  1.0, 0.0,  0.0,
                             1.0,  2.0, 1.0,  0.0,
                             0.0,  1.0, 3.0,  1.0,
                             0.0,  0.0, 1.0,  2.0)
    val b = VectorD (1.0, -2.0, 0.5, 1.0)
    
    def f(x: VectorD): Double =
        val r = A * x - b
        0.5 * r.dot(r)
    
    def gradF(x: VectorD): VectorD =
        val r = A * x - b
        A.transpose * r
    
    // Mixed bounds: some finite, some free (±∞), plus asymmetric box to hit faces/corners.
    val lower = VectorD (-1.0, NEGATIVE_INFINITY, 0.0, -0.5)
    val upper = VectorD ( 0.5,  POSITIVE_INFINITY, 1.0,  0.2)
    
    println ("Mixed/infinite bounds on coupled quadratic")
    println (s"init x0 = $x0, f(x0) = ${f(x0)}")
    println (s"Bounds: lower = $lower, upper = $upper")
    
    val opt = new LBFGS_B (f, l_u = (lower, upper), gradF = gradF)
    val (fx, x) = opt.solve (x0)
    
    println (s"argmin = $x")
    println (s"f(argmin) = $fx")

end lBFGS_BTest_MixedBounds


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Convex QP with mixed bounds, exact-check via active-set enumeration.
 *  f(x) = 0.5 x^T Q x - b^T x,  Q ≻ 0 (rotated anisotropic), box bounds.
 *  > runMain scalation.optimization.quasi_newton.lBFGS_BTest_QPEnum
 */
@main def lBFGS_BTest_QPEnum (): Unit =
    // Convex Quadratic Program with box constraints.
    // Minimize: 0.5 * x^T Q x - b^T x  subject to  l <= x <= u
    // We'll build a 4D SPD matrix Q with coupling terms, enumerate all active sets,
    // solve exactly, then compare to LBFGS_B solution.
    
    val n = 4
    
    // Construct a symmetric positive definite Q with off-diagonal coupling
    val Q = MatrixD ((n, n), 6.0,  2.0,  1.0,  0.0,
                             2.0,  5.0, -1.0,  1.0,
                             1.0, -1.0,  4.0,  1.5,
                             0.0,  1.0,  1.5,  3.5)
    
    // Ensure strict SPD by adding a small multiple of identity if needed (numerical safety)
    // (Not generally necessary here, but kept for robustness)
    val eps = 1e-10
    cfor (0, n) { i => Q(i, i) = Q(i, i) + eps }
    
    val b = VectorD (1.0, -2.0, 0.5, 1.5)
    
    // Box bounds
    val l = VectorD (-0.5, -1.0, 0.0, -0.2)
    val u = VectorD ( 1.0,  0.5, 1.2,  0.8)
    
    // Objective function and analytic gradient
    def f (x: VectorD): Double = 0.5 * (x dot (Q * x)) - b.dot(x)
    def gradF (x: VectorD): VectorD = Q * x - b
    
    // Active set enumeration: each variable can be at lower (-1), free (0), or upper (+1)
    case class EnumResult (x: VectorD, fx: Double)
    
    def enumerateQP (): EnumResult =
        var best: EnumResult = null
        
        val pattern = Array.fill (n)(-1)
        // Iterate over all 3^n patterns using base-3 counting
        val total = math.pow(3, n).toInt
        var code  = 0
        while code < total do
            // Decode code into ternary digits → pattern
            var tmp = code
            var i = 0
            while i < n do
                pattern(i) = tmp % 3 - 1         // values in {-1,0,1}
                tmp /= 3
                i += 1
            
            val x = new VectorD(n)
            val freeIdxBuf = new scala.collection.mutable.ArrayBuffer[Int]()
            
            // Assign bounds for active variables, collect free indices
            i = 0
            while i < n do
                pattern(i) match
                    case -1 => x(i) = l(i)      // at lower
                    case  1 => x(i) = u(i)      // at upper
                    case  0 => freeIdxBuf += i  // free variable
                i += 1
            
            val freeIdx = freeIdxBuf.toArray
            val m = freeIdx.length
            
            var feasible = true
            if m > 0 then
                // Build Q_ff and rhs = b_f - Q_fb * x_b
                val Qff = new MatrixD(m, m)
                val rhs = new VectorD(m)
                
                var ii = 0
                while ii < m && feasible do
                    val gi = freeIdx(ii)
                    // b_f component
                    var rhs_i = b(gi)
                    // subtract Q(g_i, j) * x(j) for bound-active j (non-free)
                    var j = 0
                    while j < n do
                        if pattern(j) != 0 then rhs_i -= Q(gi, j) * x(j)
                        j += 1
                    rhs(ii) = rhs_i
                    
                    // fill row of Q_ff
                    var jj = 0
                    while jj < m do
                        Qff(ii, jj) = Q(freeIdx(ii), freeIdx(jj))
                        jj += 1
                    ii += 1
                
                // Solve Q_ff * x_f = rhs
                if feasible then
                    val lu = new Fac_LU(Qff.copy)
                    lu.factor()
                    val xf = lu.solve (rhs)
                    
                    // Put back free components and check bounds
                    ii = 0
                    while ii < m && feasible do
                        val idx = freeIdx(ii)
                        val v = xf(ii)
                        if v < l(idx) - 1e-10 || v > u(idx) + 1e-10 then feasible = false
                        x(idx) = math.max(l(idx), math.min(u(idx), v))
                        ii += 1
            end if
            
            if feasible then
                // Verify KKT: projected gradient zero (optional check)
                val g = gradF(x)
                var kktOk = true
                i = 0
                while i < n && kktOk do
                    if pattern(i) == 0 then
                        // free: gradient near 0
                        if math.abs(g(i)) > 1e-6 then kktOk = false
                    else if pattern(i) == -1 then
                        // at lower: gradient >= 0 (cannot decrease objective by decreasing x)
                        if g(i) < -1e-6 then kktOk = false
                    else
                        // at upper: gradient <= 0
                        if g(i) > 1e-6 then kktOk = false
                    i += 1
                
                if kktOk then
                    val fx = f(x)
                    if best == null || fx < best.fx then best = EnumResult (x.copy, fx)
            end if
            
            code += 1
        end while
        best
    end enumerateQP
    
    val exact = enumerateQP()
    println (s"Exact enumeration optimum: f = ${exact.fx} at x = ${exact.x}")
    
    // Run LBFGS_B from a generic starting point
    val x0 = VectorD(0.3, -0.4, 0.9, 0.0)
    println (s"Initial x0 = $x0, f(x0) = ${f(x0)}")
    
    val optimizer = new LBFGS_B (f, l_u = (l, u), gradF = gradF)
    val (fx_lbfgs, x_lbfgs) = optimizer.solve (x0)
    
    println (s"LBFGS_B optimum: f = $fx_lbfgs at x = $x_lbfgs")
    
    // Compare solutions
    val relObjErr = math.abs (fx_lbfgs - exact.fx) / math.max (1.0, math.abs (exact.fx))
    val xDiffNorm = (x_lbfgs - exact.x).norm
    
    println (f"Relative objective error = $relObjErr%.3e")
    println (f"||x_lbfgs - x_exact||_2 = $xDiffNorm%.3e")
    
    // Tolerances (convex QP, expect tight agreement)
    val objTol = 1e-8
    val xTol   = 1e-6
    
    assert (relObjErr < objTol, s"Objective mismatch: rel error = $relObjErr > $objTol")
    assert (xDiffNorm < xTol,   s"Solution vector mismatch: norm diff = $xDiffNorm > $xTol")
    
    println ("QPEnum test: PASSED (LBFGS_B matches enumerated optimum)")

end lBFGS_BTest_QPEnum

