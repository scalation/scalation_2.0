
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

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGS_B2` companion object provides a factory method for Limited memory
 *  Broyden–Fletcher–Goldfarb–Shanno for Bounds constrained optimization.
 */
object LBFGS_B2:

    val emptyMatrix = new MatrixD (0, 0)                          // empty zero dimension matrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `LBFGS_B2` object with a given dimensionality and default lower
     *  and upper bounds of -1 and 1, respectively.
     *  @param f        the objective function to be minimized
     *  @param n        the dimensionality of the search space
     *  @param exactLS  whether to use exact (e.g., `GoldenLS`)
     *                            or inexact (e.g., `WolfeLS`) Line Search
     *  @param l_u      (vector, vector) of lower and upper bounds for all input parameters
     *  @param gradF    vector to vector functional formula for computing the gradiant, if available
     */
    def apply (f: FunctionV2S, n: Int,
               exactLS: Boolean = false, l_u_ : Bounds = null,
               gradF: FunctionV2V = null): LBFGS_B2 =
    
        val l_u = if l_u_ == null then (VectorD.fill (n)(-1), VectorD.fill (n)(1))
                  else l_u_
        new LBFGS_B2 (f, exactLS, l_u, gradF)
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

end LBFGS_B2

import LBFGS_B2.makeBounds

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGS_B2` the class implements the Limited memory Broyden–Fletcher–
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
 *  @param gradF    vector to vector functional formula for computing the gradiant, if available
 */
class LBFGS_B2 (f: FunctionV2S,
                exactLS: Boolean = false,
                private var l_u: Bounds = (null, null),
                gradF: FunctionV2V = null)
      extends Minimizer:

    private val debug           = debugf ("LBFGS_B2", false)      // debug function
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
     *  @see void getGeneralizedCauchyPoint(const TProblem &problem, const TVector &x, const TVector &g, TVector &x_cauchy, VariableTVector &c)
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
        var fDoublePrime  = max (-theta * fPrime - (p.dot (mm * p)), EPSILON)
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
            fDoublePrime  = max (EPSILON * f_dp_orig, fDoublePrime)

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
        end for
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
     *  @see void SubspaceMinimization(const TProblem &problem, TVector &x_cauchy, TVector &x, VariableTVector &c, TVector &g, TVector &SubspaceMin)
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
    def setHistorySize (hs_ : Int): Unit = { hs = hs_ }

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

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The objective function f.  Option:  plus a weighted penalty based on the
     *  constraint function g.  FIX:  better penalty scheme needed.
     *  @param x  the coordinate values of the current point
     */
    inline override def fg (x: VectorD): Double = f(x)
/*
        val f_x = f(x)
        if g == null then                             // unconstrained
            f_x
        else                                          // constrained, g(x) <= 0
            val penalty = if ineq then max (g(x), 0.0) else abs (g(x))
            f_x + abs (f_x) * WEIGHT * penalty * penalty
    end fg
*/

// FIX -- allow other line search algorithms such as MoreThuente

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform an exact `GoldenSectionLS` or inexact `WolfeLS` Line Search.
     *  Search in direction dir, returning the distance z to move in that direction.
     *  @param x     the current point
     *  @param dir   the direction to move in
     *  @param step  the initial step size
     */
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double =
        debug ("linesearch", s"x = $x, dir = $dir, step = $step")

        def f_1D (z: Double): Double = fg(x + dir * z)          // create a 1D function
        val ls = if exactLS then new GoldenSectionLS (f_1D )    // Golden Section Line Search
                 else new WolfeLS (f_1D)                        // Wolfe line search ((c1 = .0001, c2 = .9)
        ls.search (step)                                        // perform a Line Search
    end lineSearch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the following Non-Linear Programming (NLP) problem using L-BFGS_B:
     *      min { f(x) | g(x) <= 0 }.
     *  @see void minimize(TProblem &problem, TVector &x0)
     *  @param x0         the starting point
     *  @param alphaInit  the initial step size
     *  @param toler      the tolerance
     */
    def solve (x0: VectorD, alphaInit: Double = STEP, toler: Double = EPSILON): FuncVec =
        debug ("solve", s"x0 = $x0, alphaInit = $alphaInit, toler = $toler")

        var best = (MAX_VALUE, VectorD.nullv)

        dim   = x0.dim
        theta = 1.0
        if l_u == null then l_u = makeBounds (dim, NEGATIVE_INFINITY, POSITIVE_INFINITY)
//      val (l, u) = l_u

        ww = new MatrixD (dim, 0)                               // FIX - causes empty matrix warning
//      mm = new MatrixD (0, 0)                                 // find alt. to zero dimension matrix
        mm = LBFGS_B2.emptyMatrix

        val yHistory =  ArrayBuffer [VectorD] ()
        val sHistory =  ArrayBuffer [VectorD] ()
        var yHistoryMx: MatrixD = null
        var sHistoryMx: MatrixD = null

//      var (x, gr)  = (x0, ∇ (fg)(x0))                          // FIX -- differs from C++ code, could be okay
        var (x, gr)  = (x0, (if gradF != null then gradF (x0)    // by formula
                             else ∇ (fg)(x0)))                   // numerically
        var fv       = fg(x)                                     // functional value at x
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
    
                // STEP 2: compute the cauchy point
                val (xCauchy, c) = getGCP (x, gr)
                forceBounds (xCauchy)                              // FIX -- not in C++ code

                // STEP 3: compute a search direction d_k by the primal method for the sub-problem
                val subspaceMin = subspaceMinimize (x, gr, xCauchy, c)
                forceBounds (subspaceMin)                          // FIX -- not in C++ code

                // STEP 4: perform linesearch 
// FIX -- C++ sets alphaInit = 1
                val rate = lineSearch (x, subspaceMin-x, alphaInit)  // FIX -- try MoreThuente

                // STEP 5: compute gradient
                x = x - (x - subspaceMin) * rate                   // update current guess and function information
                forceBounds (x)                                    // clampToBound

                fv = fg(x)
                if blown ((fv, x)) then { best = better ((f_old, x_old), best); break () }

//              gr  = ∇ (fg)(x)
                gr  = if gradF != null then gradF (x)              // by formula
                      else ∇ (fg)(x)                               // numerically
                mgn = getMgn (x, gr)
                if mgn < toler || count > countMax then { best = better ((fv, x), best); break () }
                if abs (mgn - mgn_old) < toler then count += 1

                val newY = gr - g_old                                       // prepare for next iteration
                val newS = x - x_old

                // STEP 6
                val test = abs (newS.dot (newY))
                if test > EPSILON * newY.normSq then
// FIX -- does not look the same as C++ code
                    if yHistory.size >= hs then { yHistory.remove (0); sHistory.remove (0) }
                    yHistory append newY
                    sHistory append newS

                    // STEP 7
                    theta = newY.dot (newY) / newY.dot (newS)
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

end LBFGS_B2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_B2Test` main function is used to test the `LBFGS_B2` class.
 *      f(x) = (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.lBFGS_B2Test
 */
@main def lBFGS_B2Test (): Unit =

    val n  = 2
    val x0 = new VectorD (n)
    def f (x: VectorD): Double = (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1

    banner ("Minimize (no bounds): (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    var optimizer = new LBFGS_B2 (f)
    var opt = optimizer.solve (x0)
    println (s"o][ optimal solution (x, f(x)) = $opt")

    banner ("Minimize (bounds): (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    val lu = makeBounds (x0.dim, 3.5, 5.0)
    optimizer = new LBFGS_B2 (f, l_u = lu)
    opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

end lBFGS_B2Test


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_B2Test2` main function is used to test the `LBFGS_B2` class.
 *      f(x) = x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.lBFGS_B2Test2
 */
@main def lBFGS_B2Test2 (): Unit =

    val n  = 2
    val x0 = new VectorD (n)
    def f (x: VectorD): Double = x(0)~^4 + (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1

    banner ("Minimize (no bounds): x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    var optimizer = new LBFGS_B2 (f)
    var opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

    banner ("Minimize (bounds): x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    val lu = makeBounds (x0.dim, 3.5, 5.0)
    optimizer = new LBFGS_B2 (f, l_u = lu)
    opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

end lBFGS_B2Test2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lBFGS_B2Test3` main function is used to test the `LBFGS_B2` class.
 *      f(x) = 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1
 *  > runMain scalation.optimization.quasi_newton.lBFGS_B2Test3
 */
@main def lBFGS_B2Test3 (): Unit =

    val n  = 2
    val x0 = VectorD (0.1, 0.0)
    def f (x: VectorD): Double = 1/x(0) + x(0)~^4 + (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1

    banner ("Minimize (no bounds): 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    var optimizer = new LBFGS_B2 (f)
    var opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

    opt = optimizer.resolve (n)
    println (s"][ optimal solution (x, f(x)) = $opt")

    banner ("Minimize (bounds): 1/x_0 + x_0^4 + (x_0 - 3)^2 + (x_1 - 4)^2 + 1")
    val lu = makeBounds (x0.dim, 3.5, 5.0)
    optimizer = new LBFGS_B2 (f, l_u = lu)
    opt = optimizer.solve (x0)
    println (s"][ optimal solution (x, f(x)) = $opt")

end lBFGS_B2Test3

