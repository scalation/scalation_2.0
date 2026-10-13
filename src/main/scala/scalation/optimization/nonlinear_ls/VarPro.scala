
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sun Oct  4 10:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Variable Projection (VarPro) for Separable Nonlinear Least Squares (NLS)
 *           AI Assisted Code
 *
 *  @see Golub, G. H. and Pereyra, V. (1973), "The Differentiation of Pseudo-Inverses and
 *       Nonlinear Least Squares Problems Whose Variables Separate", SIAM J. Numerical
 *       Analysis 10 (2), 413-432.
 *  @see Kaufman, L. (1975), "A Variable Projection Method for Solving Separable Nonlinear
 *       Least Squares Problems", BIT 15, 49-57.
 *  @see O'Leary, D. P. and Rust, B. W. (2013), "Variable Projection for Nonlinear Least
 *       Squares Problems", Computational Optimization and Applications 54 (3), 579-593.
 */

package scalation
package optimization
package nonlinear_ls

import scala.math.{abs, max, sqrt}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VarPro` class solves SEPARABLE Nonlinear Least Squares problems by Variable
 *  Projection.  The model is linear in some parameters and nonlinear in the others,
 *
 *      y  =  b_0 + phi (a) b + e
 *
 *  e.g., y = b_0 + b_1 x^2 + b_2 log (x + a): b_0, b_1, b_2 are LINEAR parameters
 *  and the shift a is a NONLINEAR parameter.  For any a the best linear parameters
 *  are the solution of a linear least squares problem, b (a), so they are eliminated
 *  and only the k nonlinear parameters are searched:
 *
 *      minimize over a in [lb, ub]   || y - phi (a) b (a) ||^2 + lambda || b (a) ||^2
 *
 *  Compared with one optimizer over all the parameters, the search is in k rather
 *  than k + n dimensions, needs no starting values (and no bounds) for the linear
 *  parameters, and is far less sensitive to the starting point.
 *  The INNER problem is solved by QR factorization: the columns of phi (a) and of y
 *  are centered when `center` (an implicit, unpenalized intercept b_0, as in
 *  `CenteredRidge`), the columns of phi (a) are divided by their standard deviations
 *  when `scaled` (so the penalty does not depend on the units of a column, which the
 *  nonlinear parameters change), and a ridge penalty lambda is applied through the
 *  augmented system [ phi; sqrt (lambda) I ] b = [ y; 0 ], so the reduced problem
 *  stays a least squares problem in a.
 *  The OUTER problem is solved by `LevenbergMarquardt_TR` (bounds on a) on the
 *  reduced residual r (a) = [ y - phi (a) b (a); -sqrt (lambda) b (a) ], with the
 *  Jacobian of Kaufman's approximation when the derivative of the basis is given,
 *
 *      d r / d a_i  ~  - P (d phi / d a_i) b (a),     P = I - phi phi^+  (projector),
 *
 *  and by finite differences of r that stay inside the box otherwise.
 *  With several responses (the columns of y, e.g., one per forecasting horizon)
 *  the nonlinear parameters are SHARED and each response has its own linear
 *  parameters; the residuals are stacked response by response.
 *  The search is LOCAL: when the reduced objective has several local minima, the
 *  one found depends on the starting point.
 *  @param k       the number of nonlinear parameters
 *  @param phi     the basis function: nonlinear parameters a (size k) => m-by-n basis matrix
 *  @param y       the m-by-ny response matrix (one column per response)
 *  @param lb      the lower bounds on the nonlinear parameters (-infinity for none), lb_i < ub_i
 *  @param ub      the upper bounds on the nonlinear parameters (+infinity for none)
 *  @param dPhi    the optional derivative of the basis: a => array of k m-by-n matrices,
 *                     the i-th one holding d phi / d a_i (None => finite differences)
 *  @param lambda  the ridge penalty on the linear parameters (0 => ordinary least squares)
 *  @param center  whether to center the columns of phi (a) and y, i.e., fit an intercept
 *                     that is not penalized (defaults to true)
 *  @param scaled  whether to divide the columns of phi (a) by their standard deviations,
 *                     making the ridge penalty unit-free (defaults to false)
 */
class VarPro (k: Int, phi: VectorD => MatrixD, y: MatrixD,
              lb: VectorD, ub: VectorD,
              dPhi: Option [VectorD => Array [MatrixD]] = None,
              lambda: Double = 0.0, center: Boolean = true, scaled: Boolean = false)
      extends NonLinearLS ():

    private val debug   = debugf ("VarPro", true)                         // debug function
    private val flaw    = flawf ("VarPro")                                // flaw function
    private val FD_STEP = 1e-7                                            // relative step for finite differences

    private val ny  = y.dim2                                              // number of responses
    private val yMu = if center then y.mean else new VectorD (ny)         // response means (the intercept's part)
    private val yc  = if center then y - yMu else y                       // centered responses

    private var alpha: VectorD = null                                     // the nonlinear parameters found
    private var bb: MatrixD    = null                                     // the linear parameters at alpha (n-by-ny)
    private var b0: VectorD    = null                                     // the intercepts at alpha (one per response)
    private var steps: Map [String, Int] = Map ()                         // accepted steps of the outer solver, by kind

    if lb.dim != k || ub.dim != k then flaw ("init", s"bounds must have k = $k elements: lb.dim = ${lb.dim}, ub.dim = ${ub.dim}")
    if lambda < 0.0 then flaw ("init", s"the ridge penalty lambda = $lambda must be non-negative")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The reduced residual function r (a): the residuals of the best linear fit
     *  for nonlinear parameters a, stacked response by response (each block is
     *  followed by its penalty part -sqrt (lambda) b when lambda > 0).
     */
    val residualFn: VectorD => VectorD = (a: VectorD) => solveLinear (a).res

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The Jacobian of the reduced residual function: Kaufman's approximation
     *  when `dPhi` is given, finite differences otherwise.
     */
    val jacobianFn: Option [VectorD => MatrixD] =
        Some ((a: VectorD) => if dPhi.isDefined then kaufman (a) else finiteDiff (a))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The solution of the inner (linear) problem for given nonlinear parameters.
     *  @param b    the linear parameters (n-by-ny) for the centered/scaled basis
     *  @param res  the stacked residual vector
     *  @param pa   the system matrix (centered/scaled basis, augmented when lambda > 0)
     *  @param fac  the factorization of the system matrix
     *  @param ph   the basis matrix phi (a)
     *  @param sg   the column scales of the basis (all ones unless scaled)
     */
    private case class Inner (b: MatrixD, res: VectorD, pa: MatrixD, fac: Factorization,
                              ph: MatrixD, sg: VectorD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the column scales of a basis matrix: its standard deviations when
     *  scaled (a zero or undefined one is replaced by one, as in `CenteredRidge`),
     *  all ones otherwise.
     *  @param ph  the basis matrix
     */
    private def scales (ph: MatrixD): VectorD =
        if ! scaled then VectorD.one (ph.dim2)
        else ph.stdev.map (s => if s > 0.0 && ! s.isInfinite then s else 1.0)
    end scales

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the right-hand side of the inner problem for response j: the
     *  (centered) response, followed by n zeros when lambda > 0.
     *  @param j  the response (column of y)
     *  @param n  the number of basis columns
     */
    private def rhs (j: Int, n: Int): VectorD =
        if lambda > 0.0 then yc(?, j) ++ new VectorD (n) else yc(?, j)
    end rhs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the system matrix of the inner problem for a basis matrix: its
     *  columns centered when `center` and divided by their scales when `scaled`.
     *  @param ph  the basis matrix
     *  @param sg  the column scales of the basis (@see `scales`)
     */
    private def systemOf (ph: MatrixD, sg: VectorD): MatrixD =
        val pc = if center then ph - ph.mean else ph
        if scaled then pc / sg else pc
    end systemOf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the system matrix of the inner problem for nonlinear parameters a:
     *  the basis phi (a) with its columns centered (when `center`) and divided by
     *  their standard deviations (when `scaled`), i.e., the matrix that the
     *  linear parameters of `coefficients` multiply.
     *  @param a  the nonlinear parameters
     */
    def system (a: VectorD): MatrixD =
        val ph = phi (a)
        systemOf (ph, scales (ph))
    end system

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the responses as the inner problem sees them: centered when `center`.
     */
    def response: MatrixD = yc

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the linear parameters (n-by-ny, column j for response j) of the best
     *  fit for nonlinear parameters a in the units of the system matrix, so that
     *  response ~ system (a) * coefficients (a) (@see `linear` for the units of the basis).
     *  @param a  the nonlinear parameters
     */
    def coefficients (a: VectorD): MatrixD = solveLinear (a).b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the inner (linear) problem for nonlinear parameters a: build the
     *  system matrix (the centered/scaled basis, with sqrt (lambda) I stacked
     *  below it when lambda > 0), factor it and solve it for every response.
     *  @param a  the nonlinear parameters
     */
    private def solveLinear (a: VectorD): Inner =
        val ph  = phi (a)
        val sg  = scales (ph)
        val ps  = systemOf (ph, sg)
        val pa  = if lambda > 0.0 then ps ++ (MatrixD.eye (ps.dim2, ps.dim2) * sqrt (lambda)) else ps
        val fac = Fac_QR (pa)
        fac.factor ()
        val (ma, n) = (pa.dim, pa.dim2)
        val b_  = new MatrixD (n, ny)
        val res = new VectorD (ny * ma)
        for j <- 0 until ny do
            val yj = rhs (j, n)
            val bj = fac.solve (yj)                                       // least squares solution for response j
            val rj = yj - pa * bj
            b_(?, j) = bj
            for t <- 0 until ma do res(j * ma + t) = rj(t)
        end for
        Inner (b_, res, pa, fac, ph, sg)
    end solveLinear

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the derivative of the centered/scaled basis, given the derivative d
     *  of the basis phi (a) with respect to one nonlinear parameter.  With scaling,
     *  column j is q_j / sg_j (q_j the centered column), so its derivative is
     *  dq_j / sg_j - q_j dsg_j / sg_j^2, where the scale sg_j changes with a too.
     *  @param d   the derivative of the basis matrix with respect to one parameter
     *  @param in  the solution of the inner problem (its basis and scales)
     */
    private def dSystem (d: MatrixD, in: Inner): MatrixD =
        val dc = if center then d - d.mean else d
        if ! scaled then dc
        else
            val dev  = in.ph - in.ph.mean                                 // deviations: they define the scales
            val ddev = d - d.mean
            val q    = if center then dev else in.ph
            val ds   = new MatrixD (d.dim, d.dim2)
            for j <- d.indices2 do
                val ss  = dev(?, j).normSq
                val dsg = if ss > 0.0 then (dev(?, j) dot ddev(?, j)) * in.sg(j) / ss else 0.0
                ds(?, j) = dc(?, j) / in.sg(j) - q(?, j) * (dsg / (in.sg(j) * in.sg(j)))
            end for
            ds
        end if
    end dSystem

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Jacobian of the reduced residual by Kaufman's approximation:
     *  column i, block j is - P (d phi / d a_i) b_j, where P projects onto the
     *  orthogonal complement of the column space of the system matrix.
     *  @param a  the nonlinear parameters
     */
    private def kaufman (a: VectorD): MatrixD =
        val in  = solveLinear (a)
        val dd  = dPhi.get (a)
        val (ma, n) = (in.pa.dim, in.pa.dim2)
        val jac = new MatrixD (ny * ma, k)
        for i <- 0 until k do
            val di = dSystem (dd(i), in)                                  // derivative of the centered/scaled basis
            for j <- 0 until ny do
                val v  = di * in.b(?, j)
                val va = if lambda > 0.0 then v ++ new VectorD (n) else v
                val pv = va - in.pa * in.fac.solve (va)                   // projection P v
                for t <- 0 until ma do jac(j * ma + t, i) = -pv(t)
            end for
        end for
        jac
    end kaufman

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Jacobian of the reduced residual by finite differences: a
     *  forward difference, or a backward one where the forward step would leave
     *  the box, so the basis is never evaluated outside the bounds.
     *  @param a  the nonlinear parameters
     */
    private def finiteDiff (a: VectorD): MatrixD =
        val r0  = residualFn (a)
        val jac = new MatrixD (r0.dim, k)
        for i <- 0 until k do
            val h  = FD_STEP * max (1.0, abs (a(i)))
            val s  = if a(i) + h <= ub(i) then h else -h
            val a2 = a.copy
            a2(i)  = a(i) + s
            jac(?, i) = (residualFn (a2) - r0) / s
        end for
        jac
    end finiteDiff

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the value of the objective for nonlinear parameters a: the sum of
     *  squared errors of the best linear fit plus its ridge penalty.
     *  @param a  the nonlinear parameters
     */
    def objective (a: VectorD): Double = residualFn (a).normSq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the linear parameters (n-by-ny, column j for response j) and the
     *  intercepts (one per response; zero unless centered) of the best linear fit
     *  for nonlinear parameters a, in the units of the basis, so that
     *  y_j ~ intercept_j + phi (a) b_j.
     *  @param a  the nonlinear parameters
     */
    def linear (a: VectorD): (MatrixD, VectorD) =
        val in = solveLinear (a)
        val b_ = if scaled then (in.b.transpose / in.sg).transpose else in.b   // undo the column scaling
        val i_ = if center then yMu - (b_.transpose * in.ph.mean) else new VectorD (ny)
        (b_, i_)
    end linear

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the nonlinear parameters by minimizing the reduced residual with
     *  `LevenbergMarquardt_TR`, keep the linear parameters of the solution and
     *  return the nonlinear parameters.
     *  @param a0  the starting point for the nonlinear parameters (moved inside the bounds)
     */
    def optimize (a0: VectorD): VectorD =
        if k == 0 then
            alpha = new VectorD (0)                                       // nothing to search: a linear problem
        else
            val lm = new LevenbergMarquardt_TR (k, lb, ub, residualFn, jacobianFn)
            alpha  = lm.optimize (a0)
            steps  = lm.stepCounts
        end if
        val (b_, i_) = linear (alpha)
        bb = b_; b0 = i_
        debug ("optimize", s"nonlinear a = $alpha, objective = ${objective (alpha)}, steps = $steps")
        alpha
    end optimize

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the nonlinear parameters found by `optimize`.
     */
    def parameterNL: VectorD = alpha

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the linear parameters found by `optimize` (n-by-ny, column j for response j).
     */
    def parameterL: MatrixD = bb

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the intercepts found by `optimize` (one per response; zero unless centered).
     */
    def intercept: VectorD = b0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of accepted steps of the outer solver by kind in the last
     *  `optimize` run (@see `LevenbergMarquardt_TR.stepCounts`).
     */
    def stepCounts: Map [String, Int] = steps

end VarPro


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VarPro` companion object provides a factory method for a single response.
 */
object VarPro:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VarPro` solver for a single response vector.
     *  @param k       the number of nonlinear parameters
     *  @param phi     the basis function: nonlinear parameters a => m-by-n basis matrix
     *  @param y       the response vector
     *  @param lb      the lower bounds on the nonlinear parameters
     *  @param ub      the upper bounds on the nonlinear parameters
     *  @param dPhi    the optional derivative of the basis (None => finite differences)
     *  @param lambda  the ridge penalty on the linear parameters (0 => ordinary least squares)
     *  @param center  whether to center the columns of phi (a) and y (fit an intercept)
     *  @param scaled  whether to divide the columns of phi (a) by their standard deviations
     */
    def apply (k: Int, phi: VectorD => MatrixD, y: VectorD, lb: VectorD, ub: VectorD,
               dPhi: Option [VectorD => Array [MatrixD]] = None,
               lambda: Double = 0.0, center: Boolean = true, scaled: Boolean = false): VarPro =
        new VarPro (k, phi, MatrixD (y).transpose, lb, ub, dPhi, lambda, center, scaled)
    end apply

end VarPro


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `varProTest` main method tests the `VarPro` class on a model with two linear
 *  terms and one nonlinear parameter,
 *
 *      y = b_0 + b_1 x^2 + b_2 log (x + a),     (b_0, b_1, b_2, a) = (1, 0.5, 3, 2),
 *
 *  with noise-free data, so the parameters must be recovered.  It compares VarPro
 *  with Kaufman's Jacobian, VarPro with a finite-difference Jacobian and ONE
 *  optimizer (`LevenbergMarquardt_TR`) over all four parameters, from the same
 *  starting point for the nonlinear parameter (a = 8).
 *  > runMain scalation.optimization.nonlinear_ls.varProTest
 */
@main def varProTest (): Unit =

    val x    = VectorD.range (0, 60).map (0.5 + 0.1 * _)                  // x = 0.5, 0.6, ..., 6.4
    val beta = VectorD (1.0, 0.5, 3.0)                                    // the linear parameters b_0, b_1, b_2
    val a_   = 2.0                                                        // the nonlinear parameter (shift)
    val y    = x.map (z => beta(0) + beta(1) * z * z + beta(2) * math.log (z + a_))

    val lb = VectorD (-0.4)                                               // keeps x + a > 0 for every x
    val ub = VectorD (10.0)
    val a0 = VectorD (8.0)                                                // the starting point

    val basis = (a: VectorD) => MatrixD (x * x, (x + a(0)).log).transpose                 // columns x^2, log (x + a)
    val dBasis = (a: VectorD) =>
        Array (MatrixD (new VectorD (x.dim), (x + a(0)).recip).transpose)                 // d/da: 0, 1 / (x + a)

    banner ("VarPro with Kaufman's Jacobian")
    val vp1 = VarPro (1, basis, y, lb, ub, Some (dBasis))
    val a1  = vp1.optimize (a0)

    banner ("VarPro with a finite-difference Jacobian")
    val vp2 = VarPro (1, basis, y, lb, ub)
    val a2  = vp2.optimize (a0)

    banner ("ONE optimizer over all four parameters (b_0, b_1, b_2, a)")
    val jointRes = (w: VectorD) => y - (x * x * w(1) + (x + w(3)).log * w(2) + w(0))
    val ninf = Double.NegativeInfinity
    val pinf = Double.PositiveInfinity
    val lm   = new LevenbergMarquardt_TR (4, VectorD (ninf, ninf, ninf, -0.4), VectorD (pinf, pinf, pinf, 10.0), jointRes)
    val w    = lm.optimize (VectorD (0.0, 0.0, 0.0, 8.0))

    banner ("VarPro: results")
    println (s"true parameters:        b = $beta, a = $a_")
    println (s"VarPro (Kaufman):       b = ${vp1.intercept ++ vp1.parameterL(?, 0)}, a = $a1, " +
             s"sse = ${vp1.objective (a1)}, steps = ${vp1.stepCounts.values.sum}")
    println (s"VarPro (finite diff.):  b = ${vp2.intercept ++ vp2.parameterL(?, 0)}, a = $a2, " +
             s"sse = ${vp2.objective (a2)}, steps = ${vp2.stepCounts.values.sum}")
    println (s"one optimizer (4 par.): b = ${w(0 until 3)}, a = ${w(3)}, " +
             s"sse = ${jointRes (w).normSq}, steps = ${lm.stepCounts.values.sum}")

    assert (abs (a1(0) - a_) < 1e-5, s"VarPro (Kaufman) must recover a = $a_, got $a1")
    assert (abs (a2(0) - a_) < 1e-5, s"VarPro (finite differences) must recover a = $a_, got $a2")
    assert ((vp1.intercept ++ vp1.parameterL(?, 0) - beta).norm < 1e-4, "VarPro must recover the linear parameters")

end varProTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `varProTest2` main method tests the `VarPro` class with TWO responses that
 *  share the nonlinear parameters, a RIDGE penalty and no intercept: sums of two
 *  decaying exponentials,
 *
 *      y_j = b_1j exp (-a_1 t) + b_2j exp (-a_2 t),     (a_1, a_2) = (1, 0.2).
 *
 *  (1) With lambda = 0 the rates must be recovered.  (2) With lambda = 0.1 the
 *  minimum of the penalized objective found by VarPro must equal the one found
 *  by ONE optimizer over all six parameters on the same (augmented) residuals.
 *  > runMain scalation.optimization.nonlinear_ls.varProTest2
 */
@main def varProTest2 (): Unit =

    val t   = VectorD.range (0, 50).map (0.2 * _)                         // t = 0, 0.2, ..., 9.8
    val e1  = (t * -1.0).exp                                              // exp (-1.0 t)
    val e2  = (t * -0.2).exp                                              // exp (-0.2 t)
    val y   = MatrixD (e1 * 3.0 + e2, e1 + e2 * 2.0).transpose            // two responses (columns)
    val lam = 0.1                                                         // the ridge penalty of part (2)

    val lb = VectorD (0.01, 0.01)
    val ub = VectorD (5.0, 5.0)
    val a0 = VectorD (2.0, 0.5)                                           // the starting point

    val basis = (a: VectorD) => MatrixD ((t * -a(0)).exp, (t * -a(1)).exp).transpose
    val dBasis = (a: VectorD) =>
        Array (MatrixD ((t * -a(0)).exp * -t, new VectorD (t.dim)).transpose,              // d/da_1
               MatrixD (new VectorD (t.dim), (t * -a(1)).exp * -t).transpose)              // d/da_2

    banner ("(1) VarPro, two responses, lambda = 0, no intercept")
    val vp1 = new VarPro (2, basis, y, lb, ub, Some (dBasis), center = false)
    val a1  = vp1.optimize (a0)

    banner (s"(2) VarPro, two responses, lambda = $lam, no intercept")
    val vp2 = new VarPro (2, basis, y, lb, ub, Some (dBasis), lambda = lam, center = false)
    val a2  = vp2.optimize (a0)

    banner (s"(2) ONE optimizer over all six parameters, lambda = $lam")
    val jointRes = (w: VectorD) =>                                        // w = (a_1, a_2, b_11, b_21, b_12, b_22)
        val ph = basis (w(0 until 2))
        (y(?, 0) - ph * w(2 until 4)) ++ (w(2 until 4) * -sqrt (lam)) ++
        (y(?, 1) - ph * w(4 until 6)) ++ (w(4 until 6) * -sqrt (lam))
    val ninf = Double.NegativeInfinity
    val pinf = Double.PositiveInfinity
    val lm   = new LevenbergMarquardt_TR (6, VectorD (0.01, 0.01, ninf, ninf, ninf, ninf),
                                             VectorD (5.0, 5.0, pinf, pinf, pinf, pinf), jointRes)
    val w    = lm.optimize (VectorD (2.0, 0.5, 1.0, 1.0, 1.0, 1.0))

    banner ("VarPro, two responses: results")
    println (s"(1) true rates = VectorD(1.0, 0.2), VarPro a = $a1, sse = ${vp1.objective (a1)}")
    println (s"(1) linear parameters (column j for response j) = ${vp1.parameterL}")
    println (s"(2) VarPro:        a = $a2, objective = ${vp2.objective (a2)}, steps = ${vp2.stepCounts.values.sum}")
    println (s"(2) one optimizer: a = ${w(0 until 2)}, objective = ${jointRes (w).normSq}, steps = ${lm.stepCounts.values.sum}")

    assert ((a1 - VectorD (1.0, 0.2)).norm < 1e-5, s"VarPro must recover the rates (1.0, 0.2), got $a1")
    assert (abs (vp2.objective (a2) - jointRes (w).normSq) < 1e-6, "VarPro and the joint optimizer must reach the same penalized objective")

end varProTest2

