
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Mustafa Nural, John Miller
 *  @version 2.0
 *  @date    Fri May 26 14:32:21 EDT 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Lower-Upper (LU) Matrix Factorization
 *
 *  LU Factorization for square (n-by-n) and rectangular (m-by-n, m > n) matrices.
 *  Uses partial (row) pivoting that uses largest possible pivot for each column.
 *
 *  Also computes inverse, determinant, rank and condition number
 *
 *  Code adapted from Jama LUDecomposition.java
 *  @see github.com/fiji/Jama/blob/master/src/main/java/Jama/LUDecomposition.java
 */

package scalation
package mathstat

import scala.collection.mutable.Set
import scala.math.{abs, min, signum}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_LU` class provides methods to factor an m-by-n matrix into its
 *  lower and upper triangular products:
 *      A  = LU   when partial pivoting is not needed
 *      PA = LU   where P is the permutation matrix
 *      A  = QLU  where Q = P.inverse
 *  where a is the given matrix, l is an m-by-n lower triangular matrix, and
 *  u is an n-by-n upper triangular matrix.  The permutation matrix is represented
 *  by the piv vector.  Once factored, can be used to solve a system of linear
 *  equations.
 *     Solve for x in Ax = b: Ax = QLUx = b => LUx = Pb using steps (1) and (2)
 *     (1) Solve Ly = Pb  using forward substitution for y
 *     (2) Solve Ux = y   using backward substitution for x
 *  @param a  the given m-by-n rectangular matrix
 */
class Fac_LU (a: MatrixD)
    extends Factorization:

    private val debug  = debugf ("Fac_LU", false)               // debug function
    private val flaw   = flawf ("Fac_LU")                       // flaw function
    private val (m, n) = (a.dim, a.dim2)                        // (# rows, # columns) in matrix a

    if m < n then flaw ("init", s"requires m = $m >= n = $n")

    private val l          = a.copy                             // deep copy of matrix a to be factored
                                                                // initally holds both l and u, proper l after split
    private var u: MatrixD = null                               // the right upper triangular matrix
    private var pivsign    = 1                                  // initial value for pivot sign (used in det method)
    private val piv        = VectorI.range (0, m)               // the initial values for pivots

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor matrix a into the product of l and u.
     */
    def factor (): Fac_LU =
        if u != null then return this                           // matrix a already factored          

        cfor (0, l.dim2) { j =>                                 // for each column j
            val col_j = l(?, j)                                 // vector (copied from lu) holding column j

            cfor (0, l.dim) { i =>                              // for each row i
                val row_i = l(i)                                // vector (in l) holding row i
                val kmax  = min (i, j)
                var sum   = 0.0                                 // compute dot product truncated at kmax
                cfor (0, kmax) { k => sum += row_i(k) * col_j(k) }
                col_j(i) -= sum                                 // decrement by dot product
                row_i(j)  = col_j(i)                            // row i col j = col j row i
            } // cfor

            var p = j                                           // find pivot for column j
            cfor (j+1, m) { i =>
                if abs (col_j(i)) > abs (col_j(p)) then p = i
            } // cfor

            if p != j then
                debug ("factor", s"swap rows $j and $p")
                l.swap (p, j)                                   // swap rows j and p
                piv.swap (p, j)                                 // also swap in pivot vector
                pivsign = -pivsign
            end if

            if l(j, j) != 0.0 then                              // compute multipliers for l
                cfor (j+1, m) { i => l(i, j) /= l(j, j) }
        } // cfor

        factored = true
        split ()                                                // split l into l proper and u
        this
    end factor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split l into lower and upper triangular matrices by placing the upper
     *  portion in u and clearing, so l is properly lower triangular.
     */
    def split (): Unit =
        u = l.upper                                             // extract upper triangle including main diagonal
        cfor (0, n) { i =>
            cfor (i, n) { j => l(i, j) = is (i == j) }          // clear entries above main diagonal and set main diagonal to one
        } // cfor
    end split

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the lower l and upper u matrices.
     */
    def factors: (MatrixD, MatrixD) = (l, u)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Permute matrix c, equivalent to qc, i.e., multiplying by the inverse
     *  of the permutation matrix p.
     *  @param c  the matrix to permute
     */
    def permute (c: MatrixD): MatrixD =
        val swapped = Set [Int] ()
        cfor (0, n) { j =>
            if j != piv(j) && ! (swapped contains j) then
                val pj = piv(j)
                debug ("permute", s"swap rows $j and $pj")
                swapped += pj
                c.swap (j, pj)
        } // cfor
        c
    end permute

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Permute vector d, equivalent to pd, i.e., multiplying by the permutation matrix p.
     *  @param d  the vector to permute
     */
    def permute (d: VectorD): VectorD =
        val swapped = Set [Int] ()
        cfor (0, n) { j =>
            if j != piv(j) && ! (swapped contains j) then
                val pj = piv(j)
                debug ("permute", s"swap elements $j and $pj")
                swapped += pj
                d.swap (j, pj)
        } // cfor
        d
    end permute

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for x in the equation l*u*x = b via l*y = b and u*x = y.
     *  Return the solution vector x.
     *  @param b  the constant vector
     */
    def solve (b: VectorD): VectorD =
        if m != n then throw new IllegalArgumentException ("solve: requires a square matrix")

//      val bb = b.copy; permute (bb)                           // make a copy of b and permute by piv
        val bb = b(piv)                                         // permute b the same way as l 
        val y = new VectorD (l.dim2)                            // forward substitution
        cfor (0, y.dim) { k =>                                  // solve for y in l*y = bb
            val l_k = l(k)
            var sum = 0.0
            cfor (0, k) { j => sum += l_k(j) * y(j) }
            y(k) = bb(k) - sum
        } // cfor
        debug ("solve", s"y = $y")
        bsolve (y)
    end solve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for x using back substitution in the equation u*x = y where
     *  matrix u is upper triangular.
     *  @param y  the constant vector
     *  FIX - replace with /~
     */
    def bsolve (y: VectorD): VectorD =
        val x = new VectorD (u.dim2)                            // vector to solve for
        for k <- x.dim - 1 to 0 by -1 do                        // solve for x in u*x = y
            val u_k = u(k)
            var sum = 0.0
            cfor (k+1, u.dim2) { j => sum += u_k(j) * x(j) }
            x(k) = (y(k) - sum) / u(k, k)
        end for
        debug ("bsolve", s"x = $x")
        x
    end bsolve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the inverse of matrix a from the LU Factorization.
     */
    def inverse: MatrixD =
        if m != n then throw new IllegalArgumentException ("inverse: matrix must be square");

        factor ()
        val inv = MatrixD.eye (n, n)                            // inverse matrix - starts as identity
        cfor (0, a.dim2) { j => inv(?, j) = solve (inv(?, j)) }
        inv
    end inverse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the determinant of matrix a.  The value of the determinant
     *  indicates, among other things, whether there is a unique solution to a
     *  system of linear equations (a nonzero determinant).
     */
    def det: Double =
        if m != n then throw new IllegalArgumentException ("det: matrix must be square");

        var dt = pivsign.toDouble
        cfor (0, l.dim2) { j => dt *= u(j, j) }
        dt
    end det

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the rank (number of independent columns) of 'm-by-n' matrix 'a', by
     *  counting the number of non-zero diagonal elements in 'u'.
     *  If 'rank < n', then 'a' is singular.
     *  @see en.wikipedia.org/wiki/Rank_%28linear_algebra%29
     *  @see `Fac_QR_RR` or `SVD`
     */
    def rank: Int = n - u(?).countZero                          // u(?) return main diagonal

end Fac_LU


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_LU` companion object provides functions related to LU Factorization.
 */
object Fac_LU:

    private val flaw = flawf ("Fac_LU")                         // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for x in the equation a*x = b using LU Factorization. 
     *  Return the solution vector x.
     *  @param a  the matrix A holding the coefficients of the equations
     *  @param b  the constant vector
     */
    def solve_ (a: MatrixD, b: VectorD): VectorD =
        val lu = new Fac_LU (a)
        lu.factor ()
        lu.solve (a * b)
    end solve_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for x in the equation a*x = b in an over determined system of
     *  linear equation using least squares.  Return the solution vector x.
     *  @see people.csail.mit.edu/bkph/articles/Pseudo_Inverse.pdf
     *  @param a  the matrix A holding the coefficients of the equations
     *  @param b  the constant vector
     */
    def solveOver (a: MatrixD, b: VectorD): VectorD =
        val at = a.transpose
        val lu = new Fac_LU (at * a)
        lu.factor ()
        lu.solve (at * b)
    end solveOver

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for x in the equation a*x = b in an under determined system of
     *  linear equation by finding the smallest solution.  Return the solution
     *  vector x.
     *  @see people.csail.mit.edu/bkph/articles/Pseudo_Inverse.pdf
     *  @param a  the matrix A holding the coefficients of the equations
     *  @param b  the constant vector
     */
    def solveUnder (a: MatrixD, b: VectorD): VectorD =
        val at = a.transpose
        val lu = new Fac_LU (a * at)
        lu.factor ()
        at * lu.solve (b)
    end solveUnder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve a system of linear equations a*x = b.
     *  @param a   the matrix A holding the coefficients of the equations
     *  @param lu  LU Factorization of matrix A
     *  @param b   the constant vector
     */
    def solve (a: MatrixD, lu: Fac_LU, b: VectorD): VectorD =
        val (m, n) = (a.dim, a.dim2)
        if m == n     then lu.solve (b)                         // square
        else if m > n then solveOver (a, b)                     // more equations than variables
        else solveUnder (a, b)                                  // fewer equations then variables
    end solve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute an estimate of the L1 norm of this matrix, i.e., maximum absolute
     *  column sum.  It uses an adapted version of Hager's algorithm.
     *  @author Michael Cotterell
     *  @see Algorithm 4.1 in HIGHAM1998
     *  @see Higham, N.J. "Fortran Codes for Estimating the One-Norm of a
     *       Real or Complex Matrix, with Applications to Condition Estimation."
     *       ACM Trans. Math. Soft., 14, 1988, pp. 381-396.
     *  @see www.maths.manchester.ac.uk/~higham/narep/narep135.pdf
     *  @param a     the matrix A whose norm is sought
     *  @param a_lu  LU Factorization of matrix A
     *  @param inv   whether or not to compute for inverse (default true)
     */
    def norm1est (a: MatrixD, a_lu: Fac_LU, inv: Boolean = true): Double =
        if a.dim != a.dim2 then flaw ("norm1est", "the matrix must be square")

        if ! a_lu.isFactored then a_lu.factor ()                // factor matrix a
        val at    = a.transpose
        val at_lu = new Fac_LU (a); at_lu.factor ()             // factor the transpose matrix a

        val e  = a(0); e.set (1.0 / a.dim2)
        var v  = if inv then a_lu.solve (e) else a * e
        if a.dim2 == 1 then return v.norm

        var γ    = v.norm1
        var ξ    = v.map (signum (_))
        var x    = if inv then at_lu.solve (ξ) else at * ξ
        var k    = 2
        var done = false
        val ITER = 5

        while ! done do
            val j = x.abs.argmax ()
            cfor (0, e.dim) { k => e(k) = is (k == j) }         // one in jth position
            v     = if inv then a_lu.solve (e) else a(?, j)
            val g = γ
            γ     = v.norm1

            if v.map (signum (_)) == ξ || γ <= g then
                cfor (0, x.dim) { i =>
                    x(i) = (if i % 2 == 0 then -1.0 else 1.0) * (1.0 + ((i - 1.0) / (x.dim - 1))) }
                x = if inv then a_lu.solve (x) else a * x
                if (2.0 * x.norm1) / (3.0 * x.dim) > γ then { v = x; γ = (2.0 * x.norm1) / (3.0 * x.dim) }
            end if

            ξ  = v.map (signum (_))
            x  = if inv then at_lu.solve (ξ) else at * ξ
            k += 1
            if x.norm1 == x(j) || k > ITER then done = true
        end while
        γ
    end norm1est

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute an estimate of the L1 norm of 'this' matrix, i.e., maximum absolute
     *  column sum.  It uses an adapted version of Hager's algorithm.
     *  @param a     the matrix A whose norm is sought
     *  @param inv   whether or not to compute for inverse (default true)
     */
    def norm1est (a: MatrixD, inv: Boolean): Double = norm1est (a, new Fac_LU (a), inv)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the condition number of matrix a, which equals
     *      ||a|| ||b||  where b = a.inverse
     *  Requires a to be a square matrix.
     *  For rectangular matrices, @see `Fac_SVD`.
     *  @param a     the matrix whose condition number is sought
     *  @param a_lu  LU Factorization of matrix A
     */
    def conditionNum (a: MatrixD, a_lu: Fac_LU): Double =
        if a.dim != a.dim2 then flaw ("conditionNum", "the matrix must be square")

        val nrm1 = a.norm1                                      // norm of matrix a
        val nrm2 = a_lu.inverse.norm1                           // norm of a^-1
        nrm1 * nrm2
    end conditionNum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the condition number of matrix 'a', which equals
     *      ||a|| ||b||  where b = a.inverse
     *  Requires 'a' to be a square matrix.
     *  For rectangular matrices, @see Fac_`SVD`.
     *  @param a     the matrix whose condition number is sought
     *  @param a_lu  LU Factorization of matrix A
     */
    def conditionNum2 (a: MatrixD, a_lu: Fac_LU): Double =
        val aa = a.copy                                         // FIX - changes a unless copied
        if a.dim != a.dim2 then flaw ("conditionNum", "the matrix must be square")

        val nrm1 = aa.norm1                                     // norm of matrix a
        val nrm2 = norm1est (aa, a_lu)                          // estimate of norm of a^-1
        nrm1 * nrm2
    end conditionNum2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the LU Factorization of matrix a into l and u and its usage
     *  in solving a system of linear equations.
     *  @param a  the matrix A to be factored
     *  @param b  the constant vector in Ax = b
     */
    def test (a: MatrixD, b: VectorD): Unit =
        println (s"a = $a")
        println (s"b = $b")
        val n = a.dim2

        banner ("Factor A into L and U using LU Factorization")

        val lu = new Fac_LU (a.copy)
        lu.factor ()                                            // factor matrix A into L and U
        val (l, u) = lu.factors
        println (s"(l, u) = ($l, $u)")
        println (s"rank = ${lu.rank}")                          // rank of matrix A

        if a.dim == a.dim2 then
            println (s"cond = ${conditionNum (a, lu)}")         // condition number of matrix A
            println (s"con2 = ${conditionNum2 (a, lu)}")        // condition number of matrix A

            banner ("Solve for x in Ax = b using LUx = Pb")

            val x  = lu.solve (b)                               // solve for x
            println (s"a   = $a")
            val ax = a * x                                      // compute Ax
            println (s"x   = $x")
            println (s"a*x = $ax")
            println (s"b   = $b")
            println (s"a*x - b = ${ax - b}")
//          assert (ax =~ b)                                    // ensure Ax = Pb

            banner ("Verify that A * A^-1 = I")

            val I   = MatrixD.eye (n, n)
            val ai  = lu.inverse                                // inverse of A
            val aai = a * ai                                    // multiply A and A^-1
            println (s"ai = $ai")
            println (s"aai = $aai")
            println (s"aai - I = ${aai - I}")                   // ensure A * A^-1 = I
            assert (ai =~ inverse (a)(lu))                      // ensure both inverses are the same

            banner ("Test det and inverse functions") 

            println (s"det (a)     = ${det (a)(lu)}")
            println (s"inverse (a) = ${inverse (a)(lu)}")
        end if

        banner ("Verfify that A = QLU")

        val qlu = l * u                                         // multiply L and U
        lu.permute (qlu)                                        // permute l * u -> QLU
        println (s"q*l*u = $qlu")
        println (s"a - q*l*u = ${a - qlu}")
        assert (a =~ qlu)                                       // ensure A = QLU
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the inverse of matrix a by calling the inverse method in `Fac_LU`.
     *  Note: `Fac_LU.inverse` is generally faster and more robust than `Fac_Inv.inverse`.
     *  Usage: inverse (a)
     *  Usage: inverse (a)(lu)           // shortcut when lu is already computed
     *  @param a   the matrix whose inverse is sought
     *  @param lu  an LU factorization (use existing or make a new one)
     */
    def inverse (a: MatrixD)(lu: Fac_LU = new Fac_LU (a)): MatrixD =
        if ! lu.isFactored then lu.factor ()
        lu.inverse
    end inverse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the determinant of matrix a.  The value of the determinant
     *  indicates, among other things, whether there is a unique solution to a
     *  system of linear equations (a nonzero determinant).
     *  Usage: det (a)
     *  Usage: det (a)(lu)               // shortcut when lu is already computed
     *  @param a   the matrix whose determinant is sought
     *  @param lu  an LU factorization (use existing or make a new one)
     */
    def det (a: MatrixD)(lu: Fac_LU = new Fac_LU (a)): Double = 
        if ! lu.isFactored then lu.factor ()
        lu.det
    end det

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose matrix a looking for high correlation, high condition number,
     *  lower than expected rank, zero variance columns (there should only be one).
     *  @param a  the matrix to diagnose
     */
    def diagnoseMat (a: MatrixD): Unit =
        val lu = new Fac_LU (a)

        banner ("diagnoseMat: Matrix Dimensions")
        println (s"a.dim = ${a.dim}, a.dim2 = ${a.dim2}")

        banner ("diagnoseMat: Matrix Rank")
        println (s"lu.rank = ${lu.rank}")

        banner ("diagnoseMat: Matrix Condition Number")
        println (s"conditionNum  (a, lu) = ${conditionNum  (a, lu)}")
        println (s"conditionNum2 (a, lu) = ${conditionNum2 (a, lu)}")

        banner ("diagnoseMat: Correlation Matrix")
        println (s"a.corr = ${a.corr}")

        banner ("diagnoseMat: Variance of Matrix Columns")
        cfor (0, a.dim2) { j => println (s"a(?, $j).variance = ${a(?, j).variance}") }
    end diagnoseMat

end Fac_LU

import Fac_LU._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_LUTest` main function is used to test the `Fac_LU` class.
 *  Test a Square Matrix.
 *  > runMain scalation.mathstat.fac_LUTest
 */
@main def fac_LUTest (): Unit =

    val a = MatrixD ((4, 4), 16.0,  0.4,  0.8, -0.2,
                              1.0,  1.0,  2.0, -0.12,           // (1, 1) 3.0 -> 1.0 for pivoting
                              0.28, 1.0, 24.8, -0.12,
                              0.28, 1.4,  4.35, 1.0)

    val b = VectorD (-0.2, -0.32, 13.52, 14.17)

    test (a, b)

end fac_LUTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_LUTest2` main function is used to test the `Fac_LU` class.
 *  Test a Rectangular Matrix.
 *  > runMain scalation.mathstat.fac_LUTest2
 */
@main def fac_LUTest2 (): Unit =

    val a = MatrixD ((5, 4), 16.0,  0.4,  0.8,  -0.2,
                              1.0,  3.0,  2.0,  -0.12,
                              0.28, 1.0, 24.8,  -0.12,
                              9.2,  1.4,  1.0, -10.2,
                              0.28, 1.4,  4.35,  1.0)

    val b = VectorD (-0.2, -0.32, 13.52, 14.17, 35)

    test (a, b)

end fac_LUTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_LUTest3` main function is used to test the `Fac_LU` class.
 *  > runMain scalation.mathstat.fac_LUTest3
 */
@main def fac_LUTest3 (): Unit =

    val x = MatrixD ((2, 2), 1, 3,
                             2, 1)
    val y = VectorD (1, 7)

    println (s"x = $x")
    println (s"y = $y")
    println ("using LU factorization: Lb = Uy = " + { val lu = new Fac_LU (x); lu.factor ().solve (y) } )

end fac_LUTest3

