
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Sameer Gaherwar, John Miller
 *  @version 2.0
 *  @date    Sun Sep 16 14:09:25 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Cholesky Matrix Factorization
 */

package scalation
package mathstat

import scala.math.sqrt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_Cholesky` class provides methods to factor an n-by-n symmetric,
 *  positive definite matrix a into the product of two matrices:
 *      l   - an n-by-n left lower triangular matrix
 *      l.t - an n-by-n right upper triangular matrix - transpose of l
 *  such that a = l * l.t.
 *  @param a  the symmetric, positive definite matrix to be factored
 */
class Fac_Cholesky (a: MatrixD)
      extends Factorization:
 
    private val flaw = flawf ("Fac_Cholesky")        // flaw function
    private val EPS  = 1e-12                         // number close to zero
    private val n    = a.dim                         // the matrix should be n-by-n

    if n != a.dim2     then flaw ("init", "matrix a must be square")
    if ! a.isSymmetric then flaw ("init", "matrix a must be symmetric")

    private val l    = new MatrixD (n, n)            // for factored lower triangular matrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor matrix a into the product of l and l.t using Cholesky
     *  Factorization a = l * l.t, where l.t is l's transpose.
     *  This is a more robust Cholesky Factorization algorithm.
     *  Adpated from Scala 2 code returned by GPT-3 on the query:
     *      "more robust Cholesky Factorization algorithm in Scala"
     *  Also #see math.stackexchange.com/questions/418945/cholesky-decomposition-in-positive-semi-definite-matrix
     */
    def factor (): Fac_Cholesky =
        if factored then return this                  // matrix a is already factored

        for j <- 0 until n do
            var diff = a(j, j)
            for k <- 0 until j do diff -= sq (l(j, k))
    
            if diff > EPS then
                l(j, j) = sqrt (diff)

                for i <- j + 1 until n do
                    var sum = 0.0
                    for k <- 0 until j do sum += l(i, k) * l(j, k)
                    l(i, j) = (a(i, j) - sum) / l(j, j)
                end for
            else
                l(j, j) = sqrt (EPS)
        end for
        factored = true
        this
    end factor

    // For even more robust algorithms:
    // @see apps.dtic.mil/sti/pdfs/ADA205174.pdf
    // @see www.applied-mathematics.net/identification/papersLARS/_cholesky_shnabell.pdf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor matrix a into the product of l and l.t using Cholesky
     *  Factorization a = l * l.t, where l.t is l's transpose.
     *  It uses the Cholesky–Crout algorithm.
     */
    def factor_ (): Fac_Cholesky =
        if factored then return this                  // matrix a is already factored

        for j <- 0 until n do
            var diff = a(j, j)
            for k <- 0 until j do diff -= sq (l(j, k))

            if diff > 0.0 then
                l(j, j) = sqrt (diff)

                for i <- j + 1 until n do
                    var sum = 0.0
                    for k <- 0 until j do sum += l(i, k) * l(j, k)
                    l(i, j) = (a(i, j) - sum) / l(j, j)
                end for
            else
                flaw ("factor", s"sqrt of negative diff = $diff, setting l(j, j) to zero")
                l(j, j) = 0.0
        end for
        factored = true
        this
    end factor_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor matrix a into the product of l and l.t using Cholesky
     *  Factorization a = l * l.t, where l.t is l's transpose.
     *  It uses the Cholesky–Banachiewicz algorithm.
     *  @see introcs.cs.princeton.edu/java/95linear
     */
    def factor__ (): Fac_Cholesky =
        if factored then return this                  // matrix a is already factored

        for i <- 0 until n; j <- 0 to i do
            var sum = 0.0
            for k <- 0 until j do sum += l(i, k) * l(j, k)

            val diff = a(i, j) - sum
            if i == j then
                if diff < 0.0 then flaw ("factor", s"sqrt of negative diff = $diff")
                l(j, j) = sqrt (diff)
            else
                val l_jj = l(j, j)
                if l_jj == 0.0 then flaw ("factor", s"divide by zero l($j, $j) = $l_jj")
                l(i, j) = diff / l_jj
        end for
        factored = true
        this
    end factor__

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return both the lower triangular matrix l and its transpose l.t.
     */
    def factors: (MatrixD, MatrixD) = (l, l.transpose)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use the lower triangular matrix l from the Cholesky Factorization to
     *  solve a system of equations a * x = b. Return the solution x using
     *  forward and backward substitution.
     *  @param b  the constant vector
     */
    def solve (b: VectorD): VectorD =
        val y = new VectorD (n)
        for k <- 0 until n do                         // solve for y in l*y = b
            y(k) = (b(k) - (l(k) dot y)) / l(k, k)
        end for

        val x = new VectorD (n)
        for k <- n-1 to 0 by -1 do                    // solve for x in l.t*x = y
            x(k) = (y(k) - (l(?, k) dot x)) / l(k, k)
        end for
        x
    end solve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Efficient calculation of inverse matrix a^-1 from existing factorization.
     *      a * a^-1 = I
     */
    def inverse: MatrixD =
//      factor ()                                     // uses Cholesky–Banachiewicz algorithm
        factor_ ()                                    // uses Cholesky–Crout algorithm
        val l_inv = Fac_Inverse.inverse_lt (l)
        l_inv.transpose * l_inv                       //  l^-1^t * l^-1
    end inverse

end Fac_Cholesky


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_CholeskyTest` main function tests the `Fac_Cholesky` class.
 *  It compare the Cholesky and LU Factorization algorithms.
 *  @see ece.uwaterloo.ca/~dwharder/NumericalAnalysis/04LinearAlgebra/cholesky
 *  > runMain scalation.mathstat.fac_CholeskyTest
 */
@main def fac_CholeskyTest (): Unit =

    banner ("Compare the Cholesky and LU Factorization Algorithms")

    val a = MatrixD ((4, 4), 4.0,  0.4,   0.8, -0.2,
                             0.4,  1.04, -0.12, 0.28,
                             0.8, -0.12,  9.2,  1.4,
                            -0.2,  0.28,  1.4,  4.35)

    val b = VectorD (-0.2, -0.32, 13.52, 14.17)

    println (s"a = $a")
    println (s"b = $b")

    banner ("Cholesky Factorization")
    val chol = new Fac_Cholesky (a)
    chol.factor ()
    println ("factors = " + chol.factors)
    println ("solve   = " + chol.solve (b))

    banner ("LU Factorization")
    val lu = new Fac_LU (a)
    lu.factor ()
    println ("factors = " + lu.factors)
    println ("solve   = " + lu.solve (b))

end fac_CholeskyTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_CholeskyTest2` main function tests the `Fac_Cholesky` class.
 *  It compares the three Cholesky Factorization algorithms.
 *  > runMain scalation.mathstat.fac_CholeskyTest2
 */
@main def fac_CholeskyTest2 (): Unit = 

    banner ("Compare the Three Cholesky Factorization Algorithms")

    val a = MatrixD ((3, 3),  4.0,  12.0, -16.0,
                             12.0,  37.0, -43.0,
                            -16.0, -43.0,  98.0)

    val b = VectorD (1, 2, 3)

    println (s"a = $a")
    println (s"b = $b")

    banner ("Cholesky Factorization: factor")
    val chol = new Fac_Cholesky (a)
    chol.factor ()
    println ("factors = " + chol.factors)
    println ("solve   = " + chol.solve (b))

    banner ("Cholesky Factorization: factor_")
    chol.reset ()
    chol.factor_ ()
    println ("factors = " + chol.factors)
    println ("solve   = " + chol.solve (b))

    banner ("Cholesky Factorization: factor__")
    chol.reset ()
    chol.factor__ ()
    println ("factors = " + chol.factors)
    println ("solve   = " + chol.solve (b))

end fac_CholeskyTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_CholeskyTest3` main function tests the `Fac_Cholesky` class.
 *  It compares the three Cholesky Factorization algorithms on challenging problems.
 *  @see apps.dtic.mil/sti/pdfs/ADA205174.pdf
 *  @see www.applied-mathematics.net/identification/papersLARS/_cholesky_shnabell.pdf
 *  > runMain scalation.mathstat.fac_CholeskyTest3
 */
@main def fac_CholeskyTest3 (): Unit = 

    banner ("Compare the Three Cholesky Factorization Algorithms")

    val a = MatrixD ((3, 3), 1, 1, 2,
                             1, 2, 3,
                             2, 3, 1)

    val b = VectorD (1, 2, 3)

// Very Hard Problem

/*
    val a = MatrixD ((4, 4),  1890.3, -1705.6, -315.8,  3000.3,
                             -1705.6,  1538.3,  284.9, -2706.6,
                              -315.8,   284.9,   52.5,  -501.2,
                              3000.3, -2706.6, -501.2,  4760.8)

    val b = VectorD (1, 2, 3, 4)
*/

    println (s"a = $a")
    println (s"b = $b")

    banner ("Cholesky Factorization: factor")
    val chol = new Fac_Cholesky (a)
    chol.factor ()
    println ("factors = " + chol.factors)
    println ("solve   = " + chol.solve (b))

    banner ("Cholesky Factorization: factor_")
    chol.reset ()
    chol.factor_ ()
    println ("factors = " + chol.factors)
    println ("solve   = " + chol.solve (b))

    banner ("Cholesky Factorization: factor__")
    chol.reset ()
    chol.factor__ ()
    println ("factors = " + chol.factors)
    println ("solve   = " + chol.solve (b))

    banner ("LU Factorization")
    val lu = new Fac_LU (a)
    lu.factor ()
    println ("factors = " + lu.factors)
    println ("solve   = " + lu.solve (b))

end fac_CholeskyTest3

