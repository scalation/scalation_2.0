
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jan 22 15:18:51 EST 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Matrix Bidiagonalization
 *
 *  @see Matrix Computations:  Algorithm 5.4.2 Householder Bidiagonalization
 *
 *  This version translated from the following Algol code:
 *  @see people.duke.edu/~hpgavin/SystemID/References/Golub+Reinsch-NM-1970.pdf
 */

package scalation
package mathstat

import math.{abs, sqrt}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Bidiagonal` class is used to create a bidiagonal matrix from matrix a.
 *  It uses the Householder Bidiagonalization Algorithm to compute orthogonal
 *  matrices u and v such that 
 *      u.t * a * v = b
 *      a  =  u * b * v.t
 *  where matrix b is bidiagonal, i.e., it will only have non-zero elements on
 *  its main diagonal and its super-diagonal (the diagonal above the main).
 *      u is an m-by-n matrix
 *      b is an n-by-n matrix (bidiagonal)
 *      v is an n-by-n matrix
 *------------------------------------------------------------------------------
 *  @param a  the m-by-n matrix to bidiagonalize (requires m >= n)
 */
class Bidiagonal (a: MatrixD):

    private val flaw = flawf ("Bidiagonal")

    private val m = a.dim                     // the number of rows in matrix a
    private val n = a.dim2                    // the number of columns in matrix a

    if n > m then flaw ("init", s"Bidiagonal requires m = $m >= n = $n")

    private val u  = a.copy                   // initialize left orthogonal matrix to m-by-n matrix
    private val v  = new MatrixD (n, n)       // initialize right orthogonal matrix to n-by-n matrix
    private val e  = new VectorD (n)          // super-diagonal for b
    private val q  = new VectorD (n)          // main diagonal for b
    private val b  = new MatrixD (n, n)       // n-by-n matrix from e and q diagonals
    private var bm = 0.0                      // maximum column magnitude from the bidiagonal matrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum column magnitude from bidiagonal matrix b.
     */
    def bmax: Double = bm

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the super-diagonal e and main diagonal q as vectors.
     */
    def e_q: (VectorD, VectorD) = (e, q)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sliced dot product.  Take the dot product of two row/column
     *  vectors starting from the 'from' parameter.
     *  @param v1    the first vector
     *  @param v2    the second vector
     *  @param from  the offset from which to compute the sliced dot product
     */
    def sdot (v1: VectorD, v2: VectorD, from: Int = 0): Double =
        var sum = 0.0
        for i <- from until v1.dim do sum += v1(i) * v2(i)
        sum
    end sdot

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Bidiagonalize matrix a using the Householder Bidiagonalization Algorithm
     *  to compute orthogonal matrices u and v such that u.t * a * v = b
     *  where matrix b is bidiagonal.
     */
    def bidiagonalize (): (MatrixD, MatrixD, MatrixD) =
        var f, g = 0.0                                  // typically [ f  g ]
        var h    = 0.0                                  //           [ 0  h ]
        for i <- 0 until n do

            val l = i + 1                               // set control index l
            e(i)  = g                                   // assign ith super-diagonal element
            var s = sdot (u(?, i), u(?, i), i)          // u(i:m,i) dot u(i:m,i)

            if s < TOL then g = 0.0
            else
                f = u(i, i)
                g = if f < 0.0 then sqrt (s) else -sqrt (s)
                h = f * g - s; u(i, i) = f - g
                for j <- l until n do
                    s = sdot (u(?, i), u(?, j), i)
                    f = s / h
                    for k <- i until m do u(k, j) += f * u(k, i)
            end if

            q(i) = g                                     // assign ith main diagonal element
            s = sdot (u(i), u(i), l)                     // u(i,l:n) dot u(i,l:n)

            if s < TOL then g = 0.0
            else
                f = u(i, i+1)
                g = if f < 0 then sqrt (s) else -sqrt (s)
                h = f * g - s; u(i, i+1) = f - g
                for j <- l until n do e(j) = u(i, j) / h
                for j <- l until m do
                    s = sdot (u(i), u(j), l)
                    for k <- l until n do u(j, k) += s * e(k)
            end if

            val y = abs (q(i)) + abs (e(i)); if y > bm then bm = y
        end for
 
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Accumulate right-hand side transformations to create 'v' matrix.
         */
        def transformRHS (): Unit =
            var l = n - 1
            for i <- n-1 to 0 by -1 do
                if g != 0.0 then
                    h = u(i, i+1) * g
                    for j <- l until n do v(j, i) = u(i, j) / h
                    for j <- l until n do
                        val s = sdot (u(i), v(?, j), l)
                        for k <- l until n do v(k, j) += s * v(k, i)
                end if
                for j <- l until n do { v(i, j) = 0.0; v(j, i) = 0.0 }
                v(i, i) = 1.0
                g = e(i)
                l = i
            end for
        end transformRHS

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Accumulate left-hand side transformations to update 'u' matrix.
         */
        def transformLHS (): Unit =
            for i <- n-1 to 0 by -1 do
                val l = i + 1
                g = q(i)
                for j <- l until n do u(i, j) = 0.0
                if g != 0.0 then
                    h = u(i, i) * g
                    for j <- l until n do
                        val s = sdot (u(?, i), u(?, j), l)
                        f = s / h
                        for k <- i until m do u(k, j) += f * u(k, i)
                    end for
                    for j <- i until m do u(j, i) /= g
                else
                    for j <- i until m do u(j, i) = 0.0
                u(i, i) += 1.0
            end for
        end transformLHS

        transformRHS ()                         // transform the RHS to finalize matrix v
        transformLHS ()                         // transform the LHS to finalize matrix u
        b.setDiag (e(1 until n), 1)             // put the super-diagonal into matrix b
        b(?, ?) = q                             // put the main diagonal into matrix b
        (u, b, v)                               // return the three matrices
    end bidiagonalize

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test whether the product of factorization equals the orginal matrix.
     */
    def test (): Unit =
        val prod = u * b * v.transpose           // compute the product of the three matrices

        println (s"u    = $u")
        println (s"b    = $b")
        println (s"v    = $v")
        println (s"prod = $prod")
        assert (a == prod)
    end test

end Bidiagonal


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bidiagonalTest` main function is used to test the `Bidiagonal` class.
 *  bidiagonalization answer = [ (21.8, -.613), (12.8, 2.24, 0.) ]
 *  @see books.google.com/books?isbn=0801854148  (p. 252)
 *  > runMain scalation.mathstat.bidiagonalTest
 */
@main def bidiagonalTest (): Unit =

    val a = MatrixD ((4, 3), 1.0,  2.0,  3.0,     // orginal matrix
                             4.0,  5.0,  6.0,
                             7.0,  8.0,  9.0,
                            10.0, 11.0, 12.0)
    println (s"a = $a")

    val bid = new Bidiagonal (a)                      // Householder bidiagonalization
    bid.bidiagonalize ()                              // bidiagonalize a
    bid.test ()                                       // test the product u * b * v.t

end bidiagonalTest

