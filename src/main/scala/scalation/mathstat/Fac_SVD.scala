
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Khalid Jahangeer, John Miller
 *  @version 2.0 
 *  @date    Sun Mar 19 15:16:20 EDT 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Singular Value Decomposition (SVD) Matrix Factorization
 *           produces Compact SVD; for Full SVD see `fullSVD` method 
 *  @see eecs16b.org/notes/sp24/note15.pdf
 *
 *  @see Matrix Computations:  Algorithm 8.6.1 Golub-Kahan SVD Step Algorithm
 *  @see Matrix Computations:  Algorithm 8.6.2 SVD Algorithm
 *
 *  This version translated from the following Algol code:
 *  @see people.duke.edu/~hpgavin/SystemID/References/Golub+Reinsch-NM-1970.pdf
 */

package scalation
package mathstat

import scala.math.{abs, hypot, sqrt}
import scala.util.control.Breaks.{break, breakable}

/** Factor type contains 'u, s, v' which are the left orthogonal matrix, the diagonal
 *  matrix/vector containing singular values and the right orthogonal matrix.
 */
type FactorType     = (MatrixD, VectorD, MatrixD)
type FactorTypeFull = (MatrixD, MatrixD, MatrixD)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_SVD` class is used to compute the Singular Value Decomposition (SVD) of
 *  matrix a using the Golub-Kahan-Reinsch Algorithm.
 *  Factor/decompose matrix 'a' into the product of three matrices:
 *      a = u * q * v.t
 *  where
 *     u is an m-by-n matrix of
 *         orthogonal eigenvectors of a * a.t (LEFT SINGULAR VECTORS)
 *     q is an n-by-n diagonal matrix of square roots of
 *         eigenvalues of a.t * a & a * a.t (SINGULAR VALUES)
 *     v is an n-by-n matrix of
 *         orthogonal eigenvectors of a.t * a (RIGHT SINGULAR VECTORS)
 * The SVD algorithm implemented is based on plane rotations.  It involves transforming
 * the input matrix to its Bidiagonal form using the Householder transformation procedure.
 * The Bidiagonal matrix is then used to obtain singular values using the QR algorithm.
 *------------------------------------------------------------------------------
 *  @param a  the m-by-n matrix to factor/decompose (requires m >= n)
 */
class Fac_SVD (a: MatrixD)
    extends Factorization:

    private val flaw  = flawf ("Fac_SVD")                         // flaw function
    private val debug = debugf ("Fac_SVD", false)                 // debug function

    private val MAX_ITER       = 100                              // maximum number of iterations
    private val m              = a.dim                            // number of rows
    private val n              = a.dim2                           // number of columns

    if n > m then flaw ("init", s"Fac_SVD implementation requires m = $m >= n = $n")

    private var l              = 0                                // lower index vs. k for zeroing out super-diagonal elements
    private var f, g           = 0.0                              // typically [ f g ]
    private var h              = 0.0                              //           [ 0 h ]
    private var bmx            = 0.0                              // maximum column magnitude in the bidiagonal matrix
    private var test_fconverge = true                             // whether singular values have reached converging magnitude
    private var eps            = EPSILON                          // adjustable small value

    private var mat3: FactorType = null                           // result of factorization: 3 matrices 

// FIX: make naming of matrices consistent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor/deflate the matrix by iteratively turning elements not in the main
     *  diagonal to zero.  Save factorization result in mat3.
     */
    def factor (): Fac_SVD = { if mat3 == null then mat3 = factor123 (); this }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the two factored matrices.
     */
    def factors: (MatrixD, MatrixD) =
        throw new UnsupportedOperationException ("Fac_SVD has three, not two factors")
    end factors

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factor matrix a into the product of a matrix of left singular vectors u,
     *  a vector of singular values q and a matrix of right singular vectors v
     *  such that a = u *~ q * v.t.
     */
    def factor123 (): FactorType =
        if mat3 != null then return mat3                          // matrix a has already been factored

        val bid       = new Bidiagonal (a)                        // class for making bidiagonal matrices 
        val (u, _, v) = bid.bidiagonalize ()                      // factor a into a bidiagonal matrix b in (u, b, v)
        val (e, q)    = bid.e_q                                   // get b's super-diagonal e and main diagonal q
        bmx           = bid.bmax                                  // largest column magnitude in b
        eps          *= bmx                                       // adjust eps based on bmx
        var c, s      = 0.0                                       // cosine, sine for rotations
        var y         = 0.0                                       // holds values from q and u
        var z         = 0.0                                       // holds values from q, v and normalization of (f, h)

        for k <- n-1 to 0 by -1 do iterate (k)                    // diagonalization of the bidiagonal form

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Iterate until the super-diagonal element e(k) is (near) zero.
         *  @param k  the upper index (decremented by the loop above)
         */
        def iterate (k : Int): Unit =
            var converged = false
            var iter = 0
            cfor (! converged && iter < MAX_ITER, iter += 1) {
                testFSplitting (k, e, q)
                debug ("iterate", s"l = $l \t test_fconverge = $test_fconverge")

                if ! test_fconverge then cancellation (l, k)

                if testFConvergence (l, k) then
                    debug (s"iterate", s"for k = $k, converged after ${iter+1} iterations")
                    debug (s"iterate", s"e = $e \nq = $q")
                    converged = true
                else
                    shiftFromBottom (k, e, q)
                    qrTransform (l, k)
            } // cfor
        end iterate

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Test whether the lower index l has caught the upper index k.
         *  @param l  the lower index
         *  @param k  the upper index
         */
        def testFConvergence (l : Int, k : Int): Boolean =
            debug ("testFConvergence", s"(l, k) = ($l, $k)")

            z = q(k)
            if l == k then                                        // convergence indicated by l equaling k
                if z < 0.0 then                                   // make sure singular value is non-negative
                    q(k) = -z
                    for j <- 0 until n do v(j, k) = -v(j, k)
                true
            else false
        end testFConvergence

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Cancellation (set to zero) of super-diagonal element e(l) if l > 0.
         *  Requires test_fconverge to be false
         *  @param l  the lower index
         *  @param k  the upper index
         */
        def cancellation (l : Int, k : Int): Unit =
            c = 0.0; s = 1.0                                      // set cosine, sine
            var converged = false
            var j = l
            cfor (! converged && j <= k, j += 1) {                // each column l to k
                f     = s * e(j)                                  // sine * e(j)
                e(j) *= c                                         // cosine * e(j)

                if abs (f) <= eps then                            // f near zero => return & test f convergence
                    converged = true
                else
                    g = q(j); h = hypot (f, g)                    // hypotenuse/norm
                    q(j) = h
                    c = g / h; s = -f / h                         // reset cosine, sine
                    rotateU (l-1, j)                              // rotation for columns l-1 and j of u
                end if
            } // cfor
        end cancellation

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Perform next QR transformation.
         *  @param l  the lower index
         *  @param k  the upper index
         */
        def qrTransform (l : Int, k : Int): Unit =
            c = 1.0; s = 1.0                                      // set cosine, sine
            for j <- l+1 to k do                                  // each column l+1 to k
                g = e(j); h = s * g; g *= c                       // compute g, h
                y = q(j); z = hypot (f, h)                        // hypotenuse/norm
                e(j-1) = z                                        // update e

                c = f / z; s = h / z                              // reset cosine, sine
                f =  bmx * c + g * s                              // compute f
                g = -bmx * s + g * c; h = y * s                   // compute g, h
                y *= c
                rotateV (j-1, j)                                  // update v

                z = hypot (f, h)                                  // hypotenuse/norm
                q(j-1) = z                                        // update q

                c   = f / z; s = h / z                            // reset cosine, sine
                f   =  c * g + s * y                              // compute f
                bmx = -s * g + c * y
                rotateU (j-1, j)                                  // update u
            end for
            e(l) = 0.0
            e(k) = f
            q(k) = bmx
        end qrTransform

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Perform rotation for columns j1 and j2 in the u matrix.
         *  @param j1  the first column involved 
         *  @param j2  the second column involved 
         */
        def rotateU (j1: Int, j2: Int): Unit =
            for i <- 0 until m do                                 // each row of u
                y = u(i, j1)                                      // changes to y and z affect outer scope
                z = u(i, j2)
                u(i, j1) =  y * c + z * s
                u(i, j2) = -y * s + z * c
            end for
        end rotateU

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Perform rotation for columns j1 and j2 in the v matrix.
         *  @param j1  the first column involved 
         *  @param j2  the second column involved 
         */
        def rotateV (j1: Int, j2: Int): Unit =
            for i <- 0 until n do                                 // each row of v
                bmx = v(i, j1)                                    // changes to y and z affect outer scope
                z   = v(i, j2)
                v(i, j1) =  bmx * c + z * s
                v(i, j2) = -bmx * s + z * c
            end for
        end rotateV

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /*  Shift from bottom 2x2 minor.
         *  @param k  the upper index
         *  @param e  the super-diagonal
         *  @param q  the main diagonal
         */
        def shiftFromBottom (k : Int, e: VectorD, q: VectorD): Unit =
            bmx = q(l)
            y   = q(k-1)
            g   = e(k-1); h = e(k)
            f   = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y)
            g   = hypot (f, 1.0)
            f   = if f < 0 then ((bmx - z) * (bmx + z) + h * (y / (f - g) - h)) / bmx
                  else ((bmx - z) * (bmx + z) + h * (y / (f + g) - h)) / bmx

            debug ("shiftFromBottom", s"f = $f \t g = $g")
        end shiftFromBottom

        // final part for factor method
        flip (u, q)                                               // convert singular values to positive, if any, adjusting u accordingly
        reorder ((u, q, v))                                       // reorder so largest singular values come first
        (u, q, v)                                                 // return left matrix, singular vector and right matrix
    end factor123

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Flip negative singular values to positive and set singular values close
     *  to zero to zero.
     *  @param u  the left orthogonal matrix
     *  @param s  the vector of singular values
     */
    def flip (u: MatrixD, s: VectorD): Unit =
        for i <- s.indices do
            if abs (s(i)) < TOL then s(i) = 0.0                   // zero out
            if s(i) < 0.0 then { u(i) *= -1.0; s(i) *= -1.0 }     // flip sign
        end for
    end flip

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Flip negative main diagonal elements in the singular vectors to positive.
     *  @param u  the left orthogonal matrix
     *  @param v  the right orthogonal matrix
     */
    def flip (u: MatrixD, v: MatrixD): Unit =
        for j <- u.indices2 if u(j, j) < 0.0 do u(?, j) = u(?, j) * -1.0
        for j <- v.indices2 if v(j, j) < 0.0 do v(?, j) = v(?, j) * -1.0
    end flip

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reorder the singular values to be in non-increasing order.  Must swap
     *  singular vectors in lock step with singular values.  To minimize the
     *  number of swaps, selection sort is used.
     *  @param ft  the factored matrix (u, s, v)
     */
    def reorder (ft: FactorType): Unit =
        val n = ft._2.dim
        for i <- 0 until n do
            val j = ft._2.argmax (i , n)                          // index of largest element in s(i:n)
            if i != j then
                ft._1.swapCol (i, j)                              // u left orthogonal matrix
                ft._2.swap (i, j)                                 // s diagonal matrix
                ft._3.swapCol (i, j)                              // v right orthogonal matrix
        end for
    end reorder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test super-diagonal element e(l) and main diagonal element q(l-1)
     *  to set the lower index l.
     *  @param k  the upper index
     *  @param e  the super-diagonal
     *  @param q  the main diagonal
     */
    def testFSplitting (k : Int, e: VectorD, q: VectorD): Unit =
        breakable {
            for ll <- k to 0 by -1 do
                l = ll                                            // make global index l track loop variable ll
                test_fconverge = false
                if abs (e(ll)) <= eps then
                    debug ("Fac_SVD", s"e(ll) = ${e(ll)}")
                    test_fconverge = true
                    break ()
                if abs (q(ll-1)) <= eps then break ()
            end for
        } // breakable
    end testFSplitting

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for x in a^t*a*x = b using `Fac_SVD`.
     *  @param b  the constant vector
     */
    def solve (b: VectorD): VectorD =
        val (u, d, vt) = factor123 ()                             // factor using SVD
        val alpha = u.transpose * b                               // principle component regression
//      vt *~ d.recip * alpha                                     // estimate coefficients
        vt *~ recip (d) * alpha                                   // estimate coefficients - handles 0's
    end solve

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Efficient calculation of inverse matrix a^-1 from existing factorization.
     *      a * a^-1 = I
     */
    def inverse: MatrixD =
        val (u, d, vt) = factor123 ()                             // factor using SVD
        vt *~ d.recip * u.transpose
    end inverse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the condition number of 'this' matrix, i.e., the ratio of the
     *  largest singular value to the smallest.  Note, if not of full rank, it
     *  will be infinity.
     */
    def conditionNum: Double = { val s = factor123 ()._2; s(0) / s(s.dim -1) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the reciprocal of the given vector avoiding divide by zero.
     *  To handle the problem of a singular matrix, the singular values which are
     *  zero are left zero.  (This function is similar to 'VectorD.recip ()' except
     *  for the zeros part.)
     *  @see www.cs.princeton.edu/courses/archive/fall11/cos323/notes/cos323_f11_lecture09_svd.pdf
     *  @param d  the vector of singular values
     */
    def recip (d: VectorD): VectorD =
        val c = new VectorD (d.dim)
        for i <- d.indices do c(i) = if d(i) == 0.0 then 0.0 else 1.0 / d(i)
        c
    end recip

end Fac_SVD


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_SVD` object provides several test matrices as well as methods
 *  for making full representations, reducing dimensionality, determining rank
 *  and testing SVD factorizations.
 */
object Fac_SVD:

    val a1 = MatrixD ((2, 2), 1.00,  2.00,                        // original matrix
                              0.00,  2.00)                        // 2 by 2, bidiagonal

    val a2 = MatrixD ((3, 2), 3.0, -1.0,                          // original matrix
                              1.0,  3.0,                          // 3 by 2
                              1.0,  1.0)

    val a3 = MatrixD ((3, 3), 1.0, 1.0, 0.0,                      // original matrix
                              0.0, 2.0, 2.0,                      // 3 by 3, bidiagonal
                              0.0, 0.0, 3.0)

    val a4 = MatrixD ((3, 3), 0.0,     1.0, 1.0,                  // original matrix
                              sqrt(2), 2.0, 0.0,                  // 3 by 3
                              0.0,     1.0, 1.0)

    val a5 = MatrixD ((4, 4), 0.9501, 0.8913, 0.8214, 0.9218,     // original matrix
                              0.2311, 0.7621, 0.4447, 0.7382,     // 4 by 4
                              0.6068, 0.4565, 0.6154, 0.1763,
                              0.4860, 0.0185, 0.7919, 0.4057)

    val a6 = MatrixD ((3, 2), 4, 5,                               // original matrix
                              6, 7,                               // 3 by 2
                              9, 8)

    val a7 = MatrixD ((4, 4), 1.0, 2.0, 3.0, 4.0,                 // original matrix
                              4.0, 3.0, 2.0, 1.0,                 // 4 by 4
                              5.0, 6.0, 7.0, 8.0,
                              8.0, 7.0, 6.0, 5.0)

    val a8 = MatrixD ((5, 3), 0.44444444,  0.3333333, -1.3333333,   // original matrix
                              0.41111111, -0.3166667, -0.3333333,   // 5 by 3
                             -0.18888889,  0.4833333, -0.3333333,
                             -0.03333333, -0.6500000,  1.0000000,
                             -0.63333333,  0.1500000,  1.0000000)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert an SVD factoring to a full matrix representation, returning the result as
     *  three matrices having the following dimensions: U (m-by-n), S (n-by-n) and V (n-by-n).
     *  This is the compact SVD representation, see the next methods of a full SVD representation.
     *  @param u_s_v  the 3-way factorization
     */
    def factorFull (u_s_v: FactorType): FactorTypeFull =
        val s  = u_s_v._2                                          // vector of singular values
        val ss = new MatrixD (s.dim, s.dim); ss(?, ?) = s          // turn vector into diagonal matrix
        (u_s_v._1, ss, u_s_v._3)
    end factorFull

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a full SVD factorization (rather than a compact one) where the three
     *  matrices have the following dimensions: U (m-by-m), S (m-by-n) and V (n-by-n).
     *  @see https://eecs16b.org/notes/sp24/note15.pdf
     *  @param a  the m-by-n matrix to factor/decompose (requires m >= n)
     */
    def fullSVD (a: MatrixD): FactorTypeFull =
        val (m, n) = a.dims
        val svd = new Fac_SVD (a)
        val (u, s, v) = factorFull (svd.factor123 ())             // factor matrix a
        if m - n > 0 then
            var uu = u
            cfor (0, m - n) { _ => uu = uu :^+ gsOrtho (uu) }     // append an orthogonal column
            (uu,
             s ++ (new MatrixD (m - n, n)),                       // append rows of zeroes
             v)                                                   // use original v
        else 
            (u, s, v)
    end fullSVD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new normalized m-dimensional vector that is orthogonal to all
     *  columns in an orthonormal matrix u of dimension m-by-n (m > n),
     *  Use the Gram-Schmidt (SG) Orthogonalization Algorithm.
     *      v = a - Σ (a ⋅ uᵢ) * uᵢ
     *  @see https://www.cis.upenn.edu/~cis6100/Gram-Schmidt-Bjorck.pdf
     *  @param u  the given m-by-n orthonormal matrix
     */
    def gsOrtho (u: MatrixD): VectorD =
        val m  = u.dim
        val ut = u.transpose
        val a  = VectorD.one (m)
        val vs = VectorD (m)
        cfor (0, m) { i => vs += ut(i) * (a dot ut(i)) }
        val v = a - vs 
        v / v.norm
    end gsOrtho

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reduce the dimensionality of the u, s and v matrices from n to k.
     *  If k = rank, there is no loss of information; when k < rank, multiplying
     *  the three matrices results in an approximation (little is lost so long as
     *  the singular values set to zero (i.e., clipped) are small).
     *  @param u_s_v  the 3-way factorization
     *  @param k      the desired dimensionality
     */
    def reduce (u_s_v: FactorType, k: Int): FactorType =
        (u_s_v._1(?, 0 until k),                                  // slice columns from matrix u
         u_s_v._2(0 until k),                                     // slice elements from vector s
         u_s_v._3(?, 0 until k))                                  // slice columns from matrix v
    end reduce

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the rank of a matrix by counting the number of non-zero singular
     *  values.  The implementation assumes zero singular values are last in the vector.
     *  @param s  the vector of singular values for the matrix whose rank is sought
     */
    def rank (s: VectorD): Int =
        var i = 0
        while i < s.dim && s(i) != 0.0 do i += 1
        i
    end rank

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the SVD Factorization algorithm on matrix a by factoring the matrix
     *  into a left matrix u, a vector s, and a right matrix v.  Then multiply back
     *  to recover the original matrix u *~ s * v.t.
     *  @param a      the original matrix
     *  @param u_s_v  the given matrix a factored into three components
     *  @param name   the name of the test case
     */
    def test (a: MatrixD, svd: Fac_SVD, name: String): Unit =
        banner (name)
        println (s"factor matrix a = $a")
        val (u, s, v) = svd.factor123 ()                          // factor matrix a
        println (sline () + s"into (u, s, v) = ${(u, s, v)}")
        val prod = u *~ s * v.transpose                           // compute the product
        println (sline () + s"check: u *~ s * v.t = $prod")       // should equal the original a matrix
        println (s"prod - a = ${prod - a}")                       // difference should be close to 0
        println (sline ())
        assert (prod == a)
        println (sline ())
    end test

end Fac_SVD

import Fac_SVD._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_SVDTest` main function is used to test the `Fac_SVD` class.
 *  @see www.ling.ohio-state.edu/~kbaker/pubs/Singular_Value_Decomposition_Tutorial.pdf
 *  > runMain scalation.mathstat.fac_SVDTest
 */
@main def fac_SVDTest (): Unit =

    test (a1, new Fac_SVD (a1), "fac_SVDTest on a1")
    test (a2, new Fac_SVD (a2), "fac_SVDTest on a2")

end fac_SVDTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_SVDTest2` main function is used to test the `Fac_SVD` class.
 *  > runMain scalation.mathstat.fac_SVDTest2
 */
@main def fac_SVDTest2 (): Unit =

    test (a3, new Fac_SVD (a3), "fac_SVDTest2 on a3")
    test (a4, new Fac_SVD (a4), "fac_SVDTest2 on a4")

end fac_SVDTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_SVDTest3` main function is used to test the `Fac_SVD` class.
 *  > runMain scalation.mathstat.fac_SVDTest3
 */
@main def fac_SVDTest3 (): Unit =

    test (a5, new Fac_SVD (a5), "fac_SVDTest3 on a5")
    test (a6, new Fac_SVD (a6), "fac_SVDTest3 on a6")

end fac_SVDTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_SVDTest4` main function is used to test the `Fac_SVD` class.
 *  Test case a8 currently fails as ScalaTion does not consider 9.18964e-09
 *  to be approximately zero.
 *  > runMain scalation.mathstat.fac_SVDTest4
*/
@main def fac_SVDTest4 (): Unit =

    test (a7, new Fac_SVD (a8), "fac_SVDTest4 on 7")
    test (a8, new Fac_SVD (a7), "fac_SVDTest4 on 8")

end fac_SVDTest4

