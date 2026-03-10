
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Zhaochong Liu
 *  @version 2.0
 *  @date    Thu Jun 17 19:29:23 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    QR Matrix Factorization Using Householder Orthogonalization
 */

package scalation
package mathstat

import scala.math.min

import MatrixD.eye

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_QR` class provides methods to factor an m-by-n matrix a into the
 *  product of two matrices:
 *      q - an m-by-n orthogonal matrix and
 *      r - an n-by-n right upper triangular matrix
 *  such that a = q * r.   It uses uses Householder orthogonalization.
 *  @see 5.1 and 5.2 in Matrix Computations
 *  @see QRDecomposition.java in Jama
 *  @see www.stat.wisc.edu/~larget/math496/qr.html
 *-------------------------------------------------------------------------------
 *  This implementation improves performance by working with the transpose the
 *  original matrix and reorders operations to facilitate parallelism (see par directory).
 *  Caveat: for m < n use `Fac_LQ`, @see `Fac_QR.apply`
 *-------------------------------------------------------------------------------
 *  @param aa     the matrix to be factored into q and r
 *  @param needQ  flag indicating whether a full q matrix is needed
 */
class Fac_QR (aa: MatrixD, needQ: Boolean = false)
      extends Factorization:

    private   val debug = debugf ("Fac_QR", false)             // debug function
    private   val flaw  = flawf ("Fac_QR")                     // flaw function
    protected val at    = aa.transpose                         // transpose (for efficiency) of matrix aa
    protected val m     = aa.dim                               // number of rows
    protected val n     = aa.dim2                              // number of columns
    protected val p     = min (m, n)                           // the smallest dimension
    protected val r     = MatrixD (n)                          // the right upper triangular r matrix
    private   val q     = if needQ then eye (m, n)
                          else null                            // the orthogonal q matrix

    if m < n then flaw ("init", s"case of m = $m < n = $n => use Fac_LQ instead of Fac_QR")

    debug ("init", s"aa = ${aa.dims}, needQ = $needQ, q = ${if q == null then null else q.dims}, r = ${r.dims}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform QR Factorization and store the result in a single matrix which contains
     *  Householder vectors of each column in the lower triangle of the aa matrix. 
     *  This algorithm uses Householder orthogonalization.
     *  @see 5.1 and 5.2 in Matrix Computations
     *  @see QRDecomposition.java in Jama
     */
    def factor (): Fac_QR =
        if factored then return this

        cfor (0, at.dim) { k => colHouse (k) }                 // perform k-th factoring step
        cfor (1, p) { j =>                                     // fill in rest of r matrix
            val at_j = at.v(j)
            cfor (0, j) { i => r(i, j) = at_j(i) }
        } // cfor
        if needQ then computeQ ()
//      r.clean (TOL)                                          // comment out to avoid cleaning r matrix

        factored = true
        this
    end factor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the k-th Householder vector and perform the k-th Householder
     *  reflection.
     *  @param k  the index of the column performing the Householder reflection
     */
    protected def colHouse (k: Int): Unit =
        val at_k  = at.v(k)                                    // k-th row of transpose is k-th column
        var _norm = at(k)(k until m).norm                      // norm of A(k:m, k) column vector
        if _norm != 0.0 then
            if at_k(k) < 0.0 then _norm = -_norm               // make k-th Householder vector
            cfor (k, m) { i => at_k(i) /= _norm }
            at_k(k) += 1.0
        r(k, k) = -_norm                                       // set the diagonal of r matrix

        cfor (k+1, p) { j =>                                   // transform all the rest of aa matrix
            val at_k = at.v(k)                                 // k-th column of aa matrix
            val at_j = at.v(j)                                 // j-th column of aa matrix
            var sum = 0.0
            cfor (k, m) { i => sum += at_k(i) * at_j(i) }
            if at_k(k) != 0.0 then sum /= - at_k(k)
            cfor (k, m) { i => at_j(i) += sum * at_k(i) }
        } // cfor
    end colHouse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return both the orthogonal q matrix and the right upper triangular r matrix.
     */
    def factors: (MatrixD, MatrixD) = (q, r)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the full q orthogonal matrix based on updated values in at.
     */
    def computeQ (): Unit =
        for k <- p-1 to 0 by -1 do
            if ! (at(k, k) =~ 0.0) then                           // nearly equal =~
                cfor (k, n) { j =>
                    val at_k = at.v(k)                            // k-th column of a is k-th row of at
                    var sum  = 0.0                                // update the elements in j column of q matrix
                    cfor (k, m) { i => sum += q(i, j) * at_k(i) }
                    sum /= - at_k(k) 
                    cfor (k, m) { i => q(i, j) += sum * at_k(i) }
                } // cfor
            end if
        end for
    end computeQ
 
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for x in aa*x = b using the QR Factorization aa = q*r via
     *  r*x = q.t * b without actually calculating the q matrix.
     *  Note /~ is the back substitution operator (\ in MatLab).
     *  @param b  the constant vector
     */
    def solve (b: VectorD): VectorD = r /~ transformB (b)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Efficient calculation of inverse matrix a^-1 from existing factorization.
     *      a * a^-1 = I
     */
    def inverse: MatrixD =
        factor ()
        if q == null then computeQ ()
        val r_inv = Fac_Inverse.inverse_ut (r)                 // upper triangular matrix inverse
        r_inv * q.transpose
    end inverse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transform b into q.t * b, i.e., perform the right side of the equation
     *  r*x = q.t * b not using a full q matrix.
     *  @param b  the constant vector
     */
    private def transformB (b: VectorD): VectorD =  
        val qt_b = b.copy                                      // the result vector of q.t * b

        cfor (0, n) { j =>                                     // calculate the result of q.t * b
            val at_j = at.v(j)                                 // get the j-th Householder vector
            var sum  = 0.0                                 
            cfor (j, m) { i => sum += qt_b(i) * at_j(i) }  
            sum /= - at_j(j)
            cfor (j, m) { i => qt_b(i) += sum * at_j(i) }
        } // cfor
        qt_b
    end transformB
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the nullspace of matrix a: { x | a*x = 0 } using QR Factorization
     *  q*r*x = 0.  Gives a basis of dimension n - rank for the nullspace
     *  Caveat: requires dim >= dim2
     *  @param rank  the rank of the matrix (number of linearly independent column vectors)
     */
    def nullspace (rank: Int): MatrixD = 
        val at = aa.transpose                                  // transpose of aa
        val qq = (new Fac_QR (at, true)).factor1 ()            // using QR with needQ = true
        if qq == null then flaw ("nullspace", "orthogonal matrix qq is null")
        val ns = qq(?, rank until at.dim2 )                    // last n - rank columns
        if ns.dim2 > 0 then ns else new MatrixD (at.dim, 0)
    end nullspace

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the nullspace of matrix a: { x | a*x = 0 } using QR Factorization
     *  q*r*x = 0.  Gives only one vector in the nullspace.
     *  @param x  a vector with the correct dimension
     */
    def nullspaceV (x: VectorD): VectorD =
        x(n-1) = 1.0                                           // vector to solve for
        val b  = new VectorD (n)                               // new rhs as -r_i, n-1          
        cfor (0, n) { i => b(i) = -r(i, n-1) }
        val rr = r(?, 0 to n-2)                                // drop last column
        for k <- n-2 to 0 by -1 do                             // solve for x in rr*x = b
            x(k) = (b(k) - (rr(k) dot x)) / rr(k, k)
        end for
        x
    end nullspaceV

end Fac_QR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_QR` companion object provides a factory method and example matrices to factor.
 */
object Fac_QR:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Fac_QR` object when m >= n or a `Fac_LQ` object otherwise (more
     *  column than rows).
     *  @param aa     the matrix to be factored into q and r
     *  @param needQ  flag indicating whether a full q matrix is needed
     */
    def apply (aa: MatrixD, needQ: Boolean = false): Factorization =
        if aa.dim >= aa.dim2 then new Fac_QR (aa, needQ) else new Fac_LQ (aa)
    end apply

    val a1 = MatrixD ((4, 3), 9.0,  0.0, 26.0,
                             12.0,  0.0, -7.0,
                              0.0,  4.0,  4.0,
                              0.0, -3.0, -3.0)

    val a2 = MatrixD ((2, 2), 2.0,  1.0,
                             -4.0, -2.0)

    val a3 = MatrixD ((3, 3), 0.0,  1.0,  1.0,
                             -5.0, -2.0, -2.0,
                             -5.0, -2.0, -2.0)

    val a4 = MatrixD ((4, 4), -1.0, 1.0, 2.0,  4.0,
                               2.0, 0.0, 1.0, -7.0,
                               2.0, 0.0, 1.0, -7.0,
                               2.0, 0.0, 1.0, -7.0)

    val a5 = MatrixD ((5, 3), 0.8147, 0.0975, 0.1576,
                              0.9058, 0.2785, 0.9706,
                              0.1270, 0.5469, 0.9572,
                              0.9134, 0.9575, 0.4854,
                              0.6324, 0.9649, 0.8003)

//  since m < n, use Fac_LQ
    val a6 = MatrixD ((2, 4), 1.0, 2.0, 3.0, 4.0,
                              5.0, 6.0, 7.0, 8.0)

end Fac_QR

import Fac_QR._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_QRTest` main function is used to test the `Fac_QR` classes.
 *  @see www.ee.ucla.edu/~vandenbe/103/lectures/qr.pdf
 *  @see www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf
 *  FIX: the nullspaceV function needs to be fixed.
 *  > runMain scalation.mathstat.fac_QRTest
 */
@main def fac_QRTest (): Unit =

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the correctness of the QR Factorization.
     *  @param a   the matrix to factor
     *  @param qr  the QR Factorization class/algorithm to use
     *  @param nm  the name of test case/matrix
     */
    def test (nm: String, a: MatrixD, qr: Fac_QR): Unit =
        println ("-" * 60)
        val (q, r) = qr.factor12 ()                      // (q orthogonal, r upper triangular)
        println (s"q.dims = ${q.dims}")
        println (s"r.dims = ${r.dims}")
        val prod   = q * r                               // product of q * r

        println (nm + "    = " + a)                      // original matrix
        println ("q     = " + q)                         // orthogonal matrix
        println ("r     = " + r)                         // right upper triangular matrix
        println ("q*r   = " + prod)                      // product q * r
        println ("eq    = " + (a =~ prod))               // check that q * r  = a
    end test

    banner ("Fac_QRTest: Fac_QR")
    test ("a1", a1, new Fac_QR (a1, true))
    test ("a2", a2, new Fac_QR (a2, true))
    test ("a3", a3, new Fac_QR (a3, true))
    test ("a4", a4, new Fac_QR (a4, true))
    test ("a5", a5, new Fac_QR (a5, true))

    val a0 = MatrixD ((3, 3), 1, 2, 3,
                              2, 5, 6,
                              9, 3, 7)

    banner ("Fac_QRTest: Test Inverse")
    println (s"a0 = $a0")
    val qr0 = new Fac_QR (a0, true)
    qr0.factor ()
    println (s"qr0.factor12 () = ${qr0.factor12 ()}")
    val a0_inv = qr0.inverse
    println (s"a0_inv = $a0_inv")
    println (s"a0 * a0_inv = ${a0 * a0_inv}")

end fac_QRTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_QRTest2` main function is used to test the correctness of the solve method
 *  in the `Fac_QR` class as it would be used in Regression.
 *  > runMain scalation.mathstat.fac_QRTest2
 */
@main def fac_QRTest2 (): Unit =

    // 20 data points:   Constant     x_1     x_2    x_3      x_4
    //                                Age  Weight    Dur   Stress
    val x = MatrixD ((20, 5), 1.0,   47.0,   85.4,   5.1,    33.0,
                              1.0,   49.0,   94.2,   3.8,    14.0,
                              1.0,   49.0,   95.3,   8.2,    10.0,
                              1.0,   50.0,   94.7,   5.8,    99.0,
                              1.0,   51.0,   89.4,   7.0,    95.0,
                              1.0,   48.0,   99.5,   9.3,    10.0,
                              1.0,   49.0,   99.8,   2.5,    42.0,
                              1.0,   47.0,   90.9,   6.2,     8.0,
                              1.0,   49.0,   89.2,   7.1,    62.0,
                              1.0,   48.0,   92.7,   5.6,    35.0,
                              1.0,   47.0,   94.4,   5.3,    90.0,
                              1.0,   49.0,   94.1,   5.6,    21.0,
                              1.0,   50.0,   91.6,  10.2,    47.0,
                              1.0,   45.0,   87.1,   5.6,    80.0,
                              1.0,   52.0,  101.3,  10.0,    98.0,
                              1.0,   46.0,   94.5,   7.4,    95.0,
                              1.0,   46.0,   87.0,   3.6,    18.0,
                              1.0,   46.0,   94.5,   4.3,    12.0,
                              1.0,   48.0,   90.5,   9.0,    99.0,
                              1.0,   56.0,   95.7,   7.0,    99.0)
    //  response BP
    val y = VectorD (105.0, 115.0, 116.0, 117.0, 112.0, 121.0, 121.0, 110.0, 110.0, 114.0,
                     114.0, 115.0, 114.0, 106.0, 125.0, 114.0, 106.0, 113.0, 110.0, 122.0)

    println ("model: y = b_0 + b_1*x1 + b_2*x_ + b3*x3 + b4*x42")
//  println ("model: y = b₀ + b₁∙x₁ + b₂∙x₂ + b₃∙x₃ + b₄∙x₄")
    println (s"x = $x")
    println (s"y = $y")

    val qr = new Fac_QR(x)
    qr.factor ()
    println (qr.factors)
    println (s"b = ${qr.solve (y)}")                        // compute the b vector by using solve of `Fac_QR`

end fac_QRTest2

