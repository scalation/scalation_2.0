
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Zhaochong Liu
 *  @version 2.0
 *  @date    Sat Jul 30 22:53:47 EDT 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Rank Revealing QR Matrix Factorization
 *           Simple way to determine matrix a's rank:  Fac_QR_RR.rank (a)
 */

package scalation
package mathstat

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_QR_RR` class provides methods to factor an 'm-by-n' matrix 'a' into the
 *  product of two matrices:
 *      'q' - an 'm-by-n' orthogonal matrix and
 *      'r' - an 'n-by-n' right upper triangular matrix
 *  such that 'a = q * r'.   It uses uses Householder orthogonalization.
 *  @see 5.1 and 5.2 in Matrix Computations
 *  @see QRDecomposition.java in Jama
 *  @see www.stat.wisc.edu/~larget/math496/qr.html
 *-------------------------------------------------------------------------------
 *  This implementation extends `Fac_QR` and adds column pivoting for greater
 *  robustness and reasonably accurate rank determination (Rank Revealing QR).
 *  Caveat: for m < n use `Fac_LQ`.
 *-------------------------------------------------------------------------------
 *  @param aa     the matrix to be factored into q and r
 *  @param needQ  flag indicating whether a full q matrix is needed
 */
class Fac_QR_RR (aa: MatrixD, needQ: Boolean = true)
      extends Fac_QR (aa, needQ) with Pivoting:
 
    private val debug = debugf ("Fac_QR_RR", true)                 // debug function
    private var _rank = 0                                          // the rank of matrix aa
    private val _piv  = VectorI.range (0, n)                       // the vector storing the column pivots

    debug ("init", s"aa = ${aa.dims}, needQ = $needQ")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the pivot vector.
     */
    def piv: VectorI = _piv

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform QR Factorization and result in a single matrix which contains Householder vectors
     *  of each columns in lower triangular of 'aa' matrix. 
     *  This Rank Revealing algorithm uses Householder orthogonalization and pivoting.
     *  @see 5.1 and 5.2 in Matrix Computations
     *  @see QRDecomposition.java in Jama
     */
    override def factor (): Fac_QR_RR =
        if factored then return this

        val c = VectorD (for j <- at.indices yield at(j).normSq)   // length^2 of a's columns, at's rows

        var c_m = c.max                                            // maximum column length^2
        while _rank < n && c_m > TOL do                            // stop when max < TOL (tolerance for zero)
            val k_m = c.indexOf (c_m)                              // index of column with max length^2
            _piv.swap (k_m, _rank)                                 // swap pivot column to max
            if k_m != _rank then
                at.swap (k_m, _rank)                               // swap rows in at (columns in a)
                c.swap (k_m, _rank)                                // swap column lengths
            colHouse (_rank)                                       // perform kth factoring step                  
            for j <- _rank+1 until n do c(j) -= at(j, _rank) ~^ 2
            _rank += 1
            c_m = if _rank < n then c(_rank until n).max else 0.0
        end while

        for j <- 1 until p do                                    // fill in rest of r matrix
            val at_j = at.v(j)
            for i <- 0 until j do r(i, j) = at_j(i)
        end for
        if needQ then computeQ ()
//      r.clean (TOL)                                              // comment out to avoid cleaning r matrix

        factored = true
        this
    end factor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the rank (number of independent columns) of matrix 'aa'.
     */
    def rank: Int = _rank

end Fac_QR_RR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fac_QR_RR` object provides a convenient way to determine the rank of matrix.
 */
object Fac_QR_RR:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the rank (number of independent columns) of matrix 'a'.
     *  @param a  the matrix whose rank is to be determined
     */
    def rank (a: MatrixD): Int = new Fac_QR_RR (a, false).factor ().rank
        
end Fac_QR_RR

import Fac_QR._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fac_QR_RRTest` main function is used to test the `Fac_QR_RR` classes.
 *  @see www.ee.ucla.edu/~vandenbe/103/lectures/qr.pdf
 *  @see www.math.usm.edu/lambers/mat610/sum10/lecture9.pdf
 *  > runMain scalation.mathstat.fac_QR_RRTest
 */
@main def fac_QR_RRTest (): Unit =

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the correctness of the QR_RR Factorization.
     *  @param a   the matrix to factor
     *  @param qr  the QR Factorization class/algorithm to use
     *  @param nm  the name of test case/matrix
     */
    def test (nm: String, a: MatrixD, qr: Fac_QR_RR): Unit =
        println ("-" * 60)
        val (q, r) = qr.factor12 ()                      // (q orthogonal, r upper triangular)
        val prod   = q * r                               // product of q * r
        val ar     = qr.reorderCols (a, qr.piv)          // a matrix with columns reordered - FIX

        println (s"$nm   = $a")                          // original matrix
        println (s"q     = $q")                          // orthogonal matrix
        println (s"r     = $r")                          // right upper triangular matrix
        println (s"q*r   = $prod")                       // product q * r
        println (s"eq    = ${a == prod}")                // check that q * r == a
        println (s"ar    = $ar")                         // original matrix reordered
        println (s"eq    = ${ar == prod}")               // check that q * r == ar

        val r_est  = qr.rank                             // estimated rank
        val ns     = qr.nullspace (r_est)                // ns is a basis for the nullspace

        println (s"r_est = $r_est")                      // rank
        println (s"ns    = $ns")                         // nullspace
        println (s"a*ns  = ${a * ns}")                   // check that a * ns = 0
    end test

    println ("*" * 60)
    println ("Fac_QRTest: Fac_QR_RR")
    test ("a1", a1, new Fac_QR_RR (a1))
    test ("a2", a2, new Fac_QR_RR (a2))
    test ("a3", a3, new Fac_QR_RR (a3))
    test ("a4", a4, new Fac_QR_RR (a4))
    test ("a5", a5, new Fac_QR_RR (a5))

end fac_QR_RRTest

