
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Jun 17 19:29:23 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Experimental Implementation for Testing Efficiency of Methods
 */

package scalation
package mathstat

import scala.math.min

class MatrixD2 (val dim:  Int,
               val dim2: Int,
               var v:    Array [Array [Double]] = null):

    if v == null then
        v = Array.ofDim [Double] (dim, dim2)
    else if dim != v.length || dim2 != v(0).length then
        flaw ("init", "dimensions are wrong")

    private val flaw = flawf ("MatrixD2")                          // partial invocation of flaw function

    private val fString = "%g,\t"

    val indices  = 0 until dim
    val indices2 = 0 until dim2

    def apply (i: Int, j: Int): Double = v(i)(j)

    def update (i: Int, j: Int, a: Double): Unit = v(i)(j) = a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transpose this matrix (columns <=> rows).
     */
    def transpose: MatrixD2 = new MatrixD2 (dim2, dim, v.transpose)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this matrix and matrix y (requires y to have at least the dimensions of this).
     *  @param y  the other matrix
     */
    infix def add1 (y: MatrixD2): MatrixD2 =
        if y.dim < dim || y.dim2 < dim2 then flaw ("+", "matrix + matrix - incompatible dimensions")

        val a = Array.ofDim [Double] (dim, y.dim2)
        for i <- indices; j <- indices2 do a(i)(j) = v(i)(j) + y.v(i)(j)
        new MatrixD2 (dim, dim2, a) 
    end add1

// alias rows to avoid double subscripting

    infix def add2 (y: MatrixD2): MatrixD2 =
        if y.dim < dim || y.dim2 < dim2 then flaw ("+", "matrix + matrix - incompatible dimensions")

        val a = Array.ofDim [Double] (dim, y.dim2)
        for i <- indices do
            val v_i = v(i); val y_i = y.v(i); val a_i = a(i)
            for j <- indices2 do a_i(j) = v_i(j) + y_i(j)
        end for
        new MatrixD2 (dim, dim2, a) 
    end add2

// replace for with cfor

    infix def add3 (y: MatrixD2): MatrixD2 =
        if y.dim < dim || y.dim2 < dim2 then flaw ("+", "matrix + matrix - incompatible dimensions")

        val a = Array.ofDim [Double] (dim, y.dim2)
        cfor (0, dim) { i =>
            val v_i = v(i); val y_i = y.v(i); val a_i = a(i)
            cfor (0, dim2) { j => a_i(j) = v_i(j) + y_i(j) }
        } // for
        new MatrixD2 (dim, dim2, a) 
    end add3

/***
// use AVX vector instructions - vsz = number of 64-bit words in vector (e.g., AVX-512 has 8)

//  import jdk.incubator.vector.DoubleVector
    val species = DoubleVector.SPECIES_512
    val vsz = 8

    infix def add3 (y: MatrixD2): MatrixD2 =
        if y.dim < dim || y.dim2 < dim2 then flaw ("+", "matrix + matrix - incompatible dimensions")

        val a = Array.ofDim [Double] (dim, y.dim2)
        for i <- indices do
            val v_i = v(i); val y_i = y.v(i); val a_i = a(i)
            for j <- 0 until dim2 by vsz do
                val xv = DoubleVector.fromArray (species, v_i, i)
                val yv = DoubleVector.fromArray (species, y_i, i)
                xv.add (yv).intoArray (a_i, i)
            end for
            val left = dim2 % vsz 
            for j <- dim2 - left until dim do a_i(j) = v_i(j) + y_i(j)
        end for
        new MatrixD2 (dim, dim2, a)  
    end add3
***/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiply this matrix by matrix y.
     *  @param y  the other matrix
     */
    infix def mul1 (y: MatrixD2): MatrixD2 =
        if dim2 != y.dim then flaw ("*", "matrix * matrix - incompatible cross dimensions")

        val z = new MatrixD2 (dim, y.dim2)
        for i <- indices; j <- y.indices2 do
            var sum = 0.0
            for k <- indices2 do sum += v(i)(k) * y.v(k)(j)
            z.v(i)(j) = sum
        end for
        z
    end mul1

// transpose y first

    infix def mul2 (y: MatrixD2): MatrixD2 =
        if dim2 != y.dim then flaw ("*", "matrix * matrix - incompatible cross dimensions")

        val z  = new MatrixD2 (dim, y.dim2)
        val yt = y.v.transpose
        for i <- indices; j <- y.indices2 do
            val v_i = v(i); val y_j = yt(j)
            var sum = 0.0
            for k <- indices2 do sum += v_i(k) * y_j(k)
            z.v(i)(j) = sum
        end for
        z
    end mul2

// transpose y first and use array ops

    infix def mul3 (y: MatrixD2): MatrixD2 =
        if dim2 != y.dim then flaw ("*", "matrix * matrix - incompatible cross dimensions")

        val a  = Array.ofDim [Double] (dim, y.dim2)
        val yt = y.v.transpose

        for i <- indices do
            val v_i = v(i)
            for j <- y.indices2 do
                val y_j = yt(j)
                var sum = 0.0
                for k <- indices2 do sum += v_i(k) * y_j(k)
                a(i)(j) = sum
            end for
        end for
        new MatrixD2 (dim, y.dim2, a) 
    end mul3

// transpose y first, use array ops and blocking/tiling

    private val TSZ = 100                 // Pick a tile size T = Θ(√M)

// @see https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
// n - dim, m - dim2, p - y.dim2

    infix def mul4 (y: MatrixD2): MatrixD2 =
        if dim2 != y.dim then flaw ("*", "matrix * matrix - incompatible cross dimensions")

        val a  = Array.ofDim [Double] (dim, y.dim2)
        val yt = y.v.transpose
        for ii <- 0 until dim by TSZ do
            for jj <- 0 until y.dim2 by TSZ do 
                for kk <- 0 until dim2 by TSZ do
                    val k2 = min (kk + TSZ, dim2)

                    for i <- ii until min (ii + TSZ, dim) do
                        val v_i = v(i)
                        for j <- jj until min (jj + TSZ, y.dim2) do
                            val y_j = yt(j)
                            var sum = 0.0
                            for k <- kk until k2 do sum += v_i(k) * y_j(k)
                            a(i)(j) += sum
                        end for
                    end for

                end for
            end for
        end for
        new MatrixD2 (dim, y.dim2, a)
    end mul4

// use array ops, blocking/tiling and reorder loops

// @see  https://software.intel.com/content/www/us/en/develop/documentation/advisor-cookbook/top/optimize-memory-access-patterns-using-loop-interchange-and-cache-blocking-techniques.html

    infix def mul5 (y: MatrixD2): MatrixD2 =
        if dim2 != y.dim then flaw ("*", "matrix * matrix - incompatible cross dimensions")

        val a = Array.ofDim [Double] (dim, y.dim2)

        for ii <- 0 until dim by TSZ do
            val i2 = min (ii + TSZ, dim)
            for kk <- 0 until dim2 by TSZ do
                val k2 = min (kk + TSZ, dim2)
                for jj <- 0 until y.dim2 by TSZ do
                    val j2 = min (jj + TSZ, y.dim2)

//                  for i <- ii until i2 do
                    cfor (ii, i2) { i =>
                        val v_i = v(i); val a_i = a(i)
//                      for k <- kk until k2 do
                        cfor (kk, k2) { k =>
                            val y_k = y.v(k); val v_ik = v_i(k)
//                          for j <- jj until j2 do
                            cfor (jj, j2) { j => a_i(j) += v_ik * y_k(j) }
                        } // cfor
                    } // cfor

                end for
            end for
        end for
        new MatrixD2 (dim, y.dim2, a)
    end mul5

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this matrix to a string.
     */
    override def toString: String = 
        val sb = new StringBuilder ("\nMatrixD2(")
        if dim == 0 || dim2 == 0 then return sb.append (")").mkString
        for i <- indices; j <- indices2 do
            sb.append (fString.format (v(i)(j)))
            if j == dim2-1 then sb.replace (sb.length-1, sb.length, "\n \t")
        end for
        sb.replace (sb.length-4, sb.length, ")").mkString
    end toString

end MatrixD2


object MatrixD2:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix from repeated values.
     *  @param dim  the (row, column) dimensions
     *  @param u    the repeated values
     */
    def apply (dim: (Int, Int), u: Double*): MatrixD2 =
        val a = Array.ofDim [Double] (dim._1, dim._2)
        for i <- 0 until dim._1; j <- 0 until dim._2 do a(i)(j) = u(i * dim._2 + j)
        new MatrixD2 (dim._1, dim._2, a)
    end apply

end MatrixD2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MatrixD2Example` object provides example instances of the`MatrixD2` class.
 */
object MatrixD2Example:

    val x = MatrixD2 ((8, 8), 1, 2,  3,  4,  5,  6,  7,  8,
                             2, 3,  4,  5,  6,  7,  8,  9,
                             3, 4,  5,  6,  7,  8,  9, 10,
                             4, 5,  6,  7,  8,  9, 10, 11,
                             5, 6,  7,  8,  9, 10, 11, 12,
                             6, 7,  8,  9, 10, 11, 12, 13,
                             7, 8,  9, 10, 11, 12, 13, 14,
                             8, 9, 10, 11, 12, 13, 14, 15)

    val y = MatrixD2 ((8, 8), 1, 2,  3,  4,  5,  6,  7,  8,
                             2, 3,  4,  5,  6,  7,  8,  9,
                             3, 4,  5,  6,  7,  8,  9, 10,
                             4, 5,  6,  7,  8,  9, 10, 11,
                             5, 6,  7,  8,  9, 10, 11, 12,
                             6, 7,  8,  9, 10, 11, 12, 13,
                             7, 8,  9, 10, 11, 12, 13, 14,
                             8, 9, 10, 11, 12, 13, 14, 15)

end MatrixD2Example

import MatrixD2Example.{x, y}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixD2Test` main function tests the `MatrixD2` class.  Compares the performance
 *  of matrix addition implementations
 *  > runMain scalation.mathstat.matrixD2Test
 */
@main def matrixD2Test (): Unit =

    println (s"x add1 y = ${x add1 y}")
    println (s"x add2 y = ${x add2 y}")
    println (s"x add3 y = ${x add3 y}")

    val a = new MatrixD2 (1000, 1000)
    val b = new MatrixD2 (1000, 1000)
    for i <- a.indices; j <- a.indices2 do { a(i, j) = i + j; b(i, j) = a(i, j) }

    for it <- 1 to 10 do
        banner (s"Timing results to iteration $it")
        val t1 = gauge { a add1 b }
        println (s"a add1 b = $t1")
        val t2 = gauge { a add2 b }
        println (s"a add2 b = $t2")
        val t3 = gauge { a add3 b }
        println (s"a add3 b = $t3")
    end for

end matrixD2Test


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixD2Test2` main function tests the `MatrixD2` class.  Compares the performance
 *  of matrix multiplication implementations.
 *  > runMain scalation.mathstat.matrixD2Test2
 */
@main def matrixD2Test2 (): Unit =

    println (s"x mul1 y = ${x mul1 y}")
    println (s"x mul2 y = ${x mul2 y}")
    println (s"x mul3 y = ${x mul3 y}")
    println (s"x mul4 y = ${x mul4 y}")
    println (s"x mul5 y = ${x mul5 y}")

    val a = new MatrixD2 (1000, 1000)
    val b = new MatrixD2 (1000, 1000)
    for i <- a.indices; j <- a.indices2 do { a(i, j) = i + j; b(i, j) = a(i, j) }

    for it <- 1 to 5 do
        banner (s"Timing results to iteration $it")
        val t1 = gauge { a mul1 b }
        println (s"a mul1 b = $t1")
        val t2 = gauge { a mul2 b }
        println (s"a mul2 b = $t2")
        val t3 = gauge { a mul3 b }
        println (s"a mul3 b = $t3")
        val t4 = gauge { a mul4 b }
        println (s"a mul4 b = $t4")
        val t5 = gauge { a mul5 b }
        println (s"a mul5 b = $t5")
    end for

end matrixD2Test2

