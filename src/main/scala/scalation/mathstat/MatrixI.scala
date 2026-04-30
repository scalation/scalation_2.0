
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Dec 21 15:51:08 EST 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Matrix Data Structure of Integers (no class, only object)
 */

package scalation
package mathstat

import scala.math.round
import scala.util.control.Breaks.{break, breakable}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MatrixI object provides factory methods for class `MatrixD` with integer
 *  elements.  Low cost approach not requiring a `MatrixI` class.
 */
object MatrixI:

    private val flaw = flawf ("MatrixI")                             // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix from repeated integer values.
     *  @param dim  the (row, column) dimensions
     *  @param u    the repeated integer values
     */
    def apply (dim: (Int, Int), u: Int*): MatrixD =
        val a = Array.ofDim [Double] (dim._1, dim._2)
        for i <- 0 until dim._1; j <- 0 until dim._2 do a(i)(j) = u(i * dim._2 + j)
        new MatrixD (dim._1, dim._2, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a matrix from a variable argument list of integer vectors (row-wise).
     *  Use transpose to make it column-wise.
     *  @param vs  the vararg list of integer vectors
     */
    def apply (vs: VectorI*): MatrixD =
        val (m, n) = (vs.length, vs(0).length)
        val a = Array.ofDim [Array [Double]] (m)
        for i <- vs.indices do a(i) = vs(i).toDouble.v
        new MatrixD (m, n, a)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether the given matrix consists of only integer values.
     *  @param x  the given matrix
     */
    def isIntegerValued (x: MatrixD): Boolean =
        var allint = true
        breakable {
            for i <- x.indices; j <- x.indices2 do
                val x_ij = x(i, j)
                if x_ij != round (x_ij) then
                    flaw ("isIntegral", s"x($i, $j) = $x_ij is not integer-valued")
                    allint = false
                    break ()
            end for
        } // breakable
        allint
    end isIntegerValued

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the given matrix where each element is rounded to the nearest integer.
     *  @param x  the given matrix
     */
    def roundMat (x: MatrixD): MatrixD =
        val z = new MatrixD (x.dim, x.dim2)
        for i <- x.indices; j <- x.indices2 do z(i, j) = round (x(i, j)).toDouble
        z
    end roundMat

end MatrixI

