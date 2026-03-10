
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Aug  5 12:57:28 EDT 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Pivoting of Matrix Rows and Columns
 */

package scalation
package mathstat

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Pivoting` trait is used when row or column pivoting of a matrix is needed.
 */
trait Pivoting:

    private val debug = debugf ("Pivoting", true)                        // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reorder the rows of matrix a according to the pivot vector piv.
     *  @param a    the matrix to reorder
     *  @param piv  the vector indicating the row swaps that occurred during pivoting
     */
    def reorderRows (a: MatrixD, piv: VectorI): MatrixD =
        debug ("reorderRows", s"a = ${a.dims}, piv = $piv")
        val c = new MatrixD (a.dim, a.dim2)
        for i <- a.indices do c(i) = a(piv(i))
        c
    end reorderRows

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reorder the columns of matrix a according to the pivot vector piv.
     *  @param a    the matrix to reorder
     *  @param piv  the vector indicating the column swaps that occurred during pivoting
     */
    def reorderCols (a: MatrixD, piv: VectorI): MatrixD =
        debug ("reorderCols", s"a = ${a.dims}, piv = $piv")
        val c = new MatrixD (a.dim, a.dim2)
        for j <- a.indices2 do c(?, j) = a(?, piv(j))
        c
    end reorderCols

end Pivoting


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `pivotingTest` main function is used to test the `Pivoting` trait.
 *  > runMain scalation.mathstat.pivotingTest
 */
@main def pivotingTest (): Unit =

    object Pivoting_ extends Pivoting
    import Pivoting_._

    val a = MatrixD ((3, 3), 1, 2, 3,
                             4, 5, 6,
                             7, 8, 9)

    val piv = VectorI (2, 1, 0)

    println (s"a  = $a")
    println (s"a1 = ${reorderRows (a, piv)}")
    println (s"a2 = ${reorderCols (a, piv)}")

end pivotingTest

