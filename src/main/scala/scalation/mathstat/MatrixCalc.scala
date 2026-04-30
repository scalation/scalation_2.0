
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Jun 17 19:29:23 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Matrix Calculations
 */

package scalation
package mathstat

import scala.math.round

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MattixCalc` class supports simple simple calculations on elements in a matrix.
 *  @param x       the matrix of data
 *  @param header  the column names
 */
class MatrixCalc (x: MatrixD, header: VectorS):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform calculations for each row using the given formula.
     *  Usage:  val q = new MatrixCalc (x, hd)
     *  Usage:  q.fill (3, 2, 1, (zt: Double) => r * zt)
     *  @param c1       the column to be assigned
     *  @param c2       the column supplying the data
     *  @param offset   the row index offset applied to c2 (e.g., get past data)
     *  @param formula  the c1 = formula (c2) appropriately offset
     */
    def fill (c1: Int, c2: Int, offset: Int, formula: FunctionS2S): Unit =
        for i <- offset until x.dim do
            x(i, c1) = formula (x(i - offset, c2))
    end fill

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show header and the matrix.
     */
    def show (): Unit =
        println (header)
        println (x)
    end show

end MatrixCalc


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixCalc0` main function is used to test the `MatrixCalc` class.
 *  It performs the RW and AR(1) time series model calculations.
 *  > runMain scalation.mathstat.matrixCalc0
 */
@main def matrixCalc0 (): Unit =

//  val model = "NULL"
    val model = "RW"
//  val model = "AR"

    val y   = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3)
    val m   = y.dim
    val t   = VectorD.range (0, m) 
    val hdr = VectorS ("t", "yt", "zt", "zˆt", "yˆt", "ε", "ε2")
    val x   = new MatrixD (m, hdr.dim) 

    val ybar = y.mean
    val r    = y(0 until m-1) corr y(1 until m)           // HW - why are r, r_ different
    val r_   = y.acorr (1)                                // try using r_
    println (s"ybar = $ybar, rho_1: r = $r, r_ = $r_")

    x(?, 0) = t                                           // time
    x(?, 1) = y                                           // time series y_t

    model match
    case "NULL" =>
        for i <- 1 until m do x(i, 4) = ybar              // y_t-hat
    case "RW" =>
        for i <- 1 until m do x(i, 4) = x(i-1, 1)         // y_t-hat
    case _ => 
        x(?, 2) = y - ybar                                // centered z_t
        for i <- 1 until m do x(i, 3) = x(i-1, 2) * r_    // z_t-hat, try both r, r_
        x(?, 4) = x(?, 3) + ybar                          // uncentered y_t-hat

    x(?, 5) = y - x(?, 4)                                 // error ε
    x(?, 6) = x(?, 5) ~^ 2                                // squared error
    println (hdr)
    println (x)

    val y_   = y(1 until m)
    val csse = x(1 until m, 6).sum
    val sst  = (y_ - ybar).normSq
    val rSq  = 1 - csse / sst
    println (s"csse = $csse, sst = $sst, rSq = $rSq")

end matrixCalc0


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixCalc2` main function read a matrix from a CSV file and perform calculations,
 *  e.g. grade calculations.
 *  > runMain scalation.mathstat.matrixCalc2
 */
@main def matrixCalc2 (): Unit =

    val csvFile = ".../...csv"

    val xx = MatrixD.load (csvFile, 1, 3, fullPath = true)                        // skip 1 row and 3 columns
    val n = xx.dim2 - 3                                                           // last column should be empty
    val x = xx(?, 0 until n+1)

    println (s"x = $x")

//  val w = x(0)(0 until n)                                                       // weights for grades
//  val w = VectorD (2,2,2.5,0.25,0.25,0.6,0.6,0.6,1.2)                           // 4 projects
    val w = VectorD (2,2,2.5,0.25,0.25,0.75,0.75,1.5)                             // 3 projects
    println (s"total weight = ${w.sum}")                                          // total weight
    for i <- 0 until x.dim do x(i, n) = round (w dot x(i)(0 until n)).toDouble    // weighted total
    println (s"new x = $x")                                                       // updated matrix

end matrixCalc2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixCals3` main function allows for Custom calculations.
 *  > runMain scalation.mathstat.matrixCalc3
 */
@main def matrixCalc3 (): Unit =

   val x1 = VectorD (80,90,80,100,100,90,80,90)
   val x2 = VectorD (90,90,80,100,100,100,100,100,100)

   val w  = VectorD (2,2,2.5,0.25,0.25,0.75,0.75,1.5)
   val w2 = VectorD (2,2,2.5,0.25,0.25,0.6,0.6,0.6,1.2)

   println (s"w * x1  = ${w dot x1}")
   println (s"w2 * x2 = ${w2 dot x2}")

end matrixCalc3

