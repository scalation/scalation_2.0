
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support:  Extended Sample Autocorrelation Function (ESACF)
 *
 *  @see     `Correlogram` and `Stats4TS`
 */

package scalation
package modeling
package forecasting

import scala.annotation.tailrec
import scala.math.{abs, sqrt}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ESACF` object provides an implementation of the Extended Sample
 *  Autocorrelation Function (ESACF) to assist in determining ARIMA orders.
 */
object ESACF:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the AutoCorrelation Function (ACF) for y_.
     *  @param y_  the vector whose acf is sought
     *  @param n   the number of elements to use
     */
    def acf (y_ : VectorD, n: Int): VectorD =
        val r = new VectorD (n + 1)
        for k <- r.indices do r(k) = y_.acorr (k)
        r
    end acf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generates the ESACF Matrix for AR orders up to pMax and MA orders up to qMax.
     *  @param y     the time series data vector
     *  @param pMax  the Maximum Autoregressive lag to check
     *  @param qMax  the Maximum Moving Average lag to check
     */
    def computeMatrix (y: VectorD, pMax: Int = 10, qMax: Int = 10): MatrixD =
        val n        = y.dim
        val esacfMat = new MatrixD (pMax + 1, qMax + 1)
        esacfMat(0)  = acf (y, qMax)               // row p=0: fill with basic sample ACF values

        // Iterate through higher-order AR steps to clear out AR distortions
        for p <- 1 to pMax; q <- 0 to qMax do
            val effectiveN = n - p - q
            
            if effectiveN > p then
                val X = new MatrixD (effectiveN, p)
                val Y = new VectorD (effectiveN)

                for i <- 0 until effectiveN do
                    val t = i + p + q
                    Y(i)  = y(t)
                    for j <- 0 until p do X(i, j) = y(t - 1 - j)

                // Fit Iterated Ordinary Least Squares via ScalaTion 2.0 
                val reg = new Regression (X, Y)
                reg.train ()
                val phiHat = reg.parameter

                // Filter out the dominant AR component 
                val w = new VectorD (n)
                for t <- p until n do
                    var filteredVal = y(t)
                    for j <- 0 until p do filteredVal -= phiHat(j) * y(t - 1 - j)
                    w(t) = filteredVal

                // Measure clean residual correlation boundary at index p+q
                esacfMat(p, q) = w(p until n).acorr (p + q)
            else
                esacfMat(p, q) = 0.0
            end if
        end for

        esacfMat
    end computeMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Automated scanner that walks the matrix to find the top-left vertex 
     *  of the continuous zero triangle.
     *  @param matrix  the raw computed MatrixD from ESACF.compute
     *  @param n       the sample size/dimension of your data vector
     */
    def findOptimalOrder (matrix: MatrixD, n: Int): (Int, Int) =
        val bound = 1.96 / sqrt (n.toDouble)
        val pMax = matrix.dim - 1
        val qMax = matrix.dim2 - 1

        // Define a helper function to validate a specific triangle wedge shape
        def isValidWedge(pStart: Int, qStart: Int): Boolean =
            // Every row 'i' from pStart down to pMax must satisfy the zero boundary condition
            (pStart to pMax).forall { i =>
                val wedgeWidth = qStart + (i - pStart)
                val currentRightBound = if wedgeWidth > qMax then qMax else wedgeWidth
                
                // Every column 'j' inside the wedge must be below our significance threshold
                (qStart to currentRightBound).forall { j =>
                    abs(matrix(i, j)) <= bound
                }
            }

        // Perform a tail-recursive coordinate search to avoid non-local returns
        @tailrec
        def search (p: Int, q: Int): (Int, Int) =
            if p > pMax then 
                (0, 0)                          // default fallback safe guess if search window is exhausted
            else if q > qMax then 
                search (p + 1, 0)               // move down to the next row baseline
            else if isValidWedge (p, q) then 
                (p, q)                          // target vertex found cleanly!
            else 
                search (p, q + 1)               // slide right to the next column coordinate

        search (0, 0)                           // initiate the search pattern from the top left corner
    end findOptimalOrder

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print out a clean symbolic layout identifying the 0-vertex target.
     *  @param matrix
     *  @param n
     */
    def printTable (matrix: MatrixD, n: Int): Unit =
        val bound = 1.96 / sqrt (n.toDouble)            // consistent with Correlogram.scala bounds
        
        println ("\n>>> ESACF Diagonal Matrix Table <<<")
        print ("P\\Q\t")
        for q <- matrix.indices2 do print (s"$q\t")
        println ()

        for p <- matrix.indices do
            print(s"$p\t")
            for q <- matrix.indices2 do
                if abs (matrix(p, q)) > bound then print ("X\t") else print ("0\t")
            println ()
    end printTable

end ESACF


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `eSACFTest` main function tests the `ESACF` object.
 *  > runMain scalation.modeling.forecasting.eSACFTest
 */
@main def eSACFTest (): Unit =

    // The precise validation array from Correlogram.scala
    val y = VectorD (1, 2, 5, 8, 3, 6, 9, 4, 5, 11,
                     12, 16, 7, 6, 13, 15, 10, 8, 14, 17)

    // Compute and trace using the fully integrated ESACF layout
    val matrix = ESACF.computeMatrix (y, pMax = 4, qMax = 4)
    ESACF.printTable (matrix, y.dim)

    // Use the scanner to execute the geometric triangle pathfinding routine
    val (optimalP, optimalQ) = ESACF.findOptimalOrder (matrix, y.dim)
    
    println ("\n==============================================")
    println (s"  AUTOMATED IDENTIFICATION COMPLETE")
    println ("==============================================")
    println (s" -> Conservative AR Guess (p):  $optimalP")
    println (s" -> Conservative MA Guess (q):  $optimalQ")
    println (s" -> Recommended Base Model:     ARIMA($optimalP, d, $optimalQ)")
    println ("==============================================")

end eSACFTest

