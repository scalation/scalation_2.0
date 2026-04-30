
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Sat Dec  8 14:32:12 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Correlogram with ACF and PACF
 *
 *  @see www.jstatsoft.org/article/view/v027i03
 *  @see www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf
 *       Time Series: Interpreting ACF and PACF
 *       AR(p)      PACF drops-off after p lags
 *       MA(q)      ACF  drops-off after q lags
 *       ARMA(p, q) both tail-off (rather than drop-off)
 *                  better to use other criteria (e.g., AIC/AICc) to chose p and q
 */

package scalation
package mathstat

import scala.math.{min, sqrt}

val MAX_LAGS = 49                                    // maximum amount of lag supported

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Correlogram` trait provides functions for computing the Auto-Correlation
 *  Function (ACF) and the Partial Auto-Correlation Function (PACF).
 *  @param y         the time series data (response vector)
 *  @param adjusted  whether to adjust to account for the number of elements in the sum Σ (or use dim-1)
 *                   @see `VectorD.acov`
 */
trait Correlogram (y: VectorD, adjusted: Boolean = true):

    private val ml = min (y.dim-1, MAX_LAGS)         // maximum lag to consider (can't exceed dataset size)
    private var stats: Stats4TS = null               // statistics on time-series y
    private var psi:   MatrixD  = null               // pass in auto-covariance and max lags to Durbin-Levinson
    private var pacf:  VectorD  = null               // Partial Auto-Correlation Function (PACF) is main diagonal

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make a Correlogram, i.e., compute stats, psi and pacf.
     *  @param y_  the current (e.g., training) times-series to use (defaults to full y)
     */
    def makeCorrelogram (y_ : VectorD = y): Unit =
        stats = Stats4TS (y_, ml, adjusted)
        psi   = durbinLevinson (stats.acv, ml)
        pacf  = psi(?)
    end makeCorrelogram

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the autocorrelation vector (ACF).
     */
    inline def acF: VectorD  = stats.acr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the partial autocorrelation vector (PACF).
     */
    inline def pacF: VectorD = pacf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the psi matrix.
     */
    inline def psiM: MatrixD = psi

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return basic statistics on time-series y or y_.
     */
    inline def statsF: Stats4TS = stats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Apply the Durbin-Levinson Algorithm to iteratively compute the psi matrix.
     *  The last/p-th row of the matrix gives AR coefficients.
     *  Note, also known as Levinson-Durbin.
     *  @see www.stat.tamu.edu/~suhasini/teaching673/time_series.pdf, p. 247
     *  @param g   the auto-covariance vector (gamma)
     *  @param ml  the maximum number of lags
     */
    def durbinLevinson (g: VectorD, ml: Int): MatrixD =
        val ψ = new MatrixD (ml+1, ml+1)                       // psi matrix (ml = max lags)
        val r = new VectorD (ml+1); r(0) = g(0)

        for k <- 1 to ml do                                    // range up to max lags
            var sum = 0.0
            for j <- 1 until k do sum += ψ(k-1, j) * g(k-j)
            val a = (g(k) - sum) / r(k-1)
            ψ(k, k) = a
            for j <- 1 until k do ψ(k, j) = ψ(k-1, j) - a * ψ(k-1, k-j)
            r(k) = r(k-1) * (1.0 - a * a)
        end for
        ψ
    end durbinLevinson

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Plot a function, e.g., Auto-Correlation Function (ACF), Partial Auto-Correlation
     *  Function (PACF) with confidence bound.
     *  @param fVec  the vector given function values
     *  @param name  the name of the function
     *  @param show  whether to show the fVec values
     */
    def plotFunc (fVec: VectorD, name: String, show: Boolean = true): Unit =
        val lag_axis = VectorD.range (0, ml+1)
        val zero     = new VectorD (ml+1)
        val bound    = VectorD (for k <- 0 to ml yield 1.96 / sqrt (y.dim - k))
        val mat      = MatrixD (fVec, zero, bound, -bound)
        new PlotM (lag_axis, mat, Array ("fVec", "zero", "bound"), "PlotM of " + name, true)
        if show then println (s"$name: fVec = $fVec")
    end plotFunc

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Plot both the Auto-Correlation Function (ACF) and the Partial Auto-Correlation
     *  Function (PACF) with confidence bound.
     */
    def plotCorrelogram (): Unit =
        plotFunc (acF, "ACF")
        plotFunc (pacF, "PACF")
    end plotCorrelogram

end Correlogram


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `correlogramTest` main function tests the `Correlogram` trait on a simple
 *  dataset.
 *  > runMain scalation.mathstat.correlogramTest
 */
@main def correlogramTest (): Unit =

    val y = VectorD (1, 2, 5, 8, 3, 6, 9, 4, 5, 11,
                     12, 16, 7, 6, 13, 15, 10, 8, 14, 17)

    banner ("Plot Data")
    new Plot (null, y, null, "y vs. t", lines = true)

    banner ("Test Correlogram")
    object CT extends Correlogram (y)
    CT.makeCorrelogram ()
    CT.plotCorrelogram ()
    
end correlogramTest

