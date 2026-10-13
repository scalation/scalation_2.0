
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sat Dec  8 14:32:12 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Correlogram with ACF and PACF + Correlation Analysis
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

import scala.math.{abs, max, min, sqrt}

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
    private val r:     VectorD  = new VectorD (ml+1) // the residual variance evolution vector from Durbin-Levinson
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
        r(0)  = g(0)

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
    /** Calculate high-precision standard errors for an AR(p) model's coefficients
     *  using the variance inflation factors derived from the Durbin-Levinson work matrix.
     *  Needed for computing standard errors of coefficients (@see `summary_` in `AR`).
     *  @param nParams  the size of the parameter/coefficient vector (including intercept at index 0)
     *  @param sse      the Sum of Squared Errors from the model fit
     *  @param df       the degrees of freedom (or effective sample size N)
     */
    def stdErrAR (nParams: Int, sse: Double, df: Double): VectorD =
        val seVector = new VectorD (nParams)

        // Intercept Standard Error: Scales with the raw magnitude of the data
        seVector(0) = sqrt (sse / (df * df))

        if nParams > 1 then
            val lagOrder = nParams - 1
            val sigma2   = r(lagOrder)        // final residual variance (ψ^2) from Durbin-Levinson

            // High-Precision Lag Standard Errors using Gohberg-Semencul / Whittle property
            for j <- 1 to lagOrder do
                var varInflation = 1.0        // variance inflation
                
                // Sum squared contributions from lower-order partial correlations
                for k <- j until lagOrder do varInflation += psi(k, j)~^2

                // Precise Standard Error for Lag j: sqrt( (VIF * (ψ^2 / \ψ_0)) / N )
                seVector(j) = sqrt ((varInflation * (sigma2 / r(0))) / df)
            end for
        end if

        seVector
    end stdErrAR

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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Ljung-Box statistic Q and its p-value for the series given to
     *  `makeCorrelogram` (must be called first), i.e., a whiteness test of a
     *  residual series (@see the top-level `ljungBox` function).
     *  @param maxLag  the number of lags pooled into Q (defaults to min (10, ml))
     *  @param dof     the number of fitted ARMA parameters (p + q) to subtract from
     *                 the degrees of freedom (defaults to 0)
     */
    def ljungBoxTest (maxLag: Int = min (10, ml), dof: Int = 0): (Double, Double) =
        ljungBox (stats.y, maxLag, dof)
    end ljungBoxTest

end Correlogram


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Compute the Ljung-Box portmanteau test for the whiteness of a (residual) series e:
 *      Q = n (n + 2) Σ_{k=1}^{L} r_k^2 / (n - k)
 *  where r_k is the lag-k autocorrelation (@see `VectorD.acorr`, adjusted = false).
 *  Under H0 (no autocorrelation up to lag L) Q ~ ChiSquare (L - dof), so a small
 *  p-value (e.g., < .05) rejects whiteness.  Returns (Q, p-value); the p-value is
 *  NaN when L - dof < 1.  Note, `CDF.chiSquareCDF` integrates the density
 *  numerically: fine for reports, too slow to call for every candidate in a
 *  search (use Q or the band tests below there).
 *  @see en.wikipedia.org/wiki/Ljung%E2%80%93Box_test
 *  @param e       the series to test (e.g., the residuals of a fitted model)
 *  @param maxLag  the number of lags L pooled into Q (rule of thumb: min (10, n/5)
 *                 for non-seasonal data, 2 s for seasonal period s)
 *  @param dof     the number of fitted ARMA parameters (p + q) to subtract from
 *                 the degrees of freedom (0 for raw data or regression residuals)
 */
def ljungBox (e: VectorD, maxLag: Int, dof: Int = 0): (Double, Double) =
    val n  = e.dim
    val ml = min (maxLag, n - 1)
    val q  = n * (n + 2.0) * Σ (1, ml + 1) { k => (e.acorr (k, adjusted = false) ~^ 2) / (n - k) }
    val df = ml - dof
    val p  = if df < 1 then Double.NaN
             else if q > 100.0 + 3.0 * df then 0.0                    // tail below 1E-22: skip the integration
             else max (0.0, 1.0 - random.CDF.chiSquareCDF (q, df))
    (q, p)
end ljungBox


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the lags at which the autocorrelation of series e exceeds the
 *  approximate 95% band 2/√n, strongest first, as (lag, r_k) pairs, skipping
 *  the lags in `used` (e.g., the lags already present as predictors in the
 *  model whose residuals e are being examined).
 *  @param e       the series to scan (e.g., residuals)
 *  @param maxLag  the maximum lag to scan
 *  @param used    the lags to skip (defaults to none)
 */
def acfSpikes (e: VectorD, maxLag: Int, used: Set [Int] = Set ()): Seq [(Int, Double)] =
    val band = 2.0 / sqrt (e.dim.toDouble)
    (for k <- 1 to min (maxLag, e.dim - 1) yield (k, e.acorr (k, adjusted = false)))
        .filter ((k, r) => math.abs (r) > band && ! used (k))
        .sortBy ((_, r) => -math.abs (r))
end acfSpikes


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the lags k at which series e correlates with series x lagged k steps,
 *  corr (e_t, x_t-k), beyond the approximate 95% band 2/√n, strongest first, as
 *  (lag, correlation) pairs, skipping the lags in `used`.  A spike at lag k says
 *  x lagged k explains part of e (e.g., add x~k to the model).
 *  Uses `VectorD.ccorr`: x.ccorr (e, k) correlates x_t with e_t+k.
 *  @param e       the series to explain (e.g., residuals, n rows)
 *  @param x       the candidate explanatory series, aligned with e at t = 0
 *                 (at least n rows; extra trailing rows are ignored)
 *  @param maxLag  the maximum lag to scan
 *  @param used    the lags of x to skip (defaults to none)
 */
def ccfSpikes (e: VectorD, x: VectorD, maxLag: Int, used: Set [Int] = Set ()): Seq [(Int, Double)] =
    val n    = e.dim
    val band = 2.0 / sqrt (n.toDouble)
    val x_   = if x.dim > n then x(0 until n) else x
    (for k <- 1 to min (maxLag, n - 1) yield (k, x_.ccorr (e, k)))
        .filter ((k, r) => math.abs (r) > band && ! used (k))
        .sortBy ((_, r) => -math.abs (r))
end ccfSpikes


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Find the most collinear pair among the given columns of matrix x, i.e., the
 *  pair with the largest |corr| (@see `VectorD.corr`), returning (i, j, |corr|),
 *  or (-1, -1, 0) when fewer than two columns are given.
 *  @param x    the data/input matrix
 *  @param idx  the column indices to scan (defaults to all columns)
 */
def collinearPair (x: MatrixD, idx: IndexedSeq [Int] = null): (Int, Int, Double) =
    val cols = if idx == null then x.indices2 else idx
    var (bi, bj, bc) = (-1, -1, 0.0)
    for a <- 0 until cols.size; b <- a + 1 until cols.size do
        val c = math.abs (x(?, cols(a)) corr x(?, cols(b)))
        if c > bc then { bi = cols(a); bj = cols(b); bc = c }
    end for
    (bi, bj, bc)
end collinearPair


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return whether the spread of series e grows (or shrinks) with the level of
 *  series y, i.e., |corr (|e|, y)| exceeds the threshold -- a simple test for
 *  heteroscedastic residuals (suggesting, e.g., a log1p response transform).
 *  @param e       the residual series
 *  @param y       the response (level) series, aligned with e
 *  @param thresh  the |corr| threshold (defaults to .3)
 */
def heteroscedastic (e: VectorD, y: VectorD, thresh: Double = 0.3): Boolean =
    abs (e.abs corr y) > thresh
end heteroscedastic


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


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `correlogramTest2` main function tests the residual diagnostics: `ljungBox`
 *  on a small series with a hand-checked Q (SciPy: Q = 10.931087, p = 0.012104 at
 *  df = 3) and on white noise vs. AR(1) noise; `acfSpikes` (lag 1 for AR(1)),
 *  `ccfSpikes` (lag 2 for a series driven by x lagged twice), `collinearPair`
 *  (a proportional pair) and `heteroscedastic` (level-proportional noise).
 *  > runMain scalation.mathstat.correlogramTest2
 */
@main def correlogramTest2 (): Unit =

    banner ("ljungBox: hand-checked statistic")
    val e0 = VectorD (1, 3, 2, 5, 4, 6, 8, 7, 9, 10, 8, 12)
    val (q0, p0) = ljungBox (e0, 3)
    println (s"Q = $q0 (expect 10.931087), p-value = $p0 (expect 0.012104)")
    assert (math.abs (q0 - 10.931086991432) < 1E-9 && math.abs (p0 - 0.012104410741) < 1E-4)

    val n   = 300
    val rvg = random.Normal (0.0, 1.0, 1)                                // white noise, stream 1
    val g   = VectorD (for _ <- 0 until n yield rvg.gen)
    val e   = new VectorD (n)                                            // AR(1) noise, phi = .9
    for t <- 1 until n do e(t) = 0.9 * e(t-1) + g(t)

    banner ("ljungBox: white noise (keep H0) vs. AR(1) noise (reject H0)")
    val (qg, pg) = ljungBox (g, 10)
    val (qe, pe) = ljungBox (e, 10)
    println (s"white noise: Q = $qg, p-value = $pg;  AR(1): Q = $qe, p-value = $pe")
    assert (pg > 0.001 && pe < 0.001)

    banner ("ljungBoxTest via the Correlogram trait")
    object CT extends Correlogram (e)
    CT.makeCorrelogram ()
    println (s"CT.ljungBoxTest () = ${CT.ljungBoxTest ()} (same as ljungBox (e, 10))")
    assert (CT.ljungBoxTest ()._1 == qe)

    banner ("acfSpikes: AR(1) noise spikes at lag 1 first; used lags are skipped")
    val spikes = acfSpikes (e, 3)
    println (s"acfSpikes (e, 3) = $spikes")
    assert (spikes.nonEmpty && spikes.head._1 == 1)
    assert (acfSpikes (e, 3, Set (1, 2, 3)).isEmpty)
    println (s"acfSpikes (g, 3) = ${acfSpikes (g, 3)} (white noise: usually none)")

    banner ("ccfSpikes: residuals driven by x lagged twice")
    val rvx = random.Normal (0.0, 1.0, 2)
    val x   = VectorD (for _ <- 0 until n yield rvx.gen)
    val e2  = x.lagv (2) * 0.8 + g * 0.5
    val ccf = ccfSpikes (e2, x, 4)
    println (s"ccfSpikes (e2, x, 4) = $ccf")
    assert (ccf.nonEmpty && ccf.head._1 == 2)
    assert (ccfSpikes (e2, x, 4, Set (2)).forall (_._1 != 2))

    banner ("collinearPair: columns 0 and 1 are proportional")
    val xm = MatrixD (for t <- 0 until n yield VectorD (t.toDouble, 2.0 * t + 1.0, g(t)))
    val (bi, bj, bc) = collinearPair (xm)
    println (s"pair = ($bi, $bj), |corr| = $bc")
    assert (bi == 0 && bj == 1 && bc > 0.999)
    assert (collinearPair (xm, IndexedSeq (2)) == (-1, -1, 0.0))

    banner ("heteroscedastic: level-proportional vs. constant noise")
    val lev  = VectorD (for t <- 0 until n yield 10.0 + t)
    val eLev = lev * g                                                   // spread grows with the level
    println (s"level-proportional: ${heteroscedastic (eLev, lev)},  constant: ${heteroscedastic (g, lev)}")
    assert (heteroscedastic (eLev, lev) && ! heteroscedastic (g, lev))

end correlogramTest2

