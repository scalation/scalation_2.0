
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Tue Oct 13 16:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Exploratory Data Analysis (EDA) Profile of a Time Series
 *           (profile and residual report)
 *
 *  Everything here is computed from the TRAINING rows only (rows 0 until trSize,
 *  ScalaTion's `Model.trSize`), so nothing from the testing rows reaches a model
 *  choice or the exogenous screening (@see `ExoScreen`, which uses the same
 *  lagged cross-correlation to pick the drivers).  The
 *  profile is what a statistician would look at before writing an ARX formula:
 *  the ACF and PACF of the response (which lags matter), the periodogram's top
 *  frequencies (seasonality), the cross-correlation of the response with each
 *  exogenous variable by lag (which lag of which driver, and with which sign),
 *  the level and spread (whether a log transform is plausible), the share of
 *  zeros (whether log is safe) and a KPSS stationarity test (a formula search may
 *  render it in words for a prompt); `residualReport` applies the residual tests
 *  of `mathstat` (Ljung-Box, ACF spikes, collinearity, heteroscedasticity) to a
 *  fitted model.  Replaces the former `ResidualDiagnostics`.  rev4.1: moved from
 *  `nonlinear/llm_narx_sr` to `forecasting`; the prompt rendering (`promptBlock`)
 *  moved to the search that uses it; `maxLag` and `respName` are now fields.
 *  > runMain scalation.modeling.forecasting.eDA4TSTest
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EDA4TS` class profiles a response series and its exogenous variables on
 *  their training rows.
 *  @param y       the response series (full length; only the first trSize rows are read)
 *  @param exo     the exogenous series by name (same length as y)
 *  @param trSize  the number of leading (training) rows the profile is computed on
 *  @param maxLag    the maximum lag examined (ACF, PACF, CCF)
 *  @param nFreqs    the number of periodogram frequencies reported
 *  @param respName  the response variable's name (the key of a display-name map)
 */
class EDA4TS (y: VectorD, exo: Map [String, VectorD], trSize: Int,
              val maxLag: Int = 6, nFreqs: Int = 2, val respName: String = "y"):

    private val debug = debugf ("EDA4TS", false)                        // debug function

    require (trSize > 2 * maxLag + 2, s"EDA4TS: trSize = $trSize too small for maxLag = $maxLag")

    val yTr: VectorD = y(0 until trSize)                                // training rows of the response
    val exoTr: Map [String, VectorD] = exo.map ((k, v) => k -> v(0 until trSize))
    val n: Int = yTr.dim

    private class Corr (v: VectorD) extends Correlogram (v)            // ACF/PACF via Durbin-Levinson

    /** The ACF and PACF of the response at lags 0..maxLag (index = lag).
     */
    val (acf, pacf) =
        val ct = new Corr (yTr)
        ct.makeCorrelogram ()
        (ct.acF (0 until maxLag + 1), ct.pacF (0 until maxLag + 1))     // lags 0..maxLag (`until`: VectorD slices exclusive ends)

    /** The ~95% band for a correlation at this sample size (2 / sqrt n).
     */
    val band: Double = 2.0 / math.sqrt (n.toDouble)

    /** The top periodogram frequencies (cycles per step) and their periods.
     */
    val freqs: VectorD = EDA4TS.topFreqs (yTr, nFreqs)                  // a direct DFT (@see the companion)
    val periods: VectorD = freqs.map (f => if f > 0.0 then 1.0 / f else Double.NaN)

    /** The KPSS statistic and p-value (small p => reject level stationarity).
     */
    val (kpssStat, kpssP) =
        try
            val test = new Stationarity_KPSS (yTr, 0, "short", "c")
            (test.statistic (), test.pvalue ())
        catch case e: Exception =>
            debug ("init", s"KPSS failed: ${e.getMessage}")
            (Double.NaN, Double.NaN)

    /** The share of zeros in the response (log is unsafe when > 0) and its level.
     */
    val zeroShare: Double = yTr.count (_ == 0.0).toDouble / n
    val (yMin, yMax, yMean, yStd) = (yTr.min, yTr.max, yTr.mean, yTr.stdev)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the lags (1..maxLag) at which the PACF of the response exceeds the
     *  band, strongest first: the lags an AR term should carry.
     */
    def pacfSpikes: Seq [(Int, Double)] =
        (1 to maxLag).map (k => (k, pacf(k))).filter ((_, r) => math.abs (r) > band).sortBy ((_, r) => -math.abs (r))
    end pacfSpikes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the cross-correlation of the response y_t with x_t-k for lags
     *  k = 1..maxLag of an exogenous variable (@see `VectorD.ccorr`: x.ccorr (y, k)
     *  correlates x_i with y_i+k).
     *  @param name  the exogenous variable
     */
    def ccf (name: String): VectorD =
        val x = exoTr (name)
        VectorD (for k <- 1 to maxLag yield x.ccorr (yTr, k))
    end ccf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return, per exogenous variable, its best lag (largest |ccf|) and the
     *  correlation there, strongest variable first.
     */
    def bestLags: Seq [(String, Int, Double)] =
        exo.keys.toSeq.sorted.map { nm =>
            val c = ccf (nm)
            val k = c.abs.argmax ()
            (nm, k + 1, c(k))
        }.sortBy ((_, _, r) => -math.abs (r))
    end bestLags

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the expected sign of each exogenous variable's effect from the
     *  training correlation at its best lag (+1, -1, or 0 when inside the band).
     */
    def signs: Map [String, Int] =
        bestLags.map ((nm, _, r) => nm -> (if math.abs (r) <= band then 0 else if r > 0.0 then 1 else -1)).toMap
    end signs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Report the residual tests of a fitted model in words: Ljung-Box whiteness,
     *  ACF spikes of the residuals, the most collinear pair of design columns and
     *  heteroscedasticity (spread growing with the level).  Returns the sentences
     *  and a cleanliness score in [0, 1] (the share of the four tests passed).
     *  @param e      the (training) residuals of the model
     *  @param x      the design matrix rows the residuals belong to
     *  @param names  the column names of x
     *  @param yv     the response values the residuals belong to
     */
    def residualReport (e: VectorD, x: MatrixD, names: Array [String], yv: VectorD): (Seq [String], Double) =
        val out    = VEC [String] ()
        var passed = 0
        val (q, p) = ljungBox (e, maxLag)
        if p.isNaN || p >= 0.05 then passed += 1
        out += f"Ljung-Box Q($maxLag) = $q%.2f, p = $p%.3f" + (if ! p.isNaN && p < 0.05 then " (residuals autocorrelated)" else " (white)")
        val spikes = acfSpikes (e, maxLag)
        if spikes.isEmpty then passed += 1
        else out += "residual ACF spikes at lags " + spikes.map ((k, r) => f"$k (${r}%.2f)").mkString (", ")
        if x.dim2 > 1 then
            val (i, j, c) = collinearPair (x)
            if c < 0.95 then passed += 1
            else out += f"collinear columns ${names(i)} and ${names(j)} (|corr| = $c%.2f)"
        else passed += 1
        if heteroscedastic (e, yv) then out += "residual spread grows with the level (consider TFORM: Log1p)"
        else passed += 1
        (out.toSeq, passed / 4.0)
    end residualReport

end EDA4TS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EDA4TS` companion object holds the periodogram helper shared by the EDA
 *  profile and a formula search (its wave frequencies).
 */
object EDA4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the single-sided amplitude spectrum of a centered series by a direct
     *  DFT (O(n^2), under a second for 10,000 rows): index j = 1 .. n/2 - 1 is the
     *  frequency j/n cycles per step (index 0, the mean, is left at zero).
     *  @param y  the series
     */
    def amplitudes (y: VectorD): VectorD =
        val n    = y.dim
        val yc   = y - y.mean
        val amps = new VectorD (n / 2)
        for j <- 1 until n / 2 do
            var re = 0.0; var im = 0.0
            val w  = 2.0 * math.Pi * j / n
            for t <- 0 until n do
                re += yc(t) * math.cos (w * t)
                im -= yc(t) * math.sin (w * t)
            amps(j) = math.sqrt (re * re + im * im)
        amps
    end amplitudes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the top k frequencies (cycles per step, highest amplitude first) of a
     *  series (e.g., the training rows of the response).
     *  @param y  the series
     *  @param k  the number of frequencies
     */
    def topFreqs (y: VectorD, k: Int): VectorD =
        val amps = amplitudes (y)
        val n    = y.dim
        VectorD ((1 until amps.dim).sortBy (j => -amps(j)).take (k).map (_.toDouble / n))
    end topFreqs

end EDA4TS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `eDA4TSTest` main function tests the `EDA4TS` class on a synthetic series
 *  with a planted structure: y depends on its own lag 1, on z at lag 2 (positive)
 *  and has a 12-step cycle, so the profile must find the PACF spike at lag 1,
 *  the best lag 2 with a positive sign for z, and a period near 12; a wrong model
 *  (`ARX_D` with p = 1 and no exogenous variable, i.e., y ~= y~1) must fail at
 *  least one residual test.  (rev4.1: the prompt-block check moved to `lLMProposerTest`.)
 *  > runMain scalation.modeling.forecasting.eDA4TSTest
 */
@main def eDA4TSTest (): Unit =

    import scalation.random.Normal

    val n     = 320                                                      // 256 training rows: a power of two for the FFT
    val noise = Normal (0.0, 1.0, 3)
    val zz    = Normal (0.0, 4.0, 4)
    val z     = VectorD (for _ <- 0 until n yield 10.0 + zz.gen)
    val y     = new VectorD (n)
    y(0) = 20.0; y(1) = 21.0
    for t <- 2 until n do
        y(t) = 5.0 + 0.7 * y(t-1) + 1.5 * z(t-2) + 4.0 * math.sin (2.0 * math.Pi * t / 12.0) + noise.gen

    val eda = new EDA4TS (y, Map ("z" -> z), trSize = 256, maxLag = 6, nFreqs = 2)

    banner ("profile")
    println (s"acf   = ${eda.acf}")
    println (s"pacf  = ${eda.pacf}")
    println (s"pacf spikes = ${eda.pacfSpikes}")
    println (s"freqs = ${eda.freqs}, periods = ${eda.periods}")
    println (s"best lags   = ${eda.bestLags}")
    println (s"signs = ${eda.signs}, KPSS p = ${eda.kpssP}")

    assert (eda.pacfSpikes.nonEmpty && eda.pacfSpikes.head._1 == 1, "PACF spike at lag 1 expected")
    val (nm, k, r) = eda.bestLags.head
    assert (nm == "z" && k == 2 && r > 0.0, s"expected z at lag 2 with a positive sign, got ($nm, $k, $r)")
    assert (eda.signs ("z") == 1, "sign of z must be +1")
    assert (eda.periods.exists (p => math.abs (p - 12.0) < 2.0), s"a period near 12 expected: ${eda.periods}")

    banner ("residual report on a wrong model (y ~= y~1 only)")
    val tr  = Model.trSize (n)                                           // the training rows
    val mod = ARX_D (new MatrixD (n, 0), y, 1, hparam = AR.hp.updateReturn (("p", 1), ("px", 0), ("spec", 0)))
    mod.train (mod.getX (0 until tr), mod.getYy (0 until tr))            // fit on the training rows
    mod.test (mod.getX, mod.getYy)                                       // the h = 1 predictions of every row (yf)
    val e   = (mod.getY - mod.getYf(?, 1))(1 until tr)
    val (lines, clean) = eda.residualReport (e, mod.getX (1 until tr), mod.getFname, mod.getY (1 until tr))
    for l <- lines do println (s"  $l")
    println (s"cleanliness = $clean")
    assert (clean < 1.0, "a model missing z~2 and the cycle should fail at least one residual test")
    println ("eDA4TSTest: passed")

end eDA4TSTest
