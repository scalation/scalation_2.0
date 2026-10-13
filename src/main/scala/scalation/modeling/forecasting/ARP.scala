
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Siddhant Roy
 *  @version 2.0
 *  @date    Tue May 12 13:59:14 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Probabilistic Auto-Regressive AR(p) with Prediction Intervals
 *
 *  Extends AR to add a probabilistic layer:
 *    - Produces prediction intervals at any significance level α
 *    - Supports all six PI metrics: PICP, PINC, ACE, PINAW, MIS, WIS
 *    - Computes MA(∞) impulse response coefficients (ψ weights) from fitted φ
 *      - Uses ψ weights to compute theoretically correct h-step forecast variance
 *    - Alternatively uses Empirical (Re-sampling/Bootstrapping)
 *
 *  @see https://online.stat.psu.edu/stat510/Lesson03
 */

package scalation
package modeling
package forecasting

import scala.math.{max, min, sqrt}

import scalation.mathstat._

//import Predictor.plotPredictionInt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARP` class extends `AR` to provide probabilistic forecasting with
 *  prediction intervals.  It computes MA(∞) impulse response coefficients (ψ weights)
 *  from the fitted AR(p) parameters φ, then uses them to derive the theoretically
 *  correct h-step-ahead forecast error variance:
 *
 *      σ²_h = σ²_e * Σ (ψ²_j,  j = 0, 1, ..., h-1)
 *
 *  where ψ_0 = 1 and ψ_j = Σ_i φ_i * ψ_{j-i} for j >= 1.
 *
 *  @param y         the response vector (time series data)
 *  @param hh        the maximum forecasting horizon (h = 1 to hh)
 *  @param tRng      the time range, if relevant (time index may suffice)
 *  @param hparam    the hyper-parameters (defaults to AR.hp)
 *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
 *  @param adjusted  whether in `Correlogram` when calculating auto-covariances/auto-correlations
 *                   to adjust to account for the number of elements in the sum Σ (or use dim-1)
 *  @param tForm     the transformation applied
 *  @param resample  whether to use the empirical re-sampling or theoretical method
 */
class ARP (y: VectorD, hh: Int, tRng: Range = null,
           hparam: HyperParameter = AR.hp,
           bakcast: Boolean = false, adjusted: Boolean = true,
           tForm: Transform = null,
           resample: Boolean = false)                                   // empirical re-sampling or theoretical
    extends AR (y, hh, tRng, hparam, bakcast, adjusted, tForm):

    private val flaw_p = flawf ("ARP")                                  // flaw function for ARP


    private var psiW: VectorD = null // MA(∞) impulse response weights (ψ)
    private var sig2: Double = -1.0 // estimated innovation/error variance (σ²_e)
    private var sig2H: VectorD = null // NEW: empirical h-step variances, indexed 0..hh

    _modelName = s"ARP_$p"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `ARP` model to the time-series data in vector y_.
     *  First calls AR.train to estimate φ via Durbin-Levinson, then computes
     *  the MA(∞) impulse response coefficients (ψ weights) and error variance.
     *  @param x_null  the data/input matrix (ignored, pass null)
     *  @param y_      the training/full response vector (e.g., full y)
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =
        super.train (x_null, y_)                                        // AR train: estimates b (φ) and δ
        psiW  = computePsiWeights (b, hh)                               // compute MA(∞) ψ weights up to horizon hh
        sig2  = estimateSig2 (y_)                                       // estimate innovation variance σ²_e
        sig2H = estimateSig2H (y_, hh)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the MA(∞) impulse response coefficients (ψ weights) from the
     *  AR(p) coefficients φ (stored in b).
     *
     *  Recursion:
     *      ψ_0 = 1
     *      ψ_j = Σ_{i=1}^{min(j,p)} φ_i * ψ_{j-i}     for j = 1, 2, ..., maxH
     *
     *  Note: b is stored as [φ_1, φ_2, ..., φ_p] (i.e., b(0) = φ_1).
     *
     *  @param phi   the AR coefficient vector [φ_1, ..., φ_p]
     *  @param maxH  the maximum horizon (number of ψ weights to compute beyond ψ_0)
     */
    private def computePsiWeights (phi: VectorD, maxH: Int): VectorD =
        val psi = new VectorD (maxH + 1)                                // ψ_0 through ψ_maxH
        psi(0)  = 1.0                                                   // ψ_0 = 1 by definition
        for j <- 1 to maxH do
            var sum = 0.0
            for i <- 1 to min (j, phi.dim) do sum += phi(i-1) * psi(j-i)   // φ_i * ψ_{j-i}
            psi(j) = sum
        end for
        psi
    end computePsiWeights

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the innovation variance σ²_e from the training residuals.
     *  Uses the one-step-ahead prediction errors on the training set.
     *  @param y_  the training response vector
     */
    private def estimateSig2 (y_ : VectorD): Double =
        var sse = 0.0
        var count = 0
        for t <- p until y_.dim do                                      // start at p so full AR formula is available
            val yhat = predict (t, y_)
            val err  = y_(t) - yhat
            sse   += err * err
            count += 1
        end for
        if count > 0 then sse / count else 0.0                         // MLE-style (divide by count, not count-1)
    end estimateSig2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute empirical h-step forecast variance for h = 1..maxH by iterating
     *  the AR(p) recursion h times from real y(t-h), y(t-h-1), ..., y(t-h-p+1)
     *  and averaging the squared residuals y(t) - yhat_h(t).
     *  @param y_    the training response vector
     *  @param maxH  the maximum horizon to compute variances for
     */
    private def estimateSig2H (y_ : VectorD, maxH: Int): VectorD =
        val out = new VectorD (maxH + 1)
        out(0)  = 0.0
        for h <- 1 to maxH do
            var sse = 0.0
            var count = 0
            for t <- (p + h - 1) until y_.dim do
                // state in lag order: [y(t-h), y(t-h-1), ..., y(t-h-p+1)]
                val st = Array.tabulate (p)(i => y_(t - h - i))
                var yhat = 0.0
                for _ <- 1 to h do
                    yhat = δ
                    for i <- 0 until p do yhat += b(i) * st(i)
                    // shift state: [yhat, st(0), ..., st(p-2)]
                    var i = p - 1
                    while i >= 1 do { st(i) = st(i-1); i -= 1 }
                    st(0) = yhat
                end for
                val err = y_(t) - yhat
                sse   += err * err
                count += 1
            end for
            out(h) = if count > 0 then sse / count else 0.0
        end for
        out
    end estimateSig2H

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the h-step-ahead forecast error variance using the ψ weights.
     *
     *      σ²_h = σ²_e * Σ (ψ²_j,  j = 0, ..., h-1)
     *
     *  @param h  the forecasting horizon
     */
    def forecastVar (h: Int): Double =
        if resample then sig2H (h)                                   // empirical re-sampling
        else                                                           // default: ψ-weight (theoretical)
            var cumPsi2 = 0.0
            for j <- 0 until h do cumPsi2 += psiW(j) * psiW(j)         // Σ ψ²_j
            sig2 * cumPsi2
    end forecastVar

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the h-step-ahead forecast standard deviation.
     *  @param h  the forecasting horizon
     */
    def forecastStd (h: Int): Double = sqrt (forecastVar (h))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast intervals for all y_.dim time points at horizon h (h-steps ahead).
     *  Create prediction intervals (two vectors) for the given time points at level p.
     *  Overrides the naive approach in `Forecaster` with theoretically correct
     *  h-step variance from the MA(∞) ψ weights.
     *  @param y_   the aligned actual values to use in making forecasts
     *  @param yfh  the forecast vector at horizon h
     *  @param h    the forecasting horizon, number of steps ahead to produce forecasts
     *  @param p_   the level (1 - α) for the prediction interval, e.g., 0.9 for 90%
     */
    override def forecastAtI (y_ : VectorD, yfh: VectorD, h: Int, p_ : Double = 0.9): (VectorD, VectorD) =
        if psiW == null then
            flaw_p ("forecastAtI", "must call train before forecastAtI")
            return super.forecastAtI (y_, yfh, h, p_)                  // fallback to naive
        val sig_h = forecastStd (h)                                    // theoretically correct std dev at horizon h
        val width = z_sigma (sig_h, p_)                                // interval half width: z_{(1+p)/2} * σ_h
        (yfh - width, yfh + width)                                     // return (lower, upper) bounds
    end forecastAtI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce prediction intervals at multiple significance levels for a given horizon.
     *  Returns matrices of lower and upper bounds (one row per α level), suitable
     *  for computing WIS via `diagnose_pi`.
     *  @param y_   the actual values
     *  @param yfh  the forecast vector at horizon h
     *  @param h    the forecasting horizon
     *  @param α    the vector of significance levels (defaults to Fit.α_)
     */
    def forecastAtI_ (y_ : VectorD, yfh: VectorD, h: Int, α: VectorD = Fit.α_): (MatrixD, MatrixD) =
        val lowM = new MatrixD (α.dim, yfh.dim)
        val upM  = new MatrixD (α.dim, yfh.dim)
        for k <- α.indices do
            val p_k = 1.0 - α(k)                                       // coverage level for this α
            val (low_k, up_k) = forecastAtI (y_, yfh, h, p_k)
            lowM(k) = low_k
            upM(k)  = up_k
        end for
        (lowM, upM)
    end forecastAtI_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose all PI metrics for horizon h, including PICP, PINC, ACE, PINAW,
     *  MIS, and WIS.  Calls the `diagnose_pi` method from `Fit`.
     *  @param y_   the actual values (aligned to forecasts)
     *  @param yfh  the point forecast vector at horizon h
     *  @param h    the forecasting horizon
     *  @param α    the vector of significance levels
     *  @param iα   the index into α for the main/primary significance level
     */
    def diagnosePI (y_ : VectorD, yfh: VectorD, h: Int,
                    α: VectorD = Fit.α_, iα: Int = 2): (VectorD, Int) =
        val skip  = max (h, p)                                         // skip invalid leading entries
        val y_s   = y_(skip until y_.dim)                              // sliced actual
        val yfh_s = yfh(skip until yfh.dim)                            // sliced forecast
        val (lowM, upM) = forecastAtI_(y_s, yfh_s, h, α)               // bounds on clean data
        diagnose_pi(y_s, yfh_s, (lowM, upM), α, iα)
    end diagnosePI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose all horizons for both point forecasts and prediction intervals.
     *  Extends the standard `diagnoseAll` with PI metrics.
     *  @param y_   the actual response vector
     *  @param yf_  the forecast matrix
     *  @param α    the vector of significance levels
     *  @param iα   the index into α for the main significance level
     */
    def diagnoseAllPI (y_ : VectorD, yf_ : MatrixD,
                       α: VectorD = Fit.α_, iα: Int = 2): Unit =
        banner (s"diagnoseAllPI: Evaluate ${modelName}'s PI QoF for horizons 1 to $hh")
        for h <- 1 to hh do
            val yfh = yf_(?, h)                                        // h-step forecast vector
            val yy  = y_                                               // actual values
            val (qof, idx) = diagnosePI (yy, yfh, h, α, iα)
            println (s"Horizon h = $h, primary α index = $idx")
            showQoF (qof)
        end for
    end diagnoseAllPI

end ARP


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARP` companion object provides factory methods for the `ARP` class.
 */
object ARP:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARP` object.
     *  @param y         the response vector (time series data)
     *  @param hh        the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng      the time range, if relevant (time index may suffice)
     *  @param hparam    the hyper-parameters
     *  @param bakcast   whether a backcasted value is prepended to the time series
     *  @param adjusted  whether to adjust auto-covariance calculations
     *  @param resample  whether to use the empirical re-sampling or theoretical method
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null,
               hparam: HyperParameter = AR.hp,
               bakcast: Boolean = false, adjusted: Boolean = true,
               resample: Boolean = false): ARP =
      new ARP (y, hh, tRng, hparam, bakcast, adjusted, null, resample)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARP` object with rescaling.
     *  @param y         the response vector (time series data)
     *  @param hh        the maximum forecasting horizon (h = 1 to hh)
     *  @param tRng      the time range, if relevant (time index may suffice)
     *  @param hparam    the hyper-parameters
     *  @param bakcast   whether a backcasted value is prepended to the time series
     *  @param adjusted  whether to adjust auto-covariance calculations
     *  @param resample  whether to use the empirical re-sampling or theoretical method
     */
    def rescale (y: VectorD, hh: Int, tRng: Range = null,
                 hparam: HyperParameter = AR.hp,
                 bakcast: Boolean = false, adjusted: Boolean = true,
                 resample: Boolean = false): ARP =
        val tr_size = Model.trSize(y.dim)
        val tForm_y = NormForm(y(0 until tr_size))
        val y_scl   = tForm_y.f(y)
        new ARP (y_scl, hh, tRng, hparam, bakcast, adjusted, tForm_y, resample)
    end rescale

end ARP

import Example_LakeLevels.y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRPTest` main function tests the `ARP` class on real data:
 *  Forecasting Lake Levels with prediction intervals using In-Sample Testing.
 *  > runMain scalation.modeling.forecasting.aRPTest
 */
@main def aRPTest (): Unit =

    val hh = 3                                                         // maximum forecasting horizon

    val mod = new ARP (y, hh)                                          // create probabilistic AR model
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                // train and test on full dataset

    // Point forecasts
    mod.forecastAll ()                                                 // forecast h-steps ahead for all y
    mod.diagnoseAll (y, mod.getYf)

    // Prediction intervals (PI) for each horizon
    banner ("Prediction Interval Diagnostics")
    mod.diagnoseAllPI (y, mod.getYf)

    // Show ψ weights and variance growth
    banner ("Forecast Variance by Horizon")
    for h <- 1 to hh do
        println (s"h = $h: σ²_h = ${mod.forecastVar(h)}, σ_h = ${mod.forecastStd(h)}")

    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    // Print prediction interval bounds for each horizon
    for h <- 1 to hh do
        val yfh = mod.getYf(?, h)                                      // point forecast at horizon h
        val (low, up) = mod.forecastAtI (y, yfh, h, 0.9)               // 90% PI
        println (s"\n--- Horizon h = $h ---")
        println (s"lower = ${low.drop()}")
        println (s"upper = ${up.drop()}")
        println (s"width = ${up.drop() - low.drop()}")
        val ys = MatrixD (yfh.drop(), y.drop(), low.drop(), up.drop())
        new PlotM (null, ys, Array ("yp", "yy", "low", "up"),          // plot ordered actual, predicted, lower, upper
        s"Plot AR prediction intervals [low, up]", lines = true)
    end for

end aRPTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRPTest2` main function tests the `ARP` class on real data:
 *  Forecasting Lake Levels with prediction intervals using TnT Rolling Validation.
 *  > runMain scalation.modeling.forecasting.aRPTest2
 */
@main def aRPTest2 (): Unit =

    val hh = 3                                                         // maximum forecasting horizon

    val mod = new ARP (y, hh)                                          // create probabilistic AR model
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()                                                // train and test on full dataset

    mod.setSkip ()                                                     // don't skip any rows
    mod.rollValidate ()                                                // TnT with Rolling Validation

    // PI diagnostics on the test set
    banner ("Prediction Interval Diagnostics (TnT)")
    mod.diagnoseAllPI (y, mod.getYf)

    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRPTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRPTest3` main function tests the `ARP` class on COVID-19 data:
 *  Compares different AR orders (p = 1 to 5) with prediction intervals.
 *  > runMain scalation.modeling.forecasting.aRPTest3
 */
@main def aRPTest3 (): Unit =

    import Example_Covid.{clip, loadData_y}

    val (_, y) = clip ((null, loadData_y ()))
    val hh     = 6                                                     // maximum forecasting horizon

    for p <- 1 to 5 do
        AR.hp("p") = p
        val mod = new ARP (y, hh)
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()

        mod.forecastAll ()
        mod.diagnoseAll (y, mod.getYf)

        // PI diagnostics
        banner (s"PI Diagnostics for ${mod.modelName}")
        mod.diagnoseAllPI (y, mod.getYf)

        // Variance growth across horizons
        for h <- 1 to hh do
            println (s"  h = $h: σ²_h = ${mod.forecastVar(h)}, σ_h = ${mod.forecastStd(h)}")
    end for

end aRPTest3

