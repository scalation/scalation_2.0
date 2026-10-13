
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Nirupom Bose Roy, Lokesh Adusumilli
 *  @version 2.0
 *  @date    June 15 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive Moving Average via Kalman Filter MLE (ARMA_KF)
 *
 *  Fits ARMA(p, q) by Kalman-filter maximum-likelihood estimation (KF-MLE),
 *  matching R's arima(method="ML", transform.pars=TRUE) behaviour:
 *    - AR and MA coefficients are optimised via the Durbin-Levinson PACF
 *      reparameterization, guaranteeing stationarity and invertibility
 *      without explicit box constraints (equivalent to R's transform.pars=TRUE).
 *    - Process mean μ is optimised directly (equivalent to R's include.mean=TRUE).
 *    - Process variance σ² is concentrated out via profile MLE, reducing the
 *      optimisation dimension by one and matching R's profile log-likelihood.
 *    - The KF is initialised with the unconditional Lyapunov covariance P_L
 *      (discrete doubling algorithm), matching R's Gardner et al. (AS 154) init so that
 *      all training observations contribute to the likelihood.
 *    - A single OLS-AR warm-start (sample-mean prior) seeds the optimizer.
 *
 *  Expected rolling-validation sMAPE on COVID-19 weekly deaths (train=92, test=24, 200/n):
 *    ARMA_KF(5,0)  h=1..6:  17.007,  21.344,  27.958,  38.041,  40.604,  47.962
 *    ARMA_KF(5,1)  h=1..6:  17.654,  22.263,  28.700,  38.868,  42.530,  51.061
 *    ARMA_KF(5,3)  h=1..6:  17.469,  22.042,  28.567,  38.663,  41.432,  49.980
 *
 *  References:
 *    - Gardner, G., Harvey, A.C., Phillips, G.D.A. (1980). "Algorithm AS 154:
 *      An Algorithm for Exact ML Estimation of ARMA Models by Means of Kalman
 *      Filtering." Applied Statistics, 29(3), 311-322.   (KF-MLE + KF initialisation)
 *    - Jones, R.H. (1980). "Maximum Likelihood Fitting of ARMA Models to Time
 *      Series with Missing Observations." Technometrics, 22(3), 389-395.
 *    - Monahan, J.F. (1984). "A Note on Enforcing Stationarity in ARMA Models."
 *      Biometrika, 71(2), 403-404.   (PACF reparameterization)
 *    - Harvey, A.C. (1989). Forecasting, Structural Time Series Models and the
 *      Kalman Filter. Cambridge University Press.
 *    - Byrd, R.H., Lu, P., Nocedal, J., Zhu, C. (1995). "A Limited Memory Algorithm
 *      for Bound Constrained Optimization." SIAM J. Sci. Comput., 16(5), 1190-1208.
 *    - R: stats::arima  —
 *      https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html
 *
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest2
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest3
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest4
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest5
 */

package scalation
package modeling
package forecasting

import scala.math.{max, log, Pi, tanh, exp}
import scala.util.boundary

import scalation.mathstat._
import scalation.mathstat.MatrixD.outer
//import scalation.optimization.quasi_newton.LBFGS_B as Optimizer     // analytic-gradient L-BFGS-B
import scalation.optimization.quasi_newton.LBFGS_B_KF as Optimizer    // analytic-gradient L-BFGS-B (ARMA_KF-specific; shared LBFGS_B untouched)
import scalation.modeling.forecasting.KalmanFilterKF as KalmanFilter  // KF-MLE filter (updates x/p in predict); shadows the simulation KalmanFilter

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA_KF` companion object provides the default hyper-parameters and a
 *  factory method for the `ARMA_KF` class.
 */
object ARMA_KF:

    val hp = new HyperParameter
    hp += ("p", 1, 1)                                                      // number of AR terms
    hp += ("q", 0, 0)                                                      // number of MA terms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARMA_KF` object.
     *  @param y       the response/time-series vector
     *  @param hh      the maximum forecast horizon
     *  @param tRng    the optional time range (passed to Forecaster)
     *  @param hparam  the hyper-parameters (p, q)
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null,
               hparam: HyperParameter = hp): ARMA_KF =
        new ARMA_KF (y, hh, tRng, hparam)
    end apply

end ARMA_KF


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA_KF` class fits an ARMA(p, q) model via Kalman-filter maximum-
 *  likelihood estimation (KF-MLE) using the PACF stationarity reparameterization.
 *
 *  Unconstrained parameter vector optimised by L-BFGS-B:
 *    u = [ u₁..u_p,  v₁..v_q,  μ,  ℓ ]
 *  where:
 *    φ_i = DL(tanh(u_i))  — AR coefficients via Durbin-Levinson (stationary)
 *    θ_i = DL(tanh(v_i))  — MA coefficients via Durbin-Levinson (invertible)
 *    μ   = process mean (optimised directly)
 *    σ²  = exp(ℓ)          — process variance (always positive)
 *
 *  After training, the model exposes the full `Forecaster` interface:
 *    - `predict(t, y_)`:     sequential 1-step-ahead KF prediction
 *    - `forecast(t, y_)`:    multi-step projection from the running KF state
 *    - `forecastAt(h, y_)`:  full fresh-pass h-step forecast for all time points
 *
 *  @param y        the response/time-series vector
 *  @param hh       the maximum forecast horizon
 *  @param tRng     the optional time range
 *  @param hparam   the hyper-parameters (p, q)
 *  @param bakcast  whether to prepend a backcast value (default false)
 */
class ARMA_KF (y: VectorD, hh: Int, tRng: Range = null,
               hparam: HyperParameter = ARMA_KF.hp, bakcast: Boolean = false)
      extends Forecaster (y, hh, tRng, hparam, bakcast)
         with NoSubModels:

    private val flaw  = flawf ("ARMA_KF")
    private val p     = hparam("p").toInt
    private val q     = hparam("q").toInt
    _modelName = s"ARMA_KF($p, $q)"
    private val r_dim = max (p, q + 1)                   // companion-form state dimension

    private var kf_f_mat: MatrixD = null                 // companion-form state transition  F
    private var kf_q_mat: MatrixD = null                 // process noise covariance         Q = G G' σ²
    private var kf_h_mat: MatrixD = null                 // observation matrix               H = [1, 0, …]
    private var kf_r_mat: MatrixD = null                 // observation noise (near-zero)    R → 0⁺
    private var kf_x0:    VectorD = null                 // zero initial state vector
    private var kf_p0:    MatrixD = null                 // Lyapunov unconditional covariance P_L

    private var phi_est:   VectorD = new VectorD (0)     // estimated AR coefficients φ₁..φ_p
    private var theta_est: VectorD = new VectorD (0)     // estimated MA coefficients θ₁..θ_q
    private var mu_est     = 0.0                         // estimated process mean μ
    private var sig2_est   = 0.0                         // estimated process variance σ²

    private var kf_run:       KalmanFilter = null        // live sequential filter advanced by predict()
    private var kf_trained_x: VectorD = null             // post-training state snapshot x_{n|n}
    private var kf_trained_p: MatrixD = null             // post-training covariance snapshot P_{n|n}
    private var pre_x:        VectorD = null             // pre-update state x_{t|t-1} saved by predict()
    private var last_t        = -1                       // last t seen by predict(); detects backward jumps

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inverse hyperbolic tangent: atanh(x) = ½ ln((1+x)/(1−x)).
     *  Not in scala.math; defined here for portability.
     *  @param x  the input value; must satisfy |x| < 1
     *  @return   atanh(x)
     */
    private inline def atanh (x: Double): Double = 0.5 * log ((1.0 + x) / (1.0 - x))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Durbin-Levinson forward recursion: PACF values → AR coefficients.
     *  Guarantees stationarity when all |π_k| < 1 (matches R's pacf.to.ar step).
     *  @param pacf  partial autocorrelations π₁..π_k (0-indexed, length k)
     *  @return      AR coefficients φ₁..φ_k (0-indexed, length k)
     */
    private def pacfToAR (pacf: VectorD): VectorD =
        if pacf.dim == 0 then return new VectorD (0)
        var a = VectorD (pacf(0))                        // order-1 seed
        var i = 1
        while i < pacf.dim do
            val pi    = pacf(i)
            val new_a = new VectorD (i + 1)
            var j = 0
            while j < i do
                new_a(j) = a(j) - pi * a(i - 1 - j)
                j += 1
            end while
            new_a(i) = pi
            a = new_a
            i += 1
        end while
        a
    end pacfToAR

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Durbin-Levinson inverse recursion: AR coefficients → PACF values.
     *  Inverse of pacfToAR: given phi = [φ₁..φ_p], returns π = [π₁..π_p].
     *  Returns null if any intermediate |πₖ| ≥ 1 (non-stationary input).
     *  Used to warm-start the optimizer from the OLS AR(p) estimate.
     *  @param phi  AR coefficients [φ₁..φ_p] (0-indexed, length p)
     */
    private def arToPACF (phi: VectorD): VectorD =
        val k = phi.dim
        if k == 0 then return new VectorD (0)
        val pacf = new VectorD (k)
        var a    = phi.copy
        var step = k
        var ok   = true
        while step >= 2 && ok do
            val pk = a(step - 1)
            if math.abs (pk) >= 1.0 then ok = false
            else
                pacf(step - 1) = pk
                val denom = 1.0 - pk * pk
                val new_a = new VectorD (step - 1)
                var j = 0
                while j < step - 1 do
                    new_a(j) = (a(j) + pk * a(step - 2 - j)) / denom
                    j += 1
                end while
                a = new_a
            end if
            step -= 1
        end while
        if !ok then return null
        if a.dim > 0 then pacf(0) = a(0)
        pacf
    end arToPACF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit AR(p) by OLS on the zero-mean training series z.
     *  Design matrix: X(i, j) = z(i + p - 1 - j), response = z(p until n).
     *  Returns the p-vector of OLS AR coefficients [φ₁..φ_p].
     *  Unlike sample-PACF-based YW, OLS avoids the (1−ρ₁²) near-zero
     *  denominator amplification that occurs when ρ₁ ≈ 1.
     *  @param z  zero-mean time series of length n
     *  @param p  AR order
     */
    private def olsAR (z: VectorD, p: Int): VectorD =
        val n    = z.dim
        if p <= 0 || n <= p then return new VectorD (0)
        val nRow = n - p
        val X    = new MatrixD (nRow, p)
        for i <- 0 until nRow do
            for j <- 0 until p do X(i, j) = z(i + p - 1 - j)    // lags z_{t-1}..z_{t-p}
        val yy = z(p until n)
        Fac_LU.solveOver (X, yy)                                // (X'X)^{-1} X' yy
    end olsAR

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the Kalman filter in companion form from the unconstrained vector
     *  u = [u_AR₁..u_AR_p, u_MA₁..u_MA_q, μ, ℓ].
     *
     *  Stores all system matrices and the stationary P₀ in the class fields
     *  kf_f_mat, kf_q_mat, kf_h_mat, kf_r_mat, kf_x0, kf_p0 as a side-effect,
     *  so predict / forecast / forecastAt can use them without re-constructing.
     *
     *  The steady-state P₀ is found by iterating the prediction-correction
     *  Riccati recursion up to 300 steps. Returns (null, 0.0) if the state
     *  covariance diverges (explosive AR polynomial).
     *
     *  @param u  unconstrained parameter vector (length p + q + 2)
     *  @return   (KalmanFilter with P₀ = P_∞, μ) or (null, 0.0) on failure
     */
    private def formKF (u: VectorD): (KalmanFilter, Double) =
        val phi   = if p > 0 then pacfToAR (u(0 until p).map (tanh))     else new VectorD (0)
        val theta = if q > 0 then pacfToAR (u(p until p + q).map (tanh)) else new VectorD (0)
        val mu    = u(p + q)
        val sig2  = exp (u(p + q + 1))
        val r_obs = 1e-6                                        // R → 0⁺ (near-exact ARMA)

        val f = new MatrixD (r_dim, r_dim)                      // companion-form F
        var j = 0
        while j < p do { f(0, j) = phi(j); j += 1 }
        var i = 1
        while i < r_dim do { f(i, i - 1) = 1.0; i += 1 }

        val g = new VectorD (r_dim)                             // G vector: [1, θ₁, …, θ_q, 0, …]
        g(0) = 1.0
        i = 0
        while i < q && i + 1 < r_dim do { g(i + 1) = theta(i); i += 1 }
        val q_mat = outer (g, g) * sig2                         // Q = G G' σ²
        val h_mat = new MatrixD (1, r_dim); h_mat(0, 0) = 1.0
        val r_mat = MatrixD ((1, 1), r_obs)
        val x0    = new VectorD (r_dim)

        kf_f_mat = f
        kf_q_mat = q_mat
        kf_h_mat = h_mat
        kf_r_mat = r_mat
        kf_x0    = x0

        var p0     = MatrixD.eye (r_dim, r_dim)
        var riccOk = true
        var ri     = 0
        while ri < 1000 && riccOk do
            val s_it = h_mat * p0 * h_mat.transpose + r_mat
            if s_it(0, 0) <= 0.0 then riccOk = false
            else
                val k_it  = p0 * h_mat.transpose * s_it.inverse
                val pu_it = (MatrixD.eye (r_dim, r_dim) - k_it * h_mat) * p0
                p0        = f * pu_it * f.transpose + q_mat
                if p0(0, 0) > 1e14 then riccOk = false
            end if
            ri += 1
        end while
        if !riccOk then return (null, 0.0)

        // Lyapunov initialisation via doubling algorithm.
        //
        // R's arima(method="ML") initialises the Kalman filter with the unconditional
        // state covariance P_L (the solution to the discrete Lyapunov equation
        //   P_L = F P_L F' + Q),
        // NOT the Riccati prediction covariance (which ≈ Q for R_obs → 0).
        //
        // Simple iteration P → F P F' + Q converges as O(ρ^{2k}) per step where
        // ρ = spec_rad(F).  For near-unit-root AR (ρ ≈ 0.999) 1 000 steps still
        // leave a large relative error.  The doubling algorithm fixes this:
        //
        //   P_k  =  Σ_{j=0}^{2^k − 1}  F^j Q F'^j       (starts at P_0 = Q)
        //   A_k  =  F^{2^k}              (starts at A_0 = F)
        //
        //   P_{k+1} = P_k + A_k P_k A_k'
        //   A_{k+1} = A_k A_k
        //
        // After 60 doublings we cover ≈ 2^{60} ≈ 10^{18} impulse-response steps,
        // which is numerically indistinguishable from P_L for any ρ < 1.
        var lyapP = q_mat.copy
        var lyapA = f.copy
        var lyOk  = true
        var li    = 0
        while li < 60 && lyOk do
            val lyapP_new = lyapP + lyapA * lyapP * lyapA.transpose
            if lyapP_new(0, 0) > 1e14 then lyOk = false
            else
                lyapP = lyapP_new
                lyapA = lyapA * lyapA
            end if
            li += 1
        end while
        if !lyOk then return (null, 0.0)

        kf_p0 = lyapP
        (new KalmanFilter (f, q_mat, h_mat, r_mat, x0, lyapP), mu)
    end formKF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample PACF of a zero-mean series via the Durbin-Levinson
     *  recursion on the sample ACF.  All output values are clamped to (−1, 1).
     *  This is the same initialisation strategy used by R's arima(method="ML").
     *  @param z  zero-mean (centred) time series
     *  @param k  number of PACF lags to compute
     */
    private def samplePACF (z: VectorD, k: Int): VectorD =
        val n   = z.dim
        val out = new VectorD (k)
        if k == 0 || n < 2 then return out
        val v0  = (z dot z) / n                                 // sample variance
        if v0 < 1e-12 then return out

        val r = new VectorD (k)                                 // sample ACF lags 1..k
        var lag = 0
        while lag < k do
            var s = 0.0
            var t = lag + 1
            while t < n do { s += z(t) * z(t - lag - 1); t += 1 }
            r(lag) = s / (n * v0)
            lag += 1
        end while

        out(0) = math.max (-0.999, math.min (0.999, r(0)))
        var phi = VectorD (out(0))

        var km = 1
        while km < k do
            var num = r(km)
            var den = 1.0
            var j   = 0
            while j < km do
                num -= phi(j) * r(km - 1 - j)
                den -= phi(j) * r(j)
                j   += 1
            end while
            val pi_k = if math.abs (den) < 1e-12 then 0.0
                       else math.max (-0.999, math.min (0.999, num / den))
            out(km) = pi_k
            val new_phi = new VectorD (km + 1)
            j = 0
            while j < km do
                new_phi(j) = phi(j) - pi_k * phi(km - 1 - j)
                j += 1
            end while
            new_phi(km) = pi_k
            phi = new_phi
            km  += 1
        end while
        out
    end samplePACF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the ARMA_KF model using Kalman-filter MLE with the PACF
     *  reparameterization:
     *  1. Eight warm-starts covering OLS-AR, sample-PACF, zero, and near-unit-root
     *     regimes across multiple mean priors; whichever yields the lowest profile
     *     negLogLik wins.
     *  2. L-BFGS-B minimisation of the innovations negative log-likelihood.
     *  3. Decode the optimal u into φ, θ, μ, σ² and set b = φ ++ θ.
     *  4. Burn the sequential tracker `kf_run` in over the training window.
     *  @param x_null  ignored (no exogenous inputs; pass null)
     *  @param y_      the training response vector
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =
        banner (s"Train $modelName  —  KF-MLE + PACF reparameterization")

        val mu_guess = y_.mean

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Build an unconstrained starting vector from OLS AR(p) centred on mu_s.
         *  Profile layout: [ u_AR₁..u_AR_p | u_MA₁..u_MA_q | μ ]  (p+q+1 components).
         *  Each warm-start computes its own OLS-AR on z = y − mu_s so that the
         *  AR coefficients and the mean prior are mutually consistent.
         *  @param mu_s  the mean prior for this warm-start
         */
        def olsStart (mu_s: Double): VectorD =
            val z_s    = y_ - mu_s
            val phi_s  = if p > 0 then olsAR (z_s, p) else new VectorD (0)
            val pacf_s = if p > 0 then arToPACF (phi_s) else null
            val bv     = new VectorD (p + q + 1)
            if p > 0 && pacf_s != null then
                var ii = 0
                while ii < p do
                    bv(ii) = atanh (math.max (-0.999, math.min (0.999, pacf_s(ii))))
                    ii += 1
                end while
            else if p > 0 then
                val rho1 = samplePACF (z_s, 1)(0)
                bv(0) = atanh (rho1)
            end if
            bv(p + q) = mu_s
            bv
        end olsStart

        val start = olsStart (mu_guess)      // single OLS-AR warm-start, μ = sample mean

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Profile innovations negative log-likelihood with σ² concentrated out.
         *  σ² is replaced by its closed-form MLE σ²*(φ,θ,μ) = (1/n) Σ_t ν_t²/c_t,
         *  reducing the optimisation to p+q+1 parameters.  A unit-variance KF pass
         *  provides ν_t and c_t; the scale of c_t cancels in the profile formula.
         *  Returns ∞ on numerical failure.
         *  @param bv  unconstrained profile vector [ u_AR | u_MA | μ ] (p+q+1 components)
         */
        def negLogLikProfile (bv: VectorD): Double = boundary:
            val bv_full = bv ++ VectorD (0.0)                               // σ²=1
            val (kf, mu_c) = formKF (bv_full)
            if kf == null then boundary.break (Double.PositiveInfinity)
            var sumNu2C = 0.0
            var sumLogC = 0.0
            var tt = 0
            while tt < y_.dim do
                kf.predict ()
                val z   = VectorD (y_(tt) - mu_c)
                val nu  = (z - kf.h * kf.x)(0)
                val c_t = (kf.h * kf.p * kf.h.transpose + kf.r)(0, 0)
                if c_t <= 1e-12 then boundary.break (Double.PositiveInfinity)
                sumNu2C += nu * nu / c_t
                sumLogC += log (c_t)
                kf.update (z)
                tt += 1
            end while
            val n      = y_.dim.toDouble
            val sig2_p = sumNu2C / n
            if sig2_p <= 0.0 then boundary.break (Double.PositiveInfinity)
            n / 2.0 * log (sig2_p) + 0.5 * sumLogC + n / 2.0
        end negLogLikProfile

        val muIdx      = p + q
        val cbrtEpsLoc = math.cbrt (2.220446049250313e-16)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Analytic gradient of the profile negative log-likelihood.
         *  Central differences for the PACF/MA components and the exact
         *  sensitivity recursion for μ:
         *    ∂(−LL_p)/∂μ = −n × [Σ_t (ν_t/c_t)(1 + H Δx_{t|t-1})] / Σ_t (ν_t²/c_t)
         *  Costs ~1 Kalman-filter pass vs ~2(p+q+1) for full finite differences,
         *  which is the main per-fit speedup over a gradient-free optimizer.
         *  @param bv  unconstrained profile vector (p+q+1 components)
         */
        def analyticGrad (bv: VectorD): VectorD =
            val g = new VectorD (bv.dim)
            for k <- 0 until bv.dim if k != muIdx do
                val hk = math.max (math.abs (bv(k)), 1.0) * cbrtEpsLoc
                val xp = bv.copy; xp(k) += hk
                val xm = bv.copy; xm(k) -= hk
                g(k) = (negLogLikProfile (xp) - negLogLikProfile (xm)) / (2.0 * hk)
            end for
            val bv_full = bv ++ VectorD (0.0)
            val (kfS, mu_c) = formKF (bv_full)
            if kfS == null then
                val hk = math.max (math.abs (bv(muIdx)), 1.0) * cbrtEpsLoc
                val xp = bv.copy; xp(muIdx) += hk
                val xm = bv.copy; xm(muIdx) -= hk
                g(muIdx) = (negLogLikProfile (xp) - negLogLikProfile (xm)) / (2.0 * hk)
            else
                var d_x     = new VectorD (r_dim)
                var sumWNu  = 0.0
                var sumNu2C = 0.0
                var tt      = 0
                while tt < y_.dim do
                    kfS.predict ()
                    val s_mat = kfS.h * kfS.p * kfS.h.transpose + kfS.r
                    val S_val = s_mat(0, 0)
                    if S_val > 1e-12 then
                        val K_vec = (kfS.p * kfS.h.transpose * s_mat.inverse)(?, 0)
                        val nu_t  = y_(tt) - mu_c - (kfS.h * kfS.x)(0)
                        val H_d_x = (kfS.h * d_x)(0)
                        sumWNu   += (nu_t / S_val) * (1.0 + H_d_x)
                        sumNu2C  += nu_t * nu_t / S_val
                        val IKH   = MatrixD.eye (r_dim, r_dim)
                        for ii <- 0 until r_dim do IKH(ii, 0) -= K_vec(ii)
                        d_x = IKH * d_x - K_vec
                    end if
                    kfS.update (VectorD (y_(tt) - mu_c))
                    d_x = kf_f_mat * d_x
                    tt += 1
                end while
                val denom = if sumNu2C > 1e-12 then sumNu2C else 1e-12
                g(muIdx) = -y_.dim.toDouble * sumWNu / denom
            end if
            g
        end analyticGrad

        val lo = VectorD.fill (p + q)(-10.0) ++ VectorD (-1e6)
        val hi = VectorD.fill (p + q)( 10.0) ++ VectorD ( 1e6)

        val opt = Optimizer (negLogLikProfile, l_u_ = (lo, hi), gradF = analyticGrad)
        val (loss, u_opt_profile) = opt.solve (start)
        println (f"  warm-start OLS+mean: profile negLogLik = $loss%.4f")

        phi_est   = if p > 0 then pacfToAR (u_opt_profile(0 until p).map (tanh))     else new VectorD (0)
        theta_est = if q > 0 then pacfToAR (u_opt_profile(p until p + q).map (tanh)) else new VectorD (0)
        mu_est    = u_opt_profile(p + q)

        val bv_rec = u_opt_profile ++ VectorD (0.0)
        val (kfRec, mu_rec) = formKF (bv_rec)
        var sumNu2C_rec = 0.0
        if kfRec != null then
            var t = 0
            while t < y_.dim do
                kfRec.predict ()
                val nu = (VectorD (y_(t) - mu_rec) - kfRec.h * kfRec.x)(0)
                val ct = (kfRec.h * kfRec.p * kfRec.h.transpose + kfRec.r)(0, 0)
                if ct > 1e-12 then sumNu2C_rec += nu * nu / ct
                kfRec.update (VectorD (y_(t) - mu_rec))
                t += 1
            end while
        end if
        sig2_est = math.max (sumNu2C_rec / y_.dim, 1.0)

        val est_loss = loss + y_.dim.toDouble / 2.0 * log (2.0 * Pi)
        val u_opt    = u_opt_profile ++ VectorD (log (sig2_est))
        b            = phi_est ++ theta_est

        println (s"\nEstimated parameters for $modelName")
        println (s"  phi   = $phi_est")
        if q > 0 then println (s"  theta = $theta_est")
        println (s"  mu    = $mu_est")
        println (s"  sig2  = $sig2_est")
        println (s"  negLogLik = $est_loss   (logLik = ${-est_loss})")

        val (kf_full, _) = formKF (u_opt)
        val final_kf =
            if kf_full != null then kf_full
            else
                val (kf_unit, _) = formKF (u_opt_profile ++ VectorD (0.0))
                if kf_unit == null then
                    flaw ("train", s"formKF failed at both full and unit-variance for $modelName")
                    return
                end if
                sig2_est = 1.0
                kf_unit
        kf_run = final_kf
        var t = 0
        while t < y_.dim do
            kf_run.predict ()
            kf_run.update (VectorD (y_(t) - mu_est))
            t += 1
        end while
        kf_trained_x = kf_run.x.copy
        kf_trained_p = kf_run.p.copy
        last_t = -1
        pre_x  = null
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the model coefficient vector b = [φ₁..φ_p, θ₁..θ_q].
     *  Consistent with the `Forecaster` convention (b holds ARMA coefficients).
     *  The additional KF-specific parameters μ and σ² are printed by `train`.
     *  @return  coefficient vector b (AR then MA coefficients)
     */
    override def parameter: VectorD = b                                    // φ₁..φ_p ++ θ₁..θ_q

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a deep copy of the burned-in Kalman filter from the training run.
     *  The returned filter is independent of the live `kf_run` state, so it can
     *  be advanced for rolling-origin evaluation without side-effects on the model.
     *  @return  (deep-copy KalmanFilter with state = post-training x and P, μ)
     */
    def getTrainedFilter: (KalmanFilter, Double) =
        if kf_trained_x == null then
            flaw ("getTrainedFilter", "model has not been trained yet")
        val kfCopy = new KalmanFilter (kf_f_mat, kf_q_mat, kf_h_mat, kf_r_mat,
                                       kf_trained_x.copy, kf_trained_p.copy)
        (kfCopy, mu_est)
    end getTrainedFilter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict y_t using a one-step-ahead Kalman filter pass.  Advances the
     *  running sequential filter `kf_run` by one step:
     *    (a) KF predict step  → propagates state to x_{t|t-1}, gives ŷ_t
     *    (b) saves x_{t|t-1} in `pre_x` for subsequent multi-step forecasting
     *    (c) KF update step   → consumes y_(t) and advances to x_{t|t}
     *
     *  Auto-reset logic: `kf_run` is reset to (x₀, P₀) when t < 2 or when
     *  t ≤ last_t (backward jump detected), so in-sample sequential use
     *  (predictAll) works correctly even after a rollValidate burn-in.
     *
     *  @param t   the time point being predicted (0-indexed)
     *  @param y_  the actual series; y_(t) is consumed by the KF update
     *  @return    the 1-step-ahead forecast ŷ_t
     */
    override def predict (t: Int, y_ : VectorD): Double =
        if kf_run == null then
            flaw ("predict", "model has not been trained yet")
            return 0.0
        end if
        if t < 2 || t <= last_t then
            kf_run.x = kf_x0.copy
            kf_run.p = kf_p0.copy
        end if
        last_t = t
        kf_run.predict ()
        pre_x = kf_run.x.copy
        val y_hat = (kf_h_mat * kf_run.x)(0) + mu_est
        kf_run.update (VectorD (y_(t) - mu_est))
        yf(t, 1) = y_hat
        y_hat
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a vector of h = 1 to hh-steps-ahead forecasts at time t.
     *  Must be called immediately after `predict(t, y_)`, which fills yf(t, 1)
     *  and saves the pre-update state in `pre_x`.
     *
     *  Multi-step projection (h ≥ 2) uses the deterministic companion recursion:
     *      ŷ_{t+h-1|t} = H F^{h-1} x_{t|t-1} + μ
     *  where x_{t|t-1} = `pre_x`.
     *
     *  @param t   the time point from which to forecast
     *  @param y_  the actual series (not directly used; forecasts are model-driven)
     *  @return    vector of hh forecasts (index 0 = h=1, ..., index hh-1 = h=hh)
     */
    override def forecast (t: Int, y_ : VectorD = yb): VectorD =
        val yh = new VectorD (hh)
        yh(0) = yf(t, 1)
        if pre_x == null then return yh
        var proj = pre_x.copy
        var h = 2
        while h <= hh do
            proj     = kf_f_mat * proj
            val yhat = (kf_h_mat * proj)(0) + mu_est
            yf(t, h) = yhat
            yh(h-1)  = yhat
            h += 1
        end while
        yh
    end forecast

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast all y_.dim time points at horizon h (h-steps ahead) using a
     *  fresh Kalman filter pass starting from the stationary prior (kf_x0, kf_p0).
     *  Fills FORECAST MATRIX yf(t, h) with the h-step forecast for y_{t+h-1}
     *  made at time t having observed y_0..y_{t-1}.
     *
     *  Called by `forecastAll` for h = 2 to hh; yf(?, 1) is set by `predictAll`.
     *
     *  @param h   the forecast horizon (h ≥ 2)
     *  @param y_  the actual series used for KF update steps
     *  @return    column vector yf(?, h) of length y_.dim
     */
    override def forecastAt (h: Int, y_ : VectorD = yb): VectorD =
        if h < 2 then flaw ("forecastAt", s"horizon h = $h must be at least 2")
        if kf_run == null then flaw ("forecastAt", "model has not been trained yet")

        val tmpKf = new KalmanFilter (kf_f_mat, kf_q_mat, kf_h_mat, kf_r_mat,
                                      kf_x0.copy, kf_p0.copy)
        var tt = 0
        while tt < y_.dim do
            val snap_x = tmpKf.x.copy
            val snap_p = tmpKf.p.copy
            var step = 0
            while step < h do { tmpKf.predict (); step += 1 }
            yf(tt, h) = (kf_h_mat * tmpKf.x)(0) + mu_est
            tmpKf.x = snap_x
            tmpKf.p = snap_p
            tmpKf.predict ()
            tmpKf.update (VectorD (y_(tt) - mu_est))
            tt += 1
        end while
        yf(?, h)
    end forecastAt

end ARMA_KF

// ─────────────────────────────────────────────────────────────────────────────
// Test main functions
// ─────────────────────────────────────────────────────────────────────────────

import Example_Covid.loadData_y
import Example_LakeLevels.{y => lakeLevels}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_KFTest` main function tests the `ARMA_KF` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest
 */
@main def aRMA_KFTest (): Unit =

    val hh = 3

    for p <- 1 to 5; q <- 0 to 3 do
        ARMA_KF.hp("p") = p
        ARMA_KF.hp("q") = q
        val mod = new ARMA_KF (lakeLevels, hh)
        banner (s"In-ST: ${mod.modelName} on LakeLevels Dataset")
        mod.inSample_Test ()
    end for

end aRMA_KFTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_KFTest2` main function tests the `ARMA_KF` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest2
 */
@main def aRMA_KFTest2 (): Unit =

    val hh = 3

    val mod = new ARMA_KF (lakeLevels, hh)
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest ()()

    mod.setSkip ()                          // don't skip any rows
    mod.rollValidate ()
    mod.diagnoseAll (lakeLevels, mod.getYf, Forecaster.teRng (lakeLevels.dim))
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRMA_KFTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_KFTest3` main function tests the `ARMA_KF` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for ARMA_KF(p, q) across p = 1 to 5, q = 0 to 2.
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest3
 */
@main def aRMA_KFTest3 (): Unit =

    val yy = loadData_y ()
    val y  = yy(0 until 116)
    val hh = 6

    for p <- 1 to 5; q <- 0 to 2 do
        ARMA_KF.hp("p") = p
        ARMA_KF.hp("q") = q
        val mod = new ARMA_KF (y, hh)
        banner (s"In-ST: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()
    end for

end aRMA_KFTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_KFTest4` main function tests the `ARMA_KF` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for ARMA_KF(p, q) across p = 1 to 5, q = 0 to 2.
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest4
 */
@main def aRMA_KFTest4 (): Unit =

    val yy = loadData_y ()
    val y  = yy(0 until 116)
    val hh = 6

    for p <- 1 to 5; q <- 0 to 2 do
        ARMA_KF.hp("p") = p
        ARMA_KF.hp("q") = q
        val mod = new ARMA_KF (y, hh)
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()
    end for

end aRMA_KFTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRMA_KFTest5` main function tests the `ARMA_KF` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  Comparison of sMAPE for ARMA_KF(p, 1) (i.e., q = 1) for p = 1 to 5.
 *  > runMain scalation.modeling.forecasting.aRMA_KFTest5
 */
@main def aRMA_KFTest5 (): Unit =

    val yy = loadData_y ()
    val y  = yy(0 until 116)
    val hh = 6

    ARMA_KF.hp("q") = 1
    for p <- 1 to 5 do
        ARMA_KF.hp("p") = p
        val mod = new ARMA_KF (y, hh)
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest ()()

        mod.forecastAll ()
        mod.diagnoseAll (y, mod.getYf)
        println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")
    end for

end aRMA_KFTest5

