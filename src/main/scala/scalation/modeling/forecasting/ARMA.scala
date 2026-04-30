
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Lokesh Adusumilli, Nirupom Bose Roy
 *  @version 2.0
 *  @date    Sun Jun 30 13:27:00 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive, Moving Average (ARMA) via Kalman Filter MLE
 */

package scalation
package modeling
package forecasting

import scala.math.{max, log, Pi}
import scala.util.boundary

import scalation.mathstat._
import scalation.mathstat.MatrixD.outer
import scalation.optimization.quasi_newton.{LBFGS_B => Optimizer}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA` companion object provides factory methods and default
 *  hyperparameters for ARMA models.
 */
object ARMA:

    /** Hyper-parameters:
     *  - `p`: AR order
     *  - `q`: MA order
     */
    val hp = new HyperParameter
    hp += ("p", 1, 1)
    hp += ("q", 1, 1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARMA` model.
     *  @param y       the univariate response/time-series vector
     *  @param hh      the maximum forecast horizon
     *  @param tRng    the optional time range
     *  @param hparam  the hyper-parameters
     */
    def apply (y: VectorD, hh: Int, tRng: Range = null, hparam: HyperParameter = hp): ARMA =
        new ARMA (y, hh, tRng, hparam)

end ARMA


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARMA` class fits an Auto-Regressive Moving Average model using a
 *  state-space representation and Kalman-filter maximum likelihood.
 *
 *  Model:
 *      y_t = c + Σ_j φ_j y_{t-j} + Σ_k θ_k e_{t-k} + e_t
 *
 *  The implementation estimates:
 *  - AR coefficients `φ`
 *  - MA coefficients `θ`
 *  - intercept `c`
 *  - process variance `σ²`
 *  Notes:
 *  - The likelihood is computed from one-step-ahead Kalman innovations.
 *  - A small burn-in (`max(p, q+1)`) is excluded from the log-likelihood
 *    sum to align with the external reference implementation.
 *  - Rolling forecast evaluation is performed externally in the test driver.
 *  @param y        the response/time-series vector
 *  @param hh       the maximum forecast horizon
 *  @param tRng     the optional time range
 *  @param hparam   the hyper-parameters
 *  @param bakcast  whether a backcast value is prepended
 */
class ARMA (y: VectorD, hh: Int, tRng: Range = null,
            hparam: HyperParameter = ARMA.hp, bakcast: Boolean = false)
      extends Forecaster (y, hh, tRng, hparam, bakcast):

    private val flaw = flawf ("ARMA")                                  // flaw function
    private val p    = hparam("p").toInt                               // AR order
    private val q    = hparam("q").toInt                               // MA order

    _modelName = s"ARMA($p, $q)"

    // State dimension for the companion-form state-space model
    private val r_dim = max (p, q + 1)

    // Trained filter used as the rolling forecast state anchor
    private var kf_tracker: KalmanFilter = null

    // Stored mean-like quantity implied by the fitted parameterization
    private var mu_est = 0.0

    def getBest: BestStep = ???   // FIX -- implement or throw exception

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the ARMA model on the supplied response vector using Kalman-filter
     *  maximum likelihood and a bound-constrained L-BFGS optimizer.
     *
     *  Parameter vector layout:  [φ_1 ... φ_p, θ_1 ... θ_q, c, σ²]
     *
     *  @param x_null  ignored for univariate models
     *  @param y_      the training response vector
     */
    override def train (x_null: MatrixD, y_ : VectorD): Unit =
        banner (s"Train $modelName using Kalman Filter MLE")

        // ------------------------------------------------------------------
        // 1. OLS-based initialization
        // ------------------------------------------------------------------

        val mu_guess    = y_.mean
        val z_centered  = y_ - mu_guess
        val n_ols       = z_centered.dim - p

        val X_ols = new MatrixD (n_ols, p)
        val y_ols = new VectorD (n_ols)

        for t <- 0 until n_ols do
            y_ols(t) = z_centered(t + p)
            for j <- 0 until p do X_ols(t, j) = z_centered(t + p - 1 - j)
        end for

        val ols = new Regression (X_ols, y_ols)
        ols.train ()

        val phi_guesses = ols.parameter
        val e_ols       = y_ols - ols.predict (X_ols)
        val sigma2_guess = (e_ols dot e_ols) / n_ols

        // ------------------------------------------------------------------
        // 2. Negative log-likelihood for Kalman-filter
        // ------------------------------------------------------------------

        def negativeLogLikelihood (b_vec: VectorD): Double = boundary:
            val (kf, mu_curr) = formKalmanFilter (b_vec)
            if kf == null then boundary.break (Double.PositiveInfinity)

            var logLik = 0.0
            val burn   = max (p, q + 1)

            for t <- 0 until y_.dim do
                kf.predict ()
                val z     = VectorD (y_(t) - mu_curr)
                val y_err = z - kf.h * kf.x
                val s     = kf.h * kf.p * kf.h.transpose + kf.r
                if s(0, 0) <= 1e-12 then boundary.break (Double.PositiveInfinity)

                val err_sq  = y_err(0) * (1.0 / s(0, 0)) * y_err(0)
                val contrib = -0.5 * (log (2.0 * Pi) + log (s(0, 0)) + err_sq)
                if t >= burn then logLik += contrib
                kf.update (z)
            end for

            -logLik
        end negativeLogLikelihood

        // ------------------------------------------------------------------
        // 3. Initial parameter vector
        // ------------------------------------------------------------------

        val num_params = p + q + 2
        val b0         = new VectorD (num_params)
        val offset     = if phi_guesses.dim > p then 1 else 0

        for i <- 0 until p do b0(i) = phi_guesses(i + offset)
        for i <- 0 until q do b0(p + i) = 0.0

        val phi_init_sum = (0 until p).map (i => b0(i)).sum
        val c_guess      = mu_guess * (1.0 - phi_init_sum)

        b0(p + q)     = c_guess
        b0(p + q + 1) = sigma2_guess

        // ------------------------------------------------------------------
        // 4. Bound-constrained optimization
        // ------------------------------------------------------------------

        val lowerBounds = VectorD.fill (p + q)(-2.0) ++ VectorD (-100000.0, 1e-6)
        val upperBounds = VectorD.fill (p + q)( 2.0) ++ VectorD ( 100000.0, Double.PositiveInfinity)

        val optimizer = new Optimizer (f = negativeLogLikelihood, l_u = (lowerBounds, upperBounds))
        val (est_loss, est_params) = optimizer.solve (b0)

        // ------------------------------------------------------------------
        // 5. Persist fitted parameters and construct the tracker filter
        // ------------------------------------------------------------------

        b = est_params

        val (final_kf, final_mu) = formKalmanFilter (b)
        kf_tracker = final_kf
        mu_est     = final_mu

        for t <- 0 until y_.dim do
            kf_tracker.predict ()
            kf_tracker.update (VectorD (y_(t) - mu_est))
        end for

        val phisEst = est_params(0 until p)
        val cEst    = est_params(p + q)
        val phiSum  = phisEst.sum
        val muImp   = if math.abs (1.0 - phiSum) > 1e-8 then cEst / (1.0 - phiSum) else Double.NaN

        println (s"\nEstimated params for ARMA($p,$q)")
        println (s"phis        = $phisEst")
        if q > 0 then println (s"thetas      = ${est_params(p until p + q)}")
        println (s"intercept c = $cEst")
        println (s"implied mu  = $muImp")
        println (s"sigma2      = ${est_params(p + q + 1)}")
        println (s"negLogLik   = $est_loss")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Construct a Kalman filter from the parameter vector.
     *
     *  Parameter vector layout:  [φ_1 ... φ_p, θ_1 ... θ_q, c, σ²]
     *
     *  @param b_vec  the parameter vector
     *  @return       `(KalmanFilter, mu)` or `(null, 0.0)` if invalid
     */
    private def formKalmanFilter (b_vec: VectorD): (KalmanFilter, Double) =
        val phis   = b_vec(0 until p)
        val thetas = b_vec(p until p + q)
        val c      = b_vec(p + q)

        val phi_sum = phis.sum
        val denom   = 1.0 - phi_sum
        if math.abs (denom) <= 1e-8 then return (null, 0.0)

        val mu = c / denom

        val sig2_proc = b_vec(p + q + 1)
        val sig2_obs  = 1e-6

        if sig2_proc <= 0.0 then return (null, 0.0)

        // Companion-form state transition matrix.
        val f = new MatrixD (r_dim, r_dim)
        for j <- 0 until p do f(0, j) = phis(j)
        for i <- 1 until r_dim do f(i, i - 1) = 1.0

        // Process noise covariance Q = G G' σ².
        val g = new VectorD (r_dim)
        g(0)  = 1.0
        for i <- 0 until q if i + 1 < r_dim do g(i + 1) = thetas(i)
        val q_mat = outer (g, g) * sig2_proc

        // Observation model: observe the first state element.
        val h_mat = new MatrixD (1, r_dim)
        h_mat(0, 0) = 1.0

        val r_mat = MatrixD ((1, 1), sig2_obs)

        // Diffuse-style large-variance initialization.
        val x0 = new VectorD (r_dim)
        val p0 = MatrixD.eye (r_dim, r_dim) * 1e6

        (new KalmanFilter (f, q_mat, h_mat, r_mat, x0, p0), mu)
    end formKalmanFilter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a copy of the trained filter and the associated mean quantity.
     *  Useful for rolling-origin forecast evaluation without mutating the
     *  stored tracker state.
     */
    def getTrainedFilter: (KalmanFilter, Double) =
        if kf_tracker == null then
            flaw ("getTrainedFilter", "model has not been trained")
            (null, mu_est)
        else
            (kf_tracker.copyFilter (), mu_est)
    end getTrainedFilter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast all time points for a given horizon `h` using the stored
     *  tracker state.
     *  @param h   the forecast horizon
     *  @param y_  the observed series
     */
    override def forecastAt (h: Int, y_ : VectorD = yb): VectorD =
        if kf_tracker == null then flaw ("forecastAt", "model has not been trained")
        val (temp_kf, mu) = getTrainedFilter

        for t <- 0 until y_.dim do
            val kf_forc = new KalmanFilter (temp_kf.f, temp_kf.q, temp_kf.h, temp_kf.r,
                                            temp_kf.x.copy, temp_kf.p.copy)
            for step <- 0 until h do
                kf_forc.predict ()
                if step == h - 1 then
                    val pred = (kf_forc.h * kf_forc.x)(0) + mu
                    yf(t, h) = pred
            end for

            temp_kf.predict ()
            temp_kf.update (VectorD (y_(t) - mu))
        end for

        yf(?, h)
    end forecastAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** One-step prediction placeholder required by `Forecaster`.
     *  Horizon-specific forecasting is handled by `forecastAt`.
     *  @param t   the given time
     *  @param y_  the actual time series
     */
    override def predict (t: Int, y_ : VectorD): Double = 0.0

end ARMA

import Example_Covid.loadData_y

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Rolling-origin validation driver for the standalone Kalman-filter ARMA
 *  implementation.
 *  The logic mirrors the external Python validation flow:
 *  - fit once on the training window
 *  - forecast horizons `1..hh` before each newly revealed test observation
 *  - update the filter state sequentially without refitting
 *  > runMain scalation.modeling.forecasting.aRMA_KalmanRollingValidation
 */
@main def aRMA_KalmanRollingValidation (): Unit =

    // --------------------------------------------------------------------------
    // Configuration
    // --------------------------------------------------------------------------
    val p_max      = 5          // maximum AR order to evaluate
    val q_max      = 0          // maximum MA order to evaluate
    val hh         = 6          // maximum forecast horizon
    val train_size = 92         // length of training window

    // --------------------------------------------------------------------------
    // Load and split data
    // --------------------------------------------------------------------------
    val yy      = loadData_y ()
    val y       = yy(0 until 116)
    val y_train = y(0 until train_size)

    println (s"Data Split: Total=${y.dim}, Train=$train_size, Test=${y.dim - train_size}")

    // --------------------------------------------------------------------------
    // Evaluate all requested (p, q) configurations
    // --------------------------------------------------------------------------
    for p <- 5 to p_max; q <- 0 to q_max do

        // ----------------------------------------------------------------------
        // 1. Fit model on training data only
        // ----------------------------------------------------------------------
        ARMA.hp("p") = p
        ARMA.hp("q") = q

        val model = new ARMA (y, hh)
        model.train (null, y_train)

        // ----------------------------------------------------------------------
        // 2. Rolling-origin forecast generation
        //
        // At each origin t:
        //   - forecast horizons 1..hh before revealing y(t)
        //   - store each horizon-h forecast at the row corresponding to its
        //     target time
        //   - update the rolling filter with the newly observed value y(t)
        // ----------------------------------------------------------------------
        val (kfRolling, muFinal) = model.getTrainedFilter
        val yfMatrix             = new MatrixD (y.dim, hh)

        for t <- train_size until y.dim do
            // Clone the current rolling state so forecasting does not mutate it.
            val kfForecast = new KalmanFilter (kfRolling.f, kfRolling.q,
                                               kfRolling.h, kfRolling.r,
                                               kfRolling.x.copy, kfRolling.p.copy)

            // Generate forecasts for horizons 1..hh from the current origin.
            for h <- 0 until hh do
                kfForecast.predict ()
                val yHat = (kfForecast.h * kfForecast.x)(0) + muFinal

                val targetTime = t + h
                if targetTime < y.dim then yfMatrix(targetTime, h) = yHat
            end for

            // Reveal the next actual observation and update the rolling state.
            kfRolling.predict ()
            kfRolling.update (VectorD (y(t) - muFinal))
        end for

        // ----------------------------------------------------------------------
        // 3. Horizon-wise evaluation
        //
        // Alignment:
        //   Column h stores forecasts for horizon (h + 1), and the earliest valid
        //   row for that column is train_size + h.
        // ----------------------------------------------------------------------
        class RollingDiagnoser (dfm: Double, df: Double) extends Diagnoser (dfm, df):
            val modName = s"Rolling-ARMA($p, $q)"
        end RollingDiagnoser

        for h <- 0 until hh do
            val hStep     = h + 1
            val evalStart = train_size + h
            val evalEnd   = y.dim

            val yActual = y(evalStart until evalEnd)
            val yPred   = yfMatrix(evalStart until evalEnd, h)

            println (s"\n--- Test Set Metrics (Horizon h=$hStep) ---")

            val dfm = (p + q + 1).toDouble
            val df  = yActual.dim - dfm

            val diagnoser = new RollingDiagnoser (dfm, df)
            diagnoser.setSkip (0)

            val stats = diagnoser.diagnose (yActual, yPred)

            for i <- stats.indices do
                if qoF_names(i) != "NA" then println (f"${qoF_names(i)}%10s = ${stats(i)}%12.6f")

            val tAxis    = VectorD.range (0, y.dim)
            val plotPred = yfMatrix(?, h).copy
            new Plot (tAxis, y, plotPred, s"Rolling Forecast ARMA($p,$q) h=$hStep", lines = true)
        end for

    end for

end aRMA_KalmanRollingValidation

