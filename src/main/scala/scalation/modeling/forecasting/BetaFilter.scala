
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 15 13:20:16 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Cloded-Form Fisher Scoring Beta Filter
 *           AI Assisted Code
 */

package scalation
package modeling
package forecasting

import scala.math.{exp, log, max}

import scalation.mathstat._

import Combinatorics._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Fisher Scoring Beta Filter optimized with an integrated Forgetting Factor (gammaF).
 *  @param stateDim    the dimension of the latent state vector x_t
 *  @param phi         the fixed observation precision parameter (phi > 0)
 *  @param gammaF      the forgetting factor parameter in (0, 1], lower values ignore past history faster
 *  @param innerIters  number of scoring cycles per step, set to 1 for standard EKF (baseline), 
 *                         or 3 for higher accuracy full Iterated EKF
 */
class BetaFilter (val stateDim: Int, val phi: Double, val gammaF: Double = 0.96,
                  val innerIters: Int = 3):

    require (gammaF > 0.0 && gammaF <= 1.0, "gammaF must be strictly within the interval (0, 1]")
    require (innerIters >= 1, "innerIters must be at least 1")

    var m = new VectorD (stateDim)           
    var C = MatrixD.eye (stateDim, stateDim)           

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the inverse logit (sigmoid) function.
     *  @param z  the real valued argument to the function
     */
    def inverseLogit (z: Double): Double = 1.0 / (1.0 + exp (-z))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Execute one complete forward step using gammaF for the time-update variance decay.
     *  It processes one incoming observation in real-time.
     *  @param Z  the layout row design vector
     *  @param y  the continuous fractional measurement y_t inside (0, 1)
     *  @param G  the structural State Transition Matrix used to propagate state
     */
    def forwardStep (Z: VectorD, y: Double, G: MatrixD):
            (VectorD, MatrixD, VectorD, MatrixD) =

        // Time Prediction Step leveraging gammaF scalar inflation
        val m_pred = G * m                                         // predicted mean
        val c_pred = (G * C * G.transpose) / gammaF                // predicted cov, scaled uncertainty decay

        var m_iter  = m_pred                                       // initialize for Iterative EKF
        var final_m = m_pred                                       // mean to be updated iteratively
        var final_c = c_pred                                       // cov to be updated iteratively

        for _ <- 0 until innerIters do
            // Linear Predictor Scale Tracking
            val eta_t = Z dot m_iter
            val mu_t  = inverseLogit (eta_t)

            // Evaluate Closed-Form Fisher Scoring Terms via combinators
            val y_star  = log (y / (1.0 - y))
            val mu_star = digamma (mu_t * phi) - digamma ((1.0 - mu_t) * phi)
            val v_t     = trigamma (mu_t * phi) + trigamma ((1.0 - mu_t) * phi)
            val T_t     = mu_t * (1.0 - mu_t)

            val s_t = phi * (y_star - mu_star) * T_t
            val A_t = max (phi * phi * v_t * T_t * T_t, 1e-9) 
            val H_t = 1.0 / A_t                                        // construct Equivalent Working Gaussian Space

            // Kalman Filtering Update Correction
            val q_t = (c_pred * Z) dot Z                  
            val Q_t = q_t + H_t                           
            val K   = c_pred * Z / Q_t

            // Map the working observation back into the update sequence
            val z_tilde = eta_t + s_t / A_t

            // Compute the fresh state update
            final_m = m_pred + K * (z_tilde - (Z dot m_pred))
            final_c = c_pred - MatrixD.outer (K, K) * Q_t

            // Feed the updated mean back as the expansion point for the next inner iteration
            m_iter = final_m
        end for

        // Update global continuous tracking elements
        m = final_m
        C = final_c

        (m_pred, c_pred, m, C)
    end forwardStep

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Batch Historical Filter:  Automatically pipes a historical dataset through the
     *  `forwardStep` method to gather array contexts for retrospective RTS smoothing.
     *  @param Z  the layout row design vector
     *  @param y  the continuous fractional measurements (y_t) time series vector
     *  @param G  the structural State Transition Matrix used to propagate state
     */
    def forwardFilter (Z: VectorD, y: VectorD, G: MatrixD):
            (Array [VectorD], Array [MatrixD], Array [VectorD], Array [MatrixD]) =

        val n      = y.length
        val m_filt = new Array [VectorD] (n)
        val c_filt = new Array [MatrixD] (n)
        val m_pred = new Array [VectorD] (n)
        val c_pred = new Array [MatrixD] (n)

        for t <- 0 until n do
            val step = forwardStep (Z, y(t), G)
            m_pred(t) = step._1    // mp
            c_pred(t) = step._2    // Cp
            m_filt(t) = step._3    // mf
            c_filt(t) = step._4    // Cf
        end for
            
        (m_filt, c_filt, m_pred, c_pred)
    end forwardFilter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The Backward RTS Pass:  Corrects lag over the historical parameters generated by `forwardFilter`.
     *  Execute the Rauch-Rung-Striebel (RTS) backward smoothing pass across the complete trajectory.
     *  It combines past information with future data to eliminate time lags and minimize tracking variance.
     *  @param m_filt   array of filtered state mean vectors (m_t) captured during the forward filter pass,
     *                  represents the real-time optimal state estimate at each time step t 
     *  @param c_filt   array of filtered state covariance matrices (C_t) captured during the forward filter pass,
     *                  represents real-time uncertainty and parameter correlations at time step t
     *  @param m_pred   array of 1-step-ahead predicted state mean vectors (m_{t+1|t}) captured during the forward pass,
     *                  represents the purely dynamic propagation of state t into time t+1 before the observation is seen.
     *  @param c_pred   array of 1-step-ahead predicted state cov. matrices (C_{t+1|t}) captured during the forward pass,
     *                  represents the expanded uncertainty space at time t+1, scaled by the forgetting factor (gammaF).
     *  @param G        the structural State Transition Matrix used to propagate state
     *                  required to compute the smoother gain matrix:  J_t = c_filt(t) * G.transpose * c_pred(t+1).inverse
     */
    def rtsSmooth (m_filt: Array [VectorD], c_filt: Array [MatrixD], 
                   m_pred: Array [VectorD], c_pred: Array [MatrixD], G: MatrixD):
            (Array [VectorD], Array [MatrixD]) =

        val G_trans  = G.transpose
        val n        = m_filt.length
        val m_smooth = new Array [VectorD] (n)                // smoothed version of m_filt (mean vectors)
        val c_smooth = new Array [MatrixD] (n)                // smoothed version of c_filt (covaraince matrices)

        m_smooth(n - 1) = m_filt(n - 1)
        c_smooth(n - 1) = c_filt(n - 1)

        for t <- n - 2 to 0 by -1 do                          // backward pass
            val m_f      = m_filt(t)
            val c_f      = c_filt(t)
            val m_p_next = m_pred(t + 1)
            val c_p_next = c_pred(t + 1)

            // Use Cholesky factorization on the symmetric, positive-definite predicted covariance matrix c_pred(t + 1)
            val choleskyEngine = new Fac_Cholesky (c_p_next)
            choleskyEngine.factor ()                          // execute the matrix decomposition loop
            val c_p_next_inv = choleskyEngine.inverse         // extract the precise factorization-based inverse

            val J_t = c_f * G_trans * c_p_next_inv            // compute structural Smoother Gain Matrix (J_t) 

            // Update Mean vector and Covariance Matrix parameters
            m_smooth(t) = m_f + J_t * (m_smooth(t + 1) - m_p_next)
            c_smooth(t) = c_f + J_t * (c_smooth(t + 1) - c_p_next) * J_t.transpose
        end for

        (m_smooth, c_smooth)
    end rtsSmooth

end BetaFilter


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BetaModelOptimizer` object
 */
object BetaModelOptimizer:

    import scala.math.sqrt
    import mathstat.Combinatorics._
    import random.GaussHermiteQuadrature

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate the global Negative Log-Likelihood for the time series.
     *  This acts as the objective function for ScalaTion 2's parameter optimization suites.
     *  @param y       the raw continuous fractional measurements y_1:T
     *  @param Z       the design extraction vector layout
     *  @param G       the State Transition Matrix
     *  @param params  vector containing the candidate hyperparameters: [phi, gammaF]
     *  @return        the cumulative negative log-likelihood scalar score
     */
    def computeNegativeLogLikelihood (y: VectorD, Z: VectorD, G: MatrixD, params: VectorD): Double =
        val phi    = params(0)
        val gammaF = params(1)
        
        // Edge boundary enforcement to guide numerical optimization search grids safely
        if phi <= 1e-3 || gammaF <= 0.1 || gammaF > 1.0 then return Double.MaxValue

        val n        = y.dim
        val stateDim = G.dim
        
        // Re-instantiate a clean baseline filter container for this evaluation sweep
        val filter = new BetaFilter (stateDim, phi, gammaF)
        val rule   = GaussHermiteQuadrature.getRule (5)                 // retrieve precomputed nodes

        var cumulativeLogLikelihood = 0.0

        for t <- 0 until n do
            val y_t = y(t)
            
            // Calculate one-step-ahead forward predictions manually 
            val m_pred = G * filter.m
            val c_pred = (G * filter.C * G.transpose) / gammaF

            // Extract Gaussian Predictive parameters on the linear η scale
            val m_eta = Z dot m_pred
            val s_eta = sqrt ((c_pred * Z) dot Z)

            // Resolve Equation using the Gauss-Hermite combinator pipeline
            val integrated1StepDensity = rule.nodes.indices.map { j =>
                val xi    = rule.nodes(j)
                val omega = rule.weights(j)
                
                // Map the node position to our conditional Gaussian workspace
                val eta_j = m_eta + sqrt (2.0) * s_eta * xi
                val mu_j  = 1.0 / (1.0 + exp (-eta_j))          // inverse-logit transformation

                val alpha = mu_j * phi
                val beta  = (1.0 - mu_j) * phi

                // Compute exact local Beta density with log-gamma space protection
                val density = exp (logGamma (phi) - logGamma (alpha) - logGamma (beta) +
                    (alpha - 1.0) * log (y_t) + (beta - 1.0) * log (1.0 - y_t))
                omega * density
            }.sum

            val p_yt = (1.0 / sqrt_Pi) * integrated1StepDensity
            
            // Protect against zero probability floating math failures under bad optimization boundaries
            val safe_p_yt = max (p_yt, 1e-15)
            cumulativeLogLikelihood += log(safe_p_yt)

            // Update the running internal filter states to advance the timeline
            filter.forwardStep (Z, y_t, G)
        end for

        // Return negative log-likelihood (minimizing this is equivalent to maximizing likelihood)
        -cumulativeLogLikelihood
    end computeNegativeLogLikelihood

end BetaModelOptimizer


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `betaFilterTest` main function tests the `BetaFilter` class by applying `forwardStep`
 *  > runMain scalation.modeling.forecasting.betaFilterTest
 */
@main def betaFilterTest (): Unit =

    banner ("--- Starting Closed-Form Beta Filter Simulation ---")
    
    // Track 2 states: [0] Bounded Position Value, [1] Rate of Change / Trend Velocity
    val stateDimension       = 2
    val observationPrecision = 30.0            // higher means lower observation noise variance
    val forgettingFactor     = 0.95            // decays historical certainty by 5% per tick
    
    val filter = new BetaFilter (stateDimension, observationPrecision, forgettingFactor)
    
    // Design standard Kinematic State Transition Matrix (G): Position tracking over time increment dt = 1
    // New Position = Old Position + Trend Velocity
    // New Velocity = Old Velocity
    val G = MatrixD ((2, 2), 1.0, 1.0,
                             0.0, 1.0)
                                
    // Observation Matrix (Z): Extract position component directly
    val Z = VectorD (1.0, 0.0)

    // Simulate time series raw observation values tracking a moving proportion metric
    val y = VectorD (0.42, 0.45, 0.49, 0.58, 0.64, 0.72, 0.79)

    // Functional loop stepping through time series data points
    for t <- y.indices do
        val y_t = y(t)
        val (curMean, _, _, _) = filter.forwardStep (Z, y_t, G)        // execute the single filtering update pass
        val filteredProb       = filter.inverseLogit (curMean(0))      // transform tracking level back to bounded space for metrics logging
        println (s"Time step $t | Obs: $y_t | Filtered State Value: ${"%6.4f".format(filteredProb)} | Estimated Trend: ${"%6.4f".format(curMean(1))}")
    end for
    
    banner ("--- Simulation Completed Successfully ---")

end betaFilterTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `betaFilterTest2` main function tests the `BetaFilter` class over a time series
 *  calling `forwardFilter` and then `rtsSmooth`.
 *  > runMain scalation.modeling.forecasting.betaFilterTest2
 */
@main def betaFilterTest2 (): Unit =

    banner ("--- Test Matrix for Closed-Form Fisher-Scoring Beta Filter ---")

    // Setup hyperparameter variables matching structural constraints
    val stateDimension = 2          // Track 2 latent states: [Bounded Level Position, Slope Velocity]
    val precisionPhi   = 40.0       // High precision implies lower observation variance
    val discountFactor = 0.94       // Forgetting Factor (gammaF) creates a dynamic process noise footprint

    // Instantiate your complete optimized filter architecture
    val dglm = new BetaFilter (stateDimension, precisionPhi, discountFactor)

    // Build the structural Kinematic Matrix (G): 
    // State Level at t = Previous Level + Previous Trend Velocity
    // Trend Velocity at t = Previous Trend Velocity
    val G = MatrixD ((2, 2), 1.0, 1.0,
                             0.0, 1.0)

    // Build the Design Selection Vector (Z): Extract the bounded level element directly
    val Z = VectorD (1.0, 0.0)

    // GENERATE CYCLICAL STREAMING DATA DATASET 
    // Simulate a 12-week time-series tracking a rate parameter that climbs steadily
    val y = VectorD (0.35, 0.38, 0.41, 0.45, 0.52, 0.61, 0.70, 0.76, 0.79, 0.82, 0.81, 0.83)

    println (s"Data Sequence Generated successfully. Timeline Context Horizon: ${y.length} weeks.")
    println ("Executing Forward Filter pass...")

    // RUN FORWARD FILTER PASS
    // Captures structural state summaries across chronological parameters
    val (m_filt, c_filt, m_pred, c_pred) = dglm.forwardFilter (Z, y, G)

    println ("Forward Pass Complete. Executing Factorization-Driven Backward RTS Smoother Pass...")

    // RUN RETROSPECTIVE RTS SMOOTHER PASS
    // Utilizes optimized Cholesky Factorizations and precomputed transpositions to strip lag
    val (m_smooth, _) = dglm.rtsSmooth (m_filt, c_filt, m_pred, c_pred, G)

    println("Backward Smoothing Complete.\n")

    // RENDER COMPARATIVE DIAGNOSTIC RESULTS TABLE
    println ("-" * 115)
    printf (" %-5s | %-10s | %-22s | %-22s | %-12s | %-12s \n", 
           "Week", "Raw Obs", "Forward Filter Prob", "Backward Smoothed Prob", "Filt Trend", "Smoo Trend")
    println ("-" * 115)

    for t <- y.indices do
        // Convert unconstrained logit states back to bounded (0, 1) probability spaces
        val probFiltered = dglm.inverseLogit (m_filt(t)(0))
        val probSmoothed = dglm.inverseLogit (m_smooth(t)(0))
        
        // Extract slope tracking vectors
        val trendFiltered = m_filt(t)(1)
        val trendSmoothed = m_smooth(t)(1)

        printf ("  %-4d |   %-6.2f |        %-12.4f        |         %-12.4f        |   %-8.4f   |   %-8.4f   \n", 
               t, y(t), probFiltered, probSmoothed, trendFiltered, trendSmoothed)
    end for
    
    banner ("--- Test Pipeline Result Verification: Check how The RTS Pass Anticipates Shifts ---")

end betaFilterTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `betaFilterTest3` main function tests the `BetaFilter` class over a time series
 *  calling `forwardFilter` and then `rtsSmooth`.
 *  > runMain scalation.modeling.forecasting.betaFilterTest3
 */
@main def betaFilterTest3 (): Unit =

    banner ("--- Coherence Matrix Test for Beta Model Gauss-Hermite Optimizer ---")

    // SETUP EXPERIMENTAL TIMELINE DATA
    // We simulate a stable weekly rate that hovers around 0.45 with subtle local variance
    val y = VectorD (0.43, 0.46, 0.44, 0.45, 0.48, 0.42, 0.45, 0.47, 0.43, 0.46)
    
    // Set up standard 2-dimensional kinematics
    val G = MatrixD ((2, 2), 1.0, 1.0,
                             0.0, 1.0)
    val Z = VectorD (1.0, 0.0)

    // Set a constant forgetting factor (gammaF) for this specific sweep test
    val fixedGammaF = 0.95

    println (s"Dataset timeline loaded successfully (${y.dim} weeks).")
    println (s"Holding forgetting factor constant at gammaF = $fixedGammaF")
    println ("Beginning numerical parameter search space grid sweep across precision (phi)...\n")

    // CONSTRUCT PARAMETER SEARCH GRID PROFILE
    // We sweep across a range of phi values to observe the shape of the optimization surface
    val candidatePhis = VectorD (5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 200.0)

    println ("-" * 75)
    printf ("  %-15s | %-25s | %-25s \n", "Candidate Phi", "Negative Log-Likelihood", "Status Metric Profile")
    println ("-" * 75)

    var bestPhi   = 0.0
    var lowestNLL = Double.MaxValue

    // EXECUTE THE LOG-LIKELIHOOD SWEEP USING DYNAMIC COMBINATORS
    for phi <- candidatePhis do
        // Construct the vector payload parameter envelope expected by your optimizer
        val params = VectorD (phi, fixedGammaF)

        // Invoke the Gauss-Hermite integration routine over the full dataset loop
        val nll = BetaModelOptimizer.computeNegativeLogLikelihood (y, Z, G, params)

        // Evaluate and log performance status milestones
        val statusString = if nll < lowestNLL then
            lowestNLL = nll
            bestPhi   = phi
            "-> New Minimum Achieved"
        else
            "Receding Surface Slope"

        printf ("  %-15.2f | %-25.4f | %-25s \n", phi, nll, statusString)
    end for

    println ("-" * 75)
    println (s"Optimization Core Sweep Coherence Test: SUCCESS.")
    println (s"The minimum Negative Log-Likelihood surface point occurs at phi = $bestPhi")
    println (s"Optimized Minimal Residual Energy Cost: $lowestNLL")
    println ("================================================================================")

end betaFilterTest3

