
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jul  5 00:01:42 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Ordinary Differential Equation (ODE) Parameter Estimator
 *           AI Assisted Code
 */

package scalation
package dynamics

import scala.annotation.nowarn

import scalation.mathstat._
import scalation.optimization.quasi_newton.LBFGS_B

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `OdeEstimator` trait provides a framework for ODE parameter estimation problems
 *  involving a system of Ordinary Differential Equations (ODEs).
 */
trait OdeEstimator:

    def numParams: Int
    def timeGrid: VectorD
    def observedData: VectorD
    def initialStates: VectorD
    def obsColumnIdx: Int
    def bounds: (VectorD, VectorD)
  
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Array of functions matching the Array [DerivativeV] specification needed by DormandPrince.
     *  @param b  the parameter vector
     */
    def systemODEs (b: VectorD): Array [DerivativeV]

end OdeEstimator


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BoundedOdeSolver` class coordinates the optimization loop by taking
 *  any specific problem formulation and minimizing its Mean Squared Error (MSE).
 *  @param problem  the target ODE application to solve
 */
class BoundedOdeSolver (val problem: OdeEstimator):

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The loss function based on MSE. 
     *  @param b  the parameter vector to be optimized
     */
    private def mseLoss (b: VectorD): Double =
        val odes      = problem.systemODEs (b)
        var curState  = problem.initialStates.copy
        var totalLoss = 0.0
        
        for i <- problem.timeGrid.indices do
            val targetTime = problem.timeGrid(i)
            
            if i > 0 then
                val prevTime = problem.timeGrid(i - 1)
                curState = DormandPrince.integrateVV (odes, curState, targetTime, prevTime)   // step forward

            val simulated = curState(problem.obsColumnIdx)
            val actual    = problem.observedData(i)
            totalLoss    += (actual - simulated) ~^ 2
//          totalLoss    += ((actual - simulated) / (actual + 1e-6)) ~^ 2
        end for

        10000.0 * totalLoss / problem.timeGrid.dim   // scale up to make gradients bigger
    end mseLoss

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Estimate the parameters.
     *   @parsm b_init  the initial guess for the parameters
     */
    def estimateParameters (b_init: VectorD): (Double, VectorD) =
        val optimizer = new LBFGS_B (f = mseLoss, l_u = problem.bounds)
        optimizer.solve (b_init)
    end estimateParameters

end BoundedOdeSolver


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `odeEstimatorTest` main method test the `OdeEstimator` trait.
 *  Uses a simple Growth Rate model y(t) = x_0 e^{bt} with analytic solution 0.4023.
 *  > runMain scalation.dynamics.odeEstimatorTest
 */
@main def odeEstimatorTest (): Unit =

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concrete application implementation of `OdeEstimator` targeting a basic 
     *  growth model to feed validation scenarios into our test framework.
     */
    class SampleGrowthModel extends OdeEstimator:
        val numParams     = 1
        val timeGrid      = VectorD (0.0, 1.0, 2.0, 3.0, 4.0)
    
        // Synthetic empirical ground truth data with an authentic target rate of ~0.5
        val observedData  = VectorD (1.0, 1.5, 2.2, 3.3, 5.0) 
        val initialStates = VectorD (1.0)                      // initial population state at t = 0
        val obsColumnIdx  = 0
        
        val bounds        = (VectorD(0.01), VectorD(2.0))      // constrain search boundaries
    
        // dx/dt = b(0) * x
        @nowarn def systemODEs (b: VectorD): Array [DerivativeV] =
            Array ((t: Double, u: VectorD) => b(0) * u(0))

    end SampleGrowthModel

    println ("=== Initializing BoundedOdeSolver Testing Environment ===")
    
    // 1. Instantiate our test problem framework configuration
    val problemInstance = new SampleGrowthModel ()
    
    // 2. Feed the problem context architecture straight into our solver engine
    val solver = new BoundedOdeSolver (problemInstance)
    
    // 3. Set a non-optimal initialization point guess to evaluate fit path performance
    val initialGuess = VectorD (0.1)
    println (s"-> Configured Initial Parameter Guess: $initialGuess")
    
    // 4. Run parameter minimization routine
    val (optimizedMse, optimalParameters) = solver.estimateParameters (initialGuess)
    
    // 5. Output metrics and evaluation validation data
    println ("\n=== Optimization Output Results ===")
    println (s"-> Identified Optimal Parameter Vector: $optimalParameters")
    println (s"-> Achieved Minimum Mean Squared Error: $optimizedMse")
    
    // 6. Print Verification Data Path Comparisons
    println("\nProfile Alignment Checklist:")
    var curState = problemInstance.initialStates.copy
    val odes     = problemInstance.systemODEs(optimalParameters)
    
    for i <- problemInstance.timeGrid.indices do
        val t = problemInstance.timeGrid(i)
        if i > 0 then
            curState = DormandPrince.integrateVV(odes, curState, t, problemInstance.timeGrid(i - 1))
            
        println(s"Time: $t | Observed: ${problemInstance.observedData(i)} | Model Fit: ${curState(problemInstance.obsColumnIdx)}")

end odeEstimatorTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `odeEstimatorTest2` main method test the `OdeEstimator` trait.
 *  Uses a SEIR model with simulated data.
 *  > runMain scalation.dynamics.odeEstimatorTest2
 */
@main def odeEstimatorTest2 (): Unit =

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `SeirEpidemicModel` implements the normalized SEIR model system equations.
     *  State Index Mappings: u(0)=S, u(1)=E, u(2)=I, u(3)=R
     */
    class SeirEpidemicModel extends OdeEstimator:

        val numParams    = 3
        val timeGrid     = VectorD (0.0, 2.0, 4.0, 6.0, 8.0)            // time track mapping over 5 active monitoring phases
        val observedData = VectorD (0.010, 0.035, 0.082, 0.125, 0.110)  // experimental target data tracks the Infectious fraction population curve (u(2))
        val obsColumnIdx = 2                                            // tracking 'I' Compartment
    
        // Initial states: 98% Susceptible, 1% Exposed, 1% Infectious, 0% Recovered
        val initialStates = VectorD (0.98, 0.01, 0.01, 0.00)
    
        // Parameter box-bounds for (beta, sigma, gamma)
        val bounds = (VectorD (0.01, 0.01, 0.01),                       // lower constraints
                      VectorD (3.00, 5.00, 2.00))                       // upper constraints

        /*
         * Structural system equations matching epidemiological mechanics:
         * dS/dt = - beta * S * I
         * dE/dt =   beta * S * I - sigma * E
         * dI/dt =   sigma * E - gamma * I
         * dR/dt =   gamma * I
         */
        @nowarn def systemODEs (b: VectorD): Array [DerivativeV] =
            val beta  = b(0)
            val sigma = b(1)
            val gamma = b(2)
            
            Array ((t, u) => -beta * u(0) * u(2),                       // dS/dt
                   (t, u) => beta * u(0) * u(2) - sigma * u(1),         // dE/dt
                   (t, u) => sigma * u(1) - gamma * u(2),               // dI/dt
                   (t, u) => gamma * u(2))                              // dR/dt
        end systemODEs
    end SeirEpidemicModel

    println ("=== Initializing BoundedOdeSolver SEIR Estimation Engine ===")
    
    val problemInstance = new SeirEpidemicModel ()
    val solver = new BoundedOdeSolver (problemInstance)
    
    // Poor initial parameter guess [beta=0.1, sigma=0.1, gamma=0.1]
    val initialGuess = VectorD (0.1, 0.1, 0.1)
    println (s"-> Initial Parameters Guess: $initialGuess")
    
    // Run bounded Quasi-Newton minimization loop
    val (optimizedMse, optimalParameters) = solver.estimateParameters (initialGuess)
    
    println ("\n=== SEIR Optimization Results ===")
    println (s"-> Optimal Parameters Found [beta, sigma, gamma]: $optimalParameters")
    println (s"-> Minimum Mean Squared Error achieved: $optimizedMse")
    
    // Calculate R0 (Basic Reproduction Number) = beta / gamma
    val r0 = optimalParameters (0) / optimalParameters (2)
    println (s"-> Estimated Basic Reproduction Number (R0): $r0")
    
    println ("\nInfectious Profile Alignment [Compartment I]:")
    var curState = problemInstance.initialStates.copy
    val odes     = problemInstance.systemODEs (optimalParameters)
    
    for i <- problemInstance.timeGrid.indices do
        val t = problemInstance.timeGrid(i)
        if i > 0 then
            curState = DormandPrince.integrateVV (odes, curState, t, problemInstance.timeGrid(i - 1))
            
        val simValue = curState(problemInstance.obsColumnIdx)
        val obsValue = problemInstance.observedData(i)
        println (s"Day: $t | Observed Fraction: $obsValue | Model Fit Fraction: $simValue")

end odeEstimatorTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SEIR_Model` implements the normalized SEIR model system equations.
 *  State Index Mappings: u(0)=S, u(1)=E, u(2)=I, u(3)=R
 *  @param seirMatrix  the SEIR Training Data Matrix
 */
class SEIR_Model (val seirMatrix: MatrixD) extends OdeEstimator:

    val numParams    = 3                                     // optimizing: [beta_base, sigma, gamma]
    val timeGrid     = VectorD.range (0, seirMatrix.dim)     // time track mapping lengths matches row capacity counts
    val observedData = seirMatrix(?, 2)                      // extract engineered Infectious track (Col 2) to serve as fitting objective ground truth
    val obsColumnIdx = 2                                     // tracking 'I' (Infectious fraction)
//  val initialStates = seirMatrix(0)                        // extract complete row 0 profile to seed system boundaries exactly

    val initInfected = math.max (0.00005, seirMatrix(0, 2)) 
    val initExposed  = math.max (0.00010, seirMatrix(0, 1))

    val initialStates = VectorD (1.0 - initInfected - initExposed,   // S
                                 initExposed,                        // E
                                 initInfected,                       // I
                                 0.0)                                // R

    // Constraints to safely bind the optimization searching landscape
//  val bounds = (VectorD (0.01, 0.01, 0.01),                // lower limits
//                VectorD (3.50, 5.00, 2.00))                // upper limits
    val bounds = (VectorD (0.10, 0.40, 0.30),                // tighten lower bounds (gamma min = 0.3, approx 3 weeks max sick time)
                  VectorD (4.00, 3.00, 0.90))                // tighten upper bounds

    // Complete operational dynamic system equations

    @nowarn def systemODEs (b: VectorD): Array [DerivativeV] =
        val beta  = b(0)
        val sigma = b(1)
        val gamma = b(2)
        Array ((t, u) => -beta * u(0) * u(2),                // dS/dt
               (t, u) => beta * u(0) * u(2) - sigma * u(1),  // dE/dt
               (t, u) => sigma * u(1) - gamma * u(2),        // dI/dt
               (t, u) => gamma * u(2))                       // dR/dt
    end systemODEs

end SEIR_Model


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `odeEstimatorTest3` main method test the `OdeEstimator` trait.
 *  Uses a SEIR model with real data.
 *  > runMain scalation.dynamics.odeEstimatorTest3
 */
@main def odeEstimatorTest3 (): Unit =

    import modeling.forecasting.Example_Covid.loadData_yy    // add clip

    // Standard clinical/epidemiological boundary baselines
    val totalPopulation     = 331400000.0                    // US population
    val meanIncubationWeeks = 5.2 / 7.0                      // ~0.74 weeks
    val meanInfectiousWeeks = 10.0 / 7.0                     // ~1.43 weeks

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Extracts actual COVID time-series data and builds a clean training matrix.
     *  @return A MatrixD containing 4 columns corresponding to [S, E, I, R] fractions
     */
    def extractAndBuildData (): MatrixD =

        val features   = Array ("new_cases", "new_tests", "positive_rate")
        val dataMatrix = loadData_yy (features)

        val nSteps = dataMatrix.dim

       // Unpack individual data signals into discrete vector rows
        val newCasesVec     = dataMatrix(?, 0)
        val newDeathsVec    = dataMatrix(?, 1)
        val positiveRateVec = dataMatrix(?, 2)
        
        val sVec = new VectorD (nSteps)
        val eVec = new VectorD (nSteps)
        val iVec = new VectorD (nSteps)
        val rVec = new VectorD (nSteps)

        // Map observable columns based on recovery windows
        val recoverySteps     = math.ceil (meanInfectiousWeeks).toInt
        var cumulativeRemoved = 0.0

        for t <- 0 until nSteps do
            // I (Infectious) = Sum of new cases within active infectious period
            val startIdx   = math.max (0, t - recoverySteps)
            var activePool = 0.0
            for i <- startIdx to t do activePool += newCasesVec(i)
            iVec(t) = activePool

            // R (Recovered) = Accumulated metrics shifting out of active window
            if t >= recoverySteps then cumulativeRemoved += newCasesVec(t - recoverySteps)

            // Add direct real-world mortality data to secure the absorption boundary
            cumulativeRemoved += newDeathsVec(t)
            rVec(t) = cumulativeRemoved
        end for

        // Reverse-engineer hidden compartments using look-ahead metrics
        val incubationSteps = math.ceil (meanIncubationWeeks).toInt

        for t <- 0 until nSteps do
            // E (Exposed) = Future cases adjusted for under-reporting bias using the positivity rate.
            // A higher positivity rate indicates hidden, unobserved exposures in the community.
            val lookAhead = math.min (nSteps - 1, t + incubationSteps)
            val testingBiasCorrection = 1.0 + positiveRateVec(t)
        
            eVec(t) = newCasesVec(lookAhead) * meanIncubationWeeks * testingBiasCorrection

            // S (Susceptible) = Solved exactly using the Conservation of Mass invariant
            // S(t) = N - E(t) - I(t) - R(t)
            val remainingSusceptible = totalPopulation - eVec(t) - iVec(t) - rVec(t)
            sVec(t) = math.max (0.0, remainingSusceptible) // Enforce non-negative boundary constraint
        end for

        // Compile into a normalized training MatrixD [Rows x 4 Compartments]
        val trainingMatrix = new MatrixD (nSteps, 4)
        for t <- 0 until nSteps do
            trainingMatrix(t, 0) = sVec(t) / totalPopulation     // col 0: S
            trainingMatrix(t, 1) = eVec(t) / totalPopulation     // col 1: E
            trainingMatrix(t, 2) = iVec(t) / totalPopulation     // col 2: I
            trainingMatrix(t, 3) = rVec(t) / totalPopulation     // col 3: R
        end for

        trainingMatrix
    end extractAndBuildData

    // --- STEP 1: Extract Multi-Column Data via Pipeline ---
    banner ("[Step 1] Loading raw features using Example_Covid.loadData_yy...")
    val seirMatrix = extractAndBuildData ()
    println (s"-> Successfully built standard training matrix: ${seirMatrix.dims} rows, cols.")

    // --- STEP 2: Bind Problem Framework and Initialize Solver ---
    banner ("[Step 2] Initializing BoundedOdeSolver architecture...")

    // Limit data scope to first 25 weeks of the wave for clear epidemic growth mapping
//  val clippedMatrix = seirMatrix(0 until 25) 
    val clippedMatrix = seirMatrix(0 until 10) 

    val problemInstance = new SEIR_Model (clippedMatrix)
    val solver = new BoundedOdeSolver (problemInstance)

    // Initial non-optimized guess configurations
//  val initialGuess = VectorD (0.2, 0.4, 0.2)
    val initialGuess = VectorD (1.8, 1.0, 0.5)
    println (s"-> Configured Parameter Vector Guess: $initialGuess")

    // --- STEP 3: Execute Parameter Minimization Loop ---
    banner ("[Step 3] Launching L-BFGS-B Optimization Execution Path...")
    val (optimizedMse, optimalParameters) = solver.estimateParameters (initialGuess)

    // --- STEP 4: Output Analytical Verification Metrics ---
    banner ("=== FINAL ANALYTICAL RESULTS ===")
    println (s"-> Minimum Mean Squared Error (MSE) Achieved: $optimizedMse")
    println (s"-> Identified Optimal Parameter Vector:       $optimalParameters")
    println (f"   * Estimated Transmission Rate (beta):   ${optimalParameters(0)}%.4f")
    println (f"   * Estimated Incubation Rate (sigma):    ${optimalParameters(1)}%.4f")
    println (f"   * Estimated Recovery Rate (gamma):      ${optimalParameters(2)}%.4f")
    
    val calculatedR0 = optimalParameters(0) / optimalParameters(2)
    println (f"-> Derived Basic Reproduction Number (R0):    $calculatedR0%.4f")

    println ("\nProfile Alignment Checklist (First 8 Weeks Profile Data Comparison):")
    println ("Week \t Real Infectious \t Model Fitted Infectious \t Variance")
    println ("-----------------------------------------------------------------------")
    var curState = problemInstance.initialStates.copy
    val odes = problemInstance.systemODEs (optimalParameters)

    for t <- 0 until math.min (8, problemInstance.timeGrid.dim) do
        if t > 0 then
            val targetTime = problemInstance.timeGrid(t)
            val prevTime   = problemInstance.timeGrid(t - 1)
            // Sequence integration steps matching BoundedOdeSolver's exact trajectory track
            curState = DormandPrince.integrateVV (odes, curState, targetTime, prevTime)
        end if
            
        val simValue  = curState(problemInstance.obsColumnIdx)
        val realValue = problemInstance.observedData(t)
        val variance  = realValue - simValue
        println (f"$t \t $realValue%.6f \t\t $simValue%.6f \t\t $variance%+.6f")
    end for
        
    println ("\n=== Execution Finished Successfully ===")

end odeEstimatorTest3

