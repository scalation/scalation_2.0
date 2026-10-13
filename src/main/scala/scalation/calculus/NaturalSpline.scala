
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Aug  6 14:34:45 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Natural-Spline Basis Functions (e.g., used in Functional Data Analysis)
 *
 *  @see https://www.geeksforgeeks.org/machine-learning/natural-cubic-spline
 *  @see `B_Spline.scala`
 */

package scalation
package calculus

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NaturalSpline` class provides Natural-Spline basis functions with order 4,
 *  where the order is one more than the degree (cubic).  A spline function is a piecewise
 *  polynomial function where the pieces are stitched together at knots with the
 *  goal of maintaining continuity and differentiability.  B-Spline basis functions
 *  form a popular form of basis functions used in Functional Data Analysis.
 *  It provides natural cubic spline basis functions, restricting the standard order 4 (cubic)
 *  B-Splines by enforcing linear boundary conditions (or dampened quadratic when
 *  dq is true.
 *-----------------------------------------------------------------------------
 *  @param ττ     the time-points of the original knots in the time dimension
 *  @param dq     whether to use dampened quadratic or linear trend extension
 *  @param clamp  flag for augmenting ττ (defaults to true)
 */
class NaturalSpline (ττ: VectorD, dq: Boolean = true, clamp: Boolean = true) 
      extends B_Spline (ττ, mMax = 4, clamp):                     // fixed at order 4 (Cubic)

//  private val debug   = debugf ("NaturalSpline", true)          // debug function
    private val rawSize = super.size (4)                          // number of usable B-Spline basis functions
    private val natSize = rawSize - 2                             // Natural Splines drop this by 2
    private val tm      = new MatrixD (rawSize, natSize)          // the transformation matrix

    if dq then initTransformationMatrixDQ ()
    else initTransformationMatrix ()
    println (s"NaturalSpline: tm = $tm")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reduce the basis size by 2.  Natural boundary conditions remove 2 degrees of freedom.
     */
    override def size (m: Int = 4): Int = super.size (4) - 2

    override def range (m: Int = 4): Range = 0 to (size () - 1)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize the transformation matrix mapping parameters.
     *  Construct the restricted natural basis matrix mapping to force 2nd derivatives to 0 at endpoints.
     */
    private def initTransformationMatrix (): Unit =
        val n_1 = rawSize - 1

        // Calculate the first and last knot interval gaps (h)
        val h0 = ττ(1) - ττ(0)
        val hn = ττ(ττ.dim - 1) - ττ(ττ.dim - 2)

        // Analytical natural spline boundary conditions derived from Cox-de Boor recurrence
        val alpha1 = 1.0 + (h0 / (ττ(2) - ττ(0))) // Resolves to 2.0 for uniform knots
        val alpha2 = -(h0 / (ττ(2) - ττ(0)))      // Resolves to -1.0 for uniform knots

        val beta1  = -(hn / (ττ(ττ.dim - 1) - ττ(ττ.dim - 3)))    // resolves to -1.0 for uniform knots
        val beta2  = 1.0 + (hn / (ττ(ττ.dim - 1) - ττ(ττ.dim - 3)))  // resolves to 2.0 for uniform knots

        // Populate the left edge mappings
        tm(0, 0) = alpha1
        tm(0, 1) = alpha2

        // Populate the identity mapping rows for the inner basis functions
        for j <- 0 until natSize do
            tm(j + 1, j) = 1.0

        // Populate the right edge mappings
        tm(n_1, natSize - 2) = beta1
        tm(n_1, natSize - 1) = beta2
    end initTransformationMatrix

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize the transformation matrix mapping parameters.
     *  Construct the restricted natural basis matrix mapping to force 2nd derivatives to 0 at endpoints.
     *
    private def initTransformationMatrix2 (): Unit =
        val dbSpline = new DB_Spline (ττ, mMax = 4, clamp)
        val n_1      = rawSize - 1
        val eps      = 1e-9
        val t_left   = head + eps
        val t_right  = tail - eps

        // Left boundary second derivatives (j = 0, 1, 2) evaluated at 'head'
        val d2_L0 = dbSpline.d2bf (4)(0)(t_left)
        val d2_L1 = dbSpline.d2bf (4)(1)(t_left)
        val d2_L2 = dbSpline.d2bf (4)(2)(t_left)

        // Right boundary second derivatives (j = n_1-2, n_1-1, n_1) evaluated at 'tail'
        val d2_R_2 = dbSpline.d2bf (4)(n_1 - 2)(t_right)
        val d2_R_1 = dbSpline.d2bf (4)(n_1 - 1)(t_right)
        val d2_R0  = dbSpline.d2bf (4)(n_1)(t_right)

        // Compute the natural spline constraint boundary weights
        val alpha1 = -d2_L1 / d2_L0
        val alpha2 = -d2_L2 / d2_L0

        val beta1  = -d2_R_2 / d2_R0
        val beta2  = -d2_R_1 / d2_R0

        // Map left boundary conditions into the first column mapping positions
        tm(0, 0) = alpha1
        tm(0, 1) = alpha2

        // Map identity matrix properties to align inner coefficients
        for j <- 0 until natSize do tm(j + 1, j) = 1.0

        // Map right boundary conditions into the last column mapping positions
        tm(n_1, natSize - 2) = beta1
        tm(n_1, natSize - 1) = beta2
    end initTransformationMatrix2
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize the transformation matrix with dampened quadratic properties
     *  @param dampingFactor  the factor for the Dampaned Quadratic
     */
    private def initTransformationMatrixDQ (dampingFactor: Double = 0.85): Unit =
        val n_1 = rawSize - 1

        // Calculate the basic knot widths
        val h0 = ττ(1) - ττ(0)
        val hn = ττ(ττ.dim - 1) - ττ(ττ.dim - 2)

        // Calculate analytical base weights
        val base_alpha1 = 1.0 + (h0 / (ττ(2) - ττ(0)))
        val base_alpha2 = -(h0 / (ττ(2) - ττ(0)))

        val base_beta1  = -(hn / (ττ(ττ.dim - 1) - ττ(ττ.dim - 3)))
        val base_beta2  = 1.0 + (hn / (ττ(ττ.dim - 1) - ττ(ττ.dim - 3)))

        // Apply the damping factor to blend a quadratic curve toward a linear asymptote
        // A factor closer to 1.0 allows more quadratic curvature; closer to 0 vectors to linear.
        val alpha1 = base_alpha1 * dampingFactor
        val alpha2 = base_alpha2 * (2.0 - dampingFactor)

        val beta1  = base_beta1 * (2.0 - dampingFactor)
        val beta2  = base_beta2 * dampingFactor

        // 1. Populate the left edge mappings
        tm(0, 0) = alpha1
        tm(0, 1) = alpha2

        // 2. Populate the identity mapping rows
        for j <- 0 until natSize do tm(j + 1, j) = 1.0

        // 3. Populate the right edge mappings
        tm(n_1, natSize - 2) = beta1
        tm(n_1, natSize - 1) = beta2
    end initTransformationMatrixDQ

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Override the basis function evaluation. Project raw B-splines into natural
     *  basis space: N(t) = M^T * B(t).
     */
    override def abf_ (m: Int = 4)(t: Double): VectorD =
        val b_basis = super.abf_ (4)(t)                           // get raw B-spline basis
        tm.transpose * b_basis
    end abf_

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the Design/Basis Matrix (Φ).
     *  @param  t  the time points
     */
    def buildMatrix (t: VectorD): MatrixD =
        val Φ = new MatrixD (t.dim, size ())
        for i <- t.indices do Φ(i) = abf_ ()(t(i))                // row i contains basis evaluations at t(i)
        Φ
    end buildMatrix

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for the spline coefficients (c) using matrix factorization.
     *  c = (Φ^T * Φ)^-1 * Φ^T * actual
     *  c = (Φ.transpose * Φ).inverse * Φ.transpose * actual
     *  @param Φ  the Design/Basis Matrix
     *  @param y  the actual/observed time series
     */
    def solveSpline (Φ: MatrixD, y: VectorD): VectorD =
        val ΦtΦ = Φ.transpose * Φ
        val fac = new Fac_Cholesky (ΦtΦ) 
        fac.factor ()
        fac.solve (Φ.transpose * y)
    end solveSpline

end NaturalSpline


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naturalSplineTest` main function test the `NaturalSpline` class.
 *  > runMain scalation.calculus.naturalSplineTest
 */
@main def naturalSplineTest (): Unit =

    val τ  = VectorD (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
    val ns = new NaturalSpline (τ)
    println (s"Natural Spline Size: ${ns.size()}")
    // Further testing and plotting logic...

end naturalSplineTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naturalSplineTest2` main function test the `NaturalSpline` class.
 *  > runMain scalation.calculus.naturalSplineTest2
 */
@main def naturalSplineTest2 (): Unit =

    import scalation.random.Normal
    import scala.math.sin
    
    val normal = Normal ()
    val n      = 100                                              // 100 time points
    val t      = VectorD.range (0, n)                             // time axis
    
    // Generate fake time series data (Linear Trend + Wave + Noise)
    val actual = new VectorD (n)
    for i <- 0 until n do 
        actual(i) = 15.0 + (0.4 * t(i)) + (8.0 * sin (t(i) * 0.08)) + normal.gen

    val τ  = VectorD (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)         // define knot locations
    val ns = new NaturalSpline (τ)                                // create a natural spline model
    val Φ  = ns.buildMatrix (t)                                   // build its design/basis matrix
    val c  = ns.solveSpline (Φ, actual)                           // solve for spline coefficients
    println (s"Natural Spline Basis Size: ${ns.size ()}")
    println (s"Design/Basis Matrix Φ = $Φ")
    println (s"Spline Coefficients c = $c")
    
    // Generate the smooth spline curve points for plotting
    val spline = Φ * c

    // Plot actual points vs. the smooth fitted spline trend line
    // Parameters: x-axis, y1 (dots/scatter), y2 (continuous line)
    println (s"actual vs. spline = ${MatrixD (actual, spline).transpose}")
    new Plot (t, actual, spline, "Actual vs. Natural Spline Curve Trend")

end naturalSplineTest2

