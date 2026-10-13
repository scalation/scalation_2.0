
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Jun 24 23:19:15 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Dampened Inverse Quadratic Functions with Closed-Form Inversion
 *           Less implosive replacement for 1/x^2
 *           Rationale:  Attenuated Inverse Square Law
 *  @note    AI Assisted Code
 *
 *  @note    For more flexibility, can change mx to a * mx (e.g. a = 0.25, 0.5, 0.75, 1.25)
 *           Can either provide a hyper-parameter or include a few choices for optimizer to weigh
 */

package scalation
package mathstat

import scala.math.{abs, max, signum, sqrt}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DampenedIQuad` object implements Dampened Sign-Preserving Inverse Quadratic Functions
 *  that reduce the chances of model input exploding.  Sign-Preservation allows for
 *  closed-form inverse functions.
 */
object DampenedIQuad:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward transformation:  Follows pure 1/x^2 behavior near 0, but dampens/throttles
     *  smoothly to a linear slope as it approaches max magnitude M.
     *  Formula:  f(y) = signum(y) * sqrt(1 + k * y^2) / (1 + y^2)
     *  @param y   the value to be transformed
     *  @param mx  the maximum value
     */
    def f (y: Double, mx: Double): Double =
        if y == 0.0 then return 0.0

        val k   = 1.0 / (mx * mx)
        val ySq = y * y
        (signum (y) * sqrt (1.0 + k * ySq)) / (1.0 + ySq)
    end f

    def f (y: VectorD, mx: Double): VectorD = y.map (f(_, mx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Algebraic closed-form exact inverse transformation.
     *  Resolves the quadratic in terms of y^2: A(y^2)^2 + B(y^2) + C = 0
     *  where z is the known constant output, A = z^2, B = 2z^2 - k, C = z^2 - 1
     *  @param z   the value to be transformed back
     *  @param mx  the maximum value/characteristic transition scale
     */
    def fi (z: Double, mx: Double): Double =
        if z == 0.0 then return 0.0

        val k    = 1.0 / (mx * mx)
        val az   = abs (z)
        val azSq = az * az
        val b    = 2.0 * azSq - k                     // terms for standard quadratic formula applied to y^2
        val c    = azSq - 1.0
        val discriminant = b * b - 4.0 * azSq * c     // discriminant: b^2 - 4ac simplifies to: k^2 - 4k(azSq) + 4(azSq)
        val u = (-b + sqrt (max (0.0, discriminant))) / (2.0 * azSq)    // select positive root to ensure a real value
        signum (z) * sqrt (u)
    end fi

    def fi (z: VectorD, mx: Double): VectorD = z.map (fi(_, mx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the derivative of f(y).
     *  Derivative:  f'(y) = |y|(k - 2 - ky^2) / ((1 + y^2)^2 * sqrt(1 + ky^2))
     *  @param y   the original value
     *  @param mx  the maximum value/characteristic transition scale
     */
    def fd (y: Double, mx: Double): Double =
        if y == 0.0 then return 0.0

        val k    = 1.0 / (mx * mx)
        val ySq  = y * y
        val ySq1 = ySq + 1.0
        abs (y) * (k - 2.0 - k * ySq) / (ySq1 * ySq1 * sqrt (1.0 + k * ySq))
    end fd

    def fd (y: VectorD, mx: Double): VectorD = y.map (fd(_, mx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Partial derivative of the forward transform f w.r.t. the max-magnitude mx
     *  Derivative:  df/dmx = -signum(y) * y^2 / (mx^3 * (1 + y^2) * sqrt(1 + ky^2))
     *  @param y   the original value
     *  @param mx  the maximum value/characteristic transition scale
     */
    def df_mx (y: Double, mx: Double): Double =
        if y == 0.0 || mx == 0.0 then return 0.0

        val k    = 1.0 / (mx * mx)
        val ySq  = y * y
        val ySq1 = ySq + 1.0
        val mx3  = mx * mx * mx
        -signum (y) * ySq / (mx3 * ySq1 * sqrt (1.0 + k * ySq))
    end df_mx

    def df_mx (y: VectorD, mx: Double): VectorD = y.map (df_mx(_, mx))

end DampenedIQuad

import DampenedIQuad._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dampenedIQuadTest` main function tests the `DampenedIQuad` object
 *  > runMain scalation.mathstat.dampenedIQuadTest
 */
@main def dampenedIQuadTest (): Unit =

//  val x  = VectorD.range (1 to 100)
    val x  = VectorD.range (-10 to 11)
    val xM = x.max * 0.2
    println (s"xM = $xM")
    val x2 = x ~^ -2; x2(10) = 0.0
    val x1 = x ~^ -1; x1(10) = 0.0
    val x_ = f(x, xM)

    new PlotM (x, MatrixD (x1, x2, x_), Array ("1/x", "x^-2", "x_"), "IQuad vs. DampenedIQuad", lines = true)

    val xi = fi(x_, xM)
    println (s"x    = $x")
    println (s"x_   = $x_")
    println (s"xi   = $xi")
    println (s"xi-x = ${xi-x}")

    val xd = fd(x, xM)   

    new PlotM (x, MatrixD (x, x_, xd), Array ("x", "x_", "xd"), "DampenedIQuad Derivative", lines = true)

end dampenedIQuadTest

