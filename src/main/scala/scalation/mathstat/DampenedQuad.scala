
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Jun 24 23:19:15 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Dampened Sign-Preserving Quadratic Functions with Closed-Form Inversion
 *           Less implosive replacement for signum (x) * x^2
 *           Rationale:  Attenuated Square Law
 *  @note    AI Assisted Code
 *
 *  @note    For more flexibility, can change mx to a * mx (e.g. a = 0.25, 0.5, 0.75, 1.25)
 *           Can either provide a hyper-parameter or include a few choices for optimizer to weigh
 */

package scalation
package mathstat

import scala.math.{abs, signum, sqrt}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DampenedQuad` object implements Dampened Sign-Preserving Quadratic Functions
 *  that reduce the chances of model input exploding.  Sign-Preservation allows for
 *  closed-form inverse functions.
 */
object DampenedQuad:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward transformation:  Follows pure x^2 behavior near 0, but dampens/throttles
     *  smoothly to a linear slope as it approaches max magnitude M.
     *  Formula:  f(y) = (y * |y|) / sqrt(1 + k * y^2) where k = 1 / M^2
     *  @param y   the value to be transformed
     *  @param mx  the maximum value/characteristic transition scale
     */
    def f (y: Double, mx: Double): Double =
        val k = 1.0 / mx~^2
        y * abs (y) / sqrt (1.0 + k * y * y)
    end f

    def f (y: VectorD, mx: Double): VectorD = y.map (f(_, mx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Algebraic closed-form exact inverse transformation.
     *  Resolves the bi-quadratic equation:  y^4 - (k * z^2) * y^2 - z^2 = 0
     *  @param z   the value to be transformed back
     *  @param mx  the maximum value/characteristic transition scale
     */
    def fi (z: Double, mx: Double): Double =
        val k    = 1.0 / mx~^2
        val zSq  = z * z
        val kzSq = k * zSq
        val u = (kzSq + sqrt (kzSq * kzSq + 4.0 * zSq)) / 2.0     // quadratic formula: (-b + sqrt(b^2 - 4ac)) / 2a
        signum (z) * sqrt (u)
    end fi

    def fi (z: VectorD, mx: Double): VectorD = z.map (fi(_, mx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the derivative of f(y).
     *  Derivative:  f'(y) = |y|(2 + ky^2) / ((1 + ky^2)^1.5)
     *  @param y   the original value
     *  @param mx  the maximum value/characteristic transition scale
     */
    def fd (y: Double, mx: Double): Double =
        val k = 1.0 / mx~^2
        abs (y) * (2.0 + k * y * y) / ((1 + k * y * y)~^1.5)
    end fd

    def fd (y: VectorD, mx: Double): VectorD = y.map (fd(_, mx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Partial derivative of the forward transform f w.r.t. the max-magnitude
     *  parameter mx (used to assemble the Jacobian):
     *      df/dmx = signum (y) * y^4 / (mx^2 + y^2)^(3/2)
     *  Equivalently, in the k = 1/mx^2 form used by f:
     *      df/dmx = signum (y) * y^4 / (mx^3 * (1 + k * y^2)^(3/2))
     *  @param y   the original value
     *  @param mx  the maximum value/characteristic transition scale
     */
    def df_mx (y: Double, mx: Double): Double =
        val yy = y * y
        signum (y) * yy * yy / (mx * mx + yy)~^1.5
    end df_mx

    def df_mx (y: VectorD, mx: Double): VectorD = y.map (df_mx(_, mx))

end DampenedQuad

import DampenedQuad._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `dampenedQuadTest` main function tests the `DampenedQuad` object
 *  > runMain scalation.mathstat.dampenedQuadTest
 */
@main def dampenedQuadTest (): Unit =

//  val x  = VectorD.range (1 to 100)
    val x  = VectorD.range (-10 to 11)
    val xM = x.max
    println (s"xM = $xM")
    val x2 = x ~^ 2
    val x_ = f(x, xM)

    new PlotM (x, MatrixD (x, x2, x_), Array ("x", "x^2", "x_"), "Quad vs. DampenedQuad", lines = true)

    val xi = fi(x_, xM)
    println (s"x    = $x")
    println (s"x_   = $x_")
    println (s"xi   = $xi")
    println (s"xi-x = ${xi-x}")

    val xd = fd(x, xM)   

    new PlotM (x, MatrixD (x, x_, xd), Array ("x", "x_", "xd"), "DampenedQuad Derivative", lines = true)

end dampenedQuadTest

