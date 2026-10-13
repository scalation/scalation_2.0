
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Jul  6 22:18:54 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Cubic Splines
 */

package scalation
package calculus

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CubicSpline` class provides localized polynomial smoothing and analytical
 *  derivatives over a strictly increasing time domain index.
 *  @param t  the time vector (or horizontal axis)
 *  @param y  the value vector (or verticle axis)
 */
class CubicSpline (t: VectorD, y: VectorD):

    val n = t.dim
    private val h = new VectorD (n - 1)
    private val a = y.copy

    // Precalculate time delta steps
    for i <- 0 until n - 1 do h(i) = t(i+1) - t(i)

    // Formulate the Tridiagonal System matrix parameters to enforce smooth 2nd derivatives
    private val alpha = new VectorD (n - 1)
    for i <- 1 until n - 1 do alpha(i) = (3.0 / h(i)) * (a(i+1) - a(i)) - (3.0 / h(i-1)) * (a(i) - a(i-1))

    private val l  = new VectorD (n)
    private val mu = new VectorD (n)
    private val z  = new VectorD (n)
    private val c  = new VectorD (n)
    private val b  = new VectorD (n - 1)
    private val d  = new VectorD (n - 1)

    l(0) = 1.0; mu(0) = 0.0; z(0) = 0.0
    for i <- 1 until n - 1 do
        l(i)  = 2.0 * (t(i+1) - t(i-1)) - h(i-1) * mu(i-1)
        mu(i) = h(i) / l(i)
        z(i)  = (alpha(i) - h(i-1) * z(i-1)) / l(i)
    end for

    l(n-1) = 1.0; z(n-1) = 0.0; c(n-1) = 0.0
    
    // Back-substitution step loop
    for j <- n - 2 to 0 by -1 do
        c(j) = z(j) - mu(j) * c(j+1)
        b(j) = (a(j+1) - a(j)) / h(j) - h(j) * (c(j+1) + 2.0 * c(j)) / 3.0
        d(j) = (c(j+1) - c(j)) / (3.0 * h(j))
    end for

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate the smoothed y-value at an arbitrary time point ti.
     *  @param ti  the given time point
     */
    def eval (ti: Double): Double =
        val idx = findSegment (ti)
        val dx  = ti - t(idx)
        a(idx) + b(idx) * dx + c(idx) * dx * dx + d(idx) * dx * dx * dx
    end eval

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the analytical derivative (dy/dt) at an arbitrary time point ti.
     *  @param ti  the given time point
     */
    def derivative (ti: Double): Double =
        val idx = findSegment (ti)
        val dx  = ti - t(idx)
        b(idx) + 2.0 * c(idx) * dx + 3.0 * d(idx) * dx * dx
    end derivative

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the index of the localized interval segment containing the target time 'ti'.
     *  Uses an efficient binary search algorithm (O(log n)) to locate the bracket index 
     *  such that t(idx) <= ti <= t(idx + 1). Bounds protection automatically clamps 
     *  out-of-range times to the edge intervals.
     *  @param ti  the given time point
     */
    private def findSegment (ti: Double): Int =
        if ti <= t(0) then 0
        else if ti >= t(n-1) then n - 2
        else
            var low = 0
            var high = n - 1
            while high - low > 1 do
                val mid = (low + high) / 2
                if t(mid) <= ti then low = mid else high = mid
            low
    end findSegment
            
end CubicSpline


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cubicSplineTest` main function tests the `CubicSpline` class.
 *  > runMain scalation.calculus.cubicSplineTest
 */
@main def cubicSplineTest (): Unit =

    import scala.math.{sin, cos, abs}

    println ("=== Testing CubicSpline Class ===")

    // 1. GENERATE DUMMY SIGNAL DATA: y = sin(t)

//  val points = 6
    val t = VectorD (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    val y = VectorD (for ti <- t yield sin (ti))

    // Initialize your inline spline engine
    val spline = new CubicSpline (t, y)

    // 2. VERIFY EXACT INTERPOLATION AT NODE POINTS (y_pred == y_actual)

    println("\nChecking boundary node accuracy:")
    for i <- t.indices do
        val ti = t(i)
        val actualY = y(i)
        val predY   = spline.eval(ti)
        val error   = abs(actualY - predY)
        
        println (s"t = $ti -> Actual: ${"%6.4f".format(actualY)} | Spline: ${"%6.4f".format(predY)} | Error: ${"%6.1e".format(error)}")

    // 3. VERIFY DERIVATIVE TRACKING AT INTERMEDIATE TIME STEPS (dy/dt == cos(t))

    println("\nChecking intermediate derivative calculations (Compared against Calculus cos(t)):")
    // Evaluate halfway points between grid intervals
    val testTimes = Array (0.5, 1.5, 2.5, 3.5, 4.5)

    for ti <- testTimes do
        val expectedDeriv = cos(ti)
        val splineDeriv   = spline.derivative(ti)
        val derivError    = abs(expectedDeriv - splineDeriv)

        println (s"t = $ti -> Expected cos(t): ${"%6.4f".format(expectedDeriv)} | Spline dy/dt: ${"%6.4f".format(splineDeriv)} | Error: ${"%6.4f".format(derivError)}")

    println ("\n=== Spline Engine Test Complete ===")

end cubicSplineTest

