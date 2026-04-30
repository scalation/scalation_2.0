
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Jan 28 17:18:16 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Differential Calculus Operations and Functions
 *
 *  @see gwu.geverstine.com/pdenum.pdf
 */

package scalation
package calculus

import scala.math.sqrt

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Differential` object contains functions for computing derivatives, partial
 *  derivatives, Laplacians, gradient vectors, Hessian matrices and Jacobian matrices.
 */
object Differential:

    private var h   = 1E-6                     // default step size used for estimating derivatives
    private var h2  = h + h                    // twice the step size
    private var hh  = h * h                    // step size squared
    private var hh4 = 4.0 * hh                 // for times step size squared
    private var hl  = sqrt (h)                 // relative large step for finite difference

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the step size from its default step size to one more suitable for
     *  your function.  A heuristic for the central difference method is to let
     *  h = max (|x|,1) * (machine-epsilon)^(1/3)
     *  For double precision, the machine-epsilon is about 1E-16.
     *  @see www.karenkopecky.net/Teaching/eco613614/Notes_NumericalDifferentiation.pdf
     *  @param step  the new step size to reset h to
     */
    def resetH (step: Double): Unit = h = step; h2 = h + h; hh = h * h; hh4 = 4.0 * hh

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the large step size from its default step size to one more suitable
     *  for your function.
     *  @param largeStep  the new large step size to reset hl to
     */
    def resetHR (largeStep: Double): Unit = hl = largeStep

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Finite Difference
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the finite difference of the scalar-to-scalar function f at x
     *  using a 1-sided method (forward difference).  May work better than the
     *  derivative for noisy functions.
     *  @param f  the function whose derivative is sought
     *  @param x  the point (scalar) at which to estimate the finite difference
     */
    def fdiffernce (f: FunctionS2S, x: Double): Double = (f(x + hl) - f(x)) / hl

    inline def Δ (f: FunctionS2S)(x: Double): Double = fdiffernce (f, x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // First Order
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the derivative of the scalar-to-scalar function f at x using
     *  a 1-sided method (forward difference).  Approximate the tangent line at
     *  (x, f(x)) with the secant line through points (x, f(x)) and (x+h, f(x+h)).
     *  @param f  the function whose derivative is sought
     *  @param x  the point (scalar) at which to estimate the derivative
     */
    def derivative1 (f: FunctionS2S, x: Double): Double = (f(x + h) - f(x)) / h

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the derivative of the scalar-to-scalar function f at x using
     *  a 2-sided method (central difference).  Approximate the tangent line at
     *  (x, f(x)) with the secant line through points (x-h, f(x-h)) and (x+h, f(x+h)).
     *  Tends to be MORE ACCURATE than the 1-sided method.
     *  Although a noun, derivative was chosen over the verb differentiate.
     *  @see www.math.montana.edu/frankw/ccp/modeling/continuous/heatflow2/firstder.htm
     *  @param f  the function whose derivative is sought
     *  @param x  the point (scalar) at which to estimate the derivative
     */
    def derivative (f: FunctionS2S, x: Double): Double = (f(x + h) - f(x - h)) / h2

    inline def Ⅾ (f: FunctionS2S)(x: Double): Double = derivative (f, x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the i-th partial derivative of the vector-to-scalar function f at
     *  point x returning the value for the partial derivative for dimension i.
     *  @param i  the dimension to compute the partial derivative on
     *  @param f  the function whose partial derivative is sought
     *  @param x  the point (vector) at which to estimate the partial derivative
     */
    def partial (i: Int)(f: FunctionV2S, x: VectorD): Double = (f(x + (i, h)) - f(x - (i, h))) / h2

    inline def ∂ (i: Int)(f: FunctionV2S, x: VectorD): Double = partial (i)(f, x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the gradient of the vector-to-scalar function f at point x
     *  returning a value for the partial derivative for each dimension of x.
     *  @param f  the function whose gradient is sought
     *  @param x  the point (vector) at which to estimate the gradient
     */
    def grad (f: FunctionV2S, x: VectorD): VectorD =
        VectorD (for i <- x.indices yield (f(x + (i, h)) - f(x - (i, h))) / h2)

    inline def ∇ (f: FunctionV2S)(x: VectorD): VectorD = grad (f, x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the slope of the vector-to-scalar function f defined on mixed
     *  real/integer vectors.
     *  @param f  the function whose slope is sought
     *  @param x  the point (vector) at which to estimate the slope
     *  @param n  the number of dimensions that are real-valued (rest are integers)
     */
    def slope (f: FunctionV2S, x: VectorD, n: Int = 0): VectorD =
        val c = new VectorD (x.dim)
        for i <- x.indices do
            c(i) = if i < n then (f(x + (i, h)) - f(x - (i, h))) / h2         // derivative
                   else          (f(x + (i, 1.0)) - f(x - (i, 1.0))) / 2.0    // difference
        end for
        c
    end slope

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Jacobian matrix for a vector-valued function represented as
     *  an array of scalar-valued functions.  The i-th row in the matrix is the
     *  gradient of the i-th function.
     *  @param f  the array of functions whose Jacobian is sought
     *  @param x  the point (vector) at which to estimate the Jacobian
     */
    def jacobian (f: Array [FunctionV2S], x: VectorD): MatrixD =
        MatrixD (for i <- f.indices yield grad (f(i), x))

    inline def Ј (f: Array [FunctionV2S], x: VectorD): MatrixD = jacobian (f, x)   // want row-wise

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluate an array of vector-to-scalar functions (`FunctionV2S`) as one
     *  vector-to-vector function (`FunctionV2V`).
     *  @param f  the array of functions to be evaluated
     *  @param x  the point (vector) at which to functionally evaluate
     */
    def eval (f: Array [FunctionV2S], x: VectorD): VectorD =
        VectorD (for i <- f.indices yield f(i)(x))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert an array of vector-to-scalar functions (`FunctionV2S`) into one
     *  vector-to-vector function (`FunctionV2V`).
     *  @param f  the array of functions to be evaluated
     */
    def array2f (f: Array [FunctionV2S]): FunctionV2V = 
        (x: VectorD) => VectorD (for i <- f.indices yield f(i)(x))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Second Order
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the second derivative of the scalar-to-scalar function f at
     *  x using the central difference formula for second derivatives.
     *  @param f  the function whose second derivative is sought
     *  @param x  the point (scalar) at which to estimate the derivative
     */
    def derivative2 (f: FunctionS2S, x: Double): Double = (f(x + h) - 2.0*f(x) + f(x - h)) / hh

    inline def ⅮⅮ (f: FunctionS2S)(x: Double): Double = derivative2 (f, x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the (i,j)th second partial derivative of the vector-to-scalar
     *  function f at point x returning the value for the second partial derivative
     *  for dimensions (i, j).  If i = j, the second partial derivative is
     *  called "pure", otherwise it is a "cross" second partial derivative.
     *  @see www.uio.no/studier/emner/matnat/math/MAT-INF1100/h07/undervisningsmateriale/kap7.pdf
     *  @param i  the first dimension to compute the second partial derivative on
     *  @param j  the second dimension to compute the second partial derivative on
     *  @param f  the function whose second partial derivative is sought
     *  @param x  the point (vector) at which to estimate the second partial derivative
     */
    def partial2 (i: Int, j: Int)(f: FunctionV2S, x: VectorD): Double = 
        val (hi, hj) = ((i, h), (j, h))
        if i == j then                                                            // pure partial
            (f(x + hi) - 2.0*f(x) + f(x - hi)) / hh
        else                                                                      // cross partial
            (f(x + hi + hj) - f(x + hi - hj) - f(x - hi + hj) + f(x - hi - hj)) / hh4
    end partial2

    inline def ∂∂ (i: Int, j: Int)(f: FunctionV2S, x: VectorD): Double = partial2 (i, j)(f, x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the Hessian of the vector-to-scalar function f at point x
     *  returning a matrix of second partial derivative.
     *  @param f  the function whose Hessian is sought
     *  @param x  the point (vector) at which to estimate the Hessian
     */
    def hessian (f: FunctionV2S, x: VectorD): MatrixD =
        val h = new MatrixD (x.dim, x.dim)
        for i <- x.indices; j <- 0 to i do
            h(i, j) = ∂∂ (i, j)(f, x) 
            if j < i then h(j, i) = h(i, j)                                       // Hessian is symmetric
        end for
        h
    end hessian

    inline def Η (f: FunctionV2S, x: VectorD): MatrixD = hessian (f, x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate the Laplacian of the vector-to-scalar function f at point x
     *  returning the sum of the pure second partial derivatives.
     *  @param f  the function whose Hessian is sought
     *  @param x  the point (vector) at which to estimate the Hessian
     */
    def laplacian (f: FunctionV2S, x: VectorD): Double =
        var sum = 0.0
        for i <- x.indices do sum += (f(x + (i, h)) - 2.0*f(x) + f(x - (i, h))) / hh
        sum
    end laplacian

    inline def ∆ (f: FunctionV2S, x: VectorD): Double = laplacian (f, x)

end Differential

import Differential._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `differentialTest` main function is used to test the `Differential` object.
 *  > runMain scalation.calculus.differentialTest
 */
@main def differentialTest (): Unit =

    def g (y: Double): Double    = 2 * (y - 3)~^2
    def f (x: VectorD): Double   = 2 * (x(0) - 3)~^2 + (x(1) - 4)~^2 + x(0) * x(1)
    def gr (x: VectorD): VectorD = VectorD (4 * x(0) + x(1) - 12, 2 * x(1) + x(0) - 8)
    def f3 (x: VectorD): Double  = x(0) * x(1)

    val y  = 2.0
    val x  = VectorD (2.0, 2.0)

    banner ("Derivatives")
    println (s"Ⅾ g($y)  = ${Ⅾ (g)(y)}")                         // derivative
    println (s"ⅮⅮ g($y) = ${ⅮⅮ (g)(y)}")                        // second derivative

    banner ("Partial Derivatives")
    println (s"∂ (0)(f($x)    = ${∂ (0)(f, x)}")                // partial derivative wrt x0
    println (s"∂ (1)(f($x)    = ${∂ (1)(f, x)}")                // partial derivative wrt x1
    println (s"∂∂ (0,0)(f($x) = ${∂∂ (0,0)(f, x)}")             // second partial derivative wrt x0
    println (s"∂∂ (1,1)(f($x) = ${∂∂ (1,1)(f, x)}")             // second partial derivative wrt x1
    println (s"∂∂ (1,0)(f($x) = ${∂∂ (1,0)(f, x)}")             // cross partial derivative
    println (s"∂∂ (1,0)(f3($x) = ${∂∂ (1,0)(f3, x)}")           // cross partial derivative

    banner ("Gradient Vectors")
    val gr1 = ∇ (f)(x)                                          // gradient computed numerically
    val gr2 = gr(x)                                             // gradient from functions for partials
    println (s"∇ f($x) = $gr1")
    println (s"gr($x)  = $gr2")
    println (s"gr1 - gr2 = ${gr1 - gr2}")

    val ff = Array [FunctionV2S] ((x: VectorD) => 2 * x(0) + x(1),
                                  (x: VectorD) => 2 * x(0) - x(1))

    banner ("Evaluation of ff as one funtion")
    println (s"eval (ff, $x) = ${eval (ff, x)}")

    banner ("Jacobian Matrices")
    println (s"Ј ff($x) = ${Ј (ff, x)}")                        // Jacobian

    banner ("Laplacians")
    println (s"∆ f($x) = ${∆ (f, x)}")                          // Laplacian

    banner ("Hessian Matrices")
    println (s"Η f($x) = ${Η (f, x)}")                          // Hessian

end differentialTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `differentialTest2` main function is used to test the `Differential` object
 *  showing trade-offs of using  1-sided and 2-sided derivative approximations
 *  as well as different values for h.
 *  @see www.rose-hulman.edu/~bryan/lottamath/diffgrad.pdf
 *  > runMain scalation.calculus.differentialTest2
 */
@main def differentialTest2 (): Unit =

    import scala.math.{abs, cos, sin}

    def f (x: Double): Double = sin (x)      // the function

    def d (x: Double): Double = cos (x)      // its derivative

    val x = Array (.0, .1, .2, .3, .4, .5, .6, .7, .8, .9)
    for i <- x.indices do
        var hh = 1E-4
        println (" x \t\t h \t\t deriv \t\t 1-sided \t\t error \t\t 2-sided \t\t error")
        cfor (0, 9) { _ =>
            resetH (hh)
            val (d0, d1, d2) = (d(x(i)), derivative1 (f, x(i)), derivative (f, x(i)))
            println (s"${x(i)} \t $hh \t $d0 \t $d1 \t ${abs (d1-d0)} \t $d2 \t ${abs (d2-d0)}")
            hh /= 10.0
        } // cfor
        println ()
    end for

end differentialTest2

