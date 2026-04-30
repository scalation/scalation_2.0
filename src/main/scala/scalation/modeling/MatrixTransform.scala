
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jan 31 20:59:02 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Matrix (and Vector) Transformations
 */

package scalation
package modeling

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Top level functions used to transform vectors and matrices (e.g., scale,
 *  center and normalize).
 *  Such pre-processing of the data is required by some modeling techniques.
 */

// Vector Transformations --------------------------------------------------

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the extreme values (min, max) for vector x.
 *  @param x  the vector whose extreme values are sought
 */
inline def extreme (x: VectorD): (Double, Double) = (x.min, x.max)

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the mean and standard deviation stats for vector x
 *  @param x  the vector whose stats are sought
 */
inline def mean_stdev (x: VectorD): (Double, Double) = (x.mean, x.stdev)

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Scale vector x to the range lb to ub: x -> x_s.
 *  @param extremes  the (minimum value, maximum value) in vector x
 *  @param bounds    the desired (lower, upper) bounds
 *-------------------------------------------------------------------------------
 *  @param x         the vector to scale
 */
def scaleV (extremes: (Double, Double), bounds: (Double, Double) = (0, 1)) (x: VectorD): VectorD =
    val (min_x, max_x) = extremes
    val (lb, ub) = bounds
    val scale = (ub - lb) / (max_x - min_x)
    (x - min_x) * scale + lb                                     // shift and scale
end scaleV

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Unscale vector x_s from the range lb to ub to original range: x_s -> x.
 *  @param extremes  the (minimum value, maximum value) in original vector x
 *  @param bounds    the scaled (lower, upper) bounds
 *-------------------------------------------------------------------------------
 *  @param x_s       the vector to unscale
 */
def unscaleV (extremes: (Double, Double), bounds: (Double, Double) = (0, 1)) (x_s: VectorD): VectorD =
    val (min_x, max_x) = extremes
    val (lb, ub) = bounds
    val scale = (ub - lb) / (max_x - min_x)
    (x_s - lb) / scale + min_x                                   // shift and scale
end unscaleV

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Normalize the vector x to zero mean and unit standard deviation,
 *  by subtracting the mean and dividing by the standard deviation.
 *  Don't normalize if the standard deviation is nearly zero.
 *  @param mu_sig  the column vector's mean and standard deviation
 *-------------------------------------------------------------------------------
 *  @param x       the vector to normalize
 */
def normalizeV (mu_sig: (Double, Double)) (x: VectorD): VectorD =
    val (mu_x, sig_x) = mu_sig
    if sig_x =~ 0.0 then x else (x - mu_x) / sig_x
end normalizeV

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Denormalize the vector x_n from zero mean and unit standard deviation,
 *  by multiplying by the standard deviation and adding the mean.
 *  Don't denormalize if the standard deviation is nearly zero.
 *  @param mu_sig  the column vector's mean and standard deviation
 *-------------------------------------------------------------------------------
 *  @param x_n     the vector to denormalize
 */
def denormalizeV (mu_sig: (Double, Double)) (x_n: VectorD): VectorD =
    val (mu_x, sig_x) = mu_sig
    if sig_x =~ 0.0 then x_n else x_n * sig_x + mu_x
end denormalizeV

// ADDED FOR COVID TESTING WITH RNNs -- @author Praveen -- FIX merge with mathstat.Transform
def logTransformV (offset: Double = 1e-8)(x: VectorD): VectorD =
    x.map (v => math.log (v + offset))
end logTransformV

def expTransformV (offset: Double = 1e-8)(x_log: VectorD): VectorD =
    x_log.map (v => math.exp(v) - offset)
end expTransformV

// Matrix Transformations --------------------------------------------------

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the extreme values (min, max) for matrix x, for each column.
 *  @param x  the matrix whose extreme values are sought
 */
def extreme (x: MatrixD): (VectorD, VectorD) = (x.min, x.max)

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Center matrix x to zero mean, column-wise, by subtracting the mean.
 *  @param x     the matrix to center
 *  @param mu_x  the vector of column means of matrix x
 *
def center (x: MatrixD, mu_x: VectorD): MatrixD =
    val x_c = new MatrixD (x.dim, x.dim2)
    for j <- x.indices2 do x_c(?, j) = x(?, j) - mu_x(j)         // subtract column means
    x_c
end center
 */

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Uncenter matrix x_c from zero mean, column-wise, by adding the mean.
 *  @param x_c   the matrix to uncenter
 *  @param mu_x  the vector of column means of matrix x_c
 *
def uncenter (x_c: MatrixD, mu_x: VectorD): MatrixD =
    val x = new MatrixD (x_c.dim, x_c.dim2)
    for j <- x.indices2 do x(?, j) = x_c(?, j) + mu_x(j)         // add column means
    x
end uncenter
 */

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Scale matrix x to the range lb to ub, column-wise: x -> x_s.
 *  @param extremes  the (min_x, max_x) vectors of original matrix x
 *      where min_x  the vector of column minima of matrix x
 *            max_x  the vector of column maxima of matrix x
 *  @param bounds    the desired (lower, upper) bounds
 *-------------------------------------------------------------------------------
 *  @param x         the matrix to scale
 */
def scale (extremes: (VectorD, VectorD), bounds: (Double, Double) = (0, 1)) (x: MatrixD): MatrixD =
    val (lb, ub) = bounds
    val (min_x, max_x) = extremes
    val x_s = new MatrixD (x.dim, x.dim2)
    for j <- x.indices2 do
        if max_x(j) > min_x(j) then                              // has range?
            val scale = (ub - lb) / (max_x(j) - min_x(j))
            x_s(?, j) = (x(?, j) - min_x(j)) * scale + lb        // shift and scale
        else
            x_s(?, j) = x(?, j)                                  // no change
    end for
    x_s
end scale

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Unscale matrix x_s from the range lb to ub, column-wise: x_s -> x.
 *  @param extremes  the (min_x, max_x) vectors of original matrix x
 *      where min_x  the vector of column minima of matrix x
 *            max_x  the vector of column maxima of matrix x
 *  @param bounds    the scaled (lower, upper) bounds
 *-------------------------------------------------------------------------------
 *  @param x_s       the matrix to unscale
 */
def unscale (extremes: (VectorD, VectorD), bounds: (Double, Double) = (0, 1)) (x_s: MatrixD): MatrixD =
    val (lb, ub) = bounds
    val (min_x, max_x) = extremes
    val x = new MatrixD (x_s.dim, x_s.dim2)
    for j <- x.indices2 do
        if max_x(j) > min_x(j) then                              // has range?
            val scale = (ub - lb) / (max_x(j) - min_x(j))
            x(?, j) = (x_s(?, j) - lb) / scale + min_x(j)        // scale and shift
        else
            x(?, j) = x_s(?, j)                                  // no change
    end for
    x
end unscale

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Normalize the matrix x to zero mean and unit standard deviation,
 *  column-wise, by subtracting the mean and dividing by the standard deviation.
 *  Don't normalize if the standard deviation is nearly zero.
 *  @param mu_sig   the mean and standard deviation vectors of original matrix x
 *     where mu_x   the vector of column means of matrix x
 *           sig_x  the vector of column standard deviations of matrix x
 *-------------------------------------------------------------------------------
 *  @param x      the matrix to normalize
 */
def normalize (mu_sig: (VectorD, VectorD)) (x: MatrixD): MatrixD =
    val (mu_x, sig_x) = mu_sig
    val x_n = new MatrixD (x.dim, x.dim2)
    for j <- x.indices2 do                                       // normalize column j
        val sig_xj = sig_x(j)
        x_n(?, j) = if sig_xj =~ 0.0 then x(?, j) else (x(?, j) - mu_x(j)) / sig_xj
    end for
    x_n
end normalize

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Denormalize the matrix x_n from zero mean and unit standard deviation,
 *  column-wise, by multiplying by the standard deviation and adding the mean.
 *  Don't denormalize if the standard deviation is nearly zero.
 *  @param mu_sig   the mean and standard deviation vectors of original matrix x
 *     where mu_x   the vector of column means of matrix x
 *           sig_x  the vector of column standard deviations of matrix x
 *-------------------------------------------------------------------------------
 *  @param x_n    the matrix to denormalize
 */
def denormalize (mu_sig: (VectorD, VectorD)) (x_n: MatrixD): MatrixD =
    val (mu_x, sig_x) = mu_sig
    val x = new MatrixD (x_n.dim, x_n.dim2)
    for j <- x.indices2 do                                       // denormalize column j
        val sig_xj = sig_x(j)
        x(?, j) = if sig_xj =~ 0.0 then x_n(?, j) else x_n(?, j) * sig_xj + mu_x(j)
    end for
    x
end denormalize


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixTransformTest` main function is used to test the `MatrixTransform` object.
 *  It tests centering, scaling and normalization.
 *  > runMain scalation.modeling.matrixTransformTest
 */
@main def matrixTransformTest (): Unit =

    val x = MatrixD ((3, 3), 1, 2, 3,
                             4, 5, 6,
                             7, 8, 9)

    val (min_x, max_x) = (x.min, x.max)
    val (mu_x, sig_x)  = (x.mean, x.stdev)

    println (s"(min_x, max_x) = ($min_x, $max_x)")
    println (s"(mu_x, sig_x) = ($mu_x, $sig_x)")

//  val x_c  = center (x, mu_x)
    val x_s  = scale ((min_x, max_x), (0, 1)) (x)            // e.g., used for sigmoid activation function
    val x_s2 = scale ((min_x, max_x), (-1, 1)) (x)           // e.g., used by tanh activation function
    val x_n  = normalize ((mu_x, sig_x)) (x)                 // e.g., used for unbounded activation function

    println ("x    = " + x)
//  banner ("Center at 0")
//  println ("x_c  = " + x_c)
    banner ("Scale to (0, 1)")
    println ("x_s  = " + x_s)
    banner ("Scale to (-1, 1)")
    println ("x_s2 = " + x_s2)
    banner ("Normalize to (mu = 0, sig = 1)")
    println ("x_n  = " + x_n)

//  assert (uncenter (x_c, mu_x) == x, "uncenter")
    assert (unscale ((min_x, max_x), (0, 1)) (x_s)   == x, "unscale")
    assert (unscale ((min_x, max_x), (-1, 1)) (x_s2) == x, "unscale")
    assert (denormalize ((mu_x, sig_x)) (x_n) == x, "denormalize")

end matrixTransformTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `matrixTransformTest2` main function is used to test the `MatrixTransform` object.
 *  It tests usage of tran and itran functional variables.
 *  > runMain scalation.modeling.matrixTransformTest2
 */
@main def matrixTransformTest2 (): Unit =

    val x = VectorD (1, 2, 3, 4, 5, 6)

    val extremes = extreme (x)
    val bounds   = (0.0, 1.0)
    val mu_sig   = mean_stdev (x)

    println (s"extremes = $extremes")
    println (s"mu_sig   = $mu_sig")

    var tran  = scaleV (extremes, bounds)                   // tranform - partially applied function
    var itran = unscaleV (extremes, bounds)                 // inverse transform

    val x_s = tran (x)
    val x_u = itran (x_s)

    banner ("Test scaling")
    println (s"x_s = $x_s")
    println (s"x_u = $x_u")

    assert (x_u == x, "unscale")

    tran  = normalizeV (mu_sig)                              // tranform - partially applied function
    itran = denormalizeV (mu_sig)                            // inverse transform

    val x_n = tran (x)
    val x_d = itran (x_n)

    banner ("Test normalization")
    println (s"x_n = $x_n")
    println (s"x_d = $x_d")

    assert (x_d == x, "denormalize")

end matrixTransformTest2

