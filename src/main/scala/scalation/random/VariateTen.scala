
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Sep  6 16:52:21 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Random Variate Tensor (RVT) Generators
 */

package scalation
package random

import scalation.mathstat._

import RandomSeeds.N_STREAMS

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariateTen` abstract class serves as a base class for all the Random
 *  Variate Tensor (RVT) generators. They use one of the Random Number Generators
 *  (RNG's) from Random.scala to generate numbers following their particular
 *  multivariate distribution.
 *-----------------------------------------------------------------------------
 *  @param stream  the random number stream (0 until N_STREAMS)
 */
abstract class VariateTen (stream: Int = 0):

    protected val flaw = flawf ("VariateTen")

    /** Random number stream selected by the stream number (can't be betond last stream)
     */
    protected val r = Random (stream % N_STREAMS)

    /** Indicates whether the distribution is discrete or continuous (default)
     */
    protected var _discrete = false

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the distribution is discrete or continuous.
     */
    def discrete: Boolean = _discrete

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the tensor mean for the particular distribution.
     */
    def mean: TensorD

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability function (pf):
     *  The probability density function (pdf) for continuous RVT's or
     *  the probability mass function (pmf) for discrete RVT's.
     *  @param z  the mass point/tensor whose probability is sought
     */
    def pf (z: TensorD): Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random double tensor for the particular distribution.
     */
    def gen: TensorD

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random integer tensor for the particular distribution.
     *  It is only valid for discrete random variates.
     */
    def igen: TensorD

end VariateTen


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariateTen` companion object provides a method to add correlation to
 *  a tensor.
 *  FIX: upgrade from MatrixD to TensorD level
 *
object VariateTen:

    def corTransform (x: TensorD, cov: TensorD): TensorD =
        val fac1 = new Fac_Cholesky (cov).factor1 ()           // LL^t = Cov
        fac1 * x                                               // LX
    end corTransform

end VariateTen
 */


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NormalTen` class generates Normal (Gaussian) random variate matrices according
 *  to the Normal distribution with scalar mean 'mu' and variance 'sig2'.
 *  This continuous RVT models multiple instances of normally distributed multidimensional
 *  data and treats the variables as identical and independent.
 *  @param dim     the number of rows in the tensor
 *  @param dim2    the number of columns in the tensor
 *  @param dim3    the number of sheets in the tensor
 *  @param mu      the mean
 *  @param sig2    the variance (stdev^2)
 *  @param stream  the random number stream
 */
case class NormalTen (dim: Int, dim2: Int, dim3: Int, mu: Double, sig2: Double, stream: Int = 0)
     extends VariateTen (stream):

    private val normal = Normal (mu, sig2, stream)           // generator for standard normals

    def mean: TensorD = TensorD.fill (dim, dim2, dim3, mu)   // dim rows, dim2 columns, and dim3 sheets

    def pf (z: TensorD): Double =
        var d = 1.0                                          // density f(z)
        for i <- 0 until dim; j <- 0 until dim2; k <- 0 until dim3 do d *= normal.pf (z(i, j, k))
        d
    end pf

    def gen: TensorD =
        val y = new TensorD (dim, dim2, dim3)
        for i <- 0 until dim; j <- 0 until dim2; k <- 0 until dim3 do y(i, j, k) = normal.gen
        y
    end gen

    def igen: TensorD = gen.toInt

end NormalTen


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomTenD` class generates a random tensor of doubles.
 *  @param dim      the number of rows in the tensor
 *  @param dim2     the number of columns in the tensor
 *  @param dim3     the number of sheets in the tensor
 *  @param max      generate integers in the range 0 (inclusive) to max (inclusive)
 *  @param min      generate integers in the range 0 (inclusive) to max (inclusive)
 *  @param density  sparsity basis = 1 - density
 *  @param stream   the random number stream
 */
case class RandomTenD (dim: Int = 5, dim2: Int = 4, dim3: Int = 3, max: Double = 20.0, min: Double = 0.0,
                       density: Double = 1.0, stream: Int = 0)
     extends VariateTen (stream):

    private val mu   = (min + max) / 2.0                                            // mean
    private val rmat = RandomMatD (dim2, dim3, max, min, density, stream = stream)  // random matrix generator
    
    def mean: TensorD = TensorD.fill (dim, dim2, dim3, mu)

    def pf (z: TensorD): Double = 1.0 / (max - min) ~^ (dim + dim2 + dim3)

    def gen: TensorD = TensorD (for _ <- 0 until dim yield rmat.gen)

    def igen: TensorD = gen.toInt

end RandomTenD


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `variateTenTest` main function is used to test the Random Variate Matrix (RVT)
 *  generators from the classes derived from `VariateTen`.
 *  > runMain scalation.random.variateTenTest
 */
@main def variateTenTest (): Unit =

     var rvt: VariateTen = null                                // variate tensor

     banner ("Test: NormalTen random tensor generation")
     rvt = NormalTen (5, 4, 3, 0.0, 0.01)                      // random normal tensor generator
     println ("mean = " + rvt.mean)                            // with mean 0 and variance 0.01
     cfor (0, 10) { _ => println (rvt.gen) }

     banner ("Test: RandomTenD random tensor generation")
     rvt = RandomTenD ()                                       // random tensor generator
     println ("mean = " + rvt.mean)
     cfor (0, 10) { _ => println (rvt.gen) }

/* FIX
     import VariateTen.corTransform
     banner ("Test: RandomTenD random correlated tensor generation")
     val cor = TensorD ((2, 2), 1.0, 0.9,                      // covariance/correlation tensor
                                0.9, 1.0)
     val x = rvt.gen
     println ("x = " + x)
     val z = corTransform (x, cor)
     println ("z = " + z)
     val xx = x.toInt
     val zz = z.toInt

     Plot (xx(0), xx(1), null, "x uncorrelated")
     Plot (zz(0), zz(1), null, "z correlated")
*/

end variateTenTest

