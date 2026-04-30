
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Mar  9 19:19:53 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Random Variate Matrix (RVM) Generators
 */

package scalation
package random

import scalation.mathstat._

import RandomSeeds.N_STREAMS

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariateMat` abstract class serves as a base class for all the Random
 *  Variate Matrix (RVM) generators. They use one of the Random Number Generators
 *  (RNG's) from Random.scala to generate numbers following their particular
 *  multivariate distribution.
 *-----------------------------------------------------------------------------
 *  @param stream  the random number stream (0 until N_STREAMS)
 */
abstract class VariateMat (stream: Int = 0):

    protected val flaw = flawf ("VariateMat")

    /** Random number stream selected by the stream number (can't be beyond last stream)
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
    /** Compute the matrix mean for the particular distribution.
     */
    def mean: MatrixD

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability function (pf):
     *  The probability density function (pdf) for continuous RVM's or
     *  the probability mass function (pmf) for discrete RVM's.
     *  @param z  the mass point/matrix whose probability is sought
     */
    def pf (z: MatrixD): Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random double matrix for the particular distribution.
     */
    def gen: MatrixD

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random integer matrix for the particular distribution.
     *  It is only valid for discrete random variates.
     */
    def igen: MatrixD

end VariateMat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariateMat` companion object provides a method to add correlation to
 *  a matrix.
 */
object VariateMat:

    def corTransform (x: MatrixD, cov: MatrixD): MatrixD =
        val fac1 = new Fac_Cholesky (cov).factor1 ()           // LL^t = Cov
        fac1 * x                                               // LX
    end corTransform

end VariateMat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NormalMat` class generates Normal (Gaussian) random variate matrices according
 *  to the Normal distribution with scalar mean 'mu' and variance 'sig2'.
 *  This continuous RVM models multiple instances of normally distributed multidimensional
 *  data and treats the variables as identical and independent.
 *  @param dim     the number of rows in the matrix
 *  @param dim2    the number of columns in the matrix
 *  @param mu      the mean
 *  @param sig2    the variance (stdev^2)
 *  @param stream  the random number stream
 */
case class NormalMat (dim: Int, dim2: Int, mu: Double, sig2: Double, stream: Int = 0)
     extends VariateMat (stream):

    private val normal = Normal (mu, sig2, stream)           // generator for standard normals

    def mean: MatrixD = MatrixD.fill (dim, dim2, mu)         // m = dim rows and n = dim2 columns

    def pf (z: MatrixD): Double =
        var d = 1.0                                          // density f(z)
        for i <- 0 until dim; j <- 0 until dim2 do d *= normal.pf (z(i, j))
        d
    end pf

    def gen: MatrixD =
        val y = new MatrixD (dim, dim2)                       // m = dim rows and n = dim2 columns
        for i <- 0 until dim; j <- 0 until dim2 do y(i, j) = normal.gen
        y
    end gen

    def igen: MatrixD = gen.toInt

end NormalMat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomMatD` class generates a random matrix of doubles.
 *  @param dim      the number of rows in the matrix
 *  @param dim2     the number of columns in the matrix
 *  @param max      generate integers in the range 0 (inclusive) to max (inclusive) <-- note max arg first
 *  @param min      generate integers in the range 0 (inclusive) to max (inclusive)
 *  @param density  sparsity basis = 1 - density
 *  @param stream   the random number stream
 */
case class RandomMatD (dim: Int = 5, dim2: Int = 10, max: Double = 20.0, min: Double = 0.0,
                       density: Double = 1.0, stream: Int = 0)
     extends VariateMat (stream):

    private val mu   = (min + max) / 2.0                                      // mean
    private val rvec = RandomVecD (dim2, max, min, density, stream = stream)  // random vector generator
    
    def mean: MatrixD = MatrixD.fill (dim, dim2, mu)

    def pf (z: MatrixD): Double = 1.0 / (max - min) ~^ (dim + dim2)

    def gen: MatrixD = MatrixD (for _ <- 0 until dim yield rvec.gen)

    def igen: MatrixD = gen.toInt

end RandomMatD


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `variateMatTest` main function is used to test the Random Variate Matrix (RVM)
 *  generators from the classes derived from `VariateMat`.
 *  > runMain scalation.random.variateMatTest
 */
@main def variateMatTest (): Unit =

     import VariateMat.corTransform

     var rvm: VariateMat = null                                // variate matrix

     banner ("Test: NormalMat random matrix generation")
     rvm = NormalMat (4, 5, 0.0, 0.01)                         // random normal matrix generator
     println ("mean = " + rvm.mean)                            // with mean 0 and variance 0.01
     cfor (0, 10) { _ => println (rvm.gen) }

     banner ("Test: RandomMatD random matrix generation")
     rvm = RandomMatD (2, 100)                                 // random matrix generator
     println ("mean = " + rvm.mean)
     cfor (0, 10) { _ => println (rvm.gen) }

     val cor = MatrixD ((2, 2), 1.0, 0.9,                      // covariance/correlation matrix
                                0.9, 1.0)

     banner ("Test: RandomMatD random correlated matrix generation")
     val x = rvm.gen
     println ("x = " + x)
     val z = corTransform (x, cor)
     println ("z = " + z)
/* FIX
     val xx = x.toInt
     val zz = z.toInt

     Plot (xx(0), xx(1), null, "x uncorrelated")
     Plot (zz(0), zz(1), null, "z correlated")
*/

end variateMatTest

