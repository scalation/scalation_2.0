
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Mar 27 14:40:57 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Part: Convolutional Filter (one dimensional)
 *
 *  @see leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution
 *  @see e2eml.school/convolution_one_d.html
 */

package scalation
package modeling
package neuralnet

import scalation.mathstat._
import scalation.random.RandomVecD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CoFilter_1D` class provides a convolution filter (cofilter) for
 *  taking a weighted average over a window of an input vector.
 *  @param width  the width of the cofilter
 */
class CoFilter_1D (width: Int = 5):

    private val rvg = RandomVecD (width, 2.0)                                 // random vector genertor
    private var vec = rvg.gen                                                 // the filter's vector

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the parameters, i.e., the filter's vector.
     *  @param vec_  the new vector parameters
     */
    def update (vec_ : VectorD): Unit = vec = vec_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return/get the filter coefficients (needed for forward and gradient computation)
     */
    def coef: VectorD = vec

end CoFilter_1D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CoFilter_1D` object provides the convolution and pooling operators.
 *  @see `mathstat.VectorD` for infix implementations of
 *  conv (*+)   -- `valid` convolution, no reversal (conv_ for reversal)
 *  convs (*~+) -- `same`  convolution, with reversal
 *  convf (*++) -- `full`  convolution, with reversal
 */
object CoFilter_1D:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'valid' (no padding) convolution of cofilter vector c and
     *  input vector x.
     *  Caveat: does not include reversal.
     *  @param c  the cofilter vector of coefficients
     *  @param x  the input/data vector
     */
    def conv (c: VectorD, x: VectorD): VectorD = c *+ x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'valid' (no padding) convolution of cofilter vector c and
     *  input matrix x.
     *  Caveat: does not include reversal.
     *  @param c  the cofilter vector of coefficients
     *  @param x  the input/data matrix
     */
    def conv (c: VectorD, x: MatrixD): MatrixD = x.mmap (c *+ _)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'same' (with padding) convolution of cofilter vector c and
     *  input vector x.
     *  Same means that the size of the result is the same as the input.
     *  @param c  the cofilter vector of coefficients
     *  @param x  the input/data vector
     */
    def convs (c: VectorD, x: VectorD): VectorD = c *~+ x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'same' (with padding) convolution of cofilter vector c and
     *  input matrix x.
     *  @param c  the cofilter vector of coefficients
     *  @param x  the input/data matrix
     */
    def convs (c: VectorD, x: MatrixD): MatrixD = x.mmap (c *~+ _)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'full' convolution of cofilter vector c and input vector x.
     *  Same means that the size of the result is the same as the input.
     *  @param c  the cofilter vector of coefficients
     *  @param x  the input/data vector
     */
    def convf (c: VectorD, x: VectorD): VectorD = c *++ x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'full' convolution of cofilter vector c and input matrix x.
     *  @param c  the cofilter vector of coefficients
     *  @param x  the input/data matrix
     */
    def convf (c: VectorD, x: MatrixD): MatrixD = x.mmap (c *++ _)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the max-pooling results over all pooling windows.
     *  @param x  the input/data vector
     *  @param s  the size of the pooling window
     */
    def pool (x: VectorD, s: Int = 2): VectorD =
        val p = new VectorD (x.dim / s)
        for j <- p.indices do
            val jj = s * j
            p(j)   = x(jj until jj+s).max
        p
    end pool

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the max-pooling results over all data instances (matrix level).
     *  @param x  the input/data matrix
     *  @param s  the the size of the pooling window
     */
    def pool (x: MatrixD, s: Int): MatrixD = x.mmap (pool (_, s))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the avg-pooling results over all pooling windows.
     *  @param x  the input/data vector
     *  @param s  the size of the pooling window
     */
    def pool_a (x: VectorD, s: Int = 2): VectorD =
        val p = new VectorD (x.dim / s)
        for j <- p.indices do
            val jj = s * j
            p(j)   = x(jj until jj+s).mean
        p
    end pool_a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the avg-pooling results over all data instances (matrix level).
     *  @param x  the input/data matrix
     *  @param s  the the size of the pooling window
     */
    def pool_a (x: MatrixD, s: Int): MatrixD = x.mmap (pool_a (_, s))

end CoFilter_1D

import CoFilter_1D._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `coFilter_1DTest` main function is used to test the `CoFilter_1D` class.
 *  Test using the simple example from CNN_1D section of the ScalaTion textbook.
 *  > runMain scalation.modeling.neuralnet.coFilter_1DTest
 */
@main def coFilter_1DTest (): Unit =

    val x = MatrixD ((2, 5), 1, 2, 3, 4, 5,
                             6, 7, 8, 9, 10)
    val c = VectorD (0.5, 1, 0.5)
    val φ = conv (c, x)
    val z = pool (φ, 3)

    println (s"input       x = $x")
    println (s"filter      c = $c")
    println (s"feature map φ = $φ")
    println (s"pooled      z = $z")

end coFilter_1DTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `coFilter_1DTest2` main function is used to test the `CoFilter_1D` class.
 *  Test using the Example Calculation from the 1D CNN section of the ScalaTion textbook.
 *  > runMain scalation.modeling.neuralnet.coFilter_1DTest2
 */
@main def coFilter_1DTest2 (): Unit =

    import ActivationFun.reLU_

    val x  = VectorD (-3, -2, -1, 0, 1, 2, 3, 4)
    val c  = VectorD (0.5, 1, 0.5)

//  val φ  = c conv x
    val φ  = c *+ x
    val φa = reLU_ (φ)
    val z  = pool (φa, 2)

    println (s"input vector  x  = $x")
    println (s"conv. filter  c  = $c")
    println (s"feature map   φ  = $φ")
    println (s"feature map-a φa = $φa")
    println (s"hidden vector z  = $z")

end coFilter_1DTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `coFilter_1DTest3` main function is used to test the `CoFilter_1D` class's
 *  convolutional operator.
 *  > runMain scalation.modeling.neuralnet.coFilter_1DTest3
 */
@main def coFilter_1DTest3 (): Unit =

    val c = VectorD (1, 2, 3, 4, 5)
    val x = VectorD (1, 2, 3, 4, 5, 6, 7)

    banner (s"c convolution x")
    println (s"c = $c")
    println (s"x = $x")

    banner ("Same Convolution: convs (c, x)")
    println (s"y = ${convs (c, x)}")
    banner ("Valid Convolution: c.reverse *+ x")
    println (s"y = ${c.reverse *+ x}")

    banner ("Valid Convolution: c *+ x")       // without expected reversal
    println (s"y = ${c *+ x}")

end coFilter_1DTest3

