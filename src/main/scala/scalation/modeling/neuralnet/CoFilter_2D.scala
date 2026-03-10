
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Nov  3 16:00:47 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Part: Convolutional Filter (tow dimensional)
 *
 *  @see leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution
 *  @see e2eml.school/convolution_one_d.html
 */

package scalation
package modeling
package neuralnet

import scalation.mathstat._
import scalation.random.RandomMatD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CoFilter_2D` class provides a convolution filter (cofilter) for
 *  taking a weighted average over a window of an input matrix.
 *  @param width  the width of the cofilter
 */
class CoFilter_2D (width: Int = 5):

    private val rmg = RandomMatD (width, width, 2.0)                          // random matrix genertor
    private var mat = rmg.gen                                                 // the filter's matrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the parameters, i.e., the filter's matrix.
     *  @param vec_  the new matrix parameters
     */
    def update (mat_ : MatrixD): Unit = mat = mat_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return/get the filter coefficients (needed for forward and gradient computation)
     */
    def coef: MatrixD = mat

end CoFilter_2D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CoFilter_2D` object provides the convolution and pooling operators.
 *  @see `mathstat.MatrixD` for infix implementations of
 *  conv (*+)   -- `valid` convolution, no reversal (conv_ for reversal)
 *  convs (*~+) -- `same` convolution, with reversal
 *  convf (*++) -- `full` convolution, with reversal
 */
object CoFilter_2D:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'valid' (no padding) convolution of cofilter matrix c and
     *  input matrix x.
     *  Caveat: does not include reversal.
     *  @param c  the cofilter matrix of coefficients
     *  @param x  the input/data matrix
     */
    def conv (c: MatrixD, x: MatrixD): MatrixD = c *+ x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'valid' (no padding) convolution of cofilter matrix c and
     *  input tensor x.
     *  Caveat: does not include reversal.
     *  @param c  the cofilter matrix of coefficients
     *  @param x  the input/data tensor
     */
    def conv (c: MatrixD, x: TensorD): TensorD = x.mmap (c *+ _)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'same' (with padding) convolution of cofilter matrx c and
     *  input matrix x.
     *  Same means that the size of the result is the same as the input (via padding).
     *  @param c  the cofilter matrix of coefficients
     *  @param x  the input/data matrix
     */
    def convs (c: MatrixD, x: MatrixD): MatrixD = c *~+ x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'same' (with padding) convolution of cofilter matrix c and
     *  input tensor x.
     *  Same means that the size of the result is the same as the input (via padding).
     *  @param c  the cofilter matrix of coefficients
     *  @param x  the input/data tensor
     */
    def convs (c: MatrixD, x: TensorD): TensorD = x.mmap (c *~+ _)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'full' convolution of cofilter matrix c and input matrix x.
     *  @param c  the cofilter matrix of coefficients
     *  @param x  the input/data matrix
     */
    def convf (c: MatrixD, x: MatrixD): MatrixD = c *++ x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the 'full' convolution of cofilter c matrix and input tensor x.
     *  @param c  the cofilter matrix of coefficients
     *  @param x  the input/data tensor
     */
    def convf (c: MatrixD, x: TensorD): TensorD = x.mmap (c *++ _)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the max-pooling results over all pooling windows.
     *  @param x  the input/data matrix
     *  @param s  the size (s x s) of the pooling window
     */
    def pool (x: MatrixD, s: Int = 2): MatrixD =
        val p = new MatrixD (x.dim / s, x.dim2 / s)
        for j <- p.indices; k <- p.indices2 do
            val (jj, kk) = (s * j, s * k)
            p(j, k) = x(jj until jj+s, kk until kk+s).mmax
        p
    end pool

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the max-pooling results over all data instances (tensor level).
     *  @param x  the input/data tensor
     *  @param s  the the size of the pooling window
     */
    def pool (x: TensorD, s: Int): TensorD = x.mmap (pool (_, s))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the avg-pooling results over all pooling windows.
     *  @param x  the input/data matrix
     *  @param s  the size (s x s) of the pooling window
     */
    def pool_a (x: MatrixD, s: Int = 2): MatrixD =
        val p = new MatrixD (x.dim / s, x.dim2 / s)
        for j <- p.indices; k <- p.indices2 do
            val (jj, kk) = (s * j, s * k)
            p(j, k) = x(jj until jj+s, kk until kk+s).mmean
        p
    end pool_a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the avg-pooling results over all data instances (tensor level).
     *  @param x  the input/data tensor
     *  @param s  the the size of the pooling window
     */
    def pool_a (x: TensorD, s: Int): TensorD = x.mmap (pool_a (_, s))

end CoFilter_2D

import CoFilter_2D._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `coFilter_2DTest` main function is used to test the `CoFilter_2D` class.
 *  Test using the simple example from CNN_2D section of the ScalaTion textbook.
 *  > runMain scalation.modeling.neuralnet.coFilter_2DTest
 */
@main def coFilter_2DTest (): Unit =

    val x  = MatrixD ((5, 5), 0, 0, 2, 1, 0,
                              0, 0, 0, 1, 2,
                              1, 2, 2, 0, 2,
                              2, 0, 0, 0, 1,
                              2, 2, 2, 0, 1)

    val c  = MatrixD ((2, 2), 1, 1,
                              0, 1)

    val φ = conv (c, x)
    val z = pool (φ, 2)

    println (s"input       x = $x")
    println (s"filter      c = $c")
    println (s"feature map φ = $φ")
    println (s"pooled      z = $z")

end coFilter_2DTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `coFilter_2DTest2` main function is used to test the `CoFilter_2D` class.
 *  Test using the Example Calculation from the 2D CNN section of the ScalaTion textbook.
 *  > runMain scalation.modeling.neuralnet.coFilter_2DTest2
 */
@main def coFilter_2DTest2 (): Unit =

    import ActivationFun.f_reLU

    val x  = MatrixD ((5, 5), 0, 0, 2, 1, 0,
                              0, 0, 0, 1, 2,
                              1, 2, 2, 0, 2,
                              2, 0, 0, 0, 1,
                              2, 2, 2, 0, 1)

    val c  = MatrixD ((2, 2), 1, 1,
                              0, 1)
//  val φ  = conv (c, x)
    val φ  = c *+ x
    val φa = f_reLU.fM (φ)
    val p  = pool (φa, 2)
    val z  = p.flatten

    println (s"input matrix  x = $x")
    println (s"conv. filter  c = $c")
    println (s"feature map   φ = $φ")
    println (s"feature map-a φa = $φa")
    println (s"pooled map    p = $p")
    println (s"hidden vector z = $z")

end coFilter_2DTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `coFilter_2DTest3` main function is used to test the `CoFilter_2D` class's
 *  convolutional operator.
 *  > runMain scalation.modeling.neuralnet.coFilter_2DTest3
 */
@main def coFilter_2DTest3 (): Unit =

    val x  = MatrixD ((5, 5), 0, 0, 2, 1, 0,
                              0, 0, 0, 1, 2,
                              1, 2, 2, 0, 2,
                              2, 0, 0, 0, 1,
                              2, 2, 2, 0, 1)

    val c  = MatrixD ((2, 2), 1, 1,
                              0, 1)

    banner (s"c convolution x")
    println (s"c = $c")
    println (s"x = $x")

    banner ("Full Convolution: convf (c, x)")
    println (s"y = ${convf (c, x)}")
    banner ("Same Convolution: convs (c, x)")
    println (s"y = ${convs (c, x)}")

    banner ("Valid Convolution: conv (c, x)")
    println (s"y = ${conv (c, x)}")

end coFilter_2DTest3

