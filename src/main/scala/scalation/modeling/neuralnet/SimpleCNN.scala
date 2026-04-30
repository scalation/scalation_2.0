
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Nov  4 13:44:23 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Simple Convolutional Neural Networks (CNNs) Using Gradient Descent Optimization
 *           Tests Gradient Descent (GD) and Incremental Gradient Descent (IGD)
 *           Simplified version of `CNN_1D` for illustration/learning, not production
 *
 *  @note    the symbol ƒ indicates the derivative of function f, i.e., ƒ = f'
 *  @note    in the code below, the (improper) hidden layer is just the flattening layer
 *  @note    adding a proper hidden layer (having a weight matrix before it, a bias vector
 *           and an activation function, would likely increase the accuracy of the model
 */

package scalation
package modeling
package neuralnet

import scalation.mathstat.{VectorD, MatrixD, Plot}
import scalation.modeling.ActivationFun.{f_reLU, reLU_, sigmoid_}    // reLU, sigmoid activation functions
import scalation.modeling.forecasting.MakeMatrix4TS

import CoFilter_1D.{convs, convs_back, pool}                         // convolutional operators (s => same/padding)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleCNN` object contains a simple dataset for testing Gradient Descent (GD)
 *  and Incremental Gradient Descent (IGD) optimization algorithms.
 *  @see https://nowak.ece.wisc.edu/MFML.pdf
 */
object SimpleCNN:

    /** Sequential Data, e.g., time series or acoustic signal
     */
    val y  = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3, 4, 2, 7, 6, 3, 1, 5, 7, 9, 8)
    val yy = MatrixD (y).ᵀ

    /** Build an input/predictor matrix (default number of lags p = 3)
     */
    //  MakeMatrix4TS.hp("p") = 3
    //  val x = ARY.buildMatrix (y, MakeMatrix4TS.hp).drop (1)       // no intercept
    val x = MakeMatrix4TS.makeMatrix4L (y, 3)

end SimpleCNN


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleCNN0` main function illustrates convolution, activation and pooling.
 *  > runMain scalation.modeling.neuralnet.simpleCNN0
 */
@main def simpleCNN0 (): Unit =

    val y = VectorD (1, 3, 4, 2, 5, 7, 6, 8)
    val c = VectorD (1, -2, 1)                                // convolutional parameter/weight vector
    val α = 0.1                                               // convolutional bias scalar

    var v, u, z: VectorD = null
    v  = convs (c, y) + α                                     // convolutional pre-activation vector via 'same' convolution
    u  = reLU_ (v)                                            // convolutional vector from f0 = sigmoid activation
    z  = pool (u)
    println (s"""
    y = $y
    v = $v
    u = $u
    z = $z
    """)

end simpleCNN0

import SimpleCNN.{x, y, yy}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleCNN1` main function illustrates the use of Incremental Gradient Descent (IGD)
 *  to optimize the weights/parameters of a simple (3-layer (1 hidden) CNN.
 *
 *  Prediction Equation: z = f0(c *_c x + α)
 *                       ŷ = f1(B^T z + β)
 *
 *  where x is an input vector, z is the hidden layer vector, ŷ is a predicted output vector, f0, f1
 *  are the activation functions, c is the shared weight vector and B is the parameter matrix,
 *  and α and β are the bias vectors.
 *  @note: Stochastic Gradient Descent (SGD) adds stochastic selection to IGD.  In practice,
 *  mini-batches of size 32, 64, or 128 are commonly used.
 *  > runMain scalation.modeling.neuralnet.simpleCNN1
 */
@main def simpleCNN1 (): Unit =

    import scalation.mathstat.MatrixDOps.⊗                      // outer product of two vectors: v1 ⊗ v2 = v1 v1.ᵀ
                                                                // matrix where m_ij = v1_i * v2_j
    import scalation.mathstat.VectorDOps._

    val sst = (y - y.mean).normSq                               // sum of squares total, per column
    val c   = VectorD (0.1, 0.2, 0.1)                           // convolutional parameter/weight vector
    var α   = 0.1                                               // convolutional bias scalar
    val b   = MatrixD ((3, 1), 0.1, 0.1, 0.1)                   // parameter/weight matrix: hidden -> output
                                                                // initial weights/parameters (random in practice)
    val β   = VectorD (1.0)                                     // output bias vector

    println (s"sst = $sst, x = $x, y = $y")

    val η = 0.005                                               // sigmoid learning rate (to be tuned)
    var u, z, v, ŷ, ε, ƒ1, δ1, ƒ0, δ0: VectorD = null
    var g1: MatrixD = null
    val yp = new MatrixD (y.dim, 1)                             // save each prediction in yp

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")
        val sse = VectorD (0.0)
        for i <- x.indices do
            val (xi, yi) = (x(i), VectorD (y(i)))               // randomize i for Stochastic Gradient Descent (SGD)

            // forward prop: input -> hidden -> output
            v  = convs (c, xi) + α                              // convolutional pre-activation vector via 'same' convolution
            z  = sigmoid_ (v)                                   // convolutional vector from f0 = sigmoid activation
            u  = b.ᵀ * z + β                                    // output pre-activation vector
            ŷ  = u                                              // output/prediction vector from f1 = id activation
            ε  = yi - ŷ                                         // error vector

            // backward prop: input <- hidden <- output
            ƒ1 = VectorD.one (ŷ.dim)                            // derivative (f1') for id
            δ1 = -ε * ƒ1                                        // delta correction vector via elementwise product
            g1 = z ⊗ δ1                                         // gradients vectors (matrix)
            ƒ0 = z * (1.0 - z)                                  // derivative (f0') for sigmoid
            δ0 = (b * δ1) * ƒ0                                  // delta correction vector

            // parameter updates
            b -= g1 * η                                         // update output parameter/weight matrix
            β -= δ1 * η                                         // update output bias vector
            c -= convs (xi, δ0) * η                             // update convolutional weight vector
            α -= δ0.sum * η                                     // update convolutional bias scalar

            yp(i) = ŷ                                           // save i-th prediction
            sse  += ε.normSq                                    // sum of squared errors
        end for

        val r2 = -(sse / sst - 1.0)                             // R^2 (recoded to avoid Ops clash)

        println (s"""
        u  = $u
        z  = $z
        v  = $v
        ŷ  = $ŷ
        ε  = $ε
        ƒ1 = $ƒ1
        δ1 = $δ1
        g1 = $g1
        ƒ0 = $ƒ0
        δ0 = $δ0
        b  = $b
        β  = $β
        c  = $c
        α  = $α
        sse= $sse
        r2 = $r2
        smape = ${FitM.smapeF (y.drop(1), yp(?, 0).drop(1))}
        """)
    end for

    new Plot (null, y, yp(?, 0), "IGD for CNN y", lines = true)

    println ("Compare smape scores with Regression")
    val mod = new Regression (x, y)
    val ypr = mod.trainNtest ()()._1
    println (s"smape = ${FitM.smapeF (y.drop(1), ypr.drop(1))}")   // can't forecast first value

end simpleCNN1


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleCNN2` main function illustrates the use of Incremental Gradient Descent (IGD)
 *  to optimize the weights/parameters of a simple (3-layer (1 hidden) CNN.
 *
 *  Prediction Equation: z = f0(c *_c x + α)
 *                       ŷ = f1(B^T z + β)
 *
 *  where x is an input vector, z is the hidden layer vector, ŷ is a predicted output vector, f0, f1
 *  are the activation functions, c is the shared weight vector and B is the parameter matrix,
 *  and α and β are the bias vectors.
 *  @note: Stochastic Gradient Descent (SGD) adds stochastic selection to IGD.  In practice,
 *  mini-batches of size 32, 64, or 128 are commonly used.
 *  > runMain scalation.modeling.neuralnet.simpleCNN2
 */
@main def simpleCNN2 (): Unit =

    import scalation.mathstat.MatrixDOps.⊗                      // outer product of two vectors: v1 ⊗ v2 = v1 v1.ᵀ
                                                                // matrix where m_ij = v1_i * v2_j
    val sst = (y - y.mean).normSq                               // sum of squares total, per column
    val c   = VectorD (0.1, 0.2, 0.1)                           // convolutional parameter/weight vector
    val b   = MatrixD ((3, 1), 0.1, 0.1, 0.1)                   // parameter/weight matrix: hidden -> output
    var α   = 0.1                                               // convolutional bias scalar
                                                                // initial weights/parameters (random in practice)
    val β   = VectorD (1.0)                                     // output bias vector

    val η = 0.009                                               // reLU learning rate (to be tuned)
    var u, z, v, ŷ, ε, ƒ1, δ1, ƒ0, δ0: VectorD = null
    var g1: MatrixD = null
    val yp = new MatrixD (y.dim, 1)                             // save each prediction in yp

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")
        val sse = VectorD (0.0)
        for i <- x.indices do
            val (xi, yi) = (x(i), VectorD (y(i)))               // randomize i for Stochastic Gradient Descent (SGD)

            // forward prop: input -> hidden -> output
            v  = convs (c, xi) + α                              // convolutional pre-activation vector via 'same' convolution
            z  = reLU_ (v)                                      // convolutional vector from f0 = reLU activation
            u  = b.ᵀ * z + β                                    // output pre-activation vector
            ŷ  = u                                              // output/prediction vector from f1 = id activation
            ε  = yi - ŷ                                         // error vector

            // backward prop: input <- hidden <- output
            ƒ1 = VectorD.one (ŷ.dim)                            // derivative (f1') for id
            δ1 = -ε * ƒ1                                        // delta correction vector via elementwise product
            g1 = z ⊗ δ1                                         // gradients vectors (matrix)
            ƒ0 = z.map (t => if t > 0.0 then 1.0 else 0.0)      // derivative (f0') for reLU
            δ0 = (b * δ1) * ƒ0                                  // delta correction vector

            // parameter updates
            b -= g1 * η                                         // update output parameter/weight matrix
            β -= δ1 * η                                         // update output bias vector
            c -= convs (xi, δ0) * η                             // update convolutional weight vector
            α -= δ0.sum * η                                     // update convolutional bias scalar

            yp(i) = ŷ                                           // save i-th prediction
            sse  += ε.normSq                                    // sum of squared errors
        end for

        val r2 = -(sse / sst - 1.0)                             // R^2 (recoded to avoid Ops clash)

        println (s"""
        u  = $u
        z  = $z
        v  = $v
        ŷ  = $ŷ
        ε  = $ε
        ƒ1 = $ƒ1
        δ1 = $δ1
        g1 = $g1
        ƒ0 = $ƒ0
        δ0 = $δ0
        b  = $b
        β  = $β
        c  = $c
        α  = $α
        sse= $sse
        r2 = $r2
        smape = ${FitM.smapeF (y.drop(1), yp(?, 0).drop(1))}
        """)
    end for

    new Plot (null, y, yp(?, 0), "IGD for CNN y", lines = true)

    println ("Compare smape scores with Regression")
    val mod = new Regression (x, y)
    val ypr = mod.trainNtest ()()._1
    println (s"smape = ${FitM.smapeF (y.drop(1), ypr.drop(1))}")   // can't forecast first value

end simpleCNN2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleCNN3` main function illustrates the use of Gradient Descent (GD)
 *  to optimize the weights/parameters of a simple (3-layer (1 hidden) CNN.
 *
 *  Prediction Equation: z = f0(c *_c X + α)
 *                       ŷ = f1(Z B + β)
 *
 *  where x is the input matrix, z is the hidden layer matrix, ŷ is a predicted output matrix, f0, f1
 *  are the activation functions, c is the shared weight vector and B is the parameter matrix,
 *  and α and β are the bias vectors.
 *  > runMain scalation.modeling.neuralnet.simpleCNN3
 */
@main def simpleCNN3 (): Unit =

    val sst = (y - y.mean).normSq                               // sum of squares total, per column
    val c   = VectorD (0.1, 0.2, 0.1)                           // convolutional parameter/weight vector
    val b   = MatrixD ((3, 1), 0.1, 0.1, 0.1)                   // parameter/weight matrix: hidden -> output
    val α   = VectorD (0.1)                                     // convolutional bias scalar
                                                                // initial weights/parameters (random in practice)
    val β   = VectorD (1.0)                                     // output bias vector

    println (s"sst = $sst, x = $x, y = $y")

    val η   = 0.001                                             // reLU learning rate (to be tuned)
    val n_c = c.dim                                             // size of the convolutional filter
    var u, z, v, ŷ, ε, δ1, g1, ƒ0, δ0: MatrixD = null

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")

        // forward prop: input -> hidden -> output
        v  = convs (c, x) + α(0)                                // convolutional pre-activation vector via 'same' convolution
        z  = f_reLU.fM (v)                                      // convolutional vector from f0 = reLU activation
        u  = z * b + β                                          // output pre-activation vector
        ŷ  = u                                                  // output/prediction vector from f1 = id activation
        ε  = yy - ŷ                                             // error vector

        // backward prop: input <- hidden <- output
        δ1 = -ε                                                 // delta correction vector via elementwise product with 1
        g1 = z.ᵀ * δ1                                           // gradients vectors (matrix)
        ƒ0 = f_reLU.dM (z)                                      // derivative (f0') for reLU
        δ0 = (δ1 * b.ᵀ) ⊙ ƒ0                                    // delta correction vector

        // parameter updates
        b -= g1 * η                                             // update output parameter/weight matrix
        β -= δ1.mean * η                                        // update output bias vector
        c -= convs_back (x, δ0, n_c) * η                        // update convolutional weight vector
        α -= δ0.mean * η                                        // update convolutional bias scalar

        val sse = ε.normSq                                      // sum of squared errors
        val r2  = -(sse / sst - 1.0)                            // R^2 (recoded to avoid Ops clash)

        println (s"""
        u  = $u
        z  = $z
        v  = $v
        ŷ  = $ŷ
        ε  = $ε
        δ1 = $δ1
        g1 = $g1
        ƒ0 = $ƒ0
        δ0 = $δ0
        b  = $b
        β  = $β
        c  = $c
        α  = $α
        sse= $sse
        r2 = $r2
        smape = ${FitM.smapeF (yy(?, 0).drop(1), ŷ(?, 0).drop(1))}
        """)
    end for

    new Plot (null, yy(?, 0), ŷ(?, 0), "GD for CNN y_0", lines = true)

    println ("Compare smape scores with Regression")
    val mod = new Regression (x, y)
    val ypr = mod.trainNtest ()()._1
    println (s"smape = ${FitM.smapeF (y.drop(1), ypr.drop(1))}")   // can't forecast first value

end simpleCNN3

