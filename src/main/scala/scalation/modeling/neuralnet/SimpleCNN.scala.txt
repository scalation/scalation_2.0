
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Nov  4 13:44:23 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Simple Convolutional Neural Networks (CNNs) Using Gradient Descent Optimization
 *           Tests both Gradient Descent (GD) and Incremental Gradient Descent (IGD)
 *           Simplified version of `CNN_1D` for illustration/learning, not production
 *
 *  @note    the symbol ƒ indicates the derivative of function f, i.e., ƒ = f'
 */

package scalation.modeling.neuralnet

import scalation.?                                              // wildcard: xy(?, 3) gives column 3
import scalation.mathstat.{VectorD, MatrixD, Plot}
//import scalation.mathstat.VectorDOps._
import scalation.modeling.ActivationFun.{sigmoid_}              // sigmoid activation functions
import scalation.modeling.ActivationFun.{f_reLU, reLU_}         // reLU activation functions
import scalation.modeling.forecasting.{ARY, MakeMatrix4TS}

import CoFilter_1D.{conv, convs}                                // convolutional operators (s => same/padding)

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleCNN` object contains a simple dataset for testing Gradient Descent (GD)
 *  and Incremental Gradient Descent (IGD) optimization algorithms.
 *  @see https://nowak.ece.wisc.edu/MFML.pdf
 */
object SimpleCNN:

    /** Sequential Data, e.g., time series or acoustic signal
     */
    val yy = VectorD (1, 3, 4, 2, 5, 7, 9, 8, 6, 3, 4, 2, 7, 6, 3, 1, 5, 7, 9, 8)

    /** Build an input/predictor matrix (default number of lags p = 3)
     */
//  MakeMatrix4TS.hp("p") = 3
    val xx = ARY.buildMatrix (yy, MakeMatrix4TS.hp)

end SimpleCNN

import SimpleCNN.{yy, xx}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleCNN1` main function illustrates the use of Gradient Descent (GD) to optimize
 *  the weights/parameters of a simple neural network (3-layer (1 hidden) Neural Network).
 *  Computations done at the matrix level: X -> Z -> Y.  R^2 = .299, .432 (requires more epochs)
 *  FIX - finish implementation
 *  > runMain scalation.modeling.neuralnet.simpleCNN1
 */
@main def simpleCNN1 (): Unit =

    import scalation.mathstat.MatrixDOps._                      // may clash with VectorDOps

    val (x, y) = (xx, MatrixD.fromVector (yy))                  // input matrix, output/response matrix
    val sst    = (y - y.mean).normSq                            // sum of squares total, per column
    val a      = MatrixD ((2, 3), 0.1, 0.1, 0.1,                // parameter/weight matrix: input -> hidden
                                  0.1, 0.1, 0.1)
    val α      = VectorD (0.1, 0.1, 0.1)                        // hidden layer bias vector
    val b      = MatrixD ((3, 2), 0.1, 0.1,                     // parameter/weight matrix: hidden -> output
                                  0.2, 0.1,
                                  0.1, 0.1)                     // initial weights/parameters (random in practice)
    val β      = VectorD (0.1, 0.1)                             // output layer bias vector

    val η = 10.0                                                // learning rate (to be tuned)
    var u, z, v, ŷ, ε, ƒ1, δ1, g1, ƒ0, δ0, g0: MatrixD = null

// try f0 = reLU, f1 = id

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")

        // forward prop: input -> hidden
        u  = x * a + α                                          // hidden pre-activation matrix
        z  = f_reLU.fM (u)                                      // hidden matrix from f0 activation

        // forward prop: hidden -> output
        v  = z * b + β                                          // output pre-activation matrix
        ŷ  = f_reLU.fM (v)                                      // output/prediction matrix from f1 activation
        ε  = y - ŷ                                              // error matrix

        // backward prop: hidden <- output
        ƒ1 = ŷ ⊙ (1.0 - ŷ)                                      // derivative (f1') for sigmoid
        δ1 = -ε ⊙ ƒ1                                            // delta correction matrix via Hadamard product
        g1 = z.ᵀ * δ1                                           // transposed Jacobian matrix (gradients)

        // backward prop: input <- hidden
        ƒ0 = z ⊙ (1.0 - z)                                      // derivative (f0') for sigmoid
        δ0 = (δ1 * b.ᵀ) ⊙ ƒ0                                    // delta correction matrix
        g0 = x.ᵀ * δ0                                           // transposed Jacobian matrix (gradients)

        // parameter updates
        b -= g1 * η                                             // update output parameter/weight matrix
        β -= δ1.mean * η                                        // update output bias vector
        a -= g0 * η                                             // update hidden parameter/weight matrix
        α -= δ0.mean * η                                        // update hidden bias vector
        val sse = ε.normSq                                      // sum of squared errors
        val r2  = -(sse / sst - 1.0)                            // R^2 (recoded to avoid Ops clash)

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
        g0 = $g0
        b  = $b
        β  = $β
        a  = $a
        α  = $α
        r2 = $r2
        """)
    end for

    new Plot (null, y(?, 0), ŷ(?, 0), "GD for Three-layer Neural Net y_0", lines = true)
    new Plot (null, y(?, 1), ŷ(?, 1), "GD for Three-layer Neural Net y_1", lines = true)

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
 *  Computations done at the vector level, x -> z -> y.  R^2 = .389 vs. .409 for Regression
 *  > runMain scalation.modeling.neuralnet.simpleCNN2
 */
@main def simpleCNN2 (): Unit =

    import scalation.mathstat.MatrixDOps.⊗                      // outer product of two vectors: v1 ⊗ v2 = v1 v1.ᵀ
                                                                // matrix where m_ij = v1_i * v2_j
    import scalation.mathstat.VectorDOps._

    object MyFit extends scalation.modeling.FitM

    val nx  = xx.dim2                                           // size of input data row  
    val sst = (yy - yy.mean).normSq                             // sum of squares total, per column
    val c   = VectorD (0.2, 0.2, 0.2)                           // convolutional parameter/weight vector: input -> hidden
    val α   = VectorD (0.1, 0.1, 0.1, 0.1)                      // hidden layer bias vector
    val b   = MatrixD ((4, 1), 0.1, 0.1, 0.1, 0.1)              // parameter/weight matrix: hidden -> output
                                                                // initial weights/parameters (random in practice)
    val β   = VectorD (0.1)                                     // output layer bias vector
    val nc  = c.dim                                             // size of convolutional filter
    val nv  = nx - nc + 1                                       // size of delta 0 for a 'valid' convolution @see `simpleCNN3`

    println (s"nx = $nx, nc = $nc, sst = $sst, xx = $xx, yy = $yy")

    val η = 0.7                                                 // sigmoid learning rate (to be tuned)
//  val η = 0.012                                               // reLU learning rate (to be tuned)
    var u, z, v, ŷ, ε, ƒ1, δ1, ƒ0, δ0: VectorD = null
    var g1: MatrixD = null
    val yp = new MatrixD (yy.dim, 1)                            // save each prediction in yp

// try f0 = reLU or sigmoid, f1 = id

    for epoch <- 1 to 100 do
        println (s"Improvement step $epoch")
        val sse = VectorD (0.0)
        for i <- xx.indices do
            val (x, y) = (xx(i), VectorD (yy(i)))               // randomize i for Stochastic Gradient Descent (SGD)

            // forward prop: input -> hidden
            u  = convs (c, x) + α                               // hidden pre-activation vector via 'same' convolution
            z  = sigmoid_ (u)                                   // hidden vector from f0 = sigmoid activation
//          z  = reLU_ (u)                                      // hidden vector from f0 = reLU activation

            // forward prop: hidden -> output
            v  = b.ᵀ * z + β                                    // output pre-activation vector
            ŷ  = v                                              // output/prediction vector from f1 = id activation
            ε  = y - ŷ                                          // error vector

            // backward prop: hidden <- output
//          ƒ1 = ŷ.map (t => if t > 0.0 then 1.0 else 0.0)      // derivative (f1') for reLU
            ƒ1 = VectorD.one (ŷ.dim)                            // derivative (f1') for id
            δ1 = -ε * ƒ1                                        // delta correction vector via elementwise product
            g1 = z ⊗ δ1                                         // transposed Jacobian matrix (gradients)

            // backward prop: input <- hidden
            ƒ0 = z * (1.0 - z)                                  // derivative (f0') for sigmoid
//          ƒ0 = z.map (t => if t > 0.0 then 1.0 else 0.0)      // derivative (f0') for reLU
//          ƒ0 = VectorD.one (z.dim)                            // derivative (f0') for id
            δ0 = b * δ1 * ƒ0                                    // delta correction vector

            // parameter updates
            b -= g1 * η                                         // update output parameter/weight matrix
            β -= δ1 * η                                         // update output bias vector
//          c -= g0 * η                                         // update convolutional hidden parameter/weight vector

            for j <- c.indices do c(j) -= x(j until j+nv) ∙ δ0(j until j+nv) * η    // FIX - c_j based on what x's and δ0's

            α -= δ0 * η                                         // update hidden bias vector
            yp(i) = ŷ                                           // save i-th prediction
            sse += ε * ε                                        // sum of squared errors
            val r2 = -(sse / sst - 1.0)                         // R^2 (recoded to avoid Ops clash)

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
            smape = ${MyFit.smapeF (yy.drop(1), yp(?, 0).drop(1))}
            """)
        end for
    end for

    new Plot (null, yy, yp(?, 0), "IGD for CNN y", lines = true)

    println ("Compare smape scores with Regression")
    import scalation.modeling.Regression
    val mod = new Regression (xx, yy)
    val ypr = mod.trainNtest ()()._1
    println (s"smape = ${MyFit.smapeF (yy.drop(1), ypr.drop(1))}")   // can't forecast first value
 
end simpleCNN2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleCNN3` main function illustrates the use of Incremental Gradient Descent (IGD)
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
 *  Computations done at the vector level, x -> z -> y.  R^2 = .389 vs. .409 for Regression
 *  > runMain scalation.modeling.neuralnet.simpleCNN3
 */
@main def simpleCNN3 (): Unit =

    import scalation.mathstat.MatrixDOps.⊗                      // outer product of two vectors: v1 ⊗ v2 = v1 v1.ᵀ
                                                                // matrix where m_ij = v1_i * v2_j
//  import scalation.mathstat.VectorDOps._

    object MyFit extends scalation.modeling.FitM

    val nx  = xx.dim2                                           // size of input data row  
    val sst = (yy - yy.mean).normSq                             // sum of squares total, per column
    val c   = VectorD (0.2, 0.2, 0.2)                           // convolutional parameter/weight vector: input -> hidden
    val α   = VectorD (0.1, 0.1)                                // hidden layer bias vector
    val b   = MatrixD ((2, 1), 0.1, 0.1)                        // parameter/weight matrix: hidden -> output
                                                                // initial weights/parameters (random in practice)
    val β   = VectorD (0.1)                                     // output layer bias vector
    val nc  = c.dim                                             // size of convolutional filter
    val nδ0 = nx - nc + 1                                       // size of delta 0

    println (s"nx = $nx, nc = $nc, sst = $sst, xx = $xx, yy = $yy")

    val η = 0.7                                                 // sigmoid learning rate (to be tuned)
//  val η = 0.012                                               // reLU learning rate (to be tuned)
    var u, z, v, ŷ, ε, ƒ1, δ1, ƒ0, δ0: VectorD = null
    var g1: MatrixD = null
    val yp = new MatrixD (yy.dim, 1)                            // save each prediction in yp

// try f0 = reLU or sigmoid, f1 = id

    for epoch <- 1 to 100 do
        println (s"Improvement step $epoch")
        val sse = VectorD (0.0)
        for i <- xx.indices do
            val (x, y) = (xx(i), VectorD (yy(i)))               // randomize i for Stochastic Gradient Descent (SGD)

            // forward prop: input -> hidden
            u  = conv (c, x) + α                                // hidden pre-activation vector via 'valid' convolution
//          z  = sigmoid_ (u)                                   // hidden vector from f0 = sigmoid activation
            z  = reLU_ (u)                                      // hidden vector from f0 = reLU activation

            // forward prop: hidden -> output
            v  = b.ᵀ * z + β                                    // output pre-activation vector
            ŷ  = v                                              // output/prediction vector from f1 = id activation
            ε  = y - ŷ                                          // error vector

            // backward prop: hidden <- output
//          ƒ1 = ŷ.map (t => if t > 0.0 then 1.0 else 0.0)      // derivative (f1') for reLU
            ƒ1 = VectorD.one (ŷ.dim)                            // derivative (f1') for id
            δ1 = -ε * ƒ1                                        // delta correction vector via elementwise product
            g1 = z ⊗ δ1                                         // transposed Jacobian matrix (gradients)

            import scalation.is_

            // backward prop: input <- hidden
//          ƒ0 = z * (1.0 - z)                                  // derivative (f0') for sigmoid
//          ƒ0 = z.map (t => if t > 0.0 then 1.0 else 0.0)      // derivative (f0') for reLU
            ƒ0 = z.map (t => is_ (t > 0.0))                     // derivative (f0') for reLU
//          ƒ0 = VectorD.one (z.dim)                            // derivative (f0') for id
            δ0 = b * δ1 * ƒ0                                    // delta correction vector

            // parameter updates
            b -= g1 * η                                         // update output parameter/weight matrix
            β -= δ1 * η                                         // update output bias vector
//          c -= g0 * η                                         // update convolutional hidden parameter/weight vector

            for j <- c.indices do c(j) -= x(j until j+nδ0) ∙ δ0 * η    // CHECK - c_j based on what x's and δ0's

            α -= δ0 * η                                         // update hidden bias vector
            yp(i) = ŷ                                           // save i-th prediction
            sse += ε * ε                                        // sum of squared errors
            val r2 = -(sse / sst - 1.0)                         // R^2 (recoded to avoid Ops clash)

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
            smape= ${MyFit.smapeF (yy.drop(1), yp(?, 0).drop(1))}   // can't forecast first value
            """)
        end for
    end for

    new Plot (null, yy, yp(?, 0), "IGD for CNN y", lines = true)

    println ("Compare smape scores with Regression")
    import scalation.modeling.Regression
    val mod = new Regression (xx, yy)
    val ypr = mod.trainNtest ()()._1
    println (s"smape = ${MyFit.smapeF (yy.drop(1), ypr.drop(1))}")
 
    println ("Compare smape scores with ARY Forecaster")
    val mod2 = ARY (yy, 1)
    mod2.setSkip (1)
    mod2.trainNtest_x ()()
 
end simpleCNN3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleCNN4` main function tests the conv ('valid' convolution) and convs
 *  ('same' convolution) updated methods in `VectorD`.
 */
@main def simpleCNN4 (): Unit =

    val c = VectorD (0.5, 1.0, 0.5)                             // convolution filter
    val x = VectorD (-3, -2, -1, 0, 1, 2, 3, 4)                 // sample input (one row)
    val α = 0.0                                                 // for simplicity assume no bias
    var u = conv (c, x) + α                                     // hidden pre-activation vector φ via 'valid' convolution
    var z = reLU_ (u)                                           // hidden vector (no pooling) from f0 = reLU activation
    println ("Test the 'valid' convolution operator")
    println (s"u = $u") 
    println (s"z = $z") 

    u = convs (c, x) + α                                        // hidden pre-activation vector φ via 'same' convolution
    z = reLU_ (u)                                               // hidden vector (no pooling) from f0 = reLU activation
    println ("Test the 'same' convolution operator")
    println (s"u = $u") 
    println (s"z = $z") 

end simpleCNN4

