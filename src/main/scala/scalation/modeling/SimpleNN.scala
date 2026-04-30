
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Oct 20 20:16:25 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Simple Neural Networks Using Gradient Descent Optimization
 *           Tests both Gradient Descent (GD) and Incremental Gradient Descent (IGD)
 *           Simplified versions of `Regression`, `Perceptron`, `NeuralNet_2L`, and `NeuralNet_3L`
 *           for illustration/learning, not production
 *
 *  @note    the symbol ƒ indicates the derivative of function f, i.e., ƒ = f'
 */

package scalation
package modeling

import scalation.mathstat.{VectorD, MatrixD, Plot}
import scalation.mathstat.VectorDOps._
import ActivationFun._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleNN` object contains a simple dataset for testing Gradient Descent (GD)
 *  and Incremental Gradient Descent (IGD) optimization algorithms.
 *  @see https://nowak.ece.wisc.edu/MFML.pdf
 */
object SimpleNN:

    // 9 data points:         One    x1    x2    y1   y2
    val xy = MatrixD ((9, 5), 1.0,  0.1,  0.1,  0.5, 0.25,      // dataset
                              1.0,  0.1,  0.5,  0.3, 0.49,
                              1.0,  0.1,  1.0,  0.2, 0.64,

                              1.0,  0.5,  0.1,  0.8, 0.04,
                              1.0,  0.5,  0.5,  0.5, 0.25,
                              1.0,  0.5,  1.0,  0.3, 0.49,

                              1.0,  1.0,  0.1,  1.0, 0.0,
                              1.0,  1.0,  0.5,  0.8, 0.04,
                              1.0,  1.0,  1.0,  0.5, 0.25)

    val (x, y) = (xy(?, 0 until 3), xy(?, 3))                   // input matrix, output/response vector

end SimpleNN

import SimpleNN._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleNN1` main function illustrates the use of Gradient Descent (GD) to
 *  optimize the weights/parameters of a Multiple Linear Regression (MLR) model.
 *
 *      grad g = -x.ᵀ * (y - ŷ)    where pred ŷ = x * b
 *
 *  Computations done at the vector level: X -> y.  R^2 = .827
 *  > runMain scalation.modeling.simpleNN1
 */
@main def simpleNN1 (): Unit =

    val sst = (y - y.mean).normSq                               // sum of squares total
    val b   = VectorD (0.1, 0.2, 0.1)                           // initial weights/parameters (random in practice)

    val η = 0.1                                                 // learning rate (to be tuned)
    var ŷ, ε, δ, g: VectorD = null

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")

        // forward prop: input -> output
        ŷ = x * b                                               // prediction vector
        ε = y - ŷ                                               // error vector

        // backward prop: output -> input
        δ = -ε                                                  // delta correction vector
        g = x.ᵀ * δ                                             // gradient vector

        // parameter update
        b -= g * η                                              // update parameter vector

        val sse = ε.normSq                                      // sum of squared errors
        val r2  = 1.0 - sse / sst                               // R^2

        println (s"""
        ŷ   = $ŷ
        ε   = $ε
        δ   = $δ
        g   = $g
        b   = $b
        sse = $sse
        r2  = $r2
        """)
    end for

    new Plot (null, y, ŷ, "GD for MLR y", lines = true)

end simpleNN1


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleNN2` main function illustrates the use of Incremental Gradient Descent (IGD)
 *  to optimize the weights/parameters of a Multiple Linear Regression (MLR) model.
 *  Computations done at the vector level: X -> y.  R^2 = .933
 *  > runMain scalation.modeling.simpleNN2
 */
@main def simpleNN2 (): Unit =

    val sst = (y - y.mean).normSq                               // sum of squares total
    val b   = VectorD (0.1, 0.2, 0.1)                           // initial weights/parameters (random in practice)

    val η = 0.2                                                 // learning rate (to be tuned)
    var ŷ, ε, δ: Double = -0.0
    var g: VectorD = null
    val yp = new VectorD (y.dim)                                // save each prediction in yp

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")
        var sse = 0.0
        for i <- x.indices do
            val (xi, yi) = (x(i), y(i))                         // randomize i for Stochastic Gradient Descent (SGD)

            // forward prop: input -> output
            ŷ = xi ∙ b                                          // prediction scalar
            ε = yi - ŷ                                          // error scalar

            // backward prop: output -> input
            δ = -ε                                              // delta correction scalar
            g = xi * δ                                          // gradient vector

            // parameter update
            b -= g * η                                          // update parameter vector

            yp(i) = ŷ                                           // save i-th prediction
            sse  += ε * ε                                       // sum of squared errors
        end for
        val r2 = 1.0 - sse / sst                                // R^2

        println (s"""
        ŷ   = $ŷ
        ε   = $ε
        δ   = $δ
        g   = $g
        b   = $b
        sse = $sse
        r2  = $r2
        """)
    end for

    new Plot (null, y, yp, "IGD for MLR y", lines = true)

end simpleNN2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleNN3` main function illustrates the use of Gradient Descent (GD) to
 *  optimize the weights/parameters of a simple neural network (Perceptron).
 *  Originally, perceptrons used the Heavyside activation function for binary classification
 *  problems, but have been extended to multi-class classification and regression problems.
 *  Furthermore, when the activation function is identity, the perceptron models are equivalent
 *  multiple linear regression models.
 *
 *      grad g = -x.ᵀ * (y - ŷ) * ƒ    where pred ŷ = f(x * b)
 *
 *  Computations done at the vector level: X -> y.  R^2 = .865
 *  > runMain scalation.modeling.simpleNN3
 */
@main def simpleNN3 (): Unit =

    val sst = (y - y.mean).normSq                               // sum of squares total

    val b = VectorD (0.1, 0.2, 0.1)                             // initial weights/parameters (random in practice)

    val η = 2.5                                                 // learning rate (to be tuned)
    var u, ŷ, ε, ƒ, δ, g: VectorD = null

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")

        // forward prop: input -> output
        u = x * b                                               // pre-activation vector
        ŷ = sigmoid_ (u)                                        // prediction vector
        ε = y - ŷ                                               // error vector

        // backward prop: output -> input
        ƒ = ŷ * (1.0 - ŷ)                                       // derivative (f') for sigmoid
        δ = -ε * ƒ                                              // delta correction vector
        g = x.ᵀ * δ                                             // gradient vector

        // parameter update
        b -= g * η                                              // update parameter vector

        val sse = ε.normSq                                      // sum of squared errors
        val r2  = 1.0 - sse / sst                               // R^2

        println (s"""
        u   = $u
        ŷ   = $ŷ 
        ε   = $ε
        ƒ   = $ƒ
        δ   = $δ
        g   = $g
        b   = $b
        sse = $sse
        r2  = $r2
        """)

    end for
    new Plot (null, y, ŷ, "GD for Perceptron y", lines = true)

end simpleNN3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleNN4` main function illustrates the use of Gradient Descent (GD) to optimize
 *  the weights/parameters of a simple neural network (2-layer (no hidden) Neural Network).
 *
 *  Prediction Equation: ŷ = f(B^T x) = f(W x + b) 
 *
 *  where x is an input vector, ŷ is a predicted output vector, f is the activation function,
 *  B is the parameter matrix, W is the weight matrix, and b is the bias vector.
 *  @note, PyTorch stores the weight matrix W [num_out, num_in] transposed from B to make
 *  back-propagation more efficient.
 *  Computations done at the vector level: X -> y.  R^2 = .865, .826
 *  > runMain scalation.modeling.simpleNN4
 */
@main def simpleNN4 (): Unit =

    val (x, y) = (xy(?, 0 until 3), xy(?, 3 until 5))           // input matrix, output/response matrix
    val sst    = (y - y.mean).normSq                            // sum of squares total

    val b = MatrixD ((3, 2), 0.1, 0.1,
                             0.2, 0.1,
                             0.1, 0.1)                          // initial weights/parameters (random in practice)

    val η = 2.5                                                 // learning rate (to be tuned)
    var u, ŷ, ε, ƒ, δ, g: VectorD = null

    for epoch <- 1 to 10; k <- y.indices2 do
        println (s"Improvement step $epoch")

        // forward prop: input -> output
        u = x * b(?, k)                                         // pre-activation vector
        ŷ = sigmoid_ (u)                                        // prediction vector
        ε = y(?, k) - ŷ                                         // error vector for column k

        // backward prop: output -> input
        ƒ = ŷ * (1.0 - ŷ)                                       // derivative (f') for sigmoid
        δ = -ε * ƒ                                              // delta correction vector
        g = x.ᵀ * δ                                             // gradient vector

        // parameter update
        b(?, k) = b(?, k) - g * η                               // update parameter matrix, column k

        val sse = ε.normSq                                      // sum of squared errors
        val r2  = 1.0 - sse / sst(k)                            // R^2

        println (s"""
        for k = $k
        u   = $u
        ŷ   = $ŷ
        ε   = $ε
        ƒ   = $ƒ
        δ   = $δ
        g   = $g
        b   = $b
        sse = $sse
        r2  = $r2
        """)
    end for

    new Plot (null, y(?, 1), ŷ, "GD for Two-layer Neural Net y_1", lines = true)

end simpleNN4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleNN5` main function illustrates the use of Gradient Descent (GD) to optimize
 *  the weights/parameters of a simple neural network (2-layer (no hidden) Neural Network).
 *  Computations done at the matrix level: X -> Y.  R^2 = .865, .826
 *  > runMain scalation.modeling.simpleNN5
 */
@main def simpleNN5 (): Unit =

    import scalation.mathstat.MatrixDOps._                      // may clash with VectorDOps

    val (x, y) = (xy(?, 0 until 3), xy(?, 3 until 5))           // input matrix, output/response matrix
    val sst    = (y - y.mean).normSq                            // sum of squares total

    val b = MatrixD ((3, 2), 0.1, 0.1,
                             0.2, 0.1,
                             0.1, 0.1)                          // initial weights/parameters (random in practice)

    val η = 2.5                                                 // learning rate (to be tuned)
//  val η = 0.29                                                // learning rate (to be tuned) best for var 1 83%
    var u, ŷ, ε, ƒ, δ, g: MatrixD = null

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")

        // forward prop: input -> output
        u = x * b                                               // pre-activation matrix
        ŷ = f_sigmoid.fM (u)                                    // prediction matrix
//      ŷ = f_tanh.fM (u)                                       // prediction matrix
        ε = y - ŷ                                               // error matrix

        // backward prop: output -> input
        ƒ = ŷ ⊙ (1.0 - ŷ)                                       // derivative (f') for sigmoid
//      ƒ = 1.0 - ŷ~^2                                          // derivative (f') for tanh
        δ = -ε ⊙ ƒ                                              // delta correction matrix via Hadamard product
        g = x.ᵀ * δ                                             // gradient vectors (matrix)

        // parameter update
        b -= g * η                                              // update parameter matrix

        val sse = ε.normSq                                      // sum of squared errors
        val r2  = -(sse / sst - 1.0)                            // R^2 (recoded to avoid Ops clash)

        println (s"""
        u   = $u
        ŷ   = $ŷ
        ε   = $ε
        ƒ   = $ƒ
        δ   = $δ
        g   = $g
        b   = $b
        sse = $sse
        r2  = $r2
        """)
    end for

    new Plot (null, y(?, 0), ŷ(?, 0), "GD for Two-layer Neural Net y_0", lines = true)
    new Plot (null, y(?, 1), ŷ(?, 1), "GD for Two-layer Neural Net y_1", lines = true)

end simpleNN5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleNN6` main function illustrates the use of Gradient Descent (GD) to optimize
 *  the weights/parameters of a simple neural network (3-layer (1 hidden) Neural Network).
 *  Computations done at the matrix level: X -> Z -> Y.  R^2 = .299, .432 (requires more epochs)
 *  > runMain scalation.modeling.simpleNN6
 */
@main def simpleNN6 (): Unit =

    import scalation.mathstat.MatrixDOps._                      // may clash with VectorDOps

    val (x, y) = (xy(?, 1 until 3), xy(?, 3 until 5))           // input matrix, output/response matrix
    val sst    = (y - y.mean).normSq                            // sum of squares total

    val a = MatrixD ((2, 3), 0.1, 0.1, 0.1,                     // parameter/weight matrix: input -> hidden
                             0.1, 0.1, 0.1)
    val α = VectorD (0.1, 0.1, 0.1)                             // hidden layer bias vector
    val b = MatrixD ((3, 2), 0.1, 0.1,                          // parameter/weight matrix: hidden -> output
                             0.2, 0.1,
                             0.1, 0.1)                          // initial weights/parameters (random in practice)
    val β = VectorD (0.1, 0.1)                                  // output layer bias vector

    val η = 10.0                                                // learning rate (to be tuned)
    var u, z, v, ŷ, ε, ƒ1, δ1, g1, ƒ0, δ0, g0: MatrixD = null

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")

        // forward prop: input -> hidden
        u = x * a + α                                           // hidden pre-activation matrix
        z = f_sigmoid.fM (u)                                    // hidden matrix from f0 activation

        // forward prop: hidden -> output
        v = z * b + β                                           // output pre-activation matrix
        ŷ = f_sigmoid.fM (v)                                    // output/prediction matrix from f1 activation
        ε = y - ŷ                                               // error matrix

        // backward prop: hidden <- output
        ƒ1 = ŷ ⊙ (1.0 - ŷ)                                      // derivative (f1') for sigmoid
        δ1 = -ε ⊙ ƒ1                                            // delta correction matrix via Hadamard product
        g1 = z.ᵀ * δ1                                           // gradient vectors (matrix)

        // backward prop: input <- hidden
        ƒ0 = z ⊙ (1.0 - z)                                      // derivative (f0') for sigmoid
        δ0 = (δ1 * b.ᵀ) ⊙ ƒ0                                    // delta correction matrix
        g0 = x.ᵀ * δ0                                           // gradient vectors (matrix)

        // parameter updates
        b -= g1 * η                                             // update output parameter/weight matrix
        β -= δ1.mean * η                                        // update output bias vector
        a -= g0 * η                                             // update hidden parameter/weight matrix
        α -= δ0.mean * η                                        // update hidden bias vector

        val sse = ε.normSq                                      // sum of squared errors
        val r2  = -(sse / sst - 1.0)                            // R^2 (recoded to avoid Ops clash)

        println (s"""
        u   = $u
        z   = $z
        v   = $v
        ŷ   = $ŷ
        ε   = $ε
        ƒ1  = $ƒ1
        δ1  = $δ1
        g1  = $g1
        ƒ0  = $ƒ0
        δ0  = $δ0
        g0  = $g0
        b   = $b
        β   = $β
        a   = $a
        α   = $α
        sse = $sse
        r2  = $r2
        """)
    end for

    new Plot (null, y(?, 0), ŷ(?, 0), "GD for Three-layer Neural Net y_0", lines = true)
    new Plot (null, y(?, 1), ŷ(?, 1), "GD for Three-layer Neural Net y_1", lines = true)

end simpleNN6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleNN7` main function illustrates the use of Incremental Gradient Descent (IGD)
 *  to optimize the weights/parameters of a simple neural network (3-layer (1 hidden) Neural Network).
 *
 *  Prediction Equation: z = f0(A^T x + α)
 *                       ŷ = f1(B^T z + β)
 *
 *  where x is an input vector, z is the hidden layer vector, ŷ is a predicted output vector, f0, f1
 *  are the activation functions, A and B are the parameter matrices, and α and β are the bias vectors.
 *  @note: Stochastic Gradient Descent (SGD) adds stochastic selection to IGD.  In practice,
 *  mini-batches of size 32, 64, or 128 are commonly used.
 *  Computations done at the vector level, x -> z -> y.  R^2 = .863, .913 (for 10 epochs)
 *  > runMain scalation.modeling.simpleNN7
 */
@main def simpleNN7 (): Unit =

    import scalation.mathstat.MatrixDOps.⊗                      // outer product of two vectors: v1 ⊗ v2 = v1 v1.ᵀ
                                                                // matrix where m_ij = v1_i * v2_j

    val (xx, yy) = (xy(?, 1 until 3), xy(?, 3 until 5))         // input matrix, output/response matrix
    println (s"xx = $xx")
    println (s"yy = $yy")

    val sst      = (yy - yy.mean).normSq                        // sum of squares total
    println (s"sst = $sst")

    val a = MatrixD ((2, 3),  0.2,  0.3,  0.2,                  // parameter/weight matrix: input -> hidden
                             -0.1, -0.2, -0.1)
    val α = VectorD (-0.1, -0.1, -0.1)                          // hidden layer bias vector
    val b = MatrixD ((3, 2), 0.1, -0.1,                         // parameter/weight matrix: hidden -> output
                             0.2, -0.2,
                             0.1, -0.1)                         // initial weights/parameters (random in practice)
    val β = VectorD (-0.1, 0.1)                                 // output layer bias vector

//  val η = 20.0                                                // learning rate (to be tuned)
    val η = 2.5                                                 // learning rate (to be tuned)
    var u, z, v, ŷ, ε, ƒ1, δ1, ƒ0, δ0: VectorD = null
    var g1, g0: MatrixD = null
    val yp = new MatrixD (yy.dim, yy.dim2)                      // save each prediction in yp

//  try f0 = tanh -> f1 = id

    for epoch <- 1 to 10 do
        println (s"Improvement step $epoch")
        val sse = new VectorD (2)
        for i <- xx.indices do
            val (xi, yi) = (xx(i), yy(i))                       // randomize i for Stochastic Gradient Descent (SGD)

            // forward prop: input -> hidden
            u = a.ᵀ * xi + α                                    // hidden pre-activation vector
            z = sigmoid_ (u)                                    // hidden vector from f0 = sigmoid activation
//          z = tanh_ (u)                                       // hidden vector from f0 = tanh activation

            // forward prop: hidden -> output
            v = b.ᵀ * z + β                                     // output pre-activation vector
            ŷ = sigmoid_ (v)                                    // output prediction vector from f1 = sigmoid activation
//          ŷ = v                                               // output prediction vector from f1 = id activation
            ε = yi - ŷ                                          // error vector [ε_0, ε_0]

            // backward prop: hidden <- output
            ƒ1 = ŷ * (1.0 - ŷ)                                  // derivative (f1') for sigmoid
//          ƒ1 = 1.0 - ŷ~^2                                     // derivative (f1') for tanh
//          ƒ1 = VectorD.one (ŷ.dim)                            // derivative (f1') for id
            δ1 = -ε * ƒ1                                        // delta correction vector via elementwise product
            g1 = z ⊗ δ1                                         // gradient vectors (matrix)

            // backward prop: input <- hidden
            ƒ0 = z * (1.0 - z)                                  // derivative (f0') for sigmoid
//          ƒ0 = 1.0 - z~^2                                     // derivative (f0') for tanh
//          ƒ0 = VectorD.one (z.dim)                            // derivative (f0') for id
            δ0 = b * δ1 * ƒ0                                    // delta correction vector
            g0 = xi ⊗ δ0                                        // gradient vectors (matrix)

            // parameter updates
            b -= g1 * η                                         // update output parameter/weight matrix
            β -= δ1 * η                                         // update output bias vector
            a -= g0 * η                                         // update hidden parameter/weight matrix
            α -= δ0 * η                                         // update hidden bias vector

            yp(i)  = ŷ                                          // save i-th prediction
            sse   += ε * ε                                      // sum of squared errors
        end for
        val r2 = -(sse / sst - 1.0)                         // R^2 (recoded to avoid Ops clash)

        println (s"""
        u   = $u
        z   = $z
        v   = $v
        ŷ   = $ŷ
        ε   = $ε
        ƒ1  = $ƒ1
        δ1  = $δ1
        g1  = $g1
        ƒ0  = $ƒ0
        δ0  = $δ0
        g0  = $g0
        b   = $b
        β   = $β
        a   = $a
        α   = $α
        sse = $sse
        r2  = $r2
        """)
    end for

    new Plot (null, yy(?, 0), yp(?, 0), "IGD for Three-layer Neural Net y_0", lines = true)
    new Plot (null, yy(?, 1), yp(?, 1), "IGD for Three-layer Neural Net y_1", lines = true)

end simpleNN7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleNN8` main function illustrates the use of Incremental Gradient Descent (IGD)
 *  to optimize the weights/parameters of a simple neural network (3-layer (1 hidden) Neural Network).
 *
 *  Prediction Equation: z = f0(A^T x + α)
 *                       ŷ = f1(B^T z + β)
 *
 *  where x is an input vector, z is the hidden layer vector, ŷ is a predicted output vector, f0, f1
 *  are the activation functions, A and B are the parameter matrices, and α and β are the bias vectors.
 *  Illustrates the need for RESCALING the data.
 *  Computations done at the vector level, x -> z -> y.  R^2 = .512 (no rescaling) .813 (rescaling) (for 20 epochs)
 *  > runMain scalation.modeling.simpleNN8
 */
@main def simpleNN8 (): Unit =


    import Example_AutoMPG.xy
    import scalation.mathstat.MatrixDOps.⊗                      // outer product of two vectors: v1 ⊗ v2 = v1 v1.ᵀ
                                                                // matrix where m_ij = v1_i * v2_j
    def rescale (x: MatrixD, yes: Boolean): Unit =
        if yes then
            for j <- x.indices2 do
                val (mu_j, sig_j) = (x(?, j).mean, x(?, j).stdev)
                x(?, j) = (x(?, j) - mu_j) / sig_j              // option: subtract mean, divide by standard deviation
    end rescale

    val n       = xy.dim2 - 1                                   // last column in xy
    val (xx, y) = (xy.not(?, n), xy(?, n))                      // (data/input matrix, response column)
    val sst     = (y - y.mean).normSq                           // sum of squares total
    val yy      = MatrixD.fromVector (y)                        // turn the m-vector y into an m-by-1 matrix

    val a = MatrixD.fill (6, 3, 0.1)                            // parameter/weight matrix: input -> hidden
    val α = VectorD (0.1, 0.1, 0.1)                             // hidden layer bias vector
    val b = MatrixD ((3, 1), 0.1,                               // parameter/weight matrix: hidden -> output
                             0.1,
                             0.1)                               // initial weights/parameters (random in practice)
    val β = VectorD (0.1)                                       // output layer bias vector

    rescale (xx, true)                                          // true => rescale, false => no rescaling
    println (s"xx = $xx")

    val η = 0.1                                                 // learning rate (to be tuned)
    var x, u, z, v, ŷ, ε, ƒ1, δ1, ƒ0, δ0: VectorD = null
    var g1, g0: MatrixD = null
    val yp = new MatrixD (yy.dim, yy.dim2)                      // save each prediction in yp

    for epoch <- 1 to 20 do
        println (s"Improvement step $epoch")
        val sse = new VectorD (1)
        for i <- xx.indices do
            x     = xx(i)                                       // randomize i for Stochastic Gradient Descent (SGD)
            val y = yy(i)

            // forward prop: input -> hidden
            u = a.ᵀ * x + α                                     // hidden pre-activation vector
            z = sigmoid_ (u)                                    // hidden vector from f0 = sigmoid activation

            // forward prop: hidden -> output
            v = b.ᵀ * z + β                                     // output pre-activation vector
            ŷ = v                                               // output/prediction vector from f1 = id activation
            ε = y - ŷ                                           // error vector [ε_1, ε_2]

            // backward prop: hidden <- output
            ƒ1 = VectorD (1.0)                                  // derivative (f1') for id
            δ1 = -ε * ƒ1                                        // delta correction vector via elementwise product
            g1 = z ⊗ δ1                                         // gradient vectors (matrix)

            // backward prop: input <- hidden
            ƒ0 = z * (1.0 - z)                                  // derivative (f0') for sigmoid
            δ0 = b * δ1 * ƒ0                                    // delta correction vector
            g0 = x ⊗ δ0                                         // gradient vectors (matrix)

            // parameter updates
            b -= g1 * η                                         // update output parameter/weight matrix
            β -= δ1 * η                                         // update output bias vector
            a -= g0 * η                                         // update hidden parameter/weight matrix
            α -= δ0 * η                                         // update hidden bias vector

            yp(i) = ŷ                                           // save i-th prediction
            sse += ε * ε                                        // sum of squared errors
        end for
        val r2 = -(sse / sst - 1.0)                             // R^2 (recoded to avoid Ops clash)

        println (s"""
        x   = $x
        u   = $u
        z   = $z
        v   = $v
        ŷ   = $ŷ
        ε   = $ε
        ƒ1  = $ƒ1
        δ1  = $δ1
        g1  = $g1
        ƒ0  = $ƒ0
        δ0  = $δ0
        g0  = $g0
        b   = $b
        β   = $β
        a   = $a
        α   = $α
        sse = $sse
        r2  = $r2
        """)
    end for

    new Plot (null, yy(?, 0), yp(?, 0), "IGD for Three-layer Neural Net y_0", lines = true)

end simpleNN8

/*
        sse = VectorD(0.739686,	0.490484)
        r2  = VectorD(-0.270453,	-0.186018)

        u   = VectorD(0.0466622,	0.0881569,	0.0466622)
        z   = VectorD(0.511663,	0.522025,	0.511663)
        v   = VectorD(0.625573,	-1.14734)
        ŷ   = VectorD(0.651485,	0.240975)
        ε   = VectorD(-0.151485,	0.00902484)
        ƒ1  = VectorD(0.227052,	0.182906)
        δ1  = VectorD(0.0343950,	-0.00165070)

        sse = VectorD(0.708677,	0.448573)
        r2  = VectorD(-0.217193,	-0.0846748)


i = 0, u = VectorD(-0.0900000,	-0.0900000,	-0.0900000)
i = 1, u = VectorD(-0.126626,	-0.163253,	-0.126626)
i = 2, u = VectorD(-0.183862,	-0.276948,	-0.183862)
i = 3, u = VectorD(-0.0185846,	0.0119617,	-0.0185846)
i = 4, u = VectorD(-0.0568096,	-0.0522733,	-0.0568096)
i = 5, u = VectorD(-0.103617,	-0.149618,	-0.103617)
i = 6, u = VectorD(0.0805007,	0.175186,	0.0805007)
i = 7, u = VectorD(0.0626796,	0.142599,	0.0626796)
i = 8, u = VectorD(0.0466622,	0.0881569,	0.0466622)
*/
