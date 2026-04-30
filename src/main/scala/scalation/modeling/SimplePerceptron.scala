
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
object SimplePerceptron:

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

end SimplePerceptron

import SimplePerceptron._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simplePerceptron1` main function illustrates the use of Gradient Descent (GD) to
 *  optimize the weights/parameters of a simple neural network (Perceptron).
 *  Originally, perceptrons used the Heavyside activation function for binary classification
 *  problems, but have been extended to multi-class classification and regression problems.
 *  Furthermore, when the activation function is identity, the perceptron models are equivalent
 *  multiple linear regression models.
 *
 *      grad g = -x.ᵀ * (y - ŷ) * ƒ    where pred ŷ = f(x * b)
 *
 *  Computations done at the vector level: X -> y.  R^2 = .865
 *  > runMain scalation.modeling.simplePerceptron1
 */
@main def simplePerceptron1 (): Unit =

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

end simplePerceptron1


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simplePerceptron2` main function illustrates the use of Incremental Gradient Descent (IGD)
 *  to optimize the weights/parameters of a simple neural network (Perceptron).
 *
 *      grad g = -x.ᵀ * (y - ŷ) * ƒ    where pred ŷ = f(x * b)
 *
 *  Computations done at the scalar level: X -> y.  R^2 = .886
 *  > runMain scalation.modeling.simplePerceptron2
 */
@main def simplePerceptron2 (): Unit =

    val sst = (y - y.mean).normSq                               // sum of squares total

//  val b = VectorD (0.1, 0.2, 0.1)                             // initial weights/parameters (random in practice)
    val b = VectorD (0.2, 1.6, -1.6)                             // initial weights/parameters (random in practice)

    val η = 2.5                                                 // learning rate (to be tuned)
    var u, ŷ, ε, ƒ, δ: Double = NO_DOUBLE
    var g: VectorD = null
    val yp = new VectorD (y.dim)                                // save each prediction in yp

    for epoch <- 1 to 1 do
        println (s"Improvement step $epoch")
        var sse = 0.0
        for i <- x.indices do
            val (xi, yi) = (x(i), y(i))                         // randomize i for Pure Stochastic Gradient Descent (PSGD)

            // forward prop: input -> output
            u = xi ∙ b                                          // pre-activation scalar
            ŷ = sigmoid (u)                                     // prediction scalar
            ε = yi - ŷ                                          // error scalar

            // backward prop: output -> input
            ƒ = ŷ * (1.0 - ŷ)                                   // derivative (f') for sigmoid
            δ = -ε * ƒ                                          // delta correction scalar
            g = xi * δ                                          // gradient vector

            // parameter update
            b -= g * η                                          // update parameter vector

            yp(i) = ŷ                                           // save i-th prediction
            sse  += ε * ε                                       // sum of squared errors
        end for
        val r2 = 1.0 - sse / sst                                // R^2

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
    new Plot (null, y, yp, "IGD for Perceptron y", lines = true)

end simplePerceptron2

