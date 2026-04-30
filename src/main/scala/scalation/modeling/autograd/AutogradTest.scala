
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 19:40:13 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Unit Tests for Autograd Functionality
 */

package scalation
package modeling
package autograd

import scala.language.implicitConversions
import scala.math.ceil

import scalation.mathstat.{MatrixD, TensorD, TnT_Split, VectorD}
import scalation.modeling.neuralnet._

import AutogradOps.given
import Example_AutoMPG.{x, y}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AutogradTest` object contains various @main tests for autograd functionality.
 *  The tests validate basic arithmetic, complex expressions, activation functions,
 *  loss functions, and neural network layers with backpropagation.
 */
object AutogradTest:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest0` main function tests basic unary operations in the
     *  Autograd system.  These include absolute value, negation, floor, ceil,
     *  rounding, sign, and clipping.  For each operator, the test reports the
     *  forward results and performs gradient checking using `GradCheck.gradCheck`.
     *  This ensures that each scalar/tensor unary operation is correctly
     *  implemented in both the forward and backward passes.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest0
     */
    @main def autogradTest0 (): Unit =

        val R = TestReport ()

        R.record ("Absolute Value") {
            val data = TensorD ((2, 2, 1), -1.0, -2.0, 3.0, 4.0)
            val x = Variabl (data, name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.abs: ${x.abs.data}")
            GradCheck.gradCheck (x, () => x.abs, quiet = true)
        }
        R.record ("Negation") {
            val data = TensorD ((2, 2, 1), 1.0, -2.0, 3.0, -4.0)
            val x = Variabl (data, name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.neg: ${-x.data}")
            GradCheck.gradCheck (x, () => -x, quiet = true)
        }
        R.record ("Floor") {
            val x = Variabl (TensorD ((2, 2, 1), -1.2, -2.8, 3.4, 4.9))
            println (s"x: ${x.data}")
            println (s"x.floor: ${x.floor.data}")
            GradCheck.gradCheck (x, () => x.floor, quiet = true)
        }
        R.record ("Ceil") {
            val x = Variabl (TensorD ((2, 2, 1), -1.2, -2.8, 3.4, 4.9))
            println (s"x: ${x.data}")
            println (s"x.ceil: ${x.ceil.data}")
            GradCheck.gradCheck (x, () => x.ceil, quiet = true)
        }
        R.record ("Round") {
            val x = Variabl (TensorD ((2, 2, 1), -1.2, -2.3, 3.4, 4.7)) // not half-integers (since grad is undefined there)
            println (s"x: ${x.data}")
            println (s"x.round: ${x.round.data}")
            GradCheck.gradCheck (x, () => x.round, quiet = true)
        }
        R.record ("Sign") {
            val x = Variabl (TensorD ((2, 2, 1), -1.0,10.0, 2.0, -3.0)) // avoid zero (since grad is undefined there)
            println (s"x: ${x.data}")
            println (s"x.sign: ${x.sign.data}")
            GradCheck.gradCheck (x, () => x.sign, quiet = true)
        }
        R.record ("Clip") {
            val x = Variabl (TensorD ((2, 2, 1), -2.0, -0.5, 0.5, 3.0))
            println (s"x: ${x.data}")
            println (s"x.clip(-1.0, 1.0): ${x.clip (-1.0, 1.0).data}")
            GradCheck.gradCheck (x, () => x.clip (-1.0, 1.0), quiet = true)
        }
        R.summary ("Autograd Basic Operations - Test 0")

    end autogradTest0


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest1` main function tests basic binary arithmetic operations
     *  in the Autograd system.  These include addition, subtraction, element-wise
     *  multiplication, and element-wise division between two tensors, as well as
     *  unary operations involving constants.  For each operation, the test applies
     *  `GradCheck.gradCheck` or `gradCheckAll` to verify that both the forward
     *  and backward passes are implemented correctly.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest1
     */
    @main def autogradTest1 (): Unit =

        val R = TestReport ()

        R.record ("Addition") {
            val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val data2 = TensorD ((2, 2, 1), 5.0, 6.0, 7.0, 8.0)
            val x = Variabl (data1, name = Some ("x"))
            val y = Variabl (data2, name = Some ("y"))
            GradCheck.gradCheckAll (Seq (x, y), () => x + y, quiet = true)
        }
        R.record ("Subtraction") {
            val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val data2 = TensorD ((2, 2, 1), 5.0, 6.0, 7.0, 8.0)
            val x = Variabl (data1, name = Some ("x"))
            val y = Variabl (data2, name = Some ("y"))
            GradCheck.gradCheckAll (Seq (x, y), () => x - y, quiet = true)
        }
        R.record ("Multiplication") {
            val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val data2 = TensorD ((2, 2, 1), 5.0, 6.0, 7.0, 8.0)
            val x = Variabl (data1, name = Some ("x"))
            val y = Variabl (data2, name = Some ("y"))
            GradCheck.gradCheckAll (Seq (x, y), () => x * y, quiet = true)
        }
        R.record ("Division") {
            val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val data2 = TensorD ((2, 2, 1), 5.0, 6.0, 7.0, 8.0)
            val x = Variabl (data1, name = Some ("x"))
            val y = Variabl (data2, name = Some ("y"))
            GradCheck.gradCheckAll (Seq (x, y), () => x / y, quiet = true)
        }
        // Unary ops
        R.record ("Add constant") {
            val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val x = Variabl (data1, name = Some ("x"))
            GradCheck.gradCheck (x, () => x + 10.0, quiet = true)
        }
        R.record ("Sub constant") {
            val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val x = Variabl (data1, name = Some ("x"))
            GradCheck.gradCheck (x, () => x - 5.0, quiet = true)
        }
        R.record ("Mul constant") {
            val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val x = Variabl (data1, name = Some ("x"))
            GradCheck.gradCheck (x, () => x * 2.0, quiet = true)
        }
        R.record ("Div constant") {
            val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val x = Variabl (data1, name = Some ("x"))
            GradCheck.gradCheck (x, () => x / 2.0, quiet = true)
        }
        R.summary (title = "Autograd Basic Operations - Test 1")

    end autogradTest1


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest2` main function tests a more complex expression formed
     *  by combining several elementary operations, namely element-wise
     *  z = (x * y) + (x / y) - y.
     *  It evaluates the forward computation and then uses
     *  `GradCheck.gradCheckAll` to verify correct gradient propagation through
     *  the composite computation graph.  The test also exports the resulting
     *  computation graph in DOT format for visualization.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest2
     */
    @main def autogradTest2 (): Unit =

        val data1 = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
        val data2 = TensorD ((2, 2, 1), 5.0, 6.0, 7.0, 8.0)
        val x = Variabl (data1, name = Some ("x"))
        val y = Variabl (data2, name = Some ("y"))

        // Complex operation: z = (x * y) + (x / y) - y
        println ("\nTesting Complex Expression: z = (x * y) + (x / y) - y")
        val zFinal = (x * y) + (x / y) - y
        println (s"zFinal ((x * y) + (x / y) - y): $zFinal")

        val lossFunc = () => ((x * y) + (x / y) - y)
        val R = TestReport ()
        
        R.record ("Complex Expression") {
            GradCheck.gradCheckAll (Seq (x, y), lossFunc, quiet = false, debug = true)
        }
        R.summary (title = "Autograd Complex Operations - Test 2")

        // Export computation graph (after grad checks so gradients populated)
        val outPath = "target/autograd/visualization/computation_graph_test.dot"
        GraphExporter.writeDot (zFinal, outPath, renderSvg = true)
        println (s"Computation graph DOT written to $outPath")

    end autogradTest2


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest3` main function tests a broad range of mathematical 
     *  operations supported by the Autograd system.  These include scalar 
     *  operations (power, square root, logarithm, reciprocal, exponential), 
     *  global reductions (mean, sum, variance, standard deviation), axis-wise 
     *  reductions, and extrema extraction.  For each operation, the forward 
     *  result is displayed and `GradCheck.gradCheck` is applied to verify 
     *  the correctness of the backward pass.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest3
     */
    @main def autogradTest3 (): Unit =

        banner ("Autograd Math Operations - Test 3")

        val R = TestReport ()

        // -----------------------------------------------------------------
        // Scalar math
        // -----------------------------------------------------------------

        R.record ("Power") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x ~^ 2: ${(x ~^ 2).data}")
            GradCheck.gradCheck (x, () => x ~^ 2, quiet = true)
        }
        R.record ("Square Root") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 4.0, 8.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.sqrt: ${x.sqrt.data}")
            GradCheck.gradCheck (x, () => x.sqrt, quiet = true)
        }
        R.record ("Logarithm") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 4.0, 8.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.log: ${x.log.data}")
            GradCheck.gradCheck (x, () => x.log, quiet = true)
        }
        R.record ("Logarithm Base y") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 4.0, 8.0), name = Some ("x"))
            val y = 2.0
            println (s"x: ${x.data}")
            println (s"x.logBase($y): ${x.logBase(y).data}")
            GradCheck.gradCheck (x, () => x.logBase(y), quiet = true)
        }
        R.record ("Reciprocal") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 4.0, 8.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.reciprocal: ${x.reciprocal.data}")
            GradCheck.gradCheck (x, () => x.reciprocal, quiet = true)
        }
        R.record ("Exponential") {
            val x = Variabl (TensorD ((2, 2, 1), 0.0, 1.0, 2.0, 3.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.exp: ${x.exp.data}")
            GradCheck.gradCheck (x, () => x.exp, quiet = true)
        }

        // -----------------------------------------------------------------
        // Reductions (global)
        // -----------------------------------------------------------------

        R.record ("Mean") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 4.0, 8.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.mean: ${x.mean.data}")
            GradCheck.gradCheck (x, () => x.mean, quiet = true)
        }
        R.record ("Sum") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 4.0, 8.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.sum: ${x.sum.data}")
            GradCheck.gradCheck (x, () => x.sum, quiet = true)
        }
        R.record ("Variance") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 4.0, 8.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.variance: ${x.variance.data}")
            GradCheck.gradCheck (x, () => x.variance, quiet = true)
        }
        R.record ("Standard Deviation") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 4.0, 8.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.std: ${x.std.data}")
            GradCheck.gradCheck (x, () => x.std, quiet = true)
        }

        // -----------------------------------------------------------------
        // Axis-wise reductions
        // -----------------------------------------------------------------

        R.record ("Mean Along Axis 1") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.meanAxis(1): ${x.meanAxis(1).data}")
            GradCheck.gradCheck (x, () => x.meanAxis(1), quiet = true)
        }
        R.record ("Variance Along Axis 1") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.varAxis(1): ${x.varAxis(1).data}")
            GradCheck.gradCheck (x, () => x.varAxis(1), quiet = true)
        }
        R.record ("Std Along Axis 1") {
            val x = Variabl (TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.stdAxis(1): ${x.stdAxis(1).data}")
            GradCheck.gradCheck (x, () => x.stdAxis(1), quiet = true)
        }

        // -----------------------------------------------------------------
        // Extrema reductions
        // -----------------------------------------------------------------

        R.record ("Max Value") {
            val x = Variabl (TensorD ((2, 2, 1), -1.0, 5.0, 3.0, 2.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.maxValue: ${x.maxValue.data}")
            GradCheck.gradCheck (x, () => x.maxValue, quiet = true)
        }
        R.record ("Min Value") {
            val x = Variabl (TensorD ((2, 2, 1), -1.0, 5.0, 3.0, 2.0), name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.minValue: ${x.minValue.data}")
            GradCheck.gradCheck (x, () => x.minValue, quiet = true)
        }
        R.summary (title = "Autograd Math Operations - Test 3")

    end autogradTest3


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest4` main function tests a variety of activation functions
     *  supported by the Autograd system, including ReLU, Sigmoid, Tanh, GeLU, 
     *  Softmax, Identity, LeakyReLU, and ELU.  Each activation is applied to a 
     *  representative set of inputs to check behavior across different regions 
     *  (e.g., saturation, symmetry, negative/positive domains).  Gradient checking 
     *  via `GradCheck.gradCheck` is performed to ensure correct backward 
     *  propagation through each nonlinearity.  Note that GeLU and Softmax forward 
     *  values are currently inconsistent with PyTorch and are marked for revision.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest4
     */
    @main def autogradTest4 (): Unit =

        banner ("Autograd Activation Functions - Test 4")

        val R = TestReport ()

        R.record ("ReLU") {
            val t = VectorD (-5.0, -2.0, -1.0, -0.5, 0.5, 2.0, 5.0)     // skip 0 for gradCheck
            val data = TensorD.fromVector (t)
            val x = Variabl (data, name = Some ("x"))
            println (s"x.relu: ${x.relu.data}")
            GradCheck.gradCheck (x, () => x.relu, quiet = true)
        }
        R.record ("Sigmoid") {
            val t = VectorD (-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0)    // test saturation + center
            val data = TensorD.fromVector (t)
            val x = Variabl (data, name = Some ("x"))
            println (s"x.sigmoid: ${x.sigmoid.data}")
            GradCheck.gradCheck (x, () => x.sigmoid, quiet = true)
        }
        R.record ("Tanh") {
            val t = VectorD (-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0)      // symmetric coverage
            val data = TensorD.fromVector (t)
            val x = Variabl (data, name = Some ("x"))
            println (s"x.tanh: ${x.tanh.data}")
            GradCheck.gradCheck (x, () => x.tanh, quiet = true)
        }
        // TODO fix: Forward values are wrong compared to PyTorch
        R.record ("GeLU") {
            val t = VectorD (-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0)  // around 0 is key
            val data = TensorD.fromVector (t)
            val x = Variabl (data, name = Some ("x"))
            println (s"x.gelu: ${x.gelu.data}")
            GradCheck.gradCheck (x, () => x.gelu, quiet = true)
        }
        R.record ("Softmax") {
            val t = VectorD (-2.0, -1.0, 0.0, 1.0, 2.0)        // vector input, tests relative scale
            val data = TensorD.fromVector (t)
            val x = Variabl (data, name = Some ("x"))
            println (s"x.softmax: ${x.softmax.data}")
            GradCheck.gradCheck (x, () => x.softmax, quiet = true)
        }
        R.record ("Identity") {
            val t = VectorD (-5.0, -1.0, 0.0, 1.0, 5.0)        // trivial, but still cover range
            val data = TensorD.fromVector (t)
            val x = Variabl (data, name = Some ("x"))
            println (s"x.id: ${x.id.data}")
            GradCheck.gradCheck (x, () => x.id, quiet = true)
        }
        R.record ("Leaky ReLU") {
            val t = VectorD (-5.0, -2.0, -1.0, -0.5, 0.5, 2.0, 5.0)    // skip 0 for gradCheck
            val data = TensorD.fromVector (t)
            val x = Variabl (data, name = Some ("x"))
            println (s"x.leakyReLU: ${x.leakyReLU ().data}")
            GradCheck.gradCheck (x, () => x.leakyReLU (), quiet = true)
        }
        R.record ("ELU") {
            val t = VectorD (-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0)  // negative + positive
            val data = TensorD.fromVector (t)
            val x = Variabl (data, name = Some ("x"))
            println (s"x.elu: ${x.elu ().data}")
            GradCheck.gradCheck (x, () => x.elu (), quiet = true)
        }
        R.summary (title = "Autograd Activation Functions - Test 4")

    end autogradTest4


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest5` main function tests core regression loss functions
     *  supported by the Autograd system: SSE, MSE, and MAE.  Each loss is evaluated
     *  on simple predicted and target tensors, and `GradCheck.gradCheck` is used
     *  to validate correctness of the backward pass.  These losses form the basis
     *  of most regression models, so ensuring accurate gradients is essential.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest5
     */
    @main def autogradTest5 (): Unit =

        banner ("Autograd Loss Functions - Test 5")
        val predData   = TensorD ((2, 2, 1), 0.9, 0.1, 0.8, 0.2)
        val targetData = TensorD ((2, 2, 1), 1.0, 0.0, 1.0, 0.0)
        val pred   = Variabl (predData, name = Some ("pred"))
        val target = Variabl (targetData, name = Some ("target"))

        println ("\nTesting Loss Functions")

        val R = TestReport ()

        R.record ("SSE Loss") {
            println (s"sseLoss (pred, target): ${sseLoss (pred, target).data}")
            GradCheck.gradCheck (pred, () => sseLoss (pred, target), quiet = true)
        }
        R.record ("MSE Loss") {
            println (s"mseLoss (pred, target): ${mseLoss (pred, target).data}")
            GradCheck.gradCheck (pred, () => mseLoss (pred, target), quiet = true)
        }
        R.record ("MAE Loss") {
            println (s"maeLoss (pred, target): ${maeLoss (pred, target).data}")
            GradCheck.gradCheck (pred, () => maeLoss (pred, target), quiet = true)
        }
        R.summary (title = "Autograd Loss Functions - Test 5")

    end autogradTest5


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest6` main function verifies tensor-level linear algebra
     *  operations, including transpose, permutation, reshape, slice, concat, dot product,
     *  matrix multiplication, and batched matrix multiplication (BMM).
     *  For each operation, forward correctness is checked and gradient correctness
     *  is validated using `GradCheck.gradCheck` or `gradCheckAll`.  
     *  BMM is also compared against a manually computed expected tensor to ensure
     *  alignment with the Autograd system’s shape conventions.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest6
     */
    @main def autogradTest6 (): Unit =

        banner ("Autograd Tensor Operations - Test 6")

        val R = TestReport ()

        // Transpose
        R.record ("Transpose") {
            val C = MatrixD ((2, 3), 1, 2, 3, 4, 5, 6)
            val mat = Variabl (TensorD.fromMatrix (C, Some ((1, 2, 3))), name = Some ("mat"))
            println (s"mat: ${mat.data}")
            println (s"mat.transpose (1,2): ${mat.transpose (1, 2).data}")
            GradCheck.gradCheck (mat, () => mat.transpose (1, 2), quiet = true)
        }
        // Permute
        R.record ("Permute") {
            val data = TensorD ((2, 3, 1), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            val x = Variabl (data, name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.permute (Seq (1,0,2)): ${x.permute (Seq (1, 0, 2)).data}")
            GradCheck.gradCheck (x, () => x.permute (Seq (1, 0, 2)), quiet = true)
        }
        // Reshape
        R.record ("Reshape") {
            val data = TensorD ((2, 3, 1), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            val x = Variabl (data, name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x.reshape(Seq (6,1,1)): ${x.reshape (Seq (6, 1, 1)).data}")
            GradCheck.gradCheck (x, () => x.reshape (Seq (6, 1, 1)), quiet = true)
        }
        // Slice
        R.record ("Slice") {
            val data = TensorD ((2, 3, 1), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
            val x = Variabl (data, name = Some ("x"))
            println (s"x: ${x.data}")
            println (s"x(?, 1 until 3, ?): ${x(?, 1 until 3, ?).data}")
            GradCheck.gradCheck (x, () => x(?, 1 until 3, ?), quiet = true)
        }
        // Concat
        R.record ("Concat") {
            val dataA = TensorD ((2, 2, 1), 1.0, 2.0, 3.0, 4.0)
            val dataB = TensorD ((2, 2, 1), 5.0, 6.0, 7.0, 8.0)
            val x = Variabl (dataA, name = Some ("x"))
            val y = Variabl (dataB, name = Some ("y"))
            println (s"x: ${x.data}")
            println (s"y: ${y.data}")
            println (s"x.concat (y, axis=1): ${concat (Seq (x, y), axis=1).data}")
            GradCheck.gradCheckAll (Seq (x, y), () => concat (Seq (x, y), axis=1), quiet = true)
        }
        // Dot product
        R.record ("Dot Product") {
            val vecA = Variabl (TensorD.fromVector (VectorD (1, 2, 3)), name = Some ("vecA"))
            val vecB = Variabl (TensorD.fromVector (VectorD (4, 5, 6)), name = Some ("vecB"))
            GradCheck.gradCheckAll (Seq (vecA, vecB), () => vecA.dot (vecB), quiet = true)
        }
        // Matrix multiplication
        R.record ("Matrix Multiplication") {
            val C = MatrixD ((2, 3), 1, 2, 3, 4, 5, 6)
            val D = MatrixD ((3, 2), 7, 8, 9, 10, 11, 12)
            val matA = Variabl (TensorD.fromMatrix (C, Some ((1, 2, 3))), name = Some ("matA"))
            val matB = Variabl (TensorD.fromMatrix (D, Some ((1, 3, 2))), name = Some ("matB"))
            GradCheck.gradCheckAll (Seq (matA, matB), () => matA.matmul (matB), quiet = true)
        }
        // Batched matrix multiplication
        R.record ("Batched MatMul (BMM)") {
            // batchedA: shape (2, 3, 1)
            val batchedA = Variabl (
                TensorD ((2, 3, 1),
                    1.0, 2.0, 3.0,       // batch 0
                    4.0, 5.0, 6.0),      // batch 1
                name = Some ("bmmA"))
            
            // batchedB: shape (2, 1, 2)
            val batchedB = Variabl (
                TensorD ((2, 1, 2),
                    10.0, 20.0,          // first column (k=0) for batch0, batch1
                    30.0, 40.0),         // second column (k=1) for batch0, batch1
                name = Some ("bmmB"))

            val result = batchedA.bmm (batchedB)
            
            val expected = TensorD ((2, 3, 2),
                // k=0 column then k=1 column (TensorD’s fill order is k,i,j)
                10.0, 20.0, 30.0, 80.0, 100.0, 120.0,
                30.0, 60.0, 90.0, 160.0, 200.0, 240.0
            )
            println (s"result: ${result.data}")
            println (s"expected: $expected")
            assert (result.data == expected, s"BMM result ${result.data} did not match expected $expected")
            
            GradCheck.gradCheckAll (Seq (batchedA, batchedB), () => batchedA.bmm(batchedB), quiet = true)
        }
        // Final summary
        R.summary (title = "Autograd Tensor Operations - Test 6")

    end autogradTest6
    
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest7` main function tests autograd through a small two-layer
     *  fully connected neural network.  A Linear → ReLU → Linear architecture is
     *  constructed, and gradients are validated for inputs, weights, and biases
     *  using `GradCheck.gradCheck`.  This test ensures end-to-end correctness of
     *  forward propagation, backward propagation, and parameter gradient tracking.
     *  A computation graph for the network output is also exported for debugging.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest7
     */
    @main def autogradTest7 (): Unit =

        banner ("Autograd 2-Layer Net - Test 7")
        
        val inFeatures  = 4
        val hiddenUnits = 5
        val outFeatures = 3
        
        // Two layers
        val fc1 = Linear (inFeatures, hiddenUnits)
        val fc2 = Linear (hiddenUnits, outFeatures)
        
        // Dummy batch of 2 examples, each of size 4
        val inputData = TensorD ((2, 4, 1),
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0)
        val inputVar = Variabl (inputData, name = Some ("input"))
        
        def net (x: Variabl): Variabl =
            x ~> fc1 ~> relu ~> fc2
        
        val R = TestReport ()
        
        // Forward sanity check
        R.record ("2-Layer Net - Forward") {
            val out = net (inputVar)
            println (s"Forward output: ${out.data}")
            true
        }
        // GradCheck: input
        R.record ("2-Layer Net - Input GradCheck") {
            GradCheck.gradCheck (inputVar, () => net (inputVar).sum, quiet = true)
        }
        // GradCheck: fc1 weights
        R.record ("2-Layer Net - FC1 Weight GradCheck") {
            GradCheck.gradCheck (fc1.weight, () => net (inputVar).sum, quiet = true)
        }
        // GradCheck: fc1 bias
        R.record ("2-Layer Net - FC1 Bias GradCheck") {
            GradCheck.gradCheck (fc1.bias, () => net (inputVar).sum, quiet = true)
        }
        // GradCheck: fc2 weights
        R.record ("2-Layer Net - FC2 Weight GradCheck") {
            GradCheck.gradCheck (fc2.weight, () => net (inputVar).sum, quiet = true)
        }
        // GradCheck: fc2 bias
        R.record ("2-Layer Net - FC2 Bias GradCheck") {
            GradCheck.gradCheck (fc2.bias, () => net (inputVar).sum, quiet = true)
        }
        // Final summary
        R.summary (title = "Autograd 2-Layer Net - Test 7")

        val outPath = "target/autograd/visualization/computation_graph_2layer_net.dot"
        GraphExporter.writeDot (net (inputVar), outPath, renderSvg = true)
        println (s"Computation graph DOT written to $outPath")
    
    end autogradTest7
    
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest8` main function trains a single-layer neural network on
     *  the AutoMPG regression dataset using Autograd.  The dataset is standardized,
     *  fed through a Linear ~> Identity model, and optimized using stochastic 
     *  gradient descent.  This test validates that the autograd engine supports
     *  full training loops, gradient accumulation, parameter updates, and evaluation
     *  against regression metrics.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest8
     */
    @main def autogradTest8 (): Unit =

        banner ("Autograd 1 Layer Net with AutoMPG (Testing against Regression) - Test 8")

        println (s"x.shape: ${x.dims}")
        println (s"y.shape: ${y.dim}")
        val tensorX = TensorD.fromMatrix (x).permute (Seq (1, 2, 0))
        val tensorY = TensorD.fromVector (y)
//      val meanX  = TensorD.meanAlongAxis (tensorX, axis=0)
//      val stdX   = TensorD.stdAlongAxis (tensorX, axis=0)
        val meanY  = TensorD.meanAlongAxis (tensorY, axis=0)
        val stdY   = TensorD.stdAlongAxis (tensorY, axis=0)
        val norm_x = TensorD.standardize (tensorX, axis=0)
        val norm_y = TensorD.standardize (tensorY, axis=0)

        println (s"norm_x.shape: ${norm_x.shape}")
        println (s"norm_y.shape: ${norm_y.shape}")
        val input    = Variabl (norm_x)
        var y_actual = Variabl (norm_y)

        case class Net () extends Module with Fit (x.dim2 - 1, x.dim2 - x.dim):
            val nf1: Int = tensorX.dim2
            val outputNodes: Int = tensorY.dim2
            val fc1: Linear = Linear (nf1, outputNodes)

            override def forward (x: Variabl): Variabl = x ~> fc1 ~> identity
        end Net

        object Net:
            def apply (): Net = new Net ()
        end Net

        val net = Net ()
        val optimizer = SGD (parameters = net.parameters, lr = 0.01, momentum = 0.9)
        val permGen   = new Optimizer_SGDM {}.permGenerator (norm_x.shape(0))
        val batchSize = 64
        val nB = norm_x.shape(0) / batchSize

        for j <- 0 to 1000 do
            optimizer.zeroGrad ()
            val batches    = permGen.igen.chop (nB)
            var totalLoss  = 0.0
            var batchCount = 0
            for ib <- batches do
                val inputBatch = Variabl (norm_x(ib))
                val yBatch     = Variabl (norm_y(ib))
                val output     = net (inputBatch)
                val loss       = mseLoss (output, yBatch)
                totalLoss     += loss.data(0)(0)(0)
                batchCount    += 1
                loss.backward ()
                optimizer.step ()
            end for
            val avgLoss = totalLoss / batchCount
            if j % 100 == 0 then println (s"Epoch $j: Loss = $avgLoss")
        end for

        println (s"Model structure: $net")
        println (s"Weight shape: ${net.fc1.weight.shape}")
        println (s"Bias shape: ${net.fc1.bias.shape}")

        val outputFinal = net (input)
        val y_pred = outputFinal * Variabl (stdY) + Variabl (meanY)
        y_actual   = y_actual * Variabl (stdY) + Variabl (meanY)
        println (s"y_pred shape: ${y_pred.shape}")
        val qof = net.diagnose (y_actual.data.flattenToVector, y_pred.data.flattenToVector)
        println (FitM.fitMap (qof, qoF_names))
        println (s"grad after convergence: ${net.fc1.weight.grad}")
        println (s"weights after convergence: ${net.fc1.weight.data}")
    end autogradTest8


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest9` main function trains a two-layer neural network on the
     *  AutoMPG regression dataset.  The architecture uses a hidden layer with a
     *  sigmoid activation followed by a linear output layer.  
     *  Optimization uses the Adam optimizer, demonstrating compatibility between
     *  Autograd parameter graphs and adaptive optimizers.  
     *  The test evaluates gradient flow, training stability, and final regression
     *  quality after rescaling predictions back to the original domain.
     * > runMain scalation.modeling.autograd.AutogradTest.autogradTest9
     */
    @main def autogradTest9 (): Unit =

        banner ("Autograd 2 Layer Net with AutoMPG (Testing against NeuralNet_3L - Test 9")

        println (s"x.shape: ${x.shape}")
        println (s"y.shape: ${y.dim}")
        val tensorX = TensorD.fromMatrix (x).permute (Seq (1, 2, 0))
        val tensorY = TensorD.fromVector (y)
//      val meanX  = TensorD.meanAlongAxis (tensorX, axis = 0)
//      val stdX   = TensorD.stdAlongAxis (tensorX, axis = 0)
        val meanY  = TensorD.meanAlongAxis (tensorY, axis = 0)
        val stdY   = TensorD.stdAlongAxis (tensorY, axis = 0)
        val norm_x = TensorD.standardize (tensorX, axis = 0)
        val norm_y = TensorD.standardize (tensorY, axis = 0)

        println (s"norm_x.shape: ${norm_x.shape}")
        println (s"norm_y.shape: ${norm_y.shape}")
        val input    = Variabl (norm_x)
        var y_actual = Variabl (norm_y)

        case class Net () extends Module with Fit (x.dim2 - 1, x.dim - x.dim2):
            val nf1: Int = tensorX.dim2
            val hiddenNodes: Int = 2 * nf1 + 1
            val outputNodes: Int = tensorY.dim2
            val fc1: Linear = Linear (nf1, hiddenNodes)
            val fc2: Linear = Linear (hiddenNodes, outputNodes)

            override def forward (x: Variabl): Variabl = x ~> fc1 ~> sigmoid ~> fc2 ~> identity
        end Net

        object Net:
            def apply (): Net = new Net ()
        end Net

        val net = Net ()
        val optimizer = Adam (parameters = net.parameters, lr = 0.002, beta1 = 0.9, beta2 = 0.999)
        val batchSize = 20
        val nB = norm_x.shape(0) / batchSize

        for j <- 0 to 400 do
            val permGen = new Optimizer_SGDM {}.permGenerator (norm_x.shape(0))
            val batches = permGen.igen.chop (nB)
            var totalLoss  = 0.0
            var totalElem = 0
            for ib <- batches do
                optimizer.zeroGrad()
                val inputBatch = Variabl (norm_x(ib))
                val yBatch     = Variabl (norm_y(ib))
                val output     = net (inputBatch)
                val loss       = mseLoss (output, yBatch)
                val numel      = yBatch.shape.product
                totalLoss     += loss.data(0)(0)(0) * numel
                totalElem     += numel
                loss.backward ()
                optimizer.step ()
            end for
            val avgLoss = totalLoss / totalElem
            if j % 100 == 0 then println (s"Epoch $j: Loss = $avgLoss")
        end for

        println (s"Model structure: $net")
        println (s"Weight shape: ${net.fc1.weight.shape}")
        println (s"Bias shape: ${net.fc1.bias.shape}")

        val outputFinal = net (input)
        val y_pred = outputFinal * Variabl (stdY) + Variabl (meanY)
        y_actual   = y_actual * Variabl (stdY) + Variabl (meanY)
        println (s"y_pred shape: ${y_pred.shape}")
        val qof = net.diagnose (y_actual.data.flattenToVector, y_pred.data.flattenToVector)
        println (FitM.fitMap (qof, qoF_names))
        println (s"grad after convergence: ${net.fc1.weight.grad}")
        println (s"weights after convergence: ${net.fc1.weight.data}")

    end autogradTest9
    

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest10` main function trains a three-layer neural network on
     *  the AutoMPG dataset using SGD with momentum and an early-stopping rule.
     *  Multiple nonlinearities (tanh and sigmoid) are used to exercise diverse
     *  gradient shapes.  
     *  The test demonstrates the autograd system’s support for: multi-layer models,
     *  training loops, early stopping, loss monitoring, and post-training evaluation
     *  with rescaled predictions.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest10
     */
    @main def autogradTest10 (): Unit =

        banner ("Autograd 3 Layer Net with AutoMPG (Testing against NeuralNet_3L - Test 10")
        println (s"x.shape: ${x.shape}")
        println (s"y.shape: ${y.dim}")
        val tensorX = TensorD.fromMatrix (x).permute (Seq (1, 2, 0))
        val tensorY = TensorD.fromVector (y)
//      val meanX  = TensorD.meanAlongAxis (tensorX, axis = 0)
//      val stdX   = TensorD.stdAlongAxis (tensorX, axis = 0)
        val meanY  = TensorD.meanAlongAxis (tensorY, axis = 0)
        val stdY   = TensorD.stdAlongAxis (tensorY, axis = 0)
        val norm_x = TensorD.standardize (tensorX, axis = 0)
        val norm_y = TensorD.standardize (tensorY, axis = 0)

        println (s"norm_x.shape: ${norm_x.shape}")
        println (s"norm_y.shape: ${norm_y.shape}")
        val input = Variabl (norm_x)
        var y_actual = Variabl (norm_y)

        case class Net () extends Module with Fit (x.dim2 - 1, x.dim - x.dim2):
            val nf1: Int         = tensorX.dim2
            val hiddenNodes: Int = 2 * nf1 + 1
            val outputNodes: Int = tensorY.dim2
            val fc1: Linear = Linear (nf1, hiddenNodes)
            val fc2: Linear = Linear (hiddenNodes, hiddenNodes)
            val fc3: Linear = Linear (hiddenNodes, outputNodes)

            override def forward (x: Variabl): Variabl =
                val h1  = x ~> fc1 ~> tanh
                val h2  = h1 ~> fc2 ~> sigmoid
                val out = h2 ~> fc3 ~> identity
                out
        end Net

        object Net:
            def apply (): Net = new Net ()
        end Net

        val net = Net ()
        val optimizer = SGD (parameters = net.parameters, lr = 0.25, momentum = 0.90)
        val batchSize = 20
        val nB = ceil (norm_x.shape(0).toDouble / batchSize).toInt
        println (s"nB: $nB")

        object monitor extends MonitorLoss
        object opti extends Optimizer_SGDM
        object EarlyStopper extends StoppingRule
        val limit        = 15
        var stopTraining = false

        for j <- 0 to 400 if ! stopTraining do
            val permGen    = opti.permGenerator (norm_x.shape(0))
            val batches    = permGen.igen.chop (nB)
            var totalLoss  = 0.0
            var totalElem  = 0
            for ib <- batches do
                optimizer.zeroGrad ()
                val inputBatch = Variabl (norm_x(ib))
                val yBatch     = Variabl (norm_y(ib))
                val output     = net (inputBatch)
                val loss       = mseLoss (output, yBatch)
                val numel      = yBatch.shape.product
                totalLoss     += loss.data(0)(0)(0) * numel
                totalElem     += numel
                loss.backward ()
                optimizer.step ()
            end for
            val avgLoss = totalLoss / totalElem
            monitor.collectLoss (avgLoss)
            if j % 100 == 0 then println (s"Epoch $j: Loss = $avgLoss")
            val (stopParams, currentBestLoss) = EarlyStopper.stopWhenContinuous (net.parameters, avgLoss, limit)
            if stopParams != null then
                println (s"Early stopping triggered at epoch $j with best loss $currentBestLoss")
                net.setParameters (stopParams)
                stopTraining = true
            end if
        end for

        monitor.plotLoss ("NeuralNet4L")
        val varData  = y.variance
        val bestLoss = monitor.getBestLoss * varData
        println (s"Best Loss Unscaled: $bestLoss")
        println (s"Best Loss Scaled: ${monitor.getBestLoss}")
        println (s"Model structure: $net")
        println (s"Weight shape: ${net.fc1.weight.shape}")
        println (s"Bias shape: ${net.fc1.bias.shape}")

        val outputFinal = net (input)
        val y_pred = outputFinal * Variabl (stdY) + Variabl (meanY)
        y_actual   = y_actual * Variabl (stdY) + Variabl (meanY)
        println (s"y_pred shape: ${y_pred.shape}")
        val qof = net.diagnose(y_actual.data.flattenToVector, y_pred.data.flattenToVector)
        println (FitM.fitMap (qof, qoF_names))
        println (s"grad after convergence: ${net.fc1.weight.grad}")
        println (s"weights after convergence: ${net.fc1.weight.data}")

    end autogradTest10
    

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest11` main function tests a three-layer neural network on
     *  a train/test split of the AutoMPG dataset.  The network uses sigmoid and
     *  ReLU activations and trains with SGD + momentum and an early stopping rule
     *  based on validation loss.  
     *  This test validates the autograd system’s ability to handle data splitting,
     *  model evaluation on unseen data, and training control flow involving
     *  patience-based stopping.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest11
     */
    @main def autogradTest11 (): Unit =

        banner ("Autograd 3 Layer Net with AutoMPG (Testing against NeuralNet_XL - Test 11")

        val permGenSplit = TnT_Split.makePermGen (x.dim)
        val n_test = (0.4 * x.dim).toInt
        val splitIdx = TnT_Split.testIndices (permGenSplit, n_test)
        val (x_test, x_train, y_test, y_train) = TnT_Split (x, y, splitIdx)
        println (s"x.shape: ${x.shape}")
        println (s"y.shape: ${y.dim}")
        val tensorX = TensorD.fromMatrix (x_train).permute (Seq (1, 2, 0))
        val tensorY = TensorD.fromVector (y_train)
        val meanX  = TensorD.meanAlongAxis (tensorX, axis = 0)
        val stdX   = TensorD.stdAlongAxis (tensorX, axis = 0)
        val meanY  = TensorD.meanAlongAxis (tensorY, axis = 0)
        val stdY   = TensorD.stdAlongAxis (tensorY, axis = 0)
        val norm_x = TensorD.standardize (tensorX, axis = 0)
        val norm_y = TensorD.standardize (tensorY, axis = 0)
        println (s"norm_x.shape: ${norm_x.shape}")
        println (s"norm_y.shape: ${norm_y.shape}")
        val input    = Variabl (norm_x)
        var y_actual = Variabl (norm_y)

        case class Net () extends Module with Fit (x.dim2 - 1, x.dim - x.dim2):
            val nf1: Int = tensorX.dim2
            val hiddenNodes: Int = 2 * nf1 + 1
            val outputNodes: Int = tensorY.dim2
            val fc1: Linear = Linear (nf1, hiddenNodes)
            val fc2: Linear = Linear (hiddenNodes, hiddenNodes)
            val fc3: Linear = Linear (hiddenNodes, outputNodes)

            override def forward (x: Variabl): Variabl =
                val h1  = x ~> fc1 ~> sigmoid
                val h2  = h1 ~> fc2 ~> relu
                val out = h2 ~> fc3 ~> identity
                out
        end Net

        object Net:
            def apply (): Net = new Net ()
        end Net

        val net = Net ()
        val optimizer = SGD (parameters = net.parameters, lr = 0.1, momentum = 0.90)
        val batchSize = 20
        val nB = ceil (norm_x.shape(0).toDouble / batchSize).toInt
        println (net.parameters.toString ())
        println (s"nB: $nB")

        object monitor extends MonitorLoss
        object opti extends Optimizer_SGDM
        object EarlyStopper extends StoppingRule
        val limit        = 40
        var stopTraining = false

        for j <- 0 to 400 if ! stopTraining do
            val permGen    = opti.permGenerator (norm_x.shape(0))
            val batches    = permGen.igen.chop (nB)
            var totalLoss  = 0.0
            var totalElem = 0
            for ib <- batches do
                optimizer.zeroGrad ()
                val inputBatch = Variabl (norm_x(ib))
                val yBatch     = Variabl (norm_y(ib))
                val output     = net (inputBatch)
                val loss       = mseLoss (output, yBatch)
                val numel      = yBatch.shape.product
                totalLoss     += loss.data(0)(0)(0) * numel
                totalElem     += numel
                loss.backward ()
                optimizer.step ()
            end for
            val avgLoss = totalLoss / totalElem
            monitor.collectLoss (avgLoss)
            if j % 100 == 0 then println (s"Epoch $j: Loss = $avgLoss")
            val (stopParams, currentBestLoss) = EarlyStopper.stopWhenPatience (net.parameters, avgLoss, limit)
            if stopParams != null then
                println (s"Early stopping triggered at epoch $j with best loss $currentBestLoss")
                net.setParameters (stopParams)
                stopTraining = true
            end if
        end for

        monitor.plotLoss ("NeuralNet3L")
        val varData  = y.variance
        val bestLoss = monitor.getBestLoss * varData
        println (s"Best Loss Unscaled: $bestLoss")
        println (s"Best Loss Scaled: ${monitor.getBestLoss}")
        println (s"Model structure: $net")
        println (s"Weight shape: ${net.fc1.weight.shape}")
        println (s"Bias shape: ${net.fc1.bias.shape}")

        val outputFinal = net (input)
        val y_pred = outputFinal * Variabl (stdY) + Variabl (meanY)
        y_actual   = y_actual * Variabl (stdY) + Variabl (meanY)
        println (s"y_pred shape: ${y_pred.shape}")
        banner ("Final Train Statistics")
        val qof = net.diagnose (y_actual.data.flattenToVector, y_pred.data.flattenToVector)
        println (FitM.fitMap (qof, qoF_names))
        val testX = TensorD.fromMatrix (x_test).permute (Seq (1, 2, 0))
        val testY = TensorD.fromVector (y_test)
        val testNormX  = (testX - meanX)/stdX
        val testNormY  = (testY - meanY)/stdY
        val testInput  = Variabl (testNormX)
        val testActual = Variabl (testNormY)
        val testOutput = net (testInput)
        val testPred = testOutput * Variabl (stdY) + Variabl (meanY)
        val testActualRescaled = testActual * Variabl (stdY) + Variabl (meanY)

        println (s"test_pred shape: ${testPred.shape}")
        println (s"Final Test Loss: ${mseLoss (testOutput, testActual).data(0)(0)(0)}")
        banner ("Final Test Statistics")
        val testQoF = net.diagnose (testActualRescaled.data.flattenToVector, testPred.data.flattenToVector)
        println (FitM.fitMap (testQoF, qoF_names))

    end autogradTest11
    

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTest12` main function provides an alternative training pipeline
     *  for a three-layer neural network on AutoMPG, using randomized train/test
     *  splitting.
     *  It tests the autograd engine under variations in data sampling, training
     *  strategy, and network depth, while also exercising patience-based early 
     *  stopping and full regression evaluation on both training and test sets.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTest12
     */
    @main def autogradTest12 (): Unit =

        banner ("Autograd 3 Layer Net with AutoMPG (Testing against NeuralNet_XL - Test 12")

        val permGenSplit = TnT_Split.makePermGen (x.dim)
        val rando  = true
        val n_test = (0.2 * x.dim).toInt
        val splitIdx = TnT_Split.testIndices (permGenSplit, n_test, rando = rando)
        val (x_test, x_train, y_test, y_train) = TnT_Split (x, y, splitIdx)
        println (s"x.shape: ${x.shape}")
        println (s"y.shape: ${y.dim}")
        val tensorX = TensorD.fromMatrix (x_train).permute (Seq (1, 2, 0))
        val tensorY = TensorD.fromVector (y_train)
        val meanX  = TensorD.meanAlongAxis (tensorX, axis = 0)
        val stdX   = TensorD.stdAlongAxis (tensorX, axis = 0)
        val meanY  = TensorD.meanAlongAxis (tensorY, axis = 0)
        val stdY   = TensorD.stdAlongAxis( tensorY, axis = 0)
        val norm_x = TensorD.standardize (tensorX, axis = 0)
        val norm_y = TensorD.standardize (tensorY, axis = 0)

        println (s"norm_x.shape: ${norm_x.shape}")
        println (s"norm_y.shape: ${norm_y.shape}")
        val input = Variabl (norm_x)
        var y_actual = Variabl (norm_y)
        
        case class Net () extends Module with Fit (x.dim2 - 1, x.dim - x.dim2):
            val nf1: Int = tensorX.dim2
            val hiddenNodes: Int = 2 * nf1 + 1
            val outputNodes: Int = tensorY.dim2
            val fc1: Linear = Linear (nf1, hiddenNodes)
            val fc2: Linear = Linear (hiddenNodes, hiddenNodes)
            val fc3: Linear = Linear (hiddenNodes, outputNodes)
            
            override def forward (x: Variabl): Variabl =
                val h1  = x ~> fc1 ~> sigmoid
                val h2  = h1 ~> fc2 ~> relu
                val out = h2 ~> fc3 ~> identity
                out
        end Net
        
        object Net:
            def apply (): Net = new Net ()
        end Net
        
        val net = Net ()
        val optimizer = SGD (parameters = net.parameters, lr = 0.25, momentum = 0.90)
        val batchSize = 20
        val nB = ceil (norm_x.shape(0).toDouble / batchSize).toInt
        println (net.parameters.toString ())
        println (s"nB: $nB")
        
        object monitor extends MonitorLoss
        object opti extends Optimizer_SGDM
        object EarlyStopper extends StoppingRule
        val limit = 40
        var stopTraining = false
        
        for j <- 0 to 400 if !stopTraining do
            val permGen = opti.permGenerator (norm_x.shape(0), rando=true)
            val batches = permGen.igen.chop (nB)
            var totalLoss = 0.0
            var totalElem = 0
            for ib <- batches do
                optimizer.zeroGrad ()
                val inputBatch = Variabl (norm_x(ib))
                val yBatch = Variabl (norm_y(ib))
                val output = net (inputBatch)
                val loss   = mseLoss (output, yBatch)
                val numel  = yBatch.shape.product
                totalLoss += loss.data(0)(0)(0) * numel
                totalElem += numel
                loss.backward ()
                optimizer.step ()
            end for
            val avgLoss = totalLoss / totalElem
            monitor.collectLoss (avgLoss)
            if j % 100 == 0 then println (s"Epoch $j: Loss = $avgLoss")
            val (stopParams, currentBestLoss) = EarlyStopper.stopWhenPatience (net.parameters, avgLoss, limit)
            if stopParams != null then
                println (s"Early stopping triggered at epoch $j with best loss $currentBestLoss")
                net.setParameters (stopParams)
                stopTraining = true
            end if
        end for
        
        monitor.plotLoss ("NeuralNet3L")
        val varData  = y.variance
        val bestLoss = monitor.getBestLoss * varData
        println (s"Best Loss Unscaled: $bestLoss")
        println (s"Best Loss Scaled: ${monitor.getBestLoss}")
        println (s"Model structure: $net")
        println (s"Weight shape: ${net.fc1.weight.shape}")
        println (s"Bias shape: ${net.fc1.bias.shape}")
        
        val outputFinal = net (input)
        val y_pred = outputFinal * Variabl (stdY) + Variabl (meanY)
        y_actual = y_actual * Variabl (stdY) + Variabl (meanY)
        println (s"y_pred shape: ${y_pred.shape}")
        banner ("Final Train Statistics")
        val qof = net.diagnose (y_actual.data.flattenToVector, y_pred.data.flattenToVector)
        println (FitM.fitMap (qof, qoF_names))
        val testX = TensorD.fromMatrix (x_test).permute (Seq (1, 2, 0))
        val testY = TensorD.fromVector (y_test)
        val testNormX  = (testX - meanX) / stdX
        val testNormY  = (testY - meanY) / stdY
        val testInput  = Variabl (testNormX)
        val testActual = Variabl (testNormY)
        val testOutput = net (testInput)
        val testPred   = testOutput * Variabl (stdY) + Variabl (meanY)
        val testActualRescaled = testActual * Variabl (stdY) + Variabl (meanY)

        println (s"test_pred shape: ${testPred.shape}")
        println (s"Final Test Loss: ${mseLoss (testOutput, testActual).data(0)(0)(0)}")
        banner ("Final Test Statistics")
        val testQoF = net.diagnose (testActualRescaled.data.flattenToVector, testPred.data.flattenToVector)
        println (FitM.fitMap (testQoF, qoF_names))

    end autogradTest12
    

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `autogradTestAll` main function sequentially runs the core autograd unit
     *  tests (0–7) to verify correctness across basic ops, math operations,
     *  activations, losses, tensor algebra, and small neural networks.  
     *  Longer AutoMPG training tests (8–11) are provided but commented out to avoid
     *  extended runtime by default.
     *  > runMain scalation.modeling.autograd.AutogradTest.autogradTestAll
     */
    @main def autogradTestAll (): Unit =

        banner ("All Autograd Tests Starting")
        
        // Run core autograd unit tests to verify correctness
        autogradTest0 ()
        autogradTest1 ()
        autogradTest2 ()
        autogradTest3 ()
        autogradTest4 ()
        autogradTest5 ()
        autogradTest6 ()
        autogradTest7 ()
        
        // The following tests are more time-consuming due to training loops
        // and are commented out by default.
        // 
        // autogradTest8 ()
        // autogradTest9 ()
        // autogradTest10 ()
        // autogradTest11 ()
        
        banner ("All Autograd Tests Completed")

    end autogradTestAll

end AutogradTest

