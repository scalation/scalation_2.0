
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 19:44:13 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Core Operations for Automatic Differentiation
 */

package scalation
package modeling
package autograd

import scalation.mathstat.{TensorD, tensorize}
import scalation.modeling.ActivationFun

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AutogradOps` trait defines the core operations needed for automatic differentiation.
 *  It separates the mathematical operations on tensors (TensorD)
 *  from the autograd system (Variable, Function), allowing flexible extension.
 *  This trait is backed by a default implementation (see AutogradOps.default)
 *  using TensorD methods.
 */
trait AutogradOps:

    // ---------- Arithmetic operations ----------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the sign of each element in tensor x.
     */
    def sign (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the absolute value of each element in tensor x.
     */
    def abs (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the negation of tensor x.
     */
    def neg (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the square root of each element in tensor x.
     */
    def sqrt (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the natural logarithm of each element in tensor x.
     */
    def log (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the logarithm of tensor x with the specified base.
     */
    def logBase (x: TensorD, base: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the reciprocal of each element in tensor x.
     */
    def reciprocal (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clips the elements of tensor x to be within the range [min, max].
     */
    def clipByValue (x: TensorD, min: Double, max: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clips the elements of tensor x to have a maximum norm of maxNorm.
     */
    def clipByNorm (x: TensorD, maxNorm: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the element-wise maximum of tensors x and y.
     */
    def max (x: TensorD, y: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the element-wise maximum between tensor x and scalar s.
     */
    def maxScalar (x: TensorD, s: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the maximum value in tensor x.
     */
    def maxValue (x: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the element-wise minimum of tensors x and y.
     */
    def min (x: TensorD, y: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the element-wise minimum between tensor x and scalar s.
     */
    def minScalar (x: TensorD, s: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the minimum value in tensor x.
     */
    def minValue (x: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the mean of all elements in tensor x.
     */
    def mean (x: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the mean along the specified axis of tensor x.
     */
    def meanAlongAxis (x: TensorD, axis: Int): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the Frobenius norm of tensor x.
     */
    def normF (x: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the Frobenius norm squared of tensor x.
     */
    def normFSq (x: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the variance of all elements in tensor x.
     */
    def variance (x: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the variance along the specified axis of tensor x.
     */
    def varianceAlongAxis (x: TensorD, axis: Int): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the standard deviation of tensor x.
     */
    def std (x: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the standard deviation along the specified axis of tensor x.
     */
    def stdAlongAxis (x: TensorD, axis: Int): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Standardizes tensor x along the specified axis.
     */
    def standardize (x: TensorD, axis: Int): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rounds each element in tensor x to the nearest integer.
     */
    def round (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Applies ceiling to each element in tensor x.
     */
    def ceil (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Applies floor to each element in tensor x.
     */
    def floor (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns element-wise addition of tensors x and y.
     */
    def add (x: TensorD, y: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Adds scalar s to each element in tensor x.
     */
    def addScalar (x: TensorD, s: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns element-wise subtraction of tensor y from tensor x.
     */
    def sub (x: TensorD, y: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtracts scalar s from each element in tensor x.
     */
    def subScalar (x: TensorD, s: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns element-wise multiplication of tensors x and y.
     */
    def mul (x: TensorD, y: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Multiplies each element in tensor x by scalar s.
     */
    def mulScalar (x: TensorD, s: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns element-wise division of tensor x by tensor y.
     */
    def div (x: TensorD, y: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Divides each element in tensor x by scalar s.
     */
    def divScalar (x: TensorD, s: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Raises each element in tensor x to the power of s.
     */
    def pow (x: TensorD, s: Int): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the exponential of each element in tensor x.
     */
    def exp (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the sum of all elements in tensor x.
     */
    def sum (x: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the sum along the specified axis of tensor x.
     */
    def sumAlongAxis (x: TensorD, axis: Int): TensorD

    // ---------- Tensor operations ----------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Creates a tensor consisting of scalar value s.
     */
    def scalar (s: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Creates a tensor with the same shape as x filled with zeros.
     */
    def zerosLike (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Creates a tensor with the same shape as x filled with ones.
     */
    def onesLike (x: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Creates a tensor with the same shape as t filled with the specified value.
     */
    def fullLike (t: TensorD, value: Double): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the shape of tensor x as a list of dimension sizes.
     */
    def shape (x: TensorD): List [Int]

    def getSlice (x: TensorD, r0: Range, r1: Range, r2: Range): TensorD

    def setSlice (x: TensorD, value: TensorD, r0: Range, r1: Range, r2: Range): TensorD

    def concat (tensors: Seq [TensorD], axis: Int): TensorD

    def reshape (x: TensorD, newShape: Seq [Int]): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Permutes the axes of tensor x according to the specified order.
     */
    def permute (x: TensorD, axes: Seq [Int]): TensorD

    // FIX: Gelu doesn't pass the test
    // ---------- Activation functions ----------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Identity activation function.
     */
    def id_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rectified Linear Unit (ReLU) activation function.
     */
    def reLU_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Leaky ReLU activation function with an optional alpha parameter.
     */
    def lreLU_ (yp: TensorD, alpha: Double = 0.2): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Exponential Linear Unit (ELU) activation function with an optional alpha parameter.
     */
    def eLU_ (yp: TensorD, alpha: Double = 1.0): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Hyperbolic tangent (tanh) activation function.
     */
    def tanh_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sigmoid activation function.
     */
    def sigmoid_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Gaussian activation function.
     */
    def gaussian_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Gaussian Error Linear Unit (GeLU) activation function.
     */
    def geLU_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Softmax activation function.
     */
    def softmax_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Logit activation function.
     */
    def logit_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Logistic activation function with parameters a, b, and c.
     */
    def logistic_ (yp: TensorD, a: Double = 1.0, b: Double = 1.0, c: Double = 1.0): TensorD

    // ---------- Activation Function Derivatives ----------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Derivative of the identity activation function.
     */
    def idD_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Derivative of the ReLU activation function.
     */
    def reLUD_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Derivative of the Leaky ReLU activation function.
     */
    def lreLUD_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Derivative of the ELU activation function with an optional alpha parameter.
     */
    def eLUD_ (yp: TensorD, alpha: Double = 1.0): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Derivative of the tanh activation function.
     */
    def tanhD_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Derivative of the sigmoid activation function.
     */
    def sigmoidD_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Derivative of the GeLU activation function.
     */
    def geLUD_ (yp: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Derivative of the softmax activation function.
     */
    def softmaxD_ (yp: TensorD): TensorD

    // ---------- Loss functions ----------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the Sum of Squared Errors (SSE) loss between the prediction and target tensors.
     */
    def sseLoss (pred: TensorD, target: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the Mean Squared Error (MSE) loss between the prediction and target tensors.
     */
    def mseLoss (pred: TensorD, target: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the Mean Absolute Error (MAE) loss between the prediction and target tensors.
     */
    def maeLoss (pred: TensorD, target: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the binary cross entropy loss between the prediction and target tensors.
     */
    def binaryCrossEntropy (pred: TensorD, target: TensorD): Double

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the categorical cross entropy loss between the prediction and target tensors.
     */
    def categoricalCrossEntropy (pred: TensorD, target: TensorD): Double

    // ---------- Matrix-like operations ----------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transposes tensor x by swapping the specified axes i and j.
     */
    def transpose (x: TensorD, i: Int, j: Int): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the dot product of tensors x and y.
     */
    def dot (x: TensorD, y: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs matrix multiplication of tensors x and y.
     */
    def matmul (x: TensorD, y: TensorD): TensorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs batched matrix multiplication of tensors x and y.
     */
    def bmm (x: TensorD, y: TensorD): TensorD

end AutogradOps


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Companion object for AutogradOps that provides a default implementation.
 */
object AutogradOps:

    /** Default instance of AutogradOps.
     */
    given default: AutogradOps with

        // ---------- Arithmetic Operations ----------

        def sign (x: TensorD): TensorD = x.sign

        def abs (x: TensorD): TensorD = x.abs

        def neg (x: TensorD): TensorD = -x

        def sqrt (x: TensorD): TensorD = x.sqrt

        def log (x: TensorD): TensorD = x.log

        def logBase (x: TensorD, base: Double): TensorD = x.logBase (base)

        def reciprocal (x: TensorD): TensorD = x.reciprocal

        def clipByValue (x: TensorD, min: Double, max: Double): TensorD = x.clipByValue (min, max)

        def clipByNorm (x: TensorD, maxNorm: Double): TensorD = x.clipByNorm (maxNorm)

        def max (x: TensorD, y: TensorD): TensorD = TensorD.max (x, y)

        def maxScalar (x: TensorD, s: Double): TensorD = x.maxScalar (s)

        def maxValue(x: TensorD): Double = x.maxValue

        def min (x: TensorD, y: TensorD): TensorD = TensorD.min (x, y)

        def minScalar (x: TensorD, s: Double): TensorD = x.minScalar (s)

        def minValue (x: TensorD): Double = x.minValue

        def mean (x: TensorD): Double = x.mean

        def meanAlongAxis (x: TensorD, axis: Int): TensorD = x.meanAlongAxis (axis)

        def normF (x: TensorD): Double = x.normF

        def normFSq (x: TensorD): Double = x.normFSq

        def variance (x: TensorD): Double = x.variance

        def varianceAlongAxis (x: TensorD, axis: Int): TensorD = x.varianceAlongAxis (axis)

        def std (x: TensorD): Double = x.std

        def stdAlongAxis (x: TensorD, axis: Int): TensorD = x.stdAlongAxis (axis)

        def standardize (x: TensorD, axis: Int): TensorD = x.standardize (axis)

        def round (x: TensorD): TensorD = x.round

        def ceil (x: TensorD): TensorD = x.ceil

        def floor (x: TensorD): TensorD = x.floor

        def add (x: TensorD, y: TensorD): TensorD = x + y

        def addScalar (x: TensorD, s: Double): TensorD = x + s

        def sub (x: TensorD, y: TensorD): TensorD = x - y

        def subScalar (x: TensorD, s: Double): TensorD = x - s

        def mul (x: TensorD, y: TensorD): TensorD = x * y

        def mulScalar (x: TensorD, s: Double): TensorD = x * s

        def div (x: TensorD, y: TensorD): TensorD = x / y

        def divScalar (x: TensorD, s: Double): TensorD = x / s

        def pow (x: TensorD, s: Int): TensorD = x ~^ s

        def exp (x: TensorD): TensorD = x.exp

        def sum (x: TensorD): Double = x.sum

        def sumAlongAxis (x: TensorD, axis: Int): TensorD = x.sumAlongAxis (axis)

        // ---------- Tensor Operations ----------

        def scalar (s: Double): TensorD = TensorD.scalar (s)

        def zerosLike (x: TensorD): TensorD = x.zerosLike

        def onesLike (x: TensorD): TensorD = x.onesLike

        def fullLike (t: TensorD, value: Double): TensorD = t.fullLike (value)

        def shape (x: TensorD): List [Int] = x.shape

        def getSlice (x: TensorD, r0: Range, r1: Range, r2: Range): TensorD = x(r0, r1, r2)

        def setSlice (x: TensorD, value: TensorD, r0: Range, r1: Range, r2: Range): TensorD =
            x(r0, r1, r2) = value
            x
        end setSlice

        def concat (tensors: Seq [TensorD], axis: Int): TensorD = TensorD.concat (tensors, axis)

        def reshape (x: TensorD, newShape: Seq [Int]): TensorD = x.reshape (newShape)

        def permute (x: TensorD, axes: Seq [Int]): TensorD = x.permute (axes)

        // ---------- Activation Functions ----------

        def id_ (yp: TensorD): TensorD = yp.id

        def reLU_ (yp: TensorD): TensorD = yp.reLU

        def lreLU_ (yp: TensorD, alpha: Double = 0.2): TensorD = yp.lreLU (alpha)

        def eLU_ (yp: TensorD, alpha: Double = 1.0): TensorD = yp.eLU (alpha)

        def tanh_ (yp: TensorD): TensorD = yp.tanh

        def sigmoid_ (yp: TensorD): TensorD = yp.sigmoid

        def gaussian_ (yp: TensorD): TensorD = yp.gaussian

        def geLU_ (yp: TensorD): TensorD = yp.geLU

        def softmax_ (yp: TensorD): TensorD = yp.softmax

        def logit_ (yp: TensorD): TensorD = yp.logit

        def logistic_ (yp: TensorD, a: Double = 1.0, b: Double = 1.0, c: Double = 1.0): TensorD =
            yp.logistic (a, b, c)

        // ---------- Activation Function Derivatives ----------

        def idD_ (yp: TensorD): TensorD = yp.onesLike

        def reLUD_ (yp: TensorD): TensorD = tensorize (ActivationFun.reLUD)(yp)

        def lreLUD_ (yp: TensorD): TensorD = tensorize (ActivationFun.lreLUD)(yp)

        def eLUD_ (yp: TensorD, alpha: Double = 1.0): TensorD = tensorize (ActivationFun.eLUD)(yp)

        def tanhD_ (yp: TensorD): TensorD = tensorize (ActivationFun.tanhD)(yp)

        def sigmoidD_ (yp: TensorD): TensorD = tensorize (ActivationFun.sigmoidD)(yp)

        def geLUD_ (yp: TensorD): TensorD = tensorize (ActivationFun.geLUD)(yp)

        def softmaxD_ (yp: TensorD): TensorD = tensorize (ActivationFun.softmaxD)(yp)

        // ---------- Loss Functions ----------

        def sseLoss (pred: TensorD, target: TensorD): Double =
            val e = pred - target
            (e ~^ 2).sum

        def mseLoss (pred: TensorD, target: TensorD): Double =
            val e = pred - target
            (e ~^ 2).mean

        def maeLoss (pred: TensorD, target: TensorD): Double =
            val e = pred - target
            e.abs.mean

        def binaryCrossEntropy (pred: TensorD, target: TensorD): Double =
            val eps = 1e-15
            val predSafe = pred.clipByValue (eps, 1.0 - eps)
            val loss = TensorD.zerosLike (pred)
            cfor (pred.indices) { i =>
                cfor (pred.indices2) { j =>
                    cfor (pred.indices3) { k =>
                        loss(i, j, k) =
                            -target(i, j, k) * math.log (predSafe(i, j, k)) -
                                (1.0 - target(i, j, k)) * math.log (1.0 - predSafe(i, j, k))
                    } // cfor
                } // cfor
            } // cfor
            loss.sum / (pred.dim * pred.dim2 * pred.dim3)

        def categoricalCrossEntropy (pred: TensorD, target: TensorD): Double =
            val eps = 1e-15
            val predSafe = pred.clipByValue (eps, 1.0 - eps)
            var totalLoss = 0.0
            cfor (pred.indices) { i =>
                cfor (pred.indices2) { j =>
                    var sampleLoss = 0.0
                    cfor (pred.indices3) { k =>
                        sampleLoss += -target(i, j, k) * math.log (predSafe(i, j, k))
                    } // cfor
                    totalLoss += sampleLoss
                } // cfor
            } // cfor
            totalLoss / (pred.dim * pred.dim2)

        // ---------- Matrix-like Operations ----------

        def transpose (x: TensorD, i: Int, j: Int): TensorD = x.transpose(i, j)

        def dot (x: TensorD, y: TensorD): TensorD = x dot y

        def matmul (x: TensorD, y: TensorD): TensorD = x matmul y

        def bmm (x: TensorD, y: TensorD): TensorD = x bmm y

    end default

end AutogradOps

