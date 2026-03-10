
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 19:46:13 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Base Trait and Operations for Differentiable Functions
 */

package scalation
package modeling
package autograd

import scala.compiletime.uninitialized

import scalation.mathstat.TensorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** A utility object for generating unique numeric IDs for `Function` nodes.
 *  This is primarily used for debugging and visualization purposes in the autograd system.
 */
private [autograd] object FunctionIdGen:

    /** Counter to keep track of the current ID.
     */
    private var c = 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generates the next unique ID by incrementing the counter.
     *  @return the next unique integer ID
     */
    def next (): Int = { c += 1; c }

end FunctionIdGen


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Function` base trait for all differentiable operations in the autograd system.
 *  A Function encapsulates both the forward computation (producing outputs)
 *  and the backward computation (propagating gradients).
 *  It also provides utility methods for handling unbroadcasting of shapes
 *  during the backward pass, ensuring correct gradient flow.
 *  Every custom operation should extend this trait and implement `forward` and `backward`.
 */
trait Function (using ops: AutogradOps):

    /** Unique numeric ID for this Function node (for graph viz/debugging).
     */
    val id: Int = FunctionIdGen.next ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Human-readable name of this op (defaults to simple class name).
     */
    def opName: String =
        val n = this.getClass.getSimpleName
        if n.endsWith ("$") then n.dropRight (1) else n

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map of attributes for visualization/debugging (default: empty).
     */
    def attributes: Map [String, String] = Map.empty

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Returns the input variables to this Function.
     *  This works automatically for all case-class ops by iterating over their
     *  constructor fields and collecting those of type `Variabl`.
     *  @see Case classes and Product: https://scala-lang.org/api/3.x/scala/Product.html
     */
    def inputs: Seq [Variabl] = this match
        case p: Product =>
            p.productIterator.flatMap {
                case v: Variabl => Seq (v)
                case s: Seq[?]  => s.collect { case v: Variabl => v }
                case _          => Seq.empty
            }.toSeq
        case _ => Seq.empty
    end inputs

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs the forward pass to compute the output variable.
     *  @return a Variabl containing the output data and gradient function.
     */
    def forward (): Variabl

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs the backward pass given the upstream gradient.
     *  @param gradOutput  the gradient tensor from the next layer.
     */
    def backward (gradOutput: TensorD): Unit

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backpropagates gradients for functions with two inputs.
     *  @param v1            the first input variable.
     *  @param v2            the second input variable.
     *  @param gradOutput    the upstream gradient tensor.
     *  @param computeGrad1  function to compute the gradient for v1.
     *  @param computeGrad2  function to compute the gradient for v2.
     */
    def backpropForTwoInputs (v1: Variabl, v2: Variabl, gradOutput: TensorD,
                              computeGrad1: TensorD => TensorD, computeGrad2: TensorD => TensorD): Unit =
        v1.backward (unbroadcast (computeGrad1 (gradOutput), v1.shape))
        v2.backward (unbroadcast (computeGrad2 (gradOutput), v2.shape))
    end backpropForTwoInputs

    // Possible improvements: Add requiresGrad and retainGraph options...

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Unbroadcasts a variable's tensor data to a specified old shape.
     *  @param v         the variable to unbroadcast.
     *  @param oldShape  the target shape.
     *  @return a new Variabl with data unbroadcasted.
     */
    def unbroadcast (v: Variabl, oldShape: List [Int]): Variabl =
        Variabl (unbroadcast (data = v.data, oldShape = oldShape))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Unbroadcasts a tensor to a given shape by summing across reduced dimensions.
     *  @param data      the tensor data.
     *  @param oldShape  the original shape.
     *  @return a TensorD with shape adjusted to oldShape.
     *  @throws Exception if unbroadcasting is not feasible.
     */
    def unbroadcast (data: TensorD, oldShape: List [Int]): TensorD =
        val currentShape = ops.shape(data)
        var cur = data
        for i <- oldShape.indices do
            val (oldDim, newDim) = (oldShape(i), currentShape(i))
            if oldDim == newDim then {}                          // no change if dimensions match
            else if oldDim == 1 then
                cur = ops.sumAlongAxis (cur, i)                  // reduce dimension i by summing
            else if oldDim != newDim then
                throw new Exception (
                    s"Cannot unbroadcast from shape $currentShape to $oldShape at axis $i")
        cur
    end unbroadcast

end Function

// -----------------------------------------------------------------------
// ------------------------- ARITHMETIC FUNCTIONS ------------------------
// -----------------------------------------------------------------------

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the element-wise absolute value of a variable.
 *  @param v  the input variable.
 */
case class Abs (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the absolute value of v.
     *  @return a new Variabl containing |v|.
     */
    override def forward (): Variabl = Variabl (ops.abs (v.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: applies the chain rule using the sign of v.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit = v.backward (ops.mul (gradOutput, ops.sign (v.data)))
end Abs

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the negation of a variable.
 *  @param v  the input variable.
 */
case class Neg (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the negation of v.
     *  @return a Variabl containing -v.
     */
    override def forward (): Variabl = Variabl (ops.neg (v.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the negated gradient.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit = v.backward (ops.neg (gradOutput))
end Neg

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the floor of a variable (element-wise).
 *  @param v  the input variable.
 */
case class Floor (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes floor(v).
     *  @return a Variabl with floor applied.
     */
    override def forward (): Variabl = Variabl (ops.floor (v.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: gradient is zero almost everywhere (undefined at integers).
     *  @note:  The derivative of floor is zero almost everywhere
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit = v.backward (ops.zerosLike (v.data))
end Floor

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the ceil of a variable (element-wise).
 *  @param v  the input variable.
 */
case class Ceil (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes ceil(v).
     *  @return a Variabl with ceil applied.
     */
    override def forward (): Variabl = Variabl (ops.ceil (v.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: gradient is zero almost everywhere (undefined at integers).
     *  @note:  The derivative of ceil is zero almost everywhere
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit = v.backward (ops.zerosLike (v.data))
end Ceil

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the round of a variable (element-wise).
 *  @param v  the input variable.
 */
case class Round (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes round(v).
     *  @return a Variabl with round applied.
     */
    override def forward (): Variabl = Variabl (ops.round (v.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: gradient is zero almost everywhere (undefined at half-integers).
     *  @note:  The derivative of round is zero almost everywhere
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit = v.backward (ops.zerosLike (v.data))
end Round

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Clips the elements of a variable to the range [min, max] (element-wise).
 *  Gradient is 1 for elements strictly inside (min, max), 0 for clipped ones (ties get 0.25 via mask product heuristic).
 *  @param v   the input variable.
 *  @param min lower bound.
 *  @param max upper bound.
 */
case class Clip (v: Variabl, min: Double, max: Double)(using ops: AutogradOps) extends Function:

    override def forward (): Variabl = Variabl (ops.clipByValue (v.data, min, max), gradFn = Some (this))

    override def backward (gradOutput: TensorD): Unit =
        // diffLow > 0 and diffHigh > 0 indicates interior points
        val diffLow  = ops.subScalar (v.data, min)                          // v - min
        val diffHigh = ops.sub (ops.fullLike(v.data, max), v.data)          // max - v
        val signLow  = ops.sign (diffLow)
        val signHigh = ops.sign (diffHigh)
        val maskLow  = ops.mulScalar (ops.addScalar (signLow, 1.0), 0.5)    // 1 where v>min
        val maskHigh = ops.mulScalar (ops.addScalar (signHigh, 1.0), 0.5)   // 1 where v<max
        val interior = ops.mul (maskLow, maskHigh)                          // interior mask
        v.backward (ops.mul (gradOutput, interior))
end Clip

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the sign function element-wise.
 *  Derivative is zero almost everywhere (undefined at zero).
 *  @param v  the input variable.
 */
case class Sign (v: Variabl)(using ops: AutogradOps) extends Function:

    override def forward (): Variabl = Variabl (ops.sign (v.data), gradFn = Some (this))

    override def backward (gradOutput: TensorD): Unit = v.backward (ops.zerosLike (v.data))
end Sign

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Element-wise maximum of two variables.
 *  Gradient flows to the larger input; ties split as 0.5 / 0.5.
 *  @param v1  first input.
 *  @param v2  second input.
 */
case class Max (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    override def forward (): Variabl = Variabl (ops.max (v1.data, v2.data), gradFn = Some (this))

    override def backward (gradOutput: TensorD): Unit =
        val diff     = ops.sub (v1.data, v2.data)
        val signDiff = ops.sign (diff)                                      // -1,0,1
        val mask1    = ops.mulScalar (ops.addScalar (signDiff, 1.0), 0.5)   // 1 if v1>v2, 0 if v1<v2, 0.5 tie
        val ones     = ops.onesLike (mask1)
        val mask2    = ops.sub (ones, mask1)
        val g1 = unbroadcast (ops.mul (gradOutput, mask1), v1.shape)
        val g2 = unbroadcast (ops.mul (gradOutput, mask2), v2.shape)
        v1.backward (g1)
        v2.backward (g2)
end Max

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Element-wise minimum of two variables.
 *  Gradient flows to the smaller input; ties split as 0.5 / 0.5.
 *  @param v1  first input.
 *  @param v2  second input.
 */
case class Min (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    override def forward (): Variabl = Variabl (ops.min (v1.data, v2.data), gradFn = Some (this))

    override def backward (gradOutput: TensorD): Unit =
        val diff     = ops.sub (v1.data, v2.data)
        val signDiff = ops.sign (diff)                                // -1,0,1
        val ones     = ops.onesLike (signDiff)
        val mask1    = ops.mulScalar (ops.sub(ones, signDiff), 0.5)   // 1 if v1<v2, 0 if v1>v2, 0.5 tie
        val mask2    = ops.sub (ones, mask1)
        val g1 = unbroadcast (ops.mul (gradOutput, mask1), v1.shape)
        val g2 = unbroadcast (ops.mul (gradOutput, mask2), v2.shape)
        v1.backward (g1)
        v2.backward (g2)
end Min

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Element-wise maximum between a variable and a scalar.
 *  Gradient is 1 where v > s; 0 where v < s; 0.5 where equal.
 *  @param v  the input variable.
 *  @param s  the scalar.
 */
case class MaxScalar (v: Variabl, s: Double)(using ops: AutogradOps) extends Function:

    override def forward (): Variabl = Variabl (ops.maxScalar (v.data, s), gradFn = Some (this))

    override def backward (gradOutput: TensorD): Unit =
        val diff = ops.subScalar (v.data, s)                                // v - s
        val mask = ops.mulScalar (ops.addScalar (ops.sign(diff), 1.0), 0.5)
        v.backward (ops.mul (gradOutput, mask))
end MaxScalar

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Element-wise minimum between a variable and a scalar.
 *  Gradient is 1 where v < s; 0 where v > s; 0.5 where equal.
 *  @param v  the input variable.
 *  @param s  the scalar.
 */
case class MinScalar (v: Variabl, s: Double)(using ops: AutogradOps) extends Function:

    override def forward (): Variabl = Variabl (ops.minScalar (v.data, s), gradFn = Some (this))

    override def backward (gradOutput: TensorD): Unit =
        val diff  = ops.subScalar (v.data, s)                               // v - s
        val signD = ops.sign (diff)
        val ones  = ops.onesLike (signD)
        val mask  = ops.mulScalar (ops.sub(ones, signD), 0.5)               // (1 - sign)/2
        v.backward (ops.mul (gradOutput, mask))
end MinScalar

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the maximum value in a variable (reduces to a scalar).
 *  Gradient is distributed equally among all elements achieving the max (handles ties).
 *  @param v  the input variable.
 */
case class MaxValue (v: Variabl)(using ops: AutogradOps) extends Function:

    private var maxVal: Double = Double.NaN

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes max(v) -> scalar tensor.
     *  @return a scalar Variabl containing the maximum value.
     */
    override def forward (): Variabl =
        maxVal = ops.maxValue (v.data)
        Variabl (ops.scalar (maxVal), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: gradient flows to max positions; ties split 1/k.
     *  @param gradOutput upstream scalar gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        // sign(v - max) gives 0 at max positions, negative elsewhere.
        val diff     = ops.subScalar (v.data, maxVal)
        val signDiff = ops.sign (diff)
        val absSign  = ops.abs (signDiff)
        val ones     = ops.onesLike (absSign)
        val mask     = ops.sub (ones, absSign)         // 1 at maxima, 0 elsewhere
        val k = ops.sum (mask) match
            case 0.0 => 1.0                            // safety (should not happen)
            case kk => kk
        val normMask = ops.divScalar (mask, k)
        val scaled   = ops.mulScalar (normMask, gradOutput(0)(0)(0))
        v.backward (scaled)
end MaxValue

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the minimum value in a variable (reduces to a scalar).
 *  Gradient is distributed equally among all elements achieving the min (handles ties).
 *  @param v  the input variable.
 */
case class MinValue (v: Variabl)(using ops: AutogradOps) extends Function:

    private var minVal: Double = Double.NaN

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes min(v) -> scalar tensor.
     *  @return a scalar Variabl containing the minimum value.
     */
    override def forward (): Variabl =
        minVal = ops.minValue (v.data)
        Variabl (ops.scalar (minVal), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: gradient flows to min positions; ties split 1/k.
     *
     * @param gradOutput upstream scalar gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val diff     = ops.subScalar (v.data, minVal)     // 0 at minima, positive elsewhere
        val signDiff = ops.sign (diff)                    // 0 at minima, 1 elsewhere
        val absSign  = ops.abs (signDiff)
        val ones     = ops.onesLike (absSign)
        val mask     = ops.sub (ones, absSign)            // 1 at minima
        val k = ops.sum (mask) match
            case 0.0 => 1.0
            case kk => kk
        val normMask = ops.divScalar (mask, k)
        val scaled   = ops.mulScalar (normMask, gradOutput(0)(0)(0))
        v.backward (scaled)
end MinValue

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the square root of a variable.
 *  @param v  the input variable.
 */
case class Sqrt (v: Variabl)(using ops: AutogradOps) extends Function:

    private var sqrtCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the square root of v.
     *  @return a Variabl containing sqrt(v).
     */
    override def forward (): Variabl =
        sqrtCache = Some (ops.sqrt (v.data))
        Variabl (sqrtCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradient using the derivative of sqrt.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.div (gradOutput, ops.mulScalar (sqrtCache.get, 2.0)))
end Sqrt

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the natural logarithm of a variable.
 *  @param v  the input variable.
 */
case class Log (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes log(v).
     *  @return a Variabl containing log(v).
     */
    override def forward (): Variabl = Variabl (ops.log (v.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: applies the derivative 1/v.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.div (gradOutput, v.data))
end Log

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the reciprocal of a variable.
 *  @param v  the input variable.
 */
case class Reciprocal (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the reciprocal of v.
     *  @return a Variabl containing 1/v.
     */
    override def forward (): Variabl = Variabl (ops.reciprocal (v.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: computes the derivative -1/v&#94;2.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.div (ops.neg (gradOutput), ops.pow(v.data, 2)))

end Reciprocal

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the logarithm of a variable with a specified base.
 *  @param v     the input variable.
 *  @param base  the base for the logarithm.
 */
case class LogBase (v: Variabl, base: Double)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes log base 'base' of v.
     *  @return a Variabl containing log_base(v).
     */
    override def forward (): Variabl = Variabl (ops.logBase (v.data, base), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: adjusts the gradient by dividing by (v * log(base)).
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val denominator = ops.mulScalar (v.data, math.log (base))
        v.backward (ops.div (gradOutput, denominator))

end LogBase

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the sum of all elements in a variable.
 *  @param v  the input variable.
 */
case class Sum (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the sum and returns a scalar tensor.
     *  @return a Variabl containing the sum as a scalar.
     */
    override def forward (): Variabl = Variabl (ops.scalar (ops.sum(v.data)), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient by filling a tensor with the scalar gradient.
     *  @param gradOutput  the upstream gradient (scalar).
     */
    override def backward (gradOutput: TensorD): Unit =
        val grad = ops.fullLike (v.data, gradOutput(0)(0)(0))
        v.backward (grad)
end Sum

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes element-wise addition of two variables.
 *  @param v1  the first variable.
 *  @param v2  the second variable.
 */
case class Add (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v1 + v2.
     *  @return a Variabl containing the sum.
     */
    override def forward (): Variabl = Variabl (ops.add (v1.data, v2.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradient to both inputs (unbroadcast if necessary).
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val g1 = unbroadcast (gradOutput, v1.shape)
        v1.backward (g1)
        val g2 = unbroadcast (gradOutput, v2.shape)
        v2.backward (g2)
end Add

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Adds a constant value to a variable.
 *  @param v  the input variable.
 *  @param d  the constant to add.
 */
case class AddConstant (v: Variabl, d: Double)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v + d.
     *  @return a Variabl with the constant added.
     */
    override def forward (): Variabl = Variabl (ops.addScalar (v.data, d), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: simply propagates the upstream gradient.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit = v.backward (gradOutput)
end AddConstant

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes element-wise subtraction of two variables.
 *  @param v1  the minuend.
 *  @param v2  the subtrahend.
 */
case class Sub (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v1 - v2.
     *  @return a Variabl containing the difference.
     */
    override def forward (): Variabl = Variabl (ops.sub (v1.data, v2.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradient to v1 normally and to v2 as negative.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val g1 = unbroadcast (gradOutput, v1.shape)
        v1.backward (g1)
        val g2 = unbroadcast (gradOutput, v2.shape)
        v2.backward (ops.neg (g2))
end Sub

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Subtracts a constant value from a variable.
 *  @param v  the input variable.
 *  @param d  the constant to subtract.
 */
case class SubConstant (v: Variabl, d: Double)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v - d.
     *  @return a Variabl with the constant subtracted.
     */
    override def forward (): Variabl = Variabl (ops.subScalar (v.data, d), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: simply propagates the upstream gradient.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit = v.backward (gradOutput)
end SubConstant

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes element-wise multiplication of two variables.
 *  @param v1  the first variable.
 *  @param v2  the second variable.
 */
case class Mul (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v1 * v2.
     *  @return a Variabl containing the product.
     */
    override def forward (): Variabl = Variabl (ops.mul (v1.data, v2.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: uses the chain rule to propagate gradients.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        backpropForTwoInputs (v1, v2, gradOutput,
                             (g: TensorD) => ops.mul (g, v2.data),
                             (g: TensorD) => ops.mul (v1.data, g))
end Mul

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Multiplies a variable by a constant.
 *  @param v  the input variable.
 *  @param d  the constant multiplier.
 */
case class MulConstant (v: Variabl, d: Double)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v * d.
     *  @return a Variabl with data scaled by d.
     */
    override def forward (): Variabl = Variabl (ops.mulScalar (v.data, d), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: multiplies the gradient by d.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val g = ops.mulScalar (gradOutput, d)
        v.backward (g)
end MulConstant

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes element-wise division of two variables.
 *  @param v1  the dividend.
 *  @param v2  the divisor.
 */
case class Div (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v1 / v2.
     *  @return a Variabl with divided data.
     */
    override def forward (): Variabl = Variabl (ops.div(v1.data, v2.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradients with appropriate adjustments.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        backpropForTwoInputs (v1, v2, gradOutput,
                             (g: TensorD) => ops.div (g, v2.data),
                             (g: TensorD) => ops.div (ops.mul (ops.neg (v1.data), g), ops.pow (v2.data, 2)))
end Div

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Divides a variable by a constant.
 *  @param v  the input variable.
 *  @param d  the constant divisor.
 */
case class DivConstant (v: Variabl, d: Double)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v / d.
     *  @return a Variabl with data divided by d.
     */
    override def forward (): Variabl = Variabl (ops.divScalar (v.data, d), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradient scaled by 1/d.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val g = ops.divScalar (gradOutput, d)
        v.backward (g)
end DivConstant

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Raises a variable to an integer power.
 *  @param v  the input variable.
 *  @param s  the exponent.
 */
case class Pow (v: Variabl, s: Int)(using ops: AutogradOps) extends Function:

    private var powCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes v raised to the power s.
     *  @return a Variabl with powered data.
     */
    override def forward (): Variabl =
        powCache = Some (ops.pow (v.data, s))
        Variabl (powCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: applies the derivative of the power function.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val factor = ops.div (powCache.get, v.data)
        v.backward (ops.mul (ops.mulScalar (gradOutput, s), factor))
end Pow

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the exponential of a variable.
 *  @param v  the input variable.
 */
case class Exp (v: Variabl)(using ops: AutogradOps) extends Function:

    private var expCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes exp(v).
     *  @return a Variabl containing the exponential.
     */
    override def forward (): Variabl =
        expCache = Some (ops.exp(v.data))
        Variabl (expCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradient scaled by the exponential.
     * @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.mul (gradOutput, expCache.get))
        expCache = None
end Exp

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the mean of all elements in a variable.
 *  @param v  the input variable.
 */
case class Mean (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the mean and fills a tensor with it.
     *  @return a Variabl with data filled by the mean value.
     */
    override def forward (): Variabl =
        val out = ops.mean (v.data)
        Variabl (ops.scalar (out), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: scales the gradient and fills a tensor accordingly.
     *  @param gradOutput the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val n = ops.shape (v.data).product.toDouble
        v.backward (ops.fullLike (v.data, gradOutput(0)(0)(0) / n))
end Mean

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the variance of all elements in a variable (population variance).
 *  Uses definition Var(x) = mean((x - mean(x))^2).
 *  @param v  the input variable.
 */
case class Variance (v: Variabl)(using ops: AutogradOps) extends Function:

    private var meanCache: Double = Double.NaN
    private var varCache: Double  = Double.NaN

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes variance over all elements -> scalar tensor.
     *  @return a scalar Variabl containing variance.
     */
    override def forward (): Variabl =
        meanCache    = ops.mean (v.data)
        val centered = ops.subScalar (v.data, meanCache)
        val sq       = ops.pow (centered, 2)
        varCache     = ops.mean (sq)
        Variabl (ops.scalar (varCache), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: dVar/dx = 2 (x - mean)/N.
     *  @param gradOutput upstream scalar gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val n     = v.shape.product.toDouble
        val diff  = ops.subScalar (v.data, meanCache)
        val coeff = (2.0 / n) * gradOutput(0)(0)(0)
        val grad  = ops.mulScalar (diff, coeff)
        v.backward (grad)
end Variance

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the standard deviation of all elements in a variable.
 *  Std(x) = sqrt(Var(x)); derivative ds/dx = (x - mean)/(N * std).
 *  @param v  the input variable.
 */
case class Std (v: Variabl)(using ops: AutogradOps) extends Function:

    private var meanCache: Double = Double.NaN
    private var varCache: Double  = Double.NaN
    private var stdCache: Double  = Double.NaN

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes std over all elements -> scalar tensor.
     *  @return a scalar Variabl containing std.
     */
    override def forward (): Variabl =
        meanCache    = ops.mean (v.data)
        val centered = ops.subScalar (v.data, meanCache)
        val sq       = ops.pow (centered, 2)
        varCache     = ops.mean (sq)
        stdCache     = math.sqrt (varCache)
        Variabl (ops.scalar (stdCache), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: dStd/dx = (x - mean)/(N * std). If std=0 => gradient 0.
     *
     * @param gradOutput upstream scalar gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val n = v.shape.product.toDouble
        if stdCache == 0.0 || stdCache.isNaN then
            v.backward (ops.zerosLike (v.data))
        else
            val diff  = ops.subScalar (v.data, meanCache)
            val coeff = gradOutput(0)(0)(0) / (n * stdCache)
            val grad  = ops.mulScalar (diff, coeff)
            v.backward (grad)
end Std

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the mean of a variable along a specified axis (dimension reduced to size 1).
 *  @param v     the input variable.
 *  @param axis  the axis along which to compute the mean.
 */
case class MeanAlongAxis(v: Variabl, axis: Int)(using ops: AutogradOps) extends Function:

    private var nAxis: Int = 0

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: mean along axis -> keeps dimensionality (axis size becomes 1).
     *  @return a Variabl containing the reduced mean tensor.
     */
    override def forward (): Variabl =
        nAxis = v.shape (axis)
        Variabl (ops.meanAlongAxis (v.data, axis), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: distributes gradient equally across the axis (broadcasted).
     *  @param gradOutput upstream gradient with axis size 1.
     */
    override def backward (gradOutput: TensorD): Unit =
        val scale = 1.0 / nAxis
        // Broadcast gradOutput over axis by multiplying with a tensor of ones matching v.shape.
        val expanded = ops.mul (ops.fullLike (v.data, 1.0), gradOutput)
        val grad     = ops.mulScalar (expanded, scale)
        v.backward (grad)
end MeanAlongAxis

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the variance of a variable along a specified axis (population variance).
 *  @param v     the input variable.
 *  @param axis  the axis along which to compute variance.
 */
case class VarianceAlongAxis (v: Variabl, axis: Int)(using ops: AutogradOps) extends Function:

    private var nAxis: Int = 0
    private var meanAxis: TensorD = uninitialized

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: variance along axis -> axis dimension becomes 1.
     *  @return a Variabl containing variance along axis.
     */
    override def forward (): Variabl =
        nAxis    = v.shape (axis)
        meanAxis = ops.meanAlongAxis (v.data, axis)
        val diff = ops.sub (v.data, meanAxis)
        val sq   = ops.pow (diff, 2)
        val varAxis = ops.meanAlongAxis (sq, axis)
        Variabl (varAxis, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: dVar/dx = 2*(x - mean)/n along the specified axis (broadcasted).
     *  @param gradOutput upstream gradient with axis size 1.
     */
    override def backward (gradOutput: TensorD): Unit =
        val coeff = 2.0 / nAxis
        val diff  = ops.sub (v.data, meanAxis)
        val base  = ops.mulScalar (diff, coeff)
        val grad  = ops.mul (base, gradOutput)             // broadcast gradOutput
        v.backward (grad)
end VarianceAlongAxis

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the standard deviation of a variable along a specified axis.
 *  @param v     the input variable.
 *  @param axis  the axis along which to compute std.
 */
case class StdAlongAxis (v: Variabl, axis: Int)(using ops: AutogradOps) extends Function:

    private var nAxis: Int = 0
    private var meanAxis: TensorD = uninitialized
    private var stdAxis: TensorD  = uninitialized

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: std along axis -> axis dimension becomes 1.
     *  @return a Variabl containing std along axis.
     */
    override def forward (): Variabl =
        nAxis    = v.shape (axis)
        meanAxis = ops.meanAlongAxis (v.data, axis)
        val diff = ops.sub (v.data, meanAxis)
        val sq   = ops.pow (diff, 2)
        val varAxis = ops.meanAlongAxis (sq, axis)

        // stdAxis = sqrt(varAxis) element-wise (varAxis axis size already 1).
        // Using reciprocal + pow not needed; rely on sqrt over broadcasting by reuse Sqrt op?
        // Simpler: convert to Variabl then sqrt not desired.
        // We'll leverage math.sqrt by mapping via TensorD operations already exposed as sqrt at AutogradOps?
        // Not available for tensor element wise with shape? We have ops.sqrt.

        stdAxis = ops.sqrt (varAxis)
        Variabl (stdAxis, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: dStd/dx = (x - mean)/(N * std). Handle std=0 with zero gradient.
     *  @param gradOutput upstream gradient with axis size 1.
     */
    override def backward (gradOutput: TensorD): Unit =
        // Avoid divide by zero: clamp stdAxis.
        val denom = ops.maxScalar (ops.mulScalar (stdAxis, nAxis.toDouble), 1e-12)
        val diff  = ops.sub (v.data, meanAxis)
        val base  = ops.div (diff, denom)
        val grad  = ops.mul (base, gradOutput)
        v.backward (grad)
end StdAlongAxis

// -----------------------------------------------------------------------
// ----------------------- ACTIVATION FUNCTIONS --------------------------
// -----------------------------------------------------------------------

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the identity activation function.
 *  @param v  the input variable.
 */
case class Identity (v: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: returns v unchanged.
     *  @return a Variabl with the same data as v.
     */
    override def forward (): Variabl = Variabl (ops.id_ (v.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the upstream gradient using the identity derivative.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.mul (gradOutput, ops.idD_ (v.data)))
end Identity

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the ReLU activation function.
 *  @param v the input variable.
 */
case class ReLU (v: Variabl)(using ops: AutogradOps) extends Function:

    private var reluCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the ReLU activation of v.
     *  @return a Variabl with ReLU applied.
     */
    override def forward (): Variabl =
        reluCache = Some (ops.reLU_(v.data))
        Variabl (reluCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient through ReLU.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.mul (gradOutput, ops.reLUD_ (reluCache.get)))
end ReLU

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the LeakyReLU activation function.
 *  @param v      the input variable.
 *  @param alpha  the negative slope coefficient.
 */
case class LeakyReLU (v: Variabl, alpha: Double = 0.2)(using ops: AutogradOps) extends Function:

    private var leakyReLUCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the LeakyReLU activation of v.
     *  @return a Variabl with LeakyReLU applied.
     */
    override def forward (): Variabl =
        leakyReLUCache = Some (ops.lreLU_(v.data, alpha))
        Variabl (leakyReLUCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient using the LeakyReLU derivative.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.mul (gradOutput, ops.lreLUD_ (leakyReLUCache.get)))
end LeakyReLU

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the ELU activation function.
 *  @param v      the input variable.
 *  @param alpha  the ELU scaling parameter.
 */
case class ELU (v: Variabl, alpha: Double = 1.0)(using ops: AutogradOps) extends Function:

    private var eluCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the ELU activation of v.
     *  @return a Variabl with ELU applied.
     */
    override def forward (): Variabl =
        eluCache = Some (ops.eLU_ (v.data, alpha))
        Variabl (eluCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient using the ELU derivative.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.mul (gradOutput, ops.eLUD_(eluCache.get, alpha)))
end ELU

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the tanh activation function.
 *  @param v  the input variable.
 */
case class Tanh (v: Variabl)(using ops: AutogradOps) extends Function:

    private var tanhCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes tanh(v).
     *  @return a Variabl with tanh applied.
     */
    override def forward (): Variabl =
        tanhCache = Some (ops.tanh_ (v.data))
        Variabl (tanhCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient using the tanh derivative.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.mul (gradOutput, ops.tanhD_ (tanhCache.get)))
end Tanh

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the sigmoid activation function.
 *  @param v  the input variable.
 */
case class Sigmoid (v: Variabl)(using ops: AutogradOps) extends Function:

    private var sigmoidCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes sigmoid(v).
     *  @return a Variabl with sigmoid applied.
     */
    override def forward (): Variabl =
        sigmoidCache = Some (ops.sigmoid_(v.data))
        Variabl (sigmoidCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient using the sigmoid derivative.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.mul (gradOutput, ops.sigmoidD_ (sigmoidCache.get)))
end Sigmoid

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the GeLU activation function.
 *  @param v  the input variable.
 */
case class GeLU (v: Variabl)(using ops: AutogradOps) extends Function:

    private var geluCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes GeLU(v).
     *  @return a Variabl with GeLU applied.
     */
    override def forward (): Variabl =
        geluCache = Some (ops.geLU_ (v.data))
        Variabl (geluCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**  Backward pass: uses the GeLU derivative to propagate the gradient.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.mul (gradOutput, ops.geLUD_ (geluCache.get)))
end GeLU

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Applies the softmax activation function.
 *  @param v  the input variable.
 */
case class Softmax (v: Variabl)(using ops: AutogradOps) extends Function:

    private var softmaxCache: Option [TensorD] = None

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes softmax(v).
     *  @return a Variabl with softmax applied.
     */
    override def forward (): Variabl =
        softmaxCache = Some (ops.softmax_ (v.data))
        Variabl (softmaxCache.get, gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient using the softmax derivative.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val s = softmaxCache.get
        // Compute Jacobian-vector product for softmax derivative.
        val dot       = ops.sumAlongAxis (gradOutput * s, axis = 2)
        val dotFull   = TensorD.broadcastTo (dot, s.shape)    // Broadcast to match shape (FIX: ops should handle this)
        val gradInput = s * (gradOutput - dotFull)
        v.backward (gradInput)
end Softmax

// -----------------------------------------------------------------------
// ------------------------- LOSS FUNCTIONS ------------------------------
// -----------------------------------------------------------------------

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the Sum of Squared Errors (SSE) loss.
 *  @param pred    the prediction variable.
 *  @param target  the target variable.
 */
case class SSELoss (pred: Variabl, target: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the SSE loss.
     *  @return a Variabl with loss data.
     */
    override def forward (): Variabl =
        val loss = ops.sseLoss (pred.data, target.data)
        Variabl (ops.scalar (loss), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient scaled by 2*(pred - target).
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val grad   = ops.mulScalar (ops.sub (pred.data, target.data), 2)
        val gFinal = ops.mulScalar (grad, gradOutput(0)(0)(0))        // Since gradOutput is a scalar tensor
        pred.backward (gFinal)
end SSELoss

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the Mean Squared Error (MSE) loss.
 *  @param pred    the prediction variable.
 *  @param target  the target variable.
 */
case class MSELoss (pred: Variabl, target: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the MSE loss.
     *  @return a Variabl with loss data.
     */
    override def forward (): Variabl =
        val loss = ops.mseLoss (pred.data, target.data)
        Variabl (ops.scalar(loss), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: scales the gradient by 2*(pred-target)/batchSize.
     *  @param gradOutput the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
            val numel  = pred.shape.product.toDouble
            val grad   = ops.mulScalar (ops.sub (pred.data, target.data), 2.0 / numel)
            val gFinal = ops.mulScalar (grad, gradOutput(0)(0)(0))        // Since gradOutput is a scalar tensor
            pred.backward (gFinal)
end MSELoss

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the Mean Absolute Error (MAE) loss.
 *  @param pred    the prediction variable.
 *  @param target  the target variable.
 */
case class MAELoss (pred: Variabl, target: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the MAE loss.
     *  @return a Variabl with loss data.
     */
    override def forward (): Variabl =
        val loss = ops.maeLoss (pred.data, target.data)
        Variabl (ops.scalar (loss), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient using the sign of (pred-target).
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val grad   = ops.sign (ops.sub (pred.data, target.data))
        val prod   = ops.mulScalar (grad, gradOutput(0)(0)(0))       // Since gradOutput is a scalar tensor
        val gFinal = ops.divScalar (prod, pred.shape.product)
        pred.backward (gFinal)
end MAELoss

// -----------------------------------------------------------------------
// ------------------------- TENSOR OPERATIONS ---------------------------
// -----------------------------------------------------------------------

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the dot product of two variables.
 *  @param v1  the first variable.
 *  @param v2  the second variable.
 */
case class Dot (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes dot(v1, v2).
     *  @return a Variabl containing the dot product.
     */
    override def forward (): Variabl =
        Variabl (ops.dot (v1.data, v2.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradients for dot product.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        backpropForTwoInputs (v1, v2, gradOutput,
                             (g: TensorD) => ops.mul (g, v2.data),
                             (g: TensorD) => ops.mul (v1.data, g))
end Dot

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the matrix multiplication of two variables.
 *  @param v1  the first variable.
 *  @param v2  the second variable.
 */
case class MatMul (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes matrix multiplication of v1 and v2.
     *  @return a Variabl with the result.
     */
    override inline def forward (): Variabl =
        Variabl (ops.matmul (v1.data, v2.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradients using transposed matrices.
     *  @param gradOutput  the upstream gradient.
     */
    override inline def backward (gradOutput: TensorD): Unit =
        backpropForTwoInputs (v1, v2, gradOutput,
            (g: TensorD) => ops.matmul (g, ops.transpose (v2.data, 1, 2)),
            (g: TensorD) => ops.matmul (ops.transpose (v1.data, 1 , 2), g))
end MatMul

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Computes the batched matrix multiplication of two variables.
 *  @param v1  the first variable.
 *  @param v2  the second variable.
 */
case class BatchMatMul (v1: Variabl, v2: Variabl)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes batched matrix multiplication.
     *  @return a Variabl with the batched result.
     */
    override inline def forward (): Variabl =
        Variabl (ops.bmm (v1.data, v2.data), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradients for batched matrix multiplication, unbroadcasting as necessary.
     *  @param gradOutput  the upstream gradient.
     */
    override inline def backward (gradOutput: TensorD): Unit =
        val v1T = ops.transpose (v1.data, 1, 2)
        val v2T = ops.transpose (v2.data, 1, 2)

        val gradA = ops.bmm (gradOutput, v2T)
        val gradB = ops.bmm (v1T, gradOutput)

        val gradAFinal = unbroadcast (gradA, v1.shape)
        val gradBFinal = unbroadcast (gradB, v2.shape)

        v1.backward (gradAFinal)
        v2.backward (gradBFinal)
end BatchMatMul

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Transposes (swaps) two axes of a tensor variable.
 *  @param v  the input variable.
 *  @param i  first axis index.
 *  @param j  second axis index.
 */
case class Transpose (v: Variabl, i: Int, j: Int)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: swaps axes i and j.
     *  @return a Variabl with transposed data.
     */
    override def forward (): Variabl = Variabl (ops.transpose (v.data, i, j), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: transpose again (swap i and j) to propagate gradient.
     *  @param gradOutput upstream gradient in transposed shape.
     */
    override def backward (gradOutput: TensorD): Unit =
        v.backward (ops.transpose (gradOutput, i, j))
end Transpose

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Permutes axes of a tensor variable according to a specified ordering.
 *  @param v     the input variable.
 *  @param axes  the permutation of axes.
 */
case class Permute (v: Variabl, axes: Seq [Int])(using ops: AutogradOps) extends Function:

     private var inverse: Array[Int] = uninitialized

     //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
     /** Forward pass: reorders axes as specified.
      *  @return a Variabl with permuted data.
      */
     override def forward (): Variabl =
         // Pre-compute inverse permutation for backward.
         inverse = Array.ofDim [Int] (axes.length)
         var idx = 0
         while idx < axes.length do
             inverse(axes(idx)) = idx
             idx += 1
         Variabl (ops.permute (v.data, axes), gradFn = Some (this))

     //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
     /** Backward pass: apply inverse permutation to gradient.
      *  @param gradOutput upstream gradient in permuted layout.
      */
     override def backward (gradOutput: TensorD): Unit =
         v.backward (ops.permute (gradOutput, inverse.toSeq))
end Permute

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Reshape operation for a variable.
 *  This class represents a differentiable operation that reshapes a tensor
 *  variable to a new shape during the forward pass and reshapes the gradient
 *  back to the original shape during the backward pass.
 *  @param v         the input variable to be reshaped
 *  @param newShape  the target shape for the variable
 */
case class Reshape (v: Variabl, newShape: Seq[Int])(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: reshapes the variable to newShape.
     *  @return a Variabl with reshaped data.
     */
    override def forward (): Variabl =
        Variabl (ops.reshape (v.data, newShape), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: reshapes the gradient back to the original shape.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =
        val origShape    = v.shape
        val gradReshaped = ops.reshape (gradOutput, origShape)
        v.backward (gradReshaped)
end Reshape

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Represents a slicing operation on a tensor variable.
 *  This class performs a differentiable slicing operation during the forward pass
 *  and propagates the gradient to the sliced region during the backward pass.
 *  @param v   the input variable to be sliced
 *  @param r0  the range for the first dimension
 *  @param r1  the range for the second dimension
 *  @param r2  the range for the third dimension
 */
case class Slice (v: Variabl, r0: Range, r1: Range, r2: Range)(using ops: AutogradOps) extends Function:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: slices the variable according to the specified ranges.
     *  @return a Variabl with sliced data.
     */
    override def forward (): Variabl =
        Variabl (ops.getSlice (v.data, r0, r1, r2), gradFn = Some (this))
    end forward

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates the gradient to the sliced region.
     *  @param gradOutput  upstream gradient corresponding to the sliced output.
     */
    override def backward (gradOutput: TensorD): Unit =
        val gradInput = ops.setSlice (ops.zerosLike (v.data), gradOutput, r0, r1, r2)
        v.backward (gradInput)
end Slice

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Represents a concatenation operation on a sequence of variables along a specified axis.
 *  This class performs a differentiable concatenation operation during the forward pass
 *  and splits the gradient during the backward pass to propagate it to the input variables.
 *  @param vs    the sequence of input variables to concatenate
 *  @param axis  the axis along which to concatenate the variables
 */
case class Concat (vs: Seq[Variabl], axis: Int)(using ops: AutogradOps) extends Function:

    private var splitSizes: Seq [Int] = uninitialized

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: concatenates the input variables along the specified axis.
     *  @return a Variabl with concatenated data.
     */
    override def forward (): Variabl =
        val dataSeq = vs.map (_.data)
        splitSizes  = dataSeq.map (t => t.shape (axis))
        Variabl (ops.concat (dataSeq, axis), gradFn = Some (this))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: splits the gradient and propagates to each input variable.
     *  @param gradOutput upstream gradient corresponding to the concatenated output.
     */
    override def backward (gradOutput: TensorD): Unit =
        var cursor = 0
        for ((v, sz) <- vs.zip (splitSizes)) do
            // Build slice ranges for each axis
            val ir = if axis == 0 then cursor until cursor + sz else 0 until gradOutput.shape(0)
            val jr = if axis == 1 then cursor until cursor + sz else 0 until gradOutput.shape(1)
            val kr = if axis == 2 then cursor until cursor + sz else 0 until gradOutput.shape(2)

            // Extract gradient slice for this variable
            val gradSlice = ops.getSlice (gradOutput, ir, jr, kr)
            // Propagate gradient to the input variable
            v.backward (gradSlice)
            cursor += sz
        end for
end Concat

// -----------------------------------------------------------------------
// -------------------- LAYER LEVEL FUNCTIONS ----------------------------
// -----------------------------------------------------------------------

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNNCellFused` Function implements a single RNN cell as one fused autograd op.
 *  It fuses the input/hidden projections and activation into a single node for
 *  improved performance and reduced autograd graph size.
 *  Equation:
 *    h_t = φ(W_ih * x + b_ih + W_hh * hPrev + b_hh)
 *  where φ ∈ {tanh, relu}
 *  Shapes:
 *    input     : (B, I, 1)
 *    hidden    : (B, H, 1)
 *    W_ih      : (1, H, I)
 *    W_hh      : (1, H, H)
 *    b_ih      : (1, H, 1)
 *    b_hh      : (1, H, 1)
 *  The function caches only what is needed for the backward pass:
 *    - input and hidden states
 *    - pre-activation value
 *    - output after activation
 */
case class RNNCellFused (input: Variabl, hidden: Variabl,   
                         W_ih: Variabl,  W_hh: Variabl, 
                         b_ih: Variabl, b_hh: Variabl, 
                         activation: String = "tanh") (using ops: AutogradOps) extends Function:

    private var inputCache: TensorD  = uninitialized
    private var hiddenCache: TensorD = uninitialized
    private var preActCache: TensorD = uninitialized
    private var outputCache: TensorD = uninitialized

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: computes the RNN cell output.
     *  @return a Variabl with the RNN cell output.
     */
    override def forward (): Variabl =
        val xProj  = ops.add (ops.bmm (W_ih.data, input.data), b_ih.data)
        val hProj  = ops.add (ops.bmm (W_hh.data, hidden.data), b_hh.data)
        val preAct = ops.add (xProj, hProj)
        val hNext  = activation match
            case "tanh" => ops.tanh_ (preAct)
            case "relu" => ops.reLU_ (preAct)
            case other  => throw new IllegalArgumentException (s"Unsupported activation: $other")

        // Cache tensors for backward
        inputCache  = input.data
        hiddenCache = hidden.data
        preActCache = preAct
        outputCache = hNext
        Variabl (hNext, gradFn = Some (this))
    end forward

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backward pass: propagates gradients through the RNN cell.
     *  @param gradOutput  the upstream gradient.
     */
    override def backward (gradOutput: TensorD): Unit =         // gradOutput: (B, H, 1)
        
        // 1. Activation derivative
        val actGrad = activation match
            case "tanh" => ops.tanhD_ (outputCache)
            case "relu" => ops.reLUD_ (preActCache)
            case other  => throw new IllegalArgumentException (s"Unsupported activation: $other")
        
        // 2. Elementwise multiply: (B, H, 1)
        val gradPreAct = ops.mul (gradOutput, actGrad)
        
        // 3. dL/dinput = W_ih.T @ gradPreAct -> (1, I, H) bmm (B, H, 1) = (B, I, 1)
        val gradInput  = ops.bmm (ops.transpose (W_ih.data, 1, 2), gradPreAct)
        
        // 4. dL/dhidden = W_hh.T @ gradPreAct -> (1, H, H).T bmm (B, H, 1) = (B, H, 1)
        val gradHidden = ops.bmm (ops.transpose (W_hh.data, 1, 2), gradPreAct)
        
        // 5. Weight grads: (B, H, 1) bmm (B, 1, I) = (B, H, I), then sum across batch
        val gradWih_batch = ops.bmm (gradPreAct, ops.transpose (inputCache, 1, 2))    // (B, H, I)
        val gradWhh_batch = ops.bmm (gradPreAct, ops.transpose (hiddenCache, 1, 2))   // (B, H, H)
        val gradW_ih = ops.sumAlongAxis (gradWih_batch, 0)      // (1, H, I)
        val gradW_hh = ops.sumAlongAxis (gradWhh_batch, 0)      // (1, H, H)
        
        // 6. Bias grads: sum across batch axis (0)
        val gradB = ops.sumAlongAxis (gradPreAct, 0)            // (1, H, 1)
        
        // 7. Backpropagate
        input.backward (gradInput)
        hidden.backward (gradHidden)
        W_ih.backward (gradW_ih)
        W_hh.backward (gradW_hh)
        b_ih.backward (gradB)
        b_hh.backward (gradB)
    end backward
end RNNCellFused

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GRUCellFused` Function implements a single GRU cell as one fused autograd op.
 *  It fuses all gate computations for better performance and fewer autograd nodes.
 *  Equations:
 *    r_t = sigmoid(W_ir * x + b_ir + W_hr * hPrev + b_hr)
 *    z_t = sigmoid(W_iz * x + b_iz + W_hz * hPrev + b_hz)
 *    n_t = tanh(W_in * x + b_in + r_t ⊙ (W_hn * hPrev + b_hn))
 *    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ hPrev
 *  Shapes:
 *    input     : (B, I, 1)
 *    hidden    : (B, H, 1)
 *    W_i*      : (1, H, I)
 *    W_h*      : (1, H, H)
 *    b_i*,b_h* : (1, H, 1)
 */
case class GRUCellFused (input: Variabl, hidden: Variabl,
                         W_ir: Variabl, W_hr: Variabl, b_ir: Variabl, b_hr: Variabl,
                         W_iz: Variabl, W_hz: Variabl, b_iz: Variabl, b_hz: Variabl,
                         W_in: Variabl, W_hn: Variabl, b_in: Variabl, b_hn: Variabl)
                        (using ops: AutogradOps) extends Function:
    
    private var r_t, z_t, n_t: TensorD = uninitialized
    private var x_t, hPrev: TensorD    = uninitialized
    private var h_hn_lin: TensorD      = uninitialized
    
    override def forward (): Variabl =
        x_t   = input.data
        hPrev = hidden.data
        
        // Reset gate
        val pre_r = ops.add (ops.add (ops.bmm (W_ir.data, x_t), b_ir.data),
                             ops.add (ops.bmm (W_hr.data, hPrev), b_hr.data))
        val r = ops.sigmoid_ (pre_r)
        
        // Update gate
        val pre_z = ops.add (ops.add (ops.bmm (W_iz.data, x_t), b_iz.data),
                             ops.add (ops.bmm (W_hz.data, hPrev), b_hz.data))
        val z = ops.sigmoid_ (pre_z)
        
        // Candidate gate
        val h_hn  = ops.add (ops.bmm (W_hn.data, hPrev), b_hn.data)
        val pre_n = ops.add (ops.add (ops.bmm (W_in.data, x_t), b_in.data), ops.mul (r, h_hn))
        val n     = ops.tanh_ (pre_n)
        
        // Hidden update
        val oneMinusZ = ops.addScalar (ops.neg (z), 1.0)
        val hNext     = ops.add (ops.mul (oneMinusZ, n), ops.mul(z, hPrev))
        
        // Cache
        r_t = r; z_t = z; n_t = n; h_hn_lin = h_hn
        
        Variabl (hNext, gradFn = Some (this))
    end forward
    
    override def backward (gradOutput: TensorD): Unit =
        // dhNext/dn = (1 - z), dhNext/dz = (hPrev - n)
        val dh_dn = ops.addScalar (ops.neg (z_t), 1.0)
        val dh_dz = ops.sub (hPrev, n_t)
        
        val grad_n = ops.mul (gradOutput, dh_dn)
        val grad_z = ops.mul (gradOutput, dh_dz)
        val grad_hPrev_part = ops.mul (gradOutput, z_t)
        
        // Through activations
        val grad_pre_n = ops.mul (grad_n, ops.tanhD_ (n_t))
        val grad_pre_z = ops.mul (grad_z, ops.sigmoidD_ (z_t))
        val grad_pre_r = ops.mul (ops.mul (grad_pre_n, h_hn_lin), ops.sigmoidD_ (r_t))
        
        // --- Grad wrt hPrev ---
        val grad_h_from_n = ops.mul (r_t, ops.bmm (ops.transpose (W_hn.data, 1, 2), grad_pre_n))
        val grad_h_from_r = ops.bmm (ops.transpose (W_hr.data, 1, 2), grad_pre_r)
        val grad_h_from_z = ops.bmm (ops.transpose (W_hz.data, 1, 2), grad_pre_z)
        val grad_hPrev_total = ops.add (grad_hPrev_part, ops.add (grad_h_from_n,
                                        ops.add (grad_h_from_r, grad_h_from_z)))
        
        // --- Grad wrt input ---
        val grad_x_r = ops.bmm (ops.transpose (W_ir.data, 1, 2), grad_pre_r)
        val grad_x_z = ops.bmm (ops.transpose (W_iz.data, 1, 2), grad_pre_z)
        val grad_x_n = ops.bmm (ops.transpose (W_in.data, 1, 2), grad_pre_n)
        val grad_x_total = ops.add (ops.add (grad_x_r, grad_x_z), grad_x_n)
        
        // --- Weight and bias grads ---
        def sumB (batch: TensorD): TensorD = ops.sumAlongAxis (batch, 0)
        
        val gradW_ir = sumB (ops.bmm (grad_pre_r, ops.transpose (x_t, 1, 2)))
        val gradW_hr = sumB (ops.bmm (grad_pre_r, ops.transpose (hPrev, 1, 2)))
        val gradW_iz = sumB (ops.bmm (grad_pre_z, ops.transpose (x_t, 1, 2)))
        val gradW_hz = sumB (ops.bmm (grad_pre_z, ops.transpose (hPrev, 1, 2)))
        val gradW_in = sumB (ops.bmm (grad_pre_n, ops.transpose (x_t, 1, 2)))
        val gradW_hn = sumB (ops.bmm (ops.mul (grad_pre_n, r_t), ops.transpose (hPrev, 1, 2)))
        
        val gradB_ir = sumB (grad_pre_r)
        val gradB_hr = sumB (grad_pre_r)
        val gradB_iz = sumB (grad_pre_z)
        val gradB_hz = sumB (grad_pre_z)
        val gradB_in = sumB (grad_pre_n)
        val gradB_hn = sumB (ops.mul (grad_pre_n, r_t))
        
        // --- Backprop ---
        input.backward (grad_x_total)
        hidden.backward (grad_hPrev_total)
        
        W_ir.backward (gradW_ir); W_hr.backward (gradW_hr)
        W_iz.backward (gradW_iz); W_hz.backward (gradW_hz)
        W_in.backward (gradW_in); W_hn.backward (gradW_hn)
        
        b_ir.backward (gradB_ir); b_hr.backward (gradB_hr)
        b_iz.backward (gradB_iz); b_hz.backward (gradB_hz)
        b_in.backward (gradB_in); b_hn.backward (gradB_hn)
    end backward
end GRUCellFused

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Fused RNN over a whole input sequence (vanilla RNN).
 *  - Unrolls the sequence in a single Function.
 *  - Returns the last hidden state as the output Variabl.
 *  - On backward (), performs full BPTT and accumulates parameter grads.
 *  Shapes:
 *    input(t):  (B, I, 1)
 *    hidden:    (B, H, 1)      // initial hidden (h0)
 *    W_ih:      (1, H, I)
 *    W_hh:      (1, H, H)
 *    b_ih:      (1, H, 1)
 *    b_hh:      (1, H, 1)
 */
case class RNNFused (input: IndexedSeq [Variabl],               // sequence: x_0 ... x_{T-1}
                     hidden: Variabl,            
                     W_ih: Variabl, W_hh: Variabl,
                     b_ih: Variabl, b_hh: Variabl,
                     activation: String = "tanh", tbptt: Int = 0)
                    (using ops: AutogradOps) extends Function:

    // --------- Caches for BPTT (one entry per time step) --------------------------
    private var xCache: Array [TensorD]      = Array.empty     // x_t
    private var hPrevCache: Array [TensorD]  = Array.empty     // h_{t-1}
    private var preActCache: Array [TensorD] = Array.empty     // W_ih x_t + W_hh h_{t-1} + b
    private var hCache: Array [TensorD]      = Array.empty     // h_t
    
    private var outputsVar: IndexedSeq [Variabl] = uninitialized
    private var finalHiddenVar: Variabl = uninitialized
    
    def outputs: IndexedSeq [Variabl] = outputsVar
    def finalHidden: Variabl = finalHiddenVar

    private [autograd] def clearCaches (): Unit =
        xCache      = Array.empty
        hPrevCache  = Array.empty
        preActCache = Array.empty
        hCache      = Array.empty
    end clearCaches
    
    private class OutputNode (t: Int) extends Function:

        override def forward (): Variabl =
            throw new IllegalStateException ("" + "OutputNode.forward () should never be called.")
        
        override def backward (gradOutput: TensorD): Unit = backwardFromT (t, gradOutput)
    end OutputNode
    
    private [autograd] def backwardFromT (tStop: Int, gradAtT: TensorD): Unit =
        val T = hCache.length
        require (T > 0, "RNNFused.backwardFromT: no cached forward pass found")
        require (tStop >= 0 && tStop < T,
                 s"RNNFused.backwardFromT: tStop=$tStop out of range 0..${T - 1}")
        
        // starting gradient: dL/dh_{tStop}
        var gradNextHidden = gradAtT
        
        // Determine tMin for TBPTT
        val tMin = if tbptt > 0 then math.max (0, tStop - tbptt + 1)
                   else 0
        
        var t = tStop
        while t >= tMin do
            val x_t    = xCache (t)
            val h_prev = hPrevCache (t)
            val pre_t  = preActCache (t)
            val h_t    = hCache (t)
            
            val actGrad = activation match
                case "tanh" => ops.tanhD_(h_t)
                case "relu" => ops.reLUD_(pre_t)
                case other  => throw new IllegalArgumentException(s"Unsupported activation: $other")
            
            val gradPreAct = ops.mul (gradNextHidden, actGrad)
            
            val gradInput_t =
                ops.bmm (ops.transpose (W_ih.data, 1, 2), gradPreAct)
            
            val gradHiddenPrev =
                ops.bmm (ops.transpose (W_hh.data, 1, 2), gradPreAct)
            
            val gradWih_batch = ops.bmm (gradPreAct, ops.transpose (x_t, 1, 2))
            val gradWhh_batch = ops.bmm (gradPreAct, ops.transpose (h_prev, 1, 2))
            val gradW_ih      = ops.sumAlongAxis (gradWih_batch, 0)
            val gradW_hh      = ops.sumAlongAxis (gradWhh_batch, 0)
            val gradB         = ops.sumAlongAxis (gradPreAct, 0)
            
            input(t).backward (gradInput_t)
            W_ih.backward (gradW_ih)
            W_hh.backward (gradW_hh)
            b_ih.backward (gradB)
            b_hh.backward (gradB)
            
            gradNextHidden = gradHiddenPrev
            t -= 1
        end while
        
        // Backprop into h_{tMin-1} (which is the "incoming" hidden state seen in this chunk)
        hidden.backward (gradNextHidden)
    end backwardFromT

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass: unroll the RNN through the entire sequence.
     *  Returns the last hidden state h_T as a Variabl (with gradFn = this).
     */
    def forwardAll (): (IndexedSeq [Variabl], Variabl) =
        import scala.collection.mutable.ArrayBuffer
        require (input.nonEmpty, "RNNFused.forward: input sequence must be non-empty")

//      val T = input.length
//      val B = input.head.shape.head

        val xsBuf    = ArrayBuffer.empty [TensorD]
        val hPrevBuf = ArrayBuffer.empty [TensorD]
        val preBuf   = ArrayBuffer.empty [TensorD]
        val hBuf     = ArrayBuffer.empty [TensorD]

        // For outputs (Variabl) and state tracking
        val outputsBuf = scala.collection.mutable.ArrayBuffer.empty [Variabl]

        var hPrev = hidden.data // (B, H, 1)

        input.foreach { xVar =>
            val x = xVar.data                                             // (B, I, 1)
            val xProj = ops.add (ops.bmm (W_ih.data, x), b_ih.data)       // (B, H, 1)
            val hProj = ops.add (ops.bmm (W_hh.data, hPrev), b_hh.data)   // (B, H, 1)
            val preAct = ops.add (xProj, hProj)                           // (B, H, 1)

            val hNext = activation match
                case "tanh" => ops.tanh_ (preAct)                         // (B, H, 1)
                case "relu" => ops.reLU_ (preAct)
                case other  => throw new IllegalArgumentException (s"Unsupported activation: $other")

            // Cache per-timestep tensors
            xsBuf     += x
            hPrevBuf  += hPrev
            preBuf    += preAct
            hBuf      += hNext

            // Store output Variabl
            val t       = outputsBuf.length
            outputsBuf += Variabl (hNext, gradFn = Some (OutputNode(t)),
                                   name = Some (s"h_${outputsBuf.length}"))

            hPrev = hNext                           // Advance to next timestep
        } // foreach

        // Freeze caches to be used in backward
        xCache      = xsBuf.toArray
        hPrevCache  = hPrevBuf.toArray
        preActCache = preBuf.toArray
        hCache      = hBuf.toArray

        // Return final hidden state as the "output" of this Function
        (outputsBuf.toIndexedSeq, outputsBuf.last)
    end forwardAll
    
    override def forward (): Variabl =
        val (outs, last) = forwardAll ()
        outputsVar = outs
        finalHiddenVar = last
        last
    end forward
    
    override def backward (gradOutput: TensorD): Unit =
        val T = hCache.length
        require (T > 0, "RNNFused.backward: no cached forward pass found")
        backwardFromT (T - 1, gradOutput)
    end backward
end RNNFused

