
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

import scala.collection.immutable.ArraySeq

import scalation.calculus.Differential.grad
import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GradCheck` object provides methods the check the agreement between numerically
 *  computed gradient those computed using Automatic Differentiation (AD).
 *  @see `calculus.Differential` 
 */
object GradCheck:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check the gradient for all given variable.
     */
    def gradCheck (param: Variabl, loss: () => Variabl,
                   atol: Double = 1e-5, rtol: Double = 1e-3,
                   maxMismatches: Int = 10, quiet: Boolean = true,
                   debug: Boolean = false): Boolean =
        val fScalar    = () => loss ().data.sum
        val forwardVal = fScalar ()
        val v  = param.data.flattenToVector
        val fn = toFunctionV2S (param, fScalar)
        val numerlVec = grad (fn, v)
        val numerical = TensorD (param.data.dims, ArraySeq.unsafeWrapArray (numerlVec.toArray)*)

        // ----- Analytical gradient via backprop -----
        param.grad = TensorD.zerosLike (param.data)
        val out    = loss ()
        out.backward ()
        val analytical = param.grad

        // println for debugging
        if debug then
            println (s"param:      ${param.name.getOrElse ("unnamed")}")
            println (s"param.data: $v")
            println (s"analytical: $analytical")
            println (s"numerical:  $numerical")
        end if

        // ----- Compare -----
//      val diff = (analytical - numerical).abs
        val (d1, d2, d3) = analytical.dims
        var passed = true
        var shown  = 0

        cfor(0, d1) { i =>
            cfor(0, d2) { j =>
                cfor(0, d3) { k =>
                    val a = analytical(i, j, k)
                    val n = numerical(i, j, k)
                    val d = math.abs (a - n)
                    val tol = atol + rtol * math.abs (n)
                    if d > tol then
                        passed = false
                        if shown < maxMismatches then
                            println (f"  ❌ Mismatch at ($i,$j,$k): autograd=$a%.6g, numerical=$n%.6g, diff=$d%.3g > tol=$tol%.3g")
                            shown += 1
                    end if
                } // cfor
            } // cfor
        } // cfor

        if shown == 0 && ! quiet then
            println (s"\nGradCheck for ${param.name.getOrElse("unnamed")}")
            println (s"  Forward value: $forwardVal")
            println (s"✅ GradCheck PASSED for ${param.name.getOrElse ("unnamed")}\n")
        else if ! passed then
            println (s"\nGradCheck for ${param.name.getOrElse ("unnamed")}")
            println (s"  Forward value: $forwardVal")
            val extra = math.max (0, shown - maxMismatches)
            if extra > 0 then println (s"  ... and $extra more mismatches not shown")
            println (s"❌ GradCheck FAILED for ${param.name.getOrElse ("unnamed")}\n")

        passed
    end gradCheck

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert ? to a `FunctionV2S`
     *  @param param  the parameters
     *  @param f      the function
     */
    private def toFunctionV2S (param: Variabl, f: () => Double): FunctionV2S =
        (x: VectorD) => param.data = TensorD (param.data.dims, ArraySeq.unsafeWrapArray (x.toArray)*)
        f()
    end toFunctionV2S

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check the gradients for all the variables.
     */
    def gradCheckAll (params: Seq [Variabl], loss: () => Variabl,
                      atol: Double = 1e-5, rtol: Double = 1e-3,
                      maxMismatches: Int = 10, quiet: Boolean = true,
                      debug: Boolean = false): Boolean =
        params.forall { p => gradCheck (p, loss, atol, rtol, maxMismatches, quiet, debug) }
    end gradCheckAll

end GradCheck

