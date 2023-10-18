
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Wed Oct 11 14:31:52 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *------------------------------------------------------------------------------
 *  Method types for the user-defined functions in the L-BFGS C library shared
 *  object. Used in the FFM implementation of the Limited memory
 *  Broyden–Fletcher–Goldfarb–Shanno (BFGS) for Bound constrained optimization
 *  (L-BFGS-B) algorithm.
 */

// Package definition.
package scalation
package optimization
package LBFGS_FFM

// General imports.
import java.lang.foreign.MemorySegment
import java.lang.invoke.MethodType

// Object.
object MethodTypes:
    val EVALUATE_METHOD_TYPE: MethodType = MethodType.methodType(
        classOf[Double],
        classOf[MemorySegment],
        classOf[MemorySegment],
        classOf[MemorySegment],
        classOf[Int],
        classOf[Double]
    )

    val PROGRESS_METHOD_TYPE: MethodType = MethodType.methodType(
        classOf[Int],
        classOf[MemorySegment],
        classOf[MemorySegment],
        classOf[MemorySegment],
        classOf[Double],
        classOf[Double],
        classOf[Double],
        classOf[Double],
        classOf[Int],
        classOf[Int],
        classOf[Int]
    )
end MethodTypes
