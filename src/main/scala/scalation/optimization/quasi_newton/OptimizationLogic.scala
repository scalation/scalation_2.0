
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Tue Sep 19 09:23:45 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Optimization Logic for the L-BFGS Algorithm for Monitoring Progress.
 */

package scalation
package optimization
package quasi_newton

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `OptimizationLogic` trait specifies the requirements for the logic
 *  to be used in each step of a L-BFGS variable minimization done by the
 *  `lbfgsMain` method of the `LBFGS` object. The methods provided in this
 *  trait are called directly by the code used by the `LBFGS` class.
 *
 *  Classes mixing in this trait must implement two methods: evaluate and
 *  progress. The evaluate method is used to evaluate the gradients and
 *  objective function for a given state of the variables. The progress method
 *  is used to report on how the minimization process is progressing.
 */
trait OptimizationLogic extends EvaluationLogic:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Receives the progress of each iteration of the optimization process.  Can
     *  be used to display or record said progress and to determine if the
     *  optimization should continue or be cancelled.  A default implementation
     *  is provided to just print the contents of the current iteration of the
     *  optimization.
     *
     *  @param instance  an optional user data segment that may be provided when calling the
     *                   `LBFGS.lbfgsMain` method and may have `Any` type, but must be the same
     *                   type used by the `evaluate` method of classes extending this trait.
     *                   Note, has type `MemorySegment`in `OptimizationLogicC`.
     *  @param x         `VectorD` with the current values of the variables.
     *  @param g         `VectorD` with the current value of the gradient vector.
     *  @param fx        Current value of the objective function.
     *  @param xnorm     Euclidean norm of the variables.
     *  @param gnorm     Euclidean norm of the gradient vector.
     *  @param step      Step used by the line search routine in this iteration.
     *  @param n         The number of variables.
     *  @param k         Iteration count.
     *  @param ls        The number of evaluations called for this iteration.
     *  @return int      Determines if optimization should continue.  Zero continues
     *                   optimization. Non-zero values cancel the  optimization.
     */
    def progress (instance: Any, x: VectorD, g: VectorD, fx: Double, xnorm: Double,
                  gnorm: Double, step: Double, n: Int, k: Int, ls: Int): LBFGSReturnCode =

        println (s"""
    Iteration $k:
    instance \t\t= $instance
    x \t\t= $x
    g \t\t= $g
    fx \t\t= $fx
    xnorm \t= $xnorm
    gnorm \t= $gnorm
    step \t= $step
    n \t\t= $n
    ls \t\t= $ls
        """)

        LBFGSReturnCode.Success

end OptimizationLogic

