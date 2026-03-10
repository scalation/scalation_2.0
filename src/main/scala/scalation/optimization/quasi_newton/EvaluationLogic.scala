
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Fri Sep 22 16:15:18 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Trait Specifing Evaluation Logic for the L-BFGS Algorithm
 */

package scalation
package optimization
package quasi_newton

import scalation.mathstat.VectorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `EvaluationLogicNative` trait specifies the requirements for the logic
 *  to be used for variable evaluation against the objective function in the
 *  `lbfgsMain` method of the `LBFGS` object.  The methods provided in this
 *  trait are called directly by the code used by the `BFGS` class.
 *
 *  Classes mixing in this trait must implement the evaluate method, which is
 *  used to evaluate the gradients and objective function for a given state of
 *  the variables.
 */
trait EvaluationLogic:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Evaluates the gradients and objective function according to the state of
     *  the variables during the minimization process.
     *
     *  @param instance  an optional user data segment that may be provided when calling
     *                   the `LBFGS.lbfgsMain` method (@see `OptimizationLogic`)
     *  @param x         `VectorD` with the current values of the variables
     *  @param n         the number of variables
     *  @param step      current step chosen by the line search routine.
     *  @return          LBFGSVarEvaluationResults, results obtained from evaluating the variables
     */
    def evaluate (instance: Any, x: VectorD, n: Int, step: Double): LBFGSVarEvaluationResults

end EvaluationLogic


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGSVarEvaluationResults` case class holds results from running evaluation 
 *  logic used by the implementation of the Limited memory Broyden–Fletcher–Goldfarb–Shanno 
 *  (BFGS) for unconstrained optimization (L-BFGS) algorithm.
 */
case class LBFGSVarEvaluationResults (objFunctionValue: Double, gradientVector: VectorD)

