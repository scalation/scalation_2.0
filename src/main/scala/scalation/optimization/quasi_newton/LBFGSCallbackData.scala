
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  André Filipe Caldas Laranjeira
 *  @version 2.0
 *  @note    Mon Aug 21 13:51:22 EDT 2023
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Callback Data Used by the L-BFGS Algorithm.
 */

package scalation
package optimization
package quasi_newton

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LBFGSCallbackData` case class is used to group together the
 *  `EvaluationLogic` specified for a L-BFGS optimization done by the
 *  `LBFGS` object with values that are the parameters for the methods of the
 *  `EvaluationLogic`.  This allows the user to pass the optimization
 *  logic of the L-BFGS optimization as a parameter to different methods
 *  and classes while retaining the ability to callback the methods of said
 *  logic with the correct parameters.
 *
 *  @param n          the number of variables used in the optimization.
 *  @param instance   an optional user data segment that may be provided when calling
 *                    the `LBFGS.lbfgsMain` method (@see `OptimizationLogic`)
 *  @param evalLogic  `EvaluationLogic` that describes the optimization steps 
 *                    for the L-BFGS optimization done by the `LBFGS` object.
 */
case class LBFGSCallbackData (n: Int, instance: Any, evalLogic: EvaluationLogic)

