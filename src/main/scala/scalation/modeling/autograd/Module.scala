
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 19:50:46 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Base Classes for Neural Network Modules and Layers
 */

package scalation
package modeling
package autograd

import scalation.mathstat.TensorD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BaseModule` is a base class for all neural network modules (layers, blocks, models).
 *  Provides support for:
 *   - Parameter registration
 *   - Automatic submodule detection
 *   - Gradient management (zeroing)
 *   - Training/evaluation mode switching
 *  Modules are structured hierarchically: a module can contain submodules.
 *  @param localParameters  the parameters (Variables) directly belonging to this module
 */
abstract class BaseModule (localParameters: IndexedSeq [Variabl] = IndexedSeq.empty):

    /** Automatically detect submodules (other BaseModules) within this module. */
    lazy val subModules: IndexedSeq [BaseModule] =
        this.getClass.getDeclaredFields.flatMap { f =>
            f.setAccessible (true)
            f.get (this) match
                case module: BaseModule => Some(module)
                case _ => None
        }.toIndexedSeq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return all trainable parameters, including those from submodules.
     */
    def parameters: IndexedSeq [Variabl] = localParameters ++ subModules.flatMap (_.parameters)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the gradients of all parameters.
     */
    def gradients: IndexedSeq [TensorD] = parameters.map (_.grad)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Zero out all gradients (in-place).
     */
    def zeroGrad ()(using ops: AutogradOps): Unit = parameters.foreach (p => p.grad = ops.zerosLike (p.grad))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Flag to control training or evaluation behavior.
     */
    var inTrainingMode: Boolean = false

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the module to training mode (and all submodules recursively).
     */
    def train (mode: Boolean = true): Unit =
        this.inTrainingMode = mode
        subModules.foreach (_.train(mode))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the module to evaluation mode (and all submodules recursively).
     */
    def eval (): Unit = train (false)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Replace the current parameters with new ones.
     *  Useful for weight updates, loading saved models, etc.
     *  @param newParams  The new parameter list to assign
     */
    def setParameters(newParams: IndexedSeq [Variabl]): Unit =
        val currentParams = this.parameters
        require (currentParams.size == newParams.size, "Parameter size mismatch in setParameters!")
        for i <- currentParams.indices do currentParams(i).data = newParams(i).data
    end setParameters

end BaseModule


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Standard module for layers that take a single input (e.g., Linear, Conv1D).
 *  Defines the abstract forward function for single input.
 *  @param localParameters  the parameters (Variables) directly belonging to this module
 */
abstract class Module (localParameters: IndexedSeq [Variabl] = IndexedSeq.empty)
    extends BaseModule (localParameters):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass for a single input Variable. Must be implemented by subclasses.
     */
    def forward (input: Variabl): Variabl

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Alias for forward, allows calling the module as a function: `module(x)`.
     */
    def apply (input: Variabl): Variabl = forward (input)

end Module


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Module for layers that take multiple inputs (e.g., RNN cells, attention blocks).
 *  Defines the abstract forward function for sequence or multiple inputs.
 *  @param localParameters  the parameters (Variables) directly belonging to this module
 */
abstract class SeqModule (localParameters: IndexedSeq [Variabl] = IndexedSeq.empty)
    extends BaseModule (localParameters):

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forward pass for multiple input Variables. Must be implemented by subclasses.
     */
    def forward (inputs: IndexedSeq [Variabl]): IndexedSeq [Variabl]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Alias for forward, allows calling the module as a function: `module(xs)`.
     */
    def apply (inputs: IndexedSeq [Variabl]): IndexedSeq [Variabl] = forward (inputs)

end SeqModule

