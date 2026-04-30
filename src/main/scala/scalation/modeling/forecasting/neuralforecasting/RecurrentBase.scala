
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Mon Aug  4 21:12:40 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Base Trait for Recurrent Neural Networks
 */

package scalation
package modeling
package forecasting
package neuralforecasting

import scala.math.{log, tanh}

import scalation.mathstat.{MatrixD, TensorD, VectorD}
import scalation.random.NormalMat

import ActivationFun.{sigmoid, softmax_, tanhD}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RecurrentBase` trait defines the base structure and operations for recurrent neural networks.
 *  It includes common hyperparameters, activation functions, and methods for parameter initialization,
 *  gradient clipping, and parameter updates.
 */
trait RecurrentBase:

    protected val CLASSIF: Boolean = false                   // Whether it's classification or regression
// FIX -- use hyper-parameters
    protected val max_epochs: Int = 30                       // Maximum number of epochs
    protected val eta: Double = 0.0005                       // Learning rate
    protected val batch_size: Int = 64                       // Batch size for training
    protected val truncation_length: Int = 40                // Truncation length for BPTT
    protected val β: Double = 0.9                            // Momentum hyper parameter
    protected val threshold: Double = 5.0                    // Threshold for gradient clipping

    protected val seq_length: Int                            // Number of time steps (sequence length)
    protected val n_mem: Int                                 // Size of hidden state
    protected val n_seq: Int                                 // Number of sequences
    protected val n_var: Int                                 // Number of input variables

    protected val x: TensorD                                 // Input tensor
    protected val y: TensorD                                 // Output tensor

    protected val loss_per_epoch: VectorD                    // Loss per epoch
    protected val L: VectorD                                 // Loss per time step

// FIX -- functioanlity should be defined once
    def log_ (x: VectorD): VectorD = x.map (log)             // Log transformation for VectorD

    def log_ (x: MatrixD): MatrixD = x.map_ (log)            // Log transformation for MatrixD

    def sigmoid_ (x: MatrixD): MatrixD = x.map_ (sigmoid)    // Overloaded sigmoid for MatrixD

    def softmax_m (x: MatrixD): MatrixD = x.mmap (softmax_)  // Overloaded softmax for MatrixD

    def tanh_ (t: VectorD): VectorD = t.map (tanh)           // Overloaded tanh for VectorD
    def tanh_ (x: MatrixD): MatrixD = x.map_ (tanh)          // Overloaded tanh for MatrixD

    def tanhD_m (x: MatrixD): MatrixD = x.mmap (tanhD)       // Overloaded tanh derivative for MatrixD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Case class representing a group of parameters, including the parameter itself, its velocity, and its gradient.
     *  @param param     the parameter (e.g., weights or biases)
     *  @param velocity  the velocity associated with the parameter, used for momentum in optimization
     *  @param grad      the gradient of the parameter, used for updating the parameter during training
     */
    protected case class ParamGroup [T] (var param: T, var velocity: T, var grad: T)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Updates a specific batch of rows in a matrix with new values.
     *  @param matrix       the matrix to be updated
     *  @param batch_start  the starting index of the batch in the matrix
     *  @param batch_end    the ending index (exclusive) of the batch in the matrix
     *  @param newBatch     the new matrix containing the values to be inserted
     * @throws IllegalArgumentException If the size of newBatch doesn't match the specified batch size
     */
    protected def updateBatch (matrix: MatrixD, batch_start: Int, batch_end: Int, newBatch: MatrixD): Unit =
        if batch_end - batch_start != newBatch.dim then
            throw new IllegalArgumentException (
                "Batch size mismatch: The newBatch matrix must have the same number of rows as (batch_end - batch_start).")
        for i <- 0 until newBatch.dim do
            matrix(batch_start + i) = newBatch(i)                          // Update batch in the matrix
    end updateBatch

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize a parameter group with a specified number of rows, columns, and standard deviation.
     *  @param rows    the number of rows in the parameter matrix
     *  @param cols    the number of columns in the parameter matrix
     *  @param stdDev  the standard deviation for initializing the parameter matrix
     *  @return A ParamGroup containing the initialized parameter matrix, velocity matrix, and gradient matrix
     */
    protected def initializeParamGroup (rows: Int, cols: Int, stdDev: Double): ParamGroup [MatrixD] =
        ParamGroup (param    = NormalMat (rows, cols, 0.0, stdDev).gen,    // Initialize parameter matrix
                    velocity = new MatrixD (rows, cols),                   // Initialize velocity matrix
                    grad     = new MatrixD (rows, cols))                   // Initialize gradient matrix
    end initializeParamGroup

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Initialize a bias group with a specified size.
     *  @param size  the size of the bias vector
     *  @return A ParamGroup containing the initialized bias vector, velocity vector, and gradient vector
     */
    protected def initializeBiasGroup (size: Int): ParamGroup [VectorD] =
        ParamGroup (param    = new VectorD (size),                         // Initialize bias vector
                    velocity = new VectorD (size),                         // Initialize velocity vector
                    grad     = new VectorD (size))                         // Initialize gradient vector
    end initializeBiasGroup

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Abstract method for training the model with the given input and output tensors.
     * @param x  the input tensor
     * @param y  the output tensor
     */
    def train (x: TensorD, y: TensorD): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Abstract method to zero the gradients of the model parameters.
     */
    protected def zero_gradients (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Abstract method to clip the gradients of the model parameters to a specified threshold.
     *  @param threshold  the threshold value for gradient clipping
     */
    protected def clip_gradients (threshold: Double): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Abstract method to update the model parameters based on the gradients.
     *  @param batch_size  the size of the batch used for training
     *  @param leaky       a boolean flag indicating whether to use leaky updates (default is true)
     */
    protected def update_params (batch_size: Int, leaky: Boolean = true): Unit

end RecurrentBase

