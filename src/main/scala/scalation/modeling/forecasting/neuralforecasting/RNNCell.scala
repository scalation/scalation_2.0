
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Mon Aug  4 21:12:40 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Recurrent Neural Network (RNN) for Multivariate Time Series
 */

package scalation
package modeling
package forecasting
package neuralforecasting

import scalation.mathstat.{MatrixD, TensorD, VectorD}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNNCell` trait defines the structure and operations for a Recurrent Neural Network (RNN) cell.
 *  It includes weight matrices, bias vectors, and hidden states, along with methods for gradient clipping
 *  and parameter updates.
 */
trait RNNCell extends RecurrentBase:

    protected lazy val U: ParamGroup [MatrixD] =
        initializeParamGroup (n_mem, n_var, math.sqrt(2.0 / (n_var + n_mem)))        // Input-to-hidden weights

    protected lazy val W: ParamGroup [MatrixD] =
        initializeParamGroup (n_mem, n_mem, math.sqrt(2.0 / (n_mem + n_mem)))        // Hidden-to-hidden weights

    protected lazy val V: ParamGroup [MatrixD] =
        initializeParamGroup (y.dim2, n_mem, math.sqrt(2.0 / (n_mem + y.dim2)))      // Hidden-to-output weights

    protected lazy val b_h: ParamGroup [VectorD] = initializeBiasGroup (n_mem)       // Bias for hidden layer
    protected lazy val b_y: ParamGroup [VectorD] = initializeBiasGroup (y.dim2)      // Bias for output layer

    protected lazy val H: HiddenState = HiddenState (n_mem, batch_size, seq_length)  // Hidden state initialization

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Case class representing the hidden state of the RNN.
     *  @param n_mem         number of memory units
     *  @param n_seq         number of sequences
     *  @param seq_length    length of each sequence
     *  @param param         tensor for hidden state parameters
     *  @param grad          tensor for hidden state gradients
     *  @param pre_act_grad  matrix for pre-activation gradients
     */
    protected case class HiddenState (n_mem: Int, n_seq: Int, seq_length: Int, var param: TensorD = null,
                                      var grad: TensorD = null, var pre_act_grad: MatrixD = null):

        if param == null then param = new TensorD(n_mem, n_seq, seq_length)          // Initialize param if not provided
        if grad == null then  grad  = new TensorD(n_mem, n_seq, seq_length)          // Initialize grad if not provided
        if pre_act_grad == null then pre_act_grad = new MatrixD (n_mem, n_seq)       // Initialize pre-activation grad if not provided

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Return a new HiddenState with tensors sliced to the specified batch size.
         *  @param batch_size The batch size to slice to
         *  @return A new HiddenState with sliced tensors
         */
        def slice (batch_size: Int): HiddenState =
            val slicedParam = param(0 until n_mem, 0 until batch_size, 0 until seq_length)
            val slicedGrad  = grad(0 until n_mem, 0 until batch_size, 0 until seq_length)
            val slicedPreActGrad = pre_act_grad(0 until n_mem, 0 until batch_size)
            HiddenState (n_mem, batch_size, seq_length, param = slicedParam, grad = slicedGrad,
                         pre_act_grad = slicedPreActGrad)
        end slice

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** Reset the hidden state parameters and gradients to zero.
         */
        def reset (): Unit =
            param.set (0.0)
            grad.set (0.0)
            pre_act_grad.setAll (0.0)
        end reset

    end HiddenState

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the previous hidden state at time step t.
     *  @param H The hidden state
     *  @param t The time step
     *  @return The previous hidden state matrix
     */
    protected def get_previous_hidden_state(H: HiddenState, t: Int): MatrixD =
        require(t >= 0, "Time index t must be non-negative.")
        require(t < seq_length, "Time index t must be within the sequence length.")

        if t > 0 then H.param(?, ?, t-1)
        else H.param(?, ?, 0)
    end get_previous_hidden_state

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Zero the gradients of the model parameters.
     *  This method sets all gradients of the weight matrices and bias vectors to zero.
     */
    protected def zero_gradients (): Unit =
        for group <- List(U, V, W)  do group.grad.setAll (0.0)      // Zero gradients for weight matrices
        for group <- List(b_y, b_h) do group.grad.set (0.0)         // Zero gradients for bias vectors
    end zero_gradients

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clip the gradients of the model parameters to a specified threshold.
     *  This method ensures that the gradients do not exceed the given threshold to prevent exploding gradients.
     *  @param threshold  the threshold value for gradient clipping
     */
    override protected def clip_gradients (threshold: Double): Unit =
        for group <- List(U, V, W) do
            val norm = group.grad.normF
            println("Gradient norm: " + norm)
            if norm > threshold then
                group.grad *= (threshold / norm)
                println ("Gradient norm after clipping : " + group.grad.normF)
        end for

        for group <- List(b_y, b_h) do
            val norm = group.grad.norm
            println ("Gradient norm: " + norm)
            if norm > threshold then
                group.grad *= (threshold / norm)
                println ("Gradient norm after clipping : " + group.grad.norm)
        end for
    end clip_gradients

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the model parameters based on the gradients.
     *  This method applies momentum and updates the parameters using the specified
     *  learning rate and batch size.
     *  @param batch_size  the size of the batch used for training
     *  @param leaky       a boolean flag indicating whether to use leaky updates (default is true)
     */
    protected def update_params (batch_size: Int, leaky: Boolean = true): Unit =
        for group <- List (U, V, W) do
            group.velocity *= β
            group.velocity += group.grad * (if leaky then 1 else 1 - β)
            group.param    -= group.velocity * eta / batch_size
        end for

        for group <- List (b_y, b_h) do
            group.velocity *= β
            group.velocity += group.grad * (if leaky then 1 else 1 - β)
            group.param    -= group.velocity * eta / batch_size
        end for
    end update_params

end RNNCell

