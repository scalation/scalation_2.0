
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula, John Miller
 *  @version 2.0
 *  @date    Mon Aug  4 21:12:40 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Recurrent Neural Network (RNN) for Multivariate Time Series
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Standard RNN (GRU/LSTM) Formats for Multivariate Time Series Data:
 *  Matlab:
 *
 *  Keras: 3D format expected by GRU/LSTM is [samples, timesteps, features].
 *      => indexing [timestamp t, lags k, variable j]
 *  PyTorch:
 */

package scalation
package modeling
package forecasting
package neuralforecasting

import scalation.mathstat.{MatrixD, Plot, TensorD, VectorD}
import scalation.modeling.forecasting.MakeMatrix4TS.makeMatrix4L

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNN` class implements Recurrent Neural Network (RNN) via Back Propagation Through
 *  Time (BPTT).  At each time point x_t, there is a vector representing several variables
 *  or the encoding of a word.  Intended to work for guessing the next work in a sentence
 *  or for multi-horizon forecasting.
 *  @param x       the input sequence/time series
 *  @param y       the output sequence/time series
 *  @param y_orig  the original target matrix before any preprocessing
 *  @param fname   the feature/variable names
 *  @param n_mem   the size for hidden state (h) (dimensionality of memory)
 */
class RNN (override val x: TensorD, override val y: TensorD, val y_orig: MatrixD,
           fname: Array[String] = null, override val n_mem: Int = 8)
      extends RNNCell, FitM:

    override val CLASSIF = false                                        // Indicates whether the model is for classification (false for forecasting)
// FIX -- use hyper-parameters
    override val max_epochs = 100                                       // Maximum number of training epochs
    override val eta = 0.0054                                           // Learning rate for the optimizer
    override val batch_size = 32                                        // Size of each training batch
    override val truncation_length = 45                                 // Length of the sequence truncation for backpropagation through time
    override val β = 0.9                                                // Momentum term for the optimizer
    override val threshold = 100.0                                      // Threshold for gradient clipping to avoid exploding gradients

    override val seq_length: Int = x.dim                                // Length of the input sequences
    override val n_var: Int = x.dim2                                    // Number of variables in the input tensor
    override val n_seq: Int = x.dim3                                    // Number of sequences in the input tensor

    override val loss_per_epoch: VectorD = new VectorD (max_epochs)     // Vector to store the loss value for each epoch
    override val L: VectorD = new VectorD (seq_length)                  // Vector to store the loss value for each time step in a sequence

    private val yp: TensorD = new TensorD (seq_length, y.dim2, n_seq)   // Tensor to store the predictions made by the model
    private val L_epoch = new VectorD (max_epochs)                      // Vector to store the loss value for each epoch

    if fname != null then println (s"RNN: fname = $fname")              // Print the file names if provided

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Trains the RNN model using the provided input and target tensors.
     *  This function implements the training loop for the RNN model. It iterates over the specified
     *  number of epochs, divides the data into batches, performs forward and backward propagation,
     *  clips gradients to avoid exploding gradients, and updates the model parameters.
     *  @param x  the input tensor of shape [seq_length, n_var, n_seq]
     *  @param y  the target tensor of shape [seq_length, output_dim, n_seq]
     *  @return Unit (no return value, updates are made to class members)
     */
    override def train (x: TensorD = x, y: TensorD = y): Unit =
        for it <- 1 to max_epochs do
            val n_batches = math.ceil (n_seq.toDouble / batch_size).toInt
            for i <- 0 until n_batches do
                val batch_start        = i * batch_size
                val batch_end          = math.min (n_seq - 1, (i + 1) * batch_size - 1)
                val current_batch_size = batch_end - batch_start + 1
                println (s"batch_start = $batch_start, batch_end = $batch_end")

                val x_batch: TensorD = x(null, null, (batch_start, batch_end + 1))
                val y_batch: TensorD = y(null, null, (batch_start, batch_end + 1))

                val H_batch = if current_batch_size == batch_size then H else H.slice (current_batch_size)

                zero_gradients ()

                forward (x_batch, y_batch, batch_start, batch_end, L, H_batch)  // forward propagate: get intermediate and output results

                backward (x_batch, y_batch, batch_start, batch_end, H_batch)    // back propagate: calculate gradients (partial derivatives)

                clip_gradients (threshold)                                      // clip gradients to avoid exploding gradients

                update_params (current_batch_size, leaky = false)               // update parameters (weights and biases)

                H.reset ()
                H_batch.reset ()
            end for

            val mse = L.sum / seq_length                                        // mean squared error
            println (s"train: for epoch $it: loss function L = $L")
            banner (s"train: for epoch $it: sum of loss function L.sum = ${L.sum}")
            banner (s"train: for epoch $it: mean squared error = $mse")
            L_epoch(it - 1) = L.sum
        end for
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs the forward pass of the RNN, computing the predictions and loss for the network.
     *  This function implements the forward propagation for a batch of sequences.
     *  It calculates the predictions for each time step and computes the loss
     *  based on the difference between the predictions and the target values.
     *  @param x            the input tensor of shape [seq_length, n_var, batch_size]
     *  @param y            the target tensor of shape [seq_length, output_dim, batch_size]
     *  @param batch_start  the starting index of the current batch in the full dataset
     *  @param batch_end    the ending index (inclusive) of the current batch in the full dataset
     *  @param L            the loss vector to store the loss values for each time step
     *  @param H            the hidden state tensor, default is the class member H
     *  @return Unit (no return value, updates are made to class members)
     */
    def forward (x: TensorD = x, y: TensorD = y, batch_start: Int, batch_end: Int, L: VectorD = L,
                 H: HiddenState = H): Unit =
        val yp_batch = yp(null, null, (batch_start, batch_end + 1))                // Extract the batch of predictions
        for t <- 0 until seq_length do                                             // Iterate over each time step in the sequence
            val H_prev = get_previous_hidden_state(H, t)                           // Get the previous hidden state

            H.param (?, ?, t) = tanh_ (U.param * x(t) + W.param * H_prev +^ b_h.param)   // Compute current hidden state using the tanh activation function

            if CLASSIF then
                yp_batch(t) = softmax_m ((V.param * H.param(?, ?, t)) + b_y.param)   // Compute predictions and loss based on the task (classif or fcast)
                L(t) = (-y(t) * log_ (yp_batch(t))).sum                            // cross-entropy loss function
            else
                yp_batch(t) = (V.param * H.param(?, ?, t)) +^ b_y.param            // activation: id for forecasting
                L(t) = (y(t) - yp_batch(t)).normFSq / 2.0                          // SSE loss function
        end for

        yp(?, ?, batch_start to batch_end) = yp_batch                              // Update the class member yp with the batch predictions
    end forward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Performs the backward pass of the RNN, computing gradients for the network parameters.
     *  This function implements backpropagation through time (BPTT) for a batch of sequences.
     *  It calculates the gradients of the loss with respect to the network parameters,
     *  which are then used to update the weights in the training process.
     *  @param x            the input tensor of shape [seq_length, n_var, batch_size]
     *  @param y            the target tensor of shape [seq_length, output_dim, batch_size]
     *  @param batch_start  the starting index of the current batch in the full dataset
     *  @param batch_end    the ending index (inclusive) of the current batch in the full dataset
     *  @param H            the hidden state tensor, default is the class member H
     *  @return Unit (no return value, updates are made to class members)
     */
    def backward (x: TensorD, y: TensorD, batch_start: Int, batch_end: Int,
                  H: HiddenState = H): Unit =
        val current_batch_size = batch_end - batch_start + 1                // Calculate the current batch size
        val yp_batch = yp(null, null, (batch_start, batch_end + 1))         // Extract the batch of predictions

        val truncated_start = math.max (0, seq_length - truncation_length)  // Determine the starting point for truncation
        var dyp = yp_batch(seq_length - 1) - y(seq_length - 1)              // Initialize the gradient of the output predictions

        var dh_next = new MatrixD(n_mem, current_batch_size)                // Initialize the gradient of the next hidden state

        for t <- seq_length - 1 to truncated_start by -1 do                 // Iterate backwards through the sequence
            val H_prev = get_previous_hidden_state(H, t)                    // Get the previous hidden state
            dyp = yp_batch(t) - y(t)                                        // Calculate the gradient of the output predictions
            b_y.grad += dyp.sumVr                                           // Update the gradient of the output bias
            V.grad   += dyp * H.param(?, ?, t).ᵀ                            // Update the gradient of the output weights
            H.grad(?, ?, t) = (V.param.ᵀ * dyp) + dh_next                   // Calculate the gradient of the hidden state

            H.pre_act_grad = tanhD_m (H.param(?, ?, t)) ⊙ H.grad(?, ?, t)   // Calculate the pre-activation gradient of the hidden state
            b_h.grad += H.pre_act_grad.sumVr                                // Update the gradient of the hidden bias

            U.grad += H.pre_act_grad * x(t).ᵀ                               // Update the gradient of the input weights
            W.grad += H.pre_act_grad * H_prev.ᵀ                             // Update the gradient of the hidden weights
            dh_next = W.param.ᵀ * H.pre_act_grad                            // Calculate the gradient of the next hidden state
        end for
    end backward

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reconstructs the full-length predictions from overlapping sequences.
     *  This function takes the predictions made on overlapping sequences and
     *  reconstructs them into a single, full-length prediction matrix. It
     *  aggregates predictions for each time step and averages them based on
     *  the number of contributions.
     *  @param yp       the tensor of predictions, where each slice represents predictions
     *                  for a sequence. Dimensions: [seq_len, num_variables, num_sequences]
     *  @param seq_len  the length of each sequence used for prediction
     *  @return A matrix of reconstructed predictions, where each row represents
     *         a time step and each column represents a variable. The number of
     *         rows is equal to the original time series length, and the number
     *         of columns is equal to the number of predicted variables.
     */
    private def reconstruct_predictions (yp: TensorD, seq_len: Int): MatrixD =
        val original_len    = yp.dim3 + seq_len - 1                         // Calculate the original target length
        val ypReconstructed = new MatrixD (original_len, yp.dim2)           // Placeholder for aggregated predictions
        val contributions   = new VectorD (original_len)                    // Count contributions for each time step

        for seq_idx <- 0 until yp.dim3 do                                   // Iterate over sequences
            val start_idx = seq_idx                                         // Starting index for the current sequence

            for t <- 0 until seq_len do                                     // Aggregate predictions and count contributions
                for var_idx <- 0 until yp.dim2 do
                    ypReconstructed(start_idx + t, var_idx) += yp(t, var_idx, seq_idx)
                contributions(start_idx + t) += 1

        for t <- 0 until original_len do                                    // Average predictions by dividing by contributions
            if contributions(t) > 0 then
                for var_idx <- 0 until yp.dim2 do
                    ypReconstructed(t, var_idx) /= contributions(t)

        ypReconstructed                                                     // Return the reconstructed predictions
    end reconstruct_predictions

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tests the RNN model by reconstructing predictions and comparing them with the original data.
     *  This function reconstructs the full-length predictions from overlapping sequences,
     *  prints the dimensions of the reconstructed predictions and original data, and plots
     *  the predictions against the original data. It also calculates and prints the SMAPE
     *  and MAE values for each variable, and plots the loss function over epochs.
     *  @param original_extremes  a tuple representing the original extremes of the data for unscaling.
     *                            default is (1.0, 1.0).
     */
    def test (original_extremes: (Double, Double) = (1.0, 1.0)): Unit =
        val yp_reconstructed = reconstruct_predictions(yp, seq_length)      // Reconstruct full-length predictions from overlapping sequences

        println(s"yp_reconstructed.dims = ${yp_reconstructed.dims}")        // Print dimensions of reconstructed predictions and original data
        println(s"y_orig.dims = ${y_orig.dims}")

        new Plot (null, y_orig(?, 0), yp_reconstructed(?, 0),
                  "Plot of y vs yp for RNN", lines = true)                  // Plot first variable of the original and reconstructed predictions
        for col <- 0 until y_orig.dim2 do                                   // Iterate over each variable to calc and print SMAPE and MAE values
            val y_unscaled  = unscaleV (original_extremes, (-2.0, 2.0))(y_orig(?, col))
            val yp_unscaled = unscaleV (original_extremes, (-2.0, 2.0))(yp_reconstructed(?, col))
//          banner ("smape value = " + Fit.smapeF(y_unscaled, yp_unscaled))    how else to use smape?
            banner ("mae value = " + Fit.mae(y_unscaled, yp_unscaled))
        end for

        new Plot (VectorD.range (0, max_epochs), L_epoch, null,
                  "Plot of Loss Function vs Epoch", lines = true)           // Plot the loss function over epochs

        banner ("minimum loss epoch = " + L_epoch.argmin())                 // Print epoch with the minimum loss and the minimum loss value
        banner ("minimum loss value = " + L_epoch.min)
    end test

end RNN


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNN` companion object provides factory methods.
 */
object RNN:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Creates sequences for the RNN model from the input and output matrices.
     *  This function takes the input and output matrices and creates sequences of a specified length.
     *  Each sequence is a slice of the original matrices, and the function returns tensors containing
     *  these sequences.
     *  @param x                the input matrix of shape [n_samples, n_features]
     *  @param yy               the output matrix of shape [n_samples, n_output_features]
     *  @param sequence_length  the length of each sequence
     *  @return A tuple containing two tensors:
     *         - x_sequences: The input sequences tensor of shape [sequence_length, n_features, n_sequences]
     *         - y_sequences: The output sequences tensor of shape [sequence_length, n_output_features, n_sequences]
     */
    def create_sequences (x: MatrixD, yy: MatrixD, sequence_length: Int): (TensorD, TensorD) =
        val n_samples = x.dim
        val n_sequences = n_samples - sequence_length + 1
        val x_sequences: TensorD = new TensorD (sequence_length, x.dim2, n_sequences)
        val y_sequences: TensorD = new TensorD (sequence_length, yy.dim2, n_sequences)

        for seq <- 0 until n_sequences do
            val sequence = seq until (seq + sequence_length)
            x_sequences(?, ?, seq) = x(sequence)
            y_sequences(?, ?, seq) = yy(sequence)
        end for

        (x_sequences, y_sequences)
    end create_sequences

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Creates an output matrix for the RNN model from the target vector.
     *  This function takes a target vector and creates a matrix where each row represents
     *  a shifted version of the target vector. The number of columns in the matrix is equal
     *  to the forecasting horizon (hh). If the shifted index exceeds the length of the target
     *  vector, the value is set to -0.0.
     *  @param y   the target vector of shape [n_samples]
     *  @param hh  the forecasting horizon (number of future steps to predict)
     *  @return A matrix of shape [n_samples - 1, hh] where each row is a shifted version of the target vector
     */
    private def makeOutputMatrix (y: VectorD, hh: Int): MatrixD =
        val yy = new MatrixD (y.dim - 1, hh)
        for t <- 0 until yy.dim do
            for j <- 0 until hh do yy(t, j) = if t + 1 + j >= y.dim then -0.0 else y(t + 1 + j)
        yy
    end makeOutputMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Builds the input and output matrices for time series forecasting.
     *  This function takes a target vector and creates input and output matrices for time series forecasting.
     *  The input matrix is created using the specified number of lags, and the output matrix is created using
     *  the specified forecasting horizon. The function prints the dimensions of the input and output matrices
     *  and the value of the last element in the target vector.
     *  @param y         the target vector of shape [n_samples]
     *  @param lags      the number of lags to use for creating the input matrix
     *  @param hh        the forecasting horizon (number of future steps to predict)
     *  @param backcast  A boolean flag indicating whether to include backcasting (default is true)
     *  @return A tuple containing two matrices:
     *         - The input matrix of shape [n_samples - lags, lags]
     *         - The output matrix of shape [n_samples - 1, hh]
     */
    def buildMatrix4TS (y: VectorD, lags: Int, hh: Int, backcast: Boolean = true): (MatrixD, MatrixD) =
        val x  = makeMatrix4L (y, lags, backcast)
        val yy = makeOutputMatrix (y, hh)
        println (s"dims of x = ${x.dims}")
        println (s"dims of yy = ${yy.dims}")
        println (s"last element in y = ${y(y.dim - 1)}")
        (x, yy)
    end buildMatrix4TS

end RNN


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Main function to test the RNN model on COVID-19 new deaths data.
 *  This function loads the COVID-19 new deaths data, preprocesses it, creates sequences
 *  for the RNN model, trains the model, and tests it. It prints the dimensions of the
 *  input and output matrices, and the value of the last element in the dataset.
 *  > runMain scalation.modeling.forecasting.neuralforecasting
 */
@main def rNNTest4 (): Unit =

    val lags    = 4                                                     // Define the number of lags
    val hh      = 1                                                     // Forecasting horizon
    val seq_length = 45                                                 // Sequence length

    var y = Example_Covid.loadData_y ("new_deaths")                     // Load the COVID-19 new deaths data

    y = y(0 until 116)                                                  // Select the first 116 data points

    val original_extremes = extreme (y)                                 // Calculate the original extremes of the data

    println ("original_extremes.type = " + original_extremes.getClass)  // Print the type of the original extremes

    val y_s = scaleV (extreme(y), (-2.0, 2.0))(y)                       // Scale the data to the active domain of sigmoid and tanh functions

    val (x, yy) = RNN.buildMatrix4TS (y_s, lags, hh)                    // Build the input and output matrices for time series forecasting
    val (x_seq, y_seq) = RNN.create_sequences (x, yy, seq_length)       // Create sequences for the RNN model

    println (s"x.dims = ${x.dims}, yy.dims = ${yy.dims}")               // Print the dimensions of the input and output matrices
    println (s"x_seq.dims = ${x_seq.dims}, y_seq.dims = ${y_seq.dims}")   // Print the dimensions of the sequences

    banner ("Create a Recurrent Neural Network Unit (RNN)")             // Create and train the RNN model
    val mod = new RNN (x_seq, y_seq, yy)                                // Call constructor
    mod.train ()                                                        // Train the model
    mod.test (original_extremes)                                        // Test the model

    print ("y(115) = " + y(115))                                        // Print the value of the last element in the dataset

end rNNTest4

// ------------------------------------------------------------------------------------------------
// Made a change to line number 130 in MakeMatrix4TS due to mismatch in dimensions
// Fine tune RNN again or modify MakeMatrix4TS?
// Made a change in NormFsq in MatrixD to change the dimensions from column to row
// Made a change in TensorD to add new methods
// ------------------------------------------------------------------------------------------------

