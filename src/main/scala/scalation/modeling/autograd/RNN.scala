
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Fri April 25 19:48:13 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Recurrent Neural Networks
 */

package scalation
package modeling
package autograd

import scalation.mathstat.TensorD

import TensorInitializers._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Base class for recurrent cells.
 *  Defines the input/hidden sizes and provides utilities
 *  for initializing the tracking states (e.g., hidden, cell).
 *  Subclasses must specify the number of states, parameters,
 *  and the forward computation.
 *  @see https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/rnn.py#L1492
 */
private [autograd] abstract class RNNCellBase (val inputSize: Int, val hiddenSize: Int)(using ops: AutogradOps)
         extends SeqModule (IndexedSeq.empty):
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Number of state tensors tracked by the cell (e.g., 1 for RNN/GRU, 2 for LSTM).
     */
    def numTrackingStates: Int

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a batch of zero-initialized tracking states.
     *  You pass in the batch size to get properly shaped tensors: (batchSize, hiddenSize, 1)
     */
    def initialTrackingStates (batchSize: Int): IndexedSeq [Variabl] =
        IndexedSeq.fill (numTrackingStates) { Variabl(TensorD.fill (batchSize, hiddenSize, 1, 0.0)) }
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameters of the cell.
     *
     * @return sequence of parameters
     */
    override def parameters: IndexedSeq [Variabl]

end RNNCellBase


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNNCell` class supports a simple RNN cell that updates the hidden state:
 *  h' = activation(W_ih * x + b_ih + W_hh * h + b_hh) using two biases instead of one.
 *  @param inputSize   number of input features
 *  @param hiddenSize  number of hidden units
 *  @param activation  activation function to use: "tanh" (default) or "relu"
 *
 * @see https://pytorch.org/docs/stable/generated/torch.nn.RNNCell.html
 */
class RNNCell (inputSize: Int, hiddenSize: Int, activation: String = "tanh")
              (using ops: AutogradOps)
  extends RNNCellBase (inputSize, hiddenSize):
    
    // W_ih shape = (1, hiddenSize, inputSize)
    val W_ih: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, inputSize), name = Some ("W_ih")
    )
    // W_hh shape = (1, hiddenSize, hiddenSize)
    val W_hh: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, hiddenSize), name = Some ("W_hh")
    )
    // Biases shape = (1, hiddenSize, 1)
    val b_ih: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, 1), name = Some ("b_ih")
    )
    val b_hh: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, 1), name = Some ("b_hh")
    )
    val activationFun: String = activation
    
    override def numTrackingStates: Int = 1
    
    override def parameters: IndexedSeq [Variabl] = IndexedSeq (W_ih, W_hh, b_ih, b_hh)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform the forward pass for the RNN cell without using fused operations.
     *  Computes the next hidden state based on the input and the previous hidden state.
     *  @param inputs an indexed sequence containing:
     *                 - `input`: the input tensor at the current time step
     *                 - `hPrev`: the hidden state tensor from the previous time step
     *  @return an indexed sequence containing the next hidden state tensor
     *  @throws IllegalArgumentException if the number of inputs is not exactly 2
     */
    def forwardUnfused (inputs: IndexedSeq [Variabl]): IndexedSeq [Variabl] =
        inputs match
            case IndexedSeq (input, hPrev) =>
                val xProj  = W_ih.bmm (input) + b_ih
                val hProj  = W_hh.bmm (hPrev) + b_hh
                val preAct = xProj + hProj

                val hNext = activation match
                    case "tanh" => tanh (preAct)
                    case "relu" => relu (preAct)
                    case other  => throw new IllegalArgumentException (s"Unsupported activation: $other")

                IndexedSeq (hNext)

            case _ =>
                throw new IllegalArgumentException (s"RNNCell expects exactly 2 inputs (input, hPrev), got ${inputs.length}")
    end forwardUnfused
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform the forward pass for the RNN cell using fused operations.
     *  Computes the next hidden state based on the input and the previous hidden state.
     *  @param inputs an indexed sequence containing:
     *                 - `input`: the input tensor at the current time step
     *                 - `hPrev`: the hidden state tensor from the previous time step
     *  @return an indexed sequence containing the next hidden state tensor
     *  @throws IllegalArgumentException if the number of inputs is not exactly 2
     */
    override def forward (inputs: IndexedSeq [Variabl]): IndexedSeq [Variabl] =
        inputs match
            case IndexedSeq (input, hPrev) =>
                val fusedCell = RNNCellFused (input, hPrev, W_ih, W_hh,
                                              b_ih, b_hh, activationFun)
                val hNext = fusedCell.forward ()
                IndexedSeq (hNext)

            case _ =>
                throw new IllegalArgumentException (
                    s"RNNCell expects exactly 2 inputs (input, hPrev), got ${inputs.length}")
    end forward

end RNNCell


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNNCell` object provides a factory method for creating instances of the `RNNCell` class.
 */
object RNNCell:
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a new `RNNCell` instance.
     *  @param inputSize   number of input features
     *  @param hiddenSize  number of hidden units
     *  @param activation  activation function to use: "tanh" (default) or "relu"
     *  @param ops         implicit autograd operations
     *  @return a new instance of `RNNCell`
     */
    def apply (inputSize: Int, hiddenSize: Int, activation: String = "tanh")
              (using ops: AutogradOps): RNNCell =
        new RNNCell (inputSize, hiddenSize, activation)

end RNNCell


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GRUCell` class supports a gated recurrent unit cell:
 *  r_t = sigmoid(W_ir * x + b_ir + W_hr * h_{t-1} + b_hr)
 *  z_t = sigmoid(W_iz * x + b_iz + W_hz * h_{t-1} + b_hz)
 *  n_t = tanh(W_in * x + b_in + r_t ⊙ (W_hn * h_{t-1} + b_hn))
 *  h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}
 *  This class defines the parameters and forward computation for a GRU cell.
 *  @see https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
 *  @param inputSize   number of input features
 *  @param hiddenSize  number of hidden units
 */
class GRUCell (inputSize: Int, hiddenSize: Int)(using ops: AutogradOps)
  extends RNNCellBase (inputSize, hiddenSize):
    
    // Reset gate parameters
    /** Weight matrix for the input-to-hidden connection in the reset gate. */
    val W_ir: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, inputSize), name = Some ("W_ir")
    )
    /** Weight matrix for the hidden-to-hidden connection in the reset gate. */
    val W_hr: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, hiddenSize), name = Some ("W_hr")
    )
    /** Bias for the input-to-hidden connection in the reset gate. */
    val b_ir: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, 1), name = Some ("b_ir")
    )
    /** Bias for the hidden-to-hidden connection in the reset gate. */
    val b_hr: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, 1), name = Some ("b_hr")
    )
    
    // Update gate parameters
    /** Weight matrix for the input-to-hidden connection in the update gate. */
    val W_iz: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, inputSize), name = Some ("W_iz")
    )
    /** Weight matrix for the hidden-to-hidden connection in the update gate. */
    val W_hz: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, hiddenSize), name = Some ("W_hz")
    )
    /** Bias for the input-to-hidden connection in the update gate. */
    val b_iz: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, 1), name = Some ("b_iz")
    )
    /** Bias for the hidden-to-hidden connection in the update gate. */
    val b_hz: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, 1), name = Some ("b_hz")
    )
    
    // New gate parameters
    /** Weight matrix for the input-to-hidden connection in the new gate. */
    val W_in: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, inputSize), name = Some ("W_in")
    )
    /** Weight matrix for the hidden-to-hidden connection in the new gate. */
    val W_hn: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, hiddenSize), name = Some ("W_hn")
    )
    /** Bias for the input-to-hidden connection in the new gate. */
    val b_in: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, 1), name = Some ("b_in")
    )
    /** Bias for the hidden-to-hidden connection in the new gate. */
    val b_hn: Variabl = Variabl (
        rnnUniform (1, hiddenSize, hiddenSize, 1), name = Some ("b_hn")
    )
    
    // Initialize biases for the update gate to 0.5
    // to encourage initial retention of previous hidden state (as per common practice)
    // b_iz.data = b_iz.data.map_(_ => 0.5)
    // b_hz.data = b_hz.data.map_(_ => 0.5)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of tracking states for the GRU cell.
     *  For GRU, this is always 1.
     */
    override def numTrackingStates: Int = 1
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameters of the GRU cell.
     *  @return an indexed sequence of `Variabl` objects representing the parameters
     */
    override def parameters: IndexedSeq [Variabl] =
        IndexedSeq (
            // Reset gate
            W_ir, W_hr, b_ir, b_hr,
            // Update gate
            W_iz, W_hz, b_iz, b_hz,
            // New gate
            W_in, W_hn, b_in, b_hn
        )
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform the forward pass for the GRU cell.
     *  Computes the next hidden state based on the input and the previous hidden state.
     *  @param inputs an indexed sequence containing:
     *                 - `input`: the input tensor at the current time step
     *                 - `hPrev`: the hidden state tensor from the previous time step
     *  @return an indexed sequence containing the next hidden state tensor
     *  @throws IllegalArgumentException if the number of inputs is not exactly 2
     */
    override def forward (inputs: IndexedSeq [Variabl]): IndexedSeq [Variabl] =
        inputs match
            case IndexedSeq (input, hPrev) =>
                val fusedCell = GRUCellFused (input, hPrev,
                                              W_ir, W_hr, b_ir, b_hr,
                                              W_iz, W_hz, b_iz, b_hz,
                                              W_in, W_hn, b_in, b_hn)
                val hNext = fusedCell.forward()
                IndexedSeq (hNext)
            case _ =>
                throw new IllegalArgumentException (s"GRUCell expects exactly 2 inputs (input, hPrev), got ${inputs.length}")
    end forward
end GRUCell

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 /** The `GRUCell` object provides a factory method for creating instances of the `GRUCell` class.
  */
 object GRUCell:
 
     //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
     /** Create a new `GRUCell` instance.
      *  @param inputSize   number of input features
      *  @param hiddenSize  number of hidden units
      *  @param ops         implicit autograd operations
      *  @return a new instance of `GRUCell`
      */
     def apply (inputSize: Int, hiddenSize: Int)(using ops: AutogradOps): GRUCell =
         new GRUCell (inputSize, hiddenSize)
 
 end GRUCell

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNNBase` class serves as a wrapper for recurrent neural network cells.
 *  It manages the forward pass through the cell, including handling of input sequences,
 *  initial hidden states, and truncated backpropagation through time (TBPTT).
 *  @see https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/modules/rnn.py#L48
 *  @see https://docs.pytorch.org/docs/stable/generated/torch.nn.RNNBase.html
 *  @param cell  the recurrent cell (e.g., RNNCell, GRUCell) to be wrapped
 */
private [autograd] class RNNBase (val cell: RNNCellBase)
        extends BaseModule (IndexedSeq.empty):
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the parameters of the wrapped recurrent cell.
     *  @return an indexed sequence of `Variabl` objects representing the parameters
     */
    override def parameters: IndexedSeq [Variabl] = cell.parameters

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform the forward pass through the cell without using fused operations.
     *  This method processes the input sequence step-by-step, updating the hidden state
     *  at each time step. Optionally supports TBPTT by detaching hidden states periodically.
     *  @param xs     the input sequence, where each element is a tensor of shape (batchSize, inputDim, 1)
     *  @param h0     optional initial hidden states for the cell (default: zero-initialized)
     *  @param tbptt  the truncation interval for TBPTT (default: 0, meaning no truncation)
     *  @return a tuple containing:
     *          - the output sequence (one tensor per time step)
     *          - the final hidden state(s)
     *  @throws IllegalArgumentException if the input sequence is empty
     */
    def forwardUnfused (xs: IndexedSeq [Variabl], h0: Option [IndexedSeq [Variabl]] = None,
                        tbptt: Int = 0): (IndexedSeq [Variabl], IndexedSeq [Variabl]) =

        require (xs.nonEmpty, "Input sequence must be non-empty")
        val batchSize = xs.head.shape.head

        // initialize all tracking states (1 for RNN/GRU, 2 for LSTM)
        var h: IndexedSeq [Variabl] = h0.getOrElse (cell.initialTrackingStates (batchSize))

        inline def maybeDetach (step: Int): Unit =
            if tbptt > 0 && ((step + 1) % tbptt == 0) then
                h = h.map(_.detach())

        val outputs = xs.zipWithIndex.map { case (x, t) =>
            val nextStates = cell (IndexedSeq(x) ++ h)   // pass input + all states
            h = nextStates                               // update full state vector
            maybeDetach (t)
            nextStates.head                              // convention: first is "h"
        }
        (outputs, h)
    end forwardUnfused

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform the forward pass through the cell, using fused operations if available.
     *  This method processes the input sequence and selects the appropriate computation
     *  path based on the type of the wrapped cell.
     *  @param xs     the input sequence, where each element is a tensor of shape (batchSize, inputDim, 1)
     *  @param h0     optional initial hidden states for the cell (default: zero-initialized)
     *  @param tbptt  the truncation interval for TBPTT (default: 0, meaning no truncation)
     *  @return a tuple containing:
     *          - the output sequence (one tensor per time step)
     *          - the final hidden state(s)
     *  @throws IllegalArgumentException if the input sequence is empty
     */
    def forward (xs: IndexedSeq [Variabl], h0: Option [IndexedSeq [Variabl]] = None,
                 tbptt: Int = 0): (IndexedSeq [Variabl], IndexedSeq [Variabl]) =

        require (xs.nonEmpty, "Input sequence must be non-empty")
        val batchSize = xs.head.shape.head

        // initialize all tracking states (1 for RNN/GRU, 2 for LSTM)
        var h: IndexedSeq [Variabl] = h0.getOrElse (cell.initialTrackingStates (batchSize))

        // dispatch computation based on the type of the cell
        cell match
            // --------------------------------------------------------------------------
            case rnn: RNNCell =>
                val fused = RNNFused (input = xs, hidden = h.head,
                                      W_ih = rnn.W_ih,
                                      W_hh = rnn.W_hh,
                                      b_ih = rnn.b_ih,
                                      b_hh = rnn.b_hh,
                                      activation = rnn.activationFun,
                                      tbptt = tbptt)
                val (outs, last) = fused.forwardAll ()
                (outs, IndexedSeq (last))

            // --------------------------------------------------------------------------
            // fallback path for other cells (GRU, custom, etc.)
            case _ =>
                inline def maybeDetach (step: Int): Unit =
                    if tbptt > 0 && ((step + 1) % tbptt == 0) then
                        h = h.map(_.detach ())
                end maybeDetach

                val outputs = xs.zipWithIndex.map { case (x, t) =>
                    val nextStates = cell (IndexedSeq(x) ++ h)
                    h = nextStates
                    maybeDetach (t)
                    nextStates.head        // convention: first is "h"
                }
                (outputs, h)
        end match
    end forward

end RNNBase


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNN` class implements a multi-layer recurrent neural network (RNN).
 *  It supports stacked RNN layers, where each layer processes the input sequence
 *  and passes its output to the next layer. The class also provides methods for
 *  parameter retrieval and forward computation.
 *  @see https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
 *  @param inputSize   number of features in the input at each time step
 *  @param hiddenSize  number of features in the hidden state
 *  @param numLayers   number of stacked RNN layers (default: 1)
 *  @param activation  activation function to use: "tanh" (default) or "relu"
 *  @param ops         implicit autograd operations
 */
class RNN (inputSize: Int, hiddenSize: Int, val numLayers: Int = 1,
           activation: String = "tanh")(using ops: AutogradOps)
      extends BaseModule:
    
    /** Layers of the RNN, each represented by an `RNNBase` instance. */
    private val layers: IndexedSeq [RNNBase] =
        (0 until numLayers).map { layerIdx =>
            val inDim = if layerIdx == 0 then inputSize else hiddenSize
            RNNBase (RNNCell (inDim, hiddenSize, activation))
        }
    
    /** Retrieve a specific layer of the RNN.
     *  @param i  index of the layer to retrieve
     *  @return the `RNNBase` instance representing the layer
     */
    def layer (i: Int): RNNBase = layers(i)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the parameters of all layers in the RNN.
     *  @return an indexed sequence of `Variabl` objects representing the parameters
     */
    override def parameters: IndexedSeq [Variabl] = layers.flatMap (_.parameters)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform the forward pass through all layers of the RNN.
     *  Processes the input sequence through each layer, updating the hidden states
     *  at each time step. Optionally supports truncated backpropagation through time (TBPTT).
     *  @param inputSeq  the input sequence, where each element is a tensor of shape (batchSize, inputDim, 1)
     *  @param h0        optional initial hidden states for each layer (default: zero-initialized)
     *  @param tbptt     the truncation interval for TBPTT (default: 0, meaning no truncation)
     *  @return a tuple containing:
     *          - the output sequence from the top layer (one tensor per time step)
     *          - the final hidden states for all layers
     *  @throws IllegalArgumentException if the input sequence is empty
     */
    def forward (inputSeq: IndexedSeq [Variabl],
                 h0: Option [IndexedSeq [Variabl]] = None,
                 tbptt: Int = 0): (IndexedSeq [Variabl], IndexedSeq [Variabl]) =
        
        require (inputSeq.nonEmpty, "Input sequence cannot be empty")
        
        val batchSize = inputSeq.head.shape.head
        
        // Initialize hidden states for all layers
        val hidden: IndexedSeq [Variabl] =
            h0.getOrElse { layers.map (_.cell.initialTrackingStates (batchSize).head) }
        
        var layerInput: IndexedSeq [Variabl] = inputSeq                           // Sequence flowing upward
        val finalHidden = collection.mutable.ArrayBuffer.empty [Variabl]
        
        // ----- pass sequence through each RNN layer -----
        for (layer, hInit) <- layers.zip (hidden) do                                         // forward this layer through all time steps
            val (layerOutput, hLast) = layer.forwardUnfused (layerInput, Some (IndexedSeq(hInit)), tbptt)
            layerInput = layerOutput                                                         // becomes input to next layer
            finalHidden.append (hLast.head)                                                       // store final hidden of this layer
        
        val outputsTop = layerInput                                                          // after last layer
        (outputsTop, finalHidden.toIndexedSeq)
    end forward

end RNN


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNN` object provides a factory method for creating instances of the `RNN` class.
 */
object RNN:
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factory method for creating a standard RNN instance.
     *  @param inputSize   number of features in the input at each time step
     *  @param hiddenSize  number of features in the hidden state
     *  @param numLayers   number of stacked RNN layers (default = 1)
     *  @param activation  nonlinearity to apply ("tanh" or "relu", default = "tanh")
     *  @param ops         implicit autograd operations
     *  @return an instance of `RNN`
     */
    def apply( inputSize: Int, hiddenSize: Int, numLayers: Int = 1, activation: String = "tanh")
              (using ops: AutogradOps): RNN =
        new RNN (inputSize, hiddenSize, numLayers, activation)

end RNN


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GRU` class implements a multi-layer gated recurrent unit (GRU) network.
 *  It supports stacked GRU layers, where each layer processes the input sequence
 *  and passes its output to the next layer. The class also provides methods for
 *  parameter retrieval and forward computation.
 *  @see https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
 *  @param inputSize   number of features in the input at each time step
 *  @param hiddenSize  number of features in the hidden state
 *  @param numLayers   number of stacked GRU layers (default: 1)
 */
class GRU (inputSize: Int, hiddenSize: Int, val numLayers: Int = 1)
      extends BaseModule:
    
    /** Layers of the GRU, each represented by an `RNNBase` instance. */
    private val layers: IndexedSeq [RNNBase] =
        (0 until numLayers).map { layerIdx =>
            val inDim = if layerIdx == 0 then inputSize else hiddenSize
            RNNBase (GRUCell (inDim, hiddenSize))
        }
    
    /** Retrieve a specific layer of the GRU.
     *  @param i  index of the layer to retrieve
     *  @return the `RNNBase` instance representing the layer
     */
    def layer (i: Int): RNNBase = layers(i)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Retrieve the parameters of all layers in the GRU.
     *
     *  @return an indexed sequence of `Variabl` objects representing the parameters
     */
    override def parameters: IndexedSeq [Variabl] = layers.flatMap (_.parameters)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform the forward pass through all layers of the GRU.
     *  Processes the input sequence through each layer, updating the hidden states
     *  at each time step. Optionally supports truncated backpropagation through time (TBPTT).
     *  @param inputSeq  the input sequence, where each element is a tensor of shape (batchSize, inputDim, 1)
     *  @param h0        optional initial hidden states for each layer (default: zero-initialized)
     *  @param tbptt     the truncation interval for TBPTT (default: 0, meaning no truncation)
     *  @return a tuple containing:
     *          - the output sequence from the top layer (one tensor per time step)
     *          - the final hidden states for all layers
     *  @throws IllegalArgumentException if the input sequence is empty
     */
    def forward (inputSeq: IndexedSeq [Variabl], h0: Option [IndexedSeq [Variabl]] = None,
                 tbptt: Int = 0): (IndexedSeq [Variabl], IndexedSeq [Variabl]) =
        
        require (inputSeq.nonEmpty, "Input sequence cannot be empty")
        
        val batchSize = inputSeq.head.shape.head
        
        // Initialize hidden states for all layers
        val hidden: IndexedSeq [Variabl] =
            h0.getOrElse { layers.map (_.cell.initialTrackingStates (batchSize).head) }
        
        var layerInput: IndexedSeq [Variabl] = inputSeq
        val finalHidden = collection.mutable.ArrayBuffer.empty [Variabl]
        
        // ----- pass sequence through each GRU layer -----
        for (layer, hInit) <- layers.zip(hidden) do
            val (layerOutput, hLast) = layer.forward (layerInput, Some (IndexedSeq (hInit)), tbptt)
            layerInput = layerOutput
            finalHidden.append (hLast.head)
        
        val outputsTop = layerInput
        (outputsTop, finalHidden.toIndexedSeq)
    end forward

end GRU


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `GRU` object provides a factory method for creating instances of the `GRU` class.
 */
object GRU:
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Factory method for creating a standard GRU instance.
     *  @param inputSize   number of features in the input at each time step
     *  @param hiddenSize  number of features in the hidden state
     *  @param numLayers   number of stacked GRU layers (default = 1)
     *  @return an instance of `GRU`
     */
    def apply (inputSize: Int, hiddenSize: Int, numLayers: Int = 1): GRU =
//            (using ops: AutogradOps): GRU =
        new GRU (inputSize, hiddenSize, numLayers)

end GRU

