//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Tue Nov 11 10:44:32 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Unit Tests for Core RNN/GRU Autograd Functionality
 */

package scalation
package modeling
package autograd

import scalation.mathstat.{MatrixD, TensorD}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNNTestCore` object defines a suite of @main entrypoints that exercise the
 *  autograd system using recurrent neural network components. These tests verify:
 *    - forward computation consistency for RNNCell and GRUCell
 *    - correct propagation of hidden states through `RNNBase`
 *    - correctness of gradient backpropagation through time
 *    - multilayer RNN/GRU behavior and parameter interaction
 *    - construction and export of autograd computation graphs for debugging
 *  All tests use synthetic inputs and manually assigned weights/biases to ensure
 *  deterministic behavior to validate against PyTorch, enabling reliable 
 *  gradient-checking via finite differences using `GradCheck.gradCheck`.
 *  @note This file focuses exclusively on core autograd correctness and does not
 *        contain any real-data forecasting experiments.
 */
object RNNTestCore:
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `rnnCellTest` main function tests the `RNNCell` class.
     * > runMain scalation.modeling.autograd.RnnTest.rnnCellTest
     */
    @main def rnnCellTest (): Unit =
        banner ("Simple RNN Cell - Forward + Backward Test")
        
        val inputSize = 3
        val hiddenSize = 4
//      val batchSize = 2
        
        // Create a toy RNN cell (tanh activation)
        val cell = RNNCell(inputSize, hiddenSize, activation = "tanh")
        
        // ---- Manually set weights/biases for reproducibility ----
        
        // W_ih: shape (1, hiddenSize, inputSize) = (1,4,3)
        val WihM = new MatrixD (4, 3, Array (
            Array (0.1, 0.2, 0.3),
            Array (0.4, 0.5, 0.6),
            Array (0.7, 0.8, 0.9),
            Array (1.0, 1.1, 1.2)
        ))
        cell.W_ih.data = TensorD (WihM)
        
        // W_hh: shape (1, hiddenSize, hiddenSize) = (1,4,4)
        val WhhM = new MatrixD (4, 4, Array (
            Array (0.1, 0.0, 0.0, 0.0),
            Array (0.0, 0.1, 0.0, 0.0),
            Array (0.0, 0.0, 0.1, 0.0),
            Array (0.0, 0.0, 0.0, 0.1)
        ))
        cell.W_hh.data = TensorD (WhhM)
        
        // Biases: (1, hiddenSize, 1)
        val bIhM = new MatrixD (4, 1, Array (
            Array (0.01),
            Array (0.02),
            Array (0.03),
            Array (0.04)
        ))
        cell.b_ih.data = TensorD (bIhM)
        
        val bHhM = new MatrixD (4, 1, Array (
            Array (0.0),
            Array (0.0),
            Array (0.0),
            Array (0.0)
        ))
        cell.b_hh.data = TensorD (bHhM)
        
        // ---- Dummy batch input ----
        // x: shape (batch, inputSize, 1) = (2,3,1)
        val x0 = new MatrixD (3, 1, Array (Array (0.1), Array (0.2), Array (0.3)))
        val x1 = new MatrixD (3, 1, Array (Array (0.4), Array (0.5), Array (0.6)))
        val x = Variabl (TensorD (x0, x1), name = Some ("x"))
        
        // ---- Dummy previous hidden state ----
        // hPrev: shape (batch, hiddenSize, 1) = (2,4,1)
        val h0 = new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04)))
        val h1 = new MatrixD (4, 1, Array (Array (0.05), Array (0.06), Array (0.07), Array (0.08)))
        val hPrev = Variabl (TensorD (h0, h1), name = Some ("hPrev"))
        
        // Forward through the RNNCell
        val hNext = cell (IndexedSeq (x, hPrev)).head // Since RNNCell returns IndexedSeq of length 1
        
        println (s"Input x shape: ${x.shape}, values: ${x.data}")
        println (s"Prev hidden shape: ${hPrev.shape}, values: ${hPrev.data}")
        println (s"Next hidden shape: ${hNext.shape}, values: ${hNext.data}")
        
        val R = TestReport()
        
        // Gradient check w.r.t input
        R.record ("RNNCell - Input GradCheck") {
            GradCheck.gradCheck (x, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
        }
        
        // Gradient check w.r.t previous hidden state
        R.record ("RNNCell - Hidden GradCheck") {
            GradCheck.gradCheck (hPrev, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
        }
        
        // Gradient check w.r.t parameters
        R.record ("RNNCell - W_ih GradCheck") {
            GradCheck.gradCheck (cell.W_ih, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
        }
        R.record ("RNNCell - W_hh GradCheck") {
            GradCheck.gradCheck (cell.W_hh, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
        }
        R.record ("RNNCell - b_ih GradCheck") {
            GradCheck.gradCheck (cell.b_ih, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
        }
        R.record ("RNNCell - b_hh GradCheck") {
            GradCheck.gradCheck (cell.b_hh, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
        }

        R.summary ("RNNCell Forward + Backward Test")
        
        val outPath = "target/autograd/visualization/computation_graph_rnn_cell.dot"
        GraphExporter.writeDot(cell (IndexedSeq (x, hPrev)).head, outPath, renderSvg = true)
        println (s"Computation graph DOT written to $outPath")
    end rnnCellTest
    
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `rnnBaseTest` main function tests the `RNNBase` class.
     *  > runMain scalation.modeling.autograd.rnnBaseTest
     */
    @main def rnnBaseTest (): Unit =
        banner ("RNNBase - Forward + Backward Test")
        
        val inputSize = 3
        val hiddenSize = 4
//      val seqLen = 2
//      val batchSize = 2
        
        val cell    = RNNCell(inputSize, hiddenSize, activation = "tanh")
        val rnnBase = new RNNBase(cell)
        
        // ---- Manually set weights/biases (same as RNNCell test for reproducibility) ----
        val WihM = new MatrixD (4, 3, Array (
            Array (0.1, 0.2, 0.3),
            Array (0.4, 0.5, 0.6),
            Array (0.7, 0.8, 0.9),
            Array (1.0, 1.1, 1.2)
        ))
        cell.W_ih.data = TensorD (WihM)
        
        val WhhM = new MatrixD (4, 4, Array (
            Array (0.1, 0.0, 0.0, 0.0),
            Array (0.0, 0.1, 0.0, 0.0),
            Array (0.0, 0.0, 0.1, 0.0),
            Array (0.0, 0.0, 0.0, 0.1)
        ))
        cell.W_hh.data = TensorD (WhhM)
        
        val bIhM = new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04)))
        cell.b_ih.data = TensorD (bIhM)
        
        val bHhM = new MatrixD (4, 1, Array (Array (0.0), Array (0.0), Array (0.0), Array (0.0)))
        cell.b_hh.data = TensorD (bHhM) // all weights are shared across timesteps
        
        // ---- Sequence input ----
        // Two timesteps, each (batch, inputSize, 1)
        val x0 = new MatrixD (3, 1, Array (Array (0.1), Array (0.2), Array (0.3)))
        val x1 = new MatrixD (3, 1, Array (Array (0.4), Array (0.5), Array (0.6)))
        val xs = IndexedSeq (
            Variabl (TensorD (x0, x1), name = Some ("x_t0")),
            Variabl (TensorD (x0, x1), name = Some ("x_t1")) // reuse same input for step 1 for simplicity
        )
        
        // ---- Initial hidden ----
        val h0_batch0 = new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04)))
        val h0_batch1 = new MatrixD (4, 1, Array (Array (0.05), Array (0.06), Array (0.07), Array (0.08)))
        val h0 = IndexedSeq (Variabl (TensorD (h0_batch0, h0_batch1), name = Some ("h0")))
        
        // Forward through RNNBase
        val (outputs, hLast) = rnnBase.forward (xs, Some (h0))
        
        println (s"Outputs per timestep: ${outputs.map(_.data)}")
        println (s"Final hidden: ${hLast.head.data}")
        
        // ---- Gradient checks ----
        val R = TestReport()
        R.record ("RNNBase - x_t0 GradCheck") {
            GradCheck.gradCheck (xs(0), () => rnnBase.forward (xs, Some (h0))._1.last.sum, quiet = true)
        }
        R.record ("RNNBase - h0 GradCheck") {
            GradCheck.gradCheck (h0.head, () => rnnBase.forward (xs, Some (h0))._1.last.sum, quiet = true)
        }
        R.record ("RNNBase - W_ih GradCheck") {
            GradCheck.gradCheck (cell.W_ih, () => rnnBase.forward (xs, Some (h0))._1.last.sum, quiet = true)
        }
        R.record ("RNNBase - W_hh GradCheck") {
            GradCheck.gradCheck (cell.W_hh, () => rnnBase.forward (xs, Some (h0))._1.last.sum, quiet = true)
        }
        R.record ("RNNBase - b_ih GradCheck") {
            GradCheck.gradCheck (cell.b_ih, () => rnnBase.forward (xs, Some (h0))._1.last.sum, quiet = true)
        }
        R.record ("RNNBase - b_hh GradCheck") {
            GradCheck.gradCheck (cell.b_hh, () => rnnBase.forward (xs, Some (h0))._1.last.sum, quiet = true)
        }
        
        R.summary ("RNNBase Forward + Backward Test")
        
        val outPath = "target/autograd/visualization/computation_graph_rnn_base.dot"
        GraphExporter.writeDot(hLast.head, outPath, renderSvg = true)
        println (s"Computation graph DOT written to $outPath")
    end rnnBaseTest
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `rnnMultiLayerTest` main function tests the `RNN` class.
     * > runMain scalation.modeling.autograd.rnnMultiLayerTest
     */
    @main def rnnMultiLayerTest (): Unit =
        banner ("RNN Multi Layer (2 layers) - Forward + Backward Test")
        
        val inputSize = 3
        val hiddenSize = 4
//      val seqLen = 2
//      val batchSize = 2
        val numLayers = 2
        
        // ---- Construct 2-layer RNN ----
        val rnn = RNN(inputSize, hiddenSize, numLayers, activation = "tanh")
        
        // Layer 0
        val cell0 = rnn.layer(0).cell.asInstanceOf [RNNCell]
        val Wih0 = new MatrixD (4, 3, Array (
            Array (0.1, 0.2, 0.3),
            Array (0.4, 0.5, 0.6),
            Array (0.7, 0.8, 0.9),
            Array (1.0, 1.1, 1.2)
        ))
        cell0.W_ih.data = TensorD (Wih0)
        
        val Whh0 = new MatrixD (4, 4, Array (
            Array (0.1, 0.0, 0.0, 0.0),
            Array (0.0, 0.1, 0.0, 0.0),
            Array (0.0, 0.0, 0.1, 0.0),
            Array (0.0, 0.0, 0.0, 0.1)
        ))
        cell0.W_hh.data = TensorD (Whh0)
        
        // biases
        val bIh0 = new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04)))
        cell0.b_ih.data = TensorD (bIh0)
        
        val bHh0 = new MatrixD (4, 1, Array (Array (0.0), Array (0.0), Array (0.0), Array (0.0)))
        cell0.b_hh.data = TensorD (bHh0)
        
        // Layer 1 (note input size = hidden size = 4 here)
        val cell1 = rnn.layer(1).cell.asInstanceOf[RNNCell]
        val Wih1 = new MatrixD (4, 4, Array (
            Array (-0.1, -0.2, -0.3, -0.4),
            Array (-0.5, -0.6, -0.7, -0.8),
            Array (-0.9, -1.0, -1.1, -1.2),
            Array (-1.3, -1.4, -1.5, -1.6)
        ))
        cell1.W_ih.data = TensorD (Wih1)
        
        val Whh1 = new MatrixD (4, 4, Array (
            Array (0.11, 0.12, 0.13, 0.14),
            Array (0.15, 0.16, 0.17, 0.18),
            Array (0.19, 0.20, 0.21, 0.22),
            Array (0.23, 0.24, 0.25, 0.26)
        ))
        cell1.W_hh.data = TensorD (Whh1)
        
        // biases
        val bIh1 = new MatrixD (4, 1, Array (Array (0.1), Array (0.2), Array (0.3), Array (0.4)))
        cell1.b_ih.data = TensorD (bIh1)
        
        val bHh1 = new MatrixD (4, 1, Array (Array (0.5), Array (0.6), Array (0.7), Array (0.8)))
        cell1.b_hh.data = TensorD (bHh1)
        
        // ------------------------------------------------------------------
        // Inputs: sequence of length 2 (x_t0, x_t1), each [batch=2, input=3, 1]
        // ------------------------------------------------------------------
        val x0_batch0 = new MatrixD (3, 1, Array (Array (0.1), Array (0.2), Array (0.3)))
        val x0_batch1 = new MatrixD (3, 1, Array (Array (0.4), Array (0.5), Array (0.6)))
        val x_t0 = Variabl (TensorD (x0_batch0, x0_batch1), name = Some ("x_t0"))
        
        val x1_batch0 = new MatrixD (3, 1, Array (Array (0.7), Array (0.8), Array (0.9)))
        val x1_batch1 = new MatrixD (3, 1, Array (Array (1.0), Array (1.1), Array (1.2)))
        val x_t1 = Variabl (TensorD (x1_batch0, x1_batch1), name = Some ("x_t1"))
        
        val xs = IndexedSeq (x_t0, x_t1)
        
        // ------------------------------------------------------------------
        // Initial hidden states for both layers
        // ------------------------------------------------------------------
        val h0_layer0_batch0 = new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04)))
        val h0_layer0_batch1 = new MatrixD (4, 1, Array (Array (0.05), Array (0.06), Array (0.07), Array (0.08)))
        val h0_layer0 = Variabl (TensorD (h0_layer0_batch0, h0_layer0_batch1), name = Some ("h0_layer0"))
        
        val h0_layer1_batch0 = new MatrixD (4, 1, Array (Array (0.1), Array (0.2), Array (0.3), Array (0.4)))
        val h0_layer1_batch1 = new MatrixD (4, 1, Array (Array (0.5), Array (0.6), Array (0.7), Array (0.8)))
        val h0_layer1 = Variabl (TensorD (h0_layer1_batch0, h0_layer1_batch1), name = Some ("h0_layer1"))
        
        val h0s = IndexedSeq (h0_layer0, h0_layer1)
        
        // ---- Forward ----
        val (outputs, hLasts) = rnn.forward (xs, Some (h0s))
        
        println ("Outputs per timestep:")
        outputs.zipWithIndex.foreach { case (out, t) =>
            println (s"t=$t: ${out.data}")
        }
        println (s"Final hidden states: ${hLasts.map(_.data)}")
        
        // ---- Gradient checks ----
        val R = TestReport()
        // Input at time 0
        R.record ("RNN Multi Layer - x_t0 GradCheck") {
            GradCheck.gradCheck (xs(0), () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        
        // Initial hidden states
        R.record ("RNN Multi Layer - h0_layer0 GradCheck") {
            GradCheck.gradCheck (h0_layer0, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("RNN Multi Layer - h0_layer1 GradCheck") {
            GradCheck.gradCheck (h0_layer1, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        
        // Layer 0 params
        R.record ("RNN Multi Layer - W_ih0 GradCheck") {
            GradCheck.gradCheck (cell0.W_ih, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("RNN Multi Layer - W_hh0 GradCheck") {
            GradCheck.gradCheck (cell0.W_hh, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("RNN Multi Layer - b_ih0 GradCheck") {
            GradCheck.gradCheck (cell0.b_ih, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("RNN Multi Layer - b_hh0 GradCheck") {
            GradCheck.gradCheck (cell0.b_hh, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        
        // Layer 1 params
        R.record ("RNN Multi Layer - W_ih1 GradCheck") {
            GradCheck.gradCheck (cell1.W_ih, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("RNN Multi Layer - W_hh1 GradCheck") {
            GradCheck.gradCheck (cell1.W_hh, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("RNN Multi Layer - b_ih1 GradCheck") {
            GradCheck.gradCheck (cell1.b_ih, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("RNN Multi Layer - b_hh1 GradCheck") {
            GradCheck.gradCheck (cell1.b_hh, () => rnn.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        
        R.summary ("RNN Multi Layer (2 layers) Backward Test")
        
        // ---- Export graph ----
        val outPath = "target/autograd/visualization/computation_graph_rnn_multi_layer.dot"
        GraphExporter.writeDot(hLasts.last, outPath, renderSvg = true)
        println (s"Computation graph DOT written to $outPath")
    
    end rnnMultiLayerTest
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //                               GRU Tests
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `gruCellTest` main function tests the `GRUCell` class.
     * > runMain scalation.modeling.autograd.gruCellTest
     */
    @main def gruCellTest (): Unit =
        banner ("GRUCell - Forward + Backward Test")
    
        val inputSize  = 3
        val hiddenSize = 4
//      val batchSize  = 2
        
        // Create a toy GRU cell
        val cell = GRUCell(inputSize, hiddenSize)
        
        // ---- Manually set weights/biases for reproducibility ----
        // Reset gate
        val WirM = new MatrixD (4, 3, Array (
            Array (0.1, 0.2, 0.3),
            Array (0.4, 0.5, 0.6),
            Array (0.7, 0.8, 0.9),
            Array (1.0, 1.1, 1.2)
        ))
        val WhrM = new MatrixD (4, 4, Array (
            Array (0.1, 0.0, 0.0, 0.0),
            Array (0.0, 0.1, 0.0, 0.0),
            Array (0.0, 0.0, 0.1, 0.0),
            Array (0.0, 0.0, 0.0, 0.1)
        ))
        cell.W_ir.data = TensorD (WirM)
        cell.W_hr.data = TensorD (WhrM)
        cell.b_ir.data = TensorD (new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04))))
        cell.b_hr.data = TensorD (new MatrixD (4, 1, Array (Array (0.0), Array (0.0), Array (0.0), Array (0.0))))
        
        // Update gate
        val WizM = new MatrixD (4, 3, Array (
            Array (-0.1, -0.2, -0.3),
            Array (-0.4, -0.5, -0.6),
            Array (-0.7, -0.8, -0.9),
            Array (-1.0, -1.1, -1.2)
        ))
        val WhzM = new MatrixD (4, 4, Array (
            Array (0.2, 0.0, 0.0, 0.0),
            Array (0.0, 0.2, 0.0, 0.0),
            Array (0.0, 0.0, 0.2, 0.0),
            Array (0.0, 0.0, 0.0, 0.2)
        ))
        cell.W_iz.data = TensorD (WizM)
        cell.W_hz.data = TensorD (WhzM)
        cell.b_iz.data = TensorD (new MatrixD (4, 1, Array (Array (0.05), Array (0.06), Array (0.07), Array (0.08))))
        cell.b_hz.data = TensorD (new MatrixD (4, 1, Array (Array (0.0), Array (0.0), Array (0.0), Array (0.0))))
        
        // New gate
        val WinM = new MatrixD (4, 3, Array (
            Array (0.2, 0.1, 0.0),
            Array (0.0, -0.1, -0.2),
            Array (0.3, 0.2, 0.1),
            Array (-0.1, -0.2, -0.3)
        ))
        val WhnM = new MatrixD (4, 4, Array (
            Array (0.3, 0.0, 0.0, 0.0),
            Array (0.0, 0.3, 0.0, 0.0),
            Array (0.0, 0.0, 0.3, 0.0),
            Array (0.0, 0.0, 0.0, 0.3)
        ))
        cell.W_in.data = TensorD (WinM)
        cell.W_hn.data = TensorD (WhnM)
        cell.b_in.data = TensorD (new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04))))
        cell.b_hn.data = TensorD (new MatrixD (4, 1, Array (Array (0.0), Array (0.0), Array (0.0), Array (0.0))))
        
        
        // ---- Dummy batch input ----
        val x0 = new MatrixD (3, 1, Array (Array (0.1), Array (0.2), Array (0.3)))
        val x1 = new MatrixD (3, 1, Array (Array (0.4), Array (0.5), Array (0.6)))
        val x  = Variabl (TensorD (x0, x1), name = Some ("x"))
        
        // ---- Dummy previous hidden state ----
        val h0 = new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04)))
        val h1 = new MatrixD (4, 1, Array (Array (0.05), Array (0.06), Array (0.07), Array (0.08)))
        val hPrev = Variabl (TensorD (h0, h1), name = Some ("hPrev"))
        
        // Forward through GRUCell
        val hNext = cell (IndexedSeq (x, hPrev)).head
        
        println (s"Input x: ${x.data}")
        println (s"Prev hidden: ${hPrev.data}")
        println (s"Next hidden: ${hNext.data}")
        
        val R = TestReport()
        
        R.record ("GRUCell - Input GradCheck") {
            GradCheck.gradCheck (x, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
        }
        R.record ("GRUCell - Hidden GradCheck") {
            GradCheck.gradCheck (hPrev, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
        }
        
        for p <- cell.parameters do
            R.record (s"GRUCell - ${p.name.getOrElse ("param")} GradCheck") {
                GradCheck.gradCheck (p, () => cell (IndexedSeq (x, hPrev)).head.sum, quiet = true)
            }
        
        R.summary ("GRUCell Forward + Backward Test")
        
        val outPath = "target/autograd/visualization/computation_graph_gru_cell.dot"
        GraphExporter.writeDot(hNext, outPath, renderSvg = true)
        println (s"Computation graph DOT written to $outPath")
        
    end gruCellTest
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The `gruMultiLayerTest` main function tests the `GRU` class.
     * > runMain scalation.modeling.autograd.gruMultiLayerTest
     */
    @main def gruMultiLayerTest (): Unit =
        banner ("GRU Multi Layer (2 layers) - Forward + Backward Test")
        
        val inputSize = 3
        val hiddenSize = 4
//      val seqLen = 2
//      val batchSize = 2
        val numLayers = 2
        
        // ---- Construct 2-layer GRU ----
        val gru = GRU(inputSize, hiddenSize, numLayers)
        
        // Layer 0
        val cell0 = gru.layer(0).cell.asInstanceOf[GRUCell]
        // Reset gate
        cell0.W_ir.data = TensorD (new MatrixD (4, 3, Array (
            Array (0.1, 0.2, 0.3),
            Array (0.4, 0.5, 0.6),
            Array (0.7, 0.8, 0.9),
            Array (1.0, 1.1, 1.2)
        )))
        cell0.W_hr.data = TensorD (new MatrixD (4, 4, Array (
            Array (1.0, 0.0, 0.0, 0.0),
            Array (0.0, 1.0, 0.0, 0.0),
            Array (0.0, 0.0, 1.0, 0.0),
            Array (0.0, 0.0, 0.0, 1.0)
        )))
        cell0.b_ir.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.01),
            Array (0.02),
            Array (0.03),
            Array (0.04)
        )))
        cell0.b_hr.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.0),
            Array (0.0),
            Array (0.0),
            Array (0.0)
        )))
        
        // Update gate
        cell0.W_iz.data = TensorD (new MatrixD (4, 3, Array (
            Array (-0.1, -0.2, -0.3),
            Array (-0.4, -0.5, -0.6),
            Array (-0.7, -0.8, -0.9),
            Array (-1.0, -1.1, -1.2)
        )))
        cell0.W_hz.data = TensorD (new MatrixD (4, 4, Array (
            Array (0.2, 0.0, 0.0, 0.0),
            Array (0.0, 0.2, 0.0, 0.0),
            Array (0.0, 0.0, 0.2, 0.0),
            Array (0.0, 0.0, 0.0, 0.2)
        )))
        cell0.b_iz.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.05),
            Array (0.06),
            Array (0.07),
            Array (0.08)
        )))
        cell0.b_hz.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.0),
            Array (0.0),
            Array (0.0),
            Array (0.0)
        )))
        
        // New gate
        cell0.W_in.data = TensorD (new MatrixD (4, 3, Array (
            Array (0.2, 0.1, 0.0),
            Array (0.0, -0.1, -0.2),
            Array (0.3, 0.2, 0.1),
            Array (-0.1, -0.2, -0.3)
        )))
        cell0.W_hn.data = TensorD (new MatrixD (4, 4, Array (
            Array (0.3, 0.0, 0.0, 0.0),
            Array (0.0, 0.3, 0.0, 0.0),
            Array (0.0, 0.0, 0.3, 0.0),
            Array (0.0, 0.0, 0.0, 0.3)
        )))
        cell0.b_in.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.01),
            Array (0.02),
            Array (0.03),
            Array (0.04)
        )))
        cell0.b_hn.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.0),
            Array (0.0),
            Array (0.0),
            Array (0.0)
        )))
        
        // Layer 1 (input size = hidden size = 4)
        val cell1 = gru.layer(1).cell.asInstanceOf[GRUCell]
        // Reset gate
        cell1.W_ir.data = TensorD (new MatrixD (4, 4, Array (
            Array (-0.1, -0.2, -0.3, -0.4),
            Array (-0.5, -0.6, -0.7, -0.8),
            Array (-0.9, -1.0, -1.1, -1.2),
            Array (-1.3, -1.4, -1.5, -1.6)
        )))
        cell1.W_hr.data = TensorD (new MatrixD (4, 4, Array (
            Array (0.11, 0.0, 0.0, 0.0),
            Array (0.0, 0.11, 0.0, 0.0),
            Array (0.0, 0.0, 0.11, 0.0),
            Array (0.0, 0.0, 0.0, 0.11)
        )))
        cell1.b_ir.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.1),
            Array (0.2),
            Array (0.3),
            Array (0.4)
        )))
        cell1.b_hr.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.5),
            Array (0.6),
            Array (0.7),
            Array (0.8)
        )))
        
        // Update gate
        cell1.W_iz.data = TensorD (new MatrixD (4, 4, Array (
            Array (1.0, 0.0, 0.0, 0.0),
            Array (0.0, 1.0, 0.0, 0.0),
            Array (0.0, 0.0, 1.0, 0.0),
            Array (0.0, 0.0, 0.0, 1.0)
        )))
        cell1.W_hz.data = TensorD (new MatrixD (4, 4, Array (
            Array (0.12, 0.0, 0.0, 0.0),
            Array (0.0, 0.12, 0.0, 0.0),
            Array (0.0, 0.0, 0.12, 0.0),
            Array (0.0, 0.0, 0.0, 0.12)
        )))
        cell1.b_iz.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.05),
            Array (0.06),
            Array (0.07),
            Array (0.08)
        )))
        cell1.b_hz.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.0),
            Array (0.0),
            Array (0.0),
            Array (0.0)
        )))
        
        // New gate
        cell1.W_in.data = TensorD (new MatrixD (4, 4, Array (
            Array (1.0, 0.0, 0.0, 0.0),
            Array (0.0, 1.0, 0.0, 0.0),
            Array (0.0, 0.0, 1.0, 0.0),
            Array (0.0, 0.0, 0.0, 1.0)
        )))
        cell1.W_hn.data = TensorD (new MatrixD (4, 4, Array (
            Array (0.13, 0.0, 0.0, 0.0),
            Array (0.0, 0.13, 0.0, 0.0),
            Array (0.0, 0.0, 0.13, 0.0),
            Array (0.0, 0.0, 0.0, 0.13)
        )))
        cell1.b_in.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.01),
            Array (0.02),
            Array (0.03),
            Array (0.04)
        )))
        cell1.b_hn.data = TensorD (new MatrixD (4, 1, Array (
            Array (0.0),
            Array (0.0),
            Array (0.0),
            Array (0.0)
        )))
        
        // ------------------------------------------------------------------
        // Inputs: sequence of length 2, each [batch=2, input=3, 1]
        // ------------------------------------------------------------------
        val x0_batch0 = new MatrixD (3, 1, Array (Array (0.1), Array (0.2), Array (0.3)))
        val x0_batch1 = new MatrixD (3, 1, Array (Array (0.4), Array (0.5), Array (0.6)))
        val x_t0 = Variabl (TensorD (x0_batch0, x0_batch1), name = Some ("x_t0"))
        
        val x1_batch0 = new MatrixD (3, 1, Array (Array (0.7), Array (0.8), Array (0.9)))
        val x1_batch1 = new MatrixD (3, 1, Array (Array (1.0), Array (1.1), Array (1.2)))
        val x_t1 = Variabl (TensorD (x1_batch0, x1_batch1), name = Some ("x_t1"))
        
        val xs = IndexedSeq (x_t0, x_t1)
        
        // ------------------------------------------------------------------
        // Initial hidden states for both layers
        // ------------------------------------------------------------------
        val h0_layer0 = Variabl (TensorD (
            new MatrixD (4, 1, Array (Array (0.01), Array (0.02), Array (0.03), Array (0.04))),
            new MatrixD (4, 1, Array (Array (0.05), Array (0.06), Array (0.07), Array (0.08)))
        ), name = Some ("h0_layer0"))
        
        val h0_layer1 = Variabl (TensorD (
            new MatrixD (4, 1, Array (Array (0.1), Array (0.2), Array (0.3), Array (0.4))),
            new MatrixD (4, 1, Array (Array (0.5), Array (0.6), Array (0.7), Array (0.8)))
        ), name = Some ("h0_layer1"))
        
        val h0s = IndexedSeq (h0_layer0, h0_layer1)
        
        // ---- Forward ----
        val (outputs, hLasts) = gru.forward (xs, Some (h0s))
        
        println ("Outputs per timestep:")
        outputs.zipWithIndex.foreach { case (out, t) =>
            println (s"t=$t: ${out.data}")
        }
        println (s"Final hidden states: ${hLasts.map(_.data)}")
        
        // ---- Gradient checks ----
        val R = TestReport()
        
        R.record ("GRU Multi Layer - x_t0 GradCheck") {
            GradCheck.gradCheck (xs(0), () => gru.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("GRU Multi Layer - h0_layer0 GradCheck") {
            GradCheck.gradCheck (h0_layer0, () => gru.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        R.record ("GRU Multi Layer - h0_layer1 GradCheck") {
            GradCheck.gradCheck (h0_layer1, () => gru.forward (xs, Some (h0s))._1.last.sum, quiet = true)
        }
        
        for p <- cell0.parameters ++ cell1.parameters do
            R.record (s"GRU Multi Layer - ${p.name.getOrElse ("param")} GradCheck") {
                GradCheck.gradCheck (p, () => gru.forward (xs, Some (h0s))._1.last.sum, quiet = true)
            }
        
        R.summary ("GRU Multi Layer (2 layers) Backward Test")
    
    end gruMultiLayerTest
    
end RNNTestCore
