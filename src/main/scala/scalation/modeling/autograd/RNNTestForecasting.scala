//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Tue November 11 10:44:32 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: RNN Forecasting Tests & Utilities
 */

package scalation
package modeling
package autograd

import java.io.PrintWriter

import scalation.mathstat.{MatrixD, Plot, TensorD, VectorD}
import scalation.modeling.forecasting.MakeMatrix4TS.{makeMatrix4EXO, makeMatrix4L, makeMatrix4Y}
import scalation.modeling.forecasting.{Example_Covid, Example_ILI}
import scalation.modeling.neuralnet.StoppingRule

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RNNTestForecasting` object provides a suite of time–series utilities
 *  and forecasting experiments using Autograd–based recurrent neural networks.
 *  It includes:
 *   - lagged–window matrix builders (`buildMatrix4TS`, `buildMatrix4TSX`)
 *   - batch construction utilities for sequence models (`makeBatches`)
 *   - demonstration tests for RNN and GRU models on:
 *       • synthetic sequences
 *       • COVID–19 new-deaths data
 *       • ILI (Influenza-Like Illness) data
 *   - chronological train/test splits
 *   - rolling / walk–forward validation
 *  These tests verify correctness of data pipelines, shape handling,
 *  training loops, scaling transformations, and forecasting performance.
 */
object RNNTestForecasting:
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build lagged feature matrix and horizon-ahead target matrix for univariate
     *  time–series forecasting.
     *  @param y        the raw input series
     *  @param lags     number of past time steps used as model input
     *  @param hh       forecast horizon (steps ahead)
     *  @param backcast if `true`, constructs backward windows for reconstruction tests
     *  @return a tuple `(x, yy)` where:
     *          - `x`  is the lagged matrix (windows × features)
     *          - `yy` is the horizon-matrix aligned with `x`
     */
    def buildMatrix4TS (y: VectorD, lags: Int, hh: Int, backcast: Boolean = false): (MatrixD, MatrixD) =
        val x  = makeMatrix4L (y, lags, backcast)
        val yy = makeMatrix4Y (y, hh, backcast)
        
        println (s"dims of x = ${x.dims}")
        println (s"dims of yy = ${yy.dims}")
        println (s"last element in y = ${y(y.dim - 1)}")
        
        (x, yy)
    end buildMatrix4TS
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build lagged feature matrix for time-series with exogenous inputs.
     *  @param y        endogenous (target) series
     *  @param xe       exogenous variables (matrix)
     *  @param lags     number of lagged steps to include
     *  @param hh       forecast horizon
     *  @param backcast whether to generate backward windows
     *  @param isTest   if `true`, skip alignment trimming
     *  @return `(x_trim, yy_trim)` aligned lagged feature + horizon matrices
     */
    def buildMatrix4TSX (y: VectorD, xe: MatrixD, lags: Int, hh: Int, backcast: Boolean = false,
                         isTest: Boolean = false): (MatrixD, MatrixD) =
        // Lagged endogenous (target) and exogenous features
        val x_y  = makeMatrix4L (y, lags, backcast)          // (n, lags)
        val x_ex = makeMatrix4EXO (xe, 7, 1.0, backcast)     // (n, lags * n_exo)
        
        // Combine all lag features side by side
        val x = x_y ++^ x_ex                                 // (n, lags * (1 + n_exo))
        
        // Output horizons for the target
        val yy = makeMatrix4Y (y, hh, backcast)
        
        // Align dimensions (drop first lags rows, to match available horizons)
        val (x_trim, yy_trim) =
            if !isTest then
                println (isTest)
                println ("Trimming training data to align input and output matrices")
                (x(lags until x.dim), yy(lags until yy.dim))
            else
                (x, yy)
            end if
        
        println (s"dims of x = ${x_trim.dims}")
        println (s"dims of yy = ${yy_trim.dims}")
        println (s"last element in y = ${y(y.dim - 1)}")
        
        (x_trim, yy_trim)
    end buildMatrix4TSX
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert lagged matrices into mini-batches suitable for RNN/GRU models.
     *  Splits the dataset into batches, converts each window into an ordered
     *  input sequence `(X_t0, X_t1, ..., X_tn)` and attaches the corresponding
     *  target tensor for supervised sequence forecasting.
     *  @param xSeq      full lagged input matrix
     *  @param ySeq      horizon target matrix
     *  @param batchSize batch size for training
     *  @param nFeatures number of features at each time step (default = 1)
     *  @return an indexed sequence of `(inputSeq, target)` batch pairs
     */
    def makeBatches (xSeq: MatrixD, ySeq: MatrixD, batchSize: Int, nFeatures: Int = 1):
            IndexedSeq [(IndexedSeq[Variabl], Variabl)] =
        val nSeq = xSeq.dim
        val totalLag = xSeq.dim2
        val lags = totalLag / nFeatures
//      val horizon = ySeq.dim2
        val nBatches = (nSeq + batchSize - 1) / batchSize
        
        for b <- 0 until nBatches yield
            val start = b * batchSize
            val end = math.min (start + batchSize, nSeq)
            val batchWindow = start until end
            
            val xBatchMat = xSeq (batchWindow)
            val yBatchMat = ySeq (batchWindow)
            
            val inputSeq = (0 until lags).map { t =>
                val startCol = t * nFeatures
                val endCol = startCol + nFeatures
                val cols = xBatchMat (?, startCol until endCol)
                Variabl (
                    TensorD.fromMatrix (cols).permute (Seq (1, 2, 0)), // (batch, nFeatures, 1)
                    name = Some (s"x_b${b}_t${t}")
                )
            }
            // keep target as (batch, horizon, 1)
            val target = Variabl (
                TensorD.fromMatrix (yBatchMat).permute (Seq (1, 2, 0)),
                name = Some (s"y_batch$b")
            )
            (inputSeq, target)
        end for
    end makeBatches
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test utility functions used for RNN forecasting:
     *    - lag construction
     *    - horizon matrix construction
     *    - batch formatting into proper RNN tensors
     *  Demonstrates window creation for a simple synthetic series.
     *  Run using:
     *  > runMain scalation.modeling.autograd.rnnUtilityTest
     */
    @main def rnnUtilityTest (): Unit =
        banner ("RNN Utility Functions Test - Sequence Creation and Batching")
        
        val y = VectorD (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val lags = 3
        val hh = 2
        val (x, yy) = buildMatrix4TS (y, lags, hh)
        
        println (s"Step 0: Original series y = $y")
        println ("Step 1: Lagged X and Horizon Y")
        for i <- 0 until x.dim do
            println (f"t=$i%2d | x=${x(i)} => yy=${yy(i)}")
        println (s"Step 2: Creat batches of length = 4")
        val batches = makeBatches (x, yy, batchSize = 4)
        for (batch, i) <- batches.zipWithIndex do
            val (inputSeq, target) = batch
            println (s"Batch $i:")
            for t <- inputSeq.indices do
                println (s"  Time $t: ${inputSeq (t).data}")
            println (s"  Target: ${target.data}")
        end for
    end rnnUtilityTest
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast COVID-19 weekly new deaths using a single-layer RNN.
     *  Pipeline:
     *      1. load + scale the dataset
     *      2. convert into lagged windows
     *      3. batch for RNN input
     *      4. train with Adam + StepLR
     *      5. compute QoF metrics (R², SMAPE, MAE, RMSE)
     *      6. plot predictions & training loss
     *  Run:
     *  > runMain scalation.modeling.autograd.rnnCovidTest
     */
    @main def rnnCovidTest (): Unit =
        banner ("RNN Covid Test - Single Layer RNN on Covid Data")
        var y = Example_Covid.loadData_y ("new_deaths")
        y = y(0 until 116)
        val original_extremes = extreme (y)
        println ("original_extremes.type = " + original_extremes.getClass)
        
        val y_s = scaleV (original_extremes, (-2.0, 2.0))(y)
        
        val t = VectorD.range(0, y.dim)
        new Plot (t, y, null, "Covid New Deaths y(t)", lines=true)
        
        val lags = 10                         // how many past steps the RNN sees
        val hh = 1                            // predict 1 step ahead
        val (x, yy) = buildMatrix4TS (y_s, lags, hh)
        
        val batchSize = 32
        val batches = makeBatches (x, yy, batchSize)
        
        // Print shapes
        println (s"Number of batches: ${batches.length}")
        println (s"Each batch input shape: (${batches(0)._1.length}, ${batches(0)._1(0).shape})")
        println (s"Each batch target shape: ${batches(0)._2.shape}")
        
//      val inputSize = 1
//      val hiddenSize = 10
        
        case class RnnForecast (seqLen: Int, hiddenSize: Int, horizon: Int)
          extends SeqModule with Fit (seqLen, horizon):
            
            import scala.language.implicitConversions
            
            // RNN + output projection
            private val rnn = RNN (inputSize = 1, hiddenSize, numLayers = 1, activation = "tanh")
            private val outputLayer = Linear (hiddenSize, horizon)
            
            override def parameters: IndexedSeq [Variabl] =
                rnn.parameters ++ outputLayer.parameters
            
            override def forward (x: IndexedSeq [Variabl]): IndexedSeq[Variabl] =
                val (outputs, _) = rnn.forward (x)
                val lastOut = outputs.last                // (batch, hiddenSize, 1)
                val pred = lastOut ~> outputLayer         // (batch, horizon, 1)
                IndexedSeq (pred)
            end forward
        end RnnForecast
        
        // Instantiate the model and optimizer
        val net = RnnForecast(seqLen = lags, hiddenSize = 10, horizon = hh)
        val optimizer = Adam (parameters = net.parameters, lr = 0.02, beta1 = 0.9, beta2 = 0.999)
        val scheduler = StepLR (optimizer, stepSize = 50, gamma = 0.8)
        
        // Sanity: one forward pass on the first batch
        val (xs, target) = batches.head
        val pred = net.forward (xs).head
        println (s"Pred shape: ${pred.shape} vs target: ${target.shape}")
        
        // Training loop
        // 1. Results:  after 350 epochs, loss ~ 0.003, QoF: R2 ~ 0.97, SMAPE ~ 9.56
        //              patience = 40
        //              lr = 0.3, momentum = 0.90 (SGD) (no scheduler)
        //              maxNorm = 5.0 (grad clipping)
        //              hiddenSize = 10
        //              batchSize = 32
        //              lags = 10
        
        // 2. Results:  after 310 epochs, loss ~ 0.0027, QoF: R2 ~ 0.97, SMAPE ~ 9.2
        //              patience = 40
        //              lr = 0.005, beta1 = 0.9, beta2 = 0.99 (Adam) (no scheduler)
        //              maxNorm = 5.0 (grad clipping)
        //              hiddenSize = 10
        //              batchSize = 32
        //              lags = 10
        
        // 3. Results: after 900 epochs, loss ~ 0.0006, QoF: R2 ~ 0.995, SMAPE ~ 4.7
        //              patience = none
        //              lr = 0.01, beta1 = 0.9, beta2 = 0.999 (Adam)
        //              StepLR scheduler: stepSize = 50, gamma = 0.8
        //              maxNorm = 5.0 (grad clipping)
        //              hiddenSize = 10
        //              batchSize = 32
        //              lags = 10
        
        // 4. Results: after 341 epochs, loss ~ 0.0147, QoF: R2 ~ 0.987, SMAPE ~ 6.7
        //              patience = 60
        //              lr = 0.01, beta1 = 0.9, beta2 = 0.999 (Adam)
        //              StepLR scheduler: stepSize = 50, gamma = 0.8
        //              maxNorm = 5.0 (grad clipping)
        //              hiddenSize = 10
        //              batchSize = 32
        //              lags = 10
        
        object monitor extends MonitorLoss
        object EarlyStopper extends StoppingRule
        val patience = 60
        var stopTraining = false
        val nEpochs = 500
        val maxNorm = 5.0
        
        val lossWriter = new PrintWriter ("loss_covid_insample.txt")
        lossWriter.println ("epoch,loss")
        // Training loop
        for epoch <- 0 until nEpochs if ! stopTraining do
            var epochLoss = 0.0
            for (xs, target) <- batches do
                // Zero gradients
                optimizer.zeroGrad ()
                
                // Forward pass
                val pred = net.forward (xs).head
                
                // Compute loss (MSE)
                val loss = mseLoss (pred, target)
                epochLoss += loss.data(0)(0)(0)
                
                // Backward pass
                loss.backward ()
                
                // Clip gradients
                optimizer.clipGradNorm (maxNorm)
                
                // Update parameters
                optimizer.step ()
            end for
            
            // Step the scheduler (once per epoch)
            scheduler.step ()
            
            val avgLoss = epochLoss / batches.length
            lossWriter.println (s"$epoch,$avgLoss")
            monitor.collectLoss (avgLoss)
            
            if epoch % 10 == 0 then
                println (f"Epoch $epoch%3d | Loss: ${epochLoss / batches.length}%.6f")
            
            // early stopping check
            val (stopParams, bestLoss) = EarlyStopper.stopWhenPatience (net.parameters, avgLoss, patience)
            if stopParams != null then
                println (s"Early stopping at epoch $epoch with best loss $bestLoss")
                net.setParameters (stopParams)
                stopTraining = true
            end if
        end for
        lossWriter.close ()
        
        // ===================================================
        // Diagnostics: QoF & FitMap
        // ===================================================
        // Predict on whole dataset
        val yPredSeq: IndexedSeq [Double] =
            batches.flatMap { case (xs, _) =>
                val v = net.forward (xs).head.data.flattenToVector // VectorD
                (0 until v.dim).map (v(_))
            }
        // drop the first "lags" predictions to align with yy
        val yPredSeqAligned = yPredSeq.drop (lags)
        val yPredVec: VectorD = VectorD (yPredSeqAligned)
        // drop the first "lags" rows from yy to align
        val yyAligned = yy(lags until yy.dim)
        val yTrueVec = yyAligned.flatten
        
        val yPredVecRescaled = unscaleV (original_extremes, (-2.0, 2.0))(yPredVec)
        val yTrueVecRescaled = unscaleV (original_extremes, (-2.0, 2.0))(yTrueVec)
        
        val predWriter = new PrintWriter("predictions_covid_insample.csv")
        predWriter.println ("t,actual,predicted")
        for i <- 0 until yTrueVecRescaled.dim do
            predWriter.println (s"$i,${yTrueVecRescaled (i)},${yPredVecRescaled (i)}")
        predWriter.close ()
        
        println (s"shapes of yTrueVecRescaled = ${yTrueVecRescaled.dim}, yPredVecRescaled = ${yPredVecRescaled.dim}")
        println (s"first 10 yTrueVecRescaled = ${yTrueVecRescaled (0 until 10)}")
        println (s"first 10 yPredVecRescaled = ${yPredVecRescaled (0 until 10)}")
        
        monitor.plotLoss ("RNN-Covid-New-Deaths")
        new Plot (t(lags until t.dim), yTrueVecRescaled, yPredVecRescaled, "RNN New Deaths Forecast", lines=true)
        banner ("Final Train Statistics")
        val qof = net.diagnose (yTrueVecRescaled, yPredVecRescaled)
        println (FitM.fitMap (qof, qoF_names))
    end rnnCovidTest
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast ILI (Influenza-Like Illness) incidence using a single-layer RNN.
     *  Similar to the COVID test but uses a longer lag window (21 weeks).
     *  Includes scaling, batching, training with Adam + StepLR, diagnostics and plots.
     *  Run:
     *  > runMain scalation.modeling.autograd.rnnILITest
     */
    @main def rnnILITest (): Unit =
        banner ("RNN ILI Test - Single Layer RNN on ILI Data")
        
        val y = Example_ILI.loadData_y ("ILITOTAL")
        
        val original_extremes = extreme(y)
        println ("original_extremes.type = " + original_extremes.getClass)
        
        val y_s = scaleV (original_extremes, (-1.0, 1.0))(y)
        
        val t = VectorD.range(0, y.dim)
        new Plot (t, y, null, "Covid New Deaths y(t)", lines = true)
        
        val lags = 21 // how many past steps the RNN sees
        val hh = 1 // predict 1 step ahead
        val (x, yy) = buildMatrix4TS (y_s, lags, hh)
        
        val batchSize = 32
        val batches = makeBatches (x, yy, batchSize)
        
        // Print shapes
        println (s"Number of batches: ${batches.length}")
        println (s"Each batch input shape: (${batches(0)._1.length}, ${batches(0)._1(0).shape})")
        println (s"Each batch target shape: ${batches(0)._2.shape}")
        
//      val inputSize = 1
//      val hiddenSize = 10
        
        case class RnnForecast (seqLen: Int, hiddenSize: Int, horizon: Int)
          extends SeqModule with Fit (seqLen, horizon):
            
            import scala.language.implicitConversions
            
            // RNN + output projection
            private val rnn = RNN (inputSize = 1, hiddenSize, numLayers = 1, activation = "tanh")
            private val outputLayer = Linear (hiddenSize, horizon)
            
            override def parameters: IndexedSeq [Variabl] =
                rnn.parameters ++ outputLayer.parameters
            
            override def forward (x: IndexedSeq [Variabl]): IndexedSeq[Variabl] =
                val (outputs, _) = rnn.forward (x)
                val lastOut = outputs.last                 // (batch, hiddenSize, 1)
                val pred = lastOut ~> outputLayer          // (batch, horizon, 1)
                IndexedSeq (pred)
            end forward
        end RnnForecast
        
        // Instantiate the model and optimizer
        val net = RnnForecast(seqLen = lags, hiddenSize = 10, horizon = hh)
        val optimizer = Adam(parameters = net.parameters, lr = 0.005, beta1 = 0.9, beta2 = 0.999)
        val scheduler = StepLR(optimizer, stepSize = 80, gamma = 0.8)
        
        // Sanity: one forward pass on the first batch
        val (xs, target) = batches.head
        val pred = net.forward (xs).head
        println (s"Pred shape: ${pred.shape} vs target: ${target.shape}")
        
        object monitor extends MonitorLoss
        object EarlyStopper extends StoppingRule
        val patience = 140
        var stopTraining = false
        val nEpochs = 500
        val maxNorm = 5.0
        
        val lossWriter = new PrintWriter ("loss_ili_insample.csv")
        // Training loop
        for epoch <- 0 until nEpochs if ! stopTraining do
            var epochLoss = 0.0
            for (xs, target) <- batches do
                // Zero gradients
                optimizer.zeroGrad ()
                
                // Forward pass
                val pred = net.forward (xs).head
                
                // Compute loss (MSE)
                val loss = mseLoss (pred, target)
                epochLoss += loss.data(0)(0)(0)
                
                // Backward pass
                loss.backward ()
                
                // Clip gradients
                optimizer.clipGradNorm (maxNorm)
                
                // Update parameters
                optimizer.step ()
            
            end for
            
            // Step the scheduler (once per epoch)
            scheduler.step ()
            
            val avgLoss = epochLoss / batches.length
            lossWriter.println (s"$epoch,$avgLoss")
            monitor.collectLoss (avgLoss)
            
            if epoch % 10 == 0 then
                println (f"Epoch $epoch%3d | Loss: ${epochLoss / batches.length}%.6f")
            
            // early stopping check
            val (stopParams, bestLoss) = EarlyStopper.stopWhenPatience (net.parameters, avgLoss, patience)
            if stopParams != null then
                println (s"Early stopping at epoch $epoch with best loss $bestLoss")
                net.setParameters (stopParams)
                stopTraining = true
            end if
        end for
        lossWriter.close
        
        // ===================================================
        // Diagnostics: QoF & FitMap
        // ===================================================
        // Predict on whole dataset
        val yPredSeq: IndexedSeq [Double] =
            batches.flatMap { case (xs, _) =>
                val v = net.forward (xs).head.data.flattenToVector // VectorD
                (0 until v.dim).map (v(_))
            }
        // drop the first "lags" predictions to align with yy
        val yPredSeqAligned = yPredSeq.drop (lags)
        val yPredVec: VectorD = VectorD (yPredSeqAligned)
        // drop the first "lags" rows from yy to align
        val yyAligned = yy(lags until yy.dim)
        val yTrueVec = yyAligned.flatten
        
        val yPredVecRescaled = unscaleV (original_extremes, (-1.0, 1.0))(yPredVec)
        val yTrueVecRescaled = unscaleV (original_extremes, (-1.0, 1.0))(yTrueVec)
        
        val predWriter = new PrintWriter("predictions_ili_insample.csv")
        predWriter.println ("t,actual,predicted")
        for i <- 0 until yTrueVecRescaled.dim do
            predWriter.println (s"$i,${yTrueVecRescaled (i)},${yPredVecRescaled (i)}")
        predWriter.close ()
        println (s"shapes of yTrueVecRescaled = ${yTrueVecRescaled.dim}, yPredVecRescaled = ${yPredVecRescaled.dim}")
        println (s"first 10 yTrueVecRescaled = ${yTrueVecRescaled (0 until 10)}")
        println (s"first 10 yPredVecRescaled = ${yPredVecRescaled (0 until 10)}")
        
        monitor.plotLoss ("RNN-Covid-New-Deaths")
        new Plot (t(lags until t.dim), yTrueVecRescaled, yPredVecRescaled, "RNN New Deaths Forecast", lines = true)
        banner ("Final Train Statistics")
        val qof = net.diagnose (yTrueVecRescaled, yPredVecRescaled)
        println (FitM.fitMap (qof, qoF_names))
    end rnnILITest
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Chronological (or Temporal) train/test split forecasting using RNN.
     *  Features:
     *   - Temporal holdout (e.g., first 80% train, remaining test)
     *   - adaptive learning rate via `ReduceLROnPlateau`
     *   - early stopping
     *   - rescaling predictions back to original space
     *   - complete QoF metrics for train & test
     *  Run:
     *  > runMain scalation.modeling.autograd.rnnCovidTest1
     */
    @main def rnnCovidTest1 (): Unit =

        banner ("RNN Covid Test - Single Layer RNN on Covid Data with chronological split")
        var y = Example_Covid.loadData_y ("new_deaths")
        y = y(0 until 116)
        val split = 92
        val y_train = y(0 until split)
        val y_test = y(split until y.dim)
        val original_extremes = extreme (y_train)
        println ("original_extremes.type = " + original_extremes.getClass)
        
        val y_train_s = scaleV (original_extremes, (-2.0, 2.0))(y_train)
        val y_test_s  = scaleV (original_extremes, (-2.0, 2.0))(y_test)
        
        val t = VectorD.range(0, y.dim)
        new Plot (t, y, null, "Covid New Deaths y(t)", lines = true)
        
        val lags = 14                   // how many past steps the RNN sees
        val hh = 1                      // predict 1 step ahead
        val (x_train, yy_train) = buildMatrix4TS (y_train_s, lags, hh)
        val (x_test, yy_test) = buildMatrix4TS (y_test_s, lags, hh)
        
        val batchSize = 16
        val batchesTrain = makeBatches (x_train, yy_train, batchSize)
        
        // Print shapes
        println (s"Number of training batches: ${batchesTrain.length}")
        println (s"Each batch input shape: (${batchesTrain(0)._1.length}, ${batchesTrain(0)._1(0).shape})")
        println (s"Each batch target shape: ${batchesTrain(0)._2.shape}")
        
//      val inputSize = 1
//      val hiddenSize = 10
        
        case class RnnForecast (seqLen: Int, hiddenSize: Int, horizon: Int)
             extends SeqModule with Fit (seqLen, horizon):
            
            import scala.language.implicitConversions
            
            // RNN + output projection
            private val rnn = RNN (inputSize = 1, hiddenSize, numLayers = 1, activation = "tanh")
            private val outputLayer = Linear (hiddenSize, horizon)
            
            override def parameters: IndexedSeq [Variabl] =
                rnn.parameters ++ outputLayer.parameters
            
            override def forward (x: IndexedSeq [Variabl]): IndexedSeq[Variabl] =
                val (outputs, _) = rnn.forward (x)
                val lastOut = outputs.last                     // (batch, hiddenSize, 1)
                val pred = lastOut ~> outputLayer              // (batch, horizon, 1)
                IndexedSeq (pred)
            end forward
        end RnnForecast
        
        // Instantiate the model and optimizer
        val net = RnnForecast(seqLen = lags, hiddenSize = 5, horizon = hh)
        val optimizer = Adam (parameters = net.parameters, lr = 0.001, beta1 = 0.9, beta2 = 0.999)
        val scheduler = ReduceLROnPlateau (
            optim         = optimizer,
            mode          = "min",     // monitoring validation loss
            factor        = 0.6,       // decay factor
            patience      = 30,        // epochs to wait before reducing LR
            threshold     = 0.01,      // 1% relative improvement required
            thresholdMode = "rel",     // use relative thresholding (PyTorch style)
            cooldown      = 0,         // epochs to wait after LR has been reduced
            minLR         = 1e-4,      // minimum learning rate
            eps           = 1e-8,      // minimal decay applied to lr
            verbose       = true       // print message on each update
        )
        
        // Sanity: one forward pass on the first batch
        val (xs, target) = batchesTrain.head
        val pred = net.forward (xs).head
        println (s"Pred shape: ${pred.shape} vs target: ${target.shape}")
        
        // Training loop
        object monitor extends MonitorLoss
        object EarlyStopper extends StoppingRule
        val patience = 80
        var stopTraining = false
        val nEpochs = 500
        val maxNorm = 5.0
        
        // Training loop
        for epoch <- 0 until nEpochs if ! stopTraining do
            var epochLoss = 0.0
            for (xs, target) <- batchesTrain do
                // Zero gradients
                optimizer.zeroGrad ()
                
                // Forward pass
                val pred = net.forward (xs).head
                
                // Compute loss (MSE)
                val loss = mseLoss (pred, target)
                epochLoss += loss.data(0)(0)(0)
                
                // Backward pass
                loss.backward ()
                
                // Clip gradients
                optimizer.clipGradNorm (maxNorm)
                
                // Update parameters
                optimizer.step ()
            end for
            
            
            val avgLoss = epochLoss / batchesTrain.length
            monitor.collectLoss (avgLoss)
            
            scheduler.step (avgLoss)
            
            if epoch % 10 == 0 then
                println (f"Epoch $epoch%3d | Loss: ${epochLoss / batchesTrain.length}%.6f")
            
            // early stopping check
            val (stopParams, bestLoss) = EarlyStopper.stopWhenPatience (net.parameters, avgLoss, patience)
            if stopParams != null then
                println (s"Early stopping at epoch $epoch with best loss $bestLoss")
                net.setParameters (stopParams)
                stopTraining = true
            end if
        end for
        
        // ===================================================
        // Diagnostics: QoF & FitMap
        // ===================================================
        // Predict on test set
        val yPredTestSeq: IndexedSeq [Double] =
            makeBatches (x_test, yy_test, batchSize)
              .flatMap { case (xs, _) =>
                  val v = net.forward (xs).head.data.flattenToVector
                  (0 until v.dim).map (v(_))
              }

        val yPredTest: VectorD = VectorD (yPredTestSeq)
        val yTrueTest = yy_test.flatten

        // --- Train predictions ---
        val yPredTrainSeq =
            makeBatches (x_train, yy_train, batchSize)
              .flatMap { case (xs, _) =>
                  val v = net.forward (xs).head.data.flattenToVector
                  (0 until v.dim).map (v(_))
              }
            
        val yPredTrain = VectorD (yPredTrainSeq)
        val yTrueTrain = yy_train.flatten
        
        // --- Unscale both ---
        val yPredTrainRescaled = unscaleV (original_extremes, (-2.0, 2.0))(yPredTrain)
        val yTrueTrainRescaled = unscaleV (original_extremes, (-2.0, 2.0))(yTrueTrain)
        val yPredTestRescaled  = unscaleV (original_extremes, (-2.0, 2.0))(yPredTest)
        val yTrueTestRescaled  = unscaleV (original_extremes, (-2.0, 2.0))(yTrueTest)
        
        val yPredTrainAligned = yPredTrainRescaled.drop (lags)
        val yTrueTrainAligned = yTrueTrainRescaled.drop (lags)
        
        println (s"shapes of yTrueTrainAligned = ${yTrueTrainAligned.dim}, yPredTrainAligned = ${yPredTrainAligned.dim}")
        println (s"shapes of yTrueTestRescaled = ${yTrueTestRescaled.dim}, yPredTestRescaled = ${yPredTestRescaled.dim}")
        println (s"first 10 yTrueTrainAligned = ${yTrueTrainAligned (0 until 10)}")
        println (s"first 10 yPredTrainAligned = ${yPredTrainAligned (0 until 10)}")
        println (s"first 10 yTrueTestRescaled = ${yTrueTestRescaled (0 until 10)}")
        println (s"first 10 yPredTestRescaled = ${yPredTestRescaled (0 until 10)}")
        
        // --- QoF metrics ---
        val qofTrain = net.diagnose (yTrueTrainAligned, yPredTrainAligned)
        val qofTest  = net.diagnose (yTrueTestRescaled, yPredTestRescaled)
        
        banner ("QoF: Train (in-sample)")
        println (FitM.fitMap (qofTrain, qoF_names))
        banner ("QoF: Test (out-of-sample)")
        println (FitM.fitMap (qofTest, qoF_names))
        
        // --- Plots ---
        val t_train = VectorD.range(0, yPredTrainAligned.dim)
        val t_test = VectorD.range(0, yPredTestRescaled.dim)
        new Plot (t_train, yTrueTrainAligned, yPredTrainAligned,
            "RNN New Deaths Forecast (Train)", lines = true)
        new Plot (t_test,  yTrueTestRescaled,  yPredTestRescaled,
            "RNN New Deaths Forecast (Test)", lines = true)
    end rnnCovidTest1
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Chronological split forecasting with:
     *     - log → scale → train → unscale → exp transform pipeline
     *     - validation set carved out of training windows
     *     - ReduceLROnPlateau for LR scheduling
     *     - early stopping based on validation loss
     *  Supports both COVID and ILI depending on loaded dataset.
     *  Run:
     *  > runMain scalation.modeling.autograd.rnnCovidTest2
     */
    @main def rnnCovidTest2 (): Unit =
        banner ("RNN Covid Test - Single Layer RNN on Covid Data with chronological split")
        var y = Example_Covid.loadData_y ("new_deaths")
        y = y(0 until 116)
        
//        var y = Example_ILI.loadData_y ("ILITOTAL")
        
        val split = (0.8 * y.dim).toInt
        val y_train = y(0 until split)
        val y_test = y(split until y.dim)
//        val original_extremes = extreme(y_train)
        
//        val y_train_s = scaleV (original_extremes, (-2.0, 2.0))(y_train)
//        val y_test_s = scaleV (original_extremes, (-2.0, 2.0))(y_test)
        
        val offset = 1.0
        val scale_range = (-1.0, 1.0)
        val y_train_log = logTransformV (offset)(y_train)
        val y_test_log = logTransformV (offset)(y_test)
        val original_extremes = extreme(y_train_log)
        
        val y_train_s = scaleV (original_extremes, scale_range) (y_train_log)
        val y_test_s = scaleV (original_extremes, scale_range)  (y_test_log)
        
        val t = VectorD.range (0, y.dim)
        new Plot (t, y, null, "Covid New Deaths y(t)", lines = true)
        
        val lags = 21             // how many past steps the RNN sees
        val hh = 1                // predict 1 step ahead
        val (x_train, yy_train) = buildMatrix4TS (y_train_s, lags, hh)
        
        // ----------- Validation set --------------
        // Here we create a validation set from the end of the training set
        // by taking the last 10% of the training data
        val valFraction = 0.1               // 10% for validation
        val nTotalTrain = x_train.dim
        val nVal = (nTotalTrain * valFraction).toInt max 8 // at least 8 windows
        val nTrain = nTotalTrain - nVal
        
        val x_train_final = x_train (0 until nTrain)
        val yy_train_final = yy_train (0 until nTrain)
        val x_val = x_train (nTrain until nTotalTrain)
        val yy_val = yy_train (nTrain until nTotalTrain)
        println (s"Training windows: $nTrain, Validation windows: $nVal")
        // -----------------------------------------
        val (x_prev_dummy, yy_prev_dummy) = buildMatrix4TS(y_test_s, lags, hh)
        println (s"Previous x_test is : $x_prev_dummy")
        println (s"Previous y_test is : $yy_prev_dummy")
        
        val y_full = VectorD (y_train_s.takeRight(lags) ++ y_test_s)
        val (x_test_all, yy_test_all) = buildMatrix4TS (y_full, lags, hh)
        val x_test = x_test_all.drop (lags)
        val yy_test = yy_test_all.drop (lags)
        
        println (s"Current x_test is : $x_test")
        println (s"Current y_test is : $yy_test")
        
        val batchSize = 8 // was 8 for covid, 16 for ILI
        val batchesTrain = makeBatches (x_train_final, yy_train_final, batchSize)
        val batchesVal   = makeBatches (x_val, yy_val, batchSize)
        
        // Print shapes
        println (s"Number of training batches: ${batchesTrain.length}")
        println (s"Each batch input shape: (${batchesTrain(0)._1.length}, ${batchesTrain(0)._1(0).shape})")
        println (s"Each batch target shape: ${batchesTrain(0)._2.shape}")
        
//      val inputSize = 1
        val hiddenSize = 20           // was 20 for covid, 10 for ILI
        
        case class RnnForecast (seqLen: Int, hiddenSize: Int, horizon: Int)
             extends SeqModule with Fit (seqLen, horizon):
            
            import scala.language.implicitConversions
            
            // RNN + output projection
            private val rnn = RNN (inputSize = 1, hiddenSize, numLayers = 1, activation = "tanh")
            private val outputLayer = Linear (hiddenSize, horizon)
            
            override def parameters: IndexedSeq [Variabl] =
                rnn.parameters ++ outputLayer.parameters
            
            override def forward (x: IndexedSeq [Variabl]): IndexedSeq[Variabl] =
                val (outputs, _) = rnn.forward (x)
                val lastOut = outputs.last             // (batch, hiddenSize, 1)
                val pred = lastOut ~> outputLayer      // (batch, horizon, 1)
                IndexedSeq (pred)
            end forward
        end RnnForecast
        
        // Instantiate the model and optimizer
        val net = RnnForecast(seqLen = lags, hiddenSize = hiddenSize, horizon = hh)
        val optimizer = Adam (parameters = net.parameters,
                              lr = 0.02, // was 0.02 for covid, 0.005 for ILI
                              weightDecay = 1e-4)
        val scheduler = ReduceLROnPlateau (
            optim         = optimizer,
            mode          = "min",     // monitoring validation loss
            factor        = 0.6,       // decay factor
            patience      = 30,        // epochs to wait before reducing LR (was 30 for covid, 50 for ILI)
            threshold     = 0.02,      // % relative improvement required
            thresholdMode = "rel",     // use relative thresholding (PyTorch style)
            cooldown      = 0,
            minLR         = 1e-5,
            eps           = 1e-8,
            verbose       = true
        )
        
        def validationLoss (): Double =
            var tot = 0.0
            var n = 0
            for (xs, target) <- batchesVal do
                val pred = net.forward (xs).head
                val loss = mseLoss (pred, target)
                val batchSize = target.shape.head
                tot += loss.data(0)(0)(0) * batchSize
                n += batchSize
            end for
            tot / math.max (n, 1)
        end validationLoss
        
        // Training loop
        object monitor extends MonitorLoss
//      object EarlyStopper extends StoppingRule
        val patience = 9000+80                    // effectively disabled, it's over 9000!
        var stopTraining = false
        var bestValLoss = Double.PositiveInfinity
        var bestParams: IndexedSeq [Variabl] = null
        var badEpochs = 0
        val nEpochs = 500
        val maxNorm = 4.0
        
        val startTime = System.nanoTime ()
        
//        val lossWriter = new PrintWriter ("loss_covid_chronological.csv")
        val lossWriter = new PrintWriter ("loss_ili_chronological.csv")
        // Training loop
        for epoch <- 0 until nEpochs if ! stopTraining do
            var epochLoss = 0.0
            for (xs, target) <- batchesTrain do
                // Zero gradients
                optimizer.zeroGrad ()
                
                // Forward pass
                val pred = net.forward (xs).head
                
                // Compute loss (MSE)
                val loss = mseLoss (pred, target)
                epochLoss += loss.data(0)(0)(0)
                
                // Backward pass
                loss.backward ()
                
                // Clip gradients
                optimizer.clipGradNorm (maxNorm)
                
                // Update parameters
                optimizer.step ()
            
            end for
            
            val avgTrain = epochLoss / batchesTrain.length
            val valLoss = validationLoss ()

            lossWriter.println (s"$epoch,$avgTrain,$valLoss")
            monitor.collectLoss (valLoss)
            scheduler.step (valLoss)
            
            if epoch % 10 == 0 then
                println (f"Epoch $epoch%3d | train=$avgTrain%.6f | val=$valLoss%.6f")
            end if
            
            // early stopping check
            if valLoss < bestValLoss - 1e-6 then
                bestValLoss = valLoss
                bestParams = net.parameters.map (p => p.copy()) // deep copy
                badEpochs = 0
            else
                badEpochs += 1
                if badEpochs >= patience then
                    println (s"No improvement for $patience epochs. Early stopping at epoch $epoch with best val loss $bestValLoss")
                    net.setParameters (bestParams)
                    stopTraining = true
                end if
            end if
        end for
        // At the end of training, ensure we have the best parameters
//        if (bestParams != null) then net.setParameters(bestParams)
        lossWriter.close ()
        
        val endTime = System.nanoTime ()
        val durationSeconds = (endTime - startTime) / 1e9
        println (f"Training completed in $durationSeconds%.2f seconds.")
        
        // ===================================================
        // Diagnostics: QoF & FitMap
        // ===================================================
        // Predict on test set
        val yPredTestSeq: IndexedSeq [Double] =
            makeBatches (x_test, yy_test, batchSize)
              .flatMap { case (xs, _) =>
                  val v = net.forward (xs).head.data.flattenToVector
                  (0 until v.dim).map (v(_))
              }
        
        val yPredTest: VectorD = VectorD (yPredTestSeq)
        val yTrueTest = yy_test.flatten
        
        // --- Train predictions ---
        val yPredTrainSeq =
            makeBatches (x_train_final, yy_train_final, batchSize)
              .flatMap { case (xs, _) =>
                  val v = net.forward (xs).head.data.flattenToVector
                  (0 until v.dim).map (v(_))
              }
        
        val yPredTrain = VectorD (yPredTrainSeq)
        val yTrueTrain = yy_train_final.flatten
        
        // --- Unscale both ---
        val yPredTrainRescaled =
            expTransformV (offset) (unscaleV (original_extremes, scale_range)(yPredTrain))
        val yTrueTrainRescaled =
            expTransformV (offset) (unscaleV (original_extremes, scale_range)(yTrueTrain))
        val yPredTestRescaled =
            expTransformV (offset) (unscaleV (original_extremes, scale_range)(yPredTest))
        val yTrueTestRescaled =
            expTransformV (offset) (unscaleV (original_extremes, scale_range)(yTrueTest))
        
        val yPredTrainAligned = yPredTrainRescaled.drop (lags)
        val yTrueTrainAligned = yTrueTrainRescaled.drop (lags)
        
        println (s"shapes of yTrueTrainAligned = ${yTrueTrainAligned.dim}, yPredTrainAligned = ${yPredTrainAligned.dim}")
        println (s"shapes of yTrueTestRescaled = ${yTrueTestRescaled.dim}, yPredTestRescaled = ${yPredTestRescaled.dim}")
        println (s"first 10 yTrueTrainAligned = ${yTrueTrainAligned (0 until 10)}")
        println (s"first 10 yPredTrainAligned = ${yPredTrainAligned (0 until 10)}")
        println (s"first 10 yTrueTestRescaled = ${yTrueTestRescaled (0 until 10)}")
        println (s"first 10 yPredTestRescaled = ${yPredTestRescaled (0 until 10)}")
        
//        val trainWriter = new PrintWriter ("predictions_covid_train_chronological.csv")
        val trainWriter = new PrintWriter ("predictions_ili_train_chronological.csv")
        trainWriter.println ("t,actual,predicted")
        for i <- yTrueTrainAligned.indices do
            trainWriter.println (s"$i,${yTrueTrainAligned (i)},${yPredTrainAligned (i)}")
        trainWriter.close ()
        
//        val testWriter = new PrintWriter ("predictions_covid_test_chronological.csv")
        val testWriter = new PrintWriter ("predictions_ili_test_chronological.csv")
        testWriter.println ("t,actual,predicted")
        for i <- yTrueTestRescaled.indices do
            testWriter.println (s"$i,${yTrueTestRescaled (i)},${yPredTestRescaled (i)}")
        testWriter.close ()
        
        
        // --- QoF metrics ---
        val qofTrain = net.diagnose (yTrueTrainAligned, yPredTrainAligned)
        val qofTest = net.diagnose (yTrueTestRescaled, yPredTestRescaled)
        
        banner ("QoF: Train (in-sample)")
        println (FitM.fitMap (qofTrain, qoF_names))
        banner ("QoF: Test (out-of-sample)")
        println (FitM.fitMap (qofTest, qoF_names))
        
        // --- Plots ---
        val t_train = VectorD.range (0, yPredTrainAligned.dim)
        val t_test = VectorD.range (0, yPredTestRescaled.dim)
        new Plot (t_train, yTrueTrainAligned, yPredTrainAligned,
            "RNN New Deaths Forecast (Train)", lines = true)
        new Plot (t_test, yTrueTestRescaled, yPredTestRescaled,
            "RNN New Deaths Forecast (Test)", lines = true)
    end rnnCovidTest2
    
    // Current Best it seems to be:
    // 1. R2 (train) = 0.962, SMAPE (train) = 7.65%
    //    R2 (test)  = 0.896, SMAPE (test)  = 13.89%
    // ------------------ Hyperparameters (adjustable) ------------------
    // lags          = 21          // number of past steps (window size)
    // hiddenSize    = 20          // RNN hidden units
    // scale_range   = (-1.0, 1.0) // scaling range for normalized inputs
    // lr            = 0.02        // initial learning rate
    // weightDecay   = 1e-4        // L2 regularization strength
    // batchSize     = 8           // mini-batch size
    // nEpochs       = 500         // max training epochs
    // valFraction   = 0.1         // fraction of training data for validation
    // patienceLR    = 30          // epochs before LR reduction (ReduceLROnPlateau)
    // thresholdLR   = 0.02        // % relative improvement required for LR scheduler
    // maxNorm       = 4.0         // gradient clipping norm
    // offset        = 1.0         // log transform offset
    //-------------------------------------------------------------------
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // RNN Covid Test - Single Layer RNN on Covid Data with Rolling Validation
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Walk-forward (rolling) validation: 1-step-ahead forecasting.
     *  Each fold:
     *   - trains (or reuses) the RNN on all available data up to time t
     *   - predicts the next point t+1
     *   - logs QoF metrics
     *  Supports growing or rolling windows and periodic retraining.
     *  Run:
     *  > runMain scalation.modeling.autograd.rnnCovidTestRollVal
     */
    @main def rnnCovidTestRollVal (): Unit =
        banner ("RNN Covid Test — Walk-Forward (1-step ahead)")
        
        // ----------------------- Data -----------------------
        var y = Example_Covid.loadData_y ("new_deaths")
        y = y(0 until 116)
        
        val tAll = VectorD.range (0, y.dim)
        new Plot (tAll, y, null, "Covid New Deaths y(t)", lines = true)
        
        // -------------------- Hyperparams -------------------
        val lags         = 21               // past context
        val hh           = 1
        val batchSize    = 8
        val hiddenSize   = 20
        val nEpochs      = 500
        val maxNorm      = 4.0
        val offset       = 1.0              // log transform offset
        val scaleRange   = (-1.0, 1.0)
        val lr           = 0.02
        val weightDecay  = 1e-4
        val trainFrac    = 0.80             // initial train fraction
        val growingTrain = false
        val retrainEvery = 8                // retrain every k folds
        
        // ----------------- Train window sizes ----------------
        val n        = y.dim
        val trSize0  = math.max((trainFrac * n).toInt, lags + hh)      // ensure enough windows
        val nFolds   = n - trSize0
        if nFolds <= 0 then
            println (s"Not enough data for walk-forward with lags=$lags.")
            return
        end if
        
        println (s"Data size n = $n, " +
          s"initial train size = $trSize0, " +
          s"folds = $nFolds (walk-forward 1-step)"
        )
        
        // ----------------- Model definition -----------------
        case class RnnForecast (seqLen: Int, hiddenSize: Int, horizon: Int)
          extends SeqModule with Fit (seqLen, horizon):
            import scala.language.implicitConversions
            
            private val rnn = RNN (inputSize = 1, hiddenSize, numLayers = 1, activation = "tanh")
            private val out = Linear (hiddenSize, horizon)
            
            override def parameters: IndexedSeq [Variabl] = rnn.parameters ++ out.parameters
            
            override def forward (x: IndexedSeq [Variabl]): IndexedSeq[Variabl] =
                val (outputs, _) = rnn.forward (x)
                val last = outputs.last                 // (batch, hidden, 1)
                val pred = last ~> out                  // (batch, horizon, 1)
                IndexedSeq (pred)
        end RnnForecast
        
        // ------------------ Helpers -------------------------
        def makeB(x: MatrixD, yy: MatrixD): IndexedSeq [(IndexedSeq[Variabl], Variabl)] =
            makeBatches (x, yy, batchSize)
        
        inline def smape(yTrue: Double, yPred: Double): Double =
            val a = math.abs (yTrue); val b = math.abs (yPred); val d = math.abs(yPred - yTrue)
            d / ( (a + b) / 2.0 + 1e-9 )
        
        // Build ONE prediction batch from the last `lags` values of a (scaled) train vector.
        // Returns an IndexedSeq [Variabl] of length `lags`, each of shape (1, 1, 1).
        
        def lastLagBatch (yTrainScaled: VectorD, lags: Int): IndexedSeq [Variabl] =
            val w: VectorD = yTrainScaled (yTrainScaled.dim - lags until yTrainScaled.dim)
            (0 until lags).map { t =>
                val m = new MatrixD (1, 1)
                m(0, 0) = w(t)
                Variabl (TensorD.fromMatrix (m).permute (Seq (1, 2, 0)))   // (1,1,1)
            }
        end lastLagBatch
    
        // ----------------- Aggregation buffers ---------------
        val yTrueAll = collection.mutable.ArrayBuffer [Double]()
        val yPredAll = collection.mutable.ArrayBuffer [Double]()
        
        // ----------------- Walk-forward loop -----------------
        
        
        // ------------- Walk-forward (1-step) loop ------------
        var lastExtremes: (Double, Double) = (0.0, 0.0)
        var net:   RnnForecast  = null
        var optim: Optimizer    = null
        var sched: LRScheduler  = null
        
        for fold <- 0 until nFolds do
            // Train range
            val (trStart, trEnd) =
                if growingTrain then (0, trSize0 + fold)
                else (fold, trSize0 + fold)                   // rolling/fixed-width
            val y_tr = y(trStart until trEnd)
            
            // True next value (one step ahead)
            val y_true_next = y(trEnd)
            
            // --- transform & scale fit on TRAIN ONLY ---
            val y_tr_log = logTransformV (offset)(y_tr)
            val (extremes, y_tr_s) = {
                if fold % retrainEvery == 0 then
                    net = RnnForecast(seqLen = lags, hiddenSize = hiddenSize, horizon = hh)
                    optim = Adam(parameters = net.parameters, lr = lr, weightDecay = weightDecay)
                    sched = ReduceLROnPlateau(
                        optim = optim,
                        mode = "min",
                        factor = 0.6,
                        patience = 30,
                        threshold = 0.02,
                        thresholdMode = "rel",
                        cooldown = 0,
                        minLR = 1e-5,
                        eps = 1e-8,
                        verbose = false
                    )
                    
                    val ex = extreme(y_tr_log)
                    lastExtremes = ex
                    val y_tr_s = scaleV (ex, scaleRange)(y_tr_log)
                    (ex, y_tr_s)
                else
                    println (s"[Fold ${fold + 1}/$nFolds] Skipping retrain; reusing previous net/optim/sched")
                    
                    val ex = lastExtremes
                    val y_tr_s = scaleV (ex, scaleRange)(y_tr_log)
                    (ex, y_tr_s)
                end if
            }
            
            // --- build train matrices for many windows ---
            val (x_tr, yy_tr) = buildMatrix4TS(y_tr_s, lags, hh)
            if x_tr.dim < 2 || yy_tr.dim < 2 then
                val naivePred = y_tr.last
                yTrueAll += y_true_next
                yPredAll += naivePred
                println (f"[Fold ${fold+1}/$nFolds] skipped training (few windows). naive=${naivePred}%.4f true=${y_true_next}%.4f")
            else
                // batches + tiny val split from end of train windows
                val nTot  = x_tr.dim
                val nVal  = (nTot * 0.1).toInt max 8   // keep >=2 for train core
                val nKeep = nTot - nVal
                val bTrF  = makeB (x_tr(0 until nKeep), yy_tr(0 until nKeep))
                val bVal  = if nVal > 0 then makeB (x_tr(nKeep until nTot), yy_tr(nKeep until nTot))
                else IndexedSeq.empty
                
                def valLoss (): Double =
                    if bVal.isEmpty then Double.NaN
                    else
                        var tot = 0.0; var n = 0
                        for (xs, tgt) <- bVal do
                            val p = net.forward (xs).head
                            val L = mseLoss (p, tgt)
                            val bs = tgt.shape.head
                            tot += L.data(0)(0)(0) * bs
                            n   += bs
                        end for
                        tot / math.max (n, 1)
                    end if
                end valLoss
                
                var bestValLoss = Double.PositiveInfinity
                var bestParams = net.parameters.map (p => p.copy())
                var badEpochs = 0
                val patience = 9000+80 // effectively disabled, it's over 9000!
                var stopTraining = false
                
                if fold % retrainEvery == 0 then
                    // --- Training phase ---
                    for epoch <- 0 until nEpochs if ! stopTraining do
                        var epochLoss = 0.0
                        for (xs, tgt) <- bTrF do
                            optim.zeroGrad ()
                            val p = net.forward (xs).head
                            val L = mseLoss (p, tgt)
                            epochLoss += L.data(0)(0)(0)
                            L.backward ()
                            optim.clipGradNorm (maxNorm)
                            optim.step ()
                        end for
                        
                        val v = valLoss ()
                        if !java.lang.Double.isNaN (v) then
                            sched.step (v)
                            if v < bestValLoss - 1e-6 then
                                bestValLoss = v
                                bestParams = net.parameters.map (p => p.copy ())
                                badEpochs = 0
                            else
                                badEpochs += 1
                                if badEpochs >= patience then
                                    println (s"No improvement for $patience epochs. Early stopping at epoch $epoch : best loss $bestValLoss")
                                    net.setParameters (bestParams)
                                    stopTraining = true
                                end if
                            end if
                        end if
                        
                        if epoch % 50 == 0 || stopTraining then
                            val trainAvg = epochLoss / math.max (bTrF.length, 1)
                            if java.lang.Double.isNaN(v) then
                                println (f"[Fold ${fold + 1}/$nFolds] epoch=$epoch%3d train=$trainAvg%.6f")
                            else
                                println (f"[Fold ${fold + 1}/$nFolds] epoch=$epoch%3d train=$trainAvg%.6f val=$v%.6f (best=$bestValLoss%.6f)")
                        end if
                    end for
                else
                    println (s"[Fold ${fold + 1}/$nFolds] Skipping training (retrainEvery = $retrainEvery)")
                end if
                
                // ===================================================
                // Diagnostics: QoF & FitMap (Train within this fold)
                // ===================================================
                val yPredTrainSeq =
                    makeB (x_tr(0 until nKeep), yy_tr(0 until nKeep))
                      .flatMap { case (xs, _) =>
                          val v = net.forward (xs).head.data.flattenToVector
                          (0 until v.dim).map (v(_))
                      }
                end yPredTrainSeq
                
                val yPredTrain = VectorD (yPredTrainSeq)
                val yTrueTrain = yy_tr(0 until nKeep).flatten
                
                // --- Inverse scale + log back to original space ---
                val yPredTrainRescaled = expTransformV (offset)(unscaleV (extremes, scaleRange)(yPredTrain))
                val yTrueTrainRescaled = expTransformV (offset)(unscaleV (extremes, scaleRange)(yTrueTrain))
                
                val yPredTrainAligned = yPredTrainRescaled.drop (lags)
                val yTrueTrainAligned = yTrueTrainRescaled.drop (lags)
                
                println ( s"[Fold ${fold + 1}/$nFolds] shapes: yTrueTrainAligned=${yTrueTrainAligned.dim}, yPredTrainAligned=${yPredTrainAligned.dim}")
                println (s"[Fold ${fold + 1}/$nFolds] first few yTrueTrainAligned = ${yTrueTrainAligned (0 until math.min(10, yTrueTrainAligned.dim))}")
                println (s"[Fold ${fold + 1}/$nFolds] first few yPredTrainAligned = ${yPredTrainAligned (0 until math.min(10, yPredTrainAligned.dim))}")
                
                // --- Compute QoF (diagnostics / metrics) ---
                val qofTrain = net.diagnose (yTrueTrainAligned, yPredTrainAligned)
                println (banner (s"[Fold ${fold + 1}/$nFolds] TRAIN Diagnostics"))
                println (FitM.fitMap (qofTrain, qoF_names))
                
                
                // --- 1-step prediction using last `lags` from TRAIN (teacher-forced) ---
                val x_one = lastLagBatch (y_tr_s, lags)               // IndexedSeq [Variabl] length = lags
                val predNextScaled = net.forward (x_one).head.data.flattenToVector(0)
                // invert scale+log to original space
                val y_pred_next =
                    expTransformV (offset)( unscaleV (extremes, scaleRange)(VectorD (predNextScaled)))(0)
                
                // collect
                yTrueAll += y_true_next
                yPredAll += y_pred_next
                
                val sm = smape(y_true_next, y_pred_next) * 100.0
                println (f"[Fold ${fold+1}/$nFolds] y_true=${y_true_next}%.4f  y_pred=${y_pred_next}%.4f  SMAPE=${sm}%.2f%%")
                
//              val t_train = VectorD.range (0, yPredTrainAligned.dim)
//              new Plot (t_train, yTrueTrainAligned, yPredTrainAligned,
//                        "RNN New Deaths Forecast (Train)", lines = true)
            end if
        end for
        
        // ---------------- Overall diagnostics ----------------
        
        val yTrueAllV = VectorD (yTrueAll.toArray)
        val yPredAllV = VectorD (yPredAll.toArray)
        
        banner ("Walk-Forward (1-step): Overall QoF on concatenated test points")
        // Use a dummy net instance just to call `diagnose` (vector-only)
        val dummy = RnnForecast (seqLen = lags, hiddenSize = hiddenSize, horizon = hh)
        val qofAll = dummy.diagnose (yTrueAllV, yPredAllV)
        println (FitM.fitMap (qofAll, qoF_names))
        
        val t_oos = VectorD.range (0, yTrueAllV.dim)
        new Plot (t_oos, yTrueAllV, yPredAllV, "RNN Walk-Forward 1-step (All Test Points)", lines = true)
    end rnnCovidTestRollVal
    
    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    // Previous Run Results Summary:
    // ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    // Run #1 — RNN Forecast (lags=21, retrainEvery=inf, growingTrain=false)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // ------------------------------------------------------
    // R²: 0.896 | SMAPE: 13.89% | MAE: 1251.84 | RMSE: 1742.18
    
    // Run #1.1 — RNN Forecast (lags=21, retrainEvery=inf, growingTrain=true)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // -----------------------------------------------------
    // R²: 0.896 | SMAPE: 13.89% | MAE: 1251.84 | RMSE: 1742.18
    
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    // Run #2 — RNN Forecast (lags=21, retrainEvery=1, growingTrain=false)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // ------------------------------------------------------
    // R²: 0.723 | SMAPE: 20.72% | MAE: 2065.32 | RMSE: 2846.25
    
    // Run #2.1 — RNN Forecast (lags=21, retrainEvery=1, growingTrain=true)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // -----------------------------------------------------
    // R²: 0.861 | SMAPE: 15.83% | MAE: 1439.73 | RMSE: 2016.49
    
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    // Run #3 — RNN Forecast (lags=21, retrainEvery=4, growingTrain=false)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // -----------------------------------------------------
    // R²: 0.861 | SMAPE: 18.88% | MAE: 1511 | RMSE: 2019
    
    // Run #3.1 — RNN Forecast (lags=21, retrainEvery=4, growingTrain=true)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward
    // --------------------------------------------
    // R²: 0.889 | SMAPE: 15.55 % | MAE: 1335 | RMSE: 1803
    
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    // Run #4 — RNN Forecast (lags=21, retrainEvery=7, growingTrain=false)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // -----------------------------------------------------
    // R²: 0.829 | SMAPE: 17.38% | MAE: 1563.76 | RMSE: 2236.67
    
    // Run #4.1 — RNN Forecast (lags=21, retrainEvery=7, growingTrain=true)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // ------------------------------------------------------
    // R²: 0.898 | SMAPE: 14.30% | MAE: 1247.7 | RMSE: 1724.8
    
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    // Run #5 — RNN Forecast (lags=21, retrainEvery=8, growingTrain=false)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // -----------------------------------------------------
    // R²: 0.905 | SMAPE: 14.05 % | MAE: 1176 | RMSE: 1665
    
    // Run #5.1 — RNN Forecast (lags=21, retrainEvery=8, growingTrain=true)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // -----------------------------------------------------
    // R²: 0.899 | SMAPE: 13.91 % | MAE: 1214 | RMSE: 1719
    
    // :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    // Run #6 — RNN Forecast (lags=21, retrainEvery=12, growingTrain=false)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // -----------------------------------------------------
    // R²: 0.874 | SMAPE: 18.52 % | MAE: 1509 | RMSE: 1917
    
    // Run #6.1 — RNN Forecast (lags=21, retrainEvery=12, growingTrain=true)
    // Weekly COVID-19 Deaths — 1-Step Walk-Forward Forecast
    // -----------------------------------------------------
    // R²: 0.891 | SMAPE: 16.20% | MAE: 1357.69 | RMSE: 1788.01
    
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** GRU toy example trained on a noisy sine wave.
     *  Demonstrates:
     *   - GRUCell correctness in forecasting a smooth periodic signal
     *   - batching, training, clipping, plotting
     *  Run:
     *  > runMain scalation.modeling.autograd.gruSineTest
     */
    @main def gruSineTest (): Unit =
        banner ("GRU Toy Test - Single Layer RNN on Sine Data")
        val nPoints = 200
        val step = 0.1
        val noise = 0.0
        val y = VectorD.range(0, nPoints).map (t =>
              math.sin(t * step) + noise * (scala.util.Random.nextGaussian()))
        
        val t = VectorD.range (0, y.dim)
        new Plot (t, y, null, "Sine series y(t)", lines = true)
        
        val lags = 20                            // how many past steps the RNN sees
        val hh = 1                               // predict 1 step ahead
        val (x, yy) = buildMatrix4TS(y, lags, hh)
        
        val batchSize = 16
        val batches = makeBatches (x, yy, batchSize)
        
        // Print shapes
        println (s"Number of batches: ${batches.length}")
        println (s"Each batch input shape: (${batches(0)._1.length}, ${batches(0)._1(0).shape})")
        println (s"Each batch target shape: ${batches(0)._2.shape}")
        
//      val inputSize = 1
//      val hiddenSize = 10
//      val numLayers = 1
        
        case class GRUForecast(seqLen: Int, hiddenSize: Int, horizon: Int)
          extends SeqModule with Fit(seqLen, horizon):
            
            import scala.language.implicitConversions
            
            // RNN + output projection
            private val gru = GRU(inputSize = 1, hiddenSize, numLayers = 1)
            private val outputLayer = Linear(hiddenSize, horizon)
            
            override def parameters: IndexedSeq [Variabl] =
                gru.parameters ++ outputLayer.parameters
            
            override def forward (x: IndexedSeq [Variabl]): IndexedSeq[Variabl] =
                val (outputs, _) = gru.forward (x)
                val lastOut = outputs.last // (batch, hiddenSize, 1)
                val pred = lastOut ~> outputLayer // (batch, horizon, 1)
                IndexedSeq (pred)
            end forward
        end GRUForecast
        
        // Instantiate the model and optimizer
        val net = GRUForecast(seqLen = lags, hiddenSize = 10, horizon = hh)
        val optimizer = SGD(parameters = net.parameters, lr = 0.1, momentum = 0.90)
        
        // Sanity: one forward pass on the first batch
        val (xs, target) = batches.head
        val pred = net.forward (xs).head
        println (s"Pred shape: ${pred.shape} vs target: ${target.shape}")
        
        // Training loop
        object monitor extends MonitorLoss
        object EarlyStopper extends StoppingRule
        val patience = 20
        var stopTraining = false
        val nEpochs = 200
        val maxNorm = 5.0
        
        for epoch <- 0 until nEpochs if ! stopTraining do
            var epochLoss = 0.0
            for (xs, target) <- batches do
                // Zero gradients
                optimizer.zeroGrad()
                
                // Forward pass
                val pred = net.forward (xs).head
                
                // Compute loss (MSE)
                val loss = mseLoss (pred, target)
                epochLoss += loss.data(0)(0)(0)
                
                // Backward pass
                loss.backward ()
                
                // Clip gradients
                optimizer.clipGradNorm(maxNorm)
                
                // Update parameters
                optimizer.step ()
            end for
            
            val avgLoss = epochLoss / batches.length
            monitor.collectLoss (avgLoss)
            
            if epoch % 10 == 0 then
                println (f"Epoch $epoch%3d | Loss: ${epochLoss / batches.length}%.6f")
            end if
            
            // early stopping check
            val (stopParams, bestLoss) = EarlyStopper.stopWhenPatience(net.parameters, avgLoss, patience)
            if stopParams != null then
                println (s"Early stopping at epoch $epoch with best loss $bestLoss")
                net.setParameters(stopParams)
                stopTraining = true
            end if
        end for
        
        // ===================================================
        // Diagnostics: QoF & FitMap
        // ===================================================
        // Predict on whole dataset
        val yPredSeq: IndexedSeq [Double] =
            batches.flatMap { case (xs, _) =>
                val v = net.forward (xs).head.data.flattenToVector // VectorD
                (0 until v.dim).map (v(_))
            }
        // drop the first "lags" predictions to align with yy
        val yPredSeqAligned = yPredSeq.drop (lags)
        val yPredVec: VectorD = VectorD (yPredSeqAligned)
        // drop the first "lags" rows from yy to align
        val yyAligned = yy(lags until yy.dim)
        val yTrueVec = yyAligned.flatten
        
        println (s"shapes of yTrueVec = ${yTrueVec.dim}, yPredVec = ${yPredVec.dim}")
        println (s"first 10 yTrueVec = ${yTrueVec(0 until 10)}")
        println (s"first 10 yPredVec = ${yPredVec(0 until 10)}")
        
        monitor.plotLoss ("GRU-Sine")
        new Plot (t(lags until t.dim), yTrueVec, yPredVec, "RNN Sine Forecast", lines = true)
        banner ("Final Train Statistics")
        val qof = net.diagnose(yTrueVec, yPredVec)
        println (FitM.fitMap (qof, qoF_names))
    
    end gruSineTest
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forecast COVID-19 new deaths using a single-layer GRU.
     * 
     * Run:
     * > runMain scalation.modeling.autograd.gruCovidTest
     */
    @main def gruCovidTest(): Unit =
        banner ("GRU Covid Test - Single Layer GRU on Covid Data")
        var y = Example_Covid.loadData_y("new_deaths")
        y = y(0 until 116)
        val original_extremes = extreme(y)
        println ("original_extremes.type = " + original_extremes.getClass)
        
        val y_s = scaleV (original_extremes, (-1.0, 1.0))(y)
        
        val t = VectorD.range(0, y.dim)
        new Plot (t, y, null, "Covid New Deaths y(t)", lines = true)
        
        val lags = 10 // how many past steps the GRU sees
        val hh = 1 // predict 1 step ahead
        val (x, yy) = buildMatrix4TS(y_s, lags, hh)
        
        val batchSize = 16
        val batches = makeBatches (x, yy, batchSize)
        
        // Print shapes
        println (s"Number of batches: ${batches.length}")
        println (s"Each batch input shape: (${batches(0)._1.length}, ${batches(0)._1(0).shape})")
        println (s"Each batch target shape: ${batches(0)._2.shape}")
        
//      val inputSize = 1
        val hiddenSize = 8
        
        case class GRUForecast (seqLen: Int, hiddenSize: Int, horizon: Int)
             extends SeqModule with Fit (seqLen, horizon):
            
            import scala.language.implicitConversions
            
            // RNN + output projection
            private val gru = GRU (inputSize = 1, hiddenSize, numLayers = 1)
            private val outputLayer = Linear (hiddenSize, horizon)
            
            override def parameters: IndexedSeq [Variabl] =
                gru.parameters ++ outputLayer.parameters
            
            override def forward (x: IndexedSeq [Variabl]): IndexedSeq[Variabl] =
                val (outputs, _) = gru.forward (x)
                val lastOut = outputs.last                 // (batch, hiddenSize, 1)
                val pred = lastOut ~> outputLayer          // (batch, horizon, 1)
                IndexedSeq (pred)
            end forward
        end GRUForecast
        
        // Instantiate the model and optimizer
        val net = GRUForecast (seqLen = lags, hiddenSize = hiddenSize, horizon = hh)
        val optimizer = Adam (parameters = net.parameters, lr = 0.002, beta1 = 0.9, beta2 = 0.999)
        val scheduler = StepLR (optimizer, stepSize = 100, gamma = 0.8)
        
        // Sanity: one forward pass on the first batch
        val (xs, target) = batches.head
        val pred = net.forward (xs).head
        println (s"Pred shape: ${pred.shape} vs target: ${target.shape}")
        
        object monitor extends MonitorLoss
        object EarlyStopper extends StoppingRule
        val patience = 800
        var stopTraining = false
        val nEpochs = 500
        val maxNorm = 5.0
        
        // Training loop
        for epoch <- 0 until nEpochs if ! stopTraining do
            var epochLoss = 0.0
            for (xs, target) <- batches do
                // Zero gradients
                optimizer.zeroGrad()
                
                // Forward pass
                val pred = net.forward (xs).head
                
                // Compute loss (MSE)
                val loss = mseLoss (pred, target)
                epochLoss += loss.data(0)(0)(0)
                
                // Backward pass
                loss.backward ()
                
                // Clip gradients
                optimizer.clipGradNorm (maxNorm)
                
                // Update parameters
                optimizer.step ()
            
            end for
            
            val avgLoss = epochLoss / batches.length
            monitor.collectLoss (avgLoss)
            
            // Step the scheduler (once per epoch)
            scheduler.step ()
            
            if epoch % 10 == 0 then
                println (f"Epoch $epoch%3d | Loss: ${epochLoss / batches.length}%.6f")
            end if
            
            // early stopping check
            val (stopParams, bestLoss) = EarlyStopper.stopWhenPatience(net.parameters, avgLoss, patience)
            if stopParams != null then
                println (s"Early stopping at epoch $epoch with best loss $bestLoss")
                net.setParameters(stopParams)
                stopTraining = true
            end if
        end for
        
        // ===================================================
        // Diagnostics: QoF & FitMap
        // ===================================================
        // Predict on whole dataset
        val yPredSeq: IndexedSeq [Double] =
            batches.flatMap { case (xs, _) =>
                val v = net.forward (xs).head.data.flattenToVector // VectorD
                (0 until v.dim).map (v(_))
            }
        // drop the first "lags" predictions to align with yy
        val yPredSeqAligned = yPredSeq.drop (lags)
        val yPredVec: VectorD = VectorD (yPredSeqAligned)
        // drop the first "lags" rows from yy to align
        val yyAligned = yy(lags until yy.dim)
        val yTrueVec = yyAligned.flatten
        
        val yPredVecRescaled = unscaleV (original_extremes, (-2.0, 2.0))(yPredVec)
        val yTrueVecRescaled = unscaleV (original_extremes, (-2.0, 2.0))(yTrueVec)
        
        println (s"shapes of yTrueVecRescaled = ${yTrueVecRescaled.dim}, yPredVecRescaled = ${yPredVecRescaled.dim}")
        println (s"first 10 yTrueVecRescaled = ${yTrueVecRescaled (0 until 10)}")
        println (s"first 10 yPredVecRescaled = ${yPredVecRescaled (0 until 10)}")
        
        monitor.plotLoss ("GRU-Covid-New-Deaths")
        new Plot (t(lags until t.dim), yTrueVecRescaled, yPredVecRescaled, "RNN New Deaths Forecast", lines = true)
        banner ("Final Train Statistics")
        val qof = net.diagnose(yTrueVecRescaled, yPredVecRescaled)
        println (FitM.fitMap (qof, qoF_names))
    end gruCovidTest
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** GRU forecasting with chronological split on either COVID or ILI datasets.
     *  Uses:
     *    - log transform + scaling
     *    - validation-based LR scheduling (ReduceLROnPlateau)
     *    - truncated backprop (TBPTT) option in GRU.forward
     *    - full diagnostics and plots for predictions
     *  Run:
     *  > runMain scalation.modeling.autograd.gruCovidTest2
     */
    @main def gruCovidTest2 (): Unit =
        banner ("GRU Covid Test - Single Layer GRU on Covid Data with chronological split")
//      var y = Example_Covid.loadData_y ("new_deaths")
//      y = y(0 until 116)
        
        val y = Example_ILI.loadData_y ("ILITOTAL")
        
        val split = (0.8 * y.dim).toInt
        val y_train = y(0 until split)
        val y_test = y(split until y.dim)
//      val original_extremes = extreme (y_train)
        
//      val y_train_s = scaleV (original_extremes, (-2.0, 2.0))(y_train)
//      val y_test_s = scaleV (original_extremes, (-2.0, 2.0))(y_test)
        
        val offset = 1.0
        val scale_range = (-1.0, 1.0)
        val y_train_log = logTransformV (offset)(y_train)
        val y_test_log = logTransformV (offset)(y_test)
        val original_extremes = extreme (y_train_log)
        
        val y_train_s = scaleV (original_extremes, scale_range)(y_train_log)
        val y_test_s = scaleV (original_extremes, scale_range)(y_test_log)
        
        val t = VectorD.range(0, y.dim)
        new Plot (t, y, null, "Covid New Deaths y(t)", lines = true)
        
        val lags = 20                                 // how many past steps the GRU sees
        val hh = 1                                    // predict 1 step ahead
        val (x_train, yy_train) = buildMatrix4TS (y_train_s, lags, hh)
        
        // ----------- Validation set --------------
        // Here we create a validation set from the end of the training set
        // by taking the last 10% of the training data
        val valFraction = 0.1 // 10% for validation
        val nTotalTrain = x_train.dim
        val nVal = (nTotalTrain * valFraction).toInt max 8 // at least 8 windows
        val nTrain = nTotalTrain - nVal
        
        val x_train_final = x_train (0 until nTrain)
        val yy_train_final = yy_train (0 until nTrain)
        val x_val = x_train (nTrain until nTotalTrain)
        val yy_val = yy_train (nTrain until nTotalTrain)
        println (s"Training windows: $nTrain, Validation windows: $nVal")
        // -----------------------------------------
        
        val (x_test, yy_test) = buildMatrix4TS (y_test_s, lags, hh)
        
        val batchSize = 16
        val batchesTrain = makeBatches (x_train_final, yy_train_final, batchSize)
        val batchesVal = makeBatches (x_val, yy_val, batchSize)
        
        // Print shapes
        println (s"Number of training batches: ${batchesTrain.length}")
        println (s"Each batch input shape: (${batchesTrain(0)._1.length}, ${batchesTrain(0)._1(0).shape})")
        println (s"Each batch target shape: ${batchesTrain(0)._2.shape}")
        
//      val inputSize = 1
        val hiddenSize = 20
        
        case class GRUForecast(seqLen: Int, hiddenSize: Int, horizon: Int)
          extends SeqModule with Fit(seqLen, horizon):
            
            import scala.language.implicitConversions
            
            // GRU + output projection
            private val gru = GRU (inputSize = 1, hiddenSize, numLayers = 1)
            private val outputLayer = Linear (hiddenSize, horizon)
            
            override def parameters: IndexedSeq [Variabl] =
                gru.parameters ++ outputLayer.parameters
            
            override def forward (x: IndexedSeq [Variabl]): IndexedSeq [Variabl] =
                val (outputs, _) = gru.forward (x, tbptt = 8)
                val lastOut = outputs.last                 // (batch, hiddenSize, 1)
                val pred = lastOut ~> outputLayer          // (batch, horizon, 1)
                IndexedSeq (pred)
            end forward
        end GRUForecast
        
        // Instantiate the model and optimizer
        val net = GRUForecast(seqLen = lags, hiddenSize = hiddenSize, horizon = hh)
        val optimizer = Adam (parameters = net.parameters, lr = 0.001, weightDecay = 3e-4)
        val scheduler = ReduceLROnPlateau (
            optim = optimizer,
            mode = "min",                                 // monitoring validation loss
            factor = 0.6,                                 // decay factor
            patience = 80,                                // epochs to wait before reducing LR
            threshold = 5e-4,                             // % abs improvement required
            thresholdMode = "abs",                        // use abs thresholding
            cooldown = 20,
            minLR = 1e-5,
            eps = 1e-8,
            verbose = true
        )
        
        def validationLoss (): Double =
            var tot = 0.0
            var n = 0
            for (xs, target) <- batchesVal do
                val pred = net.forward (xs).head
                val loss = mseLoss (pred, target)
                val batchSize = target.shape.head
                tot += loss.data(0)(0)(0) * batchSize
                n += batchSize
            end for
            tot / math.max(n, 1)
        end validationLoss
        
        // Training loop
        object monitor extends MonitorLoss
//      object EarlyStopper extends StoppingRule
        val patience = 9000 + 80                     // effectively disabled, it's over 9000!
        var stopTraining = false
        var bestValLoss = Double.PositiveInfinity
        var bestParams: IndexedSeq [Variabl] = null
        var badEpochs = 0
        val nEpochs = 400
        val maxNorm = 5.0
        
        // Training loop
        for epoch <- 0 until nEpochs if ! stopTraining do
            var epochLoss = 0.0
            for (xs, target) <- batchesTrain do
                // Zero gradients
                optimizer.zeroGrad ()
                
                // Forward pass
                val pred = net.forward (xs).head
                
                // Compute loss (MSE)
                val loss = mseLoss (pred, target)
                epochLoss += loss.data(0)(0)(0)
                
                // Backward pass
                loss.backward ()
                
                // Clip gradients
                optimizer.clipGradNorm (maxNorm)
                
                // Update parameters
                optimizer.step ()
            
            end for
            
            val avgTrain = epochLoss / batchesTrain.length
            val valLoss = validationLoss ()
            
            monitor.collectLoss (valLoss)
            scheduler.step (valLoss)
            
            if epoch % 10 == 0 then
                println (f"Epoch $epoch%3d | train=$avgTrain%.6f | val=$valLoss%.6f")
            end if
            
            // early stopping check
            if valLoss < bestValLoss - 1e-6 then
                bestValLoss = valLoss
                bestParams = net.parameters.map (p => p.copy()) // deep copy
                badEpochs = 0
            else
                badEpochs += 1
                if badEpochs >= patience then
                    println (s"No improvement for $patience epochs. Early stopping at epoch $epoch with best val loss $bestValLoss")
                    net.setParameters (bestParams)
                    stopTraining = true
                end if
            end if
        end for
        // At the end of training, ensure we have the best parameters
        //        if (bestParams != null) then net.setParameters(bestParams)
        
        // ===================================================
        // Diagnostics: QoF & FitMap
        // ===================================================
        // Predict on test set
        val yPredTestSeq: IndexedSeq [Double] =
            makeBatches (x_test, yy_test, batchSize)
              .flatMap { case (xs, _) =>
                  val v = net.forward (xs).head.data.flattenToVector
                  (0 until v.dim).map (v(_))
              }
        
        val yPredTest: VectorD = VectorD (yPredTestSeq)
        val yTrueTest = yy_test.flatten
        
        // --- Train predictions ---
        val yPredTrainSeq =
            makeBatches (x_train_final, yy_train_final, batchSize)
              .flatMap { case (xs, _) =>
                  val v = net.forward (xs).head.data.flattenToVector
                  (0 until v.dim).map (v(_))
              }
        
        val yPredTrain = VectorD (yPredTrainSeq)
        val yTrueTrain = yy_train_final.flatten
        
        // --- Unscale both ---
        val yPredTrainRescaled =
            expTransformV (offset)(unscaleV (original_extremes, scale_range)(yPredTrain))
        val yTrueTrainRescaled =
            expTransformV (offset)(unscaleV (original_extremes, scale_range)(yTrueTrain))
        val yPredTestRescaled =
            expTransformV (offset)(unscaleV (original_extremes, scale_range)(yPredTest))
        val yTrueTestRescaled =
            expTransformV (offset)(unscaleV (original_extremes, scale_range)(yTrueTest))
        
        val yPredTrainAligned = yPredTrainRescaled.drop (lags)
        val yTrueTrainAligned = yTrueTrainRescaled.drop (lags)
        
        println (s"shapes of yTrueTrainAligned = ${yTrueTrainAligned.dim}, yPredTrainAligned = ${yPredTrainAligned.dim}")
        println (s"shapes of yTrueTestRescaled = ${yTrueTestRescaled.dim}, yPredTestRescaled = ${yPredTestRescaled.dim}")
        println (s"first 10 yTrueTrainAligned = ${yTrueTrainAligned (0 until 10)}")
        println (s"first 10 yPredTrainAligned = ${yPredTrainAligned (0 until 10)}")
        println (s"first 10 yTrueTestRescaled = ${yTrueTestRescaled (0 until 10)}")
        println (s"first 10 yPredTestRescaled = ${yPredTestRescaled (0 until 10)}")
        
        // --- QoF metrics ---
        val qofTrain = net.diagnose (yTrueTrainAligned, yPredTrainAligned)
        val qofTest  = net.diagnose (yTrueTestRescaled, yPredTestRescaled)
        
        banner ("QoF: Train (in-sample)")
        println (FitM.fitMap (qofTrain, qoF_names))
        banner ("QoF: Test (out-of-sample)")
        println (FitM.fitMap (qofTest, qoF_names))
        
        // --- Plots ---
        val t_train = VectorD.range (0, yPredTrainAligned.dim)
        val t_test  = VectorD.range (0, yPredTestRescaled.dim)
        new Plot (t_train, yTrueTrainAligned, yPredTrainAligned,
            "GRU New Deaths Forecast (Train)", lines = true)
        new Plot (t_test, yTrueTestRescaled, yPredTestRescaled,
            "GRU New Deaths Forecast (Test)", lines = true)
    end gruCovidTest2

end RNNTestForecasting

