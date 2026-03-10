
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Mar 16 15:13:38 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Neural Network with 2 Layers (input and output layers)
 *
 *  @see     hebb.mit.edu/courses/9.641/2002/lectures/lecture03.pdf
 */

package scalation
package modeling
package neuralnet

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import ActivationFun._
import Initializer._
import Optimizer._

//import neuralnet.{Optimizer_SGD  => OPTIMIZER}
//import neuralnet.{Optimizer_SGDM => OPTIMIZER}
import neuralnet.{Optimizer_Adam => OPTIMIZER}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_2L` class supports multi-output, 2-layer (input and output)
 *  Neural-Networks.  It can be used for both classification and prediction,
 *  depending on the activation functions used.  Given several input vectors and output
 *  vectors (training data), fit the weights/parameters b connecting the layers,
 *  so that for a new input vector z, the net can predict the output value, i.e.,
 *      yp_j = f (b dot z)
 *  where f is the activation function and the parameters b gives the weights
 *  between input and output layers.
 *  NOTE, b0 is treated as the bias, so x0 must be 1.0.
 *  @param x       the m-by-n input/data matrix (training data consisting of m input vectors)
 *  @param y       the m-by-ny output/response matrix (training data consisting of m output vectors)
 *  @param fname_  the feature/variable names (if null, use x_j's)
 *  @param hparam  the hyper-parameters for the model/network
 *  @param f       the activation function family for layers 1->2 (input to output)
 *  @param itran   the inverse transformation function returns response matrix to original scale
 */
class NeuralNet_2L (x: MatrixD, y: MatrixD, fname_ : Array [String] = null,
                    hparam: HyperParameter = Optimizer.hp,
                    f: AFF = f_sigmoid, val itran: FunctionM2M = null)
      extends PredictorMV (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):

    private val eta  = hp("eta").toDouble                                 // learning rate
            val opti = new OPTIMIZER ()                                   // parameter optimizer

//  b  = new NetParam (weightMat (x.dim2, y.dim2))                        // initialize parameters b
//  bb = Array (b.asInstanceOf [NetParam])                                // inside array 
    bb = Array (new NetParam (weightMat (x.dim2, y.dim2)))                // initialize parameters bb

    _modelName = s"NeuralNet_2L_${f.name}"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x_ and y_, fit the parameters bb.
     *  Minimize the error in the prediction by adjusting the parameters bb.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output matrix
     */
    def train (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        val epochs = opti.optimize2 (x_, y_, bb, eta, Array (f))          // optimize parameters bb
        println (s"ending epoch = $epochs")
        estat.tally (epochs._2)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x_ and y_, fit the parameters bb.
     *  Minimize the error in the prediction by adjusting the parameters bb.
     *  This version preforms an interval search for the best eta value.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output matrix
     */
    override def train2 (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        val etaI = (0.25 * eta, 4.0 * eta)                                              // quarter to four times eta
        val epochs = opti.auto_optimize (x_, y_, bb, etaI, Array (f), opti.optimize2)   // optimize parameters bb
        println (s"ending epoch = $epochs")
        estat.tally (epochs._2)
    end train2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output matrix (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : MatrixD = y): (MatrixD, MatrixD) =
        val yp = predict (x_)                                            // make predictions
        val yy = if itran == null then y_ else itran (y_)                // undo scaling, if used
        e = yy - yp                                                      // RECORD the residuals/errors (@see `PredictorMV`)
//      new Plot (null, yy(?, 0), yp(?, 0), "original scale: yy vs. yp")
        val qof = MatrixD (for k <- yy.indices2 yield diagnose (yy(?, k), yp(?, k))).transpose
        (yp, qof)                                                        // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make plots for each output/response variable (column of matrix y).
     *  Overriden as the response matrix may be transformed or rescaled.
     *  @param yy_  the testing/full actual response/output matrix (defaults to full y)
     *  @param yp   the testing/full predicted response/output matrix (defaults to full y)
     */
    override def makePlots (yy_ : MatrixD, yp: MatrixD): Unit =
        val yy = if itran == null then yy_ else itran (yy_)               // undo scaling, if used
        val (ryy, ryp) = orderByYY (yy, yp)                               // order by yy
        for k <- ryy.indices2 do
            new Plot (null, ryy(?, k), ryp(?, k), s"$modelName: y$k black/actual vs. red/predicted")
        end for
    end makePlots

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input vector v, predict the output/response vector f(v).
     *  @param v  the new input vector
     */
    def predict (v: VectorD): VectorD =
        val yp = f.f_ (bb(0) dot v)                                       // scaled prediction
        if itran == null then yp else itran (MatrixD (yp))(0)             // back to original scale
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given an input matrix v, predict the output/response matrix f(v).
     *  @param v  the input matrix
     */
    override def predict (v: MatrixD = x): MatrixD = 
        val yp = f.fM (bb(0) * v)                                         // scaled predictions
        if itran == null then yp else itran (yp)                          // back to original scale
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): NeuralNet_2L =
        new NeuralNet_2L (x_cols, y, null, hparam, f, itran)
    end buildModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor x_j
     *  and the overall Quality of Fit (QoF).
     *  FIX - only known to be valid for id activation function
     *  @see https://community.wolfram.com/groups/-/m/t/1319745
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     */
    def summary2 (x_ : MatrixD = getX, fname_ : Array [String] = fname,
                  b_ : MatrixD = parameters(0).toMatrixD): String =
        summary (x_, fname_, b_(?, 0), null)                              // summary from `Fit`
    end summary2

end NeuralNet_2L


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_2L` companion object provides factory methods for creating two-layer
 *  (no hidden layer) neural networks.  Note, 'scale' is defined in `Scaling`.
 */
object NeuralNet_2L extends Scaling:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_2L` with automatic rescaling from a combined data matrix.
     *  @param xy      the combined input and output matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Optimizer.hp)
     *  @param f       the activation function family for layers 1->2 (input to output)
     *  @param col     the first designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = Optimizer.hp, f: AFF = f_sigmoid)
               (col: Int = xy.dim2 - 1): NeuralNet_2L =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale
        val (x, y) = (xy(?, 0 until col), xy(?, col until xy.dim2))          // response matrix: col .. dim2-1

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f.bounds != null then { val y_i = rescaleY (y, f); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_2L (x_s, y_s, fname, hparam, f, itran)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_2L` with automatic rescaling from a data matrix and response matrix.
     *  @param x       the input/data m-by-n matrix
     *  @param y       the output/response m-by-ny matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Optimizer.hp)
     *  @param f       the activation function family for layers 1->2 (input to output)
     */
    def rescale (x: MatrixD, y: MatrixD, fname: Array [String] = null,
                 hparam: HyperParameter = Optimizer.hp, f: AFF = f_sigmoid): NeuralNet_2L =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f.bounds != null then { val y_i = rescaleY (y, f); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_2L (x_s, y_s, fname, hparam, f, itran)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_2L` with automatic rescaling from a data matrix and response vector.
     *  As the number of output nodes is one in this case, it is effectively a perceptron.
     *  @param x       the input/data m-by-n matrix
     *  @param y_      the output/response m-vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Optimizer.hp)
     *  @param f       the activation function family for layers 1->2 (input to output)
     */
    def perceptron (x: MatrixD, y_ : VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = Optimizer.hp, f: AFF = f_sigmoid): NeuralNet_2L =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale
        val y   = MatrixD.fromVector (y_)                                    // create a m-by-1 matrix from an vector                               

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f.bounds != null then { val y_i = rescaleY (y, f); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_2L (x_s, y_s, fname, hparam, f, itran)
    end perceptron

end NeuralNet_2L


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest` main function is used to test the `NeuralNet_2L` class.
 *  Try changing the eta and bSize hyper-parameters, as well as the activation function.
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest
 */
@main def neuralNet_2LTest (): Unit =

    val x = MatrixD ((12, 3), 1.0, 0.2, 0.3,                     // training data - input matrix (m=12 vectors)
                              1.0, 0.2, 0.5,
                              1.0, 0.2, 0.7,
                              1.0, 0.3, 0.3,
                              1.0, 0.3, 0.5,
                              1.0, 0.3, 0.7,

                              1.0, 0.4, 0.3,
                              1.0, 0.4, 0.3,
                              1.0, 0.4, 0.7,
                              1.0, 0.5, 0.5,
                              1.0, 0.5, 0.3,
                              1.0, 0.5, 0.7)

    val y0 = x.map (x_i => sigmoid (VectorD (2.0, 1.0, 2.0) dot (x_i)))
    val y1 = x.map (x_i => sigmoid (VectorD (2.0, 2.0, 2.0) dot (x_i)))
    val y  = MatrixD (y0, y1).transpose

    println (s"input  matrix x = $x")
    println (s"output matrix y = $y")

    Optimizer.hp("eta")   = 3.0                                  // set the learning rate (large for small dataset)
    Optimizer.hp("bSize") = 6.0                                  // set the batch size (small for small dataset)
//  val mod = new NeuralNet_2L (x, y)                            // create NeuralNet_2L model with sigmoid (default) 
    val mod = new NeuralNet_2L (x, y, f = f_tanh)                // create NeuralNet_2L model with tanh

    banner ("Small Example - NeuralNet_2L: trainNtest")
    mod.trainNtest ()()                                          // train and test the model
    mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs

    banner ("Small Example - NeuralNet_2L: trainNtest2")
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("neuralNet_2LTest: Compare with Linear Regression - first column of y")
    val rg0 = new Regression (x, y0)                             // create a Regression model
    rg0.trainNtest ()()                                          // train and test the model
    println (rg0.summary ())                                     // parameter/coefficient statistics

    banner ("neuralNet_2LTest: Compare with Linear Regression - second column of y")
    val rg1 = new Regression (x, y1)                             // create a Regression model
    rg1.trainNtest ()()                                          // train and test the model
    println (rg1.summary ())                                     // parameter/coefficient statistics

end neuralNet_2LTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest2` main function tests the `NeuralNet_2L` class using the
 *  Concrete dataset.  It has three outputs/response variables.
 *  There are two ways to create the model:
 *      new NeuralNet_2L (ox, y, ox_fname)       - depending on act. function user must rescale 
 *      NeuralNet_2L.rescale (ox, y, ox_fname)   - automatically rescales, assumes matrix response
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest2
 */
@main def neuralNet_2LTest2 (): Unit =

    import Example_Concrete.{ox, y, ox_fname}

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

//  val mod = new NeuralNet_2L (ox, y, ox_fname)                 // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, y, ox_fname)             // create model with intercept (else pass x) - rescales

    banner ("Concrete - NeuralNet_2L: trainNtest")
    mod.trainNtest ()()                                          // train and test the model
    mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs

    banner ("Concrete - NeuralNet_2L: trainNtest2")
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("Concrete - NeuralNet_2L: validate")
    println (Fit.showFitMap (mod.validate ()()._2))

    banner ("Concrete - NeuralNet_2L: crossValidate")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end neuralNet_2LTest2

import Example_AutoMPG._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest3` main function tests the `NeuralNet_2L` class using the
 *  AutoMPG dataset.  There are three ways to create the model:
 *      new NeuralNet_2L (ox, yy, ox_fname)       - depending on act. function user must rescale 
 *      NeuralNet_2L.rescale (ox, yy, ox_fname)   - automatically rescales, assumes matrix response
 *      NeuralNet_2L.perceptron (ox, y, ox_fname) - automatically rescales, assumes vector response
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest3
 */
@main def neuralNet_2LTest3 (): Unit =
 
//  println (s"ox = $ox")
//  println (s"yy = $yy")
    println (s"ox_fname = ${stringOf (ox_fname)}")
 
    Optimizer.hp("eta") = 0.08                                   // try 0.1 for SGDM, 0.08 for Adam
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
//  val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
    val mod = NeuralNet_2L.perceptron (ox, y, ox_fname)          // create model with intercept (else pass x) - rescales

    banner ("AutoMPG - NeuralNet_2L: In-Sample trainNtest")
    mod.trainNtest ()()                                          // train and test the model - manual tuning
    mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs

    banner ("AutoMPG NeuralNet_2L: In-Sample trainNtest2 - auto-tune")
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    mod.opti.plotLoss ("NeuralNet_2L")                           // loss function vs epochs for each eta
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("AutoMPG - NeuralNet_2L: TnT validate")
    println (Fit.showFitMap (mod.validate ()()._2))

    banner ("AutoMPG - NeuralNet_2L: crossValidate")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end neuralNet_2LTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest4` main function tests the `NeuralNet_2L` class using the
 *  AutoMPG dataset.  It tests forward selection.
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest4
 */
@main def neuralNet_2LTest4 (): Unit =

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG NeuralNet_2L")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, smape, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                     // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end neuralNet_2LTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest5` main function tests the `NeuralNet_2L` class using the
 *  AutoMPG  dataset.  It tests forward, backward and stepwise selection.
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest5
 */
@main def neuralNet_2LTest5 (): Unit =

//  println (s"ox = $ox")
//  println (s"yy = $yy")

    banner ("AutoMPG NeuralNet_2L")
//  val mod = new NeuralNet_2L (ox, yy, ox_fname)                // create model with intercept (else pass x)
    val mod = NeuralNet_2L.rescale (ox, yy, ox_fname)            // create model with intercept (else pass x) - rescales
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())

    println (s"ox_fname = ${stringOf (ox_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, smape, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${ox.dim2}")
        new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end neuralNet_2LTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest6` main function tests the `NeuralNet_2L` class using the
 *  AutoMPG dataset.  It tries all activation functions.
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest6
 */
@main def neuralNet_2LTest6 (): Unit =

//  println (s"ox = $ox")
//  println (s"yy = $yy")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    for f <- f_aff do                                            // try all activation functions for first layer
        banner (s"AutoMPG NeuralNet_2L with ${f.name}")
        val mod = NeuralNet_2L.rescale (ox, yy, ox_fname, f = f)  // create model with intercept (else pass x) - rescales
        mod.trainNtest2 ()()                                      // train and test the model - with auto-tuning

        banner ("AutoMPG Validation Test")
        println (Fit.showFitMap (mod.validate ()()._2))
    end for

end neuralNet_2LTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest7` main function is used to test the `NeuralNet_2L` class.
 *  It tests a simple case that does not require a file to be read.
 *  @see translate.google.com/translate?hl=en&sl=zh-CN&u=https:
 *       //www.hrwhisper.me/machine-learning-decision-tree/&prev=search
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest7
 */
@main def neuralNet_2LTest7 (): Unit =

    val x  = MatrixD ((10, 1), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val y  = VectorD (5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05)
    val ox = VectorD.one (x.dim) +^: x

    banner (s"Regression with intercept")
    val reg = new Regression (ox, y)
    reg.trainNtest ()()                                               // train and test the model

    banner (s"Perceptron sigmoid")
    val nn = Perceptron.rescale (ox, y)                               // FIX - plots
    nn.trainNtest ()()                                                // train and test the model

    banner (s"Perceptron tanh")
    val nn2 = Perceptron.rescale (ox, y, f = ActivationFun.f_tanh)    // FIX - plots
    nn2.trainNtest ()()                                               // train and test the model

    val ym = MatrixD (y).transpose
    Optimizer.hp ("eta") = 0.85                                       // Preceptron and NeuralNet_2L use different optimizers,
                                                                      // so different learning rates (eta) are needed.
    banner (s"NeuralNet_2L sigmoid")
    val nn3 = NeuralNet_2L.rescale (ox, ym)
    nn3.trainNtest ()()                                               // train and test the model

    banner (s"NeuralNet_2L tanh")
    val nn4 = NeuralNet_2L.rescale (ox, ym, f = ActivationFun.f_tanh)
    nn4.trainNtest ()()                                               // train and test the model

end neuralNet_2LTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest8` main function is used to test the `NeuralNet_2L` class.
 *  It test the basic vector equations for gradient descent. 
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest8
 */
@main def neuralNet_2LTest8 (): Unit =

    // 9 data points:    Constant    x1    x2    y0    y1
    val xy = MatrixD ((9, 5), 1.0,  0.0,  0.0,  0.5,  0.4,            // dataset
                              1.0,  0.0,  0.5,  0.3,  0.3,
                              1.0,  0.0,  1.0,  0.2,  0.2,

                              1.0,  0.5,  0.0,  0.8,  0.7,
                              1.0,  0.5,  0.5,  0.5,  0.5,
                              1.0,  0.5,  1.0,  0.3,  0.4,

                              1.0,  1.0,  0.0,  1.0,  0.9,
                              1.0,  1.0,  0.5,  0.8,  0.7,
                              1.0,  1.0,  1.0,  0.5,  0.5)
    val x    = xy(?, 0 until 3)                                       // matrix for predictor variables
    val y    = xy(?, 3 until 5)                                       // matrix for response variables
    val sst0 = (y(?, 0) - y(?, 0).mean).normSq                        // sum of squares total for y_:0
    val sst1 = (y(?, 1) - y(?, 1).mean).normSq                        // sum of squares total for y_:1

    val η = 1.0                                                       // learning rate
    val b = MatrixD ((3, 2), 0.1, 0.1,                                // weights/parameters
                             0.2, 0.1,
                             0.1, 0.1)

    val f = f_sigmoid

    for epoch <- 1 to 10 do
        banner (s"improvement step $epoch")
        val u  = x * b                                                // pre-activation vector
        val yp = f.fM (u)                                             // predicted response from calculation for sigmoid
        val ε  = y - yp                                               // error matrix
        val δ  = f.dM(yp) ⊙ ε                                         // delta matrix for y
        b     += x.ᵀ * δ * η                                          // parameter update (transpose (ᵀ))

        val sse0 = ε(?, 0).normSq                                     // sum of squared errors for column 0
        val sse1 = ε(?, 1).normSq                                     // sum of squared errors for column 1

        println (s"u     = $u")
        println (s"y     = $y")
        println (s"yp    = $yp")
        println (s"e     = $ε")
        println (s"δ     = $δ")
        println (s"b     = $b")
        println (s"sse0  = $sse0")
        println (s"sse1  = $sse1")
        println (s"R^2_0 = ${1 - sse0/sst0}")
        println (s"R^2_1 = ${1 - sse1/sst1}")
    end for

end neuralNet_2LTest8


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_2LTest9` main function is used to test the `NeuralNet_2L` class.
 *  It compares `NeuralNet_2L.perceptron` using sigmoid with `TransRegression` using logit.
 *  > runMain scalation.modeling.neuralnet.neuralNet_2LTest9
 *
@main def neuralNet_2LTest9 (): Unit =

//  val x  = VectorD (1, 2, 3, 4, 5, 6)
    val x  = MatrixD ((6, 1), 1, 2, 3, 4, 5, 6)
    val y  = VectorD (1, 3, 3, 5, 4, 4) / 6.0
    val yt = logit_ (y)
    val ox = MatrixD.one (x.dim) ++^ x

    println (s"ox = $ox")
    println (s"y  = $y")
    println (s"yt = $yt")

    val nn = NeuralNet_2L.perceptron (ox, y)                          // create a perceptron
//  nn.trainNtest ()()                                                // train and test the model
    nn.trainNtest2 ()()                                               // train and test the model - tunes eta

    val tr = new TranRegression (ox, y, tran = logit, itran = sigmoid) 
    tr.trainNtest ()()                                                // train and test the model

end neuralNet_2LTest9
 */

