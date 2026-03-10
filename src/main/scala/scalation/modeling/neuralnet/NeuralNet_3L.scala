
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Mar 16 15:13:38 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Neural Network with 3 Layers (input, hidden and output layers)
 *
 *  @see     hebb.mit.edu/courses/9.641/2002/lectures/lecture03.pdf
 */

package scalation
package modeling
package neuralnet

import scala.runtime.ScalaRunTime.stringOf

import scala.math.exp
import scalation.mathstat._

import ActivationFun._
import Initializer._
import Optimizer._

//import neuralnet.{Optimizer_SGD  => OPTIMIZER}
import neuralnet.{Optimizer_SGDM => OPTIMIZER}
//import neuralnet.{Optimizer_Adam => OPTIMIZER}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3L` class supports multi-output, 3-layer (input, hidden and output)
 *  Neural-Networks.  It can be used for both classification and prediction,
 *  depending on the activation functions used.  Given several input vectors and output
 *  vectors (training data), fit the parameters a and b connecting the layers,
 *  so that for a new input vector v, the net can predict the output value, i.e.,
 *      yp = f1 (b * f (a * v))
 *  where f and f1 are the activation functions and the parameter a and b
 *  are the parameters between input-hidden and hidden-output layers.
 *  Unlike `NeuralNet_2L` which adds input x0 = 1 to account for the intercept/bias,
 *  `NeuralNet_3L` explicitly adds bias.
 *  @param x       the m-by-n input/data matrix (training data consisting of m input vectors)
 *  @param y       the m-by-ny output/response matrix (training data consisting of m output vectors)
 *  @param fname_  the feature/variable names (if null, use x_j's)
 *  @param nz      the number of nodes in hidden layer (-1 => use default formula)
 *  @param hparam  the hyper-parameters for the model/network
 *  @param f       the activation function family for layers 1->2 (input to output)
 *  @param f1      the activation function family for layers 2->3 (hidden to output)
 *  @param itran   the inverse transformation function returns response matrix to original scale
 */
class NeuralNet_3L (x: MatrixD, y: MatrixD, fname_ : Array [String] = null,
                    private var nz: Int = -1, hparam: HyperParameter = Optimizer.hp,
                    f: AFF = f_sigmoid, f1: AFF = f_id,
                    val itran: FunctionM2M = null)
      extends PredictorMV (x, y, fname_, hparam)
         with Fit (dfr = x.dim2, df = x.dim - x.dim2):                    // under-estimate of degrees of freedom

    private val eta  = hp("eta").toDouble                                 // learning rate
            val opti = new OPTIMIZER ()                                   // parameter optimizer

    // Guidelines for setting the number of nodes in hidden layer:
    if nz < 1 then nz = 2 * x.dim2 + 1                                    // [1] default number of nodes for hidden layer
//  if nz < 1 then nz = 2 * x.dim2 + y.dim2                               // [2] default number of nodes for hidden layer

    private val (n, ny) = (x.dim2, y.dim2)
    bb = Array (new NetParam (weightMat (n, nz), new VectorD (nz)),       // parameters (weights & biases) in to hid
                new NetParam (weightMat (nz, ny), new VectorD (ny)))      // parameters (weights & biases) hid to out

    _modelName = s"NeuralNet_3L_${f.name}_${f1.name}"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x_ and y_, fit the parameters bb.
     *  Minimize the error in the prediction by adjusting the parameters bb.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output matrix
     */
    def train (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        val epochs = opti.optimize3 (x_, y_, bb, eta, Array (f, f1))      // optimize parameters bb
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
        val etaI = (0.25 * eta, 4.0 * eta)                                                 // quarter to four times eta
        val epochs = opti.auto_optimize (x_, y_, bb, etaI, Array (f, f1), opti.optimize3)  // optimize parameters bb
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
        println (s"T E S T: bb = ${stringOf (bb)}")
        val yp = predict (x_)                                            // make predictions
        val yy = if itran == null then y_ else itran (y_)                // undo scaling, if used
        e = yy - yp                                                      // RECORD the residuals/errors (@see `Predictor`)
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
        val yp = f1.f_ (bb(1) dot f.f_ (bb(0) dot v))                     // scaled? prediction
        if itran == null then yp
        else itran (MatrixD (yp))(0)                                      // back to original scale
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given an input matrix v, predict the output/response matrix f(v).
     *  @param v  the input matrix
     */
    override def predict (v: MatrixD = x): MatrixD =
        val yp = f1.fM (bb(1) * (f.fM (bb(0) * v)))                       // scaled? predictions
        if itran == null then yp
        else itran (yp)                                                   // back to original scale
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): NeuralNet_3L =
        new NeuralNet_3L (x_cols, y, null, -1, hparam, f, f1, itran)
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
                  b_ : MatrixD = parameter): String =
//      summary (x_, fname_, b_(?, 0), null)                              // summary from `Fit`
        "summary2 not implemented yet"
    end summary2

end NeuralNet_3L


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3L` companion object provides factory methods for creating three-layer
 *  (one hidden layer) neural networks.  Note, 'scale' is defined in `Scaling`.
 */
object NeuralNet_3L extends Scaling:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_3L` with automatic rescaling from a combined data matrix.
     *  @param xy      the combined input and output matrix
     *  @param fname   the feature/variable names
     *  @param nz      the number of nodes in hidden layer (-1 => use default formula)
     *  @param hparam  the hyper-parameters
     *  @param f       the activation function family for layers 1->2 (input to output)
     *  @param f1      the activation function family for layers 2->3 (hidden to output)
     *  @param col     the first designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               nz: Int = -1, hparam: HyperParameter = Optimizer.hp,
               f: AFF = f_sigmoid, f1: AFF = f_id)
               (col: Int = xy.dim2 - 1): NeuralNet_3L =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale
        val (x, y) = (xy(?, 0 until col), xy(?, col until xy.dim2)) 

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f1.bounds != null then { val y_i = rescaleY (y, f1); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_3L (x_s, y_s, fname, nz, hparam, f, f1, itran)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_3L` with automatic rescaling from a data matrix and response matrix.
     *  @param x       the input/data matrix
     *  @param y       the output/response matrix
     *  @param fname   the feature/variable names
     *  @param nz      the number of nodes in hidden layer (-1 => use default formula)
     *  @param hparam  the hyper-parameters
     *  @param f       the activation function family for layers 1->2 (input to output)
     *  @param f1      the activation function family for layers 2->3 (hidden to output)
     */
    def rescale (x: MatrixD, y: MatrixD, fname: Array [String] = null,
                 nz: Int = -1, hparam: HyperParameter = Optimizer.hp,
                 f: AFF = f_sigmoid, f1: AFF = f_id): NeuralNet_3L =
        var itran: FunctionM2M = null                                        // inverse transform -> original scale

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f1.bounds != null then { val y_i = rescaleY (y, f1); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new NeuralNet_3L (x_s, y_s, fname, nz, hparam, f, f1, itran)
    end rescale

end NeuralNet_3L


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest` main function is used to test the `NeuralNet_3L` class.
 *  Try changing the eta and bSize hyper-parameters, as well as the activation function.
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest
 */
@main def neuralNet_3LTest (): Unit =

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
//  val mod = new NeuralNet_3L (x, y)                            // create NeuralNet_3L model with sigmoid (default)
    val mod = new NeuralNet_3L (x, y, f = f_tanh)                // create NeuralNet_3L model with tanh

    banner ("Small Example - NeuralNet_3L: trainNtest")
    mod.trainNtest ()()                                          // train and test the model
    mod.opti.plotLoss ("NeuralNet_3L")                           // loss function vs epochs

    banner ("Small Example - NeuralNet_3L: trainNtest2")
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    mod.opti.plotLoss ("NeuralNet_3L")                           // loss function vs epochs
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("neuralNet_3LTest: Compare with Linear Regression - first column of y")
    val rg0 = new Regression (x, y0)                             // create a Regression model
    rg0.trainNtest ()()                                          // train and test the model
    println (rg0.summary ())                                     // parameter/coefficient statistics

    banner ("neuralNet_3LTest: Compare with Linear Regression - second column of y")
    val rg1 = new Regression (x, y1)                             // create a Regression model
    rg1.trainNtest ()()                                          // train and test the model
    println (rg1.summary ())                                     // parameter/coefficient statistics

end neuralNet_3LTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest2` main function tests the `NeuralNet_3L` class using the
 *  Concrete dataset.  It has three outputs/response variables.
 *  There are two ways to create the model:
 *      new NeuralNet_3L (x, y, x_fname)       - depending on act. function user must rescale
 *      NeuralNet_3L.rescale (x, y, x_fname)   - automatically rescales, assumes matrix response
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest2
 */
@main def neuralNet_3LTest2 (): Unit =

    import Example_Concrete.{x, y, x_fname}                      // don't include intercept, uses biases instead

//  println (s"x = $x")
//  println (s"y = $y")
    println (s"x_fname = ${stringOf (x_fname)}")

//  val mod = new NeuralNet_3L (x, y, x_fname)                   // create model without intercept
    val mod = NeuralNet_3L.rescale (x, y, x_fname)               // create model without intercept- rescales

    banner ("Concrete - NeuralNet_3L: trainNtest")
    mod.trainNtest ()()                                          // train and test the model
    mod.opti.plotLoss ("NeuralNet_3L")                           // loss function vs epochs

    banner ("Concrete - NeuralNet_3L: trainNtest2")
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    mod.opti.plotLoss ("NeuralNet_3L")                           // loss function vs epochs
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("Concrete - NeuralNet_3L: validate")
    println (Fit.showFitMap (mod.validate ()()._2))

    banner ("Concrete - NeuralNet_3L: crossValidate")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end neuralNet_3LTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest3` main function tests the `NeuralNet_3L` class using the
 *  AutoMPG dataset.  There are two ways to create the model:
 *      new NeuralNet_3L (x, yy, x_fname)       - depending on act. function user must rescale
 *      NeuralNet_3L.rescale (x, yy, x_fname)   - automatically rescales, assumes matrix response
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest3
 */
@main def neuralNet_3LTest3 (): Unit =

    import Example_AutoMPG.{x, yy, x_fname}                      // don't include intercept, uses biases instead

//  println (s"x  = $x")
//  println (s"yy = $yy")
    println (s"x_fname = ${stringOf (x_fname)}")

    Optimizer.hp("eta") = 0.25                                   // try 0.25 for SGDM, 0.8 for Adam
//  val mod = new NeuralNet_3L (x, yy, x_fname)                  // create model without intercept
    val mod = NeuralNet_3L.rescale (x, yy, x_fname)              // create model without intercept - rescales

    banner ("AutoMPG - NeuralNet_3L: In-Sample trainNtest")
    mod.trainNtest ()()                                          // train and test the model
    mod.opti.plotLoss ("NeuralNet_3L")                           // loss function vs epochs

    banner ("AutoMPG - NeuralNet_3L: In-Sample trainNtest2 - auto-tunes")
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
//  println (mod.summary2 ())                                    // parameter/coefficient statistics
    mod.opti.plotLoss ("NeuralNet_3L")                           // loss function vs epochs

    banner ("AutoMPG - NeuralNet_3L: TNT validate")
    println (Fit.showFitMap (mod.validate ()()._2))

/*
    banner ("AutoMPG - NeuralNet_3L: crossValidate")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)
*/

end neuralNet_3LTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest4` main function tests the `NeuralNet_3L` class using the
 *  AutoMPG dataset.  It tests forward selection.
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest4
 */
@main def neuralNet_3LTest4 (): Unit =

    import Example_AutoMPG.{x, yy, x_fname}                      // don't include intercept, uses biases instead

//  println (s"x  = $x")
//  println (s"yy = $yy")
    println (s"x_fname = ${stringOf (x_fname)}")

    banner ("AutoMPG NeuralNet_3L")
//  val mod = new NeuralNet_3L (x, yy, x_fname)                  // create model without intercept
    val mod = NeuralNet_3L.rescale (x, yy, x_fname)              // create model without intercept - rescales
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, smape, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                     // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end neuralNet_3LTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest5` main function tests the `NeuralNet_3L` class using the AutoMPG
 *  dataset.  It tests forward, backward and stepwise selection.
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest5
 */
@main def neuralNet_3LTest5 (): Unit =

    import Example_AutoMPG.{x, yy, x_fname}                      // don't include intercept, uses biases instead

//  println (s"x  = $x")
//  println (s"yy = $yy")

    banner ("AutoMPG NeuralNet_3L")
//  val mod = new NeuralNet_3L (x, yy, x_fname)                  // create model without intercept
    val mod = NeuralNet_3L.rescale (x, yy, x_fname)              // create model without intercept - rescales
//  mod.trainNtest ()()                                          // train and test the model
    mod.trainNtest2 ()()                                         // train and test the model - with auto-tuning
    println (mod.summary2 ())                                    // parameter/coefficient statistics

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())

    println (s"x_fname = ${stringOf (x_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, smape, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end neuralNet_3LTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest6` main function tests the `NeuralNet_3L` class using the
 *  AutoMPG dataset.  It tries all activation functions of the form (f, id),
 *  Ideally, eta should be initialized separately for each activation function.
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest6
 */
@main def neuralNet_3LTest6 (): Unit =

    import Example_AutoMPG.{x, yy, x_fname}                      // don't include intercept, uses biases instead

//  println (s"x  = $x")
//  println (s"yy = $yy")
    println (s"x_fname = ${stringOf (x_fname)}")

    Optimizer.hp ("eta") = 0.025                                 // some activation functions need smaller eta
    for f <- f_aff do                                            // try all activation functions for first layer
        banner (s"AutoMPG NeuralNet_3L with ${f.name}")
        val mod = NeuralNet_3L.rescale (x, yy, x_fname, f = f)   // create model without intercept - rescales
        mod.trainNtest2 ()()                                     // train and test the model - with auto-tuning

        banner ("AutoMPG Validation Test")
        println (Fit.showFitMap (mod.validate ()()._2))
    end for

end neuralNet_3LTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest7` main function tests the `NeuralNet_3L` class using the
 *  AutoMPG dataset.  It uses the best combination of two features weight and modelyear.
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest7
 */
@main def neuralNet_3LTest7 (): Unit =

    import Example_AutoMPG.{x46, y, yy, x46_fname}                   // don't include intercept, uses biases instead

    val xs = rescaleX (x46, f_sigmoid)                               // ActivationFun rescale the X matrix to active range of sigmoid
//  val (mn, mx) = (x46.min, x46.max)
//  val xs = scale ((mn, mx), (-2, 2))(x46)                          // MatrixTranform scale the X matrix to (-2, 2)

/*
    val xs = new MatrixD (x46.dim, x46.dim2)                         // low-level rescaling approach to (-2, 2)
    for j <- x46.indices2 do
        val scale = 4.0 / (mx(j) - mn(j))
        xs(?, j) = (x46(?, j) - mn(j)) * scale - 2                   // shift and scale
    end for
*/

//  println (s"xs = $xs")
//  println (s"yy = $yy")
    println (s"x46_fname = ${stringOf (x46_fname)}")

    Optimizer.hp ("eta") = 0.01                                      // some activation functions need smaller eta
    val nz = 2                                                       // number of hidden nodes
    banner (s"AutoMPG NeuralNet_3L")
    val mod = new NeuralNet_3L (xs, yy, x46_fname, nz)               // create model without intercept
    val yp  = mod.trainNtest2 ()()._1                                // train and test the model - with auto-tuning

    banner ("AutoMPG Validation Test")
    println (Fit.showFitMap (mod.validate ()()._2))

    banner ("Compare Model with Formula f_nn")
    val a  = MatrixD ((2, 2), -2.12262, -0.743867,                   // weights: input -> hidden layer
                              -0.200314, 1.62988)
    val ab = VectorD (-2.15785,	-1.65227)                            // biases: input -> hidden layer
    val b  = MatrixD ((2, 1), 15.7250,                               // weights: hidden -> output layer 
                              13.1971) 
    val bb = VectorD (13.4702)                                       // bias: hidden -> output layer

//  def f_nn (x: VectorD): VectorD = mod.predict (x)                            // result from model
    def f_nn (x: VectorD): VectorD = ((b dot sigmoid_ ((a dot x) + ab)) + bb)   // result from formula (should be close)

    val yp2 = VectorD (for i <- xs.indices yield f_nn (xs(i))(0))    // compute the response

    val yp_ = yp(?, 0)                                               // get response from model
    println (s"(yp_ - yp2).norm = ${(yp_ - yp2).norm}")              // norm of difference
    new Plot (null, yp_, yp2, "yp_ (black/model) vs. yp2 (red/formula)")

    def ff (x: VectorD): Double =
        val xx = (x(0) - 1.61300) * (4/3.537) - 2
        val yy = (x(1) - 70) * (4.0/12) - 2
        val u  = -2.12262  * xx - 0.200314 * yy - 2.15785
        val v  = -0.743867 * xx + 1.62988  * yy - 1.65227
        val uu = 1.0 / (1.0 + exp(-u))
        val vv = 1.0 / (1.0 + exp(-v))
        15.7250 * uu + 13.1971 * vv + 13.4702
    end ff

    val yp3 = VectorD (for i <- xs.indices yield ff (x46(i)))        // compute the response

    def ff2 (x: VectorD): Double =
        15.7250 / (1.0 + exp (2.12262  * ((x(0) - 1.61300) * (4/3.537) - 2) +
                              0.200314 * ((x(1) - 70) * (4.0/12) - 2) + 2.15785)) +
        13.1971 / (1.0 + exp (0.743867 * ((x(0) - 1.61300) * (4/3.537) - 2) -
                              1.62988  * ((x(1) - 70) * (4.0/12) - 2) + 1.65227)) +
        13.4702
    end ff2

    val yp4 = VectorD (for i <- xs.indices yield ff2 (x46(i)))       // compute the response

    println (s"(y - yp_).norm = ${(y - yp_).norm}")                  // norm of difference
    println (s"(y - yp2).norm = ${(y - yp2).norm}")                  // norm of difference
    println (s"(y - yp3).norm = ${(y - yp3).norm}")                  // norm of difference
    println (s"(y - yp4).norm = ${(y - yp4).norm}")                  // norm of difference

end neuralNet_3LTest7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest8` main function is used to test the `NeuralNet_3L` class.
 *  It tests a simple case that does not require a file to be read.
 *  @see translate.google.com/translate?hl=en&sl=zh-CN&u=https:
 *       //www.hrwhisper.me/machine-learning-decision-tree/&prev=search
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest8
 */
@main def neuralNet_3LTest8 (): Unit =

    val x  = MatrixD ((10, 1), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val y  = VectorD (5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05)
    val ox = VectorD.one (x.dim) +^: x

    banner (s"Regression with intercept")
    val reg = new Regression (ox, y)
    reg.trainNtest ()()                                              // train and test the model

    banner (s"Perceptron sigmoid")
    val nn = Perceptron.rescale (ox, y)
    nn.trainNtest ()()                                               // train and test the model

    banner (s"Perceptron tanh")
    val nn2 = Perceptron.rescale (ox, y, f = ActivationFun.f_tanh)
    nn2.trainNtest ()()                                              // train and test the model

    val ym = MatrixD.fromVector (y)
    Optimizer.hp ("eta") = 0.85                                      // Preceptron and NeuralNet_2L use different optimizers,
                                                                     // so different learning rates (eta) are needed.
    banner (s"NeuralNet_2L sigmoid")
    val nn3 = NeuralNet_2L.rescale (ox, ym)
    nn3.trainNtest ()()                                              // train and test the model

    banner (s"NeuralNet_2L tanh")
    val nn4 = NeuralNet_2L.rescale (ox, ym, f = ActivationFun.f_tanh)
    nn4.trainNtest ()()                                              // train and test the model

    banner (s"NeuralNet_3L sigmoid-id")
    val nn5 = NeuralNet_3L.rescale (ox, ym)
    nn5.trainNtest ()()                                              // train and test the model

    banner (s"NeuralNet_3L tanh-tanh")
    val nn6 = NeuralNet_3L.rescale (ox, ym, f = ActivationFun.f_tanh, f1 = ActivationFun.f_tanh)
    nn6.trainNtest ()()                                              // train and test the model

end neuralNet_3LTest8

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest9` main function is used to test the `NeuralNet_3L` class.
 *  It uses the matrix equations from sectioon 10.7.5 on the example problem
 *  from section 10.7.8 and 10.7.11 exercises 1 and 2.
 *  Tests 1 instance.
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest9
 */
@main def neuralNet_3LTest9 (): Unit =

    val x = MatrixD ((1, 2), 2.0, 1.0)
    val y = MatrixD ((1, 1), 0.8)

    println (s"x = $x")
    println (s"y = $y")
    val sst = (y - y.mean).normFSq                                  // sum of squares total
    println (s"sst = $sst")

    banner ("initialize")

    var aa = MatrixD ((2, 2), 0.1, 0.2,
                              0.3, 0.4)
    var a_ = VectorD (0.1, 0.1)
    var bb = MatrixD ((2, 1), 0.5,
                              0.6)
    var b_ = VectorD (0.1)

    println (s"hidden layer aa = $aa, a_ = $a_")                    // weight matrix and bias vector
    println (s"output layer bb = $bb, b_ = $b_")                    // weight matrix and bias vector

    // let the learning rate eta = 1

    val fs = f_sigmoid                                              // sigmoid family of activation functions: fM, dM
    var u, z, v, yp, e, d1, d0, df1, df0: MatrixD = null

    for epoch <- 1 to 2 do
        banner (s"Forward Propagation for step $epoch")
        u  = x * aa + a_                                            // hidden pre-activation vector
        z  = fs.fM (u)                                              // hidden vector
        v  = z * bb + b_                                            // output pre-activation vector
        yp = fs.fM (v)                                              // output vector (predicted)
        e  = yp - y                                                 // negative error

        println (s"$u = u, $z = z, $v = v, $yp = yp, $e = e")

        banner (s"Backward Propagation for step $epoch")
        df1 = fs.dM (yp)                                            // derivative of f1
        df0 = fs.dM (z)                                             // derivative of f0
        d1  = e *~ df1                                              // delta1 @ output layer
        d0  = d1 * bb.transpose *~ df0                              // delta0 @ hidden layer

        println (s"$df1 = df1, $df0 = df0, $d1 = d1, $d0 = d0")

        banner (s"Parameter Update for step $epoch")
        bb = bb - z.transpose * d1                                  // updated output layer weight matrix
        b_ = b_ - d1.mean                                           // updated output bias vector
        aa = aa - x.transpose * d0                                  // updated hidden layer weight matrix
        a_ = a_ - d0.mean                                           // updated hidden layer bias vector

        println (s"$bb = bb, $b_ = b_, $aa = aa, $a_ = a_")

        banner (s"QoF for step $epoch")
        val sse = e.normFSq                                         // sum of squared errors
        val rSq = 1 - sse / sst                                     // R-squared

        println (s"sse = $sse, rSq = $rSq")
    end for

end neuralNet_3LTest9


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest10` main function is used to test the `NeuralNet_3L` class.
 *  It uses the matrix equations from section 10.7.5 on the example problem
 *  from section 10.7.8 and 10.7.11 exercises 1 and 2.
 *  Tests 9 instances.
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest10
 */
@main def neuralNet_3LTest10 (): Unit =

    // 9 data points:         x1    x2     y
    val xy = MatrixD ((9, 3), 0.0,  0.0,  0.5,
                              0.0,  0.5,  0.3,
                              0.0,  1.0,  0.2,

                              0.5,  0.0,  0.8,
                              0.5,  0.5,  0.5,
                              0.5,  1.0,  0.3,

                              1.0,  0.0,  1.0,
                              1.0,  0.5,  0.8,
                              1.0,  1.0,  0.5)

    val (x, y) = (xy.not(?, 2), xy(?, 2))                           // input matrix, output/response vector
    println (s"x = $x")
    println (s"y = $y")
    val sst = (y - y.mean).normSq                                   // sum of squares total
    println (s"sst = $sst")

    banner ("initialize")

    var aa = MatrixD ((2, 2), 0.1, 0.2,
                              0.3, 0.4)
    var a_ = VectorD (0.1, 0.1)
    var bb = MatrixD ((2, 1), 0.5,
                              0.6)
    var b_ = VectorD (0.1)

    println (s"hidden layer aa = $aa, a_ = $a_")                    // weight matrix and bias vector
    println (s"output layer bb = $bb, b_ = $b_")                    // weight matrix and bias vector

    // let the learning rate eta = 1

    val f0 = f_sigmoid                                              // sigmoid activation function family
    val f1 = f_sigmoid                                              // sigmoid activation function family
    var u, u_, z, v, v_, yp, e, d1, d0: MatrixD = null

    for epoch <- 1 to 2 do
        banner (s"forward propagation for step $epoch")
        u  = x * aa                                                 // hidden pre-activation matrix pre-bias
        u_ = u + a_                                                 // hidden pre-activation matrix
        z  = f0.fM (u_)                                             // hidden matrix
        v  = z * bb                                                 // output pre-activation matrix pre-bias
        v_ = v + b_                                                 // output pre-activation matrix
        yp = f1.fM (v_)                                             // output matrix (predicted)

        println (s"$u <- u $u_ <- u $z <- z $v <- v $v_ <- v_ $yp <- yp")

        banner (s"backward propagation for step $epoch")
        e  = yp - y                                                 // negative error matrix
        d1 = e *~ f1.dM (yp)                                        // delta1 @ output layer
        d0 = d1 * bb.transpose *~ f0.dM (z)                         // delta0 @ hidden layer

        println (s"$e <- e $d1 <- d1 $d0 <- d0")

        banner (s"parameter update for step $epoch")
        bb = bb - z.transpose * d1                                  // updated output layer weight matrix
        b_ = b_ - d1.mean                                           // updated output bias vector
        aa = aa - x.transpose * d0                                  // updated hidden layer weight matrix
        a_ = a_ - d0.mean                                           // updated hidden layer bias vector

        println (s"$bb <- bb $b_ <- b_ $aa <- aa $a_ <- a_")

        banner (s"QoF for step $epoch")
        val sse = e.normFSq                                         // sum of squared errors
        val rSq = 1 - sse / sst                                     // R-squared

        println (s"sse = $sse, rSq = $rSq")
    end for

end neuralNet_3LTest10


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest11` main function is used to test the `NeuralNet_3L` class.
 *  It test the basic matrix equations for gradient descent.
 *  Note, there is no bias for the hidden layer, so may add a bias vector for the hidden layer.
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest11
 */
@main def neuralNet_3LTest11 (): Unit =

    // 9 data points:    Constant    x1    x2    y0    y1
    val xy = MatrixD ((9, 5), 1.0,  0.0,  0.0,  0.5,  0.4,          // dataset
                              1.0,  0.0,  0.5,  0.3,  0.3,
                              1.0,  0.0,  1.0,  0.2,  0.2,

                              1.0,  0.5,  0.0,  0.8,  0.7,
                              1.0,  0.5,  0.5,  0.5,  0.5,
                              1.0,  0.5,  1.0,  0.3,  0.4,

                              1.0,  1.0,  0.0,  1.0,  0.9,
                              1.0,  1.0,  0.5,  0.8,  0.7,
                              1.0,  1.0,  1.0,  0.5,  0.5)
    val x    = xy(?, 0 until 3)                                     // matrix for predictor variables
    val y    = xy(?, 3 until 5)                                     // matrix for response variables
    val sst0 = (y(?, 0) - y(?, 0).mean).normSq                      // sum of squares total for y_:0
    val sst1 = (y(?, 1) - y(?, 1).mean).normSq                      // sum of squares total for y_:1

//  val η = 0.4                                                     // learning rate
    val η = 0.15                                                    // learning rate
    val a = MatrixD ((3, 2), 0.1, 0.1,                              // weights/parameters X -> Z
                             0.2, 0.1,
                             0.1, 0.1)
    val b = MatrixD ((2, 2), 0.1, 0.1,                              // weights/parameters Z -> Y
                             0.1, 0.1)

//  val f0 = f_sigmoid                                              // hidden layer activation function
    val f0 = f_tanh                                                 // hidden layer activation function
    val f1 = f_id                                                   // output layer activation function

    for epoch <- 1 to 10 do
        banner (s"improvement step $epoch")
        // forward
        val u  = x * a                                              // pre-activation vector
        val z  = f0.fM (u)                                          // hidden matrix
        val v  = z * b                                              // output pre-activation matrix
        val yp = f1.fM (v)                                          // predicted response from calculation for sigmoid
        // backward
        val ε  = y - yp                                             // error matrix
        val δ1 = ε *~ f1.dM (yp)                                    // delta1 @ output layer
        val δ0 = δ1 * b.ᵀ *~ f0.dM (z)                              // delta0 @ hidden layer (transpose (ᵀ))
        b     += z.ᵀ * δ1 * η                                       // parameter update Z -> Y
        a     += x.ᵀ * δ0 * η                                       // parameter update X -> Z

        val sse0 = ε(?, 0).normSq                                   // sum of squared errors for column 0
        val sse1 = ε(?, 1).normSq                                   // sum of squared errors for column 1

        banner ("forward")
        println (s"u     = $u")
        println (s"z     = $z")
        println (s"v     = $v")
        println (s"yp    = $yp")
        banner ("backward")
        println (s"ε     = $ε")
        println (s"δ1    = $δ1")
        println (s"δ0    = $δ0")
        println (s"b     = $b")
        println (s"a     = $a")
        banner ("metrics")
        println (s"sse0  = $sse0")
        println (s"sse1  = $sse1")
        println (s"R^2_0 = ${1 - sse0/sst0}")
        println (s"R^2_1 = ${1 - sse1/sst1}")

end neuralNet_3LTest11


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3LTest12` main function is used to test the `NeuralNet_3L` class.
 *  Tests 9 instances for comparison with `AutoDiff`.
 *  @see `scalation.calculus.AutoDiff`
 *  > runMain scalation.modeling.neuralnet.neuralNet_3LTest12
 */
@main def neuralNet_3LTest12 (): Unit =

    val x1 = VectorD (1, 2, 3, 4, 5, 6, 7, 8, 9)
    val x2 = VectorD (8, 7, 6, 5, 5, 4, 4, 3, 2)
    val yv = VectorD (1, 2, 4, 7, 9, 8, 6, 5, 3)
    val x  = MatrixD (x1, x2).transpose
    val y  = MatrixD.fromVector (yv)

    println (s"x = $x")                                             // 9x2 input matrix
    println (s"y = $y")                                             // 9x1 output matrix
    val sst = (y - y.mean).normFSq                                  // sum of squares total (sst)
    println (s"sst = $sst")

    banner ("initialize")

    var aa = MatrixD.fill (2, 2, 0.1)                               // input to hidden layer 2x2 weight matrix
    val a_ = MatrixD.fill (1, 2, 0.1)                               // hidden layer 1x2 bias matrix
    var bb = MatrixD.fill (2, 1, 0.1)                               // hidden to output layer weight matrix
    val b_ = MatrixD.fill (1, 1, 0.1)                               // output layer 1x1 bias matrix

    println (s"hidden layer aa = $aa, a_ = $a_")                    // weight and bias matrices
    println (s"output layer bb = $bb, b_ = $b_")                    // weight and bias matrices

    // let the learning rate eta = 1

    val f0 = f_sigmoid                                              // sigmoid activation function family
    val f1 = f_id                                                   // identity activation function family
    var u, u_, z, v, v_, yp, e, pd1, d1, pd0, d0: MatrixD = null

    for epoch <- 1 to 1 do
        banner (s"forward propagation for step $epoch")
        u  = x * aa                                                 // hidden pre-activation vector pre-bias
        u_ = u + a_(0)                                              // hidden pre-activation vector
        z  = f0.fM (u_)                                             // hidden vector
        v  = z * bb                                                 // output pre-activation vector pre-bias
        v_ = v + b_(0)                                              // output pre-activation vector
        yp = f1.fM (v_)                                             // output vector (predicted)

        println (s"$u <- u $u_ <- u $z <- z $v <- v $v_ <- v_ $yp <- yp")

        banner (s"backward propagation for step $epoch")
        e   = yp - y                                                 // negative error
        pd1 = f1.dM (yp)                                             // partial derivative @ output layer
        d1  = e *~ pd1                                               // delta1 @ output layer
        pd0 = f0.dM (z)                                              // partial derivative @ hidden layer
        d0  = d1 * bb.transpose *~ pd0                               // delta0 @ hidden layer

        println (s"$e <- e $pd1 <- pd1 $d1 <- d1 $pd0 <- pd0 $d0 <- d0")

        banner (s"parameter for step $epoch")
        bb = bb - z.transpose * d1                                  // updated output layer weight matrix
        b_(0) = b_(0) - d1.mean                                     // updated output bias vector
        aa = aa - x.transpose * d0                                  // updated hidden layer weight matrix
        a_(0) = a_(0) - d0.mean                                     // updated hidden layer bias vector

        println (s"$bb <- bb $b_ <- b_ $aa <- aa $a_ <- a_")

        banner (s"QoF for step $epoch")
        val sse = e.normFSq                                         // sum of squared errors
        val rSq = 1 - sse / sst                                     // R-squared

        println (s"sse = $sse, rSq = $rSq")
    end for

end neuralNet_3LTest12

