
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep  9 13:30:41 EDT 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Perceptron (single output 2-layer Neural-Network)
 *
 *  @see     hebb.mit.edu/courses/9.641/2002/lectures/lecture03.pdf
 */

package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import ActivationFun._
import Initializer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Perceptron` class supports single-output, 2-layer (input and output)
 *  Neural-Networks.  Although perceptrons are typically used for classification,
 *  this class is used for prediction.  Given several input vectors and output
 *  values (training data), fit the weights/parameters b connecting the layers,
 *  so that for a new input vector z, the net can predict the output value, i.e.,
 *      z = f (b dot z)
 *  The parameter vector b (w) gives the weights between input and output layers.
 *  Note, b0 is treated as the bias, so x0 must be 1.0.
 *  @param x       the data/input m-by-n matrix (data consisting of m input vectors)
 *  @param y       the response/output m-vector (data consisting of m output values)
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters for the model/network (defaults to Perceptron.hp)
 *  @param f       the activation function family for layers 1->2 (input to output)
 *  @param itran   the inverse transformation function returns responses to original scale
 */
class Perceptron (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                  hparam: HyperParameter = Perceptron.hp,
                  f: AFF = f_sigmoid, val itran: FunctionV2V = null)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2)
         with MonitorLoss:

    private val debug     = debugf ("Perceptron", false)                // debug function
    private val flaw      = flawf ("Perceptron")                        // flaw function
    private val (m, n)    = x.dims                                      // input data matrix dimensions
    private val η         = hparam("eta").toDouble                      // the learning/convergence rate (requires adjustment)
    private val maxEpochs = hparam("maxEpochs").toInt                   // the maximum number of training epcochs/iterations

    _modelName = s"Perceptron_${f.name}"

    if y.dim != m then flaw ("init", "dimensions of x and y are incompatible")

    println (s"Create a Perceptron with $n input nodes and 1 output node")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the initial parameter/weight vector b manually before training.
     *  This is mainly for testing purposes.
     *  @param w0  the initial weights for parameter b
     */
    def setWeights (w0: VectorD): Unit = b = w0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x_ and y_, fit the parameter/weight vector b.
     *  Minimize the error in the prediction by adjusting the weight vector b.
     *  The error e is simply the difference between the target value y_ and the
     *  predicted value yp.  Minimize the dot product of error with itself using
     *  gradient-descent (move in the opposite direction of the gradient).
     *  Iterate over several epochs (no batching).
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        println (s"train: eta η = $η")
        if b == null then b = weightVec (n)                             // initialize parameters/weights
        var sse0 = Double.MaxValue

        var (go, epoch) = (true, 1)
        cfor (go && epoch <= maxEpochs, epoch += 1) {                   // epoch learning phase
            val yp = f.f_ (x_ * b)                                      // predicted output vector yp = f(Xb)
            val e  = y_ - yp                                            // error vector for y (protected var from `Predictor)
            val δ  = -f.d (yp) * e                                      // delta vector for y (protected var from `Predictor)
            b     -= x_.ᵀ * δ * η                                       // update the parameters/weights (ᵀ for transpose)

            val sse = (y_ - f.f_ (x_ * b)).normSq                       // recompute sum of squared errors
            collectLoss (sse)                                           // collect loss per epoch
            debug ("train", s"parameters for $epoch th epoch: b = $b, sse = $sse")
            if sse >= sse0 then go = false                              // stop when sse increases
            else sse0 = sse                                             // save prior sse
        } // cfor
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                           // make predictions
        val yy = if itran == null then y_ else itran (y_)               // undo scaling, if used
        (yp, diagnose (yy, yp))                                         // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.
     *  FIX - currently must override if y is transformed, @see `Predictor`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output vector (defaults to full y)
     */
    override def trainNtest (x_ : MatrixD = x, y_ : VectorD = getY)
                            (xx: MatrixD = x, yy: VectorD = y): (VectorD, VectorD) =
        train (x_, y_)
        debug ("trainNTest", s"b = $b")
        val (yp, qof) = test (xx, yy)
        println (report (qof))
        Predictor.plotPrediction (if itran == null then yy else itran (yy), yp, modelName)   // undo scaling, if used
        (yp, qof)
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input vector z, predict the output/response value f(z).
     *  @param z  the new input vector
     */
    override def predict (z: VectorD): Double =
       val yp = f.f (b dot z)                                           // scaled prediction
       if itran == null then yp else itran (VectorD (yp))(0)            // back to original scale
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input matrix z, predict the output/response value f(z).
     *  @param z  the new input matrix
     */
    override def predict (z: MatrixD = x): VectorD =
       val yp = f.f_ (z * b)                                            // scaled predictions
       if itran == null then yp else itran (yp)                         // back to original scale
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): Perceptron =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new Perceptron (x_cols, y, fname2, hparam, f, itran)
    end buildModel

end Perceptron


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Perceptron` companion object provides factory methods for creating perceptrons.
 */
object Perceptron extends Scaling:

    /** hyper-parameters for tuning the optimization algorithms - user tuning
     */
    val hp = new HyperParameter
    hp += ("eta", 0.1, 0.1)                                             // learning/convergence rate
    hp += ("maxEpochs", 400, 400)                                       // maximum number of epochs/iterations

    private val debug = debugf ("Perceptron", false)                    // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Perceptron` with automatic rescaling from a combined data matrix.
     *  @param xy      the combined data/input and response/output matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to hp)
     *  @param f       the activation function family for layers 1->2 (input to output)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp, f: AFF = f_sigmoid)
              (col: Int = xy.dim2 - 1): Perceptron =
        var itran: FunctionV2V = null                                   // inverse transform -> original scale
        val (x, y) = (xy.not(?, col), xy(?, col))                       // column col is the response

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f.bounds != null then { val y_i = rescaleY (y, f); itran = y_i._2; y_i._1 }
                  else y

        debug ("apply", s" scaled: x = $x_s \n scaled y = $y_s")
        new Perceptron (x_s, y_s, fname, hparam, f, itran)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `Perceptron` with automatic rescaling from a data matrix and response vector.
     *  @param x       the data/input matrix
     *  @param y       the response/output vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to hp)
     *  @param f       the activation function family for layers 1->2 (input to output)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp, f: AFF = f_sigmoid): Perceptron =
        var itran: FunctionV2V = null                                   // inverse transform -> original scale

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = if f.bounds != null then { val y_i = rescaleY (y, f); itran = y_i._2; y_i._1 }
                  else y

        debug ("rescale", s"scaled: x = $x_s \n scaled y = $y_s")
        new Perceptron (x_s, y_s, fname, hparam, f, itran)
    end rescale

end Perceptron

import Perceptron.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `perceptronTest` object trains a perceptron on a small dataset with variables
 *  x1 and x2.  The model equation is the following:
 *      y  =  sigmoid (b dot x)  =  sigmoid (b0 + b1*x1 + b2*x2)
 *  Does not call the train method; improvements steps for sigmoid are explicitly in code below.
 *  > runMain scalation.modeling.perceptronTest
 */
@main def perceptronTest (): Unit =

/*
    // 9 data points:         One    x1    x2     y
    val xy = MatrixD ((9, 4), 1.0,  1.0,  1.0,  0.04,           // dataset 1
                              1.0,  2.0,  1.0,  0.05,
                              1.0,  3.0,  1.0,  0.06,

                              1.0,  1.0,  2.0,  0.10,
                              1.0,  2.0,  2.0,  0.11,
                              1.0,  3.0,  2.0,  0.12,

                              1.0,  1.0,  3.0,  0.20,
                              1.0,  2.0,  3.0,  0.21,
                              1.0,  3.0,  3.0,  0.22)

    val b = VectorD ( 4.0, 0.58, 4.0)                               // initial weights/parameters
//  val b = VectorD (-5.0, -0.5, 1.5)                               // initial weights/parameters, better
*/

    // 9 data points:         One    x1    x2     y
    val xy = MatrixD ((9, 4), 1.0,  0.0,  0.0,  0.5,            // dataset 2
                              1.0,  0.0,  0.5,  0.3,
                              1.0,  0.0,  1.0,  0.2,

                              1.0,  0.5,  0.0,  0.8,
                              1.0,  0.5,  0.5,  0.5,
                              1.0,  0.5,  1.0,  0.3,

                              1.0,  1.0,  0.0,  1.0,
                              1.0,  1.0,  0.5,  0.8,
                              1.0,  1.0,  1.0,  0.5)

    val b = VectorD (0.1, 0.2, 0.1)                                 // initial weights/parameters
//  val b = VectorD (0.1, 0.1, 0.1)                                 // initial weights/parameters

    val _1 = VectorD.one (xy.dim)                                   // vector of all ones

    println (s"xy = $xy")
    val (x, y) = (xy.not(?, 3), xy(?, 3))                           // input matrix, output/response vector
    val sst = (y - y.mean).normSq                                   // sum of squares total
    println (s"sst = $sst")

    val η   = 2.0 
    hp("eta") = η                                                   // try several values for eta
//  val nn = new Perceptron (x, y, null, hp, f_reLU)                // create a perceptron, user control
//  val nn = new Perceptron (x, y, null, hp)                        // create a perceptron, user control
//  val nn = Perceptron (xy, null, hp)                              // create a perceptron, automatic scaling

    banner ("initialize")

//  nn.setWeights (b)                                               // set the parameters/weights

    var u, yp, e, fp, d, g: VectorD = null
 
    for epoch <- 1 to 10 do
        banner (s"improvement step $epoch")
        u   = x * b                                                 // pre-activation value
        yp  = sigmoid_ (u)                                          // predicted response from calculation for sigmoid
        e   = y - yp                                                // error
        fp  = yp * (_1 - yp)                                        // derivative (f') for sigmoid
        d   = - e * fp                                              // delta
        g   = x.ᵀ * d                                               // gradient
        b  -= g * η                                                 // new parameter vector
        val sse = e.normSq                                          // sum of squared errors

//      val yp2 = nn.predict ()                                     // predicted response from nn
//      val yp  = reLU_ (u)                                         // predicted response from calculation for reLU
//      assert (yp == yp2)
//      val fp  = u.map (z => is_ (z >= 0.0))                       // derivative (f') for reLU

        println (s"u   = $u")                                 
        println (s"y   = $y")
        println (s"yp  = $yp")
//      println (s"yp2 = $yp2")
        println (s"e   = $e")
        println (s"fp  = $fp")
        println (s"d   = $d")
        println (s"g   = $g")
        println (s"b   = $b")
        println (s"sse = $sse")
        println (s"R^2 = ${1 - sse/sst}")
//      nn.setWeights (b)
    end for
 
end perceptronTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `perceptronTest2` object trains a perceptron on a small dataset of
 *  temperatures from counties in Texas where the variables/factors to consider
 *  are Latitude (x1), Elevation (x2) and Longitude (x3).  The model equation
 *  is the following:
 *      y  =  sigmoid (w dot x)  =  sigmoid (w0 + w1*x1 + w2*x2 + w3*x3)
 *  This test case illustrates how to transform the columns of the matrix
 *  so that the sigmoid activation function can work effectively.
 *  > runMain scalation.modeling.perceptronTest2
 */
@main def perceptronTest2 (): Unit =

    val fname = Array ("one", "Lat", "Elev", "Long")

    // 16 data points:         one      x1      x2       x3     y
    //                                 Lat    Elev     Long  Temp        County
    val xy = MatrixD ((16, 5), 1.0, 29.767,   41.0,  95.367, 56.0,    // Harris
                               1.0, 32.850,  440.0,  96.850, 48.0,    // Dallas
                               1.0, 26.933,   25.0,  97.800, 60.0,    // Kennedy
                               1.0, 31.950, 2851.0, 102.183, 46.0,    // Midland
                               1.0, 34.800, 3840.0, 102.467, 38.0,    // Deaf Smith
                               1.0, 33.450, 1461.0,  99.633, 46.0,    // Knox
                               1.0, 28.700,  815.0, 100.483, 53.0,    // Maverick
                               1.0, 32.450, 2380.0, 100.533, 46.0,    // Nolan
                               1.0, 31.800, 3918.0, 106.400, 44.0,    // El Paso
                               1.0, 34.850, 2040.0, 100.217, 41.0,    // Collington
                               1.0, 30.867, 3000.0, 102.900, 47.0,    // Pecos
                               1.0, 36.350, 3693.0, 102.083, 36.0,    // Sherman
                               1.0, 30.300,  597.0,  97.700, 52.0,    // Travis
                               1.0, 26.900,  315.0,  99.283, 60.0,    // Zapata
                               1.0, 28.450,  459.0,  99.217, 56.0,    // Lasalle
                               1.0, 25.900,   19.0,  97.433, 62.0)    // Cameron

    val (x, y) = (xy.not (?, 4), xy(?, 4))
    println (s"x = $x")

    banner ("Perceptron with scaled y values")
    hp("eta") = 0.5                                                 // try several values for the learning rate
    val mod = Perceptron.rescale (x, y, fname)                      // factory method automatically rescales
//  val mod = new Perceptron (x, y, fname)                          // constructor does not automatically rescale

    mod.trainNtest ()()                                             // train and test the model
//  println (mod.summary ())                                        // parameter/coefficient statistics - FIX implement?

    banner ("scaled prediction")
    val yp = mod.predict ()                                         // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yp = " + yp)

    banner ("unscaled prediction")
    val (ymin, ymax) = (y.min, y.max)                               // FIX - obtain from apply
    val ypu = unscaleV ((ymin, ymax), (0, 1)) (yp)                  // unscaled predicted output values for all x
    println ("target output:   y   = " + y)
    println ("unscaled output: ypu = " + ypu)

end perceptronTest2

import Example_AutoMPG._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `perceptronTest3` main function tests the `Perceptron` class using the AutoMPG
 *  dataset.  It test cross validation.
 *  > runMain scalation.modeling.perceptronTest3
 */
@main def perceptronTest3 (): Unit =

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG Perceptron")
    hp("eta") = 0.015                                               // try several values for the learning rate
    val mod = Perceptron.rescale (ox, y, ox_fname)                  // create model with intercept (else pass x)
    mod.trainNtest ()()                                             // train and test the model
    mod.plotLoss ("Perceptron")                                     // loss function vs epochs
//  println (mod.summary ())                                        // parameter/coefficient statistics

    banner ("AutoMPG Validation Test")
    mod.validate ()()

    banner ("AutoMPG Cross-Validation Test")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end perceptronTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `perceptronTest4` main function tests the `Perceptron` class using the AutoMPG
 *  dataset.  Assumes no missing values.  It tests forward selection.
 *  > runMain scalation.modeling.perceptronTest4
 */
@main def perceptronTest4 (): Unit =

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG Perceptron")
    hp("eta") = 0.01                                                // try several values for the learning rate
    val mod = Perceptron.rescale (ox, y, ox_fname)                  // create model with intercept (else pass x)
    mod.trainNtest ()()                                             // train and test the model
//  println (mod.summary ())                                        // parameter/coefficient statistics

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                        // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, "R^2 vs n for Perceptron", lines = true)
    println (s"rSq = $rSq")

end perceptronTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `perceptronTest5` main function tests the `Perceptron` class using the AutoMPG
 *  dataset.  Assumes no missing values.  It tests forward, backward and stepwise selection.
 *  > runMain scalation.modeling.perceptronTest5
 */
@main def perceptronTest5 (): Unit =

//  println (s"ox = $ox")
//  println (s"y  = $y")

    banner ("AutoMPG Perceptron")
    hp("eta") = 0.01                                                // try several values for the learning rate
    val mod = Perceptron.rescale (ox, y, ox_fname)                  // create model with intercept (else pass x)
    mod.trainNtest ()()                                             // train and test the model
//  println (mod.summary ())                                        // parameter/coefficient statistics

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())

    println (s"ox_fname = ${stringOf (ox_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                   // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Perceptron with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end perceptronTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `perceptronTest6` main function tests the basic Perceptron equations with
 *  a Gradient Descent algorithm.
 *  > runMain scalation.modeling.perceptronTest6
 */
@main def perceptronTest6 (): Unit =

    // 9 data points:    Constant    x1    x2     y
    val xy = MatrixD ((9, 4), 1.0,  0.0,  0.0,  0.5,               // dataset
                              1.0,  0.0,  0.5,  0.3,
                              1.0,  0.0,  1.0,  0.2,

                              1.0,  0.5,  0.0,  0.8,
                              1.0,  0.5,  0.5,  0.5,
                              1.0,  0.5,  1.0,  0.3,

                              1.0,  1.0,  0.0,  1.0,
                              1.0,  1.0,  0.5,  0.8,
                              1.0,  1.0,  1.0,  0.5)
    val x   = xy.not (?, 3)                                        // matrix for predictor variables
    val y   = xy(?, 3)                                             // vector for response variable
    val sst = (y - y.mean).normSq                                  // sum of squares total

    val mod = new Regression (x, y)
    mod.trainNtest ()()

    val η = 1.0                                                    // learning rate
    val b = VectorD (0.1, 0.2, 0.1)
    val g = new VectorD (b.dim)
    for epoch <- 1 to 10 do
        banner (s"improvement step $epoch")
        val u  = x * b                                             // pre-activation vector
        val yp = sigmoid_ (u)                                      // predicted response from calculation for sigmoid
        val e  = y - yp                                            // error
        val fp = yp * (-yp + 1)                                    // derivative (f') for sigmoid
        for j <- x.indices2 do
            g(j)  = -e dot (x(?, j) * fp)                          // gradient in direction j
            b(j) -= η * g(j)                                       // update j-th parameter
        val sse = e.normSq                                         // sum of squared errors

        println (s"b   = $b")
        println (s"u   = $u")
        println (s"y   = $y")
        println (s"yp  = $yp")
        println (s"e   = $e")
        println (s"fp  = $fp")
        println (s"g   = $g")
        println (s"b   = $b")
        println (s"sse = $sse")
        println (s"R^2 = ${1 - sse/sst}")
    end for

end perceptronTest6

