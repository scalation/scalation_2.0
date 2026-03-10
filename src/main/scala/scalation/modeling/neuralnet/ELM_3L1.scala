
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jun 15 13:19:05 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Extreme-Learning Machines with Regression
 *
 *  @see     www.ntu.edu.sg/home/egbhuang/pdf/ELM-Unified-Learning.pdf
 *  @see     www.sciencedirect.com/science/article/pii/S0893608014002214
 */

package scalation
package modeling
package neuralnet

import scalation.mathstat._

import ActivationFun._
import Initializer._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ELM_3L1` class supports single-output, 3-layer (input, hidden and output)
 *  Extreme-Learning Machines.  It can be used for both classification and prediction,
 *  depending on the activation functions used.  Given several input vectors and output
 *  vectors (training data), fit the parameters a and b connecting the layers,
 *  so that for a new input vector v, the net can predict the output value, i.e.,
 *      yp = b * f (a * v)
 *  where f is the activation function and the parameter a and b
 *  are the parameters between input-hidden and hidden-output layers.
 *  Like `Perceptron` which adds input 'x0 = 1' to account for the intercept/bias,
 *  `ELM_3L1` explicitly adds bias.
 *  @param x       the m-by-n input matrix (training data consisting of m input vectors)
 *  @param y       the m output vector (training data consisting of m output scalars)
 *  @param fname_  the feature/variable names (if null, use x_j's)
 *  @param nz      the number of nodes in hidden layer (-1 => use default formula)
 *  @param hparam  the hyper-parameters for the model/network
 *  @param f       the activation function family for layers 1->2 (input to hidden)
 *  @param itran   the inverse transformation function returns responses to original scale
 */
class ELM_3L1 (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
               private var nz: Int = -1, hparam: HyperParameter = null,
               f: AFF = f_tanh, val itran: FunctionV2V = null)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):

    private val debug = debugf ("ELM_3L1", false)                       // debug function
    private val n     = x.dim2                                          // nodes in input layer
    private val s     = 8                                               // random number stream to use (0 - 999)

    if nz < 1 then nz = 2 * n + 1                                       // default number of nodes for hidden layer
    val df_r = compute_dfr (nz)                                         // degrees of freedom for regression/model (first output only)
    resetDF (df_r, x.dim - df_r)                                        // degrees of freedom for (regression/model, error)
 
    private val a = new NetParam (weightMat3 (n, nz, s),
                                  weightVec3 (nz, s))                   // parameters (weights & biases) in to hid (fixed)

    _modelName = s"ELM_3L1_${f.name}"

    println (s"Create an ELM_3L1 with $n input, $nz hidden and 1 output nodes: df_r = $df_r")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the degrees of freedom for the model (based on n, nz_, ny = 1).
     *  Rough extimate based on total number of parameters - 1.
     *  @param nz_  the number of nodes in the hidden layer
     */
    def compute_dfr (nz_ : Int): Int = nz_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameters b.  Since the a weights are fixed, only return b.
     */
    def parameters: VectorD = b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given training data x_ and y_, with parameters a fixed, fit parameters b.
     *  Use matrix factorization in `Regression` to find optimal values for the
     *  parameters/weights b.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        val z = f.fM (a *  x_)                                          // layer 1->2: Z  = f(XA)
//      debug ("train", s"x_ = $x_, \n z = $z, \n y_ = $y_")
        val reg = new Regression (z, y_)                                // layer 2-3: delegate to `Regression`
        reg.train ()
        b = reg.parameter
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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new input vector z, predict the output/response vector f(z).
     *  @param z  the new input vector
     */
    override def predict (z: VectorD): Double = b dot f.f_ (a dot z)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given an input matrix z, predict the output/response matrix f(z).
     *  @param z  the input matrix
     */
    override def predict (z: MatrixD = x): VectorD = f.fM (a * z) * b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): ELM_3L1 =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new ELM_3L1 (x_cols, y, fname2, -1, hparam, f, itran)
    end buildModel

end ELM_3L1


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ELM_3L1` companion object provides factory methods for creating three-layer
 *  (one hidden layer) extreme learning machines.  Note, 'scale' is defined in `Scalaing`.
 */
object ELM_3L1 extends Scaling:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ELM_3L1` for a combined data-response matrix.
     *  @param xy      the combined input/data and output/response matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param nz      the number of nodes in hidden layer (-1 => use default formula)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     *  @param f       the activation function family for layers 1->2 (input to hidden)
     *                 (defaults to tanh)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               nz: Int = -1, hparam: HyperParameter = Regression.hp,
               f: AFF = f_tanh)
              (col: Int = xy.dim2 - 1): ELM_3L1 =
        val (x, y) = (xy.not(?, col), xy(?, col))                       // column col is the response

        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = y                                                     // no need to scale y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new ELM_3L1 (x_s, y_s, fname, nz, hparam, f, itran = null)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ELM_3L1` for a data matrix and response vector.
     *  @param x       the m-by-n input/data matrix
     *  @param y       the m-dimensional output/response vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param nz      the number of nodes in hidden layer (-1 => use default formula)
     *  @param hparam  the hyper-parameters
     *  @param f       the activation function family for layers 1->2 (input to hidden)
     *                 (defaults to tanh)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
               nz: Int = -1, hparam: HyperParameter = Regression.hp,
               f: AFF = f_tanh): ELM_3L1 =
        val x_s = if scale then rescaleX (x, f)
                  else x
        val y_s = y                                                     // no need to scale y

//      println (s" scaled: x = $x_s \n scaled y = $y_s")
        new ELM_3L1 (x_s, y_s, fname, nz, hparam, f, itran = null)
    end rescale

end ELM_3L1


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `eLM_3L1Test` main function tests the multi-collinearity method in the
 *  `ELM_3L1` class using the following regression equation on the Blood
 *  Pressure dataset.  It also applies forward selection and backward elimination.
 *      y = b dot x = b_0 + b_1*x_1 + b_2*x_2 + b_3*x_3 + b_4 * x_4
 *  @see online.stat.psu.edu/online/development/stat501/12multicollinearity/05multico_vif.html
 *  @see online.stat.psu.edu/online/development/stat501/data/bloodpress.txt
 *  > runMain scalation.modeling.neuralnet.eLM_3L1Test
 */
@main def eLM_3L1Test (): Unit =                                     // FIX - fails

    import Example_BPressure._

    println ("model: y = b_0 + b_1*x1 + b_2*x_ + b3*x3 + b4*x4")

    banner ("ELM_3L1 vs. Regression - Example_BPressure")

    banner ("Regression")
    val reg = Regression (oxy)()
    reg.trainNtest ()()

    banner ("ELM_3L1 with scaled x values")

    val elm = ELM_3L1 (oxy)()                                        // factory method automatically rescales
//  val elm = new ELM_3L1 (ox, y)                                    // constructor does not automatically rescale
    elm.trainNtest ()()

    banner ("Collinearity Diagnostics")
    println ("ox.corr = " + ox.corr)                                 // correlations of column vectors in x

    banner ("Multi-collinearity Diagnostics")
    println ("vif     = " + elm.vif ())                              // test multi-colinearity (VIF)

    banner ("Forward Selection Test")
    elm.forwardSelAll ()

end eLM_3L1Test


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `eLM_3L1Test2` main function tests an extreme learning machine on the `Example_BasketBall`
 *  dataset, comparing its QoF with the QoF for `Regression`.
 *  > runMain scalation.modeling.neuralnet.eLM_3L1Test2
 */
@main def eLM_3L1Test2 (): Unit =

    import Example_BasketBall.{oxy, ox, y}

    banner ("ELM_3L1 vs. Regression - Example_BasketBall")

    banner ("Regression")
    val reg = Regression (oxy)()
    reg.trainNtest ()()

    banner ("prediction")                                            // not currently rescaling
    val yq = reg.predict (ox)                                        // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yq = " + yq)
    println ("error:            e  = " + (y - yq))

    banner ("ELM_3L1 with scaled x values")

    val elm = ELM_3L1 (oxy)()                                        // factory method automatically rescales
//  val elm = new ELM_3L1 (ox, y)                                    // constructor does not automatically rescale
    elm.trainNtest ()()

    banner ("prediction")
    val yp = elm.predict ()                                          // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yp = " + yp)
    println ("error:            e  = " + (y - yp))

end eLM_3L1Test2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `eLM_3L1Test3` main function tests an extreme learning machine on the `Example_AutoMPG`
 *  dataset, comparing its QoF with the QoF for `Regression`.
 *  > runMain scalation.modeling.neuralnet.eLM_3L1Test3
 */
@main def eLM_3L1Test3 (): Unit =

    import Example_AutoMPG._

    banner ("ELM_3L1 vs. Regression - Example_AutoMPG")

    banner ("Regression")
    val reg = Regression (oxy)()
    reg.trainNtest ()()

/*
    banner ("prediction")                                            // not currently rescaling
    val yq = reg.predict ()                                          // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yq = " + yq)
    println ("error:            e  = " + (y - yq))
*/

    banner ("ELM_3L1 with scaled x values")

//  val elm = new ELM_3L1 (ox, y))                                   // constructor does not automatically rescale
    val elm = ELM_3L1 (oxy)()                                        // factory method automatically rescales
    elm.trainNtest ()()

/*
    banner ("prediction")
    val yp = elm.predict ()                                          // scaled predicted output values for all x
    println ("target output:    y  = " + y)
    println ("predicted output: yp = " + yp)
    println ("error:            e  = " + (y - yp))
*/

end eLM_3L1Test3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `eLM_3L1Test4` main function tests an extreme learning machine on the `Example_AutoMPG`
 *  dataset.  It test cross-validation.
 *  > runMain scalation.modeling.neuralnet.eLM_3L1Test4
 */
@main def eLM_3L1Test4 (): Unit =

    import Example_AutoMPG._

    banner ("ELM_3L1 cross-validation - Example_AutoMPG")

    banner ("ELM_3L1 with scaled x values")

    var elm: ELM_3L1 = null
    for nz <- 11 to 31 by 2 do
        elm = ELM_3L1 (xy, nz = nz)()                                // factory method automatically rescales
//      elm = new ELM_3L1 (x, y)                                     // constructor does not automatically rescale
        elm.trainNtest ()()
    end for

//  banner ("cross-validation")
//  elm.crossVal ()

end eLM_3L1Test4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `eLM_3L1Test5` main function tests an extreme learning machine on the `Example_AutoMPG`
 *  dataset.  It tests forward feature/variable selection.
 *  > runMain scalation.modeling.neuralnet.eLM_3L1Test5
 */
@main def eLM_3L1Test5 (): Unit =

    import Example_AutoMPG._

    val nz  = 17                                                     // number of nodes in hidden layer
    banner ("ELM_3L1 feature selection - Example_AutoMPG")

    banner ("ELM_3L1 with scaled x values")
    val elm = ELM_3L1 (xy, nz = nz)()                                // factory method automatically rescales
//  val elm = new ELM_3L1 (x, y)                                     // constructor does not automatically rescale
    elm.trainNtest ()()

    banner ("Forward Selection Test")
    elm.forwardSelAll ()                                             // showing R^2, R^2 Bar, smape, R^2 cv

end eLM_3L1Test5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `eLM_3L1Test6` main function tests an extreme learning machine on the `Example_AutoMPG`
 *  dataset.  It tests forward feature/variable selection with plotting of R^2.
 *  > runMain scalation.modeling.neuralnet.eLM_3L1Test6
 */
@main def eLM_3L1Test6 (): Unit =

    import Example_AutoMPG._

    val n = ox.dim2                                                 // number of parameters/variables
    banner ("ELM_3L1 feature selection - Example_AutoMPG")

    val f_ = f_tanh                                                 // try different activation functions
//  val f_ = f_sigmoid                                              // try different activation functions
//  val f_ = f_lreLU                                                // try different activation functions

    banner ("ELM_3L1 with scaled x values")
    val elm  = ELM_3L1 (oxy, f = f_)()                              // factory method automatically rescales
//  val elm  = new ELM_3L1 (ox, y, f = f_)                          // constructor does not automatically rescale
    elm.trainNtest ()()                                             // fit the weights using training data

    banner ("Forward Selection Test")
    val (cols, rSq) = elm.forwardSelAll ()                          // R^2, R^2 bar, smape, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                    // instance index
    new PlotM (t, rSq.transpose, Regression.metrics, "R^2 vs n for ELM_3L1", lines = true)

end eLM_3L1Test6

