
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Feb 13 16:22:21 EST 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: NeuralNet_3L for Time Series
 */

package scalation
package modeling
package forecasting
package neuralforecasting

import scala.collection.mutable.{LinkedHashSet => LSET}
import scala.math.max

import scalation.mathstat._

import ActivationFun._
import neuralnet.{NeuralNet_3L, Optimizer}

import Example_Covid.{loadData, response}
import MakeMatrix4TS._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3L4TS` class provides basic time series analysis capabilities for
 *  three layer neural network models that use DIRECT (as opposed to RECURSIVE)
 *  multi-horizon forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = f1(bb dot f(aa dot x_t)) + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y and ex) @see `NeuralNet_3L4TS.apply`
 *  @param y        the response/output matrix (column per horizon) (time series data)
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param nz       the number of nodes in hidden layer (-1 => use default formula)
 *  @param hparam   the hyper-parameters (defaults to MakeMatrix4TS.hp ++ Optimizer.hp)
 *  @param f        the activation function family for layers 1->2 (input to hidden)
 *  @param f1       the activation function family for layers 2->3 (hidden to output)
 *  @param itran    the inverse transformation function returns response matrix to original scale
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 */
class NeuralNet_3L4TS (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String] = null,
                       tRng: Range = null, nz: Int = -1,
                       hparam: HyperParameter = hp ++ Optimizer.hp,
                       f: AFF = f_tanh, f1: AFF = f_id, val itran: FunctionV2V = null,
                       bakcast: Boolean = false)
      extends Forecaster_D (x, y, hh, fname, tRng, hparam, bakcast):    // no automatic backcasting, @see `NeuralNet_3L4TS.apply`

    private val debug = debugf ("NeuralNet_3L4TS", true)                // debug function
//  private val flaw  = flawf ("NeuralNet_3L4TS")                       // flaw function
    private val p     = hparam("p").toInt                               // use the last p endogenous values (p lags)
    private val q     = hparam("q").toInt                               // use the last q exogenous values (q lags)
    private val spec  = hparam("spec").toInt                            // trend terms: 0 - none, 1 - constant, 2 - linear, 3 - quadratic
                                                                        //              4 - sine, 5 cosine
    private val nneg  = hparam("nneg").toInt == 1                       // 0 => unrestricted, 1 => predictions must be non-negative
    private val nnet  = NeuralNet_3L.rescale (x, y, fname, nz, hparam, f, f1)
                                                                        // delegate training to neural network

    _modelName = s"NeuralNet_3L4TS_${p}_${q}_${f.name}_${f1.name}"

    debug ("init", s"$modelName with $n_exo exogenous variables and additional term spec = $spec with nneg = $nneg")
//  debug ("init", s"[ x | y ] = ${x ++^ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train/fit an `NeuralNet_3L4TS` model to the times-series data in matrix x and vector y_.
     *  @param x_  the data/input matrix (e.g., full x)
     *  @param y_  the training/full response vector (e.g., full y)
     */
    def train_x (x_ : MatrixD, y_ : MatrixD): Unit =
        debug ("train_x", s"$modelName, x_.dims = ${x_.dims}, y_.dims = ${y_.dims}")
//      nnet.train (x_, y_)                                              // train the neural network
        nnet.train2 (x_, y_)                                             // train the neural network (some auto-tuning)
    end train_x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a `NeuralNet_3L4TS` model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output matrix (defaults to full y)
     */
    def test_x (x_ : MatrixD = x, y_ : MatrixD = y): (MatrixD, MatrixD) =
        debug ("train_x", s"$modelName, x_.dims = ${x_.dims}, y_.dims = ${y_.dims}")
        nnet.test (x_, y_)                                              // train the neural network
    end test_x

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the forecasting model y_ = f(y-past) + e and report its QoF
     *  and plot its predictions.  Return the predictions and QoF.
     *  NOTE: must use `trainNtest_x` when an x matrix is used, such as in `ARY_D`.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output vector (defaults to full y)
     */
    def trainNtest_xx (x_ : MatrixD = x, y_ : MatrixD = y)(xx: MatrixD = x, yy: MatrixD = y):
                     (MatrixD, MatrixD) =
        train_x (x_, y_)                                                // train the model on training set
        val (yp, qof) = test_x (xx, yy)                                 // test the model on testing set
        println (nnet.report (qof))                                     // report on Quality of Fit (QoF)
        (yp, qof)
    end trainNtest_xx

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Models need to provide a means for updating the Degrees of Freedom (DF).
     *  Crude analog used for neural nets.
     *  @param size  the size of dataset (full, train, or test)
     */
    override def mod_resetDF (size: Int): Unit =
        val dfr = max (1, nnet.parameter.dim - 1)                       // degrees of freedom for regression/model
        debug ("mod_resetDF", s"dfr = $dfr, df = ${size-dfr}")
        resetDF (dfr, size - dfr)
    end mod_resetDF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict a value for y_t using the 1-step ahead forecast.
     *
     *      y_t = b_0 + b_1 y_t-1 + b_2 y_t-2 + ... + b_p y_t-p = b dot x_t
     *
     *  @param t   the time point being predicted
     *  @param y_  the actual values to use in making predictions (ignored)
     */
    def predict (t: Int, y_ : MatrixD): VectorD =
        val yp = if nneg then nnet.predict (x(t)).max0
                 else nnet.predict (x(t))
//      debug ("predict", s"@t = $t, x(t) = ${x(t)}, yp = $yp vs. y_ = ${y_(t)}")
        yp
    end predict

   //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build an `NeuralNet_3L4TS` model using the cols with the selected features.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel (cols: LSET [Int] = LSET.range (0, x.dim2)): NeuralNet_3L4TS =
        new NeuralNet_3L4TS (x(?, cols), y, hh, n_exo, cols.toArray.map (fname (_)), tRng, nz, hparam, f, f1, itran, bakcast)
    end getModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a single-horizon `Forecaster_Reg` model using the cols with the selected features.
     *  @param cols  the cols of the input matrix with selected features
     *  @param h     the number of the horizon
     */
    def getModel_h (cols: LSET [Int] = LSET.range (0, x.dim2), h: Int = 1): Forecaster_Reg =
        throw new UnsupportedOperationException ("getModel_h is not supported by NeuralNet_3L4TS")
    end getModel_h

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the underlying Regression Model to an `NeuralNet_3L4TS` Forecasting Model.
     *  @param mod  the regression model to convert, e.g., the best model after feature selection
     */
    def convertReg2Forc (mod: Model_FS = getBest.mod): NeuralNet_3L4TS =
        new NeuralNet_3L4TS (mod.getX, MatrixD.fromVector (mod.getY), 1, n_exo, fname,
                             tRng, nz, hparam, f, f1, itran, bakcast)    //, tForms)
    end convertReg2Forc

end NeuralNet_3L4TS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3L4TS` object supports 3-layer regression-like neural networks
 *  for Time Series data.  Given a response vector y, a predictor matrix x is built
 *  that consists of lagged y vectors.
 */
object NeuralNet_3L4TS extends Scaling:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_3L4TS` object from a response vector.  The input/data matrix
     *  x is formed from the lagged y vectors as columns in matrix x.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the original un-expanded output/response vector
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname    the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param nz       the number of nodes in hidden layer (-1 => use default formula)
     *  @param hparam   the hyper-parameters (use MakeMatrix4TS.hp ++ Optimizer.hp for default)
     *  @param f        the activation function family for layers 1->2 (input to hidden)
     *  @param f1       the activation function family for layers 2->3 (hidden to output)
     *  @param itran    the inverse transformation function returns response matrix to original scale
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname: Array [String] = null,
               tRng: Range = null, nz: Int = -1,
               hparam: HyperParameter = hp ++ Optimizer.hp,
               f: AFF = f_tanh, f1: AFF = f_id, itran: FunctionV2V = null,
               bakcast: Boolean = false): NeuralNet_3L4TS =
        val xy    = ARX.buildMatrix (xe, y, hparam, bakcast)
        val yy    = makeMatrix4Y (y, hh, bakcast)
        val n_exo = if xe == null then 0 else xe.dim2

        new NeuralNet_3L4TS (xy, yy, hh, n_exo, fname, tRng, nz, hparam, f, f1, itran, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_3L4TS` with automatic rescaling from a data matrix and response matrix.
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the original un-expanded output/response vector
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname    the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param nz       the number of nodes in hidden layer (-1 => use default formula)
     *  @param hparam   the hyper-parameters
     *  @param f        the activation function family for layers 1->2 (input to hidden)
     *  @param f1       the activation function family for layers 2->3 (hidden to output)
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname: Array [String] = null,
                 tRng: Range = null, nz: Int = -1,
                 hparam: HyperParameter = hp ++ Optimizer.hp,
                 f: AFF = f_tanh, f1: AFF = f_id,
                 bakcast: Boolean = false): NeuralNet_3L4TS =
        var itran: FunctionV2V = null                                        // inverse transform -> original scale (V2V or M2M)

        val x_s = if scale then rescaleX (xe, f)
                  else xe
        val y_s = if f1.bounds != null then { val y_i = rescaleY (y, f1); itran = y_i._2; y_i._1 }
                  else y

//      println (s" scaled: xe = $x_s \n scaled y = $y_s")
        apply (x_s, y_s, hh, fname, tRng, nz, hparam, f, f1, itran, bakcast)
    end rescale

end NeuralNet_3L4TS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3L4TSTest` main function tests the `NeuralNet_3L4TS` class.
 *  This test is used to CHECK that the buildMatrix4TS function is working correctly.
 *  May get NaN for some maximum lags (p) due to multi-collinearity.
 *  > runMain scalation.modeling.forecasting.neuralforecasting.neuralNet_3L4TSTest
 */
@main def neuralNet_3L4TSTest (): Unit =

    val m   = 30
    val y   = VectorD.range (1, m)                                     // used to CHECK the buildMatrix4TS function
    val hh  = 3                                                        // the forecasting horizon
    val hp2 = hp ++ Optimizer.hp

    for p <- 5 to 5 do                                                 // autoregressive hyper-parameter p
        hp2("p") = p
        banner (s"Test: NeuralNet_3L4TS with $p lags")
        val mod = NeuralNet_3L4TS (null, y, hh)                        // create model for time series data
        mod.trainNtest_xx ()()                                         // train the model on full dataset
        println (mod.summary)

        mod.forecastAll (mod.getYy)                                    // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (y, mod.getYf)                                 // diagnose for all horizons
    end for

end neuralNet_3L4TSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3L4TSTest2` main function tests the `NeuralNet_3L4TS` class on real data:
 *  Forecasting lake levels.
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.neuralforecasting.neuralNet_3L4TSTest2
 */
@main def neuralNet_3L4TSTest2 (): Unit =

    import Example_LakeLevels.y
    val hh  = 2                                                        // the forecasting horizon
    val hp2 = hp ++ Optimizer.hp

    for p <- 1 to 7 do                                                 // autoregressive hyper-parameter p
        hp2("p") = p
        banner (s"Test: NeuralNet_3L4TS with $p lags")
        val mod = NeuralNet_3L4TS (null, y, hh)                        // create model for time series data
        mod.trainNtest_xx ()()                                         // train the model on full dataset
        println (mod.summary)

        mod.forecastAll (mod.getYy)                                    // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (y, mod.getYf)                                 // diagnose for all horizons
    end for

end neuralNet_3L4TSTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3L4TSTest3` main function tests the `NeuralNet_3L4TS` class on real data:
 *  Forecasting COVID-19 Weekly Data.
 *  > runMain scalation.modeling.forecasting.neuralforecasting.neuralNet_3L4TSTest3
 */
@main def neuralNet_3L4TSTest3 (): Unit =

//  val exo_vars  = NO_EXO
    val exo_vars  = Array ("icu_patients", "hosp_patients")
//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                         // full
    val xe = xxe(0 until 116)                                            // clip the flat end
//  val y  = yy                                                          // full
    val y  = yy(0 until 116)                                             // clip the flat end
    val hh = 6                                                           // maximum forecasting horizon
    hp("lwave") = 20                                                     // wavelength (distance between peaks)

    new Plot (null, y, null, s"y (new_deaths) vs. t", lines = true)
    for j <- exo_vars.indices do
        new Plot (null, xe(?, j), null, s"x_$j (${exo_vars(j)}) vs. t", lines = true)

    val p    = 5                                                          // number of lags for endogenous variable
    val q    = 2                                                          // number of lags for exogenous variables
    val spec = 1                                                          // number of trend terms

    hp("p")    = p
    hp("q")    = q
    hp("spec") = spec
    Optimizer.hp("eta") = 0.5

    val f_ : Array [AFF] = Array (f_sigmoid, f_id)

    val mod = NeuralNet_3L4TS.rescale (xe, y, hh, f = f_(0), f1 = f_(1))    // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")

//  val (x_, y_, xx, yy) = NeuralNet_3L4TS.split_TnT (mod.getX, mod.getYY)
//  val (yp, qof) = mod.trainNtest_xx (x_, y_)(xx, yy)                    // train on (x_, y_) and test on (xx, yy)

    mod.trainNtest_xx ()()                                                // train on full and test on full

    mod.forecastAll (mod.getYy)                                           // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (y, mod.getYf)                                        // diagnose for all horizons
/*
    banner (s"Feature Selection Technique: Stepwise")
    val (cols, rSq) = mod.stepwiseSelAll (cross = false)                  // R^2, R^2 bar, sMAPE, NA
    val k = cols.size
    println (s"k = $k, n = ${mod.getX.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for NeuralNet_3L4TS with tech", lines = true)
//  println (mod.summary ())

    banner ("Feature Importance")
    println (s"tech: rSq = $rSq")
//  val imp = mod.importance (cols.toArray, rSq)
//  for (c, r) <- imp do println (s"col = $c, \t ${header(c)}, \t importance = $r")
*/

end neuralNet_3L4TSTest3

