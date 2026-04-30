
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Dec 24 13:59:45 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Quantile Regression
 */

package scalation
package modeling

import scala.math.{abs, max}
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the Pin Ball Loss based on the difference between y and yp = e
 *  @param e  the error/residual vector
 *  @param q  the quantile of interest
 */
def pinball (e: VectorD, q: Double = 0.5): Double =
    var sum = 0.0
    cfor (0, e.dim) { i =>
        val e_i = e(i)
        sum += (if e_i >= 0.0 then q * abs (e_i) else (1.0 - q) * abs (e_i))
    } // cfor
    sum
end pinball

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the Pin Ball Loss based on the difference between y and yp.
 *  @param x  the data/input m-by-n matrix
 *  @param y  the actual response/target variable m-vector
 *  @param q  the quantile of interest
 *  @param b  the parameter vector
 */
def pinball (x: MatrixD, y: VectorD, q: Double)(b: VectorD): Double =
    val yp  = x * b
    val e   = y - yp
    pinball (e, q)
end pinball


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuantileReg` class supports quantile regression.
 *  @param x       the data/input m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model)
 *  @param y       the response/output m-vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters (defaults to QuantileReg.hp)
 */
class QuantileReg (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                          hparam: HyperParameter = QuantileReg.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`

    private val debug   = debugf ("QuantileReg", true)              // debug function
    private val flaw    = flawf ("QuantileReg")                     // flaw function
    private val n       = x.dim2                                    // number of columns
    private val q       = hparam("q").toDouble                      // the quantile sought
    private val maxIter = hparam("maxIter").toInt                   // maximum number of iterations

    _modelName = s"QuantileReg_$dfr"

    if n < 1 then flaw ("init", s"dim2 = $n of the 'x' matrix must be at least 1")
    if q out (0.01, 0.99) then flaw ("init", s"quantile q = $q must be in [.01, .99]")

    debug ("init", s"_modelName with q = $q")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_1 , ... x_k] + e
     *  using the Iteratively Reweighted Least Squares 'IRLS' method.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit =
        b = VectorD.one (x.dim2)                                    // initial guess (e.g., all ones or OLS)
        val m   = x.dim
        val eps = 1e-8

        breakable {
            cfor (0, maxIter) { _ =>
                val r = y - x * b                                   // calculate residuals
        
                // Calculate weight vector (w) based on asymmetric pinball loss
                val w = new VectorD (m)
                cfor (0, m) { i =>
                    val weight = if r(i) >= 0 then q else 1.0 - q
                    w(i) = weight / (max (abs (r(i)), eps))
                } // cfor

                // Solve using ScalaTion's Weighted Regression (WLS)
                val wr = new RegressionWLS (x_, y_, fname, w)
                wr.train ()
        
                val b_new = wr.parameter
                if (b_new - b).norm < 1e-6 then break ()            // convergence check
                b = b_new
            } // cfor
        } // breakable
        b
    end train
 
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD, y_ : VectorD): (VectorD, VectorD) =
        val yp = predict (x_)                                            // make predictions
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of vector y = f(x_, b).  It is overridden for speed.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    override def predict (x_ : MatrixD): VectorD = x_ * b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String]): Predictor & Fit = ???

end QuantileReg


object QuantileReg:

    /** Base hyper-parameter specification for `Regression`
     */
    val hp = new HyperParameter
    hp += ("q", 0.5, 0.5)
    hp += ("maxIter", 50, 50)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `QuantileReg` object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): QuantileReg =
        new QuantileReg (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create two `QuantileReg` objects from a combined data-response matrix, at
     *  lower and upper quantiles to enable prediction intervals.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param α       the significance level (e.g., .1 => 2 .05 tails: .05 [ .9 ] .05
     *  @param hparam  the hyper-parameters (defaults to hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def predInterval (xy: MatrixD, fname: Array [String] = null, α: Double = 0.1,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): (QuantileReg, QuantileReg) =
        hp("q") = α / 2
        val mod1 = new QuantileReg (xy.not(?, col), xy(?, col), fname, hparam)
        hp("q") = 1 -  α / 2
        val mod2 = new QuantileReg (xy.not(?, col), xy(?, col), fname, hparam)
        (mod1, mod2)
    end predInterval

end QuantileReg


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `quantileRegTest` main function tests the `QuantileReg` class.
 *  It tests the Pin Ball Loss Function.
 *  > runMain scalation.modeling.quantileRegTest
 */
@main def quantileRegTest (): Unit =

    val y    = VectorD.range (0, 20)
    val yp   = VectorD.fill (y.dim)(y.mean)
    val loss = pinball (y - yp)
    println (s"pinball = $loss")

end quantileRegTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `quantileRegTest2` main function tests a quantile regression model on a small
 *  dataset of temperatures from counties in Texas where the variables/factors to consider
 *  are Latitude (x1), Elevation (x2) and Longitude (x3).  The model equation
 *  is the following:
 *      y  =  b dot x  =  b0 + b1*x1 + b2*x2 + b3*x3
 *  It compare POINT PREDICTIONS of `Regression` (mean) and `QuantileReg` (median).
 *  > runMain scalation.modeling.quantileRegTest2
 */
@main def quantileRegTest2 (): Unit =

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

    banner ("Texas Temperatures Regression")
    val mod = Regression (xy)()                               // create `Regression` model
    mod.inSample_Test ()                                      // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("Texas Temperatures Quantile (Median) Regression")
    val qmod = QuantileReg (xy)()                             // create `QuantileReg` model
    qmod.inSample_Test ()                                     // train and test the model
//  println (qmod.summary ())                                 // parameter/coefficient statistics

end quantileRegTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `quantileRegTest3` main function tests a quantile regression model on a small
 *  dataset of temperatures from counties in Texas where the variables/factors to consider
 *  are Latitude (x1), Elevation (x2) and Longitude (x3).  The model equation
 *  is the following:
 *      y  =  b dot x  =  b0 + b1*x1 + b2*x2 + b3*x3
 *  It compare PREDICTION INTERVALS from `Regression` and `QuantileReg`.
 *  > runMain scalation.modeling.quantileRegTest3
 */
@main def quantileRegTest3 (): Unit =

    import MatrixD.at

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

    val x = xy.not(?, 4)
    val y = xy(?, 4)

    banner ("Texas Temperatures Regression")
    val mod = Regression (xy)()                                  // create Regression model with intercept (else pass x)
    val (yp, _) = mod.inSample_Test ()                           // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    var mName = mod.modelName

    // PREDICTION INTERVAL assuming Gaussian errors and using predictInt from `Fit`

    banner ("Texas Temperatures Prediction Intervals")
    val l_u           = mod.PIbounds (yp, mod.predictInt_ (x))   // make PI lower and upper bound vectors from yp and ihw
    val (qof_all, iα) = mod.diagnose_pi (y, yp, l_u)             // compute metrics for both point and interval predictions
    mod.showQoF (qof_all)                                        // show all the QoF metrics
    Predictor.plotPredictionInt (y, yp, at (l_u, iα), mName)     // plot ordered actual, predicted, lower, upper

    // PREDICTION INTERVAL using Split Conformal Predictions (SCP) `predictCInt` from `Predictor`

    banner ("Texas Temperatures Conformal Prediction Intervals")
    var l_u_     = mod.PIbounds (yp, mod.predictCInt (x, y))     // make PI lower and upper bound vectors from yp and ihw
    var qof_all_ = mod.diagnose_ (y, yp, l_u_)                   // compute metrics for both point and interval predictions
    mod.showQoF (qof_all_)                                       // show all the QoF metrics
    Predictor.plotPredictionInt (y, yp, l_u_, mName)             // plot ordered actual, predicted, lower, upper

    // PREDICTION INTERVAL using Quantile Regression

    banner ("Texas Temperatures Quantile (PI) Regression")
    val mods = QuantileReg.predInterval (xy)()                   // create two QuantileReg models for prediction intervals
    val (yp1, _) = mods._1.trainNtest ()()                       // train and test the model (low)
    val (yp2, _) = mods._2.trainNtest ()()                       // train and test the model (up)
//  println (mods.summary ())                                    // parameter/coefficient statistics

    mName = mods._1.modelName
    l_u_     = (yp1, yp2)
    qof_all_ = mods._1.diagnose_ (y, yp, l_u_)                   // compute metrics for both point and interval predictions
    mod.showQoF (qof_all_)                                       // show all the QoF metrics
    Predictor.plotPredictionInt (y, yp, l_u_, mName)             // plot ordered actual, predicted, lower, upper
    new PlotM (null, MatrixD (y, yp1, yp2), Array ("y", "yp1 (low)", "yp2 (up)"),
               "Plot Quantile (PI) Regression", lines = true)

end quantileRegTest3

