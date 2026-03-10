
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael Cotterell
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Weighted Least Squares (WLS) Regression
 */

package scalation
package modeling

import scala.collection.mutable.IndexedSeq
import scala.math.{abs, sqrt}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import RegressionWLS._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionWLS` class supports weighted multiple linear regression.
 *  In this case, x is multi-dimensional [1, x_1, ... x_k].
 *  Weights are set to the inverse of a variable's variance, so they can compensate
 *  for such variability (heteroscedasticity).  Fit the parameter  vector b in
 *  the regression equation
 *      yy  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Weighted Least-Squares (minimizing the residuals) to fit the parameter vector
 *     b  =  fac.solve (.)
 *  The data matrix x is reweighted x = rootW * x and the response vector yy
 *  is reweighted y = rootW * yy where rootW is the square root of the weights.
 *  @see en.wikipedia.org/wiki/Least_squares#Weighted_least_squares
 *  These are then passed to Ordinary Least Squares (OLS) Regression.
 *  Five factorization techniques are provided:
 *      'QR'         // QR Factorization: slower, more stable (default)
 *      'Cholesky'   // Cholesky Factorization: faster, less stable (reasonable choice)
 *      'SVD'        // Singular Value Decomposition: slowest, most robust
 *      'LU'         // LU Factorization: better than Inverse
 *      'Inverse'    // Inverse/Gaussian Elimination, classical textbook technique
 *  @see www.markirwin.net/stat149/Lecture/Lecture3.pdf
 *  @param x       the data/input m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model)
 *  @param y       the response/output m vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param w       the weight vector (if null, compute in companion object)
 *  @param hparam  the hyper-parameters (defaults to Regression.hp)
 */
class RegressionWLS (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                     private var w: VectorD = null, hparam: HyperParameter = Regression.hp)
      extends Regression ({ setWeights (x, y, w); reweightX (x, w) },
                            reweightY (y, w), fname_, hparam):

    private val debug = debugf ("RegressionWLS", true)                       // debug function
    private val flaw  = flawf ("RegressionWLS")                              // flaw function

    _modelName = "RegressionWLS"

    if w == null then w = RegressionWLS.weights                              // adjust weights

    debug ("init", s"w = $w")
    if x != getX then println ("x has been reweighted")
    if y != getY then println ("y has been reweighted")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the weight vector.
     */
    def weights: VectorD = w

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute diagostics for the regression model.  Must override due to weights.
     *  @param y  the actual response vector to use (test/full)
     *  @param yp_  the predicted response vector (test/full), currently ignored
     *  @param w_   the weights on the instances
     *
    override def diagnose (y: VectorD, yp_ : VectorD, w_ : VectorD = w): VectorD =
        val yp = x * b
        super.diagnose (y, yp, w_)
    end diagnose
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_1 , ... x_k] + e
     *  using the ordinary least squares OLS method.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    override def train (x_ : MatrixD = getX, y_ : VectorD = getY): Unit = super.train (x_, y_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    override def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                                // make predictions
        (yp, diagnose (y_, yp))                                              // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output vector (defaults to full y)
     */
    override def trainNtest (x_ : MatrixD = getX, y_ : VectorD = getY)
                            (xx: MatrixD = x, yy: VectorD = y): (VectorD, VectorD) =
        train (x_, y_)
        debug ("trainNTest", s"b = $b")
        val (yp, qof) = test (xx, yy)
        println (report (qof))
        Predictor.plotPrediction (yy, yp, modelName)
        (yp, qof)
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use validation to compute test Quality of Fit (QoF) measures by dividing
     *  the full dataset into a TESTING-set and a TRAINING-set.
     *  The testing-set is defined by idx and the rest of the data is the training-set.
     *  @see `modeling.Predictor.validate` about the RANDOM, FIRST, and LAST options
     *  for selecting the testing-set.
     *  @param rando  flag indicating whether to use randomized or simple validation
     *  @param ratio  the ratio of the TESTING-set to the full dataset (most common 70-30 (.3), 80-20 (.2))
     *  @param idx    the prescribed TESTING-set indices (default => generate)
     */
    override def validate (rando: Boolean = true, ratio: Double = Model.TE_RATIO)
//                        (idx: IndexedSeq [Int] = testIndices ((ratio * y.dim).toInt, rando)):
                          (idx: IndexedSeq [Int] = testIndices (y.dim, (ratio * y.dim).toInt, rando)):
                          (VectorD, VectorD) =
        val (x_e, x_) = (x(idx), getX.not(idx))                              // test, training data/input matrices
        val (y_e, y_) = (y(idx), getY.not(idx))                              // test, training response/output vectors

        train (x_, y_)                                                       // train model on the TRAINING-set
        val (yp, qof) = test (x_e, y_e)                                      // test on TESTING-set and get QoF measures
        if qof(QoF.sst.ordinal) <= 0.0 then                                  // requires variation in TESTING-set
            flaw ("validate", "chosen testing set has no variability")
        println (FitM.fitMap (qof, QoF.values.map (_.toString)))
        (yp, qof)
    end validate

end RegressionWLS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionWLS` companion object provides methods for setting weights
 *  and testing.
 */
object RegressionWLS:

    private val TEST_MODE      = false                                 // all weights are one
    private var w:     VectorD = null                                  // weights for current model
    private var rootW: VectorD = null                                  // square root of weights

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionWLS` object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param ww      the weight vector (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               ww: VectorD = null, hparam: HyperParameter = Regression.hp)
              (col: Int = xy.dim2 - 1): RegressionWLS =
        new RegressionWLS (xy.not(?, col), xy(?, col), fname, ww, hparam)
    end apply

    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 ww: VectorD = null, hparam: HyperParameter = Regression.hp): RegressionWLS = ???

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the weight vector for the current model.
     */
    def weights: VectorD = w

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Root Absolute Deviations (RAD's) for each instance.
     *  @param x   the input/data m-by-n matrix
     *  @param y   the response/output m-vector
     */
    def rad (x: MatrixD, y: VectorD): VectorD =
        val ols_y = new Regression (x, y)                              // run OLS on original data
        ols_y.train ()                                                 // train the model on y
        val e = y - ols_y.predict (x)                                  // deviation/error vector
        e.map ((ei) => sqrt (abs (ei)))                                // root absolute deviations (RAD's)
    end rad

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate weights for the variables according to the reciprocal actual
     *  rad's.  Save the weight vector w and root weight vector rootW
     *  for the current model in companion object variables.
     *  @see setWeights that used predicted rad
     *  @param x   the input/data m-by-n matrix
     *  @param y   the response/output m-vector
     *  @param w0  the initial weight vector (if null, compute it)
     */
    def setWeights0 (x: MatrixD, y: VectorD, w0: VectorD = null): Unit =
        w = if w0 == null then rad (x, y).recip                        // set weight vector for WLS to reciprocal of RAD
            else w0                                                    // set weights using custom values
        rootW = w.sqrt                                                 // set root of weight vector
    end setWeights0

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Estimate weights for the variables according to the reciprocal predicted
     *  rad's.  Save the weight vector w and root weight vector rootW
     *  for the current model in companion object variables.
     *  @param x   the input/data m-by-n matrix
     *  @param y   the response/output m-vector
     *  @param w0  the initial weight vector (if null, compute it)
     */
    def setWeights (x: MatrixD, y: VectorD, w0: VectorD = null): Unit =
        if TEST_MODE then
            w = VectorD.one (y.dim)
        else if w0 == null then
            val r = rad (x, y)                                         // compute RAD
            val ols_r = new Regression (x, r)                          // run OLS on rad
            ols_r.train ()                                             // train the model on r
            val rp = ols_r.predict (x)                                 // predicted RAD
            w      = rp.recip                                          // set weight vector for WLS to reciprocal of rp
        else
            w = w0                                                     // set weights using custom values
        rootW = w.sqrt                                                 // set root of weight vector
    end setWeights

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reweight the data matrix x by multiplying by the root weight rtW.
     *  @param x   the input/data m-by-n matrix
     *  @param rW  the root weight vector (rtW: either rootW or rW)
     */
    def reweightX (x: MatrixD, rW: VectorD): MatrixD =
        if TEST_MODE then return x
        val rtW = if rW == null then rootW else rW                     // root weight vector
        rtW *~: x                                                      // vector (as if diagonal matrix) * matrix (right associative)
    end reweightX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reweight the response vector matrix y by multiplying by the root weight rtW.
     *  @param y   the response vector
     *  @param rW  the root weight vector (rtW: either rootW or rW)
     */
    def reweightY (y: VectorD, rW: VectorD): VectorD =
        if TEST_MODE then return y
        val rtW = if rW == null then rootW else rW                     // root weight vector
        rtW * y                                                        // vector * vector (element by element)
    end reweightY
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the given regression problem using WLS.
     *  @param x  the data matrix
     *  @param y  the response vector
     *  @param z  a vector to predict
     */
    def test (x: MatrixD, y: VectorD, z: VectorD): Unit =
        banner ("Fit the parameter vector b")
        val mod = new RegressionWLS (x, y)
        mod.trainNtest ()()                                            // train and test the model
        println (mod.summary ())                                       // parameter/coeffient statistics

        val yp = mod.predict (z)                                       // predict y for one point
        println (s"predict ($z) = $yp")

        val yyp = mod.predict (x)                                      // predict y for several points
        println (s"predict ($x) = $yyp")

//      new Plot (y, yyp, null)
    end test

end RegressionWLS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionWLSTest` main function tests `RegressionWLS` class using the following
 *  regression equation.
 *      y  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  > runMain scalation.modeling.regressionWLSTest
 */
@main def regressionWLSTest (): Unit =

    // 5 data points: constant term, x_1 coordinate, x_2 coordinate
    val x = MatrixD ((5, 3), 1.0, 36.0,  66.0,                         // 5-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0,  1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)
    val z = VectorD (1.0, 20.0, 80.0)
    
//  println ("model: y = b_0 + b_1*x_1 + b_2*x_2")
    println ("model: y = b₀ + b₁*x₁ + b₂*x₂")
    println (s"x = $x")
    println (s"y = $y")

    banner ("weights set internally")
    test (x, y, z)                                                     // weights set internally

//  val w0 = VectorD (0.106085, 0.0997944, 0.0831033, 0.160486, 0.171810)
//  val w0 = VectorD (0.318254, 0.299383,  0.249310,  0.481457, 0.515431)

    banner ("custom weights")
//  test (x, y, z, w0)                                                 // custom weights explicitly given - FIX: w0 not used yet
    test (x, y, z)                                                     // custom weights explicitly given

end regressionWLSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionWLSTest2` main function tests the `RegressionWLS` class using the AutoMPG
 *  dataset.  Assumes no missing values.  It test cross validation.
 *  > runMain scalation.modeling.regressionWLSTest2
 */
@main def regressionWLSTest2 (): Unit =

    import Example_AutoMPG._

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG Regression")
    val reg = new Regression (ox, y, ox_fname)                         // create model with intercept (else pass x)
    reg.trainNtest ()()                                                // train and test the model
    println (reg.summary ())                                           // parameter/coefficient statistics

    banner ("AutoMPG Regression WLS")
    val mod = new RegressionWLS (ox, y, ox_fname)                      // create model with intercept (else pass x)
    mod.trainNtest ()()                                                // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics

    banner ("AutoMPG Validation Test")
    mod.validate ()()

    banner ("AutoMPG Cross-Validation Test")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end regressionWLSTest2

