
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael E. Cotterell
 *  @version 2.0
 *  @date    Sun Jan 11 19:05:20 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Simple Exponential Regression
 */

package scalation
package modeling

// FIX -- option to include intercept and improve model accuracy

import scala.math.exp

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleExpRegression` class supports exponential regression.  In this case,
 *  x is [1, x_1].  Fit the parameter vector b in the exponential regression equation
 *      log (mu (x))  =  b dot x  =  b_0 + b_1 * x_1
 *  @see www.stat.uni-muenchen.de/~leiten/Lehre/Material/GLM_0708/chapterGLM.pdf 
 *  @param x       the data/input matrix
 *  @param y       the response/output vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters (currently none)
 *  @param nonneg  whether to check that responses are nonnegative (defaults to true)
 */
class SimpleExpRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                           hparam: HyperParameter = null, nonneg: Boolean = true)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = 1, df = x.dim - 2)
         with NoSubModels:

    private val debug     = debugf ("SimpleExpRegression", true)         // debug function
    private val flaw      = flawf ("SimpleExpRegression")                // flaw function
    private val cutoff    = 1E-5                                         // cutoff threshold
    private val eta       = 0.00005                                      // the learning/convergence rate (requires adjustment)
    private val maxEpochs = 1000                                         // the maximum number of training epcochs/iterations
//  private val eta       = hparam("eta")                                // the learning/convergence rate (requires adjustment)
//  private val maxEpochs = hparam("maxEpochs").toInt                    // the maximum number of training epcochs/iterations

    _modelName = "SimpleExpRegression"

    if nonneg && ! y.isNonnegative then flaw ("init", "response vector y must be nonnegative")

    override def getBest: BestStep = super [NoSubModels].getBest

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The estimated response value at point xi.
     *  @param xi  the point to evaluate
     *  @param b   the given parameter values
     */
    def f (xi: Double, b: VectorD): Double = b(0) * exp (b(1) * xi)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  simple exponential regression equation.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        val xx = x_(?, 0)
//      b = VectorD (50.0, -0.03)                                        // FIX - generalize initial guess
        b = VectorD (1.0, 0.3)

        var (go, epoch) = (true, 1)
        cfor (go && epoch <= maxEpochs, epoch += 1) {                    // iterate over the learning epochs
            val yp = VectorD (for i <- xx.indices yield f (xx(i), b))    // y-predicted vector
//          val yp = xx.map (f (_, b))                                   // y-predicted vector
            val e  = y_ - yp                                             // error vector
            val g0 = (e dot yp) / b(0)                                   // gradient 0 - partial of loss w.r.t. b0
            val g1 = e dot yp * xx                                       // gradient 1 - partial of loss w.r.t. b1
            b(0)  += eta/x_.dim * g0                                     // update to first parameter
            b(1)  += eta/x_.dim * g1                                     // update to second parameter
            debug ("train", s"parameters for $epoch th epoch: b = $b \n yp = $yp")
            if VectorD (g0, g1).norm < cutoff then go = false            // stopping rule, may refine
        } // cfor
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                            // make predictions
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
    end test

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = exp (b dot z),
     *  e.g., exp (b_0, b_1) dot (1, z_1).
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): Double = f (z(0), b)

end SimpleExpRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleExpRegression` companion object provides factory methods for creating
 *  simple exponential regression models.
 */
object SimpleExpRegression:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `SimpleExpRegression` object from a combined data-response matrix.
     *  The last column is assumed to be the response column.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (currently has none)
     *  @param nonneg  whether to check that responses are nonnegative (defaults to true)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = null, nonneg: Boolean = true)
               (col: Int = xy.dim2 - 1): SimpleExpRegression =
        new SimpleExpRegression (xy.not(?, col), xy(?, col), fname, hparam, nonneg)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `SimpleExpRegression` object from a data matrix and a response vector.
     *  This mathod provides data rescaling.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (default to null)
     *  @param hparam  the hyper-parameters (currently has none)
     *  @param nonneg  whether to check that responses are nonnegative (defualts to true)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = null, nonneg: Boolean = true): SimpleExpRegression =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new SimpleExpRegression (xn, y, fname, hparam, nonneg)
    end rescale

end SimpleExpRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleExpRegressionTest` main function tests `SimpleExpRegression` class
 *  using the following exponential regression problem.
 *  > runMain scalation.modeling.simpleExpRegressionTest
 */
@main def simpleExpRegressionTest (): Unit =

/*
    val x = MatrixD ((5, 3), 1.0, 36.0,  66.0,                           // 5-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0,  1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)
    val z = VectorD (1.0, 20.0, 80.0)
*/

    val x = MatrixD ((10, 2), 1.0, 1.0,
                              1.0, 2.0,
                              1.0, 3.0,
                              1.0, 4.0,
                              1.0, 5.0,
                              1.0, 6.0,
                              1.0, 7.0,
                              1.0, 8.0,
                              1.0, 9.0,
                              1.0, 10.0)
  
    val y = VectorD (2.70,  // approximate 2 * exp(0.3*1)
                     3.64,  // approximate 2 * exp(0.3*2)
                     4.93,
                     6.64,
                     8.96,
                     12.10,
                     16.33,
                     22.05,
                     29.76,
                     40.17)

    val x_ = x(?, 1 until x.dim2)
    println (s"x_ = $x_")

    val mod = new SimpleExpRegression (x_, y)                            // create a model
//  val mod = new SimpleExpRegression (x, y)                             // create a model -- with intercept
    mod.trainNtest ()()                                                  // train and test the model

    val yp = mod.predict (x_)
    println (s"y  = $y")
    println (s"yp = $yp")

    new Plot (null, y, yp, "SimpleExpRegressionTest", lines = true)

/*
    val yp2 = mod.predict (z)
    println (s"predict ($z) = $yp2")
*/

end simpleExpRegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleExpRegressionTest2` main function tests the `SimpleExpRegression` class.
 *  > runMain scalation.modeling.simpleExpRegressionTest2
 */
@main def simpleExpRegressionTest2 (): Unit =

    import scalation.random.{Uniform, Exponential, Random}

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test `SimpleExpRegression` by simulating n-many observations.
     *  @param n  number of observations (e.g., 10000)
     *  @param k  number of variables (e.g., 5)
     */
    def test (n: Int, k: Int): Unit =
        val u = new Uniform (0, 1)                                       // uniform random
        val e = new Exponential (1)                                      // exponential error
        val r = new Random ()

        val x = new MatrixD (n, k)                                       // data matrix
        val y = new VectorD (x.dim)                                      // response vector
        val b = new VectorD (k)                                          // known coefficients

        for i <- b.indices do b(i) = 1 + r.gen * 6

        for i <- x.indices; j <- x.indices2 do x(i, j) = u.gen

        for i <- y.indices do y(i) = exp (x(i) dot b) * e.gen

        banner ("Simulated exp regression problem with $k vars")
        val mod = new SimpleExpRegression (x, y)                         // create a model
        mod.trainNtest ()()                                              // train and test the model
    end test

    for k <- 1 to 10 do test (1000, k)

end simpleExpRegressionTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleExpRegressionTest3` main function tests the `SimpleExpRegression` class.
 *  @see www.cnachtsheim-text.csom.umn.edu/kut86916_ch13.pdf
 *      y = 58.6065 exp (-.03959 x) + e
 *  > runMain scalation.modeling.simpleExpRegressionTest3
 */
@main def simpleExpRegressionTest3 (): Unit =

    val xy = MatrixD ((15, 2), 2, 54,
                               5, 50,
                               7, 45,
                              10, 37,
                              14, 35,
                              19, 25,
                              26, 20,
                              31, 16,
                              34, 18,
                              38, 13,
                              45, 8,
                              52, 11,
                              53, 8,
                              60, 4,
                              65, 6)

    println (s"xy = $xy")

    val mod = SimpleExpRegression (xy)()                                 // create a model
    mod.trainNtest ()()                                                  // train and test the model

    val y  = xy(?, 1)                                                    // vector column 1
    val yp = mod.predict (xy.not(?, 1))                                  // matrix excluding column 1
    println ("y  = " + y)
    println ("yp = " + yp)

    new Plot (null, y, yp, "SimpleExpRegressionTest")

end simpleExpRegressionTest3

