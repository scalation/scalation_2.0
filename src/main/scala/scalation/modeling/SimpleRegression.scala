
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep 24 19:00:23 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Simple Linear Regression (one variable, two parameters)
 */

package scalation
package modeling


import scala.math.sqrt
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleRegression` class supports simple linear regression.  In this case,
 *  the vector x consists of the constant one and a single variable x1, i.e.,
 *  (1, x1).  Fit the parameter vector 'b' in the regression equation
 *      y  =  b dot x + e  =  [b0, b1] dot [1, x1] + e  =  b0 + b1 * x1 + e
 *  where e represents the residuals (the part not explained by the model).
 *  @param x       the data/input matrix augmented with a first column of ones
 *                 (only use the first two columns [1, x1])
 *  @param y       the response/output vector
 *  @param fname_  the feature/variable names (only use the first two names)(defaults to null)
 */
class SimpleRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null)
      extends Predictor (x, y, if fname_ == null then null else fname_.slice (0, 2), null)
         with Fit (dfr = 1, df = x.dim - 2)
         with NoSubModels:

    private val debug = debugf ("SimpleRegression", true)          // debug function
    private val flaw  = flawf ("SimpleRegression")                 // flaw function

    _modelName = s"SimpleRegression_${fname(1)}"

    if x.dim2 < 2 then flaw ("init", s"data matrix must have at least 2 columns: ${x.dim2}")

    override def getBest: BestStep = super [NoSubModels].getBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  simple regression equation
     *      y = b dot x + e  = [b0, b1] dot [1, x1] + e
     *  using the least squares method.
     *  @see www.analyzemath.com/statistics/linear_regression.html
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        val m   = x_.dim                                           // number of rows/instances
        val x1  = x_(?, 1)                                         // get column 1 of x = [1.0, x1]
        val sx  = x1.sum                                           // sum of x values
        val sy  = y_.sum                                           // sum of y values
        val ssx = x1 dot x1                                        // sum of squares x
//      val ssy = y_ dot y_                                        // sum of squares y
        val sxy = x1 dot y_                                        // sum of cross products

        b = new VectorD (2)                                        // parameter vector [b0, b1]
        b(1) = (m * sxy - sx * sy) / (m * ssx - sx*sx)             // slope
        b(0) = (sy - b(1) * sx) / m                                // intercept
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
        val yp = predict (x_)                                      // make predictions
        (yp, diagnose (y_, yp))                                    // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_0', 'x_1',
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname, b_ : VectorD = b,
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                       // summary from `Fit`
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the confidence intervals for the parameters b_0 and b_1, returning
     *  their interval half widths.
     *  @param x_  the training/full data/input matrix
     *  @param p_  the confidence level (1 - alpha)
     */
    def confInterval (x_ : MatrixD = getX, p_ : Double = .95): VectorD =
        val x1   = x_(?, 1)                                        // second column (column 1)
        val m    = x1.dim                                          // number of instances
        val df   = m - 2                                           // DoF for error e
        val s    = sqrt (sse_ / df)                                // standard deviation of e
        val s_xx = (x1 - x1.mean).normSq                           // sum of deviations squared

//      val se_0 = s * sqrt (x1.normSq / (m * s_xx))               // standard error for b_0
        val se_0 = s * sqrt (1.0 / m + x1.mean~^2 / s_xx)          // standard error for b_0
        val se_1 = s / sqrt (s_xx)                                 // standard error for b_1

        val p = 1.0 - (1.0 - p_) / 2.0                             // e.g., .95 --> .975 (two tails)
        val t = random.Quantile.studentTInv (p, df)                // critical value from Student's t distribution

        debug ("confInterval", s"s = $s, s_xx = $s_xx, se_0 = $se_0, se_1 = $se_1, t = $t")

        VectorD (t * se_0, t * se_1)
    end confInterval

end SimpleRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleRegression` companion object provides a simple factory method
 *  for building simple regression linear regression models.
 */
object SimpleRegression:

    private val flaw  = flawf ("SimpleRegression")                 // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Simple Linear Regression model from a combined data matrix.
     *  The first column of matrix xy should have all ones corresponding to the intercept
     *  (matrix from has two column vectors [1 x]).
     *  Take the first two columns for the predictor and the last column for the response.
     *  @see `SimplerRegression` for a model without an intercept parameter
     *  @param xy     the combined data matrix
     *  @param fname  the feature/variable names (defaults to null)
     */
    def apply (xy: MatrixD, fname: Array [String] = null): SimpleRegression =
        new SimpleRegression (xy(?, 0 until 2), xy(?, xy.dim2-1), fname)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Simple Linear Regression model by automatically prepending the
     *  column of ones (form matrix from two column vectors [1 x]).
     *  @param x      the data/input m-by-1 vector
     *  @param y      the response/output m-vector
     *  @param fname  the feature/variable names
     */
    def apply (x: VectorD, y: VectorD, fname: Array [String]): SimpleRegression =
        new SimpleRegression (MatrixD.one (x.dim) :^+ x, y, fname)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Simple Linear Regression model quadratic in x by automatically prepending
     *  the column of ones (form matrix from two column vectors [1 x^2]).
     *  @param x      the data/input m-by-1 vector (to be squared)
     *  @param y      the response/output m-vector
     *  @param fname  the feature/variable names (defaults to null)
     */
    def quadratic (x: VectorD, y: VectorD, fname: Array [String] = null): SimpleRegression =
        new SimpleRegression (MatrixD.one (x.dim) :^+ x~^2, y, fname)
    end quadratic

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create the Best Simple Linear Regression model using the first column of all
     *  ones and the column/variable that is the best predictor xj (matrix [1 xj]).
     *  @caveat:  assumes matrix x has a first column of all one.
     *  @param x      the m-by-n data/input matrix
     *  @param y      the response/output m-vector
     *  @param fname  the feature/variable names (defaults to null)
     */
    def best (x: MatrixD, y: VectorD, fname: Array [String] = null): SimpleRegression =
        val j = x.corr (y, 1).argmag () + 1                          // use best column
        val fname_ = Array (fname(0), fname(j))
        println (s"best: column j = $j, fname(j) = ${fname(j)}")
        new SimpleRegression (MatrixD (x(?, 0), x(?, j)).ᵀ, y, fname_)
    end best

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the `SimpleRegression` coefficients directly from the x and y vectors.
     *  @param x  the data/input m-vector
     *  @param y  the response/output m-vector
     */
    def coeff (x: VectorD, y: VectorD): VectorD =
        if x.dim != y.dim then
            flaw ("coeff", s"dimensions do not agree: x.dim = ${x.dim} != y.dim = ${y.dim}")
        val b1 = (x cov y) / x.variance
        val b0 = y.mean - b1 * x.mean
        VectorD (b0, b1)
    end coeff

end SimpleRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleRegressionTest` main function to test the `SimpleRegression` class.
 *      y = b0 + b1 * x
 *  > runMain scalation.modeling.simpleRegressionTest
 */
@main def simpleRegressionTest (): Unit =

    // 4 data points: constant x1
    val x = MatrixD ((4, 2), 1, 1,                                 // x 4-by-2 matrix
                             1, 2,
                             1, 3,
                             1, 4)
    val y = VectorD (1, 3, 3, 4)                                   // y vector

    banner ("Test1: Simple Regression Model: y = b_0 + b_1 x + e")
    println (s"x = $x")
    println (s"y = $y")

    val mod = new SimpleRegression (x, y)                          // create a simple regression model
    mod.inSample_Test ()                                           // train and test the model

    val yp = mod.predict (x)
    new Plot (x(?, 1), y, yp, "plot y and yp vs. x", lines = true)

end simpleRegressionTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleRegressionTest2` main function to test the `SimpleRegression` class.
 *      y = b0 + b1 * x
 *  > runMain scalation.modeling.simpleRegressionTest2
 */
@main def simpleRegressionTest2 (): Unit =

    // 4 data points:
    val x = VectorD (1, 2, 3, 4)
    val y = VectorD (1, 3, 3, 4)

    banner ("Test2: Simple Regression Model: y = b_0 + b_1 x + e")
    println (s"x = $x")
    println (s"y = $y")

    val mod = SimpleRegression (x, y, null)                        // automatically prepends a column of ones
    mod.inSample_Test ()                                           // train and test the model

end simpleRegressionTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleRegressionTest3` main function is used to test the `SimpleRegression` class.
 *      y = b dot x = [b0, b1] dot [1, x1]
 *  @see www.analyzemath.com/statistics/linear_regression.html
 *  > runMain scalation.modeling.simpleRegressionTest3
 */
@main def simpleRegressionTest3 (): Unit =

/*
    // 5 data points:      one x1
    val ox = MatrixD ((5, 2), 1, 0,                                // x 5-by-2 matrix
                              1, 1,
                              1, 2,
                              1, 3,
                              1, 4)
    val y = VectorD (2, 3, 5, 4, 6)                                // y vector
*/
    // 6 data points:      one x1
    val ox = MatrixD ((6, 2), 1, 1,                                // x 6-by-2 matrix
                              1, 2,
                              1, 3,
                              1, 4,
                              1, 5,
                              1, 6)
    val y = VectorD (1, 3, 4, 6, 4, 3)                             // y vector

    val x = ox(?, 1)

    banner ("Test3: Simple Regression Model: y = b_0 + b_1 x + e")
    println (s"ox = $ox")
    println (s"y  = $y")

    val mod = new SimpleRegression (ox, y)                         // create a simple regression model
    mod.inSample_Test ()                                           // train and test the model

    banner ("Low Level Calculation Check")
    val n   = y.dim
    val xm  = x.mean
    val ym  = y.mean
    val xdy = x dot y
    val xdx = x dot x
    val b1  = (xdy - n * xm * ym) / (xdx - n * xm * xm)
    val b0  = ym - b1 * xm
    println (s"xm = $xm, ym = $ym, xdy = $xdy, xdx = $xdx, b1 = $b1, b0 = $b0")

    val z = VectorD (1, 5)                                         // predict y for new point z
    println (s"predict ($z) = ${mod.predict (z)}")

    val yp = mod.predict (ox)
    new Plot (x, y, yp, "plot y and yp vs. x", lines = true)

    val qrg = SymbolicRegression.quadratic (MatrixD.fromVector (x), y)
    val yq = qrg.inSample_Test ()._1                               // train and test the model
    new Plot (x, y, yq, "plot y and yq vs. x", lines = true)

    val trg = new TranRegression (ox, y, yℱ = RootForm ())
    val yt = trg.inSample_Test ()._1                               // train and test the model
    new Plot (x, y, yt, "plot y and yt vs. x", lines = true)

end simpleRegressionTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleRegressionTest4` main function is used to test the `SimpleRegression` class.
 *      y = b dot x = b0 + b1 * x1
 *  @see mathbits.com/mathbits/tisection/Statistics2/linear.htm
 *  > runMain scalation.modeling.simpleRegressionTest4
 */
@main def simpleRegressionTest4 (): Unit =

    // 20 data points: just x1 coordinate
    val x1 = VectorD (  4.0,   9.0,  10.0,  14.0,   4.0,   7.0,  12.0,  22.0,   1.0,   3.0,
                        8.0,  11.0,   5.0,   6.0,  10.0,  11.0,  16.0,  13.0,  13.0,  10.0)
    val y  = VectorD (390.0, 580.0, 650.0, 730.0, 410.0, 530.0, 600.0, 790.0, 350.0, 400.0,
                      590.0, 640.0, 450.0, 520.0, 690.0, 690.0, 770.0, 700.0, 730.0, 640.0)

    val x = MatrixD.one (x1.dim) :^+ x1                            // form matrix x from vector x1

    banner ("Test4: Simple Regression Model: y = b_0 + b_1 x + e")
    println (s"x = $x")
    println (s"y = $y")

    val mod = new SimpleRegression (x, y)                          // create a simple regression model
    mod.inSample_Test ()                                           // train and test the model

    val z  = VectorD (1.0, 15.0)                                   // predict y for new point z
    println (s"predict ($z) = ${mod.predict (z)}")

    val yp = mod.predict (x)
    new Plot (x1, y, yp, "plot y and yp vs. x1", lines = true)

end simpleRegressionTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleRegressionTest5` main function is used to test the `SimpleRegression` class.
 *      y = b dot x = b0 + b1 * x1
 *  This version uses gradient descent to search for the optimal solution for b.
 *  > runMain scalation.modeling.simpleRegressionTest5
 */
@main def simpleRegressionTest5 (): Unit =

    // 4 data points:
    val x  = MatrixD ((4, 2), 1, 1,
                              1, 2,
                              1, 3,
                              1, 4)
    val x1 = x(?, 1)
    val y  = VectorD (1, 3, 3, 4)
    val _1 = VectorD.one (x1.dim)

    println (s"x = $x")

    val ITER = 10                                                  // number of iterations
    val eta  = 0.02                                                // try different values for the learning rate
//  val mod  = new SimpleRegression (x, y)                         // create a simple regression model, don't train
    val b    = new VectorD (x.dim2)                                // starting point [0, 0] for parameter vector b

    banner (s"Test5: Simple Regression Model: gradient descent: eta = $eta")
    for it <- 1 to ITER do
        val yp = x * b
        val e  = y - yp
        val g  = VectorD (_1 dot e, x1 dot e)
        b     += g * eta
        val sse = e dot e
        println (s"for iteration $it, b = $b, sse = $sse")
    end for

end simpleRegressionTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleRegressionTest6` main function is used to test the `SimpleRegression` class.
 *      y = b dot x = b0 + b1 * x1
 *  This version does Exploratory Data Analysis (EDA) on the AutoMPG dataset.
 *  > runMain scalation.modeling.simpleRegressionTest6
 */
@main def simpleRegressionTest6 (): Unit =

    import Example_AutoMPG.{xy, x, ox, y, x_fname, ox_fname}

    banner ("Null Model")
    val nm = new NullModel (y)
    nm.inSample_Test ()                                            // train and test the model
    val yp = nm.predict (x)
    new Plot (null, y, yp, "Null EDA: y and yp (red) vs. t", lines = true)

    banner ("Correlation Matrix for Columns of xy")
    println (s"x_fname = ${stringOf (x_fname)}")
    println (s"y_name  = MPG")
    println (s"xy.corr = ${xy.corr}")

    for j <- x.indices2 do
        banner (s"Plot response y vs. predictor variable ${x_fname(j)}")
        val xj = x(?, j)
        val mod = SimpleRegression (xj, y, Array ("one", x_fname(j)))
        mod.inSample_Test ()                                       // train and test the model
        val yp = mod.predict (mod.getX)
        new Plot (xj, y, yp, s"EDA: y and yp (red) vs. ${x_fname(j)}", lines = true)
    end for

    banner ("AutoMPG Best Simple Regression Model")
    val mod2 = SimpleRegression.best (ox, y, ox_fname)
    mod2.inSample_Test ()                                          // train and test the model

end simpleRegressionTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleRegressionTest7` main function is used to test the `SimpleRegression` class.
 *      y = b dot x = b0 + b1 * x1^2
 *  This version does Exploratory Data Analysis (EDA) on the AutoMPG dataset looking for
 *  quadratic patterns.
 *  > runMain scalation.modeling.simpleRegressionTest7
 */
@main def simpleRegressionTest7 (): Unit =

    import Example_AutoMPG.{xy, x, y, x_fname}

    banner ("Null Model")
    val nm = new NullModel (y)
    nm.inSample_Test ()                                            // train and test the model
    val yp = nm.predict (x)
    new Plot (null, y, yp, "Null Model EDA: y and yp (red) vs. t", lines = true)

    banner ("Correlation Matrix for Columns of xy")
    println (s"x_fname = ${stringOf (x_fname)}")
    println (s"y_name  = MPG")
    println (s"xy.corr = ${xy.corr}")

    for j <- x.indices2 do
        banner (s"Plot response y vs. predictor variable ${x_fname(j)}^2")
        val xj = x(?, j)
        val mod = SimpleRegression.quadratic (xj, y, Array ("one", x_fname(j) + "^2"))
        mod.inSample_Test ()                                       // train and test the model
        val yp = mod.predict (mod.getX)
        new Plot (xj, y, yp, s"EDA: y and yp (red) vs. ${x_fname(j)}^2", lines = false)
    end for

end simpleRegressionTest7


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleRegressionTest8` main function tests the `SimpleRegression` class
 *  using a simple dataset and compares it with the `NullModel`.
 *  > runMain scalation.modeling.simpleRegressionTest8
 */
@main def simpleRegressionTest8 (): Unit =

    val x  = VectorD (1, 2, 3, 4, 5, 6)
//  val y  = VectorD (1, 3, 3, 5, 4, 4)
    val y  = VectorD (1, 3, 5, 6, 4, 2)
    val ox = MatrixD.one (x.dim) :^+ x

    val x_fname  = Array ("x")
    val ox_fname = Array ("one", "x")

    banner ("NullModel")
    val nm = new NullModel (y)
    nm.inSample_Test ()                                            // train and test the model
    println (s"mse = ${nm.mse_}")
    println (nm.summary ())
    new Plot (null, y, nm.predict (nm.getX), s"${nm.modelName} y vs yp", lines = true)

    banner ("SimplerRegression")
    val reg = new SimplerRegression (MatrixD (x).ᵀ, y, x_fname)
    reg.inSample_Test ()                                           // train and test the model
    println (s"mse = ${reg.mse_}")
    println (reg.summary ())
    new Plot (null, y, reg.predict (reg.getX), s"${reg.modelName} y vs yp", lines = true)

    banner ("SimpleRegression")
    val mod = new SimpleRegression (ox, y, ox_fname)
    mod.inSample_Test ()                                           // train and test the model
    println (s"mse   = ${mod.mse_}")
    println (s"confI = ${mod.confInterval ()}")
    println (mod.summary ())
    new Plot (null, y, mod.predict (mod.getX), s"${mod.modelName} y vs yp", lines = true)

    banner ("SymbolicRegression.quadratic")
    val qrg = SymbolicRegression.quadratic (MatrixD (x).ᵀ, y, x_fname)
    qrg.inSample_Test ()                                           // train and test the model
    println (s"mse   = ${qrg.mse_}")
//  println (s"confI = ${qrg.confInterval ()}")
    println (qrg.summary ())
    new Plot (null, y, qrg.predict (qrg.getX), s"${qrg.modelName} y vs yp", lines = true)

    banner ("TranRegression")
    val trg = new TranRegression (ox, y, ox_fname, yℱ = RootForm ())
    trg.inSample_Test ()                                           // train and test the model
    println (s"mse   = ${trg.mse_}")
//  println (s"confI = ${trg.confInterval ()}")
    println (trg.summary ())
    new Plot (null, y, trg.predict (trg.getX), s"${trg.modelName} y vs yp", lines = true)

end simpleRegressionTest8

