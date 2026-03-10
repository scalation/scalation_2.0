
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jan  5 14:03:36 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Simpler Linear Regression (one variable, one parameter)
 */

package scalation
package modeling

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimplerRegression` class supports simpler linear regression.
 *  In this case,  the vector x consists of a single variable x0.
 *  Fit the parameter vector b in the regression equation
 *      y  =  b dot x + e  =  [b0] dot [x0] + e  =  b0 * x0 + e
 *  where 'e' represents the residuals (the part not explained by the model).
 *  The simpler regression model has no intercept parameter, only a slope parameter.
 *  @see `SimpleRegression` for both intercept and slope parameters
 *  @param x       the data/input matrix (only use the first column)
 *  @param y       the response/output vector
 *  @param fname_  the feature/variable names (only use the first name)(defaults to null)
 */
class SimplerRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null)
      extends Predictor (x, y, if fname_ == null then null else fname_.slice (0, 1), null)
         with Fit (dfr = 1, df = x.dim - 1)
         with NoSubModels:

    _modelName = "SimplerRegression"

    override def getBest: BestStep = super [NoSubModels].getBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  simpler regression equation
     *      y = b dot x + e  = b0 * x0 + e
     *  using the least squares method.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        val x0  = x_(?, 0)                                         // get column 0 of x = [x0]
        val ssx = x0 dot x0                                        // sum of squares x0
        val sxy = x0 dot y_                                        // sum of cross products x0, y

        b = new VectorD (1)                                        // parameter vector [b0]
        b(0) = sxy / ssx                                           // slope
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults for full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                      // y predicted for (test/full)
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

end SimplerRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimplerRegression` companion object provides a simple factory method
 *  for building simple regression linear regression models.
 */
object SimplerRegression:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Simpler Linear Regression model from a combined data matrix.
     *  Take the first column for the predictor and the last column for the response.
     *  @param xy      the combined data matrix
     *  @param fname_  the feature/variable names (defaults to null)
     */
    def apply (xy: MatrixD, fname: Array [String] = null): SimplerRegression =
        new SimplerRegression (xy(?, 0 to 1), xy(?, xy.dim2-1), fname)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Simpler Linear Regression model, automatically creating a
     *  a data/input matrix from the vector x.
     *  @param x       the data/input m-by-1 vector
     *  @param y       the response/output m-vector
     *  @param fname_  the feature/variable names
     */
    def apply (x: VectorD, y: VectorD, fname: Array [String]): SimplerRegression =
        new SimplerRegression (MatrixD (x).ᵀ, y, fname)
    end apply

end SimplerRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simplerRegressionTest` main function is used to test the `SimplerRegression` class.
 *      y = b0 * x + e
 *  > runMain scalation.modeling.simplerRegressionTest
 */
@main def simplerRegressionTest (): Unit =

    // 4 data points:       x0
    val x = MatrixD ((4, 1), 1,                                    // 4-by-1 matrix
                             2,
                             3,
                             4)
    val y = VectorD (1, 3, 3, 4)

    println (s"x = $x")
    println (s"y = $y")

    val mod = new SimplerRegression (x, y)                         // create a SimplerRegressio
    mod.trainNtest ()()                                            // train and test the model

    val yp = mod.predict (x)                                       // y-predicted
    new Plot (x(?, 0), y, yp, lines = true)                        // black for y and red for yp

end simplerRegressionTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simplerRegressionTest2` main function is used to test the `SimplerRegression` class.
 *      y = b dot x + e = [b0] dot [x0] + e
 *  > runMain scalation.modeling.simplerRegressionTest2
 */
@main def simplerRegressionTest2 (): Unit =

    val x0  = VectorD (1, 2, 3, 4)
    val y   = VectorD (1, 3, 3, 4)
    val b0  = VectorD.range (0, 200) / 100.0
    val sse = new VectorD (b0.dim)

    for i <- b0.indices do
        val e  = y - x0 * b0(i)
        sse(i) = e dot e
    end for
    new Plot (b0, sse)                                             // plot sse vs. parameter b0

end simplerRegressionTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simplerRegressionTest3` main function is used to test the `SimplerRegression` class.
 *      y = b dot x + e = [b0] dot [x0] + e
 *  > runMain scalation.modeling.simplerRegressionTest3
 */
@main def simplerRegressionTest3 (): Unit =

    // 5 data points:       x0
    val x = MatrixD ((5, 1), 0,                                    // x 5-by-1 matrix
                             1,
                             2,
                             3,
                             4)
    val y = VectorD (2, 3, 5, 4, 6)                                // y vector

    println (s"x = $x")
    println (s"y = $y")

    val mod = new SimplerRegression (x, y)                         // create a simpler regression model
    mod.trainNtest ()()                                            // train and test the model

    val yp = mod.predict (x)
    println (s"mod.predict () = $yp")
    new Plot (x(?, 0), y, yp, lines = true)

    val z   = VectorD (5.0)                                        // predict y for one point
    val yp_ = mod.predict (z)
    println (s"predict ($z) = $yp_")

end simplerRegressionTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simplerRegressionTest4` main function is used to test the `SimplerRegression` class.
 *      y = b dot x = b0 * x0
 *  @see mathbits.com/mathbits/tisection/Statistics2/linear.htm
 *  > runMain scalation.modeling.simplerRegressionTest4
 */
@main def simplerRegressionTest4 (): Unit =

    // 20 data points: just x0 coordinate
    val x0 = VectorD (  4.0,   9.0,  10.0,  14.0,   4.0,   7.0,  12.0,  22.0,   1.0,   3.0,
                        8.0,  11.0,   5.0,   6.0,  10.0,  11.0,  16.0,  13.0,  13.0,  10.0)
    val y  = VectorD (390.0, 580.0, 650.0, 730.0, 410.0, 530.0, 600.0, 790.0, 350.0, 400.0,
                      590.0, 640.0, 450.0, 520.0, 690.0, 690.0, 770.0, 700.0, 730.0, 640.0)

    println (s"x0 = $x0")
    println (s"y  = $y")

    val mod = SimplerRegression (x0, y, null)
    mod.trainNtest ()()                                            // train and test the model

    val yp = mod.predict (mod.getX)                                // predict y for several points
    println (s"mod.predict (x0) = $yp")
    new Plot (x0, y, yp, lines = true)

    val z   = VectorD (15.0)                                       // predict y for one point
    val yp_ = mod.predict (z)
    println (s"predict ($z) = $yp_")

end simplerRegressionTest4

