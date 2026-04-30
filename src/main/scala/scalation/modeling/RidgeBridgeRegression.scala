
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Thu Jul 24 11:23:31 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Multiple Linear Regression with Ridge-Bridge Regularization
 *
 *  Before calling the constructor, users should center their data; automatic by all factory methods.
 */

package scalation
package modeling

import scala.math.abs

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeBridgeRegression` class supports multiple linear regression with a hybrid
 *  of Ridge (L2) and Bridge (Lq with 0 < q < 1) regularization. It solves:
 *      y = Xb + e
 *  by minimizing:
 *      ||y - Xb||^2 + lambda * ||b||^2 + beta * sum(|b_j|^q)
 *  @param x       the centered data/input matrix
 *  @param y       the centered response/output vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the regularization hyper-parameters (lambda for ridge, beta for bridge, q)
 *  @param xℱ      the transformation applied to x (e.g., Center or Norm)
 *  @param yℱ      the transformation applied to y (e.g., Center)
 */
class RidgeBridgeRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                             hparam: HyperParameter = RidgeRegression.hp,
                             xℱ: Transform = null, yℱ: Transform = null)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2, df = x.dim - x.dim2 - 1):
         // degrees of freedom: dfr = n, df = m - n - 1 as centered x matrix has 1 less column
         // fix after training by moving a dof from error to model for each coefficient eliminated
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`

    private val debug   = debugf ("RidgeBridgeRegression", false)
    private val lambda  = hparam("lambda").toDouble                      // the L_2 shrinkage parameter
    private val beta    = hparam("beta").toDouble                        // the L_q shrinkage parameter
    private val sparse  = hparam("sparse").toInt == 1                    // whether to sparsify
    private val maxIter = hparam("maxIter").toInt                        // maximum number of iterations for IRR
    private val tol     = hparam("tol").toDouble                         // tolerance for convergence
    private val eps     = hparam("eps").toDouble                         // small constant to avoid division by zero
    private val maxW    = 1E6                                            // maximum weight
    private val q       = hparam("pow").toDouble                         // exponent/L_q norm
    private val q_2     = q - 2.0

    _modelName = s"RidgeBridgeRegression_$q"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the model using Iterative Reweighted RidgeRegression (IRR).
     *  @param x_  the input/data matrix
     *  @param y_  the output/response vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        val xtX = x_.ᵀ * x_
        val xty = x_.ᵀ * y_
        val n   = x_.dim2
        val w   = new MatrixD (n, n)                                     // diagonal weight matrix

        val ridgeMod = RidgeRegression.center (x_, y_, fname_, hparam)
        ridgeMod.trainNtest ()()
        b = ridgeMod.parameter

        var (iter, diff) = (0, Double.MaxValue)
        while iter < maxIter && diff > tol do
            cfor (0, n) { j =>
                val wj  = if abs(b(j)) > eps then (q / 2.0) * abs(b(j)) ~^ q_2 else maxW
                w(j, j) = beta * wj + lambda
            } // cfor

            val fac   = Fac_Cholesky (xtX + w).factor ()
            val b_new = fac.solve (xty)

            diff  = (b_new - b).norm
            b     = b_new
            iter += 1
        end while

        if sparse then LassoRegression.sparsify (b)
        debug ("train", s"IRR estimates parameter b = $b")
        val nz = b.countZero                                             // count number of coefficients set to zero
        if nz > 0 then resetDF (x.dim2 - nz, x.dim - x.dim2 - 1 + nz)
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
        val yp = predict_ (x_)                                  // make predictions
        (yp, diagnose (y_, yp))                                 // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z.
     *  It works on transformed values.
     *  @param z  the new vector to predict
     */
    def predict_ (z: VectorD): Double = b dot z

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of vector y = f(x_, b).  It works on transformed values.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    def predict_ (x_ : MatrixD): VectorD = x_ * b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z.
     *  It is overridden to handle transformations.
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): Double =
        val zz = if xℱ == null then z else xℱ.f(MatrixD (z))(0)
        if yℱ == null then b dot zz else yℱ.fi_(b dot zz)
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of vector y = f(x_, b).  It is overridden to handle transformations.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    override def predict (x_ : MatrixD): VectorD =
        val xx = if xℱ == null then x_ else xℱ.f(x_)
        if yℱ == null then xx * b else yℱ.fi(xx * b)
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname, b_ : VectorD = b,
                         vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                    // summary from `Fit`
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    override def buildModel (x_cols: MatrixD, fname2: Array [String] = null): RidgeBridgeRegression =
        new RidgeBridgeRegression (x_cols, y, fname2, hparam)
    end buildModel

end RidgeBridgeRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeBridgeRegression` companion object provides default hyper-parameters
 *  and convenience factory methods.
 */
object RidgeBridgeRegression extends Regularized:

    val hp = RidgeRegression.hp

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Ridge-Bridge Regression from a combined xy matrix.
     *  @param xy      the centered combines x and y matrix
     *  @param fname_  the feature/variable names (defaults to null)
     *  @param hparam  the regularization hyper-parameters (lambda for ridge, beta for bridge, q)
     *  @param col     the column used for response variable
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): RidgeBridgeRegression =
        val (x, y) = (xy.not(?, col), xy(?, col))
        val xℱ = CenterForm (x)
        val yℱ = CenterForm (y)
        new RidgeBridgeRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Ridge-Bridge Regression from an x matrix and y vector.
     *  @param x       the centered data/input matrix
     *  @param y       the centered response/output vector
     *  @param fname_  the feature/variable names (defaults to null)
     *  @param hparam  the regularization hyper-parameters (lambda for ridge, beta for bridge, q)
     */
    def center (x: MatrixD, y: VectorD, fname: Array [String] = null,
                hparam: HyperParameter = hp): RidgeBridgeRegression =
        val xℱ = CenterForm (x)
        val yℱ = CenterForm (y)
        new RidgeBridgeRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end center

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Ridge-Bridge Regression object from a data matrix and a response vector.
     *  This method provides data rescaling of x and centering of y.
     *  @param x       the un-centered data/input m-by-n matrix, NOT augmented with a first column of ones
     *  @param y       the un-centered response/output vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term 'lambda * b dot b'
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): RidgeBridgeRegression =
        val xℱ = NormForm (x)
        val yℱ = CenterForm (y)
        new RidgeBridgeRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end rescale

end RidgeBridgeRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeBridgeRegressionTest` main function tests the `RidgeBridgeRegression` class
 *  using the following regression equation.
 *      y  =  b dot x  =  b_1*x_1 + b_2*x_2.
 *  It compares `RidgeBridgeRegression` with `Regression`
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  > runMain scalation.modeling.ridgeBridgeRegressionTest
 */
@main def ridgeBridgeRegressionTest (): Unit =

    // 5 data points:         x_0    x_1
    val x = MatrixD ((5, 2), 36.0,  66.0,                              // 5-by-2 data matrix
                             37.0,  68.0,
                             47.0,  64.0,
                             32.0,  53.0,
                              1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)               // 5-dim response vector

//  println ("model: y = b_0 + b_1*x_1 + b_2*x_2")
    println ("model: y = b₀ + b₁*x₁ + b₂*x₂")                          // for Regression, remove b₀ for Ridge
    println (s"x = $x")
    println (s"y = $y")

    banner ("Regression")
    val ox  = VectorD.one (y.dim) +^: x                                // prepend a column of all 1's
    val reg = new Regression (ox, y)                                   // create a Regression model
    reg.inSample_Test ()                                               // train and test the model

    banner ("RidgeBridgeRegression with manual centering")
    val mu_x = x.mean                                                  // column-wise mean of x
    val mu_y = y.mean                                                  // mean of y
    val x_c  = x - mu_x                                                // centered x (column-wise)
    val y_c  = y - mu_y                                                // centered y
    val mod  = new RidgeBridgeRegression (x_c, y_c)                    // create a Ridge Regression model
    mod.inSample_Test ()                                               // train and test the model

    banner ("RidgeBridgeRegression with Auto-centering")
    val amod = RidgeBridgeRegression.center (x, y)                     // create an auto-centered Ridge Regression model
    amod.inSample_Test ()                                              // train and test the model

    banner ("RidgeBridgeRegression with Rescaling")
    val rmod = RidgeBridgeRegression.rescale (x, y)                    // create a rescaled Ridge Regression model
    rmod.inSample_Test ()                                              // train and test the model

    banner ("Make one OOS Predictions")
    val z   = VectorD (20.0, 80.0)                                     // new instance to predict
    val _1z = 1.0 +: z                                                 // prepend 1 to z
    val z_c = z - mu_x                                                 // center z
    println (s"reg.predict ($z) = ${reg.predict (_1z)}")               // predict using _1z
    println (s"mod.predict ($z) = ${mod.predict (z_c) + mu_y}")        // predict using z_c and add y's mean
    println (s"amod.predict ($z) = ${amod.predict (z)}")               // predict using z with auto-centering
    println (s"rmod.predict ($z) = ${rmod.predict (z)}")               // predict using z with rescaling

    banner ("Compare Summaries")
    println (reg.summary ())
    println (mod.summary ())
    println (amod.summary ())
    println (rmod.summary ())

end ridgeBridgeRegressionTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeBridgeRegressionTest2` main function tests the `RidgeBridgeRegression` class using
 *  the AutoMPG dataset.  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  Note, since x0 is automatically included in feature selection, make it an important variable.
 *  > runMain scalation.modeling.ridgeBridgeRegressionTest2
 */
@main def ridgeBridgeRegressionTest2 (): Unit =

    import scalation.modeling.Example_AutoMPG._                  // import sample dataset (x, y, x_fname, etc.)
    import RidgeRegression.hp

    hp("beta") = 10.0
    banner("AutoMPG Regression")
    val reg = new Regression(ox, y, ox_fname)
    reg.inSample_Test ()                                         // train and test the model
    println(reg.summary())

    banner("AutoMPG Ridge Regression")
    val mod1 = RidgeRegression.center (x, y, x_fname)
    mod1.inSample_Test ()                                        // train and test the model
    println (mod1.summary ())

    banner("AutoMPG Ridge + Bridge Regression")
    val mod2 = RidgeBridgeRegression.center (x, y, x_fname)
    mod2.inSample_Test ()                                        // train and test the model
    println (mod2.summary ())

end ridgeBridgeRegressionTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeBridgeRegressionTest3` main function tests the `RidgeBridgeRegression` class using
 *  the following regression equation.
 *      y  =  b dot x  =  b_1*x1 + b_2*x_2
 *  Test regression, forward selection and backward elimination.
 *  > runMain scalation.modeling.ridgeBridgeRegressionTest3
 */
@main def ridgeBridgeRegressionTest3 (): Unit =

    // 5 data points:         x_0    x_1
    val x = MatrixD ((5, 2), 36.0,  66.0,                              // 5-by-2 matrix
                             37.0,  68.0,
                             47.0,  64.0,
                             32.0,  53.0,
                              1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)

    println (s"x = $x")
    println (s"y = $y")

    // Compute centered (zero mean) versions of x and y

    val mu_x = x.mean                                                  // column-wise mean of x
    val mu_y = y.mean                                                  // mean of y
    val x_c  = x - mu_x                                                // centered x (column-wise)
    val y_c  = y - mu_y                                                // centered y

    println (s"x_c = $x_c")
    println (s"y_c = $y_c")

    banner ("RidgeBridgeRegression")
    val mod = new RidgeBridgeRegression (x_c, y_c)
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coeefficient statistics

    banner ("Forward Selection Test")
    mod.forwardSelAll (cross = "none")

    banner ("Backward Elimination Test")
    mod.backwardElimAll (cross = "none")

end ridgeBridgeRegressionTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeBridgeRegressionTest4` main function tests the `RidgeBridgeRegression` class using
 *  the following regression equation.
 *      y  =  b dot x  =  b_1*x1 + b_2*x_2
 *  > runMain scalation.modeling.ridgeBridgeRegressionTest4
 */
@main def ridgeBridgeRegressionTest4 (): Unit =

    // 4 data points:         x_1  x_2    y
    val xy = MatrixD ((4, 3), 1.0, 1.0, 6.0,                           // 4-by-3 matrix
                              1.0, 2.0, 8.0,
                              2.0, 1.0, 7.0,
                              2.0, 2.0, 9.0)
    val (x, y) = (xy.not (?, 2), xy(?, 2))                             // divides into data matrix and response vector
    val z = VectorD (2.0, 3.0)

    println (s"x = $x")
    println (s"y = $y")

    val mod = RidgeBridgeRegression (xy, null)()                       // factory method does centering
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics

    val yp = mod.predict (z)                                           // predict z
    println (s"predict ($z) = $yp")

end ridgeBridgeRegressionTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeBridgeRegressionTest5` main function tests the `RidgeBridgeRegression` class using
 *  the AutoMPG dataset.  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  Note, since x0 is automatically included in feature selection, make it an important variable.
 *  > runMain scalation.modeling.ridgeBridgeRegressionTest5
 */
@main def ridgeBridgeRegressionTest5 (): Unit =

    import Example_AutoMPG._

    banner ("AutoMPG Regression")
    val reg = new Regression (ox, y, ox_fname)                         // create a regression model (with intercept)
    reg.inSample_Test ()                                               // train and test the model
    println (reg.summary ())                                           // parameter/coefficient statistics

//  println (s"x = $x")                                                // data matrix without intercept
//  println (s"y = $y")                                                // response vector

    banner ("AutoMPG Ridge Bridge Regression")
    val mod = RidgeBridgeRegression.center (x, y, x_fname)             // create a ridge bridge regression model (no intercept)
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics
    Predictor.makePredictionInt (mod, mod.getX, y, mod.predict (x))    // make and show PREDICTION INTERVALs

    banner ("AutoMPG Validation Test")
    mod.validate ()()
/*
    banner ("AutoMPG Cross-Validation Test")
    FitM.showQofStatTable (mod.crossValidate ())

    import scala.runtime.ScalaRunTime.stringOf
    println (s"x_fname = ${stringOf (x_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                    // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq, Regression.metrics, s"R^2 vs n for RidgeBridgeRegression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
*/

end ridgeBridgeRegressionTest5

