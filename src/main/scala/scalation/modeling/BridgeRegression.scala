
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Tue Jul  1 17:54:49 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Bridge Regression (Lq for q > 0 Shrinkage/Regularization)
 *
 *  @caveat: currently only supports L0.5 Regularization (Bridge Regression with q = 0.5)
 *  Implements iterative re-weighted least squares (IRLS) to handle non-convex L0.5 penalty.
 *
 *  Model: minimize ||y - Xb||^2 + lambda * \sum |b_i|^0.5
 *
 *  Reference:
 *    - S. K. M. Wong, "Bridge regression models and IRLS",
 *      Journal of Statistical Computation and Simulation, 1995.
 *    - Hastie, Tibshirani & Friedman (2009), Elements of Statistical Learning, Sec. on Bridge.
 *
 *  Before calling the constructor, users should center their data; automatic by all factory methods.
 */

package scalation
package modeling

import scala.math.{abs, pow}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BridgeRegression` class supports L0.5 Regularization (Bridge Regression with q = 0.5)
 *  using Iterative Re-weighted Least Squares (IRLS) to handle non-convex L0.5 penalty.
 *  @param x       the centered data/input m-by-n matrix
 *  @param y       the centered response/output m-vector
 *  @param fname_  feature names
 *  @param hparam  hyper-parameters: "lambda" (penalty), "maxIter", "tol", "eps"
 *  @param xℱ      the transformation applied to x (e.g., Center or Norm)
 *  @param yℱ      the transformation applied to y (e.g., Center)
 */
class BridgeRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                        hparam: HyperParameter = RidgeRegression.hp,
                        xℱ: Transform = null, yℱ: Transform = null)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2, df = x.dim - x.dim2 - 1):
         // degrees of freedom: dfr = n, df = m - n - 1 as centered x matrix has 1 less column
         // fix after training by moving a dof from error to model for each coefficient eliminated
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`

    private val debug   = debugf ("BridgeRegression", false)
    private val lambda  = hparam("lambda").toDouble                  // shrinkage parameter
    private val sparse  = hparam("sparse").toInt == 1                // whether to sparsify
    private val maxIter = hparam("maxIter").toInt                    // maximum number of iterations for IWLS
    private val tol     = hparam("tol").toDouble                     // tolerance for convergence
    private val eps     = hparam("eps").toDouble                     // small constant to avoid division by zero
    private val q       = hparam("pow").toDouble                     // exponent/L_q norm
    private val qq      = 2 - q

    _modelName = s"BridgeRegression_$q"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train via IRLS: iterate solving weighted ridge until convergence.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit = 
        var b_old = new VectorD (x_.dim2)                            // initialize at zero
        b         = b_old.copy                                       // initial b-vector
        val xtX   = x_.ᵀ * x_                                        // form modified normal equations: X^T X + λ W
        val xty   = x_.ᵀ * y_

        var (go, it) = (true, 1)
        while go && it <= maxIter do
            val xtX_ = xtX.copy
            val w = b.map (e => pow (abs (e) + eps, qq))             // compute weights w_i = (|b_i| + eps)^(2 - q)
            for i <- w.indices do xtX_(i, i) += lambda * w(i)        // add λ * w(i) to diagonal

            val fac = new Fac_Cholesky (xtX_)                        // solve for b via Cholesky
            fac.factor ()
            b = fac.solve (xty)
            if (b - b_old).norm < tol then                           // check convergence
                debug ("train", s"converged after $it iterations")
                go = false
            b_old = b.copy
            it   += 1
        end while

        if go then debug ("train", s"completed $maxIter iterations without convergence")
        if sparse then LassoRegression.sparsify (b)
        debug ("train", s"IRLS estimates parameter b = $b")
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
        val yp = predict_ (x_)                                       // make predictions
        (yp, diagnose (y_, yp))                                      // return predictions and QoF vector
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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Quality-of-Fit summary reuses Fit.summary
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX,
                          fname_ : Array [String] = fname,
                          b_ : VectorD = b,
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): BridgeRegression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new BridgeRegression (x_cols, y, fname2, hparam)
    end buildModel
    
end BridgeRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BridgeRegression` companion object defines hyper-parameters and factory methods.
 */
object BridgeRegression extends Regularized:

    val hp = RidgeRegression.hp

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Bridge Regression object from an xy matrix and center the data.
     *  @param xy      the uncentered data/input m-by-n matrix, NOT augmented with a first column of ones
     *                     and the uncentered response m-vector (combined)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  includes the shrinkage hyper-parameter
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): BridgeRegression = 
        val (x, y) = (xy.not(?, col), xy(?, col))
        val xℱ = CenterForm (x)
        val yℱ = CenterForm (y)
        new BridgeRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Bridge Regression object from an x matrix and y vector and center the data.
     *  @param x       the uncentered data/input m-by-n matrix, NOT augmented with a first column of ones
     *  @param y       the uncentered response/output vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  includes the shrinkage hyper-parameter
     */
    def center (x: MatrixD, y: VectorD, fname: Array [String] = null,
                hparam: HyperParameter = hp): BridgeRegression = 
        val xℱ = CenterForm (x)
        val yℱ = CenterForm (y)
        new BridgeRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end center    

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Bridge Regression object from a data matrix and a response vector.
     *  This method provides data rescaling of x and centering of y.
     *  @param x       the un-centered data/input m-by-n matrix, NOT augmented with a first column of ones
     *  @param y       the un-centered response/output vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term 'lambda * norm b'
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): BridgeRegression =
        val xℱ = NormForm (x)
        val yℱ = CenterForm (y)
        new BridgeRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end rescale

end BridgeRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bridgeRegressionTest` main function tests the `BridgeRegression` class using
 *  the following regression equation.
 *      y  =  b dot x  =  b_1*x_1 + b_2*x_2.
 *  It compares `BridgeRegression` with `Regression`
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  > runMain scalation.modeling.bridgeRegressionTest
 */
@main def bridgeRegressionTest (): Unit =

    // 5 data points:         x_0    x_1
    val x = MatrixD ((5, 2), 36.0,  66.0,                              // 5-by-2 data matrix
                             37.0,  68.0,
                             47.0,  64.0,
                             32.0,  53.0,
                              1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)               // 5-dim response vector

//  println ("model: y = b_0 + b_1*x_1 + b_2*x_2")
    println ("model: y = b₀ + b₁*x₁ + b₂*x₂")                          // for Regression, remove b₀ for Bridge
    println (s"x = $x")
    println (s"y = $y")

    banner ("Regression")
    val ox  = VectorD.one (y.dim) +^: x                                // prepend a column of all 1's
    val reg = new Regression (ox, y)                                   // create a Regression model
    reg.inSample_Test ()                                               // train and test the model

    banner ("BridgeRegression with manual centering")
    val mu_x = x.mean                                                  // column-wise mean of x
    val mu_y = y.mean                                                  // mean of y
    val x_c  = x - mu_x                                                // centered x (column-wise)
    val y_c  = y - mu_y                                                // centered y
    val mod  = new BridgeRegression (x_c, y_c)                         // create a Bridge Regression model
    mod.inSample_Test ()                                               // train and test the model

    banner ("BridgeRegression with Auto-centering")
    val amod = BridgeRegression.center (x, y)                          // create an auto-centered Bridge Regression model
    amod.inSample_Test ()                                              // train and test the model

    banner ("BridgeRegression with Rescaling")
    val rmod = BridgeRegression.rescale (x, y)                         // create a rescaled Bridge Regression model
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

end bridgeRegressionTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bridgeRegressionTest2` main function tests the `BridgeRegression` class
 *  on the AutoMPG dataset.
 *  > runMain scalation.modeling.bridgeRegressionTest2
 */
@main def bridgeRegressionTest2 (): Unit =

    import Example_AutoMPG._

    banner ("AutoMPG Regression")
    val reg = new Regression (ox, y, ox_fname)                         // create a regression model (with intercept)
    reg.inSample_Test ()                                               // train and test the model
    println (reg.summary ())                                           // parameter/coefficient statistics
    
    banner ("AutoMPG Bridge Regression")
    val mod = new BridgeRegression (x, y, x_fname)                     // create a bridge regression model (no intercept)
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())

end bridgeRegressionTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bridgeRegressionTest3` main function tests the multi-collinearity method in
 *  the `BridgeRegression` class using the following regression equation.
 *      y  =  b dot x  =  b_1*x_1 + b_2*x_2
 *  Contour Plots for see, L2 penalty, see + L2 penalty, L1 penalty, sse + L1 penalty
 *                         L.5 penalty, see _ L.5 penalty
 *  > runMain scalation.modeling.bridgeRegressionTest3
 */
@main def bridgeRegressionTest3 (): Unit =

    val rvg = random.RandomVecD (100)
    val nrm = random.NormalVec_c (100, 0, 50)
    val x_1 = rvg.gen
    val x_2 = rvg.gen
    val x   = MatrixD (x_1, x_2).ᵀ

    val b_ = VectorD (4, 5)
    val y  = x * b_ + nrm.gen
    val xy = x :^+ y
    println (s"Correlation matrix for xy: rho = ${xy.corr}")

    val x_c = x - x.mean
    val y_c = y - y.mean

    banner ("Regression Model")
    val mod = new Regression (x_c, y_c)
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())
    FitM.showQofStatTable (mod.crossValidate ())
    var lambda = 0.0

    banner ("Ridge Regression Model")
    for i <- 1 to 10 do
        lambda = 200.0 * i
        RidgeRegression.hp("lambda") = lambda
        val mod2 = new RidgeRegression (x_c, y_c)
        mod2.inSample_Test ()                                          // train and test the model
        println (mod2.summary ())
        FitM.showQofStatTable (mod2.crossValidate ())
    end for

    banner ("Lasso Regression Model")
    for i <- 1 to 10 do
        lambda = 2000.0 * i
        RidgeRegression.hp("lambda") = lambda
        val mod2 = new LassoRegression (x_c, y_c)
        mod2.inSample_Test ()                                          // train and test the model
        println (mod2.summary ())
        FitM.showQofStatTable (mod2.crossValidate ())
    end for

    banner ("Bridge Regression Model")
    for i <- 1 to 10 do
        lambda = 4000.0 * i
        RidgeRegression.hp("lambda") = lambda
        val mod2 = new BridgeRegression (x_c, y_c)
        mod2.inSample_Test ()                                          // train and test the model
        println (mod2.summary ())
        FitM.showQofStatTable (mod2.crossValidate ())
    end for

    def f(b: VectorD):  Double = (y - x * b).normSq
    def f2(b: VectorD): Double = b.normSq * 2000.0
    def f3(b: VectorD): Double = f(b) + f2(b)
    def f4(b: VectorD): Double = b.norm1 * 20000.0
    def f5(b: VectorD): Double = f(b) + f4(b)
    def f6(b: VectorD): Double = b.norm_qq (0.5) * 40000.0
    def f7(b: VectorD): Double = f(b) + f6(b)

    val lb = VectorD (3, 4)
    val ub = VectorD (5, 6)
    new PlotC (f,  lb, ub, title = "Contour plot of sse")
    new PlotC (f2, lb, ub, title = "Contour plot of L2 penalty")
    new PlotC (f3, lb, ub, title = "Contour Plot of sse + L2 penalty")
    new PlotC (f4, lb, ub, title = "Contour Plot of L1 penalty")
    new PlotC (f5, lb, ub, title = "Contour Plot of sse + L1 penalty")
    new PlotC (f6, lb, ub, title = "Contour Plot of L.5 penalty")
    new PlotC (f7, lb, ub, title = "Contour Plot of sse + L.5 penalty")

end bridgeRegressionTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bridgeRegressionTest4` main function tests the `BridgeRegression` class using
 *  the following regression equation.
 *      y  =  b dot x  =  b_1*x1 + b_2*x_2
 *  > runMain scalation.modeling.bridgeRegressionTest4
 */
@main def bridgeRegressionTest4 (): Unit =

    // 4 data points:         x_1  x_2    y
    val xy = MatrixD ((4, 3), 1.0, 1.0, 6.0,                           // 4-by-3 matrix
                              1.0, 2.0, 8.0,
                              2.0, 1.0, 7.0,
                              2.0, 2.0, 9.0)
    val (x, y) = (xy.not (?, 2), xy(?, 2))                             // divides into data matrix and response vector
    val z = VectorD (2.0, 3.0)

    println (s"x = $x")
    println (s"y = $y")

    val mod = BridgeRegression (xy, null)()                            // factory method does centering
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics

    val yp = mod.predict (z)                                           // predict z
    println (s"predict ($z) = $yp")

end bridgeRegressionTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `bridgeRegressionTest5` main function tests the `BridgeRegression` class using
 *  the AutoMPG dataset.  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  Note, since x0 is automatically included in feature selection, make it an important variable.
 *  > runMain scalation.modeling.bridgeRegressionTest5
 */
@main def bridgeRegressionTest5 (): Unit =

    import Example_AutoMPG._

    banner ("AutoMPG Regression")
    val reg = new Regression (ox, y, ox_fname)                         // create a regression model (with intercept)
    reg.inSample_Test ()                                               // train and test the model
    println (reg.summary ())                                           // parameter/coefficient statistics

//  println (s"x = $x")                                                // data matrix without intercept
//  println (s"y = $y")                                                // response vector

    banner ("AutoMPG Bridge Regression")
    val mod = BridgeRegression.center (x, y, x_fname)                  // create a bridge regression model (no intercept)
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
        new PlotM (null, rSq, Regression.metrics, s"R^2 vs n for BridgeRegression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
*/

end bridgeRegressionTest5

