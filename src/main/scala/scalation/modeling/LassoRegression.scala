
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Mustafa Nural
 *  @version 2.0
 *  @date    Tue Apr 18 14:24:14 EDT 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Lasso Regression (L_1 Shrinkage/Regularization)
 *
 *  Before calling the constructor, users should center their data; automatic by all factory methods.
 */

package scalation
package modeling

import scala.math.abs

import scalation.mathstat._
import scalation.optimization.LassoAdmm

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegression` class supports multiple linear regression.  In this case,
 *  'x' is multi-dimensional [1, x_1, ... x_k].  Fit the parameter vector 'b' in
 *  the regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  where 'e' represents the residuals (the part not explained by the model).
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  @param x       the data/input m-by-n matrix
 *  @param y       the response/output m-vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the shrinkage hyper-parameter, lambda (0 => OLS) in the penalty term 'lambda * b dot b'
 *  @param xℱ      the transformation applied to x (e.g., Center or Norm)
 *  @param yℱ      the transformation applied to y (e.g., Center)
 */
class LassoRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                       hparam: HyperParameter = RidgeRegression.hp,
                       xℱ: Transform = null, yℱ: Transform = null)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2, df = x.dim - x.dim2 - 1):
         // degrees of freedom: dfr = n, df = m - n - 1 as centered x matrix has 1 less column
         // fix after training by moving a dof from error to model for each coefficient eliminated
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`

    private val debug  = debugf ("LassoRegression", true)                // debug function
    private val flaw   = flawf ("LassoRegression")                       // flaw function
    private val lambda = hparam("lambda").toDouble                       // shrinkage parameter (weight to put on regularization)
    private val sparse = hparam("sparse").toInt == 1                     // whether to sparsify

    _modelName = "LassoRegression_${lambda}"

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the value of the shrinkage parameter 'lambda'.
     */
    def lambda_ : Double = lambda

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find an optimal value for the shrinkage parameter 'lambda' using Cross Validation
     *  to minimize 'sse_cv'.  The search starts with the low default value for 'lambda'
     *  doubles it with each iteration, returning the minimum 'lambda' and it corresponding
     *  cross-validated 'sse'.
     */
    def findLambda: (Double, Double) =
        var l      = lambda
        var l_best = l
        var sse    = Double.MaxValue

        cfor (0, 20) { _ =>
            RidgeRegression.hp("lambda") = l
            val rrg   = new LassoRegression (x, y)
            val stats = rrg.crossValidate ()
            val sse2  = stats(QoF.sse.ordinal).mean
            banner (s"LassoRegression with lambda = ${rrg.lambda_} has sse = $sse2")
            if sse2 < sse then
                sse = sse2; l_best = l
            FitM.showQofStatTable (stats)
            l *= 2
        } // cfor
        (l_best, sse)
    end findLambda

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_1 , ... x_k] + e
     *  regularized by the sum of magnitudes of the coefficients.
     *  @see pdfs.semanticscholar.org/969f/077a3a56105a926a3b0c67077a57f3da3ddf.pdf
     *  @see `scalation.optimization.LassoAdmm`
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit =
        b = LassoAdmm.solve (x_, y_, lambda)                             // Alternating Direction Method of Multipliers

        if b(0).isNaN then flaw ("train", s"parameter b = $b")
        if sparse then LassoRegression.sparsify (b)
        debug ("train", s"LassoAdmm estimates parameter b = $b")
        val nz = b.countZero                                             // count number of coefficients set to zero
        if nz > 0 then resetDF (x.dim2 - nz, x.dim - x.dim2 - 1 + nz)
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
        val yp = predict_ (x_)                                           // make predictions
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
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
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): LassoRegression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new LassoRegression (x_cols, y, fname2, hparam)
    end buildModel

end LassoRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegression` companion object provides factory methods for the
 *  `LassoRegression` class.
 */
object LassoRegression extends Regularized:

    val hp = RidgeRegression.hp

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Lasso Regression object from a combined data matrix.
     *  This function centers the data.
     *  @param xy      the combined data matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): LassoRegression =
        val (x, y) = (xy.not(?, col), xy(?, col))
        val xℱ = CenterForm (x)
        val yℱ = CenterForm (y)
        new LassoRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Lasso Regression from a data matrix and response vector.
     *  This function centers the data.
     *  @param x       the un-centered data/input m-by-n matrix, NOT augmented with a first column of ones
     *  @param y       the un-centered response/output vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term 'lambda * |b|'
     */
    def center (x: MatrixD, y: VectorD, fname: Array [String] = null,
                hparam: HyperParameter = RidgeRegression.hp): LassoRegression =
        val xℱ = CenterForm (x)
        val yℱ = CenterForm (y)
        new LassoRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end center

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Lasso Regression object from a data matrix and a response vector.
     *  This method provides data rescaling on x and centering on y.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to hp)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): LassoRegression =
        val xℱ = NormForm (x)
        val yℱ = CenterForm (y)
        new LassoRegression (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Zero out small parameters/coefficients in the model that are below a threshold.
     *  Intended for use by any regularized regression, but especially Lasso and Bridge.
     *  @param b          the parameters/coefficients to sparsify
     *  @param relThresh  the relative (to the max) threshold below which parameter is set to zero
     *  @return a sparse version of the parameter vector
     */
    def sparsify (b: VectorD, relThresh: Double = 1e-3): Unit =
        val thresh = b.max * relThresh
        for i <- b.indices do if abs (b(i)) < thresh then b(i) = 0.0
    end sparsify

end LassoRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lassoRegressionTest` main function tests `LassoRegression` class using
 *  the following regression equation.
 *      y  =  b dot x  =  b_1*x_1 + b_2*x_2.
 *  It comapres `LassoRegression` to `Regression`.
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  > runMain scalation.modeling.lassoRegressionTest
 */
@main def lassoRegressionTest (): Unit =

    // 5 data points:         x_0    x_1
    val x = MatrixD ((5, 2), 36.0,  66.0,                              // 5-by-2 data matrix
                             37.0,  68.0,
                             47.0,  64.0,
                             32.0,  53.0,
                              1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)               // 5-dim response vector

//  println ("model: y = b_0 + b_1*x_1 + b_2*x_2")
    println ("model: y = b₀ + b₁*x₁ + b₂*x₂")                          // for Regression, remove b₀ for Lasso
    println (s"x = $x")
    println (s"y = $y")

    banner ("Regression")
    val ox  = VectorD.one (y.dim) +^: x                                // prepend a column of all 1's
    val reg = new Regression (ox, y)                                   // create a Regression model
    reg.inSample_Test ()                                               // train and test the model

    banner ("LassoRegression with manual centering")
    val mu_x = x.mean                                                  // column-wise mean of x
    val mu_y = y.mean                                                  // mean of y
    val x_c  = x - mu_x                                                // centered x (column-wise)
    val y_c  = y - mu_y                                                // centered y
    val mod = new LassoRegression (x_c, y_c)                           // create a Lasso Regression model
    mod.inSample_Test ()                                               // train and test the model

    banner ("LassoRegression with Auto-centering")
    val amod = LassoRegression.center (x, y)                           // create an auto-centered Lasso Regression model
    amod.inSample_Test ()                                              // train and test the model

    banner ("LasoRegression with Rescaling")
    val rmod = LassoRegression.rescale (x, y)                          // create a rescaled Lasso Regression model
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

end lassoRegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lassoRegressionTest2` main function tests `LassoRegression` class using
 *  the following regression equation.
 *      y  =  b dot x  =  b_1*x1 + b_2*x_2.
 *  Try non-default value for the 'lambda' hyper-parameter.
 *  > runMain scalation.modeling.lassoRegressionTest2
 */
@main def lassoRegressionTest2 (): Unit =

    import RidgeRegression.hp

    println (s"hp = $hp")
    val hp2 = hp.updateReturn ("lambda", 1.0)                          // try different values
    println (s"hp2 = $hp2")

    // 5 data points:        one   x_1    x_2
    val x = MatrixD ((5, 3), 1.0, 36.0,  66.0,                         // 5-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0,  1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)
    val z = VectorD (1.0, 20.0, 80.0)

    println ("x = " + x + "\ny = " + y + "\nz = " + z)

    banner ("LassoRegression")
    val mod = new LassoRegression (x, y, hparam = hp2)                 // create a Lasso regression model
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics
    println (s"predict ($z) = ${mod.predict (z)}")                     // make an out-of-sample prediction

end lassoRegressionTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lassoRegressionTest3` main function tests `LassoRegression` class using
 *  the following regression equation.
 *      y  =  b dot x  =  b_1*x1 + b_2*x_2.
 *  Test regression, forward selection and backward elimination.
 *  > runMain scalation.modeling.lassoRegressionTest3
 */
@main def lassoRegressionTest3 (): Unit =

    // 5 data points:            one   x_1    x_2
    val x = MatrixD ((5, 3), 1.0, 36.0,  66.0,                         // 5-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0,  1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)
    val z = VectorD (1.0, 20.0, 80.0)

    println ("x = " + x + "\ny = " + y + "\nz = " + z)

    banner ("LassoRegression")
    val mod = new LassoRegression (x, y)                               // create a Lasso regression model
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics
    println (s"predict ($z) = ${mod.predict (z)}")                     // make an out-of-sample prediction

    banner ("Forward Selection Test")
    mod.forwardSelAll (cross = "none")

    banner ("Backward Elimination Test")
    mod.backwardElimAll (cross = "none")

end lassoRegressionTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lassoRegressionTest4` main function tests the `LassoRegression` class using
 *  the Covid-19 dataset.  It illustrates using the `Relation`/Table class for reading
 *  the data from a .csv file "covid_19_weekly.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar, sMAPE, and R^2 cv vs. the instance index.
 *  > runMain scalation.modeling.lassoRegressionTest4
 */
@main def lassoRegressionTest4 (): Unit =

//  import scalation.database.relation.Relation
    import scalation.database.table.Table

    banner ("covid_19_weekly Table")
//  val data = Relation (DATE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    val data = Table.load ("covid_19_weekly.csv", "covid_19_weekly", 17, null) 
    data.show ()

    banner ("covid_19_weekly dataset")
    val xcols  = Array (1, 3, 4, 5, 6, 7, 8, 9, 10)
    val (x, y) = data.toMatrixV (xcols, 2)
    val fname  = xcols.map (data.schema (_))
    println (s"fname = $fname")
    println (s"y = $y")
    println (s"y = $y")

    banner ("LassoRegression for covid_19_weekly")
    val mod = new LassoRegression (x, y, fname)                        // create a Lasso regression model
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics

    banner ("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll ()                             // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                       // instance index
    new PlotM (t, rSq, Regression.metrics, "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")

end lassoRegressionTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lassoRegressionTest5` main function tests the `LassoRegression` class using
 *  the AutoMPG dataset.  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  Note, since x0 is automatically included in feature selection, make it an important variable.
 *  > runMain scalation.modeling.lassoRegressionTest5
 */
@main def lassoRegressionTest5 (): Unit =

    import Example_AutoMPG._

    banner ("AutoMPG Regression")
    val reg = new Regression (ox, y, ox_fname)                         // create a regression model (with intercept)
    reg.inSample_Test ()                                               // train and test the model
    println (reg.summary ())                                           // parameter/coefficient statistics

//  println (s"x = $x")                                                // data matrix without intercept
//  println (s"y = $y")                                                // response vector

    banner ("AutoMPG Lasso Regression")
    val mod = LassoRegression.center (x, y, x_fname)                   // create a Lasso regression model (no intercept)
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
        new PlotM (null, rSq, Regression.metrics, s"R^2 vs n for LassoRegression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for
*/

end lassoRegressionTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lassoRegressionTest6` main function tests the `LassoRegression` class using
 *  the COVID dataset.  It illustrates using the `Table` class for reading the'
 *  data from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also uses the 'findLambda' method to search for a shrinkage parameter
 *  that roughly mininizes 'sse_cv'.
 *  > runMain scalation.modeling.lassoRegressionTest6
 */
@main def lassoRegressionTest6 (): Unit =

    import scalation.database.table.Table

    banner ("auto-mpg Table")
    val ncols = 8
    val data  = Table.load ("auto_mpg.csv", "auto_mpg", ncols, null)
    data.show ()

    banner ("auto-mpg dataset")
    val xcols  = Array.range (0, ncols-1)
    val (x, y) = data.toMatrixV (xcols, ncols-1)
    val fname  = xcols.map (data.schema (_))
    println (s"y = $y")

    banner ("LassoRegression for auto-mpg")
    val mod = new LassoRegression (x, y, fname)                        // create a Lasso regression model
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics

    println (s"best (lambda, sse) = ${mod.findLambda}")

end lassoRegressionTest6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `lassoRegressionTest7` main function tests the multi-collinearity method in
 *  the `LassoRegression` class using the following regression equation.
 *      y  =  b dot x  =  b_1*x_1 + b_2*x_2
 *  Contour Plots for see, L2 penalty, see + L2 penalty, L1 penalty, sse + L1 penalty
 *  > runMain scalation.modeling.lassoRegressionTest7
 */
@main def lassoRegressionTest7 (): Unit =

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

    def f(b: VectorD):  Double = (y - x * b).normSq
    def f2(b: VectorD): Double = b.normSq * 2000.0
    def f3(b: VectorD): Double = f(b) + f2(b)
    def f4(b: VectorD): Double = b.norm1 * 20000.0
    def f5(b: VectorD): Double = f(b) + f4(b)

    val lb = VectorD (3, 4)
    val ub = VectorD (5, 6)
    new PlotC (f,  lb, ub, title = "Contour plot of sse")
    new PlotC (f2, lb, ub, title = "Contour plot of L2 penalty")
    new PlotC (f3, lb, ub, title = "Contour Plot of sse + L2 penalty")
    new PlotC (f4, lb, ub, title = "Contour Plot of L1 penalty")
    new PlotC (f5, lb, ub, title = "Contour Plot of sse + L1 penalty")

end lassoRegressionTest7

