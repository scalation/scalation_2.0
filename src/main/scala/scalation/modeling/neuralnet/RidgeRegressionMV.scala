
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Multiple Linear Regression with Multiple Response Variables
 *                  Multi-variate Multiple Linear Regression
 */

// FIX: use cholesky. QR does not work

package scalation
package modeling
package neuralnet

import scala.math.sqrt

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegressionMV` class supports multi-variate multiple linear regression.
 *  In this case, x is multi-dimensional [1, x_1, ... x_k] and y is multi-dimensional
 *  [y_0, ... y_l].
 *  Fit the parameter vector b in for each regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve the parameter vector b
 *  using the Normal Equations:
 *      x.t * x * b  =  x.t * y 
 *      b  =  fac.solve (.)
 *  with L_2 Regularization.
 *  Five factorization algorithms are provided:
 *      `Fac_QR`         QR Factorization: slower, more stable (default)
 *      `Fac_SVD`        Singular Value Decomposition: slowest, most robust
 *      `Fac_Cholesky`   Cholesky Factorization: faster, less stable (reasonable choice)
 *      `Fac_LU'         LU Factorization: better than Inverse
 *      `Fac_Inverse`    Inverse Factorization: textbook approach
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  Note, not intended for use when the number of degrees of freedom 'df' is negative.
 *  @see en.wikipedia.org/wiki/Degrees_of_freedom_(statistics)
 *------------------------------------------------------------------------------
 *  @param x       the data/input m-by-n matrix
 *                     (augment with a first column of ones to include intercept in model)
 *  @param y       the response/output m-by-ny matrix
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters (defaults to Regression.hp)
 *  @param xℱ      the transformation applied to x (e.g., Center or Norm)
 *  @param yℱ      the transformation applied to y (e.g., Center)
 */
class RidgeRegressionMV (x: MatrixD, y: MatrixD, fname_ : Array [String] = null,
                         hparam: HyperParameter = RidgeRegression.hp,
                         xℱ: Transform = null, yℱ: Transform = null)
      extends PredictorMV (x, y, fname_, hparam)
         with Fit (dfr = x.dim2, df = x.dim - x.dim2 - 1):
         // degrees of freedom: dfr = n, df = m - n - 1 as centered x matrix has 1 less column
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`
         // no intercept => correct Degrees of Freedom (DoF); as lambda get larger, need effective DoF

    private val debug     = debugf ("RidgeRegressionMV", false)          // debug function
    private val flaw      = flawf ("RidgeRegressionMV")                  // flaw function
    private val algorithm = hparam("factorization")                      // factorization algorithm
    private val lambda    = hparam ("lambda").toDouble
//                          if hparam("lambda") <= 0.0 then findLambda._1
//                          else hparam ("lambda").toDouble

    _modelName = "RidgeRegressionMV_${lambda}"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a solver for the Normal Equations using the selected factorization algorithm.
     *  @param x_  the matrix to be used by the solver
     */
    private def solver (x_ : MatrixD): Factorization =

        val xtx  = x_.transpose * x_                                     // pre-compute X.t * X
        val ey   = MatrixD.eye (x_.dim, x_.dim2)                         // identity matrix
        val xtx_ = xtx.copy                                              // copy xtx (X.t * X)
        for i <- xtx_.indices do xtx_(i, i) += lambda                    // add lambda to the diagonal

        algorithm match                                                  // select the factorization technique
            case "Fac_QR" => val xx = x_ ++ (ey * sqrt(lambda))
                             println (s"xx.dim ${xx.dim}")
                             Fac_QR(xx)                                  // QR/LQ Factorization
//          case "Fac_SVD"      => new Fac_SVD (x_)                      // Singular Value Decomposition - FIX
            case "Fac_Cholesky" => new Fac_Cholesky(xtx_)                // Cholesky Factorization
            case "Fac_LU"       => new Fac_LU(xtx_)                      // LU Factorization
            case _              => new Fac_Inverse(xtx_)                 // Inverse Factorization
            end match
    end solver

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  multiple regression equation
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [1, x_1 , ... x_k] + e
     *  using the ordinary least squares 'OLS' method.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output matrix
     */
    def train (x_ : MatrixD = x, y_ : MatrixD = y): Unit =
        val fac = solver (x_)
        fac.factor ()                                                    // factor the matrix, either X or X.t * X

        bb = Array (new NetParam (new MatrixD (x_.dim2, y_.dim2)))       // allocate parameters bb (only uses 'bb(0).w')
        for k <- y_.indices2 do
            val yk  = y_(?, k)
//          println (s"yk = ${yk.dim}")
            bb(0).w(?, k) = fac match                                    // RECORD the parameters/coefficients (@see `PredictorMV`)
            case fac: Fac_QR  => fac.solve (yk)
            case fac: Fac_SVD => fac.solve (yk)
            case _            => fac.solve (x_.transpose * yk)

            if bb(0).w(0, k).isNaN then flaw ("train", s"parameters bb(0).w = ${bb(0).w}")
        end for

        debug ("train", s"$fac estimates parameters bb(0).w = ${bb(0).w}")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output matrix (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : MatrixD = y): (MatrixD, MatrixD) =
        val yp = predict_ (x_)                                           // make predictions
        e = y_ - yp                                                      // RECORD the residuals/errors (@see `Predictor`)
        val qof = MatrixD (for k <- y_.indices2 yield diagnose (y_(?, k), yp(?, k))).transpose
        (yp, qof)                                                        // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the vector of y = f(z) by evaluating the formula y = b dot z.
     *  It works on transformed values.
     *  @param z  the new vector to predict
     */
    def predict_ (z: VectorD): VectorD = bb(0).w dot z

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of matrix y = f(x_, b).  It works on transformed values.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    def predict_ (x_ : MatrixD): MatrixD = x_ * bb(0).w

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the vector of y = f(z) by evaluating the formula y = b dot z.
     *  It is overridden to handle transformations.
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): VectorD =
        val zz = if xℱ == null then z else xℱ.f(MatrixD (z))(0)
        if yℱ == null then bb(0).w dot zz else yℱ.fi(bb(0).w dot zz)
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the matrix of vector y = f(x_, b).  It is overridden to handle transformations.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    override def predict (x_ : MatrixD): MatrixD =
        val xx = if xℱ == null then x_ else xℱ.f(x_)
        if yℱ == null then xx * bb(0).w else yℱ.fi(xx * bb(0).w)
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname,
                          b_ : VectorD = bb(0).w(?, 0),                  // FIX
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    def buildModel (x_cols: MatrixD, fname: Array [String] = null): RidgeRegressionMV =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new RidgeRegressionMV (x_cols, y, fname, hparam)
    end buildModel

end RidgeRegressionMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegressionMV` companion object provides factory methods for creating
 *  Multi-Variate (MV) Regression models.
 */
object RidgeRegressionMV extends RegularizedMV:

    val hp = RidgeRegression.hp

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Ridge RegressionMV object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     *  @param col     the first designated response column (defaults to next to last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = RidgeRegression.hp)
               (col: Int = xy.dim2 - 2): RidgeRegressionMV = 
        val (x, y) = (xy(?, 0 until col), xy(?, col until xy.dim))
        val xℱ = CenterForm (x)
        val yℱ = CenterForm (y)
        new RidgeRegressionMV (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Ridge RegressionMV from a data matrix and response vector.
     *  This function centers the data.
     *  @param x       the un-centered data/input m-by-n matrix, NOT augmented with a first column of ones
     *  @param y       the un-centered response/output matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term 'lambda * b dot b'
     */
    def center (x: MatrixD, y: MatrixD, fname: Array [String] = null,
                hparam: HyperParameter = RidgeRegression.hp): RidgeRegressionMV =
        val xℱ = CenterForm (x)
        val yℱ = CenterForm (y)
        new RidgeRegressionMV (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end center

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a Ridge RegressionMV object from a data matrix and a response matrix.
     *  This method provides data rescaling.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output matrix
     *  @param fname   the feature/variable names (use null for default)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     */
    def rescale (x: MatrixD, y: MatrixD, fname: Array [String] = null,
                 hparam: HyperParameter = RidgeRegression.hp): RidgeRegressionMV =
        val xℱ = NormForm (x)
        val yℱ = CenterForm (y)
        new RidgeRegressionMV (xℱ.f(x), yℱ.f(y), fname, hparam, xℱ, yℱ)
    end rescale

end RidgeRegressionMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeRegressionMVTest` main function tests the `RidgeRegressionMV` class using
 *  the following regression equation.
 *      y  =  b dot x  =  b_1*x_1 + b_2*x_2.
 *  It compares `RidgeRegressionMV` with `RegressionMV`
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  > runMain scalation.modeling.neuralnet.ridgeRegressionMVTest
 */
@main def ridgeRegressionMVTest (): Unit =

    // 5 data points:         x_0    x_1
    val x = MatrixD ((5, 2), 36.0,  66.0,                              // 5-by-2 data matrix
                             37.0,  68.0,
                             47.0,  64.0,
                             32.0,  53.0,
                              1.0, 101.0)

    val y = MatrixD ((5, 2), 745.0,  700.0,                            // 5-by-2 response matrix
                             895.0,  900.0,
                             442.0,  500.0,
                             440.0,  400.0,
                            1598.0, 1500.0) 

//  println ("model: y = b_0 + b_1*x_1 + b_2*x_2")
    println ("model: y = b₀ + b₁*x₁ + b₂*x₂")                          // for RegressionMV, remove b₀ for Ridge
    println (s"x = $x")
    println (s"y = $y")

    banner ("RegressionMV")
    val ox  = VectorD.one (y.dim) +^: x                                // prepend a column of all 1's
    val reg = new RegressionMV (ox, y)                                 // create a RegressionMV model
    reg.trainNtest ()()                                                // train and test the model

    banner ("RidgeRegressionMV with manual centering")
    val mu_x = x.mean                                                  // column-wise mean of x
    val mu_y = y.mean                                                  // mean of y
    val x_c  = x - mu_x                                                // centered x (column-wise)
    val y_c  = y - mu_y                                                // centered y
    val mod  = new RidgeRegressionMV (x_c, y_c)                        // create a Ridge RegressionMV model
    mod.trainNtest ()()                                                // train and test the model

    banner ("RidgeRegressionMV with Auto-centering")
    val amod = RidgeRegressionMV.center (x, y)                         // create an auto-centered Ridge RegressionMV model
    amod.trainNtest ()()                                               // train and test the model

    banner ("RidgeRegressionMV with Rescaling")
    val rmod = RidgeRegressionMV.rescale (x, y)                        // create a rescaled Ridge RegressionMV model
    rmod.trainNtest ()()                                               // train and test the model

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

end ridgeRegressionMVTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeRegressionMVTest2` main function is used to test the `RidgeRegressionMV` class.
 *  > runMain scalation.modeling.neuralnet.ridgeRegressionMVTest2
 */
@main def ridgeRegressionMVTest2 (): Unit =

    val x = MatrixD ((5, 3), 1.0, 0.35, 0.9,                     // training data - input matrix (m=5 vectors)
                             1.0, 0.20, 0.7,
                             1.0, 0.30, 0.8,
                             1.0, 0.25, 0.75,
                             1.0, 0.40, 0.95)
    val y = MatrixD ((5, 2), 0.5, 0.4,                           // training data - output matrix (m=5 vectors)
                             0.3, 0.3,
                             0.2, 0.35,
                             0.3, 0.32,
                             0.6, 0.5)

    println (s"input  matrix x = $x")
    println (s"output matrix y = $y")

    val mod = new RidgeRegressionMV (x, y)                       // create RegreesionMV model
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("ridgeRegressionMVTest: Compare with Linear Regression - first column of y")
    val y0  = y(?, 0)                                            // use first column of response matrix y
    val rg0 = new Regression (x, y0)                             // create a Regression model
    rg0.trainNtest ()()                                          // train and test the model
    println (rg0.summary ())                                     // parameter/coefficient statistics

    banner ("ridgeRegressionMVTest: Compare with Linear Regression - second column of y")
    val y1  = y(?, 1)                                            // use second column of response matrix y
    val rg1 = new Regression (x, y1)                             // create a Regression model
    rg1.trainNtest ()()                                          // train and test the model
    println (rg1.summary ())                                     // parameter/coefficient statistics

    val b_ = mod.parameters(0).w                                 // check for parameter agreements with `Regression`
    assert (b_(?, 0) == rg0.parameter)
    assert (b_(?, 1) == rg1.parameter)

end ridgeRegressionMVTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeRegressionMVTest3` main function tests the `RidgeRegressionMV` class using
 *  the Concrete dataset.
 *  > runMain scalation.modeling.neuralnet.ridgeRegressionMVTest3
 */
@main def ridgeRegressionMVTest3 (): Unit =

    import Example_Concrete._

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("Concrete RidgeRegressionMV")
    val mod = new RidgeRegressionMV (ox, y, ox_fname)            // create model with intercept (else pass x)
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("Concrete Validation Test")
    println (Fit.showFitMap (mod.validate ()()._2))

    banner ("Concrete Cross-Validation Test")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end ridgeRegressionMVTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeRegressionMVTest4` main function tests the `RidgeRegressionMV` class using
 *  the AutoMPG dataset.
 *  > runMain scalation.modeling.neuralnet.ridgeRegressionMVTest4
 */
@main def ridgeRegressionMVTest4 (): Unit =

    import Example_AutoMPG.{ox, yy, ox_fname}

//  println (s"ox = $ox")
//  println (s"yy = $yy")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG RidgeRegressionMV")
    val mod = new RidgeRegressionMV (ox, yy, ox_fname)           // create model with intercept (else pass x)
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("AutoMPG Validation Test")
    println (Fit.showFitMap (mod.validate ()()._2))

    banner ("AutoMPG Cross-Validation Test")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end ridgeRegressionMVTest4
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeRegressionMVTest5` main function tests the `RidgeRegressionMV` class using
 *  the AutoMPG dataset.  It tests forward selection.
 *  > runMain scalation.modeling.neuralnet.ridgeRegressionMVTest5
 */
@main def ridgeRegressionMVTest5 (): Unit =

    import Example_AutoMPG.{ox, yy, ox_fname}

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG RidgeRegressionMV")
    val mod = new RidgeRegressionMV (ox, yy, ox_fname)           // create model with intercept (else pass x)
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, smape, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                     // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "smape", "R^2 cv"),
               s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end ridgeRegressionMVTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ridgeRegressionMVTest6` main function tests the `RidgeRegressionMV` class using
 *  the AutoMPG dataset.  It tests forward, backward and stepwise selection.
 *  > runMain scalation.modeling.neuralnet.ridgeRegressionMVTest6
 */
@main def ridgeRegressionMVTest6 (): Unit =

    import Example_AutoMPG.{ox, yy, ox_fname}

//  println (s"ox = $ox")
//  println (s"y  = $y")

    banner ("AutoMPG RidgeRegressionMV")
    val mod = new RidgeRegressionMV (ox, yy, ox_fname)           // create model with intercept (else pass x)
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())

    println (s"ox_fname = ${stringOf (ox_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)              // R^2, R^2 bar, smape, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${ox.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "smape", "R^2 cv"),
                   s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end ridgeRegressionMVTest6

