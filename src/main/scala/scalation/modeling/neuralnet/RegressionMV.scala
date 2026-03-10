
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Multiple Linear Regression with Multiple Response Variables
 *                  Multi-variate Multiple Linear Regression
 */

package scalation
package modeling
package neuralnet

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionMV` class supports multi-variate multiple linear regression.  In this case,
 *  x is multi-dimensional [1, x_1, ... x_k] and y is multi-dimensional [y_0, ... y_l].
 *  Fit the parameter vector b in for each regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve the parameter matrix bb(0).w
 *  using the Normal Equations:
 *      x.t * x * b  =  x.t * y 
 *      b  =  fac.solve (.)
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
 */
class RegressionMV (x: MatrixD, y: MatrixD, fname_ : Array [String] = null,
                  hparam: HyperParameter = Regression.hp)
      extends PredictorMV (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):
         // if not using an intercept df = (x.dim2, x.dim-x.dim2), correct by calling 'resetDF' method from `Fit`

    private val debug     = debugf ("RegressionMV", false)               // debug function
    private val flaw      = flawf ("RegressionMV")                       // flaw function
    private val algorithm = hparam("factorization")                      // factorization algorithm
    private val n         = x.dim2                                       // number of columns

    _modelName = s"RegressionMV_$n"

    if n < 1 then flaw ("init", s"dim2 = $n of the 'x' matrix must be at least 1")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a solver for the Normal Equations using the selected factorization algorithm.
     *  @param x_  the matrix to be used by the solver
     */
    private def solver (x_ : MatrixD): Factorization =
        algorithm match                                                  // select the factorization algorithm
        case "Fac_Cholesky" => new Fac_Cholesky (x_.transpose * x_)      // Cholesky Factorization
        case "Fac_LU"       => new Fac_LU (x_.transpose * x_)            // LU Factorization
        case "Fac_Inverse"  => new Fac_Inverse (x_.transpose * x_)       // Inverse Factorization
        case "Fac_SVD"      => new Fac_SVD (x_)                          // Singular Value Decomposition
        case _              => Fac_QR (x_)                               // QR/LQ Factorization (default)
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
        val yp = predict (x_)                                            // make predictions
        e = y_ - yp                                                      // RECORD the residuals/errors (@see `Predictor`)
        val qof = MatrixD (for k <- y_.indices2 yield diagnose (y_(?, k), yp(?, k))).transpose
        (yp, qof)                                                        // return predictions and QoF vector
    end test

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
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z,
     *  e.g., (b_0, b_1, b_2) dot (1, z_1, z_2).
     *  @param z  the new vector to predict
     */
    def predict (z: VectorD): VectorD = bb(0).w dot z

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of matrix y = f(x_, b).  It is overridden for speed.
     *  @param x_  the matrix to use for making predictions, one for each row
     */
    override def predict (x_ : MatrixD): MatrixD = x_ * bb(0).w

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): RegressionMV =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new RegressionMV (x_cols, y, fname2, hparam)
    end buildModel

end RegressionMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionMV` companion object provides factory methods for creating
 *  multi-variate regression models.
 */
object RegressionMV:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionMV` object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix (predictors and response)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     *  @param col     the first designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = Regression.hp)
               (col: Int = xy.dim2 - 1): RegressionMV = 
        new RegressionMV (xy(?, 0 until col), xy(?, col until xy.dim2), fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionMV` object from a data matrix and a response matrix.
     *  This method provides data rescaling.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (use null for default)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     */
    def rescale (x: MatrixD, y: MatrixD, fname: Array [String] = null,
                 hparam: HyperParameter = Regression.hp): RegressionMV = 
        val xn = normalize ((x.mean, x.stdev)) (x)
        new RegressionMV (xn, y, fname, hparam)
    end rescale

end RegressionMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionMVTest` main function is used to test the `RegressionMV` class.
 *  > runMain scalation.modeling.neuralnet.regressionMVTest
 */
@main def regressionMVTest (): Unit =

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

    val mod = new RegressionMV (x, y)                            // create RegreesionMV model
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("regressionMVTest: Compare with Linear Regression - first column of y")
    val y0  = y(?, 0)                                            // use first column of response matrix y
    val rg0 = new Regression (x, y0)                             // create a Regression model
    rg0.trainNtest ()()                                          // train and test the model
    println (rg0.summary ())                                     // parameter/coefficient statistics

    banner ("regressionMVTest: Compare with Linear Regression - second column of y")
    val y1  = y(?, 1)                                            // use second column of response matrix y
    val rg1 = new Regression (x, y1)                             // create a Regression model
    rg1.trainNtest ()()                                          // train and test the model
    println (rg1.summary ())                                     // parameter/coefficient statistics

    val b_ = mod.parameters(0).w                                 // check for parameter agreements with `Regression`
    assert (b_(?, 0) == rg0.parameter)
    assert (b_(?, 1) == rg1.parameter)

end regressionMVTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionMVTest2` main function tests the `RegressionMV` class using the
 *  Concrete dataset.
 *  > runMain scalation.modeling.neuralnet.regressionMVTest2
 */
@main def regressionMVTest2 (): Unit =

    import Example_Concrete._

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("Concrete RegressionMV")
    val mod = new RegressionMV (ox, y, ox_fname)                 // create model with intercept (else pass x)
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("Concrete Validation Test")
    println (Fit.showFitMap (mod.validate ()()._2))

    banner ("Concrete Cross-Validation Test")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end regressionMVTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionMVTest3` main function tests the `RegressionMV` class using the
 *  AutoMPG dataset.
 *  > runMain scalation.modeling.neuralnet.regressionMVTest3
 */
@main def regressionMVTest3 (): Unit =

    import Example_AutoMPG.{ox, yy, ox_fname}

//  println (s"ox = $ox")
//  println (s"yy = $yy")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG RegressionMV")
    val mod = new RegressionMV (ox, yy, ox_fname)                // create model with intercept (else pass x)
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("AutoMPG Validation Test")
    println (Fit.showFitMap (mod.validate ()()._2))

    banner ("AutoMPG Cross-Validation Test")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end regressionMVTest3
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionMVTest4` main function tests the `RegressionMV` class using the
 *  AutoMPG dataset.  It tests forward selection.
 *  > runMain scalation.modeling.neuralnet.regressionMVTest4
 */
@main def regressionMVTest4 (): Unit =

    import Example_AutoMPG.{ox, yy, ox_fname}

//  println (s"ox = $ox")
//  println (s"y  = $y")
    println (s"ox_fname = ${stringOf (ox_fname)}")

    banner ("AutoMPG RegressionMV")
    val mod = new RegressionMV (ox, yy, ox_fname)                // create model with intercept (else pass x)
    mod.trainNtest ()()                                          // train and test the model
    println (mod.summary ())                                     // parameter/coefficient statistics

    banner ("Feature Selection Technique: Forward")
    val (cols, rSq) = mod.forwardSelAll ()                       // R^2, R^2 bar, smape, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                     // R^2, R^2 bar, smape, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end regressionMVTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionMVTest5` main function tests the `RegressionMV` class using the AutoMPG
 *  dataset.  It tests forward, backward and stepwise selection.
 *  > runMain scalation.modeling.neuralnet.regressionMVTest5
 */
@main def regressionMVTest5 (): Unit =

    import Example_AutoMPG.{ox, yy, ox_fname}

//  println (s"ox = $ox")
//  println (s"y  = $y")

    banner ("AutoMPG RegressionMV")
    val mod = new RegressionMV (ox, yy, ox_fname)                // create model with intercept (else pass x)
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
        new PlotM (null, rSq.transpose, Regression.metrics, s"R^2 vs n for ${mod.modelName} with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end regressionMVTest5

