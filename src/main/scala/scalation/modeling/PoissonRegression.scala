
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jan 11 19:05:20 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Poisson Regression
 *
 *  ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Poisson_Regression.pdf
 */

// U N D E R   D E V E L O P M E N T 

// FIX: needs improved optimization

package scalation
package modeling

import scala.math.{exp, round}

import scalation.mathstat._
import scalation.optimization.quasi_newton.BFGS

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PoissonRegression` class supports Poisson regression.  In this case, 
 *  x may be multi-dimensional [1, x_1, ... x_k].  Fit the parameter
 *  vector 'b' in the Poisson regression equation
 *      log (mu(x))  =  b dot x  =  b_0 + b_1 * x_1 + ... b_k * x_k
 *  where e represents the residuals (the part not explained by the model)
 *  and y is now integer valued.
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  @param x       the data/input matrix augmented with a first column of ones
 *  @param y       the integer response/output vector, y_i in {0, 1, ... }
 *  @param fname_  the names of the features/variables (defaults to null)
 *  @param hparam  the hyper-parameters (currently has none)
 */
class PoissonRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                         hparam: HyperParameter = null)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):

/*
    private val k          = x.dim2 - 1                         // number of variables 
    private val n          = x.dim.toDouble                     // number of data points (rows)
    private val r_df       = (n-1.0) / (n-k-1.0)                // ratio of degrees of freedom
*/
    private val debug      = debugf ("PoissonRegression", true) // debug function
    private var aic        = -1.0                               // Akaike’s Information Criterion
    private var n_dev      = -1.0                               // null dev: -LL, for null model (intercept only)
    private var r_dev      = -1.0                               // residual dev: -LL, for full model
    private var pseudo_rSq = -1.0                               // McFaffen's pseudo R-squared

    _modelName = "PoissonRegression"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For a given parameter vector 'b', compute '-Log-Likelihood' (-LL).
     *  '-LL' is the standard measure.
     *  @see dept.stat.lsa.umich.edu/~kshedden/Courses/Stat600/Notes/glm.pdf
     *  @param b  the parameters to fit
     */
    def ll (b: VectorD): Double =
        var sum = 0.0
        for i <- x.indices do
            val bx = b dot x(i)
            sum += y(i) * bx - exp (bx)                         // last term not needed [ - log (fac (y(i))) ]
        end for
        -sum                                                    // set up for minimization
    end ll
   
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For a given parameter vector 'b = [b(0)], compute -2 * Log-Likelihood' (-2LL).
     *  '-2LL' is the standard measure that follows a Chi-Square distribution. 
     *  @see dept.stat.lsa.umich.edu/~kshedden/Courses/Stat600/Notes/glm.pdf
     *  @param b  the parameters to fit
     */
    def ll_null (b: VectorD): Double =
        var sum = 0.0
        for i <- x.indices do
            val bx = b(0)                                        // only use the intercept
            sum += y(i) * bx - exp (bx)                          // last term not needed [ - log (fac (y(i))) ]
        end for
        - sum                                                    // set up for minimization
    end ll_null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For the full model, train the predictor by fitting the parameter vector
     *  (b-vector) in the Poisson regression equation using maximum likelihood.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y.toDouble): Unit =
         // FIX - currently only works for x_ = x and y_ = y
         train_null ()
         val b0   = new VectorD (x_.dim2)                        // use b_0 = 0 for starting guess for parameters
         val bfgs = new BFGS (ll)                                // minimizer for -2LL

         b     = bfgs.solve (b0)._2                              // find optimal solution for parameters
         r_dev = ll (b)                                          // measure of fitness for full model
         aic   = r_dev + 2.0 * x_.dim2
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For the null model, train the classifier by fitting the parameter vector
     *  (b-vector) in the Poisson regression equation using maximum likelihood.
     *  Do this by minimizing '-2LL'.
     */
    def train_null (): Unit =
         val b0   = new VectorD (x.dim2)                         // use b0 = 0 for starting guess for parameters
         val bfgs = new BFGS (ll_null)                           // minimizer for -2LL

         val b_n = bfgs.solve (b0)._2                            // find optimal solution for parameters
         n_dev   = ll_null (b_n)                                 // measure of fitness for null nodel
    end train_null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                    // make predictions
        (yp, diagnose (y_, yp))                                  // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the quality of fit including 'rSquared'.  Assumes both train_null and
     *  train have already been called.
     */
    override def fit: VectorD = 
        pseudo_rSq = 1.0 - r_dev / n_dev
        VectorD (n_dev, r_dev, aic, pseudo_rSq)
    end fit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the labels for the fit.
     *  FIX
    override def fitLabel: Seq [String] = Seq ("n_dev", "r_dev", "aic", "pseudo_rSq")
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Classify the value of 'y = f(z)' by evaluating the formula 'y = exp (b dot z)'.
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): Double = round (exp (b dot z)).toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): PoissonRegression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new PoissonRegression (x_cols, y, fname2)
    end buildModel

end PoissonRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PoissonRegression` companion object provides factory methods for creating
 *  Poisson regression models.
 */
object PoissonRegression:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `PoissonRegression` object from a combined data matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (currently has none)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = null)
              (col: Int = xy.dim2 - 1): PoissonRegression =
        val (x, y) = (xy.not (?, col), xy(?, col))
        new PoissonRegression (x, y, fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `PoissonRegression` object given an integer response vector.
     *  @param x       the data matrix
     *  @param y       the response vector as an integer vector
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters (currently has none)
     */
    def apply (x: MatrixD, y: VectorI, fname: Array [String],
               hparam: HyperParameter): PoissonRegression =
        new PoissonRegression (x, y.toDouble, fname, hparam)
    end apply

end PoissonRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `poissonRegressionTest` main function tests the `PoissonRegression` class.
 *  @see www.cookbook-r.com/Statistical_analysis/Logistic_regression/
 *  Answer: b = (-8.8331, 0.4304),
 *          n_dev = 43.860, r_dev = 25.533, aci = 29.533, pseudo_rSq = 0.4178
 *  > runMain scalation.modeling.poissonRegressionTest
 */
@main def poissonRegressionTest (): Unit =

    // 32 data points:        One    Mpg
    val x = MatrixD ((32, 2), 1.0,  21.0,        //  1 - Mazda RX4 
                              1.0,  21.0,        //  2 - Mazda RX4 Wa
                              1.0,  22.8,        //  3 - Datsun 710
                              1.0,  21.4,        //  4 - Hornet 4 Drive
                              1.0,  18.7,        //  5 - Hornet Sportabout
                              1.0,  18.1,        //  6 - Valiant
                              1.0,  14.3,        //  7 - Duster 360
                              1.0,  24.4,        //  8 - Merc 240D 
                              1.0,  22.8,        //  9 - Merc 230
                              1.0,  19.2,        // 10 - Merc 280
                              1.0,  17.8,        // 11 - Merc 280C
                              1.0,  16.4,        // 12 - Merc 450S
                              1.0,  17.3,        // 13 - Merc 450SL
                              1.0,  15.2,        // 14 - Merc 450SLC
                              1.0,  10.4,        // 15 - Cadillac Fleetwood
                              1.0,  10.4,        // 16 - Lincoln Continental
                              1.0,  14.7,        // 17 - Chrysler Imperial
                              1.0,  32.4,        // 18 - Fiat 128
                              1.0,  30.4,        // 19 - Honda Civic
                              1.0,  33.9,        // 20 - Toyota Corolla
                              1.0,  21.5,        // 21 - Toyota Corona
                              1.0,  15.5,        // 22 - Dodge Challenger
                              1.0,  15.2,        // 23 - AMC Javelin
                              1.0,  13.3,        // 24 - Camaro Z28
                              1.0,  19.2,        // 25 - Pontiac Firebird
                              1.0,  27.3,        // 26 - Fiat X1-9
                              1.0,  26.0,        // 27 - Porsche 914-2
                              1.0,  30.4,        // 28 - Lotus Europa
                              1.0,  15.8,        // 29 - Ford Pantera L
                              1.0,  19.7,        // 30 - Ferrari Dino
                              1.0,  15.0,        // 31 - Maserati Bora
                              1.0,  21.4)        // 32 - Volvo 142E

    val y = VectorI (0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                     0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1)

    var z: VectorD = null

    println ("x = " + x)
    println ("y = " + y)

    val rg = PoissonRegression (x, y, null, null)
    rg.train_null ()                                    // train based on null model
    rg.trainNtest ()()                                  // train based on full model

    val yp = rg.predict (x)
    println ("y  = " + y)
    println ("yp = " + yp.toInt)

    z = VectorD (1.0, 15.0)                            // predict point z
    println ("predict (" + z + ") = " + rg.predict (z))

    z = VectorD (1.0, 30.0)                            // predict point z
    println ("predict (" + z + ") = " + rg.predict (z))

end poissonRegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `poissonRegressionTest2` main function tests the `PoissonRegression` class.
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  @see www.stat.wisc.edu/~mchung/teaching/.../GLM.logistic.Rpackage.pdf
 *  > runMain scalation.modeling.poissonRegressionTest2
 */
@main def poissonRegressionTest2 (): Unit =

    // 40 data points:        One     Low  Medium    High
    val x = MatrixD ((40, 4), 1.0,  102.0,   89.0,    0.0,
                              1.0,    7.0,  233.0,    1.0,
                              1.0,    0.0,    4.0,   41.0,
                              1.0,    8.0,   37.0,   13.0,
                              1.0,   40.0,   79.0,   26.0,
                              1.0,    0.0,  625.0,  156.0,
                              1.0,    0.0,   12.0,   79.0,
                              1.0,    0.0,    3.0,  119.0,
                              1.0,  115.0,  136.0,   65.0,
                              1.0,  428.0,  416.0,  435.0,
                              1.0,   34.0,  174.0,   56.0,
                              1.0,    0.0,    0.0,   37.0,
                              1.0,   97.0,  162.0,   89.0,
                              1.0,   56.0,   47.0,  132.0,
                              1.0, 1214.0, 1515.0,  324.0,
                              1.0,   30.0,  103.0,  161.0,
                              1.0,    8.0,   11.0,  158.0,
                              1.0,   52.0,  155.0,  144.0,
                              1.0,  142.0,  119.0,   24.0,
                              1.0, 1370.0, 2968.0, 1083.0,
                              1.0,  790.0,  161.0,  231.0,
                              1.0, 1142.0,  157.0,  131.0,
                              1.0,    0.0,    2.0,   49.0,
                              1.0,    0.0,    0.0,   50.0,
                              1.0,    5.0,   68.0,   49.0,
                              1.0,    0.0,    0.0,   48.0,
                              1.0,    0.0,    6.0,   40.0,
                              1.0,    1.0,    8.0,   64.0,
                              1.0,    0.0,  998.0,  551.0,
                              1.0,  253.0,   99.0,   60.0,
                              1.0, 1395.0,  799.0,  244.0,
                              1.0,    0.0,    0.0,   50.0,
                              1.0,    1.0,   68.0,  145.0,
                              1.0, 1318.0, 1724.0,  331.0,
                              1.0,    0.0,    0.0,   79.0,
                              1.0,    3.0,   31.0,   37.0,
                              1.0,  195.0,  108.0,  206.0,
                              1.0,    0.0,   15.0,  121.0,
                              1.0,    0.0,  278.0,  513.0,
                              1.0,    0.0,    0.0,  253.0)

    val y = VectorI (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                     1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1)

    val fn = Array ("One", "Low", "Medium", "High")

    println ("x = " + x)
    println ("y = " + y)

//  val rg = PoissonRegression (x(0 until x.dim, 0 until 2), y, fn)
    val rg = PoissonRegression (x, y, fn, null)
    rg.train_null ()                                    // train based on null model
    rg.trainNtest ()()                                  // train based on full model

    val z = VectorD (1.0, 100.0, 100.0, 100.0)          // predict point z
    println ("predict (" + z + ") = " + rg.predict (z))

//  new Plot (x.col(1), y, yyp)
//  new Plot (x.col(2), y, yyp)

end poissonRegressionTest2

