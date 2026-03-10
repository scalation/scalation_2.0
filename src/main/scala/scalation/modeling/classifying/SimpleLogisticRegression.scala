
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Dec 28 21:52:38 EST 2014
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Simple Logistic Regression
 */

package scalation
package modeling
package classifying

import scala.math.{exp, log}

import scalation.mathstat._
import scalation.optimization.quasi_newton.BFGS

import ActivationFun.{sigmoid, sigmoid_}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleLogisticRegression` class supports (binomial) logistic regression.
 *  In this case, x is two-dimensional [1, x_1].  Fit the parameter vector
 *  b in the logistic regression equation
 *      logit (p_y)  =  b dot x + e  =  b_0 + b_1 * x_1 + e
 *  where e represents the residuals (the part not explained by the model)
 *  and y is now binary.
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  @param x       the input/design matrix augmented with a first column of ones
 *  @param y       the binary response vector, y_i in {0, 1}
 *  @param fname_  the names for all features/variables
 *  @param cname_  the names for both classes
 *  @param hparam  the hyper-parameters
 */
class SimpleLogisticRegression (x: MatrixD, y: VectorI, fname_ : Array [String] = Array ("one", "x1"),
                                cname_ : Array [String] = Array ("No", "Yes"),
                                hparam: HyperParameter = Classifier.hp)
      extends Classifier (x, y, fname_, 2, cname_, hparam)
         with FitC ():

    private val debug = debugf ("SimpleLogisticRegression", true)        // debug function
    private val flaw  = flawf ("SimpleLogisticRegression")               // flaw function

    if y != null && x.dim != y.dim then flaw ("init", "dimensions of x and y are incompatible")

    protected val cThresh = hparam("cThresh")                            // classification/decision threshold

    protected val n       = x.dim2                                       // number of parameters
    protected val k       = n - 1                                        // number of variables (assumes an intercept)
    protected var b       = VectorD.nullv                                // parameter vector (b_0, b_1, ... b_k)
    protected var n_dev   = -1.0                                         // null dev: -2l, for null model (intercept only)
    protected var r_dev   = -1.0                                         // residual dev: -2l, for full model
    protected var aic     = -1.0                                         // Akaike’s Information Criterion

    _modelName = s"SimpleLogisticRegression_$cThresh"                    // name of the model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute McFaffen's pseudo R-squared.
     */
    override def pseudo_rSq: Double = 1.0 - r_dev / n_dev

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a classification model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training dataset.
     *  Train the classifier by fitting the parameter vector (b-vector) in the
     *  logistic regression equation using maximum likelihood.
     *  Do this by minimizing -2l.
     *  FIX: Use improved BFGS implementation or IRWLS
     *  @see stats.stackexchange.com/questions/81000/calculate-coefficients-in-a-logistic-regression-with-r
     *  @see en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        train_null (x_, y_)

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** For a given parameter vector b, compute -2 * Log-Likelihood (-2l).
         *  -2l is the standard measure that follows a Chi-Square distribution. 
         *  @see www.stat.cmu.edu/~cshalizi/350/lectures/26/lecture-26.pdf
         *  @see www.statisticalhorizons.com/wp-content/uploads/Allison.StatComp.pdf
         *  @param b  the parameters to fit
         */
        def ll (b: VectorD): Double =
            var sum = 0.0
            var bx  = 0.0                                                // beta
            for i <- y_.indices do
                bx = b(0) + b(1) * x_(i, 1)
//              sum += y_(i) * bx - log (1.0 + exp (bx))
                sum += y_(i) * bx - bx - log (exp (-bx) + 1.0)           // less prone to overflow (infinity)
            end for
            -2.0 * sum                                                   // set up for minimization
        end ll
   
        val b0   = new VectorD (x_.dim2)                                 // use b_0 = 0 for starting guess for parameters
        val bfgs = new BFGS (ll)                                         // minimizer for -2l
        b     = bfgs.solve (b0)._2                                       // find optimal solution for parameters
        r_dev = ll (b)                                                   // measure of fitness for full model
        aic   = r_dev + 2.0 * x_.dim2
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For the null model, train the classifier by fitting the parameter vector
     *  (b-vector) in the logistic regression equation using maximum likelihood.
     *  Do this by minimizing -2l.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def train_null (x_ : MatrixD = x, y_ : VectorI = y): Unit =

        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        /** For a given parameter vector b = [b(0)], compute -2 * Log-Likelihood (-2l).
         *  -2l is the standard measure that follows a Chi-Square distribution. 
         *  @see www.stat.cmu.edu/~cshalizi/350/lectures/26/lecture-26.pdf
         *  @see www.statisticalhorizons.com/wp-content/uploads/Allison.StatComp.pdf
         *  @param b  the parameters to fit
         */
        def ll_null (b: VectorD): Double =
            var sum = 0.0
            val bx = b(0)                                                // only use the intercept
            for i <- y_.indices do
//              sum += y_(i) * bx - log (1.0 + exp (bx))
                sum += y_(i) * bx - bx - log (exp (-bx) + 1.0)           // less prone to overflow (infinity)
            end for
            -2.0 * sum                                                   // set up for minimization
        end ll_null

        val b0   = new VectorD (x_.dim2)                                 // use b0 = 0 for starting guess for parameters
        val bfgs = new BFGS (ll_null)                                    // minimizer for -2l
        val b_n  = bfgs.solve (b0)._2                                    // find optimal solution for parameters
        n_dev    = ll_null (b_n)                                         // measure of fitness for null model
    end train_null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF vector.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorI = y): (VectorI, VectorD) =
        val yp  = predictI (x_)                                          // predicted classes
        val qof = diagnose (y_.toDouble, yp.toDouble)                    // diagnose from actual and predicted
        debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of coefficient/parameter values.
     */
    override def parameter: VectorD = b

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the quality of fit.  Assumes both train and test methods have already
     *  been called.
     */
    override def fit: VectorD = 
        super.fit ++ VectorD (n_dev, r_dev, aic)
    end fit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the combined labels for the fit.
     */
    def fitLabel: Seq [String] = QoFC.values.map (_.toString).toIndexedSeq ++ Seq ("n_dev", "r_dev", "aic")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict/classify the value of y = f(z) by evaluating the formula y = sigmoid (b dot z).
     *  @param z  the new vector to predict
     */
    override def predictI (z: VectorD): Int =
        val py = sigmoid (b dot z)                                       // P(y = 1|x)
        is (py > cThresh)                                                // compare to classification threshold
    end predictI

    override def predictI (z: VectorI): Int = predictI (z.toDouble)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor x_0, x_1,
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = null, fname_ : Array [String] = null,
                          b_ : VectorD = b, vifs: VectorD = null): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

end SimpleLogisticRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SimpleLogisticRegression` companion object provides factory methods.
 */
object SimpleLogisticRegression:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SimpleLogisticRegression` object. 
     *  @param x1     the vector of values for the predictor
     *  @param y      the binary response vector, y_i in {0, 1}
     *  @param fname  the names for all features/variable
     *  @param cname  the names for both classes
     */
    def apply (x1: VectorD, y: VectorI, fname: Array [String] = Array ("one", "x1"),
               cname: Array [String] = null): SimpleLogisticRegression =
        new SimpleLogisticRegression (MatrixD (VectorD.one (x1.dim), x1).transpose, y, fname, cname)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the help string describing the Quality of Fit (QoF) measues.
     */
    def help: String = FitC.help +
        """
    n_dev =  null model deviance (-2l where l is the log-likelihood)
    r_dev =  full model deviance (-2l where l is the log-likelihood)
    aic   =  Akaike Information Criterion
        """
    end help

end SimpleLogisticRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleLogisticRegressionTest` main function tests the `SimpleLogisticRegression`
 *  class on the mtcars dataset.  Use built-in optimizer.
 *  @see `Example_MTcars`
 *  @see www.cookbook-r.com/Statistical_analysis/Logistic_regression/
 *  Answer: b = (-8.8331, 0.4304),
 *          n_dev = 43.860, r_dev = 25.533, aic = 29.533, pseudo_rSq = 0.4178
 *  > runMain scalation.modeling.classifying.simpleLogisticRegressionTest
 */
@main def simpleLogisticRegressionTest (): Unit =

    val x     = Example_MTcars.xy.not(?, 2)
    val x1    = Example_MTcars.xy(?, 1)
    val y     = Example_MTcars.xy(?, 2).toInt
    val fname = Array ("One", "Mpg")                              // feature names

    println (s"x = $x")
    println (s"y = $y")


    val lrg = new SimpleLogisticRegression (x, y, fname)          // Simple Logistic Regression classifier
    val yp  = lrg.trainNtest ()()._1                              // train and test model
    println (lrg.summary ())                                      // summary statistics

    banner ("classify new instances")
    var z = VectorD (1.0, 15.0)                                   // classify point z = [1, 15]
    println (s"classify ($z) = ${lrg.classify (z)}")
    z = VectorD (1.0, 30.0)                                       // classify point z = [1, 30]
    println (s"classify ($z) = ${lrg.classify (z)}")

/*
    banner ("perform cross-validation")
    println ("acc = " + lrg.crossValidate ())
*/

    new Plot (x1, y.toDouble, null, "y vs. x1")
    new Plot (x1, yp.toDouble, null, "yp vs. x1")
    new Plot (x1, y.toDouble, yp.toDouble, "y and yp (red) vs. x1")

end simpleLogisticRegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleLogisticRegressionTest2` main function tests the `SimpleLogisticRegression`
 *  class on the mtcars dataset.  Use grid search optimizer.
 *  To get a better solution, refine the grid.
 *  @see `Example_MTcars`
 *  @see www.cookbook-r.com/Statistical_analysis/Logistic_regression/
 *  Answer: b = (-8.8331, 0.4304),
 *          n_dev = 43.860, r_dev = 25.533, aic = 29.533, pseudo_rSq = 0.4178
 *  > runMain scalation.modeling.classifying.SimpleLogisticRegressionTest2
 *
@main def simpleLogisticRegressionTest2 (): Unit =

    val x  = Example_MTcars.xy.not (?, 2)
    val x1 = Example_MTcars.xy(?, 1)
    val y  = Example_MTcars.xy(?, 2).toInt

    println (s"x = $x")
    println (s"y = $y")

    val fname = Array ("One", "Mpg")                              // feature names

    val lrg = new SimpleLogisticRegression (x, y, fname)          // Simple Logistic Regression classifier

    banner ("Grid Search: ll(b0, b1) = -2 log-likelihood")
    var ll_min = MAX_VALUE
    var b_min  = VectorD.nullv

    for i <- -90 to -80; j <- 0 to 10 do
        val b = VectorD (i * 0.1, j * 0.1)
        val ll_b = lrg.ll (b)
        println (s"ll ($b) = $ll_b")
        if ll_b < ll_min then { b_min = b; ll_min = ll_b }
    end for
    println (s"ll_min ($b_min) = ${lrg.ll (b_min)}")

end simpleLogisticRegressionTest2
 */


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleLogisticRegressionTest3` main function tests the `SimpleLogisticRegression`
 *  class.  Compare `SimpleLogisticRegressionTest` with `SimpleRegression`.
 *  @see www.cookbook-r.com/Statistical_analysis/Logistic_regression/
 *  Answer: b = (-8.8331, 0.4304),
 *          n_dev = 43.860, r_dev = 25.533, aic = 29.533, pseudo_rSq = 0.4178
 *  > runMain scalation.modeling.classifying.simpleLogisticRegressionTest3
 */
@main def simpleLogisticRegressionTest3 (): Unit =

    // Mpg
    val x1 = VectorD (21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3, 15.2, 10.4, 10.4,
                      14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4)

    // V/S (e.g., V-6 vs. I-4)
    val y = VectorI (0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                     0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1)
    val yd = y.toDouble
    val xx = MatrixD (VectorD.one (x1.dim), x1)                   // add intercept 

    banner ("Simple Regression Results")
    val lrg = SimpleLogisticRegression (x1, y)                    // Simple Logistic Regression classifier
    val yp = lrg.trainNtest ()()._1                               // train and test the model
    println (lrg.summary ())                                      // summary statistics

    banner ("Simple Regression Results")
    val srg = SimpleRegression (x1, yd, null)                     // Simple Regression predictor
    val yq = srg.trainNtest ()()._1                               // train and test the model
    println (srg.summary ())                                      // summary statistics
 
    banner ("Rounded Simple Regression Results")
    val rrg = new RoundRegression (xx, yd)                        // Rounded Regression model
    val yr = srg.trainNtest ()()._1                               // train and test the model
    println (rrg.summary ())                                      // summary statistics

    new Plot (x1, y.toDouble + 0.02, yp.toDouble + 0.04, "Logistic: y and yp (red) vs. x1")
    new Plot (x1, y.toDouble + 0.02, yq,                 "Simple:   y and yq (red) vs. x1")
    new Plot (x1, y.toDouble + 0.02, yr,                 "Rounded:  y and yr (red) vs. x1")

end simpleLogisticRegressionTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleLogisticRegressionTest4` main function tests the `SimpleLogisticRegression`
 *  class.
 *  @see people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html
 *  > runMain scalation.modeling.classifying.simpleLogisticRegressionTest4
 */
@main def simpleLogisticRegressionTest4 (): Unit =

    // features/variable:
    //                     x - curvature
    val x  = VectorD (2.95, 2.53, 3.57, 3.16, 2.58, 2.16, 3.27)
    val y  = VectorI (   0,    0,    0,    0,    1,    1,    1)
    val xx = MatrixD (VectorD.one (x.dim), x).transpose

    val fname = Array ("one", "curvature")                        // feature names
    val cname = Array ("pass", "fail")                            // class names

    val lrg = new SimpleLogisticRegression (xx, y, fname, cname)  // SimpleLogisticRegression classifier
    val yp = lrg.trainNtest ()()._1                               // train and test the classifier
    println (lrg.summary ())                                      // summary statistics

    banner ("classify")
    val z  = VectorD (1.0, 2.81)
    println (s"classify ($z) = ${lrg.classify (z)}")

    val t = VectorD.range (0, x.dim)
    new Plot (t, y.toDouble, yp.toDouble, "y(black)/yp(red) vs. t")
    new Plot (x, y.toDouble, yp.toDouble, "y(black)/yp(red) vs. x")

end simpleLogisticRegressionTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleLogisticRegressionTest5` main function tests the `SimpleLogisticRegression`
 *  class.
 *  > runMain scalation.modeling.classifying.simpleLogisticRegressionTest5
 */
@main def simpleLogisticRegressionTest5 (): Unit =

    // features/variable:
    //                     x - curvature
    val x  = VectorD (1, 2, 3, 4, 5, 6)
    val y  = VectorI (0, 0, 1, 0, 1, 1)
    val xx = MatrixD (VectorD.one (x.dim), x).transpose

    val fname = Array ("one", "x1")                               // feature names
    val cname = Array ("pass", "fail")                            // class names

    val lrg = new SimpleLogisticRegression (xx, y, fname, cname)  // SimpleLogisticRegression classifier
    val yp = lrg.trainNtest ()()._1                               // train and test the classifier
    println (lrg.summary ())                                      // summary statistics

    banner ("classify")
    val z = VectorD (1.0, 2.81)
    println (s"classify ($z) = ${lrg.classify (z)}")

    val t = VectorD.range (0, x.dim)
    new Plot (t, y.toDouble, yp.toDouble, "y(black)/yp(red) vs. t")
    new Plot (x, y.toDouble, yp.toDouble, "y(black)/yp(red) vs. x")

end simpleLogisticRegressionTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `simpleLogisticRegressionTest6` main function tests the logistic function.
 *  > runMain scalation.modeling.classifying.simpleLogisticRegressionTest6
 */
@main def simpleLogisticRegressionTest6 (): Unit =

    val z  = VectorD.range (0, 160) / 10.0 - 8.0
    val fz = sigmoid_ (z)
    new Plot (z, fz)

end simpleLogisticRegressionTest6

