
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Dec 28 21:52:38 EST 2014
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Logistic Regression
 */

package scalation
package modeling
package classifying

import scala.math.{exp, log}

import scalation.mathstat._
import scalation.optimization.quasi_newton.BFGS

// FIX: needs improved optimization

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LogisticRegression` class supports (binomial) logistic regression.  In this
 *  case, x may be multi-dimensional [1, x_1, ... x_k].  Fit the parameter
 *  vector b in the logistic regression equation
 *      logit (p_y)  =  b dot x + e  =  b_0 + b_1 * x_1 + ... b_k * x_k + e
 *  where e represents the residuals (the part not explained by the model)
 *  and y is now binary.
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  @param x       the input/design matrix augmented with a first column of ones
 *  @param y       the binary response vector, y_i in {0, 1}
 *  @param fname_  the names for all features/variables
 *  @param cname_  the names for both classes
 *  @param hparam  the hyper-parameters
 */
class LogisticRegression (x: MatrixD, y: VectorI, fname_ : Array [String] = null,
                          cname_ : Array [String] = Array ("No", "Yes"),
                          hparam: HyperParameter = Classifier.hp)
      extends SimpleLogisticRegression (x, y, fname_, cname_, hparam):

    private val debug = debugf ("LogisticRegression", true)              // debug function

    private val r_df  = (n-1.0) / (n-k-1.0)                              // ratio of degrees of freedom

    _modelName = s"LogisticRegression_$cThresh"                          // name of the model

    debug ("init", s"r_df = $r_df")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For the full model, train the classifier by fitting the parameter vector
     *  (b-vector) in the logistic regression equation using maximum likelihood.
     *  Do this by minimizing -2l.
     *  FIX: Use improved BFGS implementation or IRWLS
     *  @see stats.stackexchange.com/questions/81000/calculate-coefficients-in-a-logistic-regression-with-r
     *  @see en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
     *  @param itestStart  the indices of test test data
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
            for i <- y_.indices do
                val bx = b dot x_(i)
    //          sum += y_(i) * bx - log (1.0 + exp (bx))
                sum += y_(i) * bx - bx - log (exp (-bx) + 1.0)           // less prone to overflow (infinity)
            end for
            -2.0 * sum                                                   // set up for minimization
        end ll
   
        val b0   = new VectorD (x_.dim2)                                 // use b_0 = 0 for starting guess for parameters
        val bfgs = new BFGS (ll)                                         // minimizer for -2l
        b        = bfgs.solve (b0)._2                                    // find optimal solution for parameters
        r_dev    = ll (b)                                                // measure of fitness for full model
        aic      = r_dev + 2.0 * x.dim2
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Variance Inflation Factor (VIF) for each variable to test
     *  for multi-collinearity by regressing xj against the rest of the variables.
     *  A VIF over 20 indicates that over 95% of the variance of xj can be predicted
     *  from the other variables, so xj is a candidate for removal from the model.
     *  FIX or remove
     */
    def vif: VectorD = ???
/*
        val vifV = new VectorD (k)                                 // VIF vector
        for (j <- 1 to k) {
            val keep = m                                           // i-value large enough to not exclude any rows in slice
            val x_j  = x.col(j)                                    // x_j is jth column in x
            val rg_j = new Regression (x.sliceEx (keep, j), x_j)   // regress with x_j removed
            rg_j.train ()
            vifV(j-1) =  1.0 / (1.0 - rg_j.fit(index_rSq))         // store vif for x_1 in vifV(0)
        } // for
        vifV
    } // vif
*/

end LogisticRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LogisticRegression` companion object provides a factory method.
 */
object LogisticRegression:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a logistic regression model for the given combined matrix where the
     *  column col is the response/classification vector.
     *  @param xy      the combined data matrix (features and response)
     *  @param fname   the names for all features/variables
     *  @param cname   the names for all classes
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               cname: Array [String]  = Array ("No", "Yes"),
               hparam: HyperParameter = Classifier.hp)
              (col: Int = xy.dim2 - 1): LogisticRegression =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new LogisticRegression (x, y, fname, cname, hparam)
    end apply

end LogisticRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `logisticRegressionTest` main function tests the `LogisticRegression` class
 *  on the mtcars dataset.
 *  @see Example_MTcars.scala
 *  @see www.cookbook-r.com/Statistical_analysis/Logistic_regression/
 *  Answer: b = (-8.8331, 0.4304),
 *          n_dev = 43.860, r_dev = 25.533, aci = 29.533, pseudo_rSq = 0.4178
 *  > runMain scalation.modeling.classifying.logisticRegressionTest
 */
@main def logisticRegressionTest (): Unit =

    val x     = Example_MTcars.xy.not(?, 2)
    val x1    = Example_MTcars.xy(?, 1)
    val y     = Example_MTcars.xy(?, 2).toInt
    val fname = Array ("One", "Mpg")                              // feature names

    println (s"x = $x")
    println (s"y = $y")

    val lrg = new LogisticRegression (x, y, fname)                // Logistic Regression classifier
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

end logisticRegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `logisticRegressionTest2` main function tests the `LogisticRegression` class.
 *  @see statmaster.sdu.dk/courses/st111/module03/index.html
 *  @see www.stat.wisc.edu/~mchung/teaching/.../GLM.logistic.Rpackage.pdf
 *  > runMain scalation.modeling.classifying.logisticRegressionTest2
 */
@main def logisticRegressionTest2 (): Unit =

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

    val fname = Array ("One", "Low", "Medium", "High")
    val cname = Array ("No", "Yes")

    println (s"x = $x")
    println (s"y = $y")

    banner ("Logistic Regression classifier")
    val lrg = new LogisticRegression (x, y, fname, cname)
    val yp  = lrg.trainNtest ()()._1                              // train and test model
    println (lrg.summary ())                                      // summary statistics

    banner ("classify new instances")
    val z  = VectorD (1.0, 100.0, 100.0, 100.0)                   // classify point z
    println (s"classify ($z) = ${lrg.classify (z)}")

    new Plot (null, y.toDouble, null, "y vs. t")
    new Plot (null, yp.toDouble, null, "yp vs. t")
    new Plot (null, y.toDouble, yp.toDouble, "y and yp (red) vs. t")

end logisticRegressionTest2

