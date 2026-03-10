
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jan  9 21:48:57 EST 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Linear Discriminant Analysis (LDA) Classifier
 *
 *  FIX - extend the code to work for k > 2.
 */

package scalation
package modeling
package classifying

import scala.math.log

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LinDiscAnalyis` class implements a Linear Discriminant Analysis (LDA) classifier.
 *  It places a vector into a group according to its maximal discriminant function.
 *  FIX - currently only works when the number of classes k = 2.
 *  @see en.wikipedia.org/wiki/Linear_discriminant_analysis
 *  @param x       the real-valued training/test data vectors stored as rows of a matrix
 *  @param y       the training/test classification vector, where y_i = class for row i of the matrix x
 *  @param fname_  the names for all features/variables
 *  @param k       the number of classes (k in {0, 1, ...k-1}
 *  @param cname_  the names for all classes
 *  @param hparam  the hyper-parameters
 */
class LinDiscAnalyis (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
           cname_ : Array [String] = Array ("No", "Yes"),
           hparam: HyperParameter = Classifier.hp)
      extends Classifier (x, y, fname_, k, cname_, hparam)
         with FitC (k):

    private val debug = debugf ("LinDiscAnalyis", true)                  // debug function
    private val flaw  = flawf ("LinDiscAnalyis")                         // flaw function

    private var mu: (VectorD, VectorD, VectorD) = null                   // column means: x1, x2, x
    private var pcovar: MatrixD  = null                                  // pooled covariance matrix
    private var ipcovar: MatrixD = null                                  // inverse of pooled covariance matrix
    private var prior: VectorD   = null                                  // prior probabilities

    if k != 2 then flaw ("init", "k must equal 2 in current implementation")

    _modelName = "LinDiscAnalyis"                                        // name of the model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the corrected covariance matrix.
     *  @param xc  the corrected martix whose corrected covariance matrix is sought 
     */
    def corrected_cov (xc: MatrixD): MatrixD = (xc.transpose * xc) / xc.dim

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the classifer by computing pcovar, ipcovar and prior
     *  that are needed to compute the discriminant functions f.
     *  These are computed in the predictI/classify method.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        val x1 = (MatrixD (for i <- x.indices if y_(i) == 0 yield x_(i)))   // group 1
        val x2 = (MatrixD (for i <- x.indices if y_(i) == 1 yield x_(i)))   // group 2

        mu  = (x1.mean, x2.mean, x_.mean)                                // columns means: x1, x2, x_
        val xc0 = x1 - mu._3                                             // corrected group matrix 1
        val xc1 = x2 - mu._3                                             // corrected group matrix 2

//      debug ("train", s" x1 = $x1 \n x2 = $x2 \n xc0 = $xc0 \n xc1 = $xc1 \n mu = $mu")

        val w1    = x1.dim / x.dim.toDouble                              // first weigth
        val w2    = 1.0 - w1                                             // second weigth
        val covar = (corrected_cov (xc0), corrected_cov (xc1))           // corrected covariances
        prior     = VectorD (w1, w2)                                     // prior probabilities
        pcovar    = covar._1 * w1 + covar._2 * w2                        // pooled covariance matrix
        ipcovar   = Fac_LU.inverse (pcovar)()                            // inverse of pooled covariance matrix   

        debug ("train", s" covar = $covar \n pcovar = $pcovar \n ipcovar = $ipcovar \n prior = $prior")
    end train

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
        val qof = diagnose (y_, yp)                                      // diagnose from actual and predicted
//      debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict/classify vector z by computing its discriminant function f for each
     *  group and return the group index with the maximun value for f.
     *  @param z  the new vector to predict
     */
    override def predictI (z: VectorD): Int =
        val fvec = (ipcovar * mu._1, ipcovar * mu._2)
        val f    = ((fvec._1 dot z) - 0.5 * (fvec._1 dot mu._1) + log (prior(0)),
                    (fvec._2 dot z) - 0.5 * (fvec._2 dot mu._2) + log (prior(1)))
        is (f._1 < f._2)
    end predictI

    override def predictI (z: VectorI): Int = predictI (z.toDouble)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for predictor x_1,
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = null, fname_ : Array [String] = null,
                          b_ : VectorD = parameter, vifs: VectorD = null): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

end LinDiscAnalyis


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LinDiscAnalyis` companion object provides a factory method.
 */
object LinDiscAnalyis:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a LinDiscAnalyis model for the given combined matrix where the column col is
     *  the response/classification vector.
     *  @param xy      the combined data matrix (features and response)
     *  @param fname   the names for all features/variables
     *  @param k       the number of classes (k in {0, 1, ...k-1}
     *  @param cname   the names for all classes
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null, k: Int = 2,
               cname: Array [String]  = Array ("No", "Yes"),
               hparam: HyperParameter = Classifier.hp)
              (col: Int = xy.dim2 - 1): LinDiscAnalyis =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new LinDiscAnalyis (x, y, fname, k, cname, hparam)
    end apply

end LinDiscAnalyis


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `linDiscAnalyisTest` main function is used to test the `LinDiscAnalyis` class.
 *  @see people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html
 *  > runMain scalation.modeling.classifying.linDiscAnalyisTest
 */
@main def linDiscAnalyisTest (): Unit =

    // features/variable: 
    // x1: curvature
    // x2: diameter
    //                       x1    x2
    val x = MatrixD ((7, 2), 2.95, 6.63,
                             2.53, 7.79,
                             3.57, 5.65,
                             3.16, 5.47,
                             2.58, 4.46,
                             2.16, 6.22,
                             3.27, 3.52)
    val y = VectorI (0, 0, 0, 0, 1, 1, 1)

    val fname = Array ("curvature", "diameter")                          // feature names
    val cname = Array ("pass", "fail")                                   // class names

    val lda = new LinDiscAnalyis (x, y, fname, 2, cname)                 // create the LDA classifier
    lda.trainNtest ()()                                                  // train the classifier
    println (lda.summary ())

    banner ("classify")
    val z  = VectorD (2.81, 5.46)
    println (s"classify ($z) = ${lda.classify (z)}")

end linDiscAnalyisTest

