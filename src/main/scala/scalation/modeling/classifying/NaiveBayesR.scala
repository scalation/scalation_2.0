
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Sep  8 13:53:16 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Gaussian Naive Bayes Classifier
 */

package scalation
package modeling
package classifying

import scala.math.{ceil, floor}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
import scalation.random.Normal

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NaiveBayesR` class implements a Gaussian Naive Bayes Classifier, which
 *  is the most commonly used such classifier for continuous input data.  The
 *  classifier is trained using a data matrix x and a classification vector y.
 *  Each data vector in the matrix is classified into one of k classes numbered
 *  0, ..., k-1.  Class probabilities are calculated based on the frequency of
 *  each class in the training-set.  Relative probabilities are computed  by
 *  multiplying these by values computed using conditional density functions
 *  based on the Normal (Gaussian) distribution.  The classifier is naive, because
 *  it assumes feature independence and therefore simply multiplies the conditional
 *  densities.
 *  @param x       the real-valued data vectors stored as rows of a matrix
 *  @param y       the class vector, where y_i = class for row i of the matrix x, x(i)
 *  @param fname_  the names for all features/variables
 *  @param k       the number of classes
 *  @param cname_  the names for all classes
 *  @param hparam  the hyper-parameters
 */
class NaiveBayesR (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
                   cname_ : Array [String] = Array ("No", "Yes"),
                   hparam: HyperParameter = NaiveBayes.hp)
      extends Classifier (x, y, fname_, k, cname_, hparam)
         with FitC (k):

    private val debug   = debugf ("NaiveBayesR", true)                   // debug function
    private val EPSILON = 1E-9                                           // number close to zero
    private val cor     = x.corr                                         // feature correlation matrix

    private val nu_y = new VectorD (k)                                   // frequency counts for classes 0, ..., k-1
    private val mean = new MatrixD (k, x.dim2)                           // mean for each class, feature
    private val varc = new MatrixD (k, x.dim2)                           // variance for each class, feature

    private val cd   = Array.ofDim [Double => Double] (k, x.dim2)        // conditional density functions

    _modelName = "NaiveBayesR"                                           // name of the model

    debug ("init", s"correlation matrix = $cor")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate statistics (sample mean and sample variance) for each class *  by feature.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def calcStats (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        for i <- y_.indices do                                           // for each data vector in training-set
            val c    = y_(i)                                             // given classification for ith data vector
            nu_y(c) += 1.0                                               // count the number in each class
            for j <- x_.indices2 do                                      // for each feature
                val d = x_(i, j)                                         // jth data value
                mean(c, j) += d                                          // running total for sum
                varc(c, j) += d * d                                      // running total for sum of squares
        end for 
    
        for c <- 0 until k do                                            // for each class
            val mc = nu_y(c)                                             // frequency of class c in training-set
            for j <- x_.indices2 do                                      // for each feature
                mean(c, j) /= mc                                         // compute mean
                val mean_cj = mean(c, j)
                varc(c, j)  = (varc(c, j) - mc * mean_cj*mean_cj) / (mc - 1.0)   // compute variance - FIX - check
        end for
    
        debug ("calcStats", s"fname = ${stringOf (fname)}, nu_y = $nu_y, mean = $mean, varc = $varc")
    end calcStats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the counts for each interval in the histogram.
     *  @param x_j  the vector for feature j given class c.
     *  @param intervals  the number intervals
     */
    def calcHistogram (x_j: VectorD, intervals: Int): VectorD =
        val minVal = floor (x_j.min)
        val maxVal = ceil (x_j.max + EPSILON)
        val intWid = (maxVal - minVal) / intervals.toDouble
        val h      = new VectorD (intervals)
        for xx <- x_j do
            val i = (floor ((xx - minVal) / intWid)).toInt
            h(i) += 1.0
        end for
        h
    end calcHistogram

    // use Discrete distribution based on histogram

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the classifier, i.e., calculate statistics and create conditional
     *  density cd functions.  Assumes that conditional densities follow the
     *  Normal (Gaussian) distribution.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        super.train (x_, y_)                                             // set class frequencies nu_y and probabilities p_y
        calcStats (x_, y_)
        for c <- 0 until k; j <- x_.indices2 do
            cd(c)(j) = (z_j => Normal (mean(c, j), varc(c, j)).pf (z_j))
        end for
        p_yz = nu_y / y_.dim.toDouble                                    // probability = class frequency / training-set size
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
        val qof = diagnose (y_.toDouble, yp.toDouble)                    // diagnose from actual and predicted
        debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the integer value of y = f(z) by computing the product of the class
     *  probabilities p_y and all the conditional probabilities P(X_j = z_j | y = c)
     *  and returning the class with the highest relative probability.
     *  Note, p_yz from `Classifier` holds the relative probabilities of y given z.
     *  @param z  the new vector to predict
     */
    override def predictI (z: VectorD): Int =
        p_yz = nu_y / p_yz.dim.toDouble                                  // reset probabilities
        for c <- 0 until k; j <- x.indices2 do p_yz(c) *= cd(c)(j)(z(j))
        p_yz.argmax ()                                                   // return class with highest probability
    end predictI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor x_0, x_1,
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = null, fname_ : Array [String] = null,
                          b_ : VectorD = p_y, vifs: VectorD = null): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

end NaiveBayesR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** `NaiveBayesR` is the companion object for the `NaiveBayesR` class.
 */
object NaiveBayesR:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NaiveBayesR` object, passing x and y together in one matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the names of the features/variables
     *  @param k       the number of classes
     *  @param cname   the names of the classes
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null, k: Int = 2,
               cname: Array [String] = Array ("No", "Yes"),
               hparam: HyperParameter = NaiveBayes.hp)
              (col: Int = xy.dim2 - 1): NaiveBayesR =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new NaiveBayesR (x, y, fname, k, cname, hparam)
    end apply

end NaiveBayesR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesRTest` main function is used to test the `NaiveBayesR` class.
 *  @see people.revoledu.com/kardi/tutorial/LDA/Numerical%20Example.html
 *  > runMain scalation.modeling.classifying.naiveBayesRTest
 */
@main def naiveBayesRTest (): Unit =

    // features/variable:
    // x1: curvature
    // x2: diameter
    // y:  classification: pass (0), fail (1)
    //                        x1    x2    y
    val xy = MatrixD ((7, 3), 2.95, 6.63, 0,
                              2.53, 7.79, 0,
                              3.57, 5.65, 0,
                              3.16, 5.47, 0,
                              2.58, 4.46, 1,
                              2.16, 6.22, 1,
                              3.27, 3.52, 1)

    val x = xy.not (?, 2)
    val y = xy(?, 2).toInt
    println (s"x = $x")
    println (s"y = $y")

    val fname = Array ("curvature", "diameter")                    // feature names
    val cname = Array ("pass", "fail")                             // class names

    val nbr = NaiveBayesR (xy, fname, 2, cname)()                  // create NaiveBayesR classifier
    nbr.trainNtest ()()
    println (nbr.summary ())

    banner ("classify")
    val z  = VectorD (2.81, 5.46)
    println (s"classify ($z) = ${nbr.classify (z)}")

end naiveBayesRTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesRTest2` main function is used to test the `NaiveBayesR` class.
 *  Ex: Classify whether a person is male (M) or female (F) based on the measured features.
 *  @see en.wikipedia.org/wiki/Naive_Bayes_classifier
 *  > runMain scalation.modeling.classifying.naiveBayesRTest2
 */
@main def naiveBayesRTest2 (): Unit =

    // training-set -----------------------------------------------------------
    // x0: Height
    // x1: Weight
    // x2: Foot-size
    // features:               x0       x1     x2
    val x = MatrixD ((8, 3), 6.00,   180.0,  12.0,                 // data matrix
                             5.92,   190.0,  11.0,
                             5.58,   170.0,  12.0,
                             5.92,   165.0,  10.0,
                             5.00,   100.0,   6.0,
                             5.50,   150.0,   8.0,
                             5.42,   130.0,   7.0,
                             5.75,   150.0,   9.0)
    val y     = VectorI (0, 0, 0, 0, 1, 1, 1, 1)                   // classification vector: 0(M), 1(F))
    val fname = Array ("Height", "Weight", "Foot-size")            // feature/value names
    val cname = Array ("M", "F")                                   // class names

    println (s"x = $x")

    val nbr = new NaiveBayesR (x, y, fname, 2, cname)              // create the classifier            

    // check independence assumption ------------------------------------------
    println (s"cor = ${x.corr}")

    // train and test the classifier ---------------------------------------------------
    nbr.trainNtest ()()
    println (nbr.summary ())

    // test sample ------------------------------------------------------------
    val z = VectorD (6.0, 130, 8.0)                                // new data vector to classify
    println (s"classify ($z) = ${nbr.classify (z)}")

end naiveBayesRTest2

