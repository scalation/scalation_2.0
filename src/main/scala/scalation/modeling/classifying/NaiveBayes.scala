
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Sat Sep  8 13:53:16 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Integer-Based Naive Bayes Classifier
 *
 *  @see eric.univ-lyon2.fr/~ricco/tanagra/fichiers/en_Tanagra_Naive_Bayes_Classifier_Explained.pdf
 */

package scalation
package modeling
package classifying

import scalation.mathstat._

import Classifier.{shift2zero, vc_fromData}
import Probability.plog

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NaiveBayes` class implements an Integer-Based Naive Bayes Classifier,
 *  which is a commonly used such classifier for discrete input data.  The
 *  classifier is trained using a data matrix x and a classification vector y.
 *  Each data vector in the matrix is classified into one of k classes numbered
 *  0, ..., k-1.  Prior probabilities are calculated based on the population of
 *  each class in the training-set.  Relative posterior probabilities are computed
 *  by multiplying these by values computed using conditional probabilities.
 *  stored in Conditional Probability Tables (CPTs).
 *------------------------------------------------------------------------------
 *  The classifier is naive, because it assumes variable/feature independence and
 *  therefore simply multiplies the conditional probabilities.
 *  @param x       the input/data m-by-n matrix with instances stored in rows
 *  @param y       the response/classification m-vector, where y_i = class for row i of matrix x
 *  @param fname_  the name for each feature/variable xj
 *  @param k       the number of classes
 *  @param cname_  the name for each class
 *  @param vc      the value count (number of distinct values) for each feature/variable xj
 *  @param hparam  the hyper-parameters
 */
class NaiveBayes (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
                  cname_ : Array [String] = Array ("No", "Yes"), private var vc: VectorI = null,
                  hparam: HyperParameter = NaiveBayes.hp)
      extends Classifier (x, y, fname_, k, cname_, hparam)
         with FitC (k):

    private val debug = debugf ("NaiveBayes", true)                      // debug function

    _modelName = "NaiveBayes"                                            // name of the model

    if vc == null then
        shift2zero (x); vc = vc_fromData (x)                             // set value counts from data

    private val me   = hparam("me").toDouble                             // m-estimates (me == 0 => regular MLE estimates)
    private val me_v = NaiveBayes.me_vc (me, vc)                         // for Laplace smoothing: me / vc_j for all j
    private var p_Xy: RTensorD = null                                    // Conditional Probability Tables (CPTs) one per feature

    debug ("init", s"distinct value count vc = $vc")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the Conditional Probability Tables (CPTs) stored in tensor p_Xy.
     *  Must call train first to get values for p_Xy.
     */
    def getCPTs: RTensorD = p_Xy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a classification model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training set
     *  Training involves estimating the model parameters or pmf.
     *  Train the classifier by computing the class probabilities for y, and
     *  the conditional probabilities for each x_j given y.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        super.train (x_, y_)                                             // set class frequencies nu_y and probabilities p_y
        val nu_Xy = RTensorD.freq (x_, vc, y, k)                         // Joint Frequency Tables (JFTs)
        p_Xy      = cProb_Xy (x_, nu_Xy)                                 // Conditional Probability Tables (CPTs)
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the conditional probability of X given y for each feature xj.
     *  @param x_     the integer-valued data vectors stored as rows of a matrix
     *  @param nu_Xy  the joint frequency of X and y for each feature xj and class value
     */
    def cProb_Xy (x_ : MatrixD, nu_Xy: RTensorD): RTensorD =
        val pXy = new RTensorD (x_.dim2, vc, k)
        for j <- x_.indices2; xj <- 0 until vc(j) do
            pXy(j, xj) = (nu_Xy(j, xj) + me_v(j)) / (nu_y + me)          // Conditional Probability Tables (CPTs)
        end for
        pXy
    end cProb_Xy

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
//      debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the integer value of y = f(z) by computing the product of the class
     *  probabilities p_y and all the conditional probabilities P(X_j = z_j | y = c)
     *  and returning the class with the highest relative probability.
     *  Note, p_yz from `Classifier` holds the relative probabilities of y given z.
     *  @param z  the new vector to predict
     */
    override def predictI (z: VectorI): Int =
        p_yz = p_y.copy                                                  // start with class (prior) probabilities
        for j <- z.indices do p_yz *= p_Xy(j, z(j))                      // multiply P(X_j = z_j | y = c)
        p_yz.argmax ()                                                   // return class with highest probability
    end predictI

    inline def predictI (z: VectorD): Int = predictI (z.toInt)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the integer value of y = f(z) by computing the product of the class
     *  probabilities p_y and all the conditional probabilities P(X_j = z_j | y = c)
     *  and returning the class with the highest relative probability.
     *  This method adds "positive log probabilities" to avoids underflow.
     *  Note, p_yz from `Classifier` holds the relative probabilities of y given z.
     *  To recover q relative probability compute 2^(-q) where q is a plog.
     *  @param z  the new vector to predict
     */
    override def lpredictI (z: VectorI): Int =
        p_yz = plog (p_y)                                                // start with class (prior) probabilities
        for j <- z.indices do p_yz += plog (p_Xy(j, z(j)))               // add plog P(X_j = z_j | y = c)
        p_yz.argmin ()                                                   // return class with lowest positive log probability
    end lpredictI

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

end NaiveBayes


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** `NaiveBayes` is the companion object for the `NaiveBayes` class.
 */
object NaiveBayes:

    val hp = new HyperParameter ()
    hp += ("me", 0.1, 0.1)                                               // Laplace smoothing

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the contribution to the fake instances for each each feature xj
     *  based on its value count.  Used for Laplace smoothing.
     *  @param me  the number/fraction of fake instances used for Laplace smooth
     *  @param vc  the value count vector
     */
    def me_vc (me: Double, vc: VectorI): VectorD = VectorD (vc.map (me / _))         // me / vc_j for all j

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NaiveBayes` object, passing x and y together in one matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the names of the features/variables
     *  @param k       the number of classes
     *  @param cname   the names of the classes
     *  @param vc      the value count (number of distinct values) for each feature
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixI, fname: Array [String] = null, k: Int = 2,
               cname: Array [String] = Array ("No", "Yes"), vc: VectorI = null,
               hparam: HyperParameter = NaiveBayes.hp)
              (col: Int = xy.dim2 - 1): NaiveBayes =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new NaiveBayes (x, y, fname, k, cname, vc, hparam)
    end apply

end NaiveBayes


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesTest` main function is used to test the `NaiveBayes` class.
 *  > runMain scalation.modeling.classifying.naiveBayesTest
 */
@main def naiveBayesTest (): Unit =

    import Example_PlayTennis._

    banner ("Play Tennis Example")
    println (s"xy = $xy")                                           // combined data matrix [ x | y ]

    val mod = NaiveBayes (xy, fname)()                              // create a classifier
    mod.trainNtest ()()                                             // train and test the classifier
    println ("CPTs = " + mod.getCPTs)                               // print the conditional probability tables
    println (mod.summary ())                                        // summary statistics

    val z = VectorI (2, 2, 1, 1)                                    // new data vector to classify
    banner (s"Classify $z")
    println (s"Use mod to classify ($z)  = ${mod.classify (z)}")    // based on highest relative probability
    println (s"Use mod to lclassify ($z) = ${mod.lclassify (z)}")   // add positive log probabilities

    banner ("Validation")
    println ("mod test accu = " + mod.validate ()())                // out-of-sample testing

/* Not enough instances for cross-validation
    banner ("Cross-validation")
    FitM.showQofStatTable (mod.crossValidate ())                    // 5-fold cross-validation (14 instances typically too few)
*/

end naiveBayesTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesTest2` main function is used to test the `NaiveBayes` class.
 *  Classify whether a car is more likely to be stolen (1) or not (1).
 *  @see www.inf.u-szeged.hu/~ormandi/ai2/06-naiveBayes-example.pdf
 *  > runMain scalation.modeling.classiying.naiveBayesTest2
 */
@main def naiveBayesTest2 (): Unit =

    // x0: Color:   Red (1), Yellow (0)
    // x1: Type:    SUV (1), Sports (0)
    // x2: Origin:  Domestic (1), Imported (0)
    // x3: Mpg:     High (1), Low (0)
    // features:             x0 x1 x2 x3
    val x = MatrixI ((10, 4), 1, 0, 1, 1,                           // data matrix
                              1, 0, 1, 0,
                              1, 0, 1, 1,
                              0, 0, 1, 1,
                              0, 0, 0, 1,
                              0, 1, 0, 0,
                              0, 1, 0, 0,
                              0, 1, 1, 1,
                              1, 1, 0, 0,
                              1, 0, 0, 0)

    val y  = VectorI (1, 0, 1, 0, 1, 0, 1, 0, 0, 1)                 // classification vector: 0(No), 1(Yes))
    val fname = Array ("Color", "Type", "Origin", "Mpg")            // feature/variable names
    val cname = Array ("No", "Yes")                                 // class names

    banner ("Stolen Car Example")
    println (s"x = $x")

    val mod = new NaiveBayes (x, y, fname, 2, cname)                // create the classifier
    mod.trainNtest ()()                                             // train and test the classifier
    println ("CPTs = " + mod.getCPTs)                               // print the conditional probability tables
    println (mod.summary ())                                        // summary statistics

    val z1 = VectorI (1, 0, 1, 1)                                   // existing data vector to classify
    val z2 = VectorI (1, 1, 1, 0)                                   // new data vector to classify
    println (s"Use mod to classify ($z1) = ${mod.classify (z1)}")
    println (s"Use mod to classify ($z2) = ${mod.classify (z2)}")

end naiveBayesTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesTest3` main function is used to test the `NaiveBayes` class.
 *  Given whether a person is Fast and/or Strong, classify them as making C = 1
 *  or not making C = 0 the football team.
 *  > runMain scalation.modeling.classiying.naiveBayesTest3
 */
@main def naiveBayesTest3 (): Unit =

    // x0: Fast
    // x1: Strong
    // y:  Classification (No/0, Yes/1)
    // features:              x0 x1  y
    val xy = MatrixI ((10, 3), 1, 1, 1,
                               1, 1, 1,
                               1, 0, 1,
                               1, 0, 1,
                               1, 0, 0,
                               0, 1, 0,
                               0, 1, 0,
                               0, 1, 1,
                               0, 0, 0,
                               0, 0, 0)

    val fname = Array ("Fast", "Strong")                            // feature names
    val cname = Array ("No", "Yes")                                 // class names

    banner ("Football Team  Example")
    println (s"xy = $xy")

    val mod = NaiveBayes  (xy, fname, 2, cname)()                   // create the classifier
    mod.trainNtest ()()                                             // train and test the classifier
    println ("CPTs = " + mod.getCPTs)                               // print the conditional probability tables
    println (mod.summary ())                                        // summary statistics

    val z = VectorI (1, 0)                                          // new data vector to classify
    println (s"Use mod  to classify ($z) = ${mod.classify (z)}")

end naiveBayesTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `naiveBayesTest4` main function is used to test the `NaiveBayes` class.
 *  @see archive.ics.uci.edu/ml/datasets/Lenses
 *  @see docs.roguewave.com/imsl/java/7.3/manual/api/com/imsl/datamining/NaiveBayesClassifierEx2.html
 *  > runMain scalation.modeling.classiying.naiveBayesTest4
 */
@main def naiveBayesTest4 (): Unit =

    // y:  Classification (1): hard contact lenses, (2) soft contact lenses, (3) no contact lenses
    // x0. Age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic
    // x1. Spectacle prescription:  (1) myope, (2) hypermetrope
    // x2. Astigmatic:     (1) no, (2) yes
    // x3. Tear production rate:  (1) reduced, (2) normal
    // features:              x0  x1  x2  x3   y
    val xy = MatrixI ((24, 5), 1,  1,  1,  1,  3,           // 1
                               1,  1,  1,  2,  2,           // 2
                               1,  1,  2,  1,  3,           // 3
                               1,  1,  2,  2,  1,           // 4
                               1,  2,  1,  1,  3,           // 5
                               1,  2,  1,  2,  2,           // 6
                               1,  2,  2,  1,  3,           // 7
                               1,  2,  2,  2,  1,           // 8
                               2,  1,  1,  1,  3,           // 9
                               2,  1,  1,  2,  2,           // 10
                               2,  1,  2,  1,  3,           // 11
                               2,  1,  2,  2,  1,           // 12
                               2,  2,  1,  1,  3,           // 13
                               2,  2,  1,  2,  2,           // 14
                               2,  2,  2,  1,  3,           // 15
                               2,  2,  2,  2,  3,           // 16
                               3,  1,  1,  1,  3,           // 17
                               3,  1,  1,  2,  3,           // 18
                               3,  1,  2,  1,  3,           // 19
                               3,  1,  2,  2,  1,           // 20
                               3,  2,  1,  1,  3,           // 21
                               3,  2,  1,  2,  2,           // 22
                               3,  2,  2,  1,  3,           // 23
                               3,  2,  2,  2,  3)           // 24

    xy -= 1                                                         // shift values to start at 0

    val fname = Array ("Age", "Spectacle", "Astigmatic", "Tear")    // feature names
    val cname = Array ("Hard", "Soft", "Neither")                   // class names

    banner ("Contact Leans Example")
    println (s"xy = $xy")

    val mod = NaiveBayes (xy, fname, 3, cname)()                    // create the classifier
    mod.trainNtest ()()                                             // train and test the classifier
    println ("CPTs = " + mod.getCPTs)                               // print the conditional probability tables
    println (mod.summary ())                                        // summary statistics

    for i <- xy.indices2 do
        val z   = xy(i).not(4).toInt                                // x-values
        val y   = xy(i, 4).toInt                                    // y-value
        val yp  = mod.classify (z)                                  // y predicted
        println (s"Use mod : yp = classify ($z) = $yp,\t y = $y,\t ${cname(y)}")
    end for

end naiveBayesTest4

