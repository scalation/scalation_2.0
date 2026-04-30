
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Hao Peng
 *  @version 2.0
 *  @date    Sat Sep  8 13:53:16 EDT 2012
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Integer-Based TAN Bayes Classifier
 *           Tree-Augmented Naive (TAN)
 *
 *  @see eric.univ-lyon2.fr/~ricco/tanagra/fichiers/en_Tanagra_TAN_Bayes_Classifier_Explained.pdf
 *
 *  for comparison @see www.bnlearn.com
 *      R-package bblearn:  bn.naive, naive.bayes, bn.tan, tree.bayes
 */

package scalation
package modeling
package classifying

import scala.runtime.ScalaRunTime.stringOf

import scalation.database.MaxSpanningTree
import scalation.mathstat._

import Classifier.{shift2zero, vc_fromData}
import Probability.plog

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TANBayes` class implements an Integer-Based TAN Bayes Classifier,
 *  which is a commonly used such classifier for discrete input data.  The
 *  classifier is trained using a data matrix x and a classification vector y.
 *  Each data vector in the matrix is classified into one of k classes numbered
 *  0, ..., k-1.  Prior probabilities are calculated based on the population of
 *  each class in the training-set.  Relative posterior probabilities are computed
 *  by multiplying these by values computed using conditional probabilities.
 *  stored in Conditional Probability Tables (CPTs).
 *------------------------------------------------------------------------------
 *  The classifier is TAN allowing each xj to utilize information from its x-parant.
 *  @param x       the input/data m-by-n matrix
 *  @param y       the class vector, where y(i) = class for row i of matrix x
 *  @param fname_  the names of the features/variables
 *  @param k       the number of classes
 *  @param cname_  the names of the classes
 *  @param vc      the value count (number of distinct values) for each feature
 *  @param hparam  the hyper-parameters
 */
class TANBayes (x: MatrixD, y: VectorI, fname_ : Array [String] = null,
                k: Int = 2, cname_ : Array [String] = Array ("No", "Yes"),
                private var vc: VectorI = null, hparam: HyperParameter = NaiveBayes.hp)
      extends Classifier (x, y, fname_, k, cname_, hparam)
         with BayesClassifier (k)
         with FitC (k):

    private val debug = debugf ("TANBayes", true)                        // debug function

    _modelName = "TANBayes"                                              // name of the model

    if vc == null then
        shift2zero (x); vc = vc_fromData (x)                             // set value counts from data

    private val me   = hparam("me").toDouble                             // m-estimates (me == 0 => regular MLE estimates)
    private val me_v = NaiveBayes.me_vc (me, vc)                         // for Laplace smoothing: me / vc_j for all j

    private var nu_Xy: RTensorD = null                                   // Joint Frequency Tables (JFTs)
    private var nu_Xpy_ : RTensor4D = null                               // Joint Frequency Tables (JFTs)

//  private var p_Xy: Array [MatrixD] = null                             // Conditional Probability Tables (CPTs) one per feature
    private var p_Xpy: Array [Array [MatrixD]] = null                    // extended Conditional Probability Tables (CPTs) one per feature
//  private var p_Xpy_ : RTensor4D = null                                // extended Conditional Probability Tables (CPTs) one per feature

    private var parent: VectorI = null                                   // parent of each feature/variable
    private var vc_p: VectorI = null                                     // parent value count

    debug ("init", s"distinct value count vc = $vc")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the best parent xpj for each feature xj based on the CMI matrix.
     *  For the Play Tennis example, parent = VectorI (-1, 0, 1, 0).
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def findParent (x_ : MatrixD, y_ : VectorI): Unit =
        val cmiMx = cmiMatrix (x_, vc, y_)                              // create Conditional Mutual Information matrix
        val stree = MaxSpanningTree (cmiMx)                             // find Maximum Spanning Tree
        parent    = stree.makeITree ()                                  // get predecessor vector
        debug ("findParent", s"parent = $parent")
    end findParent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the value counts for parents.  When there is no paremt (-1) set to 1.
     */
    def vc_parent (): Unit =
        vc_p = VectorI (for j <- x.indices2 yield { val pj = parent(j); if pj < 0 then 1 else vc(pj) })
        println (s"vc_p = $vc_p")     
    end vc_parent

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a classification model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training
     *  dataset.  Training involves estimating the model parameters or pmf.
     *  Train the classifier by computing the probabilities for y, and the
     *  conditional probabilities for each x_j.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        super.train (x_, y_)                                            // set class frequencies nu_y and probabilities p_y
        findParent (x_, y_)                                             // find the best x-parent for each xj (parent)
        vc_parent ()                                                    // set the value counts for the parents (vc_p)
        nu_Xy      = RTensorD.freq (x_, vc, y_, k)                      // Joint Frequency Tables (JFTs)
        nu_Xpy_    = RTensor4D.freq (x_, vc, parent, vc_p, y_, k)       // extended Joint Frequency Tables (JFTs)
        val nu_Xpy = freq_Xpy (x_, y_)                                  // extended Joint Frequency Tables (JFTs)
        p_Xpy      = cProb_Xpy (x_, nu_y, nu_Xy, nu_Xpy)            // extended Conditional Probability Tables (CPTs)
//      println (s"nu_Xpy  = ${stringOf (nu_Xpy)}")
//      println (s"nu_Xpy_ = $nu_Xpy_")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the joint frequency of xj, xp and y for all xj in X, where p is the
     *  unique x-parent of feature xj.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    def freq_Xpy (x_ : MatrixD, y_ : VectorI): Array [Array [MatrixD]] =
        val nu_Xpy = Array.ofDim [Array [MatrixD]] (x_.dim2)
        for j <- x_.indices2 do
            nu_Xpy(j) = Array.ofDim [MatrixD] (vc(j))
            val p     = parent(j)                                       // parent of xj
            val vc_p  = if p < 0 then 1 else vc(p)                      // parent value count
            for xj <- 0 until vc(j) do nu_Xpy(j)(xj) = new MatrixD (vc_p, k)
        end for
        for i <- x_.indices do updateFreq (x_, y_, i, nu_Xpy)
        nu_Xpy
    end freq_Xpy

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Increment frequency counters used in freq_Xpy calculations based on the i-th
     *  row of the data matrix.
     *  @param x_      the integer-valued data vectors stored as rows of a matrix
     *  @param y_      the class vector, where y(i) = class for row i of the matrix x, x(i)
     *  @param i       the index for current data row
     *  @param nu_Xpy  the joint frequency of X, p and y for each combination of features xj.
     */
    private def updateFreq (x_ : MatrixD, y_ : VectorI, i: Int,
                            nu_Xpy: Array [Array [MatrixD]]): Unit =
        val yi = y_(i)                                                  // class value for i-th instance/datapoint
        for j <- x_.indices2 if parent(j) > -1 do                       // for each feature/variable xj
            val p = parent(j)                                           // parent of xj
            val x_ij = x_(i, j).toInt                                   // i-th value for xj
            val x_ip = x_(i, p).toInt                                   // i-th value for xp
            nu_Xpy(j)(x_ij)(x_ip, yi) += 1                              // increment frequency for xj, xp, yi
        end for
    end updateFreq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the conditional probability of X given p and y for all xj in X,
     *  where p is the the unique x-parent of feature xj.
     *  @param x_     the training/full data/input matrix (defaults to full x)
     *  @param nu_y   the class frequency of y
     *  @param nu_Xy  the joint frequency of X and y for all xj in X
     *  @param nu_Xpy the joint frequency of X, p and y for all xj in X
     */
    def cProb_Xpy (x_ : MatrixD, nu_y: VectorD, nu_Xy: RTensorD,
                   nu_Xpy: Array [Array [MatrixD]]): Array [Array [MatrixD]] =
        val p_Xpy = Array.ofDim [Array [MatrixD]] (x_.dim2)
        for j <- x_.indices2 do
            p_Xpy(j) = Array.ofDim [MatrixD] (vc(j))
            val p    = parent(j)                                        // parent of xj
            val vc_p = if p < 0 then 1 else vc(p)                       // parent value count
            for xj <- 0 until vc(j) do
                p_Xpy(j)(xj) = new MatrixD (vc_p, k)
                for xp <- 0 until vc_p do
                    val d: VectorD = if p > -1 then nu_Xy(p)(xp) else nu_y
                    p_Xpy(j)(xj)(xp) = (nu_Xpy(j)(xj)(xp) + me_v(j)) / (d + me)
        end for
        p_Xpy
    end cProb_Xpy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the predictive model y_ = f(x_) + e and return its predictions and QoF vector.
     *  Testing may be in-sample (on the full dataset) or out-of-sample (on the testing set)
     *  as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorI = y): (VectorI, VectorD) =
        val yp  = predictI (x_)                                         // predicted classes
        val qof = diagnose (y_.toDouble, yp.toDouble)                   // diagnose from actual and predicted
//      debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the integer value of y = f(z) by computing the product of the class
     *  probabilities p_y and all the conditional probabilities P(X_j = z_j | X_p = z_p, y = c)
     *  and returning the class with the highest relative probability.
     *  Note, p_yz from `Classifier` holds the relative probabilities of y given z.
     *  @param z  the new vector to predict
     */
    override def predictI (z: VectorI): Int =
        p_yz = p_y.copy                                                 // start with class (prior) probabilities
        for j <- z.indices do                                           // P(X_j = z_j | X_p = z_p, y = c)
            val p    = parent(j)                                        // parent of xj
            val ecpt = p_Xpy(j)                                         // get j-th extended CPT
            if p > -1 then                                              // xj has a parent
                p_yz *= ecpt (z(j))(z(p))                               // multiply in its v = (z(j), z(p)) row
            else                                                        // xj does not have a parent
                p_yz *= ecpt (z(j))(0)                                  // multiply in its v = z(j) row
        end for
//      debug ("predictI", s"p_yz = $p_yz")
        p_yz.argmax ()                                                  // return class with highest probability
    end predictI

    inline def predictI (z: VectorD): Int = predictI (z.toInt)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the integer value of y = f(z) by computing the product of the class
     *  probabilities p_y and all the conditional probabilities P(X_j = z_j | X_p = z_p, y = c)
     *  and returning the class with the highest relative probability.
     *  This method adds "positive log probabilities" to avoids underflow.
     *  Note, p_yz from `Classifier` holds the relative probabilities of y given z.
     *  To recover q relative probability compute 2^(-q) where q is a plog.
     *  @param z  the new vector to predict
     */
    override def lpredictI (z: VectorI): Int =
        p_yz = plog (p_y)                                               // start with class (prior) plogs
        for j <- z.indices do                                           // P(X_j = z_j | X_p = z_p, y = c)
            val p    = parent(j)                                        // parent of xj
            val ecpt = p_Xpy(j)                                         // get j-th extended CPT
            if p > -1 then                                              // xj has a parent
                p_yz += plog (ecpt (z(j))(z(p)))                        // multiply in its v = (z(j), z(p)) row
            else                                                        // xj does not have a parant
                p_yz += plog (ecpt (z(j))(0))                           // multiply in its v = z(j) row
        end for
        debug ("lpredictI", s"p_yz = $p_yz")
        p_yz.argmin ()                                                  // return class with lowest positive log probability
    end lpredictI

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the extended Joint Frequenct Tables (eJFTs) by iterating over
     *  the features/variables.
     */
    def printJFTs (): Unit =
        for j <- x.indices2 do
            val pj = parent(j)
            if pj < 0 then
                println (s"regular JFT for xj [ x$j, y ]: nu_Xy(j) = ${nu_Xy(j)}")            // regular JFT
            else
                println (s"extended JFT for xj [ x$j, x$pj, y ]: nu_Xpy(j) = ${nu_Xpy_(j)}")   // extended JFT
        end for
    end printJFTs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the extended Conditional Probability Tables (eCPTs) by iterating over
     *  the features/variables.
     */
    def printCPTs (): Unit =
        for j <- x.indices2 do
            val pj = parent(j)
            if pj < 0 then
                println (s"regular CPT for xj: P(x$j | y) = ${stringOf (p_Xpy(j))}")
            else
                println (s"extended CPT for xj: P(x$j | z$pj, y) = ${stringOf (p_Xpy(j))}")
        end for
    end printCPTs

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
        super.summary (x_, fname_, b_, vifs)                            // summary from `Fit`
    end summary

end TANBayes


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** `TANBayes` is the companion object for the `TANBayes` class.
 */
object TANBayes:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TANBayes` object, passing x and y together in one matrix.
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
              (col: Int = xy.dim2 - 1): TANBayes =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                 // data matrix, response vector
        new TANBayes (x, y, fname, k, cname, vc, hparam)
    end apply

end TANBayes


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tANBayesTest` main function is used to test the `TANBayes` class on the
 *  Play Tennis example problem.
 *  > runMain scalation.modeling.classifying.tANBayesTest
 */
@main def tANBayesTest (): Unit =

    import Example_PlayTennis._

    banner ("Play Tennis Example")
    println (s"xy = $xy")                                           // combined data matrix [ x | y ]

    val mod = TANBayes (xy, fname)()                                // create a classifier
    mod.trainNtest ()()                                             // train and test the classifier
    banner ("Joint Frequency Tables:")
    mod.printJFTs ()                                                // print Joint Frequency Tables (JFTs)
    banner ("Conditional Probability Tables:")
    mod.printCPTs ()                                                // print Conditional Probability Tables (CPTs)
    println (mod.summary ())                                        // summary statistics

    val z = VectorI (2, 2, 1, 1)                                    // new data vector to classify
    banner (s"Classify $z")
    println (s"Use mod to classify ($z)  = ${mod.classify (z)}")    // based on highest relative probability
    println (s"Use mod to lclassify ($z) = ${mod.lclassify (z)}")   // add positive log probabilities

    banner ("Validation")
    println ("mod test accu = " + mod.validate ()())                // out-of-sample testing

/*  too few instances for cross-validation
    banner ("Cross-validation")
    FitM.showQofStatTable (mod.crossValidate ())                    // 5-fold cross-validation (14 instances typically too few)
*/

end tANBayesTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tANBayesTest2` main function is used to test the `TANBayes` class.
 *  Classify whether a car is more likely to be stolen (1) or not (1).
 *  @see www.inf.u-szeged.hu/~ormandi/ai2/06-tANBayes-example.pdf
 *  > runMain scalation.modeling.classiying.tANBayesTest2
 */
@main def tANBayesTest2 (): Unit =

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

    val mod = new TANBayes (x, y, fname, 2, cname)                  // create the classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printCPTs ()                                                // print the conditional probability tables
    println (mod.summary ())                                        // summary statistics

    val z1 = VectorI (1, 0, 1, 1)                                   // existing data vector to classify
    val z2 = VectorI (1, 1, 1, 0)                                   // new data vector to classify
    println (s"Use mod to classify ($z1) = ${mod.classify (z1)}")
    println (s"Use mod to classify ($z2) = ${mod.classify (z2)}")

end tANBayesTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tANBayesTest3` main function is used to test the `TANBayes` class.
 *  Given whether a person is Fast and/or Strong, classify them as making C = 1
 *  or not making C = 0 the football team.
 *  > runMain scalation.modeling.classiying.tANBayesTest3
 */
@main def tANBayesTest3 (): Unit =

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

    val mod = TANBayes  (xy, fname, 2, cname)()                     // create the classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printCPTs ()                                                // print the conditional probability tables
    println (mod.summary ())                                        // summary statistics

    val z = VectorI (1, 0)                                          // new data vector to classify
    println (s"Use mod  to classify ($z) = ${mod.classify (z)}")

end tANBayesTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tANBayesTest4` main function is used to test the `TANBayes` class.
 *  @see archive.ics.uci.edu/ml/datasets/Lenses
 *  @see docs.roguewave.com/imsl/java/7.3/manual/api/com/imsl/datamining/TANBayesClassifierEx2.html
 *  > runMain scalation.modeling.classiying.tANBayesTest4
 */
@main def tANBayesTest4 (): Unit =

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

    val mod = TANBayes (xy, fname, 3, cname)()                      // create the classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printCPTs ()                                                // print the conditional probability tables
    println (mod.summary ())                                        // summary statistics

    for i <- xy.indices2 do
        val z   = xy(i).not(4).toInt                                // x-values
        val y   = xy(i, 4).toInt                                    // y-value
        val yp  = mod.classify (z)                                  // y predicted
        println (s"Use mod : yp = classify ($z) = $yp,\t y = $y,\t ${cname(y)}")
    end for

end tANBayesTest4

