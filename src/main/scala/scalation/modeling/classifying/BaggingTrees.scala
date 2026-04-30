
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dong Yu Yu, John Miller
 *  @version 2.0
 *  @date    Fri Jan  5 16:54:27 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Bagging of Decision Trees (subsampling only)
 */

package scalation
package modeling
package classifying

import scala.collection.mutable.Set

import scalation.mathstat._
import scalation.random.RandomVecI

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BaggingTrees` class uses several randomly built descision trees for classification.
 *  It randomly selects sub-samples of bRatio * x.dim size from the data x and y
 *  to build nTrees decision trees.  The classify method uses voting from all of the trees.
 *  Note: this classifier does not select sub-features to build the trees.
 *  @param x       the data matrix (instances by features)
 *  @param y       the response/class labels of the instances
 *  @param fname_  the names of the variables/features
 *  @param k       the number of classes
 *  @param cname_  the class names
 *  @param conts   the set of feature indices for variables that are treated as continuous
 *  @param hparam  the hyper-parameters
 */
class BaggingTrees (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
                    cname_ : Array [String] = Array ("No", "Yes"),
                    conts: Set [Int] = Set [Int] (), hparam: HyperParameter = DecisionTree.hp)
      extends Classifier (x, y, fname_, k , cname_, hparam)
         with FitC (k):

    private   val debug      = debugf ("BaggingTrees", true)                     // debug function
    private   val flaw       = flawf ("BaggingTrees")                            // flaw function
    protected val nTrees     = hparam("nTrees").toInt                            // number of trees
    protected val bRatio     = hparam("bRatio").toDouble                         // bagging ratio 
    protected val height     = hparam("height").toInt                            // height limit

    protected val trees      = Array.ofDim [DecisionTree_C45] (nTrees)           // many decision trees
    protected val sampleSize = (bRatio * x.dim).toInt                            // size of matrix sub-samples

    if nTrees <= 0 then                flaw ("init", "BT number of tree must be at least one")
    if bRatio <= 0 || bRatio >= 1 then flaw ("init", "BT bagging ratio restricted to (0, 1)")

    _modelName = s"BaggingTrees_${height}_$nTrees"                               // name of the model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of model parameter vector.
     */
    override def parameter: VectorD = null                                       // FIX - to be implemented

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a classification model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training set.
     *  Build the trees by selecting a subSample for each tree.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        for l <- 0 until nTrees do
            val (sub_x, sub_y, irows) = subSample (x_, y_, sampleSize, l)        // select rows from x_ and elements from y_
            debug ("train", s"row indices for tree$l, irows = $irows")

            trees(l) = new DecisionTree_C45 (sub_x, sub_y, fname, k, cname, conts, hparam)
            trees(l).train ()
//          debug ("train", s"for tree$l === \n ${trees(l).printTree ()}")
        end for
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
        val yp  = predictI (x_)                                                  // predicted classes
        val qof = diagnose (y_.toDouble, yp.toDouble)                            // diagnose from actual and predicted
//      debug ("test", s" yp = $yp \n qof = $qof")
        (yp, qof)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict/classify the vector z by voting within randomized trees, returning
     *  the class index.
     *  @param z  the vector to be predicted
     */
    override def predictI (z: VectorD): Int =
        val vote = new VectorI (k)
        for l <- 0 until nTrees do                                               // iterate l-th tree
            val y_l = trees(l).predictI (z)                                      // get vote from l-th tree
            vote(y_l) += 1                                                       // tally the vote
//          debug ("predict", s"for tree$l, predicted class = y_l")
        end for
        vote.argmax ()                                                           // find argmax => the winner
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
        super.summary (x_, fname_, b_, vifs)                                     // summary from `Fit`
    end summary

end BaggingTrees


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BaggingTrees` companion object provides a factory method.
 */
object BaggingTrees:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a bag of trees for the given combined matrix where the column col
     *  is the response/classification vector.
     *  @param xy      the combined data matrix (features and response)
     *  @param fname   the names for all features/variables
     *  @param k       the number of classes
     *  @param cname   the names for all classes
     *  @param conts   the set of feature indices for variables that are treated as continuous
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null, k: Int = 2,
               cname: Array [String]  = Array ("No", "Yes"),
               conts: Set [Int] = Set [Int] (), hparam: HyperParameter = DecisionTree.hp)
              (col: Int = xy.dim2 - 1): BaggingTrees =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new BaggingTrees (x, y, fname, k, cname, conts, hparam)
    end apply

end BaggingTrees


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `baggingTreesTest` main function tests the `BaggingTrees` class.
 *  It tests a simple case that does not require a file to be read.
 *  > runMain scalation.modeling.classifying.baggingTreesTest
 */
@main def baggingTreesTest (): Unit =

    val x = MatrixD ((11, 11), 8.1, 0.27, 0.41,  1.45, 0.033, 11,  63.0, 0.9908, 2.99, 0.56, 12.0, 
                               8.6, 0.23, 0.40,  4.20, 0.035, 17, 109.0, 0.9947, 3.14, 0.53,  9.7, 
                               7.9, 0.18, 0.37,  1.20, 0.040, 16,  75.0, 0.9920, 3.18, 0.63, 10.8, 
                               6.6, 0.16, 0.40,  1.50, 0.044, 48, 143.0, 0.9912, 3.54, 0.52, 12.4, 
                               8.3, 0.42, 0.62, 19.25, 0.040, 41, 172.0, 1.0002, 2.98, 0.67,  9.7, 
                               6.6, 0.17, 0.38,  1.50, 0.032, 28, 112.0, 0.9914, 3.25, 0.55, 11.4, 
                               6.3, 0.48, 0.04,  1.10, 0.046, 30,  99.0, 0.9928, 3.24, 0.36,  9.6, 
                               6.2, 0.66, 0.48,  1.20, 0.029, 29,  75.0, 0.9892, 3.33, 0.39, 12.8, 
                               7.4, 0.34, 0.42,  1.10, 0.033, 17, 171.0, 0.9917, 3.12, 0.53, 11.3, 
                               6.5, 0.31, 0.14,  7.50, 0.044, 34, 133.0, 0.9955, 3.22, 0.50,  9.5, 
                               6.2, 0.66, 0.48,  1.20, 0.029, 29,  75.0, 0.9892, 3.33, 0.39, 12.8)

    val y = VectorI (5, 5, 5, 7, 5, 7, 6, 8, 6, 5, 8)                 // response/class labels
    y -= 3                                                            // shift the class labels by 3

    banner ("baggingTreesTest:  partial winequality-white dataset")
    val k     = 7
    val fname = Array ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                       "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")
    val cname = Array ("Level3", "Level4", "Level5", "Level6", "Level7", "Level8", "Level9")
    val conts = Set.range (0, x.dim2)

    DecisionTree.hp("nTrees") = 3
    val mod = new BaggingTrees (x, y, fname, k, cname, conts)
    mod.trainNtest ()()
    println (mod.summary ())

end baggingTreesTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BaggingTreesTest2` main function tests the `BaggingTrees` class.
 *  It tests the Bagging Trees classifier using well-known WineQuality Dataset.
 *  > runMain scalation.modeling.classifying.BaggingTreesTest2
 */
@main def baggingTreesTest2 (): Unit =

    val nfile  = "winequality-white.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not (?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    y -= 3                                                            // shift the class labels by 3

    banner ("baggingTreesTest2:  winequality-white dataset")
    val k     = 7
    val conts = Set.range (0, x.dim2)                                 // all features are continuous

    DecisionTree.hp("nTrees") = 3
    val mod = new BaggingTrees (x, y, null, k, null, conts)
    mod.trainNtest ()()
    println (mod.summary ())

end baggingTreesTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `baggingTreesTest3` main function tests the `BaggingTrees` class.
 *  It tests the Bagging Trees classifier using a specific numbers of trees.
 *  > runMain scalation.modeling.classifying.baggingTreesTest3
 */
@main def baggingTreesTest3 (): Unit =

    val nfile  = "winequality-white.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not (?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    y -= 3                                                            // shift the class labels by 3

    banner ("baggingTreesTest3:  winequality-white dataset")
    val k     = 7
    val conts = Set.range (0, x.dim2)                                 // all features are continuous
    val maxTrees = 3

    for nTrees <- 1 to maxTrees do
        println (s"Number of Tree = $nTrees")
        DecisionTree.hp("nTrees") = nTrees
        val mod  = new BaggingTrees (x, y, null, k, null, conts)
        mod.trainNtest ()()
        println (mod.summary ())
    end for

end baggingTreesTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `baggingTreesTest4` main function tests the `BaggingTrees` class.
 *  It tests Bagging Trees using unseen data.
 *  > runMain scalation.modeling.classifying.baggingTreesTest4
 */
@main def baggingTreesTest4 (): Unit =

    val nfile  = "winequality-white.csv"
    val xy     = MatrixD.load (nfile)
    val ycol   = xy.dim2 - 1
    for i <- xy.indices do xy(i, ycol) -= 3                           // shift the class labels by 3
    val (x, y) = (xy.not (?, ycol), xy(?, ycol))
 
    banner ("baggingTreesTest4: winequality-white dataset")
    val k     = 7
    val conts = Set.range (0, x.dim2)                                 // all features are continuous

    // Divide samples into training and testing dataset
    val trainSize  = (y.dim * 0.8).toInt
    val rvv        = RandomVecI (trainSize, y.dim - 1, 0)
    val subSample  = new MatrixD (trainSize, xy.dim2)
    val elseSample = new MatrixD (xy.dim - trainSize, xy.dim2)
    val index      = rvv.igen
    var trainCount = 0
    var elseCount  = 0

    for i <- y.indices do
        if index contains i then
            subSample.set (trainCount, xy(i))
            trainCount += 1
        else
            elseSample.set (elseCount, xy(i))
            elseCount  += 1 
    end for

    val elseFeature = elseSample(?, 0 until elseSample.dim2-1)
    val elseTarget  = elseSample(?, elseSample.dim2-1)

    val mod = new BaggingTrees (subSample(?, 0 until xy.dim2-1),
                                subSample(?, subSample.dim2-1).toInt, 
                                null, k, null, conts)
    mod.train ()

    // Print the accuracy for unseen data
    var accurateCount = 0.0
    for i <- elseFeature.indices do
        if mod.predictI (elseFeature(i)) == elseTarget(i) then accurateCount += 1
    end for
    val accuracy = accurateCount / elseFeature.dim
    println (s"Testing Accuracy = $accuracy")
    mod.test ()
    println (mod.summary ())

end baggingTreesTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `baggingTreesTest5` main function tests the `BaggingTrees` class.
 *  It tests the Bagging Trees classifier by specific numbers of trees.
 *  > runMain scalation.modeling.classifying.baggingTreesTest5
 */
@main def baggingTreesTest5 (): Unit =

    val nfile  = "breast_cancer.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not (?, xy.dim2-1), xy(?, xy.dim2-1).toInt)

    banner ("baggingTreesTest5: breast_cancer dataset")
    val maxTrees = 5

    for nTrees <- 1 to maxTrees do
        println (s"Number of Tree = $nTrees}")
        DecisionTree.hp("nTrees") = nTrees
        val mod = new BaggingTrees (x, y)
        mod.trainNtest ()()
        println (mod.summary ())
    end for

end baggingTreesTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `baggingTreesTest6` main function tests the `BaggingTrees` class.
 *  It tests the Bagging Trees classifier on Breast Cancer dataset.
 *  > runMain scalation.modeling.classifying.baggingTreesTest6
 */
@main def baggingTreesTest6 (): Unit =

    val nfile  = "breast_cancer.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not (?, xy.dim2-1), xy(?, xy.dim2-1).toInt)

    banner ("baggingTreesTest6: breast cancer dataset")

    DecisionTree.hp("bRatio") = 0.7
//  DecisionTree.hp("height") = 15
    val mod = new BaggingTrees (x, y)
    mod.trainNtest ()
    println (mod.summary ())

end baggingTreesTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `baggingTreesTest7` main function tests the `BaggingTrees` class.
 *  It tests the Bagging Trees classifier on Diabetes dataset.
 *  > runMain scalation.modeling.classifying.baggingTreesTest7
 */
@main def baggingTreesTest7 (): Unit =

    val nfile  = "diabetes.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not (?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    val fname  = Array ("pregnancies", "glucose", "blood pressure", "skin thickness", "insulin",
                        "BMI", "diabetes pedigree function", "age")   // feature names
    val cname  = Array ("tested_positive", "tested_negative")         // class names

    banner ("baggingTreesTest7: diabetes dataset")

    val conts = Set.range (0, x.dim2)
    val hp2   = DecisionTree.hp.updateReturn (("nTrees", 9.0), ("bRatio", 0.5), ("height", 7.0))
    val mod   = new BaggingTrees (x, y, fname, 2, cname, conts, hparam = hp2)
    mod.trainNtest ()
    println (mod.summary ())

end baggingTreesTest7

