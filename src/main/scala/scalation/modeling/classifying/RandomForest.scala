
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dong Yu Yu, John Miller
 *  @version 2.0
 *  @date    Fri Jan  5 16:54:27 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Random Forest of Descision Trees (subsampling & sub-features)
 *
 *  @see https://www.math.mcgill.ca/yyang/resources/doc/randomforest.pdf
 */

package scalation
package modeling
package classifying

import scala.collection.mutable.Set

import scalation.mathstat._
import scalation.random.RandomVecI

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomForest` class uses randomness for building descision trees in classification.
 *  It randomly selects sub-samples with size = bRatio * sample-size from the sample
 *  (with replacement) and uses the fbRatio fraction of sub-features to build the trees,
 *  and to classify by voting from all of the trees.
 *  @param x       the data matrix (instances by features)
 *  @param y       the response class labels of the instances
 *  @param fname_  feature names (array of string)
 *  @param k       the number of classes
 *  @param cname_  class names (array of string)
 *  @param conts  the set of feature indices for variables that are treated as continuous
 *  @param hparam  the hyper-parameters
 */
class RandomForest (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
                    cname_ : Array [String] = Array ("No", "Yes"),
                    conts : Set [Int] = Set [Int] (), hparam: HyperParameter = DecisionTree.hp)
      extends BaggingTrees (x, y, fname_, k , cname_, conts, hparam):

    private val debug   = debugf ("RandomForest", true)                       // debug function
    private val flaw    = flawf ("RandomForest")                              // flaw function
    private val fbRatio = hparam("fbRatio").toDouble                          // feature bagging ratio

    private val nFeats  = (fbRatio * x.dim2).toInt                            // number of features/columns to select
    private val rvg     = RandomVecI (nFeats, x.dim2-1, 0, -1, true)          // random vector generator
    private val jcols   = Array.ofDim [VectorI] (nTrees)                      // record column indices for each tree

    if nFeats < 0 || nFeats > x.dim2 then flaw ("init", "RF feature size restricted to 0 thru number of features")

    _modelName = s"RandomForest_${height}_$nTrees"                            // name of the model

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Select subFeatures for input of building trees, return the subFeatures
     *  and the selected features index.
     *  @param sub_x  the subsample of data matrix x to select features/columns from
     */
    def selectSubFeatures (sub_x: MatrixD): (MatrixD, VectorI) =
        val columns = rvg.igen.sorted                                         // column indices selected
        val x_sub_f = sub_x(?, columns)                                       // extraxt selected columns
        (x_sub_f, columns)
    end selectSubFeatures

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a classification model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training set.
     *  Build the trees in the forest by first selecting the subSamples, then
     *  decide which features to use in spliting, then build trees.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        for l <- 0 until nTrees do                                            // iterate l-th tree
            val (sub_x, sub_y, irows) = subSample (x_, y_, sampleSize, l)     // select rows from x_ and elements from y_
            debug ("train", s"row indices for tree$l, irows = $irows")

            val (xf, columns) = selectSubFeatures (sub_x)                     // select columns of data matrix subsample
            val fname2 = columns.map (fname(_)).toArray                       // extract corresponding feature names       
            val conts2 = conts.filter (columns contains _)                    // extract corresponding cont indicators
            jcols(l)   = columns                                              // save for use by predictI/classify

            trees(l) = new DecisionTree_C45 (xf, sub_y, fname2, k, cname, conts2, hparam)
            trees(l).train ()
//          debug ("train", s"for tree$l === \n ${trees(l).printTree ()}")
        end for
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict/classify the vector z by voting within randomized trees, returning
     *  the class index.
     *  @param z  the vector to be classified
     */
    override def predictI (z: VectorD): Int =
        val vote = new VectorI (k)
        for l <- 0 until nTrees do                                            // iterate l-th tree
            val zp  = z(jcols(l))                                             // project onto columns for l-th tree
            val y_l = trees(l).predictI (zp)                                  // get vote from l-th tree
            vote(y_l) += 1                                                    // tally the vote
//          debug ("preictI", s"for tree$l, predicted class = y_l")
        end for
        vote.argmax ()                                                        // find argmax => the winner
    end predictI

end RandomForest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomForest` companion object provides a factory method.
 */
object RandomForest:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a random forsst for the given combined matrix where the column col
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
              (col: Int = xy.dim2 - 1): RandomForest =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new RandomForest (x, y, fname, k, cname, conts, hparam)
    end apply

end RandomForest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomForestTest` main function is used to test the `RandomForest` class.
 *  It tests a simple case that does not require a file to be read.
 *  > runMain scalation.modeling.classifying.randomForestTest
 */
@main def randomForestTest (): Unit =

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

    val y = VectorI (5, 5, 5, 7, 5, 7, 6, 8, 6, 5, 8)                         // response/class labels
    y -= 3                                                                    // shift the class labels by 3

    banner ("randomForestTest:  partial winequality-white dataset")
    val fname = Array ("fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                       "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                       "density", "pH", "sulphates", "alcohol")               // feature names
    val k     = 7
    val cname = Array ("Lev3", "Lev4", "Lev5", "Lev6", "Lev7", "Lev8", "Lev9")  // class names
    val conts = Set.range (0, x.dim2)                                         // all features are continuous

    val hp2 = DecisionTree.hp.updateReturn (("nTrees", 3.0), ("bRatio", 0.9))
    println (s"hp2 = $hp2")
    val mod = new RandomForest (x, y, fname, k, cname, conts, hp2)
    mod.trainNtest ()
    println (mod.summary ())

end randomForestTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomForestTest2` main function is used to test the `RandomForest` class.
 *  It tests the Random Forest classifier using well-known WineQuality Dataset.
 *  > runMain scalation.modeling.classifying.randomForestTest2
 */
@main def randomForestTest2 (): Unit =

    val nfile  = "winequality-white.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not(?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    y -= 3                                                                    // shift the class labels by 3

    banner ("randomForestTest2:  winequality-white dataset")
    val k     = 7
    val conts = Set.range (0, x.dim2)                                         // all features are continuous

    DecisionTree.hp("nTrees") = 3.0
    val mod = new RandomForest (x, y, null, k, null, conts)
    mod.trainNtest ()
    println (mod.summary ())

end randomForestTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomForestTest3` main function is used to test the `RandomForest` class.
 *  It tests the Random Forest classifier by specific numbers of trees.
 *  > runMain scalation.modeling.classifying.randomForestTest3
 */
@main def randomForestTest3 (): Unit =

    val nfile  = "winequality-white.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not(?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    y -= 3                                                                    // shift the class labels by 3

    banner ("randomForestTest3:  winequality-white dataset")
    val k     = 7
    val conts = Set.range (0, x.dim2)                                         // all features are continuous
    val maxTrees = 3

    for numTrees <- 1 to maxTrees do
        println (s"Number of Tree = $numTrees")
        
        DecisionTree.hp("nTrees") = numTrees
        val mod = new RandomForest (x, y, null, k, null, conts)
        mod.trainNtest ()()
        println (mod.summary ())
    end for

end randomForestTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomForestTest4` main function is used to test the `RandomForest` class.
 *  It tests RF using unseen data.
 *  > runMain scalation.modeling.classifying.randomForestTest4
 */
@main def randomForestTest4 (): Unit =

    val nfile  = "winequality-white.csv"
    val xy     = MatrixD.load (nfile)
    val ycol   = xy.dim2 - 1
    for i <- xy.indices do xy(i, ycol) -= 3                                   // shift the class labels by 3
    val (x, y) = (xy.not(?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
 
    banner ("randomForestTest4: winequality-white dataset")
    val k     = 7
    val conts = Set.range (0, x.dim2)                                         // all features are continuous

    // Divide samples into training and testing dataset
    val trainSize  = (y.dim * 0.8).toInt
    val rvv        = RandomVecI (min = 0, max = y.dim-1, dim = trainSize, unique = true, stream = 223)
    val subSample  = new MatrixD (trainSize, xy.dim2)
    val elseSample = new MatrixD (xy.dim-trainSize, xy.dim2)
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

    /* Starting training Forest */
    val hp2 = DecisionTree.hp.updateReturn (("nTrees", 5.0), ("bRatio", 0.64), ("fbRatio", 0.7))
    val mod = new RandomForest (subSample(?, 0 until subSample.dim2-1), subSample(?, subSample.dim2-1).toInt, 
                                null, k, null, conts, hp2)
    mod.trainNtest ()

    // Print the accuracy for unseen data
    var accurateCount = 0.0
    for i <- elseFeature.indices do
        if mod.predictI (elseFeature(i)) == elseTarget(i) then accurateCount += 1
    end for 
    val accuracy = accurateCount / elseFeature.dim
    println (s"Testing Accuracy = $accuracy")
    println (mod.summary ())

end randomForestTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomForestTest5` main function is used to test the `RandomForest` class.
 *  It tests the Random Forest classifier by specific numbers of trees.
 *  > runMain scalation.modeling.classifying.randomForestTest5
 */
@main def randomForestTest5 (): Unit =

    val nfile  = "breast_cancer.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not(?, xy.dim2-1), xy(?, xy.dim2-1).toInt)

    banner ("randomForestTest5: breast_cancer dataset")
    val maxTrees  = 4

    for numTrees <- 1 to maxTrees do
        println (s"Number of Tree = $numTrees}")
        DecisionTree.hp("nTrees") = numTrees
        val mod = new RandomForest (x, y) 
        mod.trainNtest ()
        println (mod.summary ())
    end for

end randomForestTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomForestTest6` main function is used to test the `RandomForest` class.
 *  It tests the Random Forest classifier by specific numbers of trees.
 *  > runMain scalation.modeling.classifying.randomForestTest6
 */
@main def randomForestTest6 (): Unit =

    val nfile  = "breast_cancer.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not(?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    val fname  = Array ("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                        "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses")
    val k      = 2
    val cname  = Array ("benign", "malignant")
    val conts  = Set.range (0, xy.dim2-1)

    banner ("randomForestTest6: Breast Cancer dataset")

    val hp2 = DecisionTree.hp.updateReturn (("nTrees", 10.0), ("bRatio", 0.7), ("fbRatio", 0.9))
    val mod = new RandomForest (x, y, fname, k, cname, conts, hp2)
    mod.trainNtest ()()
    println (mod.summary ())

end randomForestTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomForestTest7` main function is used to test the `RandomForest` class.
 *  It tests the Random Forest classifier by specific numbers of trees.
 *  > runMain scalation.modeling.classifying.randomForestTest7
 */
@main def randomForestTest7 (): Unit =

    val nfile  = "diabetes.csv"
    val xy     = MatrixD.load (nfile)
    val (x, y) = (xy.not(?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    val fname  = Array ("pregnancies", "glucose", "blood pressure", "skin thickness",
                        "insulin", "BMI", "diabetes pedigree function", "age")  // feature names
    val k      = 2
    val cname  = Array ("tested_positive", "tested_negative")                 // class names
    val conts  = Set.range (0, xy.dim2-1)

    banner ("randomForestTest7: diabetes dataset")

    val hp2 = DecisionTree.hp.updateReturn (("nTrees", 9.0), ("bRatio", 0.6), ("height", 7.0), ("fbRatio", 0.9))
    val mod = new RandomForest (x, y, fname, k, cname, conts, hp2)
    mod.trainNtest ()()
    println (mod.summary ())

end randomForestTest7

