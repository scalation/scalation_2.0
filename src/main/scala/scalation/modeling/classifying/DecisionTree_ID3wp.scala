
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Susan George
 *  @version 2.0
 *  @date    Wed May 22 14:17:49 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: ID3 Decision/Classification Tree with Pruning
 */

package scalation
package modeling
package classifying

import scalation.mathstat._
import scalation.random.RandomVecI

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree_ID3wp` class extends `DecisionTree_ID3` with pruning capabilities.
 *  The base class uses the ID3 algorithm to construct a decision tree for classifying
 *  instance vectors.
 *  @param x       the input/data m-by-n matrix with instances stored in rows
 *  @param y       the response/classification m-vector, where y_i = class for row i of matrix x
 *  @param fname_  the name for each feature/variable xj
 *  @param k       the number of classes
 *  @param cname_  the name for each class
 *  @param hparam  the hyper-parameters
 */
class DecisionTree_ID3wp (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
                          cname_ : Array [String] = Array ("No", "Yes"),
                          hparam: HyperParameter = DecisionTree.hp)
      extends DecisionTree_ID3 (x, y, fname_, k, cname_, hparam):

    private val debug = debugf ("DecisionTree_ID3wp", true)      // debug function

    _modelName = "DecisionTree_ID3wp"                            // name of the model

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Prune nPrune nodes from the tree, the ones providing the least gain.
     *  @param nPrune     the number of nodes to be pruned.
     *  @param threshold  cut-off for pruning (IG < threshold, then prune)
     */
    def prune (nPrune: Int = 1, threshold: Double = 0.98): Unit =
        cfor (0, nPrune) { _ => 
            val can = candidates
            debug ("prune", s"can = $can")
            val (best, gn) = bestCandidate (can)
            println (s"prune: node $best with gain $gn identfied as bestCandidate")
            if gn < threshold then
                println (s"prune: make node $best with gain $gn into a leaf")
                makeLeaf (best)
        } // cfor
    end prune

end DecisionTree_ID3wp


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree_ID3wp` companion object provides a factory function.
 */
object DecisionTree_ID3wp:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a decision tree (with pruning) for the given combined matrix where
     *  the column col is the response/classification vector.
     *  @param xy      the combined data matrix (features and response)
     *  @param fname   the names for all features/variables
     *  @param k       the number of classes
     *  @param cname   the names for all classes
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixI, fname: Array [String] = null, k: Int = 2,
               cname: Array [String]  = Array ("No", "Yes"),
               hparam: HyperParameter = DecisionTree.hp)
              (col: Int = xy.dim2 - 1): DecisionTree_ID3wp =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new DecisionTree_ID3wp (x, y, fname, k, cname, hparam)
    end apply

end DecisionTree_ID3wp


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_ID3wpTest` main function tests the `DecisionTree_ID3wp` class.
 *  > runMain scalation.modeling.classifying.decisionTree_ID3wpTest
 */
@main def decisionTree_ID3wpTest (): Unit =

    import Example_PlayTennis.{xy, fname, k, cname}

    val tree = DecisionTree_ID3wp (xy, fname, k, cname)()
    tree.trainNtest ()
    banner ("Orignal Tree: entropy = " + tree.calcEntropy ())
    tree.printTree ()

    tree.prune (2)
    banner ("Pruned Tree: entropy = " + tree.calcEntropy ())
    tree.printTree ()

end decisionTree_ID3wpTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_ID3wpTest2` main function tests the `DecisionTree_ID3wp` class.
 *  > runMain scalation.modeling.classifying.decisionTree_ID3wpTest2
 */
@main def decisionTree_ID3wpTest2 (): Unit =

    val nfile = "breast_cancer.csv"
    val xy    = MatrixD.load (nfile)
    val fname = Array ("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                       "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses")
    val cname = Array ("benign", "malignant")
    val k     = cname.size

    banner ("create, train and print a ID3 decision tree")
    println (s"dataset xy: ${xy.dim}-by-${xy.dim2} matrix")
    val (x, y) = (xy.not (?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    val ymin   = y.min
    println (s"unadjusted ymin = $ymin")
    if ymin != 0 then y -= ymin

    val tree = new DecisionTree_ID3wp (x, y, fname, k, cname)
    tree.trainNtest ()()
    tree.printTree ()
    tree.prune ()
    tree.printTree ()

end decisionTree_ID3wpTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_ID3wpTest3` main function tests the `DecisionTree_ID3wp` class.
 *  > runMain scalation.modeling.classifying.decisionTree_ID3wpTest3
 */
@main def decisionTree_ID3wpTest3 (): Unit =

    val nfile = "breast_cancer.csv"
    val xy    = MatrixD.load (nfile)
    val fname = Array ("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                       "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses")
    val cname = Array ("benign", "malignant")
    val k     = cname.size

    val y = xy(?, xy.dim2-1).toInt
    val ymin   = y.min
    println (s"unadjusted ymin = $ymin")
    if ymin != 0 then y -= ymin

    // Divide samples into training and testing dataset
    val trainSize  = (y.dim * 0.7).toInt
    val rvv        = RandomVecI (min = 0, max = y.dim-1, dim = trainSize, unique = true, stream = 223)
    val trainData  = new MatrixI (trainSize, xy.dim2)
    val testData   = new MatrixI (xy.dim-trainSize, xy.dim2)
    val index      = rvv.igen
    var trainCount = 0
    var testCount  = 0

    for i <- y.indices do
        if index contains i then
            trainData.set (trainCount, xy(i))
            trainCount += 1
        else
            testData.set (testCount, xy(i))
            testCount  += 1
    end for

    val testFeature = testData(0 until testData.dim2)
    val testTarget  = testData(?, testData.dim2-1)

    val tree = new DecisionTree_ID3wp (trainData(0 until xy.dim2),
                                       trainData(?, trainData.dim2).toInt, fname, k, cname)
    tree.train ()

    // Print the accuracy for unseen data
    var accurateCount = 0.0
    for i <- testFeature.indices do
        if tree.classify (testFeature(i))._1 == testTarget(i) then accurateCount += 1
    end for
    var accuracy = accurateCount / testFeature.dim
    println (s"Testing Accuracy = $accuracy")

    tree.prune (5)
    accurateCount = 0.0
    for i <- testFeature.indices do
        if tree.classify (testFeature(i))._1 == testTarget(i) then accurateCount += 1
    end for
    accuracy = accurateCount / testFeature.dim
    println (s"Testing Accuracy = $accuracy")

end decisionTree_ID3wpTest3

