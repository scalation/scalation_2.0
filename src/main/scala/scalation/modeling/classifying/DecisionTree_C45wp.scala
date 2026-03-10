
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Susan George
 *  @version 2.0
 *  @date    Wed May 22 14:17:49 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: C45 Decision/Classification Tree with Pruning
 */

package scalation
package modeling
package classifying

import scala.collection.mutable.Set

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree_C45wp` class extends `DecisionTree_C45` with pruning capabilities.
 *  The base class uses the C45 algorithm to construct a decision tree for classifying
 *  instance vectors.
 *  @param x       the input/data matrix with instances stored in rows
 *  @param y       the response/classification vector, where y_i = class for row i of matrix x
 *  @param fname_  the names for all features/variables
 *  @param k       the number of classes
 *  @param cname_  the names for all classes
 *  @param conts   the set of feature indices for variables that are treated as continuous
 *  @param hparam  the hyper-parameters
 */
class DecisionTree_C45wp (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
                          cname_ : Array [String] = Array ("No", "Yes"), 
                          conts: Set [Int] = Set [Int] (), hparam: HyperParameter = DecisionTree.hp)
      extends DecisionTree_C45 (x, y, fname_, k, cname_, conts, hparam):

    private val debug = debugf ("DecisionTree_C45wp", true)      // debug function

    _modelName = "DecisionTree_C45wp"                            // name of the model

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

end DecisionTree_C45wp


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree_C45wp` companion object provides a factory function.
 */
object DecisionTree_C45wp:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a decision tree (with pruning) for the given combined matrix where
     *  the column col is the response/classification vector.
     *  @param xy      the combined data matrix (features and response)
     *  @param fname   the names for all features/variables
     *  @param k       the number of classes
     *  @param cname   the names for all classes
     *  @param conts   the set of feature indices for variables that are treated as continuous
     *  @param hparam  the hyper-parameters
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null, k: Int = 2,
               cname: Array [String] = Array ("No", "Yes"),
               conts: Set [Int] = Set [Int] (), hparam: HyperParameter = DecisionTree.hp)
              (col: Int = xy.dim2 - 1): DecisionTree_C45wp =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new DecisionTree_C45wp (x, y, fname, k, cname, conts, hparam)
    end apply

end DecisionTree_C45wp


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_C45wpTest` main function tests the `DecisionTree_C45wp` class.
 *  > runMain scalation.modeling.classifying.decisionTree_C45wpTest
 */
@main def decisionTree_C45wpTest (): Unit =

    import Example_PlayTennis.{xy, fname, k, cname}

    val tree = DecisionTree_ID3wp (xy, fname, k, cname)()
    tree.trainNtest ()
    banner ("Orignal Tree: entropy = " + tree.calcEntropy ())
    tree.printTree ()

    tree.prune (2)
    banner ("Pruned Tree: entropy = " + tree.calcEntropy ())
    tree.printTree ()

end decisionTree_C45wpTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_C45wpTest2` main function tests the `DecisionTree_C45wp` class.
 *  > runMain scalation.modeling.classifying.decisionTree_C45wpTest2
 */
@main def decisionTree_C45wpTest2 (): Unit =

    val nfile = "breast_cancer.csv"
    val xy    = MatrixD.load (nfile)
    val fname = Array ("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                       "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses")
    val cname = Array ("benign", "malignant")
    val k     = cname.size
    val conts = Set.range (0, xy.dim2 - 1)

    banner ("create, train and print a C45 decision tree")
    println (s"dataset xy: ${xy.dim}-by-${xy.dim2} matrix")
    val (x, y) = (xy.not (?, xy.dim2-1), xy(?, xy.dim2-1).toInt)
    val ymin   = y.min
    println (s"unadjusted ymin = $ymin")
    if ymin != 0 then y -= ymin

    val tree = new DecisionTree_C45wp (x, y, fname, k, cname, conts)
    tree.trainNtest ()()
    tree.printTree ()
    tree.prune ()
    tree.printTree ()

end decisionTree_C45wpTest2

