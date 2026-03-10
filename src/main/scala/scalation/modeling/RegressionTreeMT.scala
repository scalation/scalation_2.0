
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dong Yu Yu, John Miller
 *  @version 2.0
 *  @date    Wed Nov  7 17:08:17 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Regression Tree (Model Tree)
 */

package scalation
package modeling

import scala.collection.mutable.Queue
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

// FIX - rSqBar from validate is wrong

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeMT` companion object is used to count the number of leaves
 *  and provide factory methods for creating regression model trees.
 */
object RegressionTreeMT:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTree` object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the names for all features/variables (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to RegressionTree.hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = RegressionTree.hp)
              (col: Int = xy.dim2 - 1): RegressionTreeMT =
        new RegressionTreeMT (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTree` object from a data matrix and response vector,
     *  where the data is rescaled.
     *  @param x       the input/data matrix
     *  @param y       the output/response vector
     *  @param fname   the names for all features/variables (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to RegressionTree.hp)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
               hparam: HyperParameter = RegressionTree.hp): RegressionTreeMT =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new RegressionTreeMT (xn, y, fname, hparam)
    end rescale

end RegressionTreeMT

import RegressionTree.{check, fastThreshold, split}

// FIX - implemenent feature bagging

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeMT` class implements a Regression Tree (MT) that selects splitting features
 *  using minimal variance in children nodes to avoid exponential choices in the selection,
 *  supporting ordinal features currently.
 *  @param x            the m-by-n input/data matrix
 *  @param y            the output/response m-vector
 *  @param fname_       the names of the model's features/variables (defaults to null)
 *  @param hparam       the hyper-parameters for the model (defaults to RegressionTree.hp)
 *  @param curDepth     current depth (defaults to 0)
 *  @param branchValue  the branch value for the tree node (defaults to -1)
 *  @param feature      the feature for the tree's parent node (defaults to -1)
 *  @param feature      the feature for the tree's parent node (defaults to -1)
 *  @param use_r_fb     whether to use (by regression tree) feature bagging (fb) i.e.,
 *                      use a subset of the features, @see `RegressionTreeRF` with parameter `use_fb`
 *  @param leaves       the leaf counter (defaults to Counter ())
 */
class RegressionTreeMT (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                        hparam: HyperParameter = RegressionTree.hp, curDepth: Int = 0,
                        branchValue: Int = -1, feature: Int = -1,
                        use_r_fb: Boolean = false, leaves: Counter = Counter ())
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):              // call resetDF once tree is built

    private val debug     = debugf ("RegressionTreeMT", true)           // debug function
    private val depth     = hparam("maxDepth").toInt                    // the depth limit for tree
    private val thres     = hparam("threshold").toDouble                // the threshold for the tree's parent node, @see buildTree
    private val threshold = new VectorD (x.dim2)                        // store best splitting threshold for each feature
    private val score     = new VectorD (x.dim2)                        // store best splitting score for each feature

    private var root: Node = null                                       // root node   

    _modelName = s"RegressionTreeMT_$depth"

    debug ("init", s"Construct a Regression Tree (MT): curDepth = $curDepth")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of leaves in this tree.
     */
    def numLeaves = leaves.get

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the regression tree by selecting thresholds for the features/variables
     *  in matrix x_.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit =
        val ssy = y_.normSq                                             // sum of squared y
        for j <- x_.indices2 do
            val xj = x_(?, j)                                           // get j-th column     
            val (thr, sse_t) = fastThreshold (xj, y_, ssy)              // get threshold and sse_t for feature j
            if check (curDepth, j, xj, y_, thr, ssy, sse_t) then
                threshold(j) = thr                                      // set threshold for feature j
                score(j)     = sse_t                                    // set sse_t for feature j
            else
                threshold(j) = -0.0                                     // set threshold for feature j to NA
                score(j)     = Double.PositiveInfinity                  // set sse_t for feature j to infinity
        end for

        val best_j = score.argmin ()                                    // use best feature/variable
        debug ("train", s"optimal (variable, score) = ($best_j, ${score(best_j)})")
        buildTree (x_, y_, best_j)
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the next most distinguishing feature/attribute, extend the regression tree.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     *  @param j   the optimal feature/variable to use for splitting
     */
    private def buildTree (x_ : MatrixD, y_ : VectorD, j: Int): Unit =
        root =
            if curDepth == 0 then
                Node (j, -1, VectorD (y.mean), threshold(j), curDepth, -1.0, -1)
            else
                Node (j, branchValue, VectorD (y.mean), threshold(j), curDepth, thres, feature)
        debug ("buildTree", s"--> Add root = ${root}")

        val child = split (x_(?, j), threshold(j))                      // row indices of left and right children
        for i <- 0 to 1 do                                              // 0 => left, 1 => right
            val (xx, yy) = (x_(child(i)), y_(child(i)))                 // get left or right child
            if yy.dim != 0 then

                root.child +=
                    (if curDepth == depth - 1 || xx.dim <= xx.dim2 then    // leaf node
                        val b = if xx.dim <= xx.dim2 then
                            VectorD (yy.mean)                              // use mean b0
                        else
                            val mod = new Regression (xx, yy, fname)       // use regression with no intercept
//                          val mod = new Regression (VectorD.one (xx.dim) +^: xx, yy, fname)   // use regression with intercept
                            mod.train ()
                            mod.parameter

                        leaves.inc ()
                        Node (j, root.child.length, b, threshold(j), curDepth + 1,
                              threshold(j), j, true)
                    else
                        val hp = RegressionTree.hp.updateReturn ("threshold", threshold(j))
                        val subtree = new RegressionTreeMT (xx, yy, fname, hp, curDepth + 1, i, j, use_r_fb, leaves)
                        subtree.train (xx, yy)
                        subtree.root)

                debug ("buildTree", s"--> Add child = ${root.child}")
            end if
        end for
    end buildTree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = predict (x_)                                           // make predictions
        val df1 = leaves.get                                            // degrees of freedom model = number of leaves
        val df2 = y_.dim - df1                                          // degrees of freedom error
        resetDF ((df1, df2))
        (yp, diagnose (y_, yp))                                         // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the regression tree in Pre-Order using printT method. 
     */
    def printTree (): Unit =
        println (s"Regression Tree (MT): leaves = $leaves")
        println ("fname = " + stringOf (fname))
        printT (root, 0)
        println ()
    end printTree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Recursively print the regression tree nodes.
     *  @param nod     the current node
     *  @param level   the level of node nod in the tree
     */
    private def printT (nod: Node, level: Int): Unit =
        println ("\t" * level + "[ " + nod + " ]")
        for cnode <-nod.child do printT (cnode, level + 1)
    end printT

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print out the regression tree using Breadth First Search (BFS).
     */
    def printTree2 (): Unit =
        println ("RegressionTreeMT:")
        println ("fname = " + stringOf (fname))
        val queue = new Queue [Node] ()

        for cnode <- root.child do queue += cnode
        println (root)
        var level = 0

        while ! queue.isEmpty do
            val size = queue.size
            level   += 1
            cfor (0, size) { _ =>
                val nod = queue.dequeue ()
                println ("\t" * level + "[ " + nod + " ]")
                for cnode <- nod.child do queue += cnode
            } // cfor
            println ()
        end while
    end printTree2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a data vector z, predict the value by following the tree to the leaf.
     *  @param z  the data vector to predict
     */
    override def predict (z: VectorD): Double =
        var nd = root                                                   // current node
        while nd.child.length >= 2 do
            nd = if z(nd.j) <= nd.thresh then nd.child(0) else nd.child(1)
        end while
        if nd.b.dim == 1 then nd.b(0)                                   // use mean b_0
        else nd.b dot z                                                 // use regression model
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a data matrix z, predict the value by following the tree to the leaf.
     *  @param z  the data matrix to predict
     */
    override def predict (z: MatrixD = x): VectorD =
        VectorD (for i <- z.indices yield predict (z(i)))
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String]): RegressionTreeMT =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new RegressionTreeMT (x_cols, y, fname2, hparam)
    end buildModel

end RegressionTreeMT


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMTTest` main function is used to test the `RegressionTreeMT` class.
  *  @see translate.google.com/translate?hl=en&sl=zh-CN&u=https:
  *       //www.hrwhisper.me/machine-learning-decision-tree/&prev=search
  *  > runMain scalation.modeling.regressionTreeMTTest
  */
@main def regressionTreeMTTest (): Unit =

    val x  = MatrixD ((10, 1), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val y  = VectorD (5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05)
    val ox = VectorD.one (x.dim) +^: x
    val fname = Array ("x")

    banner (s"Regression no intercept")
    val reg = new Regression (x, y)
    reg.trainNtest ()()                                                 // train and test the model

    banner (s"Regression with intercept")
    val reg2 = new Regression (ox, y)
    reg2.trainNtest ()()                                                // train and test the model

    banner (s"Quadratic Regression")
    val reg3 = SymbolicRegression.quadratic (x, y, fname)
    reg3.trainNtest ()()                                                // train and test the model

    banner (s"Perceptron sigmoid")
    val nn = Perceptron.rescale (reg3.getX, y)
    nn.trainNtest ()()                                                  // train and test the model

    banner (s"Perceptron tanh")
    val nn2 = Perceptron.rescale (reg3.getX, y, f = ActivationFun.f_tanh)
    nn2.trainNtest ()()                                                 // train and test the model

    for d <- 1 to 2 do
        banner (s"Regression Tree MT with depth = $d")
        RegressionTree.hp("maxDepth") = d
        val mod = new RegressionTreeMT (x, y, fname)
        mod.trainNtest ()()                                             // train and test the model
        mod.printTree ()
    end for

end regressionTreeMTTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMTTest2` main function tests the `RegressionTreeMT` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests multiple depths.
 *  > runMain scalation.modeling.regressionTreeMTTest2
 */
@main def regressionTreeMTTest2 (): Unit =

    import Example_AutoMPG._

//  println (s"x = $o")
//  println (s"y = $y")

    val dmax    = 6                                                     // range of depths 1 to dmax
    val qual = new MatrixD (dmax, 3)

    for d <- 1 to dmax do
        banner (s"AutoMPG Regression MT Tree with d = $d")
        RegressionTree.hp ("maxDepth") = d
        val mod = new RegressionTreeMT (x, y, x_fname)                  // create model with intercept (else pass x)
        val qof = mod.trainNtest ()()._2                                // train and test the model
        mod.printTree ()                                                // print the regression tree
//      println (mod.summary ())                                        // parameter/coefficient statistics

        banner (s"AutoMPG Regression Tree MT with d = $d Validation")
        val qof2 = mod.validate ()()._2                                 // out-of-sampling testing
        val iq = QoF.rSq.ordinal                                        // index for rSq
        qual (d-1) = VectorD (qof(iq), qof(iq+1), qof2(iq))             // R^2, R^2 bar, R^2 os
    end for

    new PlotM (VectorD.range (1, dmax+1), qual.ᵀ, Array ("R^2", "R^2 bar", "R^2 os"),
               "RegressionTreeMT in-sample, out-of-sample QoF vs. depth", lines = true)
    println (s"RegressionTreeMT: qual = $qual")

end regressionTreeMTTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeMTTest3` main function tests the `RegressionTreeMT` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests forward, backward and stepwise
 *  selection.
 *  > runMain scalation.modeling.regressionTreeMTTest3
 */
@main def regressionTreeMTTest3 (): Unit =

    import Example_AutoMPG._

    val d = 5                                                           // depth of tree

//  println (s"x = $x")
//  println (s"y = $y")

    banner (s"AutoMPG Regression Tree MT with d = $d")
    RegressionTree.hp ("maxDepth") = d
    val mod = new RegressionTreeMT (x, y, x_fname)                      // create model with intercept (else pass x)
    mod.trainNtest ()()                                                 // train and test the model
    mod.printTree ()                                                    // print the regression tree

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                     // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Regression Tree with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end regressionTreeMTTest3

