
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dong Yu Yu, John Miller, Prudhvi Chekka, Lalithya Sajja
 *  @version 2.0
 *  @date    Wed Nov  7 17:08:17 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Regression Tree
 */

package scalation
package modeling

import scala.collection.mutable.{ArrayBuffer, Queue, Set}
import scala.math.abs
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
import scalation.random.RNGStream

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTree` companion object is used to count the number of leaves
 *  and provide factory methods for creating regression trees.
 */
object RegressionTree:

    val hp = new HyperParameter              // default hyper-parameter values
    hp += ("maxDepth",   5, 5)               // the maximum depth
    hp += ("threshold",  0.1, 0.1)           // threshold for parent node
    hp += ("cutoff",     0.01, 0.01)         // the cutoff (stop splitting) entropy threshold
    hp += ("bRatio",     0.7, 0.7)           // the bagging ratio (fraction of data/rows to be used in building trees)
    hp += ("fbRatio",    0.7, 0.7)           // the feature bagging ratio (fraction of features/columns to be used in building trees)
//  hp += ("nTrees",     9, 9)               // the (odd) number of trees to create for the forest (min: Supreme Court)
    hp += ("nTrees",     39, 39)             // the (odd) number of trees to create for the forest
    hp += ("iterations", 39, 39)             // number of iterations for gradient boosting
    hp += ("eta",        0.1, 0.1)           // learning rate for gradient boosting

    private val debug = debugf ("RegressionTree", false)                // debug function
    private val flaw  = flawf ("RegressionTree")                        // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTree` object from a combined data-response matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the names for all features/variables (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): RegressionTree =
        new RegressionTree (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTree` object from a data matrix and response vector.
     *  @param x       the data/input matrix
     *  @param y       the response/output vector
     *  @param fname   the names for all features/variables (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to hp)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): RegressionTree =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new RegressionTree (xn, y, fname, hparam)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given threshold thr, compute the sum of the left and right sse (sse total).
     *  Used for checking results.
     *  @param xj   the the j-th column in data matrix (used for splitting)
     *  @param y    the response vector
     *  @param thr  the given threshold
     *  @param ssy  the sum of squares for y
     */
    def sse_LR (xj: VectorD, y: VectorD, thr: Double, ssy: Double): Double =
        var nL, nR = 0
        var sL, sR = 0.0
        for i <- y.indices do
            if xj(i) <= thr then { sL += y(i); nL += 1 }
            else { sR += y(i); nR += 1 }
        end for
        ssy - (sL~^2 / nL) - (sR~^2 / nR)
    end sse_LR

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given column j, use fast threshold selection to find an optimal threshold/
     *  split point in O(NlogN) time.  Return the threshold and the sse total.
     *  @see people.cs.umass.edu/~domke/courses/sml/12trees.pdf
     *  @see www.dcc.fc.up.pt/~ltorgo/PhD/
     *  @param xj   the the j-th column in data matrix (used for splitting)
     *  @param y    the response vector
     *  @param ssy  the sum of squares for y
     */
    def fastThreshold (xj: VectorD, y: VectorD, ssy: Double): (Double, Double) =
        // indirectly sort the cases according to their value in xj
        val ord = xj.iqsort                                             // rank order according to xj   
        var (nL, nR) = (0,   xj.dim)                                    // number left, right
        var (sL, sR) = (0.0, y.sum)                                     // sum y left, right
        var (thr, hiScore) = (0.0, 0.0)                                 // threshold, its score and number left

        for i <- 0 until xj.dim - 1 do
            val yi = y(ord(i))                                          // i-th y-value according to order
            nL += 1 ; nR -= 1                                           // move one from right to left
            sL += yi; sR -= yi                                          // adjust corresponding sums
            val (xa, xb) = (xj(ord(i)), xj(ord(i+1)))
            if xb > xa then                                             // no trial if values are equal
                val newScore = (sL~^2 / nL) + (sR~^2 / nR)              // new split score
                if newScore > hiScore then                              // want higher score
                    hiScore = newScore
                    thr     = (xa + xb) / 2                             // set threshold to the midpoint
            end if
        end for

        debug ("fastThrehold", s"(thr, sse_LR) = ($thr, ${ssy - hiScore})")
        (thr, ssy - hiScore)                                            // return best (threshold, split sse)
    end fastThreshold

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check to make sure that the threshold and score are feasible.  For example,
     *  there must be at least two distinct values in xj for a split to work.
     *  @param d      the current depth
     *  @param j      the column index in data matrix
     *  @param xj     the j-th column vector in the data matrix
     *  @param y      the response vector
     *  @param thr    the selected threshhold
     *  @param ssy    the sum of squared y-values
     *  @param sse_t  the sum of squared errors total (left + right)
     */
    def check (d: Int, j: Int, xj: VectorD, y: VectorD, thr: Double, ssy: Double, sse_t: Double): Boolean =
        val (xj_lo, xj_hi) = (xj.min, xj.max)
        if thr < xj_lo || xj_hi < thr then flaw ("check", s"thr = $thr outside range of x$j: [$xj_lo, $xj_hi]")
        val sse_t_ = sse_LR (xj, y, thr, ssy)

        debug ("check", s"(d = $d) xj = x$j with threshold $xj_lo <= thr = $thr <= $xj_hi, sse_t = $sse_t, sse_t_ = $sse_t_")

        val okay = abs (sse_t - sse_t_) < 1E-6
        if ! okay then
            debug ("check", s"unable to split based on column j = $j, xj = $xj")
        okay
    end check

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split gives row indices of left and right children when splitting using thr.
     *  @param xj   the column/feature to use
     *  @param thr  the threshold for splitting (below => left, above => right) 
     */
    def split (xj: VectorD, thr: Double): Array [IndexedSeq [Int]] =
        val (sLeft, sRight) = (Set [Int] (), Set [Int] ())
        for i <- xj.indices do if xj(i) <= thr then sLeft += i else sRight += i
        Array (sLeft.toIndexedSeq, sRight.toIndexedSeq)
    end split

end RegressionTree

import RegressionTree.{check, fastThreshold, split}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Node` class contains information for a tree node.
 *  @param j        the feature/variable of the node used for splitting,
 *                      if it is leaf, contains the feature of its parent
 *  @param branch   the branch value (0 => left, 1 => right)
 *  @param b        leaf node's prediction parameters (b0 for mean or b for regression)
 *  @param thresh   the threshold for continuous feature
 *  @param depth    the current depth of the node
 *  @param pthresh  the threshold for parent node
 *  @param pfea     the feature of parent node
 *  @param leaf     whether the node is a leaf node
 */
case class Node (j: Int, branch: Int, b: VectorD, thresh: Double,
                 depth: Int, pthresh: Double, pfea: Int, leaf: Boolean = false):

    val child = new ArrayBuffer [Node] ()                               // children of node

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert node to a string.
     */
    override def toString: String =
        if child.length == 0 then
            s"Leaf (branch = $branch, feature = x$j, b = $b)"
        else if depth == 0 then
            s"Root (feature = x$j, threshold = $thresh)"
        else
            s"Node (branch = $branch, feature = x$j, threshold = $thresh)"
    end toString

end Node


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTree` class implements a Regression Tree that recursively
 *  partitions the dataset (x, y) by finding a threshold for each feature/x-variable.
 *  The threshold for a feature is the value that minimizes sseL + sseR,
 *  the sum of the "sum of squared errors".
 *  @param x            the m-by-n input/data matrix
 *  @param y            the output/response m-vector
 *  @param fname_       the names of the model's features/variables (defaults to null)
 *  @param hparam       the hyper-parameters for the model (defaults to RegressionTree.hp)
 *  @param curDepth     current depth (defaults to 0)
 *  @param branchValue  the branch value for the tree node (defaults to -1)
 *  @param feature      the feature for the tree's parent node (defaults to -1)
 *  @param use_r_fb     whether to use (by regression tree) feature bagging (fb) i.e.,
 *                      use a subset of the features, @see `RegressionTreeRF` with parameter `use_fb`
 *  @param leaves       the leaf counter (defaults to Counter ())
 */
class RegressionTree (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                      hparam: HyperParameter = RegressionTree.hp,
                      curDepth: Int = 0, branchValue: Int = -1,
                      feature: Int = -1, use_r_fb: Boolean = false,
                      leaves: Counter = Counter ())
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):              // call resetDF once tree is built

    private val debug     = debugf ("RegressionTree", false)            // debug function
    private val depth     = hparam("maxDepth").toInt                    // the depth limit for tree
    private val thres     = hparam("threshold").toDouble                // the threshold for the tree's parent node, @see buildTree
    private val fbRatio   = hparam("fbRatio").toDouble                  // the feature bagging ratio
    private val threshold = new VectorD (x.dim2)                        // store best splitting threshold for each feature
    private val score     = new VectorD (x.dim2)                        // store best splitting score for each feature

    private var root: Node = null                                       // root node   

    _modelName = s"RegressionTree_$depth"

    debug ("init", s"Construct a Regression Tree: curDepth = $curDepth")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of leaves in this tree.
     */
    def numLeaves: Int = leaves.get

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the regression tree by selecting thresholds for the features/variables
     *  in matrix x_.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit =
        if use_r_fb then return train_fb (x_, y_)                       // use the feature bagging
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
    /** Train the regression tree by selecting thresholds for the features/variables
     *  in matrix x_.  This method uses feature bagging (fb), e.g., as used in Random Forest.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train_fb (x_ : MatrixD, y_ : VectorD): Unit =
        val ssy        = y_.normSq                                      // sum of squared y
        val n_features = (x_.dim2 * fbRatio).toInt                      // number of features to select (fb)
        val strm       = RNGStream.ranStream                            // randomize the random number stream
        val (xFeat, iFeat) = featureSubSample (x_, n_features, stream = strm)  // or let strm = constant less than 1000
        val threshold_fb = new VectorD (xFeat.dim2)                     // threshold for selected features (fb)
        val score_fb     = new VectorD (xFeat.dim2)                     // score for selected features (fb)

        for j <- xFeat.indices2 do                                      // work with selected features
            val xj = xFeat(?, j)                                        // get j-th column
            val (thr, sse_t) = fastThreshold (xj, y_, ssy)              // get threshold and sse_t for feature j
            if check (curDepth, j, xj, y_, thr, ssy, sse_t) then
                threshold_fb(j) = thr                                   // set threshold for feature j
                score_fb(j)     = sse_t                                 // set sse_t for feature j
            else
                threshold_fb(j) = -0.0                                  // set threshold for feature j to NA
                score_fb(j)     = Double.PositiveInfinity               // set sse_t for feature j to infinity
        end for

        var k = 0
        for j <- x_.indices2 do                                         // map back to full feature set
            if iFeat contains j then
                threshold(j) = threshold_fb(k)
                score(j)     = score_fb(k)
                k += 1
            else
                threshold(j) = -0.0
                score(j)     = Double.PositiveInfinity

        val best_j = score.argmin ()                                    // use best feature/variable
        debug ("train", s"optimal (variable, score) = ($best_j, ${score(best_j)})")
        buildTree (x_, y_, best_j)
    end train_fb

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
                    (if curDepth == depth - 1 then                      // leaf node (removed || xx.dim <= xx.dim2)
                        val b0 = yy.mean                                // use mean
                        leaves.inc ()
                        Node (j, root.child.length, VectorD (b0), threshold(j), curDepth + 1,
                              threshold(j), j, true)
                    else
                        val hp = RegressionTree.hp.updateReturn ("threshold", threshold(j))
                        val subtree = new RegressionTree (xx, yy, fname, hp, curDepth + 1, i, j, use_r_fb, leaves)
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
        println (s"Regression Tree: leaves = $leaves")
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
        println ("RegressionTree:")
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
        nd.b(0)                                                         // b0 is the mean
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
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): RegressionTree =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new RegressionTree (x_cols, y, fname2, hparam)
    end buildModel

end RegressionTree


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeTest` main function is used to test the `RegressionTree` class.
  *  It tests a simple case that does not require a file to be read.
  *  @see translate.google.com/translate?hl=en&sl=zh-CN&u=https:
  *       //www.hrwhisper.me/machine-learning-decision-tree/&prev=search
  *  > runMain scalation.modeling.regressionTreeTest
  */
@main def regressionTreeTest (): Unit =

    val x  = MatrixD ((10, 1), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val y  = VectorD (5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05)
    val ox = VectorD.one (x.dim) +^: x
    val fname = Array ("x")

    banner (s"Regression no intercept")
    val reg = new Regression (x, y)
    reg.trainNtest ()()                                                 // train and test the model
    println (s"rSq0_ = ${reg.rSq0_}")

    banner (s"Regression with intercept")
    val reg2 = new Regression (ox, y)
    reg2.trainNtest ()()                                                // train and test the model

    banner (s"Quadratic Regression")
    val reg3 = SymbolicRegression.quadratic (x, y, fname)
    reg3.trainNtest ()()                                                // train and test the model

    banner (s"Perceptron sigmoid")
    val nn = Perceptron.rescale (ox, y)
//  val nn = Perceptron.rescale (reg3.getX, y)
    nn.trainNtest ()()                                                  // train and test the model

    banner (s"Perceptron tanh")
    val nn2 = Perceptron.rescale (ox, y, f = ActivationFun.f_tanh)
//  val nn2 = Perceptron.rescale (reg3.getX, y, f = ActivationFun.f_tanh)
    nn2.trainNtest ()()                                                 // train and test the model

    for d <- 1 to 2 do
        banner (s"Regression Tree with depth = $d")
        RegressionTree.hp("maxDepth") = d
        val mod = new RegressionTree (x, y, fname)
        mod.trainNtest ()()                                             // train and test the model
        mod.printTree ()
    end for

end regressionTreeTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeTest2` main function tests the `RegressionTree` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests multiple depths.
 *  > runMain scalation.modeling.regressionTreeTest2
 */
@main def regressionTreeTest2 (): Unit =

    import Example_AutoMPG._

//  println (s"x = $x")
//  println (s"y = $y")

    val dmax    = 6                                                     // range of depths 1 to dmax
    val qual = new MatrixD (dmax, 3)

    for d <- 1 to dmax do
        banner (s"AutoMPG Regression Tree with d = $d")
        RegressionTree.hp ("maxDepth") = d
        val mod = new RegressionTree (x, y, x_fname)                    // create model with intercept (else pass x)
        val qof = mod.trainNtest ()()._2                                // train and test the model
        mod.printTree ()                                                // print the regression tree
//      println (mod.summary ())                                        // parameter/coefficient statistics
      
        banner (s"AutoMPG Regression Tree with d = $d Validation")
        val qof2 = mod.validate ()()._2                                 // out-of-sampling testing
        val iq = QoF.rSq.ordinal                                        // index for rSq
        qual (d-1) = VectorD (qof(iq), qof(iq+1), qof2(iq))             // R^2, R^2 bar, R^2 os
    end for

    new PlotM (VectorD.range (1, dmax+1), qual.ᵀ, Array ("R^2", "R^2 bar", "R^2 os"),
               "RegressionTree in-sample, out-of-sample QoF vs. depth", lines = true)
    println (s"RegressionTree: qual = $qual")

end regressionTreeTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeTest3` main function tests the `RegressionTree` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests forward, backward and
 *  stepwise selection.
 *  > runMain scalation.modeling.regressionTreeTest3
 */
@main def regressionTreeTest3 (): Unit =

    import Example_AutoMPG._

    val d = 5                                                           // depth of tree

//  println (s"x = $x")
//  println (s"y = $y")

    banner (s"AutoMPG Regression Tree with d = $d")
    RegressionTree.hp("maxDepth") = d
    val mod = new RegressionTree (x, y, x_fname)                        // create model with intercept (else pass x)
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

end regressionTreeTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeTest4` main function tests the `RegressionTree` class using the
 *  Boston House Prices dataset.
 *  > runMain scalation.modeling.regressionTreeTest4
 */
@main def regressionTreeTest4(): Unit =

    val data = MatrixD.load ("boston_house_prices.csv", 1, 0)
    val x    = data(?, 0 until data.dim2 - 1)
    val y    = data(?, data.dim2 - 1)
    val dmax = 5

    for d <- 1 to dmax do
        banner (s"AutoMPG Regression Tree with d = $d")
        RegressionTree.hp("maxDepth") = d
        val mod = new RegressionTree (x, y)                             // create model with intercept (else pass x)
        mod.trainNtest ()()._2                                          // train and test the model
        mod.printTree ()
    end for

end regressionTreeTest4

