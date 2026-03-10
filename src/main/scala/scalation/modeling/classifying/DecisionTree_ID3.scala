
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Kevin Warrick, Susan George
 *  @version 2.0
 *  @date    Wed May 22 14:17:49 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: ID3 Decision/Classification Tree
 */

package scalation
package modeling
package classifying

import scala.collection.mutable.ArrayBuffer

import scalation.mathstat._
import scalation.mathstat.Probability.{entropy, freq}
import scalation.theory.Variable
import scalation.theory.VariableKind.Categorical

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree_ID3` class implements a Decision Tree classifier using the
 *  ID3 algorithm.  The classifier is trained using a data matrix x and a
 *  classification vector y.  Each data vector in the matrix is classified into
 *  one of k classes numbered 0, ..., k-1.  Each column in the matrix represents
 *  a feature (e.g., Humidity).
 *  @param x       the input/data m-by-n matrix with instances stored in rows
 *  @param y       the response/classification m-vector, where y_i = class for row i of matrix x
 *  @param fname_  the name for each feature/variable xj
 *  @param k       the number of classes
 *  @param cname_  the name for each class
 *  @param hparam  the hyper-parameters
 */
class DecisionTree_ID3 (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
                        cname_ : Array [String] = Array ("No", "Yes"),
                        hparam: HyperParameter = DecisionTree.hp)
      extends Classifier (x, y, fname_, k, cname_, hparam)
         with FitC (k)
         with DecisionTree:

    private val debug     = debugf ("DecisionTree_ID3", false)           // debug function
    private val height    = hparam("height").toInt                       // the maximum height of tree
    private val cutoff    = hparam("cutoff")                             // cutoff entropy threshold

    private var entropy_0 = entropy (y.freq (k)._2)                      // initial entropy of full vector y
    private val param     = ArrayBuffer [Double] ()                      // parameter vector = feature order
    private val feas      = Array.ofDim [Variable] (x.dim2)              // array of features/variables xj's
    for j <- x.indices2 do feas(j) = Variable (x(?, j), j, Categorical) 

    _modelName = s"DecisionTree_ID3_$height"                             // name of the model

    debug ("init", s"entropy of original/full y: entropy_0 = $entropy_0")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of model parameter (feature order) vector.
     */
    override def parameter: VectorD = VectorD (param)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train a classification model y_ = f(x_) + e where x_ is the data/input
     *  matrix and y_ is the response/output vector.  These arguments default
     *  to the full dataset x and y, but may be restricted to a training set.
     *  Training involves building a decision tree where the entropy of leaves is small.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        super.train (x_, y_)                                             // set class frequencies nu_y and probabilities p_y
        leaves.clear ()
        entropy_0  = entropy (y_.freq (k)._2)                            // initial entropy of vector y_ (also value for root)
        val rindex = VectorI.range (0, x_.dim)                           // initially use all rows in x_ for row index
        val cindex = VectorI.range (0, x_.dim2)                          // initially use all columns in x_ for column index
        buildTree (x_, y_, rindex, cindex)
        debug ("train", s"entropy of the ${leaves.size} leaves = ${calcEntropy ()}")
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the information gain due to using the values of a feature
     *  to distinguish the training cases (e.g., how well does Humidity with its
     *  values Normal and High indicate whether one will play tennis).
     *  @param fea     the feature to consider (e.g., 2 (Humidity))
     *  @param xj      the vector for feature fea (column j of matrix)
     *  @param y_      the training/full response/output vector
     *  @param rindex  the working row index
     */
    private def gain (fea: Variable, xj: VectorI, y_ : VectorI, rindex: VectorI): (Double, VectorI) =
        val nu  = new VectorI (k)                                             // aggregate frequency vector
        var sum = 0.0
        for v <- fea.values do
            val (frac_v, nu_v) = freq (xj, y_, k, v, rindex)                  // frequency for value v
//          debug ("gain", s" (v = $v): (frac_v, nu_v) = ($frac_v, $nu_v")
            sum += frac_v * entropy (nu_v)                                    // weighted entropy
            nu  += nu_v                                                       // aggregate frequency vector
        end for
        val igain = entropy_0 - sum                                           // the drop in entropy = information gain
//      debug ("gain", s"entropy = $sum, overall gain from feature ${fea.j} = $igain")
        (igain, nu)                                                           // return gain and aggregate frequency vector
    end gain

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the best feature j / column xj to expand the decision tree,
     *  returning that feature, its gain and its frequency vector.
     *  Note: the dataset is restricted to rindex rows and cindex columns.
     *  @param x_      the training/full data/input matrix
     *  @param y_      the training/full response/output vector
     *  @param rindex  the working row index
     *  @param cindex  the working column index
     */
    private def findBest (x_ : MatrixD, y_ : VectorI,
                          rindex: VectorI, cindex: VectorI): (Int, Double, VectorI) =
        var best = (-1, 0.0, null.asInstanceOf [VectorI])                     // best (feature, gain, frequency)
        for j <- cindex do
            val (gn, nu) = gain (feas(j), x_(?, j).toInt, y_, rindex)         // compute gain for feature j
//          debug ("findBest", s"compare ($j, $gn, $nu) to $best")
            if gn > best._2 then best = (j, gn, nu)                           // better gain => update best
        end for
//      if best._2 <= 0.0 then println ("findBest: no positive gain found")
        best
    end findBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively build the decision tree until entropy drops to the cutoff
     *  threshold or the tree depth is at the specified tree height.
     *  @param x_      the training/full data/input matrix
     *  @param y_      the training/full response/output vector
     *  @param rindex  the working row index
     *  @param cindex  the working column index
     *  @param parent  the parent node (== null => at root)
     *  @param depth   the depth of the subtree being built
     */
    private def buildTree (x_ : MatrixD, y_ : VectorI, rindex: VectorI, cindex: VectorI,
                           parent: Node = null, depth: Int = 0): Node =
        val (j, gn, nu) = findBest (x_, y_, rindex, cindex)                   // find the best feature
        debug ("buildTree", s"best feature (j, gn, nu) = ($j, $gn, $nu), depth = $depth")
        if j < 0 then return null                                             // no useful feature was found

        param += j                                                            // add feature j as next parameter             
        val node = Node (j, gn, nu, parent, nu.argmax (),                     // construct the next node
                         entropy (nu) <= cutoff || depth >= height)           // leaf or internal?
        if parent == null then
            addRoot (node)                                                    // if no parent, add node as root of tree
            debug ("buildTree", s"entropy of root node: entropy_0 = $entropy_0")

        if ! node.leaf && cindex.dim > 1 then
            val xj      = x_(?, j).toInt                                      // extract feature column j
            val cindex2 = cindex diff VectorI (j)                             // remove column j from column index
            debug ("buildTree", s"removing column j = $j gives cindex2 = $cindex2")

            for v <- feas(j).values do                                        // build subtree or leaf for each branch value
                debug ("buildTree", s"explore branch $v for feature x$j at depth $depth")
                val rindex2 = trimRows (xj, rindex, v)                        // trim row index to those matching value v
                val child = buildTree (x_, y_, rindex2, cindex2, node, depth+1)    // build a subtree
                if child != null then add (node, v, child)                    // if exists, add child to tree
            end for
        end if
        node
    end buildTree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Trim the row index by only including those where column xj == v,
     *  returning the newly trimmed row index.
     *  @param xj      the column of the data matrix to be considered
     *  @param rindex  the working row index used to create the new trimmed version
     *  @param v       the value to matched
     */
    private def trimRows (xj: VectorI, rindex: VectorI, v: Int): VectorI =
        val a = (for i <- rindex if xj(i) == v yield i).toArray
        new VectorI (a.size, a)
    end trimRows

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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a data vector z, classify it returning the class number (0, ..., k-1)
     *  by following a decision path from the root to a leaf.  If no branch found,
     *  give maximal decision of current node.
     *  @param z  the data vector to classify
     */
    override def predictI (z: VectorI): Int = predictIrec (z)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a data vector z, classify it returning the class number (0, ..., k-1)
     *  by following a decision path from the root to a leaf.  If no branch found,
     *  give maximal decision of current node.
     *  Return the best class and its name.
     *  @param z  the data vector to classify
     */
    override def predictI (z: VectorD): Int = predictIrecD (z)

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

end DecisionTree_ID3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree_ID3` companion object provides a factory method.
 */
object DecisionTree_ID3:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a decision tree for the given combined matrix where the column col
     *  is the response/classification vector.
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
              (col: Int = xy.dim2 - 1): DecisionTree_ID3 =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new DecisionTree_ID3 (x, y, fname, k, cname, hparam)
    end apply

end DecisionTree_ID3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_ID3Test` main function tests the `DecisionTree_ID3` class.
 *  Ex: Classify (No/Yes) whether a person will play tennis based on the measured features.
 *  @see www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm
 *  > runMain scalation.modeling.classifying.decisionTree_ID3Test
 */
@main def decisionTree_ID3Test (): Unit =

    // training-set -----------------------------------------------------------
    // Outlook:     Rain (0), Overcast (1), Sunny (2)
    // Temperature: Cold (0), Mild (1), Hot (2)
    // Humidity:    Normal (0), High (1)
    // Wind:        Weak (0), Strong (1)
    // features:    Outlook Temp Humidity Wind
    // classification vector: 0(no), 1(yes))

    import Example_PlayTennis._

    banner ("Play Tennis Example: DecisionTree_ID3")
    println (s"xy = $xy")                                           // combined data matrix [ x | y ]

    DecisionTree.hp("height") = 2
    val mod = DecisionTree_ID3 (xy, fname)()                        // create an ID3 classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printTree ()                                                // print the decision tree
    println (mod.summary ())                                        // summary statistics

    val z = VectorI (2, 2, 1, 1)                                    // new data vector to classify
    banner (s"Classify $z")
    println (s"classify ($z) = ${mod.classify (z)}")

    banner ("Validation")
    println ("mod test accu = " + mod.validate ()())                // out-of-sample testing

/* Not enough instances for cross-validation
    banner ("Cross-validation")
    FitM.showQofStatTable (mod.crossValidate ())                    // 5-fold cross-validation (14 instances typically too few)
*/

end decisionTree_ID3Test


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_ID3Test2` main function tests the `DecisionTree_ID3` class.
 *  Ex: Classify whether a there is breast cancer.
 *  > runMain scalation.modeling.classifying.decisionTree_ID3Test2
 */
@main def decisionTree_ID3Test2 (): Unit =

    banner ("Breast Cancer: DecisionTree_ID3")
    val nfile = "breast_cancer.csv"
    val xy    = MatrixD.load (nfile)
    val fname = Array ("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                       "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses")
    val cname = Array ("benign", "malignant")
    val k     = cname.size

    DecisionTree.hp("height") = 2 
    val mod = DecisionTree_ID3 (xy, fname, k, cname)()              // create an ID3 classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printTree ()                                                // print the decision tree
    println (mod.summary ())                                        // summary statistics

    banner ("Validation")
    println ("mod test accu = " + mod.validate ()())                // out-of-sample testing

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())                    // 5-fold cross-validation

end decisionTree_ID3Test2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_ID3Test3` main function is used to test the `DecisionTree_ID3` class.
 *  Plot entropy.
 *  > runMain scalation.modeling.classifying.decisionTree_ID3Test3
 */
@main def decisionTree_ID3Test3 (): Unit =

    import scalation.log2

    val p = VectorD.range (1, 100) / 100.0
    val h = p.map (p => -p * log2 (p) - (1-p) * log2 (1-p))
    new Plot (p, h)

end decisionTree_ID3Test3

