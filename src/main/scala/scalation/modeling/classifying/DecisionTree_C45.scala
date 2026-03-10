
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Susan George
 *  @version 2.0
 *  @date    Wed May 22 14:17:49 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: C45 Decision/Classification Tree
 */

package scalation
package modeling
package classifying

import scala.collection.mutable.{ArrayBuffer, Set}

import scalation.mathstat._
import scalation.mathstat.Probability.{entropy, freq}
import scalation.theory.Variable
import scalation.theory.VariableKind.Categorical

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree_C45` class implements a Decision Tree classifier using the
 *  C45 algorithm.  The classifier is trained using a data matrix x and a
 *  classification vector y.  Each data vector in the matrix is classified into
 *  one of k classes numbered 0, ..., k-1.  Each column in the matrix represents
 *  a feature (e.g., Humidity).
 *  @param x       the input/data matrix with instances stored in rows
 *  @param y       the response/classification vector, where y_i = class for row i of matrix x
 *  @param fname_  the names for all features/variables
 *  @param k       the number of classes
 *  @param cname_  the names for all classes
 *  @param conts   the set of feature indices for variables that are treated as continuous
 *  @param hparam  the hyper-parameters
 */
class DecisionTree_C45 (x: MatrixD, y: VectorI, fname_ : Array [String] = null, k: Int = 2,
                        cname_ : Array [String] = Array ("No", "Yes"),
                        conts: Set [Int] = Set [Int] (), hparam: HyperParameter = DecisionTree.hp)
      extends Classifier (x, y, fname_, k, cname_, hparam)
         with FitC (k)
         with DecisionTree:

    private val debug     = debugf ("DecisionTree_C45", false)                // debug function
    private val height    = hparam("height").toInt                            // the maximum height of tree
    private val cutoff    = hparam("cutoff")                                  // cutoff entropy threshold

    private var entropy_0 = entropy (y.freq (k)._2)                           // initial entropy of full vector y
    private val threshold = Array.ofDim [Double] (x.dim2)                     // threshold for continuous features (below <=, above >)
    private val param     = ArrayBuffer [Double] ()                           // parameter vector = feature order
    private val feas      = Array.ofDim [Variable] (x.dim2)                   // array of features/variables xj's
    for j <- x.indices2 do feas(j) = if conts contains j then Variable (x(?, j), j)
                                     else Variable (x(?, j), j, Categorical)

    _modelName = s"DecisionTree_C45_$height"                                  // name of the model

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
    private def gain (fea: Variable, xj: VectorD, y_ : VectorI, rindex: VectorI): (Double, VectorI) =
        val nu  = new VectorI (k)                                             // aggregate frequency vector
        var sum = 0.0
        for v <- fea.values do
            val (frac_v, nu_v) = freq (xj, y_, k, v, rindex,
                                 fea.kind != Categorical, threshold(fea.j))   // frequency for value v
//          debug ("gain", s" (v = $v): (frac_v, nu_v) = ($frac_v, $nu_v")
            sum += frac_v * entropy (nu_v)                                    // weighted entropy
            nu  += nu_v                                                       // aggregate frequency vector
        end for
        val igain = entropy_0 - sum                                           // the drop in entropy = information gain
//      debug ("gain", s"entropy = $sum, overall gain from feature ${fea.j} = $igain")
        (igain, nu)                                                           // return gain and aggregate frequency vector
    end gain

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the best feature f / column xj to expand the decision tree,
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
            val xj = x_(?, j)                                                 // column j of matrix x
            if feas(j).kind != Categorical then
                threshold(j) = DecisionTree_C45.findSplit (xj, y_, rindex, k)   // => calculate split threshold
            val (gn, nu) = gain (feas(j), xj, y_, rindex)                     // compute gain for feature j
//          debug ("findBest", s"compare ($j, $gn, $nu) to $best")
            if gn > best._2 then best = (j, gn, nu)                           // better gainb => update best
        end for
        if best._2 <= 0.0 then println ("findBest: no positive gain found")
        best
    end findBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively build the decision tree until entropy drops to the cutoff
     *  threshold cutoff or the tree depth is at the specified tree height.
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
        val leaf = entropy (nu) <= cutoff || depth >= height                  // leaf or internal? 
        val node = Node (j, gn, nu, parent, nu.argmax (), leaf)               // construct the next node
        if ! leaf && feas(j).kind != Categorical then
            node.thres = threshold (j)                                        // for continuous features, store threshold in node
        if parent == null then
            addRoot (node)                                                    // if no parent, add node as root of tree
            debug ("buildTree", s"entropy of root node: entropy_0 = $entropy_0")

        if ! node.leaf && cindex.dim > 1 then
            val xj      = x(?, j)                                             // extract feature column j
            val cindex2 = cindex diff VectorI (j)                             // remove column j from column index
            debug ("buildTree", s"removing column j = $j gives cindex2 = $cindex2")

            for v <- feas(j).values do                                        // build subtree or leaf for each branch value
                debug ("buildTree", s"explore branch $v for feature x$j at depth $depth")
                val rindex2 = trimRows (j, xj, rindex, v, threshold(j))       // trim row index to those matching value v
                val child = buildTree (x_, y_, rindex2, cindex2, node, depth+1)       // build a subtree
                if child != null then add (node, v, child)                    // if exists, add child to tree
            end for
        end if
        node
    end buildTree

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Trim the row index by only including those where column xj == vl (or
     *  above/below threshold for conts), returning the newly trimmed row index.
     *  @param j       the index for column xj
     *  @param xj      the column of the data matrix to be considered
     *  @param rindex  the working row index used to create the new trimmed version
     *  @param vl      the value to matched (for conts its 0 (up to) or 1 (beyond) threshold)
     *  @param thres   the splitting threshold
     */
    private def trimRows (j: Int, xj: VectorD, rindex: VectorI, vl: Int, thres: Double): VectorI =
        val a = if conts contains j then
            if vl == 0 then (for i <- rindex if xj(i) <= thres yield i).toArray
            else            (for i <- rindex if xj(i)  > thres yield i).toArray
        else
            (for i <- rindex if xj(i) == vl yield i).toArray

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

end DecisionTree_C45


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DecisionTree_C45` companion object provides factory methods.
 */
object DecisionTree_C45:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a decision tree for the given combined matrix where the column col
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
               cname: Array [String] = Array ("No", "Yes"), conts: Set [Int] = Set [Int] (),
               hparam: HyperParameter = DecisionTree.hp)
              (col: Int = xy.dim2 - 1): DecisionTree_C45 =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        new DecisionTree_C45 (x, y, fname, k, cname, conts, hparam)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the best split threshold 'thres' that divides feature/variable 'xj' into
     *  low (<= 'thesh') and high (> 'thres') values such that weighted entropy is minimized.
     *  @param xj    the vector for feature fea (column j of matrix)
     *  @param y_    the classification/response vector
     *  @param idx_  the index positions within x (if null, use all index positions)
     *  @param k     the number of classes
     */
    def findSplit (xj: VectorD, y_ : VectorI, idx_ : VectorI = null, k: Int = 2): Double =
        val idx = if idx_ == null then VectorI.range (0, y_.dim) else idx_
        var thres  = -0.0                                                // keep track of best threshold
        var minEnt = Double.MaxValue                                     // keep track of maximum gain
        val values = xj.distinct.sorted                                  // distinct values from vector xj
                                                                         // sort these values into increasing order
        for i <- 0 until values.dim - 1 do
            val mid            = (values(i) + values(i+1)) / 2.0         // mid point between i and i+1
            val (frac_0, nu_0) = freq (xj, y_, k, 0, idx, true, mid)     // up to threshold (v == 0)
            val (frac_1, nu_1) = freq (xj, y_, k, 1, idx, true, mid)     // beyond threhsold (v == 1)
            val ent = frac_0 * entropy (nu_0) + frac_1 * entropy (nu_1)  // compute entropy for this threshold
            if ent < minEnt then
                thres  = mid                                             // found a better threshold
                minEnt = ent                                             // save better gain
        end for

        thres                                                            // save best threshold for this feature
    end findSplit

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a decision tree for the given data matrix and response/classification
     *  vector.  Takes all integer data (no continuous features).
     *  @param x       the data matrix (features)
     *  @param y       the response/classification vector
     *  @param fname   the names for all features/variables
     *  @param k       the number of classes
     *  @param cname   the names for all classes
     *  @param hparam  the hyper-parameters
     *
    def apply (x: MatrixI, y: VectorI, fname: Array [String], k: Int, cname: Array [String],
               hparam: HyperParameter): DecisionTree_C45 =
        new DecisionTree_C45 (x.toDouble, y, fname, k, cname, hparam)
    end apply
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the decision tree on the given dataset passed in as a combined matrix.
     *  @param xy      the combined data matrix (features and response)
     *  @param fname   the names for all features/variables
     *  @param k       the number of classes
     *  @param cname   the names for all classes
     *  @param conts   the set of feature indices for variables that are treated as continuous
     *  @param hparam  the hyper-parameters
     *
    def test (xy: MatrixD, fname: Array [String], k: Int, cname: Array [String],
              conts: Set [Int] = Set [Int] (), hparam: HyperParameter = hp): DecisionTree_C45 =
        banner ("create, train and print a C45 decision tree")
        println (s"dataset xy: ${xy.dim1}-by-${xy.dim2} matrix")
        val (x, y) = pullResponse (xy)
        val ymin   = y.min ()
        println (s"unadjusted ymin = $ymin")
        if ymin != 0 then y -= ymin
        val height = hparam("height")
        println (s"height limit = $height")

        val tree = new DecisionTree_C45 (x, y.toInt, fn, k, cn, conts, hparam)
        tree.train ()
        val yp = tree.classify (x)
        tree.confusion (yp)
        tree.printTree ()

        banner ("classify all intances and show confusion matrix")
//      for i <- y.indeices do println (s"i: $i, \t y  = ${y(i)}, \t yp = ${yp(i)}")
        val ymax = y.max ()
        println (s"ymax       = $ymax")
        println (tree.report)
        println (tree.summary (tree.parameter))
        tree
    end test
     */

end DecisionTree_C45


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_C45Test` object tests the `DecisionTree_C45` class.
 *  Ex: Classify (No/Yes) whether a person will play tennis based on the measured features.
 *  @see www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm
 *  > runMain scalation.modeling.classifying.decisionTree_C45Test
 */
@main def decisionTree_C45Test (): Unit =

    // training-set -----------------------------------------------------------
    // Outlook:     Rain (0), Overcast (1), Sunny (2)
    // Temperature: Cold (0), Mild (1), Hot (2)
    // Humidity:    Normal (0), High (1)
    // Wind:        Weak (0), Strong (1)
    // features:    Outlook Temp Humidity Wind
    // classification vector: 0(no), 1(yes))

    import Example_PlayTennis._

    banner ("Play Tennis Example: DecisionTree_C45")
    println (s"xy = $xy")                                           // combined data matrix [ x | y

    DecisionTree.hp("height") = 2
    val mod = DecisionTree_C45 (xy, fname)()                        // create a classifier
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

end decisionTree_C45Test


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_C45Test2` main function tests the `DecisionTree_C45` class.
 *  Ex: Classify (No/Yes) whether a person will play tennis based on the measured features.
 *  @see www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm
 *  > runMain scalation.modeling.classifying.decisionTree_C45Test2
 */
@main def decisionTree_C45Test2 (): Unit =

    // training-set -----------------------------------------------------------
    // Outlook:     Rain (0), Overcast (1), Sunny (2)
    // Temperature: continuous
    // Humidity:    continuous
    // Wind:        Weak (0), Strong (1)
    // features:    Outlook Temp Humidity Wind
    // classification vector: 0(no), 1(yes))

    import Example_PlayTennis_Cont._

    banner ("Play Tennis Example: DecisionTree_C45")
    println (s"xy = $xy")                                           // combined data matrix [ x | y

    DecisionTree.hp("height") = 2
    val mod = DecisionTree_C45 (xy, fname, conts = conts)()         // create a classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printTree ()                                                // print the decision tree
    println (mod.summary ())                                        // summary statistics

    val z = VectorI (2, 80, 80, 1)                                  // new data vector to classify
    banner (s"Classify $z")
    println (s"classify ($z) = ${mod.classify (z)}")

    banner ("Validation")
    println ("mod test accu = " + mod.validate ()())                // out-of-sample testing

/* Not enough instances for cross-validation
    banner ("Cross-validation")
    FitM.showQofStatTable (mod.crossValidate ())                    // 5-fold cross-validation (14 instances typically too few)
*/

end decisionTree_C45Test2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_C45Test3` main function tests the `DecisionTree_C45` class.
 *  Ex: Classify whether a there is breast cancer.
 *  > runMain scalation.modeling.classifying.decisionTree_C45Test3
 */
@main def decisionTree_C45Test3 (): Unit =

    banner ("Test: DecisionTree_C45: Breast Cancer Dataset")
    val nfile = "breast_cancer.csv"
    val xy    = MatrixD.load (nfile)
    val fname = Array ("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                       "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses")
    val cname = Array ("benign", "malignant")
    val k     = cname.size
    val conts = Set.range (0, xy.dim2 - 1)

    val mod = DecisionTree_C45 (xy, fname, k, cname, conts)()       // create a classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printTree ()                                                // print the decision tree
    println (mod.summary ())                                        // summary statistics

    banner ("Validation")
    println ("mod test accu = " + mod.validate ()())                // out-of-sample testing

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())                    // 5-fold cross-validation (14 instances typically too few)

end decisionTree_C45Test3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_C45Test4` main function tests the `DecisionTree_C45` class.
 *  Ex: Classify the quality of white wine.
 *  > runMain scalation.modeling.classifying.decisionTree_C45Test4
 */
@main def decisionTree_C45Test4 (): Unit =

    val nfile  = "winequality-white.csv"
    val xy     = MatrixD.load (nfile)
    val ycol   = xy.dim2 - 1
    for i <- xy.indices do xy(i, ycol) -= 3                         // shift the class labels by 3

    val k      = 7                                                  // 7 classes
    val fname  = Array ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                    "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")    // feature names
    val cname  = Array ("Level3", "Level4", "Level5", "Level6", "Level7", "Level8", "Level9")                  // class names
    val height = 5
//  val conts  = range2muSet (0 until xy.dim2 - 1)
    val conts  = Set.range (0, xy.dim2 - 1)

    DecisionTree.hp("height") = height
    val mod = DecisionTree_C45 (xy, fname, k, cname, conts)()       // create a classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printTree ()                                                // print the decision tree
    println (mod.summary ())                                        // summary statistics

    banner ("Validation")
    println ("mod test accu = " + mod.validate ()())                // out-of-sample testing

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())                    // 5-fold cross-validation (14 instances typically too few)

end decisionTree_C45Test4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `decisionTree_C45Test5` main function tests the `DecisionTree_C45` class.
 *  Ex: Classify whether the patient has diabetes or not
 *  > runMain scalation.modeling.classifying.decisionTree_C45Test5
 */
@main def decisionTree_C45Test5 (): Unit =

    banner ("Test: DecisionTree_C45: Diabetes Dataset")
    val nfile  = "diabetes.csv"
    val xy     = MatrixD.load (nfile)
    val k      = 2                                                  // 2 classes

    val fname  = Array ("pregnancies", "glucose", "blood pressure", "skin thickness", "insulin",
                        "BMI", "diabetes pedigree function", "age")
    val cname  = Array ("tested_positive", "tested_negative")       // class names
    val height = 5
    val conts  = Set.range (0, xy.dim2 - 1)

    DecisionTree.hp("height") = height
    val mod = DecisionTree_C45 (xy, fname, k, cname, conts)()       // create a classifier
    mod.trainNtest ()()                                             // train and test the classifier
    mod.printTree ()                                                // print the decision tree
    println (mod.summary ())                                        // summary statistics

    banner ("Validation")
    println ("mod test accu = " + mod.validate ()())                // out-of-sample testing

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())                    // 5-fold cross-validation (14 instances typically too few)

end decisionTree_C45Test5

