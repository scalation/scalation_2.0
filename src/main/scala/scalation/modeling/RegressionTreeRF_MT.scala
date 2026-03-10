
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dong Yu Yu, John Miller
 *  @version 2.0
 *  @date    Fri Jan  5 16:54:27 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Random Forest (RF) of Model Trees
 */

package scalation
package modeling

//import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

import modeling.{RegressionTreeMT => REG_TREE}                               // swap for Model Tree
//import modeling.{RegressionTree  => REG_TREE}                              // swap for Regression Tree

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeRF_MT` class uses several randomly built reegression trees for prediction.
 *  It randomly selects sub-samples of 'bRatio * x.dim' size from the data x and y to
 *  build nTrees regression trees.  The predict method uses the average over all trees.
 *  Note:  By default this class does not select sub-features to build the trees (like Bagging Trees)
 *         Set use_fb (feature bagging) to true to turn this capability on
 *  @param x       the input/data matrix (instances by features)
 *  @param y       the ouput/response vector (instances)
 *  @param fname_  the names of the variables/features (defaults to null => auto-generate))
 *  @param use_fb  whether to use feature bagging (select subsets of the features)
 *  @param hparam  the hyper-parameters to the random forest (defaults to RegressionTree.hp)
 */
class RegressionTreeRF_MT (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                           use_fb: Boolean = false, hparam: HyperParameter = RegressionTree.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):                    // call resetDF once tree is built

    private val debug      = debugf ("RegressionTreeRF_MT", true)             // debug function
    private val flaw       = flawf ("RegressionTreeRF_MT")                    // flaw function
    private val depth      = hparam("maxDepth").toInt                         // the max depth for the base regression trees
    private val nTrees     = hparam("nTrees").toInt                           // number of trees
    private val bRatio     = hparam("bRatio").toDouble                        // bagging ratio 
    private val fbRatio    = hparam("fbRatio").toDouble                       // feature bagging ratio 
    private val sampleSize = (bRatio * x.dim).toInt                           // size of matrix sub-samples
    private val forest     = Array.ofDim [REG_TREE] (nTrees)                  // forest of regression trees

    if nTrees <= 0 then                  flaw ("init", "RF number of tree must be at least one")
    if bRatio <= 0  || bRatio >= 1  then flaw ("init", "RF bagging ratio restricted to (0, 1)")
    if fbRatio <= 0 || fbRatio >= 1 then flaw ("init", "RF feature bagging ratio restricted to (0, 1)")

    _modelName = s"RegressionTreeRF_MT_${depth}_${nTrees}_$use_fb"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the regression tree RF by selecting thresholds for the features/variables
     *  in matrix x_.  Build the trees of the forest by selecting a subSample for each tree.
     *  @author Prudhvi Chekka, Lalithya Sajja
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit =
        for k <- 0 until nTrees do
            val (xx, yy, imap) = subSample (x_, y_, sampleSize, k)            // select rows of data matrix
//          debug ("train", s"for tree$k, imap = ${stringOf (imap)}")
            debug ("train", s"for tree$k, imap.size = $imap.size}")

            forest(k) = new REG_TREE (xx, yy, fname, hparam, use_r_fb = use_fb)  // means/regression in leaves
            forest(k).train (xx, yy)
            debug ("train", s"for tree$k === \n ${forest(k).printTree ()}")
        end for
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp  = predict (x_)                                               // make predictions
        val df1 = forest.foldLeft (0)(_ + _.numLeaves)                       // degrees of freedom model = number of leaves
        val df2 = y_.dim - df1                                               // degrees of freedom error
        resetDF ((df1, df2))
        (yp, diagnose (y_, yp))                                              // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the response value given vector z by averaging the predictions over
     *  all the randomized trees.
     *  @param z  the vector to be predicted
     */
    override def predict (z: VectorD): Double =
        var sum = 0.0
        for k <- 0 until nTrees do sum += forest(k).predict (z)              // sum up for each tree
        sum / nTrees                                                         // take the average
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): RegressionTreeRF_MT =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new RegressionTreeRF_MT (x_cols, y, fname2, use_fb, hparam)
    end buildModel

end RegressionTreeRF_MT


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeRF_MTTest` main function is used to test the `RegressionTreeRF_MT` class.
  *  @see translate.google.com/translate?hl=en&sl=zh-CN&u=https:
  *       //www.hrwhisper.me/machine-learning-decision-tree/&prev=search
  *  > runMain scalation.modeling.regressionTreeRF_MTTest
  */
@main def regressionTreeRF_MTTest (): Unit =

    val x  = MatrixD ((10, 1), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val y  = VectorD (5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.90, 8.70, 9.00, 9.05)
    val ox = VectorD.one (x.dim) +^: x
    val fname = Array ("x")

    banner (s"Regression no intercept")
    val reg = new Regression (x, y)
    reg.trainNtest ()()                                               // train and test the model

    banner (s"Regression with intercept")
    val reg2 = new Regression (ox, y)
    reg2.trainNtest ()()                                              // train and test the model

    banner (s"Quadratic Regression")
    val reg3 = SymbolicRegression.quadratic (x, y, fname)
    reg3.trainNtest ()()                                              // train and test the model

    banner (s"Perceptron sigmoid")
    val nn = Perceptron.rescale (reg3.getX, y)
    nn.trainNtest ()()                                                // train and test the model

    banner (s"Perceptron tanh")
    val nn2 = Perceptron.rescale (reg3.getX, y, f = ActivationFun.f_tanh)
    nn2.trainNtest ()()                                               // train and test the model

    for d <- 1 to 2 do
        banner (s"Regression Tree RF with maxDepth = $d")
        RegressionTree.hp.updateReturn ("maxDepth", d)
        val mod = new RegressionTreeRF_MT (x, y, fname)
        mod.trainNtest ()()                                           // train and test the model
//      mod.printTree ()
    end for

end regressionTreeRF_MTTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeRF_MTTest2` main function tests the `RegressionTreeRF_MT` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests multiple depths.
 *  > runMain scalation.modeling.regressionTreeRF_MTTest2
 */
@main def regressionTreeRF_MTTest2 (): Unit =

    import Example_AutoMPG._

//  println (s"x = $o")
//  println (s"y = $y")

    val dmax = 6                                                        // range of depths 1 to dmax
    val qual = new MatrixD (dmax, 3)

    for d <- 1 to dmax do
        banner ("AutoMPG Regression Tree RF with depth d = $d")
        RegressionTree.hp("maxDepth") = d
        RegressionTree.hp("nTrees")   = 7
        val mod = new RegressionTreeRF_MT (x, y, x_fname)               // create model with intercept (else pass x)
        val qof = mod.trainNtest ()()._2                                // train and test the model
//      mod.printTree ()                                                // print the regression tree
//      println (mod.summary ())                                        // parameter/coefficient statistics

        banner (s"AutoMPG Regression Tree RF with d = $d Validation")
        val qof2 = mod.validate ()()._2                                 // out-of-sampling testing
        val iq = QoF.rSq.ordinal                                        // index for rSq
        qual (d-1) = VectorD (qof(iq), qof(iq+1), qof2(iq))             // R^2, R^2 bar, R^2 os
    end for

    new PlotM (VectorD.range (1, dmax+1), qual.ᵀ, Array ("R^2", "R^2 bar", "R^2 os"),
               "RegressionTreeRF_MT in-sample, out-of-sample QoF vs. depth", lines = true)
    println (s"RegressionTreeRF_MT: qual = $qual")

end regressionTreeRF_MTTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeRF_MTTest3` main function tests the `RegressionTreeRF_MT` class using
 *  the AutoMPG dataset.  Assumes no missing values.  It tests forward, backward and
 *  stepwise selection.
 *  > runMain scalation.modeling.regressionTreeRF_MTTest3
 */
@main def regressionTreeRF_MTTest3 (): Unit =

    import Example_AutoMPG._

    val d = 5                                                           // depth of tree

//  println (s"x = $x")
//  println (s"y = $y")

    banner (s"AutoMPG Regression Tree RF with d = $d")
    RegressionTree.hp("maxDepth") = d
    val mod = new RegressionTreeRF_MT (x, y, x_fname)                      // create model with intercept (else pass x)
    mod.trainNtest ()()                                                 // train and test the model
//  mod.printTree ()                                                    // print the regression tree

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                     // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Regression Tree RF with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end regressionTreeRF_MTTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeRF_MTTest4` main function tests the `RegressionTreeRF_MT` class using
 *  the Boston House Prices dataset.
 *  > runMain scalation.modeling.regressionTreeRF_MTTest4
 */
@main def regressionTreeRF_MTTest4 (): Unit =

//  println (s"x = $o")
//  println (s"y = $y")

    val data = MatrixD.load ("boston_house_prices.csv", 1, 0)
    val x    = data(?, 0 until data.dim2 - 1)
    val y    = data(?, data.dim2 - 1)
    val dmax = 5

    for d <- 1 to dmax do
        banner (s"AutoMPG Regression Tree RF with depth d = $d")
        RegressionTree.hp("maxDepth") = d
        RegressionTree.hp("nTrees") = 9
        val mod = new RegressionTreeRF_MT (x, y)                           // create model with intercept (else pass x)
        mod.trainNtest ()()._2                                          // train and test the model
    end for

end regressionTreeRF_MTTest4

