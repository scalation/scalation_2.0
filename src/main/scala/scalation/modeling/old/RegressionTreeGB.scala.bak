
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dong Yu Yu, John Miller
  * @version 2.0
  * @date    Sun Dec 16 16:09:16 EST 2018
  * @see     LICENSE (MIT style license file).
  *
  * @note    Model: Regression Tree with Gradient Boosting
  */

package scalation
package modeling

import scala.collection.mutable.ArrayBuffer

import scalation.mathstat._

import modeling.{RegressionTree  => REG_TREE}                                 // swap Regression Tree
//import modeling.{RegressionTreeMT => REG_TREE}                              // swap Model Tree

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeGB` class uses Gradient Boosting on `RegressionTree`s.
 *  @param x       the input/data matrix
 *  @param y       the output/response vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters for the model (defaults to RegressionTree.hp)
 */
class RegressionTreeGB (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                        hparam: HyperParameter = RegressionTree.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfm = x.dim2 - 1, df = x.dim - x.dim2):                    // call resetDF once tree is built

    private val depth   = hparam("maxDepth").toInt                            // the max depth for the base regression trees
    private val iter    = hparam("iterations").toInt                          // the iterations for training
    private val forest  = new ArrayBuffer [REG_TREE] ()                       // forest is a list of regression trees

    modelName = s"RegressionTreeGB ($depth, $iter)"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Using Gradient Boosting on Training, for every iteration, we evaluate the residual
     *  and form a Regression Tree where the residual is the depedent value (equal to the
     *  gradient if using SSE as loss function).
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit =
        val yp = VectorD.fill (y_.dim)(y_.mean)                               // initial value for y-predicted

        for i <- 0 until iter do
            val yres = y_ - yp                                                // y-residual
            val tree = new REG_TREE (x_, yres, fname, hparam)                 // i-th tree in forest (mean/regression)
            forest  += tree                                                   // add to forest
            tree.train (x_, yres)                                             // train the i-th tree
            yp += tree.predict (x_)                                           // add to cumulative prediction
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
        val yp  = predict (x_)                                                // make predictions
        val df1 = forest.foldLeft(0)(_ + _.numLeaves)                         // degrees of freedom model = number of leaves
        val df2 = y_.dim - df1                                                // degrees of freedom error
        resetDF ((df1, df2))
        (yp, diagnose (y_, yp))                                               // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a data vector z, predict the value by summing the predict for each tree.
     *  @param z  the data vector to predict
     */
    override def predict (z: VectorD): Double =
        var yp = y.mean
        for i <- forest.indices do yp += forest(i).predict (z)
        yp
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a data matrix z, predict the value by summing the predict for each tree,
     *  for each row of the matrix.
     *  @param z  the data matrix to predict
     */
    override def predict (z: MatrixD = x): VectorD =
        val yp = new VectorD (z.dim)
        for i <- z.indices do yp(i) = predict (z(i))
        yp
    end predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     */
    override def buildModel (x_cols: MatrixD): RegressionTreeGB =
        new RegressionTreeGB (x_cols, y, null, hparam)
    end buildModel

end RegressionTreeGB


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionTreeGB` companion object defines hyper-parameters and provides
 *  a factory methods for creating gradient boosted regression trees.
 */
object RegressionTreeGB:

    private val flaw = flawf ("RegressionTreeGB")                     // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeGB` object that uses Gradient Boosting on `RegressionTree`.
     *  One Tree is included in the model at a time wisely chosen for reducing gradient.
     *  @param xy      the combined data-response matrix
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters for the model (defaults to RegressionTree.hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = RegressionTree.hp)
              (col: Int = xy.dim2 - 1): RegressionTreeGB =
        val n = xy.dim2
        if n < 2 then
            flaw ("apply", s"dim2 = $n of the 'xy' matrix must be at least 2")
            null
        else
            val (x, y) = (xy.not (?, col), xy(?, col))
            new RegressionTreeGB (x, y, fname, hparam) 
        end if
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionTreeGB` object that uses Gradient Boosting on `RegressionTree`.
     *  One Tree is included in the model at a time wisely chosen for reducing gradient.
     *  @param x       the input/data matrix
     *  @param y       the output/response vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters for the model (defaults to RegressionTree.hp)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = RegressionTree.hp): RegressionTreeGB =
        val n = x.dim2
        if n < 1 then
            flaw ("rescale", s"dim2 = $n of the 'x' matrix must be at least 1")
            null
        else
// FIX - add rescale
            new RegressionTreeGB (x, y, fname, hparam) 
        end if
    end rescale

end RegressionTreeGB


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGBTest` main function is used to test the `RegressionTreeGB` class.
 *  @see translate.google.com/translate?hl=en&sl=zh-CN&u=https:
 *       //www.hrwhisper.me/machine-learning-decision-tree/&prev=search
 *  > runMain scalation.modeling.regressionTreeGBTest
 */
@main def regressionTreeGBTest (): Unit =

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
        banner (s"Regression Tree GB with maxDepth = $d")
        RegressionTree.hp.updateReturn ("maxDepth", d)
        val mod = new RegressionTreeGB (x, y, fname)
        mod.trainNtest ()()                                           // train and test the model
//      mod.printTree ()
    end for

end regressionTreeGBTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGBTest2` main function tests the `RegressionTreeGB` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests multiple depths.
 *  > runMain scalation.modeling.regressionTreeGBTest2
 */
@main def regressionTreeGBTest2 (): Unit =

    import Example_AutoMPG._

//  println (s"x = $o")
//  println (s"y = $y")

    val dmax = 6                                                        // range of depths 1 to dmax
    val qual = new MatrixD (dmax, 3)

    for d <- 1 to dmax do
        banner ("AutoMPG Regression Tree GB with depth d = $d")
        RegressionTree.hp("maxDepth") = d                               // depth of tree
        RegressionTree.hp("nTrees")   = 3                               // number of iterations
        val mod = new RegressionTreeGB (x, y, x_fname)                  // create model with intercept (else pass x)
        val qof = mod.trainNtest ()()._2                                // train and test the model
//      mod.printTree ()                                                // print the regression tree
//      println (mod.summary ())                                        // parameter/coefficient statistics

        banner (s"AutoMPG Regression Tree GB with d = $d Validation")
        val qof2 = mod.validate ()()                                    // out-of-sampling testing
        val iq = QoF.rSq.ordinal                                        // index for rSq
        qual (d-1) = VectorD (qof(iq), qof(iq+1), qof2(iq))             // R^2, R^2 bar, R^2 os
    end for

    new PlotM (VectorD.range (1, dmax+1), qual.transpose, Array ("R^2", "R^2 bar", "R^2 os"),
               "RegressionTreeGB in-sample, out-of-sample QoF vs. depth", lines = true)
    println (s"RegressionTreeGB: qual = $qual")

end regressionTreeGBTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGBTest3` main function tests the `RegressionTreeGB` class using the
 *  AutoMPG dataset.  Assumes no missing values.  It tests forward, backward and stepwise
 *  selection.
 *  > runMain scalation.modeling.regressionTreeGBTest3
 */
@main def regressionTreeGBTest3 (): Unit =

    import Example_AutoMPG._

    val d = 5                                                           // depth of tree

//  println (s"x = $x")
//  println (s"y = $y")

    banner (s"AutoMPG Regression Tree GB with d = $d")
    RegressionTree.hp("maxDepth") = d
    val mod = new RegressionTreeGB (x, y, x_fname)                      // create model with intercept (else pass x)
    mod.trainNtest ()()                                                 // train and test the model
//  mod.printTree ()                                                    // print the regression tree

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                     // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.transpose, Array ("R^2", "R^2 bar", "R^2 cv"),
                   s"R^2 vs n for Regression Tree GB with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end regressionTreeGBTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionTreeGBTest4` main function is used to test the `RegressionTreeGB` class.
 *  > runMain scalation.modeling.regressionTreeGBTest4
 */
@main def regressionTreeGBTest4 (): Unit =

    val x = MatrixD ((5, 1), 750, 800, 850, 900, 950)
    val y = VectorD (1160, 1200, 1280, 1450, 2000)

    val mod = new RegressionTreeGB (x, y)
    mod.trainNtest ()()                                              // train and test the model
//  mod.printTree ()                                                 // print the regression tree
//  println (mod.summary ())                                         // parameter/coefficient statistics

end regressionTreeGBTest4

