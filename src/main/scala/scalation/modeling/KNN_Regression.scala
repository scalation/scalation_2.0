
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Dec 21 14:38:32 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: K Nearest Neighbors for Regression/Prediction
 */

package scalation
package modeling

import scala.collection.mutable.IndexedSeq
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
//import scalation.random.Bernoulli

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KNN_Regression` class is used to predict a response value for new vector z.
 *  It works by finding its kappa nearest neighbors.  These neighbors essentially
 *  vote according to their prediction.  The consensus is the average individual
 *  predictions for z.  Using a distance metric, the kappa vectors nearest
 *  to z are found in the training data, which are stored row-wise in data
 *  matrix x.  The corresponding response values are given in the vector y,
 *  such that the response value for vector x(i) is given by y(i).
 *  WARNING: in-sample testing is only meaningful for large values of hyper-parsameter kappa
 *  @param x       the vectors/points of predictor data stored as rows of a matrix
 *  @param y       the response value for each vector in x
 *  @param fname_  the names for all features/variables (defaults to null)
 *  @param hparam  the number of nearest neighbors to consider (defaults to KNN_Regression.hp)
 */
class KNN_Regression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                      hparam: HyperParameter = KNN_Regression.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2):

    private val debug      = debugf ("KNN_Regression", false)                // debug function
    private val flaw       = flawf ("KNN_Regression")                        // debug function
    private val MAX_DOUBLE = Double.PositiveInfinity                         // infinity
    private val kappa      = hparam("kappa").toInt                           // the number of nearest neighbors to consider

    private val topK       = Array.fill (kappa)(-1, MAX_DOUBLE)              // top-kappa nearest points (in reserve order)
    private val d          = new VectorD (x.dim)                             // vector to hold distances

    _modelName = s"KNN_Regression_$kappa"

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a distance metric between vectors/points x and z.
     *  The squared Euclidean norm used for efficiency, but may use other norms.
     *  @param x  the first vector/point
     *  @param z  the second vector/point
     */
    def distance (x: VectorD, z: VectorD): Double = (x - z).normSq

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the kappa nearest neighbors (top-kappa) to vector z and store in
     *  the topK array.
     *  @param z     the vector whose response value is to be predicted
     *  @param i_no  the forbidden indices (e.g., in the testing set)
     */
    private def kNearest (z: VectorD, i_no: IndexedSeq [Int] = null): Unit =
        for i <- x.indices do
            d(i) = if i_no != null && (i_no contains i) then MAX_DOUBLE
                   else distance (z, x(i))                                   // distance to all points      
        end for
        val top = d.iselsort                                                 // use partial indirect selsort - FIX not partial
        for j <- 0 until kappa do topK(j) = (top(j), d(top(j)))              // assign index and distance
        debug ("kNearest", s"z = $z: topK = ${stringOf (topK)}")
    end kNearest

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the model.  It uses lazy training, so most of the work is done during
     *  prediction.
     *  @param x_  the training/full data/input matrix
     *  @param y_  the training/full response/output vector
     */
    def train (x_ : MatrixD = x, y_ : VectorD = y): Unit = {}

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val m   = y_.dim
        val df1 = m / kappa                                                  // degrees of freedom model, see ESL
        val df2 = m - df1                                                    // degrees of freedom error
        resetDF ((df1, df2))
        val yp = predict (x_)                                                // make predictions
        (yp, diagnose (y_, yp))                                              // return predictions and QoF vector
    end test

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param xe    the testing data/input matrix
     *  @param ye    the testing response/output vector
     *  @param i_no  the forbidden indices (e.g., in testing set)
     */
    def testNoSpy (xe: MatrixD = x, ye: VectorD = y, i_no: IndexedSeq [Int]): (VectorD, VectorD) =
        val m   = ye.dim
        val df1 = m / kappa                                                  // degrees of freedom model, see ESL
        val df2 = m - df1                                                    // degrees of freedom error
        resetDF ((df1, df2))
        val yp = xe.map (predictNoSpy (_, i_no))                             // make predictions
        println (s"ye.dim = ${ye.dim}, yp.dim = ${yp.dim}")
        (yp, diagnose (ye, yp))                                              // return predictions and QoF vector
    end testNoSpy

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new point/vector z, predict its response value based on the
     *  actual response values of its kappa nearest neighbors.
     *  @param z  the vector to predict
     */
    override def predict (z: VectorD): Double =
        kNearest (z)                                                         // set top-kappa to kappa nearest
        var sum = 0.0
        for i <- 0 until kappa do sum += y(topK(i)._1)                       // sum the individual predictions
        sum / kappa                                                          // divide to get average
    end predict

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new point/vector z, predict its response value based on the
     *  actual response values of its kappa nearest neighbors from the training
     *  set (no spying in testing set).
     *  @param z     the vector to predict
     *  @param i_no  the forbidden indices (e.g., in the testing set)
     */
    def predictNoSpy (z: VectorD, i_no: IndexedSeq [Int]): Double =
        kNearest (z, i_no)                                                   // set top-kappa to kappa nearest
        var sum = 0.0
        for i <- 0 until kappa do sum += y(topK(i)._1)                       // sum the individual predictions
        sum / kappa                                                          // divide to get average
    end predictNoSpy

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use validation to compute test Quality of Fit (QoF) measures by dividing
     *  the full dataset into a TESTING-set and a TRAINING-set.
     *  The testing-set is defined by idx and the rest of the data is the training-set.
     *  @see `modeling.Predictor.validate` about the RANDOM, FIRST, and LAST options
     *  for selecting the testing-set.
     *  @param rando  flag indicating whether to use randomized or simple validation
     *  @param ratio  the ratio of the TESTING-set to the full dataset (most common 70-30 (.3), 80-20 (.2))
     *  @param idx    the prescribed TESTING-set indices
     */
    override def validate (rando: Boolean = true, ratio: Double = Model.TE_RATIO)
//                        (idx : IndexedSeq [Int] = testIndices ((ratio * y.dim).toInt, rando)):
                          (idx: IndexedSeq [Int] = testIndices (y.dim, (ratio * y.dim).toInt, rando)):
                          (VectorD, VectorD) =
        val x_e = x(idx)                                                     // test data/input matrices
        val y_e = y(idx)                                                     // test response/output vectors

        val (yp, qof) = testNoSpy (x_e, y_e, idx)                            // test on TESTING-set and get its yp and QoF measures
        if qof(QoF.sst.ordinal) <= 0.0 then                                  // requires variation in test-set
            flaw ("validate", "chosen testing set has no variability")
        println (FitM.fitMap (qof, QoF.values.map (_.toString)))
        (yp, qof)
    end validate

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): KNN_Regression =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new KNN_Regression (x, y, fname2, hparam)
    end buildModel

end KNN_Regression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KNN_Regression` companion object provides factory methods for creating
 *  k-nearest neighbor regression models.
 */
object KNN_Regression:

    val hp = new HyperParameter; hp += ("kappa", 3, 3)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `KNN_Regression` object from a combined xy data-response matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the names for all features/variables (defaults to null)
     *  @param hparam  the number of nearest neighbors to consider (defaults to hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)(col: Int = xy.dim2 - 1): KNN_Regression =
        new KNN_Regression (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `KNN_Regression` object from a data matrix and a response vector.
     *  This method provides data rescaling.
     *  @param x       the data/input m-by-n matrix
     *                     (augment with a first column of ones to include intercept in model)
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (default to hp)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = hp): KNN_Regression =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new KNN_Regression (xn, y, fname, hparam)
    end rescale

end KNN_Regression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kNN_RegressionTest` main function is used to test the `KNN_Regression` class.
 *  > runMain scalation.modeling.kNN_RegressionTest
 */
@main def kNN_RegressionTest (): Unit =

    //                        x0 x1  y
    val xy = MatrixD ((10, 3), 1, 5, 1,                                   // joint data matrix
                               2, 4, 1,
                               3, 4, 1,
                               4, 4, 1,
                               5, 3, 0,
                               6, 3, 1,
                               7, 2, 0,
                               8, 2, 0,
                               9, 1, 0,
                              10, 1, 0)

    println ("xy = " + xy)

    val mod = KNN_Regression (xy)()                                       // create a KNN Regression model
    mod.trainNtest ()()                                                   // train and test the model

    val z1 = VectorD (10.0, 10.0)
    println ("z1 = " + z1)
    println ("yp = " + mod.predict (z1))

    val z2 = VectorD ( 3.0,  3.0)
    println ("z2 = " + z2)
    println ("yp = " + mod.predict (z2))

    banner ("Compare y vs. yp")
    val y  = xy(?, 2)
    val yp = mod.predict (xy.not (?, 2))
    println (s"y  = $y")
    println (s"yp = $yp")

    new Plot (xy(?, 0), y, yp, lines = true)

end kNN_RegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kNN_RegressionTest2` main function is used to test the `KNN_Regression` class.
 *  > runMain scalation.modeling.kNN_RegressionTest2
 */
@main def kNN_RegressionTest2 (): Unit =

    //                       x1 x2  y
    val xy = MatrixD ((9, 3), 0, 0, 0,
                              0, 1, 0,
                              0, 2, 1,
                              1, 0, 0,
                              1, 1, 0,
                              1, 2, 1,
                              2, 0, 1,
                              2, 1, 1,
                              2, 2, 1)

    val fn = Array ("x1", "x2")                                           // feature/variable names
    println ("xy = " + xy)

    val mod = KNN_Regression (xy, fn)()                                   // create a model
    mod.trainNtest ()()                                                   // train and test the model

    banner ("Compare y vs. yp")
    val y  = xy(?, 2)
    val yp = mod.predict (xy.not (?, 2))
    println (s"y  = $y")
    println (s"yp = $yp")

    new Plot (null, y, yp, lines = true)

end kNN_RegressionTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kNN_RegressionTest3` main function is used to test the `KNN_predictor` class.
 *  > runMain scalation.modeling.kNN_RegressionTest3
 */
@main def kNN_RegressionTest3 (): Unit =

    import Example_AutoMPG._

    println ("xy = " + xy)

    val cap = 50
    val kr  = VectorD.range (0, cap)
    val rSq = new MatrixD (cap, Fit.qofVectorSize)                        // R^2, R^2 Bar, sMAPE, R^2 cv

    for k <- 2 until cap do
        KNN_Regression.hp("kappa") = k
        val mod = KNN_Regression (xy)()                                   // create a model
        mod.trainNtest ()()                                               // train and test the model
        rSq(k) = Fit.qofVector (mod.fit, mod.crossValidate ())            // use main model, knn
    end for

    new PlotM (kr, rSq.ᵀ, Regression.metrics, lines = true)

end kNN_RegressionTest3

