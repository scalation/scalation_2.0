
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 22 18:45:44 EDT 2013 
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: K-Nearest Neighbors (KNN) Classifier
 */

package scalation
package modeling
package classifying

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KNN_Classifier` class is used to classify a new vector z into one of
 *  k classes.  It works by finding its kappa nearest neighbors.  These neighbors
 *  essentially vote according to their classification.  The class with most
 *  votes is selected as the classification of z.  Using a distance metric,
 *  the kappa vectors nearest to z are found in the training data, which is
 *  stored row-wise in the data matrix x.  The corresponding classifications
 *  are given in the vector y, such that the classification for vector x(i)
 *  is given by y(i).
 *  FIX - cross validation uses test data for decision making, so when kappa = 1, acc = 100%
 *  @param x       the input/data matrix
 *  @param y       the classification of each vector in x
 *  @param fname_  the names for all features/variables
 *  @param k       the number of classes
 *  @param cname_  the names for all classes
 *  @param kappa   the number of nearest neighbors to consider (k >= 3)
 *  @param hparam  the hyper-parameters
 */
class KNN_Classifier (x: MatrixD, y: VectorI, fname_ : Array [String] = null,
                      k: Int = 2, cname_ : Array [String] = Array ("No", "Yes"),
                      kappa: Int = 5, hparam: HyperParameter = null)
      extends Classifier (x, y, fname_, k, cname_, hparam)
         with FitC ():

    private val debug      = debugf ("KNN_Classifier", true)              // debug function
    private val flaw       = flawf ("KNN_Classifier")                     // flaw function
    private val MAX_DOUBLE = Double.PositiveInfinity                      // infinity
    private val topK       = Array.fill (kappa)(-1, MAX_DOUBLE)           // top-kappa nearest points (in reserve order)
    private val count      = new VectorI (k)                              // how many nearest neighbors in each class.
    private var d          = VectorD.nullv                                // vector to hold distances

    _modelName = s"KNN_Classifier_$kappa"                                 // name of the model

    if kappa < 3 then flaw ("init", s"number of neighbors kappa = $kappa < 3")

//  debug ("init", s" x = $x \n y = $y")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute a distance metric between vectors/points x and z.
     *  The squared Euclidean norm used for efficiency, but may use other norms.
     *  @param x  the first vector/point
     *  @param z  the second vector/point
     */
    def distance (x: VectorD, z: VectorD): Double = (x - z).normSq

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the kappa nearest neighbors (top-kappa) to vector z and store in
     *  the topK array.  Break ties by flipping a fair coin.
     *  @param z  the vector to be classified
     */
    private def nearest (z: VectorD): Unit =
        d = x.map (distance (z, _))                                       // distance from z to all points
        val tp: Array [Int] = d.iselsort (kappa)                          // use partial indirect selsort
        for j <- 0 until kappa do topK(j) = (tp(j), d(tp(j)))             // assign index and distance
    end nearest

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

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new point/vector z, determine which class it belongs to (i.e.,
     *  the class getting the most votes from its kappa nearest neighbors.
     *  Return the best class, its name and its votes
     *  @param z  the vector to classify
     */
    override def predictI (z: VectorD): Int =
        nearest (z)                                                      // set top-kappa to kappa nearest
        for i <- 0 until kappa do count(y(topK(i)._1)) += 1              // tally votes per class
        val best = count.argmax ()                                       // class with maximal count
        reset ()                                                         // reset topK and counters
        best                                                             // return the best
    end predictI

    override def predictI (z: VectorI): Int = predictI (z.toDouble)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset or re-initialize topK and counters.
     */
    def reset (): Unit =
        for i <- 0 until kappa do topK(i)  = (-1, MAX_DOUBLE)            // initialize top-kappa
        for j <- 0 until k do count(j) = 0                               // initialize counters
    end reset

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

end KNN_Classifier


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `KNN_Classifier` companion object provides a factory method.
 */
object KNN_Classifier:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `KNN_Classifier` classifier from a combined xy data matrix.
     *  @param xy     the combined data matrix [ x | y ]
     *  @param fname  the names for all features/variables
     *  @param k      the number of classes
     *  @param cname  the names for all classes
     *  @param kappa  the number of nearest neighbors to consider
     *  @param col    the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               k: Int = 2, cname: Array [String] = Array ("No", "Yes"), kappa: Int = 5)
              (col: Int = xy.dim2 - 1): KNN_Classifier =
        new KNN_Classifier (xy.not(?, col), xy(?, col).toInt, fname, k, cname, kappa)
    end apply

end KNN_Classifier


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kNN_ClassifierTest` main function is used to test the `KNN_Classifier` class.
 *  > runMain scalation.modeling.classifying.kNN_ClassifierTest
 */
@main def kNN_ClassifierTest (): Unit =

    //                        x1 x2  y
    val xy = MatrixD ((10, 3), 1, 5, 1,                      // joint data matrix
                               2, 4, 1,
                               3, 4, 1,
                               4, 4, 1,
                               5, 3, 0,
                               6, 3, 1,
                               7, 2, 0,
                               8, 2, 0,
                               9, 1, 0,
                              10, 1, 0)

//  val x = xy.not (?, xy.dim2-1)
//  val y = xy(?, xy.dim2-1).toInt

    val fname = Array ("x1", "x2")                           // feature/variable names
    val cname = Array ("No", "Yes")                          // class names

    println (s"xy = $xy")

    val knn = KNN_Classifier (xy, fname, 2, cname)()
    knn.trainNtest ()()                                      // no training; test
    println (knn.summary ())                                 // summary statistics

    val z1 = VectorD (10.0, 10.0)
    println ("knn.predictI ($z1) = " + knn.predictI (z1))

    val z2 = VectorD (3.0, 3.0)
    println ("knn.predictI ($z2) = " + knn.predictI (z2))

//  val yp = knn.predict (x)
//  new Plot (xy(?, 0), y.toDouble, yp.toDouble)

end kNN_ClassifierTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kNN_ClassifierTest2` main function is used to test the `KNN_Classifier` class.
 *  > runMain scalation.modeling.classifying.kNN_ClassifierTest2
 */
@main def kNN_ClassifierTest2 (): Unit =

    //                        x1 x2  y
    val xy = MatrixD ((9, 3), 0, 0, 0,
                              0, 1, 0,
                              0, 2, 1,
                              1, 0, 0,
                              1, 1, 0,
                              1, 2, 1,
                              2, 0, 1,
                              2, 1, 1,
                              2, 2, 1)
    val x = xy.not(?, 2)
    val y = xy(?, 2)

    val fname = Array ("x1", "x2")                           // feature/variable names
    val cname = Array ("No", "Yes")                          // class names

    println (s"xy = $xy")
    println (s"x  = $x")
    println (s"y  = $y")

    val knn = KNN_Classifier (xy, fname, 2, cname)()         // create classifier
    knn.trainNtest ()()                                      // no training; test
    println (knn.summary ())                                 // summary statistics

end kNN_ClassifierTest2

import Example_Iris.{x, yb}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kNN_ClassifierTest3` main function is used to test the `KNN_Classifier` class.
 *  It uses the Iris dataset where the classification/response y is unbalanced.
 *  > runMain scalation.modeling.classifying.kNN_ClassifierTest3
 */
@main def kNN_ClassifierTest3 (): Unit =


    println (s"x  = $x")
    println (s"yb = $yb")

    val knn = new KNN_Classifier (x, yb)                     // create classifier
    knn.trainNtest ()()                                      // no training; test
    println (knn.summary ())                                 // summary statistics

end kNN_ClassifierTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `kNN_ClassifierTest4` main function is used to test the `KNN_Classifier` class.
 *  It uses the Iris dataset where the classification/response y is imbalanced
 *  and downsampling is used to balance the classification.
 *  > runMain scalation.modeling.classifying.kNN_ClassifierTest4
 */
@main def kNN_ClassifierTest4 (): Unit =

    banner ("original x")
    println (s"x  = $x")
    banner ("original imbalanced yb")
    println (s"yb = $yb")

//  val idx = Classifier.downsample (yb, 100)                // use these indices - 100 part not implemented yet
    val idx = Classifier.downsample (yb)                     // use these indices
    val x_  = x(idx)                                         // new x-matrix
    val y_  = yb(idx)                                        // new y-vector

    banner ("donwsampled x_")
    println ("x_ = " + x_)
    banner ("downsampled y_")
    println ("y_ = " + y_)

    val knn = new KNN_Classifier (x_, y_)                    // create classifier
    knn.trainNtest ()()                                      // no training; test
    println (knn.summary ())                                 // summary statistics

end kNN_ClassifierTest4

