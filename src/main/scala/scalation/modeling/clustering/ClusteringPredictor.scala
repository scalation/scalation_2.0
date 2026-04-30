
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Dec 21 14:38:32 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Prediction based upon Clustering
 */

package scalation
package modeling
package clustering

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
//import scalation.random.Bernoulli

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ClusteringPredictor` class is used to predict a response value for new vector 'z'.
 *  It works by finding the cluster that the point 'z' would belong to.
 *  The recorded response value for 'y' is then given as the predicted response.
 *  The per cluster recorded reponse value is the consensus (e.g., average) of
 *  the individual predictions for 'z' from the members of the cluster.
 *  Training involves clustering the points in data matrix 'x' and then computing
 *  each clusters reponse.
 *  @param x       the vectors/points of predictor data stored as rows of a matrix
 *  @param y       the response value for each vector in x
 *  @param fname_  the names for all features/variables
 *  @param hparam  the number of nearest neighbors to consider
 */
class ClusteringPredictor (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                           hparam: HyperParameter = ClusteringPredictor.hp)
      extends Predictor (x, y, fname_, hparam)
         with Fit (dfr = x.dim2 - 1, df = x.dim - x.dim2)
         with NoSubModels:

    private val debug      = debugf ("ClusteringPredictor", false)   // debug flag
    private val MAX_DOUBLE = Double.PositiveInfinity                 // infinity
    private val kappa      = hparam("kappa").toInt                   // the number of nearest neighbors to consider
    private val topK       = Array.fill (kappa)(-1, MAX_DOUBLE)      // top-kappa nearest points (in reserve order)
//  private val coin       = Bernoulli ()                            // use a fair coin for breaking ties

    private val clust = new KMeansClusterer (x, kappa)               // underlying clustering algorithm
    private val yclus = new VectorD (kappa)                          // consensus response per cluster

    debug ("init", s" x = $x \n y = $y")

    // FIX - currently only works for xx = x and yy = y

    override def getBest: BestStep = super [NoSubModels].getBest

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Training involves resetting the data structures before each prediction.
     *  It uses lazy training, so most of it is done during prediction.
     *  @param xx  the data/input matrix
     *  @param yy  the response/output vector
     */
    def train (xx: MatrixD = x, yy: VectorD = y): Unit =
        clust.train ()
        val clustr = clust.cluster
        debug ("train", s"clusters = ${stringOf (clustr)}")
        assignResponse (clustr)
        debug ("train", s"yclus = $yclus")
    end train

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assign a consenus (average) response value for each cluster.
     *  @param clustr  the cluster assignments
     */
    private def assignResponse (clustr: Array [Int]): Unit =
        for i <- y.indices do yclus(clustr(i)) += y(i)
        yclus /= clust.csize.toDouble
    end assignResponse

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the error and useful diagnostics.  Requires override to adjust
     *  degrees of freedom (df1, df2).
     *  @param xx  the data matrix used in prediction
     *  @param yy  the actual response vector
     */
    override def test (xx: MatrixD = x, yy: VectorD = y): (VectorD, VectorD) =
        val yp = predict (xx)                                            // y predicted for xx (test/full)
        val df1 = kappa                                                  // degrees of freedom model = kappa
        val df2 = yy.dim - df1                                           // degrees of freedom error
        resetDF ((df1, df2))
        (yp, diagnose (yy, yp))                                          // return prediction & diagnostics
    end test

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new point/vector 'z', classify it according to the cluster it
     *  belongs to.
     *  @param z  the vector to classify
     */
    def classify (z: VectorD): Int = clust.classify (z)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a new point/vector 'z', predict its response value based on the
     *  average response from its clsuter.
     *  @param z  the vector to predict
     */
    override def predict (z: VectorD): Double = yclus (clust.classify (z))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset or re-initialize 'topK' and counters.
     */
    def reset (): Unit =
        for i <- 0 until kappa do topK(i) = (-1, MAX_DOUBLE)     // initialize top-kappa
    end reset

end ClusteringPredictor


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ClusteringPredictor` companion object provides a factory functions.
 */
object ClusteringPredictor:

    val hp = new HyperParameter; hp += ("kappa", 3, 3)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `ClusteringPredictor` object from a combined xy data-response matrix.
     *  @param xy      the combined data-response matrix
     *  @param fname   the names for all features/variables
     *  @param hparam  the number of nearest neighbors to consider
     *  @param col     the column containing the response variable
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = hp)
               (col: Int = xy.dim2-1): ClusteringPredictor =
        val (x, y) = (xy.not(?, col), xy(?, col))
        new ClusteringPredictor (x, y, fname, hparam)
    end apply

end ClusteringPredictor


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `clusteringPredictorTest` object is used to test the `ClusteringPredictor` class.
 *  > runMain scalation.modeling.clustering.clusteringPredictorTest
 */
@main def clusteringPredictorTest (): Unit =

    //                        x0 x1  y
    val xy = MatrixD ((10, 3), 1, 5, 1,       // joint data matrix
                               2, 4, 1,
                               3, 4, 1,
                               4, 4, 1,
                               5, 3, 0,
                               6, 3, 1,
                               7, 2, 0,
                               8, 2, 0,
                               9, 1, 0,
                              10, 1, 0)

    println ("-" * 60)
    println (s"xy = $xy")

//  val fn = Array ("x1", "x2")                   // feature/variable names
//  val cp = ClusteringPredictor (xy, fn)()
    val cp = ClusteringPredictor (xy)()
    cp.trainNtest ()()

    val z1 = VectorD (10.0, 10.0)
    println ("-" * 60)
    println ("z1 = " + z1)
    println ("yp = " + cp.predict (z1))

    val z2 = VectorD ( 3.0,  3.0)
    println ("-" * 60)
    println ("z2 = " + z2)
    println ("yp = " + cp.predict (z2))

    val (x, y) = (xy.not(?, 2), xy(?, 2))
    val yp = cp.predict (x)
    println ("y  = " + y)
    println ("yp = " + yp)

    new Plot (xy(?, 0), y, yp, lines = true)

end clusteringPredictorTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `clusteringPredictorTest2` object is used to test the `ClusteringPredictor` class.
 *  > runMain scalation.modeling.clustering.clusteringPredictorTest2
 */
@main def clusteringPredictorTest2 (): Unit =

    //                        x0 x1  y
    val xy = MatrixD ((9, 3), 0, 0, 0,
                              0, 1, 0,
                              0, 2, 1,
                              1, 0, 0,
                              1, 1, 0,
                              1, 2, 1,
                              2, 0, 1,
                              2, 1, 1,
                              2, 2, 1)
    val (x, y) = (xy.not(?, 2), xy(?, 2)) 

    val fn = Array ("x1", "x2")                   // feature/variable names

    println ("----------------------------------------------------")
    println ("xy = " + xy)
    println ("----------------------------------------------------")
    println ("x = " + x)

    val cp = ClusteringPredictor (xy, fn)()
    cp.trainNtest ()()

    // test samples -----------------------------------------------------------
    for i <- y.indices do
        println (s"ClusteringPredictor ($i): predict (${x(i)}) = ${cp.predict (x(i))} =? ${y(i)}")
    end for

end clusteringPredictorTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `clusteringPredictorTest3` object is used to test the `ClusteringPredictor` class.
 *  Test on AutoMPG dataset and compare with `KNN_Regression`.
 *  > runMain scalation.modeling.clustering.clusteringPredictorTest3
 */
@main def clusteringPredictorTest3 (): Unit =

    import Example_AutoMPG._

    println ("-" * 60)
    println (s"xy = $xy")

    val cap = 30
    val kr  = VectorD.range (0, cap)

    banner ("Test ClusteringPredictor on AutoMPG")
    val rSq = new MatrixD (cap, 3)                                // R^2, R^2 Bar,  R^2 cv

    for k <- 2 until cap do
        ClusteringPredictor.hp("kappa") = k
        val cp = ClusteringPredictor (xy)()
        cp.trainNtest ()

        val result = cp.crossValidate ()                          // cross-validation result
        val cv     = result(QoF.rSq.ordinal).mean                 // mean for R^2
        rSq(k)     = VectorD (100 * cp.fit(QoF.rSq.ordinal),      // R^2 percentage
                              100 * cp.fit(QoF.rSqBar.ordinal),   // R^2 Bar percentage
                              100 * cv)                           // R^2 cv percentage
    end for

    new PlotM (kr, rSq.transpose, null, "ClusteringPredictor", lines = true)


    banner ("Test KNN_Regression on AutoMPG")
    val rSQ = new MatrixD (cap, 3)                                // R^2, R^2 Bar,  R^2 cv

    for k <- 2 until cap do
        KNN_Regression.hp("kappa") = k
        val knn = KNN_Regression (xy)()
        knn.trainNtest ()()                                       // lazy/late training in KNN

        val result = knn.crossValidate ()                         // cross-validation result
        val cv     = result(QoF.rSq.ordinal).mean                 // mean for R^2
        rSQ(k)     = VectorD (100 * knn.fit(QoF.rSq.ordinal),     // R^2 percentage
                              100 * knn.fit(QoF.rSqBar.ordinal),  // R^2 Bar percentage
                              100 * cv)                           // R^2 cv percentage
    end for

    new PlotM (kr, rSQ.transpose, null, "KNN_Regression", lines = true)

end clusteringPredictorTest3

