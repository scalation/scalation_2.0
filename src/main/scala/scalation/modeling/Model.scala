
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jun  9 16:42:16 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework: Base Trait for all Models
 */

package scalation
package modeling

import java.net.URI

import scala.collection.mutable.{ArrayBuffer, IndexedSeq}
import scala.math.{abs, round}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TaskType` enum specifies the types of tasks supported by ScalaTion.
 *  @param name  the name of task type
 *  @param base  the base trait/abstract class for the given type of task
 */
enum TaskType (val name: String, val base: String):

    case Predict  extends TaskType ("Predict", "Predictor")                // FIX -- make compiler checked
    case Forecast extends TaskType ("Forecast", "forecasting.Forecaster")
    case Classify extends TaskType ("Classify", "classifying.Classifier")
    case Cluster  extends TaskType ("Cluster", "clustering.Clusterer")

end TaskType


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Order vectors y_ and yp_ according to the ascending order of y_.
 *  This can be used for graphical comparison or actual and predicted values.
 *  @param y_   the vector to order by (e.g., actual response values)
 *  @param yp_  the vector to be order by y_ (e.g., predicted response values)
 */
def orderByY (y_ : VectorD, yp_ : VectorD): (VectorD, VectorD) =
    val rank = y_.iqsort                                                 // rank order for vector y_
    (y_.reorder (rank), yp_.reorder (rank))                              // (y_ in ascending order, yp_ ordered by y_)
end orderByY

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Order matrix ys according to the ascending order of ys(0) (row vector by row vector).
 *  This can be used for graphical comparison or actual and predicted values and intervals.
 *  @param ys  the matrix of vectors to reorder based the first row (typically the y-actual value)
 */
def orderByY (ys: MatrixD): MatrixD =
    val oys  = new MatrixD (ys.dim, ys.dim2) 
    val rank = ys(0).iqsort                                              // rank order for vector y_
    oys(0)   = ys(0).reorder (rank)                                      // ys(0) in ascending order
    for i <- 1 until ys.dim do
        oys(i) = ys(i).reorder (rank)                                    // row i of ys ordered by ys(0)
    oys
end orderByY

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Order matrices y_ and yp_ according to the ascending order of y_
 *  (column vector by column vector).
 *  This can be used for graphical comparison or actual and predicted values.
 *  @param y_   the matrix to order by (e.g., actual response values)
 *  @param yp_  the matrix to be order by y_ (e.g., predicted response values)
 */
def orderByYY (y_ : MatrixD, yp_ : MatrixD): (MatrixD, MatrixD) =
    val (oy, oyp) = (new MatrixD (y_.dim, y_.dim2), new MatrixD (yp_.dim, yp_.dim2))
    for j <- y_.indices2 do
        val yj    = y_(?, j)                                             // column j of y_
        val rank  = yj.iqsort                                            // rank order for vector yj
        oy(?, j)  = yj.reorder (rank)                                    // yj in ascending order
        oyp(?, j) = yp_(?, j).reorder (rank)                             // column j of yp_ ordered by yj
    (oy, oyp)
end orderByYY


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Model` trait provides a common framework for all models and serves as
 *  base trait for `Classifier`, `Forecaster`, `Predictor`, and `PredictorMV` traits.
 *  The train and test methods must be called first, e.g.,
 *       val model = NullModel (y)
 *       model.train (null, y)
 *       model.test (null, y)
 */
trait Model:

    /** The flaw function used for writing errors and warnings
     */
    private val flaw = flawf ("Model")

    /** The optional reference to an ontological concept
     */
    var modelConcept: URI = null

    /** The name for the model (or modeling technique).  Each model should reassign.
     */
    protected var _modelName: String = "Model"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the model name.
     */
    inline def modelName: String = _modelName

    /** The type of task the model performs.  Base traits for other tasks for reassign.
     */
    protected var _taskType: TaskType = TaskType.Predict

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the type of the task performed by model.
     */
    inline def taskType: TaskType = _taskType

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used data matrix x.  Mainly for derived classes where x is expanded
     *  from the given columns in x_, e.g., `SymbolicRegression.quadratic` adds squared columns.
     */
    def getX: MatrixD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the data matrix x concatenated with response vector y.
     */
    def getXy: MatrixD = getX :^+ getY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response vector y.  Mainly for derived classes where y is
     *  transformed, e.g., `TranRegression`, `ARX`.
     */
    def getY: VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the used response matrix y, if needed.
     *  @see `neuralnet.PredictorMV`
     */
    def getYY: MatrixD = null

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the feature/variable names.
     */
    def getFname: Array [String]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the model 'y_ = f(x_) + e' on a given dataset, by optimizing the model
     *  parameters in order to minimize error '||e||' or maximize log-likelihood 'll'.
     *  @param x_  the training/full data/input matrix (impl. classes may default to x)
     *  @param y_  the training/full response/output vector (impl. classes may default to y)
     */
    def train (x_ : MatrixD, y_ : VectorD): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test/evaluate the model's Quality of Fit (QoF) and return the predictions
     *  and QoF vectors.
     *  This may include the importance of its parameters (e.g., if 0 is in a parameter's
     *  confidence interval, it is a candidate for removal from the model).
     *  Extending traits and classes should implement various diagnostics for
     *  the test and full (training + test) datasets.
     *  @param x_  the testing/full data/input matrix (impl. classes may default to x)
     *  @param y_  the testing/full response/output vector (impl. classes may default to y)
     */
    def test (x_ : MatrixD, y_ : VectorD): (VectorD, VectorD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the model equation.
     *  Single output models return `Double`, while multi-output models return `VectorD`.
     *  @param z  the new vector to predict
     */
    def predict (z: VectorD): Double | VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the model hyper-parameters (if none, return null).  Hyper-parameters
     *  may be used to regularize parameters or tune the optimizer.
     */
    def hparameter: HyperParameter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of model parameter/coefficient values.
     *  Single output models have `VectorD` parameters, while multi-output models have `MatrixD`.
     */
    def parameter: VectorD | MatrixD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the model prediction equation in readable form.
     *  Override per model.
     */
    def equation: String = s"model prediction equation $modelName: ŷ = f(${parameter})"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print the model prediction equation in LaTex form.
     *  Override per model.
     */
    def equationLaTeX: String = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a basic report on a trained and tested model.
     *  @param ftVec  the vector of qof values produced by the `Fit` trait
     */
    def report (ftVec: VectorD): String =
        val (fn, b) = (getFname, parameter.asInstanceOf [VectorD])
        println (s"b = $b, fn = $fn")
        if fn == null then flaw ("report", "no feature names given fn = null")
        else if fn.size != b.dim then flaw ("report", s"# featuers = ${fn.size} != # parameters = ${b.dim}")
        s"""
REPORT
    ----------------------------------------------------------------------------
    modelName  mn  = $modelName
    ----------------------------------------------------------------------------
    hparameter hp  = $hparameter
    ----------------------------------------------------------------------------
    features   fn  = ${stringOf (fn)}
    ----------------------------------------------------------------------------
    parameter  b   = $b
    ----------------------------------------------------------------------------
    fitMap     qof = ${FitM.fitMap (ftVec, QoF.values.map (_.toString))}
    ----------------------------------------------------------------------------
        """
    end report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a basic report on a trained and tested multi-variate model.
     *  @param ftMat  the matrix of qof values produced by the `Fit` trait
     */
    def report (ftMat: MatrixD): String =
        val (fn, b) = (getFname, parameter.asInstanceOf [MatrixD])
        if fn.size != b.dim then flaw ("report", s"# featuers = ${fn.size} != # parameters = ${b.dim}")
        s"""
REPORT
    ----------------------------------------------------------------------------
    modelName  mn  = $modelName
    ----------------------------------------------------------------------------
    hparameter hp  = $hparameter
    ----------------------------------------------------------------------------
    features   fn  = ${stringOf (fn)}
    ----------------------------------------------------------------------------
    parameter  b   = $b
    ----------------------------------------------------------------------------
    fitMap     qof = ${FitM.fitMap (ftMat, QoF.values.map (_.toString))}
    ----------------------------------------------------------------------------
        """
    end report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Screen the x-columns of matrix xy based on the two thresholds, returning
     *  the reduced matrix and the column indices/predictor variables selected.
     *  @param xy    the [ x, y ] combined data-response matrix
     *  @param thr1  the threshold used to compare the predictor x-columns to the y-column
     *               only want variables above some minimal dependency level
     *  @param thr2  the threshold used to compare the predictor x-columns with each other
     *               only want variables below some cut-off dependency/collinearity level
     *  @param dep   the variable/column dependency measure (defaults to correlation)
     */
    def screen (xy: MatrixD, thr1: Double = 0.05, thr2: Double = 0.95)
               (dep: MatrixD = xy.corr): (MatrixD, VectorI) =

        val lst  = dep.dim2 - 1                                          // the index of last column (holds y)
        val depY = dep(?, lst)                                           // the dependency sub-matrix for xy vs. y (last column)
        val depX = dep(0 until lst, 0 until lst)                         // the dependency sub-matrix for x vs. x
        val indices  = for i <- 0 until lst if abs (depY(i)) > thr1
                       yield i                                           // row indices that match (> thr1)
        val sIndices = indices.sortBy (i => -abs (depY(i)))              // sort indices from highest dep to lowest

        // only add index i if its dependency with all selected columns < thr2
        val selected = ArrayBuffer [Int] ()
        for i <- sIndices do
            if selected.forall (k => abs (depX(i, k)) < thr2) then
                selected += i                                            // row indices that also match (< thr2)
        val selected_ = selected.sorted

        (xy(?, selected_), new VectorI (selected_.size, selected_.toArray))
    end screen

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the best model found from feature selection.
     */
    def getBest: BestStep

//  T E S T I N G   S C E N A R I O S

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform In-Sample Testing, i.e., train and test on the same FULL data set.
     *  Good for initial testing and understanding variable relationships.
     *  @note May lead to over-fitting for complex models.
     *  @param skip    the number of initial data points to skip (due to insufficient information)
     *  @param showYp  whether to show the prediction vector
     */
    def inSample_Test (skip: Int = 0, showYp: Boolean = false): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use validation to compute test Quality of Fit (QoF) measures by dividing
     *  the full dataset into a TESTING-set and a TRAINING-set, returning qof and yp.
     *  The testing-set is defined by idx and the rest of the data is the training-set.
     *  @note:  Results depend on which testing-set is chosen.
     *  @see `modeling.Predictor.validate` for how to choose (1) RANDOM, (2) FIRST, or (3) LAST.
     *  @param rando  flag indicating whether to use randomized or simple validation
     *  @param ratio  the ratio of the TESTING-set to the full dataset (most common 70-30 (.3), 80-20 (.2))
     *  @param idx    the prescribed TESTING-set indices
     */
    def validate (rando: Boolean = true, ratio: Double = Model.TE_RATIO)
                 (idx: IndexedSeq [Int] = null): (VectorD | MatrixD, VectorD | MatrixD)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert QoF results into an array (of size 1) of `Statistic` for compatibility
     *  with the `crossValidate` method.
     *  @param qof  the Quality of Fit (QoF) results
     */
    def qof2Stat (qof: VectorD): Array [Statistic] =
        val stats = Fit.qofStatTable                                     // create table for QoF measures
        if qof(QoF.sst.ordinal) > 0.0 then                               // requires variation in test-set
            for q <- qof.indices do stats(q).tally (qof(q))              // tally these QoF measures
        stats
    end qof2Stat

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use k-fold cross-validation to compute test Quality of Fit (QoF) measures
     *  by iteratively dividing the full dataset into a TRAINING and a TESTING set.
     *  Each test set is defined by idx and the rest of the data is the training set.
     *  @note:  Replace with `rollValidate` for forecasting tasks.
     *  @see showQofStatTable in `Fit` object for printing the returned stats.
     *  @param k      the number of cross-validation iterations/folds (defaults to 5x).
     *  @param rando  flag indicating whether to use randomized or simple cross-validation
     */
    def crossValidate (k: Int = 5, rando: Boolean = true): Array [Statistic]

end Model


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Model` companion object provides methods useful for classes extending the `Model` trait.
 */
object Model:

    private val flaw = flawf ("Model")                                    // flaw function

    private var _TE_RATIO = 0.2                                           // ratio of TESTING-set to FULL dataset

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the TE ratio = ratio of the size of the TESTING-set to size of the FULL dataset.
     */
    inline def TE_RATIO: Double = _TE_RATIO

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the TE ratio = ratio of the size of the TESTING-set to size of the FULL dataset.
     *  @param ratio  the new ratio of the size of the TESTING-set to size of the FULL dataset
     */
    def TE_RATIO_= (ratio: Double): Unit =
        if ratio out (0.05, 0.95) then flaw ("init", s"testing ratio = $ratio should be in (0.05, 0.95)")
        _TE_RATIO = ratio
    end TE_RATIO_=

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the size (number of instances) for a testing set (round up).
     *  @param m  the size of the full dataset
     */
    inline def teSize (m: Int): Int = (round (m * TE_RATIO + 0.5)).toInt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the size (number of instances) for a training set.
     *  @param m  the size of the full dataset
     */
    inline def trSize (m: Int): Int = m - teSize (m)

end Model

