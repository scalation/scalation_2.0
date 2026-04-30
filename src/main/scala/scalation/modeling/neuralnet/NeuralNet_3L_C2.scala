
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Dec 21 12:53:44 EST 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Neural Network Classifier with 3 Layers (input, hidden and output layers)
 *           for 2-way classification.
 *
 *  @see     hebb.mit.edu/courses/9.641/2002/lectures/lecture03.pdf
 */

package scalation
package modeling
package neuralnet

import scalation.mathstat._
import scalation.modeling.classifying._

import MatrixD.fromVector
import ActivationFun._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3L_C2` class supports single-output, 3-layer (input, hidden and output)
 *  Neural-Network classifiers (for k = 2).  Given several input vectors and an output vector
 *  (training data), fit the parameters a and b connecting the layers, so that for a new
 *  input vector z, the net can classify the output value, i.e.,
 *      yp = f1 (b * f (a * z))
 *  where f and f1 (sigmoid) are the activation functions and the parameter a and b
 *  are the parameters between input-hidden and hidden-output layers.
 *  @param x       the m-by-n input/data matrix (training data consisting of m input vectors)
 *  @param y       the m output/response vector (training data consisting of m output integer values)
 *  @param fname_  the feature/variable names (if null, use x_j's)
 *  @param cname_  the names for all classes
 *  @param nz      the number of nodes in hidden layer (-1 => use default formula)
 *  @param hparam  the hyper-parameters
 *  @param f       the activation function family for layers 1->2 (input to hidden)
 *                 the activation function family for layers 2->3 (hidden to output) is sigmoid
 */
class NeuralNet_3L_C2 (x: MatrixD, y: VectorI, fname_ : Array [String] = null,
                       cname_ : Array [String] = Array ("No", "Yes"),
                       nz: Int = -1, hparam: HyperParameter = NeuralNet_3L_C2.hp,
                       f: AFF = f_id)
      extends Classifier (x, y, fname_, 2, cname_, hparam)
         with FitC ():

    private val debug   = debugf ("NeuralNet_3L_C2", true)               // debug function
    private val cThresh = hparam("cThresh").toDouble                     // classification/decision threshold

    private val ym = fromVector (y.toDouble)                             // y as a matrix

    private val nn3 = new NeuralNet_3L (x, ym, fname_, nz, hparam, f, f_sigmoid)

    _modelName = s"NeuralNet_3L_C2_${f.name}_sigmoid"                    // name of the model

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the classifier, i.e., calculate statistics and create conditional
     *  density 'cd' functions.  Assumes that conditional densities follow the
     *  Normal (Gaussian) distribution.
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     */
    override def train (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        super.train (x_, y_)                                             // set class frequencies nu_y and probabilities p_y
        nn3.train (x_, fromVector (y_.toDouble))
    end train

    override def train2 (x_ : MatrixD = x, y_ : VectorI = y): Unit =
        super.train (x_, y_)                                             // set class frequencies nu_y and probabilities p_y
        nn3.train2 (x_, fromVector (y_.toDouble))
    end train2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.
     *  FIX - currently must override if y is transformed, @see `TranRegression`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output matrix (defaults to full y)
     *
    def trainNtest (x_ : MatrixD = x, y_ : VectorI = y)
                   (xx: MatrixD = x, yy: VectorI = y): (MatrixD, MatrixD) =
        nn3.trainNtest (x_, fromVector (y_.toDouble))(xx, fromVector (yy.toDouble))
    end trainNtest
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.  This version does auto-tuning.
     *  FIX - currently must override if y is transformed, @see `TranRegression`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output matrix (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output matrix (defaults to full y)
     */
    def trainNtest2 (x_ : MatrixD = x, y_ : VectorI = y)
                    (xx: MatrixD = x, yy: VectorI = y): (MatrixD, MatrixD) =
        nn3.trainNtest2 (x_, fromVector (y_.toDouble))(xx, fromVector (yy.toDouble))
    end trainNtest2

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

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the integer value of y = f(z) by computing the sigmoid value
     *  from nn3 and taking the 0/1 based on cTresh.
     *  @param z  the new vector to predict
     */
    override def predictI (z: VectorD): Int = (nn3.predict (z)(0) + cThresh).toInt

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

end NeuralNet_3L_C2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3L_C2` companion object provides factory methods for building three-layer
 *  (one hidden layer) neural network classifiers.  Note, 'scale' is defined in `Scaling`.
 */
object NeuralNet_3L_C2 extends Scaling:

    val hp = Classifier.hp ++ Optimizer.hp                               // combine the hyper-parameters

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_3L_C2` classifier for the given combined matrix where
     *  the column col is the response/classification vector.
     *  @param xy      the combined data matrix (features and response)
     *  @param fname   the names for all features/variables
     *  @param cname   the names for all classes
     *  @param nz      the number of nodes in hidden layer
     *  @param hparam  the hyper-parameters
     *  @param f       the activation function family for layers 1->2 (input to output)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               cname: Array [String]  = Array ("No", "Yes"),
               nz: Int = -1, hparam: HyperParameter = hp, f: AFF = f_id)
              (col: Int = xy.dim2 - 1): NeuralNet_3L_C2 =
        val (x, y) = (xy.not(?, col), xy(?, col).toInt)                  // data matrix, response vector
        val x_s = if scale then rescaleX (x, f)
                  else x

//      println (s" scaled: x = $x_s \n y = $y")
        new NeuralNet_3L_C2 (x_s, y, fname, cname, nz, hparam, f)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `NeuralNet_3L_C2` classifier for a data matrix and response vector.
     *  @param x       the input/data matrix
     *  @param y       the output/response vector
     *  @param fname   the feature/variable names
     *  @param cname   the names for all classes
     *  @param nz      the number of nodes in hidden layer
     *  @param hparam  the hyper-parameters
     *  @param f       the activation function family for layers 1->2 (input to output)
     */
    def rescale (x: MatrixD, y: VectorI, fname: Array [String] = null,
                 cname: Array [String] = Array ("No", "Yes"),
                 nz: Int = -1, hparam: HyperParameter = hp,
                 f: AFF = f_id): NeuralNet_3L_C2 =
        val x_s = if scale then rescaleX (x, f)
                  else x

//      println (s" scaled: x = $x_s \n y = $y")
        new NeuralNet_3L_C2 (x_s, y, fname, cname, nz, hparam, f)
    end rescale

end NeuralNet_3L_C2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `neuralNet_3L_C2Test` main function tests the `NeuralNet_3L_C2` class.
 *  It tests the Neural Network three layer classifier on the Diabetes dataset.
 *  > runMain scalation.modeling.classifying.neuralNet_3L_C2Test
 */
@main def neuralNet_3L_C2Test: Unit =

    val nfile  = "diabetes.csv"
    val xy     = MatrixD.load (nfile)
    val fn     = Array ("pregnancies", "glucose", "blood pressure", "skin thickness", "insulin",
                        "BMI", "diabetes pedigree function", "age")    // feature names
    val cn     = Array ("tested_positive", "tested_negative")          // class names

    banner ("neuralNet_3L_C2Test: diabetes dataset")
    val hp2 = NeuralNet_3L_C2.hp.updateReturn (("cThresh", 0.48), ("eta", 0.2))
    val nnc = NeuralNet_3L_C2 (xy, fn, cn, -1, hp2)()
    nnc.trainNtest ()()

    banner ("NullModel: diabetes dataset")
    val nm = NullModel (xy)()
    nm.trainNtest ()()

end neuralNet_3L_C2Test

