
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jan  5 14:03:36 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Null Model (predict the mean)
 */

package scalation
package modeling

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NullModel` class implements the simplest type of predictive modeling technique
 *  that just predicts the response y to be the mean.
 *  Fit the parameter vector b in the null regression equation
 *  <p>
 *      y  =  b dot x + e  =  b0 + e
 *  <p>
 *  where e represents the residual/error vector (the part not explained by the model).
 *  @param y  the response/output vector
 */
class NullModel (y: VectorD)
      extends Predictor (MatrixD.one (y.dim), y, Array ("one"), null)
         with Fit (dfr = 0, df = y.dim-1)
         with NoSubModels:

    _modelName = "NullModel"

    override def getBest: BestStep = super [NoSubModels].getBest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the predictor by fitting the parameter vector (b-vector) in the
     *  null regression equation.
     *  @param x_null  the training/full data/input matrix (IGNORED by `NullModel`)
     *  @param y_      the training/full response/output vector
     */
    def train (x_null: MatrixD = null, y_ : VectorD = y): Unit =
        b = VectorD (y_.mean)                                            // parameter vector [b0]
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.
     *  @param x_null  the testing/full data/input matrix (IGNORED by `NullModel`)
     *  @param y_      the testing/full response/output vector (defaults to full y)
     */
    def test (x_null: MatrixD = null, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = VectorD.fill (y_.dim)(b(0))                             // y predicted for (test/full)
        (yp, diagnose (y_, yp))                                          // return predictions and QoF Vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z,
     *  i.e., [b0] dot [z0].
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): Double = b(0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b(0),
     *  for each row of matrix x_.
     *  @param z  the new matrix to predict (only used for dimension)
     */
    override def predict (x_ : MatrixD): VectorD = VectorD.fill (x_.dim)(b(0))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor x_0, x_1,
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname, b_ : VectorD = b,
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                       // summary from `Fit`
    end summary

end NullModel


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NullModel` companion object provides a simple factory method
 *  for building null models.
 */
object NullModel:

    private val flaw = flawf ("NullModel")                     // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a null model from a combined data matrix taking the last column as the response.
     *  @param xy  the combined data matrix
     */
    def apply (xy: MatrixD): NullModel =
        val n = xy.dim2
        if n < 1 then { flaw ("apply", "the length of the 'xy' matrix must be at least 1"); null }
        else new NullModel (xy(?, n-1))
    end apply

end NullModel


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nullModelTest` main function is used to test the `NullModel` class.
 *      y = b dot x + e = b0 + e
 *  > runMain scalation.modeling.nullModelTest
 */
@main def nullModelTest (): Unit =

    // 4 data points:
    val y = VectorD (1, 3, 3, 4)                            // response vector y
    println (s"y = $y")

    val mod = new NullModel (y)                             // create a null model
    mod.trainNtest ()()                                     // train and test the model

    val z  = VectorD (5)                                    // predict y for one point
    val yp = mod.predict (z)
    println (s"predict ($z) = $yp")

end nullModelTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nullModelTest2` main function is used to test the `NullModel` class.
 *      y = b dot x + e = b0 + e
 *  > runMain scalation.modeling.nullModelTest2
 */
@main def nullModelTest2 (): Unit =

    // 5 data points:
    val y = VectorD (2.0, 3.0, 5.0, 4.0, 6.0)               // response vector y
    println (s"y = $y")

    val mod = new NullModel (y)                             // create a null model
    mod.trainNtest ()()                                     // train and test the model

    val z  = VectorD (5.0)                                  // predict y for one point
    val yp = mod.predict (z)                                // yp (y-predicted or y-hat)
    println (s"predict ($z) = $yp")

end nullModelTest2

