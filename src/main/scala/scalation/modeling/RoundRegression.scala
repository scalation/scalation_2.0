
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Tue Oct  9 17:40:30 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Rounded Multiple Linear Regression
 */

package scalation
package modeling

import scala.math.round

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RoundRegression` class supports rounded multiple linear regression.
 *  In this case, 'x' is multi-dimensional [1, x_1, ... x_k].  Fit the parameter
 *  vector 'b' in the transformed regression equation
 *      y  =  round (b dot x) + e  =  round (b_0 + b_1 * x_1 +  b_2 * x_2 ... b_k * x_k) + e
 *  where 'e' represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to fit the parameter vector 'b'
 *  @param x       the data/input matrix
 *  @param y       the response/output vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters (defaults to Regression.hp)
 */
class RoundRegression (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                       hparam: HyperParameter = Regression.hp)
      extends Regression (x, y, fname_, hparam):

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Round the elements of the given vector to their nearest integer values.
     *  @param v  the unrounded vector
     */
    def vround (v: VectorD): VectorD = VectorD (for i <- v.indices yield round (v(i)).toDouble)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z,
     *  e.g., (b_0, b_1, b_2) dot (1, z_1, z_2).
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): Double = round (b dot z).toDouble
             def ipredict (z: VectorD): Int   = round (b dot z).toInt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z for
     *  each row of matrix z.
     *  @param z  the new matrix to predict
     */
    override def predict (z: MatrixD = x): VectorD  = vround (z * b)
             def ipredict (z: MatrixD = x): VectorI = vround (z * b).toInt

end RoundRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RoundRegression` companion object provides factory methods for creating
 *  rounded regression models.
 */
object RoundRegression:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RoundRegression` object using a combined matrix.
     *  @param xy      the combined data matrix and response vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = Regression.hp)(col: Int = xy.dim2 - 1): RoundRegression =
        new RoundRegression (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RoundRegression` object for an integer response vector.
     *  @param x       the data/input matrix
     *  @param y       the integer response/output vector
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters (it doesn't have any, but may be used by derived classes)
    */
    def apply (x: MatrixD, y: VectorI, fname: Array [String],
               hparam: HyperParameter): RoundRegression =
        new RoundRegression (x, y.toDouble, fname, hparam)
    end apply

    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = Regression.hp): RoundRegression = ???

end RoundRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `roundRegressionTest` main function tests `RoundRegression` class using the following
 *  regression equation.
 *      y  =  round (b dot x)  =  round (b_0 + b_1*x_1 + b_2*x_2).
 *  > runMain scalation.modeling.roundRegressionTest
 */
@main def roundRegressionTest (): Unit =

    //                        1 x0 x1  y
    val xy = MatrixD ((9, 4), 1, 0, 0, 1,                         // 9-by-4 matrix
                              1, 0, 1, 0,
                              1, 0, 2, 0,
                              1, 1, 0, 1,
                              1, 1, 1, 0,
                              1, 1, 2, 0,
                              1, 2, 0, 1,
                              1, 2, 1, 0,
                              1, 2, 2, 1)

    println (s"xy = $xy")
    val (x, y) = (xy.not(?, 3), xy(?, 3))                          // (all except column 3, column 3)

    val mod = new RoundRegression (x, y)                           // create a rounded regression model
    mod.inSample_Test ()                                           // train and test the model

    val yp  = mod.predict (x)
    val ypi = mod.vround (yp)
    for j <- y.indices do println (s"y_j = ${y(j)} \t ypi_j = ${ypi(j)} \t yp_j = ${yp(j)}")

end roundRegressionTest

