
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Feb  2 18:18:15 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Trigonometric Regression
 */

package scalation
package modeling

import scala.math.{cos, Pi, sin}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TrigRegression` class supports trigonometric regression.  In this case,
 *  't' is expanded to '[1, sin (wt), cos (wt), sin (2wt), cos (2wt), ...]'.
 *  Fit the parameter vector 'b' in the regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 sin (wt)  + b_2 cos (wt)  +
 *                                  b_3 sin (2wt) + b_4 cos (2wt) + ... + e
 *  where 'e' represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve for the parameter vector 'b'
 *  using the Normal Equations:
 *      x.t * x * b  =  x.t * y
 *      b  =  fac.solve (.)
 *  @see link.springer.com/article/10.1023%2FA%3A1022436007242#page-1
 *  @param t       the initial data/input m-by-1 matrix: t_i expands to x_i
 *  @param y       the response/ouput vector
 *  @param ord     the order (k), maximum multiplier in the trig function (kwt)
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters (defaults to Regression.hp)
 */
class TrigRegression (t: MatrixD, y: VectorD, ord: Int, fname_ : Array [String] = null,
                      hparam: HyperParameter = Regression.hp)
      extends Regression (TrigRegression.allForms (t, ord), y, fname_, hparam):

    private val w  = (2.0 * Pi) / (t.mmax - t.mmin)                         // base displacement angle in radians
    private val nt = TrigRegression.numTerms (ord)                          // number of terms/columns after expansion

    _modelName = s"TrigRegression_$ord"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the vector 'z' into a vector of that includes additional terms.
     *  @param z  the un-expanded vector
     */
    def expand (z: VectorD): VectorD = TrigRegression.forms (z, nt, w)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the scalar 'z', expand it and predict the response value.
     *  @param z  the un-expanded scalar
     */
    def predict (z: Double): Double = predict_ex (VectorD (z))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the vector 'z', expand it and predict the response value.
     *  @param z  the un-expanded vector
     */
    def predict_ex (z: VectorD): Double = predict (expand (z))

end TrigRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TrigRegression` companion object provides factory methods and functions
 *  for creating functional forms.
 */
object TrigRegression:

    private val debug = debugf ("TrigRegression", false)                    // debug function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TrigRegression` object from a combined data-response matrix.
     *  @param xy      the initial combined data-response matrix (before term expansion)
     *  @param ord     the number of harmomics
     *  @param fname_  the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, ord: Int, fname: Array [String] = null,
               hparam: HyperParameter = Regression.hp)(col: Int = xy.dim2 - 1): TrigRegression =
        new TrigRegression (xy.not(?, col), xy(?, col), ord, fname, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TrigRegression` object from a combined data-response matrix.
     *  @param t       the initial data/input vector: t_i expands to x_i = [1, t_i, t_i^2, ... t_i^k]
     *  @param y       the response/ouput vector
     *  @param ord     the number of harmomics
     *  @param fname_  the feature/variable names
     *  @param hparam  the hyper-parameters
     */
    def apply (t: VectorD, y: VectorD, ord: Int, fname: Array [String],
               hparam: HyperParameter): TrigRegression =
        val hp2 = if hparam == null then Regression.hp else hparam
        new TrigRegression (MatrixD (t).ᵀ, y, ord, fname, hp2)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TrigRegression` object from a data matrix and a response vector.
     *  This method provides data rescaling.
     *  @param x       the initial data/input matrix (before term expansion)
     *  @param y       the response/output m-vector
     *  @param ord     the number of harmomics
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     */
    def rescale (x: MatrixD, y: VectorD, ord: Int, fname: Array [String] = null,
                 hparam: HyperParameter = Regression.hp): TrigRegression =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new TrigRegression (xn, y, ord, fname, hparam)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a 1-vector/point 'v', compute the values for all of its trigonmetric
     *  forms/terms, returning them as a vector.
     *  '[1, sin (wt), cos (wt), sin (2wt), cos (2wt), ...]'.
     *  @param v   the vector/point (i-th row of t) for creating forms/terms
     *  @param nt  the number of terms
     *  @param w   the base displacement angle in radians
     */
    def forms (v: VectorD, nt: Int, w: Double): VectorD =
        val wt = w * v(0)
        val u  = new VectorD (nt)
        u(0)   = 1.0
        for j <- 1 to nt / 2 do
            u(2*j-1) = sin (j * wt)
            u(2*j)   = cos (j * wt)
        end for
        u
    end forms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of terms in the model.
     *  @param ord  the number of harmomics (max k in kwt)
     */
    def numTerms (ord: Int): Int = 2 * ord + 1 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create all forms/terms for each row/point placing them in a new matrix.
     *  @param x    the original un-expanded input/data matrix
     *  @param ord  the number of harmomics
     */
    def allForms (x: MatrixD, ord: Int): MatrixD =
        val t  = x(?, 0)                                             // first and only column of x
        val w  = (2.0 * Pi) / (t.max - t.min)                        // base displacement angle in radians
        val nt = numTerms (ord)
        println (s"allForms: create expanded data matrix with nt = $nt columns")
        val xe = new MatrixD (x.dim, nt)
        for i <- x.indices do xe(i) = forms (x(i), nt, w)            // vector with values for all forms/terms
        debug ("allForms", s"expanded data matrix xe = $xe")
        xe                                                           // expanded matrix
    end allForms

end TrigRegression


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `trigRegressionTest` main function tests `TrigRegression` class using the
 *  following regression equation.
 *  <p>
 *      y  =  b dot x  =  b_0 + b_1 sin wt + b_2 cos wt + ... b_2k-1 sin kwt + b_2k cos kwt + e
 *  <p>
 *  The data is generated from a noisy cubic function.
 *  > runMain scalation.modeling.trigRegressionTest
 */
@main def trigRegressionTest (): Unit =

    import scalation.random.Normal

    val noise = Normal (0.0, 10000.0)
    val t     = VectorD.range (0, 100)
    val y     = new VectorD (t.dim)
    for i <- 0 until 100 do { val x = (i - 40)/2.0; y(i) = 1000.0 + x + x*x + x*x*x + noise.gen }

    println (s"t = $t")
    println (s"y = $y")

    for harmonics <- 2 to 16 by 2 do
        val trg = TrigRegression (t, y, harmonics, null, null)
        trg.trainNtest ()()                                               // train and test the model

        val z   = 10.5                                                    // predict y for one point
        val yp1 = trg.predict (z)
        println (s"predict ($z) = $yp1")

        banner ("TrigRegressionTest: test predictions for harmonics = $harmonics")
        val yp = t.map (trg.predict (_))
//      println (s" y = $y \n yp = $yp")
        new Plot (t, y, yp, s"TrigRegression: harmonics = $harmonics")

        if harmonics == 16 then
           val stats = trg.crossValidate ()
           FitM.showQofStatTable (stats)
    end for

end trigRegressionTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `trigRegressionTest2` main function tests `TrigRegression` class using the following
 *  regression equation.
 *      y  =  b dot x  =  b_0 + b_1 sin wt + b_2 cos wt + ... b_2k-1 sin kwt + b_2k cos kwt + e
 *  The data is generated from periodic noisy cubic functions.
 *  > runMain scalation.modeling.trigRegressionTest2
 */
@main def trigRegressionTest2 (): Unit =

    import scalation.random.Normal

    val noise = Normal (0.0, 10.0)
    val t     = VectorD.range (0, 200)
    val y     = new VectorD (t.dim)
    for i <- 0 until 5 do
        for j <- 0 until 20 do { val x = j - 4;  y(40*i+j) = 100.0 + x + x*x + x*x*x + noise.gen }
        for j <- 0 until 20 do { val x = 16 - j; y(40*i+20+j) = 100.0 + x + x*x + x*x*x + noise.gen }
    end for

    println (s"t = $t")
    println (s"y = $y")

    for harmonics <- 2 to 16 by 2 do
        val trg = TrigRegression (t, y, harmonics, null, null)
        trg.trainNtest ()()                                               // train and test the model

        val z   = 10.5                                                    // predict y for one point
        val yp1 = trg.predict (z)
        println (s"predict ($z) = $yp1")

        banner ("TrigRegressionTest: test predictions for harmonics = $harmonics")
        val yp = t.map (trg.predict (_))
//      println (s" y = $y \n yp = $yp")
        new Plot (t, y, yp, s"TrigRegression: harmonics = $harmonics")

        if harmonics == 16 then
           val stats = trg.crossValidate ()
           FitM.showQofStatTable (stats)
    end for

end trigRegressionTest2

