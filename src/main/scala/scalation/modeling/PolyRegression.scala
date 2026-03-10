
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Polynomial Regression (for one variable)
 */

package scalation
package modeling

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PolyRegression` class supports polynomial regression.  In this case,
 *  t is expanded to [1, t, t^2 ... t^k].  Fit the parameter vector b in the
 *  regression equation
 *      y  =  b dot x + e  =  b_0 + b_1 * t +  b_2 * t^2 ... b_k * t^k + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve for the parameter vector b
 *  using the Normal Equations:
 *      x.t * x * b  =  x.t * y
 *      b  =  fac.solve (.)
 *  @see www.ams.sunysb.edu/~zhu/ams57213/Team3.pptx
 *  @param t       the initial data/input m-by-1 matrix: t_i expands to x_i = [1, t_i, t_i^2, ... t_i^k]
 *  @param y       the response/ouput vector
 *  @param ord     the order (k) of the polynomial (max degree)
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters (defaults to PolyRegression.hp)
 */
class PolyRegression (t: MatrixD, y: VectorD, ord: Int, fname_ : Array [String] = null,
                      hparam: HyperParameter = PolyRegression.hp)
      extends Regression (PolyRegression.allForms (t, ord), y, fname_, hparam):

    private val flaw = flawf ("PolyRegression")                    // flaw function
    private val nt   = PolyRegression.numTerms (ord)               // number of terms/columns after expansion

    _modelName = s"PolyRegression_$ord"

    if t.dim2 != 1 then flaw ("init", "matrix t must have 1 column")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the vector z into a vector of that includes additional terms,
     *  i.e., add polynomial terms.
     *  @param z  the un-expanded vector
     */
    def expand (z: VectorD): VectorD = PolyRegression.forms (z, nt) 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the scalar z, expand it and predict the response value.
     *  @param z  the un-expanded scalar
     */
    def predict (z: Double): Double = predict_ex (VectorD (z))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the vector z, expand it and predict the response value.
     *  @param z  the un-expanded vector
     */
    def predict_ex (z: VectorD): Double = predict (expand (z))

end PolyRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PolyRegression` companion object provides factory methods for creating
 ** polynomial regression models and methods for creating functional forms.
 */
object PolyRegression:

    /** Base hyper-parameter specification for `PolyRegression`
     *  See `polyRegressionTest` for rationale of picking Cholesky
     */
    val hp = new HyperParameter; hp += ("factorization", "Fac_Cholesky", "Fac_Cholesky")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `PolyRegression` object from a combined data-response matrix.
     *  @param xy      the initial combined data-response matrix (before polynomial term expansion)
     *  @param ord     the order (k) of the polynomial (max degree)
     *  @param fname_  the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to PolyRegression.hp)
     */
    def apply (xy: MatrixD, ord: Int, fname: Array [String] = null,
               hparam: HyperParameter = PolyRegression.hp): PolyRegression =
        new PolyRegression (xy.not(?, 1), xy(?, 1), ord, fname, hparam)
    end apply 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `PolyRegression` object from a combined data-response matrix.
     *  @param t       the initial data/input vector: t_i expands to x_i = [1, t_i, t_i^2, ... t_i^k]
     *  @param y       the response/ouput vector
     *  @param ord     the order (k) of the polynomial (max degree)
     *  @param fname_  the feature/variable names
     *  @param hparam  the hyper-parameters
     */
    def apply (t: VectorD, y: VectorD, ord: Int, fname: Array [String],
               hparam: HyperParameter): PolyRegression =
        new PolyRegression (MatrixD (t).ᵀ, y, ord, fname, hparam)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `PolyRegression` object from a data matrix and a response vector.
     *  This method provides data rescaling.
     *  @param x       the initial data/input matrix (before polynomial term expansion)
     *  @param y       the response/output m-vector
     *  @param ord     the order (k) of the polynomial (max degree)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to PolyRegression.hp)
     */
    def rescale (x: MatrixD, y: VectorD, ord: Int, fname: Array [String] = null,
                 hparam: HyperParameter = PolyRegression.hp): PolyRegression =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new PolyRegression (xn, y, ord, fname, hparam)
    end rescale

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The number of terms/parameters in the model (assumes `Regression` with intercept).
     *  @param k  the number of features/predictor variables (not counting intercept)
     */
    def numTerms (k: Int): Int = k + 1

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a 1-vector/point v, compute the values for all of its polynomial
     *  forms/terms, returning them as a vector.
     *  @param v   the 1-vector (e.g., i-th row of t) for creating forms/terms
     *  @param nt  the number of terms
     */
    def forms (v: VectorD, nt: Int): VectorD =
        val t = v(0)
        VectorD (for j <- 0 until nt yield t~^j)
    end forms

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create all forms/terms for each row/point placing them in a new matrix.
     *  @param x    the original un-expanded input/data matrix
     *  @param ord  the order (max degree) of the polynomial
     */
    def allForms (x: MatrixD, ord: Int): MatrixD =
        val k  = 1
        val nt = numTerms (ord)
        println (s"allForms: create expanded data matrix with nt = $nt columns from k = $k columns")
        val xe = new MatrixD (x.dim, nt)
        for i <- x.indices do xe(i) = forms (x(i), nt)        // vector with values for all forms/terms
        xe                                                    // expanded matrix
    end allForms

end PolyRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `polyRegressionTest` main function tests `PolyRegression` class using the following
 *  regression equation.
 *      y  =  b dot x  =  b_0 + b_1*t + b_2*t^2 + ... b_k*t_k
 *  Note, the order at which R-Squared drops is QR(7), Cholesky(14), SVD(6), Inverse(13).
 *  > runMain scalation.modeling.polyRegressionTest
 */
@main def polyRegressionTest (): Unit =

    import scalation.random.Normal

    val noise = Normal (0.0, 100.0)
    val t     = VectorD.range (0, 100)
    val y     = new VectorD (t.dim)
    for i <- 0 until 100 do y(i) = 10.0 - 10.0 * i + i~^2 + i * noise.gen

    println (s"t = $t")
    println (s"y = $y")

    val order = 4
    val mod   = PolyRegression (t, y, order, null, PolyRegression.hp)
    mod.trainNtest ()()                                            // train and test the model

    banner ("test for collinearity")
    println ("corr = " + mod.getX.corr)
    println ("vif  = " + mod.vif ())

    banner ("test predictions")
    val yp = t.map (mod.predict (_))
    println (s" y = $y \n yp = $yp")
    new Plot (t, y, yp, "PolyRegression")

    val z = 10.5                                                   // predict y for one point
    val yp2 = mod.predict (z)
    println (s"predict ($z) = $yp2")

    banner ("test cross-validation")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end polyRegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `polyRegressionTest2` main function tests `PolyRegression` class using the following
 *  regression equation.
 *      y  =  b dot x  =  b_0 + b_1*t + b_2*t^2 + ... b_k*t_k
 *  > runMain scalation.modeling.polyRegressionTest2
 */
@main def polyRegressionTest2 (): Unit =

    import scalation.random.Normal

    val noise = Normal (0.0, 100.0)
    val t     = VectorD.range (0, 100)
    val y     = new VectorD (t.dim)
    for i <- 0 until 100 do y(i) = 10.0 - 10.0 * i + i~^2 + i * noise.gen

    println (s"t = $t")
    println (s"y = $y")

    val order = 6
    val mod   = PolyRegression (t, y, order, null, PolyRegression.hp)
    mod.trainNtest ()()                                            // train and test the model

    banner ("test for collinearity")
    println ("corr = " + mod.getX.corr)
    println ("vif  = " + mod.vif ())

    banner ("test predictions")
    val yp = t.map (mod.predict (_))
    println (s" y = $y \n yp = $yp")
    new Plot (t, y, yp, "PolyRegression")

    val z = 10.5                                    // predict y for one point
    val yp2 = mod.predict (z)
    println (s"predict ($z) = $yp2")

    banner ("test cross-validation")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end polyRegressionTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `polyRegressionTest3` main function tests the collinearity/correlation of
 *  x1, x1^2, x1^3 under centering, standardizing, and min-max normalization,
 *  Picking an min-max interval like [-2, 2] or [-3, 3] may make the effect somewhat
 *  similar to standardization.
 *  > runMain scalation.modeling.polyRegressionTest3
 */
@main def polyRegressionTest3 (): Unit =

    import scalation.mathstat.VectorDOps._

    val x1 = VectorD (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val x  = MatrixD (x1, x1~^2, x1~^3).ᵀ
    banner (s"original: x = $x")
    println (s"original: x.corr = ${x.corr}")

    var z1 = x1 - x1.mean
    var z  = MatrixD (z1, z1~^2, z1~^3).ᵀ
    banner (s"centered: z = $z")
    println (s"centered: z.corr = ${z.corr}")

    z1 = (x1 - x1.mean) / x1.stdev
    z  = MatrixD (z1, z1~^2, z1~^3).ᵀ
    banner (s"standardized: z = $z")
    println (s"standardized: z.corr = ${z.corr}")

    banner ("minmax [a, b]: z = z")
    for a <- -2 to 2; b <- a+1 to a+4 do
        z1 = (b - a).toDouble * (x1 - x1.min) / (x1.max - x1.min) + a
        z  = MatrixD (z1, z1~^2, z1~^3).ᵀ
//      banner (s"minmax [$a, $b]: z = $z")
        println (s"minmax [$a, $b]: z.corr = ${z.corr}")

end polyRegressionTest3

