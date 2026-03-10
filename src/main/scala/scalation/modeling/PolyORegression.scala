
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Feb 20 17:39:57 EST 2013
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Orthogonal Polynomial Regression (for one variable)
 *
 *  @see en.wikipedia.org/wiki/Legendre_polynomials
 */

package scalation
package modeling

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PolyORegression` class supports orthogonal polynomial regression.
 *  In this case, 't' is expanded to an orthononalization of '[1, t, t^2 ... t^k]'.
 *  Fit the parameter vector 'b' in the regression equation
 *  <p>
 *      y  =  b dot x + e  =  b_0 + b_1 * t +  b_2 * t^2 ... b_k * t^k + e
 *  <p>
 *  where 'e' represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve for the parameter vector 'b'
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
class PolyORegression (t: MatrixD, y: VectorD, ord: Int, fname_ : Array [String] = null,
                       hparam: HyperParameter = PolyRegression.hp)
      extends Regression (PolyORegression.allForms (t, ord), y, fname_, hparam):

    private val debug = debugf ("PolyORegression", false)         // debug function
    private val flaw  = flawf ("PolyORegression")                 // flaw function
    private val nt    = PolyORegression.numTerms (ord)            // number of terms/columns after expansion
    private val a     = PolyORegression.getA                      // get the multipliers for orthogonal polynomials

    _modelName = s"PolyORegression_$ord"

    if t.dim2 != 1 then flaw ("init", "matrix t must have 1 column")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the vector 'z' into a vector of that includes additional terms,
     *  i.e., add polynomial terms.
     *  @param z  the un-expanded vector
     */
    def expand (z: VectorD): VectorD = PolyORegression.forms (z, nt) 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Follow the same transformations used to orthogonalize the data/input matrix 'x',
     *  on vector 'v', so its elements are correctly mapped.
     *  @param v  the vector to be transformed based the orthogonalize procedure
     */
    def orthoVector (v: VectorD): VectorD =
        val u = new VectorD (v.dim)
        u(0)  = v(0)
        for j <- 1 until v.dim do                                 // element to set
            u(j) = v(j)
            for k <- 0 until j do u(j) -= u(k) * a(j, k)          // apply orthogonal multiplier
        end for
        debug ("orthoVector", s"v = $v \nu = $u")
        u
    end orthoVector

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the scalar 'z', expand it and predict the response value.
     *  @param z  the un-expanded scalar
     */
    def predict (z: Double): Double = predict_ex (VectorD (z))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the vector 'z', expand it and predict the response value.
     *  @param z  the un-expanded vector
     */
    def predict_ex (z: VectorD): Double = predict (orthoVector (expand (z)))

end PolyORegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PolyORegression` companion object provides factory methods for creating
 *  orthogonal polynomial regression models and methods for creating functional forms.
 */
object PolyORegression:

    private val debug = debugf ("PolyORegression", false)         // debug flag
    private var a: MatrixD = null                                 // multipliers for orthogonal polynomials

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the multipliers for orthogonal polynomials, matrix 'a'.
     *  FIX - collecting the 'a' matrix this way may fail for parallel processing
     */
    def getA: MatrixD = a

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `PolyORegression` object from a combined data-response matrix.
     *  @param xy      the initial combined data-response matrix (before polynomial term expansion)
     *  @param ord     the order (k) of the polynomial (max degree)
     *  @param fname_  the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to PolyRegression.hp)
     */
    def apply (xy: MatrixD, ord: Int, fname: Array [String] = null,
               hparam: HyperParameter = PolyRegression.hp): PolyORegression =
        new PolyORegression (xy.not(?, 1), xy(?, 1), ord, fname, hparam)
    end apply 

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `PolyORegression` object from a combined data-response matrix.
     *  @param t       the initial data/input vector: t_i expands to x_i = [1, t_i, t_i^2, ... t_i^k]
     *  @param y       the response/ouput vector
     *  @param ord     the order (k) of the polynomial (max degree)
     *  @param fname_  the feature/variable names
     *  @param hparam  the hyper-parameters
     */
    def apply (t: VectorD, y: VectorD, ord: Int, fname: Array [String],
               hparam: HyperParameter): PolyORegression =
        new PolyORegression (MatrixD (t).ᵀ, y, ord, fname, hparam)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `PolyORegression` object from a data matrix and a response vector.
     *  This method provides data rescaling.
     *  @param x       the initial data/input matrix (before polynomial term expansion)
     *  @param y       the response/output m-vector
     *  @param ord     the order (k) of the polynomial (max degree)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to PolyRegression.hp)
     */
    def rescale (x: MatrixD, y: VectorD, ord: Int, fname: Array [String] = null,
                 hparam: HyperParameter = PolyRegression.hp): PolyORegression =
        val xn = normalize ((x.mean, x.stdev)) (x)
        new PolyORegression (xn, y, ord, fname, hparam)
    end rescale

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The number of terms/parameters in the model (assumes `Regression` with intercept).
     *  @param k  the number of features/predictor variables (not counting intercept)
     */
    def numTerms (k: Int): Int = k + 1

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a 1-vector/point 'v', compute the values for all of its polynomial
     *  forms/terms, returning them as a vector.
     *  @param v   the vector/point (i-th row of t) for creating forms/terms
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
        for i <- x.indices do xe(i) = forms (x(i), nt)            // vector with values for all forms/terms
        val za = orthogonalize (xe)
        a = za._2                                                 // save multipliers
        debug ("allForms", s"expanded data matrix za._1 = ${za._1}")
        za._1                                                     // orthogonalized expanded matrix
    end allForms

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Orthogonalize the data/input matrix x using Gram-Schmidt Orthogonalization,
     *  returning the a new orthogonal matrix z and the orthogonalization multipliers a.
     *  This will eliminate the multi-collinearity problem.
     *  @param x  the matrix to orthogonalize
     */
    def orthogonalize (x: MatrixD): (MatrixD, MatrixD) =
        val z = new MatrixD (x.dim, x.dim2)
        val a = new MatrixD (x.dim2, x.dim2)
        z(?, 0) = x(?, 0)
        for j <- 1 until x.dim2 do                                // column to set
            z(?, j) = x(?, j)
            for k <- 0 until j do                                 // make orthogonal to prior columns
                a(j, k) = (z(?, j) dot z(?, k)) / z(?, k).normSq
                z(?, j) = z(?, j) - z(?, k) * a(j, k)
            end for
        end for
        debug ("orthogonalize", s"x = $x \nz = $z")
        (z, a)
    end orthogonalize

end PolyORegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `polyORegressionTest` object tests `PolyORegression` class using the following
 *  regression equation.
 *      y  =  b dot x  =  b_0 + b_1*t + b_2*t^2 + ... b_k*t_k
 *  Note, the 'order' at which R-Squared drops is QR(7), Cholesky(14), SVD(6), Inverse(13).
 *  > runMain scalation.modeling.polyORegressionTest
 */
@main def polyORegressionTest (): Unit =

    import scalation.random.Normal

    val noise = Normal (0.0, 100.0)
    val t     = VectorD.range (0, 100)
    val y     = new VectorD (t.dim)
    for i <- 0 until 100 do y(i) = 10.0 - 10.0 * i + i~^2 + i * noise.gen

    println (s"t = $t")
    println (s"y = $y")

    val order = 4
    val mod   = PolyORegression (t, y, order, null, PolyRegression.hp)
    mod.trainNtest ()()                                            // train and test the model

    banner ("test for collinearity")
    println ("corr = " + mod.getX.corr)
    println ("vif  = " + mod.vif ())

    banner ("test predictions")
    val yp = t.map (mod.predict (_))
    println (s" y = $y \n yp = $yp")
    new Plot (t, y, yp, "PolyORegression")

    val z = 10.5                                                   // predict y for one point
    val yp2 = mod.predict (z)
    println ("predict (" + z + ") = " + yp2)

    banner ("test cross-validation")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end polyORegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `polyORegressionTest2` object tests `PolyORegression` class using the following
 *  regression equation.
 *      y  =  b dot x  =  b_0 + b_1*t + b_2*t^2 + ... b_k*t_k
 *  > runMain scalation.modeling.polyORegressionTest2
 */
@main def polyORegressionTest2 (): Unit =

    import scalation.random.Normal

    val noise = Normal (0.0, 100.0)
    val t     = VectorD.range (0, 100)
    val y     = new VectorD (t.dim)
    for i <- 0 until 100 do y(i) = 10.0 - 10.0 * i + i~^2 + i * noise.gen

    println ("t = " + t)
    println ("y = " + y)

    val order = 6
    val mod   = PolyORegression (t, y, order, null, PolyRegression.hp)
    mod.trainNtest ()()                                            // train and test the model

    banner ("test for collinearity")
    println ("corr = " + mod.getX.corr)
    println ("vif  = " + mod.vif ())

    banner ("test predictions")
    val yp = t.map (mod.predict (_))
    println (s" y = $y \n yp = $yp")
    new Plot (t, y, yp, "PolyORegression")

    val z = 10.5                                                 // predict y for one point
    val yp2 = mod.predict (z)
    println ("predict (" + z + ") = " + yp2)

    banner ("test cross-validation")
    val stats = mod.crossValidate ()
    FitM.showQofStatTable (stats)

end polyORegressionTest2

