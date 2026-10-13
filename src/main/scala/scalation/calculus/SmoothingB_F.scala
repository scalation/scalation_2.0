
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Michael Cotterell, John Miller, Hao Peng, Dong-Yu Yu
 *  @version 1.6
 *  @date    Fri Mar 24 20:24:40 2017
 *  @see     LICENSE (MIT style license file).
 *
 *  @see open.uct.ac.za/bitstream/item/16664/thesis_sci_2015_essomba_rene_franck.pdf?sequence=1
 *  @see www.jstatsoft.org/article/view/v051i04/v51i04.pdf
 *  @see Functional Data Analysis, Second Edition, Chapter 4
 *  @see link.springer.com/book/10.1007%2Fb98888
 */

package scalation
package calculus

import scalation.mathstat._

import DBasisFunction._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SmoothingMethod` enum defines various smoothing methods.
 *  @param name  the name of the smoothing method
 */
enum SmoothingMethod (val name: String):

    case ROUGHNESS extends SmoothingMethod ("ROUGHNESS") 
    case RIDGE     extends SmoothingMethod ("RIDGE") 
    case OLS       extends SmoothingMethod ("OLS") 

end SmoothingMethod

import SmoothingMethod._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SmoothingB_F` class fits a time-dependent data vector 'y' to B-Splines.
 *
 *      y(t(i)) = x(t(i)) + ε(t(i))
 *      x(t) = cΦ(t)
 *
 *  where 'x' is the signal, 'ε' is the noise, 'c' is a coefficient vector and
 *  'Φ(t)' is a vector of basis functions. 
 *  This version just used B-Splines.
 *  @param y          the (raw) data points/vector
 *  @param t          the data time points/vector
 *  @param ord        the order of the basis function (defaults to 4, cubic)
 *  @param lambda     the regularization parameter (>= 0 or -1 to use GCV)
 *  @param method     the smoothing method
 *  @param technique  the factorization technique
 */
class SmoothingB_F (y: VectorD, t: VectorD, ord: Int = 4, lambda: Double = -1,
                    method: SmoothingMethod = ROUGHNESS,
                    technique: String = "Cholesky"):

    private val debug = debugf ("SmoothingB_F", false)           // debug function
    private val flaw  = flawf ("SmoothingB_F")                   // flaw function

    private val m     = y.dim                                    // number of data time points
    private val bf    = new DB_Spline (t, ord)                   // B-Spline basis functions with derivatives

    private var Φ:   MatrixD = null                              // the design matrix
    private var Φt:  MatrixD = null                              // Φ.t (transpose)
    private var ΦtΦ: MatrixD = null                              // Φt * Φ
    private var Σ:   MatrixD = null                              // the penalty matrix, if method == ROUGHNESS
    private var Φty: VectorD = null                              // Φt * y

    private val ns    = bf.size (ord)                            // number of basis functions
    private val I     = MatrixD.eye (ns, ns)                     // identity matrix

    private var λopt  = lambda                                   // regularization parameter
    private var c: VectorD = null                                // coefficient vector
    private var e: VectorD = null                                // residual/error vector
    private var sse   = 0.0                                      // sum of squared error

    def getLambda = λopt

    if y.dim != m then flaw ("init", "require # data points == # data time points")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the partial fit for the model using one of several training 
     *  methods. If you multiply the training data by this `MatrixD`, then you
     *  get the symmetric "hat" matrix. If you multiply this `MatrixD` by the
     *  response vector, then you will get the vector estimated model
     *  coefficients.
     *  @param λ  the regularization parameter
     */
    private def pfit (λ: Double): MatrixD =
        method match
        case ROUGHNESS => (ΦtΦ + (Σ * λ)).inverse * Φt
        case RIDGE     => (ΦtΦ + (I * λ)).inverse * Φt
        case OLS       => ΦtΦ.inverse * Φt
    end pfit

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Calculate the correlation matrix for the basis functions.
     *  @param yy  data vector
     *  @param k   lag parameter for auto-covariance
     */
    def calcCov (yy: VectorD, k: Int = 1): MatrixD =
        val avar = yy.variance
        val cov  = MatrixD.eye (yy.dim, yy.dim) * avar
        val acov = yy.acov (k)
        cfor (0, yy.dim-1) { i =>
            cov(i, i+1) = acov
            cov(i+1, i) = acov
        } // cfor
        cov
    end calcCov
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Computes the "hat" matrix.
     *  @param λ  the regularization parameter
     */
    private def H (λ: Double): MatrixD = Φ * pfit (λ)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the model, i.e., determine the optimal coeifficient 'c' for the
     *  basis functions by finding optimal Lamdba to minimize gcv.
     */
    def train (): VectorD =
        if Φ == null then
            val cachedMx =
                 if method == ROUGHNESS then
                     bf.getCache (ord, t)
                 else                // no penalty matrix if not using roughness penalty
                     bf.asInstanceOf [BasisFunction].getCache (ord, t)
            Φ   = cachedMx (0)
            Φt  = cachedMx (1)
            ΦtΦ = cachedMx (2)
            if method == ROUGHNESS then Σ = cachedMx (3)
            Φty = Φt * y
        end if

        if method != OLS && λopt == -1 then useGCV ()
        updateC (λopt)
        e   = y - Φ * c
        sse = e dot e
        c
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the degrees of freedom based on the trace of hat matrix.
     *  @param λ  the regularization parameter
     */
    private def df (λ: Double): Double = H (λ).trace

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Generalized Cross Validation (GCV) score.
     *  @param λ  the regularization parameter
     */
    private def gcv (λ: Double): Double = (m / (m - df (λ))) * (sse / (m - df (λ)))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Use the Generalized Cross Validation (GCV) method to find the optimal
     *  λ value. It finds the λ that minimizes the GCV score. 
     */
    private def useGCV (): Unit =
        import scalation.optimization.GoldenSectionLS

        def f (λ: Double): Double =
            updateC (λ)
            e   = y - Φ * c
            sse = e dot e
            gcv (λ)
        end f

        val gs   = new GoldenSectionLS (f)
        val step = 1
        λopt     = gs.search (step)
        debug ("useGCV", s"λopt = $λopt")
    end useGCV

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the parameter vector 'c' using various smoothing x factorization
     *  techniques.
     *  @param λ  the regularization parameter
     */
    private def updateC (λ: Double): Unit =
        val mx = method match
            case ROUGHNESS => (ΦtΦ + (Σ * λ))
            case RIDGE     => (ΦtΦ + (I * λ))
            case OLS       => ΦtΦ

        val fac = technique match                                   // select the factorization technique
            case "Fac_Cholesky" => new Fac_Cholesky (mx)            // Cholesky Factorization
            case "Fac_LU"       => new Fac_LU (mx)                  // LU Factorization
            case _              => new Fac_Inverse (mx)             // Inverse Factorization

        fac.factor ()
        c = fac.solve (Φty)
    end updateC

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the y-value at time point 'tt'.
     *  @param tt  the given time point
     */
    def predict (tt: Double): Double =
        var sum = 0.0
        for j <- bf.range (ord) do sum += c(j) * bf (ord)(j)(tt)
        sum
    end predict

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the 1st derivative value at time point 'tt'.
     *  @param tt  the given time point
     */
    def d1predict (tt: Double): Double =
        var sum = 0.0
        for j <- bf.range (ord) do sum += c(j) * bf.d1bf (ord)(j)(tt)
        sum
    end d1predict

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the 2nd derivative value at time point 'tt'.
     *  @param tt  the given time point
     */
    def d2predict (tt: Double): Double =
        var sum = 0.0
        for j <- bf.range (ord) do sum += c(j) * bf.d2bf (ord)(j)(tt)
        sum
    end d2predict

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the n-th derivative value at time point 'tt'.
     *  @param n   the n-th derivative to be computed
     *  @param tt  the given time point
     */
    def dnpredict (n: Int)(tt: Double): Double =
        var sum = 0.0
        for j <- bf.range (ord) do sum += c(j) * bf.dnbf (n)(ord)(j)(tt)
        sum
    end dnpredict

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the y-values at all time points in vector 'tv'.
     *  @param tv  the given vector of time points
     */
    def predict (tv: VectorD): VectorD = tv.map (predict (_))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the 1st derivative values at all time points in vector 'tv'.
     *  @param tv  the given vector of time points
     */
    def d1predict (tv: VectorD): VectorD = tv.map (d1predict (_))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the 2nd derivative values at all time points in vector 'tv'.
     *  @param tv  the given vector of time points
     */
    def d2predict (tv: VectorD): VectorD = tv.map (d2predict (_))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the n-th derivative values at all time points in vector 'tv'.
     *  @param n   the n-th derivative to be computed
     *  @param tv  the given vector of time points
     */
    def dnpredict (n: Int, tv: VectorD): VectorD = tv.map (dnpredict (n)(_))

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the the basis functions
     *  @param tt  the given vector of time points
     */
    def plotBasis (tt: VectorD = t) = plot (bf, ord, tt)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the Basis Function object
     */
    def getBasis = bf

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector of residuals/errors.
     */
    def residual: VectorD = e

end SmoothingB_F


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `smoothingB_FTest` main function is used to test the `SmoothingB_F` class.
 *  > runMain scalation.calculus.smoothingB_FTest
 */
@main def smoothingB_FTest (): Unit =

    import scalation.random.Normal
    import math.pow

    val normal1 = Normal (0, 100.0)                                  // normal random variate generator
    val normal2 = Normal (0, 5000.0)                                 // normal random variate generator
    val normal3 = Normal (0, 100.0)                                  // normal random variate generator

    val t      = VectorD.range (0, 100) / 17.00                      // time points
    val tt     = VectorD.range (0, 1000) / 170.00                    // time points
    val y      = t.map ((x: Double) => pow(x-4, 5) + 5.0 * pow(x-4, 4) - 20.0 * pow(x-4, 2) + 4.0 * (x-4))
    val z      = y.map ((e: Double) => e + (if e < 2 then normal1.gen
                                            else if e < 3 then normal2.gen
                                            else normal3.gen))
    val mMin   = 4 // 2                                              // minimum order to try
    val mMax   = 4                                                   // maximum order to try
    val method = SmoothingMethod.ROUGHNESS                           // smoothing method
    val lambda = -1
//  val gap    = 10
//  val k      = 2

    new Plot (t, y, z, s"TRUE DATA vs. WITH NOISE")

//  val L    = y.max - y.min                                         // period length
//  val w    = 2.0 * Pi / L                                          // fundamental frequency estimate

//  val centers = VectorD (0.0 until 100.0/17 by 0.5)

    cfor (mMin, mMax+1) { ord =>
        val moo = new SmoothingB_F (z, t, ord, lambda = lambda, method = method)     // smoother (use GCV)
        moo.plotBasis (tt)
        moo.train (          )                                       // train -> set coefficients
        val x   = moo.predict (t)                                    // predict for all time points
        new Plot (t, z, x, s"B-Spline Fit: ord = $ord; λ = ${moo.getLambda}")
        new Plot (t, y, x, s"TRUTH vs. SMOOTHED")
    } // cfor

end smoothingB_FTest

