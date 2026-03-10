
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Jan 11 14:15:07 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Framework:  Build matrices for time series regression models
 *
 *  @see `scalation.modeling.Regression`
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer, LinkedHashSet => LSET}
import scala.math.{cos, sin}

import scalation.mathstat._

import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TransformMap` type and its extension methods provides maps of transforms.
 */
type TransformMap = Map [String, Transform | Array [Transform]]

extension (tr: Transform | Array [Transform])

    def apply (i: Int = -1): Transform =
        tr match
            case t: Transform => t
            case tArr: Array [Transform] => tArr(i)

    def length: Int =
        tr match
            case _ : Transform => 1
            case tArr: Array [Transform] => tArr.length

    def f(x: VectorD): VectorD = apply ().f(x)
    def fi(x: VectorD): VectorD = apply ().fi(x)

    def f(x: MatrixD): MatrixD = apply ().f(x)
    def fi(x: MatrixD): MatrixD = apply ().fi(x)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MakeMatrix4TSY` trait provides factory method templates for for invoking
 * `ARY*` constructors.
 */
trait MakeMatrix4TSY:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY*` object by making/building an input matrix x and then calling the
     *  `ARY*` constructor.
     *  @param y        the response vector (time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = MakeMatrix4TS.hp,
               bakcast: Boolean = false): Forecaster_Reg | Forecaster_D

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARY*` object by building an input matrix xy and then calling the
     *  `ARY*` constructor.  Also rescale the input data.
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormT   the transform for rescaling endogenous and exogenous
     */
    def rescale (y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = MakeMatrix4TS.hp,
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): Forecaster_Reg | Forecaster_D

end MakeMatrix4TSY


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MakeMatrix4TS` trait provides factory method templates for for invoking
 * `ARX*` constructors.
 */
trait MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX*` object by building an input matrix xy and then calling the
     *  `ARX*` constructor.
     *  @param xe          the matrix of exogenous variable values
     *  @param y           the endogenous/response vector (main time series data)
     *  @param hh          the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_      the feature/variable names
     *  @param tRng        the time range, if relevant (time index may suffice)
     *  @param hparam      the hyper-parameters
     *  @param fEndo_enab  the set of transforms to be used for the endogenous
     *  @param fExo_enab   the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast     whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = MakeMatrix4TS.hp,
               fEndo_enab: LSET [TransformT] = null,
               fExo_enab: Array [LSET [TransformT]] = null,
               bakcast: Boolean = false): Forecaster_Reg | Forecaster_D

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX*` object by building an input matrix xy and then calling the
     *  `ARX*` constructor.  Rescale the input data.
     *  @param xe          the matrix of exogenous variable values
     *  @param y           the endogenous/response vector (main time series data)
     *  @param hh          the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_      the feature/variable names
     *  @param tRng        the time range, if relevant (time index may suffice)
     *  @param hparam      the hyper-parameters
     *  @param fEndo_enab  the set of transforms to be used for the endogenous
     *  @param fExo_enab   the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast     whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormT      the transform for rescaling endogenous and exogenous
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = MakeMatrix4TS.hp,
                 fEndo_enab: LSET [TransformT] = null,
                 fExo_enab: Array [LSET [TransformT]] = null,
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): Forecaster_Reg | Forecaster_D

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables
     *  @param n_fExArr  the number of functions used to map exogenous variables
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int = 0, n_fExArr: Array [Int] = null): Array [String]

end MakeMatrix4TS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MakeMatrix4TS` object provides methods for making/building matrices from
 *  lagged endogenous and exogenous variables.
 */
object MakeMatrix4TS:

    /** Base hyper-parameter specification for regression based time series models.
     */
    val hp = new HyperParameter
    hp += ("p",  3, 3)                              // number of lags for the endogenous variable
    hp += ("sp", 7, 7)                              // the seasonal period (time units between repetitive behavior)
    hp += ("ps", 2, 2)                              // number of seasonal lags for the endogenous variable
    hp += ("q",  2, 2)                              // number of lags for the exogenous variables
    hp += ("spec", 1, 1)                            // trend terms: 0 - none, 1 - constant, 2 - linear, 3 - quadratic,
                                                    //              4 - sine, 5 - cosine
    hp += ("lwave", 7, 7)                           // wavelength for sine/cosine (distance between peaks)
    hp += ("cross", 0, 0)                           // 0 - no ENDO-EXO cross terms, 1 - include ENDO-EXO cross terms

    hp += ("nneg", 1, 1)                            // 0 - unrestricted, 1 - predictions must be non-negative
    hp += ("factorization", "Fac_QR", "Fac_QR")     // type of matrix factorization

    /*-------------------------------------------------------------------------------
    @see `scalation.mathstat.Transform` for Transform.hp("p") giving pow, e.g., x^pow
    Usage:
    ARY:       p, spec, lwave                          lags of endogenous variable
    ARY_Quad:  p, spec, lwave, pow                     lags of endogenous variable with power/quadratic terms
    ARX:       p, q, spec, lwave                       lags of endogenous variable and exogenous variable
    ARX_Quad:  p, q, spec, lwave, pow                  lags of endogenous variable with power/quadratic terms and exogenous variable
    ARX_SR:    p, q, spec, lwave, multiple transforms  supports transforms for all variables
    for DIRECT forecasting use ARY_D, ARY_Quad_D, ARX_D, ARX_Quad_D, ARX_SR_D
    -------------------------------------------------------------------------------*/

    private val flaw  = flawf ("MakeMatrix4TS")                            // flaw function
    val trend         = Array ("const", "lin", "quad", "sin", "cos")       // types of trends

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.  Handles all
     *  `*ARY*` models.  The `*ARX*` models require custom `formNames` methods.
     *  @param spec   the number of trend terms
     *  @param p      the number of lags for the endogenous variable (lags 1 to p)
     *  @param pow    the power to raise the lagged endogenous values to (defaults to 0.0)
     *  @param sp     the seasonal period (time units until repetitive behavior)
     *  @param start  the first seasonal lag to use (not subsumed by regular lags)
     *  @param ps     the number of seasonal lags for the endogenous variable (lags 1 to ps)
     */
    def formNames (spec: Int, p: Int, pow: Double = 0.0,
                   sp: Int = -1, start: Int = 1, ps: Int = 0): Array [String] =
        val names = ArrayBuffer [String] ()
        for j <- 0 until spec do names += s"${trend(j)}"                   // trend terms
        for j <- ps to start by -1 do names += s"yl${j*sp}"                // seasonal lags terms
        for j <- p to 1 by -1 do names += s"yl$j"                          // regular lags terms
        if pow != 0.0 then for j <- p to 1 by -1 do names += s"yl$j^$pow"  // power lags terms
        println (s"MakeMatrix4TS.formNames: names = $names")
        names.toArray
    end formNames

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the relevant given values, i.e., the actual time series values, or the
     *  same prepended with one backcasted value.
     *  @see `WeightedMovingAverage`
     *  @param y        the given output/response vector
     *  @param bakcast  whether a backcasted value is prepended to the time series
     */
    private inline def getYb (y: VectorD, bakcast: Boolean): VectorD =
        if bakcast then WeightedMovingAverage.backcast (y) +: y            // y prepended with one backcast
        else y
    end getYb

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a response vector (time series) y, make/build and return an input/predictor
     *  MATRIX x for the TREND terms.
     *  @param y        the given output/response vector
     *  @param spec     the number of trend terms (added columns)
     *                      0 - none, 1 - constant 2 - linear, 3 - quadratic, 4 - sine, 5 - cosine
     *  @param lwave    the wavelength (distance between peaks)
     *  @param bakcast  whether a backcasted value is prepended to the time series
     */
    def makeMatrix4T (y: VectorD, spec: Int, lwave: Double, bakcast: Boolean = false): MatrixD =
        if bakcast then println ("makeMatrix4T: FIX: backcasted value not accounted for in any these methods")
        val m    = y.dim
        val m2   = m / 2.0
        val w    = _2Pi / lwave                                            // 2 Pi over wavelength
        val x    = new MatrixD (m, spec)
        val t_0m = VectorD.range (0, m)                                    // vector 0, 1, ..., m-1
    
        if spec >= 1 then x(?, 0) = VectorD.one (m)                        // intercept/constant term
        if spec >= 2 then x(?, 1) = t_0m / m                               // time trend (linear)
        if spec >= 3 then x(?, 2) = (t_0m-m2)~^2 / m2~^2                   // time-squared trend (quadratic)
        if spec >= 4 then
            x(?, 3) = t_0m.map ((t: Double) => sin (t * w))                // sine wave trend
        if spec == 5 then
            x(?, 4) = t_0m.map ((t: Double) => cos (t * w))                // cosine wave trend
        x
    end makeMatrix4T

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a response vector (time series) y, make/build and return an input/predictor
     *  MATRIX x for the linear LAG terms.
     *  @param y        the given output/response vector
     *  @param p        the maximum lag included (inclusive)
     *  @param bakcast  whether a backcasted value is prepended to the time series
     */
    def makeMatrix4L (y: VectorD, p: Int, bakcast: Boolean = false): MatrixD =
        val x  = new MatrixD (y.dim, p)
        val yb = getYb (y, bakcast)
        for t <- x.indices; j <- 1 to p do
            x(t, p - j) = yb(max0 (t - j))
        x
    end makeMatrix4L

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a response vector (time series) y, make/build and return an input/predictor
     *  MATRIX x for the linear LAG terms.
     *  @param y        the given output/response vector
     *  @param p        the maximum lag included (inclusive)
     *  @param bakcast  whether a backcasted value is prepended to the time series
     */
    def makeMatrix4L (xy: MatrixD, p: Int, bakcast: Boolean): MatrixD =
        var xx = makeMatrix4L (xy(?, 0), p, bakcast)
        for j <- 1 until xy.dim2 do
            xx = xx ++^ makeMatrix4L (xy(?, j), p, bakcast)
        xx
    end makeMatrix4L

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a response vector (time series) y, make/build and return an input/predictor
     *  MATRIX x for the SEASONALLY lagged terms.
     *  @param y        the given output/response vector
     *  @param p        the maximum lag included (inclusive)
     *  @param sp       the seasonal period (time units until repetitive behavior)
     *  @param ps       the number of seasonal lags
     *  @param bakcast  whether a backcasted value is prepended to the time series
     */
    def makeMatrix4S (y: VectorD, p: Int, sp: Int, ps: Int, bakcast: Boolean = false): MatrixD =
        val n = if sp > p then ps else ps - 1                              // number of seasonal lags to use
        if n <= 0 || 2 * sp <= p then
            flaw ("makeMatrix4S", "seasonality subsumed in regular lags, so it's ignored")
            return null
        val x  = new MatrixD (y.dim, n)
        val yb = getYb (y, bakcast)

        val start = if n < ps then 2 else 1 
        for t <- x.indices; j <- start to ps do
            val js = j * sp
            x(t, ps - j) = yb(max0 (t + 1 - js))
        x
    end makeMatrix4S

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given a response vector (time series) y, make/build and return an output/response
     *  MATRIX yy for each horizon to be forecasted (needed for DIRECT forecasting).
     *  @param y        the given output/response vector
     *  @param hh       the maximum forecasting horizon (h = 1 .. hh)
     *  @param bakcast  whether a backcasted value is prepended to the time series
     */
    def makeMatrix4Y (y: VectorD, hh: Int, bakcast: Boolean = false): MatrixD =
        val yb = getYb (y, bakcast)
        val m  = y.dim
        val yy = new MatrixD (m, hh)                                       // yy = [ y_h ] for h = 1 to hh
        for t <- yy.indices; h <- yy.indices2 do
            yy(t, h) = if t + h >= m then NO_DOUBLE else yb(t + h)         // yy -> actual and horizons
        yy
    end makeMatrix4Y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make/build a part of the input matrix consisting of the q * xe.dim2 columns
     *  for the exogenous variables.
     *  @param xe       the matrix of exogenous variable values
     *  @param q        the number of lags for each exogenous variable (lags 1 to q)
     *  @param qp       the power to raise the exogensous lags to
     *  @param bakcast  whether a backcasted value is prepended to the time series
     */
    def makeMatrix4EXO (xe: MatrixD, q: Int, qp: Double, bakcast: Boolean = false): MatrixD =
        var xx: MatrixD = makeMatrix_exo_col (xe(?, 0), q, qp, bakcast)
        for j <- 1 until xe.dim2 do
            xx = xx ++^ makeMatrix_exo_col (xe(?, j), q, qp, bakcast)
        xx
    end makeMatrix4EXO

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the j-th exogenous variable vector xe_j, build and return an input/predictor
     *  MATRIX xx.
     *  The first lag responses can't be predicted due to missing past values.
     *  Therefore the number of rows in xx is reduced to xe.dim - 1.
     *  @param xej      the j-th exogenous variable vector
     *  @param q        the maximum lag included (inclusive) for the exogenous variable (1 to q)
     *  @param qp       the power to raise the lags to
     *  @param bakcast  whether a backcasted value is prepended to the time series
     *  FIX: apply bakcast if true
     */
    private def makeMatrix_exo_col (xej: VectorD, q: Int, qp: Double, bakcast: Boolean): MatrixD =
        if bakcast then
           println ("FIX handle backcast")
        val xe_j = backfill (xej)
        val xx   = new MatrixD (xe_j.dim, q)
        for i <- xx.indices; k <- xx.indices2 do
            xx(i, q - 1 - k) = xe_j(max0(i - 1 - k)) ~^ qp
        xx
    end makeMatrix_exo_col

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backfill the zero prefix of exogenous variable j (xej) by backcasting.  The zero
     *  prefix will be at least of size 1 as 0.0 is initially prepended.
     *  @param xej  the j-th exogenous variable vector
     */
    def backfill (xej: VectorD): VectorD =
        val xe_j = 0.0 +: xej                                              // prepend with zero
        val ii = xe_j.indexWhere (_ != 0.0)                                // find the first non-zero value
//      println (s"backfill: from index ii = $ii")
        for i <- ii-1 to 0 by -1 do                                        // replace zero prefix with backcasted values
            xe_j(i) = WeightedMovingAverage.backcast (xe_j, i)             // backcast from index i
        xe_j(1 until xe_j.dim)
    end backfill

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Backfill the zero prefix of the exogenous variables j (xe) by backcasting.  The zero
     *  prefix will be at least of size 1 as 0.0 is initially prepended.
     *  @param xe  the matrix of exogenous variables
     */
    def backfill (xe: MatrixD): MatrixD =
        val xe_bfill = new MatrixD (xe.dim, xe.dim2)
        for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))
        xe_bfill
    end backfill

end MakeMatrix4TS

