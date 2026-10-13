
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Thu Jan 30 21:15:45 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe with SR terms (ARX_SR_D) using OLS - Direct Forecasting
 *
 *  @see `scalation.modeling.Regression`
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_SR_D` class provides time series analysis capabilities for ARX_D Symbolic
 *  Regression (SR) models.  These models include trend, linear, power, root, and cross terms
 *  for the single endogenous (y) variable and zero or more exogenous (xe) variables.
 *  Given time series data stored in vector y and matrix xe, its next value y_t = combination
 *  of last p values of y, y^p, y^r and the last q values of each exogenous variable xe_j,
 *  again in linear, power and root forms (as well as ENDO-EXO cross terms).
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t, x_t is a vector of inputs, and e_t is the
 *  residual/error term.
 *  @see `MakeMatrix4TS` for hyper-parameter specifications.
 *  @param x        the data/input matrix (lagged columns of y and xe) @see `ARX_SR_D.apply`
 *  @param y        the response/output vector (main time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `AR.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARX_SR_D (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String],
                tRng: Range = null, hparam: HyperParameter = AR.hp,
                bakcast: Boolean = false,
                tForms: TransformMap = Map ("tForm_y" -> null))
      extends ARX_D (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms):

    private val debug = debugf ("ARX_SR_D", true)                         // debug function

    _modelName = s"ARX_SR_D_${p}_${q}_$n_exo"

    debug ("init", s"$modelName with $n_exo exogenous variables and additional trend spec = $spec")
    debug ("init", s"[ x | y ] = ${x ++^ y}")

end ARX_SR_D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_SR_D` companion object provides factory methods for the `ARX_SR_D` class.
 */
object ARX_SR_D extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_SR_D` object by building an input matrix xy and then calling the
     *  `ARX_SR_D` constructor.
     *  @caveat:  only the first set of transformations is applied for `fExo`
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous 
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = AR.hp,
               fEndo: LSET [Transform] = LSET (PowForm ()),
               fExo: Array [LSET [Transform]] = Array (),
               bakcast: Boolean = false): ARX_SR_D =

//      val (n_fExo, n_xe) = (fExo.length, xe.dim2)
//      require (n_fExo == n_xe, s"Length of fExo $n_fExo must = number of exogenous variables $n_xe")

//      val (fEndo, fExo) = ARX_SR.getTransforms (fEndo, fExo)

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val fEndo_size = fEndo.size
        val fExo_sizeArr: Array [Int] = fExo.map (_.size)

        val tForms = Map ("tForm_y" -> null, "fEndo" -> fEndo)
        val xy     = ARX_SR.buildMatrix (xe_bfil, y, hparam, fEndo, fExo, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam, fEndo_size, fExo_sizeArr) else fname_
        val yy     = makeMatrix4Y (y, hh, bakcast)
        new ARX_SR_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end apply 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_SR_D` object by building an input matrix xy and then calling the
     *  `ARX_SR_D` constructor, with rescaling of endogneous and exogenous variable values.
     *  @caveat:  only the first set of transformations is applied for `fExo`
     *  @param xe       the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hh       the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_   the feature/variable names
     *  @param tRng     the time range, if relevant (time index may suffice)
     *  @param hparam   the hyper-parameters
     *  @param fEndo    the set of transforms to be used for the endogenous
     *  @param fExo     the array containing the sets of transforms to be used for the exogenous 
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tForm    the transform for rescaling endogenous and exogenous
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
                 tRng: Range = null, hparam: HyperParameter = AR.hp,
                 fEndo: LSET [Transform] = LSET (PowForm ()),
                 fExo: Array [LSET [Transform]] = Array (),
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARX_SR_D =

        require (fExo.length == xe.dim2, s"Length of fExo must be the same as the number of exogenous variables")

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                       // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("px").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))     // backfill each exogenous variable
            if tFormScale != null then
                val tForm_exo = tFormScale (xe_bfil(0 until tr_size))
                xe_bfil       = tForm_exo.f (xe_bfil)                       // rescale the backfilled exogenous variable

        val fEndo_size = fEndo.size
        val fExo_sizeArr: Array [Int] = fExo.map (_.size)

        val tForms = Map ("tForm_y" -> tForm_y, "fEndo" -> fEndo)
        val xy     = ARX_SR.buildMatrix (xe_bfil, y_scl, hparam, fEndo, fExo, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam, fEndo_size, fExo_sizeArr) else fname_
        val yy     = makeMatrix4Y (y_scl, hh, bakcast)
        new ARX_SR_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables
     *  @param n_fExArr  the number of functions used to map exogenous variables
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int, n_fExArr: Array [Int]): Array [String] =
        ARX_SR.formNames (n_exo, hp_, n_fEn, n_fExArr)
    end formNames

end ARX_SR_D

import Example_LakeLevels.y
import AR.hp

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SR_DTest` main function tests the `ARX_SR_D` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_SR_DTest
 */
@main def aRX_SR_DTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_SR_D (EMPTY_EXO, y, hh)                               // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

end aRX_SR_DTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SR_DTest2` main function tests the `ARX_SR_D` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_SR_DTest2
 */
@main def aRX_SR_DTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_SR_D (EMPTY_EXO, y, hh)                               // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.tnT_Test ()                                                     // Train and Test with Rolling Validation

end aRX_SR_DTest2

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SR_DTest3` main function tests the `ARX_SR_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_SR_DTest3
 */
@main def aRX_SR_DTest3 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
//  hp("crx")   = 1                                                     // 1 => add cross terms
    val mx = y.max * 0.2
    Transform.hp("mx") = mx                                             // maximum value in time series
//  Transform.hp("p") = 1.8                                             // the power to use in Pow
    Transform.hp("p") = 0.2                                             // the power to use in Pow
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    val fEn: LSET [Transform] = LSET (PowForm (), DQuadForm ())         // endo Transform set
//  val fEn: LSET [Transform] = LSET (DQuadForm ())                     // endo Transform set
//  val fEn: LSET [Transform] = LSET (PowForm ())                       // endo Transform set
//  val fEx = Array (LSET (Pow.form (y)))                               // exo Transforms sets (one per exo var)

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX_SR_D (xe, y, hh, fEndo = fEn)                     // create model for time series data
//      val mod = ARX_SR_D.rescale (xe, y, hh, fEndo = fEn, tFormT = Log1p)
//      val mod = ARX_SR_D (xe, y, hh, fEndo = fEn, fExo = fEx)         // create model for time series data, with exo
        banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit
    end for

end aRX_SR_DTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SR_DTest4` main function tests the `ARX_SR_D` class on real data:
 *  Forecasting COVID-19 using Train and Test (TnT).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_SR_DTest4
 */
@main def aRX_SR_DTest4 (): Unit =

    val (xe, y) = clip (loadData (Example_Covid.NO_EXO))
//  val (xe, y) = clip (loadData (BEST_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
//  hp("crx") = 1                                                       // 1 => add cross terms
    val mx = y.max * 0.6
    Transform.hp("mx") = mx                                             // maximum value in time series
//  Transform.hp("p") = 1.8                                             // the power to use in Pow
    Transform.hp("p") = 0.2                                             // the power to use in Pow
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter

    val fEn: LSET [Transform] = LSET (PowForm (), DQuadForm ())         // endo Transform set
//  val fEn: LSET [Transform] = LSET (PowForm ())                       // endo Transform set
//  val fEn: LSET [Transform] = LSET (DQuadForm ())                     // endo Transform set
//  val fEn: LSET [Transform] = LSET (DIQuadForm ())                    // endo Transform set
//  val fEx: Array [LSET [Transform]] = Array (LSET (DQuadForm ()))     // exo Transform sets (one per exo var)

    import SelectionTech._
//  val tech = Forward
    val tech = Backward
//  val tech = Beam

    for p <- 1 to 5; q <- 0 to 0; s <- 0 to 0 do
        hp.set (("p", p), ("px", q), ("spec", s))                       // # endo lags, # exo lags, trend specification: 0, 1, 2, 3
        val mod = ARX_SR_D (xe, y, hh, fEndo = fEn)                     // create model for time series data
//      val mod = ARX_SR_D.rescale (xe, y, hh, fEndo = fEn, tFormT = Log1p)
//      val mod = ARX_SR_D (xe, y, hh, fEndo = fEn, fExo = fEx)         // create model for time series data, with exo
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.tnT_Test ()                                                 // Train and Test with Rolling Validation

        banner (s"feature selection for each horizon using $tech")
        val (colsArr, rSqArr) = mod.selectFeaturesPerH (tech, "none", first = 0)
        mod.reportSelection (colsArr, rSqArr, inSample = false)         // k, features, trace per horizon; TnT of the hh models
    end for

end aRX_SR_DTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SR_DTest5` main function tests the `ARX_SR_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs Feature Selection.
 *  > runMain scalation.modeling.forecasting.aRX_SR_DTest5
 */
@main def aRX_SR_DTest5 (): Unit =

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("p")     = 10                                                    // endo lags
    hp("px")    = 5                                                     // exo lags
    hp("spec")  = 2                                                     // trend specification: 0, 1, 2, 3
//  hp("crx")   = 1                                                     // 1 => add cross terms
    Transform.hp("p") = 1.5                                             // the power to use in Pow
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    FeatureSelection.fullset_FS = true                                  // use full dataset for FS

    val fEn = LSET (Pow.form (y))                                       // functions to apply to endo lags
    val fEx = Array (LSET (Pow.form (y)), LSET (Pow.form (y)))          // functions to apply to exo lags

//  val mod = ARX_SR_D (xe, y, hh, fEndo = fEn, fExo = fEx)             // create model for time series data
    val mod = ARX_SR_D.rescale (xe, y, hh, fEndo = fEn, fExo = fEx, tFormT = Log1p)  // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.inSample_Test ()                                                // In-sample Testing
    println (mod.summary ())                                            // statistical summary of fit

    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    mod.diagnoseAll (mod.getY, mod.getYf)                               // QoF for each horizon
//  println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

    import SelectionTech._                                              // one of Forward, Backward, Stepwise, Beam

//  for tech <- values do                                               // try all feature selection techniques
//  for tech <- values if tech == Forward do                            // try a particular one
//  for tech <- values if tech == Backward do                           // try a particular one
//  for tech <- values if tech == Stepwise do                           // try a particular one
    for tech <- values if tech == Beam do                               // try a particular one
        banner (s"Feature Selection Technique: $tech")
        val (colsArr, rSqArr) = mod.selectFeatures (tech, "none", h = 1)   // R^2, R^2 bar, sMAPE, sMAPEC
        mod.reportSelection (colsArr, rSqArr, tnt = false)              // In-ST of the selected multi-horizon model

//      val imp = mod.importance (cols.toArray, rSq)
//      println (s"feature importance imp = $imp")
//      for (c, r) <- imp do println (s"col = $c, \t ${ox_fname(c)}, \t importance = $r")
    end for

end aRX_SR_DTest5

