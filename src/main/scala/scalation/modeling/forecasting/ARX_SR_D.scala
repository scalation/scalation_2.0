
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
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARX_SR_D (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String],
                tRng: Range = null, hparam: HyperParameter = hp,
                bakcast: Boolean = false,
                tForms: TransformMap = Map ("tForm_y" -> null))
      extends ARX_D (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms):

    private val debug = debugf ("ARX_SR_D", true)                          // debug function

    _modelName = s"ARX_SR_D_${p}_${q}_$n_exo"

    debug ("init", s"$modelName with with $n_exo exogenous variables and additional term spec = $spec")
    debug ("init", s"[ x | y ] = ${x ++^ y}")

end ARX_SR_D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_SR_D` companion object provides factory methods for the `ARX_SR_D` class.
 */
object ARX_SR_D extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_SR_D` object by building an input matrix xy and then calling the
     *  `ARX_SR_D` constructor.
     *  Caveat: only the first set of transformations is applied for `fExo_enab`
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
               tRng: Range = null, hparam: HyperParameter = hp,
               fEndo_enab: LSET [TransformT] = LSET (Pow),
               fExo_enab: Array [LSET [TransformT]] = Array (LSET (Pow), LSET (Pow)),
               bakcast: Boolean = false): ARX_SR_D =

        val (n_fExo, n_xe) = (fExo_enab.length, xe.dim2)
        require (n_fExo == n_xe, s"Length of fExo_enab $n_fExo must = number of exogenous variables $n_xe")

        val (fEndo, fExo) = ARX_SR.getTransforms (fEndo_enab, fExo_enab)

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val fEndo_size = fEndo_enab.size
        val fExo_sizeArr: Array [Int] = fExo_enab.map (_.size)

        val tForms = Map ("tForm_y" -> null, "fEndo" -> fEndo)
        val xy     = ARX_SR.buildMatrix (xe_bfil, y, hparam, fEndo, fExo, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam, fEndo_size, fExo_sizeArr) else fname_
        val yy     = makeMatrix4Y (y, hh, bakcast)
        new ARX_SR_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end apply 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_SR_D` object by building an input matrix xy and then calling the
     *  `ARX_SR_D` constructor, with rescaling of endogneous and exogenous variable values.
     *  Caveat: only the first set of transformations is applied for `fExo_enab`
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
                 tRng: Range = null, hparam: HyperParameter = hp,
                 fEndo_enab: LSET [TransformT] = LSET (Pow),
                 fExo_enab: Array [LSET [TransformT]] = Array (LSET (Pow), LSET (Pow)),
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARX_SR_D =

        require (fExo_enab.length == xe.dim2, s"Length of fExo_enab must be the same as the number of exogenous variables")

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                       // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))     // backfill each exogenous variable
            if tFormScale != null then
                val tForm_exo = tFormScale (xe_bfil(0 until tr_size))
                xe_bfil       = tForm_exo.f (xe_bfil)                       // rescale the backfilled exogenous variable

        val fEndo_size = fEndo_enab.size
        val fExo_sizeArr: Array [Int] = fExo_enab.map (_.size)

        val (fEndo, fExo) = ARX_SR.getTransforms (fEndo_enab, fExo_enab)
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

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SR_DTest3` main function tests the `ARX_SR_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_SR_DTest3
 */
@main def aRX_SR_DTest3 (): Unit =

//  val exo_vars  = NO_EXO
    val exo_vars  = Array ("icu_patients")
//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
//  hp("cross") = 1                                                     // 1 => add cross terms
    Transform.hp("p") = 1.5                                             // the power to use in Pow

    val fEn = LSET (Pow)
    val fEx = Array (LSET (Pow), LSET (Pow))

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // endo lags
        hp("q")    = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5
        val mod = ARX_SR_D (xe, y, hh, fEndo_enab = fEn, fExo_enab = fEx)   // create model for time series data
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

    val exo_vars  = Array ("icu_patients")
//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
//  hp("cross") = 1                                                     // 1 => add cross terms
    Transform.hp("p") = 1.5                                             // the power to use in Pow

    val fEn = LSET (Pow)
    val fEx = Array (LSET (Pow))

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // endo lags
        hp("q")    = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5
        val mod = ARX_SR_D (xe, y, hh, fEndo_enab = fEn, fExo_enab = fEx)   // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest_x ()()                                           // use customized trainNtest_x
        println (mod.summary ())                                        // statistical summary of fit

        mod.setSkip (0)
        mod.rollValidate ()                                             // TnT with Rolling Validation
        println (s"After Roll TnT Forecast Matrix yf = ${mod.getYf}")
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))   // only diagnose on the testing set
//      println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRX_SR_DTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SR_DTest5` main function tests the `ARX_SR_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs feature selection.
 *  > runMain scalation.modeling.forecasting.aRX_SR_DTest5
 */
@main def aRX_SR_DTest5 (): Unit =

    val exo_vars  = Array ("icu_patients")
//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    val p  = 6
    val q  = 6
    Transform.hp("p") = 1.5                                             // the power to use in Pow
    RidgeRegression.hp("lambda") = 1.0                                  // regularization/shrinkage parameter
    hp("p")     = p                                                     // endo lags
    hp("q")     = q                                                     // exo lags
    hp("spec")  = 5                                                     // trend specification: 0, 1, 2, 3, 5
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
//  hp("cross") = 1                                                     // 1 => add cross terms

    val fEn = LSET (Pow)                                                // functions to apply to endo lags
    val fEx = Array (LSET (Pow))                                        // functions to apply to exo lags

    val mod = ARX_SR_D (xe, y, hh, fEndo_enab = fEn, fExo_enab = fEx)   // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset
    println (mod.summary ())                                            // statistical summary of fit

    mod.setSkip(0)
    mod.rollValidate ()                                                 // TnT with Rolling Validation
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng(y.dim))

    banner ("Feature Selection Technique: Stepwise")
    val (cols, rSq, modForc) = mod.featureSelectAtHorizon (h = 1, fsType = SelectionTech.Backward) //, cross = "many")
    val k = cols.size
    println (s"k = $k")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for ${modForc.modelName}", lines = true)
    println (s"rSq = $rSq")

end aRX_SR_DTest5

