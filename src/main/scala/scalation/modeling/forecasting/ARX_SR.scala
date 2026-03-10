
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Tue Jan 14 15:47:45 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe with SR terms (ARX_SR) using OLS
 *
 *  @see `scalation.modeling.Regression`
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{ArrayBuffer, LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_SR` class provides time series analysis capabilities for ARX Symbolic
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
 *  @param x        the data/input matrix (lagged columns of y and xe) @see `ARX_SR.apply`
 *  @param y        the response/output vector (main time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 *  @param fExo_sz  the array of the number of transformations for each exogenous
 */
class ARX_SR (x: MatrixD, y: VectorD, hh: Int, n_exo: Int, fname: Array [String],
              tRng: Range = null, hparam: HyperParameter = hp,
              bakcast: Boolean = false,
              tForms: TransformMap = Map ("tForm_y" -> null),
              fExo_sz: Array [Int] = null)
      extends ARX (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms):

    private val debug   = debugf ("ARX_SR", true)                            // debug function
    private val cross   = hparam("cross").toInt == 1                         // whether to include ENDO-EXO cross terms

    _modelName = s"ARX_SR_${p}_${q}_$n_exo"

    debug ("init", s"$modelName with with $n_exo exogenous variables and additional term spec = $spec")
//  debug ("init", s"[ x | y ] = ${x :^+ y}")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Forge a new vector from the first spec values of x, the last p-h+1 values
     *  of x (past values) and recent values 1 to h-1 from the forecasts.
     *  @param xx  the t-th row of the input matrix (lagged actual values)
     *  @param yy  the t-th row of the forecast matrix (forecasted future values)
     *  @param h   the forecasting horizon, number of steps ahead to produce forecasts
     */
    override def forge (xx: VectorD, yy: VectorD, h: Int): VectorD =
        // add terms for the endogenous variable
        val n_endo  = spec + p                                               // number of trend + endogenous values
        val x_act   = xx(n_endo-(p+1-h) until n_endo)                        // get actual lagged y-values (endogenous)
        val nyy     = p - x_act.dim                                          // number of forecasted values needed
        val x_fcast = yy(h-nyy until h)                                      // get forecasted y-values

        var xy         = x_act ++ x_fcast                                    // original values before any mapping
//      val x_fEndo    = scaleCorrection (x_fcast)                           // needed if transform first then rescaling
        val tFormsEndo = tForms("fEndo")
        val n_fEndo    = tFormsEndo.length                                   // number of functions used to map endogenous variables

        for j <- 0 until n_fEndo do
            val x_act_f = xx((j+1)*p + n_endo-(p+1-h) until (j+1)*p + n_endo)  // get transformed lagged endogenous variable
            xy = xy ++ x_act_f ++ tFormsEndo(j).f(x_fcast)  

        // add terms for the exogenous variables
        val crs    = if cross then 1 else 0                                  // whether to add endogenous-exogenous cross terms
        val n_fExo = fExo_sz.sum
        val count  = n_exo * (1 + n_fExo + crs)
        for j <- 0 until count do
            xy = xy ++ hide (xx(n_endo+n_fEndo*p + j*q until n_endo+n_fEndo*p + (j+1)*q), h)  // get actual and transformed lagged for exo variable j
        xx(0 until spec) ++ xy
    end forge

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make re-scaling corrections to the forecasted y-values.
     *  @param  x_fcast  the forecasted y-values 
     *
    def scaleCorrection (x_fcast: VectorD): Array [VectorD] =
        val x_fEndo = Array.ofDim [VectorD] (n_fEndo)
        if tForms("tForm_y") != null then
            val f_tForm = Array.ofDim [FunctionV2V] (n_fEndo)

            for i <- 0 until n_fEndo do f_tForm(i) = (tForms("fEndo")(i).f(_: VectorD)) ⚬ (tForms("tForm_y").fi(_: VectorD))

            var x_fcast_fEndo = MatrixD (f_tForm(0)(x_fcast)).ᵀ
            for i <- 1 until n_fEndo do x_fcast_fEndo = x_fcast_fEndo :^+ f_tForm(i)(x_fcast)
            x_fcast_fEndo = tForms("tForm_endo").f(x_fcast_fEndo)
            for i <- 0 until n_fEndo do x_fEndo(i) = x_fcast_fEndo(?, i)
        else
            for i <- 0 until n_fEndo do x_fEndo(i) = tForms("fEndo")(i).f(x_fcast)
        x_fEndo
    end scaleCorrection
     */

end ARX_SR

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_SR` companion object provides factory methods for the `ARX_SR` class.
 */
object ARX_SR extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_SR` object by building an input matrix xy and then calling the
     *  `ARX_SR` constructor.
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
               bakcast: Boolean = false): ARX_SR =

        val (n_fExo, n_xe) = (fExo_enab.length, xe.dim2)
        require (n_fExo == n_xe, s"Length of fExo_enab $n_fExo must = number of exogenous variables $n_xe")

        val (fEndo, fExo) = getTransforms (fEndo_enab, fExo_enab)

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))     // backfill each exogenous variable

        val fEndo_size = fEndo_enab.size
        val fExo_sizeArr: Array [Int] = fExo_enab.map (_.size)

        val tForms = Map ("tForm_y" -> null, "fEndo" -> fEndo)
        val xy     = buildMatrix (xe_bfil, y, hparam, fEndo, fExo, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam, fEndo_size, fExo_sizeArr) else fname_
        new ARX_SR (xy, y, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms, fExo_sizeArr)
    end apply 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_SR` object by building an input matrix xy and then calling the
     *  `ARX_SR` constructor, with rescaling of endogneous and exogenous variable values.
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
                 tFormT: TransformT = MinMax): ARX_SR =

        val (n_fExo, n_xe) = (fExo_enab.length, xe.dim2)
        require (n_fExo == n_xe, s"Length of fExo_enab $n_fExo must = number of exogenous variables $n_xe")

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

        val (fEndo, fExo) = getTransforms (fEndo_enab, fExo_enab)
        val tForms = Map ("tForm_y" -> tForm_y, "fEndo" -> fEndo)
        val xy     = buildMatrix (xe_bfil, y_scl, hparam, fEndo, fExo, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam, fEndo_size, fExo_sizeArr) else fname_
        new ARX_SR (xy, y_scl, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms, fExo_sizeArr)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the spec + p columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  When cross = true, additional cross terms will be added.  Columns produced
     *  by transformations will be added as well.
     *  @param xe_bfil  the matrix of exogenous variable values
     *  @param y        the endogenous/response vector (main time series data)
     *  @param hp_      the hyper-parameters
     *  @param fEndo    the transformation functions to apply on the endogenous variables
     *  @param fExo     the transformation functions to apply on the exogenous variables
     *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
     */
    def buildMatrix (xe_bfil: MatrixD, y: VectorD, hp_ : HyperParameter,
                     fEndo: Array [Transform], fExo: Array [Array [Transform]], bakcast: Boolean): MatrixD =

        val (p, q, spec, lwave, cross) = (hp_("p").toInt, hp_("q").toInt, hp_("spec").toInt, hp_("lwave").toDouble, hp_("cross").toInt == 1)

        // apply transformations fEndo to the endogenous variables and add these columns to x_endo
        var x_endo = MatrixD (y).ᵀ                                         // make a matrix out of vector y
        for tr <- fEndo do x_endo = x_endo :^+ tr.f(y)                     // add each transformation of the endogenous variable

        // make matrix xy for trend terms and lagged terms of the endogenous variable
        var xy = makeMatrix4T (y, spec, lwave, bakcast) ++^                // trend terms
                 makeMatrix4L (x_endo, p, bakcast)                         // lagged linear terms

        // apply transformations fExo to the exogenous variables and add there columns to x_exo
        if xe_bfil != null and q > 0 then
            var x_exo = new MatrixD (xe_bfil.dim, 0)
            for j <- xe_bfil.indices2 do
                val xe_j = xe_bfil(?, j)                                   // extract the (j+1)th exogenouse variable
                x_exo = x_exo :^+ xe_j                                     // add the exogenous variable
                val fExo_j = fExo(j)                                       // extract the transformations for the (j+1)th exogenouse variable
                if fExo_j.length > 0 then
                    for tr <- fExo_j do x_exo = x_exo :^+ tr.f(xe_j)       // add each transformation of the exogenous variable

            // add cross terms of the endogenous and exogenous variables
            if cross then x_exo = x_exo ++^ y *~: xe_bfil                  // element-wise multiplication of vector y and matrix xe
            xy = xy ++^ makeMatrix4L (x_exo, q, bakcast)                   // add lagged exogenous term to xy

        println (s"xy.dims = ${xy.dims}")
        xy
    end buildMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form vectors for the initial weights and their bounds for the transforms.
     *  @param fEndo_enab  the set of transforms to be used for the endogenous
     *  @param fExo_enab   the array containing the sets of transforms to be used for the exogenous
     */
    private def initializeW (fEndo_enab: LSET [TransformT], fExo_enab: Array [LSET [TransformT]]): VectorD =

        var wInit = new VectorD (0)
        // order: endo's transforms, exo1's transforms, exo2's transfroms, ...

        if fEndo_enab != null then
            for t <- fEndo_enab do wInit = wInit ++ t.wlu.w

        if fExo_enab.length > 0 then
            for set <- fExo_enab if set != null do
                for t <- set do wInit = wInit ++ t.wlu.w
        wInit
    end initializeW

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form arrays of transforms object using the vector of transform parameters.
     *  @param fEndo_enab  the set of transforms to be used for the endogenous
     *  @param fExo_enab   the array containing the sets of transforms to be used for the exogenous
     */
    def getTransforms (fEndo_enab: LSET [TransformT], fExo_enab: Array [LSET [TransformT]]):
                      (Array [Transform], Array [Array [Transform]]) =

        val wInit    = initializeW (fEndo_enab, fExo_enab)
        val listEndo = new ArrayBuffer [Transform] ()
        var i = 0
        if fEndo_enab != null then
            for t <- fEndo_enab do
                listEndo += t.form (VectorD (wInit(i), wInit(i+1)))
                i += 2

        val listExo = new Array [Array [Transform]] (fExo_enab.length)
        if fExo_enab.length > 0 then
            var k = 0
            for set <- fExo_enab do
                val listExo_k = new ArrayBuffer [Transform] ()
                if set != null then
                    for t <- set do
                        listExo_k += t.form (VectorD (wInit(i), wInit(i+1)))
                        i += 2
                listExo(k) = listExo_k.toArray
                k += 1

        (listEndo.toArray, listExo)
    end getTransforms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables
     *  @param n_fExArr  the number of functions used to map each exogenous variables
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int, n_fExArr: Array [Int]): Array [String] =

        val (spec, p, q, cross) = (hp_("spec").toInt, hp_("p").toInt, hp_("q").toInt, hp_("cross").toInt)
        val names = ArrayBuffer [String] ()
        for i <- 0 until n_fEn; j <- p to 1 by -1 do names += s"f$i(yl$j)"           // function lags endo terms

        // exogenous (match build order):
        for j <- 0 until n_exo do
            for k <- q to 1 by -1 do names += s"xe${j}l$k"                           // raw exogenous lags

            val n_fEx_j = n_fExArr(j)                                                // transformations for this exo j
            for i <- 0 until n_fEx_j do
                for k <- q to 1 by -1 do names += s"g$j,$i(xe${j}l$k)"
        end for

        if cross == 1 then
            for j <- 0 until n_exo; k <- q to 1 by -1 do names += s"xe${j}l$k*yl$k"  // lagged cross terms

        MakeMatrix4TS.formNames (spec, p) ++ names.toArray
    end formNames

end ARX_SR


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SRTest` main function tests the `ARX_SR` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_SRTest
 *
@main def aRX_SRTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_SR (y, hh)                                            // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end aRX_SRTest
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SRTest2` main function tests the `ARX_SR` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_SRTest2
 *
@main def aRX_SRTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_SR (y, hh)                                            // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.rollValidate ()                                                 // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRX_SRTest2
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SRTest3` main function tests the `ARX_SR` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_SRTest3
 */
@main def aRX_SRTest3 (): Unit =

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
    val fEx = Array (LSET (Pow))

    for p <- 6 to 6; s <- 1 to 1; q <- 6 to 6 do                        // number of lags; trend; number of exo lags
        hp("p")    = p                                                  // endo lags
        hp("q")    = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5

        val mod = ARX_SR (xe, y, hh, fEndo_enab = fEn, fExo_enab = fEx)   // create model for time series data
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit
    end for

end aRX_SRTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SRTest4` main function tests the `ARX_SR` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_SRTest4
 */
@main def aRX_SRTest4 (): Unit =

    val exo_vars  = Array ("icu_patients")
//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    Transform.hp("p") = 1.5                                             // the power to use in Pow
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
//  hp("cross") = 1                                                     // 1 => add cross terms

    val fEn = LSET (Pow)
    val fEx = Array (LSET (Pow))

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // endo lags
        hp("q")    = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5

        val mod = ARX_SR (xe, y, hh, fEndo_enab = fEn, fExo_enab = fEx)   // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest_x ()()                                           // use customized trainNtest_x

        mod.forecastAll ()                                              // forecast h-steps ahead (h = 1 to hh) for all y
        mod.diagnoseAll (mod.getY, mod.getYf)

        banner ("rollValidate")
        mod.setSkip (0)
        mod.rollValidate ()                                             // TnT with Rolling Validation
        println (s"After Roll TnT Forecast Matrix yf = ${mod.getYf}")
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))   // only diagnose on the testing set
//      println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRX_SRTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_SRTest5` main function tests the `ARX_SR` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  This version performs feature selection.
 *  > runMain scalation.modeling.forecasting.aRX_SRTest5
 */
@main def aRX_SRTest5 (): Unit =

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

    val mod = ARX_SR (xe, y, hh, fEndo_enab = fEn, fExo_enab = fEx)     // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on COVID-19 Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset
    println (mod.summary ())                                            // statistical summary of fit

    mod.setSkip(0)
    mod.rollValidate ()                                                 // TnT with Rolling Validation
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng(y.dim))

    banner ("Feature Selection Technique: Stepwise")
    val (cols, rSq) = mod.stepwiseSelAll ()                             // R^2, R^2 bar, sMAPE, R^2 cv
//  val (cols, rSq) = mod.backwardElimAll ()                            // R^2, R^2 bar, sMAPE, R^2 cv
    val k = cols.size
    println (s"k = $k")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for ${mod.modelName}", lines = true)
    println (s"rSq = $rSq")

end aRX_SRTest5

