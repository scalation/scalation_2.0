
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Wed Nov 19 22:58:47 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Auto-Regressive on lagged y and xe with SR terms (NARX_SR_D) using OLS - Direct Forecasting
 *
 *  @see `scalation.modeling.Regression`
 */

package scalation
package modeling
package forecasting
package nonlinear

import scala.runtime.ScalaRunTime.stringOf
import scala.collection.mutable.{ArrayBuffer, LinkedHashSet => LSET}

import scalation.optimization.quasi_newton.{LBFGS_B => OPTIMIZER}
import scalation.modeling.{RidgeRegression => REGRESSION}

import scalation.mathstat._
import TransformT._

import MakeMatrix4TS._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NARX_SR_D` class provides time series analysis capabilities for NARX Symbolic
 *  Regression (SR) models with Direct (D) forecasting.  These models include trend, linear,
 *  power, root, and cross terms for the single endogenous (y) variable and zero or more
 *  exogenous (xe) variables.
 *  Given time series data stored in vector y and matrix xe, its next value y_t = combination
 *  of last p values of y, y^p, y^r and the last q values of each exogenous variable xe_j,
 *  again in linear, power and root forms (as well as ENDO-EXO cross terms).
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t, x_t is a vector of inputs, and e_t is the
 *  residual/error term.
 *  @see `MakeMatrix4TS` for hyper-parameter specifications.
 *  @param x        the data/input matrix (lagged columns of y and xe) @see `NARX_SR_D.apply`
 *  @param y        the response/output vector (main time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 *  @param w_nl     the non-linear parameters
 */
class NARX_SR_D (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String],
                 tRng: Range = null, hparam: HyperParameter = hp,
                 bakcast: Boolean = false,
                 tForms: TransformMap = Map ("tForm_y" -> null), w_nl: VectorD = VectorD (0))
      extends ARX_D (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms):

    private val debug = debugf ("NARX_SR_D", true)                          // debug function

    _modelName = s"NARX_SR_D($p, $q, $n_exo)"

    debug ("init", s"$modelName with with $n_exo exogenous variables and additional term spec = $spec")
//  debug ("init", s"[ x | y ] = ${x ++^ y}")

    def parameter_nl: VectorD = w_nl
    println(s"x.dims = ${x.dims}")

end NARX_SR_D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NARX_SR_D` companion object provides factory methods for the `NARX_SR_D` class.
 */
object NARX_SR_D:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `NARX_SR_D` object by building an input matrix xy and then calling the
     *  `NARX_SR_D` constructor.
     *  @param xe             the matrix of exogenous variable values
     *  @param y              the endogenous/response vector (main time series data)
     *  @param hh             the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_         the feature/variable names
     *  @param tRng           the time range, if relevant (time index may suffice)
     *  @param hparam         the hyper-parameters
     *  @param fEndo_enabled  the set of transforms to be used for the endogenous
     *  @param fExo_enabled   the array containing the sets of transforms to be used for the exogenous 
     *  @param bakcast        whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = hp,
               fEndo_enabled: LSET[TransformT] = LSET(Root), 
               fExo_enabled: Array [LSET [TransformT]] = Array (LSET(Pow), LSET(Pow)),
               bakcast: Boolean = false): NARX_SR_D =

        val (wInit_nl, _, _) = initializeW (fEndo_enabled, fExo_enabled)
        val (fEndo, fExo) = getTransforms(wInit_nl, fEndo_enabled, fExo_enabled)

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfill = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val fEndo_size = fEndo_enabled.size
        val fExo_sizeArr: Array [Int] = fExo_enabled.map (_.size)

        val tForms = Map ("tForm_y" -> null, "fEndo" -> fEndo)   //, "fExo" -> fExo)
        val xy     = buildMatrix (xe_bfill, y, hparam, fEndo, fExo, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam, fEndo_size, fExo_sizeArr) else fname_
        val yy     = makeMatrix4Y (y, hh, bakcast)
        new NARX_SR_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `NARX_SR_D` object by building an input matrix xy and then calling the
     *  `NARX_SR_D` constructor, with rescaling of endogneous and exogenous variable values.
     *  @param xe             the matrix of exogenous variable values
     *  @param y              the endogenous/response vector (main time series data)
     *  @param hh             the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_         the feature/variable names
     *  @param tRng           the time range, if relevant (time index may suffice)
     *  @param hparam         the hyper-parameters
     *  @param fEndo_enabled  the set of transforms to be used for the endogenous
     *  @param fExo_enabled   the array containing the sets of transforms to be used for the exogenous 
     *  @param bakcast        whether a backcasted value is prepended to the time series (defaults to false)
     *  @param tFormScale     the transform for y
     */
    def rescale (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array[String] = null,
                 tRng: Range = null, hparam: HyperParameter = hp,
                 fEndo_enabled: LSET [TransformT] = LSET (Root), 
                 fExo_enabled: Array [LSET [TransformT]] = Array (LSET (Pow), LSET (Pow)),
                 bakcast: Boolean = false,
                 tFormScale: VectorD | MatrixD => Transform = MinMax.form): NARX_SR_D =

        // Rescales using rangeForm, then transforms, then lagging.
        // Uses LBFGS_B optimizer to fit all the linear and nonlinear parameters.

        // rescale y
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                       // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)
        if tForm_y.getClass.getSimpleName == "zForm" then hparam("nneg") = 0

        var xe_bfill: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfill = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfill(?, j) = backfill (xe(?, j))    // backfill each exogenous variable
            if tFormScale != null then
                val tForm_exo = tFormScale (xe_bfill(0 until tr_size))
                xe_bfill      = tForm_exo.f (xe_bfill)                      // rescale the backfilled exogenous variable          

        val fEndo_size = fEndo_enabled.size
        val fExo_sizeArr: Array [Int] = fExo_enabled.map (_.size)
        val w_nl = optimize2 (xe_bfill, y_scl, hparam, fEndo_enabled, fExo_enabled, bakcast)  // set non-linear parameters
        
        val (fEndo, fExo) = getTransforms (w_nl, fEndo_enabled, fExo_enabled)
        val tForms = Map ("tForm_y" -> tForm_y, "fEndo" -> fEndo)//, "fExo" -> fExo)
        val xy     = buildMatrix (xe_bfill, y_scl, hparam, fEndo, fExo, bakcast)
        val fname  = if fname_ == null then formNames (xe.dim2, hparam, fEndo_size, fExo_sizeArr) else fname_
        val yy     = makeMatrix4Y (y_scl, hh, bakcast)
        new NARX_SR_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms, w_nl)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build the input matrix by combining the spec + p columns for the trend and
     *  endogenous variable with the q * xe.dim2 columns for the exogenous variables.
     *  When cross = true, additional cross terms will be added.  Columns produced
     *  by transformations will be added as well.
     *  @param xe_bfill  the matrix of exogenous variable values
     *  @param y         the endogenous/response vector (main time series data)
     *  @param hp_       the hyper-parameters
     *  @param fEndo     the transformation functions to apply on the endogenous variables
     *  @param fExo      the transformation functions to apply on the exogenous variables
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     */
    def buildMatrix (xe_bfill: MatrixD, y: VectorD, hp_ : HyperParameter,
                     fEndo: Array [Transform], fExo: Array [Array [Transform]], bakcast: Boolean): MatrixD =

        val (p, q, spec, lwave, cross) = (hp_("p").toInt, hp_("q").toInt, hp_("spec").toInt, hp_("lwave").toDouble, hp_("cross").toInt == 1)

        // apply transformations fEndo to the endogenous variables and add these columns to x_endo
        var x_endo = MatrixD (y).ᵀ                                         // make a matrix out of vector y
        for tr <- fEndo do x_endo = x_endo :^+ tr.f(y)                     // add each transformation of the endogenous variable

        // make matrix xy for trend terms and lagged terms of the endogenous variable
        var xy = makeMatrix4T (y, spec, lwave, bakcast) ++^                // trend terms
                 makeMatrix4L (x_endo, p, bakcast)                         // lagged linear terms

        // apply transformations fExo to the exogenous variables and add there columns to x_exo
        if xe_bfill!= null then
            var x_exo = new MatrixD (xe_bfill.dim, 0)
            for j <- xe_bfill.indices2 do
                val xe_j = xe_bfill(?, j)                                  // extract the (j+1)th exogenouse variable
                x_exo = x_exo :^+ xe_j                                     // add the exogenous variable
                val fExo_j = fExo(j)                                       // extract the transformations for the (j+1)th exogenouse variable
                for tr <- fExo_j do x_exo = x_exo :^+ tr.f(xe_j)           // add each transformation of the exogenous variable

            // add cross terms of the endogenous and exogenous variables
            if cross then x_exo = x_exo ++^ y *~: xe_bfill                 // element-wise multiplication of vector y and matrix xe
            xy = xy ++^ makeMatrix4L (x_exo, q, bakcast)                   // add lagged exogenous term to xy

        xy                                                                 // return the built matrix xy
    end buildMatrix

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit the nonlinear + linear parameters using LBFGS_B.
     *  @param xe_bfill  the matrix of exogenous variable values
     *  @param y         the endogenous/response vector (main time series data)
     *  @param hp_       the hyper-parameters
     *  @param fEndo     the transformation functions to apply on the endogenous variables
     *  @param fExo      the transformation functions to apply on the exogenous variables
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     */
    def optimize (xe_bfill: MatrixD, y: VectorD, hparam: HyperParameter = hp,
                  fEndo_enabled: LSET [TransformT] = LSET (Root), 
                  fExo_enabled: Array [LSET [TransformT]] = Array (LSET (Pow), LSET (Pow)),
                  bakcast: Boolean = false): VectorD =
        val (p, q, spec, crs) = (hparam("p").toInt, hparam("q").toInt, hparam("spec").toInt, hparam("cross").toInt)

        val fEndo_size = fEndo_enabled.size
        val fExo_size  = fExo_enabled.map (_.size).sum
        val (wInit_nl, l_wInit_nl, u_wInit_nl) = initializeW (fEndo_enabled, fExo_enabled)
        val w_nl_size  = wInit_nl.dim

        val n_exo    = if xe_bfill != null then xe_bfill.dim2 else 0
        val q_exo    = if xe_bfill != null then q else 0
        val w_l_size = spec + p * (1 + fEndo_size) + q_exo * (n_exo + fExo_size + n_exo * crs)
                                                                           // FIX: compare with the initialization of LR  
        val wInit   = wInit_nl   ++ VectorD.fill (w_l_size)(0.1)           // add initial linear weights
        val l_wInit = l_wInit_nl ++ VectorD.fill (w_l_size)(-10.0)         // add lower bounds of linear weights
        val u_wInit = u_wInit_nl ++ VectorD.fill (w_l_size)(10.0)          // add upper bounds of linear weights
        var count = 0

        def loss: FunctionV2S = (ww: VectorD) =>
            if count % 10000 == 0 then println (s"count = $count, ww = $ww")
            count += 1
            val w_nl = ww(0 until w_nl_size)
            val w_l  = ww(w_nl_size until ww.dim)
            val (fEndo, fExo) = getTransforms (w_nl, fEndo_enabled, fExo_enabled)
            val xy = buildMatrix (xe_bfill, y, hparam, fEndo, fExo, bakcast)
            val yp = xy * w_l
            (y - yp).normSq
        end loss

        // LBFGS_B
        val optimizer = OPTIMIZER (loss, wInit.dim, l_u_ = (l_wInit, u_wInit))
        val (_, ww)   = optimizer.solve (wInit, 0.05)

        // LBFGS
//      val optimizer = OPTIMIZER (loss)
//      val (_, ww)   = optimizer.solve (wInit)

//      println (s"optimize: parameters ww = $ww")
//      println (s"optimize: loss = ${loss (ww)}")
        ww(0 until w_nl_size)
    end optimize 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit the nonlinear + linear parameters using LBFGS_B with VarPro.
     *  @param xe_bfill  the matrix of exogenous variable values
     *  @param y         the endogenous/response vector (main time series data)
     *  @param hp_       the hyper-parameters
     *  @param fEndo     the transformation functions to apply on the endogenous variables
     *  @param fExo      the transformation functions to apply on the exogenous variables
     *  @param bakcast   whether a backcasted value is prepended to the time series (defaults to false)
     */
    def optimize2 (xe_bfill: MatrixD, y: VectorD, hparam: HyperParameter = hp,
                   fEndo_enabled: LSET [TransformT] = LSET(Root), 
                   fExo_enabled: Array [LSET [TransformT]] = Array (LSET (Pow), LSET (Pow)),
                   bakcast: Boolean = false): VectorD =

        val (wInit, l_wInit, u_wInit) = initializeW (fEndo_enabled, fExo_enabled)

        def loss: FunctionV2S = (ww: VectorD) =>
            val (fEndo, fExo) = getTransforms (ww, fEndo_enabled, fExo_enabled)
            val xy  = buildMatrix (xe_bfill, y, hparam, fEndo, fExo, bakcast)
            val reg = new REGRESSION (xy, y)
            reg.train (xy, y)
            val yp = reg.predict(xy)
            (y - yp).normSq
        end loss

        // LBFGS_B
        val optimizer = OPTIMIZER (loss, wInit.dim, l_u_ = (l_wInit, u_wInit))
        val (_, ww)   = optimizer.solve (wInit, 0.05)

//      println (s"optimize: parameters ww = $ww")
//      println (s"optimize: loss = ${loss (ww)}")
        ww
    end optimize2 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form vectors for the initial weights and their bounds for the transforms.
     *  @param fEndo_enabled  the set of transforms to be used for the endogenous
     *  @param fExo_enabled   the array containing the sets of transforms to be used for the exogenous 
     */
    def initializeW (fEndo_enabled: LSET [TransformT], fExo_enabled: Array [LSET [TransformT]]):
                    (VectorD, VectorD, VectorD) =

        var wInit   = new VectorD (0)
        var l_wInit = new VectorD (0)
        var u_wInit = new VectorD (0)
        // order: endo's transforms, exo1's transforms, exo2's transfroms, ...

        if fEndo_enabled != null then
            for t <- fEndo_enabled do
                val init = t.wlu
                wInit    = wInit ++ init.w
                l_wInit  = l_wInit ++ init.l
                u_wInit  = u_wInit ++ init.u

        if fExo_enabled.length > 0 then
            for set <- fExo_enabled do    
                if set != null then
                    for t <- set do
                        val init = t.wlu
                        wInit    = wInit ++ init.w
                        l_wInit  = l_wInit ++ init.l
                        u_wInit  = u_wInit ++ init.u
        (wInit, l_wInit, u_wInit)
    end initializeW

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form arrays of transforms object using the vector of nonlinear parameters.
     *  @param w_nl           the vector of nonlinear parameters
     *  @param fEndo_enabled  the set of transforms to be used for the endogenous
     *  @param fExo_enabled   the array containing the sets of transforms to be used for the exogenous 
     */
    def getTransforms (w_nl: VectorD, fEndo_enabled: LSET [TransformT], fExo_enabled: Array [LSET [TransformT]]):
                      (Array [Transform], Array [Array [Transform]]) =
        val listEndo = new ArrayBuffer [Transform] ()
        var i = 0
        if fEndo_enabled != null then
            for t <- fEndo_enabled do
                listEndo += t.form (VectorD (w_nl(i), w_nl(i+1)))
                i += 2

        val listExo = new Array [Array [Transform]] (fExo_enabled.length)
        if fExo_enabled.length > 0 then
            var k = 0
            for set <- fExo_enabled do
                val listExo_k = new ArrayBuffer [Transform] ()    
                if set != null then
                    for t <- set do
                        listExo_k += t.form (VectorD (w_nl(i), w_nl(i+1)))
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
            // raw exogenous lags
            for k <- q to 1 by -1 do
                names += s"xe${j}l$k"

            // transformations for this exo j
            val n_fEx_j = n_fExArr(j)
            for i <- 0 until n_fEx_j do
                for k <- q to 1 by -1 do
                    names += s"g$j,$i(xe${j}l$k)"
        end for

        if cross == 1 then
            for j <- 0 until n_exo; k <- q to 1 by -1 do names += s"xe${j}l$k*yl$k"  // lagged cross terms

        MakeMatrix4TS.formNames (spec, p) ++ names.toArray
    end formNames

end NARX_SR_D

import Example_Covid._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest3` main function tests the `NARX_SR_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.nARX_SR_DTest3
 */
@main def nARX_SR_DTest3 (): Unit =

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
    hp("cross") = 1                                                     // 1 => add cross terms

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // endo lags
        hp("q")    = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5
        val mod = NARX_SR_D (xe, y, hh)                                  // create model for time series data
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit
    end for

end nARX_SR_DTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest4` main function tests the `NARX_SR_D` class on real data:
 *  Forecasting COVID-19 using Train and Test (TnT).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.nARX_SR_DTest4
 */
@main def nARX_SR_DTest4 (): Unit =

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

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // endo lags
        hp("q")    = q                                                  // exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5

        val mod = NARX_SR_D (xe, y, hh)           // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest_x ()()                                           // use customized trainNtest_x
        println (mod.summary ())                                        // statistical summary of fit

        mod.setSkip (0)
        mod.rollValidate ()                                             // TnT with Rolling Validation
        println (s"After Roll TnT Forecast Matrix yf = ${mod.getYf}")
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))   // only diagnose on the testing set
//      println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for
end nARX_SR_DTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nARX_SR_DTest5` main function fit the linear and non linear 
 *  parameters and tests the `NARX_SR_D` class on real data:
 *  Forecasting COVID-19 using Train and Test (TnT).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.nARX_SR_DTest5
 */
@main def nARX_SR_DTest5 (): Unit =

    val exo_vars  = Array ("icu_patients", "positive_rate")
//  val exo_vars  = Array ("icu_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                    // full
    val xe = xxe(0 until 116)                                       // clip the flat end
//  val y  = yy                                                     // full
    val y  = yy(0 until 116)                                        // clip the flat end
    val hh = 6                                                      // maximum forecasting horizon
    hp("lwave") = 20                                                // wavelength (distance between peaks)
    hp("cross") = 0                                                 // 1 => add cross terms
    RidgeRegression.hp("lambda") = 6.0
    // hp("pow") = 0.5
    RidgeRegression.hp("factorization") = "Fac_Cholesky"

    hp("p")    = 6                                                  // endo lags
    hp("q")    = 4                                                  // exo lags
    hp("spec") = 1                                                  // trend specification: 0, 1, 2, 3, 5

    val fEndo_enabled = LSET (Root)
    val fExo_enabled  = Array (LSET(Root), LSET(Root))              // array of transforms for exogenous (must be exo_vars.length == fExo_enabled.length)

    val mod = NARX_SR_D.rescale (xe, y, hh, fEndo_enabled = fEndo_enabled, fExo_enabled = fExo_enabled) // create model for time series data
//  mod.trainNtest_x ()()
    mod.inSample_Test ()

    println ("rollValidate")
    mod.setSkip (0)
    mod.rollValidate (rc = 2)
    mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))

    val (cols, rSq, modForc) = mod.featureSelectAtHorizon (h = 1, fsType = SelectionTech.Backward) //, cross = "many")
    println (s"cols = ${cols}")
    println (s"rSq = ${rSq}")
    modForc.setSkip (0)
    modForc.rollValidate (rc = 2)
    modForc.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))
    println (stringOf (mod.parameter_nl))

end nARX_SR_DTest5

