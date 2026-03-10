
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Mon Mar 31 23:28:32 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Quadratic, Auto-Regressive on lagged y and xe (ARX_Quad_D) using OLS - Direct Forecasting
 */

package scalation
package modeling
package forecasting

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

import MakeMatrix4TS._
import TransformT._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_Quad_D` class provides basic time series analysis capabilities for
 *  ARX_Quad_D models.  ARX_Quad_D models are often used for forecasting.
 *  `ARX_Quad_D` uses DIRECT (as opposed to RECURSIVE) multi-horizon forecasting.
 *  Given time series data stored in vector y, its next value y_t = combination of last p values.
 *
 *      y_t = b dot x_t + e_t
 *
 *  where y_t is the value of y at time t and e_t is the residual/error term.
 *  @param x        the data/input matrix (lagged columns of y) @see `ARX_Quad_D.apply`
 *  @param y        the response/output matrix (column per horizon) (time series data) 
 *  @param hh       the maximum forecasting horizon (h = 1 to hh)
 *  @param n_exo    the number of exogenous variables
 *  @param fname    the feature/variable names
 *  @param tRng     the time range, if relevant (time index may suffice)
 *  @param hparam   the hyper-parameters (defaults to `MakeMatrix4TS.hp`)
 *  @param bakcast  whether a backcasted value is prepended to the time series (defaults to false)
 *  @param tForms   the map of transformations applied
 */
class ARX_Quad_D (x: MatrixD, y: MatrixD, hh: Int, n_exo: Int, fname: Array [String] = null,
                  tRng: Range = null, hparam: HyperParameter = hp,
                  bakcast: Boolean = false,  
                  tForms: TransformMap = Map ("tForm_y" -> null))
      extends ARX_D (x, y, hh, n_exo, fname, tRng, hparam, bakcast, tForms):

    private val debug = debugf ("ARX_Quad_D", true)                     // debug function

    _modelName = s"ARX_Quad_D_${p}_${q}_$n_exo"

    debug ("init", s"$modelName with $n_exo exogenous variables and additional term spec = $spec")
//  debug ("init", s"[ x | y ] = ${x ++^ y}")

end ARX_Quad_D


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ARX_Quad_D` companion object provides factory methods for the `ARX_Quad_D` class.
 */
object ARX_Quad_D extends MakeMatrix4TS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_Quad_D` object by building an input matrix x and then calling the constructor.
     *  @param xe          the matrix of exogenous variable values
     *  @param y           the endogenous/response vector (main time series data)
     *  @param hh          the maximum forecasting horizon (h = 1 to hh)
     *  @param fname_      the feature/variable names
     *  @param tRng        the time range, if relevant (time index may suffice)
     *  @param hparam      the hyper-parameters (defaults for `MakeMatrix4TS.hp`)
     *  @param fEndo_enab  the set of transforms to be used for the endogenous
     *  @param fExo_enab   the array containing the sets of transforms to be used for the exogenous
     *  @param bakcast     whether a backcasted value is prepended to the time series (defaults to false)
     */
    def apply (xe: MatrixD, y: VectorD, hh: Int, fname_ : Array [String] = null,
               tRng: Range = null, hparam: HyperParameter = hp,
               fEndo_enab: LSET [TransformT] = null,
               fExo_enab: Array [LSET [TransformT]] = null,
               bakcast: Boolean = false): ARX_Quad_D =

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))    // backfill each exogenous variable

        val xy    = ARX_Quad.buildMatrix (xe_bfil, y, hparam, bakcast)
        val yy    = makeMatrix4Y (y, hh, bakcast)
        val fname = formNames (xe.dim2, hparam)
        new ARX_Quad_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create an `ARX_Quad_D` object by building an input matrix xy and then calling the
     * `ARX_Quad_D` constructor.  Also rescale the input data.
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
                 fEndo_enab: LSET [TransformT] = null,
                 fExo_enab: Array [LSET [TransformT]] = null,
                 bakcast: Boolean = false,
                 tFormT: TransformT = MinMax): ARX_Quad_D =

        if tFormT.name == "NormForm" then hparam("nneg") = 0

        // rescale y
        val tFormScale = tFormT.form
        val tr_size = Model.trSize (y.dim)
        val tForm_y = tFormScale (y(0 until tr_size))                       // use (mean, std) of training set for both In-sample and TnT
        val y_scl   = tForm_y.f(y)

        var xe_bfil: MatrixD = null
        if xe.dim2 > 0 and hparam("q").toInt > 0 then
            xe_bfil = new MatrixD (xe.dim, xe.dim2)
            for j <- xe.indices2 do xe_bfil(?, j) = backfill (xe(?, j))    // backfill each exogenous variable
            if tFormScale != null then
                val tForm_exo = tFormScale (xe_bfil(0 until tr_size))
                xe_bfil       = tForm_exo.f (xe_bfil)

        val powForm = PowForm (VectorD (0, Transform.hp("p").toDouble))
        val tForms  = Map ("tForm_y" -> tForm_y, "powForm" -> powForm)
        val xy      = ARX_Quad.buildMatrix (xe_bfil, y_scl, hparam, bakcast, powForm)
        val yy      = makeMatrix4Y (y_scl, hh, bakcast)
        val fname   = formNames (xe.dim2, hparam)
        new ARX_Quad_D (xy, yy, hh, xe.dim2, fname, tRng, hparam, bakcast, tForms)
    end rescale

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form an array of names for the features included in the model.
     *  @param n_exo     the number of exogenous variable
     *  @param hp_       the hyper-parameters
     *  @param n_fEn     the number of functions used to map endogenous variables (none for `ARX_Quad_D`)
     *  @param n_fExArr  the number of functions used to map exogenous variables (none for `ARX_Quad_D`)
     */
    def formNames (n_exo: Int, hp_ : HyperParameter, n_fEn: Int = 0, n_fExArr: Array [Int] = null): Array [String] =
        ARX.formNames (n_exo, hp_, n_fEn, n_fExArr)
    end formNames

end ARX_Quad_D

import Example_Covid.{loadData, response}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_DTest` main function tests the `ARX_Quad_D` class on real data:
 *  Forecasting Lake Levels using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_Quad_DTest
 *
@main def aRX_Quad_DTest (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_Quad_D (y, hh)                                             // create model for time series data
    banner (s"In-ST Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.forecastAll ()                                                  // forecast h-steps ahead (h = 1 to hh) for all y
    println (s"Final In-ST Forecast Matrix yf = ${mod.getYf}")

end aRX_Quad_DTest
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_DTest2` main function tests the `ARX_Quad_D` class on real data:
 *  Forecasting Lake Levels using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  @see cran.r-project.org/web/packages/fpp/fpp.pdf
 *  > runMain scalation.modeling.forecasting.aRX_Quad_DTest2
 *
@main def aRX_Quad_DTest2 (): Unit =

    val hh = 3                                                          // maximum forecasting horizon

    val mod = ARX_Quad_D (y, hh)                                             // create model for time series data
    banner (s"TnT Forecasts: ${mod.modelName} on LakeLevels Dataset")
    mod.trainNtest_x ()()                                               // train and test on full dataset

    mod.rollValidate ()                                                 // TnT with Rolling Validation
    println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")

end aRX_Quad_DTest2
 */


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_DTest3` main function tests the `ARX_Quad_D` class on real data:
 *  Forecasting COVID-19 using In-Sample Testing (In-ST).
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_Quad_DTest3
 */
@main def aRX_Quad_DTest3 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val exo_vars  = Array ("icu_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
    Transform.hp("p") = 1.5                                             // power on Pow transform

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1 do                        // number of lags (endo, exo); trend
        hp("p")    = p                                                  // mumber of endo lags
        hp("q")    = q                                                  // mumber of exo lags
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5
//      val mod = ARX_Quad_D (xe, y, hh)                                // create model for time series data
        val mod = ARX_Quad_D.rescale (xe, y, hh)                        // create model for time series data
        mod.inSample_Test ()                                            // In-sample Testing
        println (mod.summary ())                                        // statistical summary of fit  FIX - crashes
    end for

end aRX_Quad_DTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `aRX_Quad_DTest4` main function tests the `ARX_Quad_D` class on real data:
 *  Forecasting COVID-19 using Train-n-Test Split (TnT) with Rolling Validation.
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.aRX_Quad_DTest4
 */
@main def aRX_Quad_DTest4 (): Unit =

//  val exo_vars  = Array ("icu_patients", "hosp_patients", "new_tests", "people_vaccinated")
    val exo_vars  = Array ("icu_patients")
    val (xxe, yy) = loadData (exo_vars, response)
    println (s"xxe.dims = ${xxe.dims}, yy.dim = ${yy.dim}")

//  val xe = xxe                                                        // full
    val xe = xxe(0 until 116)                                           // clip the flat end
//  val y  = yy                                                         // full
    val y  = yy(0 until 116)                                            // clip the flat end
    val hh = 6                                                          // maximum forecasting horizon
    hp("lwave") = 20                                                    // wavelength (distance between peaks)
    Transform.hp("p") = 1.5                                             // power on Pow transform

    for p <- 6 to 6; q <- 4 to 4; s <- 1 to 1  do                       // number of lags (endo, exo); trend
        hp("p")    = p                                                  // number of endo lags
        hp("q")    = q                                                  // try various rules
        hp("spec") = s                                                  // trend specification: 0, 1, 2, 3, 5
//      val mod = ARX_Quad_D (xe, y, hh)                                // create model for time series data
        val mod = ARX_Quad_D.rescale (xe, y, hh)                        // create model for time series data
        banner (s"TnT Forecasts: ${mod.modelName} on COVID-19 Dataset")
        mod.trainNtest_x ()()                                           // use customized trainNtest_x

        mod.setSkip (0)
        mod.rollValidate ()
//      println (s"After Roll TnT Forecast Matrix yf = ${mod.getYf}")
        mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))        // only diagnose on the testing set
//      println (s"Final TnT Forecast Matrix yf = ${mod.getYf}")
    end for

end aRX_Quad_DTest4

