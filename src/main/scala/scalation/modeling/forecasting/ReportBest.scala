
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Fri Feb 27 19:50:13 EST 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Report Results from the Best Model Found
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

import SelectionTech._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Report the results for the best models given by feature selection to have
 *  in the model, returning the variables left and the new Quality of Fit (QoF)
 *  measures for all steps.
 *  @param mod     the model to be evaluated
 *  @param hh      the max forecasting horizon
 *  @param dir     the directory to save the report and qof
 *  @param toFile  whether to send to stdout or to a file
 *  @param ifTest  whether to use test set or full dataset
 *  @param fsType  the type of the feature selection to use
 *  @param cross   indicator to include the cross-validation/validation QoF measure (defaults to "none")
 *  @param first   first variable to consider for elimination
 *                     (default (1) assume intercept x_0 will be in any model)
 *  @param swap    whether to allow a swap step (swap out a feature for a new feature in one step)
 *  @param qk      index of Quality of Fit (QoF) to use for comparing quality
 */
def reportBest (mod: Forecaster_D, hh: Int, dir: String, toFile: Boolean = true,
                ifTest: Boolean = true, fsType: SelectionTech = Stepwise, cross: String = "none", 
                first: Int = 1, swap: Boolean = true) (using qk: Int): Unit =

    val ew      = new EasyWriter ("scalation", dir + "/" + "report.txt", toFile)    
    val y_org   = mod.getY_org
    val (x, y)  = (mod.getX, mod.getYy)
    val tr_size = Model.trSize (y.dim)
    val normForm_y = TransformT.Norm.form (y_org(0 until tr_size)) 
    val t_rng   = 0 until tr_size 
    val (x_tr, y_tr) = (x(t_rng), y(t_rng)) 
    val (_, qof0) = mod.trainNtest (x_tr, y_tr)(x_tr, y_tr)
    ew.println (mod.report (qof0))
    mod.setSkip ()                                                      // don't skip any rows
    mod.rollValidate ()                                            
    val ftMat0 = mod.diagnoseAll (mod.getY, mod.getYf, Forecaster.teRng (y.dim))  

    val qof = new MatrixD (1, hh+1)
    val header: Array [String] = (1 to hh).map (i => s"horizon_$i").toArray ++ Array ("mean")

    val (cols, rSqs)   = mod.selectFeaturesPerH (fsType, cross, first, swap)
    val (mods, _, ftMat) = mod.reportSelection (cols, rSqs, inSample = false)   // TnT of the hh single-horizon models
    val yf_pred = MatrixD (mods.toIndexedSeq.map (_.getYf(?, 1))).ᵀ     // column h-1 = the h-step forecasts
    val yForm   = mod.getYForm                                          // null for an unscaled model
    val yf_full = y_org +^: (if yForm == null then yf_pred else yForm.fi (yf_pred))

    val yf_test = yf_full(Model.trSize (y.dim) until y.dim)
    var yf_save = if ifTest then yf_test else yf_full
    if yf_save == null then yf_save = new MatrixD (1, 1)
    if toFile then
        yf_save.write ("log/scalation" + "/" + dir + "/" + "yf.csv", fullPath = true)
    else if ifTest then
        println (s"yf_test = $yf_save")
    else
        println(s"yf_full = $yf_save")

    ew.println (s"TnT Forecasts: ${mod.modelName}")
    ew.println ("fitMap0 QoF = ")
    ew.println (FitM.showFitMap (ftMat0.transpose, QoF.values.map (_.toString)))
    val mean_size = cols.map (_.size).sum.toDouble / cols.size
    ew.println (s"\nx.dims = ${x.dims}, cols mean_size = ${mean_size}")

    for h <- 0 until hh do
        ew.println (s"mods($h).dims = ${mods(h).getX.dims}, cols($h).size = ${cols(h).size}")
        ew.println (s"cols($h) = ${cols(h)}")
        ew.println (s"rSqs($h) = ${rSqs(h)}\n")
    end for

    ew.println ("\nfitMap QoF = ")
    ew.println (FitM.showFitMap (ftMat.transpose, QoF.values.map (_.toString)))
    val smapes_tr = ftMat(?, 8)
    qof(0, 0 until hh) = smapes_tr
    qof(0, hh) = smapes_tr.mean
    if toFile then
        qof.write ("log/scalation" + "/" + dir + "/" + "qof.csv", header, fullPath = true)
    else
        println (s"qof = $qof")
    ew.println ("\n=============Independent test=============")
    ew.println (s"yf_test.dims = ${yf_test.dims}")
    val stats = FitM.getTSResult (yf_test, hh, normForm_y)
    ew.println (s"sample sizes    = ${stats(0)}")
    ew.println (s"sMAPEs          = ${stats(1)}")
    ew.println (s"Normalized MAEs = ${stats(2)}")
    ew.println (s"Normalized MSEs = ${stats(3)}")
    ew.finish ()

end reportBest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `reportBestTest` main function fit the  parameters and tests the `ARX_D` class on real data:
 *  Forecasting Covid19 using Train and Test (TnT).
 *  Backward/Stepwise feature selection
 *  Test forecasts (h = 1 to hh steps ahead forecasts).
 *  > runMain scalation.modeling.forecasting.reportBestTest
 */
@main def reportBestTest (): Unit =

    import AR.hp
    import Example_Covid.{BEST2_EXO, clip, loadData}

    val (xe, y) = clip (loadData (BEST2_EXO))
    val hh      = 6                                                     // maximum forecasting horizon
    hp("spec")  = 1                                                     // trend specification: 0, 1, 2, 3
    val (p, q, lmb) = (6, 4, 0.5)
    hp("p")     = p                                                     // number of endo lags
    hp("q")     = q                                                     // number of exo lags
    RidgeRegression.hp("lambda") = lmb                                  // regularization/shrinkage parameter
    RidgeRegression.hp("factorization") = "Fac_Cholesky"                // matrix factorization technique

    val mod = ARX_D (xe, y, hh)                                         // create model for time series data
//  val mod = ARX_D.rescale (xe, y, hh, tForm = Log1p.form (y))         // create model for time series data
    reportBest (mod, hh, "reportBestTest", true, false, SelectionTech.Backward)

end reportBestTest

