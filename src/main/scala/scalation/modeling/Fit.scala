
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Mar 22 22:31:32 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Quality of Fit (QoF)
 *
 *  @see facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm
 *       avesbiodiv.mncn.csic.es/estadistica/ejemploaic.pdf
 *       en.wikipedia.org/wiki/Bayesian_information_criterion
 *       www.forecastpro.com/Trends/forecasting101August2011.html
 *
 *  @see github.com/scikit-learn/scikit-learn/issues/20162     // used in scikit-learn
 *       www.mdpi.com/1999-4893/13/6/132                       // defines several metrics
 *       arxiv.org/pdf/2005.12881.pdf                          // for IS and WIS
 *       https://www.sciencedirect.com/science/article/pii/S1364032120308005
 *       www.datasciencewithmarco.com/blog/conformal-prediction-in-time-series-forecasting
 */

package scalation
package modeling

import scala.collection.mutable.Map
import scala.math.{abs, log, sqrt}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
import scalation.random.CDF.{fisherCDF, studentTCDF}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QoF` enum defines the Quality of Fit (QoF) measures/metrics.
 *  @param name  the name of the parameter
 */
enum QoF (val name: String):

    case rSq    extends QoF ("rSq")                            // index  0  0-3 related to R^2
    case rSqBar extends QoF ("rSqBar")                         // index  1
    case sst    extends QoF ("sst")                            // index  2
    case sse    extends QoF ("sse")                            // index  3

    case sde    extends QoF ("sde")                            // index  4  4-8 various error metrics
    case mse0   extends QoF ("mse0")                           // index  5
    case rmse   extends QoF ("rmse")                           // index  6
    case mae    extends QoF ("mae")                            // index  7
    case smape  extends QoF ("smape")                          // index  8

    case m      extends QoF ("m")                              // index  9  9-14 degrees of freedom and information criteria
    case dfr    extends QoF ("dfr")                            // index 10
    case df     extends QoF ("df")                             // index 11
    case fStat  extends QoF ("fStat")                          // index 12
    case aic    extends QoF ("aic")                            // index 13
    case bic    extends QoF ("bic")                            // index 14

    case mape   extends QoF ("mape")                           // index 15  15-17 time series metrics (also 8)
    case mase   extends QoF ("mase")                           // index 16
    case smapeC extends QoF ("smapeC")                         // index 17

    case picp   extends QoF ("picp")                           // index 18  18-23 for prediction intervals 
    case pinc   extends QoF ("pinc")                           // index 19
    case ace    extends QoF ("ace")                            // index 20
    case pinaw  extends QoF ("pinaw")                          // index 21
    case mis    extends QoF ("mis")                            // index 22
    case wis    extends QoF ("wis")                            // index 23

end QoF

val qoF_names = QoF.values.map (_.toString)                    // The QoF names from the QoF enum

import QoF._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fit` companion object provides factory methods for assessing quality of
 *  fit for standard types of modeling techniques.
 */
object Fit:

    val MIN_FOLDS = 3                                                       // minimum number of folds for cross-validation
    val N_QoF     = QoF.values.size                                         // the number of QoF measures

    val α_ = VectorD (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  // values of α (type I error) used by WIS

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the help string that describes the Quality of Fit (QoF) measures
     *  provided by the `Fit` trait.  The QoF measures are divided into two groups:
     *  general and statistical (that often require degrees of freedom and/or
     *  log-likelihoods).
     *  @see www.ncbi.nlm.nih.gov/pmc/articles/PMC5570302/
     *  @see en.wikipedia.org/wiki/Coefficient_of_determination
     */
    def help: String =
        """
help: Quality of Fit (QoF) metrics/measures:
    rSq    =  R-squared, the Coefficient of Determination (R^2)
    rSqBar =  adjusted R-squared (R^2-bar)
    sst    =  Sum of Squares Total (SST) [ssr + sse]
    sse    =  Sum of Squares for Error (SSE = RSS)

    sde    =  Standard Deviation of Errors (SDE)
    mse0   =  raw Mean Square Error (MSE = SSE / m)
    rmse   =  Root Mean Square Error (RMSE)
    mae    =  Mean Absolute Error (MAE)
    smape  =  symmetric Mean Absolute Percentage Error (sMAPE)

    m      =  Number of Observations
    dfr    =  Degrees of Freedom (DFr) taken by the regression/model, e.g., one lost per parameter
    df     =  Degrees of Freedom (DF) left for residuals/errors
    fStat  =  Fisher's Statistic
    aic    =  Akaike Information Criterion (AIC)
    bic    =  Bayesian Information Criterion (BIC)

    mape   =  Mean Absolute Percentage Error (MAPE)
    mase   =  Mean Absolute Scaled Error (MASE)
    smapeC =  symmetric Mean Absolute Percentage Error information Criterion (sMAPE-IC)

    picp   =  Prediction Interval empirical Coverage Probability (PICP)
    pinc   =  Prediction Interval Nominal Coverage probability (PINC) [1 - α/2]
    ace    =  Average Coverage Error (ACE) [empirical - nominal coverage, i.e., picp - pinc]
    pinaw  =  Prediction Interval Normalized Average Width (PINAW)
    mis    =  Mean Interval Score (MIS) [ over given instances ]
    wis    =  Weighted Interval Score (WIS) [ over several α values ]
        """
    end help

    val maxi = Set (QoF.rSq.ordinal, QoF.rSqBar.ordinal)                    // maximize these QoF metrics (min the rest)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a contrary/starting value, -∞ for maximization, ∞ for minimization
     *  @param qk  the QoF metric index/ordinal value
     */
    inline def extreme (qk: Int): Double = if maxi contains qk then NEGATIVE_INFINITY
                                           else POSITIVE_INFINITY

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Collect QoF results for a model and return them in a vector.
     *  @param fit     the fit vector with regard to the training set
     *  @param cv_fit  the fit array of statistics for cross-validation (upon test sets)
     */
    def qofVector (fit: VectorD, cv_fit: Array [Statistic]): VectorD =
        val cv = if cv_fit == null then fit(smapeC.ordinal)                 // cv not computed => use sMAPE_IC
                 else 100 * cv_fit(rSq.ordinal).mean                        // mean for R^2 cv
        VectorD (100 * fit(rSq.ordinal),                                    // R^2 as percentage
                 100 * fit(rSqBar.ordinal),                                 // R^2 Bar as percentage
                 fit(smape.ordinal),                                        // sMAPE
                 cv)                                                        // R^2 cv as percentage, or sMAPE_IC
    end qofVector

    val qofVectorSize = 4                                                   // must correspond to size of qofVector

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a table to store statistics for QoF measures, where each row corresponds
     *  to the statistics on a particular QoF measure, e.g., rSq.
     */
    def qofStatTable: Array [Statistic] =
        val stats = Array.ofDim [Statistic] (N_QoF)                         // for collecting stats on QoF measures
        for i <- stats.indices do stats(i) = new Statistic (values(i).toString, unbiased = true)
        stats
    end qofStatTable

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the quality of fit measures/metrics for each response/output variable.
     @  @see `FitM.showFitMap`
     *  @param ftMat  the matrix of QoF values (qof x var)
     *  @param ftLab  the array of QoF labels (defaults to QoF.values.map (_.toString))
     */
    def showFitMap (ftMat: MatrixD, ftLab: Array [String] = QoF.values.map (_.toString)): String =
        FitM.showFitMap (ftMat, ftLab)
    end showFitMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tally the current QoF measures into the statistical accumulators.
     *  @param stats  the statistics table being updated
     *  @param qof    the current QoF measure vector
     */
    def tallyQof (stats: Array [Statistic], qof: VectorD): Unit =
        if qof(sst.ordinal) > 0.0 then                                      // requires variation in test set
            for q <- qof.indices do stats(q).tally (qof(q))                 // tally these QoF measures
    end tallyQof

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Mean Absolute Error (MAE) for the forecasting model under test.
     *  @param y   the given time-series (must be aligned with the forecast)
     *  @param yp  the forecasted time-series
     *  @param h   the forecasting horizon or stride (defaults to 1)
     */
    inline def mae (y: VectorD, yp: VectorD, h: Int = 1): Double =
//      println (s"mae: y.dim = ${y.dim}, yp.dim = ${yp.dim}")
        var sum = 0.0
        for t <- h until y.dim do sum += abs (y(t) - yp(t-h))
        sum / yp.dim
    end mae

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Mean Absolute Error (MAE) for the Naive Model (simple random walk)
     *  with horizon/stride h.  For comparison with the above method.
     *  @param y  the given time-series
     *  @param h  the forecasting horizon or stride (defaults to 1)
     */
    inline def mae_n (y: VectorD, h: Int = 1): Double =
        var sum = 0.0
        for t <- h until y.dim do sum += abs (y(t) - y(t-h))
        sum / (y.dim - h)
    end mae_n

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Mean Absolute Scaled Error (MASE) for the given time-series.
     *  It is the ratio of MAE of the forecasting model under test and the MAE of
     *  the Naive Model (simple random walk).
     *  @param y   the given time-series (must be aligned with the forecast)
     *  @param yp  the forecasted time-series
     *  @param h   the forecasting horizon or stride (defaults to 1)
     */
    inline def mase (y: VectorD, yp: VectorD, h: Int = 1): Double =
        mae (y, yp, h) / mae_n (y, 1)                                       // compare to Naive (one-step)
//      mae (y, yp, h) / mae_n (y, h)                                       // compare to Naive (h-steps)
    end mase

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Prediction Interval Coverage Probability (PICP) metric, i.e.,
     *  the fraction of actual values inside the prediction interval.
     *  While PINC is the nominal/desired coverage probability (1 - α), PICP is
     *  the corresponding empirical coverage probability.
     *  @param y       the given time-series (must be aligned with the interval forecast)
     *  @param low_up  the (lower, upper) bound vectors used for prediction intervals
     */
    inline def picp_ (y: VectorD, low_up: (VectorD, VectorD)): Double =
        var count = 0
        for i <- y.indices if y(i) in (low_up._1(i), low_up._2(i)) do count += 1
        count / y.dim.toDouble
    end picp_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Prediction Interval Normalised Average Deviation (PINAD) metric, i.e.,
     *  the normalized (by range) average deviation outside the prediction interval.
     *  @param y       the given time-series (must be aligned with the interval forecast)
     *  @param low_up  the (lower, upper) bound vectors used for prediction intervals
     */
    inline def pinad_ (y: VectorD, low_up: (VectorD, VectorD)): Double =
        var sum = 0.0
        for i <- y.indices do
            sum += (if y(i) < low_up._1(i) then low_up._1(i) - y(i)
                    else if y(i) > low_up._2(i) then y(i) - low_up._2(i)
                    else 0.0)
        sum / (y.dim * (y.max - y.min))
    end pinad_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Mean Interval Score (MIS) metric which starts with the average prediction
     *  interval width and adds a penalty for each true_value y(i) that is outside
     *  the prediction interval.  Smaller (in absolute value) scores are better.
     *  @see huiwenn.github.io/predictive-distributions
     *  @see arxiv.org/pdf/2005.12881.pdf
     *  @see search.r-project.org/CRAN/refmans/scoringutils/html/interval_score.html
     *
     *       score = (up − low) + α/2 * (low − true_value) ∗ is(true_value < low)
     *                          + α/2 * (true_value − up)  ∗ is(true_value > up)
     *
     *  @param y       the given time-series (must be aligned with the interval forecast)
     *  @param low_up  the (lower, upper) bound vectors used for prediction intervals
     &  @param α       the significance level (1 - p_)
     */
    inline def mis_ (y: VectorD, low_up: (VectorD, VectorD), α: Double = 0.1): Double =
        val (low, up) = low_up
//      val pf  = 2.0 / α                                                   // penalty factor
        val pf  = 4.0 / α                                                   // penalty factor - based on α/2
        var sum = 0.0
        for i <- y.indices do
            sum += up(i) - low(i)                                           // interval width
            if y(i) < low(i) then sum += pf * (low(i) - y(i))               // y_i below interval penalty
            if y(i) > up(i) then  sum += pf * (y(i) - up(i))                // y_i above interval penalty
        sum / y.dim                                                         // return the mean score
    end mis_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Weighted Interval Score (WIS) metric, i.e., a weighted average of
     *  K prediction intervals each calculated for a different alpha (α) level.
     *  WIS approximates the Continuous Ranked Probability Score (CRPS).
     *  @see arxiv.org/pdf/2005.12881.pdf
     *  @see pmc.ncbi.nlm.nih.gov/articles/PMC7880475/pdf/pcbi.1008618.pdf (equation 1)
     *  @param y       the given time-series (must be aligned with the interval forecast)
     *  @param yp      the point prediction mean/median
     *  @param low_up  the (lower, upper) bound vectors used for prediction intervals
     *  @param α       the vector of significance levels (defaults to the K = 11 prediction intervals 
     *                     used by the COVID-19 Forecast Hub)
     */
    def wis_ (y: VectorD, yp: VectorD, low_up: (MatrixD, MatrixD), α: VectorD = α_): Double =
        val w   = α * 0.25
        val ww  = 0.5
        var sum = ww * (y - yp).abs.mean
        for k <- α.indices do sum += w(k) * mis_ (y, (low_up._1(k), low_up._2(k)), α(k))
        sum / (α.dim + 0.5)
    end wis_

/*
        MAY NEED TO FIX -- use prediction median instead of prediction mean
        val kk  = α.dim
        var sum = α(0) * (y - yp).abs.mean
        for k <- 1 until kk do sum += α(k) * mis_ (y, low_up._1(k), low_up._2(k), α(k))
        sum / (2 * k + 1)
*/

end Fit

import Fit._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fit` trait provides methods to determine basic Quality of Fit QoF measures.
 *  @see reset to reset the degrees of freedom
 *  @param dfr  the degrees of freedom for regression/model
 *  @param df   the degrees of freedom for error
 */
trait Fit (protected var dfr: Double, protected var df: Double)
      extends FitM:

    private val debug   = debugf ("Fit", false)                             // debug function
    private val flaw    = flawf  ("Fit")                                    // flaw function

    private val pIC     = 2.0                                               // penalty multiplier for sMAPE IC
    private var df_t    = dfr + df                                          // total degrees of freedom
    private var r_df    = if df > 1.0 then df_t / df                        // ratio of degrees of freedom (total / error)
                         else dfr + 1.0                                     // case for for less than 1 dof error

    private var mse     = -1.0                                              // mean of squares for error MSE (unbiased)
    private var rse     = -1.0                                              // residual standard error (RSE)
    private var msr     = -1.0                                              // mean of squares for regression/model (MSR)

    private var rSqBar  = -1.0                                              // adjusted R-squared (R^2 Bar)
    private var fStat   = -1.0                                              // F statistic (Quality of Fit)
    private var p_fS    = -1.0                                              // p-value for fStat 
    private var aic     = -1.0                                              // Akaike Information Criterion (AIC)
    private var bic     = -1.0                                              // Bayesian Information Criterion (BIC)

    // Measures used for time series @see www.forecastpro.com/Trends/forecasting101August2011.html
    private var mape    = -1.0                                              // Mean Absolute Percentage Error (MAPE)
    private var mase    = -1.0                                              // Mean Absolute Scaled Error (MASE)
    private var smapeC  = -1.0                                              // symmetric Mean Absolute Percentage Error Information Criteria (sMAPE-IC)
//  private var nmae    = -1.0                                              // normalized MAE (MAD/Mean Ratio)

    private var picp    = -1.0                                              // Prediction Interval empirical Coverage Probability (PICP)
    private var pinc    = -1.0                                              // Prediction Interval Nominal Coverage probability (PINC)
    private var ace     = -1.0                                              // Average Coverage Error (picp - pinc)
    private var pinaw   = -1.0                                              // Prediction Interval Normalized Average Width (PINAW)
    private var mis     = -1.0                                              // Mean Interval Score (MIS) [over given instances]
    private var wis     = -1.0                                              // Weighted Interval Score (WIS) [over several α values]

    protected var sig2e = -1.0                                              // MLE estimate of the population variance on the residuals 

    protected var yForm: Transform = null                                   // optional transformation of the response variable y
    protected var scaledMetrics: Boolean = false                            // whether to use scaled metrics (otherwise use the default original scale)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the y-transformation.
     */
    def getYForm: Transform = yForm

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the degrees of freedom to the new updated values.  For some models,
     *  the degrees of freedom is not known until after the model is built.
     *  @param df_update  the updated degrees of freedom (regression/model, error)
     */
    def resetDF (df_update: (Double, Double)): Unit =
        dfr  = df_update._1; df = df_update._2                              // degrees of freedom
        df_t = dfr + df                                                     // total degrees of freedom
        r_df = if df > 1.0 then df_t / df                                   // ratio of degrees of freedom (total / error)
               else dfr + 1.0                                               // case for for less than 1 DoF error
        debug ("resetDF", s"dfr = $dfr, df = $df")
    end resetDF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the mean of the squares for error (sse / df).  Must call diagnose first.
     */
    inline def mse_ : Double = mse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  from the error/residual vector and the predicted & actual responses.
     *  For some models the instances may be weighted.
     *  @see `Regression_WLS`
     *  @param y_raw   the actual response/output vector to use (test/full)
     *  @param yp_raw  the predicted response/output vector (test/full)
     *  @param w       the weights on the instances (defaults to null)
     */
    override def diagnose (y_raw: VectorD, yp_raw: VectorD, w: VectorD = null): VectorD =
        val idx = y_raw.indexOf (NO_DOUBLE)                                 // skip all after NO_DOUBLE (filler)

        val (y_, yp_) = if idx < 0 then (y_raw, yp_raw) else (y_raw(0 until idx), yp_raw(0 until idx))
        val (y, yp)   = if scaledMetrics || yForm == null then (y_, yp_)
                        else (yForm.fi(y_), yForm.fi(yp_))
        super.diagnose (y, yp, w)                                           // compute `FitM` metrics

        val e = y - yp                                                      // FIX - avoid computing twice
//      println (s"Fit.diagnose:\n y = $y,\n yp = $yp,\n e = $e")

        if dfr < 0 || df < 0 then
            flaw ("diagnose", s"degrees of freedom dfr = $dfr and df = $df must be non-negative")

        msr    = if dfr == 0 then 0.0 else ssr / dfr                        // Mean Squared Regression
        mse    = sse / df                                                   // Mean squared Error

        rse    = sqrt (mse)                                                 // Residual Standard Error
        rSqBar = 1 - (1-rSq) * r_df                                         // adjusted R-squared

        fStat  = msr / mse                                                  // F statistic (quality of fit)
        p_fS = if dfr == 0 then -0.0
               else 1.0 - fisherCDF (fStat, dfr.toInt, df.toInt)            // p-value for fStat   
        if p_fS.isNaN then p_fS = -0.0                                      // NaN => check error message produced by fisherCDF

        if sig2e == -1.0 then sig2e = e.variance_

        val ln_m = log (m)                                                  // natural log of m (ln(m))
        aic    = ll() + 2 * (dfr + 1)                                       // Akaike Information Criterion
                                                                            //   the + 1 on dfr accounts for the sig2e, which is
                                                                            //   an additional parameter to be estimated in MLE
        bic    = aic + (dfr + 1) * (ln_m - 2)                               // Bayesian Information Criterion
        mape   = 100 * (e.abs / y.abs).sum / m                              // Mean Absolute Percentage Error
        mase   = Fit.mase (y, yp)                                           // Mean Absolute Scaled Error
        smapeC = smape + pIC * (dfr + 1) / y.dim.toDouble                   // sMAPE Information Criterion
        fit
    end diagnose

//      nmae   = mae / mu                                                   // normalized MAE (MAD/Mean Ratio)
//      nrmse  = rmse / mu                                                  // normalized RMSE
//  issues concerning mean: full, train or test?
//      val ym   = if ym_ == -0.0 then { debug ("diagnose", "test mean"); mu }
//                 else                { debug ("diagnose", "train mean"); ym_ }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF)
     *  metrics/measures, from the error/residual vector and the predicted &
     *  actual responses.  For some models the instances may be weighted.
     *  This method also includes PREDICTION INTERVAL (PI) metrics/measures.
     *  @see otexts.com/fpp2/prediction-intervals.html
     *  Note: `wis` should be computed separately as the bounds are matrices.
     *  @param y       the actual response/output vector to use (test/full)
     *  @param yp      the point prediction mean/median
     *  @param low_up  the predicted (lower, upper) bounds vectors
     *  @param α       the significance/nominal level of uncertainty (α) (defaults to 0.1, 10%)
     *  @param w       the weights on the instances (defaults to null)
     */
    def diagnose_ (y: VectorD, yp: VectorD, low_up: (VectorD, VectorD), α: Double = 0.1,
                   w: VectorD = null): VectorD =
        diagnose (y, yp, w)                                                 // call the main diagnose method for non-PI metrics

        picp  = picp_ (y, low_up)                                           // Prediction Interval empirical Coverage Probability
        pinc  = 1 - α/2                                                     // Prediction Interval Nominal Coverage probability
        ace   = picp - pinc                                                 // Average Coverage Error (empirical - nominal)
        pinaw = (low_up._2 - low_up._1).mean / (y.max - y.min)              // Prediction Interval Normalized Average Width
                                                                            //     average PI width / range of y values
        mis   = mis_ (y, low_up)                                            // Mean Interval Score
        fit
    end diagnose_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  specifically for the weighted interval score that allows using custom α levels.
     *  @param y       the given time-series (must be aligned with the interval forecast)
     *  @param yp      the point prediction mean/median
     *  @param low_up  the predicted (lower, upper) bounds matrices for various α levels
     *                     (column for each α level)
     *  @param α       the vector of significance levels (defaults to the K = 11 prediction
     *                     intervals used by the CDC Forecast Hub)
     */
    def diagnose_wis (y: VectorD, yp: VectorD, low_up: (MatrixD, MatrixD), α: VectorD = α_): Double =
        wis = wis_ (y, yp, low_up, α)
        wis
    end diagnose_wis

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures
     *  for both POINT PREDICTIONS and PREDICTION INTERVALS.
     *  @param y       the given time-series (must be aligned with the interval forecast)
     *  @param yp      the point prediction mean/median
     *  @param low_up  the predicted (lower, upper) bounds matrices for various α levels
     *                     (column for each α level)
     *  @param α       the vector of significance levels (defaults to the K = 11 prediction
     *                     intervals used by the CDC Forecast Hub)
     *  @param iα      the index for the main significance level out of the vector α
     */
    def diagnose_pi (y: VectorD, yp: VectorD, low_up: (MatrixD, MatrixD), α: VectorD = α_,
                     iα: Int = 2): (VectorD, Int) =
        diagnose_ (y, yp, MatrixD.at (low_up, iα), α(iα))
        wis = wis_ (y, yp, low_up, α)
        (fit, iα) 
    end diagnose_pi

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  from the predicted & actual matrix responses (output variable per column).
     *  For some models the instances may be weighted.
     *  @see `Regression_WLS`
     *  @param yy   the actual response/output matrix to use (test/full)
     *  @param yyp  the predicted response/output matrix (test/full)
     *  @param w    the weights on the instances (defaults to null)
     */
    def diagnose_mat (yy: MatrixD, yyp: MatrixD, w: VectorD = null): MatrixD =
        MatrixD (for k <- yy.indices2 yield diagnose (yy(?, k), yyp(?, k), w)).ᵀ 
    end diagnose_mat

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The log-likelihood function times -2.  Override as needed.
     *  @see www.stat.cmu.edu/~cshalizi/mreg/15/lectures/06/lecture-06.pdf
     *  @see www.wiley.com/en-us/Introduction+to+Linear+Regression+Analysis%2C+5th+Edition-p-9780470542811
     *       Section 2.11
     *  @param ms  raw Mean Squared Error
     *  @param s2  MLE estimate of the population variance of the residuals
     */
    def ll (ms: Double = mse0, s2: Double = sig2e, m2: Int = m): Double =
        -m2 / 2.0 * (log (_2Pi) + log (s2) + ms / s2)
    end ll

//  def ll (ms: Double = mse0, s2: Double = sig2e): Double = m * (log (_2Pi) + log (s2) + ms / s2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Quality of Fit (QoF) measures corresponding to the labels given.
     *  Note, if sse > sst, the model introduces errors and the rSq may be negative,
     *  otherwise, R^2 (rSq) ranges from 0 (weak) to 1 (strong).
     *  Override to add more quality of fit measures.
     */
    override def fit: VectorD = VectorD (rSq, rSqBar, sst, sse, sde, mse0, rmse, mae,
                                         smape, m, dfr, df, fStat, aic, bic, mape, mase, smapeC,
                                         picp, pinc, ace, pinaw, mis, wis)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Quality of Fit (QoF) measures corresponding to the labels given.
     *  Override to add more quality of fit measures.
     */
//  def fit_ : VectorD = fit ++ VectorD (picp, pinc, ace, pinaw, mis, wis)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the help string that describes the Quality of Fit (QoF) measures
     *  provided by the `Fit` trait.  Override to correspond to fitLabel.
     */
    override def help: String = Fit.help

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the QoF metrics/measures in vector qof.
     *  @param qof  the QoF metrics (e.g., for point and interval predictions/forecasts)
     */
    def showQoF (qof: VectorD): Unit = println (FitM.fitMap (qof, qoF_names))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make the PREDICTION INTERVAL (PI) lower and upper bound vectors from
     *  the point predictions and the interval half widths.
     *  @param yp   the vector of point predictions (y-hat)
     *  @param ihw  the vector of interval half widths (one for each prediction)
     */
    inline def PIbounds (yp: VectorD, ihw: VectorD): (VectorD, VectorD) = (yp - ihw, yp + ihw)

    inline def PIbounds (yp: VectorD, ihw_ : MatrixD): (MatrixD, MatrixD) = (-ihw_ + yp, ihw_ + yp)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a PREDICTION INTERVAL half width for each prediction yp (y-hat). 
     *  @caveat:  PREDICTION INTERVAL are built assuming Gaussian errors.
     *  Note: `Fac_Cholesky is used to compute the inverse of xtx.
     *  @see `predictCInt` in `Predictor`
     *  @see stats.stackexchange.com/questions/585660/what-is-the-formula-for-prediction-interval-in-multivariate-case
     *  @see www.geeksforgeeks.org/data-analysis/confidence-and-prediction-intervals-with-statsmodels/
     *  @param x_   the testing/full data/input matrix
     *  @param df_  the error/residual degrees of freedom
     *  @param α    the significance level α = .1 for TWO TAILS:  left tail .05 | 1 - α = .90 | .05 right tail
     *                  e.g., for AutoMPG, t_crit (385, 0.90) = 1.6488210657096942
     *                                     t_crit (385, 0.95) = 1.966
     */
    def predictInt (x_ : MatrixD, df_ : Double = df, α: Double = .1): VectorD =
        val facCho = new Fac_Cholesky (x_.ᵀ  * x_)                          // create a Cholesky factorization of xtx
        val xtxInv = facCho.inverse                                         // take inverse
        val sig2   = mse_
        val p_     = 1 - α/2                                                // need p_-th quantile
        val t_     = t_crit (df_.toInt, p_)                                 // critical value from the t-distribution (two tails)
        debug ("predictInt", s"t_crit (${df_.toInt}, $p_) = $t_")
        val ihw    = new VectorD (x_.dim)
        for i <- x_.indices do
            val x_i = x_(i)                                                 // use i-th predictor vector for ihw(i)
            ihw(i)  = t_ * sqrt (sig2 * (1 + (x_i dot xtxInv * x_i)))
        ihw                                                                 // return vector of Interval Half Widths (IHWs)
    end predictInt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a PREDICTION INTERVAL half width for each prediction yp (y-hat) and
     *  each significance level.
     *  @param x_   the testing/full data/input matrix
     *  @param df_  the error/residual degrees of freedom
     *  @param α    the significance levels to be used (defaults to `Fit.α_`)
     */
    def predictInt_ (x_ : MatrixD, df_ : Double = df, α: VectorD = α_): MatrixD =
        MatrixD (α.map (predictInt (x_, df_, _)))
    end predictInt_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF SUMMARY for a model with diagnostics for each predictor x_j
     *  and the overall Quality of Fit (QoF).
     *  Note: `Fac_Cholesky is used to compute the inverse of xtx.
     *  @param x_     the testing/full data/input matrix
     *  @param fname  the array of feature/variable names
     *  @param b      the parameters/coefficients for the model
     *  @param vifs   the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = null, fname: Array [String] = null, b: VectorD = null,
                          vifs: VectorD = null): String =

        val facCho = new Fac_Cholesky (x_.ᵀ  * x_)                          // create a Cholesky factorization of xtx
        val diag   = facCho.inverse(?)                                      // take inverse and get main diagonal
        val stdErr = (diag * mse_).sqrt                                     // standard error of coefficients

        val stats = (sumCoeff (b, stdErr, vifs), fmt(rse), fmt(rSq), fmt(rSqBar))
        debug ("summary", s"stats = $stats")

        (if fname != null then "fname = " + stringOf (fname) else "") +
        s"""
SUMMARY
    Parameters/Coefficients:
    Var      Estimate    Std. Error \t t value \t Pr(>|t|) \t VIF
----------------------------------------------------------------------------------
${stats._1}
    Residual standard error: ${stats._2} on $df degrees of freedom
    Multiple R-squared:  ${stats._3},	Adjusted R-squared:  ${stats._4}
    F-statistic: $fStat on $dfr and $df DF,  p-value: $p_fS
----------------------------------------------------------------------------------
        """
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce the summary report portion for the parameters/coefficients.
     *  @param b       the parameters/coefficients for the model
     *  @param stdErr  the standard error for parameters/coefficients
     *  @param vf      the Variance Inflation Factors (VIFs)
     */
    private def sumCoeff (b: VectorD, stdErr: VectorD, vf: VectorD): String =
        debug ("sumCoeff", s"stdErr = $stdErr")
        var t, p: VectorD = null
        if stdErr != null then
            t  = b / stdErr                                                 // Student's T statistic
            p  = if df > 0 then t.map ((x: Double) => 2.0 * studentTCDF (-abs (x), df))   // p value
                 else -VectorD.one (b.dim)
        val sb = new StringBuilder ()
        for j <- b.indices do
            sb.append ("    x" + j + "\t " + fmt(b(j)) +
                      ( if stdErr != null then
                            val p_j = if p(j).isNaN then 0.0 else p(j)
                            "\t " + fmt(stdErr(j)) +
                            "\t " + fmt(t(j)) +
                            "\t " + fmt(p_j) +
                            "\t " + (if j == 0 || vf == null then "NA" else fmt(vf(j-1))) + "\n"
                        else "?") )
        end for
        sb.mkString
    end sumCoeff

end Fit


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TestFit` class can be used for comparing two vectors on the basis of QoF.
 *  The degrees of freedom (dfr) for the "model" is assumed to be 1.
 *  Can be used when the degrees of freedom are not known.
 *  @param m  the size of vectors to compare
 */
class TestFit (m: Int) extends Fit (1, m-1):

    def testDiagnose (y: VectorD, yp: VectorD): Map [String, String] =
        FitM.fitMap (diagnose (y, yp), QoF.values.map (_.toString))
    end testDiagnose

end TestFit
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fitTest` main function is used to test the `Fit` trait on a simulated dataset.
 *  It test the `diagnose` method to get metrics on POINT PREDICTIONS.
 *  > runMain scalation.modeling.fitTest
 */
@main def fitTest (): Unit =

//  import scalation.random.Normal

    for sig2 <- 10 to 50 by 10 do                                           // test for increasing noise
//      val rv = Normal (0, sig2)
        val rv = SimpleUniform (-sig2, sig2)
        val y  = VectorD.range (1, 101) + 10.0
        val yp = y.map (_ + rv.gen) 
        val e  = y - yp
        new Plot (null, y, yp, "plot y and yp")
        new Plot (null, e, null, "plot e")

        println (new TestFit (y.dim).testDiagnose (y, yp))
    end for

end fitTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fitTest2` main function is used to test the `Fit` class on a simulated
 *  time series.
 *  It test the `diagnose_` method to get metrics on PREDICTION INTERVALS.
 *  @see `scalation.modeling.forecasting.randomWalkTest3` for another test case
 *  > runMain scalation.modeling.fitTest2
 */
@main def fitTest2 (): Unit =

    import scalation.random.Normal

    for sig2 <- 10 to 50 by 10 do                                           // test for increasing noise
        val rv  = Normal (0, sig2)
        val w   = math.sqrt (sig2) * 1.96
        val yp  = VectorD.range (1, 101) + 10.0
        val y   = yp.map (_ + rv.gen)                                       // simulated time series
        val low = yp.map (_ - w)
        val up  = yp.map (_ + w)
        new PlotM (null, MatrixD (y, yp, low, up), Array ("y", "yp", "low", "up"), "plot y, low and up")

        object ft extends Fit (1, y.dim)
        ft.diagnose_ (y, yp, (low, up))
        ft.diagnose_wis (y, yp, (MatrixD (low), MatrixD (up)), VectorD (0.1))   // FIX - WIS needs multiple α levels
        val qof = ft.fit
        println (FitM.fitMap (qof, qoF_names))
    end for

end fitTest2

