
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Mar 22 22:31:32 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Quality of Fit (QoF) Metrics/Measures Suitable for All Models
 *
 *  @see facweb.cs.depaul.edu/sjost/csc423/documents/f-test-reg.htm
 *  @see avesbiodiv.mncn.csic.es/estadistica/ejemploaic.pdf
 *  @see en.wikipedia.org/wiki/Bayesian_information_criterion
 *  @see www.forecastpro.com/Trends/forecasting101August2011.html
 *  @see www.bpa.gov/-/media/Aep/energy-efficiency/evaluation-projects-studies/uncertainty-methods-comparisons-final.pdf
 *
 *  @see `Fit` and `classifying.FitC` for more metrics/measures
 */

package scalation
package modeling

import scala.collection.mutable.{LinkedHashMap, Map}
import scala.math.{abs, sqrt}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FitM` trait provides methods to determine basic Quality of Fit 'QoF' metrics/measures
 *  suitable for all Models.  Note, to work with multiple types of models where degrees
 *  of freedom (df) may be hard to calculate, sde uses m-1 rather than df for sample estimates,
 *  while rmse uses a population formula (i.e., divide by m).  Therefore, in ScalaTion sde
 *  will be slightly larger than rmse.
 *  @see `Fit` for a more complete implementation suitable for several models.
 */
trait FitM: 

    protected var m      = -1                                // number of instances (# data points)

    protected var sse    = -1.0                              // Sum of Squares for Error (SSE or RSS)
    protected var ssr    = -1.0                              // Sum of Squares Regression/model (SSR)
    protected var sst    = -1.0                              // Sum of Squares Total (SST = SSR + SSE)
    protected var sde    = -1.0                              // Standard Deviation of Errors (standard error of estimate)
                                                             //   note sde uses sample vs. rmse uses population formulas
    protected var rSq    = -1.0                              // coefficient of determination R^2 using mean
    protected var rSq0   = -1.0                              // coefficient of determination R^2 using 0
    protected var mse0   = -1.0                              // raw/MLE Mean Squared Error (MSE0)
    protected var rmse   = -1.0                              // Root Mean Squared Error (RMSE)
    protected var mae    = -1.0                              // Mean Absolute Error (MAE or MAD)
    protected var smape  = -1.0                              // symmetric Mean Absolute Percentage Error (sMAPE)

    private val flaw = flawf ("FitM")                        // flaw function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the sum of the squares for error (sse).  Must call diagnose first.
     */
    inline def sse_ : Double = sse

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the coefficient of determination (R^2).  Must call diagnose first.
     */
    inline def rSq_ : Double  = rSq                                 // using mean 
    inline def rSq0_ : Double = rSq0                                // using 0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose and return the health of the model by computing the Quality of Fit (QoF)
     *  metrics/measures, from the error/residual vector and the predicted & actual
     *  responses.  For some models the instances may be weighted.
     *  @see `Regression_WLS`
     *  Override to add more metrics.
     *  @param y   the actual response/output vector to use (test/full)
     *  @param yp  the predicted response/output vector (test/full)
     *  @param w   the weights on the instances (defaults to null)
     */
    def diagnose (y: VectorD, yp: VectorD, w: VectorD = null): VectorD =
        m = y.dim                                            // size of response vector (test/full)
        if m < 2       then flaw ("diagnose", s"requires at least 2 responses to evaluate m = $m")
        if yp.dim != m then flaw ("diagnose", s"yp.dim = ${yp.dim} != y.dim = $m")

        val mu = y.mean                                      // Mean of y (may be zero)
        val e  = y - yp                                      // residual/error vector
        sse    = e.normSq                                    // Sum of Squares for Error
        if w == null then
            sst = (y - mu).normSq                            // Sum of Squares Total (ssr + sse)
            ssr = sst - sse                                  // Sum of Squares Regression
        else
            ssr = (w * (yp - (w * yp / w.sum).sum)~^2).sum
            sst = ssr + sse
        sde    = e.stdev                                     // Standard Deviation of Error

        rSq    = 1 - sse / sst                               // R^2 using mean
        rSq0   = 1 - sse / y.normSq                          // R^2 using 0 (used by R when no intercept)

        mse0   = sse / m                                     // raw/MLE Mean Squared Error
        rmse   = sqrt (mse0)                                 // Root Mean Squared Error
        mae    = e.norm1 / m                                 // Mean Absolute Error
        smape  = FitM.smapeF (y, yp, e)                      // symmetric Mean Absolute Percentage Error (sMAPE)
        fit                                                  // returns QoF
    end diagnose

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Quality of Fit (QoF) measures corresponding to the labels given.
     *  Note, if sse > sst, the model introduces errors and the rSq may be negative,
     *  otherwise, R^2 (rSq) ranges from 0 (weak) to 1 (strong).
     *  Override to add more quality of fit measures.
     */
    def fit: VectorD = VectorD (rSq, sst, sse, sde, mse0, rmse, mae, smape, m)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the help string that describes the Quality of Fit (QoF) metrics/measures.
     *  @see `Fit` for an implementation.  Override to correspond to fitLabel.
     */
    def help: String = "Not available for `FitM`"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_     the testing/full data/input matrix
     *  @param fname  the array of feature/variable names
     *  @param b      the parameters/coefficients for the model
     *  @param vifs   the Variance Inflation Factors (VIFs)
     */
    def summary (x_ : MatrixD, fname: Array [String], b: VectorD, vifs: VectorD = null): String =
        "Not available for `FitM`"
    end summary

end FitM


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `FitM` object provides functions for making fit maps for QoF measures.
 */
object FitM:

    private val fitLabel = Array ("rSq", "sst", "sse", "sde", "mse0", "rmse", "mae", "smape", "m")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the symmetric Mean Absolute Percentage Error (sMAPE) score.
     *  @caveat:  y_i = yp_i = 0 => no error => no percentage error
     *  @param y   the given time-series (must be aligned with the forecast)
     *  @param yp  the forecasted time-series
     *  @param e_  the error/residual vector (if null, recompute)
     */
    inline def smapeF (y: VectorD, yp: VectorD, e_ : VectorD = null): Double =
        val e = if e_ == null then y - yp else e_
        var s = 0.0
        for i <- e.indices if e(i) != 0.0 do
            s += abs (e(i)) / (abs (y(i)) + abs (yp(i)))
        200 * s / e.dim
    end smapeF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Mean Absolute Error (MAE) score on the normalized/transformed data.
     *  @caveat:  y_i = yp_i = 0 => no error => no percentage error
     *  @param y      the given time-series (must be aligned with the forecast)
     *  @param yp     the forecasted time-series
     *  @param tForm  the transformation to scale the data
     */
    inline def n_maeF (y: VectorD, yp: VectorD, tForm: Transform = null): Double =
        val m    = y.dim                                            // size of response vector (test/full)
        val y_t  = tForm.f(y)
        val yp_t = tForm.f(yp)
        val e    = y_t - yp_t
        e.norm1 / m
    end n_maeF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Mean Squared Error (MSE) score on the normalized/transformed data.
     *  @caveat:  y_i = yp_i = 0 => no error => no percentage error
     *  @param y      the given time-series (must be aligned with the forecast)
     *  @param yp     the forecasted time-series
     *  @param tForm  the transformation to scale the data
     */
    inline def n_mseF (y: VectorD, yp: VectorD, tForm: Transform = null): Double =
        val m    = y.dim                                            // size of response vector (test/full)
        val y_t  = tForm.f(y)
        val yp_t = tForm.f(yp)
        val e    = y_t - yp_t
        e.normSq / m
    end n_mseF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return common Time Series (TS) results (sample size, sMAPEs, normalized MAEs and MSEs)
     *  for all horizons (0->h1, 1->h2, ..., hh-1->hh) using the forecast matrix and return
     *  averages.  These results are commonly given in research papers.
     *  @param yf     the forecast matrix
     *  @param hh     total numer of horizons
     *  @param tForm  the transformation to scale the data
     */
    def getTSResult (yf: MatrixD, hh: Int, tForm: Transform): Array [VectorD] =
        val n_sample = new VectorD (hh)
        val smapes   = new VectorD (hh + 1)
        val maes     = new VectorD (hh + 1)
        val mses     = new VectorD (hh + 1)
        val y = yf(?, 0)
        val d = y.dim
        for h <- 0 until hh do
            val yp      = yf(?, h + 1)
            n_sample(h) = d - h
            smapes(h)   = FitM.smapeF (y(h until d), yp(0 until d-h))
            maes(h)     = FitM.n_maeF (y(h until d), yp(0 until d-h), tForm)
            mses(h)     = FitM.n_mseF (y(h until d), yp(0 until d-h), tForm)
        end for
        smapes(hh) = smapes(0 until hh).mean
        maes(hh)   = maes(0 until hh).mean
        mses(hh)   = mses(0 until hh).mean
        Array (n_sample, smapes, maes, mses)
    end getTSResult

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a map of quality of fit measures (use of `LinkedHashMap` makes it ordered).
     *  @param ftVec  the vector of QoF values
     *  @param ftLab  the array of QoF labels
     */
    def fitMap (ftVec: VectorD, ftLab: Array [String] = fitLabel): Map [String, String] =
        val lm = LinkedHashMap [String, String] ()                          // empty list map
        for i <- ftLab.indices do
            lm += ftLab(i) -> fmt(ftVec(i))
        lm
    end fitMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a map of quality of fit measures (use of `LinkedHashMap` makes it ordered).
     *  @param ftMat  the matrix of QoF values
     *  @param ftLab  the array of QoF labels
     */
    def fitMap (ftMat: MatrixD, ftLab: Array [String]): Map [String, String] =
        val lm = LinkedHashMap [String, String] ()                          // empty list map
        for i <- ftLab.indices do
            lm += ftLab(i) -> (ftMat(i).toString + "\n")
        lm
    end fitMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the quality of fit measures/metrics for each response/output variable.
     *  @param ftMat  the matrix of QoF values (qof x var)
     *  @param ftLab  the array of QoF labels
     */
    def showFitMap (ftMat: MatrixD, ftLab: Array [String]): String =
        val sb = StringBuilder ("\n")
        for i <- ftLab.indices do
            sb ++= s"\t\t${ftLab(i)} \t -> ${ftMat(i)} \n"
        sb.toString
    end showFitMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the table storing the statistics for QoF measures.
     *  @param stats  the table of statistics for QoF measures
     */
    def showQofStatTable (stats: Array [Statistic]): Unit =
        banner ("showQofStatTable: Statistical Table for QoF")
        println (Statistic.labels)
        for i <- stats.indices do
            if i == 0 then println ("-" * 88)
            println (stats(i))
        end for
        println ("-" * 88)
    end showQofStatTable

end FitM

