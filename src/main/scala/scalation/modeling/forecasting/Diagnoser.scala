
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep  1 00:38:10 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Forecast Diagnostics
 */

package scalation
package modeling
package forecasting

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Diagnoser` trait provides methods to determine basic Quality of Fit QoF measures
 *  for time series where forecasting at early time points may not be feasible.
 *  @param dfr  the degrees of freedom for regression/model (0 or more)
 *  @param df   the degrees of freedom for error
 */
abstract class Diagnoser (dfr: Double, df: Double)
         extends Fit (dfr, df):

    // For In-Sample Testing (In-ST), can't forecast for t = 0 (no past data, unless backcasting)
    // first value in time series may be atypical (but not an necessarily an outlier) => skip first 2
    // For Train-and-Test (TnT), can use values from training set, so may set skip = 0
    // Call `setSkip` to change from the DEFAULT value of 2
    // When comparing different models, should use the same skip value for all models

    // Setting skip to 'max (p, q+1)' is common for ARMA (p, q) models, e.g., the state space in
    // Kalman Filters need to hold p past values as well as the q past errors and the current error
    // to be fully populated from data (e.g., not just zeroed out).
    // For ARIMA using a Kalman Filter, it would be 'd + max(p, q+1)'.
    // Some software avoids this burn-in issue by using a diffuse Kalman filter

    private   val debug = debugf ("Diagnoser", false)                      // debug function
    protected var skip: Int = 2                                            // number of beginning elements to skip

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the number of data points/elements to skip at the beginning of a time
     *  series for purposes of diagnosis or computing a loss function.
     *  For In-Sample, the first value (time t = 0) is not forecastable without backcasting.
     *  @param skip  skip this many elements at the beginning of the time series
     */
    def setSkip (skip_ : Int): Unit = skip = skip_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Models need to provide a means for updating the Degrees of Freedom (DF).
     *  Note:  Degrees of Freedom are mainly relevant for full and train, not test.
     *  @param size  the size of dataset (full, train, or test sets)
     */
    def mod_resetDF (size: Int): Unit = resetDF (dfr, size - dfr)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Diagnose the health of the model by computing the Quality of Fit (QoF) measures,
     *  from the error/residual vector and the predicted & actual responses.
     *  For some models the instances may be weighted.
     *  For time series, the first few predictions use only part of the model, so may be skipped.
     *  @param y   the actual response/output vector to use (test/full)
     *  @param yp  the predicted response/output vector (test/full)
     *  @param w   the weights on the instances (defaults to null)
     */
    override def diagnose (y: VectorD, yp: VectorD, w: VectorD = null): VectorD =
        debug ("diagnose", s"skip = $skip")
        if skip > 0 then
            super.diagnose (y.drop (skip), yp.drop (skip),
                            if w != null then w.drop (skip) else null)
        else super.diagnose (y, yp, w)
    end diagnose

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sum of squares errors (loss function), assuming the first 'skip'
     *  errors are zero.
     *  @param y   the actual response vector
     *  @param yp  the predicted response vector (one-step ahead)
     */
    def ssef (y: VectorD, yp: VectorD): Double =
        var ss = 0.0
        for t <- skip until y.dim do ss += (y(t) - yp(t))~^2
        ss
    end ssef

end Diagnoser

