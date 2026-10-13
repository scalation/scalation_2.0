
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sun Jul 26 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Two-Space Wrapper for Centered Ridge Regression
 *
 *  Wraps a `RidgeRegression` together with its CenterForm transform pair so that
 *  ALL interaction with callers happens in ORIGINAL space, while fitting happens
 *  in CENTERED space.  The wrapper is the ONLY place where the two spaces meet:
 *      train/test  take original-space (x, y) and center them internally
 *      predict     takes original-space rows and returns original-space values
 *      QoF         is available in BOTH spaces: centered (algebra checks) and
 *                  original (reporting; sMAPE-family is meaningless when centered)
 *  Column-subset sub-models (for feature selection) are themselves fully
 *  consistent `CenteredRidge` instances: the mean of a column subset equals the
 *  subset of the column means, so transforms agree on shared columns.
 *
 *  Naming convention (matching `RidgeRegression`):
 *      no suffix  =>  original space (round trip handled internally)
 *      _ suffix   =>  transformed/centered space (caller manages the space)
 *
 *  @note    AI Assisted Code
 */

package scalation
package modeling

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CenteredRidge` class wraps a `RidgeRegression` and its CenterForm pair,
 *  presenting an original-space API while fitting in centered space.  Centering
 *  plays the role of the intercept for ridge (the level of y is never penalized),
 *  and the wrapper guarantees the (center at train, add back at predict) recipe
 *  is applied exactly once on every path -- eliminating the class of bugs where
 *  raw data reaches a centered fit or predictions are inverted twice.
 *  @param x_org   the data/input m-by-n matrix in ORIGINAL (un-centered) space,
 *                     NOT augmented with a first column of ones
 *  @param y_org   the response/output m-vector in ORIGINAL space
 *  @param fname   the feature/variable names (defaults to null)
 *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term
 *                     lambda * b dot b (defaults to `RidgeRegression.hp`)
 */
class CenteredRidge (x_org: MatrixD, y_org: VectorD,
                     fname: Array [String] = null,
                     hparam: HyperParameter = RidgeRegression.hp)
      extends Predictor (x_org, y_org, fname, hparam)
         with Fit (dfr = x_org.dim2, df = x_org.dim - x_org.dim2 - 1):

    private val debug = debugf ("CenteredRidge", false)              // debug function

    val xF = CenterForm (x_org)                                      // column-means transform, fit on construction data
    val yF = CenterForm (y_org)                                      // response-mean transform, fit on construction data

    val reg = new RidgeRegression (xF.f (x_org), yF.f (y_org),       // underlying model, fit in CENTERED space
                                   fname, hparam, xF, yF)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a data/input matrix from ORIGINAL to CENTERED space.
     *  @param x  the matrix in original space
     */
    def toC (x: MatrixD): MatrixD = xF.f (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a response vector from ORIGINAL to CENTERED space.
     *  @param y  the vector in original space
     */
    def toC (y: VectorD): VectorD = yF.f (y)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a (predicted) response vector from CENTERED back to ORIGINAL space.
     *  @param yp  the vector in centered space
     */
    def toOrg (yp: VectorD): VectorD = yF.fi (yp)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the underlying model on an ORIGINAL-space window; the parameters b
     *  are fit in CENTERED space internally (the wrapper centers both x_ and y_
     *  using the transforms fit at construction, so window fits share one set of
     *  means with the full fit and with all sub-models).
     *  @param x_  the training data/input matrix in original space (defaults to x_org)
     *  @param y_  the training response vector in original space (defaults to y_org)
     */
    def train (x_ : MatrixD = x_org, y_ : VectorD = y_org): Unit =
        reg.train (toC (x_), toC (y_))
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the model on an ORIGINAL-space window, returning original-space
     *  predictions and Quality of Fit in BOTH spaces:
     *      (yp_org, qof_centered, qof_org)
     *  rSq and rSqBar agree between the two spaces (centering both sides is
     *  algebraically equivalent to fitting an unpenalized intercept), providing a
     *  free consistency check; the sMAPE family is only meaningful in qof_org,
     *  since mean-zero data saturates sMAPE toward 200 regardless of fit quality.
     *  @param x_  the testing data/input matrix in original space (defaults to x_org)
     *  @param y_  the testing response vector in original space (defaults to y_org)
     */
    def test (x_ : MatrixD = x_org, y_ : VectorD = y_org): (VectorD, VectorD) =
        val yp_org = toOrg (reg.predict_ (toC (x_)))                 // centered predict_, single inversion
        (yp_org, diagnose (y_, yp_org))                              // diagnose on THIS instance (feeds summary)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the response for one ORIGINAL-space row, returning an ORIGINAL-space
     *  value.  Delegates to `reg.predict` (NOT `reg.predict_`), which is already
     *  transform-aware: it centers z on the way in and un-centers on the way out.
     *  Do NOT apply `toOrg` on top of this -- that would invert twice.
     *  @param z  the new input vector in original space
     */
    override def predict (z: VectorD): Double = reg.predict (z)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the responses for all rows of an ORIGINAL-space matrix, returning
     *  ORIGINAL-space values (delegates to the transform-aware `reg.predict`).
     *  @param x_  the matrix of input rows in original space
     */
    override def predict (x_ : MatrixD): VectorD = reg.predict (x_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter/coefficient vector b.
     *  @note b lives in CENTERED space: it has no intercept component, and its
     *  values are per-unit effects about the means.  For an original-space
     *  intercept use: b0 = mean(y_org) - (column means of x_org) dot b.
     */
    override def parameter: VectorD = reg.parameter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recover the ORIGINAL-space intercept b0 from the centered fit.
     *  Centering removes the (unpenalized) intercept from the ridge problem;
     *  it is recovered exactly as
     *      b0 = mean(y_org) - (column means of x_org) dot b
     *  so that  yhat = b0 + z dot b  reproduces `predict (z)` for any
     *  original-space row z.
     *  @note Must be called AFTER `train`; uses the CURRENT parameter vector b,
     *  so a sub-model recovers its own b0 from its own means.
     */
    def intercept: Double = y_org.mean - (x_org.mean dot parameter)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the full ORIGINAL-space parameter vector [b0, b1, ..., bn]:
     *  the recovered intercept prepended to the (centered-fit) slopes, which for
     *  pure centering are already per-unit, original-space effects.  This vector
     *  is directly comparable to a `Regression (ox, y)` parameter vector (same
     *  length, same meaning), apart from ridge shrinkage of the slopes.
     */
    def parameter_org: VectorD = intercept +: parameter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary for a model with diagnostics for each predictor 'x_j'
     *  and the overall Quality of Fit (QoF).
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients for the model
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname, b_ : VectorD = parameter,
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                             // summary from `Fit`
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): CenteredRidge =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new CenteredRidge (x_cols, y_org, fname2, hparam)
    end buildModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a column-subset sub-model as a fully consistent `CenteredRidge`
     *  (for feature selection).  The sub-model re-fits CenterForm on the subset
     *  columns of x_org, which yields numerically the SAME means as slicing this
     *  model's means, so the two agree on all shared columns; and the sub-model's
     *  transforms match its own width, so its `predict` works on raw subset rows
     *  (unlike sub-models built over pre-centered columns with full-width
     *  transforms).
     *  @param cols    the column indices to retain (e.g., from feature selection)
     *  @param fname2  the feature names for the retained columns (defaults to null)
     */
    def buildSub (cols: LSET [Int], fname2: Array [String] = null): CenteredRidge =
        new CenteredRidge (x_org(?, cols), y_org, fname2, hparam)
    end buildSub

end CenteredRidge


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CenteredRidge` companion object provides factory methods for creating
 *  centered ridge regression models.
 */
object CenteredRidge:

    val hp = RidgeRegression.hp

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `CenteredRidge` from a combined data matrix.
     *  @param xy      the un-centered data matrix combined with the response,
     *                     NOT augmented with a first column of ones
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the shrinkage hyper-parameter (defaults to `RidgeRegression.hp`)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = RidgeRegression.hp)
              (col: Int = xy.dim2 - 1): CenteredRidge =
        new CenteredRidge (xy.not(?, col), xy(?, col), fname, hparam)
    end apply

end CenteredRidge


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `centeredRidgeTest` main function tests the `centeredRidgeTest` class using
 *  the AutoMPG dataset.  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  Note, since x0 is automatically included in feature selection, make it an important variable.
 *  > runMain scalation.modeling.centeredRidgeTest
 */
@main def centeredRidgeTest (): Unit =

    import Example_AutoMPG._

    banner ("AutoMPG Regression")
    val reg = new Regression (ox, y, ox_fname)                         // create a regression model (with intercept)
    reg.inSample_Test ()                                               // train and test the model
    println (reg.summary ())                                           // parameter/coefficient statistics
    reg.validate()()
    RidgeRegression.hp("lambda") = 10.0

    banner ("AutoMPG Ridge Regression")
    val mod = new CenteredRidge (x, y, x_fname)                        // create a ridge regression model (no intercept)
    mod.inSample_Test ()                                               // train and test the model
    println (mod.summary ())                                           // parameter/coefficient statistics
    Predictor.makePredictionInt (mod, mod.getX, y, mod.predict (x))    // make and show PREDICTION INTERVALs

    banner ("AutoMPG Validation Test")
    mod.validate ()()

end centeredRidgeTest

