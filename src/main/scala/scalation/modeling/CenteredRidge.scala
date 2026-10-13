
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Sun Jul 26 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Two-Space Wrapper for Centered Ridge Regression
 *
 *  Wraps a `RidgeRegression` together with its centering (optionally
 *  standardizing) transform pair so that ALL interaction with callers happens
 *  in ORIGINAL space, while fitting happens in CENTERED space.  The wrapper is
 *  the ONLY place where the two spaces meet:
 *      train       takes an original-space window (x, y), fits the transform pair
 *                  on THAT window (its column means, and scales when scaled) and
 *                  then the parameters -- rows outside the window never enter
 *      test        takes original-space (x, y) and centers them internally
 *      predict     takes original-space rows and returns original-space values
 *      QoF         is reported in ORIGINAL space (sMAPE-family is meaningless
 *                  when centered)
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
/** The `CenteredRidge` class wraps a `RidgeRegression` and its transform pair,
 *  presenting an original-space API while fitting in centered space.  Centering
 *  plays the role of the intercept for ridge (the level of y is never penalized),
 *  and the wrapper guarantees the (center at train, add back at predict) recipe
 *  is applied exactly once on every path -- eliminating the class of bugs where
 *  raw data reaches a centered fit or predictions are inverted twice.  The
 *  transform pair is (re)fit by `train` on the rows it is given, so a fit on a
 *  training window (rolling validation, feature selection, hold-out scoring)
 *  uses that window's means only: no look-ahead into testing rows.  With
 *  scaled = true the columns of x are also divided by their (training) standard
 *  deviations, so the ridge penalty treats every column alike whatever its units
 *  (a zero standard deviation is replaced by one); the original-unit slopes are
 *  then recovered by `slopes`.
 *  @param x_org   the data/input m-by-n matrix in ORIGINAL (un-centered) space,
 *                     NOT augmented with a first column of ones
 *  @param y_org   the response/output m-vector in ORIGINAL space
 *  @param fname   the feature/variable names (defaults to null)
 *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term
 *                     lambda * b dot b (defaults to `RidgeRegression.hp`)
 *  @param scaled  whether to standardize the columns of x (training mean and
 *                     standard deviation) rather than only center them (defaults to false)
 */
class CenteredRidge (x_org: MatrixD, y_org: VectorD,
                     fname: Array [String] = null,
                     hparam: HyperParameter = RidgeRegression.hp,
                     scaled: Boolean = false)
      extends Predictor (x_org, y_org, fname, hparam)
         with Fit (dfr = x_org.dim2, df = x_org.dim - x_org.dim2 - 1):

    private val debug = debugf ("CenteredRidge", false)              // debug function

    private var xMu:  VectorD = null                                 // column means of x for the last training window
    private var xSig: VectorD = null                                 // column scales of x (all ones unless scaled)
    private var yMu:  Double  = 0.0                                  // mean of y for the last training window
    fitTransforms (x_org, y_org)                                     // start from the construction data (until `train`)

    val reg = new RidgeRegression (toC (x_org), toC (y_org),         // underlying model, fit in CENTERED space; the
                                   fname, hparam)                    // wrapper (not reg) maps between the spaces

    _modelName = s"CenteredRidge_${reg.lambda_}"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit the transform pair on a training window: the column means of x_, the
     *  mean of y_ and, when scaled, the column standard deviations of x_ (a zero
     *  or undefined standard deviation is replaced by one).  Called by `train`,
     *  so the statistics always belong to the rows the parameters are fit on.
     *  @param x_  the training data/input matrix in original space
     *  @param y_  the training response vector in original space
     */
    private def fitTransforms (x_ : MatrixD, y_ : VectorD): Unit =
        xMu  = x_.mean
        yMu  = y_.mean
        xSig = if ! scaled then VectorD.one (x_.dim2)
               else x_.stdev.map (s => if s > 0.0 && ! s.isInfinite then s else 1.0)
    end fitTransforms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a data/input matrix from ORIGINAL to CENTERED (standardized when scaled) space.
     *  @param x  the matrix in original space
     */
    def toC (x: MatrixD): MatrixD = if scaled then (x - xMu) / xSig else x - xMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map one data/input row from ORIGINAL to CENTERED (standardized when scaled) space.
     *  @param z  the input row in original space
     */
    def toC_x (z: VectorD): VectorD = if scaled then (z - xMu) / xSig else z - xMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a response vector from ORIGINAL to CENTERED space.
     *  @param y  the vector in original space
     */
    def toC (y: VectorD): VectorD = y - yMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a (predicted) response vector from CENTERED back to ORIGINAL space.
     *  @param yp  the vector in centered space
     */
    def toOrg (yp: VectorD): VectorD = yp + yMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the underlying model on an ORIGINAL-space window: first (re)fit the
     *  transform pair on the window (@see `fitTransforms`), then fit the
     *  parameters b in CENTERED space.
     *  @param x_  the training data/input matrix in original space (defaults to x_org)
     *  @param y_  the training response vector in original space (defaults to y_org)
     */
    def train (x_ : MatrixD = x_org, y_ : VectorD = y_org): Unit =
        fitTransforms (x_, y_)                                       // means (and scales) of THIS window only
        reg.train (toC (x_), toC (y_))
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the model on an ORIGINAL-space window, returning original-space
     *  predictions and Quality of Fit in ORIGINAL space (the sMAPE family is only
     *  meaningful there, since mean-zero data saturates sMAPE toward 200
     *  regardless of fit quality).
     *  @param x_  the testing data/input matrix in original space (defaults to x_org)
     *  @param y_  the testing response vector in original space (defaults to y_org)
     */
    def test (x_ : MatrixD = x_org, y_ : VectorD = y_org): (VectorD, VectorD) =
        val yp_org = toOrg (reg.predict_ (toC (x_)))                 // centered predict_, single inversion
        (yp_org, diagnose (y_, yp_org))                              // diagnose on THIS instance (feeds summary)
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the response for one ORIGINAL-space row, returning an ORIGINAL-
     *  space value: center (scale) z on the way in, un-center on the way out,
     *  using the transforms of the last training window.
     *  @param z  the new input vector in original space
     */
    override def predict (z: VectorD): Double = reg.predict_ (toC_x (z)) + yMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the responses for all rows of an ORIGINAL-space matrix, returning
     *  ORIGINAL-space values.
     *  @param x_  the matrix of input rows in original space
     */
    override def predict (x_ : MatrixD): VectorD = toOrg (reg.predict_ (toC (x_)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the parameter/coefficient vector b.
     *  @note b lives in CENTERED (standardized when scaled) space: it has no
     *  intercept component, and its values are per-unit effects about the means.
     *  For original-space values use `intercept` and `slopes`.
     */
    override def parameter: VectorD = reg.parameter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the slopes in ORIGINAL x units: the fitted b itself when only
     *  centering, b_j / sigma_j when scaled.
     *  @note Must be called AFTER `train`.
     */
    def slopes: VectorD = if ! scaled then parameter else parameter / xSig

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recover the ORIGINAL-space intercept b0 from the centered fit.
     *  Centering removes the (unpenalized) intercept from the ridge problem;
     *  it is recovered exactly, with the means of the last training window, as
     *      b0 = mean(y) - (column means of x) dot slopes
     *  so that  yhat = b0 + z dot slopes  reproduces `predict (z)` for any
     *  original-space row z.
     *  @note Must be called AFTER `train`; a sub-model recovers its own b0 from
     *  its own means.
     */
    def intercept: Double = yMu - (xMu dot slopes)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the full ORIGINAL-space parameter vector [b0, b1, ..., bn]:
     *  the recovered intercept prepended to the original-unit slopes.  This
     *  vector is directly comparable to a `Regression (ox, y)` parameter vector
     *  (same length, same meaning), apart from ridge shrinkage of the slopes.
     */
    def parameter_org: VectorD = intercept +: slopes

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
//      super.summary (x_, fname_, b_, vifs)                             // summary from `Fit` (no regularization)

        reg.summary (x_, fname_, b_, vifs)
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): CenteredRidge =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new CenteredRidge (x_cols, y_org, fname2, hparam, scaled)
    end buildModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a column-subset sub-model as a fully consistent `CenteredRidge`
     *  (for feature selection).  The sub-model fits its transforms on the subset
     *  columns of x_org, which yields numerically the SAME means as slicing this
     *  model's means, so the two agree on all shared columns; and the sub-model's
     *  transforms match its own width, so its `predict` works on raw subset rows.
     *  @param cols    the column indices to retain (e.g., from feature selection)
     *  @param fname2  the feature names for the retained columns (defaults to null)
     */
    def buildSub (cols: LSET [Int], fname2: Array [String] = null): CenteredRidge =
        new CenteredRidge (x_org(?, cols), y_org, fname2, hparam, scaled)
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
     *  @param scaled  whether to standardize the columns of x (defaults to false)
     *  @param col     the designated response column (defaults to the last column)
     */
    def apply (xy: MatrixD, fname: Array [String] = null,
               hparam: HyperParameter = RidgeRegression.hp, scaled: Boolean = false)
              (col: Int = xy.dim2 - 1): CenteredRidge =
        new CenteredRidge (xy.not(?, col), xy(?, col), fname, hparam, scaled)
    end apply

end CenteredRidge


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `centeredRidgeTest` main function tests the `centeredRidgeTest` class using
 *  the AutoMPG dataset.  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar, sMAPE, and R^2 cv vs. the instance index.
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


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `centeredRidgeTest2` main function tests the training-window contract and
 *  the scaled option of the `CenteredRidge` class on a small synthetic dataset:
 *  (1) a fit on the first 6 rows uses only their means: with a negligible lambda
 *      (1E-12: `RidgeRegression` treats lambda <= 0 as "search for lambda") its
 *      `parameter_org` equals the parameters of an OLS `Regression` with an
 *      intercept column on the same 6 rows, and its predictions for rows 6-7
 *      equal b0 + z dot slopes,
 *  (2) with the same lambda, scaled = true gives the same original-unit slopes
 *      and predictions as scaled = false, and
 *  (3) with lambda = 1 the scaled fit differs from the unscaled one.
 *  > runMain scalation.modeling.centeredRidgeTest2
 */
@main def centeredRidgeTest2 (): Unit =

    // y = 5 + 2 x1 + 1 x2 + noise
    val x = MatrixD ((8, 2), 1.0, 1.0,
                             2.0, 1.0,
                             3.0, 2.0,
                             4.0, 3.0,
                             5.0, 5.0,
                             6.0, 8.0,
                             7.0, 9.0,
                             8.0, 9.5)
    val y = VectorD (8.1, 10.0, 12.9, 16.2, 19.8, 25.1, 27.9, 30.6)
    val tr  = 0 until 6                                                  // training window
    val te  = 6 until 8                                                  // rows to predict
    val lam = RidgeRegression.hp("lambda")                               // remember the global lambda

    banner ("(1) window fit with lambda = 1E-12 == OLS with intercept on the same rows")
    RidgeRegression.hp("lambda") = 1E-12                                 // ~OLS (lambda <= 0 would trigger `findLambda`)
    val mod = new CenteredRidge (x, y, Array ("x1", "x2"))
    mod.train (x(tr), y(tr))                                             // means of rows 0-5 only
    val ols = new Regression (VectorD.one (6) +^: x(tr), y(tr), Array ("one", "x1", "x2"))
    ols.train ()
    println (s"CenteredRidge: parameter_org = ${mod.parameter_org}")
    println (s"Regression:    parameter     = ${ols.parameter}")
    assert ((mod.parameter_org - ols.parameter).normSq < 1E-12)
    val yp = mod.predict (x(te))
    for i <- te do
        val manual = mod.intercept + (x(i) dot mod.slopes)
        assert (math.abs (yp(i - te.start) - manual) < 1E-8, s"prediction identity failed for row $i")
        assert (math.abs (mod.predict (x(i)) - manual) < 1E-8, s"row predict failed for row $i")
    println (s"predictions for rows 6-7 = $yp")

    banner ("(2) same lambda: scaled = true gives the same original-unit slopes and predictions")
    val modS = new CenteredRidge (x, y, Array ("x1", "x2"), scaled = true)
    modS.train (x(tr), y(tr))
    println (s"scaled: b (standardized space) = ${modS.parameter}, slopes (original units) = ${modS.slopes}")
    assert ((modS.parameter_org - mod.parameter_org).normSq < 1E-12)
    assert ((modS.predict (x(te)) - yp).normSq < 1E-12)

    banner ("(3) lambda = 1: the scaled fit differs from the unscaled fit")
    RidgeRegression.hp("lambda") = 1.0
    val modR  = new CenteredRidge (x, y, Array ("x1", "x2"))
    val modRS = new CenteredRidge (x, y, Array ("x1", "x2"), scaled = true)
    modR.train (x(tr), y(tr)); modRS.train (x(tr), y(tr))
    println (s"unscaled ridge slopes = ${modR.slopes}, scaled ridge slopes = ${modRS.slopes}")
    assert ((modR.slopes - modRS.slopes).normSq > 1E-6)

    RidgeRegression.hp("lambda") = lam                                   // restore the global lambda

end centeredRidgeTest2

