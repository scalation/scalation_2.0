
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    WED Aug 05 15:00:00 EDT 2026
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Two-Space Wrapper for Centered Multi-Variate Ridge Regression
 *
 *  Multi-variate analogue of `CenteredRidge`: wraps a `RidgeRegressionMV`
 *  (response is a MATRIX -- one column per forecasting horizon in the Direct
 *  models) together with its centering (optionally standardizing) transform
 *  pair, so that ALL interaction with callers happens in ORIGINAL space while
 *  fitting happens in CENTERED space.  The wrapper is the ONLY place where the
 *  two spaces meet:
 *      train       takes an original-space window (x, y), fits the transform pair
 *                  on THAT window (its column means, and scales when scaled) and
 *                  then the parameters -- rows outside the window never enter
 *      test        takes original-space (x, y) and centers them internally
 *      predict     takes original-space rows and returns original-space values
 *      QoF         is reported in ORIGINAL space, per response column/horizon
 *                  (sMAPE-family is meaningless on centered, mean-zero data)
 *  Column-subset sub-models (for feature selection) are themselves fully
 *  consistent `CenteredRidgeMV` instances: the mean of a column subset equals
 *  the subset of the column means, so transforms agree on shared columns.
 *
 *  Naming convention (matching `RidgeRegressionMV`):
 *      no suffix  =>  original space (round trip handled internally)
 *      _ suffix   =>  transformed/centered space (caller manages the space)
 *
 *  @note    AI Assisted Code
 */

package scalation
package modeling
package neuralnet

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CenteredRidgeMV` class wraps a `RidgeRegressionMV` and its transform
 *  pair, presenting an original-space API while fitting in centered space.
 *  Centering plays the role of the (per-column) intercept for ridge -- the level
 *  of each response column is never penalized -- and the wrapper guarantees the
 *  (center at train, add back at predict) recipe is applied exactly once on
 *  every path.  The transform pair is (re)fit by `train` on the rows it is given,
 *  so a fit on a training window (rolling validation, feature selection, hold-out
 *  scoring) uses that window's means only: no look-ahead into testing rows.
 *  With scaled = true the columns of x are also divided by their (training)
 *  standard deviations, so the ridge penalty lambda * b dot b treats every
 *  column alike whatever its units (a zero standard deviation is replaced by
 *  one, leaving a constant column centered rather than divided by zero); the
 *  original-unit slopes are then recovered by `slopes`.
 *  @param x_org   the data/input m-by-n matrix in ORIGINAL (un-centered) space,
 *                     NOT augmented with a first column of ones
 *  @param y_org   the response/output m-by-ny matrix in ORIGINAL space
 *                     (e.g., ny = hh horizon columns for Direct forecasters)
 *  @param fname   the feature/variable names (defaults to null)
 *  @param hparam  the shrinkage hyper-parameter (0 => OLS) in the penalty term
 *                     lambda * b dot b (defaults to `RidgeRegression.hp`)
 *  @param scaled  whether to standardize the columns of x (training mean and
 *                     standard deviation) rather than only center them (defaults to false)
 */
class CenteredRidgeMV (x_org: MatrixD, y_org: MatrixD,
                       fname: Array [String] = null,
                       hparam: HyperParameter = RidgeRegression.hp,
                       scaled: Boolean = false)
      extends PredictorMV (x_org, y_org, fname, hparam)
         with Fit (dfr = x_org.dim2, df = x_org.dim - x_org.dim2 - 1):

    private val debug = debugf ("CenteredRidgeMV", false)            // debug function

    private var xMu:  VectorD = null                                 // column means of x for the last training window
    private var xSig: VectorD = null                                 // column scales of x (all ones unless scaled)
    private var yMu:  VectorD = null                                 // column means of y for the last training window
    fitTransforms (x_org, y_org)                                     // start from the construction data (until `train`)

    val reg = new RidgeRegressionMV (toC_x (x_org), toC_y (y_org),   // underlying model, fit in CENTERED space; the
                                     fname, hparam)                  // wrapper (not reg) maps between the spaces

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Fit the transform pair on a training window: the column means of x_ and
     *  y_ and, when scaled, the column standard deviations of x_ (a zero or
     *  undefined standard deviation is replaced by one).  Called by `train`, so
     *  the statistics always belong to the rows the parameters are fit on.
     *  @param x_  the training data/input matrix in original space
     *  @param y_  the training response matrix in original space
     */
    private def fitTransforms (x_ : MatrixD, y_ : MatrixD): Unit =
        xMu  = x_.mean
        yMu  = y_.mean
        xSig = if ! scaled then VectorD.one (x_.dim2)
               else x_.stdev.map (s => if s > 0.0 && ! s.isInfinite then s else 1.0)
    end fitTransforms

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a data/input matrix from ORIGINAL to CENTERED (standardized when
     *  scaled) space (use `toC_x` for inputs and `toC_y` for responses to avoid
     *  width mistakes).
     *  @param x  the input matrix in original space
     */
    def toC_x (x: MatrixD): MatrixD = if scaled then (x - xMu) / xSig else x - xMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map one data/input row from ORIGINAL to CENTERED (standardized when scaled) space.
     *  @param z  the input row in original space
     */
    def toC_x (z: VectorD): VectorD = if scaled then (z - xMu) / xSig else z - xMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a response matrix from ORIGINAL to CENTERED space.
     *  @param y  the response matrix in original space
     */
    def toC_y (y: MatrixD): MatrixD = y - yMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map a (predicted) response matrix from CENTERED back to ORIGINAL space.
     *  @param yp  the predictions in centered space
     */
    def toOrg (yp: MatrixD): MatrixD = yp + yMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map one (predicted) response row from CENTERED back to ORIGINAL space.
     *  @param yp  the predicted row in centered space
     */
    def toOrg (yp: VectorD): VectorD = yp + yMu

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train the underlying model on an ORIGINAL-space window: first (re)fit the
     *  transform pair on the window (@see `fitTransforms`), then fit the
     *  parameter matrix bb in CENTERED space.  The fitted bb is exposed through
     *  THIS instance, so the inherited `parameter`/`parameters` accessors (and
     *  any `PredictorMV` machinery reading bb directly) see the trained coefficients.
     *  @note the exposed bb is in CENTERED (standardized when scaled) space: no
     *  intercept row, per-unit effects about the means -- for original-space
     *  intercepts and slopes use `intercept` and `slopes`.
     *  @param x_  the training data/input matrix in original space (defaults to x_org)
     *  @param y_  the training response matrix in original space (defaults to y_org)
     */
    def train (x_ : MatrixD = x_org, y_ : MatrixD = y_org): Unit =
        fitTransforms (x_, y_)                                       // means (and scales) of THIS window only
        reg.train (toC_x (x_), toC_y (y_))
        bb = reg.parameters                                          // expose reg's fitted bb through THIS instance
    end train

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the model on an ORIGINAL-space window, returning original-space
     *  predictions and per-column (per-horizon) Quality of Fit computed in
     *  ORIGINAL space -- so the sMAPE family is meaningful (centered, mean-zero
     *  responses saturate sMAPE toward 200 regardless of fit quality).
     *  @param x_  the testing data/input matrix in original space (defaults to x_org)
     *  @param y_  the testing response matrix in original space (defaults to y_org)
     */
    def test (x_ : MatrixD = x_org, y_ : MatrixD = y_org): (MatrixD, MatrixD) =
        val yp_org = toOrg (reg.predict_ (toC_x (x_)))               // centered predict_, single inversion
        val qof    = MatrixD (for k <- y_.indices2 yield
                                  diagnose (y_(?, k), yp_org(?, k))).transpose
        (yp_org, qof)                                                // per-horizon QoF, original space
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the response vector (one value per horizon column) for one
     *  ORIGINAL-space row, returning ORIGINAL-space values: center (scale) z on
     *  the way in, un-center on the way out, using the transforms of the last
     *  training window.
     *  @param z  the new input vector in original space
     */
    override def predict (z: VectorD): VectorD = toOrg (reg.predict_ (toC_x (z)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the responses for all rows of an ORIGINAL-space matrix, returning
     *  ORIGINAL-space values.
     *  @param x_  the matrix of input rows in original space
     */
    override def predict (x_ : MatrixD): MatrixD = toOrg (reg.predict_ (toC_x (x_)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the slopes in ORIGINAL x units, one column per response/horizon k:
     *  the fitted bb itself when only centering, bb(j, k) / sigma_j when scaled.
     *  @note Must be called AFTER `train`.
     */
    def slopes: MatrixD =
        val b = parameter                                            // bb in the fitted (centered/scaled) space
        if ! scaled then b else MatrixD (for j <- b.indices yield b(j) / xSig(j))
    end slopes

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recover the ORIGINAL-space intercepts, one per response column/horizon k:
     *      b0(k) = mean(y(?, k)) - (column means of x) dot slopes(?, k)
     *  with the means of the last training window, so that
     *      yhat(k) = b0(k) + z dot slopes(?, k)  reproduces  predict (z)(k).
     *  @note Must be called AFTER `train`; a sub-model recovers its own
     *  intercepts from its own means.
     */
    def intercept: VectorD = yMu - (slopes dot xMu)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a QoF summary with diagnostics for each predictor 'x_j' and the
     *  overall Quality of Fit (QoF), for ONE designated response column.
     *  @param x_      the testing/full data/input matrix
     *  @param fname_  the array of feature/variable names
     *  @param b_      the parameters/coefficients (defaults to horizon 1's column)
     *  @param vifs    the Variance Inflation Factors (VIFs)
     */
    override def summary (x_ : MatrixD = getX, fname_ : Array [String] = fname,
                          b_ : VectorD = parameter(?, 0),
                          vifs: VectorD = vif ()): String =
        super.summary (x_, fname_, b_, vifs)                         // summary from `Fit`
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data
     *  matrix (the interface feature selection calls).
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    def buildModel (x_cols: MatrixD, fname2: Array [String] = null): CenteredRidgeMV =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new CenteredRidgeMV (x_cols, y_org, fname2, hparam, scaled)
    end buildModel

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a column-subset sub-model as a fully consistent `CenteredRidgeMV`
     *  (for feature selection).  The sub-model fits its transforms on the subset
     *  columns of x_org, which yields numerically the SAME means as slicing this
     *  model's means; its transforms match its own width, so its `predict`
     *  works on raw subset rows.
     *  @param cols    the column indices to retain (e.g., from feature selection)
     *  @param fname2  the feature names for the retained columns (defaults to null)
     */
    def buildSub (cols: LSET [Int], fname2: Array [String] = null): CenteredRidgeMV =
        new CenteredRidgeMV (x_org(?, cols), y_org, fname2, hparam, scaled)
    end buildSub

end CenteredRidgeMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CenteredRidgeMV` companion object provides factory methods for creating
 *  centered multi-variate ridge regression models.
 */
object CenteredRidgeMV:

    val hp = RidgeRegression.hp

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `CenteredRidgeMV` from a combined data matrix.
     *  @param xy      the un-centered data matrix combined with the ny response
     *                     columns (as the LAST ny columns), NOT ones-augmented
     *  @param ny      the number of response columns (e.g., horizons)
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the shrinkage hyper-parameter (defaults to `RidgeRegression.hp`)
     *  @param scaled  whether to standardize the columns of x (defaults to false)
     */
    def apply (xy: MatrixD, ny: Int, fname: Array [String] = null,
               hparam: HyperParameter = RidgeRegression.hp, scaled: Boolean = false): CenteredRidgeMV =
        val n = xy.dim2 - ny
        new CenteredRidgeMV (xy(?, 0 until n), xy(?, n until xy.dim2), fname, hparam, scaled)
    end apply

end CenteredRidgeMV


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `centeredRidgeMVTest` main function tests the `CenteredRidgeMV` class on
 *  synthetic data with TWO response columns whose levels sit well away from
 *  zero, checking the two-space contract:
 *  (1) predictions are in original scale for BOTH columns,
 *  (2) per-column original-space QoF is sane (sMAPE not saturated),
 *  (3) intercept recovery reproduces predict: yhat(k) = b0(k) + z dot bb(?, k).
 *  > runMain scalation.modeling.neuralnet.centeredRidgeMVTest
 */
@main def centeredRidgeMVTest (): Unit =

    // y0 = 5 + 2 x1 + 1 x2 + noise;  y1 = 20 + 0.5 x1 + 3 x2 + noise
    val x = MatrixD ((8, 2), 1.0, 1.0,
                             2.0, 1.0,
                             3.0, 2.0,
                             4.0, 3.0,
                             5.0, 5.0,
                             6.0, 8.0,
                             7.0, 9.0,
                             8.0, 9.5)
    val y = MatrixD ((8, 2),  8.1, 23.6,
                             10.0, 24.2,
                             12.9, 27.4,
                             16.2, 31.1,
                             19.8, 37.6,
                             25.1, 47.2,
                             27.9, 50.4,
                             30.6, 52.4)

    banner ("CenteredRidgeMV on synthetic 2-response data")
    val mod = new CenteredRidgeMV (x, y, Array ("x1", "x2"))
    mod.train ()
    val (yp, qof) = mod.test ()

    println (s"parameter bb (centered space) = ${mod.parameter}")
    println (s"intercepts b0 (original)      = ${mod.intercept}")
    println (s"predictions (original space)  =\n$yp")
    println (s"actuals                       =\n$y")
    println (s"per-column QoF (original space):")
    println (s"  rSq   = ${qof(QoF.rSq.ordinal)}")
    println (s"  smape = ${qof(QoF.smape.ordinal)}    (should be small, NOT ~200)")

    banner ("intercept identity: predict (z) == b0 + z dot bb(?, k)")
    val z  = VectorD (4.5, 4.0)
    val pz = mod.predict (z)
    val b0 = mod.intercept
    for k <- 0 until y.dim2 do
        val manual = b0(k) + (z dot mod.parameter(?, k))
        println (f"k = $k: predict = ${pz(k)}%10.4f  manual = $manual%10.4f")
        assert (math.abs (pz(k) - manual) < 1E-9, s"intercept identity failed for column $k")
    println ("intercept identity: passed")

    banner ("sub-model on column subset (width-consistent predict)")
    import scala.collection.mutable.{LinkedHashSet => LSET}
    val sub = mod.buildSub (LSET (0), Array ("x1"))
    sub.train ()
    val (_, qofS) = sub.test ()
    println (s"sub rSq = ${qofS(QoF.rSq.ordinal)}")
    println (s"sub.predict (VectorD (4.5)) = ${sub.predict (VectorD (4.5))}")

end centeredRidgeMVTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `centeredRidgeMVTest2` main function tests the training-window contract
 *  and the scaled option of the `CenteredRidgeMV` class on the data of
 *  `centeredRidgeMVTest`:
 *  (1) a fit on the first 6 rows uses only their means: with lambda = 0 its
 *      intercepts and slopes equal those of an OLS `Regression` with an
 *      intercept column on the same 6 rows, and its forecasts for rows 6-7
 *      equal b0 + z dot slopes,
 *  (2) with lambda = 0, scaled = true gives the same original-unit slopes and
 *      forecasts as scaled = false (standardization is only a reparameterization),
 *  (3) with lambda > 0 the scaled fit differs from the unscaled one (the penalty
 *      is now unit-free), and
 *  (4) a constant column with scaled = true is left centered (no NaN).
 *  > runMain scalation.modeling.neuralnet.centeredRidgeMVTest2
 */
@main def centeredRidgeMVTest2 (): Unit =

    val x = MatrixD ((8, 2), 1.0, 1.0,
                             2.0, 1.0,
                             3.0, 2.0,
                             4.0, 3.0,
                             5.0, 5.0,
                             6.0, 8.0,
                             7.0, 9.0,
                             8.0, 9.5)
    val y = MatrixD ((8, 2),  8.1, 23.6,
                             10.0, 24.2,
                             12.9, 27.4,
                             16.2, 31.1,
                             19.8, 37.6,
                             25.1, 47.2,
                             27.9, 50.4,
                             30.6, 52.4)
    val tr  = 0 until 6                                                  // training window
    val te  = 6 until 8                                                  // rows to forecast
    val lam = RidgeRegression.hp("lambda")                               // remember the global lambda

    banner ("(1) window fit with lambda = 0 == OLS with intercept on the same rows")
    RidgeRegression.hp("lambda") = 0.0
    val mod = new CenteredRidgeMV (x, y, Array ("x1", "x2"))
    mod.train (x(tr), y(tr))                                             // means of rows 0-5 only
    val ols = new Regression (VectorD.one (6) +^: x(tr), y(tr, 0), Array ("one", "x1", "x2"))
    ols.train ()
    val bOls = ols.parameter                                             // [b0, b1, b2] for response column 0
    println (s"CenteredRidgeMV: b0 = ${mod.intercept(0)}, slopes = ${mod.slopes(?, 0)}")
    println (s"Regression:      b  = $bOls")
    assert (math.abs (mod.intercept(0) - bOls(0)) < 1E-8 && (mod.slopes(?, 0) - bOls(1 until 3)).normSq < 1E-12)
    val fc = mod.predict (x(te))
    for i <- te do
        val manual = mod.intercept + (mod.slopes dot x(i))
        assert ((fc(i - te.start) - manual).normSq < 1E-12, s"forecast identity failed for row $i")
    println (s"forecasts for rows 6-7 =\n$fc")

    banner ("(2) lambda = 0: scaled = true gives the same original-unit slopes and forecasts")
    val modS = new CenteredRidgeMV (x, y, Array ("x1", "x2"), scaled = true)
    modS.train (x(tr), y(tr))
    println (s"scaled: bb (standardized space) = ${modS.parameter}")
    println (s"scaled: slopes (original units)  = ${modS.slopes}")
    assert ((modS.slopes - mod.slopes).normFSq < 1E-12 && (modS.intercept - mod.intercept).normSq < 1E-12)
    assert ((modS.predict (x(te)) - fc).normFSq < 1E-12)

    banner ("(3) lambda = 1: the scaled fit differs from the unscaled fit")
    RidgeRegression.hp("lambda") = 1.0
    val modR  = new CenteredRidgeMV (x, y, Array ("x1", "x2"))
    val modRS = new CenteredRidgeMV (x, y, Array ("x1", "x2"), scaled = true)
    modR.train (x(tr), y(tr)); modRS.train (x(tr), y(tr))
    println (s"unscaled ridge slopes = ${modR.slopes(?, 0)}")
    println (s"scaled ridge slopes   = ${modRS.slopes(?, 0)}")
    assert ((modR.slopes - modRS.slopes).normFSq > 1E-6)

    banner ("(4) a constant column with scaled = true is left centered (no NaN)")
    val xc   = x ++^ MatrixD (VectorD.fill (8)(3.0)).transpose           // add a constant third column
    val modC = new CenteredRidgeMV (xc, y, Array ("x1", "x2", "c"), scaled = true)
    modC.train (xc(tr), y(tr))
    println (s"slopes with a constant column = ${modC.slopes}")
    assert (! modC.slopes.toString.contains ("NaN") && math.abs (modC.slopes(2, 0)) < 1E-8)
    assert ((modC.predict (xc(te)) - modRS.predict (x(te))).normFSq < 1E-12)

    RidgeRegression.hp("lambda") = lam                                   // restore the global lambda

end centeredRidgeMVTest2

