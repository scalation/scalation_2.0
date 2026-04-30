
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Mustafa Nural
 *  @version 2.0
 *  @date    Sat Jan 20 15:41:27 EST 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Transformed Multiple Linear Regression (Transforms y)
 *
 *  Delegates transform (y) via y.map (tran) to Regression
 *
 *  @see data.princeton.edu/wws509/notes/c2s10.html
 *  @see scala-lang.org/api/3.x/scala/math.html
 *  @see `scalation.CommonFunctions`
 *
 *  Common transformation pairs: (tran, itran) or reversed
 *  (log, exp), (log1p, expm1), (ihs, sinh), (cbrt, cb), (sqrt, sq), (box_cox, cox_box)
 */

// FIX - remove once TranRegression is validated

package scalation
package modeling

import scala.collection.mutable.IndexedSeq
import scala.math.{exp, expm1, log, log1p, sqrt}

import scalation.mathstat._
import scalation.random.Normal

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TranRegression2` class supports transformed multiple linear regression.
 *  In this case, 'x' is multi-dimensional [1, x_1, ... x_k].  Fit the parameter
 *  vector 'b' in the transformed regression equation
 *      tran (y)  =  b dot x + e  =  b_0 + b_1 * x_1 +  b_2 * x_2 ... b_k * x_k + e
 *  where 'e' represents the residuals (the part not explained by the model) and
 *  'tran' is the function (defaults to log1p) used to transform the response vector 'y'.
 *  Common transforms include 'log (y)', 'sqrt (y)' when 'y > 0', or even 'sq (y)', 'exp (y)'.
 *  More generally, a Box-Cox Transformation may be applied.
 *  @see citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.469.7176&rep=rep1&type=pdf
 *  Use Least-Squares (minimizing the residuals) to fit the parameter vector 'b'
 *  Note: this class does not provide transformations on columns of matrix 'x'.
 *  @see www.ams.sunysb.edu/~zhu/ams57213/Team3.pptx
 *  @param x       the data/input m-by-n matrix
 *  @param y       the response/output m-vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters (defaults to Regression.hp)
 *  @param tran    the transformation function (defaults to log1p)
 *  @param itran   the inverse transformation function to rescale predictions to original y scale (defaults to expm1)
 */
class TranRegression2 (x: MatrixD, y: VectorD, fname_ : Array [String] = null,
                      hparam: HyperParameter = Regression.hp,
                      tran: FunctionS2S = log1p, itran: FunctionS2S = expm1)   // FIX - use Transform class
      extends Regression (x, y.map (tran), fname_, hparam):                // transform y and pass to `Regression`

    private val debug = debugf ("TranRegression2", true)                    // debug function
    private val flaw  = flawf ("TranRegression2")                           // flaw function
    private val inf   = getY.findInfinity                                  // infinite transformed response elements

    _modelName = "TranRegression2"

    if ! inf.isEmpty then flaw ("init", s"the transformed response vector has infinite elements at $inf")
    if ! y.isNonnegative then
        throw new IllegalArgumentException ("y must be positive for transformed regression (log, sqrt)")
                                            // FIX - may work for other transformations

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.  Test using the TRANSFORMED DATA.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    def test0 (x_ : MatrixD = x, y_ : VectorD = getY): (VectorD, VectorD) =
        val yp = x_ * b                                                  // make predictions
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
    end test0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test a predictive model y_ = f(x_) + e and return its QoF vector.
     *  Testing may be be in-sample (on the training set) or out-of-sample
     *  (on the testing set) as determined by the parameters passed in.
     *  Note: must call train before test.  Test using the ORIGINAL DATA.
     *  @param x_  the testing/full data/input matrix (defaults to full x)
     *  @param y_  the testing/full response/output vector (defaults to full y)
     */
    override def test (x_ : MatrixD = x, y_ : VectorD = y): (VectorD, VectorD) =
        val yp = (x_ * b).map (itran)                                    // make predictions
        (yp, diagnose (y_, yp))                                          // return predictions and QoF vector
    end test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Train and test the predictive model y_ = f(x_) + e and report its QoF
     *  and plot its predictions.
     *  Currently must override if y is transformed, @see `Predictor`
     *  @param x_  the training/full data/input matrix (defaults to full x)
     *  @param y_  the training/full response/output vector (defaults to full y)
     *  @param xx  the testing/full data/input matrix (defaults to full x)
     *  @param yy  the testing/full response/output vector (defaults to full y)
     */
    override def trainNtest (x_ : MatrixD = x, y_ : VectorD = getY)
                            (xx: MatrixD = x, yy: VectorD = y): (VectorD, VectorD) =
        train (x_, y_)
        debug ("trainNTest", s"b = $b")
        val (yp, qof) = test (xx, yy)
        println (report (qof))
        Predictor.plotPrediction (yy, yp, modelName, false)
        Predictor.plotPrediction (yy, yp, modelName)                     // reordered
        (yp, qof)
    end trainNtest

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z,
     *  e.g., (b_0, b_1, b_2) dot (1, z_1, z_2).
     *  @param z  the new vector to predict
     */
    override def predict (z: VectorD): Double = itran (b dot z)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Predict the value of y = f(z) by evaluating the formula y = b dot z for
     *  each row of matrix x_.
     *  @param z  the new matrix to predict
     */
    override def predict (x_ : MatrixD): VectorD = (x_ * b).map (itran)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform In-Sample Testing, i.e., train and test on the same FULL data set.
     *  Train on transformed y (via getY), test on original y.
     *  Return the prediction and the Quality of Fit.
     *  @param skip    the number of initial data points to skip (due to insufficient information)
     *  @param showYp  whether to show the prediction vector
     */
    override def inSample_Test (skip: Int = 0, showYp: Boolean = false): (VectorD, VectorD) =
        banner (s"In-Sample Test: $modelName")
        val (x_, y__, y_) = (x.drop (skip), getY.drop (skip), y.drop (skip))
        val (yp, qof) = trainNtest (x_, y__)(x_, y_)
        if showYp then
            println (s"Final In-Sample Prediction Vector yp = $yp")
        (yp, qof)
    end inSample_Test

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Use validation to compute test Quality of Fit (QoF) measures by dividing
     *  the full dataset into a TESTING-set and a TRAINING-set.
     *  The testing-set is defined by idx and the rest of the data is the training-set.
     *  @note:  currently must override if y is transformed, @see `Predictor`
     *  @see `modeling.Predictor.validate` about the RANDOM, FIRST, and LAST options
     *  for selecting the testing-set.
     *  @param rando  flag indicating whether to use randomized or simple validation
     *  @param ratio  the ratio of the TESTING-set to the full dataset (most common 70-30 (.3), 80-20 (.2))
     *  @param idx    the prescribed TESTING-set indices (default => generate)
     */
    override def validate (rando: Boolean = true, ratio: Double = Model.TE_RATIO)
//                        (idx: IndexedSeq [Int] = testIndices ((ratio * y.dim).toInt, rando)):
                          (idx: IndexedSeq [Int] = testIndices (y.dim, (ratio * y.dim).toInt, rando)):
                          (VectorD, VectorD) =
        val (x_e, x_, y_e, y_) = TnT_Split (x, y, idx)                       // Test-n-Train Split

        train (x_, y_.map (tran (_)))                                        // train model on the TRAINING-set
        val (yp, qof) = test (x_e, y_e)                                      // test on TESTING-set and get QoF measures
        if qof(QoF.sst.ordinal) <= 0.0 then                                  // requires variation in TESTING-set
            flaw ("validate", "chosen testing set has no variability")
        println (FitM.fitMap (qof, QoF.values.map (_.toString)))
        (yp, qof)
    end validate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Build a sub-model that is restricted to the given columns of the data matrix.
     *  @param x_cols  the columns that the new model is restricted to
     *  @param fname2  the variable/feature names for the new model (defaults to null)
     */
    override def buildModel (x_cols: MatrixD, fname2: Array [String] = null): TranRegression2 =
        debug ("buildModel", s"${x_cols.dim} by ${x_cols.dim2}")
        new TranRegression2 (x_cols, y, fname2, hparam, tran, itran)
    end buildModel

end TranRegression2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TranRegression2` companion object provides transformation and inverse
 *  transformation function based on the parameter λ.
 *  It support the family of Box-Cox and Yeo-Johnson Transformations.
 */
object TranRegression2:

    private val debug = debugf ("TranRegression2", false)                  // debug function
    private var _λ    = 0.5                                               // the power parameter for transformations

    inline def λ: Double = _λ                                             // getter-setter for λ
    def λ_= (λ_ : Double): Unit = _λ = λ_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transform y using the Box-Cox transformation.
     *  @param y  the value to be transformed
     */
    def box_cox (y: Double): Double =
        if λ == 0.0 then log (y)
        else (y ~^ λ - 1) / λ
    end box_cox

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inverse transform z using the Box-Cox transformation.
     *  @param z  the value to be inverse transformed
     */
    def cox_box (z: Double): Double =
        if λ == 0.0 then exp (z)
        else (λ * z + 1) ~^ (1 / λ)
    end cox_box

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Transform y using the Yeo-Johnson transformation.
     *  @param y  the value to be transformed
     */
    def yeo_john (y: Double): Double =
        if y >= 0.0 then
            if λ == 0.0 then log (1+y)
            else ((1+y) ~^ λ - 1) / λ
        else
            if λ == 2.0 then -log (1-y)
            else { val _2λ = 2-λ; -((1-y) ~^ _2λ - 1) / _2λ }
    end yeo_john

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Inverse transform z using the Yeo-Johnson transformation.
     *  @param z  the value to be inverse transformed
     */
    def john_yeo (z: Double): Double =
        if z >= 0.0 then
            if λ == 0.0 then exp (z) - 1
            else (λ*z + 1) ~^ (1/λ) - 1
        else
            if λ == 2.0 then 1 - exp (-z)
            else { val _2λ = 2-λ; 1 - (-_2λ*z + 1) ~^ (1/_2λ) }
    end john_yeo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TranRegression2` object that uses a Box-Cox Transformation.
     *  To change λ from its default value, call setter first.
     *  @param x       the data/input matrix
     *  @param y       the response/output vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     */
    def apply (x: MatrixD, y: VectorD, fname: Array [String] = null,
               hparam: HyperParameter = Regression.hp): TranRegression2 =
        new TranRegression2 (x, y, fname, hparam, box_cox, cox_box)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TranRegression2` object that uses a Yeo-Johnson Transformation.
     *  To change λ from its default value, call `set_lambda` first.
     *  @param x       the data/input matrix
     *  @param y       the response/output vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defaults to Regression.hp)
     */
    def app_yj (x: MatrixD, y: VectorD, fname: Array [String] = null,
               hparam: HyperParameter = Regression.hp): TranRegression2 =
        new TranRegression2 (x, y, fname, hparam, yeo_john, john_yeo)
    end app_yj

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TranRegression2` with automatic rescaling from a combined data matrix.
     *  @param xy       the combined data/input and response/output matrix
     *  @param fname    the feature/variable names
     *  @param hparam   the hyper-parameters
     *  @param tran     the transformation function
     *  @param itran    the inverse transformation function to rescale predictions to original y scale
     *  @param bounds   the bounds for rescaling
     */
    def apply (xy: MatrixD, fname: Array [String],
               hparam: HyperParameter, tran: FunctionS2S, itran: FunctionS2S,
               bounds: (Double, Double)): TranRegression2 =
        val hp2 = if hparam == null then Regression.hp else hparam
        val (x, y) = (xy.not(?, xy.dim2-1), xy(?, xy.dim2-1))

        val y_s =                                                          // scaled version of y
            if bounds != null then                                         // scale to bounds
                val extrem = extreme (y)
                scaleV (extrem, bounds)(y)
            else                                                           // normalize
               val (mu_y, sig_y) = (y.mean, y.stdev)
               normalizeV (mu_y, sig_y)(y)

        debug ("apply", s"scaled: scaled y = $y_s")
        new TranRegression2 (x, y_s, fname, hp2, tran, itran)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `TranRegression2` with automatic rescaling from a data matrix and
     *  response vector.
     *  @param x       the data/input matrix
     *  @param y       the response/output vector
     *  @param fname   the feature/variable names
     *  @param hparam  the hyper-parameters
     *  @param tran    the transformation function (defaults to log)
     *  @param itran   the inverse transformation function to rescale predictions to original y scale
     *  @param bounds  the bounds for rescaling
     */
    def apply (x: MatrixD, y: VectorD, fname: Array [String],
               hparam: HyperParameter, tran: FunctionS2S, itran: FunctionS2S,
               bounds: (Double, Double)): TranRegression2 =
        val hp2 = if hparam == null then Regression.hp else hparam

        val y_s =                                                         // scaled version of y
            if bounds != null then                                        // scale to bounds
                val extrem = extreme (y)
                scaleV (extrem, bounds)(y)
            else                                                          // normalize
                val (mu_y, sig_y) = (y.mean, y.stdev)
                normalizeV (mu_y, sig_y)(y)

        debug ("apply", s"scaled: scaled y = $y_s")
        new TranRegression2 (x, y_s, fname, hp2, tran, itran)
    end apply

end TranRegression2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TranRegression2Ex` provides a sample dataset for testing purposes.
 *  Move the comments on the line used to generate the response y(k) to test
 *  1D and 2D cases.
 */
object TranRegression2Ex:

    private val cap    = 30
    private val rng    = 0 until cap
    private val (m, n) = (cap * cap, 3)
    private val err    = Normal (0, cap)

    val x = new MatrixD (m, n)
    val y = new VectorD (m)
    for i <- rng; j <- rng do x(cap * i + j) = VectorD (1, i, j)
    for k <- y.indices do y(k) = sq (10 + 2 * x(k, 1) + err.gen)
//  for k <- y.indices do y(k) = sq (10 + 2 * x(k, 1) + 0.3 * x(k, 2) + err.gen)

end TranRegression2Ex


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test` main function tests `TranRegression2` class using the
 *  following regression equation.
 *      log (y)  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  > runMain scalation.modeling.tranRegression2Test
 */
@main def tranRegression2Test (): Unit =

    val x = MatrixD ((5, 3), 1.0, 36.0,  66.0,                // 5-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0,  1.0, 101.0)
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 1598.0)
    val z = VectorD (1.0, 20.0, 80.0)

    println ("x = " + x)
    println ("y = " + y)

    banner ("Parameter Estimation and Quality of Fit - log transform")
    val mod = new TranRegression2 (x, y)
    mod.inSample_Test ()                                      // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("Quality of Fit - based on transformed data")
    println (mod.report (mod.test0 ()._2))

    banner ("Prediction")
    val yp = mod.predict (x)
    println (s"predict (x) = $yp")

    val yp2 = mod.predict (z)
    println (s"predict ($z) = $yp2")

end tranRegression2Test


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test2` main function tests `TranRegression2` class using the
 *  following regression equation.
 *      sqrt (y)  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  > runMain scalation.modeling.tranRegression2Test2
 */
@main def tranRegression2Test2 (): Unit =

    // 9 data points:    Constant    x1    x2     y
    val xy = MatrixD ((9, 4), 1.0,  1.0,  1.0,  0.04,
                              1.0,  2.0,  1.0,  0.05,
                              1.0,  3.0,  1.0,  0.06,

                              1.0,  1.0,  2.0,  0.10,
                              1.0,  2.0,  2.0,  0.11,
                              1.0,  3.0,  2.0,  0.12,

                              1.0,  1.0,  3.0,  0.20,
                              1.0,  2.0,  3.0,  0.21,
                              1.0,  3.0,  3.0,  0.22)

    val x_fname = Array ("one", "x1", "x2")

    val (x, y) = (xy.not (?, 3), xy(?, 3))
    val xtx    = x.ᵀ * x
    val yy     = y.map (sqrt)
    val xtyy   = x.ᵀ * yy
    val b      = new Fac_Cholesky (xtx).inverse * xtyy

    banner ("parameters")
    println (s"xtx  = $xtx")
    println (s"xtyy = $xtyy")
    println (s"b    = $b")

    val yyp  = x * b                      // transformed
    val sst  = (yy - yy.mean).normSq
    val e    = yy - yyp
    val sse  = e.normSq
    val rSq  = 1.0 - sse / sst

    banner ("transformed")
    println (s"yy   = $yy")
    println (s"yyp  = $yyp")
    println (s"e    = $e")
    println (s"sst  = $sst")
    println (s"sse  = $sse")
    println (s"rSq  = $rSq")

    banner ("original")
    val yp   = yyp.map (sq)               // orginal
    val sst2 = (y - y.mean).normSq
    val e2   = y - yp
    val sse2 = e2.normSq
    val rSq2  = 1.0 - sse2 / sst2

    println (s"y    = $y")
    println (s"yp   = $yp")
    println (s"e2   = $e2")
    println (s"sst2 = $sst2")
    println (s"sse2 = $sse2")
    println (s"rSq2 = $rSq2")

    banner ("Parameter Estimation and Quality of Fit")
    val mod = new TranRegression2 (x, y, x_fname, Regression.hp, sqrt, sq)
    mod.inSample_Test ()                                           // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics

end tranRegression2Test2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test3` main function tests `TranRegression2` class using the
 *  following regression equation and uses the simulated data in `TranRegression2Ex`.
 *      sqrt (y)  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  @see 5.13.9 exercises 1, 2, and 3.
 *  > runMain scalation.modeling.tranRegression2Test3
 */
@main def tranRegression2Test3 (): Unit =

    import TranRegression2Ex.{x, y}

    // Phase 1 ================================================================
    banner ("Regression prediction yp")
    val reg = new Regression (x, y)
    reg.inSample_Test ()                                           // train and test the model
    println (reg.summary ())                                       // parameter/coefficient statistics

    val sst = reg.fit (QoF.sst.ordinal)

    val yp = reg.predict (x)
    val e  = y - yp

    new Plot (null, y, yp, "Original Regression y and yp vs. t", lines = true)
    new Plot (null, e, null, "Original e vs. t", lines = true)

    // Phase 2 ================================================================
    banner ("Transform y to y2")
    val y2  = y.map (sqrt)
    val trg = new Regression (x, y2)
    trg.inSample_Test ()                                           // train and test the model
    println (trg.summary ())                                       // parameter/coefficient statistics

    val yp2 = trg.predict (x)
    val e2  = y2 - yp2

    new Plot (null, y2, yp2, "Transformed Regression y2 and yp2 vs. t", lines = true)
    new Plot (null, e2, null, "Transformed e2 vs. t", lines = true)

    // Phase 3 ================================================================
    banner ("Inverse Transform yp2 to yp3")
    val yp3 = yp2.map (sq)
    val e3  = y - yp3

    val sse = e3 dot e3
    println (s"R^2 = ${1 - (sse / sst)}")

    new Plot (null, y, yp3, "Tran-back Regression y and yp3 vs. t", lines = true)
    new Plot (null, e3, null, "Tran-back e3 vs. t", lines = true)

    val ys2 = MatrixD (y2, yp2)
    val ys3 = MatrixD (y, yp3, yp)
    new PlotM (null, ys2.ᵀ, null, "Transformed", lines = true)
    new PlotM (null, ys3.ᵀ, null, "Tran-back", lines = true)

end tranRegression2Test3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test4` main function tests `TranRegression2` class using the
 *  following regression equation.
 *      sqrt (y)  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  > runMain scalation.modeling.tranRegression2Test4
 */
@main def tranRegression2Test4 (): Unit =

    import TranRegression2Ex.{x, y}

    banner ("Regression")
    val reg = new Regression (x, y)
    reg.inSample_Test ()                                      // train and test the model
    println (reg.summary ())                                  // parameter/coefficient statistics

    val yp = reg.predict (x)
    val e  = y - yp

    banner ("TranRegression2 with sqrt")
    val mod = new TranRegression2 (x, y, null, Regression.hp, sqrt, sq)
    mod.inSample_Test ()                                      // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("Quality of Fit - based on transformed data")
    println (mod.report (mod.test0 ()._2))                    // test on transformed data

    val yp2 = mod.predict (x)
    val e2  = y - yp2

    println (s"e2.dim = ${e2.dim}")

    val ys = MatrixD (y, yp, yp2)
    new PlotM (null, ys.ᵀ, lines = true)

    new Plot (null, e, null, "e vs. t", lines = true)
    new Plot (null, e2, null, "e2 vs. t", lines = true)

end tranRegression2Test4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test5` main function tests `TranRegression2` class using the
 *  following regression equation.
 *      sigmoid^{-1} (y)  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2.
 *  > runMain scalation.modeling.tranRegression2Test5
 */
@main def tranRegression2Test5 (): Unit =

    import Example_AutoMPG._

    def f (u: Double): Double  = -log (1/u - 1)               // transform
    def fi (t: Double): Double = 1 / (1 + exp (-t))           // inverse transform

    val extrem = extreme (y)                                  // (min, max) for y
    val bounds = (0.01, 0.99)                                 // transform function domain bounds
 
    val yy = scaleV (extrem, bounds)(y)                       // rescale to domain of transform
    println (s"yy = $yy")

    banner ("Regression")
    val reg = new Regression (ox, yy)
    reg.inSample_Test ()                                      // train and test the model
    println (reg.summary ())                                  // parameter/coefficient statistics

    val yp = reg.predict (ox)
    val e  = yy - yp

    banner ("TranRegression2")
//  val mod = new Regression (ox, yy.map (f))                         // rescale & transform
    val mod = new TranRegression2 (ox, yy,  ox_fname, Regression.hp, f, fi)   // rescale
    mod.inSample_Test ()                                      // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics

    banner ("Quality of Fit - based on transformed data")
    println (mod.report (mod.test0 ()._2))                    // test with transformed

    val yp2 = mod.predict (ox)
    val e2  = yy - yp2

    val yp_  = scaleV (bounds, extrem)(yp)
    val yp2_ = scaleV (bounds, extrem)(yp2)

    val rnk  = y.iqsort                                       // rank order for vector y
    val ry   = y.reorder (rnk)                                // actual - red
    val ryp  = yp_.reorder (rnk)                              // Regression - green
    val ryp2 = yp2_.reorder (rnk)                             // TranRegression2 - blue

    val ys = MatrixD (ry, ryp, ryp2)
    new PlotM (null, ys.ᵀ, lines = true)

    new Plot (null, e, null, "e vs. t", lines = true)
    new Plot (null, e2, null, "e2 vs. t", lines = true)

end tranRegression2Test5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test6` main function tests `TranRegression2` class using the
 *  following regression equation on the beer foam dataset.
 *  @see www.tf.uni-kiel.de/matwis/amat/iss/kap_2/articles/beer_article.pdf
 *      exp (y)  =  b dot x  =  b_0 + b_1*x_1.
 *  > runMain scalation.modeling.tranRegression2Test6
 */
@main def tranRegression2Test6 (): Unit =

    val x1 = VectorD (0, 15, 30, 45, 60, 75, 90, 105, 120, 150, 180, 210, 250, 300, 360)
    val y  = VectorD (14.0, 12.1, 10.9, 10.0, 9.3, 8.6, 8.0, 7.5,
                       7.0,  6.2,  5.5, 4.5, 3.5, 2.0, 0.9)
    val _1 = VectorD.one (x1.dim)
    val x  = MatrixD (_1, x1).ᵀ

    banner ("SimpleRegression")
    val reg = new SimpleRegression (x, y)
    reg.inSample_Test ()                                      // train and test the model
    println (reg.summary ())                                  // parameter/coefficient statistics
    val yp = reg.predict (x)
    new Plot (x1, y, yp, "SimpleRegression", lines = true)

    banner ("TranRegression2 (log)")
    val mod = new TranRegression2 (x, y)
    mod.inSample_Test ()                                      // train and test the model
    println (mod.summary ())                                  // parameter/coefficient statistics
    val yp2 = mod.predict (x)
    new Plot (x1, y, yp2, "TranRegression2", lines = true)

    banner ("ExpRegression")
    val erg = new ExpRegression (x, y)
    erg.inSample_Test ()                                      // train and test the model
    println (erg.summary ())                                  // parameter/coefficient statistics
    val yp3 = erg.predict (x)
    new Plot (x1, y, yp3, "ExpRegression", lines = true)

end tranRegression2Test6


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test7` main function tests the `TranRegression2` class using
 *  the AutoMPG dataset.  It also combines feature selection with cross-validation
 *  and plots R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.modeling.tranRegression2Test7
 */
@main def tranRegression2Test7 (): Unit =

    import Example_AutoMPG._
    banner ("AutoMPG TranRegression2 feature selection")

//  val f = (id,    id,    "id")
//  val f = (recip, recip, "recip")
//  val f = (log,   exp,   "log")
//  val f = (log1p, expm1, "log1p")
//  val f = (sqrt,  sq,    "sqrt")
//  val f = (sq,    sqrt,  "sq")
//  val f = (exp,   log,   "exp")
    import TranRegression2.{box_cox, cox_box}
    TranRegression2.λ = 0.4; val f = (box_cox, cox_box, "box_cox")   // try 0.2, 0.3, 0.4, 0.5, 0.6

    banner (s"TranRegression2 with ${f._3} transform")
    val mod = new TranRegression2 (ox, y, ox_fname, Regression.hp, f._1, f._2)
    mod.inSample_Test ()                                           // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics

    banner ("Validation Test")
    mod.validate ()()

/*
    banner ("Forward Selection Test")
    val (cols, rSq) = mod.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
    val k = cols.size
    println (s"k = $k, n = ${ox.dim2}")
    new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for TranRegression2 ${f._3}", lines = true)
    println (s"rSq = $rSq")
*/

end tranRegression2Test7


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test8` main function tests and compares `Regression` vs.
 *  `SymbolicRegression.quadratic` vs. `TranRegression2`.
 *  using the following regression equations.
 *      y = b dot x  = b_0 + b_1*x
 *      y = b dot x' = b_0 + b_1*x + b_2*x^2
 *      y = b dot x' = b_0 + b_1*x + b_2*x^2 + b_3*x^3
 *  > runMain scalation.modeling.tranRegression2Test8
 */
@main def tranRegression2Test8 (): Unit =

    // 8 data points:         x   y
    val xy = MatrixD ((8, 2), 1,  2,                              // 8-by-2 matrix
                              2,  5,
                              3, 10,
                              4, 15,
                              5, 20,
                              6, 30,
                              7, 50,
                              8, 60)

    val x_fname  = Array ("x")                                     // names of features/variables
    val ox_fname = Array ("_1", "x")                               // names of features/variables

    println ("model: y = b0 + b1*x1 + b2*x1^2")
    println (s"xy = $xy")

    val oxy     = VectorD.one (xy.dim) +^: xy                      // combined data matrix with ones column prepended
    val (ox, y) = (oxy.not(?, 2), oxy(?, 2))                       // (data matrix, response column)
    val x       = xy.not(?, 1)                                     // data matrix with no ones column

    banner ("Regression")
    val reg = Regression (oxy, ox_fname)()                         // create a Regression model
    reg.inSample_Test ()                                           // train and test the model
    println (reg.summary ())                                       // parameter/coefficient statistics
    val yp = reg.predict (ox)                                      // y predicted for Regression
    println (s"predict = $yp")

    banner ("Quadratic Regression")
    val qrg = SymbolicRegression.quadratic (x, y, x_fname)         // create a Quadratic Regression model
    qrg.inSample_Test ()                                           // train and test the model
    println (qrg.summary ())                                       // parameter/coefficient statistics
    val yp2 = qrg.predict (qrg.getX)                               // y predicted for Quadratic Regression
    println (s"predict = $yp2")

    banner ("Transformed Regression")
    val mod = new TranRegression2 (ox, y, ox_fname, Regression.hp, sqrt, sq)   // sqrt Transformed Regression model
    mod.inSample_Test ()                                           // train and test the model
    println (mod.summary ())                                       // parameter/coefficient statistics
    val yp3 = mod.predict (ox)                                     // y predicted for Transformed Regression
    println (s"predict = $yp2")

    val mat = MatrixD (y, yp, yp2, yp3)
    println (s"mat = $mat")
    new PlotM (null, mat, Array ("actual", "linear", "quad", "sqrt"),
                          "y vs. yp vs. yp2 vs. yp3", true)

    banner ("Expanded Form")
    println (s"expanded x = ${mod.getX}")
    println (s"y = ${mod.getY}")

end tranRegression2Test8


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `tranRegression2Test9` main function tests the Box-Cox and Yeo-Johnson Transformations.
 *  > runMain scalation.modeling.tranRegression2Test9
 */
@main def tranRegression2Test9 (): Unit =

    import TranRegression2._

    banner ("Text Transformations on Positive Values")
    var y = VectorD.range (1 until 11)
    println (s"y  = $y")

    banner ("Positive Values -- Box-Cox Transformation and Inverse: y -> z -> y'")
    var z = y.map (box_cox (_))
    println (s"z  = $z")
    println (s"y' = ${z.map (cox_box (_))}")

    banner ("Positive Values -- Yeo-Johnson Transformation and Inverse: y -> z -> y'")
    z = y.map (yeo_john (_))
    println (s"z  = $z")
    println (s"y' = ${z.map (john_yeo (_))}")

    banner ("Text Transformations on Negative Values")
    y = -y
    println (s"y  = $y")

    banner ("Negative Values -- Box-Cox Transformation and Inverse: y -> z -> y'")
    z = y.map (box_cox (_))
    println (s"z  = $z")
    println (s"y' = ${z.map (cox_box (_))}")

    banner ("Negative Values -- Yeo-Johnson Transformation and Inverse: y -> z -> y'")
    z = y.map (yeo_john (_))
    println (s"z  = $z")
    println (s"y' = ${z.map (john_yeo (_))}")
 
end tranRegression2Test9

