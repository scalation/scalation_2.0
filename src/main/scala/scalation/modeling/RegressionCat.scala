
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Jan  4 23:09:27 EST 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Regression Containing Categorical Variables
 *
 *  Convert string columns into ordinal or dummy columns.
 */

package scalation
package modeling

import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._
import scalation.theory.Variable

type MatrixI = MatrixD             // MatrixI object exists, MatrixI class uses MatrixD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExpandableVariable` trait provides the framwork for replacing categorical
 *  variables with dummy variables.  A dummy variable having nl levels is replaced
 *  with nl-1 dummy variables.
 */
trait ExpandableVariable:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the vector zt into a vector of terms/columns including dummy variables.
     *  @param zt    the vector with categorical values (at the end) to expand
     *  @param nCat  the number of categorical variables in the zt
     */
    def expand (zt: VectorD, nCat: Int): VectorD

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the vector zt, expand it and predict the response value.
     *  @param zt  the vector with categorical values (at the end) to expand
     */
    def predict_ex (zt: VectorD): Double

end ExpandableVariable


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionCat` class supports Regression that contains Categorical Variables.
 *  somtimes called  ANalysis of COVAriance (ANCOVA).  It allows the addition of 
 *  categorical (treatment) variables t into a multiple linear regression.
 *  This is done by introducing dummy variables dj to distinguish the treatment level.
 *  The problem is again to fit the parameter vector b in the augmented regression equation
 *      y  =  b dot x + e  =  b0  +  b_1   * x_1  +  b_2   * x_2  +  ... b_k * x_k
 *                                +  b_k+1 * d_1  +  b_k+2 * d_2  +  ... b_k+l * d_l + e
 *  where e represents the residuals (the part not explained by the model).
 *  Use Least-Squares (minimizing the residuals) to solve for the parameter vector b
 *  using the Normal Equations:
 *      x.t * x * b  =  x.t * y
 *      b  =  fac.solve (.)
 *  t has categorical values/levels, e.g., treatment levels (0, ... t.max)
 *  @see see.stanford.edu/materials/lsoeldsee263/05-ls.pdf
 *  @param x_      the data/input matrix of continuous variables
 *  @param t       the treatment/categorical variable matrix 
 *  @param y       the response/output vector
 *  @param fname_  the feature/variable names (defaults to null)
 *  @param hparam  the hyper-parameters (defaults to Regression.hp)
 */
class RegressionCat (x_ : MatrixD, t: MatrixI, y: VectorD, fname_ : Array [String] = null,
                     hparam: HyperParameter = Regression.hp)
      extends Regression (x_ ++^ RegressionCat.dummyVars (t), y, fname_, hparam)
         with ExpandableVariable:

    private val debug = debugf ("RegressionCat", true)                      // the flaw function
    private val flaw  = flawf ("RegressionCat")                             // the flaw function
    private val m     = y.dim                                               // number of instances
    private val (shift, tmax) = RegressionCat.get_shift_tmax                // save shift and tmax

    if x_.dim != m then flaw ("init", s"dimensions of x_ = ${x_.dim} and y = $m are incompatible")
    if t.dim  != m then flaw ("init", s"dimensions of t  = ${t.dim}  and y = $m are incompatible")

    _modelName = s"RegressionCat_${t.dim2}"

    debug ("init", s"$modelName on x_t = $getX, y = $y")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the vector zt into a vector of terms/columns including dummy variables.
     *  @param zt    the vector with categorical values (at the end) to expand
     *  @param nCat  the index at which the categorical variables start
     */
    def expand (zt: VectorD, nCat: Int = t.dim2): VectorD =
        val cat = zt.dim - nCat                                             // start index of categorical variables
        val (z, t) = zt.split (cat)
        z ++ RegressionCat.dummyVar (t.toInt, shift, tmax)
    end expand

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the vector zt, expand it and predict the response value.
     *  @param zt  the vector with categorical values (at the end) to expand
     */
    def predict_ex (zt: VectorD): Double = predict (expand (zt))

end RegressionCat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RegressionCat` companion object provides factory methods and other helper methods.
 */
object RegressionCat:

    private val debug = debugf ("RegressionCat", true)                    // debug function
    private var shift: VectorI = null                                     // for saving shift
    private var tmax:  VectorI = null                                     // for saving tmax

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionCat` object from a single data matrix.
     *  @param xt      the data/input matrix of continuous and categorical variables
     *  @param y       the response/output vector
     *  @param nCat    the index at which the categorical variables start in xt
     *                     requires the cont vars to be first, followed by the cat vars
     *  @param fname   the feature/variable names (defaults to null)
     *  @param hparam  the hyper-parameters (defualts to Regression.hp)
     */
    def apply (xt: MatrixD, y: VectorD, nCat: Int, fname: Array [String] = null,
               hparam: HyperParameter = Regression.hp): RegressionCat =
        val x = xt(?, 0 until nCat)                                        // cont vars
        val t: MatrixI = xt(?, nCat until xt.dim2)                         // cat vars
        new RegressionCat (x, t, y, fname, hparam)
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `RegressionCat` object from a continuous a data matrix and a categorical
     *  data matrix.  This method does rescaling.
     *  @param x       the data/input matrix of continuous variables
     *  @param t       the treatment/categorical variable matrix
     *  @param y       the response/output vector
     *  @param fname   the feature/variable names (defualts to null)
     *  @param hparam  the hyper-parameters (defualts to Regression.hp)
     */
    def rescale (x: MatrixD, t: MatrixI, y: VectorD, fname: Array [String] = null,
                 hparam: HyperParameter = Regression.hp): RegressionCat = ???      // FIX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the shift in categorical/treatment variables to make them start at zero
     *  as well as the maximum values after shifting.  Must call dummyVars first.
     */
    def get_shift_tmax: (VectorI, VectorI) = (shift, tmax)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assign values for the dummy variables based on the categorical/treatment vector t.
     *  @see `Variable`
     *  Note: To maintain consistency `Variable` is the only place where values for
     *  dummy variables should be set.
     *  @param t  the categorical/treatment level matrix
     */
    def dummyVars (t: MatrixI): MatrixD =
        shift  = t.min.toInt
        tmax   = t.max.toInt - shift
        val xd = new MatrixD (t.dim, tmax.sum)

        for i <- t.indices do
            var col = 0
            for j <- t.indices2 do
                val t_ij = t(i, j).toInt
                val td   = Variable.dummyVar (t_ij, shift(j), tmax(j))
                debug ("dummyVars", s"for (row $i, column $j) t_ij = $t_ij => td = $td")
                for k <- td.indices do
                    xd(i, col) = td(k); col += 1
            end for
        end for
        xd
    end dummyVars

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Assign values for dummy variables based on a single categorical/treatment
     *  vector t.
     *  @see `Variable`
     *  Note:  To maintain consistency `Variable` is the only place where values for
     *  dummy variables should be set.
     *  @param t     the categorical/treatment vector
     *  @param shft  the amount to shift the vector
     *  @param tmx   the maximum vector categorical/treatment after shifting
     */
    def dummyVar (t: VectorI, shft: VectorI = shift, tmx: VectorI = tmax): VectorD =
        val xd  = new VectorD (tmx.sum)
        var col = 0
        for j <- t.indices do
            val td = Variable.dummyVar (t(j), shft(j), tmax(j))
            for k <- td.indices do
                xd(col) = td(k); col += 1
        end for
        xd
    end dummyVar

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert a string vector into the corresponding dummy variable matrix columns.
     *  Usage:  stringVec2Dummy (VectorS ("biology", "chemistry", "physics", "biology", "physics"))
     *  @param svec  the string vector to be converted
     */
    def stringVec2Dummy (svec: VectorS): MatrixD =
        val ivec = VectorS (svec).map2Int._1                         // VectorS to VectorI
        debug ("stringVec2Dummy", s"svec = $svec -> ivec = $ivec")
        val imat = MatrixI (ivec).ᵀ                                  // VectorI as column in MatrixI 
        dummyVars (imat)                                             // MatrixD of dummy columns
    end stringVec2Dummy

end RegressionCat


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionCatTest` main function tests the `RegressionCat` class using the
 *  following regression equation.
 *      y  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2 + b_3*d_1 + b_4*d_2
 *  > runMain scalation.modeling.regressionCatTest
 */
@main def regressionCatTest (): Unit =

    // 6 data points: constant term, x_1 coordinate, x_2 coordinate
    val x = MatrixD ((6, 3), 1.0, 36.0,  66.0,                        // 6-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0, 42.0,  83.0,
                             1.0,  1.0, 101.0)
    val t = MatrixI ((6, 1), 0, 0, 1, 1, 2, 2)                        // treatments levels
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 643.0, 1598.0)       // response vector

    val z = VectorD (1.0, 20.0, 80.0, 1)                              // new instance to predict response

    println (s"x = $x")
    println (s"t = $t")
    println (s"y = $y")

    val xt = x ++^ t                                                  // .toDouble

    banner ("Regression Model")
    val reg = new Regression (xt, y)                                  // create model with with interecept (else pass x)
    println (s"xt = $xt")
    reg.trainNtest ()()                                               // train and test the model
    println (reg.summary ())                                          // parameter/coefficient statistics

    banner ("Make Predictions")
    val yp = reg.predict (z)
    println (s"predict ($z) = $yp")

    banner ("RegressionCat Model")
    val mod = new RegressionCat (x, t, y)                             // treated as categorical
    println (s"xt = ${mod.getX}")
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

    banner ("Make Predictions")
    val ze = VectorD (1.0, 20.0, 80.0, 2, 1)                          // expanded vector
    assert (ze == mod.expand (z))

    println (s"predict ($ze)   = ${mod.predict (ze)}")
    println (s"predict_ex ($z) = ${mod.predict_ex (z)}")

end regressionCatTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionCatTest2` main function tests the `RegressionCat` class using the
 *  following regression equation.
 *      y  =  b dot x  =  b_0 + b_1*x_1 + b_2*x_2 + b_3*d_1 + b_4*d_2
 *  This version needs the treatment levels to be shift down to zero.
 *  > runMain scalation.modeling.regressionCatTest2
 */
@main def regressionCatTest2 (): Unit =

    // 6 data points:        one   x_1    x_2
    val x = MatrixD ((6, 3), 1.0, 36.0,  66.0,                        // 6-by-3 matrix
                             1.0, 37.0,  68.0,
                             1.0, 47.0,  64.0,
                             1.0, 32.0,  53.0,
                             1.0, 42.0,  83.0,
                             1.0,  1.0, 101.0)
    val t = MatrixI ((6, 1), 1, 1, 2, 2, 3, 3)                        // treatments levels
    val y = VectorD (745.0, 895.0, 442.0, 440.0, 643.0, 1598.0)       // response vector
    val z = VectorD (1.0, 20.0, 80.0, 2)                              // new instance to predict response

    println (s"x = $x")
    println (s"t = $t")
    println (s"y = $y")

    val xt = x ++^ t                                                  // combine x and t

    banner ("Regression Model")
    val reg = new Regression (xt, y)                                  // treated as ordinal
    println (s"xt = $xt")
    reg.trainNtest ()()                                               // train and test the model
    println (reg.summary ())                                          // parameter/coefficient statistics
    val yp = reg.predict (z)
    println (s"predict ($z) = $yp")

    banner ("RegressionCat Model")
    val mod = new RegressionCat (x, t, y)                             // treated as categorical
    println (s"xt = ${mod.getX}")
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics
    val ze = VectorD (1.0, 20.0, 80.0, 2, 1)                          // expanded vector
    assert (ze == mod.expand (z))

    println (s"predict ($ze)   = ${mod.predict (ze)}")
    println (s"predict_ex ($z) = ${mod.predict_ex (z)}")

end regressionCatTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionCatTest3` main function tests the `RegressionCat` object related
 *  to encoding a column x1 of strings.
 *  > runMain scalation.modeling.regressionCatTest3
 */
@main def regressionCatTest3 (): Unit =

    val x1 = VectorS ("English", "French", "German", "Spanish")
    val (xe, map) = x1.map2Int                                        // map strings to integers
    println (s"map = $map")
    val xm = MatrixI (xe).ᵀ                                           // form a matrix from vector
    val xd = RegressionCat.dummyVars (xm)                             // make dummy variable columns

    println (s"encoded        xe = $xe")                              // encoded
    println (s"matrix encoded xm = $xm")                              // matrix encoded column
    println (s"matrix dummy   xd = $xd")                              // matrix dummy columns

end regressionCatTest3

import Example_AutoMPG.{oxr, y, oxr_fname}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionCatTest4` main function tests the `RegressionCat` class using the AutoMPG
 *  dataset.  Assumes no missing values.  It tests forward, backward and stepwise selection.
 *  > runMain scalation.modeling.regressionCatTest4
 */
@main def regressionCatTest4 (): Unit =


//  println (s"oxr = $oxr")
//  println (s"y   = $y")

    banner ("AutoMPG Regression")
    val mod = new Regression (oxr, y, oxr_fname)                      // create model with intercept (else pass x)
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())

    println (s"oxr_fname = ${stringOf (oxr_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                   // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${oxr.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end regressionCatTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionCatTest5` main function tests the `RegressionCat` class using the AutoMPG
 *  dataset.  Assumes no missing values.  It tests forward, backward and stepwise selection.
 *  > runMain scalation.modeling.regressionCatTest5
 */
@main def regressionCatTest5 (): Unit =

//  println (s"oxr = $oxr")
//  println (s"y   = $y")

    banner ("AutoMPG RegressionCat")
    val mod = RegressionCat (oxr, y, 6, oxr_fname)                    // create model with intercept (else pass x)
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

    banner ("Cross-Validation")
    FitM.showQofStatTable (mod.crossValidate ())

    println (s"oxr_fname = ${stringOf (oxr_fname)}")

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                   // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${oxr.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for RegressionCat with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end regressionCatTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionCatTest6` main function tests the `RegressionCat` class
 *  regarding conflicts between Dummy and Intercept columns (the Dummy Variable Trap).
 *  Compare parameters/coefficients for keeping/removing columns in matrix x.
 *  @see www.statlect.com/fundamentals-of-statistics/dummy-variable.
 *  > runMain scalation.modeling.regressionCatTest6
 */
@main def regressionCatTest6 (): Unit =

//                          one x1 d0 d1
    val x = MatrixD ((5, 4), 1,  5, 0, 1,                             // x0 = one, the intercept
                             1,  2, 0, 1,                             // x1 = years of work experience
                             1,  2, 1, 0,                             // d0 = some college (0 => false, 1 => true))
                             1,  3, 1, 0,                             // d1 = no college   (0 => false, 1 => true)
                             1, 15, 0, 1)

    val y = VectorD (65, 50, 70, 120, 80)                             // annual income in $1000

    banner ("Regression with Cat Var - Remove One Dummy")
    var mod = new Regression (x(?, 0 until 3), y)                     // create model with one dummy removed
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

    banner ("Regression with Cat Var - Remove Intercept")
    mod = new Regression (x(?, 1 until 4), y)                         // create model with intercept removed
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

    println (s"[0, 1, 0]: ${mod.predict (VectorD (0, 1, 0))}") 
    println (s"[0, 0, 1]: ${mod.predict (VectorD (0, 0, 1))}") 

    banner ("Encode Dummy Variables")
    val x_enc = VectorI (0, 0, 1, 1, 0)                               // encoded vector
    val dm    = MatrixI (x_enc).ᵀ                                     // form a dummy matrix (dummy columns) from encoded vector
    val t     = RegressionCat.dummyVars (dm)
    println (s"encoded        x_enc = $x_enc")                        // encoded vector
    println (s"matrix encoded dm    = $dm")                           // matrix encoded column
    println (s"matrix dummy   t     = $t")                            // matrix of dummy variables

    banner ("Regression with Cat Var - Use Regression")
    mod = new Regression (x(?, 0 until 2) ++^ dm, y)                  // create model from matrices x prefix and dm
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

    banner ("Regression with Cat Var - Use RegressionCat")
    mod = new RegressionCat (x(?, 0 until 2), dm, y)                  // create model from matrices x prefix and dm 
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

    banner ("Regression with Cat Var - Keep Intercept and All Dummies")
    mod = new Regression (x, y)                                       // create model with all dummies
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

end regressionCatTest6
 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `regressionCatTest7` main function tests the `RegressionCat` class' ability
 *  to handle string columns by converting them into dummy variable columns.
 *  Note, for larger datasets read from a CSV file using the `MatrixD.loadStr` method.
 *  @see `SymbolicRegression`
 *  > runMain scalation.modeling.regressionCatTest7
 */
@main def regressionCatTest7 (): Unit =

//                                            one x1  major
    val xs = Array [Array [ValueType]] (Array (1,  5, "biology"),
                                        Array (1,  2, "chemistry"),
                                        Array (1,  2, "physics"),
                                        Array (1,  3, "biology"),
                                        Array (1, 15, "physics")).transpose

    val y = VectorD (65, 50, 70, 120, 80)                             // annual income in #1000
    println (s"xs = ${stringOf (xs)}")                                // data containing strings
    println (s"y  = $y")                                              // response vector

    val x   = MatrixD (for j <- 0 to 1 yield VectorD.fromValueTypes (xs(j))).ᵀ 
    val xs2 = VectorS.fromValueTypes (xs(2))
    val dm  = RegressionCat.stringVec2Dummy (xs2)

    println (s"x  = $x")                                              // matrix of original predictors
    println (s"dm = $dm")                                             // matrix of dummy variable columns 

    banner ("Regression with Cat Var - Use RegressionCat")
    val mod = new RegressionCat (x, dm, y)                            // create model from matrices x prefix and dm 
    mod.trainNtest ()()                                               // train and test the model
    println (mod.summary ())                                          // parameter/coefficient statistics

end regressionCatTest7

