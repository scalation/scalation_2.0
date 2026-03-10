
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Dec 23 13:54:30 EST 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model: Symbolic Lasso Regression, including Quadratic and Cubic Lasso Regression
 */

package scalation
package modeling

import scala.collection.mutable.{LinkedHashSet => LSET}

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegression` class supports symbolic ridge regression that allows
 *  variables/columns to be raised to various powers, e.g., x^2, x^3, x^.5.
 *  Note, x~^p is a column-wise power function (each column raised to p-th power).
 *  IMPORTANT:  must not include INTERCEPT (column of ones) in initial data matrix),
 *  i.e., DO NOT include a column of ones in x (will cause singularity in expanded matrix).
 *  Method signatures are the as same as for `SymbolicRegression`, except there is
 *  NO intercept ARGUMENT.
 *  @param x       the initial data/input m-by-n matrix (before expansion)
 *                     must not include an intercept column of all ones
 *  @param y       the response/output m-vector
 *  @param fname   the feature/variable names
 *  @param powers  the set of powers to raise matrix x to
 *  @param hparam  the hyper-parameters (use Regression.hp for default)
 */
class SymLassoRegression (x: MatrixD, y: VectorD, fname: Array [String],
                          powers: LSET [Double], hparam: HyperParameter = LassoRegression.hp)
      extends LassoRegression (x, y, fname, hparam):

    _modelName = s"SymLassoRegression_$powers"

end SymLassoRegression


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SymLassoRegression` object provides factory methods.
 */
object SymLassoRegression:

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SymLassoRegression` object from a data matrix and a response vector.
     *  Partial support for "Symbolic Lasso Regression" as matrix x can be raised
     *  to several powers (e.g., x^1 and x^2).  Note, x^1 is automatically included.
     *  @see `SymbolicRegression.buildMatrix`
     *  @param x          the initial data/input m-by-n matrix (before expansion)
     *                        must not include an intercept column of all ones
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (defaults to null)
     *  @param intercept  whether to include the intercept term (column of ones) _1 (defaults to true)
     *  @param powers     the set of powers to raise matrix x to (defaults to null)
     *  @param cross      whether to include 2-way cross/interaction terms x_i x_j (defaults to true)
     *  @param cross3     whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param hparam     the hyper-parameters (defaults to LassoRegression.hp)
     *  @param terms      custom terms to add into the model, e.g., Array ((0, 1.0), (1, -2.0))
     *                        adds x0 x1^(-2)
     */
    def apply (x: MatrixD, y: VectorD, fname: Array [String] = null,
               powers: LSET [Double] = null, intercept: Boolean = true,
               cross: Boolean = true, cross3: Boolean = false,
               hparam: HyperParameter = LassoRegression.hp,
               terms: Array [Xj2p]*): SymLassoRegression =
        val fname_ = if fname != null then fname
                     else x.indices2.map ("x" + _).toArray                // default feature/variable names

        val (xx, f_name) = SymbolicRegression.buildMatrix (x, fname_, powers, null, intercept,
                                                           cross, cross3, terms*)
        new SymLassoRegression (xx, y, f_name, powers, hparam)
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SymLassoRegression` object from a data matrix and a response vector.
     *  This method provides data rescaling.
     *  @param x          the data/input m-by-n matrix
     *                        (augment with a first column of ones to include intercept in model)
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (defaults to null)
     *  @param intercept  whether to include the intercept term (column of ones) _1 (defaults to true)
     *  @param powers     the set of powers to raise matrix x to (defualts to null)
     *  @param cross      whether to include 2-way cross/interaction terms x_i x_j (defaults to true)
     *  @param cross3     whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param hparam     the hyper-parameters (defaults to Regression.hp)
     *  @param terms      custom terms to add into the model, e.g., Array ((0, 1.0), (1, -2.0))
     *                        adds x0 x1^(-2)
     */
    def rescale (x: MatrixD, y: VectorD, fname: Array [String] = null,
                 powers: LSET [Double] = null, intercept: Boolean = true,
                 cross: Boolean = true, cross3: Boolean = false,
                 hparam: HyperParameter = Regression.hp,
                 terms: Array [Xj2p]*): SymLassoRegression =
        val xn = normalize ((x.mean, x.stdev)) (x)
        apply (xn, y, fname, powers, intercept, cross, cross3, hparam, terms*)
    end rescale

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SymLassoRegression` object that uses multiple regression to fit a quadratic
     *  surface to the data.  For example in 2D, the quadratic regression equation is
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [x_0, x_0^2, x_1, x_1^2] + e
     *  @param x       the initial data/input m-by-n matrix (before quadratic term expansion)
     *                     must not include an intercept column of all ones
     *  @param y       the response/output m-vector
     *  @param fname   the feature/variable names (defaults to null)
     *  @param intercept  whether to include the intercept term (column of ones) _1 (defaults to true)
     *  @param cross   whether to include cross terms x_i * x_j (defaults to false)
     *  @param hparam  the hyper-parameters (defaults to LassoRegression.hp)
     */
    def quadratic (x: MatrixD, y: VectorD, fname: Array [String] = null,
                   intercept: Boolean = true, cross: Boolean = false,
                   hparam: HyperParameter = LassoRegression.hp): SymLassoRegression =
        apply (x, y, fname, LSET (1, 2), intercept, cross, false, hparam)
    end quadratic

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `SymLassoRegression` object that uses multiple regression to fit a cubic
     *  surface to the data.  For example in 2D, the cubic regression equation is
     *      y  =  b dot x + e  =  [b_0, ... b_k] dot [x_0, x_0^2, x_0^3,
     *                                                x_1, x_1^2, x_1^3,
     *                                                x_0*x_1, x_0^2*x_1, x_0*x_1^2] + e
     *  @param x          the initial data/input m-by-n matrix (before quadratic term expansion)
     *                        must not include an intercept column of all ones
     *  @param y          the response/output m-vector
     *  @param fname      the feature/variable names (defaults to null)
     *  @param intercept  whether to include the intercept term (column of ones) _1 (defaults to true)
     *  @param cross      whether to include 2-way cross/interaction terms x_i x_j (defaults to false)
     *  @param cross3     whether to include 3-way cross/interaction terms x_i x_j x_k (defaults to false)
     *  @param hparam     the hyper-parameters (defaults to LassoRegression.hp)
     */
    def cubic (x: MatrixD, y: VectorD, fname: Array [String] = null,
               intercept: Boolean = true, cross: Boolean = false, cross3: Boolean = false,
               hparam: HyperParameter = LassoRegression.hp): SymLassoRegression =
        apply (x, y, fname, LSET (1, 2, 3), intercept, cross, cross3, hparam)
    end cubic

end SymLassoRegression

import Example_AutoMPG._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Symbolic Lasso Regression", with powers specified in "LSET (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.symLassoRegressionTest
 */
@main def symLassoRegressionTest (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("AutoMPG Symbolic Lasso Regression")
    val mod = SymLassoRegression (x, y, x_fname, LSET (-2, -1, 0.5, 2))   // add cross-terms and given powers
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Symbolic Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end symLassoRegressionTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest2` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Quadratic Lasso Regression" (with cross = false) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.symLassoRegressionTest2
 */
@main def symLassoRegressionTest2 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("AutoMPG Quadratic Lasso Regression")
    val mod = SymLassoRegression.quadratic (x, y, x_fname)                // add x^2 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Quadratic Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end symLassoRegressionTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest3` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Quadratic X Lasso Regression" (with cross = true) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.symLassoRegressionTest3
 */
@main def symLassoRegressionTest3 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("AutoMPG Quadratic X Lasso Regression")
    val mod = SymLassoRegression.quadratic (x, y, x_fname, true)          // add cross-terms and x^2 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Quadratic X Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end symLassoRegressionTest3


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest4` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Cubic Lasso Regression" (with cross = false) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.symLassoRegressionTest4
 */
@main def symLassoRegressionTest4 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("AutoMPG Cubic Lasso Regression")
    val mod = SymLassoRegression.cubic (x, y, x_fname)                    // add x^2 and x^3 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Cubic Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end symLassoRegressionTest4


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest5` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Cubic X Lasso Regression" (with cross = true) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  > runMain scalation.modeling.symLassoRegressionTest5
 */
@main def symLassoRegressionTest5 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("AutoMPG Cubic X Lasso Regression")
    val mod = SymLassoRegression.cubic (x, y, x_fname, true)              // add cross-terms, x^2 and x^3 terms
    mod.trainNtest ()()                                                   // train and test the model
    println (mod.summary ())                                              // parameter/coefficient statistics

    for tech <- SelectionTech.values do 
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                       // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Cubic X Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end symLassoRegressionTest5


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest6` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests "Cubic XX Lasso Regression" (with cross, cross3 = true) and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  WARNING: setting cross3 = true can lead to an explotion of terms.
 *  > runMain scalation.modeling.symLassoRegressionTest6
 */
@main def symLassoRegressionTest6 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("AutoMPG Cubic XX Lasso Regression")
    val mod = SymLassoRegression.cubic (x, y, x_fname,                  // add x^2 and x^3 terms
                                        true, true)                     // add cross and cross3 terms
    mod.trainNtest ()()                                                 // train and test the model
    println (mod.summary ())                                            // parameter/coefficient statistics

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                     // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Cubic XX Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end symLassoRegressionTest6


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest7` main function tests the `SymLassoRegression`
 *  object using the AutoMPG dataset.  Assumes no missing values.
 *  It tests custom "Symbolic Lasso Regression", with powers specified in "LSET (...)" and
 *  applies forward selection, backward elimination, or stepwise regression.
 *  This test case performs data rescaling.
 *  > runMain scalation.modeling.symLassoRegressionTest7
 */
@main def symLassoRegressionTest7 (): Unit =

//  println (s"x = $x")
//  println (s"y = $y")

    banner ("AutoMPG Symbolic Lasso Regression")
    val mod = SymLassoRegression.rescale (x, y, x_fname,
                                          LSET (-2, -1, 0.5, 2))         // add cross-terms and given powers
    mod.trainNtest ()()                                                  // train and test the model
    println (mod.summary ())                                             // parameter/coefficient statistics

    for tech <- SelectionTech.values do
        banner (s"Feature Selection Technique: $tech")
        val (cols, rSq) = mod.selectFeatures (tech)                      // R^2, R^2 bar, R^2 cv
        val k = cols.size
        println (s"k = $k, n = ${x.dim2}")
        new PlotM (null, rSq.ᵀ, Regression.metrics, s"R^2 vs n for Symbolic Lasso Regression with $tech", lines = true)
        println (s"$tech: rSq = $rSq")
    end for

end symLassoRegressionTest7


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest8` main function tests the `SymLassoRegression`
 *  object using a simulated gravity dataset.
 *  It tests custom "Symbolic Regression", with a custom term: x0 x1^(-2)
 *  FIX - acquire a real gravity dataset
 *  > runMain scalation.modeling.symLassoRegressionTest8
 */
@main def symLassoRegressionTest8 (): Unit =

    import random.{Uniform, Normal}

    val noise = Normal (0, 10)                                           // random noise
    val rad   = Uniform (6370, 7000)                                     // distance from the center of the Earth in km
    val mas   = Uniform (50, 150)                                        // mass of person

    val m1 = 5.97219E24                                                  // mass of Earth in kg
    val G  = 6.67408E-11                                                 // gravitational constant in m^3 kg^-1 s^-2

    val xy = new MatrixD (100, 3)                                        // simulated gravity data
    for i <- xy.indices do
        val m2 = mas.gen                                                 // unit of kilogram (kg)
        val r  = 1000 * rad.gen                                          // unit of meter (m)
        xy(i, 0) = m2                                                    // mass of person
        xy(i, 1) = r                                                     // radius/distance
        xy(i, 2) = G * m1 * m2 / r~^2 + noise.gen                        // force of gravity
    end for

    val fname = Array ("mass2", "radius")

    println (s"xy = $xy")
    val (x, y) = (xy.not (?, 2), xy(?, 2))

    banner ("Newton's Universal Gravity Symbolic Lasso Regression")
    val mod = SymLassoRegression (x, y, fname, null, false, false,
              terms = Array ((0, 1.0), (1, -2.0)))                       // add one custom term

    mod.trainNtest ()()                                                  // train and test the model
    println (mod.summary ())                                             // parameter/coefficient statistics
    println (s"b =~ GM = ${G * m1}")                                     // Gravitational Constant * Mass of the Earth

end symLassoRegressionTest8


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `symLassoRegressionTest9` main function tests the `SymLassoRegression`
 *  object using a simple dataset to compare Lasso Regression, Quadratic Lasso Regression
 *  and Cubic Lasso Regression.
 *  > runMain scalation.modeling.symLassoRegressionTest9
 */
@main def symLassoRegressionTest9 (): Unit =

    val x  = VectorD (1, 2, 3, 4, 5)
    val y  = VectorD (1, 3, 3, 5, 4)

    val mu_x = x.mean                                                      // center the data
    val mu_y = y.mean
    val x_c  = x - mu_x
    val y_c  = y - mu_y

    val xx = MatrixD (x_c).ᵀ

    banner ("Lasso Regression")
    var mod = new LassoRegression (xx, y_c)
    mod.trainNtest ()()
    println (mod.summary ())
    var yp  = mod.predict (mod.getX)
    var yp2 = yp + mu_y
    new Plot (null, y_c, yp, s"${mod.modelName} y_c vs yp", lines = true)
    new Plot (null, y,  yp2, s"${mod.modelName} y vs yp2", lines = true)

    banner ("Quadratic Lasso Regression")
    val fname = Array ("x")
    mod = SymLassoRegression.quadratic (xx, y_c, fname)
    mod.trainNtest ()()
    println (mod.summary ())
    yp  = mod.predict (mod.getX)
    yp2 = yp + mu_y
    new Plot (null, y_c, yp, s"${mod.modelName} y_c vs yp", lines = true)
    new Plot (null, y,  yp2, s"${mod.modelName} y vs yp2", lines = true)

    banner ("Cubic Lasso Regression")
    mod = SymLassoRegression.cubic (xx, y_c, fname)
    mod.trainNtest ()()
    println (mod.summary ())
    yp  = mod.predict (mod.getX)
    yp2 = yp + mu_y
    new Plot (null, y_c, yp, s"${mod.modelName} y_c vs yp", lines = true)
    new Plot (null, y,  yp2, s"${mod.modelName} y vs yp2", lines = true)

end symLassoRegressionTest9

