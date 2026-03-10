
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Yousef Fekri Dabanloo
 *  @version 2.0
 *  @date    Thu Mar 13 14:06:11 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Support for Common Transformation Functions with their Inverse
 *  @see     `modeling.TranRegression` for `box_cox` and `yeo_john` transformations
 *
 *  https://www.infoq.com/news/2023/10/foreign-function-and-memory-api/
 */

package scalation
package mathstat

import scala.annotation.unused

import VectorDOps._

type VecMat = VectorD | MatrixD

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `⚬` extension method performs function composition f ⚬ g.
 *  The composition f ⚬ g = f (g (.)) maps A -g-> B -f-> R.
 *  @see www.scala-lang.org/api/current/scala/Function1.html
 *  @tparam A  the type to which function `g` can be applied
 *  @tparam B  the type to which function `f` can be applied
 *  @tparam R  the return type for f ⚬ g
 *  @param  f  the function from domain B to range R
 */
extension [A, B, R] (f: B => R)

    def ⚬ (g: A => B): A => R = f compose g


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `mu_sig` and `min_max` extend the methods defined for vectors and matrices
 *  to work for types that are either `VectorD` or `MatrixD`.
 *  @param x  the argument that is either a `VectorD` or `MatrixD`.
 */
extension (x: VecMat)

    def mu_sig: VecMat =
        x match
            case xV: VectorD => xV.mu_sig
            case xM: MatrixD => xM.mu_sig

    def min_max: VecMat =
        x match
            case xV: VectorD => xV.min_max
            case xM: MatrixD => xM.min_max


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Transform` object defines hyper-parameters for Transforms.
 */
object Transform:

    /** Base hyper-parameter specification for regression based time series models.
     */
    val hp = new HyperParameter
    hp += ("a",   1.0,   1.0)                          // the multiplier (positive)
    hp += ("an", -1.0,  -1.0)                          // the multiplier (negative)
    hp += ("c",  -1.0,  -1.0)                          // the reciprocal power
    hp += ("f",   0.1,   0.1)                          // the frequency for cos, sin
    hp += ("l",   0.4,   0.4)                          // the Box-Cox lambda (l)
    hp += ("p",   1.5,   1.5)                          // the real power
    hp += ("q",   0.67,  0.67)                         // the rational power
    hp += ("r",   0.5,   0.5)                          // the root
    hp += ("s",   0.0,   0.0)                          // the amount of shift
    hp += ("ss",  1.0,   1.0)                          // the amount of shift for log

    val a  = hp("a").toDouble
    val an = hp("an").toDouble
    val c  = hp("c").toDouble
    val f  = hp("f").toDouble
    val l  = hp("l").toDouble
    val p  = hp("p").toDouble
    val q  = hp("q").toDouble
    val r  = hp("r").toDouble
    val s  = hp("s").toDouble
    val ss = hp("ss").toDouble

end Transform

import Transform._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TranformT` enum defines the types of transforms.
 *  @param name  the name of the transform
 *  @param form  a function to instantiate a `Transform`
 *  @param wlu   the default vectors of parameters (w) and their lower (l) and upper bounds (u)
 */
enum TransformT (val name: String, val form: VecMat => Transform, val wlu: TransformS):

    case Norm   extends TransformT ("NormForm",                                                 // x -> (x - μ)/σ
                                    x => NormForm (x),
                                    null)                                                       // NormForm (linear rescaling)

    case MinMax extends TransformT ("MinMaxForm",                                               // x -> l + (x-min)(u-l)/(max-min)
                                    x => MinMaxForm (x),
                                    null)                                                       // MinMaxForm (linear rescaling)

    case Center extends TransformT ("CenterForm",                                               // x -> x - μ
                                    x => CenterForm (x),
                                    null)                                                       // CenterForm (linear rescaling)

    case Pow    extends TransformT ("PowForm",                                                  // x -> (x + s)^p
                                    x => PowForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((s, p), (0.0, 1.1), (10.0, 4.0)))           // PowForm real p in [1.1, 4.0]

    case PowR   extends TransformT ("PowRForm",                                                 // x -> (x + s)↑q
                                    x => PowRForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((s, q), (0.0, 0.1), (10.0, 0.9)))           // PowRForm rational q in [0.1, 0.9]

    case Boxcox extends TransformT ("BoxcoxForm",                                               // x -> (x^l - 1) / l
                                    x => BoxcoxForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((l, 0.0), (-4.0, 0.0), (4.0, 0.0)))         // PowForm real l in [-4.0, 4.0]

    case Root   extends TransformT ("RootForm",                                                 // x -> (x + s)^r
                                    x => PowForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((s, r), (0.0, 0.1), (10.0, 0.9)))           // RootForm (PowForm r in [0.1, 0.9])

    case Recip  extends TransformT ("RecipForm",                                                // x -> (x + s)^c
                                    x => PowForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((s, c), (0.0, -4.0), (10.0, -0.1)))         // RecipForm (PowForm c in [-4.0, -0.1])

    case Log    extends TransformT ("LogForm",                                                  // x -> log (a x + ss)
                                    x => LogForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((ss, a), (0.0, 0.1), (10.0, 10.0)))         // LogForm (replaces PowForm in (-0.1, 0.1))

    case Log1p  extends TransformT ("Log1pForm",                                                // x -> log (x + 1)
                                    x => Log1pForm (x),
                                    null)                                                       // LogForm (replaces PowForm in (-0.1, 0.1))

    case Exp    extends TransformT ("ExpForm",                                                  // x -> exp (a x + s)
                                    x => ExpForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((s, a), (0.0, 0.1), (10.0, 10.0)))          // ExpForm (replaces PowForm above 4)

    case NExp   extends TransformT ("NExpForm",                                                 // x -> exp (an x + s)
                                    x => ExpForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((s, an), (-10.0, -10.0), (0.0, -0.1)))      // NExpForm (replaces RecipForm below -4)

    case Cos    extends TransformT ("CosForm",                                                  // x -> cos (2π f x + s)
                                    x => CosForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((s, f), (0.0, 0.01), (10.0, 10.0)))         // CosForm (for wave forms)

    case Sin    extends TransformT ("SinForm",                                                  // x -> sin (2π f x + s)
                                    x => SinForm (x.asInstanceOf [VectorD]),
                                    new TransformS ((s, f), (0.0, 0.01), (10.0, 10.0)))         // SinForm (for wave forms)

end TransformT


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TransformS` class holds a transform specification that contains default values
 *  for the parameters w and their bounds, e.g., w in x -> f(w_0 + w_1 * x)
 *  @param w  the vector of two nonlinear parameters: shift w_0 and scale w_1 parameters
 *  @param l  the vector containing the lower bounds of shift and scale parameters
a*  @param u  the vector containing the upper bounds of shift and scale parameters
 */
case class TransformS (w: VectorD = VectorD (0, 1),
                       l: VectorD = VectorD (0, 0.1),
                       u: VectorD = VectorD (10, 10)):

    def this (wlu: (Double, Double)*) = this (VectorD (wlu(0)), VectorD (wlu(1)), VectorD (wlu(2)))

end TransformS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Transform` trait supports the use of transformation functions and makes it
 *  is easy to take the inverse transform.  When a transformation uses arguments,
 *  they are remembered for use by the inverse transformation.
 *
 *  @note: the CENTERED lu interval [-2, 2] reduces collinearity, but may yield NaN
 *  due to a negative base (x < 0) raised to a `Double` power (x~^p).
 *  Switch to a `Rat` power (x↑r) with an odd denominator, e.g., r = Rat(1, 3) to avoid NaN.
 *  Otherwise, use a positive interval such as [0, 4] or [1, 5].
 *  @see `Rat`, `pow_` in CommonFunctions.scala, and `↑` in ValueType.scala
 *
 *  @param w         the transform argument vector or matrix (in subclasses pass x_ for input or w for weights)
 *  @param centered  whether the normalization should CENTER the data (defaults to true)
 */
trait Transform (w: VecMat, centered: Boolean = true):

    protected var lu: VectorD =                                          // min-max default range/bounds [l .. u]
        if centered then VectorD (-2, 2)                                 // centered to reduce collinearity, similar range to z  
        else VectorD (0, 4)                                              // shifted to positive values, allows (-x)^.5
//      else VectorD (1, 5)                                              // shifted away from zero, allows log (x)

    protected val b: MatrixD  =                                          // transform argument matrix
        w match
            case wV: VectorD => MatrixD (wV).ᵀ
            case wM: MatrixD => wM

    inline def b_ : VectorD = b(?, 0)                                    // get 0-th column of the argument matrix
    def setLU (_lu: VectorD): Unit = lu = _lu                            // set the default bounds to custom value

    def f  (x: MatrixD): MatrixD                                         // transformation function (matrix level)
    def fi (y: MatrixD): MatrixD                                         // inverse transformation function

    val f:  FunctionV2V = (x: VectorD) => f(MatrixD (x).ᵀ)(?, 0)         // vector level
    val fi: FunctionV2V = (y: VectorD) => fi(MatrixD (y).ᵀ)(?, 0)

    val f_ :  FunctionS2S = (x: Double) => f(MatrixD ((1, 1), x))(0, 0)   // scalar level
    val fi_ : FunctionS2S = (y: Double) => fi(MatrixD ((1, 1), y))(0, 0)

    def df (x: VectorD): MatrixD = null                                  // partial derivative of f

    def df (x: MatrixD): MatrixD =                                       // column-by-column partial derivative of f wrt to w
        var jMatrix = df (x(?, 0))
        for j <- 1 until x.dim2 do jMatrix = jMatrix ++^ df (x(?, j))
        jMatrix
    end df

    def df (x: MatrixD, i: Int): MatrixD =                               // partial derivative of each column wrt wi
        if i == 0 || i == 1 then
            var jMatrix = MatrixD (df (x(?, 0))(?, i)).ᵀ
            for j <- 1 until x.dim2 do jMatrix = jMatrix :^+ df (x(?, j))(?, i)
            jMatrix
        else
            df(x)
    end df

    def testV (x: VectorD): Unit =
        val y = f (x)
        val z = fi (y)
        println (s"y = $y, \nz = $z")
    end testV

    def testM (x: MatrixD): Unit =
        val y = f (x)
        val z = fi (y)
        println (s"y = $y, \nz = $z")
    end testM

end Transform 

// @note: NormForm will make some of the normalized data negative
//        MinMaxForm will do the same when centered = true

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NormForm` class applies the Z-transformation/normalization/standardization
 *  (subtract mean b(0) and divide by standard deviation b(1)).
 *  Like ''StandardScalar'' in sk-learn.
 *  @see www.geeksforgeeks.org/machine-learning/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/
 *
 *      x -> (x - μ)/σ    robust version:  (x - μ)/(σ + ε)  or  (x - μ)/√(σ² + ε) 
 *
 *  @param x_      the input vector or matrix to be transformed (needed to get w)
 *  @param robust  whether to add a small value (ε) to the standard deviation to avoid DBZ
 *                 when a whole column has zero stdev (σ) it should be removed in pre-processing
 */
class NormForm (x_ : VecMat, robust: Boolean = true) extends Transform (x_.mu_sig):

    val ε = 1E-8
    def f (x: MatrixD): MatrixD  =
        if robust then (x - b(0)) / (b(1) + ε)
        else           (x - b(0)) / b(1)
    def fi (y: MatrixD): MatrixD =
        if robust then y *~ (b(1) + ε) + b(0)
        else           y *~ b(1) + b(0)

end NormForm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MinMaxForm` class applies the MIN-MAX-transformation to move the data into the range [l .. u]
 *  (subtract min, multiply by bounds lu range over min-max range and add lower bound).
 *  Like ''MinMaxScaler'' in sk-learn.
 *  @see www.geeksforgeeks.org/machine-learning/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/
 *
 *      x -> l + (x-min)(u-l)/(max-min)
 *
 *  @param x_        the input vector or matrix to be transformed (needed to get w)
 *  @param centered  whether the normalization should CENTER the data (defaults to true)
 */
class MinMaxForm (x_ : VecMat, centered: Boolean = true) extends Transform (x_.min_max, centered):

    def f (x: MatrixD): MatrixD  = (x - b(0)) * (lu(1) - lu(0)) / (b(1) - b(0))  + lu(0)
    def fi (y: MatrixD): MatrixD = (y - lu(0)) *~ (b(1) - b(0)) /(lu(1) - lu(0)) + b(0)

end MinMaxForm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CenterForm` class applies the CENTER-transformation so the new mean will be zero
 *  (subtract original mean b(0)).
 *
 *      x -> x - μ
 *
 *  @param x_  the input vector or matrix to be transformed (needed to get w)
 */
class CenterForm (x_ : VecMat) extends Transform (x_.mu_sig):

    def f (x: MatrixD): MatrixD  = x - b(0)
    def fi (y: MatrixD): MatrixD = y + b(0)

end CenterForm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PowForm` class applies a shifted s = b_(0) and scaled p = b_(1) POWER-transformation.
 *  For shift s and power p,
 *
 *       x -> (x + s)^p
 *
 *  @param w  the transform argument vector (w -> b)
 */
class PowForm (w: VectorD = VectorD (s, p)) extends Transform (w):

    def f (x: MatrixD): MatrixD  = (x + b_(0)) ~^ b_(1)
    def fi (y: MatrixD): MatrixD = y ~^ (1/b_(1)) - b_(0)
    override def df (x: VectorD): MatrixD = MatrixD (((x + b_(0)) ~^ (b_(1) - 1)) * b_(1),
                                                     f(x) * (x + b_(0)).log).ᵀ
end PowForm

class RootForm (w: VectorD = VectorD (s, r)) extends PowForm (w)

class RecipForm (w: VectorD = VectorD (s, c)) extends PowForm (w)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PowRForm` class applies a shifted s = b_(0) and scaled q = b_(1) POWER-transformation.
 *  For shift s and power q where q is converted to r a nearby rational number with an odd denominator
 *
 *       x -> (x + s)↑q
 *
 *  @param w  the transform argument vector (w -> b)
 */
class PowRForm (w: VectorD = VectorD (s, q)) extends Transform (w):

    private val r  = Rat.fromDouble3 (b_(1))         // nearby rational number with an odd denominator
    private val ri = r.recip

    def f (x: MatrixD): MatrixD  = (x + b_(0)) ↑ r
    def fi (y: MatrixD): MatrixD = y ↑ ri - b_(0)
    override def df (x: VectorD): MatrixD = MatrixD (((x + b_(0)) ↑ (r - 1)) * r.toDouble,
                                                     f(x) * (x + b_(0)).log).ᵀ
end PowRForm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BoxcoxForm` class applies a and scaled p = b_(0) BOX-COX-transformation.
 *  When l is zero, switch to a log-transformation.  For power l (λ in textbooks),
 *
 *       x -> (x^l - 1) / l
 *
 *  @param w  the transform argument vector (w -> b)
 */
class BoxcoxForm (w: VectorD = VectorD (l)) extends Transform (w):

    private val l = b_(0)
    
    def f (x: MatrixD): MatrixD  = (x ~^ l - 1) / l
    def fi (y: MatrixD): MatrixD = (y * l + 1) ~^ (1/l)
    override def df (x: VectorD): MatrixD = MatrixD (x ~^ (l-1),
                                                     (l * x + 1) ~^ (1/l-1)).ᵀ
end BoxcoxForm


// FIX - TBD: add (1) Modulus Transformation and (2) Inverse Hyperbolic Sine (asinh)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LogForm` class applies a shifted b_(0) and scaled b_(1) LOG-transformation.
 *  Note: the default w = (1, 1) corresponds to log1p and its inverse to expm1.
 *  For shift s and scale a,
 *
 *      x -> log (ax + s)
 *
 *  @param w  the transform argument vector (w -> b)
 */
class LogForm (w: VectorD = VectorD (ss, a)) extends Transform (w):

    def f (x: MatrixD): MatrixD  = (x * b_(1) + b_(0)).log
    def fi (y: MatrixD): MatrixD = (y.exp - b_(0)) / b_(1)
    override def df (x: VectorD): MatrixD = MatrixD (1 / (x * b_(1) + b_(0)), 
                                                     x / (x * b_(1) + b_(0))).ᵀ
end LogForm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Log1pForm` class applies a shifted b_(0) and scaled b_(1) LOG-transformation.
 *  @note:  the given w = (1, 1) corresponds to log1p and its inverse to expm1.
 *  For shift s and scale a,
 *
 *      x -> log (ax + s)
 *
 *  @param x_  the transform argument is null
 */
class Log1pForm (@unused x_ : VecMat = null) extends Transform (VectorD (1, 1)):

    def f (x: MatrixD): MatrixD  = (x * b_(1) + b_(0)).log
    def fi (y: MatrixD): MatrixD = (y.exp - b_(0)) / b_(1)
    override def df (x: VectorD): MatrixD = MatrixD (1 / (x * b_(1) + b_(0)),
                                                     x / (x * b_(1) + b_(0))).ᵀ
end Log1pForm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ExpForm` class applies a shifted b_(0) and scaled b_(1) EXP-transformation.
 *  For shift s and scale a,
 *
 *       x -> exp (ax + s)
 *
 *  @param w  the transform argument vector (w -> b)
 */
class ExpForm (w: VectorD = VectorD (s, a)) extends Transform (w):

    def f (x: MatrixD): MatrixD  = (x * b_(1) + b_(0)).exp
    def fi (y: MatrixD): MatrixD = (y.log - b_(0)) / b_(1)
    override def df (x: VectorD): MatrixD = MatrixD (f(x),
                                                     f(x) * x).ᵀ
end ExpForm

class NExpForm (w: VectorD = VectorD (s, an)) extends ExpForm (w)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CosForm` class applies a shifted b_(0) and scaled b_(1) COSINE-transformation.
 *  For phase shift b_(0) = φ = s and frequency b_(1) = f with scaling 2πf,
 *
 *      x -> cos (2πfx + φ)
 *
 *  @param w  the transform argument vector (w -> b)
 */
class CosForm (w: VectorD = VectorD (s, f)) extends Transform (w):

    def f (x: MatrixD): MatrixD  = (x * (b_(1) * _2Pi) + b_(0)).cos
    def fi (y: MatrixD): MatrixD = (y.acos - b_(0)) / (b_(1) * _2Pi) 
    override def df (x: VectorD): MatrixD = MatrixD (-(x * (b_(1) * _2Pi) + b_(0)).sin,
                                                     -(x * (b_(1) * _2Pi) + b_(0)).sin * (x * _2Pi)).ᵀ
end CosForm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SinForm` class applies a shifted b_(0) and scaled b_(1) SINE-transformation.
 *  For phase shift b_(0) = φ = s and frequency b_(1) = f with scaling 2πf,
 *
 *      x -> sin (2πfx + φ)
 *
 *  @param w  the transform argument vector (w -> b)
 */
class SinForm (w: VectorD = VectorD (s, f)) extends Transform (w):

    def f (x: MatrixD): MatrixD  = (x * (b_(1) * _2Pi) + b_(0)).sin
    def fi (y: MatrixD): MatrixD = (y.asin - b_(0)) / (b_(1) * _2Pi) 
    override def df (x: VectorD): MatrixD = MatrixD ((x * (b_(1) * _2Pi) + b_(0)).cos,
                                                     (x * (b_(1) * _2Pi) + b_(0)).cos * (x * _2Pi)).ᵀ
end SinForm


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `transformTest` tests the classes extending the `Transform` trait at the vector level.
 *  > runMain scalation.mathstat.transformTest
 */
@main def transformTest (): Unit =

    import TransformT._

    val x = VectorD (1.0, 1.5, 2.0, 2.5)
    var form: Transform = null
    println (s"x = $x")

    banner ("NormForm Transformation: (x - μ)/σ")
    form = Norm.form (x)
    form.testV (x)

    banner ("MinMaxForm Transformation: l + (x-min)(l-u)/(max-min)")
    form = MinMax.form (x)
    form.testV (x)

    banner ("CenterForm Transformation: x - μ")
    form = Center.form (x)
    form.testV (x)

    banner ("PowForm Transformation: x^1.5")
    form = Pow.form (Pow.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")
    form.testV (-x)                                             // fails for negative numbers (gives NaN)

    banner ("PowRForm Transformation: x↑.67")
    form = PowR.form (PowR.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")
    form.testV (-x)                                             // works for negative numbers

    banner ("BoxcoxForm Transformation: x^.4")
    form = Boxcox.form (Boxcox.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")
    form.testV (-x)                                             // fails for negative numbers (gives NaN)

    banner ("RootForm Transformation: x^.5")
    form = Root.form (Root.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")

    banner ("RecipForm Transformation: x^-1")
    form = Recip.form (Recip.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")

    banner ("LogForm Transformation: log1p (x)")
    form = Log.form (Log.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")

    banner ("Log1pForm Transformation: log1p (x)")
    form = Log1p.form (x)
    form.testV (x)
    println (s"df: ${form.df (x)}")

    banner ("ExpForm Transformation: exp (x)")
    form = Exp.form (Exp.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")

    banner ("NExpForm Transformation: exp (-x)")
    form = NExp.form (NExp.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")

    banner ("CosForm Transformation: cos (2πfx + φ)")
    form = Cos.form (Cos.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")

    banner ("SinForm Transformation: sin (2πfx + φ)")
    form = Sin.form (Sin.wlu.w)
    form.testV (x)
    println (s"df: ${form.df (x)}")

end transformTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `transformTest2` tests the classes extending the `Transform` trait at the matrix level.
 *  > runMain scalation.mathstat.transformTest2
 */
@main def transformTest2 (): Unit =

    import TransformT._

    val x = MatrixD ((4, 2), 1.0, 1,
                             1.5, 2,
                             2.0, 3,
                             2.5, 4)
    var form: Transform = null
    println (s"x = $x")

    banner ("NormForm Transformation: (x - μ)/σ")
    form = Norm.form (x)
    form.testM (x)

    banner ("MinMaxForm Transformation: a + (x-min)(b-a)/(max-min)")
    form = MinMax.form (x)
    form.testM (x)

    banner ("CenterForm Transformation: x - μ")
    form = Center.form (x)
    form.testM (x)

    banner ("PowForm Transformation: x^1.5")
    form = Pow.form (Pow.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

    banner ("PowRForm Transformation: x↑.67")
    form = PowR.form (PowR.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

    banner ("RootForm Transformation: x^.5")
    form = Root.form (Root.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

    banner ("RecipForm Transformation: x^-1")
    form = Recip.form (Recip.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

    banner ("LogForm Transformation: log1p (x)")
    form = Log.form (Log.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

    banner ("ExpForm Transformation: exp (x)")
    form = Exp.form (Exp.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

    banner ("NExpForm Transformation: exp (-x)")
    form = NExp.form (NExp.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

    banner ("CosForm Transformation: cos (2πfx + φ)")
    form = Cos.form (Cos.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

    banner ("SinForm Transformation: sin (2πfx + φ)")
    form = Sin.form (Sin.wlu.w)
    form.testM (x)
    println (s"df: ${form.df (x)}")

end transformTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `transformTest3` tests the `Transform` class' ability to compose transformations. 
 *  > runMain scalation.mathstat.transformTest3
 */
@main def transformTest3 (): Unit =

    val x = VectorD (3, 5, 6, 2, 1, 3, 2, 4, 6, 87, 1000)
    println (s"x = $x")

    banner ("NormForm (z) Transformation")
    val zForm = NormForm (x)
    var y = zForm.f (x)
    var z = zForm.fi (y)
    println (s"y = $y, \nz = $z")

    banner ("PowForm Transformation")
    val powForm = PowForm ()
    y = powForm.f (x)
    z = powForm.fi (y)
    println (s"y = $y, \nz = $z")

    val fsc = (zForm.f(_: VectorD)) ⚬ (powForm.f(_: VectorD)) ⚬ (zForm.fi(_: VectorD))
    val ysc = fsc (y)
    println (s"ysc = ${ysc}")

    val ysc2 = fsc (y(0 until 3))
    println (s"ysc = ${ysc2}")

end transformTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `transformTest4` tests the `Transform` class by plotting the transformations.
 *  > runMain scalation.mathstat.transformTest4
 */
@main def transformTest4 (): Unit =

    import TransformT._

    val x = VectorD.range (1 until 100) / 25.0 + 1.0
    var form: Transform = null
    println (s"x = $x")

    banner ("NormForm Transformation: (x - μ)/σ")
    form = Norm.form (x)
    val y1 = form.f (x)

    banner ("MinMaxForm Transformation: a + (x-min)(b-a)/(max-min)")
    form = MinMax.form (x)
    val y2 = form.f (x)

    banner ("PowForm Transformation: x^1.5")
    form = Pow.form (Pow.wlu.w)
    val y3 = form.f (x)

    banner ("RootForm Transformation: x^.5")
    form = Root.form (Root.wlu.w)
    val y4 = form.f (x)

    banner ("RecipForm Transformation: x^-1")
    form = Recip.form (Recip.wlu.w)
    val y5 = form.f (x)

    banner ("LogForm Transformation: log1p (x)")
    form = Log.form (Log.wlu.w)
    val y6 = form.f (x)

/*
    banner ("ExpForm Transformation: exp (x)")
    form = Exp.form (Exp.wlu.w)
    val y7 = form.f (x)
*/

    banner ("NExpForm Transformation: exp (-x)")
    form = NExp.form (NExp.wlu.w)
    val y8 = form.f (x)

    banner ("CosForm Transformation: cos (2πfx + φ)")
    form = Cos.form (Cos.wlu.w)
    val y9 = form.f (x)

    banner ("SinForm Transformation: sin (2πfx + φ)")
    form = Sin.form (Sin.wlu.w)
    val y10 = form.f (x)

    new PlotM (x, MatrixD (y1, y2, y3, y4, y5, y6, y8, y9, y10),
                    Array ("Norm", "MinMax", "Pow", "Root", "Recip", "Log", "NExp", "Cos", "Sin"))

end transformTest4

