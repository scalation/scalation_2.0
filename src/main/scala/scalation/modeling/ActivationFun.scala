
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Dec 28 12:00:07 EST 2014
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Activation Functions for Neural Networks
 *
 *  @see     `MatrixD` for `matricize`
 */

package scalation
package modeling

import scala.math.{cosh, exp, log, max, min, tanh}

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `AFF` class holds an Activation Function Family (AFF).
 *  @param name    the name of the activation function
 *  @param f       the activation function itself (scalar version)
 *  @param f_      the vector version of the activation function
 *  @param d       the vector version of the activation function derivative
 *  @param bounds  the (lower, upper) bounds on the output range of the activation function,
 *                     e.g., (0, 1) for sigmoid, defaults to null => no limit
 *  @param arange  the (lower, upper) bounds on the input (active) range of the activation function
 *                     e.g., (-2, 2) for sigmoid, defaults to null => no limit
 */
case class AFF (name: String, f: FunctionS2S, f_ : FunctionV2V, d: FunctionV2V,
                bounds: (Double, Double) = null, arange: (Double, Double) = null):

    val fM = matricize (f_)             // the matrix version of the activation function
    val dM = matricize (d)              // the matrix version of the activation function derivative

//  val fM2 = matrixize (f_)            // the matrix version of the activation function via FP
//  val dM2 = matrixize (d)             // the matrix version of the activation function derivative via FP

end AFF


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ActivationFun` object contains common Activation functions and provides
 *  both scalar and vector versions.
 *  @see en.wikipedia.org/wiki/Activation_function
 *  Convention: fun   activation function (e.g., sigmoid)
 *              fun_  vector version of activation function (e.g., sigmoid_)
 *              funD  vector version of dervivative (e.g., sigmoidD)
 *----------------------------------------------------------------------------------
 * Supports: id, reLU, lreLU, eLU, tanh, sigmoid, gaussian, softmax
 * Related functions: logistic, logit
 */
object ActivationFun:

    private val debug = debugf ("ActivationFun", true)                            // debug function

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// id: Identity functions

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value of the Identity id function at scalar t.
     *  @param t  the id function argument
     */
    inline def id (t: Double): Double    = t
    def id_ (t: VectorD): VectorD = t

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for id function at vector yp where
     *  yp is pre-computed by yp = id (t).
     *  @param yp  the derivative function vector argument
     */
    def idD (yp: VectorD): VectorD = VectorD.one (yp.dim)

    val f_id = AFF ("id", id, id_, idD)                                           // id family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// reLU: Rectified Linear Unit functions

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value of the Rectified Linear Unit reLU function at scalar t.
     *  @param t  the reLU function argument
     */
    inline def reLU (t: Double): Double = max0 (t)
    def reLU_ (t: VectorD): VectorD = t.max0
//  val _reLU_ : FunctionV2V = vectorize (reLU)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for reLU function at vector yp where
     *  yp is pre-computed by yp = reLU_ (t).
     *  @param yp  the derivative function vector argument
     */
    def reLUD (yp: VectorD): VectorD = yp.map (y => if y > 0.0 then 1.0 else 0.0)

    val f_reLU = AFF ("reLU", reLU, reLU_, reLUD)                                 // reLU family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// lreLU: Leaky Rectified Linear Unit functions
// @see stackoverflow.com/questions/64735352/details-about-alpha-in-tf-nn-leaky-relu-features-alpha-0-2-name-none
// "The default values in Tensorflow and Keras are 0.2 and 0.3 respectively"

//  private var a = 0.01             // the lreLU alpha parameter (0, 1] indicating how leaky the function is
//  private var a = 0.3              // the lreLU alpha parameter (0, 1] default values used in Keras
                                     // @see keras.io/layers/advanced-activations
    private var a = 0.2              // the lreLU alpha parameter (0, 1] default values used in Tensorflow and ScalaTion

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the lreLU a (alpha) parameter for the Leaky Rectified Linear Unit functions.
     *  @param a  the rleLU alpha parameter (0, 1] indicating how leaky the function is
     */
    def setA (a_ : Double): Unit = a = min (a_, 1.0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value of the Leaky Rectified Linear Unit lreLU function at scalar t.
     *  @param t  the lreLU function argument
     */
    inline def lreLU (t: Double): Double = max (a * t, t)
    def lreLU_ (t: VectorD): VectorD = t.map (lreLU (_))
//  val _lreLU_ : FunctionV2V = vectorize (lreLU)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for lreLU function at vector yp where
     *  yp is pre-computed by yp = lreLU_ (t).
     *  @param yp  the derivative function vector argument
     */
    def lreLUD (yp: VectorD): VectorD = yp.map (y => if y >= 0.0 then 1.0 else a)

    val f_lreLU = AFF ("lreLU", lreLU, lreLU_, lreLUD)                            // lreLU family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// eLU: Exponential Linear Unit functions
// @see arxiv.org/pdf/1511.07289.pdf

    private var a2 = 1.0             // the eLU alpha parameter (0, infinity) indicating how leaky the function is

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set the eLU a2 (alpha) parameter for the Exponential Linear Unit functions.
     *  @param a_  the eLU alpha parameter (0, infinity) indicating how leaky the function is
     */
    def setA2 (a_ : Double): Unit = a2 = a_ 

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value of the Exponential Linear Unit eLU function at scalar t.
     *  @param t  the eLU function argument
     */
    inline def eLU (t: Double): Double = if t > 0.0 then t else a2 * (exp (t) - 1)
    def eLU_ (t: VectorD): VectorD = t.map (eLU (_))
//  val _eLU_ : FunctionV2V = vectorize (eLU)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for eLU function at vector yp where
     *  yp is pre-computed by yp = eLU_ (t).
     *  @param yp  the derivative function vector argument
     */
    def eLUD (yp: VectorD): VectorD = yp.map (y => if y > 0.0 then 1.0 else y + a2)
 
    val f_eLU = AFF ("eLU", eLU, eLU_, eLUD)                                      // eLU family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// tanh: Hyperbolic Tangent functions
  
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /*  Compute the value of the tanh function at scalar t.
     *  @param t  the tanh function argument
     */
    //  @see scala.math.tanh

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the vector of values of the tanh function applied to vector t.
     *  @param t  the tanh function vector argument
     */
    def tanh_ (t: VectorD): VectorD = t.map (tanh (_))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for tanh function at vector yp where
     *  yp is pre-computed by yp = tanh_ (t).
     *  @param yp  the derivative function vector argument
     */
    def tanhD (yp: VectorD): VectorD = VectorD.one (yp.dim) - yp~^2

    val f_tanh = AFF ("tanh", tanh, tanh_, tanhD, (-1, 1), (-2, 2))               // tanh family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// sigmoid: Sigmoid functions

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value of the Sigmoid function at t.  This is a special case of
     *  the logistic function, where a = 0 and b = 1.  It is also referred to as
     *  the standard logistic function.  It is also the inverse of the logit function.
     *  @param t  the sigmoid function argument
     */
    inline def sigmoid (t: Double): Double = 1.0 / (1.0 + exp (-t))
    def sigmoid_ (t: VectorD): VectorD = t.map (sigmoid (_))
//  val _sigmoid_ : FunctionV2V = vectorize (sigmoid)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for sigmoid function at vector yp where
     *  yp is pre-computed by yp = sigmoid_ (t).
     *  @param yp  the derivative function vector argument
     */
    def sigmoidD (yp: VectorD): VectorD = yp * (VectorD.one (yp.dim) - yp)

    val f_sigmoid = AFF ("sigmoid", sigmoid, sigmoid_, sigmoidD, (0, 1), (-2, 2))   // sigmoid family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Common Activation Function Families

    val f_aff = Array (f_id, f_reLU, f_lreLU, f_eLU, f_tanh, f_sigmoid)

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// gaussian: Gaussian functions

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value of the Gaussian function at scalar t.
     *  @param t  the Gaussian function argument
     */
    inline def gaussian (t: Double): Double = exp (-t * t)
    def gaussian_ (t: VectorD): VectorD = t.map (gaussian (_))
//  val _gaussian_ : FunctionV2V = vectorize (gaussian)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for Gaussian function at vector yp where
     *  yp is pre-computed by yp = gaussian_ (t).
     *  @param yp  the derivative function vector argument
     *  @param t   the domain value for the function
     */
    def gaussianD (yp: VectorD, t: VectorD): VectorD = t * yp * -2.0              // non-standard signature

//  val f_gaussain = AFF ("gaussian", gaussian, gaussian_, gaussianD)             // gaussian family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// geLU: Gaussain Error Linear Unit (geLU) functions

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Approximately compute the value of the geLU function at t.
     *  @param t  the geLU function argument
     */
    inline def geLU (t: Double): Double = 0.5 * t * (1.0 + tanh (sqrt_2byPi * (t + 0.044715 * t~^3)))
    def geLU_ (t: VectorD): VectorD = t.map (geLU (_))
//  val _geLU_ : FunctionV2V = vectorize (geLU)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for geLU function at vector yp where
     *  yp is pre-computed by yp = geLU_ (t).
     *  @param t   the domain value for the function
     */
    def geLUd (t: Double): Double =
        val t3 = t~^3
        0.5 * tanh (0.0356774 * t3 + 0.797885 * t) + 0.5 +
        (0.0535161 * t3 + 0.398942 * t) / cosh (0.0356774 * t3 + 0.797885 * t)~^2
    end geLUd

    val geLUD : FunctionV2V = vectorize (geLUd)

    val f_geLU = AFF ("geLU", geLU, geLU_, geLUD)                                 // geLU family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// softmax: Softmax functions - FIX

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the vector of values of the Softmax function applied to vector t.
     *  @see en.wikipedia.org/wiki/Softmax_function
     *  Note, scalar function version softmax is not needed.
     *  @param t  the softmax function vector argument
     */
    inline def softmax_ (t: VectorD): VectorD = 
        val ts = t - t.max                                  // shift for numeric stability
        val et = ts.map (exp (_))
        et / et.sum
    end softmax_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative vector for the Softmax function at vector yp where
     *  yp is pre-computed by yp = softmax_ (t).
     *  @param yp  the derivative function vector argument
     */
    def softmaxD (yp: VectorD): VectorD = yp.map (t => t * (1 - t))
//      VectorD (for i <- yp.indices yield yp(i) * (1.0 - yp(i)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the derivative matrix (Jacobian) for Softmax function at vector yp where
     *  yp is pre-computed by yp = softmax_ (t).
     *  @see eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
     *  @param yp  the derivative function vector argument
     */
    def softmaxDM (yp: VectorD): MatrixD = 
        val z = new MatrixD (yp.dim, yp.dim)
        for i <- yp.indices; j <- yp.indices do
            z(i, j) = if i == j then yp(i) * (1.0 - yp(j))
                      else          -yp(i) * yp(j)
        z
    end softmaxDM

    val f_softmax = AFF ("softmax", null, softmax_, softmaxD)                     // softmax family

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// logistic: Logistic functions - related function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the value of the Logistic function at scalar t.
     *  With the default settings, it is identical to sigmoid.
     *  Note, it is not typically used as an activation function
     *  @see www.cs.xu.edu/math/math120/01f/logistic.pdf
     *  @param t  the logistic function argument
     *  @param a  the shift parameter (1 => mid at 0, <1 => mid shift left, >= mid shift right
     *  @param b  the spread parameter (1 => sigmoid rate, <1 => slower than, >1 => faster than)
     *            althtough typically positive, a negative b will cause the function to decrease
     *  @param c  the scale parameter (range is 0 to c)
     */
    inline def logistic (t: Double, a: Double = 1, b: Double = 1, c: Double = 1): Double = c / (1 + a * exp (-b*t))
    def logistic_ (t: VectorD, a: Double = 1, b: Double = 1, c: Double = 1): VectorD = t.map (logistic (_, a, b, c))

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// logit: Logit functions - related function

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the log of the odds (Logit) of an event occurring (e.g., success, 1).
     *  The inverse of the logit function is the standard logistic function
     *  (sigmoid function).
     *  Note, it is not typically used as an activation function
     *  @param p  the probability, a number between 0 and 1.
     */
    inline def logit (p: Double): Double = log (p / (1 - p))
    def logit_ (p: VectorD): VectorD = p.map (logit (_))
//  val _logit_ : FunctionV2V = vectorize (logit)                 // vector version

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Utility rescaling functions based on the active range of activation functions

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rescale the input/data matrix x to the arange (active range) of the "first"
     *  activation function f; otherwise normalize.  Return the rescaled matrix.
     *  @param x  the input/data matrix
     *  @param f  the activation function family (first)
     */
    def rescaleX (x: MatrixD, f: AFF): MatrixD =
        if f.arange != null then                                // scale to arange of f
            val (min_x, max_x) = (x.min, x.max)
            debug ("rescaleX", s"from ($min_x, $max_x) to ${f.arange}")
            scale ((min_x, max_x), f.arange) (x)
        else                                                    // normalize: Normal (0, 1)
            val (mu_x, sig_x) = (x.mean, x.stdev)
            normalize ((mu_x, sig_x)) (x)
    end rescaleX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rescale the input/data tensor x to the arange (active range) of the "first"
     *  activation function f; otherwise normalize.  Return the rescaled tensor.
     *  @param x  the input/data matrix
     *  @param f  the activation function family (first)
     */
    def rescaleX (x: TensorD, f: AFF): TensorD =
/*                                                              // FIX - need to adapt for tensor level
        if f.arange != null then                                // scale to arange of f
            val (min_x, max_x) = (x.min, x.max)
            debug ("rescaleX", s"from ($min_x, $max_x) to ${f.arange}")
            scale ((min_x, max_x), f.arange) (x)
        else                                                    // normalize: Normal (0, 1)
            val (mu_x, sig_x) = (x.mean, x.stdev)
            normalize ((mu_x, sig_x)) (x)
*/
        null
    end rescaleX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rescale the output/response vector y to the bounds of the "last" activation
     *  function f; otherwise normalize.  Return the rescaled vector and the
     *  rescaling inverse function.
     *  @param y  the output/response vector
     *  @param f  the activation function family (last)
     */
    def rescaleY (y: VectorD, f: AFF): (VectorD, FunctionV2V) =
        if f.bounds != null then                                // scale to bounds of f
            val (min_y, max_y) = (y.min, y.max)
            debug ("rescaleY", s"from ($min_y, $max_y) to ${f.bounds}")
            (scaleV ((min_y, max_y), f.bounds) (y),
             unscaleV ((min_y, max_y), f.bounds))               // rescaling inverse
        else                                                    // normalize: Normal (0, 1)
            val (mu_y, sig_y) = (y.mean, y.stdev)
            (normalizeV ((mu_y, sig_y)) (y),
             denormalizeV ((mu_y, sig_y)))                      // rescaling inverse
    end rescaleY

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Rescale the output/response matrix y to the bounds of the "last" activation
     *  function f; otherwise normalize.  Return the rescaled matrix and the
     *  rescaling inverse function.
     *  @param y  the output/response matrix
     *  @param f  the activation function family (last layer)
     */
    def rescaleY (y: MatrixD, f: AFF): (MatrixD, FunctionM2M) =
        if f.bounds != null then                                // scale to bounds of f
            val (min_y, max_y) = (y.min, y.max)
            debug ("rescaleY", s"from ($min_y, $max_y) to ${f.bounds}")
            (scale ((min_y, max_y), f.bounds) (y),
             unscale ((min_y, max_y), f.bounds))                // rescaling inverse
        else                                                    // normalize: Normal (0, 1)
            val (mu_y, sig_y) = (y.mean, y.stdev)
            (normalize ((mu_y, sig_y)) (y),
             denormalize ((mu_y, sig_y)))                       // rescaling inverse
    end rescaleY

end ActivationFun


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `activationFunTest` main function tests the `ActivationFun` object.
 *  This test individually plots the vector versions of the activation functions f_(t).
 *  > runMain scalation.modeling.activationFunTest
 */
@main def activationFunTest (): Unit =

    import ActivationFun._

    val t = VectorD.range (-50, 50) / 10.0

    // Test the vector version of activation functions
    val idf       = id_ (t)
    val reLUf     = reLU_ (t)
    val lreLUf    = lreLU_ (t)
    val eLUf      = eLU_ (t)
    val tanhf     = tanh_ (t)
    val sigmoidf  = sigmoid_ (t)
    val gaussianf = gaussian_ (t)
    val geLUf     = geLU_ (t)
    val softmaxf  = softmax_ (t)

    new Plot (t, idf,       null, "t vs. id_") 
    new Plot (t, reLUf,     null, "t vs. reLU_") 
    new Plot (t, lreLUf,    null, "t vs. lreLU_") 
    new Plot (t, eLUf,      null, "t vs. eLU_") 
    new Plot (t, tanhf,     null, "t vs. tanh-") 
    new Plot (t, sigmoidf,  null, "t vs. sigmoid_")
    new Plot (t, gaussianf, null, "t vs. gaussian_")
    new Plot (t, geLUf,     null, "t vs. geLU_")
    new Plot (t, softmaxf,  null, "t vs. softmax_")

end activationFunTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `activationFunTest2` main function tests the `ActivationFun` object.
 *  This test plots similar pairs of activation functions f(t).
 *  > runMain scalation.modeling.activationFunTest2
 */
@main def activationFunTest2 (): Unit =

    import ActivationFun._

    val t = VectorD.range (-50, 50) / 10.0

    // Test the vector version of activation functions
    val idf       = id_ (t)
    val reLUf     = reLU_ (t)
    val lreLUf    = lreLU_ (t)
    val eLUf      = eLU_ (t)
    val tanhf     = tanh_ (t)
    val sigmoidf  = sigmoid_ (t)
    val gaussianf = gaussian_ (t)
    val geLUf     = geLU_ (t)
    val softmaxf  = softmax_ (t)

    new Plot (t, idf,       null,     "t vs. id_")
    new Plot (t, lreLUf,    reLUf,    "t vs. lreLU_, reLU_ (red)")
    new Plot (t, eLUf,      geLUf,    "t vs. eLU_, geLU_ (red)")
    new Plot (t, tanhf,     sigmoidf, "t vs. tanh_, sigmoid_ (red)")
    new Plot (t, gaussianf, softmaxf, "t vs. gaussian_, softmax_ (red)")

end activationFunTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `activationFunTest3` main function tests the `ActivationFun` object.
 *  This test plots the derivatives of the vector versions of the activation functions f'(t).
 *  > runMain scalation.modeling.activationFunTest2
 */
@main def activationFunTest3 (): Unit =

    import ActivationFun._

    val t = VectorD.range (-50, 50) / 10.0

    val idf       = id_ (t)
    val reLUf     = reLU_ (t)
    val lreLUf    = lreLU_ (t)
    val eLUf      = eLU_ (t)
    val tanhf     = tanh_ (t)
    val sigmoidf  = sigmoid_ (t)
    val gaussianf = gaussian_ (t)
//  val geLUf     = geLU_ (t)                                    // not need for derivative
    val softmaxf  = softmax_ (t)

    // Test the vector version of activation function derivatives
    val idDf       = idD (idf)
    val reLUDf     = reLUD (reLUf)
    val lreLUDf    = lreLUD (lreLUf)
    val eLUDf      = eLUD (eLUf)
    val tanhDf     = tanhD (tanhf)
    val sigmoidDf  = sigmoidD (sigmoidf)
    val gaussianDf = gaussianD (gaussianf, t)
    val geLUDf     = geLUD (t)
    val softmaxDf  = softmaxD (softmaxf)

    // print out all the derivatives
    println (s"idD       = $idDf")
    println (s"reLUD     = $reLUDf")
    println (s"lreLUD    = $lreLUDf")
    println (s"eLUD      = $eLUDf")
    println (s"tanhD     = $tanhDf")
    println (s"sigmoidD  = $sigmoidDf")
    println (s"gaussianD = $gaussianDf")
    println (s"geLUD     = $geLUDf")
    println (s"softmaxD  = $softmaxDf")

    new Plot (t, idDf,       null, "t vs. idD") 
    new Plot (t, reLUDf,     null, "t vs. reLUD") 
    new Plot (t, lreLUDf,    null, "t vs. lreLUD") 
    new Plot (t, eLUDf,      null, "t vs. eLUD") 
    new Plot (t, tanhDf,     null, "t vs. tanhD") 
    new Plot (t, sigmoidDf,  null, "t vs. sigmoidD")
    new Plot (t, gaussianDf, null, "t vs. gaussianD")
    new Plot (t, geLUDf,     null, "t vs. geLUD")
    new Plot (t, softmaxDf,  null, "t vs. softmaxD")

end activationFunTest3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `activationFunTest4` main function tests the `ActivationFun` object softmax
 *  activation function.
 *  @see en.wikipedia.org/wiki/Softmax_function
 *  > runMain scalation.modeling.activationFunTest4
 */
@main def activationFunTest4 (): Unit =

    import ActivationFun.softmax_
    import Probability.centropy

    banner ("Example Softmax Calculation")
    val t = VectorD (1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0)
    println (s"softmax_ ($t) = \n ${softmax_ (t)}")

    banner ("Neural Network Classifier, see section 10.8 in ScalaTion Textbook")
    val y    = VectorD (0, 0, 1)
    val u    = VectorD (2.1, 1.2, 8.3)
    val yp   = softmax_ (u)
    val loss = centropy (y, yp)

    println (s"y    = given            = $y")
    println (s"u    = given            = $u")
    println (s"yp   = softmax_ (u)     = $yp")
    println (s"loss = centropy (y, yp) = $loss")

end activationFunTest4


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `activationFunTest5` main function tests the `ActivationFun` object.
 *  It tests the `logit_` and `logistic_` functions.
 *  > runMain scalation.modeling.activationFunTest5
 */
@main def activationFunTest5 (): Unit =

    import ActivationFun.{logit_, logistic_}

    val t = VectorD.range (-50, 50) / 10.0
    val p = VectorD.range (1, 99) / 100.0

    // Test the vector version of related functions
    val logitf    = logit_ (p);    new Plot (p, logitf,    null, "p vs. logit_")
    val logisticf = logistic_ (t); new Plot (t, logisticf, null, "t vs. logistic_")

end activationFunTest5


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `activationFunTest6` main function tests the `ActivationFun` object.
 *  It tests the `matricise` or `matrixize` functions.
 *  > runMain scalation.modeling.activationFunTest6
 */
@main def activationFunTest6 (): Unit =

    import ActivationFun.sigmoid_
    
    val x = MatrixD (VectorD.range (-5, 5), VectorD.range(0, 10))

    val fM = matricize (sigmoid_)(x)
    println (s"fM = $fM")

    val fM2 = matrixize (sigmoid_)
    println (s"fM2 (x) = ${fM2 (x)}")

end activationFunTest6

