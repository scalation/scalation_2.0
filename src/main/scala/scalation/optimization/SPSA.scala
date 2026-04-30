
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Yulong Wang
 *  @version 2.0
 *  @date    Thursday Feb 17 13:32:52 EDT 2022
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Simultaneous Perturbation Stochastic Approximation
 */

package scalation
package optimization

import scala.math.pow

import scalation.mathstat.{FunctionV2S, VectorD}
import scalation.random.{Bernoulli, Uniform}
//import scalation.random.{Bernoulli, Normal, Uniform}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SPSA` class implements the Simultaneous Perturbation Stochastic Approximation
 *  algorithm for rough approximation of gradients.
 *  @see https://www.jhuapl.edu/spsa/PDF-SPSA/Matlab-SPSA_Alg.pdf
 *
 *      minimize f(x)
 *
 *  @param f         the vector to scalar function whose approximate gradient is sought
 *  @param max_iter  the maximum number of iterations
 *  @param checkCon  whether to check bounds contraints
 *  @param lower     the lower bounds vector
 *  @param upper     the upper bounds vector
 *  @param debug_    the whether to call in debug mode (does tracing)j
 */
class SPSA (f: FunctionV2S, max_iter: Int = 100, checkCon: Boolean = false,
            lower: VectorD = null, upper: VectorD = null, debug_ : Boolean = true)
      extends Minimizer
         with BoundsConstraint (lower, upper)
         with MonitorEpochs:

    private val debug = debugf ("SPSA", debug_)                        // debug function
    private val flaw  = flawf ("SPSA")                                 // flaw function

    private val EPS   = 1E-6
    private var alpha = 0.602
    private var gamma = 0.101
    private var A     = 100.0
    private var a     = 0.16       // these numbers are from Spall (1998) DOI: 10.1109/7.705889
    private var c     = 1.0

    private var f_best = Double.MaxValue

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the parameters.
     *  @param params  the given starting parameters of a VectorD
     */
    def reset (params: VectorD = VectorD (0.602, 0.101, 10.0, 0.16, 1.0)): Unit =
        if params.length != 5 then flaw ("reset", "failed! did not pass 5 parameters")
        alpha  = params(0)
        gamma  = params(1)
        A      = params(2)
        a      = params(3)
        c      = params(4)
    end reset

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** This method is not supported.
     */
    def lineSearch (x: VectorD, dir: VectorD, step: Double = STEP): Double =
        throw new UnsupportedOperationException ("lineSearch: not provided by this optimizer")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a random vector of {-1, 1} values.
     *  @param n       the size of the vector
     *  @param p       the probability of 1
     *  @param stream  the random number stream
     */
    def bernoulliVec (n: Int, p: Double = 0.5, stream: Int = 0): VectorD =
        val coin = Bernoulli (p, stream)                               // Bernoulli (0/1) RVG
        VectorD (for _ <- 0 until n yield 2.0 * coin.gen - 1.0)
    end bernoulliVec

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve for an optimal point by moving a distance ak in the -ghat direction.
     *  @param x0     initial point
     *  @param step   steps for iteration
     *  @param toler  tolerance
     */
    def solve (x0: VectorD, step: Double = STEP, toler: Double = EPS): FuncVec =

        var x_old  = x0.copy                                           // old point
        var x_best = x0.copy                                           // best point so far
        val x      = x0.copy                                           // new point

        var (k, go) = (1, true)
        cfor (k <= max_iter && go, k += 1) {
            val ak      = a / pow (A + k + 1, alpha)                   // how far to move along gradient
            val ck      = c / pow (k + 1, gamma)                       // for x distance
            val delta   = bernoulliVec (x0.dim)                        // random direction
            val x_plus  = x + delta * ck                               // x moved + delta direction
            val x_minus = x - delta * ck                               // x moved - delta direction
            val y_plus  = f(x_plus)                                    // functional value for x_plus
            val y_minus = f(x_minus)                                   // functional value for x_minus

            val ghat = delta * (y_plus - y_minus) / (2 * ck)           // rough/approx. gradient
            x_old    = x.copy                                          // save previous location x
            x       -= ghat * ak                                       // update x: move opposite gradient

            if checkCon then constrain (x)                             // enforce contraints, may move x

            val f_x = f(x)                                             // new functional value
            debug ("solve", s"iteration k = $k, x = $x, f(x) = $f_x vs. f_best = $f_best")

            if f_x < f_best then
                x_best = x.copy                                        // copy by value
                f_best = f_x
            epochLoss += f_best                                           // record best for k-th epoch
            if (x - x_old).norm < toler then go = false                // stopping rule
        } // cfor

        println (s"x_last is $x and y(x_last) at the end is ${f(x)} and \n " +
                 s"lowest is $x_best and $f_best")
        (f_best, x_best)
    end solve

end SPSA


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sPSATest` main function tests the `SPSA` class.
 *  > runMain scalation.optimization.sPSATest
 */
@main def sPSATest (): Unit =

    banner ("Minimize: (x_0 - 3)^2 + (x_1 - 4)^2 + 1")

//  val noisen = Normal (0.0, 0.1)
    val noise  = Uniform (-0.1, 0.1)

    def f (x: VectorD): Double = (x(0) - 3)~^2 + (x(1) - 4)~^2 + 1 + noise.gen

    val x0 = VectorD (1, 2)
    val optimizer = new SPSA (f)
    optimizer.reset ()
    val opt = optimizer.solve (x0)
    println (s"][ optimal solution (f(x), x) = $opt")

    optimizer.plotLoss ()

end sPSATest

