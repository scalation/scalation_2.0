
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Sep 30 18:41:26 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Random Variate (RV) Generators
 *
 *  Many of the algorithms used are from:
 *    Averill M. Law and W. David Kelton
 *    Simulation Modeling and Analysis, 2nd Edition
 *    McGraw-Hill, Inc., NY, 1991.
 */

package scalation
package random

import scala.math.{exp, floor, log, max, Pi, round, sqrt, tan}
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat.{Histogram, Plot, VectorD}
import scalation.mathstat.Combinatorics.{betaF, choose, fac, gammaF}

import RandomSeeds.N_STREAMS

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Variate` abstract class serves as a base class for all the Random Variate
 *  (RV) generators.  They use one of the Random Number Generators (RNG's) from
 *  `Random` to generate numbers following their particular distribution.
 *  Random Variate Generators (RVG's) for thirty popular probability distributions
 *  are implemented as extensions of Variate.  Still need to add one.
 *  @see http://www.math.uah.edu/stat/special/index.html
 *  @see `VariateVec` for Random MultiVariate Generators (RMVG's).
 *-----------------------------------------------------------------------------
 *  @param  stream  the random number stream (0 until N_STREAMS)
 */
abstract class Variate (stream: Int = 0):

    protected val flaw = flawf ("Variate")

    /** Random number stream selected by the stream number (can't be beyond last stream)
     */
    protected val r = Random (stream % N_STREAMS)

    /** Allow (lax) calling igen on continuous distributions
     */
    private val LAX = true

    /** Indicates whether the distribution is discrete or continuous (default)
     */
    protected var _discrete = false

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the distribution is discrete or continuous.
     */
    def discrete: Boolean = _discrete

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Precompute the mean for the particular distribution.
     */
    val mean: Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability function (pf):  Either
     *  (a) the probability density function (pdf) for continuous RV's or
     *  (b) the probability mass function (pmf) for discrete RV's.
     *  @param z  the mass point whose probability density/mass is sought
     */
    def pf (z: Double): Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the entire probability mass function (pmf) for finite discrete RV's.
     *  FIX: make abstract and require extending classes to implement it.
     *  @param k  number of objects of the first type
     */
    def pmf (k: Int = 1): Array [Double] = Array.fill (k)(1.0/k.toDouble)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random number for the particular distribution.
     */
    def gen: Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random number for the particular distribution.
     *  This version allows one parameter.
     *  @param z  the limit parameter
     */
    def gen1 (z: Double): Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random integer for the particular distribution.
     *  It is only valid for discrete random variates.
     */
    def igen: Int =
        if LAX || _discrete then
            round (gen).toInt
        else 
            flaw ("igen", "should not be invoked on continuous RV's")
            0
    end igen

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random integer for the particular distribution.
     *  It is only valid for discrete random variates.
     *  This version allows one parameter.
     *  @param z  the limit parameter
     */
    def igen1 (z: Double): Int =
        if LAX || _discrete then
            round (gen1 (z)).toInt
        else 
            flaw ("igen", "should not be invoked on continuous RV's")
            0
    end igen1

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random string for the particular distribution.
     *  For better random strings, overide this method.
     */
    def sgen: String = "s" + "%g".format (gen)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random string for the particular distribution.
     *  For better random strings, overide this method.
     *  This version allows one parameter.
     *  @param z  the limit parameter
     */
    def sgen1 (z: Double): String = "s" + "%g".format (gen1 (z))

end Variate


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Bernoulli` random variates.
 *  This discrete RV models the one trial (success is 1, failure is 0).
 *  Note, the gen2 method gives a range of 2 (-1 to 1).
 *  @see http://www.math.uah.edu/stat/bernoulli/Introduction.html
 *  @param p       the probability of success
 *  @param stream  the random number stream
 */
case class Bernoulli (p: Double = .5, stream: Int = 0)
     extends Variate (stream):

    if p < 0.0 || p > 1.0 then flaw ("init", "parameter p must be in [0, 1]")
    _discrete = true
    private val q = 1.0 - p                                     // probability of failure

    val mean = p

    def pf (z: Double): Double = if z =~ 0.0 then q else if z =~ 1.0 then p else 0.0

    override def pmf (k: Int): Array [Double] = Array (q, p)

    def gen: Double = is (r.gen < p)                            // returns 0 or 1

    def gen1 (z: Double): Double = is (r.gen < z)

    def gen2 : Double = if r.gen < p then -1 else 1             // returns -1 or 1

end Bernoulli


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Beta` random variates.
 *  This continuous RV models random proportions.
 *  Beta =  Gamma1 / (Gamma1 + Gamma2).
 *  @see http://www.math.uah.edu/stat/special/Beta.html
 *  @param alpha   the shape parameter for Gamma1
 *  @param beta    the shape parameter for Gamma2
 *  @param stream  the random number stream
 */
case class Beta (alpha: Double = 2, beta: Double = 3, stream: Int = 0)
     extends Variate (stream):

    if alpha <= 0 || beta <= 0 then flaw ("init", "parameters alpha and beta must be positive")
    private val gamma1 = Gamma (alpha, 1.0, stream)
    private val gamma2 = Gamma (beta, 1.0, stream)

    val mean = alpha / (alpha + beta)

    def pf (z: Double): Double =
        if 0.0 < z && z < 1.0 then z~^(alpha-1.0) * (1.0-z)~^(beta-1.0) / betaF (alpha, beta)
        else 0.0
    end pf

    def gen: Double = { val g1 = gamma1.gen; g1 / (g1 + gamma2.gen) }

    def gen1 (z: Double): Double =
        val gamma1 = Gamma (z, 1.0, stream)
        val g1 = gamma1.gen; g1 / (g1 + gamma2.gen)
    end gen1

end Beta


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Binomial` random variates.
 *  This discrete RV models the number of successes in n trials.
 *  @see http://www.math.uah.edu/stat/bernoulli/Binomial.html
 *  @param p       the probability of success
 *  @param n       the number of independent trials
 *  @param stream  the random number stream
 */
case class Binomial (p: Double = .5, n: Int = 10, stream: Int = 0)
     extends Variate (stream):

    if p < 0.0 || p > 1.0 then flaw ("init", "parameter p must be in [0, 1]")
    if n <= 0 then             flaw ("init", "parameter n must be positive")
    _discrete = true
    private val q    = 1.0 - p                  // probability of failure
    private val p_q  = p / q                    // the ratio p divided by q
    private val coin = Bernoulli (p, stream)    // coin with prob of success of p

    val mean = p * n

    def pf (z: Double): Double = { val k = z.toInt; if z == k then pf (k) else 0.0 }

    def pf (k: Int): Double = if k in (0, n) then choose (n, k) * p~^k * q~^(n-k) else 0.0

    override def pmf (k: Int): Array [Double] =
        val d = Array.ofDim [Double] (n+1)      // array to hold pmf distribution
        d(0)  = q~^n
        for k <- 1 to n do d(k) = d(k-1) * p_q * (n-k+1) / k.toDouble
        d
    end pmf

    def gen: Double = summation (n)(coin.gen)

    def gen1 (z: Double): Double = summation (z.toInt)(coin.gen)

end Binomial


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Cauchy` (or Lorentz) random variates.
 *  This continuous RV models data with heavier tails than normally distributed.
 *  @see http://www.math.uah.edu/stat/special/Cauchy.html
 *  @param alpha   the location parameter (median)
 *  @param beta    the scale parameter 
 *  @param stream  the random number stream
 */
case class Cauchy (alpha: Double = 2.5, beta: Double = 1.0, stream: Int = 0)
     extends Variate (stream):

    if beta <= 0.0 then flaw ("init", "parameter beta must be positive")

    val mean = alpha                       // but, technically does not exist

    def pf (z: Double): Double = beta / (((z-alpha)~^2.0 + beta~^2.0) * Pi)

    def gen: Double = beta * tan (Pi * (r.gen-.5)) + alpha

    def gen1 (z: Double): Double = beta * tan (Pi * (r.gen-.5)) + z

end Cauchy


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `ChiSquare` random variates.
 *  This continuous RV models the variance of a distribution.
 *  @see www.math.uah.edu/stat/special/ChiSquare.html
 *  @param df      the degrees of freedom 
 *  @param stream  the random number stream
 */
case class ChiSquare (df: Int = 2, stream: Int = 0)
     extends Variate (stream):

    if df <= 0 then flaw ("init", "parameter df must be positive")

    private val gamma  = Gamma (df/2, 2.0, stream)
    private val normal = Normal (0.0, 1.0, stream)
    private val k      = df/2.0

    val mean = df.toDouble

    def pf (z: Double): Double = 0.5~^k * z~^(k-1) * exp (-z/2.0) / gammaF (k)

    def gen: Double = gamma.gen + (if df % 2 == 0 then 0.0 else normal.gen~^2.0)

    def gen1 (z: Double): Double = gamma.gen + (if z.toInt % 2 == 0 then 0.0 else normal.gen~^2.0)

end ChiSquare


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Dice` random variates for a given distribution specified
 *  using a cumulative distribution function (cdf). This discrete RV models the
 *  roll of dice numbered 0, 1, ..., n-1.  Add 1 for 1 to n.
 *  @see `Randi` for an easy way to use fair dice 'Randi (1, 6)'.
 *  @param cdf     the distribution function (cdf)
 *  @param stream  the random number stream
 */
case class Dice (cdf: Array [Double] = Array (.1, .3, .5, .7, .9, 1.0), stream: Int = 0)
     extends Variate (stream):

    _discrete = true
    private val n = cdf.length

    val mean = { var sum = 0.0; for i <- 1 until n do sum += i * (cdf(i) - cdf(i-1)); sum }

    def pf (z: Double): Double =
        val j = floor (z).toInt
        if j == z then if j == 0 then cdf(0) else cdf(j) - cdf(j-1)
        else 0.0
    end pf

    def gen: Double =
        var j = -1.0
        val y = r.gen
        var (go, i) = (true, 0)
        cfor (go && i < n, i += 1) {
            if y <= cdf(i) then { j = i.toDouble; go = false }
        } // cfor
        if j < 0.0 then n.toDouble - 1.0 else j
    end gen

    def gen1 (z: Double): Double =
        val ri = Randi0 (z.toInt, stream)
        ri.gen
    end gen1

end Dice


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates generalized `Discrete` random variates for a given
 *  distribution specified using either a probability mass function (pmf)
 *  or a cumulative distribution function (cdf).
 *  This discrete RV models arbitrary experiments with discrete outcomes.
 *  @param dist        the distribution function (pdf or cdf)
 *  @param x           the x-coordinate values (mass points)
 *  @param cumulative  whether 'dist' is cumulative (cdf) or not (pmf)
 *  @param stream      the random number stream
 */
case class Discrete (dist: VectorD = VectorD (.2, .2, .2, .2, .2), x: VectorD = null,
                     cumulative: Boolean = false, stream: Int = 0)
     extends Variate (stream):

    private val cdf = if cumulative then dist else dist.cumulate
    private val n   = dist.dim
    private val xx = if x == null || x.dim == 0 then VectorD.range (0, dist.dim) else x

    if xx.dim != dist.dim then flaw ("Discrete", "dist and xx must have the same length")
    _discrete = true

    val mean = { var sum = xx(0) * cdf(0); for i <- 1 until n do sum += xx(i) * (cdf(i) - cdf(i-1)); sum }

    def pf (z: Double): Double =
        var j = -1
        for i <- 0 until n if xx(i) == z do j = i
        if j >= 0 then if j == 0 then cdf(0) else cdf(j) - cdf(j-1)
        else 0 
    end pf

    def gen: Double =
        var xx_i = -0.0
        val y    = r.gen
        var (go, i) = (true, 0)
        cfor (go && i < n, i += 1) {
            if y <= cdf(i) then { xx_i = xx(i); go = false }
        } // cfor
        if go then xx(cdf.dim - 1)
        else xx_i
    end gen

    def gen1 (z: Double): Double =
        throw new UnsupportedOperationException ("'gen1' not implemented for Discrete")
    end gen1

end Discrete


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `DiscreteF` class generates generalized `Discrete` random variates for a
 *  distribution specified via an array of functions.  At a particular time 't',
 *  the functions must sum to one.
 *  This discrete RV models time-varying experiments with discrete outcomes.
 *  @param f       the array of time-based functions 
 *  @param stream  the random number stream
 *
case class DiscreteF (f: Array [Double => Double] = Array ((x: Double) => x), stream: Int = 0)
     extends Variate (stream):

    _discrete = true

    val n = f.length                        // the number of functions
    val x = VectorD.range (0, n)            // outcomes range 0 until n
    var t = 0.0                             // time

    val mean = sum (x) / n.toDouble

    def setTime (tt: Double): Unit = { t = tt }

    def pf (z: Double): Double = f(math.min (n-1, floor (z).toInt))(t)
   
    def gen: Double =
        val y = r.gen
        var sum = 0.0
        for i <- 0 until n do
            sum += f(i)(t)
            if y <= sum then return i.toDouble
        end for
        n
    end gen

    def gen1 (z: Double): Double =
        throw new UnsupportedOperationException ("'gen1' not implemented for DiscreteF")
    end gen1

    private def sum (w: VectorD): Double =
       var s = 0.0
       for i <- 0 until n do s += f(i)(t) * w(i)
       s 
    end sum

end DiscreteF
 */


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Erlang` random variates.
 *  This continuous RV models the time until k stages complete.
 *  Can also pass mu/k to provide an Exponential like distribution with cv < 1.
 *  @see http://www.math.uah.edu/stat/poisson/Gamma.html
 *  @param mu      the mean of exponential samples (Erlang mean = mu * k)
 *  @param k       the number of stages (or Exponential samples)
 *  @param stream  the random number stream
 */
case class Erlang (mu: Double = 1.0, k: Int = 2, stream: Int = 0)
     extends Variate (stream):

    if mu <= 0.0 || k <= 0 then flaw ("init", "parameters mu and k must be positive")

    private val λ = 1.0 / mu                     // lambda, the rate parameter

    val mean = mu * k

    def pf (z: Double): Double = λ~^k * z~^(k-1) * exp (-λ*z) / fac (k-1)

    def gen: Double =
        var prod = 1.0
        cfor (0, k) { _ => prod *= r.gen }
        -mu * log (prod)
    end gen

    def gen1 (z: Double): Double =
        var prod = 1.0
        cfor (0, k) { _ => prod *= r.gen }
        -z * log (prod)
    end gen1

end Erlang


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Erlang2S` (Shifted right) random variates.
 *  This continuous RV models the time until 2 stages complete.
 *  @param mu      the original unshifted mean of exponential samples (Erlang mean = mu * 2)
 *  @param tau     the time shift, generated values Y = tau + Erlang2
 *  @param stream  the random number stream
 */
case class Erlang2S (mu: Double = 1.0, tau: Double = 0.2, stream: Int = 0)
     extends Variate (stream):

    if tau <= 0.0 then flaw ("init", "parameter tau must be positive")
    if tau >= mu  then flaw ("init", "parameter tau must be less than mu")

    private val λ = 1.0 / mu             // lambda, the rate parameter

    val mean = tau + 2 * mu              // adjusted mean

    def pf (z: Double): Double = if z >= tau then λ~^2 * (z-tau) * exp (-λ*(z-tau))
                                 else 0.0

    def gen: Double = tau - mu * log (r.gen * r.gen)

    def gen1 (z: Double): Double = tau - z * log (r.gen * r.gen)

end Erlang2S


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Erlang2T` (lower-Truncated) random variates.
 *  This continuous RV models the time until 2 stages complete.
 *  It produces Erlang2 random variates until one is greater than tau.
 *  @param mu      the original untruncated mean of exponential samples (Erlang mean = mu * 2)
 *  @param tau     the time threshold, require generated values Y >= tau
 *  @param stream  the random number stream
 */
case class Erlang2T (mu: Double = 1.0, tau: Double = 0.2, stream: Int = 0)
     extends Variate (stream):

    if tau <= 0.0 then flaw ("init", "parameter tau must be positive")
    if tau >= mu  then flaw ("init", "parameter tau must be less than mu")

    private val λ = 1.0 / mu                                               // lambda, the rate parameter

    val mean = (λ * tau~^2 + 2 * tau + 2.0 / λ) / (1 + λ * tau)            // adjusted mean

    def pf (z: Double): Double = if z >= tau then (λ~^2 * z * exp (-λ * z)) / ((1 + λ * tau) * exp (-λ * tau))
                                 else 0.0

    def gen: Double =
        var x = 0.0
        while x < tau do x = -mu * log (r.gen * r.gen)
        x
    end gen

    def gen1 (z: Double): Double =
        var x = 0.0
        while x < tau do x = -z * log (r.gen * r.gen)
        x
    end gen1

end Erlang2T


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Erlang2T_` (lower-Truncated) random variates.
 *  This discrete/continuous RV models the time until 2 stages complete.
 *  It produces an Erlang2 random variate and takes the max with tau (causes a Mass Point).
 *  FIX - fails Mean Test
 *  @param mu      the original untruncated mean of exponential samples (Erlang mean = mu * 2)
 *  @param tau     the time threshold, require generated values Y >= tau
 *  @param stream  the random number stream
 */
case class Erlang2T_ (mu: Double = 1.0, tau: Double = 0.2, stream: Int = 0)
     extends Variate (stream):

    if tau <= 0.0 then flaw ("init", "parameter tau must be positive")
    if tau >= mu  then flaw ("init", "parameter tau must be less than mu")

    private val λ     = 1.0 / mu                                           // lambda, the rate parameter
    private val p_tau = 1 - exp (-λ * tau) - λ * tau * exp (-λ * tau)      // probability Y = tau (mass point)

    val mean = tau + (1 - p_tau) * 2 * mu                                  // adjusted mean

    def pf (z: Double): Double = if z > tau then λ~^2 * (z-tau) * exp (-λ*(z-tau))
                                 else if z == tau then POSITIVE_INFINITY   // 'p_tau * delta(y-tau)' Dirac delta function
                                 else 0.0

    def gen: Double = max (tau, -mu * log (r.gen * r.gen))

    def gen1 (z: Double): Double = max (tau, -z * log (r.gen * r.gen))

end Erlang2T_


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Exponential` random variates.
 *  This continuous RV models the time until an event occurs.
 *  As its stdev = mean, its coefficient of variation cv = 1.
 *  @see www.math.uah.edu/stat/poisson/Exponential.html
 *  @param mu      the mean (e.g., mean inter-arrival time)
 *  @param stream  the random number stream
 */
case class Exponential (mu: Double = 1.0, stream: Int = 0)
     extends Variate (stream):

    if mu <= 0.0 then flaw ("init", "parameter mu must be positive")

    private val λ = 1.0 / mu             // lambda, the rate parameter

    val mean = mu

    def pf (z: Double): Double = if z >= 0 then λ * exp (-λ*z) else 0.0

    def gen: Double = -mu * log (r.gen)

    def gen1 (z: Double): Double = -z * log (r.gen)

end Exponential


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `ExponentialS` (Shifted right) random variates.
 *  This continuous RV models the time until an event occurs.
 *  @note:  Due to memoryless property, this RV is the same as producing Exponential RVs
 *  until one is greater than tau.
 *  @param mu      the original unshifted mean (e.g., mean inter-arrival time)
 *  @param tau     the time shift, generated values Y = tau + Exponential
 *  @param stream  the random number stream
 */
case class ExponentialS (mu: Double = 1.0, tau: Double = 0.2, stream: Int = 0)
     extends Variate (stream):

    if tau <= 0.0 then flaw ("init", "parameter tau must be positive")
    if tau >= mu  then flaw ("init", "parameter tau must be less than mu")

    private val λ = 1.0 / mu             // lambda, the rate parameter

    val mean = tau + mu                  // adjusted mean

    def pf (z: Double): Double = if z >= tau then λ * exp (-λ*(z-tau))
                                 else 0.0

    def gen: Double = tau - mu * log (r.gen)

    def gen1 (z: Double): Double = tau - z * log (r.gen)

end ExponentialS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `ExponentialT_` (lower-Truncated) random variates.
 *  This discrete/continuous RV models the time until an event occurs.
 *  It produces an Exponential random variate and takes the max with tau (causes a Mass Point).
 *  @param mu      the original untruncated mean (e.g., mean inter-arrival time)
 *  @param tau     the time threshold, require generated values Y >= tau
 *  @param stream  the random number stream
 */
case class ExponentialT_ (mu: Double = 1.0, tau: Double = 0.2, stream: Int = 0)
     extends Variate (stream):

    if tau <= 0.0 then flaw ("init", "parameter tau must be positive")
    if tau >= mu  then flaw ("init", "parameter tau must be less than mu")

    private val λ     = 1.0 / mu                 // lambda, the rate parameter
    private val p_tau = 1 - exp (-λ * tau)       // probability Y = tau (mass point)

    val mean = tau + (1 - p_tau) * mu            // adjusted mean

    def pf (z: Double): Double = if z > tau then λ * exp (-λ*(z-tau))
                                 else if z == tau then POSITIVE_INFINITY   // 'p_tau * delta(y-tau)' Dirac delta function
                                 else 0.0

    def gen: Double = max (tau, -mu * log (r.gen))

    def gen1 (z: Double): Double = max (tau, -z * log (r.gen))

end ExponentialT_


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Fisher` (F-Distribution) random variates.
 *  This continuous RV models the ratio of variances.
 *  @see http://www.math.uacombined discrete-continuoush.edu/stat/special/Fisher.html
 *  @param df1     the degrees of freedom for numerator Chi-Square
 *  @param df2     the degrees of freedom for denominator Chi-Square
 *  @param stream  the random number stream
 */
case class Fisher (df1: Int = 6, df2: Int = 4, stream: Int = 0)
     extends Variate (stream):

    if df1 <= 0 || df2 <= 2 then
        flaw ("init", "parameters df1 and df2 must be at least 1 and 3, respectively")

    private val chi1 = ChiSquare (df1, stream)
    private val chi2 = ChiSquare (df2, stream)

    val mean = df2 / (df2-2.0)

    def pf (z: Double): Double =
        if z >= 0.0 then sqrt ((df1*z)~^df1 * df2.toDouble~^df2 / (df1*z + df2)~^(df1+df2)) /
                              (z * betaF (df1/2.0, df2/2.0))
        else 0.0
    end pf

    def gen: Double =
        val c1 = chi1.gen
        val c2 = chi2.gen
        (df2 * c1) / (df1 * c2)
    end gen

    def gen1 (z: Double): Double =
        val c1 = chi1.gen
        val c2 = chi2.gen
        (df2 * c1) / (z.toInt * c2)
    end gen1

end Fisher


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Gamma` random variates.
 *  This continuous RV models the time until an event occurs.
 *  Note: variance = alpha * beta ^ 2.
 *  @see http://www.math.uah.edu/stat/poisson/Gamma.html
 *  @param alpha   the shape parameter
 *  @param beta    the scale parameter
 *  @param stream  the random number stream
 */
case class Gamma (alpha: Double = 1.0, beta: Double = 1.0, stream: Int = 0)
     extends Variate (stream):

    if alpha <= 0.0 || beta <= 0.0 then flaw ("init", "parameters alpha and beta must be positive")

    private val a    = floor (alpha).toInt          // integral part
    private val b    = alpha - a                    // fractional part  
    private val erl1 = Erlang (beta, a, stream)
    private val erl2 = Erlang (beta, a + 1, stream)

    val mean = alpha * beta
    
    def pf (z: Double): Double =
        if z > 0.0 then beta~^(-alpha) * z~^(alpha-1.0) * exp (-z/beta) / gammaF (alpha)
        else 0.0
    end pf

    def gen: Double =
        var x = 0.0
        var y = 0.0
        if alpha < 1.0 then                               // 0 < alpha < 1
            while
                x = r.gen~^(1.0/alpha)
                y = r.gen~^(1.0/(1.0-alpha))
                x + y > 1
            do ()
            (x / (x + y)) * (-log (r.gen)) * beta
        else if alpha < 5.0 then                          // 1 <= alpha < 5
            while
                x = alpha / a
                var prod = 1.0
                cfor (0, a) { _ => prod *= r.gen }
                x *= -log (prod)
                r.gen > (x / alpha)~^b * exp (-b * x / (alpha-1.0))
            do ()
            x * beta
        else                                              // alpha >= 5
            if r.gen >= b then erl1.gen else erl2.gen
    end gen

    def gen1 (z: Double): Double =
        throw new UnsupportedOperationException ("'gen1' not implemented for Gamma")
    end gen1

end Gamma


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Geometric` random variates.
 *  This discrete RV models the number of failures before the first success.
 *  @see http://www.math.uah.edu/stat/bernoulli/Geometric.html
 *  @param p       the probability of success
 *  @param stream  the random number stream
 */
case class Geometric (p: Double = .5, stream: Int = 0)
     extends Variate (stream):

    if p < 0 || p > 1 then flaw ("init", "parameter p must be in [0, 1]")
    _discrete = true

    private val q      = 1.0 - p
    private val log_q  = log (q)
    private val q_by_p = q / p

    val mean = q_by_p

    def pf (z: Double): Double = 
        val k = floor (z).toInt
        if z == k && k >= 0 then p * q~^k else 0.0
    end pf

    def gen: Double = floor (log (r.gen) / log_q).toInt

    def gen1 (z: Double): Double =
        val q      = 1.0 - z
        val log_q  = log (q)
        floor (log (r.gen) / log_q).toInt
    end gen1

end Geometric


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `HyperExponential` random variates (two rates).
 *  This continuous RV models the time until an event occurs (higher coefficient
 *  of variation than exponential distribution).  
 *  @param p       the probability of first vs. second rates
 *  @param mu1     the first mean (1 / lambda1)
 *  @param mu2     the second mean (1 / lambda2)
 *  @param stream  the random number stream
 */
case class HyperExponential (p: Double = .5, mu1: Double = 1, mu2: Double = 2, stream: Int = 0)
     extends Variate (stream):

    if p < 0.0 || p > 1.0 then       flaw ("init", "parameter p must be in [0, 1]")
    if mu1 <= 0.0 || mu2 <= 0.0 then flaw ("init", "parameters mu1 and mu2 must be positive")

    private val q  = 1.0 - p
    private val l1 = 1.0 / mu1      // lambda 1
    private val l2 = 1.0 / mu2      // lambda 2

    val mean = p * mu1 + q * mu2

    def pf (z: Double): Double = if z >= 0.0 then p * l1 * exp (-l1*z) + q * l2 * exp (-l2*z)
                                 else 0.0

    def gen: Double = log (r.gen) * (if r.gen < p then -mu1 else -mu2)

    def gen1 (z: Double): Double = log (r.gen) * (if r.gen < z then -mu1 else -mu2)

end HyperExponential


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `HyperExponential_` random variates.
 *  This continuous RV models the time until an event occurs (higher coefficient
 *  of variation than exponential distribution). FIX
 *  @param mu      the mean
 *  @param sigma   the standard deviation
 *  @param stream  the random number stream
 */
case class HyperExponential_ (mu: Double = 1.0, sigma: Double = 2, stream: Int = 0)
     extends Variate (stream):

    if mu <= 0.0 || sigma <= 0.0 then flaw ("init", "parameters mu and sigma must be positive")

    private val cv2 = (sigma / mu)~^2.0
    private val p   = .5 * (1.0 - sqrt ((cv2-1.0) / (cv2+1.0)))

    private val l = 2.0 * p / mu   // adjusted lambda

    val mean = mu

    def pf (z: Double): Double = if z >= 0.0 then l * exp (-l*z) else 0.0

    def gen: Double =
        val z = if r.gen > p then mu / (1 - p) else mu / p
        -0.5 * z * log (r.gen)
    end gen

    def gen1 (z: Double): Double =
        val cv2 = (sigma / mu)~^2.0
        val p   = .5 * (1.0 - sqrt ((cv2-1.0) / (cv2+1.0)))
        val zz = if r.gen > p then z / (1 - p) else z / p
        -0.5 * zz * log (r.gen)
    end gen1

end HyperExponential_


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `HyperGeometric` random variates.
 *  This discrete RV models the number of successes in n draws from a finite population.
 *  @see http://www.math.uah.edu/stat/urn/Hypergeometric.html
 *  @param p       the probability of success (red balls)
 *  @param n       the number of draws (balls drawn)
 *  @param pop     the size of the finite population (total number of balls)
 *  @param stream  the random number stream
 */
case class HyperGeometric (p: Double = .5, n: Int = 5, pop: Int = 10, stream: Int = 0)
     extends Variate (stream):

    if p < 0.0 || p > 1.0 then flaw ("init", "parameter p must be in [0, 1]")
    if n <= 0 || pop <= 0 then flaw ("init", "parameters n and pop must be positive")
    _discrete = true

    private val reds: Int = floor (p * pop).toInt    // number of red balls

    val mean = n * reds / pop.toDouble

    def pf (z: Double): Double =
        val k = floor (z).toInt
        if k == z && 0 <= k && k <= reds && k <= n then
            choose (reds, k) * choose (pop-reds, n-k) / choose (pop, n).toDouble
        else 0.0
    end pf

    def gen: Double =
        var b: Double = pop                      // population of number of balls
        var rd = reds                            // number of red/success balls in population
        var s = 0                                // count number of successes
        cfor (0, n) { _ =>
            if r.gen <= rd / b then { s += 1; rd -= 1 }
            b -= 1
        } // cfor
        s
    end gen

    def gen1 (z: Double): Double =
        val reds: Int = floor (z * pop).toInt    // number of red balls
        var b: Double = pop                      // population of number of balls
        var rd = reds                            // number of red/success balls in population
        var s = 0                                // count number of successes
        cfor (0, n) { _ =>
            if r.gen <= rd / b then { s += 1; rd -= 1 }
            b -= 1
        } // cfor
        s
    end gen1

end HyperGeometric


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Known` random variates.
 *  The values come from a know array of values (useful for testing).
 *  @param values  the known array of values
 *  @param stream  the random number stream
 */
case class Known (values: Array [Double], stream: Int = 0)
     extends Variate (stream):

    private var i = -1
    private val n = values.length

    def pf (z: Double): Double = if values contains z then 1.0/n else 0.0

    val mean = values.sum / n

    def gen: Double = { i = (i + 1) % n; values(i) }

    def gen1 (z: Double): Double =
        throw new UnsupportedOperationException ("'gen1' not implemented for Known")
    end gen1

end Known


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Logistic` random variates.
 *  This continuous RV models logistically distributed data (stretched Normal).
 *  @see http://www.math.uah.edu/stat/special/Logistic.html
 *  @param a       the location parameter
 *  @param b       the scale parameter
 *  @param stream  the random number stream
 */
case class Logistic (a: Double = 0.0, b: Double = 1.0, stream: Int = 0)
     extends Variate (stream):

    val mean = a

    def pf (z: Double): Double =
        val y = exp ((z - a) / b)
        y / (b * (1.0 + y)~^2.0)
    end pf

    def gen: Double =
        val y = r.gen
        a + b * log (y / (1.0 - y))
    end gen

    def gen1 (z: Double): Double =
        val y = r.gen
        z + b * log (y / (1.0 - y))
    end gen1

end Logistic


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `LogNormal` random variates.
 *  This continuous RV models data that is normally distributed after a log transformation.
 *  @see http://www.math.uah.edu/stat/special/LogNormal.html
 *  @param mu      the mean for underlying Normal
 *  @param sigma2  the variance (sigma squared) for underlying Normal
 *  @param stream  the random number stream
 */
case class LogNormal (mu: Double = 0.0, sigma2: Double = 1.0, stream: Int = 0)
     extends Variate (stream):

    if sigma2 <= 0.0 then flaw ("init", "parameter sigma2 must be positive")

    private val _2sigma2 = 2.0 * sigma2
    private val normal   = Normal (mu, sigma2, stream)   // associated Normal distribution

    val mean = exp (mu + sigma2/2.0)

    def pf (z: Double): Double =
        val denom = z * sqrt (Pi * _2sigma2)
        if z > 0.0 then exp (-(log (z) - mu)~^2.0 / _2sigma2) / denom else 0.0
    end pf

    def gen: Double = exp (normal.gen)

    def gen1 (z: Double): Double =
        val normal = Normal (z, sigma2, stream)          // associated Normal distribution
        exp (normal.gen)
    end gen1

end LogNormal


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `NegativeBinomial` random variates.
 *  This discrete RV models the number of failures before s-th success.
 *  @see http://www.math.uah.edu/stat/bernoulli/NegativeBinomial.html
 *  @param p       the probability of success
 *  @param s       the number of successes
 *  @param stream  the random number stream
 */
case class NegativeBinomial (p: Double = .5, s: Int = 2, stream: Int = 0)
     extends Variate (stream):

    if p < 0.0 || p > 1.0 then flaw ("init", "parameter p must be in [0, 1]")
    if s <= 0 then             flaw ("init", "parameter s must be positive")
    _discrete = true

    private val q    = 1.0 - p
    private val geom = Geometric (p, stream)

    val mean = s * q / p

    def pf (z: Double): Double =
        val k = floor (z).toInt
        if k == z && k >= 0 then choose (s+k-1, k) * p~^s * q~^k
        else 0.0
    end pf

    def gen: Double = 
        var sum = 0
        cfor (0, s) { _ => sum += geom.gen.toInt }
        sum
    end gen

    def gen1 (z: Double): Double =
        val geom = Geometric (z, stream)
        var sum = 0
        cfor (0, s) { _ => sum += geom.gen.toInt }
        sum
    end gen1

end NegativeBinomial


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Normal` (Gaussian) random variates.
 *  This continuous RV models normally distributed data (bell curve).
 *  When summed, most distributions tend to Normal (Central Limit Theorem).
 *  @see http://www.math.uah.edu/stat/special/Normal.html
 *  @param mu      the mean
 *  @param sigma2  the variance (sigma squared)
 *  @param stream  the random number stream
 */
case class Normal (mu: Double = 0.0, sigma2: Double = 1.0, stream: Int = 0)
     extends Variate (stream):

    if sigma2 < 0.0 then flaw ("init", "parameter sigma2 must be nonnegative")

    private val sigma    = sqrt (sigma2)
    private val _2sigma2 = - 2.0 * sigma2
    private var computed = true                      // toggle, since values gen in pairs
    private var save     = 0.0                       // save second in pair

    val mean = mu

    def pf (z: Double): Double = (1.0 / (sigma * sqrt_2Pi)) * exp ((z-mu)~^2 / _2sigma2)

    def gen: Double =                                // use acceptance-rejection method
        var (a, b, w, t) = (0.0, 0.0, 0.0, 0.0)

        computed = ! computed
        if computed then return save * sigma + mu
        while
            a = 2.0 * r.gen - 1.0
            b = 2.0 * r.gen - 1.0
            w = a*a + b*b
            w > 1.0
        do ()
        t    = sqrt (-2.0 * log (w) / w)
        save = b * t
        (a * t) * sigma + mu
    end gen

    def gen0: Double = Quantile.normalInv (r.gen)     // use inverse transform method

    def gen1 (z: Double): Double =
        var (a, b, w, t) = (0.0, 0.0, 0.0, 0.0)

        computed = ! computed
        if computed then return save * sigma + z
        while
            a = 2.0 * r.gen - 1.0
            b = 2.0 * r.gen - 1.0
            w = a*a + b*b
            w > 1.0
        do ()
        t    = sqrt (-2.0 * log (w) / w)
        save = b * t
        (a * t) * sigma + z
    end gen1

end Normal


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Pareto` random variates.
 *  This continuous RV models Pareto distributed data.
 *  @see http://www.math.uah.edu/stat/special/Pareto.html
 *  @param a       the shape parameter
 *  @param b       the scale parameter
 *  @param stream  the random number stream
 *
case class Pareto (a: Double = 0.5, b: Double = 1.0, stream: Int = 0)
     extends Variate (stream):

    if a <= 0.0 || b < 0.0 then flaw ("init", "parameters 'a' and 'b' must be positive, nonegative")
    if a >= 1.0 then flaw ("init", "parameter 'a' must be less than 1")

    val mean = a * b / (a-1.0)

    def pf (z: Double): Double = if z >= b then a * b~^a / z~^(a+1.0) else 0.0

    def gen: Double = b / (1.0 - r.gen)~^(1.0/a)

    def gen1 (z: Double): Double = b / (1.0 - r.gen)~^(1.0/z)

end Pareto
 */


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Poisson` random variates (discrete).
 *  This discrete RV models the number of events in a time interval of unit length.
 *  @see http://www.math.uah.edu/stat/poisson/Poisson.html
 *  @param mu      the mean
 *  @param stream  the random number stream
 */
case class Poisson (mu: Double = 2.0, stream: Int = 0)
     extends Variate (stream):

    if mu <= 0.0 then flaw ("init", "parameter mu must be positive")
    _discrete = true

    private val cutoff = exp (-mu)

    val mean = mu

    def pf (z: Double): Double =
        val k = floor (z).toInt
        if k == z && k >= 0 then exp (-mu) * mu~^k / fac (k) else 0.0
    end pf

    def gen: Double = 
        var n = -1
        var prod = 1.0
        while
            prod *= r.gen
            n    += 1
            prod >= cutoff
        do ()
        n
    end gen

    def gen1 (z: Double): Double =
        val cutoff = exp (-z)
        var n      = -1
        var prod   = 1.0
        while
           prod *= r.gen
           n    += 1
           prod >= cutoff
        do  ()
        n
    end gen1

end Poisson


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `PowerLaw` random variates:  'cx^-y' for 'x in [a, b]'.
 *  This continuous RV models power-law distributions.
 *  @see http://mathworld.wolfram.com/RandomNumber.html
 *  @param a       the minimum value parameter
 *  @param b       the maximum value parameter
 *  @param y       the power parameter to be used
 *  @param stream  the random number stream
 */
case class PowerLaw (a: Double = 1.0, b: Double = 10.0, y: Double = 2.1, stream: Int = 0)
     extends Variate (stream):

    if a >= b then  flaw ("init", "requires parameter a < b")
    if y < 1.0 then flaw ("init", "parameter y must be >= 1.0")

    private val exp  = 1.0 - y                            // exponent
    private val a_up = a~^exp
    private val b_up = b~^exp
    private val diff = b~^exp - a_up
    private val root = 1.0 / exp.toDouble
    private val c    = exp / diff                         // coefficient

    val mean = c * (b*b_up - a*a_up) / (exp + 1.0)

    def pf (z: Double): Double = if z >= a && z <= b then c * z~^(-y) else 0.0

    def gen: Double = (diff * r.gen  + a_up) ~^ root

    def gen1 (z: Double): Double =
        val a_up = z~^exp
        val diff = b~^exp - a_up
        (diff * r.gen  + a_up) ~^ root
    end gen1

end PowerLaw


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Randi` random variates (random integers: a, ..., b).
 *  This discrete RV models equi-probable integral outcomes.
 *  @see http://www.math.uah.edu/stat/special/UniformDiscrete.html
 *  @param a       the lower bound (inclusive)
 *  @param b       the upper bound (inclusive)
 *  @param stream  the random number stream
 */
case class Randi (a: Int = 0, b: Int = 5, stream: Int = 0)
     extends Variate (stream):

    if a > b then flaw ("init", "parameter a must not be greater than b")
    _discrete = true

    private val width = b + 1 - a

    val mean = (a + b) / 2.0

    def pf (z: Double): Double =
        val k = floor (z).toInt
        if k == z && a <= k && k <= b then 1.0 / width.toDouble else 0.0
    end pf

    def gen: Double = floor (a + width * r.gen)

    def gen1 (z: Double): Double = { val width = z.toInt + 1 - a; floor (a + width * r.gen) }

end Randi

object Randi:

    def apply (a_b: (Int, Int), stream: Int): Randi = Randi (a_b._1, a_b._2, stream)

end Randi 


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Randi0` random variates (random integers: 0, ..., b).
 *  This discrete RV models equi-probable integral outcomes starting with 0.
 *  @param b       the upper bound (>= 0) (inclusive)
 *  @param stream  the random number stream
 */
case class Randi0 (b: Int = 5, stream: Int = 0)
     extends Variate (stream):

    if b < 0 then flaw ("init", "parameter b must be non-negative")
    _discrete = true

    private val width = b + 1

    val mean = b / 2.0

    def pf (z: Double): Double =
        val k = floor (z).toInt
        if k == z && k <= b then 1.0 / width.toDouble else 0.0
    end pf

    def gen: Double = floor (width * r.gen)

    def gen1 (z: Double): Double = floor ((z.toInt + 1) * r.gen)

end Randi0


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `RandiU0` random variates (unique random integers: 0, ..., b).
 *  This discrete RV models equi-probable integral outcomes starting with 0.
 *  @param b       the upper bound (>= 0) (inclusive)
 *  @param stream  the random number stream
 */
case class RandiU0 (b: Int = 1000, stream: Int = 0)
     extends Variate (stream):

    if b < 0 then flaw ("init", "parameter b must be non-negative")
    _discrete = true

    val previous = collection.mutable.Set [Int] ()

    private val width = b + 1

    val mean = b / 2.0

    def pf (z: Double): Double =
        val k = floor (z).toInt
        if k == z && k <= b then 1.0 / width.toDouble else 0.0
    end pf

    def gen: Double = igen.toDouble

    def gen1 (z: Double): Double = igen1 (z).toDouble

    override def igen: Int = igen1 (b)

    override def igen1 (z: Double): Int = 
        val bb = z.toInt
        if previous.size == bb then
            flaw ("igen1", "all unique values have been exhausted - starting over")
            previous.clear ()
        var i = -1
        while
            i = floor ((bb + 1) * r.gen).toInt
            previous contains i
        do ()
        previous += i
        i
    end igen1

end RandiU0


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Sharp` (Deterministic) random variates.
 *  This discrete RV models the case when the variance is 0.
 *  @param x       the value for this constant distribution
 *  @param stream  the random number stream
 */
case class Sharp (x: Double = 1.0, stream: Int = 0)
     extends Variate (stream):

    _discrete = true

    val mean = x

    def pf (z: Double): Double = is (z =~ x)

    def gen: Double = x

    def gen1 (z: Double): Double = z

end Sharp


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `StdNormal` random variates using the acceptance-rejection
 *  method.
 *  @see http://www.columbia.edu/~ks20/4703-Sigman/4703-07-Notes-ARM.pdf
 *  @see https://www.math.arizona.edu/~tgk/mc/book_chap4.pdf
 *  @param stream  the random number stream
 */
case class StdNormal (stream: Int = 0)
     extends Variate (stream):

    val mean = 0.0

    def pf (z: Double): Double = (1.0 / sqrt_2Pi) * exp (z~^2 / -2.0)

    def gen: Double =
        var z = -0.0
        var cont = true
        while
            val y1 = -log (r.gen)                     // Exponential distribution
            val y2 = -log (r.gen)
            if y2 >= (y1 - 1)~^2 / 2.0 then
                z = y1
                cont = false
            cont
        do ()
        if r.gen <= 0.5 then z else -z
    end gen

    def gen1 (z: Double): Double =
        throw new UnsupportedOperationException ("'gen1' not implemented for StdNormal")
    end gen1

end StdNormal


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `StudentT` (Student's t) random variates.
 *  This continuous RV models cases where data are normally distributed, but
 *  variability increases since the variance is unknown.
 *  @see http://www.math.uah.edu/stat/special/Student.html
 *  @param df      the degrees of freedom
 *  @param stream  the random number stream
 */
case class StudentT (df: Int = 4, stream: Int = 0)
     extends Variate (stream):

    if df <= 0 then flaw ("init", "parameter df must be positive")

    private val _df    = (df + 1) / 2.0
    private val normal = Normal (0.0, 1.0, stream)
    private val chi    = ChiSquare (df, stream)

    val mean = 0.0

    def pf (z: Double): Double =
        gammaF (_df) * (1.0 + (z*z)/df)~^(-_df) / (sqrt (df*Pi) * gammaF (df/2.0))
    end pf

    def gen: Double = normal.gen / sqrt (chi.gen / df) 

    def gen1 (z: Double): Double =
        val chi = ChiSquare (z.toInt, stream)
        normal.gen / sqrt (chi.gen / z.toInt)
    end gen1

end StudentT


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Trapezoidal` random variates.
 *  This continuous RV models cases where outcomes cluster between two modes.
 *  Both `Uniform and `Triangular` are special cases of `Trapezoidal`.
 *  @see iopscience.iop.org/0026-1394/44/2/003/pdf/0026-1394_44_2_003.pdf
 *  @param a       the minimum
 *  @param c       the first mode
 *  @param d       the second mode
 *  @param b       the maximum
 *  @param stream  the random number stream
 */
case class Trapezoidal (a: Double = 0.0, c: Double = 2.0, d: Double = 7.0, b: Double = 10.0,
                        stream: Int = 0)
     extends Variate (stream):

    private val l  = b + d - a - c                     // length factor
    private val h  = 2.0 / l                           // maximum height
    private val lf = c - a                             // triangle on left
    private val rt = b - d                             // triangle on right

    val mean = (b~^2 - a~^2 + d~^2 - c~^2 - a*c + b*d) / (3.0 * l)

    def pf (z: Double): Double =
        if a <= z && z <= c then h * (z - a) / lf 
        else if z <= d      then h
        else if z <= b      then h * (b - z) / rt
        else 0.0
    end pf

    def gen: Double = 
        val y   = r.gen
        val h_2 = h / 2.0
        if y <= h_2 * lf            then a + sqrt (2.0 * lf * y / h)
        else if y <= 1.0 - h_2 * rt then (a + c) / 2.0 + y / h
        else b - sqrt (2.0 * rt * (1.0 - y) / h)
    end gen

    def gen1 (z: Double): Double =
        val l  = z + d - a - c                         // length factor
        val h  = 2.0 / l                               // maximum height
        val lf = c - a                                 // triangle on left
        val rt = z - d                                 // triangle on right
        val y   = r.gen
        val h_2 = h / 2.0
        if y <= h_2 * lf            then a + sqrt (2.0 * lf * y / h)
        else if y <= 1.0 - h_2 * rt then (a + c) / 2.0 + y / h
        else z - sqrt (2.0 * rt * (1.0 - y) / h)
    end gen1

end Trapezoidal


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates simple `Triangular` random variates with the mode in the middle.
 *  This continuous RV models cases where outcomes cluster around the mode.
 *  @see http://www.math.uah.edu/stat/special/Triangle.html
 *  @param a       the lower bound
 *  @param b       the upper bound
 *  @param c       the mode
 *  @param stream  the random number stream
 */
case class Triangular (a: Double = 0, b: Double = 5, c: Double = Double.MaxValue, stream: Int = 0)
     extends Variate (stream):

    if a > b then flaw ("init", "parameter a must not be greater than b")

    private val width = b - a
    private val mode  = if c < Double.MaxValue then c else (a + b) / 2.0
    private val left  = mode - a
    private val right = b - mode

    val mean = (a + mode + b) / 3.0

    def pf (z: Double): Double =
        if a <= z && z <= mode then 2 * (z - a) / (width * left)
        else if z <= b         then 2 * (b - z) / (width * right)
        else 0
    end pf

    def gen: Double = 
        val y = r.gen    
        if y <= left / width then a + sqrt (left * width * y)
        else b - sqrt (right * width * (1.0 - y))
    end gen

    def gen1 (z: Double): Double =
        val width = z - a
        val mode  = if c < Double.MaxValue then c else (a + z) / 2.0
        val left  = mode - a
        val right = z - mode
        val y = r.gen
        if y <= left / width then a + sqrt (left * width * y)
        else z - sqrt (right * width * (1.0 - y))
    end gen1

end Triangular


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Trinomial` random variates.  While Binomial is based on
 *  trials with two outcomes, success (1) or failure (0).  Trinomial is based on
 *  trials with three outcomes, high (2), medium (1) or low (0).
 *  This discrete RV models the result of 'n' trials.
 *  @see onlinecourses.science.psu.edu/stat414/node/106
 *  FIX - fails distribution test - maybe due to 'pf' method
 *  @param p       the probability of high (2)
 *  @param q       the probability of medium (1)
 *  @param n       the number of independent trials
 *  @param stream  the random number stream
 */
case class Trinomial (p: Double = 1.0/3.0, q: Double = 1.0/3.0, n: Int = 5, stream: Int = 0)
     extends Variate (stream):

    if p < 0.0   then flaw ("init", "parameter p must be non-negative")
    if q < 0.0   then flaw ("init", "parameter q must be non-negative")
    if p+q > 1.0 then flaw ("init", "p+q must not exceed one")
    if n <= 0    then flaw ("init", "parameter n must be positive")
    _discrete = true

    private val qq   = 1.0 - p - q               // the probability of low (0)
//  private val p_qq = p / qq                    // the ratio of high to low
    private val q_qq = q / qq                    // the ratio of medium to low
    private val dice = Dice (Array (qq, qq+q, 1.0), stream)

    val mean = (q + 2.0*p) * n

    def pf (z: Double): Double =
        val k = floor (z).toInt                  // k and l in lock step (one dimensional)
        val l = floor (z).toInt
        if z == k && 0 <= k && k+l <= n then
            choose (n, k, l) * p~^k * q~^l * qq~^(n-k-l)
        else 0.0
    end pf

    def pf (y: Double, z: Double): Double =
        val k = floor (y).toInt   // k and l independent (two dimensional)
        val l = floor (z).toInt
        if y == k && z == l && 0 <= k && 0 <= l && k+l <= n then
            choose (n, k, l) * p~^k * q~^l * qq~^(n-k-l)
        else 0.0
    end pf

    def pf (k: Int, l: Int): Double =                   // ex: n = 10, (k, l, m) = (2, 3, 5)
        if 0 <= k && 0 <= l && k+l <= n then
            choose (n, k, l) * p~^k * q~^l * qq~^(n-k-l)
        else 0.0
    end pf

    override def pmf (k: Int): Array [Double] =
        val d = Array.ofDim [Double] (n-k+1)            // array to hold pmf distribution
        d(0) = choose (n, k) * p~^k * qq~^n-k
        for l <- 1 to n-k do d(l) = d(l-1) * q_qq * (k-l+1) / l.toDouble
        d
    end pmf

    def gen: Double =
        var sum = 0.0
        cfor (0, n) { _ => sum += dice.gen }            // add 0, 1 or 2
        sum
    end gen

    def gen1 (z: Double): Double =
        var sum = 0.0
        cfor (0, z.toInt) { _ => sum += dice.gen }      // add 0, 1 or 2
        sum
    end gen1

end Trinomial


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Uniform` random variates in the range (a, b).
 *  This continuous RV models equi-probable outcomes.
 *  @see http://www.math.uah.edu/stat/special/UniformContinuous.html
 *  @param a       the lower bound
 *  @param b       the upper bound
 *  @param stream  the random number stream
 */
case class Uniform (a: Double = 0.0, b: Double = 5.0, stream: Int = 0)
     extends Variate (stream):

    if a > b then flaw ("init", "parameter a must not be greater than b")

    private val width = b - a

    val mean = (a + b) / 2.0

    def pf (z: Double): Double = if a <= z && z <= b then 1.0 / width else 0.0

    def gen: Double = a + width * r.gen

    def gen1 (z: Double): Double = { val width = z - a; a + width * r.gen }

end Uniform

object Uniform:

    def apply (a_b: (Double, Double), stream: Int): Uniform = Uniform (a_b._1, a_b._2, stream)

end Uniform


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates `Weibull` random variates.
 *  This continuous RV models the time for an event to occur.
 *  @see http://www.math.uah.edu/stat/special/Weibull.html
 *  @param alpha   the shape parameter
 *  @param beta    the scale parameter
 *  @param stream  the random number stream
 */
case class Weibull (alpha: Double = 1.0, beta: Double = 2.0, stream: Int = 0)
     extends Variate (stream):

    if alpha <= 0.0 || beta <= 0.0 then flaw ("init", "parameters alpha and beta must be positive")

    private val shape_recip = 1.0 / alpha

    val mean = beta * shape_recip * gammaF (shape_recip)

    def pf (z: Double): Double =
        if z > 0.0 then alpha * beta~^(-alpha) * z~^(alpha-1.0) * exp (-(z/beta)~^alpha)
        else 0.0
    end pf

    def gen: Double = -beta * log (r.gen)~^shape_recip

    def gen1 (z: Double): Double = -z * log (r.gen)~^shape_recip

end Weibull


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `variateTest` main function conducts two simple tests of the Random Variate
 *  Generators: (1) Means Test and (2) Chi-square Goodness of Fit Test.
 *  FIX: need to add (3) Variance Test and (4) K-S Goodness of Fit Test.
 *  > runMain scalation.random.variateTest
 */
@main def variateTest (): Unit =

    import mathstat.{Statistic, z_meanTest}

    val SIGNIFICANCE  = 0.99

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform a Means Test (average of generated rv's close to mean for distribution).
     *  @param rv  the random variate to test
     */
    def meansTest (rv: Variate): Unit =
        banner ("Mean Test the `" + rv.getClass.getSimpleName () + "` random variate generator")

        var ran = 0.0
        val stat = new Statistic ()
        val rep = 100000
        print ("rv.gen  = {")
        for i <- 1 to rep do
            ran = rv.gen
            if i <= 10 then print (" " + ran)
            stat.tally (ran)
        end for
        println (" ... }")
        assert (z_meanTest (rv.mean, stat.mean, stat.stdev, stat.num, SIGNIFICANCE))
    end meansTest

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform a Chi-square Goodness of Fit Test.  Compare the random variable's
     *  histogram (generated by repeatedly calling 'gen') to the probability
     *  function pf (either pmf or pdf).
     *  @param rv  the random variate to test
     */
    def distributionTest (rv: Variate): Unit =
        banner ("Fit Test the " + rv.getClass.getSimpleName () + " random variate generator")

        val rep    = 50000          // replications
        var j      = 0              // interval number
        var x      = 0.0            // x coordinate
        var o      = 0.0            // observed value: height of histogram
        var e      = 0.0            // expected value: pf (x)
        var chi2   = 0.0            // ChiSquare statistic
        var n      = 0              // number of nonzero intervals
        val offset = if rv.isInstanceOf [StudentT] ||
                        rv.isInstanceOf [Normal] ||
                        rv.isInstanceOf [StdNormal] then 25 else 0
        val sum    = new Array [Int] (51)

        cfor (0, rep) { _ =>
            j = floor (rv.gen * 10.0).toInt + offset
            if 0 <= j && j <= 50 then sum (j) += 1
        } // cfor

        for i <- 0 until sum.length do
            x = (i - offset) / 10.0
            o = sum(i)
            e = round (if rv.discrete then 50000 * rv.pf (x)
                       else 5000 * rv.pf (x + .05)).toDouble
            if e >= 5 then
                chi2 += (o-e)~^2.0 / e
                n += 1
            print (s"\tsum ($x) = $o : $e ")
            if i % 5 == 4 then println ()
        end for
        n -= 1
        if n < 2  then n = 2
        if n > 49 then n = 49 
        
        val criticalValue = Quantile.chiSquareInv (SIGNIFICANCE, n)
        println (s"\nchi^2 = $chi2 <? chi^2($SIGNIFICANCE, $n) = $criticalValue")
        assert (chi2  < criticalValue)
    end distributionTest

    val distribution = Array (Bernoulli (),
                              Beta (),
                              Binomial (),
                              Cauchy (),
                              ChiSquare (),
                              Dice (),
                              Discrete (),
// useful?                    DiscreteF (),
                              Erlang (),
                              Erlang2S (),
                              Erlang2T (),
// fails                      Erlang2T_ (),
                              Exponential (),
                              ExponentialS (),
// fails Fit Test             ExponentialT_ (),
                              Fisher (),
                              Gamma (),
                              Geometric (),
                              HyperExponential (),
                              HyperGeometric (),
// barely fails               LogNormal (),
                              Logistic (),
                              NegativeBinomial (),
                              Normal (),
// fails                      Pareto (),
                              Poisson (),
                              PowerLaw (),
                              Randi (),
                              Randi0 (),
                              RandiU0 (),
                              Sharp (),
                              StdNormal (),
                              StudentT (),
                              Trapezoidal (),
                              Triangular (),
//                            Trinomial (),
                              Uniform (),
                              Weibull ())

    val testAll = true                        // test all distributions, else only those in 'include' set
    val include = Set (7, 11)                 // adjust as needed
    val nTests  = if testAll then distribution.length else include.size
    println ("Testing " + nTests + " Random Variate Generators")

    for i <- distribution.indices do
        if testAll || (include contains i) then
            meansTest (distribution (i))
            distributionTest (distribution (i))
    end for

    val trinom = Trinomial (1.0/3.0, 1.0/3.0, 9)
    println ("trinom = " + stringOf (trinom.pmf(0)))

end variateTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `diceTest` main function is used to test the `Randi` random variate generator
 *  for the case of rolling two dice.  The probability mass function (pmf) is plotted.
 *  > runMain scalation.random.diceTest
 */
@main def diceTest (): Unit =

    val dice  = Randi (1, 6)
    val x     = VectorD.range (0, 7)
    val freq  = new VectorD (7)
    val x2    = VectorD.range (0, 13)
    val freq2 = new VectorD (13)
    cfor (0, 10000) { _ =>
        val sum  = dice.igen
        val sum2 = dice.igen + dice.igen
        freq(sum)   += 1
        freq2(sum2) += 1
    } // cfor
    println (s"x    = $x")
    println (s"freq = $freq")
    println (s"x2    = $x2")
    println (s"freq2 = $freq2")
    new Plot (x, freq)
    new Plot (x2, freq2)

end diceTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cLTTest` main function is used to test the `Uniform` random variate generator
 *  for illustrating the Central Lmiit Theorem.  Try adding difference numbers of
 *  random values and other distributions.
 *  > runMain scalation.random.cLTTest
 */
@main def cLTTest (): Unit =

    import mathstat.Plot

    val rvg = Uniform ()
    val x   = VectorD (for _ <- 0 until 100000 yield rvg.gen + rvg.gen + rvg.gen + rvg.gen)
    new Histogram (x)

    val nrm = Normal ()
    val y   = VectorD (for i <- 0 until 100 yield (i/20.0 - 2.5))
    val p   = VectorD (for i <- 0 until 100 yield nrm.pf (y(i)))
    new Plot (y, p)

    val snr = StdNormal ()
    val y2   = VectorD (for i <- 0 until 100 yield (i/20.0 - 2.5))
    val p2   = VectorD (for i <- 0 until 100 yield snr.pf (y(i)))
    new Plot (y2, p2)

end cLTTest

