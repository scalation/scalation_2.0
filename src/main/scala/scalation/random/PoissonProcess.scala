
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Casey Bowman
 *  @version 2.0
 *  @date    Sat Mar 15 15:28:44 EDT 2014
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Poisson Process Arrival Time Generator
 *
 *  Many of the algorithms used are from:
 *    Averill M. Law and W. David Kelton
 *    Simulation Modeling and Analysis, 2nd Edition
 *    McGraw-Hill, Inc., NY, 1991.
 */

package scalation
package random

import scala.math.{exp, floor, log, Pi, pow, round}

import scalation.mathstat.VectorD
import scalation.mathstat.Combinatorics.{fac, logfac}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeVariate` abstract class serves as a superclass for time-based
 *  random variates such Poisson Processes.
 *  @param stream  the random number stream
 */
abstract class TimeVariate (stream: Int)
         extends Variate (stream):

    protected val MAXFAC = 170       // maximum factorial for Double data type
              val mean  = -1.0       // mean changes with time, use meanF function instead

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean as a function of time.
     *  @param tt  the time point for computing the mean
     */
    def meanF (tt: Double): Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability function (pf):
     *  The probability density function (pdf) for continuous RV's or
     *  the probability mass function (pmf) for discrete RV's.
     *  @param z  the mass point whose probability is sought
     */
    def pf (z: Double): Double = pf (floor (z).toInt)

    def pf (z: Int): Double

    def pf (z: Int, tt: Double): Double

    def pf (z: Int, aa: Double, bb: Double): Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count then number of sample until the time exceeds tt.
     *  @param tt  the time point
     */
    def count (tt: Double): Int =
        var i = 0
        while gen < tt do i += 1
        i
    end count

    def count (a: Double, b: Double): Int =
        var i = 0
        var done = false
        while ! done do
            val c = gen
            if c >= a && c < b then i += 1
            if c >= b then done = true
        end while
        i
    end count

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the time-based process to the beginning.
     */
    def reset (): Unit

end TimeVariate


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PoissonProcess` class generates arrival times according to a Poisson Process.
 *  Given the current arrival time 't', generate the next arrival time.
 *  @see  http://en.wikipedia.org/wiki/Poisson_process
 *  @param lambda  the arrival rate (arrivals per unit time)
 *  @param stream  the random number stream
 */
case class PoissonProcess (lambda: Double, stream: Int = 0)
     extends TimeVariate (stream):

    private val flaw = flawf ("PoissonProcess")               // flaw function

    if lambda <= 0.0 then flaw ("init", "parameter lambda must be positive")

    private val e_rv  = Exponential (1.0 / lambda, stream)    // exponential rv generator
    private val _1_30 = 1.0 / 30.0                            // one thirtieth
    private val _1_6  = 1.0 / 6.0                             // one sixth
    private var t     = 0.0                                   // current time
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean number of arrivals for amount of time 'tt'.
     *  @param tt  a number of intervals
     */
    def meanF (tt: Double): Double = lambda * tt

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability P[ N(t) = k ] using a general factorial function
     *  implemented with the Gamma function and Ramanujan's Approximation.
     *  @see  http://en.wikipedia.org/wiki/Poisson_process
     *  @param k    the number of arrivals in the interval
     */
    def pf (k: Int): Double = (lambda * t)~^k * exp (-lambda * t) / fac (k)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability P[ (N(t + tau) - N(t)) = k] using a general
     *  factorial function implemented with the Gamma function and Ramanujan's
     *  Approximation.  Switches to pf_ln for k >= 170 to handle large k-values.
     *  @param k    the number of arrivals in the interval
     *  @param tau  the length of the interval
     */
    def pf (k: Int, tau: Double): Double = 
        if k < MAXFAC then (lambda * tau)~^k * exp (-lambda * tau) / fac (k)
        else pf_ln (k, tau)
    end pf

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability P[ (N(t + tau) - N(t)) = k] using the log of 
     *  Ramanujan's Approximation formula.
     *  @param k    the number of arrivals in the interval
     *  @param tau  the length of the interval
     */
    def pf_ln (k: Int, tau: Double): Double =
        val n = k.toDouble
        exp (-lambda * tau + n + n * log (lambda * tau) - n * log (n) - 0.5 * log (Pi)
            - _1_6 * log (((8.0 * n + 4.0) * n + 1.0) * n + _1_30))
    end pf_ln

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability P [ (N(b) - N(a)) = k ].
     *  @param k  the number of arrivals in the interval
     *  @param a  the left end of the interval
     *  @param b  the right end of the interval
     */
    def pf (k: Int, a: Double, b: Double): Double = pf (k, b - a)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate Poisson arrival times using and exponential random variable.
     */
    def gen: Double = { t += e_rv.gen; t }

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random number for the particular distribution.
     *  This version allows one paramater.
     *  @param z  the limit paramater
     */
    def gen1 (z: Double): Double = 
        val e_rv = Exponential (1.0 / z, stream)    // exponential rv generator
        t += e_rv.gen
        t
    end gen1

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the global time value to zero.
     */
    def reset (): Unit = { t = 0.0 }

end PoissonProcess


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** This class generates arrival times according to a `NHPoissonProcess`, an
 *  Non-Homogeneous Process Process (NHPP), where the arrival rate function
 *  lambda(t) is piece-wise constant.  Rates are constant over basic time
 *  intervals of length 'dt'.
 *  @see  http://en.wikipedia.org/wiki/Poisson_process#Non-homogeneous
 *  @param lambda  the vector of arrival rates
 *  @param dt      the length the basic time intervals
 *  @param stream  the random number stream
 */
case class NHPoissonProcess (lambda: VectorD, dt: Double = 1.0, stream: Int = 0)
     extends TimeVariate (stream):

    private val flaw = flawf ("NHPoissonProcess")             // flaw function

    if ! lambda.isNonnegative then flaw ("init", "parameter vector lambda must be nonnegative")

    private val lsum    = lambda.cumulate * dt                // cumulative lambda
    private val e_rv    = Exponential (1.0, stream)           // exponential rv generator with mean 1
    private var e       = 0.0                                 // cumulative exponential rv's
    private var tlast   = 0.0                                 // previous arrival time
    private var t       = 0.0                                 // current arrival time

    def meanF (tt: Double): Double = 
        val i1 = floor (tt / dt).toInt
        val i2 = i1 + 1
        if i2 >= lsum.dim then
            flaw ("meanF", "i2 is beyond the end of lsum vector")
            return -1.0
        val t1 = i1 * dt
        val t2 = t1 + dt
        val l1 = lsum(i1 - 1)
        val l2 = lsum(i2 - 1)
        (l2 - l1) * (tt - t1) / (t2 - t1) + l1                
    end meanF

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
    /** Compute the probability of k arrivals occurring in the time
     *  interval [0, t] where t is the current time.
     *  @param k  the number of arrivals in the time interval
     */
    def pf (k: Int): Double = pf (k, 0.0, t)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
    /** Compute the probability of k arrivals occurring in the time
     *  interval [0, tt].
     *  @param k   the number of arrivals in the time interval
     *  @param tt  the upper bound time value
     */
    def pf (k: Int, tt: Double): Double = pf (k, 0.0, tt)
    
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability P[ (N(b) - N(a)) = k ].
     *  @param k  the number of arrivals in interval [a,b]
     *  @param a  the left end of the interval
     *  @param b  the right end of the interval
     */
    def pf (k: Int, a: Double, b: Double): Double =
        if a < 0.0 || b > lambda.dim * dt then { flaw ("pf", "time bounds are outside total time interval"); -1.0 }
        else
            val aRes = a % dt
            val bRes = b % dt
            val iLow  = ((a - aRes) / dt).toInt
            val iHigh = ((b - bRes) / dt).toInt 
            var i1    = iLow
            val i2    = iHigh - 1
            if aRes > TOL then i1 = iLow + 1

            var sum   = 0.0
            if aRes > TOL then sum += lambda(iLow) * (dt - aRes)
            for i <- i1 to i2 do sum += lambda (i) * dt
            if bRes > TOL then sum += lambda (iHigh) * (bRes)
            if k < MAXFAC then exp (-sum) * sum~^k / fac (k) 
            else
                val s = -sum + k * log (sum) - logfac (k)
                exp (s)
        end if
    end pf

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute arrival times of the NHPP.
     */
    def genTime: Double =
        var useT = false
        tlast = t
        e += e_rv.gen                                           // add next exponential rv

        var (go, i) = (true, 0)
        cfor (go && i < lsum.dim, i += 1) {
            if e <= lsum(i) then                                // find where lsum(i-1) < e <= lsum(i)
                val lsum_i_1 = if i == 0 then 0.0 else lsum(i-1)
                val d = e - lsum_i_1                            // distance past lsum(i-1)
                t = dt * (i + d / (lsum(i) - lsum_i_1))         // calculate the new arrival time
                useT = true
                go = false
            end if
        } // cfor
        if useT then t
        else { flaw ("genTime", "cumulative e value larger than last lsum"); -1.0 }
    end genTime

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute inter-arrival times of the NHPP. 'tlast' is a global
     *  variable.
     */
    def gen: Double = genTime - tlast

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random number for the particular distribution.
     *  This version allows one paramater.
     *  @param z  the limit paramater
     */ 
    def gen1 (z: Double): Double =
        throw new UnsupportedOperationException ("gen1 not implemented for NHPoissonProcess")
    end gen1

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reset the NHPP by resetting 'e' to zero.
     */
    def reset (): Unit = { e = 0.0 }

end NHPoissonProcess


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `poissonProcessTest` main function is used to test both the `PoissonProcess` and
 *  `NHPoissonProcess` classes.
 */
@main def poissonProcessTest (): Unit =

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform a means test (average of generated rv's close to mean for distribution).
     *  @param rv  the random variate to test
     *  @param tt  the time point for the test
     */
    def meansTest (rv: TimeVariate, tt: Double): Unit =
        println ("\nTest the " + rv.getClass.getSimpleName () + " random variate generator at " + tt)

        var sum = 0.0
        val rep = 10000
        cfor (0, rep) { _ =>
            rv.reset ()
            sum += rv.count (tt)
        } // cfor
        println ("rv.mean = " + rv.meanF (tt) + " estimate = " + sum / rep.toDouble)
    end meansTest

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform a distribution test
     *  @param rv  the random variate to test
     *  @param a   the lower bound
     *  @param b   the upper bound
     */
    def distrTest (rv: TimeVariate, a: Double, b: Double): Unit =
        val tt   = b - a
        val name = rv.getClass.getSimpleName ()

        println ("\nTest the " + name + " random variate generator at " + tt)

        val rep  = 50000              // replications
        var j    = 0                  // interval number
//      var x    = 0.0                // x coordinate
        var o    = 0.0                // observed value: height of histogram
        var e    = 0.0                // expected value: pf (x)
        var chi2 = 0.0                // ChiSquare statistic
        var n    = 0                  // number of nonzero intervals
        val sum  = new Array [Int] (51)

        cfor (0, rep) { _ =>
            rv.reset ()
            if name == "PoissonProcess" then j = rv.count (tt)
            else j = rv.count (a, b)
            if 0 <= j && j <= 50 then sum (j) += 1
        } // cfor

        for i <- 0 until sum.length do
//          x = i / 10.0
            o = sum(i)
            rv.gen
            if name == "PoissonProcess" then e = round (rep * rv.pf (i, tt)).toDouble
            else e = round (rep * rv.pf (i, a, b)).toDouble
            if e >= 5 then
                chi2 += pow (o - e, 2) / e
                n += 1
            print ("\tsum (" + i + ") = " + o + " : " + e + " ")
            if i % 5 == 4 then println ()
        end for
        n -= 1
        if n < 2 then  n = 2
        if n > 49 then n = 49 
        println ("\nchi2 = " + chi2 + " : chi2(0.95, " + n + ") = " + Quantile.chiSquareInv (0.95, n))
    end distrTest

    val lambda = VectorD (2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          4, 4, 4, 4, 4, 4, 4, 4, 4, 4)

    val pp   = PoissonProcess (lambda.sum / lambda.dim.toDouble)
    val nhpp = NHPoissonProcess (lambda, 1.0, 1)

    meansTest (pp, 12.0)
    distrTest (pp, 0.0, 12.0)
//  distrTest (nhpp, args(0).toDouble, args(1).toDouble)

//  println ("nhpp.pf (17, " + args(0) + ", " + args(1) + ") = " + nhpp.pf (17, args (0).toDouble, args(1).toDouble))

    var v = 0.0
    while v >= 0.0 do
        v = nhpp.genTime
        println (v)
    end while  

end poissonProcessTest

