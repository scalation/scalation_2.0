
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael E. Cotterell
 *  @version 2.0
 *  @date    Fri Jul 24 14:35:58 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Cumulative Distribution Function (CDF)
 */

package scalation
package random

import scala.collection.mutable.ArrayBuffer
import scala.math.{abs, atan, exp, floor, Pi, pow}

import scalation.mathstat.{Plot, VectorD}
import scalation.mathstat.Combinatorics.{rBetaC, rBetaF}

/** Type definition for parameters to a distribution. `Vector` is used instead
 *  of `Array` since they are covariant, while Scala arrays are not.
 */
type Parameters = Vector [Double]

/** The function type for distribution functions, including
 *  (1) Cumulative Distribution Function (CDF)
 *  (2) inverse Cumulative Distribution Function (iCDF)
 *  The arguments are `Double` for coordinate 'x' or probability 'p' and a
 *  `Vector` of parameters, e.g., degrees of freedom.
 */
type Distribution = (Double, Parameters) => Double

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CDF` object contains methods for computing 'F(x)', the Cumulative
 *  Distribution Functions 'CDF's for popular distributions:
 *  `Uniform`
 *  `Exponential`
 *  `Weibel`
 *  `Empirical`
 *  `StandardNormal`
 *  `StudentT`
 *  `ChiSquare`
 *  `Fisher`
 *  For a given CDF 'F' with argument 'x', compute 'p = F(x)'.
 */
object CDF:

    private val flaw = flawf ("CDF")                           // flaw function

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function 'CDF' 'F(x)' for the
     *  Uniform distribution.
     *  @param x  the x coordinate, argument to F(x)
     *  @param b  the upper end-point of the uniform distribution 
     *  @param a  the lower end-point of the uniform distribution 
     */
    def uniformCDF (x: Double, a: Double, b: Double): Double =
        if a >= b then flaw ("uniformCDF", "requires parameter b > parameter a")
        if x <= a then     0.0
        else if x < b then (x - a) / (b - a)
        else 1.0
    end uniformCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function 'CDF' 'F(x)' for the
     *  Uniform distribution.
     *  @param x   the x coordinate, argument to F(x)
     *  @param pr  parameters giving the end-points of the uniform distribution 
     */
    def uniformCDF (x: Double, pr: Parameters = null): Double =
        val (a, b) = if pr == null then (0.0, 1.0) else (pr(0), pr(1))
        uniformCDF (x, a, b)
    end uniformCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function 'CDF' 'F(x)' for the
     *  `Exponential` distribution.
     *  @param x  the x coordinate, argument to F(x)
     *  @param λ  the rate parameter
     */
    def exponentialCDF (x: Double, λ: Double): Double =
        if λ <= 0.0 then flaw ("exponentialCDF", "requires parameter lambda λ > 0")
        if x > 0 then 1.0 - exp (-λ * x) else 0.0
    end exponentialCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function 'CDF' 'F(x)' for the
     *  `Exponential` distribution.
     *  @param x     the x coordinate, argument to F(x)
     *  @param parm  parameter giving the rate 
     */
    def exponentialCDF (x: Double, pr: Parameters = null): Double =
        val λ = if pr == null then 1.0 else pr(0)
        exponentialCDF (x, λ)
    end exponentialCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function 'CDF' 'F(x)' for the
     *  `Weibull` distribution.
     *  @param x  the x coordinate, argument to F(x)
     *  @param α  the shape parameter
     *  @param β  the scale parameter
     */
    def weibullCDF (x: Double, α: Double, β: Double): Double =
        if α <= 0.0 || β <= 0.0 then flaw ("weibullCDF", "parameters α and β must be positive")
        if x > 0.0 then 1.0 - exp (-(x/β)~^α) else 0.0
    end weibullCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function 'CDF' 'F(x)' for the
     *  `Weibull` distribution.
     *  @param x     the x coordinate, argument to F(x)
     *  @param parm  parameters giving the shape and scale
     */
    def weibullCDF (x: Double, pr: Parameters = null): Double =
        val (α, β) = if pr == null then (2.0, 2.0) else (pr(0), pr(1))
        weibullCDF (x, α, β)
    end weibullCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    /** Build an empirical CDF from input data vector 'x'.
     *  Ex: x = (2, 1, 2, 3, 2) -> cdf = ((1, .2), (2, .8), (3, 1.))
     *  @param x  the input data vector
     */
    def buildEmpiricalCDF (x: VectorD): (VectorD, VectorD) =
        val zbuf = ArrayBuffer [Double] ()
        val cbuf = ArrayBuffer [Double] ()
        val z    = x.sorted
        for i <- z.indices do
            if i == 0 || z(i) != z(i-1) then
                zbuf += z(i)
                cbuf += 1.0
            else
                cbuf(cbuf.size - 1) += 1.0
        end for
        (new VectorD (zbuf.size, zbuf.toArray),
         new VectorD (cbuf.size, cbuf.toArray).cumulate / x.dim.toDouble)
    end buildEmpiricalCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function 'CDF' 'F(x)' for the
     *  Empirical distribution 'eCDF'.
     *  @param x     the x coordinate, argument to F(x)
     *  @param eCDF  the Empirical CDF
     */
    def empiricalCDF (x: Double, eCDF: (VectorD, VectorD)): Double =
        if x < eCDF._1(0) then 0.0
        else if x < eCDF._1(eCDF._1.dim-1) then eCDF._2 (eCDF._1.indexWhere (_ > x) - 1)
        else 1.0
    end empiricalCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    /*  Code below recoded in Scala from Apache Java code.
     *  @see mail-archives.apache.org/mod_mbox/commons-dev/200401.mbox/%3C20040126030431.92035.qmail@minotaur.apache.org%3E
     *  License for Apache Java code given below:
     *-------------------------------------------------------------------------
     * The Apache Software License, Version 1.1
     *
     * Copyright (c) 2004 The Apache Software Foundation.  All rights
     * reserved.
     *
     * Redistribution and use in source and binary forms, with or without
     * modification, are permitted provided that the following conditions
     * are met:
     *
     * 1. Redistributions of source code must retain the above copyright
     *    notice, this list of conditions and the following disclaimer.
     *
     * 2. Redistributions in binary form must reproduce the above copyright
     *    notice, this list of conditions and the following disclaimer in
     *    the documentation and/or other materials provided with the
     *    distribution.
     *
     * 3. The end-user documentation included with the redistribution, if
     *    any, must include the following acknowledgement:
     *       "This product includes software developed by the
     *        Apache Software Foundation (http://www.apache.org/)."
     *    Alternately, this acknowledgement may appear in the software itself,
     *    if and wherever such third-party acknowledgements normally appear.
     *
     * 4. The names "The Jakarta Project", "Commons", and "Apache Software
     *    Foundation" must not be used to endorse or promote products derived
     *    from this software without prior written permission. For written
     *    permission, please contact apache@apache.org.
     *
     * 5. Products derived from this software may not be called "Apache"
     *    nor may "Apache" appear in their name without prior written
     *    permission of the Apache Software Foundation.
     *
     * THIS SOFTWARE IS PROVIDED ``AS IS'' AND ANY EXPRESSED OR IMPLIED
     * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
     * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
     * DISCLAIMED.  IN NO EVENT SHALL THE APACHE SOFTWARE FOUNDATION OR
     * ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
     * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
     * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
     * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
     * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
     * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
     * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
     * SUCH DAMAGE.
     * ====================================================================
     *
     * This software consists of voluntary contributions made by many
     * individuals on behalf of the Apache Software Foundation.  For more
     * information on the Apache Software Foundation, please see
     * <http://www.apache.org/>.
     *-------------------------------------------------------------------------
     * Implements Cody algorithm to calculate CDF (cumulative distribution function)
     * for Normal (Gauss) distribution.<p> 
     * Provided implementation is adapted from 
     * <a href="http://www.r-project.org/">R statistical package</a> function
     * <code>pnorm(...)</code>.<p>
     * References:
     * <ul> 
     * <li>Cody's algorithm: Cody, W. D. (1993).<br>
     *    <a href="http://www.acm.org/toms/cgi-bin/TOMSbibget?Cody:1993:ASE">
     *  ALGORITHM 715: SPECFUN - A Portable FORTRAN Package of
     *    Special Function Routines and Test Drivers".
     *    ACM Transactions on Mathematical Software. 19, 22-32</a>.</li>
     * <li>FORTRAN code is available at 
     * <a href="http://www.netlib.org/toms/715">http://www.netlib.org/toms/715</a>
     * </li>
     * </ul>
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    /** Compute the Cumulative Distribution Function 'CDF' 'F(x)' for the Standard
     *  Normal distribution using the Hart function.  Recoded in Scala from Java code.
     *  Apache license given above.
     *  @see mail-archives.apache.org/mod_mbox/commons-dev/200401.mbox/%3C20040126030431.92035.qmail@minotaur.apache.org%3E
     *  @param x   the x coordinate, argument to F(x)
     */
    def normalCDF (x: Double): Double =
        val a = Array (2.2352520354606839287,   1.6102823106855587881e2,
                       1.0676894854603709582e3, 1.8154981253343561249e4,
                       6.5682337918207449113e-2)

        val b = Array (4.7202581904688241870e1, 9.7609855173777669322e2,
                       1.0260932208618978205e4, 4.5507789335026729956e4)

        val c = Array (3.9894151208813466764e-1, 8.8831497943883759412,
                       9.3506656132177855979e1,  5.9727027639480026226e2,
                       2.4945375852903726711e3,  6.8481904505362823326e3,
                       1.1602651437647350124e4,  9.8427148383839780218e3,
                       1.0765576773720192317e-8)

        val d = Array (2.2266688044328115691e1, 2.3538790178262499861e2,
                       1.5193775994075548050e3, 6.4855582982667607550e3,
                       1.8615571640885098091e4, 3.4900952721145977266e4,
                       3.8912003286093271411e4, 1.9685429676859990727e4)

        val p = Array (2.1589853405795699e-1, 1.274011611602473639e-1,
                       2.2235277870649807e-2, 1.421619193227893466e-3,
                       2.9112874951168792e-5, 2.307344176494017303e-2)

        val q = Array (1.28426009614491121,    4.68238212480865118e-1,
                       6.59881378689285515e-2, 3.78239633202758244e-3,
                       7.29751555083966205e-5)

        val sixteen      = 16.0
        val M_1_SQRT_2PI = 0.398942280401432677939946059934        // 1/sqrt(2pi)
        val M_SQRT_32    = 5.656854249492380195206754896838        // sqrt(32)    
  
//      val eps = 0.5 * pow (2, 1.0-60.0)
        val eps = EPSILON                                          // machine epsilon
        val min = -MAX_VALUE

        var (ccum, cum) = (0.0, 0.0)
        var (del, temp) = (0.0, 0.0)
        var (xden, xnum, xsq) = (0.0, 0.0, 0.0)

        val y = abs (x)

        // 1st case: |x| <= qnorm(3/4)
        if y <= 0.67448975 then
            if y > eps then xsq = x * x
            xnum = a(4) * xsq
            xden = xsq
            for i <- 0 until 3 do
                xnum = (xnum + a(i)) * xsq
                xden = (xden + b(i)) * xsq
            end for
            temp = x * (xnum + a(3)) / (xden + b(3))
            cum  = 0.5 + temp
            ccum = 0.5 - temp

        // 2nd case: qnorm(3/4)1 <= |x| <= sqrt(32)
        else if y <= M_SQRT_32 then
            xnum = c(8) * y
            xden = y
            for i <- 0 until 7 do
                xnum = (xnum + c(i)) * y
                xden = (xden + d(i)) * y
            end for
            temp = (xnum + c(7)) / (xden + d(7))
            xsq = floor (y*sixteen) / sixteen
            del = (y - xsq) * (y + xsq)
            cum = exp (-(xsq*xsq*0.5)) * exp(-(del*0.5)) * temp
            ccum = 1.0 - cum
            if x > 0.0 then { temp = cum; cum = ccum; ccum = temp }

        // 3rd case: -37.5193 < x && x < 8.2924 || -8.2924  < x && x < 37.5193
        else if -37.5193 < x && x < 8.2924 || -8.2924 < x && x < 37.5193 then
            xsq = 1.0 / (x * x)
            xnum = p(5) * xsq
            xden = xsq
            for i <- 0 until 4 do
                xnum = (xnum + p(i)) * xsq
                xden = (xden + q(i)) * xsq
            end for
            temp = xsq * (xnum + p(4)) / (xden + q(4))
            temp = (M_1_SQRT_2PI - temp) / y
            xsq = floor (x*sixteen) / sixteen
            del = (x - xsq) * (x + xsq)
            cum = exp (-(xsq*xsq*0.5)) * exp (-(del*0.5)) * temp
            ccum = 1.0 - cum
            if x > 0.0 then { temp = cum; cum  = ccum; ccum = cum }

        // 4th case: high x values
        else
            if x > 0 then { cum = 1.0; ccum = 0.0 }
            else          { cum = 0.0; ccum = 1.0 }

        if cum < min then  cum  = 0.0
        if ccum < min then ccum = 0.0

        cum
    end normalCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    /** Compute the Cumulative Distribution Function (CDF) 'F(x)' for the Standard
     *  Normal distribution using the Hart function.  Recoded in Scala from C code
     *  @see stackoverflow.com/questions/2328258/cumulative-normal-distribution-function-in-c-c
     *  which was recoded from VB code.
     *  @see www.codeplanet.eu/files/download/accuratecumnorm.pdf
     *  @param x   the x coordinate, argument to F(x)
     */  
    def _normalCDF (x: Double): Double =
        val z = abs (x)
        if z > 37.0 then return 0.0

        val RT2PI = 2.506628274631           // sqrt (4.0* acos (0.0))
        val SPLIT = 7.07106781186547

        val N0 = 220.206867912376
        val N1 = 221.213596169931
        val N2 = 112.079291497871
        val N3 = 33.912866078383
        val N4 = 6.37396220353165
        val N5 = 0.700383064443688
        val N6 = 3.52624965998911e-02
        val M0 = 440.413735824752

        val M1 = 793.826512519948
        val M2 = 637.333633378831
        val M3 = 296.564248779674
        val M4 = 86.7807322029461
        val M5 = 16.064177579207
        val M6 = 1.75566716318264
        val M7 = 8.83883476483184e-02

        val e = exp (-z * z / 2.0)

        val c = if z < SPLIT then
            val n = (((((N6*z + N5)*z + N4)*z + N3)*z + N2)*z + N1)*z + N0
            val d = ((((((M7*z + M6)*z + M5)*z + M4)*z + M3)*z + M2)*z + M1)*z + M0
            e * n / d
        else
            val f = z + 1.0 / (z + 2.0/(z + 3.0/(z + 4.0/(z + 13.0/20.0))))
            e / (RT2PI * f)

        if x <= 0.0 then c else 1.0 - c
    end _normalCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    /** Compute the Cumulative Distribution Function (CDF) for the Standard Normal
     *  distribution using a composite fifth-order Gauss-Legendre quadrature.
     *  @author John D. Cook (Adapted to Scala by Michael E. Cotterell)
     *  @see www.johndcook.com/blog/cpp_phi
     *  @see [AS 1965] Abramowitz & Stegun. "Handbook of Mathematical Functions" (June) (1965) 
     *  @param x   the x coordinate, argument to F(x)
     */  
//  def normalCDF (x: Double): Double =
//      val a1 =  0.254829592
//      val a2 = -0.284496736
//      val a3 =  1.421413741
//      val a4 = -1.453152027
//      val a5 =  1.061405429
//      val p  =  0.3275911
//
//      val sign = if x < 0 then -1 else 1
//      val y = abs (x) / math.sqrt (2.0)
//
//      // A&S formula 7.1.26
//      val t = 1.0 / (1.0 + p * y)
//      val z = 1.0 - (((((a5*t + a4) * t) + a3) * t + a2) * t + a1) * t * exp (-y*y)
//
//      0.5 * (1.0 + sign * z)
//  end normalCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for the Normal
     *  distribution.
     *  @param x   the x coordinate, argument to F(x)
     *  @param pr  parameters for the mean and standard deviation
     */
    def normalCDF (x: Double, pr: Parameters = null): Double =
        val (μ, σ) = if pr == null then (0.0, 1.0) else (pr(0), pr(1))
        if σ <= 0.0 then flaw ("normalCDF", "requires parameter σ > 0")
        normalCDF ((x - μ) / σ)
    end normalCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for the Normal
     *  distribution.
     *  @param x   the x coordinate, argument to F(x)
     *  @param pr  parameters for the mean and standard deviation
     */
    def _normalCDF (x: Double, pr: Parameters = null): Double =
        val (μ, σ) = if pr == null then (0.0, 1.0) else (pr(0), pr(1))
        if σ <= 0.0 then flaw ("_normalCDF", "requires parameter σ > 0")
        _normalCDF ((x - μ) / σ)
    end _normalCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for "Student's t"
     *  distribution.
     *  @author Michael Cotterell
     *  @see [JKB 1995] Johnson, Kotz & Balakrishnan "Continuous Univariate 
     *       Distributions" (Volume 2) (2nd Edition) (Chapter 28) (1995) 
     *  @param x   the x coordinate, argument to F(x)
     *  @param df  the degrees of freedom (must be > 0.0)
     */    
    def studentTCDF (x: Double, df: Double): Double =
        if df <= 0.0 then
            flaw ("studentTCDF", "parameter df must be strictly positive")
            return -0.0

        if df =~ 1.0 then                                  // Cauchy CDF
            0.5 + (1.0/Pi) * atan (x)
        else if df =~ 2.0 then                            // Explicit Formula
            0.5 + (x/2.0) * pow (2.0 + x*x, -0.5)
        else if df < 2.0*x*x then                         // [JKB 1995]
            val z = 0.5 * rBetaF (df / (df + x*x), 0.5*df, 0.5)
            if x > 0 then 1.0 - z else z
        else if df < 30.0 then                            // [JKB 1995]
            val z = 0.5 * rBetaC (x*x / (df + x*x), 0.5, 0.5*df)
            if x > 0 then 1.0 - z else z
        else                                               // Ordinary Normal Approximation (ONA)
            normalCDF (x)                  
    end studentTCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for "Student's t"
     *  distribution.
     *  @param x   the x coordinate, argument to F(x)
     *  @param pr  parameter for the degrees of freedom
     */
    def studentTCDF (x: Double, pr: Parameters = null): Double =
        val df = if pr == null then 9 else pr(0).toInt
        studentTCDF (x, df)
    end studentTCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for "Noncentral t"
     *  distribution.
     *  @see https://en.wikipedia.org/wiki/Noncentral_t-distribution
     *  @param x   the x coordinate, argument to F(x)
     *  @param mu  the noncentrality parameter (or mean)
     *  @param df  the degrees of freedom (must be > 0.0)
     */    
    def noncentralTCDF (x: Double, mu: Double, df: Double): Double =
        if df <= 0.0 then
            flaw ("noncentralTCDF", "parameter df must be strictly positive")
            return -0.0

        throw new UnsupportedOperationException ("noncentralTCDF in CDF not implemented yet")  // FIX
    end noncentralTCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for the ChiSquare
     *  distribution by numerically integrating the ChiSquare probability
     *  density function (pdf).  See Variate.scala.
     *  @param x   the x coordinate, argument to F(x)
     *  @param df  the degrees of freedom
     */
    def chiSquareCDF (x: Double, df: Int): Double =
        if x < 0.0 then
            flaw ("chiSquareCDF", "coordinate x should be nonnegative")
            return 0.0
        if df <= 0 then
            flaw ("chiSquareCDF", "parameter df must be strictly positive")
            return -0.0

        val chi  = ChiSquare (df)        // ChiSquare distribution
        val step = 0.0001
        var sum  = 0.0
        var xx   = EPSILON               // small number (machine epsilon)
        var y1   = 0.0
        var y2   = chi.pf (xx)           // pdf for ChiSquare distribution
        while xx <= x && sum < 1.0 do
            y1  = y2
            xx  += step
            y2  = chi.pf (xx)
            sum += step * (y1 + y2) / 2.0
        end while
        sum
    end chiSquareCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for the ChiSquare
     *  distribution.
     *  @param x   the x coordinate, argument to F(x)
     *  @param df  parameter for the degrees of freedom
     */
    def chiSquareCDF (x: Double, pr: Parameters = null): Double =
        val df = if pr == null then 9 else pr(0).toInt
        chiSquareCDF (x, df)
    end chiSquareCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for the Fisher (F)
     *  distribution using beta functions.
     *  @param x    the x coordinate, argument to F(x)
     *  @param df1  the degrees of freedom 1 (numerator)
     *  @param df2  the degrees of freedom 2 (denominator)
     */
    def fisherCDF (x: Double, df1: Int, df2: Int): Double =
        if x < 0.0 then
            flaw ("fisherCDF", s"F(x) requires coordinate x = $x to be nonnegative")
            return 0.0
        if df1 <= 0 || df2 <= 0 then
            flaw ("fisherCDF", "parameters df1 and df2 must be strictly positive")
            return -0.0
       
        val ff = rBetaF (df1 * x / ((df1 * x) + df2), df1 / 2.0, df2 / 2.0)
        if ff > 1.0 then 1.0 else ff                                   // handle possible round-off errors
    end fisherCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for the Fisher (F)
     *  distribution using beta functions.
     *  @param x   the x coordinate, argument to F(x)
     *  @param df  the pair of degrees of freedom ('df1', 'df2')
     */
    def fisherCDF (x: Double, df: (Int, Int)): Double =
        fisherCDF (x, df._1, df._2)
    end fisherCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Cumulative Distribution Function (CDF) for the Fisher (F)
     *  distribution using beta functions.
     *  @param x   the x coordinate, argument to F(x)
     *  @param df  parameters for degrees of freedom numerator and denominator
     */
    def fisherCDF (x: Double, pr: Parameters = null): Double =
        val (df1, df2) = if pr == null then (2, 9) else (pr(0).toInt, pr(1).toInt)
        fisherCDF (x, df1, df2)
    end fisherCDF

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the given CDF 'ff' over a range of 'x' values for the given parameters,
     *  e.g., degrees of freedom 'df'.
     *  @param ff     the CDF F(.)
     *  @param name   the name of CDF F(.)
     *  @param x_min  the minimum of value of x to test
     *  @param x_max  the maximum of value of x to test
     *  @param pr     parameters for the distribution, e.g., the degrees of freedom
     */
    def test_df (ff: Distribution, name: String, x_min: Double, x_max: Double, pr: Parameters = null): Unit =
        println ("-----------------------------------------------------------")
        println (s"Test the $name function")
        val n  = 32
        val x  = new VectorD (n + 1)
        val p  = new VectorD (n + 1)
        val dx = (x_max - x_min) / n.toDouble
        for i <- 0 to n do
            x(i) = x_min + i * dx
            p(i) = ff(x(i), pr)
            println (s"$name (${x(i)}, $pr)\t = ${p(i)}")
        end for
        new Plot (x, p, null, name + ": p = F(x)")
    end test_df

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the difference CDF 'ff1' and 'ff2' over a range of 'x' values for
     *  the given parameters. e.g., degrees of freedom 'df'.
     *  @param ff     the CDF F(.)
     *  @param name   the name of CDF F(.)
     *  @param x_min  the minimum of value of x to test
     *  @param x_max  the maximum of value of x to test
     *  @param pr     parameters for the distribution, e.g., the degrees of freedom
     */
    def test_diff (ff1: Distribution, ff2: Distribution,
                   name: String, x_min: Double, x_max: Double, pr: Parameters = null): Unit =
        println ("-----------------------------------------------------------")
        println (s"Test the $name function")
        val n = 128
        val x = new VectorD (n + 1)
        val p = new VectorD (n + 1)
        val q = new VectorD (n + 1)
        val dx = (x_max - x_min) / n.toDouble
        for i <- 0 to n do
            x(i) = x_min + i * dx
            p(i) = ff1(x(i), pr)
            q(i) = ff2(x(i), pr)
            println ("%s: %7.3f, %26.18g, %26.18g, %26.18g"format (name, x(i), p(i), q(i), (p(i) - q(i))))
//          println (s"$name: ${x(i)}, \t ${p(i)}, \t ${q(i)}, \t${abs (p(i) - q(i))}")
        end for
        new Plot (x, p, q, name + ": p = F(x)")
    end test_diff

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the given CDF with name 'cdf'
     *  @param cdf  the name of the CDF to test
     */
    def test (cdf: String): Unit =
        cdf match
        case "uniformCDF" =>
            test_df (uniformCDF, cdf, -0.5, 1.5)
        case "exponentialCDF" =>
            test_df (exponentialCDF, cdf, 0.0, 4.0)
        case "empiricalCDF" =>
            println (s"distribution $cdf currently is not yet implemented")
        case "weibullCDF" =>
            test_df (weibullCDF, cdf, 0.0, 4.0)
        case "normalCDF" =>
            test_df (normalCDF, cdf, -4.0, 4.0)
        case "_normalCDF" =>
            test_df (_normalCDF, cdf, -4.0, 4.0)
        case "studentTCDF" =>
            for df <- 1 to 30 do test_df (studentTCDF, cdf, -4.0, 4.0, Vector (df))
        case "chiSquareCDF" =>
            for df <- 1 to 30 do test_df (chiSquareCDF, cdf, 0.0, 2.0*df, Vector (df))
        case "fisherCDF" =>
            for df1 <- 1 to 10; df2 <- 1 to 10 do
                test_df (fisherCDF, cdf, 0.0, 4.0*(df1/df2.toDouble), Vector (df1, df2))
        case _ =>
            println (s"distribution $cdf currently is not supported")
        end match
    end test

end CDF

import CDF._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_Uniform` main function tests the 'CDF.uniformCDF' method.
 *  > runMain scalation.random.cDFTest_Uniform
 */
@main def cDFTest_Uniform (): Unit = test ("uniformCDF")

 
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_Exponential` main function tests the 'CDF.exponentialCDF' method.
 *  > runMain scalation.random.cDFTest_Exponential
 */
@main def cDFTest_Exponential (): Unit = test ("exponentialCDF")


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_Weibull` main function tests the 'CDF.weibullCDF' method.
 *  > runMain scalation.random.cDFTest_Weibull
 */
@main def cDFTest_Weibull (): Unit = test ("weibullCDF")


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_Empirical` main function tests the 'CDF.buildEmpiricalCDF' method.
 *  > runMain scalation.random.cDFTest_Empirical
 */
@main def cDFTest_Empirical (): Unit =

    val eCDF = buildEmpiricalCDF (VectorD (2.0, 1.0, 2.0, 3.0, 2.0))
    println (s"F(x)   = $eCDF")
    for i <- 1 to 7 do println (s"empiricalCDF (${i/2.0}, eCDF) = ${empiricalCDF (i/2.0, eCDF)}")

end cDFTest_Empirical
     
 
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_Normal` main function tests the 'CDF.normalCDF' method.
 *  > runMain scalation.random.cDFTest_Normal
 */
@main def cDFTest_Normal (): Unit = test ("normalCDF")

 
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_Normal` main function tests difference between different
 *  implementations of the 'normalCDF' method in the `CDF` object.
 *  > runMain scalation.random.cDFTest_Normal_Diff
 */
@main def cDFTest_Normal_Diff (): Unit = test_diff (normalCDF, _normalCDF, "normalCDF", -8.0, 8.0)

 
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_StudentT` main function tests the 'CDF.studentTCDF' method.
 *  > runMain scalation.random.cDFTest_StudentT
 */
@main def cDFTest_StudentT (): Unit = test ("studentTCDF")

 
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_ChiSquare` main function tests the 'CDF.chiSquareCDF' method.
 *  > runMain scalation.random.cDFTest_ChiSquare
 */
@main def cDFTest_ChiSquare (): Unit = test ("chiSquareCDF")

 
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_Fisher` main function tests the 'CDF.fisherCDF' method.
 *  > runMain scalation.random.cDFTest_Fisher
 */
@main def cDFTest_Fisher (): Unit = test ("fisherCDF")


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cDFTest_Fisher2` main function tests the 'CDF.fisherCDF' method.
 *  > runMain scalation.random.cDFTest_Fisher2
 */
@main def cDFTest_Fisher2 (): Unit =

    val fStat = 670.36
    val df    = (3, 31)
    println (s"fisherCDF ($fStat, ${df._1}, ${df._2}) = ${fisherCDF (fStat, df._1, df._2)}")
    println (s"fisherCDF (${1.0/fStat}, ${df._2}, ${df._1}) = ${fisherCDF (fStat, df._2, df._1)}")

end cDFTest_Fisher2

