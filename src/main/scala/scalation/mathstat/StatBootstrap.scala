
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Nov  2 13:27:50 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Bootstrap Resampling (with Replacement)
 *
 *  @see     en.wikipedia.org/wiki/Bootstrapping_(statistics)
 */

package scalation
package mathstat

import scala.math.sqrt

import scalation.random.{RandomVecD, RandomVecI, RandomVecIR}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `StatBootstrap` class is used to create bootstrap samples and compute
 *  statistics based on these pseudo-samples.  Given a sample y from a population
 *  (typically unknown), create several pseudo-samples (bootstrap samples).
 *  This allows confidence intervals to be created with requiring distribution
 *  assumptions, such as the data are Normally distributed.
 *  @param y         the original sample of data
 *  @param unbiased  whether the estimators are restricted to be unbiased
 *  @param stream    the random number stream to use
 */
class StatBootstrap (y: VectorD, unbiased: Boolean = true, stream: Int = 0)
      extends Statistic ("bootstrap", unbiased):

    private val len   = y.dim                                     // size of the vector
    private val r_idx = RandomVecIR (len, len-1, 0, stream)       // random index generator

    private var samples: MatrixD = null                           // will be a n by len matrix

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Make/generate n bootstrap samples (resampling with replacement).
     *  @param n  the number of sample to make
     */
    def makeSamples (n: Int): Unit =
        samples = MatrixD (for _ <- 0 until n yield y(r_idx.igen))
    end makeSamples

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the bootstrap sample mean.
     */
    inline def bmean: Double = samples.mmean

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the bootstrap sample variance.
     */
    def bvariance: Double = samples.mvariance

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the bootstrap sample standard deviation.
     */
    def bstdev: Double = samples.mstdev

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the bootstrap confidence interval (lo, hi) for the given confidence level
     *  using the bootstrap percentile method.
     *  @see stat.rutgers.edu/home/mxie/RCPapers/bootstrap.pdf
     *  @param p_  the confidence level
     */
    def binterval (p_ : Double = .95): (Double, Double) =
        val α_2 = (1.0 - p_) / 2.0
        val (lo, hi) = (α_2, 1.0 - α_2)
        val means = samples.mean
        (means.quantile (lo), means.quantile (hi))
    end binterval

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the bootstrap confidence interval half-width (ihw) for the given confidence
     *  level using the t-distribution.  Assumes the data follows a Normal distribution.
     *  @param p_  the confidence level
     */
    def binterval_ (p_ : Double = .95): Double =
        val df = samples.dim - 1                                  // degrees of freedom
        t_sigma (bstdev, df, p_) / sqrt (samples.dim)
    end binterval_

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the summary bootstrap statistics as a row/Array.
     */
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a row of bootstrap statistical results as a string.
     */
    def toString2: String =
        "| %11s | %5s | %10.3f | %10.3f | %10.3f | %10.3f | %10.3f |".format (
        name, num, min, max, bmean, bstdev, binterval_ ())
    end toString2

end StatBootstrap


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `statBootstrapTest` main function is used to test the `StatBootstrap` class.
 *  Test population, sample, and bootstrap statistics.
 *  > runMain scalation.mathstat.statBootstrapTest
 */
@main def statBootstrapTest (): Unit =

    val (popSize, maxValue) = (500, 100)
    val sampSize = 100

    banner ("Generate a Population Vector")
    val pgen = RandomVecD (popSize, maxValue, 0) 
    val pop  = pgen.gen
    println (s"population pop = $pop")

    banner ("Take a Sample from the Population Vector")
    val sgen = RandomVecI (sampSize, maxValue, 0)
    val samp = pop (sgen.igen)
    println (s"sample samp = $samp")

    banner ("Compute Population Statistics")
    val stat0 = new Statistic ("pop", false)
    stat0.tallyVec (pop)
    println (Statistic.labels)
    println (stat0)

    banner ("Compute Sample Statistics")
    val stat1 = new StatBootstrap (samp)
    stat1.tallyVec (samp)
    println (Statistic.labels)
    println (stat1)

    banner ("Compute Bootstrap Statistics")
    stat1.makeSamples (200)
    println (Statistic.labels)
    println (stat1.toString2)
    val (mu, ihw) = (stat1.bmean, stat1.binterval_ ())
    val lo_hi     = (mu - ihw, mu + ihw)
    println (s"binterval_ = $lo_hi")
    println (s"binterval  = ${stat1.binterval ()}")

end statBootstrapTest

