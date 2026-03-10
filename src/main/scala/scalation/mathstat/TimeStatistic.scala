
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Aug 26 18:41:26 EDT 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Time-Persistent Statistics Collection
 */

package scalation
package mathstat

import scala.collection.mutable.{ArrayBuffer => VEC}
//import scala.collection.mutable.{ListBuffer => VEC}
import scala.math.abs

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeStatistic` class is used to collect values and compute time-persistent
 *  statistics on them (e.g., Number in Queue).
 *  @see staff.unak.is/andy/Year%203%20Simulation/Laboratories/v4manual/internal.htm
 *  @param name        the name for this statistic (e.g., 'numberInQueue' or 'tellerQ')
 *  @param _lastTime   the time of last observation
 *  @param _startTime  the time observation began
 */
class TimeStatistic (override val name: String = "p-stat",
                      private var _lastTime:  Double = 0.0,
                      private val _startTime: Double = 0.0)
      extends Statistic (name):

    private val flaw = flawf ("TimeStatistic")                     // flaw function

    def n_     = n
    def sum_   = sum
    def sumAb_ = sumAb
    def sumSq_ = sumSq
    def minX_  = minX
    def maxX_  = maxX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the last time statistics were recorded by this collector.
     */
    def lastTime: Double = _lastTime

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the time observation began.
     */
    def startTime: Double = _startTime

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Disable the tally method (it is for sample statistics, not time-persistent
     *  statistics.
     *  @param x  the value to tally
     */
    override def tally (x: Double): Unit =
        flaw ("tally", "this method must not be called from TimeStatistic")
    end tally

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Accumulate the next value weighted by its time duration and update accumulators.
     *  @param x  the value to accumulate (e.g., system occupancy)
     *  @param t  the time of the observation (e.g., event times)
     */
    def accum (x: Double, t: Double): Unit =
        val duration = t - _lastTime        // the duration of value x
        _lastTime    = t

        n     += 1
        sum   += x * duration
        sumAb += abs (x) * duration
        sumSq += x * x * duration
        if x < minX then minX = x
        if x > maxX then maxX = x
    end accum

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the time-weighted mean.
     */
    override def mean: Double =
        val totalTime = _lastTime - _startTime
        if totalTime <= 0.0 then 0.0 else sum / totalTime
    end mean

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute/estimate the time-weighted variance.
     */
    override def variance: Double =
        val totalTime = _lastTime - _startTime
        if totalTime <= 0.0 then 0.0 else (sumSq - sum*sum/totalTime) / totalTime
    end variance

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the values of this collector's accumulators.
     */
    override def show: String = "Time" + super.show + ", " + _lastTime + ", " + _startTime

end TimeStatistic


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TimeStatistic` companion object provides additional functions.
 */
object TimeStatistic:

    private val flaw = flawf ("TimeStatistic")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a statistical object and set its accumulators and extreme values.
     *  @param n           the number of samples
     *  @param sum         sum of the sample values
     *  @param sumAb       sum of the sample absolute values
     *  @param sumSq       sum of the sample values squared
     *  @param minX        the minimum sample value
     *  @param maxX        the maximum sample value
     *  @param name        the name for this statistic (e.g., 'waitingTime')
     *  @param _lastTime   the time of latest observation
     *  @param _startTime  the time observation began
     */
    def apply (n: Int, sum: Double, sumAb: Double, sumSq: Double, minX: Double, maxX: Double,
               name: String = "p-stat", _lastTime: Double = 0.0, _startTime: Double = 0.0): TimeStatistic =
        val stat   = new TimeStatistic (name, _lastTime, _startTime)
        stat.set (n, sum, sumAb, sumSq, minX, maxX)
        stat
    end apply

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Accumulate all the values weighted by their time durations and return the sum.
     *  @param x  the vector of values to accumulate (e.g., system occupancy)
     *  @param t  the vector of times of the observation (e.g, event times)
     */
    def accumAll (x: VectorD, t: VectorD): Double = 
        var s = 0.0
        for i <- 0 until x.dim-1 do s += x(i) * (t(i+1) - t(i))
        s
    end accumAll

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Aggregate the sub-statistics by summing all accumulators and determining
     *  new overall extreme values.
     *  @param subStats  the list of sub-statistics
     *  @param name      the name for the aggregated statistic
     */
    def aggregate (subStats: VEC [TimeStatistic], name: String = "ap-stat"): TimeStatistic =
        val m = subStats.size
        if m < 1 then { flaw ("aggregate", "there are no subparts to average"); return null }
        var n     = 0
        var sum   = 0.0
        var sumAb = 0.0
        var sumSq = 0.0
        var minX  = Double.MaxValue
        var maxX  = 0.0
        var starT = 0.0
        var lastT = 0.0
        for s <- subStats do
            n     += s.n_
            sum   += s.sum_
            sumAb += s.sumAb_
            sumSq += s.sumSq_
            if s.minX_ < minX      then minX  = s.minX_
            if s.maxX_ > maxX      then maxX  = s.maxX_
            if s.startTime < starT then starT = s.startTime
            if s.lastTime  > lastT then lastT = s.lastTime
        end for
        TimeStatistic (n, sum, sumAb, sumSq, minX, maxX, name, lastT, starT)
    end aggregate

end TimeStatistic


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `timeStatisticTest` main function is used to test the `TimeStatistic` class.
 *  > runMain scalation.mathstat.timeStatisticTest
 */
@main def timeStatisticTest (): Unit =

    val rv = SimpleUniform (0, 10)

    //:: Test ordinary sample statistics

    banner ("Test sample statistics")
    val stat1 = new Statistic ()
    cfor (0, 1000) { _ => stat1.tally (rv.gen) }

    println (Statistic.labels)
    println (stat1)
    println (stat1.show)
    println (s"ma = ${stat1.ma}, rms = ${stat1.rms}")

    //:: Test time-persistent statistics

    banner ("Test time-persistent statistics")
    val stat2 = new TimeStatistic ()
    for i <- 1 to 1000 do stat2.accum (rv.gen, 2.0 * i)

    println (Statistic.labels)
    println (stat2)
    println (stat2.show)
    println (s"ma = ${stat2.ma}, rms = ${stat2.rms}")

end timeStatisticTest

