
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Mon Sep 20 15:47:16 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Statistical Trait
 */

package scalation
package simulation.agent

import scala.collection.mutable.{ArrayBuffer => VEC}

import scalation.mathstat.{Statistic, TimeStatistic}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Statistical` trait maintain statistical collectors.
 *  @param name  the name of this statistical object
 */
trait Statistical (name: String):

    private val sampStatistic = new Statistic (name)                   // sample statistics
    private val persStatistic = new TimeStatistic ("p-"+name)          // time-persistent statistics

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tally sample statistics.
     *  @param duration  the time duration to tally
     */
    private [agent] def tallyStats (duration: Double): Unit =
        sampStatistic.tally (duration)
    end tallyStats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Accumulate time-persistent statistics.
     *  @param count  the numerical count to accumulate
     *  @param time   the relevant time
     */
    private [agent] def accumStats (count: Int, time: Double): Unit =
        persStatistic.accum (count, time)
    end accumStats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Tally sample statistics and accumulate time-persistent statistics.
     *  @param duration  the time duration to tally
     *  @param count     the numerical count to accumulate
     *  @param time      the relevant time
     */
    private [agent] def collectStats (duration: Double, count: Int, time: Double): Unit =
        sampStatistic.tally (duration)
        if ! this.isInstanceOf [Source | Sink | Gate ] then persStatistic.accum (count, time)
    end collectStats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add the sample statistics and time-persistent statistic to the list.
     *  @param list  the collection of statistical objects
     */
    private [agent] def addStats (list: VEC [Statistic]): Unit =
        list += sampStatistic
        if ! this.isInstanceOf [Source | Sink | Gate ] then list += persStatistic
    end addStats

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return sample statistics for durations for this component (e.g., Time in queue).
     */
    def durationStat: Statistic = sampStatistic

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return time persistent statistics for value for this component (e.g. Number in queue).
     */
    def persistentStat: TimeStatistic = persStatistic

end Statistical

