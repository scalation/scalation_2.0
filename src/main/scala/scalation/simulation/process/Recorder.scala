

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Casey Bowman
 *  @version 2.0
 *  @date    Tue Mar 12 21:43:42 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Records the Flow of Actors/Vehicles (Counts and Speed)
 */

package scalation
package simulation
package process

import scala.math.floor

import scalation.mathstat.{MatrixD, Statistic}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Recorder` trait allows Components/Nodes to easily record the flow of actors/entities
 *  (e.g., vehicles) in terms of counts and optionally average speed (or other property
 *  of interest).
 *  @param nt      the number of time intervals (defaults to 60)
 *                     15-minute or 900-second intervals over 6:00 AM to 9:00 PM
 *  @param nLanes  the number of lanes
 */
trait Recorder (nt: Int = 60, nLanes: Int = 4):

    protected val r_counts = new MatrixD (nt, nLanes)                     // record counts in time interval
    protected val r_speeds = new MatrixD (nt, nLanes)                     // record average speed in time interval

    private val timeConv = 54000.0 / nt                                   // 60 * 60 * 15 = 54000 seconds per busy part of the day
//  private val timeConv = 86400.0 / nt                                   // 60 * 60 * 24 = 86400 seconds per day
    private var i_pre = 0                                                 // the current and previous time intervals
    private val lane_stat = Array.fill (nLanes) (new Statistic ("lane"))  // array of `Statistic`

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Get the recorder matrices.
     */
    def getRecorderMat: (MatrixD, MatrixD) = (r_counts, r_speeds)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Record the entity and optionally its speed (or other property of interest).
     *  @param ctime  the clock time the entity entered the component (e.g., Junction, Sink)
     *  @param speed  the speed at which entity entered the component (e.g., Junction, Sink)
     *  @param lane   the lane the vehicle is in
     */
    def record (ctime: Double, speed: Double, lane: Int): Unit =
        val i_cur = floor (ctime / timeConv).toInt                        // determine the current time interval
        if i_cur > i_pre then                                             // detected start of new time interval
            recordInMatrix (i_pre)                                        // put stats in recorder matrices
            i_pre = i_cur                                                 // update i_pre
        lane_stat(lane).tally (speed)                                     // record the speed/property of interest
    end record

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Record the vehicle entity and optionally its speed (or other property of interest).
     *  @param actor  the actor/vehicle being recorded
     *  @param ctime  the clock time the entity entered the component (e.g., Junction, Sink)
     */
    inline def record (actor: SimActor, ctime: Double): Unit =
        if actor.isInstanceOf [Vehicle] then
            val car = actor.asInstanceOf [Vehicle]
            record (ctime, car.velocity, car.subtype)
        else
            if actor.prop != null then
                record (ctime, actor.prop.head._2, actor.subtype)         // record value of first property
            else
                record (ctime, -0.0, actor.subtype)                       // no property to record
    end record

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Put the lane statistics in recorder matrices and reset the statistical counters
     *  at the end of each (ii-th) observation/time interval.
     *  @param ii    the relevant observation/time interval
     */
    private def recordInMatrix (ii: Int): Unit =
        for l <- r_counts.indices2 do                                     // for each lane
            r_counts(ii, l) = lane_stat(l).num                            // vehicles counted during the time interval
            r_speeds(ii, l) = lane_stat(l).mean                           // average speed during the time interval
            lane_stat(l).reset ()                                         // reset statistical counters
    end recordInMatrix

end Recorder


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `recorderTest` main function tests the `Recorder` trait.
 *  Creates fake simulated and actual values to test `Recorder` and `Fit` methods.
 *  > runMain scalation.simulation.process.recorderTest
 */
@main def recorderTest (): Unit =

    import scalation.modeling.Fit
    import scalation.random.{Normal, Randi, Uniform}

    val nparams = 5                                                       // number of parameters in fake simulation model

    val rlane  = Randi (0, 3, 1)                                          // stream 1
    val rspeed = Uniform (22.0, 34.0, 2)                                  // stream 2
    val noise  = Normal (0.0, 2.0, 3)                                     // stream 3 (use independent streams)

    object TestRec extends Recorder ()                                    // create a `Recorder` object

    var ctime = 0.0                                                       // initialize simulated time to zero
    for _ <- 0 until 10800 do                                             // for each of 10800 fake cars
        ctime += 5.0                                                      // time between cars = 5 (should use Erlang process)
        val lane  = rlane.igen                                            // randomly pick lane
        val speed = lane + rspeed.gen                                     // randomly set speed in m/s
        TestRec.record (ctime, speed, lane)                               // record information about this fake car
    end for

    val (cmat, smat) = TestRec.getRecorderMat                             // get fake simulated values
    banner ("Recorder Matrix for counts")
    println (s"cmat = $cmat")
    banner ("Recorder Matrix for speeds")
    println (s"smat = $smat")

    val cmat_ = new MatrixD (cmat.dim, cmat.dim2)                         // make fake actual values
    val smat_ = new MatrixD (smat.dim, smat.dim2)

    for i <- cmat.indices; j <- cmat.indices2 do
        cmat_(i, j) = 45.0 + noise.gen
        smat_(i, j) = 29.5 + noise.gen

    object TestFit extends Fit (dfr = nparams , df = cmat.dim - nparams)  // create a `Fit` object

    val cqof = TestFit.diagnose_mat (cmat_, cmat)                         // diagnostics for counts
    val sqof = TestFit.diagnose_mat (smat_, smat)                         // diagnostics for speeds

//  println (cqof)
//  println (sqof)

    banner ("Quality of Fit (QoF) for counts")
    println (Fit.showFitMap (cqof))
    banner ("Quality of Fit (QoF) for speeds")
    println (Fit.showFitMap (sqof))

end recorderTest

