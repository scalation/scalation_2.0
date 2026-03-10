
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Dec 30 14:48:41 EST 2009
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model for Table-Oriented Simulation Models
 */

package scalation
package simulation
package tableau

import scala.collection.mutable.ArrayBuffer

import scalation.mathstat._
import scalation.random.{Randi, Variate}
import scalation.random.RandomSeeds.N_STREAMS

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Model` class support tableau oriented simulation models in which each
 *  simulation entity's events are recorded in tabular form (in a matrix).
 *  This is analogous to Spreadsheet simulation.
 *  @see http://www.informs-sim.org/wsc06papers/002.pdf
 *  @param name    the name of simulation model
 *  @param m       the number entities to process before stopping
 *  @param rv      the random variate generators to use
 *  @param label_  the column labels for the matrix
 */
class Model (name: String, m: Int, rv: Array [Variate], label_ : Array [String] = null)
      extends Modelable:

    private val label = if label_ != null then label_
                        else Array ("      ID-0",                // entity identifier
                                    "IArrival-1",                // inter-arrival time
                                    " Arrival-2",                // arrival time
                                    "   Begin-3",                // begin service
                                    "    Wait-4",                // waiting time
                                    " Service-5",                // service time
                                    "Departure-6",               // departure time
                                    "   Total-7")                // total time in system
    val (e_a, e_d)  = (2, 6)                                     // the event columns (arrival, departure)
    private val mm  = m.toDouble                                 // m as a double
    private val n   = label.length                               // number of column labels
    private val len = 15 * n + 3                                 // length of "-" line
    private var tau = 0.0                                        // simulation duration (time of last event)

    /** The table/matrix holding information about entity timing
     *  row  0      - before any real entities
     *  rows 1 to m - entities 1 to m timings
     *  row  m+1    - column sums
     *  row  m+2    - column sample averages
     *  row  m+3    - column time averages
     */
    protected val tab = new MatrixD (m+4, n)

    for i <- 1 to m do tab(i, 0) = i                             // ID-0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Perform tableau-based simulation by recording timing information about
     *  the i-th entity in the i-th row of the matrix.  This method may need to
     *  be overridden for other models:  Copy this code and modify the equations
     *  as needed.
     *  @param startTime  may be used to initilize a start time other than zero
     */
    def simulate (startTime: Double = 0.0): Unit =
        for i <- 1 to m do
            tab(i, 1) = rv(0).gen                                // IArrival-1
            tab(i, 2) = tab(i-1, 2) + tab(i, 1)                  // Arrival-2
            tab(i, 3) = tab(i, 2) max tab(i-1, 6)                // Begin-3
            tab(i, 4) = tab(i, 3) - tab(i, 2)                    // Wait-4
            tab(i, 5) = rv(1).gen                                // Service-5
            tab(i, 6) = tab(i, 3) + tab(i, 5)                    // Departure-6
            tab(i, 7) = tab(i, 6) - tab(i, 2)                    // Total-7
        end for
    end simulate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print a vector (one row of the tab matrix) nicely formatted.
     *  @param x  the vector to print
     */
    private def printVec (x: VectorD): Unit =
        for j <- x.indices do print ("\t%10.3f".format (x(j)))
        println ()
    end printVec

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Report the simulation results as a entity time trace followed by column
     *  sums and averages.
     */
    def report (): Unit =
        var l = m
        while l > 0 && tab(l, e_d) == 0 do l -= 1                // last non-zero departure
        tau = tab(l, e_d)                                        // set end of simulation
        banner (s"$name Tableau Simulation Model Report - t_start = 0.0, t_end = $tau")

        for j <- 0 until n do
            tab(m+1, j) = tab(?, j).sum                          // column sums
            tab(m+2, j) = tab(m+1, j) / mm                       // column sample averages
            tab(m+3, j) = tab(m+1, j) / tau                      // column time averages
        end for

        println ("-" * len)
        for j <- label.indices do 
            print ("\t" + label(j))
            if label(j).length < 8 then print ("\t")
        end for
        println ()
        println ("-" * len)

        for i <- tab.indices do
            if i == 1 then println ("-" * len)
            if i == tab.dim - 3 then println ("-" * len)
            printVec (tab(i))
        end for
        println ("-" * len)

        for j <- label.indices do 
            print ("\t" + label(j))
            if label(j).length < 8 then print ("\t")
        end for
        println ()
        println ("-" * len)
    end report

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Format a double.
     *  @param x  the double to format
     */
    private def fmt (x: Double): String = "%10.3f".format (x)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Summarize the number and time in the queue (q), service (s) and system (y).
     *  Must call report before summary.
     *  Override as needed.
     */
    def summary (): Unit =
        val lambda_e = m / tau                                    // effective arrival rate
        val tt = VectorD (tab(m+2, 4), tab(m+2, 5), tab(m+2, 7))  // time in q, s, y
        val ll = tt * lambda_e                                    // number in q, s, y
        println (s"Simulation of $m entities in $tau units of time:")
        println (s"lambda_e = $lambda_e")
        println (s"""
        ---------------------------------------------------------
        |  Queue    |  L_q  | ${fmt(ll(0))}  |  T_q  | ${fmt(tt(0))}  |
        |  Service  |  L_s  | ${fmt(ll(1))}  |  T_s  | ${fmt(tt(1))}  |
        |  System   |  L_y  | ${fmt(ll(2))}  |  T_y  | ${fmt(tt(2))}  |
        ---------------------------------------------------------
        """)
    end summary

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a time-line of the simualtion giving all the event times in order
     *  and occupancy of (number in) the system L_y(t) at these event times.
     */
    def timeLine (): (VectorD, VectorD) =
        val et, lt = ArrayBuffer (0.0)
        var n = 0
        var i, j = 1
        while i <= m || j <= m do
           val (at, dt) = (tab(i, e_a), tab(j, e_d))
           if at == dt then
              et += at; lt += n
              i += 1; j += 1
           else if i <= m && at < dt then
              n  += 1
              et += at; lt += n
              i += 1
           else
              n  -= 1
              et += dt; lt += n
              j += 1
        end while
        (VectorD (et), VectorD (lt))
    end timeLine

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Save the table/matrix in a Comma Separated Value (.csv) file suitable for
     *  opening in a spreadsheet.  Note, the file 'data/tableau.csv' is overwritten.
     */
    def save (): Unit =
        tab.write (DATA_DIR + "tableau.csv")
    end save

end Model


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Model` companion object provides a function for computing the occupancy
 *  (number in system) L_y.
 */
object Model:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the occupancy of the model.
     *  @param (et, lt)  the event time and corresponding occupancy vectors
     */
    def occupancy (et_lt: (VectorD, VectorD)): Unit =
        val (et, lt) = et_lt
        val end      = et.last
        val sum_L_y  = TimeStatistic.accumAll (lt, et)
        println (s"event times: et = $et")
        println (s"occupancy:   lt = $lt")
        println (s"sum L_y = $sum_L_y")
        println (s"avg L_y = ${sum_L_y/end}")
        new Plot (et, lt, null, "occupancy L_y(t) versus time t", lines = true)
    end occupancy

end Model


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runModelTest` main function is used to test the `Model` class.
 *  Note make sure the streams are different for two Random Variables (RV)
 *  > runMain scalation.simulation.tableau.runModelTest
 */
@main def runModelTest (): Unit =

    val stream     = 0                                           // random number stream (0 to 999)
    val maxCusts   = 20                                          // stopping rule: maxCusts customers
    val iArrivalRV = Randi (1, 10, stream)                       // customer interarrival time 
    val serviceRV  = Randi (1, 10, (stream + 1) % N_STREAMS)     // customer service time
    val gg1 = new Model ("G/G/1 Queue", maxCusts, Array (iArrivalRV, serviceRV))
    gg1.simulate ()
    gg1.report ()
    gg1.summary ()
    Model.occupancy (gg1.timeLine ())
    gg1.save ()

end runModelTest

