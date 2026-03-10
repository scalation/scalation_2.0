
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Apr  6 17:25:30 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Erlang Process (Counting Process with Erlang Inter-arrivale Times)
 */

package scalation
package simulation

import scala.collection.mutable.ArrayBuffer

import scalation.mathstat._
import scalation.random.{Erlang, VariateVec}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `ErlangProcess` class generates data following a Erlng Process.
 *  @param t       the terminal time
 *  @param lambda  the arrival rate
 *  @param stream  the random number stream to use
 */
class ErlangProcess (t: Double, lambda: Double = 1.0, stream: Int = 0)
      extends VariateVec (stream):

    private   val mu   = 1.0 / lambda                     // mean interarrival time
    private   val k    = 2                                // number of stages/level of averaging
    protected val t_ia = Erlang (mu/k, k, stream)         // interarrival time distribution
    protected var t_a  = VectorD.nullv                    // arrival time vector

    def mean: VectorD = VectorD.fill (1)(lambda * t)      // mean of N(t)

    def pf (z: VectorD): Double = ???

    def igen: VectorI = gen.toInt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate the arrival times in the time interval [0, t], returning them
     *  as a vector.
     */
    def gen: VectorD =
        val atime = ArrayBuffer [Double] ()
        var now   = 0.0
        while now <= t do
            now   += t_ia.gen
            atime += now 
        end while
        t_a = VectorD (atime)
        t_a
    end gen

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of arrivals by time tt, i.e., N(tt)
     *  @param tt  the inquiry time (how many arrivals by time tt)
     */
    def num (tt: Double): Int = 
        if t_a == null then gen
        val i = t_a.indexWhere (_ > tt)
        if i < 0 then t_a.dim else i
    end num

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the arrivals/events occurring during each time interval of length t_span.
     *  @param t_span  the time span for an interval (e.g., 5 minute time span)
     */
    def flow (t_span: Double): VectorI =
        if t_a == null then gen
        val flw = ArrayBuffer [Int] ()
        var now = 0.0
        var n1  = 0
        while now <= t do
            val n2 = num (now)
            flw   += n2 - n1
            now   += t_span
            n1     = n2
        end while
        VectorI (flw)
    end flow

end ErlangProcess


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `erlangProcessTest` main function is used to test the `ErlangProcess` class.
 *  Example of car arrivals and determination of traffic flow (car per 5-minutes
 *  passing by a sensor).
 *  > runMain scalation.simulation.erlangProcessTest
 */
@main def erlangProcessTest (): Unit =

    banner ("running erlangProcessTest")
    val t_end  = 50.0                                  // simulate for 50 minutes
    val lambda = 1.0                                   // arrival rate 2 cars per minute
    val pp = new ErlangProcess (t_end, lambda)
    println (s"pp.gen     = ${pp.gen}")
    println (s"pp.num (5) = ${pp.num (5)}")

    banner ("Plot the Erlng Process: total cars")
    val t  = VectorD.range (0, 501) / 10.0 
    val nt = new VectorI (t.dim)
    for i <- t.indices do nt(i) = pp.num (t(i))
    new Plot (t, nt.toDouble, null, "ErlangProcess total cars", lines = true)

    banner ("Plot the flow of cars per 5 min.")
    val flw  = pp.flow (5.0)
    val tflw = VectorD.range (0, 11) * 5.0
    new Plot (tflw, flw.toDouble, null, "ErlangProcess cars per 5 min.", lines = true)

end erlangProcessTest

