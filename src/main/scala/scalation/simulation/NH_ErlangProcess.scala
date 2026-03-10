
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Apr  6 18:21:33 EDT 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Non-Homogeneous (changing arrival rate) Erlang Process (NHEP)
 */

package scalation
package simulation

import scala.collection.mutable.ArrayBuffer

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NH_ErlangProcess` class generates data following a Non-Homogeneous Erlang
 *  Process.
 *  @param t        the terminal time
 *  @param lambdaf  the arrival rate function, lambda(t)
 *  @param stream   the random number stream to use
 */
class NH_ErlangProcess (t: Double, lambdaf: FunctionS2S, stream: Int = 0)
      extends ErlangProcess (t, 1.0, stream):                      // use rate = 1 as it will be adjusted

    private val lambdaBar = func2vector (lambdaf, (0, t)).mean

    override def mean: VectorD = VectorD.fill (1)(lambdaBar * t)   // mean of N(t)

    override def pf (z: VectorD): Double = ???

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate all arrival times in the time interval [0, t], returning them
     *  as a vector.
     */
    override def gen: VectorD =
        val atime = ArrayBuffer [Double] ()
        var now   = 0.0
        while now <= t do
            val lamb = lambdaf (now)                               // current value of the lambda function
            println (s"lamb = $lamb")
            now     += t_ia.gen / lamb                             // adjust by dividing current lambda
            atime   += now 
        end while
        t_a = VectorD (atime)
        t_a
    end gen

end NH_ErlangProcess


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nH_ErlangProcessTest` main function is used to test the `NH_ErlangProcess` class.
 *  Example of car arrivals and determination of traffic flow (car per 5-minutes
 *  passing by a sensor).
 *  > runMain scalation.simulation.nH_ErlangProcessTest
 */
@main def nH_ErlangProcessTest (): Unit =

    val t_end = 50.0                                               // simulate for 50 minutes
    val tl    = VectorD.range (0, 101) / 2.0 
    def lambdaf (t: Double): Double = 1.5 - 0.001 * (t - 25.0)~^2
    new Plot (tl, func2vector (lambdaf, (0, t_end)), null, "Arrival Rate Function: lambdaf", lines = true)

    val pp = new NH_ErlangProcess (t_end, lambdaf)
    println (s"pp.gen     = ${pp.gen}")
    println (s"pp.num (5) = ${pp.num (5)}")

    val t  = VectorD.range (0, 501) / 10.0 
    val nt = new VectorI (t.dim)
    for i <- t.indices do nt(i) = pp.num (t(i))
    new Plot (t, nt.toDouble, null, "NH_ErlangProcess total cars", lines = true)

    val flw  = pp.flow (5.0)
    val tflw = VectorD.range (0, 11) * 5.0
    new Plot (tflw, flw.toDouble, null, "NH_ErlangProcess cars per 5 min.", lines = true)

end nH_ErlangProcessTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nH_ErlangProcessTest2` main function is used to test the `NH_ErlangProcess` class.
 *  Example showing how to use the `PolyRegression` class to create a lambda function
 *  based on traffic data.
 *  > runMain scalation.simulation.nH_ErlangProcessTest2
 */
@main def nH_ErlangProcessTest2 (): Unit =

    import scalation.modeling._

    val fileName = "travelTime.csv"
    val data = MatrixD.load (fileName)
    val ord  = 19

    val (t, y) = (data(?, 0) * 60.0, data(?, 1))                   // (time, vehicle count)
    new Plot (t, y, null, "traffic data")
    val mod = PolyRegression (t, y, ord, null, Regression.hp)
    mod.train ()
    val (yp, qof) = mod.test ()
    println (mod.report (qof))
    new Plot (t, y, yp, "traffic: actual vs. predicted")

    def lambdaf (tt: Double): Double = mod.predict (tt)

    val pp = new NH_ErlangProcess (t.dim-1, lambdaf)
    val flw  = pp.flow (1.0).toDouble
    new Plot (t, y, flw, "NH_ErlangProcess cars per 1 min.")

    val ft = new TestFit (y.dim)
    ft.diagnose (y, flw)
    println (FitM.fitMap (ft.fit, QoF.values.map (_.toString)))
    
end nH_ErlangProcessTest2

