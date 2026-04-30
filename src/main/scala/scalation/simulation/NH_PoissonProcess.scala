
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Aug 25 15:38:28 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Non-Homogeneous (changing arrival rate) Poisson Process (NHPP)
 */

package scalation
package simulation

import scala.collection.mutable.ArrayBuffer

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NH_PoissonProcess` class generates data following a Non-Homogeneous Poisson
 *  Process.
 *  @param t        the terminal time
 *  @param lambdaf  the arrival rate function, lambda(t)
 *  @param stream   the random number stream to use
 */
class NH_PoissonProcess (t: Double, lambdaf: FunctionS2S, stream: Int = 0)
      extends PoissonProcess (t, 1.0, stream):                     // use rate = 1 as it will be adjusted

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

end NH_PoissonProcess


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nH_PoissonProcessTest` main function is used to test the `NH_PoissonProcess` class.
 *  Example of car arrivals and determination of traffic flow (car per 5-minutes
 *  passing by a sensor).
 *  > runMain scalation.simulation.nH_PoissonProcessTest
 */
@main def nH_PoissonProcessTest (): Unit =

    val t_end = 50.0                                               // simulate for 50 minutes
    val tl    = VectorD.range (0, 101) / 2.0 
    def lambdaf (t: Double): Double = 1.5 - 0.001 * (t - 25.0)~^2
    new Plot (tl, func2vector (lambdaf, (0, t_end)), null, "Arrival Rate Function: lambdaf", lines = true)

    val pp = new NH_PoissonProcess (t_end, lambdaf)
    println (s"pp.gen     = ${pp.gen}")
    println (s"pp.num (5) = ${pp.num (5)}")

    val t  = VectorD.range (0, 501) / 10.0 
    val nt = new VectorI (t.dim)
    for i <- t.indices do nt(i) = pp.num (t(i))
    new Plot (t, nt.toDouble, null, "NH_PoissonProcess total cars", lines = true)

    val flw  = pp.flow (5.0)
    val tflw = VectorD.range (0, 11) * 5.0
    new Plot (tflw, flw.toDouble, null, "NH_PoissonProcess cars per 5 min.", lines = true)

end nH_PoissonProcessTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `nH_PoissonProcessTest2` main function is used to test the `NH_PoissonProcess` class.
 *  Example showing how to use the `PolyRegression` class to create a lambda function
 *  based on traffic data.
 *  > runMain scalation.simulation.nH_PoissonProcessTest2
 */
@main def nH_PoissonProcessTest2 (): Unit =

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

    val pp = new NH_PoissonProcess (t.dim-1, lambdaf)
    val flw  = pp.flow (1.0).toDouble
    new Plot (t, y, flw, "NH_PoissonProcess cars per 1 min.")

    val ft = new TestFit (y.dim)
    ft.diagnose (y, flw)
    println (FitM.fitMap (ft.fit, QoF.values.map (_.toString)))
    
end nH_PoissonProcessTest2

