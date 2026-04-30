
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jul 29 14:33:21 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Chemical Reactions Pathway
 */

package scalation
package dynamics

import scala.annotation.nowarn
import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `reactions` main function simulates a simple chemical pathway for making H2O.
 *  Compares ode44 and ode45.  FIX - add ode23s
 *  @see the KINSOLVER Paper: www.cs.uga.edu/~thiab/paper25.pdf
 *  > runMain scalation.dynamics.reactions
 */
@main def reactions (): Unit =

    val t0 = 0.0                                  // initial time
    val tf = 5.0                                  // final time
    val n  = 100                                  // number of time steps
    val _h2o = new VectorD (n+1)                  // vector to record concentrations of H2O for RK
    val h2o = new VectorD (n+1)                   // vector to record concentrations of H2O for DP

    val kf = (1.0,  1.0,  0.5)                    // forward reaction rates
    val kb = (0.02, 0.02, 0.01)                   // backward reaction rates

    // concentrations    H2   O2   O    H     OH   H2O
    //                    0    1    2    3     4    5
    var _c  = VectorD   (6.0, 6.0, 0.0, 0.01, 0.0, 1.0)
    var c   = VectorD   (6.0, 6.0, 0.0, 0.01, 0.0, 1.0)
    _h2o(0) = _c.last
    h2o(0)  = c.last

    // define the system of Ordinary Differential Equations (ODEs)

    // d[H2]/dt = - kf1 [H2] [O] + kb1 [H] [OH] - kf3 [H2] [OH] + kb3 [H2O] [H]
    @nowarn def dh2_dt (t: Double, c: VectorD)  = -kf._1*c(0)*c(2) + kb._1*c(3)*c(4) - kf._3*c(0)*c(4) + kb._3*c(5)*c(3) 

    // d[O2]/dt = - kf2 [H] [O2] + kb2 [O] [OH]
    @nowarn def do2_dt (t: Double, c: VectorD)  = -kf._2*c(3)*c(1) + kb._2*c(2)*c(4)

    // d[O]/dt = - kf1 [H2] [O] + kb1 [H] [OH] + kf2 [H] [O2] - kb2 [O] [OH]
    @nowarn def do_dt (t: Double, c: VectorD)   = -kf._1*c(0)*c(2) + kb._1*c(3)*c(4) + kf._2*c(3)*c(1) - kb._2*c(2)*c(4)

    // d[H]/dt = + kf1 [H2] [O] - kb1 [H] [OH] - kf2 [H] [O2] + kb2 [O] [OH] + kf3 [H2] [OH] - kb3 [H2O] [H]
    @nowarn def dh_dt (t: Double, c: VectorD)   =  kf._1*c(0)*c(2) - kb._1*c(3)*c(4) - kf._2*c(3)*c(1) + kb._2*c(2)*c(4) + kf._3*c(0)*c(4) - kb._3*c(5)*c(3)

    // d[OH]/dt = + kf1 [H2] [O] - kb1 [H] [OH] + kf2 [H] [O2] - kb2 [O] [OH] - kf3 [H2] [OH] + kb3 [H2O] [H]
    @nowarn def doh_dt (t: Double, c: VectorD)  =  kf._1*c(0)*c(2) - kb._1*c(3)*c(4) + kf._2*c(3)*c(1) - kb._2*c(1)*c(4) - kf._3*c(0)*c(4) + kb._3*c(5)*c(3)

    // d[H2O]/dt = + kf3 [H2] [OH] - kb3 [H2O] [H]
    @nowarn def dh2o_dt (t: Double, c: VectorD) =  kf._3*c(0)*c(4) - kb._3*c(5)*c(3)

    val odes: Array [DerivativeV] = Array (dh2_dt, do2_dt, do_dt, dh_dt, doh_dt, dh2o_dt)

    println ("dh2_dt  = " + dh2_dt (0.0, c))
    println ("do2_dt  = " + do2_dt (0.0, c))
    println ("do_dt   = " + do_dt (0.0, c))
    println ("dh_dt   = " + dh_dt (0.0, c))
    println ("doh_dt  = " + doh_dt (0.0, c))
    println ("dh2o_dt = " + dh2o_dt (0.0, c))

    println ("                           H2,  O2,   O,    H,  OH,  H2O")
    println (s"> at t = ${"%6.3f".format(t0)}, c = $c")
    val dt = tf / n                                        // time step
    var t  = t0 + dt                                       // next time point to examine

    for i <- 1 to n do
        _c = RungeKutta.integrateVV (odes, _c, dt)     // compute new concentrations using RK
        c  = DormandPrince.integrateVV (odes, c, dt)   // compute new concentrations using DP
        _h2o(i) = _c.last
        h2o(i)  = c.last

        println (s"> at t = ${"%6.3f".format(t)}, c = $c")
        t += dt
    end for

    new Plot (null, _h2o, h2o,  "Plot x vs. t (black-RK, red-DP)")

end reactions

