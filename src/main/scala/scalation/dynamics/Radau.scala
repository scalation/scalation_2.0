
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Dec 30 18:23:13 EST 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Radau IIA ODE Solver for Moderately Stiff Systems
 *
 *  @see     http://users.bart.nl/users/termaten/Publications/Coached/JdS_Radau_1997.pdf
 *  @see     http://www.dm.uniba.it/~testset/solvers/radau5.php
 *  @see     A simple ODE solver based on 2-stage Radau IIA  (Radau3)
 *           https://www.sciencedirect.com/science/article/pii/S0377042797001416
 */

// U N D E R   D E V E L O P M E N T

package scalation
package dynamics

import scala.math.sqrt
import scala.util.control.Breaks.{breakable, break}

import scalation.calculus.Differential._
import scalation.mathstat.{MatrixD, VectorD}

import MatrixD.eye

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Radau` object implements Radau IIA, which is a simple Ordinary Differential
 *  Equation ODE solver for moderately stiff systems.  Solve for y given
 *
 *      d/dt  y = f(t, y).
 */
object Radau
       extends Integrator:

//    import Derivatives._

    private val EPSILON  = 1E-7
    private val MAX_ITER = 100
    private val root6    = sqrt (6.0)
    private val _1_3     = 1.0 / 3.0
    private val _1_4     = 1.0 / 4.0
    private val _3_4     = 3.0 / 4.0
    private val _1_12    = 1.0 / 12.0
    private val _5_12    = 5.0 / 12.0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**
     */
    def integrate (f: Derivative, y0: Double, t: Double,
                   t0: Double = 0.0, step: Double = defaultStepSize): Double =
        0.0                                                           // FIX - to be implemented
    end integrate

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /**
     */
    def integrateVV (f: Array [DerivativeV], y0: VectorD, t: Double,
                     t0: Double = 0.0, step: Double = defaultStepSize): VectorD =
        y0                                                           // FIX - to be implemented
    end integrateVV

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Jacobian matrix for a vector-valued derivative function
     *  represented as an array of scalar-valued functions.  The i-th row in the
     *  matrix is the gradient of the i-th function.
     *  @param f  the array of functions whose Jacobian is sought
     *  @param y  the point (vector) at which to estimate the Jacobian
     */
    def jacobian (f: Array [DerivativeV], y: VectorD, t: Double): MatrixD =
        val ja = new MatrixD (f.length, y.dim)
        for i <- ja.indices do ja(i) = grad (f(i)(t, _: VectorD), y)
        ja
    end jacobian

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** 
     */
    def solve (f: Array [DerivativeV], yn_1: VectorD, fn_1: VectorD, tn_1: Double, tn: Double, h: Double): Unit =
        val h1_3  = _1_3 * h
        val h1_4  = _1_4 * h
        val h3_4  = _3_4 * h
        val h1_12 = _1_12 * h
        val h5_12 = _5_12 * h

        val gn    = yn_1 + fn_1 * h1_3
        val yn    = yn_1 + fn_1 * h

        val jacob = jacobian (f, yn_1, tn_1)
        val ident = eye (f.length, f.length)
        val lu    = ident - jacob * (root6 * h / 6.0)

        breakable {
            for _ <- 1 to MAX_ITER do
                val fg = new VectorD (f.length)
                for i <- fg.indices do fg(i) = f(i) (tn_1 + h1_3, gn)
                val fy = new VectorD (f.length)
                for i <- fy.indices do fy(i) = f(i) (tn, yn)
                val dg = lu.inverse * (yn_1 - gn + fg * h5_12 - fy * h1_12)
                val dy = lu.inverse * (dg * (4.0 * root6 - 8.0) + yn_1 - yn + fg * h3_4 + fy * h1_4)
    
                if dy.norm < EPSILON then break ()
    
                gn += dg
                yn += dg * (8.0 - 4.0 * root6) + dy
            end for
        } // breakable

        println ("yn = " + yn)
    end solve 

end Radau


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `radauTest` main function is used to test the `Radau` object.
 */
@main def radauTest (): Unit =

    println ("Radau is not implemented yet")
    // call solve

end radauTest

