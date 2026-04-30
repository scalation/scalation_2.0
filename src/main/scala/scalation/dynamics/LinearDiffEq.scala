
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Jan 29 18:36:48 EST 2010
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Linear Differential Equations Solver
 */

package scalation
package dynamics

import scala.math.exp

import scalation.mathstat._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LinearDiffEq` class may be used for solving a system of linear differential
 *  equations that are ordinary and first-order with constant coefficients of the form
 *      d/dt y(t) = a * y(t)
 *  y(t) is the vector function of time and a is the coefficient matrix.  The
 *  initial value vector y0 = y(0) must also be given.  Note, higher-order differential
 *  equations may be converted to first-order by introducing additional variables.
 *  The above equation is the homogeneous case.
 *  @caveat:  the following `LinearDiffEq` cases are not currently handled:
 *            (1) The non-homogeneous equation: d/dt y(t) = a * y(t) + f(t).
 *            (2) Complex or repeated eigenvalues.
 *  @param a   the coefficient matrix
 *  @param y0  the initial value vector
 */
class LinearDiffEq (a: MatrixD, y0: VectorD):

     private val flaw = flawf ("LinearDiffEq")               // flaw function

     if a.dim2 != y0.dim then flaw ("init", "incompatible dimensions")

     /** Vector of eigenvalues
      */
     private val e = VectorD.nullv  // FIX (new Eigenvalue (a)).getE ()

     /** Matrix of eigenvectors
      */
     private val v = MatrixD.nullm  // FIX (new Eigenvector (a, e)).getV

     /** Vector of constants
      */
     private val c = VectorD.nullv  // FIX  v.solve (y0) { val lu =new Fac_LU (v); lu.facror (); lu.solve (y0) }

     /** Matrix of transformed/final constants
      */
     private val k = v *~ c

     //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
     /** Apply the exponential `exp` function to each element of a vector.
      *  @param v  the vector to apply the exp function to
      */
     def expV (v: VectorD): VectorD = v.map (exp (_))

     //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
     /** Evaluate the solution for y(t) at time t.
      *  @param t  the time point
      */
     def eval (t: Double): VectorD = k * expV (e * t)

     //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
     /** Print the solution to the differential equation.
      */
     def printSol (): Unit =
         println ("---------------------------------------")
         println ("System of Linear Differential Equations")
         println ("Solve: y(t)' = a * y(t) where y(0) = y0")
         println ("coefficient matrix    a  = " + a)
         println ("initial state vector y0 = " + y0)
         println ("eigenvalue vector     e  = " + e)
         println ("eigenvector matrix    v  = " + v)
         println ("constant vector       c  = " + c)
         println ("constant matrix       k  = " + k)
         println ("---------------------------------------")
     end printSol

end LinearDiffEq


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `linearDiffEqTest` object to test the `LinearDiffEq` class using example at
 *  @see biomed.tamu.edu/faculty/wu/BMEN_452/Eigenvalue%20Problems.doc
 *  The eigenvalues should be (-3, -1)
 *  The constant matrix should be [ (.375, .625), (-.75, 1.25) ]
 *  > runMain scalation.dynamics.linearDiffEqTest
 */
@main def linearDiffEqTest (): Unit =

    val a  = MatrixD ((2, 2), -2.0,  0.5,                    // 2-by-2 matrix
                               2.0, -2.0)
    val y0 = VectorD (1.0, 0.5)
    val de = new LinearDiffEq (a, y0)
    de.printSol ()

    val n = 60                     // number of iterations
    val p = new MatrixD (n, 2)     // n 2D vectors (x, y)
    val t = new VectorD (n)        // time points

    for i <- 0 until n do
        t(i) = .25 * i
        p(i) = de.eval (t(i))
        println ("at t = " + t(i) + " trajectory = " + p(i))
    end for

    println ("Plot (x, y) vs. t")
    new Plot (t, p(?, 0), p(?, 1), "Plot (x, y) vs. t")
    println ("Plot y vs. x")
    new Plot (p(?, 0), p(?, 1), null, "Plot y vs. x")

end linearDiffEqTest

