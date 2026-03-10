
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep  4 21:57:30 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note   Check Linaer Programming (LP) Solutions
 *
 *  @see Linear Programming and Network Flows, Bazaraa and Jarvis
 *  @see www.wiley.com/WileyCDA/WileyTitle/productCd-0470462728,subjectCd-BA04.html
 *  @see Algorithms, 4th Edition, Robert Sedgewick and Kevin Wayne
 *  @see www.cs.princeton.edu/algs4/63or/Simplex.java.html
 *  @see en.wikibooks.org/wiki/Operations_Research/The_Simplex_Method
 */

package scalation
package optimization
package linear_opt

import scala.math.abs

import scalation.mathstat._

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CheckLP` class checks the solution to Linear Programming (LP) problems.
 *  Given a constraint matrix 'a', limit/RHS vector 'b' and cost vector 'c',
 *  determine if the values for the solution/decision vector 'x' minimizes the
 *  objective function 'f(x)', while satisfying all of the constraints, i.e.,
 *
 *  minimize    f(x) = c x
 *  subject to  a x <= b, x >= 0
 *
 *  Check the feasibility and optimality of the solution.
 *  @param a  the M-by-N constraint matrix
 *  @param b  the M-length limit/RHS vector (make b_i negative for ">=" constraint => surplus)
 *  @param c  the N-length cost vector
 */
class CheckLP (a: MatrixD, b: VectorD, c: VectorD):

    private val debug   = debugf ("CheckLP", true)     // debug function
    private val flaw    = flawf ("CheckLP")            // flaw function
    private val EPSILON = 1E-9                         // number close to zero
    private val M       = a.dim                        // the number of constraints (row in a matrix)
    private val N       = a.dim2                       // the number of decision variables (columns in a matrix)

    if b.dim != M then flaw ("init", s"b.dim = ${b.dim} != $M")
    if c.dim != N then flaw ("init", s"c.dim = ${c.dim} != $N")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the solution primal feasible 'x >= 0 and a x [<= | >=] b'.
     *  @param x  the N-length primal solution vector
     */
    def isPrimalFeasible (x: VectorD): Boolean =
        if x.dim != N then flaw ("init", s"x.dim = ${x.dim} != $N")

        var feas = true
        // non-negativity constraints: check that x >= 0
        for j <- 0 until N if x(j) < 0.0 do
            flaw ("isPrimalFeasible", s"x($j) = ${x(j)} is negative")
            feas = false
        end for

        val ax = a * x
        // resource limit constraints: check that ax_i <= b_i
        for i <- 0 until M do
            val ax_i = ax(i)
            val b_i  = b(i)
            if ax_i > b_i + EPSILON then
                flaw ("isPrimalFeasible", s"constraint ax_i <= b_i violated for row $i: $ax_i > $b_i")
                feas = false
        end for
        feas
    end isPrimalFeasible

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the solution dual feasible 'y <= 0 and y a <= c'.
     *  @param y  the M-length dual solution vector
     */
    def isDualFeasible (y: VectorD): Boolean =
        if y.dim != M then flaw ("init", s"y.dim = ${y.dim} != $M")

        var feas = true
        // non-positivity constraints: check that y <= 0
        for i <- 0 until M if y(i) > 0.0 do
            flaw ("isDualFeasible", s"y($i) = ${y(i)} is positive")
            feas = false
        end for

        val ya = y *: a
        // dual constraints: check that ya_j <= c_j
        for j <- 0 until N do
            val ya_j = ya(j)
            val c_j  = c(j)
            if ya_j > c_j + EPSILON then
                flaw ("isDualFeasible", s"constraint ya_j <= c_j violated for col $j: $ya_j > $c_j")
                feas = false
        end for
        feas
    end isDualFeasible

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the optimum objective function value f == c x == y b.
     *  @param x  the N-length primal solution vector
     *  @param y  the M-length dual solution vector
     *  @param f  the optimum (minimum) value of the objective function
     */
    def isOptimal (x: VectorD, y: VectorD, f: Double): Boolean =
       val cx = c dot x                                         // c x
       val yb = y dot b                                         // y b

       var opti = true
       if abs (f - cx) > EPSILON then
           flaw ("isOptimal", s"failed since f = $f != c x = $cx")
           opti = false
       if abs (f - yb) > EPSILON then
           flaw ("isOptimal", s"failed since f = $f != y b = $yb")
           opti = false
       opti
    end isOptimal

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether the solution is correct, feasible and optimal.
     *  @param x  the N-length primal solution vector
     *  @param y  the M-length dual solution vector
     *  @param f  the optimum (minimum) value of the objective function
     */
    def isCorrect (x: VectorD, y: VectorD, f: Double): Boolean =
        val pFeas = isPrimalFeasible (x)
        val dFeas = isDualFeasible (y)
        val optim = isOptimal (x, y, f)
        debug ("isCorrect", s"isPrimalFeasible = $pFeas")
        debug ("isCorrect", s"isDualFeasible   = $dFeas")
        debug ("isCorrect", s"isOptimal        = $optim")
        pFeas && dFeas & optim
    end isCorrect

end CheckLP

