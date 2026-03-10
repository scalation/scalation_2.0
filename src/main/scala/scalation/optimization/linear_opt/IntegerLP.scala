

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Sep 11 22:43:04 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Integer Linear Programming (ILP) and Mixed Integer Linear Programming (MILP)
 */

package scalation
package optimization
package linear_opt

import scala.math.{abs, ceil, floor, round}

import scalation.mathstat.{MatrixD, VectorD}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `IntegerLP` class solves Integer Linear Programming (ILP) and Mixed Integer
 *  Linear Programming (MILP) problems recursively using the Simplex algorithm.
 *  First, an LP problem is solved.  If the optimal solution vector x is
 *  entirely integer valued, the ILP is solved.  If not, pick the first x_j
 *  that is not integer valued.  Define two new LP problems which bound x_j
 *  to the integer below and above, respectively.  Branch by solving each of
 *  these LP problems in turn.  Prune by not exploring branches less optimal
 *  than the currently best integer solution.  This technique is referred to
 *  as Branch and Bound.  An exclusion set may be optionally provided for
 *  MILP problems.
 *  FIX: Use the Dual Simplex Algorithm for better performance.
 *
 *  Given a constraint matrix a, limit/RHS vector b and cost vector c,
 *  find values for the solution/decision vector x that minimize the
 *  objective function f(x), while satisfying all of the constraints, i.e.,
 *
 *  minimize    f(x) = c x
 *  subject to  a x <= b, x >= 0, some x_i must be integer valued
 *
 *  Make b_i negative to indicate a >= constraint.
 *
 *  @param a     the M-by-N constraint matrix
 *  @param b     the M-length limit/RHS vector
 *  @param c     the N-length cost vector
 *  @param excl  the set of variables to be excluded from the integer requirement 
 */
class IntegerLP (a: MatrixD, b: VectorD, c: VectorD, excl: Set [Int] = Set ()):

    type Constraints = (MatrixD, VectorD)                         // a x <= b (-b_i for >=)

    private val EPSILON = 1E-9                                    // number close to zero
//  private val M       = a.dim                                   // # rows in the original constraint matrix
    private val N       = a.dim2                                  // # columns in the original constraint matrix
                                                                  // N = # decision variables

    // best integer solution so far
    private var best: (VectorD, Double) = (null, Double.PositiveInfinity)

    val x_le = new VectorD (N); x_le.set (-1.0)                   // constraint x_j <= value 
    val x_ge = new VectorD (N); x_ge.set (-1.0)                   // constraint x_j >= value

    println (">>>>>>>>>>>>>> root: dp = 0")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add a new constraint to the current set of bounding constraints: x_j <= bound
     *  or x_j >= bound (e.g., x_1 <= 2. or x_0 >= 4.).
     *  @param j      the index of variable x_j
     *  @param le     whether it is a "less than or equal to" le constraint
     *  @param bound  the bounding value
     */
    def addConstraint (j: Int, le: Boolean, bound: Double): Boolean =
        val low = x_le(j)
        val hi  = x_ge(j)
        if le then
            if low < 0.0 && hi < 0.0 then x_le(j) = bound                   // add "<=" constraint
            else if bound >= hi then      x_le(j) = bound                   // add "<=" constraint
            else if bound < hi then     { x_le(j) = bound; x_ge(j) = -1 }   // replace ">=" constraint
            else if bound < low then      x_le(j) = bound                   // replace "<=" constraint
            else return false
        else
            if low < 0.0 && hi < 0.0 then x_ge(j) = bound                   // add ">=" constraint
            else if bound <= low then     x_ge(j) = bound                   // add ">=" constraint
            else if bound > low then    { x_ge(j) = bound; x_le(j) = -1 }   // replace "<=" constraint
            else if bound > hi then       x_ge(j) = bound                   // replace ">=" constraint
            else return false
        end if
        true
    end addConstraint

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Starting with the original constraints (a, b) add the current set of bounding
     *  constraints.
     */
    def formConstraints: Constraints =
        var (aa, bb) = (a, b)                                     // start with the original constraints
        for j <- 0 until N do                                     // loop over the variables x_j
            if x_le(j) >= 0.0 then                                // check for x_j <= bound
                println (s"x_$j <= ${x_le(j)}")
                aa = aa :+ VectorD.oneAt (j, c.dim)               // add row to constraint matrix
                bb = bb :+ x_le(j)                                // add element to limit vector
                
            if x_ge(j) >= 0.0 then                                // check for x_j >= bound
                println ("x_$j >= ${x_ge(j)}")
                aa = aa :+ VectorD.oneAt (j, c.dim)               // add row to constraint matrix
                bb = bb :+ -x_ge(j)                               // add element to limit vector
        end for
        (aa, bb)                                                  // return the full set of constraints
    end formConstraints

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return j such that x_j has a fractional (non-integer) value, -1 otherwise.
     *  Make sure that j is not in the exclusion list.
     *  @param x  the vector to check
     */
    def fractionalVar (x: VectorD): Int =
        var (go, j, jj) = (true, 0, -1)
        cfor (go && j < x.dim, j += 1) {
            if ! (excl contains j) && abs (x(j) - round (x(j))) > EPSILON then { jj = j; go = false }
        } // cfor
        jj
    end fractionalVar

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the Integer Linear Programming (ILP) problem by using Branch and Bound
     *  and the Two-Phase Simplex Algorithm, recursively.
     *  @param dp    the current depth of recursion
     *  @param cons  the current set of original and new constraints (a x <= b)
     */
    def solve (dp: Int, cons: Constraints): Unit =
        val MAX_DEPTH = 4 * N                                     // limit on depth of recursion  FIX ??
        val lp = new Simplex2P (cons._1, cons._2, c)              // set up a new LP problem
        val x  = lp.solve ()                                      // optimal primal solution vector for this LP
        val y  = lp.dual                                          // optimal dual solution vector for this LP
        val f  = lp.objF (x)                                      // optimal objective function value for this LP
        if ! lp.check (x, y, f) then return                       // check the new LP, return if no solution
        
        val j = fractionalVar (x)                                 // find j such that x_j is not an integer
        var bound = 0.0

        println (s"IntegerLP.solve: x = $x, f = $f, j = $j")

        if j != -1 && f < best._2 && dp < MAX_DEPTH then          // x_j is not an integer => bound on both sides

            // add lower bound constraint: x_j <= floor (x(j))
            bound = floor (x(j))
            if addConstraint (j, true, bound) then
                println (">>>>>>>>>>>>>> left branch:  dp = " + (dp + 1))
                println (">>>>>>>>>>>>>> add constraint x_" + j + " <= " + bound)
                solve (dp + 1, formConstraints)

            // add upper bound constraint: x_j >= -ceil (x(j)) where "-" => ">=" constraint
            bound = ceil (x(j))
            if addConstraint (j, false, bound) then
                println (">>>>>>>>>>>>>> right branch: dp = " + (dp + 1))
                println (">>>>>>>>>>>>>> add constraint x_" + j + " >= " + bound)
                solve (dp + 1, formConstraints)
        end if

        if j == -1 then
            println ("###############################################################")
            println ("IntegerLP.solve: found an INTEGER solution (x, f) = " + (x, f))
            println ("###############################################################")
            if f < best._2 then best = (x, f)                     // save the best result
        end if
    end solve

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the optimal (minimal) integer solution.
     */
    def solution: (VectorD, Double) = best

end IntegerLP


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `IntegerLPTest` main function is used to test the `IntegerLP` class.
 *  real solution    x = (.8, 1.6), f = 8.8
 *  integer solution x = (2, 1),    f = 10
 *  @see Linear Programming and Network Flows, Example 6.14
 *  > runMain scalation.optimization.linear_opt.integerLPTest
 */
@main def integerLPTest (): Unit =

    val a = MatrixD ((2, 2), 3.0, 1.0,
                             1.0, 2.0)
    val c = VectorD         (3.0, 4.0) 
    val b = VectorD (-4.0, -4.0)

    val ilp = new IntegerLP (a, b, c)
    ilp.solve (0, (a, b))
    println ("###############################################################")
    println ("optimal solution = " + ilp.solution)
    println ("###############################################################")

end integerLPTest

