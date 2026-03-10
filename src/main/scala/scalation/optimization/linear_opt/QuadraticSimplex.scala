
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Aug 24 15:05:08 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Quadratic Simplex Method for QP with Linear Constraints
 */

package scalation
package optimization
package linearopt

import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{breakable, break}

import scalation.mathstat.{MatrixD, VectorD}

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `QuadraticSimplex` class solves Quadratic Programming (QP) problems using the
 *  Quadratic Simplex Algorithm.  Given a constraint matrix a, constant vector b,
 *  cost matrix q and cost vector c, find values for the solution/decision
 *  vector x that minimize the objective function f(x), while satisfying all of
 *  the constraints, i.e.,
 *
 *  minimize    f(x) = 1/2 x q x + c x
 *  subject to  a x <= b, x >= 0
 *
 *  Creates an MM-by-NN simplex tableau.  This implementation is restricted to
 *  linear constraints a x <= b and q being a positive semi-definite matrix.
 *  Pivoting must now also handle non-linear complementary slackness.
 *  @see www.engineering.uiowa.edu/~dbricker/lp_stacks.html
 *
 *  @param a    the M-by-N constraint matrix
 *  @param b    the M-length constant/limit vector
 *  @param q    the N-by-N cost/revenue matrix (second order component)
 *  @param c    the N-length cost/revenue vector (first order component)
 *  @param x_B  the initial basis (set of indices where x_i is in the basis)
 */
class QuadraticSimplex (a: MatrixD, b: VectorD, q: MatrixD, c: VectorD, var x_B: Array [Int] = null):

    private val flaw     = flawf ("QuadraticSimplex")   // if true, show each pivot
    private val M        = a.dim                        // number of constraints (rows in a)
    private val N        = a.dim2                       // number of original variables (columns in a)
    private val MM       = M + q.dim                    // # rows in tableau
    private val NN       = N + M + 2 * q.dim2 + 2       // # columns in tableau
    private val MAX_IT   = 200 * N                      // maximum number of iterations

    if b.dim != M then  flaw ("init", "b.dim  = " + b.dim  + " != " + M)
    if c.dim != N then  flaw ("init", "c.dim  = " + c.dim  + " != " + N)
    if q.dim  != N then flaw ("init", "q.dim  = " + q.dim  + " != " + N)
    if q.dim2 != N then flaw ("init", "q.dim2 = " + q.dim2 + " != " + N)

    /** The MM-by-NN simplex tableau [ x, w, y, v, r | bc ]
     *  The complementary variables are x_i v_i = 0 = w_i y_i
     */
    private val t = new MatrixD (MM, NN)
    for i <- 0 until M do                               // fill the top part of the tableau
        t.set (i, a(i))                                 // x: constraint matrix a
        t(i, i+M+N) = 1.0                               // y: slack identity matrix
        t(i, NN-1)  = b(i)                              // bc: constant vector b

    for i <- M until MM do                              // fill the bottom part of the tableau
        t.set (i, q(i-M) * -1.0)                        // x: cost matrix q (quadratic part)
        set (t, i, a(?, i-M) * -1, N)                   // w: multipliers for a x <= b (transpose of a)
        t(i, i+M+N) = 1.0                               // v: multipliers for a x >= 0
        t(i, NN-1)  = c(i-M)                            // bc: cost vector c (linear part)

    for i <- 0 until MM if t(i, NN - 1) < 0 do t(i, NN - 2) = -1.0   // r: artificial variable

    if x_B == null then x_B = setBasis (MM, MM)

    showTableau ()                                      // before pivoting the artificial variable in
    val l = NN - 2                                      // entering column/variable (artificial variable)
    val k = argminNeg (t(?, NN - 1))                    // leaving row/variable
    pivot (k, l)                                        // special pivot puts artificial variable in basis

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set matrix mat's i-th row starting at column j to the vector u.
     *  @param i  the row index
     *  @param u  the vector value to assign
     *  @param j  the starting column index
     */
    def set (mat: MatrixD, i: Int, u: VectorD, j: Int = 0): Unit = 
        for k <- 0 until u.dim do mat(i, k+j) = u(k)
    end set

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the argument minimum of vector v or -1 if its value is not negative.
     *  @param v  the vector to examine
     *  @param e  the ending index (exclusive) for the search
     */
    private def argminNeg (v: VectorD): Int =
        val j = v.argmin (); if v(j) < 0.0 then j else -1
    end argminNeg

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** There are M+N variables, N decision and M slack variables, of which,
     *  for each iteration, M are chosen for a Basic Feasible Solution (BFS).
     *  The variables not in the basis are set to zero.  Setting j to N
     *  will start with the slack variables in the basis (only works if b >= 0).
     *  @param j  the offset to start the basis
     *  @param l  the size of the basis
     */
    def setBasis (j: Int = N, l: Int = M): Array [Int] =
        val ba = Array.ofDim [Int] (l)
        for i <- 0 until l do ba(i) = i + j
        ba
    end setBasis

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find a variable x_l to enter the basis.  Determine the index of entering
     *  variable corresponding to column l.  Neither the variable nor its complement
     *  may be in the current basis.   Return -1 to indicate no such column.
     */
    def entering (): Int =
        var (go, l) = (true, 0)
        cfor (go && l < NN-1, l += 1) { 
            var found = true
            breakable {
                for k <- 0 until x_B.length if l == x_B(k) || comple (l) == x_B(k) do
                    found = false; break ()
            } // breakable
            if found then go = false                // found x_l => quit early
        } // cfor
        if go then -1 else l-1                      // l-1 due to increment at end
    end entering

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return l's complementary variable.
     *  @param l  whose complement
     */
    def comple (l: Int): Int =
        if l < N then l + 3 * M
        else if l < N + M then l + M
        else if l < N + 2 * M then l - M
        else l - 3 * M
    end comple

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the best variable x_k to leave the basis given that x_l is entering.
     *  Determine the index of the leaving variable corresponding to row k using
     *  the Min-Ratio Rule.  Return -1 to indicate no such row.
     *  @param l  the entering variable (column)
     */
    def leaving (l: Int): Int =
        var k = -1
        for i <- 0 until MM if t(i, l) > 0 do
            if k == -1 then k = i
            else if t(i, NN - 1) / t(i, l) < t(k, NN - 1) / t(k, l) then k = i
        if k == -1 then println ("the solution is unbounded")
        k
    end leaving

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Pivot on entry (k, l) using Gauss-Jordan elimination to replace variable
     *  x_k with x_l in the basis.
     *  @param k  the leaving variable (row)
     *  @param l  the entering variable (column)
     */
    def pivot (k: Int, l: Int): Unit =
        println ("pivot: " + k + " leaves and " + l + " enters")
        // everything but row k and column l
        for i <- 0 until MM; j <- 0 until NN if i != k && j != l do
            t(i, j) -= t(k, j) * t(i, l) / t(k, l)

        for i <- 0 until MM if i != k do t(i, l) = 0.0         // zero out column l
        for j <- 0 until NN if j != l do t(k, j) /= t(k, l)    // scale row k
        t(k, l) = 1.0
        x_B(k) = l                                             // update basis (l replaces k)
    end pivot

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Run the simplex algorithm starting from the initial BFS and iteratively
     *  find a non-basic variable to replace a variable in the current basis
     *  so long as the objective improves.
     */
    def solve (): (VectorD, Double) =
        var k = -1                                             // the leaving variable (row)
        var l = -1                                             // the entering variable (column)
        showTableau ()

        breakable {
            for _ <- 0 until MAX_IT do
                l = entering (); if l == -1 then break ()      // optimal solution found
                k = leaving (l); if k == -1 then break ()      // solution is unbounded
                pivot (k, l)                                   // pivot: k leaves and l enters
        } // breakable

        showTableau ()
        val x = primal
        (x, objValue (x))                  // return the primal solution and the optimal value
    end solve

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the tableau t.
     */
    def tableau: MatrixD = t

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the primal solution vector x.
     */
    def primal: VectorD =
        val x = new VectorD (N)
        for i <- 0 until MM if x_B(i) < N do x(x_B(i)) = t(i, NN - 1)
        x
    end primal

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the dual solution vector (y).
     */
    def dual: VectorD = null   // FIX

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the optimal objective function value (f(x) = 1/2 x q x + c x).
     *  @param x  the primal solution vector
     */
    def objValue (x: VectorD): Double = (x dot (q * x)) * .5 + (c dot x)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Show the current basis and tableau.
     */
    def showTableau (): Unit =
        println ("basis x_B = " + stringOf (x_B))
        println ("tableau t = " + t)
    end showTableau

end QuadraticSimplex


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `quadraticSimplexTest` main function is used to test the `QuadraticSimplex` class.
 *  > runMain scalation.optimization.linearOpt.quadraticSimplexTest
 */
@main def quadraticSimplexTest (): Unit =

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test the `QuadraticSimplex` Algorithm for solving Quadratic Programs:
     *  min { 1/2 x q x + c x | a x <= b, x >= 0 }.
     *  @param a  the constraint matrix
     *  @param b  the constant vector
     *  @param q  the cost matrix
     *  @param c  the cost vector
     */
    def test (a: MatrixD, b: VectorD, q: MatrixD, c: VectorD): Unit =
        val lp = new QuadraticSimplex (a, b, q, c)
        val x  = lp.solve ()._1
        println ("tableau         t = " + lp.tableau)
        println ("primal vector   x = " + lp.primal)
        println ("dual vector     u = " + lp.dual)
        println ("objective value f = " + lp.objValue (x))
    end test

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test case 1: solution x = (.222222), 1.55556), z = -8.44444.
     *  @see www.engineering.uiowa.edu/~dbricker/Stacks_pdf2/QP_LCP_Example.pdf
     *  min x^2 - 2xy + y^2 - 4x - 6y
     *  st  2x + y <= 2
     *      -x + y <= 4
     */
    def test1 (): Unit =
        val a = MatrixD ((2, 2),  2.0, 1.0,
                                 -1.0, 1.0)
        val b = VectorD (2.0, 4.0)
        val q = MatrixD ((2, 2),  2.0, -2.0,
                                 -2.0,  2.0)
        val c = VectorD (-4.0, -6.0)
        test (a, b, q, c)
    end test1

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Test case 2: solution x = (0, 5), z = 20.
     *  @see http://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
     *  min .5x^2 + 3x + 4y
     *  st    x + 3y >=  15
     *       2x + 5y <= 100
     *       3x + 4y <=  80
     */
    def test2 (): Unit =
        val a = MatrixD ((3, 2), -1.0, -3.0,
                                  2.0,  5.0,
                                  3.0,  4.0)
        val b = VectorD (-15.0, 100.0, 80.0)
        val q = MatrixD ((2, 2),  1.0,  0.0,
                                  0.0,  0.0)
        val c = VectorD (3.0, 4.0)
        test (a, b, q, c)
    end test2

    println ("---------------------------------------------------------------")
    test1 ()
    println ("---------------------------------------------------------------")
    println (" test1: Answer should be x = (.222222), 1.55556), z = -8.44444")

    println ("---------------------------------------------------------------")
    test2 ()
    println ("---------------------------------------------------------------")
    println (" test2: Answer should be x = (0, 5), z = 20")
    println ("---------------------------------------------------------------")

end quadraticSimplexTest

