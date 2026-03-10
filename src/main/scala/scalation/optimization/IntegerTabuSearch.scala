
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Oct 13 12:03:21 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Integer Tabu Search
 */

package scalation
package optimization

import scala.collection.mutable.HashSet
import scala.math.max
import scala.util.control.Breaks.{break, breakable}

import scalation.mathstat.VectorI

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `IntegerTabuSearch` class performs tabu search to find minima of functions
 *  defined on integer vector domains z^n.  Tabu search will not re-visit points
 *  already deemed sub-optimal.
 *
 *  minimize    f(x)
 *  subject to  g(x) <= 0, x in Z^n
 *
 *  @param f        the objective function to be minimize (f maps an integer vector to a double)
 *  @param g        the constraint function to be satisfied, if any
 *  @param maxStep  the maximum/starting step size (make larger for larger domains)
 */
class IntegerTabuSearch (f: VectorI => Double,
                         g: VectorI => Double = null,
                         maxStep: Int = 10):

    type Vec_Func = (VectorI, Double)                           // pair: (integer vector, its functional value)

    private val debug   = debugf ("IntegerTabuSearch", false)   // debug function
    private val weight  = 1000                                  // weight on penalty for constraint violation
    private val maxIter = 4 * maxStep * maxStep                 // maxStep = 5 => MaxIter = 100
    private val tabu    = HashSet [VectorI] ()                  // set that keeps track of already visited points in search space

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** The objective function f re-scaled by a weighted penalty, if constrained.
     *  @param x  the coordinate values of the current point
     */
    def fg (x: VectorI): Double =
        if tabu contains x then                                 // already visited
            Double.PositiveInfinity                             // return worst possible value
        else 
            tabu += x                                           // indicate point x has been visited
            var sum = f(x)                                      // unconstrained value
            if g != null then                                   // if constrained
                sum += f(x) * weight * (max (g(x), 0))~^2       // add penalty
            sum
        end if
    end fg

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find a minimal neighbor of the current point x that is a distance step away.
     *  Let x be the current point with y being a step down and x being a step up
     *  in dimension i.  Recurse to handle all of the dimensions.
     *  @param x_f0  the current pair (the point and its functional value)
     *  @param i     the i-th dimension or coordinate (facilitates recursion)
     *  @param step  examine points that are this far away
     */
    def minNeighbor (x_f0: Vec_Func, i: Int, step: Int = 1): Vec_Func =
        val x = x_f0._1                                         // current point
        val y = x - (i, step)                                   // step down in dimension i: x_i - step
        val z = x + (i, step)                                   // step up in dimension i:   x_i + step

        var x_f = x_f0                                          // current pair (vector and its function)
        var y_f = (y, fg(y))                                    // down pair   
        var z_f = (z, fg(z))                                    // up pair
        debug ("minNeighbor", s"candidates at i = $i: $x_f, $y_f, $z_f")

        if i < x_f._1.dim - 1 then
            x_f = minNeighbor (x_f, i + 1, step)                // min in neighborhood of x_f
            y_f = minNeighbor (y_f, i + 1, step)                // min in neighborhood of y_f
            z_f = minNeighbor (z_f, i + 1, step)                // min in neighborhood of z_f

        if x_f._2 < y_f._2 then                                 // find smallest of 3 functional value
            if x_f._2 < z_f._2 then x_f else z_f
        else 
            if y_f._2 < z_f._2 then y_f else z_f
    end minNeighbor

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Solve the minimization problem by repeatedly moving to a minimal neighbor
     *  until there is no improvement.
     *  @param x  the starting point for the search
     */
    def solve (x: VectorI): Vec_Func =
        var x_f  = (x, fg(x))                                   // starting pair: vector x and its functional value f(x)
        var step = maxStep                                      // start with larger steps
        breakable {
            for k <- 1 to maxIter do
                println (s"+ k = $k, step = $step, x_f = $x_f")
                val y_f = minNeighbor (x_f, 0, step)
                if x_f._2 <= y_f._2 then                                // no improvement
                    if step == 1 then { print ("optimal "); break () }  // => return solution when step is 1
                    else step -= 1                                      // => decrease step size otherwise
                x_f = y_f                                               // move to improved point
            end for
        } // breakable
        println (s"value x_f = $x_f")                           // at maxIter could mean sub-optimal
        x_f                                                     // return optimal/sub-optimal solution
    end solve

end IntegerTabuSearch


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `integerTabuSearchTest` main method is used to test the `IntegerTabuSearch`
 *  class (unconstrained).
 *  > runMain scalation.optimization.integerTabuSearchTest
 */
@main def integerTabuSearchTest (): Unit =

    def f (x: VectorI): Double = (x(0) - 10)~^2 + (x(1) - 20)~^2 + 1
    val x0  = new VectorI (2)
    val ils = new IntegerTabuSearch (f)
    println ("optimal solution = " + ils.solve (x0))

end integerTabuSearchTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `integerTabuSearchTest2` main method is used to test the `IntegerTabuSearch`
 *  class (constrained).
 *  > runMain scalation.optimization.integerTabuSearchTest2
 */
@main def integerTabuSearchTest2 (): Unit =

    def f (x: VectorI): Double = (x(0) - 10)~^2+ (x(1) - 20)~^2 + 1
    def g (x: VectorI): Double = -max (x(0) - 1, x(1) - 1)    // require x(i) >= 1
    val x0  = new VectorI (2); x0.set (1)
    val ils = new IntegerTabuSearch (f, g)
    println ("optimal solution = " + ils.solve (x0))

end integerTabuSearchTest2

