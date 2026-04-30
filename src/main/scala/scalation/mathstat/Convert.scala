
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Aug 27 22:55:37 EDT 2021
 *  @see     LICENSE (MIT style license file). 
 *
 *  @note    Conversion Untilities
 */

package scalation
package mathstat

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Sample a function at n+1 equally spaced points in the domain and return
 *  their functional values in a vector.
 *  Ex: sample f(x) on x in [0, 10] at 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
 *  @param f   the function to be sampled
 *  @param ab  the interval of the domain [a, b]
 *  @param n   the number of points to be evaluated
 */
def func2vector (f: FunctionS2S, ab: Interval, n: Int = 100): VectorD =
    val step = (ab._2 - ab._1) / n
    var x    = ab._1 - step
    VectorD (for _ <- 0 to n yield { x += step; f(x) })
end func2vector

