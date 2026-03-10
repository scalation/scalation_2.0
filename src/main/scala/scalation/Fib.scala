
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Nov  6 12:03:37 EST 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Fibonacci Used to Illustrate Recursion - Time O(2^n) and Space O(n) Complexity
 *
 *  @see https://en.wikipedia.org/wiki/Fibonacci_sequence
 */

package scalation

//import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Fib` object provides a method for computing Fibonacci numbers.
 */
object Fib:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the n-th Fibonacci number, e.g., f(4) = f(3) + f(2) = 2 + 1 = 3.
     *  It uses the Naive Approach, see link for more efficient approaches.
     *  @see https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/
     *  @param n  the number to compute (for n >= 0)
     */
    def fib (n: Long): Long =
        if n <= 1 then n else fib (n-1) + fib (n-2)
    end fib

    def min (a: Array [Int], i: Int): Int =
        if i == 0 then a(0)
        else
            val amin = min (a, i-1)
            if a(i) < amin then a(i) else amin
    end min

end Fib


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `fibTest` main function test the `Fib` object.
 *  > runMain scalation.fibTest
 */
@main def fibTest (): Unit =

    import Fib.{fib, min}

    val nn  = 5               // takes too long for nn = 60
    var f_n = 0l
    
    for n <- 5 to nn by 5 do
        time { f_n = fib (n) }
        println (s"fib ($n) = $f_n")

    val rig = random.Randi0 (100000)
    val a = (for _ <- 0 until 10023 yield rig.igen).toArray     // 10023 works, 10024 stack overflow
//  println (s"a = ${stringOf (a)}")
    val m = min (a, a.length-1)
    println (s"min = $m")

end fibTest

