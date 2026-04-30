
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Mar 30 13:55:49 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Top-Level Constants and Methods for Recording Run-Times and Memory Usage
 *
 *  Java's `nanoTime` method is used for approximate timings based on elapsed-times.
 *  @see https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/lang/System.html#nanoTime()
 *  @see https://cs.colby.edu/courses/S20/cs231-labs/labs/lab04/timing.php
 *  @see http://vision.psych.umn.edu/users/boyaci/Guide_v00/guidePrcsTime.pdf
 *
 *  Consensus:  the resolution of nanoTime (in units of nano-seconds) is somewhere in
 *  the micro-second (10-6 second) range.
 */

package scalation

import java.lang.System.nanoTime

/** The number of units in one second
 *  Use 1.0 to make SECOND the base unit, 1.0 / 60.0 to make MINUTE the base unit, etc. 
 *  @see `TimeNum` for alternative definitions
 */
val SECOND = 1.0
// val SECOND = 1.0 / 60.0

/** The number of seconds in one minute
 */
val MINUTE = 60.0 * SECOND

/** The number of minutes in one hour
 */
val HOUR = 60.0 * MINUTE

/** The number of hours in one day
 */
val DAY = 24.0 * HOUR

/** The number of days in one week
 */
val WEEK = 7.0 * DAY

/** The number of millisecond (10-3) per second
 */
val MS_PER_SEC = 1E-3

/** The number of nanoseconds (10-9) per millisecond
 */
val NS_PER_MS = 1E-6

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Print the elapsed time in milliseconds (ms) for the execution of an arbitrary
 *  block of code:  'time { block }'.  Return any result produced by the block of code.
 *  @see http://stackoverflow.com/questions/9160001/how-to-profile-methods-in-scala
 *  @param block  the block of code to be executed
 */
def time [R] (block: => R): R =
    val t0 = nanoTime ()                                   // start time
    val result = block                                     // call-by-name
    val t1 = nanoTime ()                                   // end time
    println ("Elapsed time: " + (t1 - t0) * NS_PER_MS + " ms")
    result
end time

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Print the elapsed time in milliseconds (ms) for the execution of an arbitrary
 *  block of code:  'time (10) { block }'.  Return any result produced by the block of code.
 *  Executes the block of code multiple times for better resolution and accuracy.
 *  @see http://stackoverflow.com/questions/9160001/how-to-profile-methods-in-scala
 *  @param amplify  the number of times to execute the block of code
 *  @param skip     whether to skip the first execution due to slowness of the JIT-Compiler
 *  @param block    the block of code to be executed
 */
def time [R] (amplify: Int, skip: Boolean = true) (block: => R): R =
    var result: R = null.asInstanceOf [R]
    if skip then result = block                            // skip measuring first time due to JIT-Compiler
    val t0 = nanoTime ()                                   // start time
    cfor (0, amplify) { _ => result = block }              // exercise code amplify times
    val t1 = nanoTime ()                                   // end time
    println ("Elapsed time: " + (t1 - t0) * NS_PER_MS / amplify + " ms")
    result
end time

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Calculate the elapsed time in milliseconds (ms) for the execution of an
 *  arbitrary block of code:  'timed { block }'.  Return any result produced
 *  by the block of code and its elapsed time.
 *  @see http://stackoverflow.com/questions/9160001/how-to-profile-methods-in-scala
 *  @param block  the block of code to be executed
 */
def timed [R] (block: => R): (R, Double) =
    val t0 = nanoTime ()                                   // start time
    val result = block                                     // call-by-name
    val t1 = nanoTime ()                                   // end time
    (result, (t1 - t0) * NS_PER_MS)
end timed

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Calculate the elapsed time in milliseconds (ms) for the execution of an
 *  arbitrary block of code:  'timed { block }'.  Return any result produced
 *  by the block of code and the average elapsed time.
 *  @see http://stackoverflow.com/questions/9160001/how-to-profile-methods-in-scala
 *  Executes the block of code multiple times for better resolution and accuracy.
 *  @param amplify  the number of times to execute the block of code
 *  @param skip     whether to skip the first execution due to slowness of the JIT-Compiler
 *  @param block    the block of code to be executed
 */
def timed [R] (amplify: Int, skip: Boolean = true) (block: => R): (R, Double) =
    var result: R = null.asInstanceOf [R]
    if skip then result = block                            // skip measuring first time due to JIT-Compiler
    val t0 = nanoTime ()                                   // start time
    cfor (0, amplify) { _ => result = block }              // exercise code amplify times
    val t1 = nanoTime ()                                   // end time
    (result, (t1 - t0) * NS_PER_MS / amplify)
end timed

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Calculate the elapsed time in milliseconds (ms) for the execution of an
 *  arbitrary block of code:  'timedX { block }'.  Return any result produced
 *  by the block of code and the average elapsed time.
 *  @see http://stackoverflow.com/questions/9160001/how-to-profile-methods-in-scala
 *  Executes the block of code multiple times for better resolution and accuracy,
 *  throwing the slowest execution to improve consistency (e.g., may be slow due to GC).
 *  @param amplify  the number of times to execute the block of code
 *  @param skip     whether to skip the first execution due to slowness of the JIT-Compiler
 *  @param block    the block of code to be executed
 */
def timedX [R] (amplify: Int, skip: Boolean = true) (block: => R): (R, Double) =
    var result: R = null.asInstanceOf [R]
    if skip then result = block                            // skip measuring first time due to JIT-Compiler
    val dt = Array.ofDim [Long] (amplify+1)
    var t0 = 0L
    cfor (0, amplify+1) { i =>
        t0 = nanoTime ()                                   // start time
        result = block
        dt(i)  = nanoTime () - t0                          // time difference
    } // cfor
    (result, (dt.sum - dt.max) * NS_PER_MS / amplify)
end timedX

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Calculate the elapsed time in milliseconds (ms) for the execution of an
 *  arbitrary block of code:  'gauge { block }'.  Return the block of code's
 *  elapsed time.
 *  @see http://stackoverflow.com/questions/9160001/how-to-profile-methods-in-scala
 *  @param block  the block of code to be executed
 */
def gauge [R] (block: => R): Double =
    val t0 = nanoTime ()                                   // start time
    block                                                  // call-by-name
    val t1 = nanoTime ()                                   // end time
    (t1 - t0) * NS_PER_MS
end gauge

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Calculate the elapsed time in milliseconds (ms) for the execution of an
 *  arbitrary block of code:  'gauge { block }'.  Return the block of code's
 *  elapsed time.
 *  @see http://stackoverflow.com/questions/9160001/how-to-profile-methods-in-scala
 *  @param amplify  the number of times to execute the block of code
 *  @param skip     whether to skip the first execution due to slowness of the JIT-Compiler
 *  @param block    the block of code to be executed
 */
def gauge [R] (amplify: Int, skip: Boolean = true) (block: => R): Double =
    if skip then block                                     // skip measuring first time due to JIT-Compiler
    val t0 = nanoTime ()                                   // start time
    cfor (0, amplify) { _ => block }                       // exercise code amplify times
    val t1 = nanoTime ()                                   // end time
    (t1 - t0) * NS_PER_MS / amplify
end gauge

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the free, total, and max memory in MB (actually MiB).
 */
def memoryUsed: Array [Long] =
    val MB = 1024 * 1024
    val runtime = Runtime.getRuntime
    Array (runtime.freeMemory / MB, runtime.totalMemory / MB, runtime.maxMemory / MB)
end memoryUsed


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `timerTest` main function tests the timer methods.
 *  Due to the short execution time of the quadraticEq method, the amplify value should be
 *  1000 to 1000000.
 *  > runMain scalation.timerTest
 */
@main def timerTest (): Unit =

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the real roots of a quadratic equation ax^2 + bx + c = 0.
     *  Note there exist more numerically accurate algorithms.
     *  @see https://en.wikipedia.org/wiki/Quadratic_formula
     */
    def quadraticEq (a: Double, b: Double, c: Double): (Double, Double) =
        if a == 0.0 then
            (-c / b, NO_DOUBLE)                            // one real root
        else
            val _2a = 2 * a                                // denominator
            val d   = b~^2 - 4 * a * c                     // discriminant
            if d == 0.0 then
                (-b / _2a, NO_DOUBLE)                      // one real root
            else if d < 0.0 then
                (NO_DOUBLE, NO_DOUBLE)                     // two complex roots
            else 
                val rd = math.sqrt (d)                     // root of discriminant
                ((-b + rd) / _2a, (-b - rd) / _2a)         // two real roots
    end quadraticEq

    val (a, b, c) = (1, -5, 6)
    banner (s"time: Quadratic Equation ax^2 + bx + c = 0 where a = $a, b = $b, c = $c") 
    val roots = time (10000) { quadraticEq (1, -5, 6) }
    println (s"roots = $roots")

    banner (s"timed: Quadratic Equation ax^2 + bx + c = 0 where a = $a, b = $b, c = $c") 
    var result = timed (10000) { quadraticEq (1, -5, 6) }
    println (s"result = $result")

    banner (s"timedX: Quadratic Equation ax^2 + bx + c = 0 where a = $a, b = $b, c = $c") 
    result = timedX (10000) { quadraticEq (1, -5, 6) }
    println (s"result = $result")

end timerTest

