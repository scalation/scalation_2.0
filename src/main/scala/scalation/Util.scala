
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Jul  9 17:40:18 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Util Contains Basic Utilities: Generally Useful Top-Level Constants and Methods.
 *
 *  Supports multiple cfor loops that are much faster than Scala's for-comprehensions.
 */

package scalation

import java.io.File
import java.net.{MalformedURLException, URI} // URL

import scala.collection.mutable.ArrayBuffer
import scala.math.{max, min}
import scala.io.Source.{fromFile, fromURL}
//import scala.runtime.ScalaRunTime.stringOf
import scala.util.Properties.envOrElse

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/*  Top-level definitions of values/constants and functions of utilities for files and
 *  language extensions such as cfor.
 *  @see ValueType.scala for top-level definitions related to basic datatypes.
 *  @see CommonFunctions.scala for top-level definitions related to common scalar functions.
 *  @see Timer.scala for top-level defination related to recording run-times and memory usage.
 */

/** Type definition for an interval [a, b]
 */
type Interval = (Double, Double)

/** Universal symbol for the ? character (e.g., meaning all rows of column 2 in x(?, 2) for matrix x)
 */
val ? = '?'

/** The file path separation character: '/' for Linux/Mac, '\' for Windows
 *  Use either 'SEP' or '⁄' for portability, i.e., do not use '/' or '\'.
 */
val SEP = File.separator
val ⁄   = File.separator                                    // Unicode symbol

/** File-name extension for serialized tables
 *  FIX: investigate using more efficient serialization, e.g.,
 *  @see github.com/EsotericSoftware/kryo
 */
val SER = ".ser"

/** File-name extension for CSV data files
 */
val CSV = ".csv"

/** File-name extension for JSON data files
 */
val JSON = ".json"

/** The token/element separation character (',' for CSV)
 */
val SP = ','

/** Base directory for ScalaTion (pick one, comment out the rest)
 */
val BASE = System.getProperty ("user.dir")                  // absolute path
// val BASE = "."                                           // relative path

/** File system path for input/output data directory
 *  Use 'SCALATION_HOME' environment variable or else BASE directory "."
 */
val DATA_DIR = envOrElse ("SCALATION_HOME", BASE) + ⁄ + "data" + ⁄

/** File system path for log (log/file output) directory
 *  Use 'SCALATION_HOME' environment variable or else BASE directory "."
 */
val LOG_DIR = envOrElse ("SCALATION_HOME", BASE) + ⁄ + "log" + ⁄

/** File system path for src (source code) directory
 *  Use 'SCALATION_HOME' environment variable or else BASE directory "."
 */
val SRC_DIR = envOrElse ("SCALATION_HOME", BASE) + ⁄ + "src" + ⁄
val SRC_SCALA_DIR = SRC_DIR + ⁄ + "main" + ⁄ + "scala" + ⁄

/** File system path for database storage directory
 *  Use 'SCALATION_HOME' environment variable or else BASE directory "."
 */
val STORE_DIR = envOrElse ("SCALATION_HOME", BASE) + ⁄ + "store" + ⁄

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Remove the last path element from the file/directory pathname.
 *  @param s  the string to be so truncated
 */
def removeLast (s: String): String = s.substring (0, s.lastIndexOf (SEP))

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return a line iterator for a line-oriented data source (e.g., CSV file).
 *  The data source is accessed via (1) URL, (2) file's absolute path, or
 *  (3) file's relative path (relative to 'DATA-DIR').
 *  @see stackoverflow.com/questions/5713558/detect-and-extract-url-from-a-string
 *  @param path  the path name of the data source (via URL or file's path name)
 */
def getFromURL_File (path: String): Iterator [String] =
    val urlPat = "(?i)((https?|ftp|file)://|file:/).*"     // (?i) => case insensitive
    if path.matches (urlPat) then
        try
//          return fromURL (new URL (path)).getLines ()
            return fromURL (new URI (path).toURL).getLines ()
        catch
            case _ : MalformedURLException => 
        end try    
    end if

    val file = new File (path)
//  if file.isAbsolute () then fromFile (file).getLines ()
    if file.exists ()     then fromFile (file).getLines ()
    else
        println (s"getFromURL_File: file '$path' does not exist, try prefixing DATA-DIR")
        fromFile (DATA_DIR + path).getLines ()
end getFromURL_File

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Make a string for use in printing a line of '-'s.
 *  @param n  the number of '-'s to use
 */
def sline (n: Int = 60): String = "-" * n + "\n"

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Print a banner, i.e., a string in a box.
 *  @param str  the string to put in the banner
 */
def banner (str: String): Unit =
    val len = str.size + 4
    println ("-" * len)
    println ("| " + str + " |")
    println ("-" * len)
end banner

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Show the flaw by printing the error message and returning false.
 *  Usage: def flaw: Unit = flawf ("MyClass")
 *         flaw ("myMethod", "Houston we have a problem")
 *  @param clsName  the class/trait/object containing the method
 *  @param method   the method where the error occurred
 *  @param message  the error message
 */
inline def flawf (clsName: String)(method: String, message: String): Boolean =
    import scala.Console.{RED, RESET}
    println (s"${RED}ERROR @ $clsName.$method: $message ${RESET}")
    false
end flawf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Show the debugging information that can be enabled or disabled.
 *  Usage: def debug: Unit = debugf ("MyClass", true)
 *         debug ("myMethod", s"x = $x")
 *  @param clsName  the class/trait/object containing the method
 *  @param enabled  indicates whether the debug message is enabled
 *  @param method   the method where the error occurred
 *  @param message  the error message
 */
inline def debugf (clsName: String, enabled: Boolean)(method: String, message: String): Unit =
    if enabled then
        println (s"DEBUG @ $clsName.$method: $message")
end debugf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Return the type of an object as string, e.g., "Double".
 *  @param o  the object whose type is sought
 */
def typeOf (o: Any): String = o.getClass.getSimpleName

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Remove 'n' characters starting at position 'i' in string 'str' and
 *  return the new string.
 *  @param str  the source string 
 *  @param i    the starting position for character removal
 *  @param n    the number of characters to remove
 */
def removeAt (str: String, i: Int, n: Int = 1): String =
    str.substring (0, i) + str.substring (i + n)
end removeAt

def removeAt (str: Array [Char], i: Int, n: Int): Array [Char] =
    str.slice (0, i) ++ str.slice (i + n, str.length)
end removeAt

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** C/Java style for loop provides improved performance.
 *  usage: var i = 0; cfor (i < 10, i += 1) { a(i) = 2 * i }
 *  @see www.reddit.com/r/scala/comments/4m2qgb/will_the_new_optimizer_in_scala_212_help_optimize/
 *  @see `cforTest` main function at the end of this file
 *  @param pred  loop continuation predicate/condition
 *  @param step  increment step
 *  @param body  main body of the loop
 */
inline def cfor (pred: => Boolean, step: => Unit) (inline body: => Unit): Unit =
    while pred do { body; step }
end cfor

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** C/Java style for loop provides improved performance (assumes i is already defined).
 *  usage: cfor ({i = 0}, i < 10, i += 1) { a(i) = 2 * i }
 *  @see www.reddit.com/r/scala/comments/4m2qgb/will_the_new_optimizer_in_scala_212_help_optimize/
 *  @param init  initialization part
 *  @param pred  loop continuation predicate/condition
 *  @param step  increment step
 *  @param body  main body of the loop
 *
inline def cfor (init: => Unit, pred: => Boolean, step: => Unit) (inline body: => Unit): Unit =
    init; while pred do { body; step }
end cfor
 */

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** C/Java style for loop (over a range) provides improved performance.
 *  usage: cfor (0, 10) { i => a(i) = 2 * i }
 *  usage: cfor (0, 10) { _ => println (rng.gen) }
 *  usage: cfor (0, 10, 2) { i => a(i) = 2 * i }
 *  @see august.nagro.us/scala-for-loop.html
 *  @param start  initialization value (i = start)
 *  @param end    ending value (i < end)
 *  @param inc    increment value (i += c)
 *  @param body   main body of the loop
 */
inline def cfor (start: Int, end: Int, inc: Int = 1) (inline body: Int => Unit): Unit =
    var i = start; while i < end do { body (i); i += inc }
end cfor

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** C/Java style for loop (over a range) provides improved performance.
 *  usage: cfor (0 until 10) { i => a(i) = 2 * i }
 *  usage: cfor (0 until 10 by 2) { i => a(i) = 2 * i }
 *  @see august.nagro.us/scala-for-loop.html
 *  @param r     the loop range
 *  @param body  main body of the loop
 */
inline def cfor (r: Range) (inline body: Int => Unit): Unit =
    var i = r.start; while i < r.end do { body (i); i += r.step }
end cfor

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** C/Java style for loop (0 until n) collecting results in an array provides improved performance.
 *  Faster replacement: for ... yield 
 *  usage: cfor (10) { i => 2 * i }
 *  @param n        the number of elements/results
 *  @param formula  the formula to apply, repeatedly
 */
inline def cfor (n: Int) (inline formula: Int => Double): Array [Double] =
    val a = Array.ofDim [Double] (n)
    cfor (0, n) { i => a(i) = formula (i) }
    a
end cfor

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Summation of the formula applied n times over a range of values.
 *  usage: sumAll (0, 10) { i => 2 * i }
 *  @param start    initialization value (i = start)
 *  @param end      ending value (i < end)
 *  @param formula  the formula to apply, repeatedly
 */
inline def sumAll (start: Int, end: Int)(inline formula: Int => Double): Double =
    var sum_ = 0.0
    var i = start; while i < end do { sum_ += formula (i); i += 1 }
    sum_
end sumAll

inline def Σ (start: Int, end: Int)(inline formula: Int => Double): Double =
    var sum_ = 0.0
    var i = start; while i < end do { sum_ += formula (i); i += 1 }
    sum_
end Σ

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Summation of the formula applied n times over a range of values.
 *  usage: sumAll (0 until 10) { i => 2 * i }
 *  @param r        the loop range
 *  @param formula  the formula to apply, repeatedly
 */
inline def sumAll (r: Range)(inline formula: Int => Double): Double =
    var sum_ = 0.0
    var i = r.start; while i < r.end do { sum_ += formula (i); i += r.step }
    sum_
end sumAll

inline def Σ (r: Range)(inline formula: Int => Double): Double =
    var sum_ = 0.0
    var i = r.start; while i < r.end do { sum_ += formula (i); i += r.step }
    sum_
end Σ 

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Summation of the formula applied n times.
 *  @param n        the number of iterations
 *  @param formula  the formula (taking no arguments) to apply, repeatedly
 */
inline def summation (n: Int)(inline formula: => Double): Double =
    var sum_ = 0.0
    cfor (0, n) { _ => sum_ += formula }
    sum_
end summation

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Convert `Boolean` to `Int` (true -> 1, false -> 0).
 *  @param p  the predicate to convert
 */
inline def is (p: Boolean): Int = if p then 1 else 0
inline def is_ (p: Boolean): Double = if p then 1.0 else 0.0

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Swap two elements in a regular array.
 *  @param i  first element to swap
 *  @param j  second element to swap
 */
inline def swap [T] (a: Array [T], i: Int, j: Int): Unit =
    val t = a(i); a(i) = a(j); a(j) = t
end swap

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Swap two elements in a resizable array.
 *  @param i  first element to swap
 *  @param j  second element to swap
 */
inline def swap [T] (a: ArrayBuffer [T], i: Int, j: Int): Unit =
    val t = a(i); a(i) = a(j); a(j) = t
end swap

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Find the median of three unordered numbers.
 *  @param a1  the first number 
 *  @param a2  the second number 
 *  @param a3  the third number 
 */
@inline def median3 (a1: Double, a2: Double, a3: Double): Double =
    max (min (a1, a2), min (max (a1, a2), a3))
end median3


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cforTest` main function tests the cfor loops that are faster replacements for Scala's
 *  for-comprehension.
 *  Run this code for performance results.  Note, due to JIT, reordering the code may change
 *  the relative performance.  Generally, case 4 "cfor (0, 100000) { ... }" is the fastest.
 *  @see Timer.scala
 *  > runMain scalation.cforTest
 */
@main def cforTest (): Unit =

    import scala.math.sqrt

    val n    = 1                          // number of repeats to get average time (gauge's amplify parameter)
    val skip = false                      // whether to skip timing the first time through code (slow due to JIT)

    val tims = Array.ofDim [Double] (9)
    val sums = Array.ofDim [Double] (9)
    var sum  = 0.0

// Collect results

    tims(0) = gauge (n, skip) {
        sum = 0.0
        var i = 0; while i < 100000 do { sum += sqrt (i); i += 1 } }
    sums(0) = sum

    tims(1) = gauge (n, skip) {
        sum = 0.0
        for i <- 0 until 100000 do sum += sqrt (i) }
    sums(1) = sum

    tims(2) = gauge (n, skip) {
        sum = 0.0
        (0 until 100000).foreach { i => sum += sqrt (i) } }
    sums(2) = sum

    tims(3) = gauge (n, skip) {
        sum = 0.0
        var i = 0; cfor (i < 100000, i += 1) { sum += sqrt (i) } }
    sums(3) = sum

    tims(4) = gauge (n, skip) {
        sum = 0.0
        cfor (0, 100000) { i => sum += sqrt (i) } }
    sums(4) = sum

    tims(5) = gauge (n, skip) {
        sum = 0.0
        cfor (0 until 100000) { i => sum += sqrt (i) } }
    sums(5) = sum

    tims(6) = gauge (n, skip) {
        sum = sumAll (0, 100000) { i => sqrt (i) } }
    sums(6) = sum
    
    tims(7) = gauge (n, skip) {
        sum = Σ (0, 100000) { i => sum + sqrt (i) } }
    sums(7) = sum

    tims(8) = gauge (n, skip) {
        sum = (0 until 100000).foldLeft (0.0) { (sum, i) => sum + sqrt (i) } }
    sums(8) = sum

// Show results

    banner ("var i = 0; while i < 100000 do { sum += sqrt (i); i += 1 }")
    println (s"case 0: sum  = ${sums(0)}")
    println (s"case 0: time = ${tims(0)}")
    
    banner ("for i <- 0 until 100000 do sum += sqrt (i)")
    println (s"case 1: sum  = ${sums(1)}")
    println (s"case 1: time = ${tims(1)}")

    banner ("(0 until 100000).foreach { i => sum += sqrt (i) }")
    println (s"case 2: sum  = ${sums(2)}")
    println (s"case 2: time = ${tims(2)}")

    banner ("var i = 0; cfor (i < 100000, i += 1) { sum += sqrt (i) ")
    println (s"case 3: sum  = ${sums(3)}")
    println (s"case 3: time = ${tims(3)}")

    banner ("cfor (0, 100000) { i => sum += sqrt (i) }")
    println (s"case 4: sum  = ${sums(4)}")
    println (s"case 4: time = ${tims(4)}")

    banner ("cfor (0 until 100000) { i => sum += sqrt (i) }")
    println (s"case 5: sum  = ${sums(5)}")
    println (s"case 5: time = ${tims(5)}")

    banner ("sum = sumAll (0, 100000) { i => sqrt (i) }")
    println (s"case 6: sum  = ${sums(6)}")
    println (s"case 6: time = ${tims(6)}")

    banner ("sum = Σ (0, 100000) { i => sqrt (i) }")
    println (s"case 7: sum  = ${sums(7)}")
    println (s"case 7: time = ${tims(7)}")

    banner ("sum = (0 until 100000).foldLeft (0.0) { (sum, i) => sum + sqrt (i) } }")
    println (s"case 8: sum  = ${sums(8)}")
    println (s"case 8: time = ${tims(8)}")

end cforTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cforTest2` main function test the cfor loops that are faster replacements for Scala's
 *  for-yield.
 *  Run this code for performance results.  Note, due to JIT, reordering the code may change
 *  the relative performance.  Generally, case 4 "cfor (0, 100000) { ... }" is the fastest.
 *  @see Timer.scala
 *  Note, uncomment to compile and run.  Commented as there is a reverse dependency on mathstat (VectorD).
 *  > runMain scalation.cforTest2
 *
@main def cforTest2 (): Unit =

    import scala.math.sqrt
    import scalation.mathstat.VectorD

    val n    = 1                          // number of repeats to get average time (gauge's amplify parameter)
    val skip = false                      // whether to skip timing the first time through code (slow due to JIT)

    val tims = Array.ofDim [Double] (6)
    val sums = Array.ofDim [Double] (6)
    var x: VectorD = null

// Collect results

    tims(0) = gauge (n, skip) {
        x = new VectorD (10000); var i = 0; while i < 10000 do { x(i) = sqrt (i); i += 1 } }
    sums(0) = x.sum

    tims(1) = gauge (n, skip) {
        x = VectorD ((0 until 10000).map { i => sqrt (i) }) }
    sums(1) = x.sum

    tims(2) = gauge (n, skip) {
        x = VectorD (for i <- 0 until 10000 yield sqrt (i)) }
    sums(2) = x.sum

    tims(3) = gauge (n, skip) {
        val a = Array.ofDim [Double] (10000); cfor (0, 10000) { i => a(i) = sqrt (i) }; x = new VectorD (10000, a) }
    sums(3) = x.sum

    tims(4) = gauge (n, skip) {
        x = new VectorD (10000); cfor (0, 10000) { i => x(i) = sqrt (i) } }
    sums(4) = x.sum

    tims(5) = gauge (n, skip) {
        x = new VectorD (10000, cfor (10000) { i => sqrt (i) }) }
    sums(5) = x.sum

// Show results

    banner ("x = new VectorD (10000); var i = 0; while i < 10000 do { x(i) = sqrt (i); i += 1 } }")
    println (s"case 0: sum  = ${sums(0)}")
    println (s"case 0: time = ${tims(0)}")

    banner ("x = VectorD ((0 until 10000).map { i => sqrt (i) }) }")
    println (s"case 1: sum  = ${sums(1)}")
    println (s"case 1: time = ${tims(1)}")

    banner ("x = VectorD (for i <- 0 until 10000 yield sqrt (i)) }")
    println (s"case 2: sum  = ${sums(2)}")
    println (s"case 2: time = ${tims(2)}")

    banner ("val a = Array.ofDim [Double] (10000); cfor (0, 10000) { i => a(i) = sqrt (i) }; x = new VectorD (10000, a) }")
    println (s"case 3: sum  = ${sums(3)}")
    println (s"case 3: time = ${tims(3)}")

    banner ("x = new VectorD (10000); cfor (0, 10000) { i => x(i) = sqrt (i) } }")
    println (s"case 4: sum  = ${sums(4)}")
    println (s"case 4: time = ${tims(4)}")

    banner ("x = new VectorD (10000, cfor (10000) { i => sqrt (i) })")
    println (s"case 5: sum  = ${sums(5)}")
    println (s"case 5: time = ${tims(5)}")

end cforTest2
 */

