
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Praveen Rangavajhula
 *  @version 2.0
 *  @date    Tue Nov 11 11:18:42 EST 2025
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Autograd: Lightweight Testing Utilities
 *
 *  Provides simple tools for recording and summarizing unit tests.  Includes
 *  status enumeration, result containers, ANSI color support, and a reporting
 *  utility for structured test output.
 */

package scalation
package modeling
package autograd

import scala.collection.mutable.ArrayBuffer
import scala.util.control.NonFatal

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Enumeration representing the status of a test:
 *      - Passed
 *      - Failed
 */
enum Status:
    case Passed, Failed
end Status


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Result container for a single test execution.
 *  @param name   the name of the test
 *  @param status the status of the test (Passed or Failed)
 *  @param ms     the execution time in milliseconds
 *  @param note   optional note or error message (default empty)
 */
final case class TestResult (name: String, status: Status, ms: Long, note: String = "")

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** ANSI color codes for colored console output.
 */
object ConsoleColor:
    val RESET  = "\u001B[0m"
    val RED    = "\u001B[31m"
    val GREEN  = "\u001B[38;5;34m"
    val YELLOW = "\u001B[33m"
end ConsoleColor


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** A test report utility for recording and summarizing test results.
 *  Stores a collection of `TestResult` objects and provides support for
 *  timing tests, capturing failures, and printing formatted summary reports.
 */
final class TestReport:

    private val buf = ArrayBuffer.empty [TestResult]
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Record and execute a test, capturing its execution time and status.
     *  @param name the name of the test
     *  @param body the test code to execute (returns true on success)
     *  @return     true if the test passed, false otherwise
     */
    inline def record (name: String)(inline body: => Boolean): Boolean =
        val t0 = System.nanoTime()
        var note = ""
        val ok =
            try body
            catch
                case NonFatal(e) =>
                    note = s"${e.getClass.getSimpleName}: ${e.getMessage}"
                    false
        val dtMs = ((System.nanoTime() - t0) / 1e6).toLong
        val status = if ok then Status.Passed else Status.Failed
        buf += TestResult(name, status, dtMs, note)
        ok
    end record
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Print a formatted summary of all recorded tests.
     *  @param title         title of the test report (default "Test Report")
     *  @param onlyFailures  whether to print only failed tests (default false)
     */
    def summary (title: String = "Test Report", onlyFailures: Boolean = false): Unit =
        println (s"\n================== $title ==================")
        val (passed, failed) = buf.partition(_.status == Status.Passed)
        val rows = if onlyFailures then failed else buf
        
        if rows.isEmpty then
            if onlyFailures && buf.nonEmpty then
                println ("All tests passed ✅  (no failures to show)")
            else
                println ("No tests recorded.")
        else
            rows.foreach { r =>
                import ConsoleColor.*
                val (tag, color) =
                    if r.status == Status.Passed then ("✅ PASSED", GREEN)
                    else ("❌ FAILED", RED)
                
                val note = if r.note.nonEmpty then s"  [$YELLOW${r.note}$RESET]" else ""
                
                println (f"$color${r.name}%-40s $tag$RESET  (${r.ms}ms)$note")
            }
        
        println ("-------------------------------------------------")
        val total = buf.size
        val pass  = passed.size
        val fail  = failed.size
        val rate  = if total == 0 then 0.0 else pass * 100.0 / total
        println (f"Total: $total  Passed: $pass  Failed: $fail  |  Pass rate: $rate%.1f%%")
        println ("=================================================")
    end summary
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check whether any recorded tests have failed.
     *  @return true if any test has status Failed, false otherwise
     */
    def hasFailures: Boolean = buf.exists (_.status == Status.Failed)
    
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Clear all recorded test results.
     */
    def reset (): Unit = buf.clear ()

end TestReport


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Companion object for creating TestReport instances.
 */
object TestReport:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a new TestReport instance.
     *  @return a new TestReport
     */
    def apply (): TestReport = new TestReport

end TestReport

