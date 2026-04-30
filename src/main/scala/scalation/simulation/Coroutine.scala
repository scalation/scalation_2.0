
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller, Michael Cotterell, Yulong Wang
 *  @version 2.0
 *  @date    Sat Mar 21 20:34:23 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Coroutine implementation options: (1) Java Threads, (2) Java Virtual Threads
 */

package scalation
package simulation

import java.util.concurrent.{ExecutorService, Executors, Semaphore, ThreadPoolExecutor}
//import java.util.concurrent.{Executors, ExecutorService, Future, Semaphore, ThreadPerTaskExecutor}

import scala.collection.mutable.ArrayBuffer
import scala.language.postfixOps

import scalation.{debugf, flawf}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Coroutine` class supports (one-at-a-time) quasi-concurrent programming.
 *  A coroutine runs/acts until it yields control from 'this' to 'that' coroutine.
 *  When resumed, a coroutines continues its execution where it left off.
 *  @param label  the label for the class of coroutines to be created.
 */
abstract class Coroutine (label: String = "cor")
    extends Runnable:

    import Coroutine._

    private val debug   = debugf ("Coroutine", true)           // debug function
    private val _sema   = new Semaphore (0)                    // waiting semaphore
    private var started = false                                // whether this coroutine has started

    nCreated += 1
    protected val cor_id = label + "." + nCreated
    debug ("init", s"===> $cor_id waits to be STARTed")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the Coroutine counts.
     */
    def counts: (Int, Int, Int) = (nCreated, nStarted, nTerminated)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Thread's 'run' method delegates to the 'act' method.  Upon interruption
     *  the 'act' method is run again from the beginning.
     */
    def run (): Unit =
        nStarted += 1
        try
            act ()
        catch case _ : InterruptedException =>
            debug ("run", s"INTERRUPTED coroutine $cor_id")
        end try
        nTerminated +=1
//      pool.remove (this)
        debug ("run", s"TERMINATE coroutine $cor_id")
    end run

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Actor model features the 'act' method, even though threads are used.
     *  This abstract method must be implemented in application models.
     */
    def act (): Unit

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** For multiple sources and more than one replication otherwise, after one
     *  replication, all the sources are already started the first source from agenda
     *  will be invoked by the yield to null but the second resource from the won't
     *  so needs to be reset
     */
    def resetStart (): Unit = started = false

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Yield control from 'this' to 'that' coroutine.
     *  @param that  the other coroutine to yield control to
     *  @param quit  whether 'this' coroutine is to terminate (true)
     *                                  or wait to be resumed (false)
     */
    def yyield (that: Coroutine, quit: Boolean = false): Unit =
        if that != null then
            if that.started then
                debug ("yyield", s"$cor_id RESUMEs that coroutine ${that.cor_id}")
                that.resume ()
            else
                debug ("yyield", s"$cor_id STARTs that new coroutine ${that.cor_id}")
                that.start ()
        end if

        if quit then
            debug ("yyield", s"$cor_id TERMINATEs")
            return
        else
            debug ("yyield", s"$cor_id WAITs on semaphore")
            _sema.acquire ()                                   // wait until resumed
    end yyield

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Start this coroutine, i.e., invoke its 'run' -> 'act' method.
     */
    def start (): Unit =
        started = true

//      if threadArr == null then
//          flaw ("start", "the coroutine system must be started using Coroutine.startup; expect undefined behavior.")

        if useVirtualThread then
            val vt = Thread.ofVirtual ().unstarted (this)
            threadArr += vt
            vt.start ()
        else
            pool.submit (this)

    end start

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Check this thread to see if it is virtual.
     */
    def isVirtual: Boolean  = Thread.currentThread ().isVirtual

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Interrupt this waiting coroutine.
     */
    def interrupt (): Unit = Thread.currentThread ().interrupt ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Resume this coroutine.
     */
    private def resume (): Unit = _sema.release ()

end Coroutine


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Coroutine` companion object provides functions to start-up and shutdown
 *  the coroutine system as well as counters for the `Coroutine` class.
 */
object Coroutine:

    private [simulation]  var useVirtualThread = true          // whether to use regular or virtual threads

    private val debug = debugf ("Coroutine", false)            // debug function
    private val flaw  = flawf ("Coroutine")                    // flaw function
    private var pool: ExecutorService = null                   // thread pool
    private val CORE_THREADS = 0                               // number of core threads
    private val threadArr    = ArrayBuffer [Thread] ()         // array buffer to hold threads
    private var nCreated     = 0                               // number of Coroutines created
    private var nStarted     = 0                               // number of Coroutines started
    private var nTerminated  = 0                               // number of Coroutines terminated
//  var scope = new StructuredTaskScope[Unit]()

    if ! useVirtualThread then startup ()                      // automatic startup at program start

//  sys.addShutdownHook ({                                     // automatic shutdown at program end
//      pool.shutdown ()
//      pool.shutdownNow ()
//  })

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the thread pool executor.
     */
    private def threadPoolExecutor = pool.asInstanceOf [ThreadPoolExecutor]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Start-up the coroutine system.  This function can also set the core
     *  number of threads for the internal cached thread pool.
     *  @param nCoreThreads  the new core size
     */
    private def startup (nCoreThreads: Int = CORE_THREADS): Unit =
        if pool == null then
            pool = Executors.newCachedThreadPool ()
            if nCoreThreads != CORE_THREADS then threadPoolExecutor.setCorePoolSize (nCoreThreads)
        else
            flaw ("startup", "coroutine system is already started")
    end startup

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the number of threads that have terminated.
     */
    inline def numTerminted: Int = nTerminated

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Wait for threads in threadArr to finish and then clear the array buffer,
     *  otherwise, there will be a memory leak
     */
    def waitThreadFinish (): Unit =
        if threadArr != null then
            for thread <- threadArr do thread.join ()
            threadArr.clear ()
    end waitThreadFinish

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shutdown the coroutine thread pool and return the largest number of threads
     *  that have ever simultaneously been in the pool.  Must be called at program
     *  end or application will hang.
     */
    def shutdown (): Int =
        var lps = 0                                            // largest pool size
        debug ("Coroutine", "shutdown")

        if threadArr != null && useVirtualThread then
            for thread <- threadArr do thread.join ()
            lps = -1                                           // ThreadPerTaskExecutor does not track this

        if pool != null && !useVirtualThread then
            pool.shutdown ()                                   // prevent new submissions to pool
//          pool.awaitTermination (Int.MaxValue, TimeUnit.NANOSECONDS)
            pool.shutdownNow ()                                // interrupt all threads remaining in pool
            lps  = threadPoolExecutor.getLargestPoolSize
            pool = null
//      else flaw ("shutdown", "coroutine system is already shutdown")
        lps
    end shutdown

end Coroutine


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `CoroutineTest` object is used to test the `Coroutine` class.
 *  Should print:
 *   start coroutines
 *   Cor1: phase 1
 *   inside Cor1 the id is 22  isVirtual true
 *   VirtualThread[#22]/runnable@ForkJoinPool-1-worker-1
 *   Cor2: phase 1
 *   inside Cor2 the id is 26  isVirtual true
 *   VirtualThread[#26]/runnable@ForkJoinPool-1-worker-3
 *   Cor1: phase 2
 *   Cor2: phase 2
 *  > runMain scalation.simulation.runCoroutineTest
 */
object CoroutineTest:                      // requires object since it needs forward reference

    class Cor1 extends Coroutine:
        override def act (): Unit =
            println ("Cor1: phase 1")
            println (s"inside Cor1 the id is ${Thread.currentThread ().threadId ()} " +
                    s"isVirtual ${Thread.currentThread ().isVirtual}")
            println (Thread.currentThread ().toString ())
            yyield (cor2)
            println ("Cor1: phase 2")
            yyield (cor2, true)
        end act
    end Cor1

    class Cor2 extends Coroutine:
        override def act (): Unit =
            println ("Cor2: phase 1")
            println (s"inside Cor2 the id is ${Thread.currentThread ().threadId ()} " +
                    s"isVirtual ${Thread.currentThread ().isVirtual}")
            println (Thread.currentThread ().toString ())
            yyield (cor1)
            println ("Cor2: phase 2")
            yyield (null, true)
        end act
    end Cor2

    val cor1 = Cor1 ()                     // initialization, hold the reference to the object/instance
    val cor2 = Cor2 ()

    println ("start coroutines")

    @main def runCoroutineTest (): Unit =
        println ("runCoroutineTest: start test from main func")
        cor1.start ()                                            // start the coroutine cor1
        if Coroutine.useVirtualThread then Coroutine.shutdown ()
//      if ! Coroutine.useVirtualThread then Coroutine.shutdown ()
    end runCoroutineTest

end CoroutineTest

