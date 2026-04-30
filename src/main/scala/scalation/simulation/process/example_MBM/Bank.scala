
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Fri Sep 17 23:52:26 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Example Model: Bank for Process Simulation using 'jump'
 */

package scalation
package simulation
package process
package example_MBM                                     // Method of Batch Means (MBM)

import scalation.mathstat.VectorD
import scalation.random.{Exponential, Sharp}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `runBank` function is used to launch the `BankModel` class.
 *  > runMain scalation.simulation.process.example_MBM.runBank
 */
@main def runBank (): Unit = new BankModel ()


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `testBank` function is used to test the autocorrelation between the batches
 *  in the `BankModel` class.
 *  Obtain the batchMeans vector from statV by calling reportV (true) in `Model_MBM`
 *  > runMain scalation.simulation.process.example_MBM.testCorrBank
 */
@main def testCorrBank (): Unit =
    val batchMeans = VectorD (
        41.0430, 34.1852, 27.8945, 20.7674, 29.5358, 28.1378, 39.3704, 22.8593, 38.9610, 35.9986,
        30.0565, 29.4135, 37.4061, 31.1298, 26.1880, 23.1076, 28.7468, 21.2382, 29.1787, 27.7081,
        54.9413, 16.9537, 26.2421, 27.1526, 39.5139, 18.0606, 34.2399, 27.5326, 21.1054, 32.2369,
        18.5720, 56.1361, 37.0753, 29.6010, 50.7372, 29.5670, 52.8855, 21.3998, 39.4527, 31.7850)
    println (s"batchMeans.acorr () = ${batchMeans.acorr ()}")
end testCorrBank


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `BankModel` class defines a simple process-interaction model of a bank
 *  where service is provided by one or more tellers.
 *  This version reduces the impact of transports on simulation by
 *  (1) using the transport's 'jump' method rather than its 'move' method and
 *  (2) reducing the time through the transport by an order of magnitude.
 *  It also has animation turned off, by default.
 *  @caveat:  must add 'from' and 'to' components before transport!!
 *  @param name       the name of the simulation model
 *  @param nBatch     the number of batches to run
 *  @param sizeB      the size of each batch
 *  @param animating  whether to animate the model
 *  @param aniRatio   the ratio of simulation speed vs. animation speed
 *  @param stream     the base random number stream (0 to 999)
 */
class BankModel (name: String = "Bank", nBatch: Int = 50, sizeB: Int = 1000,
                 animating: Boolean = false, aniRatio: Double = 8.0, stream: Int = 0)
      extends Model_MBM (name, nBatch, sizeB, animating, aniRatio):

    val nStop = nBatch * sizeB                          // the number arrivals before stopping the Source

    //--------------------------------------------------
    // Initialize Model Constants

    val lambda   = 12.0  // 6.0                         // customer arrival rate (per hour)
    val mu       = 7.5                                  // customer service rate (per hour)
    val nTellers = 2     // 1                           // the number of bank tellers (servers)

    //--------------------------------------------------
    // Create Random Variables (RVs)

    val iArrivalRV = Exponential (HOUR / lambda, stream)
    val serviceRV  = Exponential (HOUR / mu, stream + 1)
    val moveRV     = Sharp (SECOND, stream + 2)

    //--------------------------------------------------
    // Create Model Components

    val entry     = Source ("entry", this, () => Customer (), 0, nStop, iArrivalRV, (100, 290))
    val tellerQ   = WaitQueue ("tellerQ", (330, 290))
    val teller    = Resource ("teller", tellerQ, nTellers, serviceRV, (350, 285))
    val door      = Sink ("door", (600, 290))
    val toTellerQ = Transport ("toTellerQ", entry, tellerQ, moveRV)
    val toDoor    = Transport ("toDoor", teller, door, moveRV)

    addComponent (entry, tellerQ, teller, door, toTellerQ, toDoor)

    //--------------------------------------------------
    // Specify Scripts for each Type of Simulation Actor

    case class Customer () extends SimActor ("c", this):

        override def act (): Unit =
            toTellerQ.jump ()                           // jump (one step) to teller queue
            if teller.busy then tellerQ.waitIn ()       // wait when all tellers are busy
            else tellerQ.noWait ()                      // record no waiting time
            teller.utilize ()                           // process transaction with teller
            teller.release ()                           // release the teller
            toDoor.jump ()                              // jump to the door
            door.leave ()                               // exit the bank
        end act

    end Customer

    simulate ()
    waitFinished ()
    Model.shutdown ()

end BankModel

