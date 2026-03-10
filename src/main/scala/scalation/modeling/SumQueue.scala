
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sat Aug 31 15:18:54 EDT 2019
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Model Support: Queue for Moving Averages
 */

package scalation
package modeling

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SumQueue` class retains the last q elements as well as the running total
 *  in sum and number of elements in size_, making it efficient to compute
 *  moving averages.
 *  @param q  the number of elements to retain in the queue
 */
class SumQueue (q: Int = 5):

    private val queue = new CircularQueue [Double] (q)       // circular queue to hold elements
    private var size_ = 0                                    // number of elements in the queue
    private var sum   = 0.0                                  // running total (sum)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size (number of elements) in the queue.
     */
    def size: Int = size_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the mean of the elements in the queue.
     */
    def mean: Double = sum / size_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Enqueue the next element y in the queue, removing the first element when
     *  the queue is full.
     *  @param y  the next element to place in the queue
     */
    def += (y: Double): Unit =
        if size_ < q then size_ += 1 else sum -= queue.dequeue ()
        sum   += y
        queue += y
    end +=

end SumQueue


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `SumQueue` class retains the last q elements as well as the running total
 *  in sum and number of elements in size_, making it efficient to compute
 *  moving averages.
 *  @param q  the number of elements to retain in the queue
 */
class SumSqQueue (q: Int = 5):

    private val queue = new CircularQueue [Double] (q)       // circular queue to hold elements
    private var size_ = 0                                    // number of elements in the queue
    private var sum   = 0.0                                  // running total (sum)
    private var sumSq = 0.0                                  // running total (sum of squares)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the size (number of elements) in the queue.
     */
    def size: Int = size_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the mean of the elements in the queue.
     */
    def mean: Double = sum / size_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the variance of the elements in the queue.
     */
    def variance: Double = (sumSq - sum ~^ 2 / q) / (q-1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Enqueue the next element y in the queue, removing the first element when
     *  the queue is full.
     *  @param y  the next element to place in the queue
     */
    def += (y: Double): Unit =
        if size_ < q then size_ += 1
        else
            val yy = queue.dequeue ()
            sum   -= yy
            sumSq -= yy ~^ 2
        sum   += y
        sumSq += y ~^ 2
        queue += y
    end +=

end SumSqQueue


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sumQueueTest` main function is used to test the `SumQueue` class.
 *  > runMain scalation.modeling.sumQueueTest
 */
@main def sumQueueTest (): Unit =

    val sumq = new SumQueue ()
    for i <- 1 to 20 do
        val y = 2.0 * i
        sumq += y
        println (s"$i: mean = ${sumq.mean}")
    end for
       
end sumQueueTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sumQueueTest2` main function is used to test the `SumSqQueue` class.
 *  > runMain scalation.modeling.sumQueueTest2
 */
@main def sumQueueTest2 (): Unit =

    val sumq = new SumSqQueue ()
    for i <- 1 to 20 do
        val y = 2.0 * i
        sumq += y
        println (s"$i: mean = ${sumq.mean}")
        println (s"$i: variance = ${sumq.variance}")
    end for
       
end sumQueueTest2

