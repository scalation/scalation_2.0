
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Sun Aug  7 20:40:29 EDT 2016
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Random String/Word Generators
 */

package scalation
package random

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomStr` class generates random strings.
 *  @param lRange  the range of string lengths to generate
 *  @param cRange  the range of characters to generate
 *  @param stream  the random number stream
 */
case class RandomStr (lRange: Range = 4 to 7, cRange: Range = 97 to 122, stream: Int = 0)
     extends Variate (stream):

    private val lrng = Randi (lRange.start, lRange.end, stream)       // random integer generator
    private val crng = Randi (cRange.start, cRange.end, stream)       // random integer generator

    val mean = Double.NaN

    def pf (s: Double): Double = throw new UnsupportedOperationException ("'pf' not implemented")

    def gen: Double = throw new UnsupportedOperationException ("'gen' not implemented, use 'sgen' instead")

    def gen1 (z: Double): Double = throw new UnsupportedOperationException ("'gen1' not implemented, use 'sgen1' instead")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a random string.
     */
    override def sgen: String =
        val sb = new StringBuilder ()
        cfor (0, lrng.igen) { _ => sb.append (crng.igen.toChar) }
        sb.toString
    end sgen

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random string for the particular distribution.
     *  It is only valid for discrete random variates.
     *  This version allows one parameter.
     *  @param z  the limit parameter as a range
     */
    def sgen1 (z: Range): String =
        val lrng = Randi (z.start, z.end, stream)       // random integer generator
        val sb = new StringBuilder ()
        cfor (0, lrng.igen) { _ => sb.append (crng.igen.toChar) }
        sb.toString
    end sgen1

end RandomStr


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomWord` class generates random words from a predetermined set.
 *  @param nWords  the numbers of words to predetermine.
 *  @param lRange  the range of string lengths to generate
 *  @param cRange  the range of characters to generate
 *  @param stream  the random number stream
 */
case class RandomWord (nWords: Int = 10, lRange: Range = 4 to 6, cRange: Range = 97 to 122, stream: Int = 0)
     extends Variate (stream):

    private val rig   = Randi0 (nWords-1, stream)                       // random integer generator
    private val rsg   = RandomStr (lRange, cRange, stream)              // random string generator
    private val words = Array.ofDim [String] (nWords)

    for i <- 0 until nWords do
        while 
            val w = rsg.sgen; words(i) = w
            words contains w
        do ()
    end for

    val mean = Double.NaN

    def pf (s: Double): Double = throw new UnsupportedOperationException ("'pf' not implemented")

    def gen: Double = throw new UnsupportedOperationException ("'gen' not implemented, use 'sgen' instead")

    def gen1 (z: Double): Double = throw new UnsupportedOperationException ("'gen1' not implemented, use 'sgen1' instead")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a random string.
     */
    override def sgen: String = words (rig.igen)

end RandomWord


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomStrTest` main function is used to test the Random Variate String (RVS)
 *  generator from the `RandomStr` class.
 *  > runMain scalation.random.randomStrTest
 */
@main def randomStrTest (): Unit =

     val rsg = RandomStr ()                            // variate string generator
     var rs: String = null                             // variate string

     println ("Test: RandomStr random string generation ------------------------")
     cfor (0, 30) { _ => rs = rsg.sgen; println (rs) }

end randomStrTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `randomWordTest` main function is used to test the Random Variate Word (RVW)
 *  generator from the `RandomWord` class.
 *  > runMain scalation.random.randomWordTest
 */
@main def randomWordTest (): Unit =

     val rsg = RandomWord ()                           // variate word generator
     var rs: String = null                             // variate word

     println ("Test: RandomWord random string generation ------------------------")
     cfor (0, 30) { _ => rs = rsg.sgen; println (rs) }

end randomWordTest

