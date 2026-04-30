
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Sep 16 16:12:08 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Random Variate Set (RVS) Generators
 */

package scalation
package random

import scala.collection.mutable.Set

import RandomSeeds.N_STREAMS

//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VariateSet` abstract class serves as a base class for all the Random
 *  Variate Set (RVS) generators.  They use one of the Random Number Generators
 *  (RNG's) from Random.scala to generate numbers following their particular
 *  multivariate distribution.
 *  @param stream  the random number stream (0 until N_STREAMS)
 */
abstract class VariateSet (stream: Int = 0):

    /** Flaw function
     */
    protected val flaw = flawf ("VariateSet")

    /** Random number stream selected by the stream number (can't be betond last stream)
     */
    protected val r = Random (stream % N_STREAMS)

    /** Indicates whether the distribution is discrete (default) or continuous
     */
    protected var _discrete = true

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the distribution is discrete or continuous.
     */
    def discrete: Boolean = _discrete

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean for the particular distribution.
     */
    def mean: Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the probability function (pf):
     *  The probability density function (pdf) for continuous RVV's or
     *  the probability mass function (pmf) for discrete RVV's.
     *  @param z  the mass point/set whose probability is sought
     */
    def pf (s: Set [Int]): Double

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random double set for the particular distribution.
     */
    def gen: Set [Double]

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine the next random integer set for the particular distribution.
     *  It is only valid for discrete random variates.
     */
    def igen: Set [Int]

end VariateSet


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomSet` class generates a random set/subset of integers.
 *  @param count   the size of the set (number of integer elements)
 *  @param max     generate integers in the range 0 (inclusive) to max (inclusive)
 *  @param skip    skip this number, i.e, do not use it
 *  @param stream  the random number stream
 */
case class RandomSet (count: Int = 10, max: Int = 20, skip: Int = -1, stream: Int = 0)
     extends VariateSet (stream):

    if count > max then
        flaw ("init", "requires count <= max")
        throw new IllegalArgumentException ("RandomSet: count shouldn't be larger than max")

    private val rng = Randi0 (max, stream)              // random integer generator

    def mean: Double =  max / 2.0 

    def pf (s: Set [Int]): Double = throw new UnsupportedOperationException ("'pf' not implemented")

    def gen: Set [Double] = throw new UnsupportedOperationException ("'gen' not implemented, use 'igen' instead")

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a random set of unique integers 'r' in the range '0 to max'.
     *  The size of the set is given by 'count'.
     */
    def igen: Set [Int] =
        val r = Set [Int] ()
        var e = 0
        for i <- 0 until count do
            while { e = rng.igen; e == skip || i > 0 && (r contains e) } do ()
            r += e
        end for
        r
    end igen

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a random set of unique integers 'r' in the range '0 to mx'.
     *  @param n    the size of the resultant random subset r
     *  @param mx   generate integers in the range 0 (inclusive) to 'mx' (inclusive)
     *  @param skp  skip this number, i.e, do not use it
     */
    def igen (n: Int, mx: Int, skp: Int = -1): Set [Int] =
        val r = Set [Int] ()
        var e = 0
        for i <- 0 until n do
            while { e = rng.igen1 (mx); e == skp || i > 0 && (r contains e) } do ()
            r += e
        end for
        r
    end igen

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Generate a set of unique integers 'r' that is a random subset of set 's'.
     *  @param s  the given set
     *  @param n  the size of the resultant random subset r
     */
    def igen (s: Set [Int], n: Int): Set [Int] =
        val sa = s.toArray
        val mx = sa.length - 1
        val r  = Set [Int] ()
        var e  = 0
        for i <- 0 until n do
            while { e = sa(rng.igen1 (mx)); i > 0 && (r contains e) } do ()
            r += e
        end for
        r
    end igen

end RandomSet


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomSetS` class generates a random set/subset of strings.
 *  @param count   the size of the set (number of strings)
 *  @param stream  the random number stream
 */
case class RandomSetS (count: Int = 10, stream: Int = 0)
     extends VariateSet (stream) :

    _discrete = true

    private val rng = RandomStr (stream = stream)       // random string generator

    def mean: Double = throw new UnsupportedOperationException ("'mean' not implemented")

    def pf (s: Set [Int]): Double = throw new UnsupportedOperationException ("'pf' not implemented")

    def gen: Set [Double] = sgen.map (_.toDouble)

    def igen: Set [Int] = sgen.map (_.toInt)

    def sgen: Set [String] =
        val y   = Set [String] ()
        var str = ""
        for i <- 0 until count do
            while { str = rng.sgen; i > 0 && (y contains str) } do ()
            y += str
        end for
        y
    end sgen

end RandomSetS


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RandomSetW` class generates a random set/subset of words.
 *  @param count   the size of the set (number of words)
 *  @param nWords  the numbers of words to predetermine.
 *  @param lRange  the range of string lengths to generate
 *  @param cRange  the range of characters to generate
 *  @param stream  the random number stream
 */
case class RandomSetW (count: Int = 10, nWords: Int = 20, lRange: Range = 4 to 6,
                       cRange: Range = 97 to 122,stream: Int = 0)
     extends VariateSet (stream):

    _discrete = true

    private val rng = RandomWord (nWords, lRange, cRange, stream)       // random word generator

    def mean: Double = throw new UnsupportedOperationException ("'mean' not implemented")

    def pf (s: Set [Int]): Double = throw new UnsupportedOperationException ("'pf' not implemented")

    def gen: Set [Double] = sgen.map (_.toDouble)

    def igen: Set [Int] = sgen.map (_.toInt)

    def sgen: Set [String] =
        val y   = Set [String] ()
        var str = ""
        for i <- 0 until count do
            while { str = rng.sgen; i > 0 && (y contains str) } do ()
            y += str
        end for
        y
    end sgen

end RandomSetW


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `variateSetTest` object is used to test the Random Variate Set (RVS)
 *  generators from the classes derived from `VariateSet`.
 *  > runMain scalation.random.variateSetTest
 */
@main def variateSetTest (): Unit =

     val rsg = RandomSet (10)                           // variate set generator
     var rs:  Set [Int]  = null                         // variate set

     println ("Test: RandomSet random set generation ------------------------")
     println ("mean = " + rsg.mean)                     // random set generator
     cfor (0, 30) { _ => rs = rsg.igen; println (rs) }

     println ("Test: RandomSet random subset generation ---------------------")
     println ("mean = " + rsg.mean)                     // random set generator
     cfor (0, 30) { _ => println (rsg.igen (rs, 5)) }

end variateSetTest


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `variateSetTest2` object is used to test the Random Variate Set (RVS)
 *  generators from the classes derived from `VariateSet`.
 *  > runMain scalation.random.variateSetTest2
 */
@main def variateSetTest2 (): Unit =

     val rsg = RandomSetS (10)                          // variate set generator
     var rs:  Set [String]  = null                      // variate set

     println ("Test: RandomSetS random set generation ------------------------")
     cfor (0, 30) { _ => rs = rsg.sgen; println (rs) }

end variateSetTest2


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `variateSetTest3` object is used to test the Random Variate Set (RVS)
 *  generators from the classes derived from `VariateSet`.
 *  > runMain scalation.random.variateSetTest3
 */
@main def variateSetTest3 (): Unit =

     val rsg = RandomSetW (10)                          // variate set generator
     var rs:  Set [String]  = null                      // variate set

     println ("Test: RandomSetW random set generation ------------------------")
     cfor (0, 30) { _ => rs = rsg.sgen; println (rs) }

end variateSetTest3

