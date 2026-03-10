
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Aug 25 13:58:23 EDT 2011
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Monte Carlo Simulation of Poker Hands
 */

package scalation
package simulation
package monte_carlo

import scala.collection.mutable.Map
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat.VectorD
import scalation.random.Randi

import Cards._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Cards` class is used to represent a deck of playing cards.
 */
class Cards:

    private val NUM_CARDS = 52                               // number of cards in deck
    private val card      = Array.range (0, NUM_CARDS)       // the cards themselves
    private val rvg       = Randi (0, NUM_CARDS - 1)         // random variate generator
    private var top       = 0                                // index of top card

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Draw the top card from the deck and return it.  Return -1 if no cards are
     *  left in the deck.
     */
    def draw (): Int = if top == NUM_CARDS then -1
                       else { val c = card(top); top += 1; c }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Shuffle the deck of cards.
     */
    def shuffle (): Unit =
        for i <- card.indices do swap (card, i, rvg.igen)
        top = 0
    end shuffle

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the card deck to a string.
     */
    override def toString: String = "Cards ( " + stringOf (for c <- card yield value (c)) + " )"

end Cards


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `Cards` companion object provides function that operate on cards and hands.
 *  @see https://rosettacode.org/wiki/Poker_hand_analyser
 */
object Cards:

    val suit = Array ('C', 'D', 'H', 'S')                    // suits: Clubs, Diamonds, Hearts, Spades

    val htype = Array ("high-card", "one-pair", "two-pair", "3-of-a-kind", "straight",
                       "flush", "full-house", "4-of-a-kind", "straight-flush")

    given intOrd: Ordering [Int] = new Ordering [Int]:
        def compare (s: Int, t: Int) = s compare t

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert the integer card number c (0 to 51) to the
     *  face value: 1(A), 2, 3, ..., 10, 11(J), 12(Q), 13(K) and
     *  suit: 0(C), 1(D), 2(H), 3(S).
     *  @param c  the card number
     */
    def value (c : Int): (Int, Char) = (c % 13 + 1, suit (c / 13))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Turn the hand into a map of face-value -> frequency in hand, e.g., 3 Kings.
     *  @param hand  the card hand
     */
    def handMap (hand: IndexedSeq [Int]): Map [Int, Int] =
        val hmap = Map [Int, Int] ()
        for c <- hand do { val k = value(c)._1; hmap += k -> (hmap.getOrElse (k, 0) + 1) }
        hmap
    end handMap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the hand is a flush (all cards have the same suit).
     *  @param hand  the card hand
     */
    def isFlush (hand: IndexedSeq [Int]): Boolean =
        val suit = value (hand(0))._2
        hand.forall (value (_)._2 == suit)
    end isFlush

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether the hand is a straight (consecutive cards).
     *  An Ace (1) can be on either end, e.g., (A, 2, 3, 4, 5) or (10, J, Q, K, A).
     *  @param hand  the card hand
     */
    def isStraight (hmap: Map [Int, Int]): Boolean =
        val keys = hmap.keys.toIndexedSeq
        keys.max - keys.min == 4 || keys.toSet == Set (10, 11, 12, 13, 1)     // 10, J, Q, K, A
    end isStraight

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Classify the Poker Hand from highest to lowest:
     *  @param hand  the card hand
     */
    def classify (hand: IndexedSeq  [Int]): Int =
        val flush = isFlush (hand)
        val hmap  = handMap (hand)
//      val freq  = hmap.values.toIndexedSeq.sorted ((x, y) => y.compare (x))
        val freq  = hmap.values.toIndexedSeq.sorted (using intOrd)
     
        freq(0) match
        case 4 => 7
        case 3 => if freq(1) == 2 then 6
                  else if flush then 5
                  else 3
        case 2 => if flush then 5 
                  else if freq(1) == 2 then 2 
                  else 1
        case 1 => if isStraight (hmap) then if flush then 8 else 4
                  else if flush then 5
                  else 0
    end classify

end Cards


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cardsTest` main function is used to test the `Cards` class.
 *  > runMain scalation.simulation.monte_carlo.cardsTest
 */
@main def cardsTest (): Unit =

    val deck = new Cards ()
    println ("\nOrdered deck of cards:")
    println (deck)
    deck.shuffle ()
    println ("\nShuffled deck of cards:")
    println (deck)

    val hand  = for _ <- 1 to 5 yield deck.draw ()
    val cards = hand.map (value (_))
    val hmap  = handMap (hand)
    println ("\n hand  = " + hand)
    println ("\n cards = " + cards)
    println ("\n hmap  = " + hmap)

end cardsTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cardsTest2` main function is used to test the `Cards` class.
 *  Classification of Poker Hands.
 *  > runMain scalation.simulation.monte_carlo.cardsTest2
 */
@main def cardsTest2 (): Unit =

    val deck = new Cards ()
    println ("\nOrdered deck of cards:")
    println (deck)
    deck.shuffle ()
    println ("\nShuffled deck of cards:")
    println (deck)

    for h <- 1 to 1000 do
        val hand  = for _ <- 1 to 5 yield deck.draw ()
        val cards = hand.map (value (_))
        val kind  = classify (hand)
        if kind > 1 then                                    // skip common hands
            banner (s"Hand $h")
            println ("\n hand  = " + hand)
            println ("\n cards = " + cards)
            println ("\n kind  = " + htype (kind))
        end if
        deck.shuffle ()
    end for

end cardsTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `cardsTest3` main function is used to test the `Cards` class.
 *  Monte Carlo simulation to estimate probabilities of Poker Hands.
 *  > runMain scalation.simulation.monte_carlo.cardsTest3
 */
@main def cardsTest3 (): Unit =

    val deck = new Cards ()
    println ("\nOrdered deck of cards:")
    println (deck)
    deck.shuffle ()
    println ("\nShuffled deck of cards:")
    println (deck)

    val iter  = 30000000
    val count = new VectorD (htype.length)

    cfor (0, iter) { _ =>
        val hand = for _ <- 1 to 5 yield deck.draw ()
        count(classify (hand)) += 1
        deck.shuffle ()
    } // cfor

    banner ("Monte Carlo Simulation Poker Hand Precentages")
    val mul = 100.0 / iter
    for k <- htype.indices do
        val prec = "%10.6f".format (count(k) * mul)
        println (s"kind = $k: $prec \t for ${htype(k)}")
    end for

end cardsTest3

