
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Wed Mar 13 21:00:26 EDT 2024
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Extension Method for Scala's Mutable SortedSets
 */

package scalation

import scala.collection.mutable.SortedSet

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Extend `SortedSet` to include Unicode symbols for subset and proper subset (⊆, ⊂, ⊈, ⊄),
 *  interection/union (∩, ∪), and quantifiers exists, not exists and forall (∃, ∄, ∀).
 *  @see www.scala-lang.org/api/3.7.4/scala/collection/mutable/SortedSet.html
 *  @see `scalation.Unicode`
 */
extension [T] (x: SortedSet [T])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether set x is a subset of set y.
     *  @param y  the other set
     */
    inline def ⊆ (y: SortedSet [T]): Boolean = x subsetOf y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether set x is a proper subset of set y.
     *  @param y  the other set
     */
    inline def ⊂ (y: SortedSet [T]): Boolean = (x subsetOf y) && x != y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether set x is not a subset of set y.
     *  @param y  the other set
     */
    inline def ⊈ (y: SortedSet [T]): Boolean = ! (x subsetOf y)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether set x is not a proper subset of set y.
     *  @param y  the other set
     */
    inline def ⊄ (y: SortedSet [T]): Boolean = ! (x subsetOf y) || x == y

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether there exists at least one element of set x for which predicate p holds.
     *  @param p  the predicate to check
     */
    inline def ∃ (p: T => Boolean): Boolean = x.exists (p)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether there does not exists at least one element of set x for which predicate p holds.
     *  @param p  the predicate to check
     */
    inline def ∄ (p: T => Boolean): Boolean = ! x.exists (p)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether predicate p holds for all elements of set x.
     *  @param p  the predicate to check
     */
    inline def ∀ (p: T => Boolean): Boolean = x.forall (p)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the intersetion of set x and set y (x & y).
     *  @param y  the other set
     */
    inline def ∩ (y: SortedSet [T]): SortedSet [T] = x.intersect (y)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the union of set x and set y (x | y).
     *  @param y  the other set
     */
    inline def ∪ (y: SortedSet [T]): SortedSet [T] = x.union (y)


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `sortedSetExtTest` main function test the SortedSetExt extension methods.
 *  > runMain scalation.sortedSetExtTest
 */
@main def sortedSetExtTest (): Unit =

    val x = SortedSet (1, 2)
    val y = SortedSet (1, 2, 3)

    println (s"x = $x")
    println (s"y = $y")

    println (s"x ⊆ y = ${x ⊆ y}")
    println (s"x ⊆ x = ${x ⊆ x}")

    println (s"x ⊂ y = ${x ⊂ y}")
    println (s"x ⊂ x = ${x ⊂ x}")

    println (s"x ⊈ y = ${x ⊈ y}")
    println (s"x ⊈ x = ${x ⊈ x}")

    println (s"x ⊄ y = ${x ⊄ y}")
    println (s"x ⊄ x = ${x ⊄ x}")

end sortedSetExtTest

