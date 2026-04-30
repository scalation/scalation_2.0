
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Hao Peng, John Miller
 *  @version 2.0
 *  @date    Fri Sep 30 12:27:14 EDT 2018
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Indirect Merge Sort 
 *
 *  An efficient stable sorting algorithm - useful for sorting on multiple fields/columns.
 *
 *  Translated from the following Java code:
 *  @see https://github.com/AbhijithMadhav/Data-Structures-And-Algorithms/blob/master/
 *       Data%20Structures%20and%20Algorithms/src/sorting/mergesort/MergeSortIndirect.java
 */

package scalation

import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `MergeSortIndirect` class provides an implementation of Indirect Merge Sort,
 *  a version of mergesort that does not re-arrange the array, but modifies an Int
 *  array called perm such that perm(i) is the index of the i-th smallest entry in
 *  the array (i.e., a(perm(i)) <= a(perm(i+1)).
 *  The isort method returns the permutation that sorts array a in perm.
 *  @param a     the array to be indirectly sorted
 *  @param perm  the initial permutation, either 0 until a.length or result of last isort
 */
class MergeSortIndirect (a: Array [ValueType]) (perm: Array [Int] = Array.range (0, a.length)):

    private val aux = Array.ofDim [Int] (a.length)             // non-updated copy of perm

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort the array 'a' by updating the permutation array 'perm'.
     */
    def isort (): Array [Int] = { indirectSort (0, a.length - 1); perm }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Merge a(lo to mid) with a(mid+1 to hi).
     *  @param lo   the lower bound
     *  @param mid  the middle
     *  @param hi   the upper bound
     */
    private def mergeIndirect (lo: Int, mid: Int, hi: Int): Unit =
        var i = lo                                             // current element in left half
        var j = mid + 1                                        // current element in right half

        for k <- lo to hi do aux(k) = perm(k)                  // copy perm(lo to hi) to aux(lo to hi)

        for k <- lo to hi do                                   // merge back to perm(lo to hi)
            if i > mid  then                                   // left half exhausted
                perm(k) = aux(j); j += 1
            else if j > hi then                                // right half exhausted
                perm(k) = aux(i); i += 1
            else if a(aux(j)) < a(aux(i)) then                 // take smaller element
                perm(k) = aux(j); j += 1
            else
                perm(k) = aux(i); i += 1
        end for
    end mergeIndirect

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursive helper method that sorts 'perm(lo to hi).
     *  @param lo  the lower bound
     *  @param hi  the upper bound (e.g., a.length at beginning)
     */
    private def indirectSort (lo: Int, hi: Int): Unit =
        if hi <= lo then return
        val mid = lo + (hi - lo) / 2

        indirectSort  (lo, mid)                                // sort left half
        indirectSort  (mid + 1, hi)                            // sort right half
        mergeIndirect (lo, mid, hi)                            // merge the two halves
    end indirectSort

end MergeSortIndirect


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `mergeSortIndirectTest` is used to test the `MergeSortIndirect` class.
 *  This test illustrates sorting multiple types.
 *  > runMain scalation.mergeSortIndirectTest
 */
@main def mergeSortIndirectTest (): Unit =

    val a1 = Array [ValueType] (5, 3, 7, 1, 10, 2, 9, 8, 6, 4)
    val a2 = Array [ValueType] (5.0, 3.0, 7.0, 1.0, 10.0, 2.0, 9.0, 8.0, 6.0, 4.0)

    banner ("Indirectly sort array of Int 'a1'")
    val perm1 = (new MergeSortIndirect (a1)()).isort ()        // the sorting permutation
    println (s"a1(0 until n) = ${stringOf (a1)}")
    print ("a1(perm1(0 until n)) = ")
    for ip <- perm1 do print (s"${a1(ip)} ")
    println ()

    banner ("Indirectly sort array of Double 'a2'")
    val perm2 = (new MergeSortIndirect (a2)()).isort ()        // the sorting permutation
    println (s"a2(0 until n) = ${stringOf (a2)}")
    print ("a2(perm2(0 until n)) = ")
    for ip <- perm2 do print (s"${a2(ip)} ")
    println ()

end mergeSortIndirectTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `mergeSortIndirectTest2` is used to test the `MergeSortIndirect` class.
 *  This test illustrates sorting multiple columns.
 *  > runMain scalation.mergeSortIndirectTest2
 */
@main def mergeSortIndirectTest2 (): Unit =

    import scalation.mathstat.MatrixI                          // WARNING: package inversion (but only for testing)

    val mat = MatrixI ((9, 2), 2, 2,
                               1, 1,
                               1, 3,
                               3, 3,
                               2, 1,
                               3, 1,
                               1, 2,
                               3, 2,
                               2, 3)

/*
    val mat = Array (Array [ValueType] (2, 1, 1, 3, 2, 3, 1, 2, 3),   // col 0
                     Array [ValueType] (2, 1, 3, 3, 1, 1, 2, 2, 3))   // col 1

*/
    println (s"mat = $mat")

    banner ("Indirectly sort mat column 1")                    // first, sort on the secondary ordering
    val col1 = mat(?, 1).toArray [ValueType]
    var perm = (new MergeSortIndirect (col1)()).isort ()
    println ("mat(perm(0 until n)) = ")
    for ip <- perm do println (s"${mat(ip)} ")

    banner ("Indirectly sort mat column 0 after column 1")     // second, sort on the primary ordering
    val col0 = mat(?, 0).toArray [ValueType]
    perm = (new MergeSortIndirect (col0)(perm)).isort ()       // must pass in the previous perm array
    println ("mat(perm(0 until n)) = ")
    for ip <- perm do println (s"${mat(ip)} ")

end mergeSortIndirectTest2

