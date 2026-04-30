 
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Jun 17 19:29:23 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Vector Data Structure of Strings
 */

package scalation
package mathstat

import java.util.Arrays.copyOf

import scala.collection.generic.DefaultSerializable
import scala.collection.immutable.{IndexedSeq => IIndexedSeq, Set}
import scala.collection.mutable.IndexedSeq
import scala.runtime.ScalaRunTime.stringOf

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VectorS` class stores and operates on Numeric Vectors of base type `String`.
 *  @param dim  the dimension/size of the vector
 *  @param v    the 1D array used to store vector elements
 */
class VectorS (val dim: Int,
               private [mathstat] var v: Array [String] = null)
      extends IndexedSeq [String]
         with PartiallyOrdered [VectorS]
         with DefaultSerializable:

    private val flaw = flawf ("VectorS")                          // partial invocation of flaw function

    if v == null then
        v = Array.ofDim [String] (dim)
    else if dim > v.length then
        flaw ("init", s"vector dimension is larger than space: dim = $dim > v.length = $v.length")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the length of this vector.
     */
    inline def length: Int = dim
    inline def nd          = dim.toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the size (dim) of this vector by more elements.
     *  @param more  the number of new elements to add
     */
    def expand (more: Int = dim): VectorS =
        if more < 1 then this                                     // no change
        else new VectorS (dim + more, Array.concat (v, new Array [String] (more)))
    end expand

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a deep copy of this vector (note: clone may not be deep).
     *  Uses Java's native `Arrays.copyOf` for efficiency.
     */
    def copy: VectorS = new VectorS (dim, copyOf (v, dim))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th element of this vector.
     *  @param i  the index of the element to return
     */
    def apply (i: Int): String = v(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements in range r of this vector.
     *  @param r  the index range of elements to return
     */
    def apply (r: Range): VectorS = new VectorS (r.end - r.start, v.slice (r.start, r.end))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements in index sequence idx of this vector.
     *  @param idx  the index sequence of elements to return
     */
    def apply (idx: IndexedSeq [Int]): VectorS = VectorS (for i <- idx.indices yield v(idx(i)))
    def apply (idx: IIndexedSeq [Int]): VectorS = VectorS (for i <- idx.indices yield v(idx(i)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements not equal to index ix of this vector.
     *  @param ix  the index to skip
     */
    def not (ix: Int): VectorS =
        VectorS (for i <- indices if i != ix yield v(i))
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements not in index sequence idx of this vector.
     *  @param idx  the index sequence of elements to skip
     */
    def not (idx: IndexedSeq [Int]): VectorS =
        VectorS (for i <- indices if ! (idx `contains` i) yield v(i))
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a vector containing all but the first n elements of this vector.
     *  @param n  the number of elements to be dropped
     */
    override def drop (n: Int): VectorS = new VectorS (dim - n, v.drop (n))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the elements in
     *  idx (e.g., testing set) and the other from elements not in idx (e.g., training set).
     *  Note split and split_ produce different element orders.
     *  @param idx  the element indices to include/exclude
     */
    def split (idx: Set [Int]): (VectorS, VectorS) =
        val len = idx.size
        val a   = new VectorS (len)
        val b   = new VectorS (dim - len)
        var j, k = 0
        for i <- indices do
            if idx `contains` i then
                a.v(j) = v(i)
                j += 1
            else
                b.v(k) = v(i)
                k += 1
        end for
        (a, b)
    end split

    inline def split (idx: IndexedSeq [Int]): (VectorS, VectorS) = split (idx.toSet [Int])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the elements in
     *  idx (e.g., testing set) and the other from elements not in idx (e.g., training set).
     *  Concise, but less efficient than split
     *  @param idx  the element indices to include/exclude
     */
    def split_ (idx: IndexedSeq [Int]): (VectorS, VectorS) = (this(idx), not(idx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the 
     *  the first i elements and the other from the rest of the elements.
     *  @param i  the split index
     */
    def split (i: Int): (VectorS, VectorS) = (new VectorS (i, v.slice (0, i)),
                                              new VectorS (dim - i, v.slice (i, dim)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Chop this vector into k sub-vectors of equal sizes (perhaps except for the last one).
     *  @param k  the number of pieces to chop this vector into
     */
    def chop (k: Int): Array [VectorS] =
        if k <= 0 then flaw ("chop", s"k = $k must be at least one")
        val pieces = Array.ofDim [VectorS] (k)
        val size = dim / k
        for i <- 0 until k-1 do pieces(i) = this (i*size until (i+1)*size)
        pieces(k-1) = this ((k-1)*size until dim)
        pieces
    end chop

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the i-th element (or in range) of this vector.
     *  @param i  the index of the element to update
     *  @param a  the updated value to assign
     */
    def update (i: Int, a: String): Unit = v(i) = a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the i-th element (or in range) of this vector.
     *  @param i  the index of the element to update
     *  @param a  the updated value to assign
     */
    def update (r: Range, a: String): Unit = for i <- r do v(i) = a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the i-th element (or in range) of this vector.
     *  @param i  the index of the element to update
     *  @param y  the update vector/indexed sequence to assign
     */
    def update (r: Range, y: IndexedSeq [String]): Unit = for i <- r do v(i) = y(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all elements in this vector to a (or vector y.
     *  @param a  the value to be assigned
     */
    def set (a: String): Unit = for i <- v.indices do v(i) = a
    def set (y: IndexedSeq [String]): Unit   = for i <- v.indices do v(i) = y(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map the elements of this vector by applying the mapping function f.
     *  @param f  the function to apply
     */
    def map (f: String => String): VectorS = new VectorS (v.size, v.map (f))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Iterate over this vector element by element applying the given function.
     *  @param f  the function to apply
     */
    override def foreach [U] (f: String => U): Unit = { var i = 0; while i < dim do { f (v(i)); i += 1 } }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Try to compare this vector to bb (return None if they are not comparable).
     *  As a partial order some vectors may not be comparable.
     *  @param bb  the other vector
     */
    infix def tryCompareTo [B >: VectorS: AsPartiallyOrdered] (bb: B): Option [Int] =
        if ! bb.isInstanceOf [VectorS] then return None
        val b  = bb.asInstanceOf [VectorS]
        var le = true
        var ge = true
        for i <- v.indices do
            if      ge && v(i) < b(i) then ge = false
            else if le && v(i) > b(i) then le = false
        end for
        if ge && le then Some (0)
        else if le  then Some (-1)
        else if ge  then Some (1)
        else None
    end tryCompareTo

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this is non-negative (contains no negative values).
     */
    def isNonnegative: Boolean = false

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate this vector and vector y.
     *  @param y  the other vector/indexed sequence
     */
    def ++ (y: IndexedSeq [String]): VectorS = new VectorS (dim + y.size, v ++ y)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Prepend (or append) this vector with scalar a.
     *  @param a  the scalar second operand
     */
    def +: (a: String): VectorS = new VectorS (dim + 1, a +: v)
    def :+ (a: String): VectorS = new VectorS (dim + 1, v :+ a)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the negative of this vector (unary minus).
     */
    def unary_- : VectorS = VectorS (for i <- v.indices yield -v(i))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of vectors this and y.
     *  @param y  the other vector/indexed sequence
     */
    def + (y: IndexedSeq [String]): VectorS = VectorS (for i <- v.indices yield v(i) + y(i))
    def - (y: IndexedSeq [String]): VectorS = VectorS (for i <- v.indices yield v(i) - y(i))
//  def * (y: IndexedSeq [String]): VectorS = VectorS (for i <- v.indices yield v(i) * y(i))
//  def / (y: IndexedSeq [String]): VectorS = VectorS (for i <- v.indices yield v(i) / y(i))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of this and scalar a.
     *  @param a  the scalar second operand
     */
    def + (a: String): VectorS = VectorS (for i <- v.indices yield v(i) + a)
    def - (a: String): VectorS = VectorS (for i <- v.indices yield v(i) - a)
    def * (a: Int): VectorS = VectorS (for i <- v.indices yield
          { println (s"v(i) = ${v(i)}, a = $a"); val z = v(i) * a; println (s"z = $z"); z })
    def / (a: Int): VectorS = VectorS (for i <- v.indices yield v(i) / a)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of vectors this and y.
     *  Perform operations in-place (destructive) to reduce memory allocations.
     *  @param y  the other vector/indexed sequence
     */
    def += (y: IndexedSeq [String]): VectorS = { for i <- v.indices do v(i) += y(i); this }
    def -= (y: IndexedSeq [String]): VectorS = { for i <- v.indices do v(i) -= y(i); this }
//  def *= (y: IndexedSeq [String]): VectorS = { for i <- v.indices do v(i) *= y(i); this }
//  def /= (y: IndexedSeq [String]): VectorS = { for i <- v.indices do v(i) /= y(i); this }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of this and scalar a.
     *  Perform operations in-place (destructive) to reduce memory allocations.
     *  @param a  the scalar second operand
     */
    def += (a: String): VectorS = { for i <- v.indices do v(i) += a; this }
    def -= (a: String): VectorS = { for i <- v.indices do v(i) -= a; this }
    def *= (a: Int): VectorS = { for i <- v.indices do v(i) *= a; this }
    def /= (a: Int): VectorS = { for i <- v.indices do v(i) /= a; this }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this vector and scalar a only at position i, e.g., x + (3, 5.5).
     *  @param ia = (i, a)  the (index position, scalar) to add
     */
    def + (ia: (Int, String)): VectorS = { val c = copy; c.v(ia._1) += ia._2; c }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract from this vector the scalar a only at position i, e.g., x - (3, 5.5).
     *  @param ia = (i, a)  the (index position, scalar) to subtract
     */
    def - (ia: (Int, String)): VectorS = { val c = copy; c.v(ia._1) -= ia._2; c }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise power function of this vector raised to scalar a.
     *  @param  the scalar second operand
     */
    def ~^ (a: String): VectorS = VectorS (for i <- v.indices yield v(i) ~^ a)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector consisting of the square root of each element of this vector.
     */
    def sqrt: VectorD = VectorD (for i <- v.indices yield math.sqrt (v(i).mkDouble))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector consisting of the reciprocal of each element of this vector.
    def recip: VectorD = VectorD (for i <- v.indices yield 1.0 / v(i).mkDouble)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the difference between this vector and vector y.
     *  @param y  the other vector/indexed sequence
     */
    infix def diff (y: IndexedSeq [String]): VectorS = { val a = v `diff` y; new VectorS (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the intersection of this vector and vector y.
     *  @param y  the other vector/indexed sequence
     */
    infix def intersect (y: IndexedSeq [String]): VectorS = { val a = v `intersect` y; new VectorS (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form a new vector consisting of the unique values in this vector.
     */
    override def distinct: VectorS = { val a = v.distinct; new VectorS (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the number of unique/distinct values in this vector.
     */
    def countDistinct: Int = v.distinct.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reverse the elements in this vector.
     */
    override def reverse: VectorS = { val a = v.reverse; new VectorS (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this vector is sorted in ascending order.
     */
    def isSorted: Boolean =
        var i = 0
        while i < dim-1 do if v(i) > v(i+1) then return false else i += 1
        true
    end isSorted

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort the elements in this vector according to ord.lt (ascending order).
     */
    def sorted: VectorS = { val a = v.sorted; new VectorS (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort the elements in this vector according to cmp (use '_ > _' for descending order.
     *  @param cmp  the comparison operator.
     */
    override def sortWith (cmp: (String, String) => Boolean): VectorS =
        val a = v.sortWith (cmp)
        new VectorS (a.size, a)
    end sortWith

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the predicate (or not predicate).
     *  @param  the filter predicate
     */
    override def filter (p: String => Boolean): VectorS = { val a = v.filter (p); new VectorS (a.size, a) }
    override def filterNot (p: String => Boolean): VectorS = { val a = v.filterNot (p); new VectorS (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the predicate, returning index positions.
     *  @param  the filter predicate
     */
    def filterPos (p: String => Boolean): IIndexedSeq [Int] = for i <- v.indices if p(v(i)) yield i

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the dot (inner) product of vectors this and y.
     *  @param  the other vector/indexed sequence
    infix def dot (y: IndexedSeq [String]): String =
        var sum = ""
        for i <- v.indices do sum += v(i) * y(i)
        sum
    end dot
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new vector consisting of the maximum of this and y's corresponding elements.
     *  @param y  the other vector/indexed sequence
     */
    infix def maxv (y: IndexedSeq [String]): VectorS =
        VectorS (for i <- indices yield if v(i) >= y(i) then v(i) else y(i))
    end maxv

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new vector consisting of the minimum of this and y's corresponding elements.
     *  @param y  the other vector/indexed sequence
     */
    infix def minv (y: IndexedSeq [String]): VectorS =
        VectorS (for i <- indices yield if v(i) <= y(i) then v(i) else y(i))
    end minv

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument maximum of this vector (index of maximum element).
     *  @param e  the ending index (exclusive) for the search
     */
    def argmax (e: Int = dim): Int =
        var j = 0
        for i <- 1 until e if v(i) > v(j) do j = i
        j
    end argmax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument maximum of this vector (index of maximum element).
     *  @param s  the starting index (inclusive) for the search
     *  @param e  the ending index (exclusive) for the search
     */
    def argmax (s: Int, e: Int): Int =
        var j = s
        for i <- s + 1 until e if v(i) > v(j) do j = i
        j
    end argmax

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument minimum of this vector (index of minimum element).
     *  @param e  the ending index (exclusive) for the search
     */
    def argmin (e: Int = dim): Int =
        var j = 0
        for i <- 1 until e if v(i) < v(j) do j = i
        j
    end argmin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Find the argument minimum of 'this' vector (index of minimum element).
     *  @param s  the starting index (inclusive) for the search
     *  @param e  the ending index (exclusive) for the search
     */
    def argmin (s: Int, e: Int): Int =
        var j = s
        for i <- s + 1 until e if v(i) < v(j) do j = i
        j
    end argmin

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the magnitude of this vector, i.e., the element value farthest from zero.
     */
    def mag: String = "NaN"

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute sum for the elements in this vector.
     */
    def sum: String =
        var s = ""
        for i <- v.indices do s += v(i)
        s
    end sum
        
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute three sums for the k-prefix, middle and k-suffix of this vector.
     *  @param k  the integer specifying the size of the prefix
     */
    def sums (k: Int): (String, String, String) =
        var s0, s1, s2 = ""
        for i <- v.indices do
           if i < k then          s0 += v(i)
           else if i < dim-k then s1 += v(i)
           else s2 += v(i)
        end for
        (s0, s1, s2)
    end sums

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute three squared norms for the k-prefix, middle and k-suffix of this vector.
     *  @param k  the integer specifying the size of the prefix
    def normSqs (k: Int): (String, String, String) =
        var s0, s1, s2 = ""
        for i <- v.indices do
           if i < k then          s0 += v(i) * v(i)
           else if i < dim-k then s1 += v(i) * v(i)
           else s2 += v(i) * v(i)
        end for
        (s0, s1, s2)
    end normSqs
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Euclidean norm (2-norm) (or its square) of this vector.
    def normSq: String = v.fold ("")((s, e) => s + e*e)
    def norm: Double = math.sqrt (normSq.mkDouble)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Manhattan norm (1-norm) of this vector.
     */
    def norm1: String = v.fold ("")((s, e) => s + e)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector that is the element-wise absolute value of this vector.
     */
    def abs: VectorS = copy

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cumulate the values of 'this' vector from left to right (e.g., create a
     *  CDF from a pmf).  Example: (4, 2, 3, 1) --> (4, 6, 9, 10)
     */
    def cumulate: VectorS = { var s = ""; VectorS (for i <- v.indices yield { s += v(i); s })}

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `String` vector to an `Int` vector.
     */
    def toInt: VectorI = new VectorI (dim, v.map (_.toInt))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `VectorS` into a `VectorI` by mapping each distinct value in
     *  `VectorS` into a distinct numeric integer value, returning the new vector
     *  and the bidirectional mapping.  Use the from method in `BiMap` to recover
     *  the original string.
     *  e.g., VectorS ("A", "B", "C", "A", "D") will be mapped to VectorI (0, 1, 2, 0, 3)
     */
    def map2Int: (VectorI, BiMap [String, Int]) =
        val map   = new BiMap [String, Int] ()
        var count = 0
        for i <- indices if ! (map `contains` (v(i))) do
            map   += v(i) -> count
            count += 1
        end for
        val vec = VectorI (for i <- indices yield map(v(i)))
        (vec, map)
    end map2Int

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `String` vector to a `Double` vector.
     */
    def toDouble: VectorD = new VectorD (dim, v.map (_.mkDouble))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert to a probability vector, by normalizing so that it sums to one.
     */
    def toProbability: VectorD = toDouble.toProbability

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Normalize this vector so its length is one (unit vector).
    def normalize: VectorD = toDouble.normalize
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Normalize this vector to have a maximum of one.
    def normalize1: VectorD = toDouble.normalize1
     */

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the number of zero elements in the this vector.
     */
    def countZero: Int =
        var count = 0
        for e <- v if e == "0" do count += 1
        count
    end countZero

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert vector to a string.
     */
    override def toString: String = super.toString.replace ("IndexedSeq", "VectorS")

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap (in-place) elements i and k in this vector.
     *  @param i  the first element in the swap
     *  @param k  the second element in the swap
     */
    def swap (i: Int, k: Int): Unit =
        val tmp = v(i); v(i) = v(k); v(k) = tmp
    end swap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly find the k-median of the p to r partition of array v
     *  using the QuickSelect algorithm.
     *  @see http://en.wikipedia.org/wiki/Quickselect
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     *  @param k   the type of median (k-th smallest element)
     */
    private def median (rk: Array [Int], p: Int, r: Int, k: Int): String =
        if p == r then return v(rk(p))
        iswap (rk, r, med3 (p, (p+r)/2, r))                // use median-of-3, comment out for simple pivot
        val q = ipartition (rk, p, r)                      // partition into left (<=) and right (>=)
        if q == k-1 then     return v(rk(q))               // found k-median
        else if q > k-1 then median (rk, p, q - 1, k)      // recursively find median in left partition
        else median (rk, q + 1, r, k)                      // recursively find median in right partition
    end median

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly find the k-median (k-th smallest element) of array v.
     *  @param k  the type of median (e.g., k = (dim+1)/2 is the median)
     */
    def median (k: Int = (dim+1)/2): String =
        if dim <= 0 then flaw ("median", s"no vector to take the median of k = $k, dim = $dim")
        median (Array.range (0, dim), 0, dim-1, k)
    end median

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the fraction quantile.
     *  @param fraction  the fraction/percentile to take
     */
    def quantile (fraction: Double): String =
        var k = (fraction * dim).toInt
        if k >= dim then k = dim - 1
        if k <= 0   then k = 1
        median (k)
    end quantile

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the averaged median, which is the median when dim is odd and
     *  the average of the median and the next k-median when dim is even.
     */
    def median_ : String = median ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the rank order for this vector, return its elements in that order.
     *  The rank order may be established using indirect sorting (e.g., iqsort).
     *  @param rank  the rank order of elements in this vector
     */
    def reorder (rank: Array [Int]): VectorS = VectorS (for i <- indices yield v(rank(i)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively and indirectly sort the p to r partition of array v 
     *  using QuickSort.
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     */
    private def iqsort (rk: Array [Int], p: Int, r: Int): Array [Int] =
        if r - p > 5 then
            iswap (rk, r, med3 (p, (p+r)/2, r))            // use median-of-3, comment out for simple pivot
            val q = ipartition (rk, p, r)                  // partition into left (<=) and right (>=)
            iqsort (rk, p, q - 1)                          // recursively sort left partition
            iqsort (rk, q + 1, r)                          // recursively sort right partition
        else
            iselsort (rk, p, r)                            // use simple sort when small
        rk
    end iqsort

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort this vector using QuickSort, returning the rank order.
     */
    inline def iqsort: Array [Int] = iqsort (Array.range (0, dim), 0, dim-1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort the p to r partition of array v using SelectionSort.
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     */
    private def iselsort (rk: Array [Int], p: Int, r: Int): Array [Int] =
        for i <- p to r do
            var k = i
            for j <- i+1 to r if v(rk(j)) < v(rk(k)) do k = j
            if i != k then iswap (rk, i, k)
        end for
        rk
    end iselsort

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort this vector using SelectionSort, returning the rank order.
     */
    def iselsort: Array [Int] = iselsort (Array.range (0, dim), 0, dim-1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort this vector using Selection Sort, returning the rank order
     *  of the stop smallest elements.
     *  @param stop  only sort stop number of smallest elements
     */
    def iselsort (stop: Int = dim): Array [Int] =
        val rk = Array.range (0, dim)
        for i <- 0 until stop do
            var k = i
            for j <- i+1 until dim if v(rk(j)) < v(rk(k)) do k = j
            if i != k then iswap (rk, i, k)
        end for
        rk.slice (0, stop)
    end iselsort

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly partition the array from 'p' to 'r' into a left partition
     *  (<= 'x') and a right partition (>= 'x').
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     */
    private def ipartition (rk: Array [Int], p: Int, r: Int): Int =
        val x = v(rk(r))                                   // pivot
        var i = p - 1
        for j <- p until r if v(rk(j)) <= x do
            i += 1; iswap (rk, i, j)
        end for
        iswap (rk, i + 1, r)
        i + 1
    end ipartition

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the index of the median of three elements.
     *  @param i  element index 1
     *  @param j  element index 2
     *  @param k  element index 3
     */
    inline private def med3 (i: Int, j: Int, k: Int): Int =
        if v(i) < v(j) then
            if v(j) < v(k) then j else if v(i) < v(k) then k else i
        else
            if v(j) > v(k) then j else if v(i) > v(k) then k else i
    end med3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly swap the elements at i and j, i.e., rk(i) <-> rk(j).
     *  @param rk  the rank order
     *  @param i   the first index position
     *  @param j   the second index position
     */
    inline private def iswap (rk: Array [Int], i: Int, j: Int): Unit =
        val t = rk(i); rk(i) = rk(j); rk(j) = t
    end iswap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample mean (also the population mean, they are the same).
     *  >> E(X)
    inline def mean: Double = toDouble.mean
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the centered norm-squared of this vector.
    def cnormSq: Double =
        var e, s, ss = 0.0
        for i <- indices do { e = v(i); s += e; ss += e * e }
        ss - s * s / nd
    end cnormSq
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample variance (and population population).
     *  >> E(X - μ)^2
    def variance: Double   = toDouble.variance
    def variance_ : Double = toDouble.variance_
    def variance (mu: Double): Double  = toDouble.variance (mu)
    def variance_ (mu: Double): Double = toDouble.variance_ (mu)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample covariance (or population covariance) of this vector with vector y.
     *  @param y  the other vector
    inifix def cov (y: IndexedSeq [String]): Double  = toDouble cov VectorD.fromStrings (y)
    inifix def cov_ (y: IndexedSeq [String]): Double = toDouble cov_ VectorD.fromStrings (y)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean square (ms) (or root mean square (rms)) of this vector.
    def ms: Double  = normSq.toDouble / nd
    def rms: Double = math.sqrt (ms)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample standard deviation (and population standard deviation).
    def stdev: Double   = math.sqrt (variance)
    def stdev_ : Double = math.sqrt (variance_)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for stationary and
     *  non-stationary series (acov_).
     *  @param k  the lag parameter
    def acov (k: Int = 1): Double = toDouble.acov (k)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for stationary series.
     *  This follows an intuitive defintion that divides by the number of elements summed dim-k.
     *  @param k  the lag parameter (0 <= k < n)
    def acov2 (k: Int = 1): Double =
        val n  = dim - k
        val mu = mean
        var s  = 0.0
        for i <- 0 until n do s += (v(i) - mu) * (v(i+k) - mu)
        s / n
    end acov2
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for non-stationary series.
     *  @param k  the lag parameter (0 <= k < n)
    def acov_ (k: Int = 1): Double = toDouble.acov_ (k)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute Pearson's correlation of this vector with vector y.
     *  If either variance is zero, will result in Not-a-Number (NaN), return
     *  one if the vectors are the same, or -0 (indicating undefined).
     *  @param y  the other vector
    infix def corr (y: VectorS): Double = toDouble.corr (y.toDouble)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-correlation of this vector.
     *  Assumes a stationary process vector, if not its an approximation.
     *  @param k  the lag parameter
    def acorr (k: Int = 1): Double  = toDouble.acorr (k)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag cross-correlation of this vector (assumes a stationary
     *  process vector, if not its an approximation).
     *  @param y  the other vector
     *  @param k  the lag parameter (0 <= k < n)
    def ccorr (y: VectorS, k: Int = 1): Double =
        val mu   = mean
        val mu_y = y.mean
        ccov (k, mu, y, mu_y) / (stdev * y.stdev)
    end ccorr
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-correlation of this vector for a non-stationary series.
     *  @param k  the lag parameter (0 <= k < n)
    def acorr_ (k: Int = 1): Double = toDouble.acorr_ (k)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute Spearman's rank correlation of this vector with vector y.
     *  The `iqsort` method gives the rank order of a vector.
     *  @see  en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
     *  @param y  the other vector
    infix def scorr (y: VectorS): Double = toDouble.scorr (y.toDouble)
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the skewness of this vector.  Negative skewness indicates the
     *  distribution is elongated on the left, zero skewness indicates it is
     *  symmetric, and positive skewness indicates it is elongated on the right.
     *  @see www.mathworks.com/help/stats/skewness.html
     *  >> E(X - μ)^3 / σ^3
    def skew: Double   = toDouble.skew
    def skew_ : Double = toDouble.skew_
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the kurtosis of 'x' vector.  High kurtosis (> 3) indicates a 
     *  distribution with heavier tails than a Normal distribution.
     *  @see www.mathworks.com/help/stats/kurtosis.html
     *  >> E(X - μ)^4 / σ^4
    def kurtosis: Double   = toDouble.kurtosis
    def kurtosis_ : Double = toDouble.kurtosis_
     */

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a standardized version of the vector by subtracting the mean and
     *  dividing by the standard deviation (e.g., Normal -> Standard Normal).
    def standardize: VectorD =toDouble.standardize
    def standardize2: VectorD = (this - mean) / (stdev + EPSILON)
     */

end VectorS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VectorS` object is the companion object for the `VectorS` class.
 */
object VectorS:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorS` from an immutable indexed sequence of `String`s.
     *  @param xs  the sequence/array of the `String` numbers
     */
    def apply (xs: collection.immutable.IndexedSeq [String]): VectorS = new VectorS (xs.size, xs.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorS` from a mutable indexed sequence of `String`s.
     *  @param xs  the sequence/array of the `String` numbers
     */
    def apply (xs: IndexedSeq [String]): VectorS = new VectorS (xs.size, xs.toArray)


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorS` from one or more values (repeated values `String`*).
     *  @param x   the first `String` number
     *  @param xs  the varargs of `String` numbers
     */
    def apply (x: String, xs: String*): VectorS = new VectorS (xs.size + 1, x +: xs.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorS` from a mutable indexed sequence of String.
     *  FIX - for numeric types, assign missing value indicator upon format failure.
     *  @param xs  the sequence/array of the `String` numbers
     */
    def fromStrings (xs: IndexedSeq [String]): VectorS = VectorS (xs.map (_.toString))
    def fromValueTypes (xs: IndexedSeq [ValueType]): VectorS = VectorS (xs.map (_.toString))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a `VectorS` consisting of a sequence of integers in a range.
     *  @param r  the range of values
     */
    def range (r: Range): VectorS = VectorS (for i <- r yield i.toString)
    def range (i1: Int, i2: Int): VectorS = VectorS (for i <- i1 until i2 yield i.toString)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorS` with n elements and fill it with the value x.
     *  @param n  the number of elements
     *  @param x  the value to assign to all elements
     */
    def fill (n: Int)(x: String): VectorS = new VectorS (n, Array.fill (n)(x))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a one vector (all elements are one) of length n.
     *  @param size  the size of the new vector
     */
    def one (n: Int): VectorS = new VectorS (n, Array.fill (n)("1"))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a vector of the form (0, ... 1, ... 0) where the 1 is at position j.
     *  @param j     the position to place the 1
     *  @param size  the size of the vector (upper bound = size - 1)
     */
    def oneAt (j: Int, size: Int): VectorS =
        val x = new VectorS (size)
        x.v(j) = "1"
        x
    end oneAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a null vector of type `VectorS`.
     */
    val nullv = null.asInstanceOf [VectorS]

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert an `Array [String]` into an `Array [Int]` by mapping each distinct value in
     *  `Array [String]` into a distinct numeric integer value, returning the new vector
     *  and the bidirectional mapping.  Use the from method in `BiMap` to recover
     *  the original string.
     *  e.g., Array ("A", "B", "C", "A", "D") will be mapped to Array (0, 1, 2, 0, 3)
     *  @param a  the array of strings to be converted
     */
    def map2Int (a: Array [String]): (Array [Int], BiMap [String, Int]) =
        val map   = new BiMap [String, Int] ()
        var count = 0
        for i <- a.indices if ! (map `contains` (a(i))) do
            map   += a(i) -> count
            count += 1
        end for
        val arr = Array.ofDim [Int] (a.size)
        for i <- a.indices do arr(i) = map(a(i))
        (arr, map)
    end map2Int

end VectorS


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorSTest` main function tests the operations provided by the `VectorS` class.
 *  Only the most commonly used inherited operations are shown.
 *  @see mutable.IndexedSeq for a complete list
 *  > runMain scalation.mathstat.vectorSTest
 */
@main def vectorSTest (): Unit =

    val x = VectorS ("1", "2", "3")
    val y = VectorS ("4", "6", "5")
    val z = VectorS ("4", "6", "5")
    val w = VectorS ("3", "4", "5", "5")
    val u = VectorS ("1", "2", "3", "4")
    val a = "2"
    val b = 2

    banner ("Given Vectors:")
    println (s"x = $x")
    println (s"y = $y")
    println (s"z = $z")
    println (s"w = $w")
    println (s"u = $u")

    banner ("Inherited Operations:")

    println (s"x == y              = ${x == y}")                 // vector equality
    println (s"y == z              = ${y == z}")
    println (s"x != y              = ${x != y}")                 // vector inequality
    println (s"x < y               = ${x < y}")                  // less than
    println (s"x <= y              = ${x <= y}")                 // less than or equal
    println (s"x > y               = ${x > y}")                  // greater then
    println (s"x >= y              = ${x >= y}")                 // greater then or equal

    println (s"x `contains` a      = ${x `contains` a}")         // element contained in vector
    println (s"x `contains` 4      = ${x `contains` "4"}")
    println (s"x.exists (_ > a)    = ${x.exists (_ > a)}")       // existence of element satisfying predicate
    println (s"x.groupBy (_ > a)   = ${x.groupBy (_ > a)}")      // group according function values
    println (s"x.indexOf (a)       = ${x.indexOf (a)}")          // index of first element equaling
    println (s"x.indexWhere (_ > a) = ${x.indexWhere (_ > a)}")  // index of first element satisfying predicate
    println (s"x.indices           = ${x.indices}")              // indices of the vector
    println (s"u.map (_ + a)       = ${u.map (_ + a)}")          // map vector elements using the function
    println (s"x.max               = ${x.max}")                  // maximum element
    println (s"x.min               = ${x.min}")                  // minimum element
//  println (s"x.product           = ${x.product}")              // product of all elements
    println (s"x.sum               = ${x.sum}")                  // sum of all elements
    println (s"w.toArray           = ${stringOf (w.toArray)}")   // convert to array - use stringOf to print arrays
    println (s"w.toSet             = ${w.toSet}")                // convert to set

    banner ("Implemented Operations:")

    println (s"x(2)                = ${x(2)}")                   // value at index
    println (s"x(0 to 2)           = ${x(0 to 2)}")              // values in exclusive range

    println (s"-x                  = ${-x}")                     // unary minus
    println (s"x + y               = ${x + y}")                  // element-wise vector addition
    println (s"x - y               = ${x - y}")                  // element-wise vector subtraction
//  println (s"x * y               = ${x * y}")                  // element-wise vector multiplication
//  println (s"x / y               = ${x / y}")                  // element-wise vector division
    println (s"x ++ y              = ${x ++ y}")                 // concatenate vectors

    println (s"x + a               = ${x + a}")                  // add scalar a
    println (s"x - a               = ${x - a}")                  // subtract scalar a
    println (s"x * b               = ${x * b}")                  // multiply by scalar integer b
    println (s"x / b               = ${x / b}")                  // divide by scalar a
    println (s"x ~^ a              = ${x ~^ a}")                 // raise to power of scalar a
    println (s"a +: y              = ${a +: x}")                 // prepend scalar a
    println (s"x :+ a              = ${x :+ a}")                 // append scalar a

    println (s"(x-y).abs           = ${(x-y).abs}")              // absolute value
    println (s"x.cumulate          = ${x.cumulate}")             // cumulate all prior values
    println (s"x diff w            = ${x diff w}")               // multi-set difference
    println (s"w.distinct          = ${w.distinct}")             // extract distinct values
//  println (s"x dot y             = ${x dot y}")                // dot product
    println (s"u.filter (_ > a)    = ${u.filter (_ > a)}")       // filter on predicate
    println (s"u.filterNot (_ > a) = ${u.filterNot (_ > a)}")    // filter on not predicate
    println (s"u.filterPos (_ > a) = ${u.filterPos (_ > a)}")    // filter return indices
    println (s"x intersect w       = ${x intersect w}")          // multi-set intersection
    println (s"y.iselsort          = ${stringOf (y.iselsort)}")  // indirect QuickSort in ascending order
    println (s"y.iqsort            = ${stringOf (y.iqsort)}")    // indirect SelectionSort in ascending order
//  println (s"x.norm              = ${x.norm}")                 // Euclidean norm
//  println (s"x.normalize         = ${x.normalize}")            // normalize the vector to a unit vector
//  println (s"x.normalize1        = ${x.normalize1}")           // normalize the vector to max 1 vector
//  println (s"x.normSq            = ${x.normSq}")               // Euclidean norm squared
    println (s"x.norm1             = ${x.norm1}")                // Manhattan norm
//  println (s"x.recip             = ${x.recip}")                // reciprocal
    println (s"x.reverse           = ${x.reverse}")              // reverse elements in vector
    println (s"y.sorted            = ${y.sorted}")               // sort in ascending order
    println (s"y.sortWith (_ > _)  = ${y.sortWith (_ > _)}")     // sort in descending order
//  println (s"y.toProbabilty      = ${y.toProbability}")        // convert to a probability vector

/***
    banner ("Implemented Statistical Operations: (_ => population")

    println (s"x.acorr ()          = ${x.acorr ()}")             // auto-correlation
    println (s"x.acorr_ ()         = ${x.acorr_ ()}")            // auto-correlation - non-stationary case
    println (s"x.acov ()           = ${x.acov ()}")              // auto-covariance
    println (s"x.acov_ ()          = ${x.acov_ ()}")             // auto-covariance - non-stationary case
    println (s"x corr y            = ${x corr y}")               // correlation
    println (s"x cov y             = ${x cov y}")                // covariance      
    println (s"x cov_ y            = ${x cov_ y}")
    println (s"x.kurtosis          = ${x.kurtosis}")             // kurtosis
    println (s"x.mean              = ${x.mean}")                 // mean
    println (s"u.median ()         = ${u.median ()}")            // median
    println (s"u.median_           = ${u.median_}")              // averaged median
    println (s"x ms y              = ${x.ms}")                   // mean square
    println (s"x rms y             = ${x.rms}")                  // root mean square
    println (s"x scorr y           = ${x scorr y}")              // Spearman's rank correlation
    println (s"x.skew              = ${x.skew}")                 // skewness
    println (s"x.skew_             = ${x.skew_}")
    println (s"x.standardize       = ${x.standardize}")          // standardize the vector
    println (s"x.stdev             = ${x.stdev}")                // standard deviation
    println (s"x.stdev_            = ${x.stdev_}")
    println (s"x.variance          = ${x.variance}")             // variance
    println (s"x.variance_         = ${x.variance_}")
***/

end vectorSTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorSTest2` main function tests the operations provided by the `VectorS` class.
 *  It tests the map2Int method (String vector to Int vector).
 *  > runMain scalation.mathstat.vectorSTest2
 */
@main def vectorSTest2 (): Unit =

    val months = VectorS ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

    val (vec, map) = months.map2Int

    println (s"vec = $vec") 
    println (s"map = $map") 
    for vi <- vec do println (s"from ($vi) = ${map.from (vi)}")

end vectorSTest2

