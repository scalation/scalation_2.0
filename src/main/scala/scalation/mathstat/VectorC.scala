
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  John Miller
 *  @version 2.0
 *  @date    Thu Jun 17 19:29:23 EDT 2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @note    Vector Data Structure of Complex Numbers
 *
 *  https://www.infoq.com/news/2023/10/foreign-function-and-memory-api/
 */

package scalation
package mathstat

import java.util.Arrays.copyOf

import scala.collection.generic.DefaultSerializable
import scala.collection.immutable.{IndexedSeq => IIndexedSeq, Set}
import scala.collection.mutable.IndexedSeq
import scala.runtime.ScalaRunTime.stringOf
import scala.util.control.Breaks.{break, breakable}

import Complex._

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VectorC` class stores and operates on Numeric Vectors of base type `Complex`.
 *  @param dim  the dimension/size of the vector
 *  @param v    the 1D array used to store vector elements
 */
class VectorC (val dim: Int,
               private [mathstat] var v: Array [Complex] = null)
      extends IndexedSeq [Complex]
         with PartiallyOrdered [VectorC]
         with DefaultSerializable:
    
    private val EPSILON = 1E-9                                    // number close to zero
    private val flaw    = flawf ("VectorC")                       // partial invocation of flaw function
    private val fString = "%g,\t"                                 // output format spec

    if v == null then
        v = Array.ofDim [Complex] (dim)
    else if dim > v.length then
        flaw ("init", s"vector dimension is larger than space: dim = $dim > v.length = ${v.length}")
        assert (dim <= v.length)                                  // make this a fatal flaw

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the length of this vector.
     */
    inline def length: Int = dim
    inline def nd          = dim.toDouble

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Expand the size (dim) of this vector by more elements.
     *  @param more  the number of new elements to add
     */
    def expand (more: Int = dim): VectorC =
        if more < 1 then this                                     // no change
        else new VectorC (dim + more, Array.concat (v, new Array [Complex] (more)))
    end expand

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a deep copy of this vector (note: clone may not be deep).
     *  Uses Java's native `Arrays.copyOf` for efficiency.
     */
    def copy: VectorC = new VectorC (dim, copyOf (v, dim))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the i-th element of this vector.
     *  @param i  the index of the element to return
     */
    def apply (i: Int): Complex = v(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements in range r of this vector, making sure not to go beyond
     *  the end of vector.
     *  @param r  the index range of elements to return
     */
    def apply (r: Range): VectorC = 
        if r.end > dim then
            flaw ("apply", s"Range beyond end: r.start = ${r.start}, r.end = ${r.end} > dim = $dim")
            new VectorC (dim - r.start, v.slice (r.start, dim))
        else if r.end - r.start <= 0 then
//          flaw ("apply", s"Range is empty: r.start = ${r.start}, r.end = ${r.end} > dim = $dim")
            new VectorC (0, Array.ofDim [Complex] (0))
        else
            new VectorC (r.end - r.start, v.slice (r.start, r.end))
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements in index sequence idx of this vector.
     *  @param idx  the index sequence of elements to return
     */
    def apply (idx: IndexedSeq [Int]): VectorC = VectorC (for i <- idx.indices yield v(idx(i)))
    def apply (idx: IIndexedSeq [Int]): VectorC = VectorC (for i <- idx.indices yield v(idx(i)))
    def apply (idx: Array [Int]): VectorC = VectorC (for i <- idx.indices yield v(idx(i)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements not equal to index ix of this vector.
     *  @param ix  the index to skip
     */
    def not (ix: Int): VectorC =
        VectorC (for i <- indices if i != ix yield v(i))
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the elements not in index sequence idx of this vector.
     *  @param idx  the index sequence of elements to skip
     */
    def not (idx: IndexedSeq [Int]): VectorC =
        VectorC (for i <- indices if ! (idx `contains` i) yield v(i))
    end not

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a vector containing all but the first n elements of this vector.
     *  @param n  the number of elements to be dropped
     */
    override def drop (n: Int): VectorC = new VectorC (dim - n, v.drop (n))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the elements in
     *  idx (e.g., testing set) and the other from elements not in idx (e.g., training set).
     *  Note split and split_ produce different element orders.
     *  @param idx  the set of element indices to include/exclude
     */
    def split (idx: Set [Int]): (VectorC, VectorC) =
        val len = idx.size
        val a   = new VectorC (len)
        val b   = new VectorC (dim - len)
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

    inline def split (idx: IndexedSeq [Int]): (VectorC, VectorC) = split (idx.toSet [Int])

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the elements in
     *  idx (e.g., testing set) and the other from elements not in idx (e.g., training set).
     *  Concise, but less efficient than split
     *  @param idx  the element indices to include/exclude
     */
    def split_ (idx: IndexedSeq [Int]): (VectorC, VectorC) = (this(idx), not(idx))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Split the elements from this vector to form two vectors:  one from the
     *  the first i elements and the other from the rest of the elements.
     *  @param i  the split index
     */
    def split (i: Int): (VectorC, VectorC) = (new VectorC (i, v.slice (0, i)),
                                              new VectorC (dim - i, v.slice (i, dim)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Chop this vector into k sub-vectors of equal sizes (perhaps except for the last one).
     *  @param k  the number of pieces to chop this vector into
     */
    def chop (k: Int): Array [VectorC] =
        if k <= 0 then flaw ("chop", s"k = $k must be at least one")
        val pieces = Array.ofDim [VectorC] (k)
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
    def update (i: Int, a: Complex): Unit = v(i) = a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the i-th element (or in range) of this vector.
     *  @param i  the index of the element to update
     *  @param a  the update value to assign
     */
    def update (r: Range, a: Complex): Unit = for i <- r do v(i) = a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Update the i-th element (or in range) of this vector.
     *  @param i  the index of the element to update
     *  @param y  the update vector/indexed sequence to assign
     */
    def update (r: Range, y: IndexedSeq [Complex]): Unit = for i <- r do v(i) = y(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all elements in this vector to scalar a.
     *  @param a  the scalar value to be assigned
     */
    def set (a: Complex): Unit = for i <- v.indices do v(i) = a

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Set all elements in this vector to vector y.
     *  @param y  the vector value to be assigned
     */
    def set (y: IndexedSeq [Complex]): Unit   = for i <- v.indices do v(i) = y(i)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Iterate over this vector element by element applying the given function.
     *  @param f  the function to apply
     */
    override def foreach [U] (f: Complex => U): Unit = { var i = 0; while i < dim do { f (v(i)); i += 1 } }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Map the elements of this vector by applying the mapping function f.
     *  @param f  the function to apply
     */
//  def map (f: FunctionS2S): VectorC = new VectorC (v.size, v.map (f))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Try to compare this vector to bb (return None if they are not comparable).
     *  As a partial order some vectors may not be comparable.
     *  @param bb  the other vector
     */
    infix def tryCompareTo [B >: VectorC: AsPartiallyOrdered] (bb: B): Option [Int] =
        if ! bb.isInstanceOf [VectorC] then return None
        val b  = bb.asInstanceOf [VectorC]
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
    /** Find values in this vector of infinite magnitude, returning all such index positions.
     */
//  def findInfinity: IIndexedSeq [Int] = for i <- indices if v(i).isInfinite yield i

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return whether this vector is non-negative (contains no negative values).
     */
    def isNonnegative: Boolean = v.forall (_ >= Complex._0)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Concatenate this vector and vector y.
     *  @param y  the other vector/indexed sequence
     */
    def ++ (y: IndexedSeq [Complex]): VectorC = new VectorC (dim + y.size, v ++ y)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Prepend (or append) this vector with scalar a.
     *  @param a  the scalar second operand
     */
    def +: (a: Complex): VectorC = new VectorC (dim + 1, a +: v)
    def :+ (a: Complex): VectorC = new VectorC (dim + 1, v :+ a)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the negative of this vector (unary minus).
     */
    def unary_- : VectorC = VectorC (for i <- v.indices yield -v(i))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of vectors this and y.
     *  @param y  the other vector/indexed sequence
     */
    def + (y: IndexedSeq [Complex]): VectorC = VectorC (for i <- v.indices yield v(i) + y(i))
    def - (y: IndexedSeq [Complex]): VectorC = VectorC (for i <- v.indices yield v(i) - y(i))
    def * (y: IndexedSeq [Complex]): VectorC = VectorC (for i <- v.indices yield v(i) * y(i))
    def / (y: IndexedSeq [Complex]): VectorC = VectorC (for i <- v.indices yield v(i) / y(i))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of this and scalar a.
     *  @param a  the scalar second operand
     */
    def + (a: Complex): VectorC = VectorC (for i <- v.indices yield v(i) + a)
    def - (a: Complex): VectorC = VectorC (for i <- v.indices yield v(i) - a)
    def * (a: Complex): VectorC = VectorC (for i <- v.indices yield v(i) * a)
    def / (a: Complex): VectorC = VectorC (for i <- v.indices yield v(i) / a)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of vectors this and y.
     *  Perform operations in-place (destructive) to reduce memory allocations.
     *  @param y  the other vector/indexed sequence
     */
    def += (y: IndexedSeq [Complex]): VectorC = { for i <- v.indices do v(i) += y(i); this }
    def -= (y: IndexedSeq [Complex]): VectorC = { for i <- v.indices do v(i) -= y(i); this }
    def *= (y: IndexedSeq [Complex]): VectorC = { for i <- v.indices do v(i) *= y(i); this }
    def /= (y: IndexedSeq [Complex]): VectorC = { for i <- v.indices do v(i) /= y(i); this }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise sum (or difference, product, quotient) of this and scalar a.
     *  Perform operations in-place (destructive) to reduce memory allocations.
     *  @param a  the scalar second operand
     */
    def += (a: Complex): VectorC = { for i <- v.indices do v(i) += a; this }
    def -= (a: Complex): VectorC = { for i <- v.indices do v(i) -= a; this }
    def *= (a: Complex): VectorC = { for i <- v.indices do v(i) *= a; this }
    def /= (a: Complex): VectorC = { for i <- v.indices do v(i) /= a; this }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Add this vector and scalar a only at position i, e.g., x + (3, 5.5).
     *  @param ia = (i, a)  the (index position, scalar) to add
     */
    def + (ia: (Int, Complex)): VectorC = { val c = copy; c.v(ia._1) += ia._2; c }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Subtract from this vector the scalar a only at position i, e.g., x - (3, 5.5).
     *  @param ia = (i, a)  the (index position, scalar) to subtract
     */
    def - (ia: (Int, Complex)): VectorC = { val c = copy; c.v(ia._1) -= ia._2; c }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the element-wise power function of this vector raised to scalar a.
     *  @param  the scalar second operand
     */
    def ~^ (a: Double): VectorC = VectorC (for i <- v.indices yield v(i) ~^ a)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this vector and vector y are nearly equal.
     *  @param y  the other vector
     */
    def =~ (y: VectorC): Boolean =
        if dim != y.dim then return false
        var close = true
        breakable {
            for i <- indices do
                if ! (v(i) =~ y.v(i)) then { close = false; break () }
            end for
        } // breakable
        close
    end =~

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector consisting of the square root of each element of this vector.
     */
    def sqrt: VectorC = VectorC (for i <- v.indices yield Complex.sqrt (v(i)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector consisting of the reciprocal of each element of this vector.
     */
    def recip: VectorC = VectorC (for i <- v.indices yield Complex._1 / v(i))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the difference between this vector and vector y.
     *  @param y  the other vector/indexed sequence
     */
    infix def diff (y: IndexedSeq [Complex]): VectorC = { val a = v `diff` y; new VectorC (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the intersection of this vector and vector y.
     *  @param y  the other vector/indexed sequence
     */
    infix def intersect (y: IndexedSeq [Complex]): VectorC = { val a = v `intersect` y; new VectorC (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Form a new vector consisting of the unique values in this vector.
     */
    override def distinct: VectorC = { val a = v.distinct; new VectorC (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the number of unique/distinct values in this vector.
     */
    def countDistinct: Int = v.distinct.size

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Reverse the elements in this vector.
     */
    override def reverse: VectorC = { val a = v.reverse; new VectorC (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Determine whether this vector is sorted in ascending order.
     */
    def isSorted: Boolean = 
        var i = 0
        while i < dim-1 do if v(i) > v(i+1) then return false else i += 1
        true
    end isSorted

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort the elements in this vector according to ord.lt (ascending order)
     *  returning a new sorted vector.
     */
//  def sorted: VectorC = { val a = v.sorted; new VectorC (a.size, a) }

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Sort the elements in this vector according to cmp (use '_ > _' for descending order).
     *  @param cmp  the comparison operator.
     */
    override def sortWith (cmp: (Complex, Complex) => Boolean): VectorC =
        val a = v.sortWith (cmp)
        new VectorC (a.size, a)
    end sortWith

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the predicate.
     *  @param  the filter predicate
     */
    override def filter (p: Complex => Boolean): VectorC = 
        val a = v.filter (p)
        new VectorC (a.size, a)
    end filter

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the negation of the predicate.
     *  @param  the filter predicate
     */
    override def filterNot (p: Complex => Boolean): VectorC = 
        val a = v.filterNot (p)
        new VectorC (a.size, a)
    end filterNot

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Filter the elements in this vector based on the predicate, returning index positions.
     *  @param  the filter predicate
     */
    def filterPos (p: Complex => Boolean): IIndexedSeq [Int] = for i <- v.indices if p(v(i)) yield i

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the dot (inner) product of vectors this and y.
     *  @param  the other vector/indexed sequence
     */
    infix def dot (y: IndexedSeq [Complex]): Complex =
        var s = Complex._0
        for i <- v.indices do s += v(i) * y(i)
        s
    end dot

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the maximum element of this vector.
     */
    def max: Complex =
        var mx = v(0)
        for i <- 1 until dim do if mx > v(i) then mx = v(i)
        mx
    end max

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the minimum element of this vector.
     */
    def min: Complex =
        var mn = v(0)
        for i <- 1 until dim do if mn < v(i) then mn = v(i)
        mn
    end min

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new vector consisting of the maximum of this vector elements and zero.
     */
    infix def max0: VectorC =
        VectorC (for i <- indices yield if v(i) < Complex._0 then Complex._0 else v(i))
    end max0

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new vector consisting of the maximum of this and y's corresponding elements.
     *  @param y  the other vector/indexed sequence
     */
    infix def maxv (y: IndexedSeq [Complex]): VectorC =
        VectorC (for i <- indices yield if v(i) >= y(i) then v(i) else y(i))
    end maxv

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a new vector consisting of the minimum of this and y's corresponding elements.
     *  @param y  the other vector/indexed sequence
     */
    infix def minv (y: IndexedSeq [Complex]): VectorC =
        VectorC (for i <- indices yield if v(i) <= y(i) then v(i) else y(i))
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
    /** Find the argument maximum magnitude of this vector (index of most extreme element).
     *  @param e  the ending index (exclusive) for the search
     */
    def argmag (e: Int = dim): Int =
        var j = 0
        for i <- 1 until e if Complex.abs (v(i)) > Complex.abs (v(j)) do j = i
        j
    end argmag

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the magnitude of this vector, i.e., the element value farthest from zero.
     */
//  def mag: Complex = Complex.max (Complex.abs (min), Complex.abs (max))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute product of the elements of this vector.
     */
    def product: Complex =
        var p = Complex._1
        for i <- v.indices do p *= v(i)
        p
    end product

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute sum of the elements of this vector.
     */
    def sum: Complex =
        var s = Complex._0
        for i <- v.indices do s += v(i)
        s
    end sum

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute three sums for the k-prefix, middle and k-suffix of this vector.
     *  @param k  the integer specifying the size of the prefix
     */
    def sums (k: Int): (Complex, Complex, Complex) =
        var s0, s1, s2 = Complex._0
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
     */
    def normSqs (k: Int): (Complex, Complex, Complex) =
        var s0, s1, s2 = Complex._0
        for i <- v.indices do
           if i < k then          s0 += v(i) * v(i)
           else if i < dim-k then s1 += v(i) * v(i)
           else s2 += v(i) * v(i)
        end for
        (s0, s1, s2)
    end normSqs

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Euclidean norm (2-norm) (or its square) of this vector.
     */
    def normSq: Complex = v.fold (Complex._0)((s, e) => s + e*e)
    def norm: Complex   = Complex.sqrt (normSq)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the Manhattan norm (1-norm) of this vector.
     */
    def norm1: Complex = v.fold (Complex._0)((s, e) => s + Complex.abs (e))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector that is the element-wise absolute value of this vector.
     */
    def abs: VectorC = VectorC (for i <- v.indices yield Complex.abs (v(i)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return the vector containing the mid-points between adjacent elements.
     */
    def mids: VectorC = VectorC (for i <- 1 until dim yield (v(i) + v(i-1)) * 0.5)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Cumulate the values of 'this' vector from left to right (e.g., create a
     *  CDF from a pmf).  Example: (4, 2, 3, 1) --> (4, 6, 9, 10)
     */
    def cumulate: VectorC = { var s = Complex._0; VectorC (for i <- v.indices yield { s += v(i); s })}

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `Complex` vector to an `Int` vector.
     */
    def toInt: VectorI = new VectorI (dim, v.map (_.toInt))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert this `Complex` vector to a `Double` vector.
     */
    def toDouble: VectorD = new VectorD (dim, v.map (_.toDouble))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert 'this' `VectorC` into a `VectorC`.
     */
    def toComplex: VectorC = this

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert to a probability vector, by normalizing so that it sums to one.
     */
    def toProbability: VectorC = this * (Complex._1 / sum)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Normalize this vector so its length is one (unit vector).
     */
    def normalize: VectorC = this * (Complex._1 / norm)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Normalize this vector to have a maximum of one.
     */
    def normalize1: VectorC = this * (Complex._1 / max)

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Count the number of zero elements in the this vector.
     */
    def countZero: Int =
        var count = 0
        for e <- v if e == Complex._0 do count += 1
        count
    end countZero

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert vector to a string.
     */
    override def toString: String =
        val sb = new StringBuilder ("VectorC(")
        if dim == 0 then return sb.append (")").mkString
        for i <- indices do sb.append (fString.format (v(i)))
        sb.replace (sb.length-2, sb.length, ")").mkString
    end toString

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Convert vector to vector of strings.
     */
    def toString2: VectorS = new VectorS (dim, v.map (_.toString))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Swap (in-place) elements i and k in this vector.
     *  @param i  the first element in the swap
     *  @param k  the second element in the swap
     */
    def swap (i: Int, k: Int): Unit =
        val tmp = v(i); v(i) = v(k); v(k) = tmp
    end swap

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly find the k-median of the p to r partition of array v
     *  using the QuickSelect algorithm.
     *  @see http://en.wikipedia.org/wiki/Quickselect
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     *  @param k   the type of median (k-th smallest element)
     */
    private def median (rk: Array [Int], p: Int, r: Int, k: Int): Complex =
        if p == r then return v(rk(p))
        iswap (rk, r, med3 (p, (p+r)/2, r))                // use median-of-3, comment out for simple pivot
        val q = ipartition (rk, p, r)                      // partition into left (<=) and right (>=)
        if q == k-1 then     return v(rk(q))               // found k-median
        else if q > k-1 then median (rk, p, q - 1, k)      // recursively find median in left partition
        else median (rk, q + 1, r, k)                      // recursively find median in right partition
    end median

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly find the k-median (k-th smallest element) of array v.
     *  @param k  the type of median (e.g., k = (dim+1)/2 is the median)
     */
    def median (k: Int = (dim+1)/2): Complex =
        if dim <= 0 then flaw ("median", s"no vector to take the median of k = $k, dim = $dim")
        median (Array.range (0, dim), 0, dim-1, k)
    end median

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the fraction quantile.
     *  @param fraction  the fraction/percentile to take
     */
    def quantile (fraction: Complex): Complex =
        var k = (fraction * dim).toInt
        if k >= dim then k = dim - 1
        if k <= 0   then k = 1
        median (k)
    end quantile

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the averaged median, which is the median when dim is odd and
     *  the average of the median and the next k-median when dim is even.
     */
    def median_ : Complex = if dim % 2 == 0 then (median () + median ((dim+2)/2)) / 2.0
                           else median ()

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Given the rank order for this vector, return its elements in that order.
     *  The rank order may be established using indirect sorting (e.g., iqsort).
     *  @param rank  the rank order of elements in this vector
     */
    def reorder (rank: Array [Int]): VectorC = VectorC (for i <- indices yield v(rank(i)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Recursively and indirectly sort the p to r partition of array v 
     *  using QuickSort.  This version avoids stack overflow.
     *  @author Sulaiman Owodunni
     *  @param rk  the rank order
     *  @param p   the left cursor
     *  @param r   the right cursor
     */
    private def iqsort (rk: Array [Int], p: Int, r: Int): Array [Int] =
        if r - p > 5 then
            iswap (rk, r, med3 (p, (p+r)/2, r))            // use median-of-3, comment out for simple pivot
            var (p_, r_) = (p, r)                          // use local cursors
            while p_ < r_ do
                val pivot = ipartition (rk, p_, r_)        // partition into left (<=) and right (>=)
                if pivot - p_ < r_ - pivot then            // recurse on the smaller subarray
                    iqsort (rk, p_, pivot - 1)             // recursively sort left partition
                    p_ = pivot + 1
                else
                    iqsort (rk, pivot + 1, r_)             // recursively sort right partition
                    r_ = pivot - 1
            end while
        else
            iselsort (rk, p, r)                            // use simple sort when small
        rk
    end iqsort

/*
    private def iqsort_ (rk: Array [Int], p: Int, r: Int): Array [Int] =
        if r - p > 5 then
            iswap (rk, r, med3 (p, (p+r)/2, r))            // use median-of-3, comment out for simple pivot
            val pivot = ipartition (rk, p, r)              // partition into left (<=) and right (>=)
            iqsort (rk, p, pivot - 1)                      // recursively sort left partition
            iqsort (rk, pivot + 1, r)                      // recursively sort right partition
        else
            iselsort (rk, p, r)                            // use simple sort when small
        rk
    end iqsort_
*/

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort this vector using QuickSort, returning the rank order.
     */
    inline def iqsort: Array [Int] = iqsort (Array.range (0, dim), 0, dim-1)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Indirectly sort the p to r partition of array v using Selection Sort.
     *  @param rk  the rank order (index order, index of smallest element, etc.)
     *  @param p   the left cursor (inclusive)
     *  @param r   the right cursor (inclusive)
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
     */
    inline def mean: Complex = sum / nd

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean of a subvector.
     *  @param j  the starting index (inclusive)
     *  @param k  the ending index (exclusive)
     */
    def mean (j: Int, k: Int = dim): Complex =
        if k <= j then v(j)
        else
            var sum = Complex._0
            for i <- j until k do sum += v(i)
            sum / (k - j)
    end mean

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the centered norm-squared of this vector.
     */
    def cnormSq: Complex =
        var e, s, ss = Complex._0
        for i <- indices do { e = v(i); s += e; ss += e * e }
        ss - s * s / nd
    end cnormSq

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample variance (and population population).
     *  >> E(X - μ)^2
     */
    def variance: Complex   = cnormSq / (nd-1)
    def variance_ : Complex = cnormSq / nd
    def variance (mu: Complex): Complex  = (normSq - mu * mu * nd) / (nd-1)
    def variance_ (mu: Complex): Complex = (normSq - mu * mu * nd) / nd

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample covariance (or population covariance) of this vector with vector y.
     *  @param y  the other vector
     */
    infix def cov (y: VectorC): Complex  = ((this dot y) - sum * y.sum / nd) / (nd-1)
    infix def cov_ (y: VectorC): Complex = ((this dot y) - sum * y.sum / nd) / nd

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the mean square (ms) (or root mean square (rms)) of this vector.
     */
    def ms: Complex  = normSq / nd
    def rms: Complex = Complex.sqrt (ms)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the sample standard deviation (and population standard deviation).
     */
    def stdev: Complex   = Complex.sqrt (variance)
    def stdev_ : Complex = Complex.sqrt (variance_)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for stationary series.
     *  This follows the standard defintion that divides by the number of elements dim
     *  to avoid singularity issues.  IT ALWAYS DIVIDES BY 'dim-1'
     *  @see stats.stackexchange.com/questions/56238/question-about-sample-autocovariance-function
     *  @param k  the lag parameter (0 <= k < n) 
     */
    def acov (k: Int = 1): Complex =
        val mu = mean
        var s  = Complex._0
        for i <- 0 until dim-k do s += (v(i) - mu) * (v(i+k) - mu)
        s / (dim-1)
    end acov

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for stationary series
     *  with the mean pre-computed.
     *  @param k   the lag parameter (0 <= k < n) 
     *  @param mu  the pre-computed mean
     */
    def acov (k: Int, mu: Complex): Complex =
        var s  = Complex._0
        for i <- 0 until dim-k do s += (v(i) - mu) * (v(i+k) - mu)
        s / (dim-1)
    end acov

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag cross-covariance of this vector and vector y for stationary
     *  series with both means pre-computed.
     *  @param k     the lag parameter (0 <= k < n)
     *  @param mu    the pre-computed mean for this
     *  @param y     the other vector
     *  @param mu_y  the pre-computed mean for y
     */
    def ccov (k: Int, mu: Complex, y: VectorC, mu_y: Complex): Complex =
        var s  = Complex._0
        for i <- 0 until dim-k do s += (v(i) - mu) * (y.v(i+k) - mu_y)
        s / (dim-1)
    end ccov

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for stationary series.
     *  This follows an intuitive defintion that divides by the number of elements summed dim-k.
     *  @param k  the lag parameter (0 <= k < n) 
     */
    def acov2 (k: Int = 1): Complex =
        val n  = dim - k
        val mu = mean
        var s  = Complex._0
        for i <- 0 until n do s += (v(i) - mu) * (v(i+k) - mu)
        s / n
    end acov2

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-covariance of this vector for non-stationary series.
     *  @param k  the lag parameter (0 <= k < n) 
     */
    def acov_ (k: Int = 1): Complex =
        val n   = dim - k
        val ss  = sums (k)
        val mu  = ((ss._1 + ss._2) / n, (ss._2 + ss._3) / n)
        var s = Complex._0
        for i <- 0 until n do s += (v(i) - mu._1) * (v(i+k) - mu._2)
        s / n
    end acov_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute Pearson's correlation of this vector with vector y.
     *  If either variance is zero, will result in Not-a-Number (NaN), return
     *  one if the vectors are the same, or -0 (indicating undefined).
     *  @param y  the other vector
     */
    infix def corr (y: VectorC): Complex =
        val c = cov (y) / Complex.sqrt (variance * y.variance)
        if c.isNaN then if this == y then Complex._1 else -Complex._0 else c
    end corr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-correlation of this vector (assumes a stationary
     *  process vector, if not its an approximation).
     *  @param k  the lag parameter (0 <= k < n) 
     */
    def acorr (k: Int = 1): Complex =
        val mu = mean
        acov (k, mu) / acov (0, mu)
    end acorr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag cross-correlation of this vector (assumes a stationary
     *  process vector, if not its an approximation).
     *  @param y  the other vector
     *  @param k  the lag parameter (0 <= k < n)
     */
    def ccorr (y: VectorC, k: Int = 1): Complex =
        val mu   = mean
        val mu_y = y.mean
        ccov (k, mu, y, mu_y) / (stdev * y.stdev)
    end ccorr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the 'k'-lag auto-correlation of this vector for a non-stationary series.
     *  @param k  the lag parameter (0 <= k < n) 
     */
    def acorr_ (k: Int = 1): Complex =
        val n   = dim - k
        val ss  = sums (k)
        val sq  = normSqs (k)
        val mu  = ((ss._1 + ss._2) / n, (ss._2 + ss._3) / n)
        val vr  = ((sq._1 + sq._2 - (mu._1 * mu._1) * n) / n,
                   (sq._2 + sq._3 - (mu._2 * mu._2) * n) / n)
        var s = Complex._0
        for i <- 0 until n do s += (v(i) - mu._1) * (v(i+k) - mu._2)
        (s / n) / (Complex.sqrt (vr._1) * Complex.sqrt (vr._2))
    end acorr_

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute Spearman's rank correlation of this vector with vector y.
     *  The `iqsort` method gives the rank order of a vector.
     *  @see  en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
     *  @param y  the other vector
     */
    infix def scorr (y: VectorC): Complex =
        val rk1 = iqsort                                       // rank order for this vector
        val rk2 = y.iqsort                                     // rank order for vector y
        var s = Complex._0
        for i <- v.indices do s += (rk1(i) - rk2(i))~^2
        Complex._1 - Complex._6 * s / (nd * (nd*nd - 1))
    end scorr

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the skewness of this vector.  Negative skewness indicates the
     *  distribution is elongated on the left, zero skewness indicates it is
     *  symmetric, and positive skewness indicates it is elongated on the right.
     *  @see www.mathworks.com/help/stats/skewness.html
     *  >> E(X - μ)^3 / σ^3
     */
    def skew: Complex   = ((this - mean)~^3).sum / (stdev_ ~^3 * nd)
    def skew_ : Complex = skew * math.sqrt (nd * (nd-1)) / (nd-2)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Compute the kurtosis of 'x' vector.  High kurtosis (> 3) indicates a 
     *  distribution with heavier tails than a Normal distribution.
     *  @see www.mathworks.com/help/stats/kurtosis.html
     *  >> E(X - μ)^4 / σ^4
     */
    def kurtosis: Complex = ((this - mean)~^4).sum / (variance_ ~^2 * nd)
    def kurtosis_ : Complex = kurtosis * (nd-1) * ((nd+1) - 3 * (nd-1)) / ((nd-2) * (nd-3)) + 3

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Produce a standardized version of the vector by subtracting the mean and
     *  dividing by the standard deviation (e.g., Normal -> Standard Normal).
     */
    def standardize: VectorC = (this - mean) / stdev
    def standardize2: VectorC = (this - mean) / (stdev + EPSILON)

end VectorC


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `VectorC` object is the companion object for the `VectorC` class.
 */
object VectorC:

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorC` from an immutable indexed sequence of `Complex`s.
     *  @param xs  the sequence/array of the `Complex` numbers
     */
    def apply (xs: collection.immutable.IndexedSeq [Complex]): VectorC = new VectorC (xs.size, xs.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorC` from a mutable indexed sequence of `Complex`s.
     *  @param xs  the sequence/array of the `Complex` numbers
     */
    def apply (xs: IndexedSeq [Complex]): VectorC = new VectorC (xs.size, xs.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorC` from one or more values (repeated values `Complex`*).
     *  @param x   the first `Complex` number
     *  @param xs  the varargs of `Complex` numbers
     */
    def apply (x: Complex, xs: Complex*): VectorC = new VectorC (xs.size + 1, x +: xs.toArray)

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorC` from one or more values (repeated values String*).
     *  For numeric types, assign missing value indicator upon format failure.
     *  @param x   the first string
     *  @param xs  the varargs of strings
     */
    def apply (x: String, xs: String*): VectorC =
        val y = new VectorC (1 + xs.length)
        for i <- y.indices do
            y(i) = if i == 0 then Complex (x) else Complex (xs(i-1))
        end for
        y
    end apply

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorC` from a mutable indexed sequence of `Double`.
     *  @param xs  the sequence/array of the `Double` numbers
     */
    def fromDoubles (xs: IndexedSeq [Double]): VectorC = VectorC (xs.map (Complex (_)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorC` from a mutable indexed sequence of `String`.
     *  @param xs  the sequence/array of the `String` numbers
     */
    def fromStrings (xs: IndexedSeq [String]): VectorC = VectorC (xs.map (Complex (_)))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Return a `VectorC` consisting of a sequence of integers in a range.
     *  @param r  the range of values
     */
    def range (r: Range): VectorC = VectorC (for i <- r yield Complex (i))
    def range (i1: Int, i2: Int): VectorC = VectorC (for i <- i1 until i2 yield Complex (i))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a `VectorC` with n elements and fill it with the value x.
     *  @param n  the number of elements
     *  @param x  the value to assign to all elements
     */
    def fill (n: Int)(x: Complex): VectorC = new VectorC (n, Array.fill (n)(x))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a one vector (all elements are one) of length n.
     *  @param size  the size of the new vector
     */
    def one (n: Int): VectorC = new VectorC (n, Array.fill (n)(Complex._1))

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** Create a vector of the form (0, ... 1, ... 0) where the 1 is at position j.
     *  @param j     the position to place the 1
     *  @param size  the size of the vector (upper bound = size - 1)
     */
    def oneAt (j: Int, size: Int): VectorC =
        val x = new VectorC (size)
        x.v(j) = Complex._1
        x
    end oneAt

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    /** A null vector of type `VectorC`.
     */
    val nullv: VectorC = null.asInstanceOf [VectorC]

end VectorC


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorCTest` main function tests the operations provided by the `VectorC` class.
 *  Only the most commonly used inherited operations are shown.
 *  @see mutable.IndexedSeq for a complete list
 *  > runMain scalation.mathstat.vectorCTest
 */
@main def vectorCTest (): Unit =

    val x = VectorC (_1, _2, _3)
    val y = VectorC (_4, _6, _5)
    val z = VectorC (_4, _6, _5)
    val w = VectorC (_3, _4, _5, _5)
    val u = VectorC ("1", "2", "3", "4")
    val a = _2

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

    println (s"x `contains` _2     = ${x `contains` _2}")        // element contained in vector
    println (s"x `contains` _4     = ${x `contains` _4}")
    println (s"x.exists (_ > _2)   = ${x.exists (_ > _2)}")      // existence of element satisfying predicate
    println (s"x.groupBy (_ > _2)  = ${x.groupBy (_ > _2)}")     // group according function values
    println (s"x.indexOf (_2)      = ${x.indexOf (_2)}")         // index of first element equaling
    println (s"x.indexWhere (_ > _2) = ${x.indexWhere (_ > _2)}")  // index of first element satisfying predicate
    println (s"x.indices           = ${x.indices}")              // indices of the vector
    println (s"u.map (_ * _2)      = ${u.map (_ * _2)}")         // map vector elements using the function
    println (s"x.max               = ${x.max}")                  // maximum element
    println (s"x.min               = ${x.min}")                  // minimum element
    println (s"x.product           = ${x.product}")              // product of all elements
    println (s"x.sum               = ${x.sum}")                  // sum of all elements
    println (s"w.toArray           = ${stringOf (w.toArray)}")   // convert to array - use stringOf to print arrays
    println (s"w.toSet             = ${w.toSet}")                // convert to set

    banner ("Implemented Operations:")

    println (s"x(2)                = ${x(2)}")                   // value at index
    println (s"x(0 to 2)           = ${x(0 to 2)}")              // values in exclusive range

    println (s"-x                  = ${-x}")                     // unary minus
    println (s"x + y               = ${x + y}")                  // element-wise vector addition
    println (s"x - y               = ${x - y}")                  // element-wise vector subtraction
    println (s"x * y               = ${x * y}")                  // element-wise vector multiplication
    println (s"x / y               = ${x / y}")                  // element-wise vector division
    println (s"x ++ y              = ${x ++ y}")                 // concatenate vectors

    println (s"x + a               = ${x + a}")                  // add scalar a
    println (s"x - a               = ${x - a}")                  // subtract scalar a
    println (s"x * a               = ${x * a}")                  // multiply by scalar a
    println (s"x / a               = ${x / a}")                  // divide by scalar a
    println (s"x ~^ 2              = ${x ~^ 2}")                 // raise to power of scalar a
    println (s"a +: y              = ${a +: x}")                 // prepend scalar a
    println (s"x :+ a              = ${x :+ a}")                 // append scalar a

    println (s"(x-y).abs           = ${(x-y).abs}")              // absolute value
    println (s"x.cnormSq           = ${x.cnormSq}")              // center norm squared
    println (s"x.cumulate          = ${x.cumulate}")             // cumulate all prior values
    println (s"x diff w            = ${x diff w}")               // multi-set difference
    println (s"w.distinct          = ${w.distinct}")             // extract distinct values
    println (s"x dot y             = ${x dot y}")                // dot product
    println (s"u.filter (_ > _2)   = ${u.filter (_ > _2)}")      // filter on predicate
    println (s"u.filterNot (_ > _2) = ${u.filterNot (_ > _2)}")  // filter on not predicate
    println (s"u.filterPos (_ > _2) = ${u.filterPos (_ > _2)}")  // filter return indices
    println (s"x intersect w       = ${x intersect w}")          // multi-set intersection
    println (s"y.iselsort          = ${stringOf (y.iselsort)}")  // indirect QuickSort in ascending order
    println (s"y.iqsort            = ${stringOf (y.iqsort)}")    // indirect SelectionSort in ascending order
    println (s"x.norm              = ${x.norm}")                 // Euclidean norm
    println (s"x.normalize         = ${x.normalize}")            // normalize the vector to a unit vector
    println (s"x.normalize1        = ${x.normalize1}")           // normalize the vector to max 1 vector
    println (s"x.normSq            = ${x.normSq}")               // Euclidean norm squared
    println (s"x.norm1             = ${x.norm1}")                // Manhattan norm
    println (s"x.recip             = ${x.recip}")                // reciprocal
    println (s"x.reverse           = ${x.reverse}")              // reverse elements in vector
//  println (s"y.sorted            = ${y.sorted}")               // sort in ascending order
    println (s"y.sortWith (_ > _)  = ${y.sortWith (_ > _)}")     // sort in descending order
    println (s"y.toProbabilty      = ${y.toProbability}")        // convert to a probability vector

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

end vectorCTest


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorCTest2` main function tests the indirect sorting operations provided
 *  by the `VectorC` class.
 *  > runMain scalation.mathstat.vectorCTest2
 */
@main def vectorCTest2 (): Unit =

    val y  = VectorC (_4, _6, _5, _2, _8, _3, _9, _7, _5, _2)
    val is = y.iselsort
    val iq = y.iqsort

    banner ("Index Order")
    println (s"y.iselsort = ${stringOf (is)}")              // indirect QuickSort in ascending order
    println (s"y.iqsort   = ${stringOf (iq)}")              // indirect SelectionSort in ascending order

    banner ("Value Order")
    println (s"y.iselsort = ${stringOf (is.map (y(_)))}")   // indirect QuickSort in ascending order
    println (s"y.iqsort   = ${stringOf (iq.map (y(_)))}")   // indirect SelectionSort in ascending order

end vectorCTest2


//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `vectorCTest3` main function tests auto-covariance and auto-correlation methods
 *  provided by the `VectorC` class.
 *  > runMain scalation.mathstat.vectorCTest3
 */
@main def vectorCTest3 (): Unit =

    val y    = VectorC (_1, _2, _4, _7, _9, _8, _6, _5, _3)
    val z    = VectorC (_2, _1, _9, _7, _4, _8, _6, _3, _5)
    val mu_y = y.mean
    val mu_z = z.mean
    val sd_y = y.stdev
    val sd_z = z.stdev
    val sd_p = sd_y * sd_z                                   // product of stdev's

    println (s"y    = $y")
    println (s"z    = $z")
    println (s"mu_y = $mu_y")
    println (s"mu_z = $mu_z")
    println (s"sd_y = $sd_y")
    println (s"sd_z = $sd_z")
    println (s"sd_p = $sd_p")

    banner (s"Regular Covariance and Correlation of y and z")
    println (s"Covariance   y cov z          = ${y cov z}")
    println (s"Correlation  y corr z         = ${y corr z}")
    println (s"Correlation  (y cov z) / sd_p = ${(y cov z) / sd_p}")

    for k <- 0 to 2 do
        banner (s"Lag-$k Auto-Covariance")
        println (s"Stationary     y acov $k   = ${y.acov (k)}")
        println (s"Non-Stationary y acov_ $k  = ${y.acov_ (k)}")

        banner (s"Lag-$k Auto-Correlation")
        println (s"Stationary     y acorr $k  = ${y.acorr (k)}")
        println (s"Non-Stationary y acorr_ $k = ${y.acorr (k)}")

        banner (s"Lag-$k Cross-Covariance")
        println (s"Stationary     y.ccov ($k, mu_y, z, mu_z)  = ${y.ccov (k, mu_y, z, mu_z)}")

        banner (s"Lag-$k Cross-Correlation")
        println (s"Stationary     y.ccorr (z, $k) = ${y.ccorr (z, k)}")
    end for

end vectorCTest3

